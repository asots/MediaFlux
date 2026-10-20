"""批量文件规则走既有 Agent 确认卡和持久队列，不触碰真实云盘。"""

from __future__ import annotations

import json
from copy import deepcopy
from unittest import mock

import pytest

from app.agent.domain_catalog import build_tool_specs
from app.agent.domains.guangya import fs_change_actions, workspace_actions
from app.agent.errors import AgentToolError
from app.agent.models import ToolContext, ToolResult
from app.agent.session_context import SQLiteAgentSessionContextRepository
from app.clients.guangya import GuangYaFile
from app.modules import guangya_fs_change, guangya_workspace
from tests.support import isolated_test_database
from tests.test_agent_guangya_fs_gateway import FakeGatewayClient

_REF = "OBJ" + "A" * 24
_THRESHOLD = 200 * 1024 * 1024


def _arguments(**values):
    return fs_change_actions.guangya_fs_change_arguments({
        "operations": [{"op": "batch_files", "scope_ref": _REF, **values}],
        "trigger_strm": False,
    })


def test_batch_rules_normalize_without_implicit_deletion_or_kind_protection():
    batch = _arguments(target_path="/source", cleanup_empty_dirs=True)["operations"][0]
    assert "trash_filter" not in batch
    assert batch["move_filter"] == {}
    assert batch["conflict"] == "keep_both"
    batch = _arguments(trash_filter={"size_lt_bytes": _THRESHOLD})["operations"][0]
    assert batch["trash_filter"] == {"size_lt_bytes": _THRESHOLD}
    assert "target_path" not in batch
    batch = _arguments(trash_filter={
        "kinds": [" VIDEO ", "video"], "extensions": ["MP4"], "name_contains": ["广告", "AD"],
    })["operations"][0]
    assert batch["trash_filter"] == {
        "kinds": ["video"], "extensions": ["mp4"], "name_contains": ["广告", "ad"],
    }
    assert fs_change_actions._operation_object_refs([batch]) == (_REF,)


@pytest.mark.parametrize("values", [
    {}, {"scope_ref": "0", "cleanup_empty_dirs": True},
    {"trash_filter": {}}, {"trash_filter": None}, {"trash_filter": {"size": 1}},
    {"trash_filter": {"size_lt_bytes": True}}, {"trash_filter": {"size_lt_bytes": -1}},
    {"trash_filter": {"size_lt_bytes": 200.5}}, {"trash_filter": {"size_lt_bytes": "200M"}},
    {"trash_filter": {"size_lt_bytes": 2**53}},
    {"trash_filter": {"size_lt_bytes": 10, "size_gte_bytes": 10}},
    {"trash_filter": {"kinds": ["directory"]}}, {"trash_filter": {"kinds": []}},
    {"trash_filter": {"name_contains": [""]}}, {"trash_filter": {"name_contains": [" "]}},
    {"trash_filter": {"name_contains": [3]}}, {"trash_filter": {"name_contains": ["x"] * 17}},
    {"trash_filter": {"extensions": [".mp4"]}}, {"trash_filter": {"extensions": ["*"]}},
    {"move_filter": {}}, {"target_path": "../relative"}, {"target_path": 123},
    {"cleanup_empty_dirs": "true"}, {"cleanup_empty_dirs": True, "conflict": "overwrite"},
    {"cleanup_empty_dirs": True, "purge": True},
])
def test_invalid_or_ambiguous_rules_are_rejected(values):
    with pytest.raises(AgentToolError):
        _arguments(**values)


def test_batch_cannot_mix_manual_operations_or_relax_manual_limit():
    batch = _arguments(cleanup_empty_dirs=True)["operations"][0]
    with pytest.raises(AgentToolError, match="独占"):
        fs_change_actions.guangya_fs_change_arguments({"operations": [batch, {"op": "trash", "object_ref": _REF}]})
    with pytest.raises(AgentToolError, match="200"):
        fs_change_actions.guangya_fs_change_arguments({"operations": [{"op": "trash", "object_ref": _REF}] * 201})


def test_existing_confirmation_tool_exposes_batch_rule_schema():
    spec = next(spec for spec in build_tool_specs() if spec.name == "guangya.fs.change.execute")
    assert spec.requires_confirmation and spec.risk.value == "danger"
    assert spec.context_handler is None and spec.handler is None
    schema = next(branch for branch in spec.parameters["properties"]["operations"]["items"]["oneOf"]
                  if branch["properties"]["op"]["enum"] == ["batch_files"])
    assert schema["required"] == ["op", "scope_ref"]
    assert schema["properties"]["trash_filter"]["minProperties"] == 1
    assert "directory" not in schema["properties"]["trash_filter"]["properties"]["kinds"]["items"]["enum"]
    assert "覆盖" in schema["properties"]["conflict"]["description"]


@pytest.mark.parametrize("message", [
    "光鸭newnsfw低于200M的都删掉，不限字幕封面种子",
    "把媒体平铺到根目录，重名保留两份并清理空子目录",
])
def test_batch_intents_find_the_existing_tool(message):
    from app.agent.kernel.capabilities import CapabilityRetriever
    from app.agent.kernel.ports import catalog_from_tool_specs

    catalog = catalog_from_tool_specs(build_tool_specs())
    assert "guangya.fs.change.execute" in CapabilityRetriever().retrieve(message, catalog).names


@pytest.fixture
def environment(tmp_path, monkeypatch):
    with isolated_test_database(), mock.patch("socket.socket.connect", side_effect=AssertionError("禁止外联")):
        for module, dirname in ((guangya_workspace, "observations"), (guangya_fs_change, "changes")):
            monkeypatch.setattr(module, "_directory", lambda name=dirname: tmp_path / name)
            monkeypatch.setattr(module, "get_web_secret", lambda: "test-secret")
        monkeypatch.setattr(guangya_workspace, "organize_operation_owner_digest", lambda owner: f"digest:{owner}")
        monkeypatch.setattr(guangya_fs_change, "_owner_digest", lambda owner: f"digest:{owner}")
        monkeypatch.setattr(fs_change_actions, "organize_operation_owner_digest", lambda owner: f"digest:{owner}")
        client = FakeGatewayClient()
        repository = SQLiteAgentSessionContextRepository(secret_provider=lambda: "test-secret")
        workspace = workspace_actions.GuangyaWorkspaceActions(repository)
        actions = fs_change_actions.GuangyaFsChangeActions(repository, workspace=workspace)
        context = ToolContext(owner="owner", session_id="session")
        monkeypatch.setattr(workspace_actions, "GuangYaClient", lambda: client)
        monkeypatch.setattr(fs_change_actions, "GuangYaClient", lambda: client)
        observed = workspace.query_guangya_filesystem(workspace_actions.guangya_fs_query_arguments({
            "operation": "list", "path": "/source", "page_size": 1,
        }), context)
        scope_ref = observed.data["scope_objects"][0]["object_ref"]
        yield client, repository, actions, context, scope_ref


def test_first_page_scope_expands_all_files_and_restored_confirmation_queues_same_manifest(environment):
    client, repository, actions, context, scope_ref = environment
    # 初次查询仅展示一项；确认预览由后端扫描新增的完整目录，而非让模型逐页枚举。
    client.directories["source"] = [
        GuangYaFile(f"file-{i}", f"part-{i}.mp4", False, parent_id="source", size=_THRESHOLD - 1, etag=str(i))
        for i in range(225)
    ]
    client.directories["source"].append(GuangYaFile("boundary", "keep.mp4", False, parent_id="source", size=_THRESHOLD))
    before = deepcopy(client.directories)
    arguments = _arguments(scope_ref=scope_ref, trash_filter={"size_lt_bytes": _THRESHOLD})
    confirmation, fingerprint = actions.prepare_guangya_fs_change_confirmation(arguments, context)
    assert confirmation.status == "confirmation_required"
    assert confirmation.data["trash_count"] == 225
    assert confirmation.data["cloud_write"] is False
    assert str(_THRESHOLD) in confirmation.data["batch_summary"]
    assert confirmation.data["batch_summary"] in confirmation.data["effects"]
    assert client.directories == before

    # 进程重建后仍用原冻结清单；确认时提交的新规则及新出现的匹配文件都不能扩权。
    restored = fs_change_actions.GuangyaFsChangeActions(repository, workspace=actions.workspace)
    client.directories["source"].append(GuangYaFile("later", "new.mp4", False, parent_id="source", size=1))
    manager = mock.Mock()
    manager.start_durable_operation.return_value = {"ok": True, "task_id": "b" * 32, "queued": True, "queue_position": 1}
    with mock.patch("app.modules.organize_tasks.get_organize_manager", return_value=manager):
        accepted = restored.execute_guangya_fs_change_confirmed(
            {"operations": [{"op": "trash", "object_ref": "arbitrary"}]}, fingerprint, context,
        )
    assert accepted.status == "accepted"
    assert accepted.data["batch_summary"] == confirmation.data["batch_summary"]
    kwargs = manager.start_durable_operation.call_args.kwargs
    assert kwargs["job_kind"] == "agent_guangya_fs_change"
    assert kwargs["ttl_seconds"] == 6 * 60 * 60
    plan = guangya_fs_change.load_fs_change_plan(kwargs["payload"]["plan_id"], owner="owner")
    assert len(plan["operations"]) == 225
    assert {row["source"]["file_id"] for row in plan["operations"]} == {f"file-{i}" for i in range(225)}
    assert len(client.list_dir("source")) == 227


def test_no_matching_files_has_no_confirmation_or_queue(environment):
    _client, _repo, actions, context, scope_ref = environment
    arguments = _arguments(scope_ref=scope_ref, trash_filter={"size_lt_bytes": 0})
    result, fingerprint = actions.prepare_guangya_fs_change_confirmation(arguments, context)
    assert result.status == "no_changes"
    assert not fingerprint and result.data["cloud_write"] is False
    assert result.data["total"] == 0
    assert result.data["batch_summary"]


def test_bulk_preview_does_not_resolve_shared_target_per_file(environment):
    client, _repo, actions, context, scope_ref = environment
    client.directories["source"] = [
        GuangYaFile(f"file-{i}", f"part-{i}.mp4", False, parent_id="source", size=_THRESHOLD, etag=str(i))
        for i in range(225)
    ]
    arguments = _arguments(scope_ref=scope_ref, target_path="/target")
    with mock.patch.object(client, "list_dir", wraps=client.list_dir) as listing:
        confirmation, _ = actions.prepare_guangya_fs_change_confirmation(arguments, context)
    assert confirmation.data["move_count"] == 225
    root_reads = sum(str(call.args[0]) == "0" for call in listing.call_args_list)
    assert root_reads <= 6  # 根路径查找不随媒体数量增长；写前验证不受此只读缓存影响。
    assert client.file_info("file-224").parent_id == "source"


def test_scope_from_other_session_is_not_writable(environment):
    _client, _repo, actions, context, scope_ref = environment
    arguments = _arguments(scope_ref=scope_ref, cleanup_empty_dirs=True)
    with pytest.raises(AgentToolError):
        actions.prepare_guangya_fs_change_confirmation(arguments, ToolContext(owner=context.owner, session_id="other"))


def test_nested_scope_rules_remain_visible_on_public_confirmation(environment):
    from app.agent.kernel.projection import DefaultProjector

    client, _repo, actions, context, _scope_ref = environment
    client.move(["source"], "target")
    observation = actions.workspace.query_guangya_filesystem(workspace_actions.guangya_fs_query_arguments({
        "operation": "list", "path": "/target/source",
    }), context)
    arguments = _arguments(scope_ref=observation.data["scope_objects"][0]["object_ref"],
                           trash_filter={"size_lt_bytes": _THRESHOLD})
    confirmation, _ = actions.prepare_guangya_fs_change_confirmation(arguments, context)
    data = DefaultProjector().project(confirmation).public_content["data"]
    assert data["batch_summary"] == confirmation.data["batch_summary"]
    assert str(_THRESHOLD) in data["effects"][0]
    assert "target" in data["effects"][0] and "source" in data["effects"][0]


def test_large_result_keeps_failure_totals_with_bounded_public_receipt():
    from app.agent.kernel.projection import DefaultProjector

    rows = [{"position": index, "operation": "move", "status": "completed", "label": "影片" * 21}
            for index in range(1, 2001)]
    rows[-1].update(status="failed", reason="execution_error")
    stats = {"total": 2000, "moved": 1999, "failed": 1}
    result = ToolResult(False, "partial", "光鸭后台任务部分完成", data={"operation_items": rows, "stats": stats})
    public = DefaultProjector().project(result).public_content
    assert public["status"] == "partial" and public["ok"] is False
    assert public["data"]["stats"] == stats
    assert public["truncated"] is True  # 仅会话展示压缩，完整回执仍在持久队列。
    assert len(json.dumps(public, ensure_ascii=False).encode()) < 128 * 1024
