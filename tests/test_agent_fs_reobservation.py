"""fs.change 来源恢复：离线 Provider、真实 SQLite 与签名 plan，不是生产实测。

只替换 Provider IO/隔离落盘；单次 prepare、query 与原生 pipeline 都走真实实现。
不执行确认、不伪造预检或写入结果；读取工厂的 spy 原样调用真实实现。
"""
from __future__ import annotations

import asyncio
import json
import time
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
import requests

from app.agent import guangya_fs_change_actions as changes
from app.agent import guangya_workspace_actions as files
from app.agent.confirmation import SQLiteConfirmationStore
from app.agent.domain_catalog import build_tool_specs
from app.agent.errors import AgentToolError
from app.agent.kernel.capabilities import ToolCatalog, ToolEffect
from app.agent.kernel.effects import ConfirmationEffectPlanStore, PreparedEffect
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.pipeline import ToolCallContext, ToolPipeline
from app.agent.kernel.ports.existing_actions import adapt_tool_spec
from app.agent.kernel.projection import DefaultProjector
from app.agent.kernel.state import CancellationToken
from app.agent.models import ToolContext
from app.agent.session_context import SQLiteAgentSessionContextRepository
from app.clients.guangya import GuangYaFile
from app.modules import guangya_fs_change as fs_plans
from app.modules import guangya_workspace as workspace
from tests.support import initialized_web_runtime
from tests.test_agent_reobservation_queries import QueryProvider
from tests.test_guangya_observation_sources import FILMS, SHOWS

FS_READ = "guangya.fs.query"
FS_EXECUTE = "guangya.fs.change.execute"
DESTINATION = "private-provider-destination"
NESTED = "private-provider-nested"
MOVIE = "private-provider-movie"
SUBTITLE = "private-provider-episode-subtitle"
OFFLINE = "fs.change 离线回归（不是生产实测）"


@pytest.fixture
def case(monkeypatch):
    def no_network(*_args, **_kwargs):
        raise AssertionError(OFFLINE + "：禁止网络 IO")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    with initialized_web_runtime() as paths:
        provider = QueryProvider()
        provider.directories["0"].append(
            GuangYaFile(DESTINATION, "Destination", True, parent_id="0"))
        provider.directories[DESTINATION] = []

        def no_write(*args, **kwargs):
            provider.writes.append({"args": args, "kwargs": kwargs})
            raise AssertionError(OFFLINE + "：未获新确认，禁止 Provider 写入")

        for method in ("rename", "copy", "move", "create_directory", "mkdir", "delete"):
            monkeypatch.setattr(provider, method, no_write, raising=False)
        monkeypatch.setattr(workspace, "PATHS", paths)
        monkeypatch.setattr(fs_plans, "PATHS", paths)
        monkeypatch.setattr(files, "GuangYaClient", lambda: provider)
        monkeypatch.setattr(changes, "GuangYaClient", lambda: provider)
        files.reset_guangya_workspace_context_for_tests()
        changes.reset_guangya_fs_change_context_for_tests()
        repository = SQLiteAgentSessionContextRepository()
        files.configure_guangya_workspace_context(repository)
        changes.configure_guangya_fs_change_context(repository)
        current = SimpleNamespace(
            root=paths.data_dir, provider=provider,
            context=ToolContext(owner="offline-fs-source-owner", session_id="offline-fs-source-session"))
        try:
            yield current
        finally:
            files.reset_guangya_workspace_context_for_tests()
            changes.reset_guangya_fs_change_context_for_tests()
            assert provider.writes == [], OFFLINE + "：诊断不能触发写入"


def read(case, arguments=None, *, source=None, context=None):
    context = context or case.context
    if source is not None:
        context = replace(context, reobservation_source=source)
    return files.query_guangya_filesystem(
        files.guangya_fs_query_arguments(arguments or {
            "operation": "list", "path": FILMS["path"], "max_items": 17}), context)


def object_ref(result, name):
    return next(item["object_ref"] for item in result.data["entries"] if item["object_name"] == name)


def prepare(case, observation, operations, *, source=None):
    arguments = changes.guangya_fs_change_arguments({
        "observation_ref": observation.data["observation_ref"],
        "operations": operations, "trigger_strm": False})
    context = replace(case.context, reobservation_source=source) if source is not None else case.context
    result, fingerprint = changes.prepare_guangya_fs_change_confirmation(arguments, context)
    assert result.ok is True and result.data["total"] == len(operations)
    assert result.data["cloud_write"] is False
    assert result.ok is True and result.status == "confirmation_required"
    assert len(fingerprint) == 64
    source = result.effect_metadata.get("reobservation_source")
    assert isinstance(source, dict), "fs.change prepare 必须从冻结 plan 输出私有 reobservation_source"
    assert (source["version"], source["kind"], source["owner"], source["session_id"],
            source["credential_generation"]) == (
        1, "guangya_files", case.context.owner, case.context.session_id, 41)
    assert source["available"] is True and source["read"]["tool"] == FS_READ
    assert isinstance(source["baseline"], dict)
    assert json.loads(json.dumps(source)) == source
    # 冷恢复读取已冻结的确认卡，而不是通过 prepare({}) 隐式重新编译旧计划。
    case.confirmation_plan = ConfirmationEffectPlanStore(SQLiteConfirmationStore()).freeze(
        owner=context.owner, session_id=context.session_id, generation=1,
        tool_name=FS_EXECUTE, effect=ToolEffect.DANGER, arguments=arguments,
        prepared=PreparedEffect(preview=result.to_dict(), snapshot_fingerprint=fingerprint,
                                metadata=result.effect_metadata),
        public_result=result.to_dict(),
    )
    return result, source, fingerprint


def restore_confirmation(case):
    stored = ConfirmationEffectPlanStore(SQLiteConfirmationStore()).get_active_plan(
        owner=case.context.owner, session_id=case.context.session_id,
        generation=1, plan_id=case.confirmation_plan.plan_id,
    )
    assert stored is not None
    assert stored == case.confirmation_plan
    return stored


def invalidate(case, observation_ref, mode):
    path = workspace._plan_path(workspace.observation_plan_id(observation_ref))
    if mode == "deleted":
        workspace.discard_observation(observation_ref)
        assert not path.exists()
    else:
        # 只令真实签名 OBS 过期；不改 plan、来源、预检或执行回执。
        payload = workspace.load_directory_observation(observation_ref, owner=case.context.owner)
        payload["expires_at_epoch"] = time.time() - 1
        workspace._atomic_write(payload)
        assert path.exists()
    with pytest.raises(workspace.GuangYaWorkspaceError):
        workspace.load_directory_observation(observation_ref, owner=case.context.owner)


def frozen_plan(case):
    paths = list((case.root / "agent-guangya-fs-change").glob("*.json"))
    assert len(paths) == 1
    payload = fs_plans.load_fs_change_plan(paths[0].stem, owner=case.context.owner)
    assert payload["auth"] and payload["status"] == "previewed"
    return payload


def assert_private(*results):
    for result in results:
        projected = DefaultProjector().project(result)
        encoded = json.dumps({"public": result.to_dict(), "model": result.to_model_dict(),
                              "projected": projected.public_content, "projected_model": projected.model_content}, ensure_ascii=False)
        for private in ("reobservation_source", "source_root", "source_observations",
                        "credential_generation", "file_id", "parent_id", "owner_digest",
                        "private-provider-", "offline-fs-source-owner", "offline-fs-source-session",
                        FILMS["path"], SHOWS["path"]):
            assert private not in encoded, f"{OFFLINE}：公共 DTO 泄露 {private}"


@pytest.mark.parametrize("operation", ["rename", "copy", "move"])
@pytest.mark.parametrize("mode", ["deleted", "expired"])
@pytest.mark.parametrize("restored_at", ["before_invalidation", "after_invalidation"])
def test_single_directory_plan_reobserves_after_original_obs_is_unusable(case, operation, mode, restored_at):
    observed = read(case)
    old_ref = object_ref(observed, "Nested")
    change = {"op": operation, "object_ref": old_ref}
    change.update({"new_name": "Nested-Renamed"} if operation == "rename"
                  else {"target_path": "/Destination"})
    prepared, source, fingerprint = prepare(case, observed, [change])
    plan = frozen_plan(case)
    assert plan["operations"][0]["source"]["file_id"] == NESTED
    assert plan["operations"][0]["source"]["is_dir"] is True
    previous = restore_confirmation(case) if restored_at == "before_invalidation" else None
    invalidate(case, observed.data["observation_ref"], mode)
    # 丢弃进程内缓存，必须由真实 SQLite flow + 签名冻结计划恢复，不能重载旧 OBS。
    changes.clear_guangya_fs_change_context(owner=case.context.owner)
    stored = restore_confirmation(case)
    assert stored.preview == prepared.to_dict() and stored.snapshot_fingerprint == fingerprint
    assert stored.metadata["reobservation_source"] == source
    source = stored.metadata["reobservation_source"]
    if previous is not None:
        assert stored == previous
    frozen = deepcopy(source)
    case.provider.live_file(NESTED).name = "Nested-External-Rename"
    case.provider.trace.clear()
    renewed = read(case, {"operation": "reobserve"}, source=source)
    mapping = renewed.data["reobservation"]
    assert renewed.ok is True and renewed.data["observation_ref"] != observed.data["observation_ref"]
    assert mapping["basis"] == "provider_identity" and mapping["complete"] is True
    assert len(mapping["items"]) == 1
    item = mapping["items"][0]
    assert item["previous_ref"] == old_ref and item["object_ref"] != old_ref
    assert item["state"] == "same_identity" and "name" in item["changed_fields"]
    saved = workspace.load_directory_observation(renewed.data["observation_ref"], owner=case.context.owner)
    assert workspace.observation_entry_map(saved)[item["object_ref"]]["file_id"] == NESTED
    assert (saved["operation"], saved["query"], saved["kinds"], saved["max_depth"], saved["max_items"]) == (
        "list", "", [], 0, 17)
    assert saved["roots"] == [FILMS]
    assert [row["id"] for row in case.provider.trace if row["method"] == "list_dir"] == [
        "0", FILMS["parent_id"], FILMS["file_id"]]
    assert source == frozen and frozen_plan(case) == plan
    assert_private(observed, prepared, renewed)


@pytest.mark.parametrize("primary", [0, 1])
@pytest.mark.parametrize("mode", ["deleted", "expired"])
def test_mixed_objects_restore_each_observation_query_without_primary_scope_or_filter_guessing(
        case, monkeypatch, primary, mode):
    deep = "private-provider-shows-deep"
    deeper = "private-provider-shows-deeper"
    case.provider.directories[SHOWS["file_id"]].append(
        GuangYaFile(deep, "Deep", True, parent_id=SHOWS["file_id"]))
    case.provider.directories[deep] = [GuangYaFile(deeper, "Deeper", True, parent_id=deep)]
    case.provider.directories[deeper] = [GuangYaFile(
        SUBTITLE, "Episode.srt", False, parent_id=deeper, size=10)]
    queries = [
        {"operation": "search", "path": FILMS["path"], "query": "Movie", "kinds": ["video"],
         "max_depth": 0, "max_items": 1},
        {"operation": "search", "path": SHOWS["path"], "query": "Episode", "kinds": ["subtitle"],
         "max_depth": 2, "max_items": 9},
    ]
    observations = [read(case, arguments) for arguments in queries]
    refs = [object_ref(observations[0], "Movie.mkv"), object_ref(observations[1], "Episode.srt")]
    operations = [{"op": "rename", "object_ref": refs[0], "new_name": "Movie-Renamed.mkv"},
                  {"op": "move", "object_ref": refs[1], "target_path": "/Destination"}]
    prepared, source, fingerprint = prepare(case, observations[primary], operations)
    plan = frozen_plan(case)
    assert {item["source"]["file_id"] for item in plan["operations"]} == {MOVIE, SUBTITLE}
    for observed in observations:
        invalidate(case, observed.data["observation_ref"], mode)
    changes.clear_guangya_fs_change_context(owner=case.context.owner)
    stored = restore_confirmation(case)
    assert stored.preview == prepared.to_dict() and stored.snapshot_fingerprint == fingerprint
    assert stored.metadata["reobservation_source"] == source
    source = stored.metadata["reobservation_source"]
    frozen = deepcopy(source)
    actual_queries = []
    real_create = files.create_directory_observation

    def traced_create(*args, **kwargs):
        actual = {key: deepcopy(kwargs[key]) for key in queries[0]}
        actual["kinds"] = list(actual["kinds"])
        actual_queries.append(actual)
        payload = real_create(*args, **kwargs)
        # 中间查询真实落盘并签名后立即过期；最终合并 OBS 不得再依赖其引用 TTL。
        payload["expires_at_epoch"] = time.time() - 1
        workspace._atomic_write(payload)
        return payload

    monkeypatch.setattr(files, "create_directory_observation", traced_create)
    case.provider.trace.clear()
    renewed = read(case, {"operation": "reobserve"}, source=source)
    assert sorted(actual_queries, key=lambda item: item["path"]) == queries
    mapping = renewed.data["reobservation"]
    assert mapping["basis"] == "provider_identity" and mapping["complete"] is True
    assert {item["previous_ref"] for item in mapping["items"]} == set(refs)
    assert len(mapping["items"]) == 2 and all(item["state"] == "same_identity" for item in mapping["items"])
    saved = workspace.load_directory_observation(renewed.data["observation_ref"], owner=case.context.owner)
    entries = workspace.observation_entry_map(saved)
    assert {entries[item["object_ref"]]["file_id"] for item in mapping["items"]} == {MOVIE, SUBTITLE}
    assert {item["object_name"] for item in renewed.data["entries"]} == {"Movie.mkv", "Episode.srt"}
    assert {entry["source_root"]["path"] for entry in entries.values()} == {FILMS["path"], SHOWS["path"]}
    scanned = [row["id"] for row in case.provider.trace if row["method"] == "list_dir"]
    assert set(scanned) == {"0", FILMS["parent_id"], FILMS["file_id"], SHOWS["parent_id"], SHOWS["file_id"], deep, deeper}
    assert NESTED not in scanned and DESTINATION not in scanned
    assert {path.stem for path in (case.root / "agent-guangya-workspace").glob("*.json")} == {saved["plan_id"]}
    assert source == frozen and frozen_plan(case) == plan
    assert_private(prepared, renewed)
    # 新 OBS 必须可直接用于真实混合准备；这里只读预检，不执行或自动确认。
    new_refs = {item["previous_ref"]: item["object_ref"] for item in mapping["items"]}
    fresh_prepared, _fresh_source, _fresh_fingerprint = prepare(
        case, renewed, [{**item, "object_ref": new_refs[item["object_ref"]]} for item in operations], source=source)
    assert_private(fresh_prepared)


@pytest.mark.parametrize("mode", ["deleted", "expired"])
def test_create_directory_only_retains_original_diagnostic_scope_without_object_refs_or_write(case, mode):
    observed = read(case, {"operation": "search", "path": FILMS["path"], "query": "Movie",
                          "kinds": ["video"], "max_depth": 0, "max_items": 7})
    prepared, source, fingerprint = prepare(case, observed, [{
        "op": "create_directory", "parent_path": SHOWS["path"], "name": "New Folder"}])
    plan = frozen_plan(case)
    assert all("object_ref" not in item for item in plan["operations"])
    invalidate(case, observed.data["observation_ref"], mode)
    changes.clear_guangya_fs_change_context(owner=case.context.owner)
    stored = restore_confirmation(case)
    assert stored.preview == prepared.to_dict() and stored.snapshot_fingerprint == fingerprint
    assert stored.metadata["reobservation_source"] == source
    source = stored.metadata["reobservation_source"]
    frozen = deepcopy(source)
    case.provider.trace.clear()
    renewed = read(case, {"operation": "reobserve"}, source=source)
    saved = workspace.load_directory_observation(renewed.data["observation_ref"], owner=case.context.owner)
    assert saved["roots"] == [FILMS]
    assert (saved["operation"], saved["query"], saved["kinds"], saved["max_depth"], saved["max_items"]) == (
        "search", "Movie", ["video"], 0, 7)
    assert renewed.data["cloud_write"] is False
    assert renewed.data["reobservation"]["items"] == []
    assert renewed.data["reobservation"]["complete"] is False, "范围诊断不能冒充目录已创建"
    assert [row["id"] for row in case.provider.trace if row["method"] == "list_dir"] == [
        "0", FILMS["parent_id"], FILMS["file_id"]]
    assert not any(item.name == "New Folder" for item in case.provider.directories[SHOWS["file_id"]])
    assert source == frozen and frozen_plan(case) == plan
    fresh_prepared, _fresh_source, _fresh_fingerprint = prepare(case, renewed, [{
        "op": "create_directory", "parent_path": SHOWS["path"], "name": "New Folder"}], source=source)
    note = fresh_prepared.data["observation_note"]
    assert "没有" in note and "选中对象" in note and "确认" in note
    assert "已按稳定标识核对" not in note
    assert note in fresh_prepared.data["effects"]
    assert_private(prepared, renewed, fresh_prepared)


def test_same_name_replacement_is_not_remapped_to_the_old_object_or_accepted_as_its_ref(case):
    observed = read(case)
    old_ref = object_ref(observed, "Nested")
    operation = {"op": "rename", "object_ref": old_ref, "new_name": "Nested-Renamed"}
    _prepared, source, _fingerprint = prepare(case, observed, [operation])
    frozen = deepcopy(source)
    invalidate(case, observed.data["observation_ref"], "deleted")
    replacement = case.provider.live_file(NESTED)
    replacement.file_id = "private-provider-replacement"
    case.provider.directories[replacement.file_id] = []
    renewed = read(case, {"operation": "reobserve"}, source=source)
    mapping = renewed.data["reobservation"]
    assert mapping["complete"] is False and len(mapping["items"]) == 1
    item = mapping["items"][0]
    assert item["previous_ref"] == old_ref and item["state"] == "same_name_other_identity"
    assert "object_ref" not in item
    assert object_ref(renewed, "Nested") != old_ref
    with pytest.raises(AgentToolError) as raised:
        prepare(case, renewed, [operation])
    assert raised.value.code in {"precondition_failed", "confirmation_stale"}
    with pytest.raises(AgentToolError) as raised:
        prepare(case, renewed, [{**operation, "object_ref": object_ref(renewed, "Nested")}], source=source)
    assert raised.value.code == "observation_selection_changed"
    assert source == frozen
    assert_private(renewed)


@pytest.mark.parametrize("boundary", ["root_replaced", "root_moved", "root_kind", "credential", "owner", "session"])
def test_root_identity_and_credential_or_session_changes_reject_before_any_scope_scan(case, boundary):
    observed = read(case)
    _prepared, source, _fingerprint = prepare(case, observed, [
        {"op": "rename", "object_ref": object_ref(observed, "Nested"), "new_name": "New-Nested"}])
    frozen = deepcopy(source)
    invalidate(case, observed.data["observation_ref"], "deleted")
    context = case.context
    root = case.provider.live_file(FILMS["file_id"])
    if boundary == "root_replaced":
        root.file_id = "private-provider-replaced-root"
        case.provider.directories[root.file_id] = []
    elif boundary == "root_moved":
        case.provider.directories[FILMS["parent_id"]].remove(root)
        root.parent_id = SHOWS["parent_id"]
        case.provider.directories[SHOWS["parent_id"]].append(root)
    elif boundary == "root_kind":
        root.is_dir = False
    elif boundary == "credential":
        case.provider.credential_generation += 1
    elif boundary == "owner":
        context = replace(context, owner="another-owner")
    else:
        context = replace(context, session_id="another-session")
    case.provider.trace.clear()
    with pytest.raises(AgentToolError) as raised:
        read(case, {"operation": "reobserve"}, source=source, context=context)
    assert raised.value.code == "observation_scope_changed"
    assert not any(row["method"] == "list_dir" for row in case.provider.trace)
    assert not list((case.root / "agent-guangya-workspace").glob("*.json"))
    assert source == frozen



@pytest.mark.parametrize("failed_root", ["replaced", "moved"])
def test_missing_one_root_retains_other_observation_but_reports_failed_selection_as_unknown(case, failed_root):
    observations = [read(case), read(case, {
        "operation": "search", "path": SHOWS["path"], "query": "Episode", "kinds": ["video"],
        "max_depth": 0, "max_items": 9})]
    refs = [object_ref(observations[0], "Nested"), object_ref(observations[1], "Episode.mkv")]
    _prepared, source, _fingerprint = prepare(case, observations[0], [
        {"op": "rename", "object_ref": refs[0], "new_name": "New-Nested"},
        {"op": "copy", "object_ref": refs[1], "target_path": "/Destination"}])
    frozen = deepcopy(source)
    for observation in observations:
        invalidate(case, observation.data["observation_ref"], "deleted")
    root = case.provider.live_file(SHOWS["file_id"])
    if failed_root == "replaced":
        root.file_id = "private-provider-replaced-shows-root"
        case.provider.directories[root.file_id] = []
    else:
        case.provider.directories[SHOWS["parent_id"]].remove(root)
        root.parent_id = DESTINATION
        case.provider.directories[DESTINATION].append(root)
    case.provider.trace.clear()
    renewed = read(case, {"operation": "reobserve"}, source=source)
    comparison = renewed.data["reobservation"]
    assert renewed.ok is True and comparison["complete"] is False
    assert len(comparison["failed_sources"]) == 1
    failure = comparison["failed_sources"][0]
    assert failure["observation_ref"] == observations[1].data["observation_ref"]
    assert failure["code"] == "observation_scope_changed"
    items = {item["previous_ref"]: item for item in comparison["items"]}
    assert items[refs[0]]["state"] == "same_identity" and items[refs[0]]["object_ref"] != refs[0]
    assert items[refs[1]]["state"] == "inconclusive" and "object_ref" not in items[refs[1]]
    saved = workspace.load_directory_observation(renewed.data["observation_ref"], owner=case.context.owner)
    assert saved["roots"] == [FILMS]
    scanned = [row["id"] for row in case.provider.trace if row["method"] == "list_dir"]
    assert set(scanned) == {"0", FILMS["parent_id"], FILMS["file_id"]}
    assert source == frozen
    assert_private(renewed)


@pytest.mark.parametrize("operation", ["rename", "copy"])
def test_hot_to_real_sqlite_cold_flow_preserves_confirmation_fingerprint_counts_and_source_note(case, operation):
    observed = read(case)
    change = {"op": operation, "object_ref": object_ref(observed, "Nested")}
    change.update({"new_name": "Nested-Renamed"} if operation == "rename"
                  else {"target_path": "/Destination"})
    _prepared, source, _fingerprint = prepare(case, observed, [change])
    renewed = read(case, {"operation": "reobserve"}, source=source)
    fresh_prepared, fresh_source, fresh_fingerprint = prepare(
        case, renewed, [{**change, "object_ref": object_ref(renewed, "Nested")}], source=source)
    plan = frozen_plan(case)
    assert fresh_prepared.data["observation_note"]
    # 单次准备已保存热 flow、SQLite flow 和确认卡；恢复不再调用 prepare。
    changes.configure_guangya_fs_change_context(None)
    hot = changes._flow(case.context.owner)
    hot_card = case.confirmation_plan
    assert hot is not None
    hot_fingerprint = changes._confirmation_fingerprint(hot, plan)
    changes.configure_guangya_fs_change_context(SQLiteAgentSessionContextRepository())
    changes.clear_guangya_fs_change_context(owner=case.context.owner)
    cold = changes._flow(case.context.owner)
    cold_card = restore_confirmation(case)
    assert cold is not None
    cold_fingerprint = changes._confirmation_fingerprint(cold, plan)
    assert hot_fingerprint == cold_fingerprint == fresh_fingerprint == cold_card.snapshot_fingerprint
    assert hot_card.metadata["reobservation_source"] == cold_card.metadata["reobservation_source"] == fresh_source
    assert hot.preview_safe == cold.preview_safe
    assert hot_card.preview == cold_card.preview == fresh_prepared.to_dict()
    assert hot_card.public_result == cold_card.public_result == fresh_prepared.to_dict()
    counters = ("total", "unchanged_count", "rename_count", "move_count", "copy_count",
                "relocate_count", "trash_count", "create_directory_count")
    for key in counters:
        assert hot.preview_safe[key] == cold.preview_safe[key] == fresh_prepared.data[key]
    assert cold.preview_safe[operation + "_count"] == 1
    note = fresh_prepared.data["observation_note"]
    assert hot.preview_safe["observation_note"] == cold.preview_safe["observation_note"] == note
    assert note in hot_card.preview["data"]["effects"] and note in cold_card.preview["data"]["effects"]
    assert frozen_plan(case) == plan
    assert_private(fresh_prepared)
    public = json.dumps([hot_card.public_dict(), cold_card.public_dict(), cold_card.public_result])
    assert "reobservation_source" not in public and "private-provider-" not in public

def test_credentials_changed_before_prepare_cannot_issue_a_source_from_old_observation(case):
    observed = read(case)
    arguments = changes.guangya_fs_change_arguments({
        "observation_ref": observed.data["observation_ref"], "trigger_strm": False,
        "operations": [{"op": "copy", "object_ref": object_ref(observed, "Nested"), "target_path": "/Destination"}],
    })
    case.provider.credential_generation += 1
    with pytest.raises(AgentToolError) as raised:
        changes.prepare_guangya_fs_change_confirmation(arguments, case.context)
    assert raised.value.code == "confirmation_stale"
    assert changes._flow(case.context.owner) is None
    assert not list((case.root / "agent-guangya-fs-change").glob("*.json"))


def test_native_pipeline_persists_private_source_in_real_sqlite_without_public_dto_leak(case):
    observed = read(case)
    arguments = changes.guangya_fs_change_arguments({
        "observation_ref": observed.data["observation_ref"], "trigger_strm": False,
        "operations": [{"op": "move", "object_ref": object_ref(observed, "Nested"), "target_path": "/Destination"}],
    })
    assert_private(observed)

    async def native_round_trip():
        specs = [adapt_tool_spec(spec) for spec in build_tool_specs() if spec.name in {FS_READ, FS_EXECUTE}]
        assert len(specs) == 2
        state = SQLiteKernelStore(secret_provider=lambda: "offline-fs-sqlite-secret")
        pipeline = ToolPipeline(catalog=ToolCatalog(specs), state_store=state, reference_store=state,
                                effect_store=ConfirmationEffectPlanStore(SQLiteConfirmationStore()))
        lease, _ = await state.begin_turn(owner=case.context.owner, session_id=case.context.session_id,
                                         request_id="offline-fs-prepare")

        async def progress(_event):
            pass

        context = ToolCallContext(owner=case.context.owner, session_id=case.context.session_id,
                                  request_id=lease.request_id, turn_id=lease.turn_id, lease=lease,
                                  cancellation=CancellationToken(), report_progress=progress)
        result = await pipeline.execute(FS_EXECUTE, arguments, context=context)
        assert result.effect_plan is not None
        stored = ConfirmationEffectPlanStore(SQLiteConfirmationStore()).get_active_plan(
            owner=case.context.owner, session_id=case.context.session_id, generation=lease.generation,
            plan_id=result.effect_plan.plan_id)
        assert stored is not None
        source = deepcopy(result.effect_plan.metadata["reobservation_source"])
        assert stored.metadata["reobservation_source"] == source == result.effect_plan.metadata["reobservation_source"]
        assert stored.arguments == arguments
        assert stored.preview["status"] == "confirmation_required"
        assert stored.preview["data"]["total"] == 1 and stored.preview["data"]["cloud_write"] is False
        public = json.dumps({"approval": stored.public_approval_dict(), "outcome": result.outcome.public_content})
        for private in ("reobservation_source", "credential_generation", "private-provider-", "source_root",
                        case.context.owner, case.context.session_id):
            assert private not in public
        invalidate(case, observed.data["observation_ref"], "deleted")
        renewed = await pipeline.execute(FS_READ, {"operation": "reobserve"}, context=replace(
            context, reobservation_source=stored.metadata["reobservation_source"]))
        assert renewed.outcome.public_content["data"]["reobservation"]["complete"] is True
        assert stored.metadata["reobservation_source"] == source

    asyncio.run(native_round_trip())


@pytest.fixture
def point_case(case, monkeypatch):
    """冻结真实 search 选择；同 hash 的未选中对象不能成为 ID 查找的替身。"""
    case.provider.live_file(MOVIE).etag = "offline-stable-content-hash"
    case.provider.directories[FILMS["file_id"]].append(GuangYaFile(
        "private-provider-unselected-same-hash", "Unselected.mkv", False,
        parent_id=FILMS["file_id"], size=100, etag="offline-stable-content-hash"))
    query = {"operation": "search", "path": FILMS["path"], "query": "Movie",
             "kinds": ["video"], "max_depth": 0, "max_items": 7}
    observed = read(case, query)
    old_ref = object_ref(observed, "Movie.mkv")
    _prepared, source, _fingerprint = prepare(
        case, observed, [{"op": "rename", "object_ref": old_ref, "new_name": "Planned.mkv"}])
    invalidate(case, observed.data["observation_ref"], "deleted")
    actual_queries = []
    real_create = files.create_directory_observation
    real_info = case.provider.file_info

    def traced_create(*args, **kwargs):
        actual = {key: deepcopy(kwargs[key]) for key in query}
        actual["kinds"] = list(actual["kinds"])
        actual_queries.append(actual)
        return real_create(*args, **kwargs)

    def bounded_info(file_id):
        # 本夹具父链至多两层；明确上限让错误的循环回溯快速失败而非挂住测试。
        assert sum(row["method"] == "file_info" for row in case.provider.trace) < 16, "point lookup 必须有界"
        return real_info(file_id)

    monkeypatch.setattr(files, "create_directory_observation", traced_create)
    monkeypatch.setattr(case.provider, "file_info", bounded_info)
    case.provider.trace.clear()
    return SimpleNamespace(**vars(case), query=query, actual_queries=actual_queries,
                           source=source, old_ref=old_ref)


def assert_point_private(*results):
    """changed_fields 可公开字段名，但实际 ID/路径字段和私有值仍不得进入任何 DTO。"""
    private_keys = {"reobservation_source", "source_root", "source_observations", "credential_generation",
                    "file_id", "parent_id", "parent_path", "owner_digest"}
    private_values = ("private-provider-", "offline-stable-content-hash", "offline-fs-source-owner",
                      "offline-fs-source-session", FILMS["path"], SHOWS["path"])
    for result in results:
        projected = DefaultProjector().project(result)
        pending = [result.to_dict(), result.to_model_dict(), projected.public_content,
                   json.loads(projected.model_content)]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                assert not private_keys.intersection(value), f"{OFFLINE}：公共 DTO 泄露私有字段"
                pending.extend(value)
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
            elif isinstance(value, str):
                assert not any(private in value for private in private_values), f"{OFFLINE}：公共 DTO 泄露私有值"


@pytest.mark.parametrize("change", ["renamed", "deeper_inside", "outside_root", "same_name_same_hash", "parent_cycle"])
def test_selected_id_point_reobservation_proves_original_root_without_widening_search(point_case, change):
    current = point_case
    source = deepcopy(current.source)
    target = current.provider.live_file(MOVIE)
    if change == "same_name_same_hash":
        target.file_id = "private-provider-same-name-same-hash-replacement"
    else:
        if change != "deeper_inside":
            target.name = "Changed.mkv"
        if change != "renamed":
            current.provider.directories[FILMS["file_id"]].remove(target)
            target.parent_id = DESTINATION if change == "outside_root" else NESTED
            current.provider.directories[target.parent_id].append(target)
        if change == "parent_cycle":
            current.provider.live_file(NESTED).parent_id = NESTED

    result = read(current, {"operation": "reobserve"}, source=current.source)
    assert current.actual_queries == [current.query], "原 query/kinds/depth/items 必须原样重查，不能用扩大搜索补漏"
    comparison = result.data["reobservation"]
    assert comparison["basis"] == "provider_identity" and len(comparison["items"]) == 1
    item = comparison["items"][0]
    assert item["previous_ref"] == current.old_ref
    saved = workspace.load_directory_observation(result.data["observation_ref"], owner=current.context.owner)
    entries = workspace.observation_entry_map(saved)
    if change in {"renamed", "deeper_inside"}:
        assert item["state"] == "same_identity", "原 query 未命中但 ID 仍在原根：必须做有界身份核验，不能只报 missing"
        assert comparison["complete"] is True and item["object_ref"] != current.old_ref
        selected = entries[item["object_ref"]]
        expected_name = "Movie.mkv" if change == "deeper_inside" else "Changed.mkv"
        assert (selected["file_id"], selected["name"], selected["etag"]) == (
            MOVIE, expected_name, "offline-stable-content-hash")
        assert selected["parent_path"] == FILMS["path"] + ("/Nested" if change == "deeper_inside" else "")
        assert ("parent_id" if change == "deeper_inside" else "name") in item["changed_fields"]
        assert selected["observed_by"] == item["observed_by"] == "identity"
        assert selected["source_root"] == FILMS, "身份补充只能归属已经核验的原根，不能扩展来源"
        assert selected["is_dir"] is False
        assert selected["parent_id"] == (NESTED if change == "deeper_inside" else FILMS["file_id"])
        assert {entry["file_id"] for entry in entries.values()} == {MOVIE}, "point 读取不得带出未选中对象"
        info_ids = {row["id"] for row in current.provider.trace if row["method"] == "file_info"}
        assert MOVIE in info_ids
        if change == "deeper_inside":
            assert NESTED in info_ids, "超原 depth 的目标必须回溯父链证明属于原根"
    else:
        assert comparison["complete"] is False and item["state"] != "same_identity"
        assert "object_ref" not in item, "范围外、同 hash 替代或循环父链都不能绑定原选择的新引用"
        assert MOVIE not in {entry["file_id"] for entry in entries.values()}
        if change == "same_name_same_hash":
            assert item["state"] == "same_name_other_identity"

    listed = {row["id"] for row in current.provider.trace if row["method"] == "list_dir"}
    assert listed <= {"0", FILMS["parent_id"], FILMS["file_id"], NESTED}
    assert DESTINATION not in listed and SHOWS["file_id"] not in listed, "禁止盲搜其它根或读取范围外目录"
    info_ids = [row["id"] for row in current.provider.trace if row["method"] == "file_info"]
    assert "private-provider-unselected-same-hash" not in info_ids
    assert len(info_ids) <= 16
    assert current.source == source
    assert_point_private(result)
    assert "offline-stable-content-hash" not in json.dumps(result.to_dict())

    if change in {"renamed", "deeper_inside"}:
        # 从最终签名 OBS 单次 prepare 获取下一轮来源；删除 OBS 后仍可冷恢复重查，
        # 不依赖按对象分裂出的 stat OBS 或第一次补充读取的内存 entry。
        fresh_prepared, fresh_source, fresh_fingerprint = prepare(current, result, [{
            "op": "rename", "object_ref": item["object_ref"], "new_name": "Planned.mkv"}], source=current.source)
        frozen_source = deepcopy(fresh_source)
        invalidate(current, result.data["observation_ref"], "deleted")
        changes.clear_guangya_fs_change_context(owner=current.context.owner)
        cold_card = restore_confirmation(current)
        assert cold_card.preview == fresh_prepared.to_dict()
        cold_source = cold_card.metadata["reobservation_source"]
        cold_fingerprint = cold_card.snapshot_fingerprint
        assert (cold_source, cold_fingerprint) == (fresh_source, fresh_fingerprint)
        current.provider.trace.clear()
        current.actual_queries.clear()
        continued = read(current, {"operation": "reobserve"}, source=cold_source)
        assert current.actual_queries == [current.query], "下一轮仍须保持原查询，而非改成全盘或放宽 depth"
        compared = continued.data["reobservation"]
        assert compared["complete"] is True and len(compared["items"]) == 1
        continued_item = compared["items"][0]
        assert continued_item["previous_ref"] == item["object_ref"]
        assert continued_item["object_ref"] != item["object_ref"]
        assert continued_item["state"] == "same_identity" and continued_item["observed_by"] == "identity"
        continued_saved = workspace.load_directory_observation(
            continued.data["observation_ref"], owner=current.context.owner)
        continued_entries = workspace.observation_entry_map(continued_saved)
        assert {entry["file_id"] for entry in continued_entries.values()} == {MOVIE}
        continued_selected = continued_entries[continued_item["object_ref"]]
        for key in ("file_id", "name", "is_dir", "etag", "parent_id", "parent_path", "source_root", "observed_by"):
            assert continued_selected[key] == selected[key]
        continued_info = [row["id"] for row in current.provider.trace if row["method"] == "file_info"]
        assert MOVIE in continued_info and len(continued_info) <= 16
        assert "private-provider-unselected-same-hash" not in continued_info
        if change == "deeper_inside":
            assert NESTED in continued_info
        assert {row["id"] for row in current.provider.trace if row["method"] == "list_dir"} <= {
            "0", FILMS["parent_id"], FILMS["file_id"], NESTED}
        assert cold_source == fresh_source == frozen_source and current.source == source
        assert_point_private(fresh_prepared, continued)


@pytest.mark.parametrize("interruption", ["credential", "cancelled"])
def test_selected_id_point_reobservation_stops_if_credentials_change_or_cancel_during_lookup(
        point_case, monkeypatch, interruption):
    current = point_case
    current.provider.live_file(MOVIE).name = "Changed.mkv"
    source = deepcopy(current.source)
    interrupted = False
    real_info = current.provider.file_info
    real_list = current.provider.list_dir

    def interrupt_point(file_id):
        nonlocal interrupted
        assert not interrupted, "point 读取失去凭据/取消后不得继续回溯父链"
        result = real_info(file_id)
        if str(file_id) == MOVIE:
            interrupted = True
            if interruption == "credential":
                current.provider.credential_generation += 1
        return result

    def guarded_list(parent_id="0"):
        assert not interrupted, "point 读取失去凭据/取消后不得继续 stat 或目录扫描"
        return real_list(parent_id)

    monkeypatch.setattr(current.provider, "file_info", interrupt_point)
    monkeypatch.setattr(current.provider, "list_dir", guarded_list)
    context = replace(current.context, cancelled=lambda: interrupted and interruption == "cancelled")
    with pytest.raises(AgentToolError) as raised:
        read(current, {"operation": "reobserve"}, source=current.source, context=context)
    assert interrupted, "应实际进入原选中稳定 ID 的 point 读取，而非伪造停止结果"
    assert raised.value.code == ("observation_scope_changed" if interruption == "credential" else "stopped")
    assert current.actual_queries == [current.query]
    assert not list((current.root / "agent-guangya-workspace").glob("*.json")), "中断后不得发布新 OBJ 或遗留中间观察"
    assert current.source == source
