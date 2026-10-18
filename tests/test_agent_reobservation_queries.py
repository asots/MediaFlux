"""#18 reobserve 查询：真实 prepare 来源/签名 OBS/SQLite 引用，Provider 明确离线。

不是生产实测。不伪造 source、成功回执、模型观察结果或 execute_confirmed 返回。
只替换 Provider IO；原生模型用 NativeScript/MockTransport，所有业务 handler 保持真实。
"""
from __future__ import annotations

import asyncio
import json
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import requests

from app.agent.confirmation import SQLiteConfirmationStore
from app.agent.domain_catalog import build_tool_specs
from app.agent.domain_catalog.catalog import DomainActions
from app.agent.domains.guangya import share_actions as shares
from app.agent.domains.guangya import workspace_actions as files
from app.agent.errors import AgentToolError, ToolPipelineError
from app.agent.kernel.adapters import TurnViewBuilder
from app.agent.kernel.capabilities import CapabilityRetriever, ToolCatalog
from app.agent.kernel.effects import ConfirmationEffectPlanStore
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports.existing_actions import adapt_tool_spec
from app.agent.kernel.ports.mediaflux_effects import MediaFluxEffectLifecycle
from app.agent.kernel.ports.mediaflux_policy import (
    MediaFluxAuthorizationPolicy,
    MediaFluxToolRateLimiter,
    MediaFluxTurnAdmission,
)
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state_types import AgentInput, CancellationToken
from app.agent.kernel.tool_calls import ToolCallContext
from app.agent.models import ToolContext
from app.clients.guangya import GuangYaFile
from app.modules import guangya_workspace as workspace
from tests.support import initialized_web_runtime
from tests.test_agent_task_first_slice import OWNER, NativeScript
from tests.test_guangya_observation_sources import FILMS, SHOWS, OfflineProvider

FS_READ = "guangya.fs.query"
CREATE = "guangya.share.create"
SHARE_READ = "guangya.share.list"
REVOKE = "guangya.share.revoke"
MOVIE = "private-provider-movie"
GOAL = "为 /Films/Shared/Movie.mkv 创建光鸭分享；引用失效后重新观察原对象，再给新确认卡，不要自动创建。"
OFFLINE = "离线 Provider 原生回归（不是生产实测）"
EVIDENCE = Path(__file__).resolve().parents[1] / ".superpowers/tmp/agent-reobservation-queries-20261012"


class QueryProvider(OfflineProvider):
    """复用真实 GuangYaFile DTO 树；改变的是离线 Provider 状态，不是 handler 回执。"""
    logged_in = True

    def __init__(self):
        super().__init__()
        self.trace = []
        self.writes = []
        self.raw_shares = [{"id": f"private-share-{letter}", "title": f"Share {letter}", "status": "active",
                            "updateTime": "t1", "fileCount": 1} for letter in "ABC"]

    def live_file(self, file_id):
        return next((item for rows in self.directories.values() for item in rows if item.file_id == file_id), None)

    def list_dir(self, parent_id="0"):
        self.trace.append({"method": "list_dir", "id": str(parent_id)})
        return super().list_dir(parent_id)

    def file_info(self, file_id):
        self.trace.append({"method": "file_info", "id": str(file_id)})
        return deepcopy(self.live_file(str(file_id)))

    def list_user_shares(self, *, max_items):
        self.trace.append({"method": "list_user_shares", "max_items": max_items})
        return deepcopy(self.raw_shares[:max_items])

    def create_user_share(self, *args, **kwargs):
        self.writes.append({"method": "create_user_share", "args": args, "kwargs": kwargs})
        raise AssertionError(OFFLINE + "：未获新确认不能执行 Provider 写入")

    def delete_user_shares(self, *args, **kwargs):
        self.writes.append({"method": "delete_user_shares", "args": args, "kwargs": kwargs})
        raise AssertionError(OFFLINE + "：本测试不授权撤销分享")

    def close(self):
        self.trace.append({"method": "close"})


class QueryCase:
    def __init__(self, root):
        self.root = root
        self.domain_actions = DomainActions()
        self.provider = QueryProvider()
        self.context = ToolContext(owner=OWNER, session_id="actual-reobservation-queries")
        self.artifacts = []
        self.pipeline = None
        self.kernel_context = None
        self.script = None
        self.events = []

    def record(self, label, value):
        self.artifacts.append({"label": label, "value": deepcopy(value)})
        return value

    def file_read(self, arguments=None, *, source=None, context=None):
        context = context or self.context
        if source is not None:
            context = replace(context, reobservation_source=source)
        arguments = files.guangya_fs_query_arguments(arguments or {"operation": "list", "path": "/Films/Shared"})
        result = self.domain_actions.workspace.query_guangya_filesystem(arguments, context)
        self.record("file_read", result.to_dict())
        return result

    def file_prepare(self, result, *, refs=None, source=None):
        if refs is None:
            refs = [next(item["object_ref"] for item in result.data["entries"] if item["object_name"] == "Movie.mkv")]
        arguments = shares.guangya_share_create_arguments({"observation_ref": result.data["observation_ref"],
                                                         "object_refs": refs, "title": "Offline new approval"})
        prepared, fingerprint = shares.prepare_create_guangya_share(
            arguments, replace(self.context, reobservation_source=source), workspace_actions=self.domain_actions.workspace)
        self.record("file_prepare", {"public": prepared.to_dict(), "source": prepared.effect_metadata["reobservation_source"],
                                     "fingerprint": fingerprint})
        assert prepared.ok is True and prepared.status == "confirmation_required"
        return prepared.effect_metadata["reobservation_source"], fingerprint

    def invalidate(self, observation_ref, mode):
        path = workspace._plan_path(workspace.observation_plan_id(observation_ref))
        if mode == "deleted":
            workspace.discard_observation(observation_ref)
            assert not path.exists()
        else:
            # 只构造真实签名 OBS 的过期条件；不修改 effect 回执或 prepare 来源。
            payload = workspace.load_directory_observation(observation_ref, owner=self.context.owner)
            payload["expires_at_epoch"] = time.time() - 1
            workspace._atomic_write(payload)
            assert path.exists()
        return path

    def build_pipeline(self):
        if self.pipeline is not None:
            return
        specs = [adapt_tool_spec(spec) for spec in build_tool_specs(domain_actions=self.domain_actions) if spec.name in {FS_READ, CREATE, SHARE_READ, REVOKE}]
        assert len(specs) == 4
        self.catalog = ToolCatalog(specs)
        self.store = SQLiteKernelStore(secret_provider=lambda: "offline-query-source-only")
        self.pipeline = ToolPipeline(catalog=self.catalog, state_store=self.store, reference_store=self.store,
            effect_store=ConfirmationEffectPlanStore(SQLiteConfirmationStore(ttl_seconds=300), record_actions=True),
            authorization=MediaFluxAuthorizationPolicy(), rate_limiter=MediaFluxToolRateLimiter(),
            effect_lifecycle=MediaFluxEffectLifecycle())

    async def call(self, tool, arguments, *, source=None):
        self.build_pipeline()
        if self.kernel_context is None:
            lease, _ = await self.store.begin_turn(owner=self.context.owner, session_id=self.context.session_id,
                                                 request_id="trusted-handler-test")

            async def progress(_payload):
                pass

            self.kernel_context = ToolCallContext(owner=self.context.owner, session_id=self.context.session_id,
                request_id=lease.request_id, turn_id=lease.turn_id, lease=lease,
                cancellation=CancellationToken(), report_progress=progress)
        result = await self.pipeline.execute(tool, arguments, context=replace(self.kernel_context, reobservation_source=source))
        self.record(tool, {"public": result.outcome.public_content,
                          "plan_id": result.effect_plan.plan_id if result.effect_plan else None,
                          "source": result.effect_plan.metadata.get("reobservation_source") if result.effect_plan else None})
        return result

    async def shares_source(self, indices):
        listed = await self.call(SHARE_READ, {"page": 1, "page_size": 3})
        old_ref = listed.outcome.public_content["reference_arguments"]["guangya_shares_ref"]
        preview = await self.call(REVOKE, {"guangya_shares_ref": old_ref, "indices": indices})
        source = preview.effect_plan.metadata["reobservation_source"]
        assert source["available"] is True
        return source, old_ref

    def dump(self, name):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        signed = [json.loads(path.read_text()) for path in (self.root / "agent-guangya-workspace").glob("*.json")]
        (EVIDENCE / (name.replace("/", "_") + ".json")).write_text(json.dumps({
            "scope": OFFLINE, "artifacts": self.artifacts, "provider_reads": self.provider.trace,
            "provider_writes": self.provider.writes, "signed_observations": signed,
            "native_wire": self.script.requests if self.script else [],
            "model_requests": self.script.model_requests if self.script else [],
            "events": [event.to_dict() for event in self.events],
        }, ensure_ascii=False, indent=2))


@pytest.fixture
def case(tmp_path, monkeypatch, request):
    def no_network(*_args, **_kwargs):
        raise AssertionError(OFFLINE + "：禁止真实网络")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    with initialized_web_runtime():
        monkeypatch.setattr(workspace, "PATHS", SimpleNamespace(data_dir=tmp_path))
        current = QueryCase(tmp_path)
        monkeypatch.setattr(files, "GuangYaClient", lambda: current.provider)
        monkeypatch.setattr(shares, "GuangYaClient", lambda: current.provider)
        try:
            yield current
        finally:
            current.dump(request.node.name)


def assert_private(result):
    text = json.dumps({"public": result.to_dict(), "model": result.to_model_dict()}, ensure_ascii=False)
    for private in ("private-provider-", "private-share-", "reobservation_source", "credential_generation"):
        assert private not in text, f"{OFFLINE}：私有来源不得泄露到模型或公开 DTO"


@pytest.mark.parametrize("mode", ["deleted", "expired"])
@pytest.mark.parametrize("operation", ["list", "stat"])
def test_actual_prepare_source_reobserves_renamed_identity_after_old_obs_is_gone(case, mode, operation):
    initial = case.file_read({"operation": operation,
                             "path": "/Films/Shared/Movie.mkv" if operation == "stat" else "/Films/Shared"})
    source, fingerprint = case.file_prepare(initial)
    frozen = deepcopy(source)
    old_ref = initial.data["observation_ref"]
    old_obj = source["baseline"]["selected"][0]["old_object_ref"]
    old_path = case.invalidate(old_ref, mode)
    case.provider.live_file(MOVIE).name = "Movie-Renamed.mkv"
    case.provider.trace.clear()
    result = case.file_read({"operation": "reobserve"}, source=source)
    assert result.ok is True and result.data["observation_ref"] != old_ref
    mapping = result.data["reobservation"]
    assert mapping["basis"] == "provider_identity" and mapping["complete"] is True
    assert len(mapping["items"]) == 1
    match = mapping["items"][0]
    assert match["previous_ref"] == old_obj and match["object_ref"] != old_obj
    assert match["state"] == "same_identity" and "name" in match["changed_fields"]
    saved = workspace.load_directory_observation(result.data["observation_ref"], owner=case.context.owner)
    assert workspace.observation_entry_map(saved)[match["object_ref"]]["file_id"] == MOVIE
    assert not old_path.exists(), f"{OFFLINE}：新观察不能复活旧 OBS"
    with pytest.raises(workspace.GuangYaWorkspaceError):
        workspace.load_directory_observation(old_ref, owner=case.context.owner)
    new_source, new_fingerprint = case.file_prepare(result, refs=[match["object_ref"]], source=source)
    assert new_fingerprint != fingerprint and new_source["baseline"]["observation_ref"] == result.data["observation_ref"]
    assert source == frozen and case.provider.writes == []
    assert_private(result)


def test_same_name_different_file_identity_is_not_mapped_or_accepted_by_create_prepare(case):
    initial = case.file_read()
    source, _ = case.file_prepare(initial)
    case.invalidate(initial.data["observation_ref"], "deleted")
    item = case.provider.live_file(MOVIE)
    item.file_id = "private-provider-replacement"
    result = case.file_read({"operation": "reobserve"}, source=source)
    comparison = result.data["reobservation"]
    assert comparison["complete"] is False
    row = comparison["items"][0]
    assert row["state"] == "same_name_other_identity" and "object_ref" not in row
    replacement_ref = next(entry["object_ref"] for entry in result.data["entries"] if entry["object_name"] == "Movie.mkv")
    with pytest.raises(AgentToolError) as raised:
        case.file_prepare(result, refs=[replacement_ref], source=source)
    case.record("rejected", {"code": raised.value.code, "message": str(raised.value)})
    assert raised.value.code == "observation_selection_changed"
    assert case.provider.writes == []
    assert_private(result)


@pytest.mark.parametrize("boundary", ["root_replaced", "root_moved", "generation", "owner", "session"])
def test_file_source_boundary_rejects_before_any_directory_scan(case, boundary):
    initial = case.file_read()
    source, _ = case.file_prepare(initial)
    case.invalidate(initial.data["observation_ref"], "deleted")
    context = case.context
    if boundary == "root_replaced":
        case.provider.live_file(FILMS["file_id"]).file_id = "private-provider-new-root"
    elif boundary == "root_moved":
        root = case.provider.directories[FILMS["parent_id"]].pop()
        root.parent_id = SHOWS["parent_id"]
        case.provider.directories[SHOWS["parent_id"]].append(root)
    elif boundary == "generation":
        case.provider.credential_generation += 1
    elif boundary == "owner":
        context = replace(context, owner="another-owner")
    else:
        context = replace(context, session_id="another-session")
    case.provider.trace.clear()
    with pytest.raises(AgentToolError) as raised:
        case.file_read({"operation": "reobserve"}, source=source, context=context)
    case.record("rejected", {"code": raised.value.code, "message": str(raised.value)})
    assert raised.value.code == "observation_scope_changed"
    assert not any(row["method"] == "list_dir" for row in case.provider.trace), f"{OFFLINE}：拒绝前不得开始扫描或回退全盘"
    assert not list((case.root / "agent-guangya-workspace").glob("*.json"))
    assert case.provider.writes == []


def test_root_resolution_race_does_not_scan_same_name_replacement(case):
    initial = case.file_read()
    source, _ = case.file_prepare(initial)
    case.invalidate(initial.data["observation_ref"], "deleted")
    # Provider 精确 ID 查询仍返回原对象，但路径目录项已换成新 ID：必须在根内容扫描前再次核对。
    old_root = case.provider.directories[FILMS["parent_id"]].pop()
    case.provider.directories["private-provider-detached"] = [old_root]
    case.provider.directories[FILMS["parent_id"]] = [GuangYaFile("private-provider-new-root", "Shared", True,
                                                                            parent_id=FILMS["parent_id"])]
    case.provider.directories["private-provider-new-root"] = []
    case.provider.trace.clear()
    with pytest.raises(AgentToolError) as raised:
        case.file_read({"operation": "reobserve"}, source=source)
    # fs.query 的现有公开错误映射把 GuangYaWorkspaceStale 映射为 precondition_failed。
    case.record("rejected", {"code": raised.value.code, "message": str(raised.value)})
    assert raised.value.code == "precondition_failed"
    assert "未读取同名替代目录" in str(raised.value)
    assert [row["id"] for row in case.provider.trace if row["method"] == "list_dir"] == ["0", FILMS["parent_id"]]
    assert case.provider.writes == []


def test_same_basename_roots_keep_query_filter_depth_and_count_bounds(case):
    case.provider.live_file("private-provider-episode").name = "Movie-Episode.mkv"
    case.provider.directories[FILMS["file_id"]].append(
        GuangYaFile("private-provider-other", "Other.mkv", False, parent_id=FILMS["file_id"]))
    initial = case.file_read({"operation": "search", "paths": ["/Films/Shared", "/Shows/Shared"],
                             "query": "Movie", "kinds": ["video"], "max_items": 2, "max_depth": 0})
    source, _ = case.file_prepare(initial, refs=[entry["object_ref"] for entry in initial.data["entries"]])
    frozen = deepcopy(source)
    case.invalidate(initial.data["observation_ref"], "deleted")
    case.provider.live_file(MOVIE).name = "Movie-Changed.mkv"
    case.provider.trace.clear()
    result = case.file_read({"operation": "reobserve"}, source=source)
    loaded = workspace.load_directory_observation(result.data["observation_ref"], owner=case.context.owner)
    assert [{k: root[k] for k in FILMS} for root in loaded["roots"]] == [FILMS, SHOWS]
    assert loaded["scope_names"] == ["目录 1 · Shared", "目录 2 · Shared"]
    assert (loaded["operation"], loaded["query"], loaded["kinds"], loaded["max_items"], loaded["max_depth"]) == (
        "search", "Movie", ["video"], 2, 0)
    assert len(loaded["entries"]) == 2 and all(entry["media_kind"] == "video" for entry in loaded["entries"])
    assert {entry["source_root"]["path"] for entry in loaded["entries"]} == {FILMS["path"], SHOWS["path"]}
    assert [row["id"] for row in case.provider.trace if row["method"] == "list_dir"] == [
        "0", FILMS["parent_id"], "0", SHOWS["parent_id"], FILMS["file_id"], SHOWS["file_id"]]
    assert result.data["reobservation"]["complete"] is True and source == frozen
    assert case.provider.writes == []
    assert_private(result)


@pytest.mark.parametrize("override", [{"path": "/"}, {"max_items": 2000}, {"max_depth": 12}, {"kinds": []}, {"query": ""}])
def test_reobserve_cannot_accept_model_scope_overrides(case, override):
    initial = case.file_read()
    source, _ = case.file_prepare(initial)
    case.provider.trace.clear()
    with pytest.raises(AgentToolError):
        case.file_read({"operation": "reobserve", **override}, source=source)
    assert case.provider.trace == [] and case.provider.writes == []


def test_share_reorder_maps_stable_ids_to_new_ref_and_only_original_selection(case):
    async def run():
        source, old_ref = await case.shares_source([2, 3])
        frozen = deepcopy(source)
        case.provider.raw_shares = [case.provider.raw_shares[index] for index in (2, 0, 1)]
        result = await case.call(SHARE_READ, {"operation": "reobserve"}, source=source)
        public = result.outcome.public_content
        new_ref = public["reference_arguments"]["guangya_shares_ref"]
        assert new_ref != old_ref
        mapping = public["data"]["reobservation"]
        assert mapping["complete"] is True and mapping["basis"] == "provider_identity"
        assert [(row["previous_index"], row["index"], row["state"]) for row in mapping["items"]] == [
            (2, 1, "same_identity"), (3, 2, "same_identity")]
        collection = await case.store.resolve(new_ref, owner=case.context.owner, session_id=case.context.session_id,
                                             expected_kind="guangya_shares")
        assert [row["share_id"] for row in collection["items"]] == ["private-share-B", "private-share-C"]
        preview = await case.call(REVOKE, {"guangya_shares_ref": new_ref, "indices": [1, 2]}, source=source)
        assert preview.effect_plan.plan_id and preview.outcome.public_content["ok"] is True
        assert [row["share_id"] for row in preview.effect_plan.metadata["reobservation_source"]["baseline"]["selected"]] == [
            "private-share-B", "private-share-C"]
        assert source == frozen and case.provider.writes == []
        assert "private-share-" not in result.outcome.model_content + json.dumps(public)
    asyncio.run(run())


@pytest.mark.parametrize("inventory", ["reobserved", "ordinary-new-list"])
def test_old_share_index_cannot_silently_select_another_identity(case, inventory):
    async def run():
        source, _ = await case.shares_source([2])
        case.provider.raw_shares = [case.provider.raw_shares[index] for index in (2, 0, 1)]
        result = await case.call(SHARE_READ, {"operation": "reobserve"} if inventory == "reobserved" else {"page": 1, "page_size": 3}, source=source)
        new_ref = result.outcome.public_content["reference_arguments"]["guangya_shares_ref"]
        with pytest.raises(ToolPipelineError) as raised:
            await case.call(REVOKE, {"guangya_shares_ref": new_ref, "indices": [2]}, source=source)
        case.record("rejected", {"code": raised.value.code, "message": str(raised.value)})
        assert raised.value.code == ("confirmation_stale" if inventory == "reobserved" else "observation_selection_changed")
        assert case.provider.writes == []
    asyncio.run(run())


def test_missing_share_is_not_replaced_by_same_title_and_fresh_targets_stay_selected(case):
    async def run():
        source, _ = await case.shares_source([2, 3])
        case.provider.raw_shares = [case.provider.raw_shares[2],
            {**case.provider.raw_shares[1], "id": "private-share-replacement"}, case.provider.raw_shares[0]]
        result = await case.call(SHARE_READ, {"operation": "reobserve"}, source=source)
        public = result.outcome.public_content
        rows = public["data"]["reobservation"]["items"]
        assert rows[0]["previous_index"] == 2 and rows[0]["state"] == "not_observed" and "index" not in rows[0]
        assert rows[1]["previous_index"] == 3 and rows[1]["state"] == "same_identity" and rows[1]["index"] == 1
        assert public["data"]["reobservation"]["complete"] is False and public["data"]["total"] == 1
        new_ref = public["reference_arguments"]["guangya_shares_ref"]
        collection = await case.store.resolve(new_ref, owner=case.context.owner, session_id=case.context.session_id,
                                             expected_kind="guangya_shares")
        assert [row["share_id"] for row in collection["items"]] == ["private-share-C"]
        preview = await case.call(REVOKE, {"guangya_shares_ref": new_ref, "indices": [1]}, source=source)
        assert preview.effect_plan.metadata["reobservation_source"]["baseline"]["selected"][0]["share_id"] == "private-share-C"
        assert case.provider.writes == []
    asyncio.run(run())


@pytest.mark.parametrize("boundary", ["generation", "owner", "session"])
def test_share_source_boundary_rejects_before_listing(case, boundary):
    async def run():
        source, _ = await case.shares_source([2])
        context = replace(case.context, reobservation_source=source)
        if boundary == "generation":
            case.provider.credential_generation += 1
        elif boundary == "owner":
            context = replace(context, owner="another-owner")
        else:
            context = replace(context, session_id="another-session")
        case.provider.trace.clear()
        with pytest.raises(AgentToolError) as raised:
            shares.list_guangya_user_shares(shares.guangya_share_list_arguments({"operation": "reobserve"}), context)
        case.record("rejected", {"code": raised.value.code, "message": str(raised.value)})
        assert raised.value.code == "observation_scope_changed"
        assert not any(row["method"] == "list_user_shares" for row in case.provider.trace)
        assert case.provider.writes == []
    asyncio.run(run())


class ReobserveNativeScript(NativeScript):
    def __init__(self):
        super().__init__("fixture")
        self.steps = ["query", "preview", "new_preview"]

    async def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        assert self.steps, OFFLINE + "：发生额外模型续行"
        step = self.steps.pop(0)
        if step in {"query", "reobserve"}:
            name = FS_READ
            arguments = {"operation": "list", "path": "/Films/Shared"} if step == "query" else {"operation": "reobserve"}
        else:
            observations = [json.loads(row["content"])["data"] for row in body["messages"]
                if row.get("role") == "tool" and json.loads(row["content"]).get("data", {}).get("observation_ref")]
            latest = observations[-1]
            obj = (latest["reobservation"]["items"][0]["object_ref"] if step == "new_preview" else
                   next(entry["ref"] for entry in latest["entries"] if entry["name"] == "Movie.mkv"))
            name, arguments = CREATE, {"observation_ref": latest["observation_ref"], "object_refs": [obj]}
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "content": "", "tool_calls": [{"id": "native-" + step, "type": "function",
                "function": {"name": name.replace(".", "__"), "arguments": json.dumps(arguments)}}],
        }}]})


@pytest.mark.parametrize("retrieved_source", [True, False])
def test_native_failed_confirmation_passes_actual_private_prepare_source_to_reobserve_and_new_card(case, retrieved_source):
    async def run():
        case.build_pipeline()
        case.script = ReobserveNativeScript()
        session = AgentSession(model=case.script.adapter, catalog=case.catalog,
            retriever=CapabilityRetriever(minimum=4, maximum=4), pipeline=case.pipeline,
            state_store=case.store, journal=case.store, turn_admission=MediaFluxTurnAdmission())
        first = [event async for event in session.run(AgentInput(owner=case.context.owner,
            session_id=case.context.session_id, message=GOAL, request_id="original-user"))]
        case.events.extend(first)
        assert first[-1].payload["status"] == "approval_required"
        initial = await case.store.load(owner=case.context.owner, session_id=case.context.session_id)
        source_users = [deepcopy(row) for row in initial.conversation if row.get("role") == "user"]
        assert len(source_users) == 1 and source_users[0]["history_id"]
        assert source_users == [{"role": "user", "content": GOAL, "history_id": source_users[0]["history_id"]}]
        old = await case.pipeline.get_pending_plan(owner=case.context.owner, session_id=case.context.session_id)
        source = deepcopy(old.metadata["reobservation_source"])
        case.record("native_original_source", source)
        original_ref = source["baseline"]["observation_ref"]
        case.invalidate(original_ref, "expired")
        case.provider.live_file(MOVIE).name = "Movie-Renamed.mkv"
        if not retrieved_source:
            retrieve = session.retriever.retrieve
            def without_source(*args, **kwargs):
                selected = retrieve(*args, **kwargs)
                return replace(selected, tools=tuple(tool for tool in selected.tools if tool.name != FS_READ))
            session.retriever = SimpleNamespace(maximum=4, retrieve=without_source)
        events = [event async for event in session.confirm(owner=case.context.owner,
            session_id=case.context.session_id, plan_id=old.plan_id, request_id="confirm-old")]
        case.events.extend(events)
        assert events[-1].payload["status"] == "approval_required"
        failed = [event for event in events if event.type is AgentEventType.EFFECT_FAILED]
        assert len(failed) == 1 and failed[0].payload["result"]["status"] == "confirmation_stale"
        observed = [event for event in events if event.type is AgentEventType.TOOL_COMPLETED and event.payload.get("tool") == FS_READ]
        assert len(observed) == 1 and observed[0].payload["result"]["data"]["reobservation"]["complete"] is True
        assert len({event.turn_id for event in events}) == 1
        new = await case.pipeline.get_pending_plan(owner=case.context.owner, session_id=case.context.session_id)
        assert new.plan_id != old.plan_id and new.snapshot_fingerprint != old.snapshot_fingerprint
        assert new.arguments["observation_ref"] != original_ref
        assert new.metadata["reobservation_source"]["baseline"]["selected"][0]["file_id"] == MOVIE
        assert not workspace._plan_path(workspace.observation_plan_id(original_ref)).exists()
        cold = SQLiteKernelStore(secret_provider=lambda: "offline-query-source-only")
        state = await cold.load(owner=case.context.owner, session_id=case.context.session_id)
        assert [row for row in state.conversation if row.get("role") == "user"] == source_users
        assert len([row for row in state.conversation if row.get("completion_receipt_id") == old.plan_id]) == 1
        view = TurnViewBuilder()
        for event in events:
            view.apply(event)
        assert view.build().to_dict()["effect_result"] == failed[0].payload["result"]
        assert len(case.script.requests) == len(case.script.model_requests) == 3 and not case.script.steps
        resumed_wire = case.script.requests[2]
        assert FS_READ.replace(".", "__") in {tool["function"]["name"] for tool in resumed_wire["tools"]}
        assert "只传 operation=reobserve" not in resumed_wire["messages"][0]["content"]
        assert events.index(observed[0]) < next(i for i, event in enumerate(events)
                                              if event.type is AgentEventType.MODEL_STARTED)
        assert any(json.loads(row["content"]).get("data", {}).get("reobservation", {}).get("complete") is True
                   for row in resumed_wire["messages"] if row.get("role") == "tool")
        case.script.assert_user_inputs([GOAL])
        serialized = json.dumps(case.script.requests + [event.to_dict() for event in case.events])
        assert "private-provider-" not in serialized and "reobservation_source" not in serialized
        assert case.provider.writes == []
        assert len([row for row in case.provider.trace if row["method"] == "file_info"]) == 3
        assert len([event for event in case.events if event.type is AgentEventType.EFFECT_APPROVAL_REQUIRED]) == 2
    asyncio.run(run())


@pytest.mark.parametrize("change", ["moved", "root_replaced"])
def test_ordinary_new_scope_is_not_reported_as_original_scope_reobservation(case, change):
    old = case.file_read()
    source, _ = case.file_prepare(old)
    case.invalidate(old.data["observation_ref"], "deleted")
    if change == "moved":
        movie = case.provider.live_file(MOVIE)
        case.provider.directories[FILMS["file_id"]].remove(movie)
        movie.parent_id = SHOWS["file_id"]
        case.provider.directories[SHOWS["file_id"]].append(movie)
        path = SHOWS["path"]
        reobserved = case.file_read({"operation": "reobserve"}, source=source)
        assert reobserved.data["reobservation"]["items"][0]["state"] == "missing"
    else:
        new_root = "private-provider-new-root"
        case.provider.live_file(FILMS["file_id"]).file_id = new_root
        contents = case.provider.directories.pop(FILMS["file_id"])
        for item in contents:
            item.parent_id = new_root
        case.provider.directories[new_root] = contents
        path = FILMS["path"]
        with pytest.raises(AgentToolError) as caught:
            case.file_read({"operation": "reobserve"}, source=source)
        assert caught.value.code == "observation_scope_changed"
    # A separate explicit READ can supply a new scope, but it must not claim old-scope recovery.
    fresh = case.file_read({"operation": "list", "path": path}, source=source)
    obj = next(item["object_ref"] for item in fresh.data["entries"] if item["object_name"] == "Movie.mkv")
    args = shares.guangya_share_create_arguments({"observation_ref": fresh.data["observation_ref"], "object_refs": [obj]})
    prepared, _ = shares.prepare_create_guangya_share(args, replace(case.context, reobservation_source=source), workspace_actions=case.domain_actions.workspace)
    assert prepared.status == "confirmation_required"
    effects = "\n".join(prepared.data["effects"])
    assert "观察范围已变化" in effects and "新确认" in effects
    assert "已按稳定标识核对原选中对象" not in effects
    assert prepared.effect_metadata["reobservation_source"]["baseline"]["roots"] != source["baseline"]["roots"]
    assert not case.provider.writes
