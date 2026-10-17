"""候选业务容量独立于模型/展示预算，搜索到确认与后置生命周期不丢尾项。"""
from __future__ import annotations

import asyncio
import json
import shutil
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.agent.domain_catalog import build_tool_specs
from app.agent.errors import AgentToolError
from app.agent.indexer_candidate_actions import (
    present_candidates,
    present_candidates_arguments,
)
from app.agent.ingest_actions import (
    AgentIngestSessionStore,
    IngestActions,
    ingest_inspect_arguments,
    ingest_submit_arguments,
)
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.pipeline import (
    ToolCallContext,
    ToolPipelineError,
    _validate_json_schema,
)
from app.agent.kernel.projection import DefaultProjector
from app.agent.kernel.state import AgentInput, CancellationToken
from app.agent.kernel.transports import QueryEnvelope
from app.agent.kernel.ux_selection import current_candidate_view, normalize_selection
from app.agent.missing_media_workflow_runtime import MissingMediaWorkflowRuntime
from app.agent.models import ToolContext, ToolReference, ToolResult
from app.agent.public_view import candidate_result_lines
from app.agent.recent_resource_candidates import (
    RecentResourceCandidateStore,
    attach_resource_candidate_reference,
    new_resource_search_id,
    restore_resource_candidate_reference,
    safe_resource_snapshot,
    validate_safe_resource_snapshot,
)
from app.indexers.models import IndexerItem
from app.indexers.result_store import IndexerResultStore
from tests import test_agent_ux_backend as ux
from tests.test_agent_recent_resource_confirmation import _candidate, _single_result
from tests.test_agent_resource_batch import _prepare_batch


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("候选完整性回归不允许访问网络")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def store(tmp_path):
    yield from ux.store.__wrapped__(tmp_path)


def _resources(count):
    result = ToolResult(True, "success", f"找到 {count} 项资源", data={"items": [
        {"result_id": f"capacity-resource-{position:03d}", "title": f"Example {position:03d}",
         "site_id": "demo", "site_name": "Demo", "size_text": "1 GiB",
         "download_state": "ready", "download_kinds": ["magnet"]}
        for position in range(1, count + 1)
    ]})
    result.references.append(ToolReference("resource_candidates", safe_resource_snapshot(
        result, search_id=new_resource_search_id(),
    )))
    return result


@pytest.mark.parametrize("count", [20, 50])
def test_full_business_snapshot_survives_projection_and_validation(count):
    result = _resources(count)
    snapshot = result.references[0].value
    assert len(snapshot["candidates"]) == count
    assert [item["position"] for item in snapshot["candidates"]] == list(range(1, count + 1))
    assert validate_safe_resource_snapshot(snapshot) == snapshot
    projected = DefaultProjector(max_model_chars=2000).project(result)
    assert len(projected.model_content) <= 2000
    assert projected.refs[0].value == snapshot
    assert len(result.data["items"]) == count


@pytest.mark.parametrize("count", [20, 50])
def test_episodic_snapshot_keeps_all_eligible_candidates(count):
    result = _single_result(*[_candidate(f"capacity-episode-{position:03d}", rank=position)
                              for position in range(1, count + 1)])
    attach_resource_candidate_reference(result, result_store=IndexerResultStore())
    snapshot = restore_resource_candidate_reference(result.references[0].value)
    assert len(snapshot["candidates"]) == count
    assert snapshot["candidates"][-1]["position"] == count
    assert snapshot["candidates"][-1]["_verification_context"]["episode"] == 3


@pytest.mark.parametrize("count", [20, 50])
def test_private_snapshot_restores_every_handle_after_restart_and_rejects_tampered_tail(store, count):
    original = IndexerResultStore()
    result = _resources(count)
    for position, public in enumerate(result.data["items"], 1):
        public["result_id"] = original.put(IndexerItem(
            site_id="demo", site_name="Demo", title=public["title"],
            download_state="ready", download_kinds=("magnet",),
            magnet=f"magnet:?xt=urn:btih:{position:040x}",
        ))
    attach_resource_candidate_reference(result, result_store=original, query_arguments={"title": "Example", "resolution": "2160p"})
    private = result.references[0].value
    ids = [item["result_id"] for item in result.data["items"]]
    assert [item["result_id"] for item in private["_private_items"]] == ids

    async def reload_reference():
        reference = await store.put(owner=ux.OWNER, session_id=ux.SESSION, kind="resource_candidates", value=private, ttl_seconds=900)
        reloaded = SQLiteKernelStore(secret_provider=lambda: ux.SECRET)
        return await reloaded.resolve(reference.ref, owner=ux.OWNER, session_id=ux.SESSION, expected_kind="resource_candidates")

    private = asyncio.run(reload_reference())
    assert private["_query_arguments"] == {"title": "Example", "resolution": "2160p"}
    restarted = IndexerResultStore()
    with patch("app.indexers.runtime.get_indexer_service", return_value=SimpleNamespace(result_store=restarted)):
        restored = restore_resource_candidate_reference(private)
        assert [item["result_id"] for item in restored["candidates"]] == ids
        assert [restarted.get(result_id).title for result_id in ids] == [item["title"] for item in result.data["items"]]
        private["_private_items"][-1]["result_id"] = "tampered-resource-tail"
        assert restore_resource_candidate_reference(private) is None


def test_schema_and_validators_share_resource_capacity_without_reducing_share_capacity():
    specs = {spec.name: spec for spec in build_tool_specs(RecentResourceCandidateStore(), AgentIngestSessionStore())}
    reference = "ref_" + "a" * 24
    positions = list(range(1, 51))
    arguments = {
        "indexer.present_candidates": {"resource_candidates_ref": reference, "positions": positions},
        "ingest.inspect": {"source_type": "resource_candidates", "resource_candidates_ref": reference, "positions": positions},
        "ingest.submit": {"source_type": "resource_candidates", "resource_candidates_ref": reference, "positions": positions, "target": "guangya"},
    }
    assert specs["indexer.search_resources"].parameters["properties"]["limit"]["maximum"] == 50
    for name, valid in arguments.items():
        spec = specs[name]
        _validate_json_schema(valid, spec.parameters)
        assert spec.validator(valid)["positions"] == positions
        invalid = {**valid, "positions": [51]}
        with pytest.raises(ToolPipelineError):
            _validate_json_schema(invalid, spec.parameters)
        with pytest.raises(AgentToolError):
            spec.validator(invalid)
    share = {"source_type": "guangya_share", "ingest_snapshot_ref": reference, "positions": [200]}
    _validate_json_schema(share, specs["ingest.submit"].parameters)
    assert ingest_submit_arguments(share)["positions"] == [200]
    selection = {"ref": reference, "positions": positions[::-1], "target": "guangya"}
    assert normalize_selection(selection)["positions"] == positions


def test_displaying_only_tail_does_not_shrink_or_restore_business_reference():
    value = _resources(20).references[0].value
    arguments = present_candidates_arguments({"resource_candidates_ref": "ref_" + "a" * 24, "positions": [20, 13]})
    arguments["resource_candidates"] = value
    with patch("app.indexers.runtime.get_indexer_service", side_effect=AssertionError("展示不能恢复 Provider")):
        presented = present_candidates(arguments)
    assert presented.data["positions"] == [20, 13]
    assert presented.references[0].value == value
    assert len(presented.references[0].value["candidates"]) == 20


def test_low_model_budget_still_persists_full_reference_and_presents_tail(store):
    async def exercise():
        with patch.object(ux, "_resources", side_effect=lambda **_: _resources(50)):
            _, pipeline, states = ux._runtime(store)
            pipeline.projector = DefaultProjector(max_model_chars=2000)
            lease, _ = await states.begin_turn(owner=ux.OWNER, session_id=ux.SESSION, request_id="budget")
            context = ToolCallContext(owner=ux.OWNER, session_id=ux.SESSION, request_id="budget", turn_id=lease.turn_id,
                                      lease=lease, cancellation=CancellationToken(), report_progress=AsyncMock())
            searched = await pipeline.execute("indexer.search_resources", {"title": "Example"}, context=context)
        assert len(searched.outcome.model_content) <= 2000
        arguments = searched.outcome.public_content["reference_arguments"]
        private = await store.resolve(arguments["resource_candidates_ref"], owner=ux.OWNER, session_id=ux.SESSION, expected_kind="resource_candidates")
        assert len(private["candidates"]) == 50
        presented = await pipeline.execute("indexer.present_candidates", {**arguments, "positions": [50, 13]}, context=context)
        assert [item["position"] for item in presented.outcome.public_content["candidate_view"]["items"]] == [13, 50]
        reference = presented.outcome.public_content["reference_arguments"]["resource_candidates_ref"]
        assert await store.resolve(reference, owner=ux.OWNER, session_id=ux.SESSION, expected_kind="resource_candidates") == private
    asyncio.run(exercise())


def test_inspect_reads_entire_snapshot_or_tail_in_original_request_order():
    resources = RecentResourceCandidateStore()
    search_id = resources.capture(owner=ux.OWNER, result=_resources(20))
    actions = IngestActions(store=AgentIngestSessionStore(), recent_resource_store=resources)
    for positions in (None, [20, 13, 20]):
        arguments = {"source_type": "resource_candidates"}
        if positions is not None:
            arguments["positions"] = positions
        result = actions.inspect(ingest_inspect_arguments(arguments), ToolContext(owner=ux.OWNER))
        expected = list(range(1, 21)) if positions is None else [20, 13]
        assert result.data["search_id"] == search_id
        assert result.data["count"] == len(expected)
        assert result.data["total"] == 20
        assert [item["position"] for item in result.data["items"]] == expected


@pytest.mark.parametrize("natural", [False, True])
@pytest.mark.parametrize("count", [20, 50])
def test_full_batch_is_frozen_in_one_plan_and_only_dispatched_once_after_confirmation(store, natural, count):
    class Model:
        def __init__(self):
            self.intent = {}
            self.calls = 0

        async def stream(self, request, *, cancellation):
            self.calls += 1
            if any("已确认操作的可信系统结果" in row.content for row in request.messages):
                yield ModelEvent(ModelEventType.TEXT_DELTA, text="整批已处理。")
                yield ModelEvent(ModelEventType.FINISH, finish_reason="stop")
                return
            yield ModelEvent(ModelEventType.TOOL_CALL_COMPLETED, tool_call=ModelToolCall("batch", "ingest.submit", self.intent))
            yield ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")

    async def exercise():
        model = Model()
        with patch.object(ux, "_resources", side_effect=lambda **_: _resources(count)):
            session, pipeline, states = ux._runtime(store, model=model)
            view, context = await ux._publish_candidates(pipeline, states)
        positions = list(range(1, count + 1))
        assert [item["position"] for item in view["items"]] == positions
        assert await current_candidate_view(state=await states.load(owner=ux.OWNER, session_id=ux.SESSION), store=store) == view
        # 新搜索不能替换已持有引用的业务身份；仍按旧快照整批预检与确认。
        if natural:
            with patch.object(ux, "_resources", side_effect=lambda **_: _resources(1)):
                await pipeline.execute("indexer.search_resources", {"title": "Other"}, context=context)
        model.intent = {"source_type": "resource_candidates", "resource_candidates_ref": view["ref"], "positions": positions, "target": "guangya"}
        message = f"把这 {count} 项全部推送光鸭"
        user = AgentInput(owner=ux.OWNER, session_id=ux.SESSION, message=message) if natural else QueryEnvelope(
            owner=ux.OWNER, session_id=ux.SESSION, message=message, selection=ux._selection(view, positions),
        ).to_agent_input()
        result = ToolResult(True, "accepted", "整批处理完毕", data={"target": "guangya", "items": [
            {"status": "submitted", "request_id": position, "succeeded": ["guangya"], "failed": []}
            for position in positions
        ]})
        with patch("app.agent.indexer_candidate_actions.prepare_submit_resource_batch", side_effect=_prepare_batch) as prepare, \
             patch("app.agent.indexer_candidate_actions.submit_resource_batch_confirmed", return_value=result) as execute:
            events = await ux._events(session.run(user))
            approvals = [event for event in events if event.type is AgentEventType.EFFECT_APPROVAL_REQUIRED]
            assert len(approvals) == 1
            assert model.calls == int(natural)
            ids = [f"capacity-resource-{position:03d}" for position in positions]
            assert prepare.call_args.args[0]["result_ids"] == ids
            assert approvals[0].payload["plan"]["preview"]["data"]["count"] == count
            execute.assert_not_called()
            plan_id = approvals[0].payload["plan"]["plan_id"]
            confirmed = await ux._events(session.confirm(owner=ux.OWNER, session_id=ux.SESSION, plan_id=plan_id))
            completed = next(event.payload["result"] for event in confirmed if event.type is AgentEventType.EFFECT_COMPLETED)
            assert execute.call_args.args[0]["result_ids"] == ids
            assert [item["position"] for item in completed["data"]["items"]] == positions
            await ux._events(session.confirm(owner=ux.OWNER, session_id=ux.SESSION, plan_id=plan_id))
            execute.assert_called_once()
    asyncio.run(exercise())


def test_lifecycle_retains_every_slot_and_captures_tail_results_without_shifting_verification():
    repository, recent, enqueue = Mock(), Mock(), Mock(return_value=True)
    repository.select_candidate.return_value = None
    runtime = MissingMediaWorkflowRuntime(repository=repository, recent_download_store=recent, verification_enqueuer=enqueue)
    verification = {"title": "Example", "tmdb_id": "12345", "season": 1, "episode": 20, "as_of": "2026-10-15"}
    candidates = [{"verification": None} for _ in range(19)] + [{"verification": verification, "candidate_title": "Tail", "target": "guangya"}]
    followups = runtime.stage_candidates(owner=ux.OWNER, candidates=candidates)
    assert len(followups) == 20
    assert followups[:19] == (None,) * 19
    assert followups[19] is not None
    items = [{"position": pos, "title": f"Example {pos}", "target": "guangya", "status": "submitted", "request_id": pos} for pos in range(1, 21)]
    runtime.complete_submission(owner=ux.OWNER, tool_name="ingest.submit", result=ToolResult(True, "accepted", "done", data={"source_type": "resource_candidates", "items": items}), followups=followups)
    assert recent.capture.call_count == 20
    assert [call.kwargs["result"].data["request_id"] for call in recent.capture.call_args_list] == list(range(1, 21))
    enqueue.assert_called_once()
    assert enqueue.call_args.args[0].data["request_id"] == 20
    assert len(candidate_result_lines({"source_type": "resource_candidates", "items": items})) == 20


def test_lifecycle_failure_returns_full_aligned_empty_slots():
    repository = Mock()
    repository.select_candidate.side_effect = RuntimeError("isolated repository failure")
    runtime = MissingMediaWorkflowRuntime(repository=repository, recent_download_store=Mock(), verification_enqueuer=Mock())
    candidates = [{} for _ in range(49)] + [{"verification": {"title": "Example"}}]
    assert runtime.stage_candidates(owner=ux.OWNER, candidates=candidates) == (None,) * 50
    repository.select_candidate.assert_called_once()


def test_multi_work_search_keeps_all_thirty_six_candidates_and_private_handles():
    from app.agent.episode_resource_actions import (
        missing_season_resource_arguments,
        search_missing_season_resources,
    )
    from tests.test_agent_missing_episode_resources import _audit_result, _search_result

    names = [f"CapacityShow{index:02d}" for index in range(12)]
    provider = SimpleNamespace(result_store=IndexerResultStore())
    arguments = missing_season_resource_arguments({"items": [
        {"query": name, "season": 1, "episodes": [1, 2, 3]} for name in names
    ], "as_of": "2026-10-15"})

    def audit(arguments, **_kwargs):
        result = _audit_result(missing=[{"season": 1, "episode": episode} for episode in (1, 2, 3)])
        result.data.update(title=arguments["query"], tmdb_id=str(1000 + names.index(arguments["query"])))
        return result

    def search(arguments, **_kwargs):
        name = arguments["title"]
        assert name in names
        assert arguments["season"] == 1 and arguments["episode"] is None
        assert arguments["limit"] == 50
        # 单季共享查询返回完整候选池，不能继续按旧逐集协议格式化 episode=None。
        titles = [f"{name}.S01E{episode:02d}.1080p.WEB-DL" for episode in (1, 2, 3)]
        titles += ["UnrelatedShow.S01E01.1080p.WEB-DL",
                   f"{name}.S02E01.1080p.WEB-DL", f"{name}.S01E04.1080p.WEB-DL"]
        items = []
        for offset, title in enumerate(titles, 1):
            item = IndexerItem(
                "demo", "Demo", title, download_state="ready", download_kinds=("magnet",),
                magnet=f"magnet:?xt=urn:btih:{names.index(name) * len(titles) + offset:040x}",
            )
            items.append({**item.to_public_dict(), "result_id": provider.result_store.put(item)})
        return _search_result(query=f"{name} S01", items=items)

    with patch("app.agent.episode_resource_actions.audit_series_episodes", side_effect=audit) as audits, \
         patch("app.agent.episode_resource_actions.search_resources", side_effect=search) as searches, \
         patch("app.agent.episode_resource_actions.get_indexer_service", return_value=provider):
        result = search_missing_season_resources(arguments)
    assert audits.call_count == searches.call_count == len(names)
    assert {call.args[0]["query"] for call in audits.call_args_list} == set(names)
    assert all(call.kwargs == {"refresh": True} for call in audits.call_args_list)
    assert {call.args[0]["title"] for call in searches.call_args_list} == set(names)
    assert result.ok and result.status == "success"
    assert result.data["recommended_positions"] == list(range(1, 37))
    assert result.data["review_positions"] == []
    assert len(result.references) == 1
    snapshot = result.references[0].value
    assert len(snapshot["candidates"]) == len(snapshot["_private_items"]) == 36
    assert [item["position"] for item in snapshot["candidates"]] == list(range(1, 37))
    assert [item["result_id"] for item in snapshot["_private_items"]] == [
        item["result_id"] for item in snapshot["candidates"]
    ]
    expected = [(name, episode) for name in names for episode in (1, 2, 3)]
    for candidate, (name, episode) in zip(snapshot["candidates"], expected, strict=True):
        assert candidate["title"] == f"{name}.S01E{episode:02d}.1080p.WEB-DL"
        assert candidate["match"] == "exact_episode"
        assert candidate["_verification_context"]["tmdb_id"] == str(1000 + names.index(name))
        assert candidate["_verification_context"]["episode"] == episode
        assert candidate["_verification_context"]["season"] == 1
        assert provider.result_store.get(candidate["result_id"]).title == candidate["title"]
    assert all(len(group["positions"]) == 3 and not group["uncovered_episodes"] for group in result.data["groups"])
    assert [group["positions"] for group in result.data["groups"]] == [list(range(start, start + 3)) for start in range(1, 37, 3)]


def test_web_candidate_data_filter_keeps_tail_without_any_visual_changes():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node 不可用，不能执行现有 Web 候选过滤逻辑")
    source = Path("app/static/js/agent.js").read_text()
    body = source.split("function renderCandidateView(turn, view) {", 1)[1].split("const previousGroup", 1)[0]
    javascript = f"""
const pick = new Function('turn', 'view', {json.dumps(body + 'return items;')});
const items = Array.from({{length: 50}}, (_, i) => ({{title: String(i + 1), position: i + 1}}));
const result = pick({{}}, {{ref: 'ref_capacity', expires_at: 1, items: [...items, {{title: 'bad', position: 0}}]}});
if (JSON.stringify(result.map(item => item.position)) !== JSON.stringify(items.map(item => item.position))) process.exit(1);
"""
    completed = subprocess.run([node, "--input-type=module", "-e", javascript], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
