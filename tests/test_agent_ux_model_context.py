"""真实领域 DTO 经 Kernel 到模型下一轮的兼容性；不调用真实 Provider。"""

from __future__ import annotations
from typing import Any

import asyncio
import json
from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from app.agent.domains.library.episode_resource_actions import (
    missing_episode_resource_arguments,
    missing_season_resource_arguments,
    search_missing_episode_resources,
    search_missing_season_resources,
)
from app.agent.domains.resources.indexer_actions import (
    search_arguments,
    search_resources,
)
from app.agent.domains.resources.presentation import (
    MediaFluxProjector,
    ResourceSelection,
)
from app.agent.kernel.capabilities import (
    CapabilityRetriever,
    KernelToolSpec,
    ToolCatalog,
    ToolEffect,
)
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.projection import DefaultProjector
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import InMemorySessionStateStore
from app.agent.kernel.state_types import AgentInput
from app.indexers.models import AggregatedIndexerResult, IndexerProviderError
from tests.test_agent_indexer_actions import (
    _SECRET_DETAIL_URL,
    _SECRET_MAGNET,
    _SECRET_TORRENT_URL,
    _resource_item,
)
from tests.test_agent_missing_episode_resources import _audit_result
from tests.test_agent_task_read_observations import control, declared


class FixtureIndexer:
    """仅替换 I/O，搜索 DTO、推荐排序与引用封装仍运行正式实现。"""

    enabled_site_ids = frozenset({"nyaa"})

    def __init__(self, *, empty=False, titles=None):
        self.empty = empty
        self.titles = titles
        self.items = {}
        self.result_store = Mock(get=lambda result_id: self.items[result_id])

    def site_ids_for_scope(self, scope="manual"):
        return tuple(self.enabled_site_ids)

    async def search_media(self, request, sites=None, *, scope="manual", timeout_seconds=None):
        label = (
            f"S{request.season:02d}E{request.episode:02d}"
            if request.season is not None and request.episode is not None
            else request.title.rsplit(" ", 1)[-1]
        )
        label = label if label.startswith("S02E") else "S02E03"
        items = [] if self.empty else [
            _resource_item(
                result_id=f"fixture-{label}-candidate-{index}",
                title=title.format(label=label),
                seeders=40 - index,
            )
            for index, title in enumerate(self.titles or (
                "Example.{label}.2160p.WEB-DL", "Example.{label}.1080p.WEB-DL",
            ), 1)
        ]
        self.items.update({item.result_id: item for item in items})
        return AggregatedIndexerResult(
            query=request.title,
            page=request.page,
            items=items,
            sites_attempted=("nyaa", "broken"),
            sites_succeeded=("nyaa",),
            errors=[IndexerProviderError("broken", "unavailable", "站点暂不可用")],
            partial=True,
            cached=True,
            has_more=True,
        )


def real_dto(kind, *, titles=None, original_name="Example"):
    service = FixtureIndexer(empty=kind == "empty", titles=titles)

    def verified_audit(*, missing, target_missing=None):
        audit = _audit_result(missing=missing, target_missing=target_missing)
        # 英文名必须来自审计元数据，不能根据资源站返回标题反推作品别名。
        audit.effect_metadata["episode_mapping_context"] = {
            "detail": {"name": "示例剧", "original_name": original_name},
            "season_detail": {"season_number": 2, "episodes": [
                {"episode_number": number, "air_date": "2020-01-01"} for number in range(1, 7)]},
            "as_of": "2026-08-15",
        }
        return audit

    with (
        patch("socket.socket.connect", side_effect=AssertionError("network forbidden")),
        patch("sqlite3.connect", side_effect=AssertionError("database forbidden")),
        patch("app.agent.domains.resources.indexer_actions.config.get_bool", return_value=True),
        patch("app.agent.domains.resources.indexer_actions.get_indexer_service", return_value=service),
        patch("app.agent.domains.library.episode_resource_actions.get_indexer_service", return_value=service),
    ):
        if kind in {"search", "empty"}:
            return search_resources(search_arguments({"title": "Example S02E03", "page": 2, "limit": 10}))
        if kind in {"episode", "not_missing"}:
            audit = verified_audit(missing=[{"season": 2, "episode": 3}], target_missing=kind != "not_missing")
            with patch("app.agent.domains.library.episode_resource_actions.audit_series_episodes", return_value=audit):
                return search_missing_episode_resources(missing_episode_resource_arguments({
                    "query": "示例剧", "season": 2, "episode": 3, "as_of": "2026-08-15",
                }))
        audit = verified_audit(missing=[{"season": 2, "episode": number} for number in (3, 4, 5, 6)])
        with patch("app.agent.domains.library.episode_resource_actions.audit_series_episodes", return_value=audit):
            arguments = {"query": "示例剧", "season": 2}
            return search_missing_season_resources(missing_season_resource_arguments({
                **({"items": [arguments]} if kind == "batch" else arguments),
                "max_episodes": 2, "as_of": "2026-08-15",
            }))


class NextRoundModel:
    def __init__(self, name):
        self.name = name
        self.requests = []

    async def stream(self, request, *, cancellation):
        cancellation.raise_if_cancelled()
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelEvent(ModelEventType.TOOL_CALL_COMPLETED, tool_call=ModelToolCall("read-resources", self.name, {}))
            yield ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")
        else:
            yield ModelEvent(ModelEventType.TEXT_DELTA, text="已读取本轮结构化工具结果。")
            yield ModelEvent(ModelEventType.FINISH, finish_reason="stop")


def next_model_round(dto, *, kind="search", maximum=24_000, tracked=False):
    async def exercise():
        name = {
            "episode": "library.search_missing_episode_resources",
            "season": "library.search_missing_season_resources",
            "batch": "library.search_missing_season_resources",
            "not_missing": "library.search_missing_episode_resources",
        }.get(kind, "indexer.search_resources")
        # 只隔离工具参数/外部 I/O；模型循环、projector、refs、事件与提交均为正式 Kernel。
        tool = KernelToolSpec(
            name=name, domain="indexer", description="搜索资源并返回已验证的检索上下文",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            effect=ToolEffect.READ, read=lambda _args, _context: dto,
            metadata={"source_kind": "resource_index"},
        )
        catalog = ToolCatalog(
            [tool],
            projector=MediaFluxProjector(max_model_chars=maximum),
            selection=ResourceSelection(),
        )
        states = InMemorySessionStateStore()
        if tracked:
            lease, _ = await states.begin_turn(owner="ux-model-owner", session_id="ux-model-session", request_id="seed")
            await states.commit(lease, task_patch=control(declared()))
        pipeline = ToolPipeline(catalog=catalog, state_store=states)
        model = NextRoundModel(name)
        session = AgentSession(model=model, catalog=catalog, retriever=CapabilityRetriever(), pipeline=pipeline, state_store=states)
        events = [event async for event in session.run(AgentInput(message="搜索资源", owner="ux-model-owner", session_id="ux-model-session"))]
        assert not any(event.type in {AgentEventType.TURN_FAILED, AgentEventType.TOOL_FAILED} for event in events)
        assert len(model.requests) == 2
        message = next(message for message in reversed(model.requests[1].messages) if message.role == "tool")
        payload = json.loads(message.content)
        task = payload.pop("task", None)
        assert bool(task) is tracked
        auxiliary = {key: payload.pop(key) for key in ("opaque_refs", "reference_arguments", "candidate_numbers", "recommended_ingest_arguments", "result_handle", "read_tool") if key in payload}
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        suffix = json.dumps(auxiliary, ensure_ascii=False, separators=(",", ":")) if auxiliary else ""
        public = next(event.payload["result"] for event in events if event.type is AgentEventType.TOOL_COMPLETED)
        if tracked:
            assert task["task_id"] == public["task"]["task_id"]
        if kind in {"empty", "not_missing"}:
            from tests.test_agent_public_text_decoding import assert_scoped_snapshot
            expected = json.loads(DefaultProjector().project(dto).model_content)
            assert auxiliary["result_handle"] == public["result_handle"] and auxiliary["read_tool"] == "agent.read_result"
            assert bool(auxiliary.get("reference_arguments")) is (kind == "empty")
            expected.update({key: auxiliary[key] for key in ("opaque_refs", "reference_arguments") if key in auxiliary})
            await assert_scoped_snapshot(pipeline, auxiliary["result_handle"], expected,
                                         owner="ux-model-owner", session_id="ux-model-session")
        return json.loads(encoded), encoded, suffix, message.content, public, model.requests[1]
    return asyncio.run(exercise())


def assert_private_absent(content):
    for private in (_SECRET_DETAIL_URL, _SECRET_MAGNET, _SECRET_TORRENT_URL, "/private/", "10.0.0.9", "SECRET", "rawresult", "raw_result"):
        assert private not in content


def model_resource_candidates(payload):
    """读取实际模型回执中的候选事实；同一候选的嵌套副本不能冒充多项。"""
    rows = {}

    def collect(value):
        if isinstance(value, dict):
            if type(value.get("position")) is int and "technical_evidence" in value:
                identity = (value["position"], value["title"])
                if identity in rows:
                    assert rows[identity]["technical_evidence"] == value["technical_evidence"]
                rows[identity] = value
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(payload)
    return list(rows.values())


def assert_original_facts_preserved(actual, expected):
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual
            assert_original_facts_preserved(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for value, source in zip(actual, expected):
            assert_original_facts_preserved(value, source)
    else:
        assert actual == expected


@pytest.mark.parametrize("count", [20, 50])
def test_candidate_evidence_is_single_copy_without_losing_original_facts(count):
    from app.agent.domains.resources.selection import (
        annotate_resource_candidates,
        candidate_item,
    )

    items = [{"position": number, "result_id": f"resource-{number:04d}",
              "title": f"Example.S01E{number:02d}.1080p", "seeders": 0,
              "download_state": "ready", "size_bytes": None} for number in range(1, count + 1)]
    payload = {"ok": True, "data": {"items": deepcopy(items), "page": 2, "has_more": True}}
    original = deepcopy(payload)
    annotate_resource_candidates(payload, [candidate_item(item, item["position"]) for item in items])
    assert "candidate_numbers" not in payload
    assert_original_facts_preserved(payload, original)
    encoded = json.dumps(payload)
    assert all(encoded.count(item['title']) == 1 for item in items)
    for row in model_resource_candidates(payload):
        assert row["technical_evidence"]["resolution"] == "1080p"
        assert row["technical_evidence"]["dynamic_range"] is None
        assert row["technical_evidence"]["source"] is None
        assert row["technical_evidence"]["stream_verified"] is False
    # 只读投影既不修改源DTO，也不把同一编号的另一标题覆盖成当前候选。
    assert items == original["data"]["items"]
    conflicting: dict[str, Any] = {"data": {"items": [{"position": 1, "title": "Other title"}]}}
    annotate_resource_candidates(conflicting, [candidate_item(items[0], 1)])
    assert conflicting["data"]["items"] == [{"position": 1, "title": "Other title"}]
    assert conflicting["candidate_numbers"][0]["title"] == items[0]["title"]


def test_conflicting_search_hits_stay_in_frozen_scope_but_not_default_recommendation():
    from app.agent.domains.resources.candidate_context import (
        restore_resource_candidate_reference,
    )
    from app.indexers.models import IndexerMediaSearchRequest
    from app.indexers.ranking import rank_item
    from app.indexers.result_store import IndexerResultStore
    from tests.test_agent_indexer_actions import FakeIndexerService

    request = IndexerMediaSearchRequest.create(title="Example", year=2026, season=2, episode=1)
    titles = ["Example.2026.S04E01.1080p", "Example.2025.S02E01.1080p", "Example.2026.S02E01.1080p"]
    service = FakeIndexerService(items=[rank_item(_resource_item(title=title, result_id=f"opaque-resource-{i:04d}"),
                                                 media=request, fallback_query=request.title) for i, title in enumerate(titles)])
    real_results = IndexerResultStore()
    for item in service.items:
        assert item.result_id is not None
        real_results.restore(item.result_id, item)
    with patch.object(service, 'result_store', real_results), patch('app.agent.domains.resources.indexer_actions.config.get_bool', return_value=True), patch('app.agent.domains.resources.indexer_actions.get_indexer_service', return_value=service):
        dto = search_resources(search_arguments({"title": "Example", "year": 2026, "season": 2, "episode": 1}))
    assert [row['title'] for row in dto.data['items']] == titles
    assert dto.model_data is not None
    assert [row['position'] for row in dto.model_data['items']] == [2, 3]
    # 系列首播年不等于后续季发布年；年份冲突仍是观察证据，不能据此隐藏精确季集。
    assert dto.model_data is not None
    assert 'year_conflict' in dto.model_data['items'][0]['match_reasons']
    assert 'episode_exact' in dto.model_data['items'][0]['match_reasons']
    assert dto.model_data['query_scope']['known_conflicts'] == 1
    assert dto.model_data['query_scope']['global_availability'] == 'unknown'
    assert dto.suggestions == []
    actual, _, suffix, content, _, _ = next_model_round(dto)
    numbers = model_resource_candidates({**actual, **json.loads(suffix)})
    assert [row['position'] for row in numbers] == [2, 3]
    assert titles[0] not in content and titles[1] in content
    # 原始研究快照可精确检查，不破坏依赖发布季映射的缺集业务。
    with patch.object(service, 'result_store', real_results), patch('app.indexers.runtime.get_indexer_service', return_value=service):
        restored = restore_resource_candidate_reference(dto.references[0].value)
    assert restored is not None
    assert [row['position'] for row in restored['candidates']] == [1, 2, 3]
    assert restored is not None
    assert [row['title'] for row in restored['candidates']] == titles
    assert restored is not None
    assert [real_results.get(row['result_id']).title for row in restored['candidates']] == titles


@pytest.mark.parametrize("kind", ["episode", "season", "batch"])
@pytest.mark.parametrize("has_match", [False, True])
def test_missing_search_model_omits_unmatched_titles_without_losing_scope_or_handles(kind, has_match):
    titles = (["Example.{label}.2160p.WEB-DL"] if has_match else []) + [
        "Example.UnverifiedArc.1080p.WEB-DL", "Example.KnownOld.S02E01.2160p.WEB-DL",
    ]
    dto = real_dto(kind, titles=titles)
    original = deepcopy(dto.to_dict())
    actual, _, suffix, content, public, _ = next_model_round(dto, kind=kind)
    assert "UnverifiedArc" not in content and "KnownOld" not in content
    assert_original_facts_preserved(actual["data"], dto.model_data)
    assert dto.to_dict() == original, "精简模型视图不能删除原始查询证据"
    if kind == "batch":
        assert actual["data"]["groups"][0]["remaining"] == 2
        assert actual["data"]["groups"][0]["remaining_episodes"] == [{"season": 2, "episode": number} for number in (5, 6)]
    elif kind == "season":
        assert actual["data"]["remaining"] == 2
        assert actual["data"]["remaining_episodes"] == [{"season": 2, "episode": number} for number in (5, 6)]
    else:
        assert actual["data"]["verification"]["verified_missing"] is True
    appendix = json.loads(suffix)
    assert appendix["reference_arguments"]["resource_candidates_ref"]
    numbers = model_resource_candidates({**actual, **appendix})
    assert bool(numbers) is has_match
    assert bool(appendix.get("recommended_ingest_arguments")) is has_match
    for item in numbers:
        assert item["match"] in {"exact_episode", "episode_pack"}
        assert "Example." in item["title"]
    if not has_match:
        assert public["candidate_view"] is None
        assert any("UnverifiedArc" in item["title"] for item in dto.references[0].value["candidates"])
    assert all("KnownOld" not in item["title"] for item in dto.references[0].value["candidates"])
    reference = dto.references[0].value
    assert len(reference["_private_items"]) == len(reference["candidates"])
    assert [item["result_id"] for item in reference["_private_items"]] == [item["result_id"] for item in reference["candidates"]]
    assert [(item["title"], item["site_id"]) for item in reference["_private_items"]] == [
        (item["title"], item["site_id"]) for item in reference["candidates"]]


@pytest.mark.parametrize("kind", ["season", "batch"])
def test_unregistered_english_title_is_not_recommended_or_kept_for_review(kind):
    dto = real_dto(kind, titles=["Example.{label}.2160p.WEB-DL"], original_name="")
    original = deepcopy(dto.to_dict())
    actual, _, suffix, content, public, _ = next_model_round(dto, kind=kind)
    assert dto.references == []
    assert dto.to_dict() == original
    assert actual["data"] == dto.model_data
    assert "resource_candidates_ref" not in suffix
    assert "Example." not in content
    assert public["candidate_view"] is None


@pytest.mark.parametrize("kind", ["search", "episode", "season"])
def test_real_resource_dto_reaches_next_model_request_without_business_field_loss(kind):
    dto = real_dto(kind)
    expected = json.loads(DefaultProjector().project(dto).model_content)
    actual, _, suffix, content, public, request = next_model_round(dto, kind=kind)
    # 对真正下一轮 ModelRequest 做全结构比较，而非只检查领域输出或 refs 存在。
    assert_original_facts_preserved(actual, expected)
    assert "data" in actual and not actual.get("truncated")
    assert request.round_index == 1
    assert "resource_candidates_ref" in suffix
    assert "candidate_view" not in actual and "selection" not in content
    assert_private_absent(content)
    assert "result_id" not in json.dumps(public)
    if kind == "search":
        assert public["candidate_view"] is None
        assert '"candidate_numbers":' not in suffix
        assert len(model_resource_candidates(actual)) == len(actual["data"]["items"])
    else:
        assert public["candidate_view"]["selection_ref"] not in content
    data = actual["data"]
    if kind == "search":
        assert data["page"] == 2 and data["has_more"] is True
        assert data["cached"] is True and data["partial"] is True
        assert data["sites_attempted"] == ["nyaa", "broken"] and data["sites_succeeded"] == ["nyaa"]
        assert data["errors"][0]["code"] == "unavailable"
        assert data["items"][0]["result_id"] == "fixture-S02E03-candidate-1"
        assert data["items"][0]["seeders"] == 39 and data["items"][0]["download_state"] == "ready"
    elif kind == "episode":
        assert data["verification"]["verified_missing"] is True
        assert data["verification"]["tmdb_id"] == "12345"
        search = data["search"]
        assert search["recommendation"]["candidate_count"] == 2
        assert search["recommendation"]["selected"]["confidence"] == "high"
        assert search["recommendation"]["alternatives"][0]["rank"] == 2
        assert search["download_plan"]["requires_confirmation"] is True
        assert search["download_plan"]["auto_submit"] is False
        assert search["download_plan"]["supported_targets"] == ["qb", "guangya", "both"]
    else:
        assert {key: data[key] for key in ("missing_total", "processed", "remaining", "failed", "truncated")} == {
            "missing_total": 4, "processed": 2, "remaining": 2, "failed": 0, "truncated": True,
        }
        assert data["episodes"][0]["search"]["download_plan"]["mode"] == "read_only"
        assert data["episodes"][1]["search"]["has_more"] is True


@pytest.mark.parametrize("kind", ["empty", "not_missing"])
def test_empty_query_and_not_missing_preserve_facts_without_download_candidates(kind):
    dto = real_dto(kind)
    assert bool(dto.references) is (kind == "empty")
    expected = json.loads(DefaultProjector().project(dto).model_content)
    actual, _, suffix, content, public, _ = next_model_round(dto, kind=kind)
    assert actual == expected
    assert json.loads(suffix)["result_handle"] == public["result_handle"]
    assert json.loads(suffix)["read_tool"] == "agent.read_result"
    assert public["candidate_view"] is None
    assert bool(public["refs"]) is (kind == "empty")
    assert not json.loads(suffix).get("candidate_numbers")
    assert_private_absent(content)
    if kind == "empty":
        assert actual["data"]["page"] == 2 and actual["data"]["has_more"] is True
        assert actual["data"]["partial"] is True and actual["data"]["errors"]
    else:
        assert actual["status"] == "not_missing"
        assert actual["data"]["verification"]["verified_missing"] is False


@pytest.mark.parametrize("has_refs", [False, True])
@pytest.mark.parametrize("tracked", [False, True])
def test_model_data_override_is_kept_but_private_branches_and_ui_tickets_are_removed(has_refs, tracked):
    dto = real_dto("search")
    if tracked:
        dto.status = "success"  # 成功 READ 才会生成任务观察补丁。
    if not has_refs:
        dto.references.clear()
    safe = {
        "paging": {"page": 7, "has_more": True},
        "verification": {"verified_missing": True, "episode": 3},
        "recommendation": {"selected": {"result_id": "fixture-S02E03-candidate-1", "confidence": "high"}},
        "download_plan": {"auto_submit": False, "requires_confirmation": True},
        "remaining": 2,
    }
    dto.model_data = {
        **deepcopy(safe),
        "rawresult": {"download_url": "https://10.0.0.9/SECRET"},
        "raw_result": {"cookie": "SECRET"},
        "_private_items": [{"magnet": "magnet:?xt=urn:btih:SECRET"}],
        "details": {
            "downloadUrl": "https://10.0.0.9/SECRET", "torrent_url": "https://10.0.0.9/SECRET",
            "api_key": "SECRET", "credentials": {"password": "SECRET"},
            "note": "局部提示 https://10.0.0.9/SECRET", "location": "/private/SECRET",
            "safe_counter": 3,
        },
        "candidate_view": {"selection": {"ref": "ref_ui_only_ticket_1234567890", "position": 1}},
        "selection": {"ref": "ref_ui_only_ticket_1234567890", "position": 1},
    }
    actual, _, _, content, _, _ = next_model_round(dto, tracked=tracked)
    assert {key: actual["data"][key] for key in safe} == safe
    assert "items" not in actual["data"], "不能用 public.data 覆盖 model_data"
    assert "selection" not in content and "ref_ui_only_ticket" not in content
    assert actual["data"]["details"]["safe_counter"] == 3
    assert_private_absent(content)
    assert set(actual["data"]["details"]) == {"note", "location", "safe_counter"}


@pytest.mark.parametrize("has_refs", [False, True])
@pytest.mark.parametrize("tracked", [False, True])
def test_model_next_request_still_respects_projector_length_budget(has_refs, tracked):
    dto = real_dto("search")
    if tracked:
        dto.status = "success"
    if not has_refs:
        dto.references.clear()
    dto.model_data = {"safe_details": ["x" * 1500 for _ in range(40)]}
    maximum = 2000
    expected = json.loads(DefaultProjector(max_model_chars=maximum).project(dto).model_content)
    actual, encoded, suffix, content, public, _ = next_model_round(dto, maximum=maximum, tracked=tracked)
    assert expected.get("truncated") is True and actual.get("truncated") is True
    assert actual["data"]["safe_details"]  # 有界明细，不再整块丢弃data
    assert len(encoded) <= maximum
    # 引用与候选附录也在同一预算内；完整明细可经result_handle续取。
    # Session 附加的任务元数据单独计算，不放宽业务结果的 projector 预算。
    task = json.loads(content).get("task")
    task_chars = len(json.dumps({"task": task}, ensure_ascii=False, separators=(",", ":"))) - 1 if task else 0
    assert len(content) <= maximum + task_chars
    assert "result_handle" in suffix
    assert public["candidate_view"] is None
    assert ('"candidate_numbers":' in suffix) is has_refs
    assert "candidate_view" not in content and "selection" not in content
