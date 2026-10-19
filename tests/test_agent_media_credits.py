from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock, patch

import pytest

from app.agent.domain_catalog import build_tool_specs
from app.agent.domains.discovery.credits_actions import (
    get_media_credits,
    media_credits_arguments,
)
from app.agent.domains.discovery.mapping_actions import get_discovery_detail
from app.agent.errors import AgentToolError
from app.agent.kernel.capabilities import ToolEffect
from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
from app.agent.models import ToolContext
from app.clients.tmdb import TMDBClient
from app.discovery.models import (
    MediaCard,
    ProviderAuthenticationError,
    ProviderInvalidResponse,
    ProviderNotConfigured,
    ProviderTimeout,
)


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.proxies = {}
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        result = Mock()
        result.status_code = 200
        result.json.return_value = self.payload
        return result

    def close(self):
        self.closed = True


def _call(payload, args=None):
    session = _Session(payload)
    client = TMDBClient(
        api_key="private-secret",
        base_url="https://api.themoviedb.org/3",
        proxy_url="",
        session=session,
    )
    with (
        patch("app.agent.domains.discovery.credits_actions.config.get_bool", return_value=True),
        patch("app.agent.domains.discovery.credits_actions.TMDBClient", return_value=client),
    ):
        result = get_media_credits(args or {"tmdb_id": "272938", "media_type": "tv"})
    assert session.closed
    return result, session


def test_movie_credits_preserve_roles_director_and_use_existing_client():
    result, session = _call(
        {
            "cast": [
                {
                    "id": 2,
                    "name": "演员乙",
                    "character": "配角",
                    "order": 2,
                    "profile_path": "/private.jpg",
                },
                {
                    "id": 1,
                    "name": "演员甲",
                    "original_name": "Actor A",
                    "character": "主角",
                    "order": 0,
                },
            ],
            "crew": [
                {"id": 3, "name": "编剧", "job": "Writer", "department": "Writing"},
                {"id": 4, "name": "导演", "job": "Director", "department": "Directing"},
            ],
        },
        {"tmdb_id": "550", "media_type": "movie"},
    )
    assert result.ok and result.data["complete"]
    assert result.data["credits_state"] == "available"
    assert result.data["scope"] == "movie"
    assert [person["name"] for person in result.data["cast"]] == ["演员甲", "演员乙"]
    assert result.data["cast"][0]["roles"] == [{"character": "主角"}]
    assert result.data["crew"][0]["jobs"] == [{"job": "Director"}]
    assert result.data["collected_at"] and result.evidence[0].source == "tmdb_credits"
    rendered = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "private.jpg" not in rendered and "private-secret" not in rendered
    url, kwargs = session.calls[0]
    assert url.endswith("/movie/550/credits")
    assert kwargs["params"]["language"] == "zh-CN"
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "season,suffix,scope",
    [
        (None, "/tv/272938/aggregate_credits", "series_aggregate"),
        (0, "/tv/272938/season/0/aggregate_credits", "season_aggregate"),
        (2, "/tv/272938/season/2/aggregate_credits", "season_aggregate"),
    ],
)
def test_tv_aggregate_roles_and_optional_season(season, suffix, scope):
    args = {"tmdb_id": "272938", "media_type": "tv"}
    if season is not None:
        args["season_number"] = season
    result, session = _call(
        {
            "cast": [
                {
                    "id": 1,
                    "name": "演员甲",
                    "order": 0,
                    "total_episode_count": 26,
                    "roles": [{"character": "李长寿", "episode_count": 26}],
                }
            ],
            "crew": [
                {
                    "id": 2,
                    "name": "导演乙",
                    "department": "Directing",
                    "total_episode_count": 26,
                    "jobs": [{"job": "Director", "episode_count": 26}],
                }
            ],
        },
        args,
    )
    assert result.ok and result.data["scope"] == scope
    assert session.calls[0][0].endswith(suffix)
    assert result.data["cast"][0]["roles"] == [
        {"character": "李长寿", "episode_count": 26}
    ]
    assert result.data["crew"][0]["jobs"] == [{"job": "Director", "episode_count": 26}]
    assert result.data["cast"][0]["episode_count"] == 26


def test_empty_provider_arrays_are_not_official_absence_claim():
    result, _ = _call({"cast": [], "crew": []})
    assert result.ok and result.data["credits_state"] == "empty"
    assert result.data["cast_coverage"]["state"] == "empty"
    assert "空表不代表官方未官宣" in result.data["coverage_note"]
    assert any("web.search" in item for item in result.suggestions)


@pytest.mark.parametrize(
    "payload", [{}, {"cast": []}, {"cast": None, "crew": []}, {"cast": {}, "crew": []}]
)
def test_missing_or_invalid_fields_do_not_become_empty_cast(payload):
    result, _ = _call(payload)
    assert not result.ok and result.status == "invalid_response"
    assert result.data["credits_state"] == "failed"
    assert "cast" not in result.data
    assert "不代表源站为空" in result.evidence[0].description


def test_projection_is_bounded_and_marks_invalid_unknown_and_truncated():
    roles = [
        {"character": f"角色{index}", "episode_count": index} for index in range(25)
    ]
    result, _ = _call(
        {
            "cast": [
                {"id": 1, "name": "甲", "roles": roles},
                {
                    "id": 2,
                    "name": "乙",
                    "roles": [{"character": "乙", "episode_count": -3}],
                },
                {"id": 3, "name": "丙"},
                None,
            ],
            "crew": [{"name": "导演", "jobs": "not-a-list"}],
        },
        {"tmdb_id": "1", "media_type": "tv", "cast_limit": 2},
    )
    assert result.ok and result.status == "partial"
    assert not result.data["complete"] and result.data["truncated"]
    assert result.data["cast_coverage"]["total"] == 4
    assert result.data["cast_coverage"]["discarded_invalid"] == 1
    assert len(result.data["cast"][0]["roles"]) == 20
    assert result.data["cast"][0]["roles_truncated"]
    assert result.data["cast"][1]["roles"][0]["episode_count"] is None
    assert result.data["crew_coverage"]["state"] == "partial"


@pytest.mark.parametrize("bounded", ["cast", "crew", "roles"])
def test_display_limit_does_not_make_successful_credits_query_partial(bounded):
    payload: dict[str, list[dict[str, Any]]] = {
        "cast": [{"name": "演员", "character": "主角"}],
        "crew": [{"name": "导演", "job": "Director"}],
    }
    args: dict[str, Any] = {"tmdb_id": "1", "media_type": "tv"}
    if bounded == "roles":
        payload["cast"][0]["roles"] = [{"character": f"角色{i}"} for i in range(25)]
    else:
        payload[bounded] *= 3
        args[f"{bounded}_limit"] = 2
    result, _ = _call(payload, args)
    assert result.ok and result.status == "completed"
    assert result.data["truncated"] and not result.data["complete"]
    coverage = result.data[f"{'cast' if bounded == 'roles' else bounded}_coverage"]
    assert coverage["state"] == "available" and coverage["truncated"]


@pytest.mark.parametrize("limits", [(2,), (2, 50)])
def test_limited_credits_and_expanded_read_finish_without_spurious_failure(limits):
    import asyncio

    from app.agent.kernel.capabilities import CapabilityRetriever, ToolCatalog
    from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
    from app.agent.kernel.pipeline import ToolPipeline
    from app.agent.kernel.session import AgentSession
    from app.agent.kernel.state import InMemorySessionStateStore
    from app.agent.kernel.state_types import AgentInput
    from tests.test_agent_kernel_core import ScriptedModel, collect

    async def scenario():
        payload = {"cast": [{"name": f"演员{i}", "character": f"角色{i}"} for i in range(30)], "crew": []}
        wires = [_Session(payload) for _ in limits]
        clients = [TMDBClient(api_key="test", base_url="https://api.themoviedb.org/3", proxy_url="", session=wire) for wire in wires]
        tool = catalog_from_tool_specs(build_tool_specs()).get("discovery.credits")
        catalog = ToolCatalog([tool])
        model = ScriptedModel([
            *[[ModelEvent(ModelEventType.TOOL_CALL_COMPLETED, tool_call=ModelToolCall(
                f"credits-{limit}", tool.model_name, {"tmdb_id": "1", "media_type": "movie", "cast_limit": limit})),
               ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")] for limit in limits],
            [ModelEvent(ModelEventType.TEXT_DELTA, text="两位主演是演员0和演员1。"),
             ModelEvent(ModelEventType.FINISH, finish_reason="stop")],
        ])
        store = InMemorySessionStateStore()
        session = AgentSession(model=model, catalog=catalog, state_store=store,
            pipeline=ToolPipeline(catalog=catalog, state_store=store),
            retriever=CapabilityRetriever(minimum=1, maximum=1))
        with (
            patch("app.agent.domains.discovery.credits_actions.config.get_bool", return_value=True),
            patch("app.agent.domains.discovery.credits_actions.TMDBClient", side_effect=clients),
        ):
            events = await collect(session.run(AgentInput(owner="owner", session_id="credits", message="只需要电影的前两位演员")))
        assert all(len(wire.calls) == 1 and wire.closed for wire in wires)
        assert events[-1].payload["status"] == "success", events[-1].payload
        assert events[-1].payload["answer"] == "两位主演是演员0和演员1。"
        saved = await store.load(owner="owner", session_id="credits")
        results = [json.loads(row["content"]) for row in saved.conversation if row["role"] == "tool"]
        assert len(results) == len(limits)
        assert results[0]["data"]["cast_coverage"]["returned"] == 2
        assert results[0]["data"]["truncated"]
        if len(limits) > 1:
            assert results[-1]["data"]["cast_coverage"]["returned"] == 30
            assert results[-1]["data"]["complete"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "exc,state,status",
    [
        (ProviderNotConfigured("未配置 TMDB_API_KEY"), "not_queried", "not_configured"),
        (
            ProviderTimeout("请求超时", detail="api_key=private-secret"),
            "failed",
            "timeout",
        ),
        (
            ProviderAuthenticationError("认证失败", detail="secret-token"),
            "failed",
            "authentication",
        ),
    ],
)
def test_failures_are_factual_sanitized_and_close_client(exc, state, status):
    client = Mock()
    client.media_credits.side_effect = exc
    with (
        patch("app.agent.domains.discovery.credits_actions.config.get_bool", return_value=True),
        patch("app.agent.domains.discovery.credits_actions.TMDBClient", return_value=client),
    ):
        result = get_media_credits({"tmdb_id": "1", "media_type": "movie"})
    assert not result.ok and result.status == status
    assert result.data["credits_state"] == state
    assert "private-secret" not in json.dumps(result.to_dict())
    assert "secret-token" not in json.dumps(result.to_dict())
    client.close.assert_called_once()


def test_disabled_does_not_instantiate_provider_or_claim_empty_cast():
    with (
        patch("app.agent.domains.discovery.credits_actions.config.get_bool", return_value=False),
        patch("app.agent.domains.discovery.credits_actions.TMDBClient") as factory,
    ):
        result = get_media_credits({"tmdb_id": "1", "media_type": "movie"})
    factory.assert_not_called()
    assert result.status == "disabled" and result.data["credits_state"] == "not_queried"


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"tmdb_id": True},
        {"tmdb_id": "0"},
        {"tmdb_id": "１２３"},
        {"tmdb_id": "1/credits"},
        {"tmdb_id": "1", "media_type": []},
        {"tmdb_id": "1", "media_type": "movie", "season_number": 1},
        {"tmdb_id": "1", "media_type": "tv", "season_number": True},
        {"tmdb_id": "1", "media_type": "tv", "season_number": None},
        {"tmdb_id": "1", "media_type": "tv", "season_number": 101},
        {"tmdb_id": "1", "media_type": "movie", "cast_limit": 51},
        {"tmdb_id": "1", "media_type": "movie", "crew_limit": True},
        {"tmdb_id": "1", "media_type": "movie", "arbitrary_path": "/secret"},
    ],
)
def test_strict_arguments_reject_invalid_identity_or_unbounded_reads(bad):
    with pytest.raises(AgentToolError):
        media_credits_arguments(bad)


def test_detail_explicitly_marks_credits_as_not_queried():
    service = Mock()
    service.get_detail.return_value = MediaCard(
        provider="tmdb",
        external_id="272938",
        media_type="tv",
        title="真人剧",
    )
    with (
        patch("app.agent.domains.discovery.mapping_actions.config.get_bool", return_value=True),
        patch(
            "app.agent.domains.discovery.mapping_actions.get_discovery_service",
            return_value=service,
        ),
    ):
        result = get_discovery_detail(
            {"provider": "tmdb", "media_type": "tv", "external_id": "272938"},
            ToolContext(),
        )
    assert result.data["credits_state"] == "not_queried"
    assert "不表示源站为空" in result.data["credits_note"]


def test_registered_credits_is_read_only_and_related_to_identity_and_web():
    catalog = catalog_from_tool_specs(build_tool_specs())
    tool = catalog.get("discovery.credits")
    assert tool.effect == ToolEffect.READ
    assert "discovery.search" in tool.metadata["related_tools"]
    assert "web.search" in tool.metadata["related_tools"]
    assert (
        "discovery.credits" in catalog.get("discovery.search").metadata["related_tools"]
    )
    assert (
        "discovery.credits" in catalog.get("discovery.detail").metadata["related_tools"]
    )


def test_client_rejects_movie_season_before_http():
    session = _Session({"cast": [], "crew": []})
    client = TMDBClient(api_key="private", proxy_url="", session=session)
    try:
        with pytest.raises(ValueError):
            client.media_credits("1", "movie", season_number=1)
        with pytest.raises(ProviderInvalidResponse):
            session.payload = {"id": 1}
            client.media_credits("1", "tv")
    finally:
        client.close()
    assert len(session.calls) == 1
