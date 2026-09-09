"""追漫日历的真实目录、中文召回与只读管线；配置隔离，禁止网络。"""

from __future__ import annotations

import asyncio
import json
import socket
from datetime import date, timedelta
from unittest.mock import Mock, patch

import pytest

import tests  # noqa: F401 -- 必须在 app 导入前建立临时 DB/config 隔离。
from app.agent.calendar_actions import anime_calendar, anime_calendar_arguments
from app.agent.domain_catalog import build_tool_specs
from app.agent.kernel.capabilities import CapabilityRetriever, ToolEffect
from app.agent.kernel.discovery import CapabilityDiscovery
from app.agent.kernel.pipeline import ToolCallContext, ToolPipeline, ToolPipelineError
from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
from app.agent.kernel.ports.mediaflux_policy import MediaFluxToolRateLimiter
from app.agent.kernel.session import DEFAULT_SYSTEM_PROMPT
from app.agent.kernel.state import CancellationToken, InMemorySessionStateStore
from app.agent.models import RiskLevel
from app.agent.public_safety import public_tool_label, sanitize_public_text
from app.agent.rate_limit import (
    AgentRateLimiter,
    allow_agent_tool,
    tool_rate_limit_policy,
)

NEW_TOOL = "discovery.anime_calendar"
OLD_TOOL = "bangumi.calendar"
MODEL_TOOL = "discovery__anime_calendar"


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("追漫日历接线测试禁止真实网络")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.fixture(scope="module")
def specs():
    return build_tool_specs()


@pytest.fixture(scope="module")
def catalog(specs):
    return catalog_from_tool_specs(specs)


def _snapshot(*, loading=False):
    start = date(2026, 9, 7)
    days = [
        {"date": (start + timedelta(days=index)).isoformat(), "items": []}
        for index in range(7)
    ]
    if not loading:
        days[3]["items"] = [{
            "source": "tencent", "source_id": "calendar-test-1",
            "title": "测试动画", "category": "animation", "free_progress": "",
            "events": [
                {"date": "2026-09-10", "update_time": "12:00",
                 "schedule": "会员更新一集", "audience": "member"},
                {"date": "2026-09-10", "update_time": "20:00",
                 "schedule": "非会员更新一集", "audience": "free"},
            ],
            "url": "https://v.qq.com/x/cover/not-for-agent",
            "poster_key": "private-test-poster", "cookie": "private-test-cookie",
        }]
    return {
        "timezone": "Asia/Shanghai", "today": "2026-09-10",
        "week_start": "2026-09-07", "days": days, "refreshing": loading,
        "sources": [
            {"id": source, "status": "loading" if loading else "ok",
             "message": "后台获取中" if loading else "已读取平台公开排期",
             "fetched_at": "" if loading else "2026-09-10T08:00:00+08:00"}
            for source in ("tencent", "iqiyi", "youku")
        ],
    }


async def _execute(catalog, name, arguments, *, finder=None, rate_limiter=None):
    state = InMemorySessionStateStore()
    lease, _ = await state.begin_turn(
        owner="calendar-test-owner", session_id="calendar-test-session",
        request_id="calendar-test-request",
    )

    async def progress(_payload):
        return None

    context = ToolCallContext(
        owner=lease.owner, session_id=lease.session_id,
        request_id="calendar-test-request", turn_id=lease.turn_id, lease=lease,
        cancellation=CancellationToken(), report_progress=progress,
        capability_search=finder.search if finder else None,
    )
    pipeline = ToolPipeline(catalog=catalog, state_store=state, rate_limiter=rate_limiter)
    return await pipeline.execute(name, arguments, context=context)


def test_catalog_registers_distinct_read_handler_and_model_alias(specs, catalog):
    by_name = {spec.name: spec for spec in specs}
    assert len(by_name) == len(specs) == len(catalog)
    spec = by_name[NEW_TOOL]
    tool = catalog.get(NEW_TOOL)
    assert spec.handler is anime_calendar
    assert spec.validator is anime_calendar_arguments
    assert spec.risk is RiskLevel.READ
    assert not spec.requires_confirmation
    assert tool.effect is ToolEffect.READ
    assert tool.read is not None
    assert tool.prepare is tool.execute_confirmed is None
    assert tool.domain == "discovery"
    assert tool.metadata["source_kind"] == "public_calendar"
    assert tool.metadata["freshness"] == "cached"
    assert catalog.get(MODEL_TOOL) is tool
    assert tool.model_definition()["name"] == MODEL_TOOL
    assert catalog.get(OLD_TOOL).domain == "bangumi"
    assert by_name[OLD_TOOL].handler is not anime_calendar
    assert set(by_name[OLD_TOOL].parameters["properties"]) == {"weekday", "page", "limit"}
    assert all("Bangumi" in example for example in by_name[OLD_TOOL].examples)


def test_declared_schema_defaults_match_validator_and_bounded_event_contract(catalog):
    tool = catalog.get(NEW_TOOL)
    schema = tool.input_schema
    props = schema["properties"]
    defaults = {name: prop["default"] for name, prop in props.items()}
    assert defaults == tool.validator({}) == {
        "day": "today", "source": "all", "query": "", "page": 1, "limit": 20,
    }
    assert schema["additionalProperties"] is False
    assert set(props) == {"day", "source", "query", "page", "limit"}
    assert props["source"]["enum"] == ["all", "tencent", "iqiyi", "youku"]
    assert set(props["day"]["anyOf"][0]["enum"]) == {
        "today", "tomorrow", "week", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday",
    }
    assert props["day"]["anyOf"][1]["pattern"] == "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
    for name, maximum in (("page", 100), ("limit", 20)):
        assert props[name]["type"] == "integer"
        assert (props[name]["minimum"], props[name]["maximum"]) == (1, maximum)
    assert (props["query"]["minLength"], props["query"]["maxLength"]) == (0, 80)
    assert "真实事件" in tool.description


@pytest.mark.parametrize("message", [
    "今天动漫更新", "追漫日历", "今天有哪些动画更新", "本周追番日历",
    "本周腾讯/爱奇艺/优酷排期", "本周腾讯排期", "本周爱奇艺排期", "本周优酷排期",
    "明天优酷有哪些动漫更新", "本周腾讯视频测试动画哪天更新",
])
def test_domestic_calendar_survives_real_retrieval_and_discovery_window(catalog, message):
    context = {"recent_tool_names": (), "reference_kinds": ()}
    selection = CapabilityRetriever().retrieve(message, catalog, context=context)
    window = CapabilityDiscovery(catalog, context=context).window(selection.tools)
    names = [tool.name for tool in window]
    assert 6 <= len(names) <= 10
    assert names[0] == "agent.capabilities"
    assert NEW_TOOL in names
    assert selection.scores[NEW_TOOL] > selection.scores[OLD_TOOL]
    assert names[1] == NEW_TOOL


@pytest.mark.parametrize("message", [
    "看看 Bangumi 本周放送日历", "Bangumi 星期六有哪些日番放送",
])
def test_explicit_bangumi_remains_the_preferred_calendar(catalog, message):
    selection = CapabilityRetriever().retrieve(message, catalog)
    assert OLD_TOOL in selection.names
    assert selection.scores[OLD_TOOL] > selection.scores[NEW_TOOL]
    assert selection.names[0] == OLD_TOOL


@pytest.mark.parametrize(("message", "expected"), [
    ("从我的媒体库推荐没看过的动漫", "media.recommend_from_library"),
    ("创建一个追番RSS订阅", "rss.create_subscription"),
])
def test_calendar_does_not_replace_library_recommendations_or_rss(catalog, message, expected):
    selection = CapabilityRetriever().retrieve(message, catalog)
    assert expected in selection.names
    assert selection.scores[expected] > selection.scores[NEW_TOOL]


@pytest.mark.parametrize("arguments", [{"query": "追漫日历"}, {"tool_names": [NEW_TOOL]}])
def test_discovery_pipeline_loads_real_tool_without_executing_calendar(catalog, arguments):
    finder = CapabilityDiscovery(catalog, context={})
    with patch("app.agent.calendar_actions.get_calendar_service") as service:
        result = asyncio.run(_execute(catalog, "agent.capabilities", arguments, finder=finder))
    service.assert_not_called()
    data = result.outcome.public_content["data"]
    assert NEW_TOOL in data["loaded_tools"]
    declaration = next(tool for tool in data["tools"] if tool["name"] == NEW_TOOL)
    assert declaration["effect"] == "read"
    window = finder.window((), finder.consume())
    assert MODEL_TOOL in [tool.model_name for tool in window]
    assert len(window) <= 10


@pytest.mark.parametrize("tool_name", [NEW_TOOL, MODEL_TOOL])
def test_real_pipeline_reads_single_cached_snapshot_and_projects_flat_events(catalog, tool_name):
    service = Mock()
    service.get_week.return_value = _snapshot()
    with (
        patch("app.agent.calendar_actions.config.get_bool", return_value=True),
        patch("app.agent.calendar_actions.get_calendar_service", return_value=service),
        patch("app.agent.discovery_actions.get_discovery_service") as bangumi,
    ):
        result = asyncio.run(_execute(catalog, tool_name, {"source": "tencent"}))
    service.get_week.assert_called_once_with()
    bangumi.assert_not_called()
    assert result.tool.name == NEW_TOOL
    assert result.arguments == {"day": "today", "source": "tencent", "query": "", "page": 1, "limit": 20}
    public = result.outcome.public_content
    assert public["ok"] and public["status"] == "success"
    assert result.outcome.effect_plan is None
    assert not result.outcome.refs
    data = public["data"]
    assert data["calendar_url"] == "/discovery/calendar"
    assert data["requested_dates"] == ["2026-09-10"]
    assert (data["total"], data["returned"], data["total_programmes"]) == (2, 2, 1)
    assert [item["audience"] for item in data["items"]] == ["member", "free"]
    assert all("events" not in item and item["free_progress"] == "" for item in data["items"])
    assert public["evidence"][0]["source"] == "anime_calendar"
    model = json.loads(result.outcome.model_content)
    assert model["data"]["items"] == data["items"]
    assert len(result.outcome.model_content) < 24_000
    for forbidden in ("not-for-agent", "private-test-poster", "private-test-cookie", "poster_key"):
        assert forbidden not in json.dumps(public)
        assert forbidden not in result.outcome.model_content


def test_cold_calendar_pipeline_returns_loading_without_retry_or_bangumi(catalog):
    service = Mock()
    service.get_week.return_value = _snapshot(loading=True)
    with (
        patch("app.agent.calendar_actions.config.get_bool", return_value=True),
        patch("app.agent.calendar_actions.get_calendar_service", return_value=service),
        patch("app.agent.discovery_actions.get_discovery_service") as bangumi,
    ):
        result = asyncio.run(_execute(catalog, NEW_TOOL, {}))
    service.get_week.assert_called_once_with()
    bangumi.assert_not_called()
    public = result.outcome.public_content
    assert not public["ok"] and public["status"] == "loading"
    assert public["data"]["items"] == []
    assert public["data"]["retry_after"] == 5
    assert {source["status"] for source in public["data"]["sources"]} == {"loading"}


@pytest.mark.parametrize("arguments", [
    {"force": True}, {"source": "bangumi"}, {"day": "2026-09-31"},
    {"page": 0}, {"limit": 21}, {"limit": True}, {"query": "字" * 81},
])
def test_registered_validator_rejects_unsafe_arguments_before_service(catalog, arguments):
    with (
        patch("app.agent.calendar_actions.get_calendar_service") as service,
        pytest.raises(ToolPipelineError),
    ):
        asyncio.run(_execute(catalog, NEW_TOOL, arguments))
    service.assert_not_called()


def test_rate_policy_has_independent_six_per_minute_bucket_and_canonical_alias(catalog):
    assert tool_rate_limit_policy(NEW_TOOL) == ("discovery-anime-calendar", 6, 1)
    assert tool_rate_limit_policy(OLD_TOOL) == ("bangumi-calendar", 6, 1)
    clock = [100.0]
    limiter = AgentRateLimiter(clock=lambda: clock[0], shared=False)
    service = Mock()
    service.get_week.return_value = _snapshot()
    with (
        patch("app.agent.rate_limit.agent_rate_limiter", limiter),
        patch("app.agent.calendar_actions.config.get_bool", return_value=True),
        patch("app.agent.calendar_actions.get_calendar_service", return_value=service),
    ):
        for index in range(6):
            name = NEW_TOOL if index % 2 else MODEL_TOOL
            asyncio.run(_execute(catalog, name, {}, rate_limiter=MediaFluxToolRateLimiter()))
        with pytest.raises(ToolPipelineError) as error:
            asyncio.run(_execute(catalog, MODEL_TOOL, {}, rate_limiter=MediaFluxToolRateLimiter()))
        assert error.value.code == "rate_limited"
        assert service.get_week.call_count == 6
        assert allow_agent_tool("calendar-test-owner", OLD_TOOL)
        assert allow_agent_tool("other-owner", NEW_TOOL)
        clock[0] += 61
        assert allow_agent_tool("calendar-test-owner", NEW_TOOL)


def test_public_label_and_kernel_guidance_keep_calendars_and_links_distinct():
    assert public_tool_label(NEW_TOOL) == "追漫日历"
    assert public_tool_label(OLD_TOOL) == "番剧放送日历"
    assert sanitize_public_text(f"正在执行 {NEW_TOOL}") == "正在执行 追漫日历"
    assert sanitize_public_text("请打开 /discovery/calendar") == ""
    for contract in (
        NEW_TOOL, OLD_TOOL, "Asia/Shanghai", "每项为一条排期事件", "不等于免费进度",
        "本回合不要循环调用", "calendar_url", "Telegram", "不猜站点地址",
    ):
        assert contract in DEFAULT_SYSTEM_PROMPT
