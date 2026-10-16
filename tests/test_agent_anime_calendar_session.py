"""追漫日历进入实际AgentSession工具循环；模型为本地桩，不调用外部LLM或平台。"""
from __future__ import annotations

import asyncio
import copy
import json
import threading
import unittest
from datetime import datetime
from unittest.mock import ANY, Mock, patch

import tests  # noqa: F401 -- 导入应用前隔离运行目录和配置。
from app.agent.domain_catalog import build_tool_specs
from app.agent.kernel.capabilities import CapabilityRetriever
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports.existing_actions import adapt_tool_spec, catalog_from_tool_specs
from app.agent.kernel.ports.mediaflux_policy import MediaFluxAuthorizationPolicy
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import AgentInput, InMemorySessionStateStore
from app.discovery.cache import DiscoveryCache
from app.discovery.calendar.models import CalendarEntry, CalendarEvent, SourceResult
from app.discovery.calendar.service import CalendarService
from tests.support import isolated_test_database
from tests.test_agent_anime_calendar import card, model_rows, snapshot
from tests.test_agent_kernel_capability_discovery import call, session_for
from tests.test_agent_kernel_core import ScriptedModel, collect


class CalendarEchoModel:
    """用工具实际返回值生成测试答复，避免硬编码成功文本掩盖链路失败。"""
    def __init__(self, arguments):
        self.arguments = arguments
        self.requests = []
        self.facts = None

    async def stream(self, request, *, cancellation):
        self.requests.append(request)
        cancellation.raise_if_cancelled()
        if len(self.requests) == 1:
            assert "discovery__anime_calendar" in {tool["name"] for tool in request.tools}
            yield ModelEvent(ModelEventType.TOOL_CALL_COMPLETED,
                             tool_call=ModelToolCall("calendar-read", "discovery__anime_calendar", self.arguments))
            yield ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")
            return
        assert len(self.requests) == 2
        messages = [message for message in request.messages if message.role == "tool"]
        assert len(messages) == 1
        self.facts = json.loads(messages[0].content)
        data = self.facts.get("data", {})
        if self.facts["status"] == "loading":
            text = "来源正在后台获取，暂不能确认今天的排期，不代表今天没有更新。"
        elif self.facts["status"] == "partial":
            row = model_rows(data)[0]
            text = f'{row["source_name"]} {row["date"]} 已收录 {data["total"]} 条排期：{row["title"]}；会员排期不等于免费观看。'
            text += f' [打开追漫日历]({data["calendar_url"]})'
        else:
            text = self.facts["summary"]
        await asyncio.sleep(0)
        yield ModelEvent(ModelEventType.TEXT_DELTA, text=text)
        yield ModelEvent(ModelEventType.FINISH, finish_reason="stop")


class AgentAnimeCalendarSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.data = snapshot()
        self.data["days"][3]["items"] = [card(title="离线回放动漫")]
        self.service = Mock(get_week=Mock(side_effect=lambda **kwargs: copy.deepcopy(self.data)))
        self.getter = self.enterContext(patch("app.agent.calendar_actions.get_calendar_service", return_value=self.service))
        self.enabled = self.enterContext(patch("app.agent.calendar_actions.config.get_bool", return_value=True))
        self.enterContext(patch("app.agent.kernel.ports.mediaflux_policy.is_agent_enabled", return_value=True))
        self.network_attempts = []
        def forbidden(*args, **kwargs):
            self.network_attempts.append(True)
            raise AssertionError("AgentSession追漫日历回归禁止真实网络")
        for name in ("getaddrinfo", "create_connection", "socket.connect", "socket.connect_ex"):
            self.enterContext(patch("socket." + name, side_effect=forbidden))

    def tearDown(self):
        self.assertEqual(self.network_attempts, [])

    def session(self, model):
        catalog = catalog_from_tool_specs(build_tool_specs())
        store = InMemorySessionStateStore()
        pipeline = ToolPipeline(catalog=catalog, state_store=store, authorization=MediaFluxAuthorizationPolicy())
        return AgentSession(model=model, catalog=catalog, retriever=CapabilityRetriever(), pipeline=pipeline, state_store=store)

    async def test_natural_calendar_query_uses_registered_read_tool_and_projects_actual_facts(self):
        model = CalendarEchoModel({"source": "youku"})
        session = self.session(model)
        events = await collect(session.run(AgentInput(message="今天优酷有哪些动漫更新，打开追漫日历", owner="webk:v1:" + "a" * 64,
                                                       session_id="calendar-session")))
        self.service.get_week.assert_called_once_with(wait_sources=("youku",), cancelled=ANY)
        self.assertEqual(len(model.requests), 2)
        facts = model.facts
        self.assertEqual((facts["status"], facts["data"]["total"], facts["data"]["returned"]), ("partial", 1, 1))
        self.assertEqual(facts["data"]["items"][0]["audience"], "member")
        self.assertEqual(facts["data"]["items"][0]["free_progress"], "")
        self.assertEqual(facts["data"]["sources"][0]["fetched_at"], "2026-09-10T04:24:05+08:00")
        self.assertEqual(facts["data"]["calendar_url"], "/discovery/calendar")
        self.assertEqual(events[-1].type, AgentEventType.TURN_COMPLETED)
        self.assertEqual(sum(event.type == AgentEventType.TOOL_STARTED for event in events), 1)
        self.assertFalse(any(event.type in {AgentEventType.TOOL_FAILED, AgentEventType.EFFECT_APPROVAL_REQUIRED} for event in events))
        content = json.dumps([event.to_dict() for event in events], ensure_ascii=False)
        self.assertIn("离线回放动漫", content)
        for private in ("private-poster-key", "should-not-be-returned", "private-token", "trackInfo"):
            self.assertNotIn(private, content)

    async def test_source_still_loading_after_bounded_wait_explains_gap_without_bangumi(self):
        self.data["sources"][2]["status"] = "loading"
        self.data["days"][3]["items"] = []
        self.data["refreshing"] = True
        model = CalendarEchoModel({"source": "youku"})
        events = await collect(self.session(model).run(AgentInput(message="今天优酷动漫排期", owner="webk:v1:" + "b" * 64,
                                                                session_id="loading-session")))
        self.service.get_week.assert_called_once_with(wait_sources=("youku",), cancelled=ANY)
        self.assertEqual(model.facts["status"], "loading")
        self.assertEqual(model.facts["data"]["retry_after"], 5)
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(sum(event.type == AgentEventType.TOOL_STARTED for event in events), 1)
        self.assertEqual(events[-1].type, AgentEventType.TURN_COMPLETED)

    async def test_cold_week_finishes_in_one_turn_from_shared_collector_without_user_refresh(self):
        self.enterContext(isolated_test_database())
        started, release = threading.Event(), threading.Event()
        calls = []
        dates = [day["date"] for day in self.data["days"]]

        class Provider:
            source = "tencent"
            allowed_hosts = frozenset({"v.qq.com"})

            async def fetch(self, http):
                calls.append(True)
                started.set()
                await asyncio.to_thread(release.wait, 2)
                return SourceResult(tuple(CalendarEntry(
                    "tencent", f"show_{i}", f"回放动漫{i}", "animation",
                    f"https://v.qq.com/x/cover/show_{i}.html",
                    evidence="线程测试平台排期回放",
                    events=(CalendarEvent(dates[i % 7], "10:00", "会员更新一话", "member"),),
                ) for i in range(53)), "partial", "公开排期有限覆盖")

        class NoNetwork:
            def __init__(self, hosts):
                pass

            async def aclose(self):
                pass

        class Metadata:
            def enrich(self, entries):
                return [entry.to_dict() for entry in entries]

        now = datetime(2026, 9, 10, 12)
        service = CalendarService(providers={"tencent": Provider()}, cache=DiscoveryCache(clock=lambda: now),
                                  http_factory=NoNetwork, metadata=Metadata(), clock=lambda: now)
        self.addCleanup(service.shutdown)
        self.getter.return_value = service
        model = CalendarEchoModel({"day": "week", "source": "tencent"})
        session = self.session(model)
        turn = asyncio.create_task(collect(session.run(AgentInput(
            message="腾讯动漫周一到周日追更表", owner="webk:v1:" + "a" * 64, session_id="cold-week",
        ))))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            self.assertFalse(turn.done(), "冷加载不能直接结束用户请求")
            self.assertEqual(service.get_week()["sources"][0]["status"], "loading", "页面读取保持非阻塞")
        finally:
            release.set()
        events = await asyncio.wait_for(turn, 5)
        facts = model.facts["data"]
        self.assertEqual((facts["total"], facts["returned"]), (53, 53))
        self.assertFalse(facts["has_more"])
        self.assertEqual({row["date"] for row in model_rows(facts)}, set(dates))
        self.assertEqual((len(calls), len(model.requests)), (1, 2))
        self.assertEqual(sum(event.type == AgentEventType.TOOL_STARTED for event in events), 1)
        self.assertEqual(events[-1].type, AgentEventType.TURN_COMPLETED)

    async def test_missing_initial_capability_can_be_discovered_then_called_in_same_turn(self):
        target = adapt_tool_spec(next(spec for spec in build_tool_specs() if spec.name == "discovery.anime_calendar"))
        session, model = session_for([target], [
            [call("agent.capabilities", {"tool_names": ["discovery.anime_calendar"]}, "find-calendar")],
            [call("discovery__anime_calendar", {"source": "youku"}, "calendar-read")],
            [ModelEvent(ModelEventType.TEXT_DELTA, text="按工具返回的公开排期及来源状态回答。")],
        ])
        events = await collect(session.run(AgentInput(message="查询追漫日历优酷今天的排期", owner="fixture-owner", session_id="find-session")))
        self.service.get_week.assert_called_once_with(wait_sources=("youku",), cancelled=ANY)
        self.assertNotIn("discovery__anime_calendar", {tool["name"] for tool in model.requests[0].tools})
        self.assertIn("discovery__anime_calendar", {tool["name"] for tool in model.requests[1].tools})
        self.assertEqual(events[-1].type, AgentEventType.TURN_COMPLETED)
        data = json.loads(next(message.content for message in model.requests[2].messages
                               if message.role == "tool" and message.tool_call_id == "calendar-read"))
        self.assertEqual(data["data"]["total"], 1)
        self.assertEqual(data["data"]["calendar_url"], "/discovery/calendar")

    async def test_read_tool_still_requires_production_principal_authorization(self):
        for owner in ("not-a-web-principal", "tg:v1:123\x1f456"):
            with self.subTest(owner=owner), patch("app.agent.kernel.ports.mediaflux_policy.telegram_owner_route_is_currently_authorized", return_value=False):
                model = ScriptedModel([
                    [call("discovery__anime_calendar", {"source": "youku"}), ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")],
                    [ModelEvent(ModelEventType.TEXT_DELTA, text="当前身份无权读取。"), ModelEvent(ModelEventType.FINISH, finish_reason="stop")],
                ])
                events = await collect(self.session(model).run(AgentInput(message="今天优酷追漫日历", owner=owner, session_id="denied-session")))
                failed = [event for event in events if event.type == AgentEventType.TOOL_FAILED]
                self.assertEqual(len(failed), 1)
                self.assertEqual(failed[0].payload["code"], "authorization_denied")
                self.getter.assert_not_called()

    async def test_feature_disabled_is_explained_by_tool_without_calendar_access(self):
        model = CalendarEchoModel({"source": "youku"})
        session = self.session(model)
        self.enabled.return_value = False
        events = await collect(session.run(AgentInput(message="今天优酷追漫日历", owner="webk:v1:" + "c" * 64,
                                                       session_id="disabled-session")))
        self.getter.assert_not_called()
        self.assertEqual(model.facts["status"], "disabled")
        self.assertEqual(events[-1].type, AgentEventType.TURN_COMPLETED)
