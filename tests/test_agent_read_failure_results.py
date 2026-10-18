"""READ 服务失败 DTO 保留领域事实，并阻止模型把未完成查询宣告成功。"""
from __future__ import annotations

import asyncio
import json
import socket
from collections import defaultdict, deque
from functools import partial
from unittest.mock import patch

import httpx
import pytest

from app.agent.domain_catalog import build_tool_specs
from app.agent.domains.discovery import web_search_actions as actions
from app.agent.errors import ToolPipelineError
from app.agent.kernel.capabilities import CapabilityRetriever, ToolCatalog
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports.existing_actions import adapt_tool_spec
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state_types import AgentInput
from app.agent.models import RiskLevel, ToolResult, ToolSpec
from app.clients.http import FixedHostHttpClient
from tests.test_agent_kernel_core import ScriptedModel, collect
from tests.test_agent_tool_argument_ingress import offline as offline
from tests.test_agent_tool_argument_ingress import store as store

CLAIMED_SUCCESS = "全部检索已经完成，所有来源都已核对。"
FINAL = [ModelEvent(ModelEventType.TEXT_DELTA, text=CLAIMED_SUCCESS),
         ModelEvent(ModelEventType.FINISH, finish_reason="stop")]


def round_calls(calls):
    return [*(ModelEvent(ModelEventType.TOOL_CALL_COMPLETED, tool_call=call) for call in calls),
            ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")]


def session_for(model, tool, store):
    catalog = ToolCatalog([tool])
    return AgentSession(model=model, catalog=catalog, state_store=store, journal=store,
        pipeline=ToolPipeline(catalog=catalog, state_store=store),
        retriever=CapabilityRetriever(minimum=1, maximum=1))


async def run_steps(store, steps, *, together=False, parallel=False):
    outcomes = defaultdict(deque)
    for query, value in steps:
        outcomes[query].append(value)
    observed = []

    def read(arguments):
        query = arguments["query"]
        observed.append(query)
        value = outcomes[query].popleft()
        if isinstance(value, ToolPipelineError):
            raise value
        return value

    spec = ToolSpec(name="fixture.lookup", description="核验指定对象", risk=RiskLevel.READ,
        parameters={"type": "object", "required": ["query"], "additionalProperties": False,
                    "properties": {"query": {"type": "string"}}},
        validator=dict, handler=read, parallel_read=parallel)
    tool = adapt_tool_spec(spec)
    calls = [ModelToolCall(f"call-{index}", tool.model_name, {"query": query})
             for index, (query, _) in enumerate(steps)]
    rounds = [round_calls(calls)] if together else [round_calls([call]) for call in calls]
    model = ScriptedModel([*rounds, FINAL])
    events = await collect(session_for(model, tool, store).run(AgentInput(
        owner="owner", session_id="read-result", message="核验指定对象，不以其他对象结果代替")))
    saved = await store.load(owner="owner", session_id="read-result")
    rows = [row for row in saved.conversation if row.get("role") == "tool"]
    assert sorted(observed) == sorted(query for query, _ in steps)
    assert [row["tool_call_id"] for row in rows] == [call.call_id for call in calls]
    # 失败仍为原始领域 DTO，没有在 adapter 或 pipeline 中改造成丢失资料的异常。
    for row, (_, value) in zip(rows, steps):
        if isinstance(value, ToolResult):
            assert json.loads(row["content"]) == value.to_dict()
    assert "_recovery_tracked" not in json.dumps(saved.conversation)
    return events, saved, model


@pytest.mark.parametrize("status,code", [(503, "unavailable"), (429, "rate_limited")])
def test_real_web_search_http_failure_cannot_be_announced_as_success(status, code, store):
    async def scenario():
        requests = []

        def transport(request):
            requests.append((request.headers["host"], json.loads(request.content)["query"]))
            return httpx.Response(status, json={"error": "fixture outage"})

        def factory(**kwargs):
            return FixedHostHttpClient(**kwargs, transport=httpx.MockTransport(transport),
                resolver=lambda host, port: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))])

        settings = {"WEB_SEARCH_ENABLED": "1", "TAVILY_API_KEY": "offline-key", "TAVILY_DAILY_CREDIT_LIMIT": "100"}
        original = actions._search_tavily
        tool = adapt_tool_spec(next(spec for spec in build_tool_specs() if spec.name == "web.search"))
        query = f"offline outage {status}"
        model = ScriptedModel([round_calls([ModelToolCall("lookup", tool.model_name, {"query": query})]), FINAL])
        with patch.object(actions, "get", side_effect=lambda key, default="": settings.get(key, default)), patch.object(
            actions, "_search_tavily", partial(original, client_factory=factory),
        ):
            events = await collect(session_for(model, tool, store).run(AgentInput(
                owner="owner", session_id="web-outage", message="请搜索并核对公开来源")))
        saved = await store.load(owner="owner", session_id="web-outage")
        result = json.loads(next(row["content"] for row in saved.conversation if row["role"] == "tool"))
        assert requests == [("api.tavily.com", query)]
        assert result["ok"] is False and result["status"] == code
        assert result["summary"] == ("网页搜索服务暂时不可用" if status == 503 else "网页搜索服务请求过于频繁，请稍后重试")
        assert events[-1].type.value == "turn.failed" and events[-1].payload["code"] == code
        assert "answer" not in events[-1].payload
        assert not any(row.get("content") == CLAIMED_SUCCESS for row in saved.conversation)
        assert len(model.requests) == 2
    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["unavailable", "timeout", "authentication", "invalid_response", "http_error",
    "provider_timeout", "upstream_timeout", "provider_auth_failed", "indexer_error", "configuration_missing"])
def test_explicit_failed_read_service_codes_remain_terminal_blockers(status, store):
    async def scenario():
        value = ToolResult(False, status, "查询服务尚未完成", data={"retryable": True}, error="查询服务失败")
        events, _, _ = await run_steps(store, [("object-x", value)])
        assert events[-1].type.value == "turn.failed"
        assert events[-1].payload["code"] == status
        assert events[-1].payload["message"] == "查询服务失败"
    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["not_found", "disabled", "loading", "empty", "blocked"])
def test_negative_business_observation_is_not_a_service_exception(status, store):
    async def scenario():
        events, _, _ = await run_steps(store, [("object-x", ToolResult(False, status, "已取得业务观察"))])
        assert events[-1].payload["status"] == "success"
        assert events[-1].payload["answer"] == CLAIMED_SUCCESS
    asyncio.run(scenario())


@pytest.mark.parametrize("first_kind", ["dto", "exception"])
@pytest.mark.parametrize("later_status", ["not_found", "loading"])
def test_another_failed_result_for_same_arguments_cannot_erase_service_failure(first_kind, later_status, store):
    async def scenario():
        first = (ToolResult(False, "provider_timeout", "对象x查询超时") if first_kind == "dto"
                 else ToolPipelineError("对象x查询超时", code="provider_timeout"))
        events, _, _ = await run_steps(store, [
            ("object-x", first), ("object-x", ToolResult(False, later_status, "尚未取得该对象的完整查询结果")),
        ])
        assert events[-1].type.value == "turn.failed"
        assert events[-1].payload["code"] == "provider_timeout"
        assert events[-1].payload["message"] == "对象x查询超时"
    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["ok", "empty"])
def test_success_for_same_arguments_recovers_failed_read_including_successful_empty_result(status, store):
    async def scenario():
        events, _, _ = await run_steps(store, [
            ("object-x", ToolResult(False, "unavailable", "对象x服务暂不可用")),
            ("object-x", ToolResult(True, status, "已核验对象x", data={"results": []})),
        ])
        assert events[-1].payload["status"] == "success"
        assert events[-1].payload["answer"] == CLAIMED_SUCCESS
    asyncio.run(scenario())


def test_success_for_other_arguments_does_not_recover_failed_object(store):
    async def scenario():
        events, _, _ = await run_steps(store, [
            ("object-x", ToolResult(False, "unavailable", "对象x服务暂不可用")),
            ("object-y", ToolResult(True, "ok", "已核验对象y", data={"query": "object-y"})),
        ])
        terminal = events[-1].payload
        assert terminal["status"] == "partial" and terminal["finish_reason"] == "unavailable"
        assert "已核验对象y" in terminal["answer"]
        assert "对象x服务暂不可用" in terminal["answer"]
        assert "未完成的查询不能据此判断没有内容" in terminal["answer"]
    asyncio.run(scenario())


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_mixed_read_batch_retains_partial_results_and_discloses_service_failure(parallel, store):
    async def scenario():
        events, _, model = await run_steps(store, [
            ("object-x", ToolResult(True, "ok", "已核验对象x", data={"query": "object-x"})),
            ("object-y", ToolResult(False, "unavailable", "对象y查询服务不可用")),
        ], together=True, parallel=parallel)
        terminal = events[-1].payload
        assert terminal["status"] == "partial" and terminal["finish_reason"] == "unavailable"
        assert "已核验对象x" in terminal["answer"] and "对象y查询服务不可用" in terminal["answer"]
        assert terminal["answer"] != CLAIMED_SUCCESS
        assert model.requests[0].allow_parallel_tool_calls is parallel
    asyncio.run(scenario())
