"""原生Session预算控制实验；真实LLM验收另行记录，不把脚本模型冒充实测。"""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from app.agent.errors import ToolPipelineError
from app.agent.kernel.capabilities import CapabilityRetriever, ToolCatalog
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.session import AgentSession, SessionLimits
from app.agent.kernel.state import InMemorySessionStateStore
from app.agent.kernel.state_types import AgentInput
from tests.test_agent_kernel_capability_discovery import capability_tool
from tests.test_agent_kernel_core import read_tool
from tests.test_agent_task_read_observations import control, declared


def run_pages(*, pages=13, mode="pages", limits=None, status="success", tracked=False):
    requests, attempted, read = [], [], []

    def handler(arguments, _context):
        attempted.append(arguments["page"])
        if mode == "exception":
            raise ToolPipelineError("测试源不可用", code="unavailable")
        if status in {"success", "completed", "empty"}:
            read.append(arguments["page"])
        return {"ok": status != "failed", "status": status,
                "data": {"page": arguments["page"], "has_more": arguments["page"] < pages}}

    tool = replace(read_tool("records.page", handler=handler), input_schema={
        "type": "object", "properties": {"page": {"type": "integer", "minimum": 1}},
        "required": ["page"], "additionalProperties": False,
    }, metadata={"kernel_utility": mode == "utility"})

    class Model:
        async def stream(self, request, *, cancellation):
            requests.append(request)
            cancellation.raise_if_cancelled()
            if not request.tools or (mode == "pages" and len(read) >= pages):
                yield ModelEvent(ModelEventType.TEXT_DELTA, text=f"实际读取{len(set(read))}页。")
                yield ModelEvent(ModelEventType.FINISH, finish_reason="stop")
                return
            name, arguments = tool.model_name, {"page": 1 if mode in {"repeat", "alias_repeat"} else len(requests)}
            if mode == "alias_repeat" and len(requests) % 2:
                name = tool.name
            if mode == "discovery":
                name, arguments = "agent__capabilities", {"query": f"读取记录 {len(requests)}"}
            yield ModelEvent(ModelEventType.TOOL_CALL_COMPLETED,
                             tool_call=ModelToolCall(f"call-{len(requests)}", name, arguments))
            yield ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")

    catalog = ToolCatalog([tool, capability_tool()] if mode == "discovery" else [tool])
    state = InMemorySessionStateStore()
    session = AgentSession(model=Model(), catalog=catalog, state_store=state,
                           retriever=CapabilityRetriever(minimum=1, maximum=1),
                           pipeline=ToolPipeline(catalog=catalog, state_store=state), limits=limits)

    async def run():
        if tracked:
            lease, _ = await state.begin_turn(owner="test-owner", session_id="read-pages", request_id="seed")
            await state.commit(lease, task_patch=control(declared()))
        return [event async for event in session.run(AgentInput(
            owner="test-owner", session_id="read-pages", message="读取全部记录后统计数量"))]

    events = asyncio.run(run())
    if mode == "discovery":
        assert not attempted
        assert any(event.type is AgentEventType.TOOL_COMPLETED
                   and event.payload["tool"] == "agent.capabilities"
                   and event.payload["result"]["ok"] for event in events)
    return events, requests, attempted, read


@pytest.mark.parametrize("pages", [13, 16])
@pytest.mark.parametrize("tracked", [False, True])
def test_distinct_successful_reads_can_finish_beyond_twelve_rounds(pages, tracked):
    events, requests, attempted, read = run_pages(pages=pages, tracked=tracked)
    assert attempted == read == list(range(1, pages + 1))
    assert len(requests) == pages + 1
    assert events[-1].payload["status"] == "success"


@pytest.mark.parametrize("tracked", [False, True])
def test_explicit_small_hard_round_limit_is_not_extended(tracked):
    _events, requests, attempted, _read = run_pages(limits=SessionLimits(max_model_rounds=2), tracked=tracked)
    assert len(requests) == 2
    assert attempted == [1]
    assert not requests[-1].tools


@pytest.mark.parametrize("tracked", [False, True])
def test_progress_does_not_increase_the_business_tool_budget(tracked):
    _events, requests, attempted, _read = run_pages(pages=17, tracked=tracked)
    assert len(attempted) == SessionLimits().max_tool_calls == 16
    assert len(requests) == 17
    assert not requests[-1].tools


@pytest.mark.parametrize("mode,status,ceiling", [
    ("repeat", "success", 13),
    ("alias_repeat", "success", 13),
    ("exception", "success", 12),
    ("pages", "failed", 12),
    ("pages", "partial", 12),
    ("pages", "submitted", 12),
    ("pages", "running", 12),
    ("discovery", "success", 12),
    ("utility", "success", 12),
])
@pytest.mark.parametrize("tracked", [False, True])
def test_repeats_failures_and_discovery_do_not_earn_more_observation_rounds(mode, status, ceiling, tracked):
    _events, requests, _attempted, _read = run_pages(pages=99, mode=mode, status=status, tracked=tracked)
    assert len(requests) <= ceiling
    assert not requests[-1].tools


@pytest.mark.parametrize("mode,status,expected", [
    ("pages", "success", 3), ("repeat", "success", 1),
    ("alias_repeat", "success", 1), ("exception", "success", 0),
    ("pages", "failed", 0), ("pages", "partial", 0), ("discovery", "success", 0),
    ("utility", "success", 0),
])
@pytest.mark.parametrize("tracked", [False, True])
def test_observation_progress_is_measured_without_broadcasting_a_countdown(mode, status, expected, tracked):
    events, requests, _attempted, _read = run_pages(pages=3, mode=mode, status=status, tracked=tracked)
    terminal = next(e for e in reversed(events) if "successful_read_queries" in e.payload)
    assert terminal.payload["successful_read_queries"] == expected
    assert "实际剩余工具调用" not in requests[0].system_prompt
    assert "只有已完成的新业务只读观察可扩展规划额度" not in requests[0].system_prompt


def test_budget_warning_is_only_exposed_near_the_real_limit():
    _events, requests, _attempted, _read = run_pages(pages=16)
    assert all("实际剩余工具调用" not in request.system_prompt for request in requests[:10])
    assert "实际剩余工具调用 2 次" in requests[-3].system_prompt
    assert "实际剩余工具调用 0 次" in requests[-1].system_prompt
