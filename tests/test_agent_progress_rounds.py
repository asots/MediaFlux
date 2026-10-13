"""原生Session预算控制实验；真实LLM验收另行记录，不把脚本模型冒充实测。"""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from app.agent.kernel.capabilities import CapabilityRetriever, ToolCatalog
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
from app.agent.kernel.pipeline import ToolPipeline, ToolPipelineError
from app.agent.kernel.session import AgentSession, SessionLimits
from app.agent.kernel.state import AgentInput, InMemorySessionStateStore
from tests.test_agent_kernel_core import read_tool


def run_pages(*, pages=13, mode="pages", limits=None, status="success"):
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
    })

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
                name, arguments = "agent__capabilities", {"query": "读取记录"}
            yield ModelEvent(ModelEventType.TOOL_CALL_COMPLETED,
                             tool_call=ModelToolCall(f"call-{len(requests)}", name, arguments))
            yield ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")

    catalog, state = ToolCatalog([tool]), InMemorySessionStateStore()
    session = AgentSession(model=Model(), catalog=catalog, state_store=state,
                           retriever=CapabilityRetriever(minimum=1, maximum=1),
                           pipeline=ToolPipeline(catalog=catalog, state_store=state), limits=limits)

    async def run():
        return [event async for event in session.run(AgentInput(
            owner="test-owner", session_id="read-pages", message="读取全部记录后统计数量"))]

    return asyncio.run(run()), requests, attempted, read


@pytest.mark.parametrize("pages", [13, 16])
def test_distinct_successful_reads_can_finish_beyond_twelve_rounds(pages):
    events, requests, attempted, read = run_pages(pages=pages)
    assert attempted == read == list(range(1, pages + 1))
    assert len(requests) == pages + 1
    assert events[-1].payload["status"] == "success"


def test_explicit_small_hard_round_limit_is_not_extended():
    _events, requests, attempted, _read = run_pages(limits=SessionLimits(max_model_rounds=2))
    assert len(requests) == 2
    assert attempted == [1]
    assert not requests[-1].tools


def test_progress_does_not_increase_the_business_tool_budget():
    _events, requests, attempted, _read = run_pages(pages=17)
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
])
def test_repeats_failures_and_discovery_do_not_earn_more_observation_rounds(mode, status, ceiling):
    _events, requests, _attempted, _read = run_pages(pages=99, mode=mode, status=status)
    assert len(requests) <= ceiling
    assert not requests[-1].tools


@pytest.mark.parametrize("mode,status,expected", [
    ("pages", "success", 3), ("repeat", "success", 1),
    ("alias_repeat", "success", 1), ("exception", "success", 0),
    ("pages", "failed", 0), ("pages", "partial", 0), ("discovery", "success", 0),
])
def test_model_sees_observation_progress_separate_from_attempt_budget(mode, status, expected):
    _events, requests, _attempted, _read = run_pages(pages=3, mode=mode, status=status)
    assert f"已完整取得并落盘 {expected} 个不同的成功业务只读观察" in requests[-1].system_prompt
    assert "调用尝试次数不是成功查询数、已读页数或记录总数" in requests[-1].system_prompt
    assert "不含历史回合" in requests[-1].system_prompt
