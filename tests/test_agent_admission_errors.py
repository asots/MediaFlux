"""准入拒绝须保留未完成事实；只有原调用真实恢复才能清算错误。"""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from app.agent.errors import ToolPipelineError
from app.agent.kernel.capabilities import (
    CapabilityRetriever,
    CapabilitySelection,
    KernelToolSpec,
    ToolCatalog,
    ToolEffect,
)
from app.agent.kernel.effects import PreparedEffect
from app.agent.kernel.events import AgentEventType as E
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state_types import AgentInput
from app.agent.public_view import public_conversation_messages
from tests.test_agent_kernel_capability_discovery import capability_tool
from tests.test_agent_recovered_progress import reply
from tests.test_agent_tool_argument_ingress import PROTOCOLS, Wire, encode
from tests.test_agent_tool_argument_ingress import offline as offline  # noqa: PLC0414
from tests.test_agent_tool_argument_ingress import store as store  # noqa: PLC0414

OWNER, SID = "admission-owner", "admission-errors"
LOOKUP, HEALTH = "library.lookup", "library.health"
FIND = ("agent__capabilities", {"tool_names": [LOOKUP]})
X, Y = {"ref": "object-x"}, {"ref": "object-y"}
CLAIM = "所有目录均已核验完成，没有任何错误。"


class HealthRetriever(CapabilityRetriever):
    """词法初选遗漏目标，后续必须经过真实发现加载它。"""

    def retrieve(self, message, catalog, *, context=None):
        return CapabilitySelection((catalog.get(HEALTH),), {HEALTH: 1.0})


def reader(name, calls):
    def read(arguments, context):
        assert (context.owner, context.session_id) == (OWNER, SID)
        calls.append((name, dict(arguments)))
        return {"ok": True, "status": "success", "summary": "已核验指定对象",
                "data": {"title": "可信查询结果"}}

    return KernelToolSpec(
        name=name, domain="library", description="核验指定对象", effect=ToolEffect.READ,
        input_schema={"type": "object", "properties": {
            "ref": {"type": "string"}, "limit": {"type": "integer"}}, "additionalProperties": False},
        read=read,
    )


def make_session(store, adapter, tools):
    catalog = ToolCatalog([capability_tool(), *tools])
    return AgentSession(model=adapter, catalog=catalog, state_store=store, journal=store,
        pipeline=ToolPipeline(catalog=catalog, state_store=store), retriever=HealthRetriever())


async def cold_history(store, events):
    cold = SQLiteKernelStore(secret_provider=lambda: "offline-tool-argument-test")
    state = await cold.load(owner=OWNER, session_id=SID)
    assert await cold.list_events(owner=OWNER, session_id=SID) == [event.to_dict() for event in events]
    calls = [call for row in state.conversation for call in row.get("tool_calls", ())]
    results = [row for row in state.conversation if row["role"] == "tool"]
    assert [call["call_id"] for call in calls] == [row["tool_call_id"] for row in results]
    return state


async def run_calls(store, protocol, tools, steps, *, malformed_first=False):
    replies = [reply(protocol, name=name, arguments=arguments, index=index)
               for index, (name, arguments) in enumerate(steps)]
    if malformed_first:
        body = json.loads(replies[0][0])
        if protocol == "chat_completions":
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"ref":'
        else:
            body["output"][0]["arguments"] = '{"ref":'
        replies[0] = encode(body).encode(), "application/json"
    wire = Wire(protocol, [*replies, reply(protocol, text=CLAIM)])
    session = make_session(store, wire.adapter, tools)
    events = [event async for event in session.run(AgentInput(
        owner=OWNER, session_id=SID, message="核验指定对象，缺少能力时先发现，再完成原查询。"))]
    wire.verify(len(steps) + 1)
    assert not wire.replies
    return events, await cold_history(store, events), wire


def assert_unfinished(events, state, *, partial=False, code="tool_not_available"):
    terminal = events[-1]
    if partial:
        assert terminal.type is E.TURN_COMPLETED
        assert terminal.payload["status"] == "partial"
        assert terminal.payload["finish_reason"] == code
        assert CLAIM not in terminal.payload["answer"]
        assert "已核验指定对象" in terminal.payload["answer"]
    else:
        assert terminal.type is E.TURN_FAILED
        assert terminal.payload["code"] == code
        assert terminal.payload["successful_read_queries"] == 0
    assert CLAIM not in encode(public_conversation_messages(state.conversation))
    assert CLAIM not in encode([event.to_dict() for event in events])


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("kind", ["unauthorized", "unknown"])
def test_rejected_native_call_cannot_persist_a_zero_query_success(store, protocol, kind):
    async def run():
        calls = []
        target = replace(reader(LOOKUP, calls), authorize=lambda _context: False)
        tools = [reader(HEALTH, calls), *([target] if kind == "unauthorized" else [])]
        name = target.model_name if kind == "unauthorized" else "unregistered__lookup"
        events, state, _wire = await run_calls(store, protocol, tools, [(name, X)])
        assert calls == []
        assert [event.payload["code"] for event in events if event.type is E.TOOL_FAILED] == ["tool_not_available"]
        result = next(row for row in state.conversation if row["role"] == "tool")
        assert json.loads(result["content"])["code"] == "tool_not_available"
        assert_unfinished(events, state)

    asyncio.run(run())


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("arguments", [{}, X], ids=["implicit-object", "explicit-object"])
@pytest.mark.parametrize("reversed_names", [False, True], ids=["alias-first", "canonical-first"])
def test_discover_then_complete_same_native_call_recovers_its_admission_error(
    store, protocol, arguments, reversed_names,
):
    async def run():
        calls = []
        target = replace(reader(LOOKUP, calls), validator=lambda raw: {"limit": 20, **raw})
        rejected_name, retry_name = (target.model_name, target.name)
        if reversed_names:
            rejected_name, retry_name = retry_name, rejected_name
        steps = [(rejected_name, arguments), FIND, (retry_name, arguments)]
        events, state, wire = await run_calls(store, protocol, [reader(HEALTH, calls), target], steps)
        assert calls == [(LOOKUP, {"limit": 20, **arguments})]
        assert events[-1].type is E.TURN_COMPLETED and events[-1].payload["status"] == "success"
        assert events[-1].payload["answer"] == CLAIM
        assert events[-1].payload["successful_read_queries"] == 1
        assert any(event.type is E.MODEL_DELTA and event.payload["delta"] == CLAIM for event in events)
        assert public_conversation_messages(state.conversation)[-1]["content"] == CLAIM
        results = [row for row in state.conversation if row["role"] == "tool"]
        assert json.loads(results[0]["content"])["code"] == "tool_not_available"
        assert json.loads(results[-1]["content"])["ok"] is True
        assert "library__lookup" not in [tool.get("function", tool)["name"] for tool in wire.requests[0]["tools"]]
        assert "library__lookup" in [tool.get("function", tool)["name"] for tool in wire.requests[2]["tools"]]

    asyncio.run(run())


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("case", ["discover-only", "other-object", "other-tool", "implicit-context", "different-raw-default"])
def test_discovery_or_a_different_call_cannot_clear_the_original_admission_error(store, protocol, case):
    async def run():
        calls = []
        target = replace(reader(LOOKUP, calls), validator=lambda raw: {"limit": 20, **raw})
        steps = {
            "discover-only": [(target.model_name, X), FIND],
            "other-object": [(target.model_name, X), FIND, (LOOKUP, Y)],
            "other-tool": [(target.model_name, X), (HEALTH, {})],
            "implicit-context": [(target.model_name, {}), (HEALTH, {}), FIND, (LOOKUP, {})],
            "different-raw-default": [(target.model_name, X), FIND, (LOOKUP, {**X, "limit": 20})],
        }[case]
        events, state, _wire = await run_calls(store, protocol, [reader(HEALTH, calls), target], steps)
        assert_unfinished(events, state, partial=case != "discover-only")
        if case == "discover-only":
            assert calls == []
        elif case == "other-object":
            assert calls == [(LOOKUP, {**Y, "limit": 20})]
        elif case == "other-tool":
            assert calls == [(HEALTH, {})]
        elif case == "implicit-context":
            assert calls == [(HEALTH, {}), (LOOKUP, {"limit": 20})]
        else:
            assert calls == [(LOOKUP, {**X, "limit": 20})]

    asyncio.run(run())


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("dto", [False, True], ids=["exception", "dto"])
def test_a_failed_same_object_retry_does_not_become_success(store, protocol, dto):
    async def run():
        calls = []

        def failed(arguments, _context):
            calls.append(dict(arguments))
            if dto:
                return {"ok": False, "status": "provider_timeout", "summary": "指定对象查询超时"}
            raise ToolPipelineError("指定对象查询超时", code="provider_timeout")

        target = replace(reader(LOOKUP, []), read=failed)
        events, state, _wire = await run_calls(store, protocol, [reader(HEALTH, []), target],
            [(target.model_name, X), FIND, (LOOKUP, X)])
        assert calls == [X]
        assert_unfinished(events, state, code="provider_timeout")

    asyncio.run(run())


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_an_unparsed_unavailable_call_cannot_be_reinterpreted_as_an_empty_object(store, protocol):
    async def run():
        calls = []
        target = reader(LOOKUP, calls)
        events, state, _wire = await run_calls(store, protocol, [reader(HEALTH, calls), target],
            [(target.model_name, X), FIND, (LOOKUP, {})], malformed_first=True)
        assert calls == [(LOOKUP, {})]
        assert_unfinished(events, state, partial=True)
        call = next(call for row in state.conversation for call in row.get("tool_calls", ()))
        assert call["arguments"] is None and call["argument_error"] == "invalid_json"

    asyncio.run(run())


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_scoped_confirmation_completion_uses_the_original_admitted_arguments(store, protocol):
    async def run():
        writes = []
        tool = KernelToolSpec(name="library.change", domain="library", description="变更指定对象", effect=ToolEffect.WRITE,
            input_schema={"type": "object", "required": ["ref"], "properties": {
                "ref": {"type": "string"}, "enabled": {"type": "boolean"}}, "additionalProperties": False},
            validator=lambda raw: {"enabled": True, **raw},
            prepare=lambda arguments, _context: PreparedEffect(
                preview={"ok": True, "summary": "确认后变更指定对象"}, arguments=arguments, snapshot_fingerprint="original"),
            execute_confirmed=lambda arguments, _fingerprint, _context: (
                writes.append(dict(arguments)) or {"ok": True, "status": "completed", "summary": "指定对象已变更"}),
            metadata={"risk": "low_write"}, approval_scope=lambda _name, arguments, _metadata: {"ref": arguments["ref"]})
        preview_wire = Wire(protocol, [reply(protocol, name=tool.model_name, arguments=X)])
        session = make_session(store, preview_wire.adapter, [reader(HEALTH, []), tool])
        session.retriever = CapabilityRetriever(minimum=len(session.catalog), maximum=len(session.catalog))
        preview = [event async for event in session.run(AgentInput(
            owner=OWNER, session_id=SID, message="为指定对象生成确认卡。"))]
        preview_wire.verify()
        assert preview[-1].payload["status"] == "approval_required"
        plan_id = preview[-1].payload["plan_id"]
        granted = await session.set_scoped_trust(owner=OWNER, session_id=SID, operation="grant",
            plan_id=plan_id, tool_name=tool.name, scope=X, ttl_seconds=300, max_actions=5)
        assert granted["active"] is True and writes == []
        assert await session.cancel_effect(owner=OWNER, session_id=SID, plan_id=plan_id)

        replay = Wire(protocol, [reply(protocol, name=tool.model_name, arguments=X, index=1),
            reply(protocol, name="agent__capabilities", arguments={"tool_names": [tool.name]}, index=2),
            reply(protocol, name=tool.name, arguments=X, index=3), reply(protocol, text=CLAIM)])
        session.model, session.retriever = replay.adapter, HealthRetriever()
        events = [event async for event in session.run(AgentInput(
            owner=OWNER, session_id=SID, message="完成已预授权的指定对象变更，以实际回执为准。"))]
        replay.verify(4)
        assert writes == [{**X, "enabled": True}]
        assert any(event.type is E.EFFECT_COMPLETED for event in events)
        assert [event.payload["code"] for event in events if event.type is E.TOOL_FAILED] == ["tool_not_available"]
        assert events[-1].type is E.TURN_COMPLETED and events[-1].payload["status"] == "success"
        assert events[-1].payload["finish_reason"] == ("end_turn" if protocol == "anthropic_messages" else "stop")
        cold = SQLiteKernelStore(secret_provider=lambda: "offline-tool-argument-test")
        state = await cold.load(owner=OWNER, session_id=SID)
        assert state.metadata["scoped_trust"]["remaining"] == 4
        assert not state.pending_effect_plan_id
        calls = [call for row in state.conversation for call in row.get("tool_calls", ())]
        results = [row for row in state.conversation if row["role"] == "tool"]
        assert len(calls) == len(results) == 4
        assert json.loads(results[1]["content"])["code"] == "tool_not_available"
        assert json.loads(results[-1]["content"])["status"] == "completed"

    asyncio.run(run())
