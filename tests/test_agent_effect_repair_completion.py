"""参数修复只在原计划真实完成并保存回执后解除错误阻塞。"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import replace

import pytest

from app import database as db
from app.agent.errors import ToolPipelineError
from app.agent.kernel import tool_batches
from app.agent.kernel.capabilities import KernelToolSpec, ToolCatalog, ToolEffect
from app.agent.kernel.effects import PreparedEffect
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelRequest, ModelToolCall
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.state import InMemorySessionStateStore
from app.agent.kernel.state_types import AgentInput, CancellationToken
from app.agent.kernel.tool_calls import ToolCallContext
from tests.test_agent_kernel_core import collect
from tests.test_agent_scoped_trust import (
    FINAL,
    OWNER,
    SCOPE,
    SESSION,
    TOOL,
    arguments,
    call,
    rig as scoped_rig,
    task_store as isolated_store,
)

task_store = isolated_store
rig = scoped_rig


class RepairModel:
    def __init__(self, rounds: list[list[ModelEvent] | Callable[[ModelRequest], list[ModelEvent]]]) -> None:
        self.rounds = rounds

    async def stream(self, request: ModelRequest, *, cancellation: CancellationToken) -> AsyncIterator[ModelEvent]:
        cancellation.raise_if_cancelled()
        reply = self.rounds.pop(0)
        for event in reply(request) if callable(reply) else reply:
            yield event


def repair(arguments):
    def reply(request: ModelRequest) -> list[ModelEvent]:
        ticket = next(json.loads(message.content)["repair"]["ticket"] for message in reversed(request.messages)
                      if message.role == "tool" and "repair" in json.loads(message.content))
        return [ModelEvent(ModelEventType.TOOL_CALL_COMPLETED,
                          tool_call=ModelToolCall("repair", "agent.repair", {"ticket": ticket, "arguments": arguments})),
                ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")]
    return reply


async def trusted_session(rig, rounds):
    session = rig.session([])
    session.model = RepairModel([call(arguments()), *rounds])
    preview = await collect(session.run(AgentInput(owner=OWNER, session_id=SESSION, message="把这部剧加入探索收藏")))
    await session.set_scoped_trust(owner=OWNER, session_id=SESSION, operation="grant",
        plan_id=preview[-1].payload["plan_id"], tool_name=TOOL, scope=SCOPE, max_actions=5, ttl_seconds=300)
    return session


@pytest.mark.parametrize("mode", ["direct", "repair", "lineage", "receipt_readback"])
def test_real_watchlist_repair_settles_only_after_one_confirmed_write(rig, mode):
    async def run():
        if mode == "direct":
            rounds = [call({**arguments(501), "unsupported": True}), call(arguments(501)), FINAL]
        elif mode == "repair":
            rounds = [call({**arguments(501), "provider": "invalid"}), repair(arguments(501)), FINAL]
        elif mode == "lineage":
            rounds = [call({"external_id": "501"}),
                      repair({**arguments(501), "media_type": 4}), repair(arguments(501)), FINAL]
        else:
            rounds = [call(arguments(501)), call({**arguments(501), "unsupported": True}), call(arguments(501)), FINAL]
        session = await trusted_session(rig, rounds)
        events = await collect(session.run(AgentInput(owner=OWNER, session_id=SESSION, message="把另一部电视剧也加入探索收藏")))
        assert db.get_media_watchlist("tmdb", "501", "tv") is not None
        assert rig.attempts == [arguments(501)] and await rig.remaining() == 4
        assert events[-1].type is AgentEventType.TURN_COMPLETED
        assert events[-1].payload["status"] == "success", events[-1].payload
        assert "未完成的查询" not in events[-1].payload["answer"]
        assert not any(event.type is AgentEventType.EFFECT_APPROVAL_REQUIRED for event in events)
    asyncio.run(run())


def test_other_objects_failure_survives_successful_automatic_write(rig):
    async def run():
        session = await trusted_session(rig, [call({**arguments(501), "unsupported": True}), call(arguments(502)), FINAL])
        events = await collect(session.run(AgentInput(owner=OWNER, session_id=SESSION, message="收藏这些电视剧")))
        assert db.get_media_watchlist("tmdb", "501", "tv") is None
        assert db.get_media_watchlist("tmdb", "502", "tv") is not None
        assert rig.attempts == [arguments(502)] and await rig.remaining() == 4
        assert events[-1].payload["status"] == "partial"
        assert events[-1].payload["finish_reason"] == "invalid_arguments"
    asyncio.run(run())


@pytest.mark.parametrize("status", ["failed", "partial", "accepted"])
def test_nonterminal_or_failed_execution_does_not_settle_repair(rig, status):
    async def run():
        def execute(args, *_):
            rig.attempts.append(args)
            return {"ok": status != "failed", "status": status, "summary": "执行未完整完成"}
        rig.pipeline = ToolPipeline(catalog=ToolCatalog([replace(rig.tool, execute_confirmed=execute)]),
            state_store=rig.store, effect_store=rig.pipeline.effect_store)
        session = await trusted_session(rig, [call({**arguments(501), "unsupported": True}), call(arguments(501)), FINAL])
        events = await collect(session.run(AgentInput(owner=OWNER, session_id=SESSION, message="收藏这部电视剧")))
        assert rig.attempts == [arguments(501)] and await rig.remaining() == 4
        assert events[-1].payload["status"] == "partial"
        assert events[-1].payload["finish_reason"] == "invalid_arguments"
    asyncio.run(run())


def test_receipt_save_failure_does_not_record_recovery(rig, monkeypatch):
    async def run():
        session = await trusted_session(rig, [call({**arguments(501), "unsupported": True}), call(arguments(501))])
        successful_records = []
        original_record = tool_batches.record_result

        def record(turn, name, args, result, **kwargs):
            if kwargs.get("error") is None:
                successful_records.append(result)
            original_record(turn, name, args, result, **kwargs)

        async def reject_receipt(*_args, **_kwargs):
            return False
        monkeypatch.setattr(tool_batches, "record_result", record)
        monkeypatch.setattr("app.agent.kernel.turn.remember_result", reject_receipt)
        events = await collect(session.run(AgentInput(owner=OWNER, session_id=SESSION, message="收藏这部电视剧")))
        assert db.get_media_watchlist("tmdb", "501", "tv") is not None
        assert successful_records == []
        assert events[-1].payload["status"] == "partial"
        assert events[-1].payload["finish_reason"] == "receipt_unavailable"
    asyncio.run(run())


@pytest.mark.parametrize("explicit", [False, True], ids=["direct", "explicit-repair"])
@pytest.mark.parametrize("status", ["completed", "failed", "partial", "accepted"])
def test_plan_binding_survives_normalization_and_frozen_arguments(explicit, status):
    async def run():
        writes = []
        tool = KernelToolSpec(name="fixture.change", domain="fixture", description="冻结字段转换",
            input_schema={"type": "object", "required": ["target"], "properties": {
                "target": {"type": "string"}, "limit": {"type": "integer"}}, "additionalProperties": False},
            effect=ToolEffect.WRITE,
            validator=lambda args: {"target": args["target"].strip(), "limit": args.get("limit", 1)},
            prepare=lambda args, _: PreparedEffect(preview={"summary": "修改预览"}, snapshot_fingerprint="snapshot",
                arguments={"target_number": args["target"].upper()}),
            execute_confirmed=lambda args, *_: writes.append(args) or {
                "ok": status != "failed", "status": status, "summary": "执行结果"})
        store = InMemorySessionStateStore()
        lease, _ = await store.begin_turn(owner="owner", session_id="normalization", request_id="r")
        context = ToolCallContext(lease.owner, lease.session_id, lease.request_id, lease.turn_id, lease,
                                 CancellationToken(), lambda _: asyncio.sleep(0))
        pipeline = ToolPipeline(catalog=ToolCatalog([tool]), state_store=store)
        with pytest.raises(ToolPipelineError) as failure:
            await pipeline.execute(tool.name, {"target": " x ", "unsupported": True}, context=context)
        ticket = failure.value.repair_ticket
        corrected = {"target": " x "}
        preview = await pipeline.execute("agent.repair" if explicit else tool.name,
            {"ticket": ticket, "arguments": corrected} if explicit else corrected, context=context)
        assert not writes and preview.recovered_failures == ()
        assert preview.arguments == {"target": "x", "limit": 1}
        assert preview.effect_plan is not None and dict(preview.effect_plan.arguments) == {"target_number": "X"}
        if explicit:
            with pytest.raises(ToolPipelineError, match="不属于本轮"):
                await pipeline.execute("agent.repair", {"ticket": ticket, "arguments": corrected}, context=context)
        result = await pipeline.execute_confirmed(preview.effect_plan.plan_id, context=context)
        assert writes == [{"target_number": "X"}]
        assert result.recovered_failures == ((ticket,) if status == "completed" else ())
    asyncio.run(run())
