"""effect READ quota anchor：原生 pipeline 拒限，离线、无模型、无云盘。"""
from __future__ import annotations

from typing import cast

import operator

from typing import Any

import asyncio
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from app.agent.errors import ToolPipelineError, ToolRateLimitError, _ReadPermitRejected
from app.agent.feature_gate import RuntimeFence
from app.agent.kernel.capabilities import KernelToolSpec, ToolCatalog, ToolEffect
from app.agent.kernel.continuation import (
    _ContinuationAction,
    anchor_for,
    apply_actions,
    history_call,
    register_effect,
    release_effect,
)
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.session_state import SessionState
from app.agent.kernel.state import InMemorySessionStateStore
from app.agent.kernel.state_types import CancellationToken, PublicationLease
from app.agent.kernel.tool_calls import ToolCallContext
from app.agent.rate_limit import RateLimitDecision

NOW = 100.0
READ = "library.inspect"


@pytest.fixture
def effect_case():
    binding = SimpleNamespace(
        fence=RuntimeFence(1, "a" * 32, 1, "stable", "b" * 64),
        effective_digest="effect-read-anchor",
    )
    state = SessionState("anchor-owner", "anchor-session", generation=2,
        conversation=[{"role": "user", "content": "后台任务完成后读取实际结果。"}])
    lease = PublicationLease(state.owner, state.session_id, 2, "resume-turn",
        "source-request", "accepted-effect", "claim-1", "effect")
    origin = dict(owner=state.owner, session_id=state.session_id,
        source_request_id=lease.request_id, source_turn_id="source-turn",
        origin_generation=1, expected_task=None, admission_fence=asdict(binding.fence),
        effective_digest=binding.effective_digest, model_policy={}, channel="web",
        model_limit=8, tool_limit=8)
    row: dict[str, Any] = {"expires_at": NOW + 300}
    register_effect(row, SimpleNamespace(origin=origin, plan_id=lease.continuation_id,
        lease=lease), NOW)
    state.metadata["effect_waits"] = {lease.continuation_id: row}
    record = row["continuation"]
    assert "anchor" not in record
    row["final_result"] = {"ok": True, "summary": "后台任务已完成"}
    release_effect(row, NOW)
    assert record["state"] == "pending" and "anchor" not in record
    # 本切片只测执行器已领取后的 transition，不触发后台任务或加密来源解封。
    record.update(state="claimed", claim_id=lease.continuation_claim_id,
        resume_turn_id=lease.turn_id, publication_generation=state.generation)
    return SimpleNamespace(state=state, lease=lease, binding=binding, record=record)


def closed_group(state, call_id="inspect-1", tool=READ, arguments=None):
    call = {"call_id": call_id, "name": tool,
        "arguments": deepcopy(arguments if arguments is not None else
            {"item_ids": ["original"], "options": {"include_size": True}})}
    state.conversation.extend([
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call_id, "name": tool,
            "content": '{"ok":false,"code":"rate_limited"}'},
    ])
    return call


def native_denial(*, effect=ToolEffect.READ, name=READ, reason="quota",
                  due=NOW + 10, in_handler=False):
    """经真实 execute/acquire 边界取异常，不能直接伪造私有 permit。"""
    async def run():
        error = ToolRateLimitError(RateLimitDecision(False, 1, 1, 0, 10, due, reason))
        started = []

        class Limiter:
            async def acquire(self, **_kwargs):
                if not in_handler:
                    raise error

        def handler(_arguments, _context):
            started.append(name)
            if in_handler:
                raise error
            raise AssertionError("被拒限的工具不能进入 handler")

        async def progress(_value):
            pass

        tool = KernelToolSpec(name=name, domain="library", description="离线读取测试",
            input_schema={"type": "object", "properties": {}}, effect=effect,
            read=handler if effect is ToolEffect.READ else None,
            prepare=handler if effect is not ToolEffect.READ else None,
            execute_confirmed=handler if effect is not ToolEffect.READ else None)
        store = InMemorySessionStateStore(clock=lambda: NOW)
        lease, _ = await store.begin_turn(owner="anchor-owner", session_id="anchor-session",
            request_id="native-denial")
        pipeline = ToolPipeline(catalog=ToolCatalog([tool]), state_store=store,
            rate_limiter=Limiter())
        context = ToolCallContext(lease.owner, lease.session_id, lease.request_id,
            lease.turn_id, lease, CancellationToken(), progress)
        with pytest.raises(ToolPipelineError) as caught:
            await pipeline.execute(name, {}, context=context)
        assert started == ([name] if in_handler else [])
        return caught.value

    return asyncio.run(run())


def park(case, permit, call_id="inspect-1", now=NOW):
    apply_actions(case.state, case.lease, (
        _ContinuationAction("", {"operation": "park", "permit": permit,
            "call_id": call_id}, binding=case.binding),
    ), now)


def test_initial_effect_followup_has_no_read_anchor(effect_case):
    case = effect_case
    apply_actions(case.state, case.lease, (
        _ContinuationAction("", {"operation": "reserve", "model_limit": 8,
            "tool_limit": 8, "model_spent": 1}, binding=case.binding),
    ), NOW)
    assert case.record["state"] == "running"
    assert "anchor" not in case.record
    assert "read_wait" not in case.state.metadata


def test_read_permit_has_stable_kernel_type_and_preserves_denial():
    permit = native_denial()
    another = native_denial()
    assert type(permit) is type(another) is _ReadPermitRejected
    assert isinstance(permit, ToolRateLimitError)
    assert type(permit.__cause__) is ToolRateLimitError
    assert permit.rate_limit is permit.__cause__.rate_limit
    assert permit.code == permit.__cause__.code == "rate_limited"
    assert str(permit) == str(permit.__cause__)
    assert permit.tool_name == READ
    with pytest.raises(TypeError):
        operator.setitem(cast(dict[str, object], permit.rate_limit), "remaining", 1)
    with pytest.raises(AttributeError):
        setattr(permit, "rate_limit", {})


def test_native_quota_park_restores_original_arguments(effect_case):
    case = effect_case
    original = deepcopy(closed_group(case.state))
    permit = native_denial()
    assert isinstance(permit, _ReadPermitRejected)
    park(case, permit)
    assert case.record["state"] == "pending"
    assert case.record["anchor"] == anchor_for(case.state, original["call_id"], READ)
    assert case.record["due_at"] == NOW + 10
    assert case.record["budget"]["wait_segments"] == 1
    assert case.record["budget"]["wait_seconds_spent"] == 10
    assert "read_wait" not in case.state.metadata
    # 冷序列化后的 record 仍可恢复参数，返回值不与历史共享可变对象。
    cold = SessionState.from_payload(owner=case.state.owner, session_id=case.state.session_id,
        generation=case.state.generation, payload=case.state.to_payload())
    restored = history_call(cold, cold.metadata["effect_waits"][case.lease.continuation_id]["continuation"])
    assert restored == original
    restored["arguments"]["item_ids"].append("mutated")
    assert history_call(case.state, case.record) == original


def test_repark_replaces_old_anchor(effect_case):
    case = effect_case
    closed_group(case.state)
    park(case, native_denial())
    previous = deepcopy(case.record["anchor"])
    case.record.update(state="running", last_checked_at=NOW + 10)
    call = closed_group(case.state, "inspect-2", "library.stat",
        {"path": "new-item", "fields": ["size"]})
    park(case, native_denial(name="library.stat", due=NOW + 20), "inspect-2", NOW + 10)
    assert case.record["anchor"] != previous
    assert case.record["anchor"] == anchor_for(case.state, "inspect-2", "library.stat")
    assert history_call(case.state, case.record) == call
    assert case.record["budget"]["wait_segments"] == 2
    assert case.record["budget"]["wait_seconds_spent"] == 20


@pytest.mark.parametrize("effect,name,in_handler", [
    (ToolEffect.WRITE, "library.change", False),
    (ToolEffect.DANGER, "library.remove", False),
    (ToolEffect.READ, "agent.inspect", False),
    (ToolEffect.READ, READ, True),
])
def test_real_nonread_utility_or_handler_denial_has_no_resume_authority(
        effect_case, effect, name, in_handler):
    case = effect_case
    closed_group(case.state, tool=name)
    permit = native_denial(effect=effect, name=name, in_handler=in_handler)
    assert not isinstance(permit, _ReadPermitRejected)
    before = deepcopy(case.record)
    with pytest.raises(ValueError, match="untrusted READ rejection"):
        park(case, permit)
    assert case.record == before
    assert "anchor" not in case.record


@pytest.mark.parametrize("reason", ["concurrency", "invalid_cost"])
def test_native_nonquota_rejection_has_no_resume_authority(effect_case, reason):
    case = effect_case
    closed_group(case.state)
    permit = native_denial(reason=reason)
    assert isinstance(permit, _ReadPermitRejected)
    before = deepcopy(case.record)
    with pytest.raises(ValueError, match="untrusted READ rejection"):
        park(case, permit)
    assert case.record == before


@pytest.mark.parametrize("due", [None, True, float("nan"), float("inf"),
    float("-inf"), NOW - 1, NOW + 300])
def test_invalid_retry_deadline_has_no_resume_authority(effect_case, due):
    case = effect_case
    closed_group(case.state)
    before = deepcopy(case.record)
    with pytest.raises(ValueError, match="wait budget exhausted"):
        park(case, native_denial(due=due))
    assert case.record == before


def test_rate_limit_projection_is_not_a_native_permit(effect_case):
    case = effect_case
    closed_group(case.state)
    fake = SimpleNamespace(tool_name=READ,
        rate_limit={"reason": "quota", "retry_at_epoch": NOW + 10})
    before = deepcopy(case.record)
    with pytest.raises(ValueError, match="untrusted READ rejection"):
        park(case, fake)
    assert case.record == before


@pytest.mark.parametrize("invalid_history", ["missing_call", "open_group", "duplicate_call", "no_user"])
def test_invalid_native_history_cannot_park_or_spend_budget(effect_case, invalid_history):
    case = effect_case
    closed_group(case.state)
    if invalid_history == "missing_call":
        case.state.conversation = case.state.conversation[:1]
    elif invalid_history == "open_group":
        case.state.conversation.pop()
    elif invalid_history == "duplicate_call":
        closed_group(case.state)
    else:
        case.state.conversation.pop(0)
    before = deepcopy(case.record)
    with pytest.raises(ValueError, match="read wait"):
        park(case, native_denial())
    assert case.record == before


@pytest.mark.parametrize("changed", ["arguments", "user", "result"])
def test_history_changes_invalidate_saved_anchor(effect_case, changed):
    case = effect_case
    call = closed_group(case.state)
    park(case, native_denial())
    if changed == "arguments":
        call["arguments"]["item_ids"] = ["replacement"]
    elif changed == "user":
        case.state.conversation[0]["content"] = "另一个目标"
    else:
        case.state.conversation[-1]["content"] = '{"ok":true}'
    with pytest.raises(ValueError, match="history changed"):
        history_call(case.state, case.record)
