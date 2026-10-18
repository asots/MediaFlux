"""Behavioral controls for R0; these pass before the deduplication.

capture_session_contract is also used by the offline, fixed-base differential
probe. No test constructs a second runner or requires Git/history at runtime.
"""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import asdict
from itertools import count
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.agent.domains.resources.presentation import ResourceSelection
from app.agent.domains.resources.selection import ValidatedSelection
from app.agent.errors import ToolPipelineError
from app.agent.kernel.capabilities import (
    CapabilityRetriever,
    KernelToolSpec,
    ToolCatalog,
    ToolEffect,
)
from app.agent.kernel.effects import PreparedEffect
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports.existing_actions import adapt_tool_spec
from app.agent.kernel.session import AgentSession, SessionLimits
from app.agent.kernel.session_state import CandidateSelectionGuard, SessionState
from app.agent.kernel.state import InMemorySessionStateStore
from app.agent.kernel.state_types import AgentInput, StateUpdate
from app.agent.models import RiskLevel, ToolResult, ToolSpec
from tests.test_agent_kernel_core import ScriptedModel

SESSION_CASES = (
    "read_alias", "unavailable", "invalid_alias", "pipeline_failure",
    "approval_siblings", "over_budget", "selection",
)


def _normalize(value: Any) -> Any:
    """Ignore only wall-clock/duration values, never field presence or order."""
    if isinstance(value, dict):
        return {
            key: (f"<{type(child).__name__}>" if key in {"elapsed_ms", "expires_at"}
                  else _normalize(child))
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(child) for child in value]
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, str) and value.startswith("{"):
        try:
            return _normalize(json.loads(value))
        except ValueError:
            pass
    return value


async def capture_session_contract(case):
    timeline = []
    executions = []

    class RecordingStore(InMemorySessionStateStore):
        async def commit(self, lease, *, conversation=None, updates=()):
            result = await super().commit(lease, conversation=conversation, updates=updates)
            timeline.append(["commit", asdict(result)])
            return result

    class Journal:
        async def append(self, event, *, owner):
            current = await store.load(owner=owner, session_id=event.session_id)
            timeline.append([event.type.value, deepcopy(dict(event.payload)),
                             deepcopy(current.conversation)])

    def read(arguments, context):
        executions.append(["read", dict(arguments)])
        if case == "pipeline_failure":
            raise ToolPipelineError("查询失败", code="lookup_failed")
        return {"ok": True, "status": "success", "summary": "读取完成", "data": {"count": 3}}

    def prepare(arguments, context):
        executions.append(["prepare", dict(arguments)])
        return PreparedEffect(
            preview={"summary": "准备写入"}, snapshot_fingerprint="r0-snapshot",
            metadata={"purpose": "compatibility"},
        )

    def execute(arguments, snapshot, context):
        executions.append(["execute", dict(arguments)])
        raise AssertionError("No scenario authorizes writes")

    schema = {"type": "object", "properties": {"query": {"type": "string"}},
              "required": ["query"], "additionalProperties": False}
    reader = KernelToolSpec(
        name="library.inspect", model_name="inspect_alias", domain="library",
        description="读取状态", input_schema=schema, effect=ToolEffect.READ, read=read,
    )
    writer = KernelToolSpec(
        name="ingest.submit", domain="library", description="准备写入",
        input_schema={"type": "object", "properties": {}}, effect=ToolEffect.WRITE,
        prepare=prepare, execute_confirmed=execute,
    )
    catalog = ToolCatalog([reader, writer], selection=ResourceSelection())
    store = RecordingStore()
    calls = [ModelToolCall("call-1", "inspect_alias", {"query": "fixture"})]
    if case == "unavailable":
        calls = [ModelToolCall("call-1", "not_selected", {})]
    elif case == "invalid_alias":
        calls = [ModelToolCall("call-1", "inspect_alias", {})]
    elif case in {"approval_siblings", "over_budget"}:
        calls = [ModelToolCall("call-1", "ingest__submit", {}),
                 ModelToolCall("call-2", "inspect_alias", {"query": "fixture"})]
    model = ScriptedModel([
        [*[ModelEvent(ModelEventType.TOOL_CALL_COMPLETED, tool_call=call) for call in calls],
         ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")],
        [ModelEvent(ModelEventType.TEXT_DELTA, text="已核对。"),
         ModelEvent(ModelEventType.FINISH, finish_reason="stop")],
    ])
    pipeline = ToolPipeline(catalog=catalog, state_store=store)
    session = AgentSession(
        model=model, catalog=catalog, pipeline=pipeline, state_store=store,
        journal=Journal(), retriever=CapabilityRetriever(), system_prompt="R0 compatibility fixture",
        limits=SessionLimits(max_model_rounds=2, max_tool_calls=1 if case == "over_budget" else 4),
    )
    metadata = {}
    # Session-only selection control: admission retains the real generation/ref
    # guard, while reference validation is covered by test_agent_ux_backend.
    selection = ValidatedSelection(
        CandidateSelectionGuard(1, "ref_fixture", 4_000_000_000.0),
        {"source_type": "resource_candidates", "positions": [1], "target": "guangya"},
    )
    serial = count(1)
    with patch("secrets.token_urlsafe", side_effect=lambda *args: f"fixture_{next(serial):024d}"), \
            patch("app.agent.domains.resources.selection.validate_selection", new=AsyncMock(return_value=selection)):
        if case == "selection":
            lease, _ = await store.begin_turn(owner="owner", session_id="session", request_id="seed")
            await store.commit(lease, updates=(StateUpdate("metadata.ux_candidate_view", {
                "generation": 1, "ref": "ref_fixture", "expires_at": 4_000_000_000.0,
            }),))
            timeline.clear()
            metadata["selection"] = {"fixture": True}
        events = [event async for event in session.run(AgentInput(
            message="读取状态并准备写入", owner="owner", session_id="session", request_id="r0",
            metadata=metadata,
        ))]
    state = await store.load(owner="owner", session_id="session")
    return _normalize({
        "events": [{"type": event.type.value, "sequence": event.sequence,
                    "payload": dict(event.payload)} for event in events],
        "requests": [asdict(request) for request in model.requests],
        "state": asdict(state), "timeline": timeline, "executions": executions,
    })


def _assert_paired(messages):
    calls = [call["call_id"] for row in messages for call in row.get("tool_calls", [])]
    results = [row["tool_call_id"] for row in messages if row["role"] == "tool"]
    assert results == calls
    assert len(set(results)) == len(results)


@pytest.mark.parametrize("case", SESSION_CASES)
def test_session_closure_preserves_protocol_events_and_persistence(case):
    result = asyncio.run(capture_session_contract(case))
    events = result["events"]
    tool_events = [event for event in events if event["type"].startswith(("tool.", "effect."))]
    expected = {
        "read_alias": ["tool.started", "tool.completed"],
        "unavailable": ["tool.failed"],
        "invalid_alias": ["tool.started", "tool.failed"],
        "pipeline_failure": ["tool.started", "tool.failed"],
        "approval_siblings": ["effect.preview_started", "tool.started", "effect.approval_required", "tool.failed"],
        "over_budget": ["tool.failed", "tool.failed"],
        "selection": ["tool.started", "effect.approval_required"],
    }
    assert [event["type"] for event in tool_events] == expected[case]
    assert events[-1]["type"] in {"turn.completed", "turn.failed"}
    assert [item[0] for item in result["executions"]] == {
        "read_alias": ["read"], "pipeline_failure": ["read"],
        "approval_siblings": ["prepare"], "selection": ["prepare"],
    }.get(case, [])
    _assert_paired(result["state"]["conversation"])
    for request in result["requests"]:
        _assert_paired(request["messages"])
    for entry in result["timeline"]:
        _assert_paired(entry[1]["conversation"] if entry[0] == "commit" else entry[2])
    for event in tool_events:
        payload = event["payload"]
        assert {"call_id", "tool", "label"} <= payload.keys()
        if event["type"] == "effect.approval_required":
            assert "plan" in payload and "elapsed_ms" not in payload
        if event["type"] == "tool.completed":
            assert "elapsed_ms" in payload and "plan" not in payload
            published = next(row for row in result["timeline"] if row[0] == "tool.completed")
            assert any(row.get("tool_call_id") == payload["call_id"] for row in published[2])
    if case == "read_alias":
        assert all(event["payload"]["tool"] == "library.inspect" for event in tool_events)
        assert result["requests"][1]["messages"][-1]["tool_name"] == "inspect_alias"
    if case == "invalid_alias":
        payload = result["requests"][1]["messages"][-1]["content"]
        assert payload["expected_input_schema"]["required"] == ["query"]
        assert tool_events[-1]["payload"]["tool"] == "library.inspect"
        assert "expected_input_schema" not in tool_events[-1]["payload"]
    if case == "approval_siblings":
        assert tool_events[-1]["payload"]["code"] == "not_executed_after_approval"
        assert tool_events[-1]["payload"]["tool"] == "inspect_alias"
    if case == "over_budget":
        assert {event["payload"]["code"] for event in tool_events} == {"tool_budget_exceeded"}
        assert events[-1]["payload"]["tool_calls"] == 0
    if case == "selection":
        assert result["requests"] == []
        assert events[-1]["payload"]["model_calls"] == 0
        assert events[-1]["payload"]["tool_calls"] == 1
        assert events[-1]["payload"]["answer"] == ""


@pytest.mark.parametrize("risk,confirmation,effect", [
    (RiskLevel.READ, False, ToolEffect.READ),
    (RiskLevel.READ, True, ToolEffect.WRITE),
    (RiskLevel.WRITE, True, ToolEffect.WRITE),
    (RiskLevel.DANGER, True, ToolEffect.DANGER),
])
@pytest.mark.parametrize("freshness", ["live", "snapshot"])
@pytest.mark.parametrize("parallel_read", [False, True])
def test_adapted_specs_keep_shared_fields_costs_and_disjoint_handlers(risk, confirmation, effect, freshness, parallel_read):
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    value = ToolResult(True, "success", "兼容对照")
    spec = ToolSpec(
        name="fixture.inspect", model_name="fixture_alias", description="兼容字段",
        risk=risk, requires_confirmation=confirmation, parameters=schema,
        validator=lambda arguments: arguments, handler=lambda arguments: value,
        runtime_status=lambda: {"status": "ready"}, domains=("Custom Domain",),
        context_confirmation_preparer=lambda arguments, context: (value, "snapshot"),
        context_confirmed_handler=lambda arguments, snapshot, context: value,
        examples=("例子",), source_kind="fixture_source", freshness=freshness, parallel_read=parallel_read,
        workflow="fixture_workflow", workflow_stage=3,
        related_tools=("fixture.next", "fixture.next", " "),
    )
    adapted = adapt_tool_spec(spec)
    assert adapted.name == spec.name and adapted.model_name == spec.model_name
    assert adapted.description == spec.description and adapted.domain == "custom_domain"
    assert adapted.examples == spec.examples and adapted.effect is effect
    assert adapted.input_schema is schema and adapted.validator is spec.validator
    assert adapted.runtime_status is spec.runtime_status
    assert adapted.metadata == {
        "source_kind": "fixture_source", "freshness": freshness, "parallel_read": parallel_read,
        "workflow": "fixture_workflow", "workflow_stage": 3,
        "related_tools": ("fixture.next",),
        "retrieval_terms": ("Custom Domain", "fixture_source"), "risk": risk.value,
    }
    if effect is ToolEffect.READ:
        assert adapted.cost == (1.25 if freshness == "live" else 1.0)
        assert adapted.read is not None
        assert adapted.prepare is None and adapted.execute_confirmed is None and adapted.verify is None
        assert adapted.read({}, None) is value
    else:
        assert adapted.cost == (2.0 if effect is ToolEffect.DANGER else 1.5)
        assert adapted.read is None
        assert adapted.prepare is not None and adapted.execute_confirmed is not None and adapted.verify is not None


@pytest.mark.parametrize("mutate_clone", [True, False])
def test_state_clone_keeps_all_fields_and_isolates_nested_mutations(mutate_clone):
    original = SessionState(
        owner="owner", session_id="session", generation=7, summary="summary",
        pending_effect_plan_id="plan", recent_refs=["ref_a"], ref_kinds={"resource"},
        conversation=[{"role": "tool", "content": "fact", "nested": {"items": [1]}}],
        metadata={"nested": {"items": [{"values": {"old"}}]}},
    )
    baseline = deepcopy(original)
    cloned = original.clone()
    assert cloned == original and cloned is not original
    changed, untouched = (cloned, original) if mutate_clone else (original, cloned)
    changed.conversation[0]["nested"]["items"].append(2)
    changed.metadata["nested"]["items"][0]["values"].add("new")
    changed.recent_refs.append("ref_b")
    changed.ref_kinds.add("result")
    changed.generation += 1
    changed.pending_effect_plan_id = "new-plan"
    assert untouched == baseline
    assert changed != baseline
