"""R1 compatibility controls: green on the original defaults, not feature RED."""
from __future__ import annotations

import asyncio
import inspect
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from app.agent.confirmation import ConfirmationStore
from app.agent.errors import ToolPipelineError
from app.agent.kernel.capabilities import KernelToolSpec, ToolCatalog, ToolEffect
from app.agent.kernel.effects import (
    ConfirmationEffectPlanStore,
    EffectPlan,
    PreparedEffect,
)
from app.agent.kernel.pipeline import EffectLifecycle, NoopEffectLifecycle, ToolPipeline
from app.agent.kernel.ports.mediaflux_effects import MediaFluxEffectLifecycle
from app.agent.kernel.state import InMemorySessionStateStore
from tests.test_agent_kernel_safety import _context

SIGNATURES = {
    "prepared": ([('tool', 'KernelToolSpec'), ('arguments', 'Mapping[str, Any]'),
                  ('prepared', 'PreparedEffect'), ('context', 'ToolCallContext')], 'PreparedEffect'),
    "prepare_failed": ([('prepared', 'PreparedEffect'), ('context', 'ToolCallContext')], 'None'),
    "completed": ([('plan', 'EffectPlan'), ('value', 'Any'), ('elapsed_ms', 'int')], 'None'),
    "failed": ([('plan', 'EffectPlan'), ('code', 'str'), ('elapsed_ms', 'int')], 'None'),
    "interrupted": ([('plan', 'EffectPlan')], 'None'),
    "cancelled": ([('plan', 'EffectPlan')], 'None'),
}


class RecordingLifecycle:
    """Intentionally structural: no Protocol/Noop base and no dynamic dispatch."""

    def __init__(self):
        self.calls = []

    def prepared(self, *, tool, arguments, prepared, context):
        self.calls.append(("prepared", dict(tool=tool, arguments=arguments,
                                            prepared=prepared, context=context)))
        return replace(prepared, metadata={**prepared.metadata, "injected": True})

    def prepare_failed(self, *, prepared, context):
        self.calls.append(("prepare_failed", dict(prepared=prepared, context=context)))

    def completed(self, *, plan, value, elapsed_ms):
        self.calls.append(("completed", dict(plan=plan, value=value, elapsed_ms=elapsed_ms)))

    def failed(self, *, plan, code, elapsed_ms):
        self.calls.append(("failed", dict(plan=plan, code=code, elapsed_ms=elapsed_ms)))

    def interrupted(self, *, plan):
        self.calls.append(("interrupted", dict(plan=plan)))

    def cancelled(self, *, plan):
        self.calls.append(("cancelled", dict(plan=plan)))


def _fixture(lifecycle=None, *, terminal="completed"):
    prepared = PreparedEffect(
        preview={"summary": "fixture preview", "data": {"nested": [1, {"a": [2]}]}},
        snapshot_fingerprint="r1-snapshot",
        arguments={"nested": [{"a": [3]}]}, metadata={"nested": [{"b": [4]}]},
    )
    value = {"ok": True, "status": "completed", "summary": "fixture complete",
             "data": {"nested": [{"c": [5]}]}}

    async def execute(arguments, snapshot, context):
        if terminal == "failed":
            raise ToolPipelineError("fixture failure", code="fixture_failed")
        if terminal == "interrupted":
            raise asyncio.CancelledError("fixture interruption")
        return value

    tool = KernelToolSpec(
        name="fixture.write", domain="fixture", description="offline lifecycle control",
        input_schema={"type": "object", "properties": {}}, effect=ToolEffect.WRITE,
        prepare=lambda arguments, context: prepared, execute_confirmed=execute,
    )
    state = InMemorySessionStateStore()
    pipeline = ToolPipeline(
        catalog=ToolCatalog([tool]), state_store=state,
        effect_store=ConfirmationEffectPlanStore(ConfirmationStore()),
        effect_lifecycle=lifecycle,
    )
    return pipeline, tool, prepared, value


@pytest.mark.parametrize("name", SIGNATURES)
def test_six_hook_signatures_remain_explicit_keyword_only(name):
    expected, result = SIGNATURES[name]
    for cls in (EffectLifecycle, NoopEffectLifecycle):
        signature = inspect.signature(getattr(cls, name))
        parameters = list(signature.parameters.values())
        assert parameters[0].name == "self"
        assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameters[0].annotation is inspect.Parameter.empty
        assert [(p.name, p.annotation) for p in parameters[1:]] == expected
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters[1:])
        assert all(p.default is inspect.Parameter.empty for p in parameters)
        assert signature.return_annotation == result


def test_protocol_is_not_instantiable_and_default_pipeline_uses_named_noop():
    with pytest.raises(TypeError, match="Protocols cannot be instantiated"):
        EffectLifecycle()
    pipeline, _, _, _ = _fixture()
    assert type(pipeline.effect_lifecycle) is NoopEffectLifecycle
    assert type(NoopEffectLifecycle()).__name__ == "NoopEffectLifecycle"
    assert EffectLifecycle is not NoopEffectLifecycle


def test_noop_returns_identity_and_never_mutates_nested_payloads():
    async def check():
        pipeline, tool, prepared, value = _fixture()
        context = await _context(pipeline.state_store)
        from app.agent.kernel.error_repair import SchemaReplacement
        context.schema_replacements["fixture"] = SchemaReplacement((), tool.name, {("nested",): {"keep": [True]}})
        context = replace(context, selection_arguments={"nested": [{"d": [6]}]})
        arguments = {"nested": [{"e": [7]}]}
        plan = EffectPlan(
            plan_id="r1-plan", owner=context.owner, session_id=context.session_id,
            generation=context.lease.generation, tool_name=tool.name, effect=tool.effect,
            arguments=arguments, snapshot_fingerprint=prepared.snapshot_fingerprint,
            preview=prepared.preview, expires_at=0, metadata=prepared.metadata,
            confirmation_contract={"nested": [{"f": [8]}]}, public_result=value,
        )
        payloads = (arguments, prepared, plan, value, context.selection_arguments,
                    context.schema_replacements, tool.input_schema, tool.metadata)
        before = deepcopy(payloads)
        lifecycle = pipeline.effect_lifecycle
        assert lifecycle.prepared(tool=tool, arguments=arguments,
                                  prepared=prepared, context=context) is prepared
        assert lifecycle.prepare_failed(prepared=prepared, context=context) is None
        assert lifecycle.completed(plan=plan, value=value, elapsed_ms=3) is None
        assert lifecycle.failed(plan=plan, code="fixture", elapsed_ms=3) is None
        assert lifecycle.interrupted(plan=plan) is None
        assert lifecycle.cancelled(plan=plan) is None
        assert payloads == before
    asyncio.run(check())


@pytest.mark.parametrize("behavior", ["exception", "wrong_type", "replacement", "cancel"])
def test_prepared_boundary_keeps_fallback_and_baseexception_behavior(behavior):
    async def check():
        lifecycle = RecordingLifecycle()
        pipeline, tool, prepared, _ = _fixture(lifecycle)
        context = await _context(pipeline.state_store)
        replacement = replace(prepared, metadata={"replacement": True})
        with patch.object(lifecycle, "prepared") as hook:
            if behavior == "exception":
                hook.side_effect = RuntimeError("auxiliary failure")
            elif behavior == "cancel":
                hook.side_effect = asyncio.CancelledError("do not swallow")
            else:
                hook.return_value = None if behavior == "wrong_type" else replacement
            if behavior == "cancel":
                with pytest.raises(asyncio.CancelledError, match="do not swallow"):
                    pipeline._effect_prepared(tool=tool, arguments={}, prepared=prepared, context=context)
            else:
                result = pipeline._effect_prepared(tool=tool, arguments={}, prepared=prepared, context=context)
                assert result is (replacement if behavior == "replacement" else prepared)
            hook.assert_called_once_with(tool=tool, arguments={}, prepared=prepared, context=context)
    asyncio.run(check())


@pytest.mark.parametrize("name", [name for name in SIGNATURES if name != "prepared"])
@pytest.mark.parametrize("exception", [RuntimeError, asyncio.CancelledError])
def test_notification_boundary_isolates_exception_not_cancellation(name, exception):
    lifecycle = RecordingLifecycle()
    pipeline, _, _, _ = _fixture(lifecycle)
    payload = {key: object() for key, _ in SIGNATURES[name][0]}
    with patch.object(lifecycle, name, side_effect=exception("auxiliary failure")) as hook:
        if exception is asyncio.CancelledError:
            with pytest.raises(asyncio.CancelledError, match="auxiliary failure"):
                pipeline._notify_effect(name, **payload)
        else:
            assert pipeline._notify_effect(name, **payload) is None
        hook.assert_called_once_with(**payload)


@pytest.mark.parametrize("kind", ["structural", "mediaflux"])
@pytest.mark.parametrize("terminal", ["prepare_failed", "completed", "failed", "interrupted", "cancelled"])
def test_injected_hooks_run_on_real_pipeline_paths(kind, terminal):
    async def check():
        runtime = Mock()
        lifecycle = RecordingLifecycle() if kind == "structural" else MediaFluxEffectLifecycle(
            missing_media_runtime=runtime,
        )
        pipeline, _, prepared, value = _fixture(lifecycle, terminal=terminal)
        assert pipeline.effect_lifecycle is lifecycle
        context = await _context(pipeline.state_store)
        module = "app.agent.kernel.ports.mediaflux_effects"
        with ExitStack() as stack:
            # Real MediaFlux hooks run; only domain services are isolated.
            stack.enter_context(patch(f"{module}.config.all_items", return_value={}))
            stack.enter_context(patch(f"{module}.compensation_candidate", return_value=None))
            audit_failed = stack.enter_context(patch(f"{module}.record_confirmation_error"))
            audit_interrupted = stack.enter_context(patch(f"{module}.record_confirmation_interrupted"))
            hooks = {name: stack.enter_context(patch.object(lifecycle, name, wraps=getattr(lifecycle, name)))
                     for name in SIGNATURES}
            if terminal == "prepare_failed":
                stack.enter_context(patch.object(pipeline.effect_store, "freeze", side_effect=ValueError("freeze failed")))
                with pytest.raises(ValueError, match="freeze failed"):
                    await pipeline.execute("fixture.write", {}, context=context)
                plan = None
            else:
                preview = await pipeline.execute("fixture.write", {}, context=context)
                plan = preview.effect_plan
                assert plan is not None
                if kind == "structural":
                    assert plan.metadata["injected"] is True
                else:
                    assert plan.metadata["configuration_fingerprint"]
                if terminal == "cancelled":
                    assert await pipeline.cancel_effect(plan.plan_id, lease=context.lease)
                elif terminal == "failed":
                    with pytest.raises(ToolPipelineError) as caught:
                        await pipeline.execute_confirmed(plan.plan_id, context=context)
                    assert caught.value.code == "fixture_failed"
                elif terminal == "interrupted":
                    with pytest.raises(asyncio.CancelledError, match="fixture interruption"):
                        await pipeline.execute_confirmed(plan.plan_id, context=context)
                else:
                    result = await pipeline.execute_confirmed(plan.plan_id, context=context)
                    assert result.outcome.public_content["ok"] is True
            assert hooks["prepared"].call_count == 1
            assert hooks["prepared"].call_args.kwargs["prepared"] is prepared
            hooks[terminal].assert_called_once()
            for name, hook in hooks.items():
                if name not in {"prepared", terminal}:
                    hook.assert_not_called()
            payload = hooks[terminal].call_args.kwargs
            if plan is not None:
                assert payload["plan"].plan_id == plan.plan_id
            if terminal == "completed":
                assert payload["value"] is value
                assert isinstance(payload["elapsed_ms"], int)
            if kind == "structural":
                assert [name for name, _ in lifecycle.calls] == ["prepared", terminal]
            elif terminal == "completed":
                runtime.complete_submission.assert_called_once_with(
                    owner=context.owner, tool_name="fixture.write", result=value, followups=None,
                )
            else:
                runtime.release_followups.assert_called_once_with(owner=context.owner, followups=None)
                if terminal == "failed":
                    assert audit_failed.call_args.kwargs["code"] == "fixture_failed"
                elif terminal == "interrupted":
                    audit_interrupted.assert_called_once()
    asyncio.run(check())
