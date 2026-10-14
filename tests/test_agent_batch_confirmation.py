"""F16：真实 Kernel/SQLite 确认链路，领域写入仅使用内存 handler。"""

import asyncio
import json

import pytest

from app.agent.confirmation import SQLiteConfirmationStore
from app.agent.domain_catalog import build_tool_specs
from app.agent.kernel.adapters import TurnViewBuilder
from app.agent.kernel.capabilities import CapabilityRetriever, KernelToolSpec, ToolCatalog, ToolEffect
from app.agent.kernel.effects import ConfirmationEffectPlanStore, PreparedEffect
from app.agent.kernel.events import AgentEventType as Event
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.pipeline import ToolPipeline, ToolPipelineError
from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import AgentInput
from app.agent.models import RiskLevel, ToolResult, ToolSpec
from tests.test_agent_kernel_core import ScriptedModel, collect
from tests.test_agent_task_cas import task_store as isolated_store

task_store = isolated_store

OWNER, SESSION = "batch-regression-owner", "batch-regression-session"
NAMES = ("test.watchlist", "test.preferences", "test.third", "test.fourth")
BACKGROUND = "test.background"
FIELDS = dict(zip((*NAMES, BACKGROUND), ("item", "quality", "label", "tag", "path")))
FINAL = [ModelEvent(ModelEventType.TEXT_DELTA, text="请按系统回执核对结果。"),
         ModelEvent(ModelEventType.FINISH, finish_reason="stop")]


def step(name):
    return {"tool": name, "arguments": {FIELDS.get(name, "item"): f" {name} "}}


def batch(*names):
    return {"steps": [step(name) for name in names]}


def call(name, arguments):
    return [ModelEvent(ModelEventType.TOOL_CALL_COMPLETED,
                       tool_call=ModelToolCall("regression-call", name, arguments)),
            ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")]


async def capture(stream):
    events = await collect(stream)
    builder = TurnViewBuilder()
    for event in events:
        builder.apply(event)
    return builder.build(), events


async def confirm(session, plan_id):
    return await capture(session.confirm(owner=OWNER, session_id=SESSION, plan_id=plan_id))


class BatchRig:
    def __init__(self, store):
        self.store = store
        self.values = dict.fromkeys(FIELDS, "before")
        self.writes, self.attempted, self.verified, self.trace = [], [], [], []
        self.fail_at, self.fail_verify, self.denied = set(), set(), set()

    def tool(self, name):
        field = FIELDS[name]

        def normalized(arguments):
            return {field: arguments[field].strip()}

        def snapshot(arguments):
            return json.dumps([name, self.values[name], normalized(arguments)], sort_keys=True)

        def prepare(arguments, _context):
            self.trace.append(("prepare", name))
            return PreparedEffect(
                preview=ToolResult(True, "preview", f"将写入 {name}",
                                   data={"target": normalized(arguments)}).to_dict(),
                arguments=normalized(arguments), snapshot_fingerprint=snapshot(arguments))

        def execute(arguments, expected, _context):
            self.attempted.append(name)
            self.trace.append(("execute", name))
            if expected != snapshot(arguments):
                raise ToolPipelineError("领域快照已变化", code="confirmation_stale")
            if name in self.fail_at:
                raise ToolPipelineError("注入领域失败", code="fixture_failure")
            self.values[name] = arguments[field]
            self.writes.append((name, dict(arguments)))
            return ToolResult(True, "completed", f"已写入 {name}",
                              data={"receipt": f"receipt:{name}", "value": self.values[name]})

        def verify(arguments, result, _context):
            self.verified.append(name)
            self.trace.append(("verify", name))
            assert self.values[name] == arguments[field]
            return False if name in self.fail_verify else result

        return KernelToolSpec(
            name=name, domain="test", description=f"同步组合回归操作 {name}",
            effect=ToolEffect.WRITE,
            input_schema={"type": "object", "required": [field], "additionalProperties": False,
                          "properties": {field: {"type": "string", "minLength": 1}}},
            prepare=prepare, execute_confirmed=execute, verify=verify,
            authorize=lambda _context: name not in self.denied,
            metadata={"batch_confirmation": name != BACKGROUND})

    def session(self, rounds, store=None):
        store = store if store is not None else self.store
        catalog = ToolCatalog([self.tool(name) for name in FIELDS])
        pipeline = ToolPipeline(catalog=catalog, state_store=store,
                                effect_store=ConfirmationEffectPlanStore(SQLiteConfirmationStore()))
        return AgentSession(model=ScriptedModel(rounds), catalog=catalog,
                            retriever=CapabilityRetriever(), pipeline=pipeline, state_store=store)

    async def start(self, arguments=None, *, tool="agent.batch", followups=None):
        arguments = batch(*NAMES[:2]) if arguments is None else arguments
        session = self.session([call(tool, arguments), *(followups or [FINAL])])
        view, events = await capture(session.run(AgentInput(
            message=f"执行 {tool} 同步组合回归操作，确认前不得写入。", owner=OWNER, session_id=SESSION)))
        return session, view, events


@pytest.fixture
def rig(task_store, monkeypatch):
    monkeypatch.setattr("app.modules.web_secret.get_web_secret", lambda: "batch-test-only")
    return BatchRig(task_store)


def test_production_whitelist_and_strict_real_subschemas(rig):
    default = ToolSpec(name="test.default", description="默认不组合", risk=RiskLevel.WRITE,
                       parameters={}, validator=dict)
    assert default.batch_confirmation is False
    specs = build_tool_specs()
    allowed = {"discovery.add_watchlist", "media.set_preferences"}
    assert {spec.name for spec in specs if spec.batch_confirmation} == allowed
    catalog = catalog_from_tool_specs(specs)
    pipeline = ToolPipeline(catalog=catalog, state_store=rig.store)
    assert {tool.name for tool in catalog.visible() if tool.metadata.get("batch_confirmation")} == allowed
    schema = pipeline.catalog.get("agent.batch").input_schema
    steps = schema["properties"]["steps"]
    assert schema["required"] == ["steps"] and schema["additionalProperties"] is False
    assert steps["minItems"] == 2 and 2 <= steps["maxItems"] <= 4
    branches = steps["items"]["oneOf"]
    assert len(branches) == 2
    for branch in branches:
        assert set(branch["required"]) == {"tool", "arguments"}
        assert branch["additionalProperties"] is False
        name, = branch["properties"]["tool"]["enum"]
        assert branch["properties"]["arguments"] == catalog.get(name).input_schema


def test_one_card_two_writes_and_single_use_ticket(rig):
    async def run():
        session, preview, events = await rig.start()
        assert preview.status == "approval_required" and preview.approval.tool_name == "agent.batch"
        assert preview.approval.confirmation["risk"] == "write"
        assert sum(e.type is Event.EFFECT_APPROVAL_REQUIRED for e in events) == 1
        assert [s["tool"] for s in preview.approval.preview["data"]["steps"]] == list(NAMES[:2])
        assert not rig.writes and not rig.attempted and not rig.verified
        final, events = await confirm(session, preview.approval.plan_id)
        assert final.status == "success" and final.approval is None
        assert sum(e.type is Event.EFFECT_COMPLETED for e in events) == 1
        assert not any(e.type is Event.EFFECT_APPROVAL_REQUIRED for e in events)
        assert rig.writes == [(n, {FIELDS[n]: n}) for n in NAMES[:2]]
        assert rig.verified == list(NAMES[:2])
        assert rig.trace[:4] == [("prepare", n) for n in (*NAMES[:2], *NAMES[:2])]
        data = final.effect_result["data"]
        assert data["completed"] == data["total"] == 2
        assert [s["result"]["data"]["receipt"] for s in data["steps"]] == [f"receipt:{n}" for n in NAMES[:2]]
        replay, _ = await confirm(session, preview.approval.plan_id)
        assert replay.status == "failed"
        assert rig.attempted == rig.verified == list(NAMES[:2]) and len(rig.writes) == 2
    asyncio.run(run())


def test_cancelled_batch_never_writes(rig):
    async def run():
        session, preview, _ = await rig.start()
        assert await session.cancel_effect(owner=OWNER, session_id=SESSION, plan_id=preview.approval.plan_id)
        assert not await session.cancel_effect(owner=OWNER, session_id=SESSION, plan_id=preview.approval.plan_id)
        result, _ = await confirm(session, preview.approval.plan_id)
        assert result.status == "failed"
        assert not rig.writes and not rig.attempted and not rig.verified
    asyncio.run(run())


@pytest.mark.parametrize("change", ["snapshot", "authorization"])
def test_second_step_preflight_failure_means_zero_writes(rig, change):
    async def run():
        session, preview, _ = await rig.start()
        if change == "snapshot":
            rig.values[NAMES[1]] = "external-change"
        else:
            rig.denied.add(NAMES[1])
        _, events = await confirm(session, preview.approval.plan_id)
        code = "confirmation_stale" if change == "snapshot" else "authorization_denied"
        assert any(e.type is Event.EFFECT_FAILED and e.payload.get("code") == code for e in events)
        assert not rig.writes and not rig.attempted and not rig.verified
        await confirm(session, preview.approval.plan_id)
        assert not rig.attempted
    asyncio.run(run())


@pytest.mark.parametrize("arguments", [
    batch(), batch(NAMES[0]), batch(*NAMES, NAMES[0]),
    batch(NAMES[0], NAMES[0]), batch(NAMES[0], BACKGROUND), batch(*NAMES[:2], BACKGROUND),
    batch(NAMES[0], "test.unknown"), batch(NAMES[0], "agent.batch"),
    {"steps": [step(NAMES[0]), {"tool": NAMES[1], "arguments": {"item": "wrong-schema"}}]},
    {"steps": [step(NAMES[0]), {"tool": NAMES[1]}]},
    {"steps": [step(NAMES[0]), {**step(NAMES[1]), "extra": True}]},
    {"steps": [step(NAMES[0]), {"tool": NAMES[1], "arguments": {"quality": True}}]},
    {**batch(*NAMES[:2]), "extra": True},
])
def test_invalid_or_unsupported_batch_is_rejected_before_writes(rig, arguments):
    async def run():
        _, view, events = await rig.start(arguments)
        assert view.approval is None
        assert any(e.type is Event.TOOL_FAILED and e.payload.get("code") == "invalid_arguments" for e in events)
        assert not rig.writes and not rig.attempted and not rig.verified
        assert await rig.session([]).pipeline.get_pending_plan(owner=OWNER, session_id=SESSION) is None
    asyncio.run(run())


def test_four_distinct_tools_fit_one_confirmation(rig):
    async def run():
        session, preview, _ = await rig.start(batch(*NAMES))
        assert preview.approval is not None and not rig.writes
        final, _ = await confirm(session, preview.approval.plan_id)
        assert final.effect_result["data"]["completed"] == 4
        assert rig.attempted == rig.verified == list(NAMES)
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["execute", "verify"])
def test_second_step_failure_keeps_first_receipt_and_stops_third(rig, failure):
    async def run():
        session, preview, _ = await rig.start(batch(*NAMES[:3]))
        (rig.fail_at if failure == "execute" else rig.fail_verify).add(NAMES[1])
        final, _ = await confirm(session, preview.approval.plan_id)
        data = final.effect_result["data"]
        assert final.status == "partial" and data["completed"] == 1 and data["total"] == 3
        assert rig.attempted == list(NAMES[:2])
        assert rig.values[NAMES[0]] == NAMES[0] and rig.values[NAMES[2]] == "before"
        assert len(rig.writes) == (1 if failure == "execute" else 2)
        assert rig.verified == list(NAMES[:1 if failure == "execute" else 2])
        first, failed, skipped = data["steps"]
        assert first["result"]["data"]["receipt"] == f"receipt:{NAMES[0]}"
        assert failed["result"]["status"] == "failed" and skipped["result"]["status"] == "not_started"
        await confirm(session, preview.approval.plan_id)
        assert rig.attempted == list(NAMES[:2])
    asyncio.run(run())


def test_unlisted_third_write_requires_its_own_approval(rig):
    async def run():
        session, preview, _ = await rig.start(followups=[call(NAMES[2], step(NAMES[2])["arguments"]), FINAL])
        next_view, _ = await confirm(session, preview.approval.plan_id)
        assert rig.attempted == list(NAMES[:2]) and rig.values[NAMES[2]] == "before"
        assert next_view.status == "approval_required" and next_view.approval.tool_name == NAMES[2]
        assert next_view.approval.plan_id != preview.approval.plan_id
        await confirm(session, preview.approval.plan_id)
        assert rig.attempted == list(NAMES[:2])
        pending = await session.pipeline.get_pending_plan(owner=OWNER, session_id=SESSION)
        assert pending is not None and pending.plan_id == next_view.approval.plan_id
    asyncio.run(run())


def test_ordinary_single_tool_still_requires_confirmation_and_verifies_once(rig):
    async def run():
        session, preview, events = await rig.start(step(NAMES[0])["arguments"], tool=NAMES[0])
        assert preview.approval.tool_name == NAMES[0] and not rig.writes
        assert sum(e.type is Event.EFFECT_APPROVAL_REQUIRED for e in events) == 1
        final, _ = await confirm(session, preview.approval.plan_id)
        assert final.status == "success" and final.effect_result["data"]["receipt"] == f"receipt:{NAMES[0]}"
        await confirm(session, preview.approval.plan_id)
        assert rig.attempted == rig.verified == [NAMES[0]] and len(rig.writes) == 1
    asyncio.run(run())


@pytest.mark.parametrize("drift", [False, True])
def test_cold_recovery_restores_public_plan_snapshots_and_consumption(rig, drift):
    async def run():
        _, preview, _ = await rig.start()
        cold = rig.session([FINAL], SQLiteKernelStore(secret_provider=lambda: "task-test-only"))
        restored = await cold.pipeline.get_pending_plan(owner=OWNER, session_id=SESSION)
        assert restored is not None and restored.public_approval_dict() == preview.approval.to_dict()
        assert not rig.writes and len(rig.trace) == 2
        if drift:
            rig.values[NAMES[1]] = "changed-after-restart"
        final, events = await confirm(cold, restored.plan_id)
        if drift:
            assert any(e.type is Event.EFFECT_FAILED and e.payload.get("code") == "confirmation_stale" for e in events)
            assert not rig.attempted and not rig.writes
        else:
            assert final.status == "success" and final.effect_result["data"]["completed"] == 2
            assert rig.attempted == rig.verified == list(NAMES[:2])
        again = rig.session([], SQLiteKernelStore(secret_provider=lambda: "task-test-only"))
        replay, _ = await confirm(again, restored.plan_id)
        assert replay.status == "failed" and len(rig.writes) == (0 if drift else 2)
        assert await again.pipeline.get_pending_plan(owner=OWNER, session_id=SESSION) is None
    asyncio.run(run())


def test_production_audit_records_each_step_before_finalizing(rig, monkeypatch):
    from app import database
    from app.agent.kernel.ports.mediaflux_effects import MediaFluxEffectLifecycle

    original = rig.session

    def audited_session(*args, **kwargs):
        session = original(*args, **kwargs)
        session.pipeline.effect_lifecycle = MediaFluxEffectLifecycle()
        session.pipeline.effect_store.record_actions = True
        return session

    monkeypatch.setattr(rig, "session", audited_session)

    async def run():
        session, preview, _ = await rig.start()
        final, _ = await confirm(session, preview.approval.plan_id)
        assert final.status == "success"
        with database.get_conn() as conn:
            rows = [dict(r) for r in conn.execute(
                "SELECT tool_name,status,ok,confirmation_id FROM agent_action_history ORDER BY id")]
        assert len(rows) == 3
        assert [r["tool_name"] for r in rows] == ["agent.batch", "unknown.confirmed_action", "unknown.confirmed_action"]
        assert all(r["ok"] and r["status"] == "completed" for r in rows)
        assert len({r["confirmation_id"] for r in rows}) == 3
        assert len(rig.writes) == 2

    asyncio.run(run())
