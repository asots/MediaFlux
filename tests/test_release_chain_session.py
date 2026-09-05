"""确认终态的会话回写不能清除同 generation 的另一张票据。"""
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from app.agent.confirmation import ConfirmationStore
from app.agent.kernel.capabilities import (
    CapabilityRetriever, KernelToolSpec, ToolCatalog, ToolEffect,
)
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.effects import ConfirmationEffectPlanStore, PreparedEffect
from app.agent.kernel.pipeline import ToolCallContext, ToolPipeline
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import CancellationToken, InMemorySessionStateStore
from tests.support import isolated_test_database


class ConfirmationSessionPointerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(isolated_test_database())
        self.ordinal = 0

    async def fixture(self, kind):
        self.ordinal += 1
        store = (InMemorySessionStateStore() if kind == "memory" else
                 SQLiteKernelStore(secret_provider=lambda: "isolated-session-secret"))
        lease, _ = await store.begin_turn(owner="owner", session_id=f"s{self.ordinal}", request_id="prepare")

        async def progress(_payload):
            pass

        context = ToolCallContext(owner=lease.owner, session_id=lease.session_id,
            request_id=lease.request_id, turn_id=lease.turn_id, lease=lease,
            cancellation=CancellationToken(), report_progress=progress)
        execute = Mock(return_value={"ok": True, "summary": "accepted"})
        catalog = ToolCatalog([KernelToolSpec(name="downloads.submit", domain="downloads",
            description="test write", input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.WRITE,
            prepare=lambda *_: PreparedEffect(preview={"summary": "preview"}, snapshot_fingerprint="stable"),
            execute_confirmed=execute)])
        pipeline = ToolPipeline(catalog=catalog, state_store=store,
            effect_store=ConfirmationEffectPlanStore(ConfirmationStore()))
        session = AgentSession(model=Mock(), catalog=catalog,
            retriever=CapabilityRetriever(minimum=1, maximum=1), pipeline=pipeline, state_store=store)
        prepared = await pipeline.execute("downloads.submit", {}, context=context)
        return session, context, prepared.effect_plan.plan_id, execute

    async def test_failed_old_confirmation_retains_current_ticket_and_result_history(self):
        for kind in ("memory", "sqlite"):
            with self.subTest(store=kind):
                session, context, old, execute = await self.fixture(kind)
                new = await session.pipeline.execute("downloads.submit", {}, context=context)
                events = [e async for e in session.confirm(owner=context.owner,
                    session_id=context.session_id, plan_id=old)]
                state = await session.state_store.load(owner=context.owner, session_id=context.session_id)
                self.assertEqual(state.pending_effect_plan_id, new.effect_plan.plan_id)
                self.assertEqual(events[-1].type.value, "turn.failed")
                self.assertTrue(state.conversation)
                execute.assert_not_called()
                claimed = session.pipeline.effect_store.claim(owner=context.owner, session_id=context.session_id,
                    generation=context.lease.generation, plan_id=new.effect_plan.plan_id)
                self.assertEqual(claimed.plan_id, new.effect_plan.plan_id)

    async def test_same_generation_replacement_at_terminal_write_is_not_erased(self):
        for kind in ("memory", "sqlite"):
            with self.subTest(store=kind):
                session, context, old, execute = await self.fixture(kind)
                original = session.state_store.commit
                replacement = []

                async def commit(lease, **kwargs):
                    if kwargs.get("conversation") and not replacement:
                        new = await session.pipeline.execute("downloads.submit", {}, context=context)
                        replacement.append(new.effect_plan.plan_id)
                    return await original(lease, **kwargs)

                with patch.object(session.state_store, "commit", side_effect=commit):
                    events = [e async for e in session.confirm(owner=context.owner,
                        session_id=context.session_id, plan_id=old)]
                state = await session.state_store.load(owner=context.owner, session_id=context.session_id)
                self.assertEqual(state.pending_effect_plan_id, replacement[0])
                self.assertEqual(events[-1].type.value, "turn.completed")
                self.assertEqual(execute.call_count, 1)
                self.assertTrue(state.conversation)
