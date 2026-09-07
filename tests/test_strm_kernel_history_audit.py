"""真实 STRM 工具声明→Kernel确认→SQLite票据/会话/审计的重建验证。"""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from app import database as db
from app.agent.action_history import action_history_owner_digest
from app.agent.confirmation import SQLiteConfirmationStore
from app.agent.domain_catalog import build_tool_specs
from app.agent.domain_catalog import strm_runtime
from app.agent.kernel.capabilities import ToolCatalog
from app.agent.kernel.effects import ConfirmationEffectPlanStore
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.pipeline import (
    ConfirmationClaimError,
    ToolCallContext,
    ToolPipeline,
)
from app.agent.kernel.ports.existing_actions import adapt_tool_spec
from app.agent.kernel.ports.mediaflux_effects import MediaFluxEffectLifecycle
from app.agent.kernel.state import CancellationToken
from app.modules.scheduler import STRMScheduler
from tests.support import isolated_test_database


class STRMKernelHistoryAuditTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(isolated_test_database())
        self.scheduler = STRMScheduler()
        self.scheduler._run_lock = threading.Lock()
        self.enterContext(
            patch("app.modules.scheduler.get_scheduler", return_value=self.scheduler)
        )
        self.enterContext(
            patch.object(self.scheduler, "validate_config", return_value="")
        )
        self.enterContext(
            patch.object(self.scheduler, "status", return_value={"running": False})
        )
        original = strm_runtime.config.get
        self.enterContext(
            patch.object(
                strm_runtime.config,
                "get",
                side_effect=lambda key, default="": (
                    "synthetic"
                    if key in strm_runtime._STRM_CONFIRMATION_KEYS
                    else original(key, default)
                ),
            )
        )
        self.spec = next(
            spec for spec in build_tool_specs() if spec.name == "strm.run_once"
        )
        self.owner = "synthetic-kernel-owner"

    def pipeline(self):
        state = SQLiteKernelStore(
            secret_provider=lambda: "synthetic-kernel-history-secret"
        )
        return ToolPipeline(
            catalog=ToolCatalog([adapt_tool_spec(self.spec)]),
            state_store=state,
            reference_store=state,
            effect_store=ConfirmationEffectPlanStore(
                SQLiteConfirmationStore(), record_actions=True
            ),
            effect_lifecycle=MediaFluxEffectLifecycle(),
        ), state

    async def prepare(self, arguments):
        pipeline, state = self.pipeline()
        lease, _state = await state.begin_turn(
            owner=self.owner, session_id="history", request_id="prepare"
        )

        async def ignore(_payload):
            pass

        context = ToolCallContext(
            owner=self.owner,
            session_id="history",
            request_id=lease.request_id,
            turn_id=lease.turn_id,
            lease=lease,
            cancellation=CancellationToken(),
            report_progress=ignore,
        )
        result = await pipeline.execute("strm.run_once", arguments, context=context)
        self.assertIsNotNone(result.effect_plan)
        return context, result.effect_plan.plan_id

    def audit(self):
        return db.list_agent_action_history(
            owner_digest=action_history_owner_digest(self.owner), limit=10
        )

    async def test_restored_confirmation_preserves_failed_receipt_and_cannot_replay(
        self,
    ):
        context, plan_id = await self.prepare({})
        db.init_db()
        rebuilt, state = self.pipeline()
        self.assertEqual(
            (
                await state.load(owner=self.owner, session_id="history")
            ).pending_effect_plan_id,
            plan_id,
        )
        self.scheduler._stop_event.set()
        failed = await rebuilt.execute_confirmed(plan_id, context=context)
        self.assertFalse(failed.outcome.public_content["ok"])
        self.assertEqual(failed.outcome.public_content["status"], "failed")
        self.assertIn("停止", failed.outcome.public_content["error"])
        rows = self.audit()
        self.assertEqual(len(rows), 1)
        self.assertFalse(bool(rows[0]["ok"]))
        self.assertEqual(rows[0]["status"], "failed")
        db.init_db()
        restarted, _state = self.pipeline()
        with self.assertRaises(ConfirmationClaimError):
            await restarted.execute_confirmed(plan_id, context=context)
        self.assertEqual(len(self.audit()), 1)
        self.assertEqual(self.audit()[0]["status"], "failed")

    async def test_restored_scoped_confirmation_submits_once_and_persists_success(self):
        with (
            patch(
                "app.modules.strm.configured_strm_source_plans",
                return_value=([{"id": "source-a", "name": "Alpha"}], ""),
            ),
            patch.object(
                self.scheduler, "trigger", return_value={"ok": True}
            ) as trigger,
            patch(
                "app.agent.kernel.ports.mediaflux_effects.invalidate_agent_runtime_generation"
            ) as invalidate,
        ):
            context, plan_id = await self.prepare({"source_names": ["Alpha"]})
            db.init_db()
            rebuilt, state = self.pipeline()
            await rebuilt.execute_confirmed(plan_id, context=context)
            trigger.assert_called_once_with("manual", selected_source_ids=["source-a"])
            invalidate.assert_called_once()
            self.assertFalse(
                (
                    await state.load(owner=self.owner, session_id="history")
                ).pending_effect_plan_id
            )
            db.init_db()
            restarted, _state = self.pipeline()
            with self.assertRaises(ConfirmationClaimError):
                await restarted.execute_confirmed(plan_id, context=context)
            self.assertEqual(trigger.call_count, 1)
        rows = self.audit()
        self.assertEqual(len(rows), 1)
        self.assertTrue(bool(rows[0]["ok"]))
        self.assertEqual(rows[0]["status"], "accepted")
