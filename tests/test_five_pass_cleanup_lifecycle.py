"""残留清理预览不能删除已确认执行凭据；GC 与确认共用状态锁。"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

from app.agent import guangya_cleanup_actions as actions
from app.agent.models import ToolContext
from app.modules import guangya_residual_cleanup as cleanup
from tests.support import IsolatedDatabaseTestCase, isolated_test_database
from tests.test_guangya_residual_cleanup import FakeCleanupClient


class CleanupPlanLifecycleTests(IsolatedDatabaseTestCase):
    def setUp(self):
        self.enterContext(isolated_test_database())
        root = self.enterContext(tempfile.TemporaryDirectory(prefix="five-pass-cleanup-"))
        self.plan_dir = Path(root) / "plans"
        self.enterContext(patch.object(cleanup, "_plan_directory", return_value=self.plan_dir))
        self.enterContext(patch.object(cleanup, "get_web_secret", return_value="synthetic-cleanup-secret"))
        self.enterContext(patch.object(actions, "_configured_sources", return_value=[{"id": "source", "name": "整理源"}]))
        actions.reset_guangya_cleanup_context_for_tests()
        self.addCleanup(actions.reset_guangya_cleanup_context_for_tests)
        self.client = FakeCleanupClient()
        self.client.directories["source"] = [row for row in self.client.directories["source"] if row.file_id != "residual"]
        self.context = ToolContext(owner="five-pass-owner", session_id="synthetic-session")

    def _preview(self):
        with patch.object(actions, "GuangYaClient", return_value=self.client):
            actions.preview_guangya_cleanup({"max_candidates": 20}, self.context)
        flow = actions._flow(self.context.owner)
        self.assertIsNotNone(flow)
        return cleanup.load_cleanup_plan(flow.plan_id, owner=self.context.owner)

    def _confirm(self, plan):
        return cleanup.confirm_cleanup_plan(
            plan["plan_id"], owner=self.context.owner, expected_fingerprint=plan["fingerprint"],
        )

    def test_new_preview_keeps_confirmed_previous_plan_and_journal(self):
        first = self._confirm(self._preview())
        cleanup._append_journal(first["plan_id"], {"action": "accepted", "task_id": "synthetic-job"})
        journal = cleanup._journal_path(first["plan_id"]).read_bytes()
        second = self._preview()
        self.assertNotEqual(second["plan_id"], first["plan_id"])
        recovered = cleanup.load_cleanup_plan(
            first["plan_id"], owner=self.context.owner, require_confirmed=True,
        )
        self.assertEqual(recovered["status"], "confirmed")
        self.assertEqual(cleanup._journal_path(first["plan_id"]).read_bytes(), journal)

    def test_new_preview_still_removes_unconfirmed_previous_plan(self):
        first = self._preview()
        self._preview()
        self.assertFalse(cleanup._plan_path(first["plan_id"]).exists())

    def test_running_or_terminal_plan_cannot_be_reconfirmed(self):
        plan = self._confirm(self._preview())
        for status in ("running", "completed", "partial", "manual_review", "failed"):
            with self.subTest(status=status):
                cleanup._update_execution(plan["plan_id"], status, {"marker": "preserve"})
                before = cleanup._plan_path(plan["plan_id"]).read_bytes()
                with self.assertRaises(cleanup.GuangYaCleanupPlanStale):
                    self._confirm(plan)
                self.assertEqual(cleanup._plan_path(plan["plan_id"]).read_bytes(), before)

    def test_preview_discard_waits_for_confirmation_then_preserves_confirmed_plan(self):
        plan = self._preview()
        entered, release, discarded = threading.Event(), threading.Event(), threading.Event()
        original = cleanup._atomic_write
        errors = []
        result = []

        def delayed_write(payload):
            if payload.get("status") == "confirmed":
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("test confirmation barrier")
            return original(payload)

        def confirm():
            try:
                self._confirm(plan)
            except BaseException as exc:
                errors.append(exc)

        def discard():
            try:
                result.append(cleanup.discard_cleanup_plan(plan["plan_id"], preview_only=True))
            except BaseException as exc:
                errors.append(exc)
            finally:
                discarded.set()

        with patch.object(cleanup, "_atomic_write", side_effect=delayed_write):
            writer = threading.Thread(target=confirm)
            remover = threading.Thread(target=discard)
            writer.start()
            try:
                self.assertTrue(entered.wait(2))
                remover.start()
                self.assertFalse(discarded.wait(0.1))
            finally:
                release.set()
                writer.join(3)
                if remover.ident is not None:
                    remover.join(3)
        self.assertFalse(writer.is_alive())
        self.assertFalse(remover.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result, [False])
        self.assertEqual(cleanup.load_cleanup_plan(plan["plan_id"], require_confirmed=True)["status"], "confirmed")

    def test_history_gc_preserves_running_plan_after_old_ttl(self):
        plan = self._confirm(self._preview())
        cleanup._update_execution(plan["plan_id"], "running", {"started_at": "2026-09-07T10:00:00+08:00"})
        with patch.object(cleanup.time, "time", return_value=plan["execute_until_epoch"] + 86400):
            summary = cleanup.maintain_cleanup_plans()
        self.assertEqual(summary["active"], 1)
        self.assertTrue(cleanup._plan_path(plan["plan_id"]).exists())
