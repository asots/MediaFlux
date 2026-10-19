from __future__ import annotations

from app import database as db
from app.agent.action_history import action_history_owner_digest
from app.agent.domains.activity.undo import (
    attach_undo_receipt,
    compensation_candidate,
    execute_undo,
    inspect_undo,
    prepare_undo,
)
from app.agent.domains.subscriptions.consumption_actions import (
    prepare_set_preferences,
    set_preferences_confirmed,
)
from app.agent.models import ToolContext, ToolResult
from app.repositories.media_experience import get_media_preferences
from tests.support import IsolatedDatabaseTestCase


class ActionUndoTests(IsolatedDatabaseTestCase):
    def setUp(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM agent_compensations")
            conn.execute("DELETE FROM agent_media_preferences")
        self.context = ToolContext(owner="undo-test", session_id="session")

    def forward(self):
        args = {"preferred_download_target": "qb"}
        preview, expected = prepare_set_preferences(args, self.context)
        candidate = compensation_candidate(
            "media.set_preferences", args, preview.to_dict()
        )
        result = set_preferences_confirmed(args, expected, self.context)
        attach_undo_receipt(result, candidate, key="receipt-test", context=self.context)
        return {"undo_receipt": result.references[0].value}

    def test_settings_can_be_undone_once_and_restore_implicit_defaults(self):
        arguments = self.forward()
        before = get_media_preferences(action_history_owner_digest(self.context.owner))
        self.assertEqual(before["preferred_download_target"], "qb")
        _, frozen = prepare_undo(arguments, self.context)
        result = execute_undo(arguments, frozen, self.context)
        self.assertTrue(result.ok)
        after = get_media_preferences(action_history_owner_digest(self.context.owner))
        self.assertFalse(after["explicit"])
        with self.assertRaises(ValueError):
            prepare_undo(arguments, self.context)

    def test_intervening_changes_make_receipt_stale(self):
        arguments = self.forward()
        args = {"preferred_download_target": "both"}
        _, expected = prepare_set_preferences(args, self.context)
        set_preferences_confirmed(args, expected, self.context)
        with self.assertRaises(ValueError):
            prepare_undo(arguments, self.context)
        self.assertEqual(
            get_media_preferences(action_history_owner_digest(self.context.owner))[
                "preferred_download_target"
            ],
            "both",
        )

    def test_owner_cannot_consume_another_owners_receipt(self):
        arguments = self.forward()
        with self.assertRaises(ValueError):
            prepare_undo(arguments, ToolContext(owner="other", session_id="session"))

    def test_mutating_frozen_payload_is_rejected(self):
        arguments = self.forward()
        _, expected = prepare_undo(arguments, self.context)
        arguments["undo_receipt"]["receipt_id"] = "unknown-receipt"
        with self.assertRaises(ValueError):
            execute_undo(arguments, expected, self.context)

    def test_unsupported_irreversible_tools_do_not_issue_compensation(self):
        for name in (
            "ingest.submit",
            "guangya.recycle.clear",
            "rss.delete_subscription",
        ):
            self.assertIsNone(compensation_candidate(name, {}, {}))
        result = inspect_undo(
            {"activity_selection": {"items": [{"kind": "download", "id": 1}]}},
            self.context,
        )
        self.assertFalse(result.data["reversible"])
        self.assertFalse(result.references)

    def test_incomplete_result_does_not_offer_undo(self):
        result = ToolResult(True, "accepted", "已接收")
        attach_undo_receipt(
            result, {"tool": "media.set_preferences"}, key="a", context=self.context
        )
        self.assertFalse(result.references)

    def test_newer_write_before_receipt_creation_is_not_captured_as_our_write(self):
        args = {"preferred_download_target": "qb"}
        preview, expected = prepare_set_preferences(args, self.context)
        candidate = compensation_candidate(
            "media.set_preferences", args, preview.to_dict()
        )
        original_result = set_preferences_confirmed(args, expected, self.context)
        other_args = {"preferred_download_target": "both"}
        _, other_expected = prepare_set_preferences(other_args, self.context)
        set_preferences_confirmed(other_args, other_expected, self.context)
        attach_undo_receipt(
            original_result, candidate, key="late-receipt", context=self.context
        )
        self.assertFalse(original_result.references)
        self.assertEqual(
            get_media_preferences(action_history_owner_digest(self.context.owner))[
                "preferred_download_target"
            ],
            "both",
        )

    def test_missing_transactional_after_snapshot_does_not_issue_receipt(self):
        args = {"preferred_download_target": "qb"}
        preview, expected = prepare_set_preferences(args, self.context)
        candidate = compensation_candidate(
            "media.set_preferences", args, preview.to_dict()
        )
        result = set_preferences_confirmed(args, expected, self.context)
        result.effect_metadata.clear()
        attach_undo_receipt(result, candidate, key="no-proof", context=self.context)
        self.assertFalse(result.references)

    def organize_receipt(self):
        from unittest.mock import patch

        from app.repositories import agent_compensations as receipts

        value = {"title": "示例", "version": 1, "changes": [], "affected": 1}
        receipts.create(
            "organize-receipt", action_history_owner_digest(self.context.owner)
        )
        args = {
            "undo_receipt": {
                "kind": "organize",
                "receipt_id": "organize-receipt",
                "target": {"kind": "organize", "id": 1},
                "expected": "version-one",
                "version": 1,
            }
        }
        with patch(
            "app.agent.domains.activity.undo._organize_state", return_value=(value, "version-one")
        ):
            _, expected = prepare_undo(args, self.context)
        return value, args, expected

    def test_organize_second_read_cannot_adopt_new_unapproved_version(self):
        from unittest.mock import patch

        first, args, token = self.organize_receipt()
        second = {**first, "version": 2}
        with (
            patch(
                "app.agent.domains.activity.undo._organize_state",
                side_effect=[(first, "version-one"), (second, "version-two")],
            ),
            patch(
                "app.modules.organize_correction.OrganizeCorrectionService"
            ) as service,
            patch("app.modules.organize_tasks.get_organize_manager") as manager,
            self.assertRaises(ValueError),
        ):
            execute_undo(args, token, self.context)
        service.assert_not_called()
        manager.assert_not_called()

    def test_claim_database_failure_does_not_create_external_service(self):
        import sqlite3
        from unittest.mock import patch

        first, args, token = self.organize_receipt()
        with (
            patch(
                "app.agent.domains.activity.undo._organize_state",
                return_value=(first, "version-one"),
            ),
            patch(
                "app.agent.domains.activity.undo.receipts.claim",
                side_effect=sqlite3.OperationalError("busy"),
            ),
            patch(
                "app.modules.organize_correction.OrganizeCorrectionService"
            ) as service,
            self.assertRaises(sqlite3.OperationalError),
        ):
            execute_undo(args, token, self.context)
        service.assert_not_called()

    def test_queued_organize_uses_frozen_version_and_closes_service(self):
        from unittest.mock import Mock, patch

        first, args, token = self.organize_receipt()
        client = Mock()
        client.revert_latest.return_value = {"success": True}
        callbacks = []
        manager = Mock()
        manager.start_operation.side_effect = lambda _name, _title, callback: (
            callbacks.append(callback) or {"ok": True, "task_id": "private-undo-task"}
        )
        with (
            patch(
                "app.agent.domains.activity.undo._organize_state",
                return_value=(first, "version-one"),
            ),
            patch(
                "app.modules.organize_correction.OrganizeCorrectionService",
                return_value=client,
            ),
            patch(
                "app.modules.organize_tasks.get_organize_manager", return_value=manager
            ),
        ):
            result = execute_undo(args, token, self.context)
            self.assertEqual(result.status, "accepted")
            self.assertEqual(
                result.effect_metadata["completion"],
                {
                    "kind": "guangya_organize_task",
                    "task_id": "private-undo-task",
                    "operation": "undo",
                },
            )
            client.revert_latest.assert_not_called()
            callbacks[0]()
            client.revert_latest.assert_called_once_with(1, "undo-organize-receipt", 1)
            client.close.assert_called_once()

    def test_busy_organize_can_reinspect_and_retry_same_receipt_exactly_once(self):
        from unittest.mock import Mock, patch

        from app.modules.organize_tasks import OrganizeTaskManager
        from app.repositories import agent_compensations as receipts

        public = {"title": "示例", "version": 1, "changes": [], "affected": 1}
        selection = {"activity_selection": {"items": [{"kind": "organize", "id": 1}]}}
        manager = OrganizeTaskManager()
        self.addCleanup(manager.shutdown, timeout=2)
        unused, worker_client = Mock(), Mock()
        worker_client.revert_latest.return_value = {"success": True}
        digest = action_history_owner_digest(self.context.owner)
        with (
            patch("app.agent.domains.activity.undo._organize_state", return_value=(public, "version-one")),
            patch("app.modules.organize_correction.OrganizeCorrectionService", side_effect=[unused, worker_client]),
            patch("app.modules.organize_tasks.get_organize_manager", return_value=manager),
        ):
            receipt = inspect_undo(selection, self.context).references[0].value
            arguments = {"undo_receipt": receipt}
            _, expected = prepare_undo(arguments, self.context)
            self.assertTrue(manager._lock.acquire(blocking=False))
            try:
                rejected = execute_undo(arguments, expected, self.context)
            finally:
                manager._lock.release()
            self.assertEqual(rejected.status, "busy")
            unused.revert_latest.assert_not_called()
            unused.close.assert_called_once()
            self.assertEqual(receipts.state(receipt["receipt_id"], digest), "available")
            fresh_receipt = inspect_undo(selection, self.context).references[0].value
            self.assertEqual(receipt["receipt_id"], fresh_receipt["receipt_id"])
            _, expected = prepare_undo({"undo_receipt": fresh_receipt}, self.context)
            accepted = execute_undo({"undo_receipt": fresh_receipt}, expected, self.context)
            self.assertEqual(accepted.status, "accepted")
            self.assertTrue(manager.shutdown(timeout=2))
            worker_client.revert_latest.assert_called_once()
            worker_client.close.assert_called_once()
            self.assertEqual(receipts.state(receipt["receipt_id"], digest), "completed")
            with self.assertRaises(ValueError):
                prepare_undo(arguments, self.context)

    def test_explicit_organize_admission_rejection_keeps_receipt_available(self):
        from unittest.mock import Mock, patch

        from app.modules.organize_tasks import OrganizeTaskManager
        from app.repositories import agent_compensations as receipts

        public = {"title": "示例", "version": 1, "changes": [], "affected": 1}
        selection = {"activity_selection": {"items": [{"kind": "organize", "id": 1}]}}
        for rejection in ("shutdown", "thread_start_failed"):
            with self.subTest(rejection=rejection):
                context = ToolContext(owner=self.context.owner, session_id=rejection)
                manager, client = OrganizeTaskManager(), Mock()
                self.addCleanup(manager.shutdown, timeout=2)
                if rejection == "shutdown":
                    manager.begin_shutdown()
                thread = Mock()
                thread.start.side_effect = RuntimeError("cannot start")
                with (
                    patch("app.agent.domains.activity.undo._organize_state", return_value=(public, "version-one")),
                    patch("app.modules.organize_correction.OrganizeCorrectionService", return_value=client),
                    patch("app.modules.organize_tasks.get_organize_manager", return_value=manager),
                    patch("app.modules.organize_tasks.threading.Thread", return_value=thread),
                ):
                    receipt = inspect_undo(selection, context).references[0].value
                    arguments = {"undo_receipt": receipt}
                    _, expected = prepare_undo(arguments, context)
                    result = execute_undo(arguments, expected, context)
                    self.assertEqual(result.status, "busy")
                    self.assertEqual(receipts.state(receipt["receipt_id"], action_history_owner_digest(context.owner)), "available")
                    prepare_undo(arguments, context)
                client.revert_latest.assert_not_called()
                client.close.assert_called_once()

    def test_unknown_admission_exception_does_not_release_receipt(self):
        from unittest.mock import Mock, patch

        from app.repositories import agent_compensations as receipts

        public, arguments, expected = self.organize_receipt()
        manager = Mock()
        manager.start_operation.side_effect = RuntimeError("admission response lost")
        with (
            patch("app.agent.domains.activity.undo._organize_state", return_value=(public, "version-one")),
            patch("app.modules.organize_correction.OrganizeCorrectionService"),
            patch("app.modules.organize_tasks.get_organize_manager", return_value=manager),
            self.assertRaises(RuntimeError),
        ):
            execute_undo(arguments, expected, self.context)
        self.assertEqual(receipts.state("organize-receipt", action_history_owner_digest(self.context.owner)), "outcome_unknown")
        with self.assertRaises(ValueError):
            prepare_undo(arguments, self.context)

    def test_accepted_callback_failure_stays_closed_to_retries(self):
        from unittest.mock import Mock, patch

        from app.repositories import agent_compensations as receipts

        public = {"title": "示例", "version": 1, "changes": [], "affected": 1}
        selection = {"activity_selection": {"items": [{"kind": "organize", "id": 1}]}}
        for raises in (False, True):
            with self.subTest(raises=raises):
                context = ToolContext(owner=self.context.owner, session_id=f"failed-{raises}")
                manager, client, callbacks = Mock(), Mock(), []
                client.revert_latest.return_value = {"success": False}
                if raises:
                    client.revert_latest.side_effect = OSError("write outcome unknown")
                manager.start_operation.side_effect = lambda _operation, _title, callback: (
                    callbacks.append(callback) or {"ok": True, "task_id": "private-task"}
                )
                with (
                    patch("app.agent.domains.activity.undo._organize_state", return_value=(public, "version-one")),
                    patch("app.modules.organize_correction.OrganizeCorrectionService", return_value=client),
                    patch("app.modules.organize_tasks.get_organize_manager", return_value=manager),
                ):
                    receipt = inspect_undo(selection, context).references[0].value
                    arguments = {"undo_receipt": receipt}
                    _, expected = prepare_undo(arguments, context)
                    self.assertEqual(execute_undo(arguments, expected, context).status, "accepted")
                    if raises:
                        with self.assertRaises(OSError):
                            callbacks[0]()
                    else:
                        callbacks[0]()
                    self.assertEqual(receipts.state(receipt["receipt_id"], action_history_owner_digest(context.owner)), "outcome_unknown")
                    with self.assertRaises(ValueError):
                        prepare_undo(arguments, context)
                client.revert_latest.assert_called_once()
                client.close.assert_called_once()

    def test_unaccepted_release_cannot_reopen_settled_or_other_owner_receipts(self):
        from app.repositories import agent_compensations as receipts

        digest = action_history_owner_digest(self.context.owner)
        for completed in (False, True):
            key = f"settled-{completed}"
            receipts.create(key, digest)
            self.assertTrue(receipts.claim(key, digest))
            self.assertFalse(receipts.claim(key, digest))
            self.assertFalse(receipts.release_unaccepted(key, "other-owner"))
            self.assertEqual(receipts.state(key, digest), "executing")
            receipts.finish(key, digest, completed=completed)
            self.assertFalse(receipts.release_unaccepted(key, digest))
            receipts.create(key, digest)
            self.assertFalse(receipts.claim(key, digest))
