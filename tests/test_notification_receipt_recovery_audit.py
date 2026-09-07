"""回调已提供真实消息身份时，未知首次投递应安全恢复为原位编辑。"""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from app import database as db
from app.modules import telegram_notification_center as center
from app.notifier import NotificationAction, NotificationEvent, TelegramSendResult
from app.repositories import telegram_notifications as repository
from tests.support import isolated_test_database


class NotificationReceiptRecoveryAuditTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(isolated_test_database())
        self.enterContext(patch.object(center, "_dispatch_stop", threading.Event()))
        self.enterContext(
            patch.object(center, "allows_notification", return_value=True)
        )
        self.enterContext(patch.object(center, "wake_telegram_notification_dispatcher"))
        self.sender = self.enterContext(
            patch.object(
                center,
                "send_event_result",
                return_value=TelegramSendResult(
                    ok=False, status_code=408, error="ReadTimeout"
                ),
            )
        )
        self.editor = self.enterContext(
            patch.object(
                center, "edit_event_result", return_value=TelegramSendResult(ok=True)
            )
        )
        self.initial = NotificationEvent(
            "待确认", actions=(NotificationAction("确认", "orgc:audit:0"),)
        )
        self.terminal = NotificationEvent("已完成", state="completed")

    def publish(self, event, *, message_id=0, deliver_now=True):
        return center.publish_notification_thread(
            "confirmation:receipt-audit",
            event,
            topic="confirmation",
            chat_id="100",
            preferred_message_id=message_id,
            deliver_now=deliver_now,
        )

    def test_changed_terminal_with_callback_message_id_recovers_unknown_first_send(
        self,
    ):
        first = self.publish(self.initial)
        self.assertEqual(first.status, "outcome_unknown")
        result = self.publish(self.terminal, message_id=77)
        self.assertTrue(result.delivered, result)
        self.sender.assert_called_once()
        self.editor.assert_called_once()
        self.assertEqual(self.editor.call_args.kwargs["message_id"], 77)
        self.assertEqual(self.editor.call_args.args[0].actions, ())
        row = repository.get_notification(first.event_key)
        self.assertEqual(
            (row["revision"], row["delivered_revision"], row["message_id"]), (2, 2, 77)
        )
        self.assertEqual(row["last_error"], "")

    def test_same_payload_with_message_receipt_recovers_without_revision_inflation(
        self,
    ):
        first = self.publish(self.initial)
        result = self.publish(self.initial, message_id=77)
        self.assertTrue(result.delivered)
        self.assertEqual(repository.get_notification(first.event_key)["revision"], 1)
        self.sender.assert_called_once()
        self.editor.assert_called_once()

    def test_no_receipt_never_replays_unknown_send_even_after_reopen(self):
        first = self.publish(self.initial)
        db.init_db()
        repository.recover_notifications()
        result = self.publish(self.terminal)
        self.assertEqual(result.status, "outcome_unknown")
        self.assertFalse(result.queued)
        self.sender.assert_called_once()
        self.editor.assert_not_called()
        self.assertFalse(repository.get_notification(first.event_key)["message_id"])

    def test_later_preferred_id_cannot_replace_the_already_bound_thread(self):
        first = self.publish(self.initial)
        self.publish(self.terminal, message_id=77)
        result = self.publish(NotificationEvent("后处理完成"), message_id=99)
        self.assertTrue(result.delivered)
        self.assertEqual(repository.get_notification(first.event_key)["message_id"], 77)
        self.assertEqual(self.editor.call_args.kwargs["message_id"], 77)
        self.assertEqual(self.editor.call_count, 2)
        self.sender.assert_called_once()

    def test_receipt_during_inflight_keeps_lease_then_delivers_latest_revision_only(
        self,
    ):
        first = self.publish(self.initial, deliver_now=False)
        claimed = repository.claim_due_notifications(event_key=first.event_key)[0]
        updated = self.publish(self.terminal, message_id=77, deliver_now=False)
        self.assertEqual(updated.status, "sending")
        self.assertTrue(
            repository.mark_outcome_unknown(
                claimed["id"],
                lease_generation=claimed["lease_generation"],
                claimed_revision=claimed["revision"],
                error="old send outcome unknown",
            )
        )
        self.assertTrue(center.drain_telegram_notifications(event_key=first.event_key))
        row = repository.get_notification(first.event_key)
        self.assertEqual(
            (row["status"], row["revision"], row["delivered_revision"]), ("sent", 2, 2)
        )
        self.sender.assert_not_called()
        self.editor.assert_called_once()
