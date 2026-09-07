"""通知规则重置/重建复用整数版本时，确认不能覆盖另一份业务状态。"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

from app import database as db
from app.agent import media_consumption_actions as actions
from app.agent.errors import AgentToolError
from app.agent.models import ToolContext
from app.repositories import media_experience as repository
from tests.support import IsolatedDatabaseTestCase


class NotificationRuleReconfirmAuditTests(IsolatedDatabaseTestCase):
    def setUp(self) -> None:
        with db.get_conn() as conn:
            conn.execute("DELETE FROM media_subscription_notification_rules")
            conn.execute("DELETE FROM media_subscriptions")
        self.context = ToolContext(owner="audit", session_id="isolated")
        self.sid = db.add_media_subscription(
            provider="tmdb",
            external_id="606",
            tmdb_id="606",
            media_type="tv",
            title="Reconfirm",
            action="confirm",
            download_target="guangya",
            sites=("mikan",),
        )
        current = repository.get_notification_rule(self.sid)
        assert current is not None
        self.initial = repository.set_notification_rule(
            self.sid,
            expected_rule_revision=0,
            expected_subscription_revision=current["subscription_revision"],
            updates={"enabled": True},
        )
        assert self.initial is not None

    def _replace_after_read(self, number):
        current = repository.get_notification_rule(number)
        assert current is not None
        self.assertTrue(
            repository.reset_notification_rule(
                number,
                expected_rule_revision=current["revision"],
                expected_subscription_revision=current["subscription_revision"],
            )
        )
        replacement = repository.set_notification_rule(
            number,
            expected_rule_revision=0,
            expected_subscription_revision=current["subscription_revision"],
            updates={"enabled": False},
        )
        assert replacement is not None
        self.assertEqual(replacement["revision"], current["revision"])
        self.assertEqual(
            replacement["subscription_revision"], current["subscription_revision"]
        )
        return current

    def test_reset_recreate_between_precheck_and_set_rejects_old_confirmation(
        self,
    ) -> None:
        arguments = {
            "subscription_number": self.sid,
            "enabled": True,
            "notify_on_satisfied": True,
        }
        _, fingerprint = actions.prepare_set_subscription_notification_rule(
            arguments, self.context
        )
        with patch.object(actions, "get_notification_rule", self._replace_after_read):
            with self.assertRaises(AgentToolError) as raised:
                actions.set_subscription_notification_rule_confirmed(
                    arguments, fingerprint, self.context
                )
        self.assertEqual(raised.exception.code, "confirmation_stale")
        actual = repository.get_notification_rule(self.sid)
        assert actual is not None
        self.assertFalse(actual["enabled"])
        self.assertFalse(actual["notify_on_satisfied"])
        self.assertEqual(actual["revision"], 1)

    def test_reset_recreate_between_precheck_and_reset_keeps_new_rule(self) -> None:
        arguments = {"subscription_number": self.sid}
        _, fingerprint = actions.prepare_reset_subscription_notification_rule(
            arguments, self.context
        )
        with patch.object(actions, "get_notification_rule", self._replace_after_read):
            with self.assertRaises(AgentToolError) as raised:
                actions.reset_subscription_notification_rule_confirmed(
                    arguments, fingerprint, self.context
                )
        self.assertEqual(raised.exception.code, "confirmation_stale")
        actual = repository.get_notification_rule(self.sid)
        assert actual is not None
        self.assertTrue(actual["explicit"])
        self.assertFalse(actual["enabled"])

    def test_write_failure_rolls_back_and_restart_can_retry_same_confirmation(
        self,
    ) -> None:
        arguments = {"subscription_number": self.sid, "notify_on_satisfied": True}
        _, fingerprint = actions.prepare_set_subscription_notification_rule(
            arguments, self.context
        )
        with db.get_conn() as conn:
            conn.execute(
                "CREATE TRIGGER audit_rule_failure BEFORE UPDATE ON media_subscription_notification_rules BEGIN SELECT RAISE(ABORT,'audit interruption'); END"
            )
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                actions.set_subscription_notification_rule_confirmed(
                    arguments, fingerprint, self.context
                )
            self.assertEqual(repository.get_notification_rule(self.sid), self.initial)
        finally:
            with db.get_conn() as conn:
                conn.execute("DROP TRIGGER audit_rule_failure")
        db.init_db()
        receipt = actions.set_subscription_notification_rule_confirmed(
            arguments, fingerprint, self.context
        )
        self.assertTrue(receipt.ok)
        with self.assertRaises(AgentToolError) as raised:
            actions.set_subscription_notification_rule_confirmed(
                arguments, fingerprint, self.context
            )
        self.assertEqual(raised.exception.code, "confirmation_stale")
        actual = repository.get_notification_rule(self.sid)
        assert actual is not None
        self.assertTrue(actual["notify_on_satisfied"])
        self.assertEqual(actual["revision"], 2)

    def test_new_confirmation_after_reset_recreate_is_valid(self) -> None:
        self._replace_after_read(self.sid)
        arguments = {"subscription_number": self.sid, "enabled": True}
        _, fingerprint = actions.prepare_set_subscription_notification_rule(
            arguments, self.context
        )
        result = actions.set_subscription_notification_rule_confirmed(
            arguments, fingerprint, self.context
        )
        self.assertTrue(result.ok)
        _, reset_fingerprint = actions.prepare_reset_subscription_notification_rule(
            {"subscription_number": self.sid},
            self.context,
        )
        reset = actions.reset_subscription_notification_rule_confirmed(
            {"subscription_number": self.sid},
            reset_fingerprint,
            self.context,
        )
        self.assertTrue(reset.ok)
        current = repository.get_notification_rule(self.sid)
        assert current is not None
        self.assertFalse(current["explicit"])
        self.assertEqual(current["revision"], 0)
