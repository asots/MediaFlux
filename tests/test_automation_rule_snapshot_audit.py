"""主动规则保存回执不能被提交后的另一更新/删除替换。"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from unittest.mock import patch

from app import database as db
from app.repositories import media_automation_rules as repository
from tests.support import IsolatedDatabaseTestCase


class AutomationRuleSnapshotAuditTests(IsolatedDatabaseTestCase):
    def setUp(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM media_automation_rules")
        self.stamp = datetime.now().astimezone().isoformat()

    def _save(self, **updates):
        return repository.save_rule(
            "receipt-audit",
            kind="daily_summary",
            settings={"hour": 21, "minute": 0},
            next_run_at=self.stamp,
            enabled=True,
            **updates,
        )

    def test_insert_receipt_reports_own_version_when_next_writer_updates(self):
        connect = db.get_conn
        changed = False

        @contextmanager
        def hooked():
            nonlocal changed
            with connect() as conn:
                yield conn
                wrote = conn.in_transaction
            if wrote and not changed:
                changed = True
                with connect() as writer:
                    writer.execute(
                        "UPDATE media_automation_rules SET revision=revision+1,enabled=0 WHERE owner_digest='receipt-audit'"
                    )

        with patch.object(db, "get_conn", hooked):
            result = self._save()
        self.assertTrue(changed)
        assert result is not None
        self.assertEqual((result["revision"], result["enabled"]), (1, True))
        actual = repository.get_rule("receipt-audit", result["id"])
        assert actual is not None
        self.assertEqual((actual["revision"], actual["enabled"]), (2, False))

    def test_update_receipt_does_not_disappear_after_later_delete(self):
        saved = self._save()
        assert saved is not None
        connect = db.get_conn
        deleted = False

        @contextmanager
        def hooked():
            nonlocal deleted
            with connect() as conn:
                yield conn
                wrote = conn.in_transaction
            if wrote and not deleted:
                deleted = True
                with connect() as writer:
                    writer.execute(
                        "DELETE FROM media_automation_rules WHERE id=?", (saved["id"],)
                    )

        with patch.object(db, "get_conn", hooked):
            result = self._save(rule_id=saved["id"], expected_revision=1)
        self.assertTrue(deleted)
        assert result is not None
        self.assertEqual(result["revision"], 2)
        self.assertEqual(result["id"], saved["id"])
        self.assertIsNone(repository.get_rule("receipt-audit", saved["id"]))

    def test_conflict_preserves_existing_rule_and_normal_receipt_shape(self):
        saved = self._save()
        assert saved is not None
        self.assertEqual(saved, repository.get_rule("receipt-audit", saved["id"]))
        self.assertIsNone(self._save(rule_id=saved["id"], expected_revision=2))
        self.assertEqual(saved, repository.get_rule("receipt-audit", saved["id"]))
