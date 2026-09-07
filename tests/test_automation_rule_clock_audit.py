"""主动规则按真实时间领取，不能用不同 ISO 时区/精度的字面值比较。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app import database as db
from app.repositories import media_automation_rules as repository
from tests.support import IsolatedDatabaseTestCase


class AutomationRuleClockAuditTests(IsolatedDatabaseTestCase):
    def setUp(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM media_automation_rules")
        self.clock = datetime(2026, 1, 2, 0, 0, 0, 123456, tzinfo=timezone.utc)

    def _save(self, value):
        result = repository.save_rule(
            "clock-audit",
            kind="daily_summary",
            settings={"hour": 21, "minute": 0},
            enabled=True,
            next_run_at=value,
        )
        assert result is not None
        return result

    def test_equal_microsecond_deadline_is_due_but_next_microsecond_is_not(self):
        due = self._save(self.clock.isoformat())
        future = self._save((self.clock + timedelta(microseconds=1)).isoformat())
        claimed = repository.claim_due_rules(self.clock)
        self.assertEqual([item["id"] for item in claimed], [due["id"]])
        self.assertEqual(repository.claim_due_rules(self.clock), [])
        self.assertEqual(
            [
                item["id"]
                for item in repository.claim_due_rules(
                    self.clock + timedelta(microseconds=1)
                )
            ],
            [future["id"]],
        )

    def test_batch_order_and_limit_use_instants_not_timezone_spelling(self):
        oldest = self._save(
            (self.clock - timedelta(minutes=2))
            .astimezone(timezone(timedelta(hours=14)))
            .isoformat()
        )
        middle = self._save(
            (self.clock - timedelta(minutes=1))
            .astimezone(timezone(timedelta(hours=-10)))
            .isoformat()
        )
        latest = self._save(
            self.clock.astimezone(timezone(timedelta(hours=8))).isoformat()
        )
        future = self._save(
            (self.clock + timedelta(minutes=1))
            .astimezone(timezone(timedelta(hours=-12)))
            .isoformat()
        )
        first = repository.claim_due_rules(self.clock, limit=2)
        self.assertEqual([row["id"] for row in first], [oldest["id"], middle["id"]])
        second = repository.claim_due_rules(self.clock, limit=2)
        self.assertEqual([row["id"] for row in second], [latest["id"]])
        self.assertNotIn(future["id"], [row["id"] for row in first + second])

    def test_restart_in_other_offset_preserves_exact_lease_expiry_and_rejects_old_receipt(
        self,
    ):
        row = self._save((self.clock - timedelta(seconds=1)).isoformat())
        leased = repository.claim_due_rules(self.clock)[0]
        db.init_db()
        another_clock = self.clock.astimezone(timezone(timedelta(hours=-7)))
        self.assertEqual(
            repository.claim_due_rules(
                another_clock + timedelta(minutes=5) - timedelta(microseconds=1)
            ),
            [],
        )
        second = repository.claim_due_rules(another_clock + timedelta(minutes=5))
        self.assertEqual([item["id"] for item in second], [row["id"]])
        self.assertNotEqual(second[0]["lease_token"], leased["lease_token"])
        tomorrow = (another_clock + timedelta(days=1)).isoformat()
        self.assertFalse(
            repository.finish_rule(row["id"], leased["lease_token"], tomorrow)
        )
        self.assertTrue(
            repository.finish_rule(row["id"], second[0]["lease_token"], tomorrow)
        )
        self.assertEqual(
            repository.claim_due_rules(self.clock + timedelta(minutes=6)), []
        )

    def test_historical_offset_and_invalid_dates_are_not_rewritten_or_block_valid_rows(
        self,
    ):
        old_spelling = self.clock.astimezone(timezone(timedelta(hours=8))).isoformat()
        good = self._save(old_spelling)
        broken = self._save(old_spelling)
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE media_automation_rules SET next_run_at='not-a-date' WHERE id=?",
                (broken["id"],),
            )
        db.init_db()
        result = repository.claim_due_rules(self.clock)
        self.assertEqual([row["id"] for row in result], [good["id"]])
        self.assertEqual(
            repository.get_rule("clock-audit", good["id"])["next_run_at"], old_spelling
        )
        self.assertEqual(
            repository.get_rule("clock-audit", broken["id"])["next_run_at"],
            "not-a-date",
        )
