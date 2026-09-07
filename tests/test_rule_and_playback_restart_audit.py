"""用独立进程和真实事务验证规则租约、坏记录修复及播放记录回滚。"""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from app import database as db
from app.repositories import media_automation_rules as rules
from app.repositories import media_proxy
from tests.support import IsolatedDatabaseTestCase


class RuleAndPlaybackRestartAuditTests(IsolatedDatabaseTestCase):
    def setUp(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM media_automation_rules")
            conn.execute("DELETE FROM media_proxy_playback_records")
            conn.execute("DELETE FROM media_proxy_playback_sessions")
        self.clock = datetime(2026, 1, 2, 8, tzinfo=timezone.utc)

    def _rule(self):
        row = rules.save_rule(
            "restart-audit",
            kind="daily_summary",
            settings={"hour": 8, "minute": 0},
            enabled=True,
            next_run_at=self.clock.isoformat(),
        )
        assert row is not None
        return row

    def _child(self, body, *args, expected_exit=0):
        script = (
            """
import tests
import json, os, sys
from datetime import datetime
from app import database as db
from app.repositories import media_automation_rules as rules
db.configure_database(sys.argv[1], test_mode=True)
db.init_db()
"""
            + body
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.test_db_path), *args],
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, expected_exit, result.stderr)
        return result.stdout

    def test_committed_child_lease_survives_abrupt_exit_and_expires_once(self):
        good = self._rule()
        broken = self._rule()
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE media_automation_rules SET settings_json='null' WHERE id=?",
                (broken["id"],),
            )
        output = self._child(
            "claimed=rules.claim_due_rules(datetime.fromisoformat(sys.argv[2]))\n"
            "print(json.dumps(claimed), flush=True)\nos._exit(17)\n",
            self.clock.isoformat(),
            expected_exit=17,
        )
        first = json.loads(output)
        self.assertEqual([row["id"] for row in first], [good["id"]])
        self.assertEqual(rules.claim_due_rules(self.clock + timedelta(minutes=4)), [])
        second = rules.claim_due_rules(self.clock + timedelta(minutes=5))
        self.assertEqual([row["id"] for row in second], [good["id"]])
        self.assertNotEqual(first[0]["lease_token"], second[0]["lease_token"])
        tomorrow = (self.clock + timedelta(days=1)).isoformat()
        self.assertFalse(
            rules.finish_rule(good["id"], first[0]["lease_token"], tomorrow)
        )
        self.assertTrue(
            rules.finish_rule(good["id"], second[0]["lease_token"], tomorrow)
        )
        self.assertEqual(rules.claim_due_rules(self.clock + timedelta(minutes=6)), [])
        self.assertTrue(rules.get_rule("restart-audit", broken["id"])["settings_error"])

    def test_uncommitted_child_claim_is_rolled_back_after_abrupt_exit(self):
        good = self._rule()
        self._child(
            """
with db.get_conn() as conn:
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "UPDATE media_automation_rules SET lease_token='abandoned',"
        "lease_until='2099-01-01T00:00:00+00:00' WHERE id=?", (sys.argv[2],)
    )
    os._exit(17)
""",
            good["id"],
            expected_exit=17,
        )
        current = rules.get_rule("restart-audit", good["id"])
        self.assertEqual((current["lease_token"], current["lease_until"]), ("", ""))
        self.assertEqual(
            [row["id"] for row in rules.claim_due_rules(self.clock)], [good["id"]]
        )

    def test_repair_after_reopen_fences_the_previous_worker(self):
        row = self._rule()
        old = rules.claim_due_rules(self.clock)[0]
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE media_automation_rules SET settings_json='[]' WHERE id=?",
                (row["id"],),
            )
        db.init_db()
        damaged = rules.get_rule("restart-audit", row["id"])
        self.assertTrue(damaged["settings_error"])
        fixed = rules.save_rule(
            "restart-audit",
            kind="daily_summary",
            settings={"hour": 9, "minute": 0},
            enabled=True,
            next_run_at=self.clock.isoformat(),
            rule_id=row["id"],
            expected_revision=damaged["revision"],
        )
        self.assertNotIn("settings_error", fixed)
        self.assertFalse(rules.owns_lease(row["id"], old["lease_token"]))
        self.assertFalse(
            rules.finish_rule(row["id"], old["lease_token"], self.clock.isoformat())
        )
        self.assertEqual(len(rules.claim_due_rules(self.clock)), 1)
        self.assertEqual(rules.claim_due_rules(self.clock), [])

    def test_failed_playback_transaction_never_leaves_a_partial_session(self):
        original = media_proxy.get_conn

        @contextmanager
        def interrupted():
            with original() as conn:
                yield conn
                raise RuntimeError("fixture interrupted before commit")

        arguments = dict(
            instance_id=1,
            playback_session_key="restart-session",
            route_class="guangya_direct",
            method="GET",
            status_code=302,
            source="guangya",
        )
        with (
            patch.object(media_proxy, "get_conn", interrupted),
            self.assertRaisesRegex(RuntimeError, "before commit"),
        ):
            media_proxy.record_media_proxy_playback_attempt(**arguments)
        db.init_db()
        self.assertEqual(media_proxy.list_media_proxy_playback_records()["total"], 0)
        self.assertEqual(media_proxy.list_media_proxy_playback_sessions()["total"], 0)
        media_proxy.record_media_proxy_playback_attempt(**arguments)
        sessions = media_proxy.list_media_proxy_playback_sessions()
        self.assertEqual(sessions["total"], 1)
        self.assertEqual(sessions["items"][0]["request_count"], 1)
        self.assertEqual(sessions["items"][0]["redirect_request_count"], 1)
