"""规则与播放历史经真实 ZIP/SQLite 恢复后保留身份、租约和单一读取合同。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from app import database as db
from app.modules import backup
from app.repositories import media_automation_rules as rules
from app.repositories import media_proxy
from app.runtime_paths import RuntimePaths
from tests.support import isolated_test_database


class RuleAndPlaybackRestoreAuditTests(unittest.TestCase):
    @staticmethod
    def _paths(database_path):
        root = database_path.parent
        paths = RuntimePaths(
            root / "program",
            root,
            root,
            root / "cache",
            root / "logs",
            root / "strm",
            root / "trash",
        )
        paths.ensure_writable_dirs()
        paths.env_file.write_text("WEB_PORT=1258\n", encoding="utf-8")
        return paths

    def _seed(self):
        self.clock = datetime(2026, 1, 2, 8, tzinfo=timezone.utc)
        saved = []
        for _ in range(2):
            row = rules.save_rule(
                "restore-audit",
                kind="daily_summary",
                settings={"hour": 8, "minute": 0},
                enabled=True,
                next_run_at=self.clock.isoformat(),
            )
            assert row is not None
            saved.append(row)
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE media_automation_rules SET settings_json=? WHERE id=?",
                ('{"private_fixture":', saved[0]["id"]),
            )
        for session in ("restored-session", "restored-session", ""):
            media_proxy.record_media_proxy_playback_attempt(
                instance_id=1,
                playback_session_key=session,
                route_class="guangya_direct",
                method="GET",
                status_code=302,
                source="guangya",
                total_latency_ms=20,
            )
        return saved[0], saved[1]

    @staticmethod
    def _snapshot():
        with db.get_conn() as conn:
            raw = {
                table: [
                    dict(row)
                    for row in conn.execute(f"SELECT * FROM {table} ORDER BY id")
                ]
                for table in (
                    "media_automation_rules",
                    "media_proxy_playback_records",
                    "media_proxy_playback_sessions",
                )
            }
        return {
            "raw": raw,
            "rules": rules.list_rules("restore-audit"),
            "records": media_proxy.list_media_proxy_playback_records(instance_id=1),
            "sessions": media_proxy.list_media_proxy_playback_sessions(instance_id=1),
            "failure_summary": media_proxy.get_media_proxy_playback_failure_summary(
                instance_id=1
            ),
        }

    def test_backup_restore_preserves_bad_payload_and_valid_lease_without_replay(self):
        with isolated_test_database("mediaflux.db") as path:
            paths = self._paths(path)
            broken, good = self._seed()
            lease = rules.claim_due_rules(self.clock)[0]
            expected = self._snapshot()
            archive = backup.create_backup(paths)
            media_proxy.clear_media_proxy_playback_records()
            rules.delete_rule(
                "restore-audit", good["id"], expected_revision=good["revision"]
            )
            self.assertNotEqual(self._snapshot(), expected)
            backup.restore_backup(paths, archive)
            db.configure_database(path, test_mode=True)
            db.init_db()
            self.assertEqual(self._snapshot(), expected)
            self.assertEqual(expected["records"]["total"], 3)
            self.assertEqual(expected["sessions"]["items"][0]["request_count"], 2)
            self.assertEqual(expected["sessions"]["unlinked_total"], 1)
            self.assertEqual(
                rules.claim_due_rules(self.clock + timedelta(minutes=4)), []
            )
            reclaimed = rules.claim_due_rules(self.clock + timedelta(minutes=5))
            self.assertEqual([row["id"] for row in reclaimed], [good["id"]])
            self.assertFalse(
                rules.finish_rule(
                    good["id"], lease["lease_token"], self.clock.isoformat()
                )
            )
            self.assertTrue(
                rules.get_rule("restore-audit", broken["id"])["settings_error"]
            )
            self.assertFalse(backup.recover_pending_restore(paths))

    def test_interrupted_restore_recovers_live_projection_and_raw_history_once(self):
        with isolated_test_database("mediaflux.db") as path:
            paths = self._paths(path)
            _, good = self._seed()
            archive = backup.create_backup(paths)
            rules.save_rule(
                "restore-audit",
                kind="daily_summary",
                settings={"hour": 9, "minute": 30},
                enabled=False,
                next_run_at=self.clock.isoformat(),
                rule_id=good["id"],
                expected_revision=good["revision"],
            )
            media_proxy.clear_media_proxy_playback_records()
            expected = self._snapshot()
            replace = backup.os.replace
            interrupted = False

            def replace_then_interrupt(source, destination):
                nonlocal interrupted
                result = replace(source, destination)
                if (
                    not interrupted
                    and Path(destination) == path
                    and Path(source).name.endswith(".tmp")
                ):
                    interrupted = True
                    raise KeyboardInterrupt("fixture after database replacement")
                return result

            with (
                patch.object(backup.os, "replace", replace_then_interrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                backup.restore_backup(paths, archive)
            self.assertTrue(interrupted)
            self.assertTrue(backup.recover_pending_restore(paths))
            db.configure_database(path, test_mode=True)
            db.init_db()
            self.assertEqual(self._snapshot(), expected)
            self.assertFalse(backup.recover_pending_restore(paths))
            self.assertEqual(self._snapshot(), expected)
