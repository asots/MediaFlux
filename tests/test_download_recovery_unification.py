"""启动与存活期恢复共享策略，已确认后端事实不得被恢复器改写。"""

from __future__ import annotations

import unittest
from app import database as db
from tests.support import isolated_test_database


class DownloadRecoveryUnificationTests(unittest.TestCase):
    def setUp(self):
        self.path = self.enterContext(isolated_test_database("mediaflux.db"))

    def share(self, key, *, status="submitting", backend="submitting", stale=True):
        request, _ = db.create_share_transfer_request(key, title=key, origin="web")
        db.update_download_request(
            request,
            status=status,
            gy_status=backend,
            gy_task_id="acknowledged-task" if backend not in ("", "submitting") else "",
            gy_task_ids='["acknowledged-task"]'
            if backend not in ("", "submitting")
            else "[]",
            error="保留的诊断",
            gy_target_dir="preserved-target",
        )
        if stale:
            with db.get_conn() as conn:
                conn.execute(
                    "UPDATE download_requests SET updated_at='2000-01-01 00:00:00' WHERE id=?",
                    (request,),
                )
        return request

    def test_boot_preserves_confirmed_share_backend_facts(self):
        requests = {
            backend: self.share(backend, backend=backend)
            for backend in ("submitted", "downloading", "completed", "failed")
        }
        before = {
            key: dict(db.get_download_request(rid)) for key, rid in requests.items()
        }
        db.init_db()
        for backend, rid in requests.items():
            with self.subTest(backend=backend):
                row = db.get_download_request(rid)
                self.assertEqual(row["status"], "manual_review")
                for field in (
                    "gy_status",
                    "gy_task_id",
                    "gy_task_ids",
                    "gy_target_dir",
                ):
                    self.assertEqual(row[field], before[backend][field], field)
                self.assertIn("保留的诊断", row["error"])

    def test_live_recovery_covers_stale_legacy_shares_but_not_recent_or_terminal(self):
        expected = [
            self.share(f"old-{status}-{backend}", status=status, backend=backend)
            for status in ("pending", "submitting")
            for backend in ("", "submitting", "completed", "failed")
        ]
        controls = [self.share("recent", stale=False)]
        controls += [
            self.share(f"terminal-{status}", status=status, backend="completed")
            for status in ("completed", "failed", "cancelled", "resubmitted")
        ]
        before = {rid: dict(db.get_download_request(rid)) for rid in controls}
        self.assertEqual(db.recover_stale_submitting_download_requests(), len(expected))
        for rid in expected:
            self.assertEqual(db.get_download_request(rid)["status"], "manual_review")
        for rid in controls:
            self.assertEqual(dict(db.get_download_request(rid)), before[rid])
        self.assertEqual(db.recover_stale_submitting_download_requests(), 0)

    def regular(self, key="regular"):
        rid, _ = db.create_download_request(key, "magnet")
        self.assertTrue(db.claim_download_request(rid, "both"))
        db.update_download_request(rid, qb_status="completed", qb_task_id="known-qb")
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE download_requests SET updated_at='2000-01-01 00:00:00' WHERE id=?",
                (rid,),
            )
        return rid

    def test_boot_and_live_project_the_same_mixed_history_without_late_reactivation(
        self,
    ):
        snapshots = []
        fields = (
            "status",
            "qb_status",
            "gy_status",
            "qb_task_id",
            "gy_task_id",
            "gy_task_ids",
            "error",
        )
        for mode in ("boot", "live"):
            with self.subTest(mode=mode), isolated_test_database():
                ids = [self.regular()]
                ids += [
                    self.share(f"share-{value}", backend=value)
                    for value in ("", "submitting", "submitted", "completed", "failed")
                ]
                ordinary, _ = db.create_download_request("untouched", "magnet")
                if mode == "boot":
                    db.init_db()
                else:
                    self.assertEqual(
                        db.recover_stale_submitting_download_requests(), len(ids)
                    )
                snapshots.append(
                    [
                        tuple(db.get_download_request(rid)[key] for key in fields)
                        for rid in ids
                    ]
                )
                self.assertEqual(
                    db.count_download_requests_requiring_attention(), len(ids)
                )
                self.assertEqual(db.get_download_request(ordinary)["status"], "pending")
                before = [dict(db.get_download_request(rid)) for rid in ids]
                for rid in ids:
                    self.assertIsNone(
                        db.finalize_download_request_submission(
                            rid,
                            ("guangya",),
                            gy_status="completed",
                            error="late-success",
                        )
                    )
                self.assertEqual(
                    [dict(db.get_download_request(rid)) for rid in ids], before
                )
                db.init_db()
                self.assertEqual(
                    [dict(db.get_download_request(rid)) for rid in ids], before
                )
                self.assertEqual(db.recover_stale_submitting_download_requests(), 0)
        self.assertEqual(snapshots[0], snapshots[1])

    def test_live_empty_timeout_keeps_recent_submission_active(self):
        from typing import Any, cast

        rid = self.share("recent-default-timeout", stale=False)
        before = dict(db.get_download_request(rid))
        for value in (None, 0, 15):
            with self.subTest(value=value):
                self.assertEqual(
                    db.recover_stale_submitting_download_requests(cast(Any, value)), 0
                )
                self.assertEqual(dict(db.get_download_request(rid)), before)

    def test_second_family_sql_failure_rolls_back_all_request_facts(self):
        import sqlite3

        ids = [self.regular(), self.share("share-fault", backend="completed")]
        before = [dict(db.get_download_request(rid)) for rid in ids]
        with db.get_conn() as conn:
            conn.execute(
                f"CREATE TRIGGER fail_share_recovery BEFORE UPDATE ON download_requests WHEN NEW.id={ids[-1]} AND NEW.status='manual_review' BEGIN SELECT RAISE(ABORT,'recovery-write-fault'); END"
            )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "recovery-write-fault"):
            db.recover_stale_submitting_download_requests()
        self.assertEqual([dict(db.get_download_request(rid)) for rid in ids], before)
        with db.get_conn() as conn:
            conn.execute("DROP TRIGGER fail_share_recovery")
        self.assertEqual(db.recover_stale_submitting_download_requests(), 2)
        self.assertEqual(db.get_download_request(ids[-1])["gy_status"], "completed")
        self.assertEqual(db.recover_stale_submitting_download_requests(), 0)

    def test_real_exit_mid_recovery_and_zip_restore_preserve_backend_facts(self):
        import subprocess
        import sys
        from app.modules import backup
        from tests.test_ten_pass_process_restore_audit import runtime_paths

        ids = [self.regular(), self.share("share-process", backend="completed")]
        before = [dict(db.get_download_request(rid)) for rid in ids]
        script = r"""
import tests
import os, sys, socket
from contextlib import contextmanager
from unittest.mock import patch
from app import database as db
socket.socket.connect = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("network forbidden"))
db.configure_database(sys.argv[1], test_mode=True)
actual = db.get_conn
count = 0
def interrupt(sql):
 global count
 if sql.startswith("UPDATE download_requests SET status='manual_review'"):
  count += 1
  if count == 2: os._exit(47)
@contextmanager
def traced():
 with actual() as conn:
  conn.set_trace_callback(interrupt)
  yield conn
with patch.object(db, "get_conn", traced):
 db.recover_stale_submitting_download_requests()
raise AssertionError("recovery window not reached")
"""
        child = subprocess.run(
            [sys.executable, "-c", script, str(self.path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(child.returncode, 47, child.stderr)
        self.assertEqual([dict(db.get_download_request(rid)) for rid in ids], before)
        paths = runtime_paths(self.path)
        archive = backup.create_backup(paths, reason="interrupted-recovery")
        backup.verify_backup(archive)
        fields = (
            "status",
            "qb_status",
            "gy_status",
            "qb_task_id",
            "gy_task_id",
            "gy_task_ids",
            "error",
        )
        db.init_db()
        expected = [
            tuple(db.get_download_request(rid)[key] for key in fields) for rid in ids
        ]
        self.assertEqual(db.get_download_request(ids[-1])["gy_status"], "completed")
        backup.restore_backup(paths, archive)
        db.init_db()
        self.assertEqual(
            [tuple(db.get_download_request(rid)[key] for key in fields) for rid in ids],
            expected,
        )
        self.assertEqual(db.recover_stale_submitting_download_requests(), 0)
