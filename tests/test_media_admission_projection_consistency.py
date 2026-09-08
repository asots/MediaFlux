"""运行中批量巡检与即时/启动投影使用一致的媒体下载准入状态。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app import database as db
from tests.support import isolated_test_database


class MediaAdmissionProjectionConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.path = self.enterContext(isolated_test_database("mediaflux.db"))
        self.enterContext(patch.object(db, "now", return_value="2026-09-08 12:00:00"))
        self.subscription = db.add_media_subscription(
            provider="tmdb",
            external_id="1",
            tmdb_id="1",
            media_type="tv",
            title="投影一致性",
            monitor_mode="missing",
            action="confirm",
            download_target="qb",
            check_interval_minutes=60,
        )

    def seed(self, episode=1, **fields):
        key = f"tmdb:1:tv:S01E{episode:03d}"
        candidate = db.replace_media_subscription_candidates(
            self.subscription,
            key,
            season=1,
            episode=episode,
            candidates=[{"result_id": f"candidate-{episode}", "title": key}],
            expires_at="2099-01-01 00:00:00",
        )[0]
        admission = db.claim_media_download_admission(
            media_key=key,
            tmdb_id="1",
            media_type="tv",
            subscription_id=self.subscription,
            candidate_id=candidate,
            season=1,
            episode=episode,
            subscription_revision=1,
        )
        self.assertTrue(
            db.begin_media_download_dispatch(
                admission, subscription_id=self.subscription, subscription_revision=1
            )
        )
        request, _ = db.create_download_request(
            f"projection-{episode}", "magnet", admission_id=admission
        )
        db.update_download_request(request, **fields)
        return request, admission, key

    @staticmethod
    def view(admission):
        with db.get_conn() as conn:
            return tuple(
                conn.execute(
                    "SELECT status,error,completed_at FROM media_download_admissions WHERE id=?",
                    (admission,),
                ).fetchone()
            )

    def test_batch_and_immediate_projection_have_identical_business_results(self):
        cases = [
            {"status": "failed", "error": "明确提交失败"},
            {"status": "manual_review", "error": "远端结果未知，请人工核验"},
            {
                "status": "completed",
                "organize_status": "failed",
                "organize_error": "整理失败",
            },
            {"status": "completed", "organize_started": -1},
            {"status": "completed", "strm_status": "failed", "strm_error": "STRM失败"},
            {"status": "completed", "local_import_status": "failed"},
            {"status": "completed"},
            {"status": "submitted"},
            {"status": "downloading"},
        ]
        for episode, fields in enumerate(cases, 1):
            with self.subTest(fields=fields):
                request, admission, _ = self.seed(episode, **fields)
                self.assertEqual(
                    db.sync_media_download_admission_for_request(request), 1
                )
                expected = self.view(admission)
                db.update_media_download_admission(
                    admission, status="dispatching", error="", completed_at=None
                )
                db.reconcile_media_download_admissions(
                    self.subscription, set(), expected_revision=1
                )
                self.assertEqual(self.view(admission), expected)

    def test_live_timeout_is_visible_without_restart_or_releasing_duplicate_guard(self):
        request, admission, _ = self.seed(status="pending")
        self.assertTrue(db.claim_download_request(request, "qb"))
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE download_requests SET updated_at='2000-01-01 00:00:00' WHERE id=?",
                (request,),
            )
        self.assertEqual(db.recover_stale_submitting_download_requests(), 1)
        self.assertEqual(
            db.reconcile_media_download_admissions(
                self.subscription, set(), expected_revision=1
            ),
            1,
        )
        state, error, completed = self.view(admission)
        self.assertEqual(state, "processing")
        self.assertIn("结果未知", error)
        self.assertIsNone(completed)
        self.assertEqual(
            len(db.list_active_media_download_admissions(self.subscription)), 1
        )

    def test_mixed_batch_keeps_local_presence_precedence_and_bounded_reads(self):
        from contextlib import contextmanager

        entries = [
            self.seed(
                i, status=("manual_review" if i % 2 else "failed"), error="batch-error"
            )
            for i in range(1, 51)
        ]
        local_keys = {key for _, _, key in entries[:3]}
        statements = []
        actual = db.get_conn

        @contextmanager
        def traced():
            with actual() as conn:
                conn.set_trace_callback(statements.append)
                try:
                    yield conn
                finally:
                    conn.set_trace_callback(None)

        with patch.object(db, "get_conn", traced):
            self.assertEqual(
                db.reconcile_media_download_admissions(
                    self.subscription, local_keys, expected_revision=1
                ),
                50,
            )
        reads = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
        self.assertEqual(len(reads), 1, reads)
        for index, (_, admission, _) in enumerate(entries):
            status, error, completed = self.view(admission)
            self.assertEqual(
                status,
                "completed"
                if index < 3
                else "processing"
                if index % 2 == 0
                else "failed",
            )
            self.assertEqual(completed is not None, status != "processing")
            if status == "processing":
                self.assertEqual(error, "batch-error")

    def test_restart_and_zip_recovery_preserve_unknown_result_without_duplicate_admission(
        self,
    ):
        import subprocess
        import sys
        from app.modules import backup
        from tests.test_ten_pass_process_restore_audit import runtime_paths

        request, admission, key = self.seed(status="pending")
        paths = runtime_paths(self.path)
        self.assertEqual(self.path, paths.database_path)
        script = r"""
import tests
import os, sys, socket
from app import database as db
socket.socket.connect = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("network forbidden"))
db.configure_database(sys.argv[1], test_mode=True)
assert db.claim_download_request(int(sys.argv[2]), "qb")
with db.get_conn() as conn:
 conn.execute("UPDATE download_requests SET updated_at='2000-01-01 00:00:00' WHERE id=?", (int(sys.argv[2]),))
assert db.recover_stale_submitting_download_requests() == 1
os._exit(43)
"""
        child = subprocess.run(
            [sys.executable, "-c", script, str(paths.database_path), str(request)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(child.returncode, 43, child.stderr)
        archive = backup.create_backup(paths, reason="unprojected-unknown")
        db.init_db()
        self.assertEqual(
            db.reconcile_media_download_admissions(
                self.subscription, set(), expected_revision=1
            ),
            1,
        )
        expected = self.view(admission)
        self.assertEqual(expected[0], "processing")
        self.assertIn("结果未知", expected[1])
        backup.restore_backup(paths, archive)
        db.init_db()
        projected, released = db.reconcile_startup_media_download_admissions()
        self.assertEqual((projected, released), (1, 0))
        self.assertEqual(self.view(admission), expected)
        self.assertEqual(
            len(db.list_active_media_download_admissions(self.subscription)), 1
        )
        self.assertEqual(
            db.reconcile_media_download_admissions(
                self.subscription, {key}, expected_revision=1
            ),
            1,
        )
        self.assertEqual(self.view(admission)[0], "completed")
