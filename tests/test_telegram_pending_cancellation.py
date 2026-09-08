"""下载尚未提交时的用户取消必须直接持久化，不经过submitting中间态。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app import database as db
from app.bot import handlers
from tests.support import isolated_test_database


class TelegramPendingCancellationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(isolated_test_database())
        self.bot = Mock()

    def invoke(self, request_id, decision="cancel"):
        call = SimpleNamespace(
            id="cancel",
            data="tgc:opaque",
            from_user=SimpleNamespace(id=9),
            message=SimpleNamespace(chat=SimpleNamespace(id=100), message_id=23),
        )
        store = SimpleNamespace(
            claim=lambda *a, **kw: {
                "operation": "download_request",
                "decision": decision,
                "value": {"request_id": request_id, "target": "qb"},
            }
        )
        with patch(
            "app.modules.telegram_write_confirmations.get_telegram_write_confirmation_store",
            return_value=store,
        ):
            handlers._handle_write_confirmation_callback(
                self.bot, call, SimpleNamespace()
            )

    def test_cancel_does_not_depend_on_a_second_database_commit(self):
        request, _ = db.create_download_request(
            "cancel-pending", "magnet", chat_id="100", user_id="9"
        )
        with patch.object(
            db,
            "update_download_request",
            side_effect=OSError("second commit unavailable"),
        ):
            self.invoke(request)
        self.assertEqual(db.get_download_request(request)["status"], "cancelled")
        db.init_db()
        self.assertEqual(db.get_download_request(request)["status"], "cancelled")
        self.assertEqual(db.count_download_requests_requiring_attention(), 0)

    def test_webpage_rejection_does_not_leave_a_submitting_orphan(self):
        request, _ = db.create_download_request(
            "webpage-pending",
            "http",
            source_value="http://192.168.0.195:1258/guangya/offline",
            chat_id="100",
            user_id="9",
        )
        with patch.object(
            db,
            "update_download_request",
            side_effect=OSError("second commit unavailable"),
        ):
            self.invoke(request, "confirm")
        row = db.get_download_request(request)
        self.assertEqual(row["status"], "cancelled")
        self.assertIn("普通网页", row["error"])
        db.init_db()
        self.assertEqual(db.get_download_request(request)["status"], "cancelled")

    def test_atomic_cancel_and_submission_are_mutually_exclusive(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading

        for index in range(8):
            request, _ = db.create_download_request(f"cancel-race-{index}", "magnet")
            gate = threading.Barrier(2)

            def cancel():
                gate.wait(timeout=5)
                return db.cancel_pending_download_request(request)

            def submit():
                gate.wait(timeout=5)
                return db.claim_download_request(request, "qb")

            with ThreadPoolExecutor(max_workers=2) as pool:
                first, second = pool.submit(cancel), pool.submit(submit)
                cancelled, submitted = first.result(timeout=5), second.result(timeout=5)
            self.assertNotEqual(cancelled, submitted)
            row = db.get_download_request(request)
            self.assertEqual(row["status"], "cancelled" if cancelled else "submitting")
            self.assertEqual(row["qb_status"], "" if cancelled else "submitting")
            self.assertFalse(db.cancel_pending_download_request(request))

    def test_cancel_is_not_an_alternate_submission_target(self):
        request, _ = db.create_download_request("reject-cancel-target", "magnet")
        self.assertFalse(db.claim_download_request(request, "cancelled"))
        self.assertEqual(db.get_download_request(request)["status"], "pending")
        self.assertTrue(db.cancel_pending_download_request(request))
        self.assertFalse(db.claim_download_request(request, "qb"))


class TelegramCancellationProcessRestoreTests(unittest.TestCase):
    def test_process_exit_after_cancel_commit_keeps_terminal_intent_through_zip_restore(
        self,
    ):
        from pathlib import Path
        import subprocess
        import sys
        from app.modules import backup
        from tests.test_ten_pass_process_restore_audit import runtime_paths

        with isolated_test_database("mediaflux.db") as path:
            paths = runtime_paths(path)
            script = r"""
import tests
import os, sys, socket
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch
from app import database as db
from app.bot import handlers
from app.repositories import download_requests as repository
socket.socket.connect = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("network forbidden"))
db.configure_database(sys.argv[1], test_mode=True)
db.init_db()
request, _ = db.create_download_request("process-cancel", "magnet", chat_id="100", user_id="9")
actual = repository.get_conn
@contextmanager
def exit_after_cancel_commit():
 with actual() as conn:
  yield conn
  cancelled = conn.execute("SELECT status FROM download_requests WHERE id=?", (request,)).fetchone()[0] == "cancelled"
 if cancelled:
  os._exit(41)
call = SimpleNamespace(id="cancel", data="tgc:opaque", from_user=SimpleNamespace(id=9), message=SimpleNamespace(chat=SimpleNamespace(id=100), message_id=23))
store = SimpleNamespace(claim=lambda *a, **kw: {"operation":"download_request", "decision":"cancel", "value":{"request_id":request,"target":"qb"}})
with patch("app.modules.telegram_write_confirmations.get_telegram_write_confirmation_store", return_value=store), patch.object(repository, "get_conn", exit_after_cancel_commit):
 handlers._handle_write_confirmation_callback(Mock(), call, SimpleNamespace())
raise AssertionError("cancel commit window not reached")
"""
            result = subprocess.run(
                [sys.executable, "-c", script, str(path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 41, result.stderr)
            with db.get_conn() as conn:
                row = conn.execute("SELECT * FROM download_requests").fetchone()
                request = row["id"]
                self.assertEqual(row["status"], "cancelled")
                self.assertIsNotNone(row["completed_at"])
            archive = backup.create_backup(paths, reason="committed-cancel")
            backup.verify_backup(archive)
            db.update_download_request(
                request, status="failed", error="post-backup-change"
            )
            backup.restore_backup(paths, archive)
            db.init_db()
            restored = dict(db.get_download_request(request))
            self.assertEqual(restored["status"], "cancelled")
            self.assertFalse(db.cancel_pending_download_request(request))
            self.assertFalse(db.claim_download_request(request, "qb"))
            self.assertEqual(db.count_download_requests_requiring_attention(), 0)
            db.init_db()
            self.assertEqual(dict(db.get_download_request(request)), restored)
            self.assertTrue(Path(path).exists())


class LegacyPendingCancellationRecoveryTests(unittest.TestCase):
    def test_boot_and_live_recovery_decode_old_cancel_intent_but_not_unknown_backend_work(
        self,
    ):
        for boot, legacy_status in (
            (False, "submitting"),
            (True, "submitting"),
            (False, "manual_review"),
            (True, "manual_review"),
        ):
            with (
                self.subTest(boot=boot, legacy_status=legacy_status),
                isolated_test_database(),
            ):
                cancelled, _ = db.create_download_request("old-cancel", "magnet")
                ordinary, _ = db.create_download_request("ordinary-pending", "magnet")
                unknown, _ = db.create_download_request("unknown-work", "magnet")
                with db.get_conn() as conn:
                    conn.execute(
                        "UPDATE download_requests SET status='submitting',targets='cancelled',updated_at='2000-01-01 00:00:00' WHERE id IN (?,?)",
                        (cancelled, unknown),
                    )
                    conn.execute(
                        "UPDATE download_requests SET status=? WHERE id=?",
                        (legacy_status, cancelled),
                    )
                    conn.execute(
                        "UPDATE download_requests SET qb_status='submitting' WHERE id=?",
                        (unknown,),
                    )
                identity_only, _ = db.create_download_request(
                    "unknown-with-id", "magnet"
                )
                db.update_download_request(
                    identity_only,
                    status="submitting",
                    targets="cancelled",
                    qb_task_id="a" * 40,
                )
                with db.get_conn() as conn:
                    conn.execute(
                        "UPDATE download_requests SET updated_at='2000-01-01 00:00:00' WHERE id=?",
                        (identity_only,),
                    )
                if boot:
                    db.init_db()
                else:
                    self.assertEqual(db.recover_stale_submitting_download_requests(), 2)
                self.assertEqual(
                    db.get_download_request(cancelled)["status"], "cancelled"
                )
                self.assertIsNotNone(db.get_download_request(cancelled)["completed_at"])
                self.assertEqual(db.get_download_request(ordinary)["status"], "pending")
                self.assertEqual(
                    db.get_download_request(unknown)["status"], "manual_review"
                )
                self.assertEqual(
                    db.get_download_request(identity_only)["status"], "manual_review"
                )
                self.assertEqual(db.count_download_requests_requiring_attention(), 2)
                self.assertEqual(db.recover_stale_submitting_download_requests(), 0)
