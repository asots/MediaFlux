"""分享转存必须使用与普通下载一致的持久认领和迟到结果收尾合同。"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from app import database as db
from app.modules.share_transfer import ShareTransferPreviewStore, create_share_request
from tests.support import isolated_test_database
from tests.test_telegram_guangya_share import FakeShareClient


class ShareTransferSubmissionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(isolated_test_database())
        self.store = ShareTransferPreviewStore()
        self.client = FakeShareClient()
        self.preview = self.store.create(self.client.inspected, "chat", "user")
        self.enterContext(
            patch("app.modules.share_transfer.get", return_value="archive")
        )
        self.tracker = self.enterContext(
            patch(
                "app.modules.download_tracker.get_download_tracker", return_value=Mock()
            )
        ).return_value

    def submit(self, *, injected=True):
        kwargs = {"client": self.client} if injected else {}
        return create_share_request(
            self.preview,
            ["file-1"],
            "target",
            "chat",
            user_id="user",
            store=self.store,
            **kwargs,
        )

    @staticmethod
    def request():
        with db.get_conn() as conn:
            return conn.execute(
                "SELECT * FROM download_requests WHERE kind='guangya_share'"
            ).fetchone()

    def test_request_is_durably_claimed_before_first_provider_write(self):
        observed = []
        actual = self.client.create_dir

        def create(name, parent):
            row = self.request()
            observed.append((row["status"], row["gy_status"], row["targets"]))
            return actual(name, parent)

        self.client.create_dir = create
        result = self.submit()
        self.assertTrue(result["success"])
        self.assertEqual(observed, [("submitting", "submitting", "guangya")])

    def test_recovered_or_cancelled_request_rejects_late_success_and_followup(self):
        for status in ("manual_review", "cancelled", "resubmitted"):
            with self.subTest(status=status):
                with db.get_conn() as conn:
                    conn.execute("DELETE FROM download_log")
                    conn.execute("DELETE FROM download_request_keys")
                    conn.execute("DELETE FROM download_requests")
                self.tracker.reset_mock()

                def complete_after_recovery(*args, **kwargs):
                    row = self.request()
                    db.update_download_request(
                        row["id"],
                        status=status,
                        gy_status=status,
                        error="recovered-owner",
                    )
                    return {"success": True}

                self.client.restore_share = complete_after_recovery
                result = self.submit()
                row = self.request()
                self.assertEqual(row["status"], status)
                self.assertEqual(row["gy_status"], status)
                self.assertEqual(row["error"], "recovered-owner")
                self.assertFalse(result["success"])
                self.assertEqual(result["status"], status)
                self.tracker.reload.assert_not_called()

    def test_client_preparation_failure_is_retryable_without_unknown_cloud_result(self):
        with patch(
            "app.modules.share_transfer.GuangYaClient",
            side_effect=OSError("isolated client startup failure"),
        ):
            first = self.submit(injected=False)
        self.assertFalse(first["success"])
        self.assertEqual(first["status"], "failed")
        self.assertEqual(self.request()["status"], "failed")
        self.assertEqual(self.client.restore_calls, [])
        self.tracker.reload.assert_not_called()
        with patch(
            "app.modules.share_transfer.GuangYaClient", return_value=self.client
        ):
            second = self.submit(injected=False)
        self.assertTrue(second["success"])
        self.assertTrue(second["retried"])
        self.assertEqual(first["request_id"], second["request_id"])
        self.assertEqual(len(self.client.restore_calls), 1)
        self.assertEqual(self.client.close_calls, 1)
        third = self.submit()
        self.assertTrue(third["duplicate"])
        self.assertEqual(len(self.client.restore_calls), 1)

    def test_first_submission_is_visible_to_live_timeout_recovery(self):
        recovered_counts = []

        def complete_after_timeout(*args, **kwargs):
            row = self.request()
            with db.get_conn() as conn:
                conn.execute(
                    "UPDATE download_requests SET updated_at=? WHERE id=?",
                    ("2000-01-01 00:00:00", row["id"]),
                )
            recovered_counts.append(db.recover_stale_submitting_download_requests())
            return {"success": True}

        self.client.restore_share = complete_after_timeout
        result = self.submit()
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "manual_review")
        self.assertEqual(recovered_counts, [1])
        self.assertEqual(db.recover_stale_submitting_download_requests(), 0)
        self.assertEqual(db.list_download_logs(source="guangya_share"), [])
        self.tracker.reload.assert_not_called()

    def test_cancel_during_directory_creation_stops_before_restore(self):
        actual = self.client.create_dir

        def create_after_cancel(*args):
            target = actual(*args)
            db.update_download_request(
                self.request()["id"], status="cancelled", gy_status="cancelled"
            )
            return target

        self.client.create_dir = create_after_cancel
        result = self.submit()
        self.assertEqual(result["status"], "cancelled")
        self.assertFalse(result["success"])
        self.assertEqual(self.client.restore_calls, [])
        self.tracker.reload.assert_not_called()

    def test_concurrent_repeat_does_not_acquire_or_finish_first_submission(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading

        entered, release = threading.Event(), threading.Event()
        actual = self.client.restore_share

        def blocked_restore(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test release timeout")
            return actual(*args)

        self.client.restore_share = blocked_restore
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(self.submit)
            try:
                self.assertTrue(entered.wait(5))
                duplicate = self.submit()
                self.assertTrue(duplicate["duplicate"])
                self.assertTrue(duplicate["accepted"])
                self.assertFalse(duplicate["success"])
                self.assertEqual(duplicate["status"], "submitting")
                self.tracker.reload.assert_not_called()
            finally:
                release.set()
            completed = first.result(timeout=5)
        self.assertEqual(completed["request_id"], duplicate["request_id"])
        self.assertTrue(completed["success"])
        self.assertEqual(len(self.client.restore_calls), 1)
        self.tracker.reload.assert_called_once()

    def test_creation_claim_failure_rolls_back_request_and_key_together(self):
        with patch(
            "app.repositories.download_requests._claim_download_request_conn",
            side_effect=RuntimeError("claim failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "claim failure"):
                self.submit()
        with db.get_conn() as conn:
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM download_requests").fetchone()[0], 0
            )
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM download_request_keys").fetchone()[
                    0
                ],
                0,
            )
        self.assertEqual(self.client.restore_calls, [])

    def test_log_insert_failure_rolls_back_finalization_and_allows_only_same_claim_finish(
        self,
    ):
        request_id, _ = db.create_share_transfer_request("atomic-log", title="atomic")
        fields = dict(
            success=True,
            target_dir_id="target",
            target_dir_name="target",
            title="atomic",
        )
        with db.get_conn() as conn:
            conn.execute(
                "CREATE TRIGGER reject_share_log BEFORE INSERT ON download_log BEGIN SELECT RAISE(ABORT,'injected log failure'); END"
            )
        import sqlite3

        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected log failure"):
            db.finish_share_transfer_request(request_id, **fields)
        self.assertEqual(db.get_download_request(request_id)["status"], "submitting")
        with db.get_conn() as conn:
            conn.execute("DROP TRIGGER reject_share_log")
        self.assertTrue(db.finish_share_transfer_request(request_id, **fields))
        self.assertFalse(db.finish_share_transfer_request(request_id, **fields))
        self.assertEqual(len(db.list_download_logs(source="guangya_share")), 1)

    def test_legacy_pending_duplicate_and_boot_recovery_do_not_replay(self):
        from app.modules.share_transfer import share_request_key

        key = share_request_key("demo123", ["file-1"], "target", "chat", "user")
        legacy, _ = db.create_download_request(key, "guangya_share", title="legacy")
        regular, _ = db.create_download_request("ordinary-pending", "magnet")
        self.assertEqual(self.submit()["status"], "pending")
        self.assertEqual(self.client.restore_calls, [])
        db.init_db()
        self.assertEqual(self.submit()["status"], "manual_review")
        self.assertEqual(db.get_download_request(regular)["status"], "pending")
        recovered = dict(db.get_download_request(legacy))
        db.init_db()
        self.assertEqual(dict(db.get_download_request(legacy)), recovered)
        self.assertEqual(self.client.restore_calls, [])


class ShareTransferProcessRestoreTests(unittest.TestCase):
    def test_exit_during_provider_write_and_zip_restore_keep_manual_gate(self):
        import subprocess
        import sys
        from pathlib import Path
        from app.modules import backup
        from tests.test_ten_pass_process_restore_audit import runtime_paths

        with isolated_test_database("mediaflux.db") as path:
            paths = runtime_paths(path)
            script = r"""
import tests
import os, sys, socket
from pathlib import Path
from unittest.mock import patch
from app import database as db
from app.modules.share_transfer import ShareTransferPreviewStore, create_share_request
from tests.test_telegram_guangya_share import FakeShareClient
socket.socket.connect = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("network forbidden"))
db.configure_database(sys.argv[1], test_mode=True)
db.init_db()
client = FakeShareClient()
store = ShareTransferPreviewStore()
preview = store.create(client.inspected, "chat", "user")
def accepted_then_exit(*args):
 Path(sys.argv[1]).with_suffix(".cloud-marker").write_text("accepted-once")
 os._exit(29)
client.restore_share = accepted_then_exit
with patch("app.modules.share_transfer.get", return_value="archive"):
 create_share_request(preview, ["file-1"], "target", "chat", user_id="user", store=store, client=client)
raise AssertionError("provider window not reached")
"""
            child = subprocess.run(
                [sys.executable, "-c", script, str(path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(child.returncode, 29, child.stderr)
            self.assertEqual(
                Path(path).with_suffix(".cloud-marker").read_text(), "accepted-once"
            )
            with db.get_conn() as conn:
                row = conn.execute("SELECT * FROM download_requests").fetchone()
                request_id = row["id"]
                self.assertEqual(
                    (row["status"], row["gy_status"]), ("submitting", "submitting")
                )
                self.assertEqual(row["gy_target_dir"], "staging-target")
            archive = backup.create_backup(paths, reason="share-interrupted")
            backup.verify_backup(archive)
            db.init_db()
            self.assertEqual(
                db.get_download_request(request_id)["status"], "manual_review"
            )
            db.update_download_request(
                request_id, status="cancelled", gy_status="cancelled"
            )
            backup.restore_backup(paths, archive)
            db.init_db()
            recovered = dict(db.get_download_request(request_id))
            self.assertEqual(recovered["status"], "manual_review")
            self.assertEqual(recovered["gy_target_dir"], "staging-target")
            self.assertFalse(db.claim_failed_share_transfer_request(request_id))
            self.assertFalse(
                db.finish_share_transfer_request(
                    request_id,
                    success=True,
                    target_dir_id="target",
                    target_dir_name="target",
                    title="late",
                )
            )
            client = FakeShareClient()
            store = ShareTransferPreviewStore()
            preview = store.create(client.inspected, "chat", "user")
            result = create_share_request(
                preview,
                ["file-1"],
                "target",
                "chat",
                user_id="user",
                store=store,
                client=client,
            )
            self.assertFalse(result["success"])
            self.assertEqual(result["status"], "manual_review")
            self.assertEqual(client.restore_calls, [])
            db.init_db()
            self.assertEqual(dict(db.get_download_request(request_id)), recovered)
