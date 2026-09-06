"""下载跟踪批次隔离：坏数据不阻断其他请求，持久游标跨重建后仍可重试。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from app import database as db
from app.modules.download_tracker import _TRACKER_CURSOR_KEY, DownloadTracker
from tests.support import IsolatedDatabaseTestCase


class DownloadTrackerBatchIsolationTests(IsolatedDatabaseTestCase):
    def setUp(self) -> None:
        with db.get_conn() as conn:
            conn.execute("DELETE FROM download_request_keys")
            conn.execute("DELETE FROM download_requests")
            conn.execute("DELETE FROM download_log")
        db.kv_set(_TRACKER_CURSOR_KEY, "0")
        for name in ("_run_staging_reconciliation_if_due", "_run_torrent_data_cleanup_if_due"):
            mock = patch.object(DownloadTracker, name, return_value=0)
            mock.start()
            self.addCleanup(mock.stop)
        for name in ("_notify_completion", "_start_local_import", "_publish_lifecycle"):
            mock = patch.object(DownloadTracker, name)
            mock.start()
            self.addCleanup(mock.stop)

    def _request(self, title: str) -> int:
        request_id, _ = db.create_download_request(title, "magnet", title=title)
        db.update_download_request(
            request_id, status="submitted", qb_status="submitted", qb_task_id=title,
        )
        return request_id

    @staticmethod
    def _task(title: str, progress=0.5):
        return SimpleNamespace(hash=title, name=title, progress=progress, state="downloading")

    def test_malformed_task_does_not_starve_batch_or_lose_retry_after_restart(self) -> None:
        bad = self._request("bad")
        good = self._request("good")
        with patch.object(
            DownloadTracker, "_qb_tasks", return_value=(True, [self._task("bad", "invalid"), self._task("good")]),
        ):
            self.assertEqual(DownloadTracker().run_once(), 2)
        self.assertEqual(db.get_download_request(bad)["qb_status"], "submitted")
        self.assertEqual(db.get_download_request(good)["qb_status"], "downloading")
        self.assertEqual(db.kv_get(_TRACKER_CURSOR_KEY), str(good))

        # 模拟重启：新 tracker 使用同一持久游标；坏记录恢复后会绕回处理。
        with patch.object(
            DownloadTracker, "_qb_tasks", return_value=(True, [self._task("bad"), self._task("good")]),
        ):
            self.assertEqual(DownloadTracker().run_once(), 2)
            self.assertEqual(DownloadTracker().run_once(), 2)
        self.assertEqual(db.get_download_request(bad)["qb_status"], "downloading")
        self.assertEqual(db.get_download_request(good)["qb_status"], "downloading")

    def test_one_database_failure_does_not_skip_other_requests(self) -> None:
        first = self._request("first")
        second = self._request("second")
        actual = DownloadTracker._update_request
        seen = []

        def process(tracker, row, *args, **kwargs):
            seen.append(row["id"])
            if row["id"] == first:
                raise RuntimeError("isolated state transition failure")
            return actual(tracker, row, *args, **kwargs)

        with (
            patch.object(DownloadTracker, "_update_request", process),
            patch.object(DownloadTracker, "_qb_tasks", return_value=(True, [self._task("first"), self._task("second")])),
        ):
            self.assertEqual(DownloadTracker().run_once(), 2)
        self.assertEqual(seen, [first, second])
        self.assertEqual(db.get_download_request(second)["qb_status"], "downloading")
        self.assertEqual(db.kv_get(_TRACKER_CURSOR_KEY), str(second))

    def test_process_interrupt_is_not_swallowed_or_marked_as_finished_batch(self) -> None:
        self._request("interrupted")
        with (
            patch.object(DownloadTracker, "_qb_tasks", return_value=(True, [])),
            patch.object(DownloadTracker, "_update_request", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            DownloadTracker().run_once()
        self.assertEqual(db.kv_get(_TRACKER_CURSOR_KEY), "0")
