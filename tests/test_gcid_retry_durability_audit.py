"""GCID 重试必须使用持久凭据与任务 CAS；未知在途写入不得重放。"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app import database as db
from app.modules import gcid_import
from tests.support import IsolatedDatabaseTestCase
from tests.test_gcid_import_api import FakeImporter, FakeOutcome


def _retry_in_process(database_path, task_id, token, marker, entered, release, results):
    db.configure_database(database_path, test_mode=True)

    class Importer:
        available = True
        unavailable_reason = ""

        def import_file(self, **kwargs):
            with open(marker, "a", encoding="utf-8") as stream:
                stream.write(token + "\n")
            return FakeOutcome(True, remote_file_id="synthetic-remote")

    original_execute = gcid_import._execute_items

    def execute_after_gate(*args, **kwargs):
        entered.set()
        if not release.wait(15):
            raise AssertionError("process gate timed out")
        return original_execute(*args, **kwargs)

    try:
        with (
            patch.object(gcid_import, "_execute_items", side_effect=execute_after_gate),
            patch.object(gcid_import, "get_private_importer", return_value=Importer()),
            patch.object(gcid_import.notifier, "notify_gcid_import_started"),
            patch.object(gcid_import.notifier, "notify_gcid_import_finished"),
        ):
            task, replayed = gcid_import.retry_task(task_id, operation_token=token)
            results.put((token, task["status"], replayed))
    except BaseException as exc:
        results.put((token, "error", type(exc).__name__))
        raise


class GCIDRetryDurabilityAuditTests(IsolatedDatabaseTestCase):
    def setUp(self):
        gcid_import.reset_runtime_state()
        self.addCleanup(patch.stopall)
        patch.object(gcid_import.notifier, "notify_gcid_import_started").start()
        patch.object(gcid_import.notifier, "notify_gcid_import_finished").start()

    def _task(self, suffix=""):
        task_id = db.create_gcid_import_task(
            operation_token=self._testMethodName + suffix,
            manifest_digest="a" * 64,
            target_dir_id="target",
            file_count=1,
            total_size=1,
        )
        db.replace_gcid_import_items(
            task_id,
            [{"path": "Film.mkv", "size": 1, "gcid": "gcid-film", "status": "failed"}],
        )
        gcid_import._finish_task(task_id)
        return task_id

    def test_retry_token_survives_runtime_reset_even_when_retry_failed(self):
        task_id = self._task()
        fake = FakeImporter({"Film.mkv": [FakeOutcome(False), FakeOutcome(False)]})
        with patch.object(gcid_import, "get_private_importer", return_value=fake):
            first, replayed = gcid_import.retry_task(
                task_id, operation_token="retry-once"
            )
            self.assertFalse(replayed)
            self.assertEqual(first["status"], "failed")
            gcid_import.reset_runtime_state()
            again, replayed = gcid_import.retry_task(
                task_id, operation_token="retry-once"
            )
        self.assertTrue(replayed)
        self.assertEqual(again["status"], "failed")
        self.assertEqual(len(fake.calls), 1)

    def test_running_task_is_not_retried_or_advertised_as_retryable(self):
        task_id = self._task()
        db.update_gcid_import_task(task_id, status="running")
        fake = FakeImporter()
        with patch.object(gcid_import, "get_private_importer", return_value=fake):
            current, replayed = gcid_import.retry_task(
                task_id, operation_token="while-running"
            )
        self.assertEqual(fake.calls, [])
        self.assertTrue(replayed)
        self.assertEqual(current["status"], "running")
        self.assertFalse(current["can_retry"])
        self.assertEqual(db.get_gcid_import_task(task_id)["status"], "running")

    def test_replayed_reply_rereads_after_another_process_claims(self):
        task_id = self._task()

        def claimed_between_reads(selected_id, token):
            self.assertTrue(
                gcid_import._claim_task_for_run(selected_id, retry_token=token)
            )
            return True

        with patch.object(
            gcid_import, "_retry_was_recorded", side_effect=claimed_between_reads
        ):
            current, replayed = gcid_import.retry_task(
                task_id, operation_token="interleaved"
            )
        self.assertTrue(replayed)
        self.assertEqual(current["status"], "running")
        self.assertFalse(current["can_retry"])

    def test_new_token_can_retry_and_identical_tokens_on_other_tasks_are_independent(
        self,
    ):
        first = self._task("-a")
        second = self._task("-b")
        fake = FakeImporter(
            {"Film.mkv": [FakeOutcome(False), FakeOutcome(True), FakeOutcome(True)]}
        )
        before = db.list_gcid_import_items(first)[0]["id"]
        with patch.object(gcid_import, "get_private_importer", return_value=fake):
            gcid_import.retry_task(first, operation_token="old")
            done, replayed = gcid_import.retry_task(first, operation_token="new")
            other, other_replayed = gcid_import.retry_task(
                second, operation_token="new"
            )
        self.assertEqual((done["status"], other["status"]), ("success", "success"))
        self.assertFalse(replayed or other_replayed)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(db.list_gcid_import_items(first)[0]["id"], before)

    def test_persistent_replay_lru_keeps_recent_hits_and_stays_bounded(self):
        task_id = self._task()
        fake = FakeImporter({"Film.mkv": [FakeOutcome(False), FakeOutcome(True)]})
        with patch.object(gcid_import, "get_private_importer", return_value=fake):
            gcid_import.retry_task(task_id, operation_token="keep-recent")
            ledger = json.loads(db.kv_get(gcid_import._RETRY_REPLAY_KEY))
            self.assertIn([task_id, "keep-recent"], ledger)
            ledger = [
                [task_id, "keep-recent"],
                *[[10000 + i, f"historical-{i}"] for i in range(511)],
            ]
            db.kv_set(gcid_import._RETRY_REPLAY_KEY, json.dumps(ledger))
            _old, replayed = gcid_import.retry_task(
                task_id, operation_token="keep-recent"
            )
            self.assertTrue(replayed)
            gcid_import.reset_runtime_state()
            final, replayed = gcid_import.retry_task(
                task_id, operation_token="new-explicit"
            )
        self.assertFalse(replayed)
        self.assertEqual(final["status"], "success")
        stored = json.loads(db.kv_get(gcid_import._RETRY_REPLAY_KEY))
        self.assertEqual(len(stored), 512)
        self.assertIn([task_id, "keep-recent"], stored)
        self.assertIn([task_id, "new-explicit"], stored)
        self.assertNotIn([10000, "historical-0"], stored)
        self.assertEqual(len(fake.calls), 2)

    def test_two_real_processes_cannot_execute_the_same_failed_item(self):
        task_id = self._task()
        context = multiprocessing.get_context("spawn")
        entered = context.Event()
        release = context.Event()
        results = context.Queue()
        with TemporaryDirectory() as directory:
            marker = str(Path(directory) / "calls.txt")
            workers = [
                context.Process(
                    target=_retry_in_process,
                    args=(
                        str(self.test_db_path),
                        task_id,
                        token,
                        marker,
                        entered,
                        release,
                        results,
                    ),
                )
                for token in ("process-a", "process-b")
            ]
            try:
                workers[0].start()
                self.assertTrue(entered.wait(10))
                workers[1].start()
                # 新消费者必须能立即读到“已有执行者”，而不是进入第二次私有写入。
                try:
                    early = results.get(timeout=3)
                except Exception:
                    early = None
            finally:
                release.set()
                for worker in workers:
                    if worker.pid is not None:
                        worker.join(10)
                        if worker.is_alive():
                            worker.terminate()
                            worker.join(5)
            self.assertEqual([worker.exitcode for worker in workers], [0, 0])
            calls = Path(marker).read_text().splitlines()
            self.assertEqual(calls, ["process-a"])
            self.assertEqual(early, ("process-b", "running", True))
            self.assertEqual(db.get_gcid_import_task(task_id)["status"], "success")
        results.close()
        results.join_thread()
