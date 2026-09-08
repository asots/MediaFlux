"""R03 失败归属围栏：隔离 DB、内存云盘及临时 STRM，不调用外部服务。"""

from __future__ import annotations

# isort: off
import tests  # noqa: F401  # 必须先建立测试隔离，再导入应用。

# isort: on
import unittest
from unittest.mock import patch

from app import database as db
from app.clients.guangya import GuangYaFile
from app.modules.strm_metadata_worker import STRMMetadataWorker
from app.repositories.strm_request_ownership import INTERRUPTED_ERROR
from tests import test_strm_request_ownership_recovery as recovery_tests


class STRMFailureLeaseScopeTests(unittest.TestCase):
    def setUp(self):
        # 复用业务夹具但不继承 TestCase，避免重复收集已有 16 个用例。
        self.fixture = recovery_tests.STRMRequestOwnershipRecoveryTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.enterContext(
            patch(
                "app.modules.scheduler.threading.Thread",
                recovery_tests._ParkedThread,
            )
        )

    def execute_recovered(self):
        scheduler = self.fixture.scheduler()
        return self.fixture.execute(
            scheduler, self.fixture.recovered_options(scheduler)
        )

    def queue_two_targets(self):
        request, admission, _ = self.fixture.seed()
        self.fixture.cloud.tree["source"] = []
        changes = []
        for episode, dirname in enumerate(("A", "B"), start=1):
            parent = "dir-" + dirname
            self.fixture.cloud.tree["source"].append(
                GuangYaFile(parent, dirname, True, 0, "", "source")
            )
            self.fixture.cloud.tree[parent] = [
                GuangYaFile(
                    f"v{episode}",
                    f"Show.S01E{episode:02d}.mkv",
                    False,
                    128,
                    f"e{episode}",
                    parent,
                )
            ]
            change = self.fixture.change(episode, dirname)
            change["parent_id"] = parent
            changes.append(change)
        scheduler = self.fixture.scheduler()
        self.fixture.queue(scheduler, [request], changes)
        return request, admission, changes

    def exhaust_first_target(self, *, include_second_in_last_failure=False):
        request, admission, changes = self.queue_two_targets()
        db.reschedule_strm_change_targets([changes[1]], not_before_seconds=3600)
        db.init_db()
        db.reconcile_startup_media_download_admissions()
        error = "target A permanent failure"
        for attempt in range(5):
            if attempt:
                db.reschedule_strm_change_targets([changes[0]], not_before_seconds=0)
            if include_second_in_last_failure and attempt == 4:
                db.reschedule_strm_change_targets([changes[1]], not_before_seconds=0)
            # scheduler() 装配真实生成入口后再注入本次失败。
            resumed = self.fixture.scheduler()
            with patch(
                "app.modules.scheduler.sync_strm_incremental",
                side_effect=OSError(error),
            ):
                result = self.fixture.execute(
                    resumed, self.fixture.recovered_options(resumed)
                )
            self.assertFalse(result["ok"], result)
        with db.get_conn() as conn:
            targets = {
                row["rel_dir"]: dict(row)
                for row in conn.execute("SELECT * FROM strm_change_queue")
            }
        self.assertEqual(targets["A"]["state"], "failed")
        self.assertEqual(targets["A"]["attempts"], 5)
        self.assertEqual(targets["B"]["state"], "queued")
        self.assertEqual(
            targets["B"]["lease_generation"],
            1 if include_second_in_last_failure else 0,
        )
        self.assertEqual(db.get_download_request(request)["strm_status"], "failed")
        self.assertEqual(self.fixture.admission(admission)["status"], "failed")
        return request, admission, changes[1], error

    def assert_second_target_does_not_clear_first_failure(self, *, also_failed):
        request, admission, second, error = self.exhaust_first_target(
            include_second_in_last_failure=also_failed
        )
        db.reschedule_strm_change_targets([second], not_before_seconds=0)
        result = self.execute_recovered()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["stats"]["generated"], 1)
        self.assertEqual(len(list(self.fixture.root.rglob("*.strm"))), 1)
        self.assertEqual(db.count_pending_strm_change_targets(), 0)
        row = db.get_download_request(request)
        self.assertEqual((row["strm_status"], row["strm_error"]), ("failed", error))
        self.assertEqual(self.fixture.admission(admission)["status"], "failed")

    def test_split_failed_targets_recover_after_each_actual_target_succeeds(self):
        """保留其它目标错误不能阻断合法分批重试；最后一个失败目标可正常恢复。"""
        request, admission, changes = self.queue_two_targets()
        resumed = self.fixture.scheduler()
        error = "shared initial failure"
        with patch(
            "app.modules.scheduler.sync_strm_incremental", side_effect=OSError(error)
        ):
            first = self.fixture.execute(
                resumed, self.fixture.recovered_options(resumed)
            )
        self.assertFalse(first["ok"], first)
        db.reschedule_strm_change_targets([changes[1]], not_before_seconds=3600)
        db.reschedule_strm_change_targets([changes[0]], not_before_seconds=0)
        first_retry = self.execute_recovered()
        self.assertTrue(first_retry["ok"], first_retry)
        self.assertEqual(first_retry["stats"]["generated"], 1)
        row = db.get_download_request(request)
        self.assertEqual((row["strm_status"], row["strm_error"]), ("failed", error))
        self.assertEqual(self.fixture.admission(admission)["status"], "failed")

        db.reschedule_strm_change_targets([changes[1]], not_before_seconds=0)
        last_retry = self.execute_recovered()
        self.assertTrue(last_retry["ok"], last_retry)
        self.assertEqual(last_retry["stats"]["generated"], 1)
        self.assertEqual(db.count_pending_strm_change_targets(), 0)
        self.assertEqual(len(list(self.fixture.root.rglob("*.strm"))), 2)
        row = db.get_download_request(request)
        self.assertEqual((row["strm_status"], row["strm_error"]), ("completed", ""))
        self.assertEqual(self.fixture.admission(admission)["status"], "processing")

    def test_unclaimed_target_cannot_clear_an_exhausted_target_failure(self):
        self.assert_second_target_does_not_clear_first_failure(also_failed=False)

    def test_retry_of_one_failed_target_keeps_other_exhausted_failure(self):
        # B 确实失败过也不能单独释放 A 的错误，不能只把 >=0 改成 >0。
        self.assert_second_target_does_not_clear_first_failure(also_failed=True)

    def test_same_text_independent_failure_revokes_cold_change_recovery(self):
        request, admission, _ = self.fixture.seed()
        scheduler = self.fixture.scheduler()
        self.fixture.queue(scheduler, [request], [self.fixture.change()])
        db.init_db()
        db.reconcile_startup_media_download_admissions()
        self.assertEqual(
            db.get_download_request(request)["strm_error"], INTERRUPTED_ERROR
        )
        db.update_download_request(
            request, strm_status="failed", strm_error=INTERRUPTED_ERROR
        )
        db.init_db()  # 已 failed 的同文本独立写入不能在再次启动时补发凭据。
        result = self.execute_recovered()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["stats"]["generated"], 1)
        self.assertEqual(db.count_pending_strm_change_targets(), 0)
        row = db.get_download_request(request)
        self.assertEqual(
            (row["strm_status"], row["strm_error"]), ("failed", INTERRUPTED_ERROR)
        )
        self.assertEqual(self.fixture.admission(admission)["status"], "failed")

    def test_legacy_queued_change_work_recovers_on_first_startup(self):
        request, admission, _ = self.fixture.seed()
        scheduler = self.fixture.scheduler()
        self.fixture.queue(scheduler, [request], [self.fixture.change()])
        self.assertEqual(db.get_download_request(request)["strm_status"], "queued")
        with db.get_conn() as conn:
            # 升级前 work 的真实默认形态，不通过新代码 seed 的 0 掩盖兼容性。
            conn.execute(
                "UPDATE strm_request_work SET failed_lease_generation=-1 WHERE request_id=?",
                (request,),
            )
        db.init_db()
        db.reconcile_startup_media_download_admissions()
        result = self.execute_recovered()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["stats"]["generated"], 1)
        self.assertEqual(db.count_pending_strm_change_targets(), 0)
        row = db.get_download_request(request)
        self.assertEqual((row["strm_status"], row["strm_error"]), ("completed", ""))
        self.assertEqual(self.fixture.admission(admission)["status"], "processing")

    def test_legacy_queued_refresh_work_recovers_on_first_startup(self):
        request, admission = self.cold_refresh(legacy_work=True)
        STRMMetadataWorker()._flush_media_refresh(force=True)
        self.assertEqual(db.count_strm_refresh_paths(), 0)
        row = db.get_download_request(request)
        self.assertEqual((row["strm_status"], row["strm_error"]), ("completed", ""))
        self.assertEqual(self.fixture.admission(admission)["status"], "processing")

    def test_startup_only_arms_matching_active_request_ownership(self):
        requests = []
        for episode, mismatch in enumerate(
            ("generation", "task", "cancelled", "failed", "resubmitted"), start=1
        ):
            request, _, _ = self.fixture.seed(episode)
            self.fixture.owned_queue(request, [self.fixture.change(episode)])
            with db.get_conn() as conn:
                conn.execute(
                    "UPDATE strm_request_work SET failed_lease_generation=-1 WHERE request_id=?",
                    (request,),
                )
                if mismatch == "generation":
                    conn.execute(
                        "UPDATE strm_request_work SET generation=generation-1 WHERE request_id=?",
                        (request,),
                    )
                elif mismatch == "task":
                    conn.execute(
                        "UPDATE strm_request_work SET organize_task_id='previous-task' WHERE request_id=?",
                        (request,),
                    )
            if mismatch not in {"generation", "task"}:
                db.update_download_request(request, status=mismatch)
            requests.append(request)
        db.init_db()
        with db.get_conn() as conn:
            for request in requests:
                proof = conn.execute(
                    "SELECT failed_lease_generation FROM strm_request_work WHERE request_id=?",
                    (request,),
                ).fetchone()[0]
                self.assertEqual(proof, -1, request)

    def cold_refresh(self, *, legacy_work=False):
        request, admission, _ = self.fixture.seed()
        owners = self.fixture.owned_queue(request, [])
        path = self.fixture.root / "cold.strm"
        path.write_text("http://media.invalid/v1\n", encoding="utf-8")
        db.enqueue_strm_refresh_paths([str(path)], request_owners=owners)
        if legacy_work:
            self.assertEqual(db.get_download_request(request)["strm_status"], "queued")
            with db.get_conn() as conn:
                conn.execute(
                    "UPDATE strm_request_work SET failed_lease_generation=-1 WHERE request_id=?",
                    (request,),
                )
        else:
            self.assertTrue(
                db.update_strm_request_state(owners[0], strm_status="running")
            )
        db.init_db()
        db.reconcile_startup_media_download_admissions()
        self.assertEqual(db.get_download_request(request)["strm_status"], "failed")
        self.assertEqual(
            db.get_download_request(request)["strm_error"], INTERRUPTED_ERROR
        )
        self.assertEqual(self.fixture.admission(admission)["status"], "failed")
        return request, admission

    def test_cold_outbox_ack_recovers_owned_interruption(self):
        request, admission = self.cold_refresh()
        STRMMetadataWorker()._flush_media_refresh(force=True)
        self.assertEqual(db.count_strm_refresh_paths(), 0)
        row = db.get_download_request(request)
        self.assertEqual((row["strm_status"], row["strm_error"]), ("completed", ""))
        self.assertEqual(self.fixture.admission(admission)["status"], "processing")

    def test_same_text_independent_failure_blocks_cold_outbox_ack(self):
        request, admission = self.cold_refresh()
        db.update_download_request(
            request, strm_status="failed", strm_error=INTERRUPTED_ERROR
        )
        db.init_db()  # 已 failed 的同文本独立写入不能在再次启动时补发凭据。
        STRMMetadataWorker()._flush_media_refresh(force=True)
        self.assertEqual(db.count_strm_refresh_paths(), 0)
        row = db.get_download_request(request)
        self.assertEqual(
            (row["strm_status"], row["strm_error"]), ("failed", INTERRUPTED_ERROR)
        )
        self.assertEqual(self.fixture.admission(admission)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
