"""深审：重命名计划确认、替换预览和清理的跨进程状态边界。"""
from __future__ import annotations

import json
import multiprocessing
import tempfile
import time
import threading
import unittest
from pathlib import Path
from unittest import mock

from app import database as db
from app.modules import guangya_rename as plans
from app.repositories.organize_operation_jobs import (
    claim_organize_operation_job,
    count_pending_organize_operation_jobs,
    enqueue_organize_operation_job,
    finish_organize_operation_job,
    organize_operation_owner_digest,
    recover_orphaned_organize_operation_jobs,
)
from tests.support import IsolatedDatabaseTestCase
from tests.test_agent_guangya_rename import FakeGuangYaClient


SECRET = "deep-rename-lifecycle-synthetic-secret"
OWNER = "deep-rename-lifecycle-owner"


def confirm_paused_after_read(directory, plan_id, fingerprint, entered, proceed):
    # spawn 入口只读测试目录，与主进程相同的计划状态锁；不打开SDK/数据库。
    with mock.patch.object(plans, "_plan_directory", return_value=Path(directory)), \
            mock.patch.object(plans, "get_web_secret", return_value=SECRET):
        original = plans.load_rename_plan

        def paused_read(*args, **kwargs):
            payload = original(*args, **kwargs)
            entered.set()
            if not proceed.wait(10):
                raise RuntimeError("测试未释放确认窗口")
            return payload

        with mock.patch.object(plans, "load_rename_plan", side_effect=paused_read):
            plans.confirm_rename_plan(plan_id, owner=OWNER, expected_fingerprint=fingerprint)


class RenamePlanLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(plans, "_plan_directory", return_value=self.directory))
        self.enterContext(mock.patch.object(plans, "get_web_secret", return_value=SECRET))
        self.plan = plans.build_rename_plan(
            FakeGuangYaClient(), owner=OWNER, targets=["/整理/动漫"], mode="remove_bitrate",
        )

    def test_unconfirmed_preview_is_still_removable(self):
        self.assertTrue(plans.discard_rename_plan(self.plan["plan_id"], preview_only=True))
        self.assertFalse(plans._plan_path(self.plan["plan_id"]).exists())
        self.assertFalse(plans.discard_rename_plan(self.plan["plan_id"], preview_only=True))

    def test_completed_history_cannot_be_reconfirmed_or_deleted_by_preview_replacement(self):
        plan_id = self.plan["plan_id"]
        plans.confirm_rename_plan(plan_id, owner=OWNER, expected_fingerprint=self.plan["fingerprint"])
        plans.update_rename_plan_execution(plan_id, status="completed", execution={"renamed": 2})
        with self.assertRaises(plans.GuangYaRenamePlanStale):
            plans.confirm_rename_plan(plan_id, owner=OWNER, expected_fingerprint=self.plan["fingerprint"])
        self.assertFalse(plans.discard_rename_plan(plan_id, preview_only=True))
        plans.maintain_rename_plans()
        self.assertEqual(plans._read_plan(plan_id)["execution"], {"renamed": 2})

    def test_preview_deletion_waits_for_cross_process_confirmation_and_rechecks_state(self):
        context = multiprocessing.get_context("spawn")
        entered, proceed = context.Event(), context.Event()
        child = context.Process(
            target=confirm_paused_after_read,
            args=(str(self.directory), self.plan["plan_id"], self.plan["fingerprint"], entered, proceed),
        )
        started, deleted = threading.Event(), threading.Event()
        result, failures = [], []

        def discard():
            started.set()
            try:
                result.append(plans.discard_rename_plan(self.plan["plan_id"], preview_only=True))
            except BaseException as exc:
                failures.append(exc)
            finally:
                deleted.set()

        worker = threading.Thread(target=discard, daemon=True)
        child.start()
        try:
            self.assertTrue(entered.wait(10))
            worker.start()
            self.assertTrue(started.wait(2))
            self.assertFalse(deleted.wait(0.1), "确认持锁期间不能按旧preview状态删除")
            proceed.set()
            child.join(10)
            worker.join(10)
            self.assertFalse(child.is_alive())
            self.assertFalse(worker.is_alive())
            self.assertEqual(child.exitcode, 0)
            self.assertFalse(failures)
            self.assertEqual(result, [False])
            confirmed = plans.load_rename_plan(self.plan["plan_id"], require_confirmed=True)
            self.assertEqual(confirmed["status"], "confirmed")
            self.assertEqual(len(confirmed["entries"]), 2)
        finally:
            proceed.set()
            child.join(10)
            if child.is_alive():
                child.terminate()
                child.join(5)
            if worker.ident is not None:
                worker.join(5)


class RenameQueueRecoveryPlanProjectionTests(IsolatedDatabaseTestCase):
    """旧 rename 队列终态必须收束同一份已签名 plan。"""

    def setUp(self):
        super().setUp()
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.plan_patch = mock.patch.object(
            plans, "_plan_directory", return_value=self.directory
        )
        self.secret_patch = mock.patch.object(
            plans, "get_web_secret", return_value=SECRET
        )
        self.plan_patch.start()
        self.secret_patch.start()
        self.addCleanup(self.plan_patch.stop)
        self.addCleanup(self.secret_patch.stop)
        with db.get_conn() as conn:
            conn.execute("DELETE FROM organize_operation_jobs")

    @staticmethod
    def _plan(plan_id: str, owner: str, status: str) -> dict:
        return {
            "version": 1,
            "plan_id": plan_id,
            "owner_digest": organize_operation_owner_digest(owner),
            "created_at": "2026-09-19T00:00:00+08:00",
            "created_at_epoch": time.time(),
            "expires_at_epoch": time.time() + 900,
            "confirmed_at": "2026-09-19T00:00:01+08:00",
            "confirmed_at_epoch": time.time(),
            "execute_until_epoch": time.time() + 3600,
            "status": status,
            "credential_generation": 7,
            "mode": "replace_text",
            "recursive": True,
            "targets": ["/library"],
            "limit": 1,
            "summary": "test",
            "stats": {"rename_count": 1},
            "extension_counts": {"mkv": 1},
            "entries": [],
            "conflicts": [],
            "samples": [],
            "transform": {},
            "rollback": {"available": True, "basis": "file_id,parent_id"},
            "fingerprint": "f" * 64,
            "execution": {},
        }

    def _write_plan(self, plan_id: str, owner: str, status: str) -> dict:
        payload = self._plan(plan_id, owner, status)
        plans._atomic_write_plan(plans._plan_path(plan_id), payload)
        return payload

    def _enqueue(self, plan: dict, owner: str):
        return enqueue_organize_operation_job(
            job_kind="agent_guangya_rename",
            owner=owner,
            operation="旧批量名称转换",
            reference="test-reference",
            payload={
                "version": 1,
                "plan_id": plan["plan_id"],
                "plan_fingerprint": plan["fingerprint"],
                "owner_digest": plan["owner_digest"],
                "credential_generation": plan["credential_generation"],
            },
            dedupe_key=f"rename-recovery:{plan['plan_id']}",
        )[0]

    def test_recovery_cancel_expiry_tamper_and_terminal_plan_projection(self):
        running_owner = "rename-recovery-running"
        running = self._write_plan("1" * 32, running_owner, "running")
        running_row = self._enqueue(running, running_owner)
        self.assertIsNotNone(claim_organize_operation_job(str(running_row["job_id"])))

        cancel_owner = "rename-recovery-cancel"
        cancel = self._write_plan("2" * 32, cancel_owner, "confirmed")
        cancel_row = self._enqueue(cancel, cancel_owner)
        self.assertIsNotNone(claim_organize_operation_job(str(cancel_row["job_id"])))
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE organize_operation_jobs SET cancel_requested=1 WHERE job_id=?",
                (str(cancel_row["job_id"]),),
            )

        expired_owner = "rename-recovery-expired"
        expired = self._write_plan("3" * 32, expired_owner, "confirmed")
        expired_row = self._enqueue(expired, expired_owner)
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE organize_operation_jobs SET expires_at=1 WHERE job_id=?",
                (str(expired_row["job_id"]),),
            )

        tampered_owner = "rename-recovery-tampered"
        tampered = self._write_plan("4" * 32, tampered_owner, "running")
        tampered_row = self._enqueue(tampered, tampered_owner)
        self.assertIsNotNone(claim_organize_operation_job(str(tampered_row["job_id"])))
        tampered_path = plans._plan_path(tampered["plan_id"])
        tampered_json = json.loads(tampered_path.read_text(encoding="utf-8"))
        tampered_json["status"] = "completed"
        tampered_path.write_text(
            json.dumps(tampered_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        queue_tampered_owner = "rename-recovery-queue-tampered"
        queue_tampered = self._write_plan("7" * 32, queue_tampered_owner, "running")
        queue_tampered_row = self._enqueue(queue_tampered, queue_tampered_owner)
        self.assertIsNotNone(
            claim_organize_operation_job(str(queue_tampered_row["job_id"]))
        )
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE organize_operation_jobs SET payload_json=? WHERE job_id=?",
                (
                    json.dumps(
                        {
                            "version": 1,
                            "plan_id": queue_tampered["plan_id"],
                            "plan_fingerprint": "0" * 64,
                            "owner_digest": queue_tampered["owner_digest"],
                            "credential_generation": 7,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    str(queue_tampered_row["job_id"]),
                ),
            )

        completed_owner = "rename-recovery-completed"
        completed = self._write_plan("5" * 32, completed_owner, "completed")
        completed_row = self._enqueue(completed, completed_owner)
        self.assertIsNotNone(claim_organize_operation_job(str(completed_row["job_id"])))

        manual_owner = "rename-recovery-manual"
        manual = self._write_plan("6" * 32, manual_owner, "manual_review")
        manual_row = self._enqueue(manual, manual_owner)
        self.assertIsNotNone(claim_organize_operation_job(str(manual_row["job_id"])))

        self.assertEqual(recover_orphaned_organize_operation_jobs(), 6)
        count_pending_organize_operation_jobs()

        self.assertEqual(plans._read_plan(running["plan_id"])["status"], "manual_review")
        self.assertEqual(plans._read_plan(cancel["plan_id"])["status"], "cancelled")
        self.assertEqual(plans._read_plan(expired["plan_id"])["status"], "cancelled")
        self.assertEqual(
            json.loads(tampered_path.read_text(encoding="utf-8"))["status"],
            "completed",
        )
        self.assertEqual(
            plans._read_plan(queue_tampered["plan_id"])["status"], "running"
        )
        self.assertEqual(plans._read_plan(completed["plan_id"])["status"], "completed")
        self.assertEqual(plans._read_plan(manual["plan_id"])["status"], "manual_review")

    def test_finish_projects_trusted_terminal_without_rewriting_completed_plan(self):
        owner = "rename-recovery-finish"
        plan = self._write_plan("8" * 32, owner, "confirmed")
        row = self._enqueue(plan, owner)
        claimed = claim_organize_operation_job(str(row["job_id"]))
        self.assertIsNotNone(claimed)
        self.assertTrue(
            finish_organize_operation_job(
                str(row["job_id"]),
                expected_lease_generation=int(claimed["lease_generation"]),
                status="completed",
                result={"stats": {"renamed": 1}},
            )
        )
        self.assertEqual(plans._read_plan(plan["plan_id"])["status"], "completed")
        self.assertEqual(
            plans._read_plan(plan["plan_id"])["execution"]["queue_status"],
            "completed",
        )
