"""深审：重命名计划确认、替换预览和清理的跨进程状态边界。"""
from __future__ import annotations

import multiprocessing
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from app.modules import guangya_rename as plans
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
