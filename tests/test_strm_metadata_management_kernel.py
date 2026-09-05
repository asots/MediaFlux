"""积压取消经过真实 ToolPipeline/EffectPlan，不能模型直接执行写入。"""

from unittest.mock import patch

from app import database as db
from app.agent.errors import AgentToolError
from tests.agent_kernel_test_harness import KernelDomainTestHarness
from tests.support import IsolatedDatabaseTestCase
from tests.test_strm_metadata_queue import _job


class MetadataKernelTests(IsolatedDatabaseTestCase):
    def setUp(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM strm_metadata_queue")
        self.kernel = KernelDomainTestHarness()

    def test_queue_cancel_prepares_and_rechecks_after_explicit_confirmation(self):
        db.enqueue_strm_metadata_jobs([_job()])
        with (
            patch(
                "app.modules.web_secret.get_web_secret",
                return_value="metadata-kernel-secret",
            ),
            patch("app.config.get_bool", return_value=False),
        ):
            plan = self.kernel.prepare(
                "strm.metadata.cancel_pending", {}, owner="metadata-owner"
            )
            self.assertEqual(db.count_strm_metadata_jobs()["queued"], 1)
            plan_id = plan["action_plan"]["plan_id"]
            with self.assertRaises(AgentToolError):
                self.kernel.confirm(plan_id, owner="different-owner")
            db.enqueue_strm_metadata_jobs([_job(file_id="after-preview")])
            result = self.kernel.confirm(plan_id, owner="metadata-owner")
            self.assertTrue(result["result"]["ok"])
            self.assertEqual(result["result"]["data"]["cancelled"], 1)
            self.assertFalse(result["result"]["data"]["enabled"])
            self.assertEqual(db.count_strm_metadata_jobs()["queued"], 1)
            with self.assertRaises(AgentToolError):
                self.kernel.confirm(plan_id, owner="metadata-owner")

    def test_changed_snapshot_fails_closed_in_confirmation_pipeline(self):
        db.enqueue_strm_metadata_jobs([_job()])
        with (
            patch(
                "app.modules.web_secret.get_web_secret",
                return_value="metadata-kernel-secret",
            ),
            patch("app.config.get_bool", return_value=False),
        ):
            plan = self.kernel.prepare(
                "strm.metadata.cancel_pending", {}, owner="metadata-owner"
            )
            db.enqueue_strm_metadata_jobs([_job(etag="changed")])
            with self.assertRaises(AgentToolError):
                self.kernel.confirm(
                    plan["action_plan"]["plan_id"], owner="metadata-owner"
                )
            self.assertEqual(db.count_strm_metadata_jobs()["queued"], 1)
