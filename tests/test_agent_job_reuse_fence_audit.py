"""确认复用的是具体旧任务，不能在写事务等待后悄悄创建或接管另一任务。"""

from __future__ import annotations

import json
import socket
import threading
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

from app import database as db
from app.agent.confirmation import SQLiteConfirmationStore
from app.agent.errors import AgentToolError
from app.agent.durable_job_actions import (
    prepare_start_episode_audit,
    start_episode_audit_confirmed,
)
from app.agent.library_patrol_progress import empty_patrol_projection
from app.agent.models import ToolContext
from tests.support import isolated_test_database

OWNER = "r7-temporary-owner"
ARGUMENTS = {"as_of": "2026-01-01", "max_series": 2}


class AgentJobReuseFenceAuditTests(unittest.TestCase):
    def test_confirmed_reuse_cannot_create_second_job_after_original_commits(self):
        with (
            patch.object(
                socket.socket,
                "connect",
                side_effect=AssertionError("network forbidden"),
            ),
            patch.object(
                socket, "getaddrinfo", side_effect=AssertionError("DNS forbidden")
            ),
            isolated_test_database(),
            patch(
                "app.agent.durable_job_actions.get_agent_jobs_scheduler",
                return_value=Mock(),
            ),
        ):
            projection = json.dumps(empty_patrol_projection(as_of=ARGUMENTS["as_of"]))
            first, created = db.create_agent_job(
                owner=OWNER,
                job_type="library_episode_audit",
                dedupe_key="2026-01-01:2",
                input_json=json.dumps(ARGUMENTS),
                checkpoint_json=json.dumps(
                    {"as_of": ARGUMENTS["as_of"], "cursor": "", "stall_attempts": 0}
                ),
                projection_json=projection,
            )
            self.assertTrue(created)
            first_id = str(first["job_id"])
            running = db.claim_due_agent_job(job_type="library_episode_audit")
            self.assertEqual(running["job_id"], first_id)
            context = ToolContext(owner=OWNER)
            preview, fingerprint = prepare_start_episode_audit(ARGUMENTS, context)
            self.assertTrue(preview.data["reused"])
            self.assertIn("确认后不会创建新任务", " ".join(preview.data["effects"]))

            # 模拟真实入口已经消费有效、owner 匹配的一次性确认票据。
            store = SQLiteConfirmationStore()
            ticket = store.issue(
                owner=OWNER,
                tool_name="library.start_episode_audit",
                arguments=ARGUMENTS,
                context_fingerprint=fingerprint,
            )
            claimed_ticket = store.claim_and_rotate_owner(
                owner=OWNER, confirmation_id=ticket.confirmation_id
            )
            self.assertEqual(store.list_active_tickets(owner=OWNER), [])

            completion_uncommitted = threading.Event()
            allow_completion_commit = threading.Event()
            confirmation_checked = threading.Event()
            outcomes = {}
            failures = []
            real_get_conn = db.get_conn
            real_create = db.create_agent_job

            @contextmanager
            def gated_connection():
                with real_get_conn() as conn:
                    yield conn
                    if threading.current_thread().name == "r7-existing-job-completion":
                        # 正式 complete_agent_job 已执行 UPDATE，但其 SQLite 提交尚未完成。
                        completion_uncommitted.set()
                        if not allow_completion_commit.wait(3):
                            raise AssertionError("completion gate timed out")

            def observed_create(**kwargs):
                # 此时正式 confirmed handler 已通过 _start_context 指纹复核；
                # 随后的真实 BEGIN IMMEDIATE 必须等待旧 job 的成功事务提交。
                confirmation_checked.set()
                return real_create(**kwargs)

            def complete_existing():
                try:
                    outcomes["completed"] = db.complete_agent_job(
                        first_id,
                        expected_lease_generation=int(running["lease_generation"]),
                        projection_json=projection,
                        progress_current=0,
                        progress_total=0,
                        summary="原有全库巡检正常完成",
                    )
                except BaseException as exc:
                    failures.append(exc)

            def confirm_reuse():
                try:
                    outcomes["confirmed"] = start_episode_audit_confirmed(
                        claimed_ticket.arguments,
                        claimed_ticket.context_fingerprint,
                        context,
                    )
                except BaseException as exc:
                    failures.append(exc)

            completer = threading.Thread(
                target=complete_existing, name="r7-existing-job-completion"
            )
            confirmer = threading.Thread(
                target=confirm_reuse, name="r7-confirm-existing-job"
            )
            with (
                patch.object(db, "get_conn", gated_connection),
                patch.object(db, "create_agent_job", observed_create),
            ):
                try:
                    completer.start()
                    self.assertTrue(completion_uncommitted.wait(2))
                    confirmer.start()
                    self.assertTrue(
                        confirmation_checked.wait(2), "WAL 读取没有到达确认后的创建入口"
                    )
                finally:
                    allow_completion_commit.set()
                    completer.join(4)
                    if confirmer.ident is not None:
                        confirmer.join(4)
            self.assertFalse(completer.is_alive() or confirmer.is_alive())
            self.assertTrue(outcomes["completed"])
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], AgentToolError)
            self.assertEqual(failures[0].code, "confirmation_stale")
            self.assertNotIn("confirmed", outcomes)
            self.assertEqual(
                db.get_agent_job(owner=OWNER, job_id=first_id)["status"], "succeeded"
            )
            self.assertEqual(len(db.list_agent_jobs(owner=OWNER)), 1)
            self.assertEqual(store.list_active_tickets(owner=OWNER), [])

    def test_current_reuse_and_other_owner_keep_single_owner_scoped_job(self):
        from app.agent import durable_job_actions as actions

        with (
            isolated_test_database(),
            patch.object(actions, "get_agent_jobs_scheduler"),
        ):
            args = dict(ARGUMENTS)
            context = ToolContext(owner=OWNER)
            _, fingerprint = prepare_start_episode_audit(args, context)
            first = start_episode_audit_confirmed(args, fingerprint, context)
            self.assertTrue(first.data["created"])
            _, reuse_fingerprint = prepare_start_episode_audit(args, context)
            reused = start_episode_audit_confirmed(args, reuse_fingerprint, context)
            self.assertFalse(reused.data["created"])
            self.assertEqual(reused.data["job_id"], first.data["job_id"])
            other = ToolContext(owner="other-owner")
            _, other_fingerprint = prepare_start_episode_audit(args, other)
            second = start_episode_audit_confirmed(args, other_fingerprint, other)
            self.assertTrue(second.data["created"])
            self.assertNotEqual(second.data["job_id"], first.data["job_id"])
            self.assertEqual(len(db.list_agent_jobs(owner=OWNER)), 1)

    def test_cancelled_reuse_cannot_attach_to_new_active_replacement(self):
        from app.agent import durable_job_actions as actions

        with (
            isolated_test_database(),
            patch.object(actions, "get_agent_jobs_scheduler"),
        ):
            args = dict(ARGUMENTS)
            context = ToolContext(owner=OWNER)
            _, fingerprint = prepare_start_episode_audit(args, context)
            first = start_episode_audit_confirmed(args, fingerprint, context)
            _, reuse_fingerprint = prepare_start_episode_audit(args, context)
            real_create = db.create_agent_job

            def replace_before_transaction(**kwargs):
                with db.get_conn() as conn:
                    conn.execute(
                        "UPDATE agent_jobs SET status='cancelled' WHERE job_id=?",
                        (first.data["job_id"],),
                    )
                replacement_kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key != "expected_active_job_id"
                }
                replacement, created = real_create(**replacement_kwargs)
                self.assertTrue(created)
                self.assertNotEqual(replacement["job_id"], first.data["job_id"])
                return real_create(**kwargs)

            with patch.object(
                db, "create_agent_job", side_effect=replace_before_transaction
            ):
                with self.assertRaises(AgentToolError) as raised:
                    start_episode_audit_confirmed(args, reuse_fingerprint, context)
            self.assertEqual(raised.exception.code, "confirmation_stale")
            self.assertEqual(len(db.list_agent_jobs(owner=OWNER)), 2)
