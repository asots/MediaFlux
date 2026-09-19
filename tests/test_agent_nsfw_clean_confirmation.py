"""自动清洗只扩展光鸭人工清洗入口；冻结、竞争及开写边界回归。"""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import database as db
from app.clients.guangya import GuangYaFile
from app.modules import organize_confirmations as confirmations
from app.modules.directory_scrape_errors import DirectoryScrapeConflictError
from app.modules.organize import OrganizePlan, Organizer, OrganizeRules
from app.modules.organize_execution import execute_organize_plans
from app.modules.scraper import MatchResult
from tests.support import IsolatedDatabaseTestCase
from tests.test_agent_nsfw_clean_review import clean_payload


class CleanConfirmationTests(IsolatedDatabaseTestCase):
    def setUp(self):
        super().setUp()
        with db.get_conn() as conn:
            conn.execute("DELETE FROM organize_confirmations")
        self.rules = OrganizeRules(
            target_dir_id="archive",
            nsfw_enabled=True,
            nsfw_source_ids='["source"]',
            clean_empty=True,
        ).for_source("source")
        self.payload = clean_payload()
        self.payload["rules"] = confirmations.organize_rules_snapshot(self.rules)
        self.payload["_notification_suppressed"] = True
        self.candidate = self.payload["candidates"][0]
        self.enabled = True
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.object(
                confirmations,
                "_recognition_review_is_enabled",
                return_value=True,
            )
        )
        self.stack.enter_context(
            patch(
                "app.modules.nsfw_clean_review.nsfw_clean_review_enabled",
                side_effect=lambda: self.enabled,
            )
        )
        self.stack.enter_context(
            patch.object(
                confirmations.OrganizeRules,
                "from_config",
                return_value=self.rules,
            )
        )
        self.stack.enter_context(
            patch.object(confirmations, "wake_confirmation_dispatcher")
        )

    def ticket(self, *, running=False):
        token = "frozen-clean-test"
        db.create_organize_confirmation(
            token=token,
            fingerprint="clean-test",
            chat_id="100",
            source_name="source",
            directory_path="/待整理",
            payload=self.payload,
            expires_at=(
                datetime.now(timezone.utc).astimezone() + timedelta(hours=1)
            ).strftime("%Y-%m-%d %H:%M:%S"),
        )
        db.update_organize_confirmation(token, review_status="running")
        if running:
            db.claim_organize_confirmation(
                token, chat_id="100", selected_index=0, actor="agent"
            )
            self.assertIsNotNone(db.claim_queued_organize_confirmation(token))
        return token

    def pending_manual(self):
        with db.get_conn() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM organize_confirmations WHERE status='pending'"
                )
            ]

    def test_disabled_scope_cannot_claim_but_human_still_can(self):
        token = self.ticket()
        self.enabled = False
        with self.assertRaises(DirectoryScrapeConflictError):
            confirmations.start_confirmation(token, 0, chat_id="100", actor="agent")
        self.assertEqual(db.get_organize_confirmation(token)["status"], "pending")
        manager = SimpleNamespace(
            start_operation=lambda *_a, **_kw: {"ok": True, "task_id": "t"}
        )
        with patch(
            "app.modules.organize_tasks.get_organize_manager", return_value=manager
        ):
            confirmations.start_confirmation(token, 0, chat_id="100", actor="human")
        self.assertEqual(
            db.get_organize_confirmation(token)["confirmation_actor"], "human"
        )
        self.enabled = True
        with self.assertRaisesRegex(ValueError, "人工"):
            confirmations.start_confirmation(token, 0, chat_id="100", actor="agent")

    def test_numeric_prefix_candidate_keeps_agent_authorization_boundary(self):
        self.payload = clean_payload("300MIUM-1474.mp4")
        self.payload["rules"] = confirmations.organize_rules_snapshot(self.rules)
        self.payload["_notification_suppressed"] = True
        self.candidate = self.payload["candidates"][0]
        token = self.ticket()
        self.enabled = False

        with self.assertRaises(DirectoryScrapeConflictError):
            confirmations.start_confirmation(token, 0, chat_id="100", actor="agent")
        self.assertEqual(db.get_organize_confirmation(token)["status"], "pending")

    def test_disable_queued_review_hands_off_without_network_or_auto_retry(self):
        token = self.ticket(running=True)
        self.enabled = False
        with (
            patch.object(confirmations, "GuangYaClient") as client,
            self.assertRaises(DirectoryScrapeConflictError),
        ):
            confirmations._execute_guangya_confirmation(
                token,
                self.payload,
                self.candidate,
                selected_index=0,
                chat_id="100",
                actor="agent",
            )
        client.assert_not_called()
        self.assertEqual(db.get_organize_confirmation(token)["status"], "failed")
        manual = self.pending_manual()
        self.assertEqual(len(manual), 1)
        self.assertNotEqual(manual[0]["review_status"], "pending")
        self.assertNotEqual(manual[0]["token"], token)

    def test_stale_rules_or_file_cannot_create_an_unusable_retry_ticket(self):
        from dataclasses import replace

        with patch.object(
            confirmations.OrganizeRules,
            "from_config",
            return_value=replace(self.rules, target_dir_id="changed"),
        ):
            self.assertFalse(
                confirmations._clean_confirmation_retry_is_current(self.payload, None)
            )
        remote = GuangYaFile(
            "file-0", "ABC-123.changed.mp4", False, size=1024, parent_id="parent"
        )
        self.assertFalse(
            confirmations._clean_confirmation_retry_is_current(
                self.payload,
                SimpleNamespace(file_info=lambda _: remote),
            )
        )
        token = self.ticket(running=True)
        with (
            patch.object(
                confirmations,
                "GuangYaClient",
                return_value=SimpleNamespace(file_info=lambda _: remote),
            ),
            self.assertRaises(DirectoryScrapeConflictError),
        ):
            confirmations._execute_guangya_confirmation(
                token,
                self.payload,
                self.candidate,
                selected_index=0,
                chat_id="100",
                actor="agent",
            )
        self.assertEqual(db.get_organize_confirmation(token)["status"], "failed")
        self.assertEqual(self.pending_manual(), [])

    def run_worker(self, *, decision="new", revoke=False, fail_after_commit=False):
        token = self.ticket(running=True)
        calls = []
        outer = self
        plan = SimpleNamespace(
            file_id="file-0", action="move", conflict_decision=decision
        )

        class FakeOrganizer:
            def __init__(self, **kwargs):
                self.guard = kwargs.get("before_plan_write")

            def _validate_target_outside_source(self, *_args):
                pass

            def organize(self, _parent, rules, **kwargs):
                calls.append((kwargs["dry_run"], rules.clean_empty))
                if not kwargs["dry_run"]:
                    if revoke:
                        outer.enabled = False
                    self.guard(plan, "commit")
                    if fail_after_commit:
                        raise RuntimeError("injected move failure")
                return [plan], {} if kwargs["dry_run"] else {"moved": 1}

            @staticmethod
            def trigger_post_actions(*_args, **_kwargs):
                return None

        remote = GuangYaFile(
            "file-0", "ABC-123.mp4", False, size=1024, parent_id="parent"
        )
        with (
            patch.object(
                confirmations,
                "GuangYaClient",
                return_value=SimpleNamespace(file_info=lambda _: remote),
            ),
            patch.object(
                confirmations,
                "ScopedGuangYaClient",
                return_value=SimpleNamespace(begin_source_scan=lambda: None),
            ),
            patch.object(
                confirmations,
                "_resolve_guangya_confirmation_candidate",
                return_value=(object(), object(), {}, "clean_title"),
            ),
            patch.object(confirmations, "FixedMatchScraper", return_value=object()),
            patch.object(confirmations, "Organizer", FakeOrganizer),
        ):
            result = confirmations._execute_guangya_confirmation(
                token,
                self.payload,
                self.candidate,
                selected_index=0,
                chat_id="100",
                actor="agent",
            )
        return token, result, calls

    def test_success_disables_empty_directory_cleanup_and_labels_fallback(self):
        token, result, calls = self.run_worker()
        self.assertEqual(calls, [(True, False), (False, False)])
        self.assertEqual(result["stats"]["moved"], 1)
        row = db.get_organize_confirmation(token)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["confirmation_actor"], "agent")
        event = confirmations._confirmation_result_event(
            self.payload,
            self.candidate,
            {"moved": 1},
            actor="agent",
        )
        self.assertIn("Agent", event.title)
        self.assertIn("清洗入库 · 无完整元数据", str(event.fields))

    def test_preflight_conflict_and_late_revocation_remain_manual(self):
        with self.assertRaises(DirectoryScrapeConflictError):
            self.run_worker(decision="replace")
        self.assertEqual(len(self.pending_manual()), 1)

    def test_after_preflight_authorization_can_be_revoked(self):
        with self.assertRaises(DirectoryScrapeConflictError):
            self.run_worker(revoke=True)
        self.assertEqual(len(self.pending_manual()), 1)

    def test_after_write_attempt_does_not_issue_blind_retry(self):
        with self.assertRaises(RuntimeError):
            self.run_worker(fail_after_commit=True)
        self.assertEqual(self.pending_manual(), [])

    def real_execution(self, *, snapshot_hook=None, conflict="new", target_files=()):
        guard = confirmations._AgentCleanWriteBoundary(self.payload, self.candidate)
        client = Mock()
        client.list_dir.return_value = list(target_files)
        organizer = Organizer(client=client, scraper=object(), before_plan_write=guard)
        plan = OrganizePlan(
            file_id="file-0",
            original_name="ABC-123.mp4",
            original_path="parent",
            original_parent_id="parent",
            size=1024,
            target_path="adult/ABC-123",
            new_name="ABC-123.cleaned.mp4",
            action="move",
            conflict_decision="new",
            match=MatchResult(
                title="ABC-123",
                media_type="movie",
                provider="clean_title",
                external_id="ABC-123",
            ),
        )
        stats = dict.fromkeys(
            (
                "moved",
                "renamed",
                "metadata_moved",
                "subtitle_moved",
                "stopped",
                "skipped",
                "conflict",
                "failed",
            ),
            0,
        )
        existing = (
            GuangYaFile("old", "ABC-123.old.mp4", False, size=100)
            if conflict == "replace"
            else None
        )
        with (
            patch.object(organizer, "_ensure_dir_chain", return_value="target"),
            patch.object(
                organizer, "_verify_remote_snapshot", side_effect=snapshot_hook
            ),
            patch.object(
                organizer,
                "_resolve_variant_conflict",
                return_value=(existing, conflict, ""),
            ),
        ):
            execute_organize_plans(organizer, [plan], self.rules, stats, {}, None)
        return guard, client, stats

    def test_last_snapshot_read_can_revoke_authorization_without_write(self):
        count = 0

        def snapshot(*_args, **_kwargs):
            nonlocal count
            count += 1
            if count == 2:
                self.enabled = False

        guard, client, stats = self.real_execution(snapshot_hook=snapshot)
        self.assertEqual(count, 2)
        self.assertFalse(guard.media_write_attempted)
        client.move.assert_not_called()
        client.rename.assert_not_called()
        client.delete.assert_not_called()
        self.assertEqual(stats["failed"], 1)

    def test_last_snapshot_read_failure_never_counts_as_a_write_attempt(self):
        guard, client, _ = self.real_execution(
            snapshot_hook=[None, RuntimeError("source changed")]
        )
        self.assertFalse(guard.media_write_attempted)
        client.move.assert_not_called()
        client.rename.assert_not_called()
        client.delete.assert_not_called()

    def test_live_conflict_cannot_backup_replace_or_recycle_old_media(self):
        guard, client, _ = self.real_execution(conflict="replace")
        self.assertFalse(guard.media_write_attempted)
        client.move.assert_not_called()
        client.rename.assert_not_called()
        client.delete.assert_not_called()

    def test_orphan_target_metadata_is_not_implicitly_overwritten(self):
        guard, client, _ = self.real_execution(
            target_files=[
                GuangYaFile("old-nfo", "ABC-123.nfo", False, size=100),
            ]
        )
        self.assertFalse(guard.media_write_attempted)
        client.move.assert_not_called()
        client.rename.assert_not_called()
        client.delete.assert_not_called()

    def test_already_moved_part_of_same_frozen_group_is_not_a_foreign_conflict(self):
        guard = confirmations._AgentCleanWriteBoundary(self.payload, self.candidate)
        guard(
            SimpleNamespace(action="move", conflict_decision="new"),
            "conflict",
            target_files=[GuangYaFile("file-0", "ABC-123.mp4", False, size=1024)],
        )
        self.assertFalse(guard.media_write_attempted)

    def test_real_engine_success_marks_attempt_only_before_move(self):
        guard, client, stats = self.real_execution()
        self.assertTrue(guard.media_write_attempted)
        client.move.assert_called_once_with(["file-0"], "target")
        self.assertEqual(stats["moved"], 1)
        client.delete.assert_not_called()

    def test_real_organizer_and_confirmation_share_one_safe_clean_write_path(self):
        from tests.test_organize_multiversion import _VariantTreeClient

        # 仅替换云端传输/探测/通知；保留冻结解析、扫描、规划、移动与审计代码。
        self.rules.small_file_mb = 0
        self.rules.link_strm = False
        self.rules.emby_refresh = False
        self.payload["rules"] = confirmations.organize_rules_snapshot(self.rules)
        token = self.ticket(running=True)
        incoming = GuangYaFile(
            "file-0", "ABC-123.mp4", False, size=1024, parent_id="parent"
        )
        client = _VariantTreeClient(
            incoming, GuangYaFile("unused", "Unused.mp4", False)
        )
        incoming.parent_id = "parent"
        client.tree = {
            "0": [
                GuangYaFile("source", "专用源", True, parent_id="0"),
                GuangYaFile("archive", "归档", True, parent_id="0"),
            ],
            "source": [GuangYaFile("parent", "待整理", True, parent_id="source")],
            "parent": [incoming],
            "archive": [],
        }
        with (
            patch.object(confirmations, "GuangYaClient", return_value=client),
            patch.object(Organizer, "_probe_move_plan_profiles"),
            patch.object(Organizer, "trigger_post_actions"),
            patch.object(confirmations, "NsfwRecognizer") as metatube,
        ):
            result = confirmations._execute_guangya_confirmation(
                token,
                self.payload,
                self.candidate,
                selected_index=0,
                chat_id="100",
                actor="agent",
            )
        metatube.assert_not_called()
        self.assertEqual(result["stats"]["moved"], 1)
        self.assertEqual(len(client.moves), 1)
        self.assertEqual(client.deleted, [])
        self.assertIn("parent", client.tree)  # 原待整理空目录不自动清理。
        self.assertNotEqual(incoming.parent_id, "parent")
        rows = [
            dict(r)
            for r in db.list_organize_logs_by_operation_token(
                f"recognition-confirm:{token}"
            )
        ]
        self.assertTrue(rows)
        self.assertTrue(all(row["confirmation_actor"] == "agent" for row in rows))
        self.assertTrue(all(row["provider"] == "clean_title" for row in rows))
