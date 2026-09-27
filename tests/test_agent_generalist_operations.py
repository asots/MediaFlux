"""项目级 Agent 跨模块执行能力的安全回归。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from app import database as db
from app.agent.errors import AgentToolError
from app.agent.effect_completion import wait_for_effect_completion
from app.agent.models import ToolContext
from app.agent.local_media_scan_actions import (
    local_media_scan_arguments,
    prepare_scan_local_media_sources,
    scan_local_media_sources_confirmed,
)
from app.agent.media_proxy_actions import (
    media_proxy_restart_arguments,
    prepare_restart_media_proxy_instance,
    restart_media_proxy_instance_confirmed,
)
from app.modules.local_media_scheduler import LocalMediaScheduler
from tests.support import IsolatedDatabaseTestCase


class AgentGeneralistOperationTests(IsolatedDatabaseTestCase):
    def setUp(self) -> None:
        with db.get_conn() as conn:
            conn.execute("DELETE FROM local_media_tasks")
            conn.execute("DELETE FROM local_library_targets")
            conn.execute("DELETE FROM local_media_sources")
            conn.execute("DELETE FROM media_proxy_instances")

    def tearDown(self) -> None:
        pass

    def test_local_media_scan_is_limited_to_confirmed_configured_sources(self) -> None:
        self.assertEqual(
            local_media_scan_arguments({}), {"source_numbers": [], "query": ""}
        )
        self.assertEqual(
            local_media_scan_arguments({"source_numbers": [2, 1, 2]}),
            {"source_numbers": [2, 1], "query": ""},
        )
        with self.assertRaises(AgentToolError):
            local_media_scan_arguments({"source_numbers": [True]})
        with (
            tempfile.TemporaryDirectory() as root_raw,
            tempfile.TemporaryDirectory() as target_raw,
        ):
            root = Path(root_raw)
            source_id = db.create_local_media_source(
                name="下载来源",
                qb_profile="",
                qb_path_prefix="",
                local_root=str(root),
                stable_seconds=0,
                owner="admin",
            )
            db.upsert_local_library_target(
                source_id, "default", target_raw, owner="admin"
            )
            preview, fingerprint = prepare_scan_local_media_sources(
                {"source_numbers": [1]}
            )
            self.assertEqual(preview.status, "confirmation_required")
            self.assertEqual(preview.data["eligible"], 1)
            scheduler = Mock()
            scheduler.status.return_value = {"running": False}
            scheduler.enqueue_manual_scan_candidates.return_value = {
                "scanned_sources": 1,
                "candidate_count": 2,
                "queued_count": 2,
            }
            with patch(
                "app.agent.local_media_scan_actions.get_local_media_scheduler",
                return_value=scheduler,
            ):
                result = scan_local_media_sources_confirmed(
                    {"source_numbers": [1]}, fingerprint
                )
            self.assertTrue(result.ok)
            self.assertEqual(result.data["queued_tasks"], 2)
            scheduler.enqueue_manual_scan_candidates.assert_called_once_with(
                silent=True, source_ids={source_id}, candidate_query=""
            )
            scheduler.start.assert_called_once_with()

    def test_scan_filter_preserves_titles_and_rejects_invalid_nonempty_filters(self):
        for title in ("Alpha.Show", "The-Last-of-Us", "MF-Audit-Local", "绝命毒师"):
            with self.subTest(title=title):
                self.assertEqual(local_media_scan_arguments({"query": title})["query"], title)
        for query in (123, None, "/private/media", "access_token=secret", "x" * 121):
            with self.subTest(query=query), self.assertRaises(AgentToolError):
                local_media_scan_arguments({"query": query})
        self.assertEqual(local_media_scan_arguments({"query": "  "})["query"], "")

    def test_scheduler_filters_manual_scan_by_media_name(self) -> None:
        with (
            tempfile.TemporaryDirectory() as root_raw,
            tempfile.TemporaryDirectory() as target_raw,
        ):
            root = Path(root_raw)
            (root / "Alpha.Show.S01E01.mkv").write_bytes(b"alpha")
            (root / "Beta.Show.S01E01.mkv").write_bytes(b"beta")
            source_id = db.create_local_media_source(
                name="下载来源",
                qb_profile="",
                qb_path_prefix="",
                local_root=str(root),
                stable_seconds=0,
                owner="admin",
            )
            db.upsert_local_library_target(
                source_id, "default", target_raw, owner="admin"
            )
            result = LocalMediaScheduler(service=Mock()).enqueue_manual_scan_candidates(
                silent=True, source_ids={source_id}, candidate_query=local_media_scan_arguments({"query": "Alpha.Show"})["query"]
            )
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["queued_count"], 1)
        task = db.get_local_media_task(result["task_ids"][0], owner="admin")
        self.assertIn("Alpha.Show", task.content_path)

    def test_real_scan_receipt_drives_completion_and_empty_candidates_need_no_wait(self):
        with (
            tempfile.TemporaryDirectory() as root_raw,
            tempfile.TemporaryDirectory() as target_raw,
        ):
            root = Path(root_raw)
            source_id = db.create_local_media_source(
                name="扫描来源", qb_profile="", qb_path_prefix="",
                local_root=root_raw, stable_seconds=0, owner="admin",
            )
            db.upsert_local_library_target(source_id, "default", target_raw, owner="admin")
            scheduler = LocalMediaScheduler(service=Mock())
            _, fingerprint = prepare_scan_local_media_sources({})
            with (
                patch("app.agent.local_media_scan_actions.get_local_media_scheduler", return_value=scheduler),
                patch.object(scheduler, "status", return_value={"running": False}),
                patch.object(scheduler, "start") as start,
            ):
                empty = scan_local_media_sources_confirmed({}, fingerprint)
                self.assertEqual(empty.status, "completed")
                self.assertEqual(empty.data["candidates"], 0)
                self.assertEqual(empty.effect_metadata, {})
                self.assertIs(asyncio.run(wait_for_effect_completion(
                    empty, tool="local_media.scan_sources", context=ToolContext(owner="scan-test")
                )), empty)
                start.assert_not_called()

                (root / "Local.Scan.S01E01.mkv").write_bytes(b"isolated-test-video")
                accepted = scan_local_media_sources_confirmed({}, fingerprint)
                start.assert_called_once_with()
            self.assertEqual(accepted.status, "accepted")
            self.assertEqual(accepted.effect_metadata["completion"], {
                "kind": "local_media_scan", "scan_ref": accepted.data["scan_ref"],
            })
            self.assertNotIn("effect_metadata", accepted.to_dict())
            self.assertNotIn("completion", accepted.to_model_dict())
            from app.modules.local_media_scan_runs import resolve_local_media_scan

            scan = resolve_local_media_scan(accepted.data["scan_ref"])
            self.assertEqual(len(scan["task_ids"]), 1)
            task_id = scan["task_ids"][0]
            db.add_local_media_task_item(
                task_id, str(root / "Local.Scan.S01E01.mkv"),
                str(Path(target_raw) / "Local.Scan.S01E01.mkv"), role="video", action="move",
            )
            db.update_local_media_task(task_id, status="completed")
            completed = asyncio.run(wait_for_effect_completion(
                accepted, tool="local_media.scan_sources", context=ToolContext(owner="scan-test")
            ))
            self.assertTrue(completed.ok)
            self.assertEqual(completed.status, "completed")
            self.assertEqual(completed.data["stats"]["archived_video_count"], 1)
            self.assertNotIn(root_raw, str(completed.to_dict()))
            self.assertNotIn("task_ids", str(completed.to_dict()))

    def test_media_proxy_restart_queues_runtime_rebuild_without_config_write(
        self,
    ) -> None:
        self.assertEqual(
            media_proxy_restart_arguments({"instance_number": 1}),
            {"instance_number": 1},
        )
        instance_id = db.add_media_proxy_instance(
            name="Jellyfin 反代",
            server_type="jellyfin",
            upstream_url="http://jellyfin.invalid:8096",
            api_key="secret",
            listen_host="127.0.0.1",
            listen_port=19096,
            local_root="/media",
            enabled=1,
        )
        preview, fingerprint = prepare_restart_media_proxy_instance(
            {"instance_number": 1}
        )
        manager = Mock()
        manager.request_restart.return_value = True
        with (
            patch(
                "app.agent.media_proxy_actions.clear_signed_url_cache", return_value=3
            ),
            patch(
                "app.agent.media_proxy_actions.get_media_proxy_manager",
                return_value=manager,
            ),
        ):
            result = restart_media_proxy_instance_confirmed(
                {"instance_number": 1}, fingerprint
            )
        self.assertEqual(preview.status, "confirmation_required")
        self.assertTrue(result.ok)
        self.assertEqual(result.data["cache_entries_cleared"], 3)
        manager.request_restart.assert_called_once_with(instance_id)


class MediaProxyRuntimeRestartTests(unittest.IsolatedAsyncioTestCase):
    async def test_restart_instance_rebuilds_only_selected_runtime(self) -> None:
        from app.modules import media_proxy

        row = {
            "id": 7,
            "enabled": 1,
            "listen_host": "127.0.0.1",
            "listen_port": 18097,
            "upstream_url": "http://127.0.0.1:8096",
        }
        previous = MagicMock()
        replacement = MagicMock()
        manager = media_proxy.MediaProxyManager()
        manager._runtimes[7] = previous
        with (
            patch.object(
                media_proxy.database, "get_media_proxy_instance", return_value=row
            ),
            patch.object(media_proxy.database, "update_media_proxy_instance"),
            patch.object(media_proxy, "resolve_proxy_instance", return_value=row),
            patch.object(manager, "_stop_runtime", new=AsyncMock()) as stop,
            patch.object(
                manager, "_start_runtime", new=AsyncMock(return_value=replacement)
            ) as start,
        ):
            result = await manager.restart_instance(7)
        self.assertEqual(result, {"restarted": True, "reason": ""})
        stop.assert_awaited_once_with(previous)
        start.assert_awaited_once_with(row)
        self.assertIs(manager._runtimes[7], replacement)
