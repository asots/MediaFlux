"""真实子进程退出和 ZIP 恢复，验证本轮重试/通知/游标的持久边界。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

from app import database as db
from app.modules import backup
from app.modules.local_media_scheduler import LocalMediaScheduler
from app.repositories import telegram_notifications as notifications
from app.runtime_paths import RuntimePaths
from tests.support import isolated_test_database


def runtime_paths(database_path):
    root = database_path.parent
    paths = RuntimePaths(
        root / "program",
        root,
        root,
        root / "cache",
        root / "logs",
        root / "strm",
        root / "trash",
    )
    paths.ensure_writable_dirs()
    paths.env_file.write_text("WEB_PORT=1258\n", encoding="utf-8")
    return paths


class TenPassProcessRestoreAuditTests(unittest.TestCase):
    def child(self, script, *args, expected_exit):
        completed = subprocess.run(
            [sys.executable, "-c", script, *map(str, args)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, expected_exit, completed.stderr)

    def test_process_exit_after_target_retirement_preserves_manual_verification_gate(
        self,
    ):
        with isolated_test_database("mediaflux.db") as path:
            script = r"""
import tests
import os, sys, json, socket
from pathlib import Path
from unittest.mock import patch
from app import database as db
from app.modules.local_media_service import LocalMediaService
from app.modules.local_move_transaction import LocalMoveTransaction
from app.modules.organize import OrganizeRules
from app.modules.scraper import MatchResult
from tests.test_local_media_service import FakeScraper
root = Path(sys.argv[1]).parent
socket.socket.connect = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("network forbidden"))
db.configure_database(sys.argv[1], test_mode=True)
db.init_db()
incoming_root, library = root / "incoming", root / "library"
incoming_root.mkdir(); library.mkdir()
incoming = incoming_root / "Movie.2026.mkv"
incoming.write_bytes(b"new-process-media-larger")
source = db.create_local_media_source(name="process-audit", qb_profile="", qb_path_prefix="", local_root=str(incoming_root), owner="admin", media_type="movie")
db.upsert_local_library_target(source, "movie", str(library), owner="admin")
service = LocalMediaService(scraper=FakeScraper(MatchResult(tmdb_id="1", title="Movie", year="2026", media_type="movie", confidence=1.0)))
rules = OrganizeRules(region_split=False, year_split=False, naming_scope="both", conflict_strategy=2, emby_refresh=False, media_probe_enabled=False, clean_empty=False)
with patch("app.modules.local_media_service.OrganizeRules.from_config", return_value=rules), patch("app.modules.local_media_service.probe_local_media_profile", return_value=None):
 inspection = service.inspect_source("admin", source, incoming)
 preview = service.preview("admin", inspection["inspection_id"], tmdb_id="1", media_type="movie")
 target = Path(preview["plans"][0]["target_path"])
 target.parent.mkdir(parents=True); target.write_bytes(b"old-process-media")
 task = service.create_manual_task("admin", inspection["inspection_id"], tmdb_id="1", media_type="movie", rules_snapshot=preview["rules_snapshot"])
 assert db.claim_local_media_task(task, owner="admin")
 (root / "process-manifest.json").write_text(json.dumps({"task": task, "source": source, "incoming": str(incoming), "target": str(target)}))
 original = LocalMoveTransaction._backup_replaced_target
 def retire_then_exit(transaction, *args, **kwargs):
  original(transaction, *args, **kwargs)
  os._exit(23)
 with patch.object(LocalMoveTransaction, "_backup_replaced_target", retire_then_exit):
  service.execute_task("admin", task)
raise AssertionError("child did not reach retirement")
"""
            self.child(script, path, expected_exit=23)
            manifest = json.loads((path.parent / "process-manifest.json").read_text())
            task_id = manifest["task"]
            self.assertEqual(db.get_local_media_task(task_id).status, "moving")
            db.init_db()
            recovered = db.get_local_media_task(task_id)
            self.assertEqual(recovered.status, "requires_manual")
            self.assertTrue(db.is_interrupted_local_media_write_error(recovered.error))
            with self.assertRaisesRegex(ValueError, "核验"):
                db.prepare_manual_local_media_task(
                    manifest["source"], manifest["incoming"], owner="admin"
                )
            self.assertFalse(db.reset_local_media_task(task_id, owner="admin"))
            db.init_db()
            self.assertEqual(db.get_local_media_task(task_id), recovered)
            target = Path(manifest["target"])
            retired = list(target.parent.glob(".*.mediaflux-replaced-*"))
            self.assertEqual(len(retired), 1)
            self.assertEqual(retired[0].read_bytes(), b"old-process-media")
            self.assertFalse(target.exists())
            self.assertEqual(
                Path(manifest["incoming"]).read_bytes(), b"new-process-media-larger"
            )

    def test_archive_preserves_cursor_manual_token_and_unknown_notification_then_recovers(
        self,
    ):
        with isolated_test_database("mediaflux.db") as path:
            paths = runtime_paths(path)
            sub_id = db.add_rss_subscription(
                "history-feed", "https://feed.invalid/rss", refresh_interval_minutes=10
            )
            db.kv_set("rss.scheduler.last_admitted_id", str(sub_id))
            source = db.create_local_media_source(
                name="history-local",
                qb_profile="",
                qb_path_prefix="",
                local_root="/synthetic",
                owner="admin",
            )
            task_id = db.create_local_media_task(
                source,
                "",
                "/synthetic/file.mkv",
                trigger="scan",
                operation_token="silent-manual-scan:history",
            )
            db.update_local_media_task(task_id, status="failed")
            old_task = db.get_local_media_task(task_id)
            notification = notifications.upsert_notification(
                "restore-receipt",
                topic="confirmation",
                importance="result",
                chat_id="100",
                event_json='{"title":"old"}',
                replace=True,
            )
            claimed = notifications.claim_due_notifications(
                event_key="restore-receipt"
            )[0]
            notifications.mark_outcome_unknown(
                claimed["id"],
                lease_generation=claimed["lease_generation"],
                claimed_revision=1,
                error="lost response",
            )
            archive = backup.create_backup(paths)
            db.kv_set("rss.scheduler.last_admitted_id", "9999")
            db.reset_local_media_task(task_id)
            notifications.upsert_notification(
                "restore-receipt",
                topic="confirmation",
                importance="result",
                chat_id="100",
                event_json='{"title":"new"}',
                preferred_message_id=77,
                replace=True,
            )
            backup.restore_backup(paths, archive)
            db.configure_database(path, test_mode=True)
            db.init_db()
            self.assertFalse(backup.recover_pending_restore(paths))
            self.assertEqual(db.kv_get("rss.scheduler.last_admitted_id"), str(sub_id))
            self.assertEqual(db.get_local_media_task(task_id), old_task)
            restored = notifications.get_notification("restore-receipt")
            self.assertEqual(
                (restored["id"], restored["status"]),
                (notification["id"], "outcome_unknown"),
            )
            updated = notifications.upsert_notification(
                "restore-receipt",
                topic="confirmation",
                importance="result",
                chat_id="100",
                event_json='{"title":"confirmed terminal"}',
                preferred_message_id=77,
                replace=True,
            )
            self.assertEqual(updated["status"], "pending")
            self.assertTrue(db.reset_local_media_task(task_id))
            retried = db.get_local_media_task(task_id)
            self.assertTrue(LocalMediaScheduler._is_silent_task(retried))
            self.assertNotEqual(retried.operation_token, old_task.operation_token)
            self.assertFalse(db.reset_local_media_task(task_id))

    def test_process_loss_mid_restore_rolls_back_live_state_once(self):
        with isolated_test_database("mediaflux.db") as path:
            paths = runtime_paths(path)
            db.kv_set("rss.scheduler.last_admitted_id", "4")
            archive = backup.create_backup(paths)
            db.kv_set("rss.scheduler.last_admitted_id", "9")
            script = r"""
import tests
import os, sys
from pathlib import Path
from unittest.mock import patch
from app import database as db
from app.modules import backup
from app.runtime_paths import RuntimePaths
path = Path(sys.argv[1]); root = path.parent
db.configure_database(path, test_mode=True)
paths = RuntimePaths(root / "program", root, root, root / "cache", root / "logs", root / "strm", root / "trash")
original = backup.os.replace
def replace_then_exit(source, destination):
 result = original(source, destination)
 if Path(destination) == path and Path(source).name.endswith(".tmp"):
  os._exit(24)
 return result
with patch.object(backup.os, "replace", replace_then_exit):
 backup.restore_backup(paths, Path(sys.argv[2]))
raise AssertionError("restore replacement not reached")
"""
            self.child(script, path, archive, expected_exit=24)
            self.assertTrue(backup.recover_pending_restore(paths))
            db.configure_database(path, test_mode=True)
            db.init_db()
            self.assertEqual(db.kv_get("rss.scheduler.last_admitted_id"), "9")
            self.assertFalse(backup.recover_pending_restore(paths))
            self.assertEqual(db.kv_get("rss.scheduler.last_admitted_id"), "9")
