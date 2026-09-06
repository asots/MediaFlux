"""STRM 路径迁移的进程中断恢复与历史副本校准；只使用临时数据库/假云盘。"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import database as db
from app.clients.guangya import GuangYaFile
from app.modules import strm
from app.modules.scheduler import STRMScheduler
from tests.support import isolated_test_database
from tests.support import PagedDirectoryTestMixin

BASE_URL = "http://play.invalid"


class MoveCloud(PagedDirectoryTestMixin):
    def __init__(self, location="Old", source="source", fail_source=""):
        self.location = location
        self.source = source
        self.fail_source = fail_source

    def list_dir(self, fid):
        if fid == self.fail_source:
            raise RuntimeError("isolated scan failure")
        if fid in {"source", "other"}:
            return (
                [GuangYaFile("series", self.location, True, parent_id=fid)]
                if fid == self.source
                else []
            )
        return [self.file_info("video")]

    def file_info(self, fid):
        return GuangYaFile(fid, "Ultraman.S01E01.mkv", False, 100, "etag", "series")


def sync(root, location="Old", source="source", **kwargs):
    return strm.sync_strm(
        source, BASE_URL, str(root), client=MoveCloud(location, source), **kwargs
    )


def crash_move(db_path, root, stage="after_index"):
    code = """
import os, sys
import tests
from pathlib import Path
from unittest.mock import patch
from app import database as db
from app.modules import strm
from tests.test_strm_move_recovery import sync

db.configure_database(Path(sys.argv[1]), test_mode=True)
real_delete = strm._delete_owned_file
def crash(*args, **kwargs):
    if sys.argv[3] == "after_delete":
        real_delete(*args, **kwargs)
    os._exit(73)
target = "app.modules.strm.db.upsert_strm_index" if sys.argv[3] == "before_index" else "app.modules.strm._delete_owned_file"
with patch(target, side_effect=crash):
    sync(sys.argv[2], "New")
os._exit(74)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(db_path), str(root), stage],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 73, (result.returncode, result.stdout, result.stderr)


class StrmMoveRecoveryTests(unittest.TestCase):
    def test_process_exit_at_each_install_boundary_converges(self):
        for stage in ("before_index", "after_index", "after_delete"):
            with (
                self.subTest(stage=stage),
                isolated_test_database() as database_path,
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                sync(root)
                old = next(root.rglob("*.strm"))
                crash_move(database_path, root, stage)
                stats = sync(root, "New")
                self.assertEqual(stats["failed"], 0, stats)
                self.assertFalse(old.exists(), stats)
                self.assertEqual(len(list(root.rglob("*.strm"))), 1)
                self.assertIn(str(old), stats["changed_strm_paths"], stats)
                again = sync(root, "New")
                self.assertEqual(again["generated"], 0)
                self.assertEqual(again["cleaned"], 0)
                self.assertFalse(again["clean_skipped"], again)

    def test_full_sync_removes_historical_unindexed_exact_copy(self):
        with isolated_test_database(), tempfile.TemporaryDirectory() as root:
            sync(root)
            old = next(Path(root).rglob("*.strm"))
            db.delete_strm_index_ids("guangya:source", ["video"])
            stats = sync(root, "New")
            self.assertFalse(old.exists())
            self.assertEqual(stats["cleaned"], 1)
            self.assertIn(str(old), stats["changed_strm_paths"])
            self.assertEqual(len(list(Path(root).rglob("*.strm"))), 1)

    def test_historical_cleanup_preserves_foreign_changed_and_non_strm_files(self):
        with isolated_test_database(), tempfile.TemporaryDirectory() as root:
            sync(root, "New")
            target = next(Path(root).rglob("*.strm"))
            old_dir = target.parent.parent / "Old"
            old_dir.mkdir()
            contents = {
                "foreign.strm": b"http://other.invalid/play/video",
                "edited.strm": target.read_bytes() + b"# manual edit",
                "episode.nfo": b"<movie/>",
                "poster.jpg": b"jpeg",
            }
            for name, payload in contents.items():
                (old_dir / name).write_bytes(payload)
            stats = sync(root, "New")
            for name, payload in contents.items():
                self.assertEqual((old_dir / name).read_bytes(), payload)
            self.assertEqual(stats["cleaned"], 0)
            self.assertTrue(stats["clean_skipped"], stats)
            self.assertTrue(stats["error_samples"], stats)

    def test_incomplete_scan_does_not_clean_historical_copy(self):
        with isolated_test_database(), tempfile.TemporaryDirectory() as root:
            sync(root)
            old = next(Path(root).rglob("*.strm"))
            db.delete_strm_index_ids("guangya:source", ["video"])
            cloud = MoveCloud("New")
            original = cloud.list_dir

            def entries(fid):
                if fid == "source":
                    return original(fid) + [GuangYaFile("broken", "Broken", True)]
                if fid == "broken":
                    raise RuntimeError("isolated incomplete listing")
                return original(fid)

            cloud.list_dir = entries
            stats = strm.sync_strm("source", BASE_URL, root, client=cloud)
            self.assertTrue(old.is_file())
            self.assertTrue(stats["clean_skipped"])
            self.assertEqual(stats["generated"], 0)

    def test_interrupted_old_file_user_edit_is_preserved_and_reported(self):
        with (
            isolated_test_database() as database_path,
            tempfile.TemporaryDirectory() as root,
        ):
            sync(root)
            old = next(Path(root).rglob("*.strm"))
            crash_move(database_path, root)
            old.write_text("user changed this pointer", encoding="utf-8")
            stats = sync(root, "New")
            self.assertEqual(old.read_text(), "user changed this pointer")
            self.assertTrue(stats["clean_skipped"], stats)
            self.assertTrue(stats["error_samples"], stats)

    def test_symlink_and_file_outside_strm_root_are_preserved(self):
        with (
            isolated_test_database(),
            tempfile.TemporaryDirectory() as root,
            tempfile.TemporaryDirectory() as external,
        ):
            sync(root, "New")
            target = next(Path(root).rglob("*.strm"))
            outside = Path(external) / "other.strm"
            outside.write_bytes(target.read_bytes())
            link_dir = target.parent.parent / "Linked"
            link_dir.symlink_to(external, target_is_directory=True)
            link = target.parent.parent / "old.strm"
            link.symlink_to(outside)
            stats = sync(root, "New")
            self.assertTrue(link.is_symlink())
            self.assertTrue(outside.is_file())
            self.assertEqual(stats["cleaned"], 0)

    def test_another_sources_indexed_copy_is_not_an_orphan(self):
        with isolated_test_database(), tempfile.TemporaryDirectory() as root:
            sync(root, "New")
            target = next(Path(root).rglob("*.strm"))
            other = target.parent.parent / "Other.strm"
            other.write_bytes(target.read_bytes())
            db.upsert_strm_index(
                "guangya:other",
                "video",
                "etag",
                100,
                "Ultraman.S01E01.mkv",
                str(other),
                strm._content_fingerprint(other),
            )
            stats = sync(root, "New")
            self.assertTrue(other.is_file())
            self.assertEqual(stats["cleaned"], 0)

    def test_scheduler_full_round_sweeps_cross_source_orphans_once(self):
        for scoped in (False, True):
            with (
                self.subTest(scoped=scoped),
                isolated_test_database(),
                tempfile.TemporaryDirectory() as root,
            ):
                sources = [
                    {"id": "source", "name": "A", "rel_prefix": "A"},
                    {"id": "other", "name": "B", "rel_prefix": "B"},
                ]
                sync(root, rel_prefix="A")
                old = next(Path(root).rglob("*.strm"))
                db.delete_strm_index_ids("guangya:source", ["video"])
                selected = sources[1:] if scoped else sources
                scheduler = STRMScheduler()
                scheduler._source_runtime = [
                    {**s, "status": "pending", "completed": 0, "total": 0}
                    for s in selected
                ]

                def execute(**kwargs):
                    return strm.sync_strm(client=MoveCloud("New", "other"), **kwargs)

                with (
                    patch(
                        "app.modules.scheduler.configured_strm_source_plans",
                        return_value=(sources, ""),
                    ),
                    patch("app.modules.scheduler.sync_strm", side_effect=execute),
                ):
                    stats, _, stopped = scheduler._run_full_sources(
                        selected,
                        base_url=BASE_URL,
                        strm_root=root,
                        exts={"mkv"},
                        metadata_exts=set(),
                        threshold=0,
                        active_ids_complete=not scoped,
                    )
                self.assertFalse(stopped)
                self.assertEqual(old.exists(), scoped, stats)
                self.assertEqual(stats["cleaned"], 0 if scoped else 1)

    def test_incremental_recovery_is_scoped_to_confirmed_ids(self):
        with (
            isolated_test_database() as database_path,
            tempfile.TemporaryDirectory() as root,
        ):
            sync(root)
            old = next(Path(root).rglob("*.strm"))
            crash_move(database_path, root)
            change = {
                "source_id": "source",
                "kind": "video",
                "action": "upsert",
                "file_id": "video",
                "parent_id": "series",
                "rel_dir": "New",
                "name": "Ultraman.S01E01.mkv",
                "etag": "etag",
                "size": 100,
            }
            stats = strm.sync_strm_incremental(
                "source", [change], BASE_URL, root, client=MoveCloud("New")
            )
            self.assertFalse(stats["fallback_required"], stats)
            self.assertFalse(old.exists())
            self.assertEqual(stats["cleaned"], 1)
            self.assertEqual(db.list_strm_path_cleanup("guangya:source"), [])

    def test_recovery_keeps_old_copy_when_new_pointer_is_missing(self):
        from app.modules.strm_recovery import recover_pending_paths

        with (
            isolated_test_database() as database_path,
            tempfile.TemporaryDirectory() as root,
        ):
            sync(root)
            old = next(Path(root).rglob("*.strm"))
            crash_move(database_path, root)
            current = Path(db.list_strm_index("guangya:source")[0]["strm_path"])
            current.unlink()
            stats = STRMScheduler._empty_stats()
            recover_pending_paths("guangya:source", root, stats, valid_ids={"video"})
            self.assertTrue(old.exists())
            self.assertTrue(stats["clean_skipped"])
            self.assertEqual(len(db.list_strm_path_cleanup("guangya:source")), 1)
            restored = sync(root, "New")
            self.assertTrue(current.is_file())
            self.assertFalse(old.exists(), restored)

    def test_cleanup_journal_rejects_symlink_replacement_of_old_path(self):
        with (
            isolated_test_database() as database_path,
            tempfile.TemporaryDirectory() as root,
        ):
            sync(root)
            old = next(Path(root).rglob("*.strm"))
            crash_move(database_path, root)
            current = Path(db.list_strm_index("guangya:source")[0]["strm_path"])
            other = current.parent / "user.data"
            other.write_bytes(old.read_bytes())
            old.unlink()
            old.symlink_to(other)
            stats = sync(root, "New")
            self.assertTrue(other.is_file())
            self.assertTrue(old.is_symlink())
            self.assertTrue(stats["clean_skipped"])
            self.assertEqual(len(db.list_strm_path_cleanup("guangya:source")), 1)

    def test_recovery_refresh_handoff_failure_preserves_journal_for_retry(self):
        with (
            isolated_test_database() as database_path,
            tempfile.TemporaryDirectory() as root,
        ):
            sync(root)
            old = next(Path(root).rglob("*.strm"))
            crash_move(database_path, root)

            def unavailable(paths):
                raise OSError("isolated outbox failure")

            stats = sync(root, "New", on_refresh_paths=unavailable)
            self.assertFalse(old.exists())
            self.assertTrue(stats["stopped"])
            self.assertEqual(len(db.list_strm_path_cleanup("guangya:source")), 1)
            refreshed = []
            stats = sync(
                root, "New", on_refresh_paths=lambda paths: refreshed.extend(paths)
            )
            self.assertIn(str(old.parent), refreshed)
            self.assertFalse(stats["clean_skipped"])
            self.assertEqual(db.list_strm_path_cleanup("guangya:source"), [])

    def test_recovery_scan_is_bounded_and_preserves_unvisited_files(self):
        from app.modules.strm_recovery import reconcile_historical_strm

        with isolated_test_database(), tempfile.TemporaryDirectory() as root:
            sync(root, "New")
            current = next(Path(root).rglob("*.strm"))
            for i in range(5):
                target = current.parent.parent / f"Old{i}" / "Copy.strm"
                target.parent.mkdir()
                target.write_bytes(current.read_bytes())
            stats = STRMScheduler._empty_stats()
            with patch.object(strm, "_scan_limits", return_value=(2, 100, 100, 30)):
                reconcile_historical_strm(root, BASE_URL, [{"id": "source"}], stats)
            self.assertTrue(stats["clean_skipped"])
            self.assertTrue(stats["error_samples"])
            self.assertGreater(len(list(Path(root).rglob("Copy.strm"))), 0)

    def test_recovery_preserves_old_if_new_copy_changes_during_scan(self):
        from app.modules import strm_recovery

        with isolated_test_database(), tempfile.TemporaryDirectory() as root:
            sync(root, "New")
            current = next(Path(root).rglob("*.strm"))
            old = current.parent.parent / "Old.strm"
            old.write_bytes(current.read_bytes())

            def walk(*args):
                current.write_text("changed by an external writer", encoding="utf-8")
                yield old

            stats = STRMScheduler._empty_stats()
            with patch.object(strm_recovery, "_local_pointers", side_effect=walk):
                strm_recovery.reconcile_historical_strm(
                    root, BASE_URL, [{"id": "source"}], stats
                )
            self.assertTrue(old.is_file())
            self.assertTrue(stats["clean_skipped"])

    def test_recovery_rechecks_old_fingerprint_at_delete_boundary(self):
        from app.modules import strm_recovery

        with isolated_test_database(), tempfile.TemporaryDirectory() as root:
            sync(root, "New")
            current = next(Path(root).rglob("*.strm"))
            old = current.parent.parent / "Old.strm"
            old.write_bytes(current.read_bytes())
            real_delete = strm._delete_owned_file

            def external_write(path, owners, action):
                path.write_text("external pointer", encoding="utf-8")
                return real_delete(path, owners, action)

            stats = STRMScheduler._empty_stats()
            with patch.object(strm, "_delete_owned_file", side_effect=external_write):
                strm_recovery.reconcile_historical_strm(
                    root, BASE_URL, [{"id": "source"}], stats
                )
            self.assertEqual(old.read_text(), "external pointer")
            self.assertTrue(stats["clean_skipped"])

    def test_stale_file_delete_error_is_reported_and_remains_retryable(self):
        with isolated_test_database(), tempfile.TemporaryDirectory() as root:
            sync(root)
            old = next(Path(root).rglob("*.strm"))
            cloud = MoveCloud("New")
            cloud.file_info = lambda fid: GuangYaFile(
                "replacement", "Ultraman.S01E01.mkv", False, 100, "etag", "series"
            )
            real_delete = strm._delete_owned_file

            def permission_error(path, rows, action):
                if path == old:
                    raise PermissionError("isolated NAS directory permission error")
                return real_delete(path, rows, action)

            with patch.object(strm, "_delete_owned_file", side_effect=permission_error):
                stats = strm.sync_strm("source", BASE_URL, root, client=cloud)
            self.assertTrue(old.exists())
            self.assertEqual(stats["generated"], 1)
            self.assertTrue(stats["clean_skipped"], stats)
            self.assertTrue(stats["error_samples"], stats)
            retry = strm.sync_strm("source", BASE_URL, root, client=cloud)
            self.assertFalse(old.exists(), retry)
            self.assertFalse(retry["clean_skipped"], retry)

    def test_normal_move_does_not_duplicate_removed_file_details(self):
        with isolated_test_database(), tempfile.TemporaryDirectory() as root:
            sync(root)
            stats = sync(root, "New")
            removed = [item for item in stats["changes"] if item["action"] == "removed"]
            self.assertEqual(stats["cleaned"], 1)
            self.assertEqual(len(removed), 1, stats)
            self.assertEqual(len(stats["changed_strm_paths"]), 2, stats)
