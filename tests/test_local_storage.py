"""本地文件快照与安全扫描测试。"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.modules.local_storage import (
    LocalContentChanged,
    LocalFilesystemAdapter,
    LocalScanLimitExceeded,
    LocalStorageError,
    snapshot_digest,
)


class LocalStorageTests(unittest.TestCase):
    def test_cached_root_entries_recheck_kind_before_descent(self) -> None:
        for operation_name in ("contains_video", "scan"):
            with self.subTest(operation=operation_name):
                self._assert_root_entry_recheck(operation_name)


    def _assert_root_entry_recheck(self, operation_name: str) -> None:
        with tempfile.TemporaryDirectory(prefix="a5-boundary-root-") as root_raw, tempfile.TemporaryDirectory(
            prefix="a5-boundary-outside-"
        ) as outside_raw:
            root = Path(root_raw)
            outside = Path(outside_raw)
            replaced = root / "Volatile"
            parked = root / "Volatile.original"
            replaced.mkdir()
            (outside / "outside.mkv").write_bytes(b"outside")
            adapter = LocalFilesystemAdapter(root, cache_directory_names=True)

            original_is_dir = Path.is_dir
            original_scandir = os.scandir
            external_scandir: list[Path] = []
            swapped = False

            def replace_before_kind_check(path: Path) -> bool:
                nonlocal swapped
                if path == replaced and not swapped:
                    replaced.rename(parked)
                    replaced.symlink_to(outside, target_is_directory=True)
                    swapped = True
                return original_is_dir(path)

            def scandir(path):
                candidate = Path(path)
                resolved = candidate.resolve(strict=False)
                if resolved == outside or outside in resolved.parents:
                    external_scandir.append(resolved)
                return original_scandir(path)

            try:
                with patch.object(Path, "is_dir", new=replace_before_kind_check), patch.object(
                    os, "scandir", side_effect=scandir
                ):
                    result = getattr(adapter, operation_name)(root)
            finally:
                if replaced.is_symlink():
                    replaced.unlink()
                if parked.exists():
                    parked.rename(replaced)

            self.assertTrue(swapped, "未触发根层 entries 产生后的目录替换")
            self.assertEqual(
                external_scandir,
                [],
                "根层缓存的旧目录 kind 不得驱动源外 symlink 枚举",
            )
            if operation_name == "contains_video":
                self.assertFalse(result)
            else:
                self.assertEqual(result, [])


    def test_cached_path_rechecks_kind_on_next_consume(self) -> None:
        for operation_name in ("contains_video", "scan"):
            with self.subTest(operation=operation_name):
                self._assert_next_consume_recheck(operation_name)


    def _assert_next_consume_recheck(self, operation_name: str) -> None:
        with tempfile.TemporaryDirectory(prefix="a5-boundary-root-") as root_raw, tempfile.TemporaryDirectory(
            prefix="a5-boundary-outside-"
        ) as outside_raw:
            root = Path(root_raw)
            outside = Path(outside_raw)
            container = root / "Container"
            replaced = container / "CachedChild"
            parked = container / "CachedChild.original"
            container.mkdir()
            replaced.mkdir()
            (outside / "outside.mkv").write_bytes(b"outside")
            adapter = LocalFilesystemAdapter(root, cache_directory_names=True)

            self.assertFalse(adapter.contains_video(root))
            replaced.rename(parked)
            replaced.symlink_to(outside, target_is_directory=True)

            original_scandir = os.scandir
            external_scandir: list[Path] = []

            def scandir(path):
                candidate = Path(path)
                resolved = candidate.resolve(strict=False)
                if resolved == outside or outside in resolved.parents:
                    external_scandir.append(resolved)
                return original_scandir(path)

            with patch.object(os, "scandir", side_effect=scandir):
                result = getattr(adapter, operation_name)(root)

            if replaced.is_symlink():
                replaced.unlink()
            if parked.exists():
                parked.rename(replaced)

            self.assertEqual(
                external_scandir,
                [],
                "下一次消费不得信任缓存 path 的旧目录 kind",
            )
            if operation_name == "contains_video":
                self.assertFalse(result)
            else:
                self.assertEqual(result, [])


    def test_directory_name_cache_rechecks_names_and_in_place_video_sizes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            adapter = LocalFilesystemAdapter(root, cache_directory_names=True)
            self.assertFalse(adapter.contains_video(root))
            video = root / "Movie.mkv"
            video.touch()
            self.assertFalse(adapter.contains_video(root))
            video.write_bytes(b"video")
            self.assertTrue(adapter.contains_video(root))
            self.assertEqual([item.path for item in adapter.scan()], [video])
            video.write_bytes(b"")
            self.assertFalse(adapter.contains_video(root))
            video.unlink()
            replacement = root / "Other.mkv"
            replacement.write_bytes(b"new video")
            self.assertEqual([item.path for item in adapter.scan()], [replacement])

    def test_independent_operations_refresh_empty_names_with_same_directory_stat(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            adapter = LocalFilesystemAdapter(root, cache_directory_names=True)
            directory_info = root.lstat()
            original_lstat = Path.lstat

            def lstat(path, *args, **kwargs):
                # Model a filesystem timestamp collision, not a cached scan result.
                return directory_info if path == root else original_lstat(path, *args, **kwargs)

            with patch.object(Path, "lstat", new=lstat):
                self.assertFalse(adapter.contains_video())
                video = root / "Movie.mkv"
                video.touch()
                self.assertFalse(adapter.contains_video())
                video.write_bytes(b"video")
                self.assertTrue(adapter.contains_video())
                self.assertEqual([item.path for item in adapter.scan()], [video])

    def test_independent_operations_refresh_nonempty_names_with_same_directory_stat(self):
        for operation in ("contains_video", "scan", "directory_entries"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                other = root / "Other"
                other.mkdir()
                note = root / "readme.txt"
                note.write_text("not media")
                adapter = LocalFilesystemAdapter(root, cache_directory_names=True)
                infos = {path: path.lstat() for path in (root, other)}
                original_lstat = Path.lstat

                def lstat(path, *args, **kwargs):
                    return infos[path] if path in infos else original_lstat(path, *args, **kwargs)

                def assert_names(expected):
                    if operation == "contains_video":
                        self.assertEqual(adapter.contains_video(root), bool(expected))
                    elif operation == "scan":
                        self.assertEqual(
                            [item.path for item in adapter.scan(root, new_inspection=True)], expected,
                        )
                    else:
                        self.assertEqual(set(adapter.directory_entries(root)), {note, other, *expected})

                with patch.object(Path, "lstat", new=lstat):
                    assert_names([])
                    video = root / "Movie.mkv"
                    video.write_bytes(b"video")
                    assert_names([video])
                    renamed = root / "Renamed.mkv"
                    video.rename(renamed)
                    assert_names([renamed])
                    # A separate directory must not inherit the first one's names.
                    self.assertFalse(adapter.contains_video(other))
                    renamed.unlink()
                    assert_names([])
                    nested = other / "Nested.mkv"
                    nested.write_bytes(b"nested")
                    self.assertTrue(adapter.contains_video(other))
                    self.assertEqual([item.path for item in adapter.scan(other)], [nested])

    def test_scan_drops_removed_or_renamed_names_with_same_directory_stat(self):
        for mutation in ("remove", "rename", "replace"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                video = root / "Movie.mkv"
                video.write_bytes(b"video")
                adapter = LocalFilesystemAdapter(root, cache_directory_names=True)
                directory_info = root.lstat()
                original_lstat = Path.lstat

                def lstat(path, *args, **kwargs):
                    return directory_info if path == root else original_lstat(path, *args, **kwargs)

                with patch.object(Path, "lstat", new=lstat):
                    self.assertEqual([item.path for item in adapter.scan()], [video])
                    replacement = root / "Replacement.mkv"
                    if mutation == "rename":
                        video.rename(replacement)
                    else:
                        video.unlink()
                        if mutation == "replace":
                            replacement.write_bytes(b"replacement")
                    expected = [] if mutation == "remove" else [replacement]
                    self.assertEqual(
                        [item.path for item in adapter.scan(new_inspection=True)], expected,
                    )
                    self.assertEqual(adapter.contains_video(), bool(expected))

    def test_directory_name_scope_discards_names_after_normal_or_exceptional_exit(self):
        for exceptional in (False, True):
            with self.subTest(exceptional=exceptional), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                video = root / "Movie.mkv"
                video.write_bytes(b"video")
                adapter = LocalFilesystemAdapter(root, cache_directory_names=True)
                directory_info = root.lstat()
                original_lstat = Path.lstat

                def lstat(path, *args, **kwargs):
                    return directory_info if path == root else original_lstat(path, *args, **kwargs)

                with patch.object(Path, "lstat", new=lstat):
                    try:
                        with adapter.directory_name_scope():
                            self.assertEqual([item.path for item in adapter.scan()], [video])
                            if exceptional:
                                raise RuntimeError("abort discovery")
                    except RuntimeError as exc:
                        self.assertTrue(exceptional)
                        self.assertEqual(str(exc), "abort discovery")
                    replacement = root / "Replacement.mkv"
                    video.rename(replacement)
                    self.assertEqual([item.path for item in adapter.scan()], [replacement])
                    with adapter.directory_name_scope():
                        self.assertEqual([item.path for item in adapter.scan()], [replacement])

    def test_nested_directory_name_scope_reuses_names_but_reads_live_sizes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "Movie.mkv"
            video.touch()
            adapter = LocalFilesystemAdapter(root, cache_directory_names=True)
            with patch("os.scandir", wraps=os.scandir) as scandir:
                with adapter.directory_name_scope():
                    self.assertFalse(adapter.contains_video())
                    with adapter.directory_name_scope():
                        video.write_bytes(b"video")
                        self.assertTrue(adapter.contains_video())
                    self.assertEqual([item.path for item in adapter.scan()], [video])
                scandir.assert_called_once()
                self.assertEqual(Path(scandir.call_args.args[0]), root)

    def test_disabled_directory_name_cache_stays_fresh_inside_explicit_scope(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            adapter = LocalFilesystemAdapter(root)
            with adapter.directory_name_scope():
                self.assertFalse(adapter.contains_video())
                video = root / "Movie.mkv"
                video.write_bytes(b"video")
                self.assertTrue(adapter.contains_video())

    def test_snapshot_detects_size_mtime_or_inode_change(self):
        with tempfile.TemporaryDirectory() as root_raw:
            root = Path(root_raw)
            video = root / "Movie.mkv"
            video.write_bytes(b"before")
            adapter = LocalFilesystemAdapter(root)
            snapshot = adapter.snapshot(video)
            replacement = root / "replacement"
            replacement.write_bytes(b"changed-content")
            os.replace(replacement, video)
            with self.assertRaises(LocalContentChanged):
                adapter.verify_snapshot(snapshot)

    def test_scan_filters_temporary_zero_and_symlink_but_keeps_companions(self):
        with tempfile.TemporaryDirectory() as root_raw, tempfile.TemporaryDirectory() as outside_raw:
            root = Path(root_raw)
            (root / "Show.S01E01.mkv").write_bytes(b"video")
            (root / "Show.S01E01.zh.ass").write_text("subtitle")
            (root / "poster.jpg").write_bytes(b"image")
            (root / "readme.txt").write_text("not media")
            (root / "archive.zip").write_bytes(b"archive")
            (root / "empty.mkv").touch()
            (root / "downloading.mkv.!qB").write_bytes(b"partial")
            outside = Path(outside_raw) / "outside.mkv"
            outside.write_bytes(b"outside")
            try:
                (root / "linked.mkv").symlink_to(outside)
            except (OSError, NotImplementedError):
                pass
            snapshots = LocalFilesystemAdapter(root).scan()
            self.assertEqual(
                [(item.relative_path, item.role) for item in snapshots],
                [
                    ("poster.jpg", "image"),
                    ("Show.S01E01.mkv", "video"),
                    ("Show.S01E01.zh.ass", "subtitle"),
                ],
            )
            self.assertEqual(snapshot_digest(snapshots), snapshot_digest(list(reversed(snapshots))))

    def test_scan_rejects_partial_directory_walk_instead_of_silent_success(self):
        with tempfile.TemporaryDirectory() as root_raw:
            root = Path(root_raw)
            (root / "Movie.mkv").write_bytes(b"video")
            unreadable = root / "Season 02"
            unreadable.mkdir()
            (unreadable / "Episode.mkv").write_bytes(b"episode")
            original_scandir = os.scandir

            def scandir(path):
                if Path(path) == unreadable:
                    raise PermissionError("unreadable directory")
                return original_scandir(path)

            with patch("app.modules.local_storage.os.scandir", side_effect=scandir):
                for include_non_media in (False, True):
                    with self.subTest(include_non_media=include_non_media):
                        with self.assertRaises(LocalStorageError):
                            LocalFilesystemAdapter(root).scan(include_non_media=include_non_media)
            self.assertEqual((root / "Movie.mkv").read_bytes(), b"video")
            self.assertEqual((unreadable / "Episode.mkv").read_bytes(), b"episode")

    def test_scan_can_include_non_media_for_single_pass_cleanup_planning(self):
        with tempfile.TemporaryDirectory() as root_raw:
            root = Path(root_raw)
            (root / "Movie.mkv").write_bytes(b"video")
            (root / "下载必看.txt").write_bytes(b"ad")
            (root / "fonts.zip").write_bytes(b"font")
            (root / "empty.tmp").touch()

            snapshots = LocalFilesystemAdapter(root).scan(
                include_non_media=True,
            )

            self.assertEqual(
                [(item.path.name, item.role, item.size) for item in snapshots],
                [
                    ("empty.tmp", "other", 0),
                    ("fonts.zip", "other", 4),
                    ("Movie.mkv", "video", 5),
                    ("下载必看.txt", "other", 2),
                ],
            )

    def test_contains_video_filters_non_media_only_directories(self):
        with tempfile.TemporaryDirectory() as root_raw:
            root = Path(root_raw)
            non_media = root / "Documents"
            nested_media = root / "Show" / "Season 01"
            non_media.mkdir()
            nested_media.mkdir(parents=True)
            (non_media / "readme.txt").write_text("notes")
            (non_media / "archive.zip").write_bytes(b"archive")
            (nested_media / "Show.S01E01.mkv").write_bytes(b"video")

            adapter = LocalFilesystemAdapter(root)

            self.assertFalse(adapter.contains_video(non_media))
            self.assertTrue(adapter.contains_video(nested_media.parent))
            self.assertFalse(adapter.contains_video(non_media / "readme.txt"))
            self.assertTrue(adapter.contains_video(nested_media / "Show.S01E01.mkv"))

    def test_single_video_scan_includes_only_its_uniquely_matched_subtitles(self):
        with tempfile.TemporaryDirectory() as root_raw:
            root = Path(root_raw)
            episode_one = root / "Show.S01E01.mkv"
            episode_two = root / "Show.S01E02.mkv"
            episode_one.write_bytes(b"episode-one")
            episode_two.write_bytes(b"episode-two")
            (root / "Show.S01E01.zh-Hans.ass").write_bytes(b"subtitle-one")
            (root / "Show.S01E02.en.srt").write_bytes(b"subtitle-two")
            (root / "unmatched.srt").write_bytes(b"unmatched")
            (root / "poster.jpg").write_bytes(b"poster")

            snapshots = LocalFilesystemAdapter(root).scan(episode_one)

            self.assertEqual(
                [(item.path.name, item.role) for item in snapshots],
                [
                    ("Show.S01E01.mkv", "video"),
                    ("Show.S01E01.zh-Hans.ass", "subtitle"),
                ],
            )

    def test_contains_video_distinguishes_missing_and_unreadable_paths(self):
        with tempfile.TemporaryDirectory() as root_raw:
            root = Path(root_raw)
            blocked = root / "blocked"
            blocked.mkdir()
            adapter = LocalFilesystemAdapter(root)

            with self.assertRaisesRegex(LocalContentChanged, "扫描路径不存在"):
                adapter.contains_video(root / "Movie.mkv")

            with patch("app.modules.local_storage.os.scandir", side_effect=PermissionError("denied")):
                with self.assertRaisesRegex(LocalStorageError, "目录暂时不可完整读取"):
                    adapter.contains_video(blocked)

    def test_scan_prunes_system_and_temporary_directories_by_exact_name(self):
        with tempfile.TemporaryDirectory() as root_raw:
            root = Path(root_raw)
            normal = root / "Tempest.Show"
            normal.mkdir()
            (normal / "Tempest.Show.S01E01.mkv").write_bytes(b"video")
            for name in ("@eaDir", "temp", "TMP", ".mediaflux-trash", "#recycle"):
                ignored = root / name
                ignored.mkdir()
                (ignored / "Ignored.mkv").write_bytes(b"ignored")

            snapshots = LocalFilesystemAdapter(root).scan()

            self.assertEqual(
                [item.relative_path for item in snapshots],
                ["Tempest.Show/Tempest.Show.S01E01.mkv"],
            )
            self.assertEqual(LocalFilesystemAdapter(root).scan(root / "temp"), [])

    def test_item_and_depth_limits_are_enforced(self):
        with tempfile.TemporaryDirectory() as root_raw:
            root = Path(root_raw)
            for index in range(3):
                (root / f"{index}.mkv").write_bytes(b"x")
            with self.assertRaises(LocalScanLimitExceeded):
                LocalFilesystemAdapter(root, item_limit=2).scan()

            deep = root / "a" / "b" / "c"
            deep.mkdir(parents=True)
            (deep / "A.mkv").write_bytes(b"x")
            with self.assertRaises(LocalScanLimitExceeded):
                LocalFilesystemAdapter(root, depth_limit=1).scan()

    def test_same_filesystem_and_available_space(self):
        with tempfile.TemporaryDirectory() as root_raw:
            root = Path(root_raw)
            source = root / "A.mkv"
            source.write_bytes(b"x")
            self.assertTrue(LocalFilesystemAdapter.same_filesystem(source, root / "library" / "A.mkv"))
            self.assertGreater(LocalFilesystemAdapter.available_space(root / "library"), 0)


if __name__ == "__main__":
    unittest.main()
