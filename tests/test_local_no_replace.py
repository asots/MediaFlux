"""回收恢复与整理事务必须共用无覆盖发布及失败收尾合同。"""

from __future__ import annotations

import errno
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.modules.local_move_transaction import LocalMoveTransaction
from app.modules.local_storage import LocalStorageError, move_entry_no_replace_at


class LocalNoReplaceTests(unittest.TestCase):
    @contextmanager
    def _case(self, entrypoint):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, target = root / "source.mkv", root / "target.mkv"
            source.write_bytes(b"original-media")
            fd = os.open(root, os.O_RDONLY)
            try:

                def publish():
                    if entrypoint == "transaction":
                        return LocalMoveTransaction._publish_no_replace(source, target)
                    return move_entry_no_replace_at(
                        source.name,
                        target.name,
                        source_dir_fd=fd,
                        target_dir_fd=fd,
                        is_directory=False,
                    )

                with patch("ctypes.CDLL", return_value=SimpleNamespace(renameat2=None)):
                    yield source, target, publish
            finally:
                os.close(fd)

    def test_fallback_moves_bytes_and_never_overwrites_existing_target(self):
        for entrypoint in ("storage", "transaction"):
            with (
                self.subTest(entrypoint=entrypoint),
                self._case(entrypoint) as (source, target, publish),
            ):
                target.write_bytes(b"existing-media")
                with self.assertRaises(FileExistsError):
                    publish()
                self.assertEqual(source.read_bytes(), b"original-media")
                self.assertEqual(target.read_bytes(), b"existing-media")
                target.unlink()
                publish()
                self.assertFalse(source.exists())
                self.assertEqual(target.read_bytes(), b"original-media")

    def test_unlink_failure_removes_own_alias_and_allows_retry(self):
        for entrypoint in ("storage", "transaction"):
            with (
                self.subTest(entrypoint=entrypoint),
                self._case(entrypoint) as (source, target, publish),
            ):
                real_unlink = os.unlink

                def denied(path, **kwargs):
                    if Path(path).name == source.name:
                        raise PermissionError(errno.EACCES, "source busy")
                    return real_unlink(path, **kwargs)

                with (
                    patch("os.unlink", side_effect=denied),
                    self.assertRaises(PermissionError),
                ):
                    publish()
                self.assertEqual(source.read_bytes(), b"original-media")
                self.assertFalse(
                    target.exists(), "failed publish must not obstruct retry"
                )
                publish()
                self.assertEqual(target.read_bytes(), b"original-media")
                self.assertFalse(source.exists())

    def test_concurrent_source_removal_never_deletes_last_media_copy(self):
        for entrypoint in ("storage", "transaction"):
            with (
                self.subTest(entrypoint=entrypoint),
                self._case(entrypoint) as (source, target, publish),
            ):
                real_unlink = os.unlink

                def removed_by_another_actor(path, **kwargs):
                    if Path(path).name == source.name:
                        # 另一进程先删除源目录项；本次 unlink 随后真实得到 ENOENT。
                        real_unlink(path, **kwargs)
                    return real_unlink(path, **kwargs)

                with (
                    patch("os.unlink", side_effect=removed_by_another_actor),
                    self.assertRaises(FileNotFoundError),
                ):
                    publish()
                self.assertFalse(source.exists())
                self.assertTrue(
                    target.exists(),
                    "do not delete the last link after source disappears",
                )
                self.assertEqual(target.read_bytes(), b"original-media")

    def test_changed_source_keeps_original_target_and_external_replacement(self):
        for entrypoint in ("storage", "transaction"):
            with (
                self.subTest(entrypoint=entrypoint),
                self._case(entrypoint) as (source, target, publish),
            ):
                real_unlink = os.unlink

                def source_replaced(path, **kwargs):
                    if Path(path).name == source.name:
                        real_unlink(path, **kwargs)
                        source.write_bytes(b"new-download")
                        raise PermissionError(errno.EACCES, "replacement busy")
                    return real_unlink(path, **kwargs)

                with (
                    patch("os.unlink", side_effect=source_replaced),
                    self.assertRaises(PermissionError),
                ):
                    publish()
                self.assertEqual(source.read_bytes(), b"new-download")
                self.assertTrue(target.exists())
                self.assertEqual(target.read_bytes(), b"original-media")

    def test_changed_target_is_not_removed_during_compensation(self):
        for entrypoint in ("storage", "transaction"):
            with (
                self.subTest(entrypoint=entrypoint),
                self._case(entrypoint) as (source, target, publish),
            ):
                real_unlink = os.unlink

                def target_replaced(path, **kwargs):
                    if Path(path).name == source.name:
                        real_unlink(target)
                        target.write_bytes(b"external-target")
                        raise PermissionError(errno.EACCES, "source busy")
                    return real_unlink(path, **kwargs)

                with (
                    patch("os.unlink", side_effect=target_replaced),
                    self.assertRaises(PermissionError),
                ):
                    publish()
                self.assertEqual(source.read_bytes(), b"original-media")
                self.assertEqual(target.read_bytes(), b"external-target")

    def test_directory_fallback_refuses_without_mutating_entries(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "source").mkdir()
            (root / "source" / "Movie.mkv").write_bytes(b"media")
            fd = os.open(root, os.O_RDONLY)
            try:
                with patch("ctypes.CDLL", return_value=SimpleNamespace(renameat2=None)):
                    with self.assertRaises(LocalStorageError):
                        move_entry_no_replace_at(
                            "source",
                            "target",
                            source_dir_fd=fd,
                            target_dir_fd=fd,
                            is_directory=True,
                        )
                self.assertEqual((root / "source" / "Movie.mkv").read_bytes(), b"media")
                self.assertFalse((root / "target").exists())
            finally:
                os.close(fd)
