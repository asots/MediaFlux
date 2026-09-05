"""STRM 迁移凭据恢复与历史副本校准；复用 STRM 的所有权和变化交接逻辑。"""

from __future__ import annotations

import hashlib
import os
import time
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path

from app import database as db

# 只读小型指针文件，不把伪装成 .strm 的媒体文件加载进内存。
MAX_POINTER_BYTES = 16 * 1024


def _blocked(stats: dict, path: object, message: str) -> None:
    from app.modules import strm

    stats["clean_skipped"] = True
    stats["recovery_pending"] = int(stats.get("recovery_pending", 0)) + 1
    strm._append_error_sample(stats, "恢复 STRM 清理", path, RuntimeError(message))


def _stopped(stats: dict, should_stop: Callable[[], bool] | None) -> bool:
    if stats.get("stopped") or (should_stop and should_stop()):
        stats.update(stopped=True, stop_stage="recovery-cleanup", clean_skipped=True)
        return True
    return False


def _managed_path(value: object, strm_root: str) -> Path | None:
    """限定光鸭输出子树；不把索引或扫描时出现的符号链接解析成删除目标。"""
    raw = Path(str(value or "")).expanduser()
    if not str(value or "").strip():
        return None
    if not raw.is_absolute():
        raw = Path(strm_root).expanduser() / raw
    try:
        from app.modules.strm import STRM_SUBDIR

        resolved = raw.resolve(strict=False)
        resolved.relative_to(
            (Path(strm_root).expanduser() / STRM_SUBDIR).resolve(strict=False)
        )
        if raw.absolute() != resolved or raw.is_symlink():
            return None
        return resolved
    except (OSError, ValueError, RuntimeError):
        return None


def recover_pending_paths(
    source_key: str,
    strm_root: str,
    stats: dict,
    *,
    valid_ids: set[str],
    only_file_ids: set[str] | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_refresh_paths: Callable[[list[str]], object] | None = None,
) -> None:
    """完整扫描或可信增量成功后消费旧路径凭据，绝不删除仍被索引引用的路径。

    valid_ids 是本轮确认仍有效的 ID；精准增量必须同时给 only_file_ids，
    避免把未核查的其他 ID 当成远端删除。当前新文件存在且指纹正确才删旧
    副本；已确认远端删除的 ID 可以直接使用原索引指纹清理。
    """
    from app.modules import strm

    current = None
    reported_paths = set(stats.get("changed_strm_paths") or [])
    after_id = 0
    while not _stopped(stats, should_stop):
        batch = db.list_strm_path_cleanup(source_key, after_id=after_id)
        if not batch:
            return
        after_id = int(batch[-1]["id"])
        completed: list[int] = []
        changed_dirs: set[str] = set()
        for row in batch:
            if _stopped(stats, should_stop):
                break
            file_id = str(row["file_id"])
            if only_file_ids is not None and file_id not in only_file_ids:
                continue
            path = _managed_path(row["strm_path"], strm_root)
            if path is None:
                _blocked(
                    stats, row["strm_path"], "旧路径不在当前 STRM 根内，已保留清理凭据"
                )
                continue
            try:
                if db.list_strm_path_owners(str(path)):
                    # 普通异常已回滚，或路径已被其他活跃对象复用；凭据不再适用。
                    completed.append(int(row["id"]))
                    continue
                if path.exists() and file_id in valid_ids:
                    if current is None:
                        current = {
                            str(item["file_id"]): item
                            for item in db.list_strm_index(source_key)
                        }
                    active = current.get(file_id)
                    replacement = (
                        _managed_path(active["strm_path"], strm_root)
                        if active
                        else None
                    )
                    if not replacement or not strm._fingerprint_matches(
                        replacement, active["content_fingerprint"]
                    ):
                        _blocked(
                            stats, path, "新文件尚未验证，已保留旧 STRM 和清理凭据"
                        )
                        continue
                if path.is_symlink() or (path.exists() and not path.is_file()):
                    _blocked(stats, path, "旧路径已变为链接或非普通文件，需人工核对")
                    continue
                deleted = strm._delete_owned_file(path, [row], "恢复旧 STRM 清理")
                stats[
                    "metadata_cleaned"
                    if source_key.startswith("guangya-meta:")
                    else "cleaned"
                ] += int(deleted)
                # 已删但未确认的凭据同样重放路径变化，防止中断后媒体库漏刷新。
                if deleted or str(path) not in reported_paths:
                    strm._track_change(
                        stats,
                        "removed",
                        path,
                        strm_root,
                        on_refresh_paths=on_refresh_paths,
                    )
                    if len(reported_paths) < strm._MAX_TRACKED_CHANGED_PATHS:
                        reported_paths.add(str(path))
                changed_dirs.add(str(path.parent))
                if _stopped(stats, should_stop):
                    break
                completed.append(int(row["id"]))
            except (OSError, RuntimeError, ValueError) as exc:
                _blocked(stats, path, str(exc))
        # 生产调用使用已固定刷新策略的 sink；持久化失败时不确认凭据。
        if completed:
            if changed_dirs and on_refresh_paths:
                try:
                    on_refresh_paths(sorted(changed_dirs))
                except Exception as exc:
                    _blocked(
                        stats,
                        source_key,
                        f"旧目录刷新交接失败（{type(exc).__name__}），保留清理凭据",
                    )
                    stats.update(stopped=True, stop_stage="refresh-persist")
                    return
            db.delete_strm_path_cleanup(completed)


def _pointer(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(MAX_POINTER_BYTES + 1)


def _local_pointers(roots: list[Path], stats: dict, should_stop) -> Iterator[Path]:
    """每轮只遍历一次所属目录；条目、目录数、时间有界，不跟随符号链接。"""
    from app.modules import strm

    max_dirs, max_entries, _candidates, seconds = strm._scan_limits()
    deadline = time.monotonic() + seconds
    pending = deque(roots)
    visited: set[Path] = set()
    entries = 0
    while pending and not _stopped(stats, should_stop):
        directory = pending.popleft()
        if directory in visited:
            continue
        if directory.is_symlink():
            continue
        if len(visited) >= max_dirs or time.monotonic() >= deadline:
            _blocked(
                stats, directory, "历史 STRM 校准达到目录或时间上限，未检查项已保留"
            )
            return
        visited.add(directory)
        if not directory.exists():
            continue
        try:
            with os.scandir(directory) as children:
                for entry in children:
                    if _stopped(stats, should_stop):
                        return
                    if entries >= max_entries or time.monotonic() >= deadline:
                        _blocked(
                            stats,
                            directory,
                            "历史 STRM 校准达到条目或时间上限，未检查项已保留",
                        )
                        return
                    entries += 1
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if len(pending) + len(visited) >= max_dirs:
                            _blocked(
                                stats,
                                directory,
                                "历史 STRM 校准达到目录上限，未检查项已保留",
                            )
                            return
                        pending.append(Path(entry.path))
                    elif entry.is_file(
                        follow_symlinks=False
                    ) and entry.name.lower().endswith(".strm"):
                        yield Path(entry.path)
        except OSError as exc:
            _blocked(
                stats, directory, f"历史 STRM 目录读取失败（{type(exc).__name__}）"
            )


def reconcile_historical_strm(
    strm_root: str,
    base_url: str,
    sources: list[dict[str, str]],
    stats: dict,
    *,
    all_sources: bool = False,
    should_stop: Callable[[], bool] | None = None,
    on_refresh_paths: Callable[[list[str]], object] | None = None,
) -> None:
    """只在完整扫描和索引清理均成功后校准，绝不用于快速同步。

    旧副本必须与本轮有效索引、实际安装内容、当前签名播放 URL 三者精确
    一致。其他来源的索引路径、手工内容、第三方指针和伴随文件一律保留。
    """
    from app.modules import strm

    if stats.get("clean_skipped") or _stopped(stats, should_stop):
        return
    root = Path(strm_root).expanduser() / strm.STRM_SUBDIR
    if not sources or not root.is_dir() or root.is_symlink():
        return
    source_keys = {f"guangya:{source['id']}" for source in sources}
    rows = db.list_strm_index_by_prefix("")
    indexed_paths: set[Path] = set()
    replacements: dict[bytes, list[tuple[Path, str]]] = {}
    for row in rows:
        if _stopped(stats, should_stop):
            return
        path = strm._safe_indexed_path(row["strm_path"], strm_root)
        if not path:
            continue
        indexed_paths.add(path)
        if _managed_path(row["strm_path"], strm_root) is None:
            continue
        if str(row["source"]) not in source_keys or path.suffix.lower() != ".strm":
            continue
        try:
            url = strm.build_play_url(
                base_url, row["file_id"], row["etag"], row["size"], row["filename"]
            )
            payload = url.encode("utf-8")
            fingerprint = f"sha256:{hashlib.sha256(payload).hexdigest()}"
            if (
                len(payload) <= MAX_POINTER_BYTES
                and row["content_fingerprint"] == fingerprint
            ):
                # 这里只整理索引与签名 URL 的候选关系；遇到实际旧副本时
                # 才在删除边界校验新文件，避免每次全量校准重读所有有效指针。
                replacements.setdefault(payload, []).append((path, fingerprint))
        except (OSError, ValueError, TypeError):
            continue
    if all_sources:
        roots = [root]
    else:
        roots = list(
            dict.fromkeys(
                root / strm.safe_path_component(source["rel_prefix"])
                if source.get("rel_prefix", "").strip()
                else root
                for source in sources
            )
        )
    for candidate in _local_pointers(roots, stats, should_stop):
        try:
            path = _managed_path(candidate, strm_root)
            if path is None:
                _blocked(stats, candidate, "历史 STRM 路径已变化或越界，已保留")
                continue
            if path in indexed_paths or candidate.is_symlink():
                continue
            payload = _pointer(path)
            matches = replacements.get(payload, [])
            if not matches:
                # 只提示形似本项目的失联指针；其他工具生成的指针不算同步失败。
                if b"/playgy/" in payload:
                    _blocked(
                        stats,
                        path,
                        "历史 STRM 无索引且无法验证有效新副本，已保留，请人工核对",
                    )
                continue
            # 删除前重查新副本及全部来源路径所有者，避免扫描期间外部改写。
            if not any(
                strm._fingerprint_matches(new, fingerprint)
                for new, fingerprint in matches
            ):
                _blocked(stats, path, "历史 STRM 的有效新副本已变化，已保留旧文件")
                continue
            if db.list_strm_path_owners(str(path)):
                continue
            owner = {"content_fingerprint": matches[0][1]}
            if _stopped(stats, should_stop):
                return
            if strm._delete_owned_file(path, [owner], "清理历史 STRM 副本"):
                stats["cleaned"] += 1
                strm._track_change(
                    stats, "removed", path, strm_root, on_refresh_paths=on_refresh_paths
                )
        except (OSError, RuntimeError, ValueError) as exc:
            _blocked(stats, candidate, str(exc))
