"""本地整理文件事实投影；任务结束并不等于本次归档了视频。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _field(value: Any, name: str, default: Any = "") -> Any:
    try:
        return value[name]
    except (KeyError, IndexError, TypeError):
        return getattr(value, name, default)


def _basename(value: Any) -> str:
    # 本地任务也可能来自 Windows 下载器；不把源目录或控制字符带到通知。
    text = str(value or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return " ".join(text.split())[:255]


def local_media_task_outcome(task: Any, items: Iterable[Any]) -> dict[str, Any]:
    """共享给通知和 Agent 的有限文件事实，不含 ID、路径、哈希或原始错误。"""
    videos = [item for item in items if _field(item, "role") == "video"]
    names = list(
        dict.fromkeys(_basename(_field(item, "source_path")) for item in videos)
    )
    status = str(_field(task, "status"))
    preview_only = "仅预览模式：未移动文件" in str(_field(task, "warning"))
    completed = status == "completed" and not preview_only
    archived = (
        sum(_field(item, "action") in {"move", "replace"} for item in videos)
        if completed
        else 0
    )
    skipped = (
        sum(_field(item, "action") == "skip" for item in videos) if completed else 0
    )
    unknown = len(videos) - archived - skipped
    if preview_only:
        outcome = "preview_only"
    elif status != "completed":
        outcome = "pending" if status not in {"failed", "cancelled"} else status
    elif not videos or unknown:
        outcome = "unknown" if not archived and not skipped else "partial"
    elif skipped == len(videos):
        outcome = "conflict_skipped"
    elif archived == len(videos):
        outcome = "archived"
    else:
        outcome = "partial"
    return {
        "original_filename": _basename(_field(task, "content_path")),
        "file_names": [name for name in names[:20] if name],
        "file_names_truncated": len(names) > 20,
        "file_outcome": outcome,
        "video_count": len(videos),
        "archived_video_count": archived,
        "skipped_video_count": skipped,
        "unknown_video_count": unknown,
        "created_at": str(_field(task, "created_at")),
        "updated_at": str(_field(task, "updated_at")),
        "completed_at": str(_field(task, "completed_at")),
    }
