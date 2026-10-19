"""本地媒体任务只读查询、公开摘要与确认后完成状态。"""

from __future__ import annotations

from typing import Any

from app import database as db
from app.agent.models import Evidence, ToolContext, ToolResult
from app.agent.public_safety import sanitize_public_text
from app.modules.local_media_models import (
    LOCAL_BUSY_TASK_STATUSES,
    LOCAL_TASK_STATUSES,
    LocalMediaTask,
)
from app.modules.local_media_outcomes import local_media_task_outcome

from . import common


def _safe_title(value: Any) -> str:
    text = sanitize_public_text(value, limit=120)
    if not text or "/" in text or "\\" in text:
        return "未识别媒体"
    return text


def _status_reason(status: str) -> str:
    return {
        "requires_manual": "manual_match_required",
        "failed": "processing_failed",
        "waiting_stable": "waiting_to_run",
        "recognizing": "recognizing",
        "planned": "plan_ready",
        "moving": "moving_files",
        "verifying": "verifying_files",
        "refreshing": "refreshing_library",
        "rolling_back": "rolling_back",
        "completed": "processing_completed",
    }.get(status, "unknown")


def _outcome_public(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "original_filename": sanitize_public_text(
            outcome.get("original_filename"), limit=255
        ),
        "file_names": [
            name
            for value in (outcome.get("file_names") or [])[:20]
            if (name := sanitize_public_text(value, limit=255))
        ],
        "file_names_truncated": bool(outcome.get("file_names_truncated")),
        "file_outcome": outcome.get("file_outcome")
        if outcome.get("file_outcome")
        in {
            "archived",
            "conflict_skipped",
            "partial",
            "unknown",
            "pending",
            "failed",
            "cancelled",
            "preview_only",
        }
        else "unknown",
        **{
            key: max(0, int(outcome.get(key) or 0))
            for key in (
                "video_count",
                "archived_video_count",
                "skipped_video_count",
                "unknown_video_count",
            )
        },
        **{
            key: sanitize_public_text(outcome.get(key), limit=40)
            for key in ("created_at", "updated_at", "completed_at")
        },
    }


def _task_public(
    task: Any, number: int, *, outcome: dict[str, Any] | None = None
) -> dict[str, Any]:
    if outcome is None:
        outcome = local_media_task_outcome(
            task, db.list_local_media_task_items(int(task.id), owner=common._WORKSPACE_OWNER)
        )
    return {
        **_outcome_public(outcome),
        "task_number": number,
        "status": str(task.status),
        "reason_code": _status_reason(str(task.status)),
        "title": _safe_title(task.title),
        "media_type": str(task.media_type)
        if str(task.media_type) in {"movie", "tv"}
        else "unknown",
        "season": int(task.season_override)
        if task.season_override is not None
        else None,
        "episode": int(task.episode_override)
        if task.episode_override is not None
        else None,
        "trigger": str(task.trigger)
        if str(task.trigger) in {"qb_completed", "scan", "manual"}
        else "unknown",
        "attempts": max(0, int(task.attempts)),
        "can_inspect": str(task.status) == "requires_manual",
        "can_retry": str(task.status) in common._RETRYABLE,
        "can_refresh_library": str(task.status) == "completed",
        "can_verify_library": str(task.status) == "completed",
    }


def _scope_statuses(scope: str) -> set[str]:
    if scope == "attention":
        return set(common._RETRYABLE)
    if scope == "active":
        return set(LOCAL_BUSY_TASK_STATUSES)
    if scope == "history":
        return {"completed", "failed"}
    if scope == "skipped":
        return {"completed"}
    return {scope} if scope in LOCAL_TASK_STATUSES else set()


def _scope_tasks(scope: str, limit: int, offset: int = 0) -> list[Any]:
    # 过滤和更新时间排序必须先于 LIMIT，不能先截断最近创建的历史任务。
    sql = "SELECT t.* FROM local_media_tasks t WHERE t.owner=?"
    params: list[Any] = [common._WORKSPACE_OWNER]
    statuses = sorted(_scope_statuses(scope))
    if statuses:
        sql += " AND t.status IN (" + ",".join("?" for _ in statuses) + ")"
        params.extend(statuses)
    if scope == "skipped":
        sql += (
            " AND EXISTS (SELECT 1 FROM local_media_task_items i WHERE i.task_id=t.id"
            " AND i.owner=t.owner AND i.role='video' AND i.action='skip')"
            " AND t.warning NOT LIKE '%仅预览模式：未移动文件%'"
        )
    sql += " ORDER BY COALESCE(NULLIF(t.updated_at,''),t.created_at) DESC,t.id DESC LIMIT ? OFFSET ?"
    params.extend((limit, offset))
    with db.get_conn() as conn:
        return [
            LocalMediaTask.from_row(row) for row in conn.execute(sql, params).fetchall()
        ]


def _scan_tasks(
    scan: dict[str, Any], scope: str, limit: int, offset: int = 0
) -> tuple[list[Any], dict[int, dict[str, Any]], int, int]:
    membership = list(dict.fromkeys(int(value) for value in scan.get("task_ids", [])))
    found: dict[int, Any] = {}
    with db.get_conn() as conn:
        for batch_offset in range(0, len(membership), 500):
            batch = membership[batch_offset : batch_offset + 500]
            sql = (
                "SELECT * FROM local_media_tasks WHERE owner=? AND id IN ("
                + ",".join("?" for _ in batch)
                + ")"
            )
            for row in conn.execute(sql, [common._WORKSPACE_OWNER, *batch]).fetchall():
                task = LocalMediaTask.from_row(row)
                found[task.id] = task
    summary = scan.get("summary") or {}
    snapshots = {
        (item["task_id"]): item
        for item in summary.get("task_outcomes", [])
        if isinstance(item, dict)
        and isinstance(item.get("task_id"), int)
        and item["task_id"] in found
    }
    statuses = _scope_statuses(scope)
    tasks = []
    for task_id in membership:
        task = found.get(task_id)
        if task is None:
            continue
        snapshot = snapshots.get(task_id)
        status = snapshot.get("status", task.status) if snapshot else task.status
        if statuses and status not in statuses:
            continue
        if scope == "skipped":
            outcome = snapshot or local_media_task_outcome(
                task, db.list_local_media_task_items(task.id, owner=common._WORKSPACE_OWNER)
            )
            if not outcome.get("skipped_video_count"):
                continue
        tasks.append(task)
    return (
        tasks[offset : offset + limit],
        snapshots,
        len(membership) - len(found),
        len(tasks),
    )


def local_media_completion_status(
    completion: dict[str, Any], _context: ToolContext
) -> ToolResult:
    """读取确认后重试任务的安全终态；内部任务 ID 不进入公开结果。"""
    task_id = int(completion.get("task_id") or 0)
    task_number = int(completion.get("task_number") or 0)
    if task_id <= 0 or task_number <= 0:
        return ToolResult(False, "unknown", "本地媒体任务状态暂时无法确认")
    task = db.get_local_media_task(task_id, owner=common._WORKSPACE_OWNER)
    if task is None:
        return ToolResult(False, "unknown", "本地媒体任务状态暂时无法确认")
    public_task = _task_public(task, task_number)
    raw_status = str(task.status or "")
    if raw_status in LOCAL_BUSY_TASK_STATUSES:
        status, ok, summary = (
            "running",
            True,
            f"本地媒体任务 {task_number} 正在处理",
        )
    elif raw_status == "completed":
        status = "completed"
        outcome = public_task["file_outcome"]
        archived = public_task["archived_video_count"]
        skipped = public_task["skipped_video_count"]
        unknown = public_task["unknown_video_count"]
        if outcome == "preview_only":
            ok, summary = True, f"本地媒体任务 {task_number} 预览已完成，未移动文件，归档 0 个视频"
        else:
            ok = outcome in {"archived", "conflict_skipped", "partial"} and unknown == 0
            summary = f"本地媒体任务 {task_number} 已结束：归档 {archived} 个视频，冲突跳过 {skipped} 个视频"
            if not ok:
                summary += f"，{unknown} 个视频结果待核验" if unknown else "，缺少可核验的视频整理记录"
    elif raw_status == "requires_manual":
        status, ok, summary = (
            "manual_review",
            False,
            f"本地媒体任务 {task_number} 仍需人工确认",
        )
    elif raw_status == "failed":
        status, ok, summary = (
            "failed",
            False,
            f"本地媒体任务 {task_number} 整理失败",
        )
    else:
        status, ok, summary = (
            "unknown",
            False,
            f"本地媒体任务 {task_number} 状态暂时无法确认",
        )
    public_task["source_status"] = public_task["status"]
    public_task["status"] = status
    return ToolResult(
        ok,
        status,
        summary,
        data={"task": public_task},
        evidence=[
            Evidence(
                "sqlite:local_media_tasks",
                "按确认时冻结的内部任务身份读取当前状态；不返回路径、哈希或数据库 ID。",
                common._now(),
            )
        ],
        suggestions=(
            ["请在本地媒体待确认页继续处理。"]
            if status == "manual_review"
            else ["可查看本地媒体任务详情后再决定是否重试。"]
            if status == "failed" or (status == "completed" and not ok)
            else []
        ),
        error=(
            "本地媒体任务未成功完成。" if status in {"manual_review", "failed"}
            else "视频归档结果尚无法核验。" if status == "completed" and not ok else ""
        ),
    )
