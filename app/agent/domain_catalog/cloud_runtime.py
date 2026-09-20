"""光鸭整理运行状态的安全领域投影。"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from app.agent.models import Evidence, ToolContext, ToolResult
from app.agent.public_safety import sanitize_public_text

from .shared import _bounded_int, _now, _safe_choice, _safe_timestamp

_GY_OPERATION_REF_RE = re.compile(r"GY-(?:[0-9A-F]{4}-){7}[0-9A-F]{4}")
_GY_WAITABLE_STATUSES = {"accepted", "queued", "running"}
_GY_ACTIVE_STATUSES = {"queued", "running", "stopping"}
_GY_TERMINAL_STATUSES = {
    "completed", "partial", "failed", "cancelled", "manual_review", "stopped"
}
_GY_WAIT_TIMEOUT_SECONDS = 30 * 60
_GY_WAIT_INTERVAL_SECONDS = 1.0


def guangya_organize_status(
    arguments: dict[str, Any], context: ToolContext | None = None
) -> ToolResult:
    """读取光鸭整理任务、持久化操作与调度器的脱敏运行快照。"""
    context = context or ToolContext()
    from app.modules.organize_tasks import get_organize_manager

    manager = get_organize_manager()
    operation_ref = str(arguments.get("operation_ref") or "").strip().upper()
    overview = manager.status()
    raw = (
        manager.task_result(operation_ref, owner=context.owner)
        if operation_ref
        else overview
    )
    if operation_ref and raw is None:
        return ToolResult(
            ok=False,
            status="empty",
            summary="没有找到这个光鸭操作编号",
            data={"operation_ref": operation_ref, "found": False},
            evidence=[
                Evidence(
                    "guangya_organizer",
                    "已按公开操作编号查询持久化任务；未返回目录、内部任务标识或错误正文。",
                    _now(),
                )
            ],
            suggestions=["请核对操作编号，或直接查看当前光鸭整理状态。"],
        )
    raw = raw or {}
    task_status = _safe_choice(
        raw.get("status"),
        {
            "idle",
            "queued",
            "running",
            "stopping",
            "completed",
            "partial",
            "stopped",
            "failed",
            "cancelled",
            "manual_review",
        },
        "idle",
    )
    running = task_status in {"running", "stopping"}
    allowed_stats = {
        "total",
        "matched",
        "need_confirm",
        "moved",
        "renamed",
        "rename_failed",
        "metadata_moved",
        "stopped",
        "skipped",
        "conflict",
        "failed",
        "subtitle_moved",
        "subtitle_skipped",
        "replacement_cleanup_failed",
        "empty_dir_cleanup_failed",
        "source_dir_cleanup_failed",
        "audit_failures",
        "copied",
        "relocated",
        "created",
        "trashed",
        "strm_triggered",
        "strm_trigger_failed",
        "strm_scope_unknown",
        "strm_trigger_skipped",
        "quarantined",
        "empty_deleted",
        "verification_failed",
        "precondition_failed",
    }
    stats = (
        {
            key: _bounded_int(value)
            for key, value in (raw.get("stats") or {}).items()
            if key in allowed_stats
        }
        if isinstance(raw.get("stats"), dict)
        else {}
    )
    if not stats and isinstance(raw.get("result"), dict):
        persisted_stats = raw["result"].get("stats")
        if isinstance(persisted_stats, dict):
            stats = {
                key: _bounded_int(value)
                for key, value in persisted_stats.items()
                if key in allowed_stats
            }

    schedule_raw = (
        overview.get("schedule") if isinstance(overview.get("schedule"), dict) else {}
    )
    schedule = {
        "enabled": bool(schedule_raw.get("enabled")),
        "configured": not bool(schedule_raw.get("config_error")),
        "cron_valid": bool(schedule_raw.get("cron_valid")),
        "next_run": _safe_timestamp(schedule_raw.get("next_run")),
    }
    queue_raw = overview.get("operation_queue")
    queue_total = (
        _bounded_int(queue_raw.get("total")) if isinstance(queue_raw, dict) else 0
    )

    if running:
        ok, status, summary = True, "running", "光鸭整理任务正在运行"
        suggestions: list[str] = []
    elif task_status == "queued":
        ok, status, summary = True, "queued", "光鸭整理操作正在排队"
        suggestions = ["任务会在当前整理操作结束后自动执行。"]
    elif task_status == "manual_review":
        ok, status, summary = False, "attention", "光鸭操作在进程中断后需要人工核验"
        suggestions = ["请先核对光鸭目标目录，确认远端结果后再决定是否重新执行。"]
    elif task_status == "failed":
        ok, status, summary = False, "attention", "最近一次光鸭整理任务未成功"
        suggestions = ["请到网盘整理页查看任务详情后再决定是否重试。"]
    elif task_status == "completed":
        ok, status, summary = True, "completed", "最近一次光鸭整理任务已完成"
        suggestions = []
    elif task_status == "partial":
        ok, status, summary = False, "attention", "最近一次光鸭整理任务部分完成"
        suggestions = ["请到网盘整理页核对失败项后再决定是否重试。"]
    elif task_status in {"stopped", "cancelled"}:
        ok, status, summary = True, "stopped", "最近一次光鸭整理任务已停止"
        suggestions = []
    else:
        ok, status, summary = (
            True,
            "idle",
            (
                f"光鸭整理任务当前空闲，另有 {queue_total} 项操作排队"
                if queue_total
                else "光鸭整理任务当前空闲"
            ),
        )
        suggestions = []

    if stats.get("strm_scope_unknown"):
        suggestions.append("文件变更结果已记录，但同步范围未能确认，本次未触发 STRM 联动；请核对同步目录后手动同步。")

    task_data = {
        "status": task_status,
        "running": running,
        "stoppable": bool(raw.get("stoppable")) if running else False,
        "trigger_type": _safe_choice(
            raw.get("trigger_type"), {"manual", "cron", "telegram"}
        ),
        "started_at": _safe_timestamp(raw.get("started_at")),
        "finished_at": _safe_timestamp(raw.get("finished_at")),
        "stats": stats,
    }
    if operation_ref:
        task_data["operation_ref"] = operation_ref
    return ToolResult(
        ok=ok,
        status=status,
        summary=summary,
        data={
            "task": task_data,
            "queue": {"pending_count": queue_total},
            "schedule": schedule,
        },
        evidence=[
            Evidence(
                "guangya_organizer",
                "读取光鸭整理任务脱敏快照；仅在用户提供时返回公开操作编号，不返回目录、内部任务标识或错误正文。",
                _now(),
            )
        ],
        suggestions=suggestions,
    )


def _background_task_snapshot(snapshot: ToolResult) -> tuple[str, dict[str, Any]]:
    payload = snapshot.data if isinstance(snapshot.data, dict) else {}
    task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
    status = str(task.get("status") or snapshot.status or "").strip().casefold()
    return status, dict(task)


def _background_job_data(
    result: ToolResult,
    operation_ref: str,
    status: str,
    task: dict[str, Any],
    *,
    timed_out: bool = False,
) -> dict[str, Any]:
    data = dict(result.data) if isinstance(result.data, dict) else {}
    # execute 的公开范围沿用了 preview DTO；最终/未知状态不能继续声称未写云端。
    data.pop("cloud_write", None)
    data["operation_ref"] = operation_ref
    job = {
        "status": status,
        "last_status": status,
        "timed_out": timed_out,
    }
    for key in ("stats", "started_at", "finished_at"):
        value = task.get(key)
        if value not in (None, ""):
            job[key] = dict(value) if key == "stats" and isinstance(value, dict) else value
    if isinstance(task.get("stats"), dict):
        data["stats"] = dict(task["stats"])
    data["background_job"] = job
    return data


def _background_model_data(result: ToolResult, data: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(result.model_data, dict):
        return result.model_data
    model_data = dict(result.model_data)
    model_data.pop("cloud_write", None)
    for key in ("operation_ref", "stats", "background_job"):
        if key in data:
            model_data[key] = data[key]
    return model_data


async def wait_for_guangya_operation(
    result: ToolResult,
    *,
    tool: str,
    context: ToolContext,
    report_progress: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    timeout_seconds: float = _GY_WAIT_TIMEOUT_SECONDS,
) -> ToolResult:
    """等待 owner 绑定的 GY 持久任务；非 GY accepted 结果原样返回。"""
    data = result.data if isinstance(result.data, dict) else {}
    operation_ref = str(data.get("operation_ref") or "").strip().upper()
    status = str(result.status or "").strip().casefold()
    if (
        status not in _GY_WAITABLE_STATUSES
        or not _GY_OPERATION_REF_RE.fullmatch(operation_ref)
    ):
        return result

    async def report(snapshot: ToolResult, task_status: str) -> None:
        if report_progress is None:
            return
        payload = {
            "phase": "background_job",
            "operation_ref": operation_ref,
            "tool": str(tool or "")[:120],
            "status": task_status,
            "summary": sanitize_public_text(snapshot.summary, limit=240)
            or "光鸭后台任务状态已更新",
        }
        try:
            await report_progress(payload)
        except Exception:  # noqa: BLE001 - 进度通道故障不应改写业务终态
            # 进度通道不是业务终态；状态查询仍须继续完成。
            return

    def unknown(last_status: str, *, timed_out: bool = False) -> ToolResult:
        job_status = last_status or "unknown"
        data = _background_job_data(
            result, operation_ref, job_status, {}, timed_out=timed_out
        )
        return replace(
            result,
            ok=False,
            status="outcome_unknown",
            summary=(
                "光鸭后台任务仍在运行，等待已达上限，结果尚未确认"
                if timed_out and last_status in _GY_ACTIVE_STATUSES
                else "光鸭后台任务状态暂时未知，结果尚未确认"
            ),
            data=data,
            model_data=_background_model_data(result, data),
            error=result.error or "请稍后查询这个光鸭操作编号的状态",
        )

    terminal_summary = {
        "completed": "光鸭后台任务已完成",
        "partial": "光鸭后台任务部分完成",
        "failed": "光鸭后台任务执行失败",
        "cancelled": "光鸭后台任务已取消",
        "manual_review": "光鸭后台任务结果未知，需要人工核验",
        "stopped": "光鸭后台任务已停止",
    }
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, float(timeout_seconds))
    last_status = ""
    while True:
        if context.cancelled():
            raise asyncio.CancelledError
        try:
            snapshot = await asyncio.to_thread(
                guangya_organize_status,
                {"operation_ref": operation_ref},
                context=context,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - 查询异常只能安全降级为未知
            return unknown(last_status)
        task_status, task = _background_task_snapshot(snapshot)
        if task_status in _GY_TERMINAL_STATUSES:
            await report(snapshot, task_status)
            data = _background_job_data(result, operation_ref, task_status, task)
            return replace(
                result,
                ok=task_status in {"completed", "stopped"},
                status=task_status,
                summary=terminal_summary[task_status],
                data=data,
                model_data=_background_model_data(result, data),
                suggestions=list(dict.fromkeys([*result.suggestions, *snapshot.suggestions])),
                error=result.error
                or ("请核对任务统计和失败项" if task_status != "completed" else ""),
            )
        if task_status not in _GY_ACTIVE_STATUSES:
            return unknown(
                last_status or (
                    "unknown" if task_status in {"", "empty", "idle"} else task_status
                )
            )
        last_status = task_status
        await report(snapshot, task_status)
        remaining = deadline - loop.time()
        if remaining <= 0:
            return unknown(last_status, timed_out=True)
        await asyncio.sleep(min(_GY_WAIT_INTERVAL_SECONDS, remaining))
