"""持有单一会话上下文的本地媒体任务确认动作。"""

from __future__ import annotations

import secrets
from typing import Any

from app import database as db
from app.agent.errors import AgentToolError
from app.agent.models import Evidence, ToolContext, ToolResult
from app.agent.public_safety import sanitize_public_text
from app.agent.session_context import AgentSessionContextRepository
from app.clients.base import close_media_server_client
from app.modules import local_media_scan_runs
from app.modules.local_media_models import LOCAL_TASK_STATUSES
from app.modules.local_media_scheduler import get_local_media_scheduler
from app.modules.local_media_service import (
    LocalMediaServiceError,
    get_local_media_service,
)
from app.modules.local_storage import LocalStorageError

from . import common, task_library, task_queries
from .task_context import LocalMediaAgentContextStore, _InspectionRef, _TaskRef


def _require_owner(context: ToolContext) -> str:
    owner = (context.owner or "").strip()
    if not owner or not context.session_id.strip():
        raise AgentToolError(
            "本地媒体任务操作需要已登录会话", code="precondition_failed"
        )
    return owner


def _task_source_digest(task: Any) -> str:
    digest = str(task.snapshot_digest or "")
    return digest.split(":")[1] if digest.startswith("preview:") else digest


def _episode_position(season: int | None, episode: int | None) -> str:
    if season is None or episode is None:
        return "未明确"
    return f"S{season:02d}E{episode:02d}"


def _retry_confirmation_snapshot(
    snapshot: dict[str, Any],
    task: Any,
    *,
    season: int | None,
    episode: int | None,
) -> dict[str, Any]:
    if season is None or episode is None:
        if task.trigger != "manual" or not task.confirmation_actor or snapshot["interrupted_write"]:
            return snapshot
        # TG 确认后的手动任务普通重试同样重新安装票据，不依赖清空后的 actor。
        season, episode = task.season_override, task.episode_override
    else:
        _validate_episode_remap(task, snapshot, season=season, episode=episode)
    service = get_local_media_service()
    try:
        inspection = service.inspect_source(common._WORKSPACE_OWNER, task.source_id, task.content_path)
        if inspection["digest"] != _task_source_digest(task):
            raise AgentToolError("源文件范围已变化，请重新检查并确认整理预览", code="precondition_failed")
        preview = service.preview(
            common._WORKSPACE_OWNER, inspection["inspection_id"], task.tmdb_id, task.media_type,
            rules_snapshot=task.rules_snapshot, season_override=season, episode_override=episode,
            numbering_mode=task.numbering_mode,
        )
    except AgentToolError:
        raise
    except (LocalMediaServiceError, LocalStorageError, ValueError, OSError):
        raise AgentToolError("无法生成重试预览，请重新检查任务", code="precondition_failed") from None
    if preview.get("status") != "planned" or preview.get("pending_confirmations"):
        raise AgentToolError("重试计划仍需人工确认，请重新检查任务", code="precondition_failed")
    return {
        **snapshot,
        "preview_digest": preview["preview_digest"],
        "rules_snapshot": preview["rules_snapshot"],
        "media_type": str(task.media_type or ""),
        "tmdb_id": str(task.tmdb_id or ""),
        "current_season": task.season_override,
        "current_episode": task.episode_override,
        "target_season": season,
        "target_episode": episode,
    }


def _validate_episode_remap(
    task: Any, snapshot: dict[str, Any], *, season: int, episode: int,
) -> None:
    if task.status != "requires_manual":
        raise AgentToolError(
            "只有待人工确认的任务可以修正季集映射", code="precondition_failed"
        )
    if str(task.media_type or "").strip().lower() != "tv":
        raise AgentToolError("只有剧集任务可以修正季集映射", code="precondition_failed")
    tmdb_id = str(task.tmdb_id or "").strip()
    if not tmdb_id.isascii() or not tmdb_id.isdigit():
        raise AgentToolError(
            "任务缺少已确认的 TMDB 剧集身份，请先完成媒体匹配",
            code="precondition_failed",
        )
    if snapshot["interrupted_write"]:
        raise AgentToolError(
            "任务曾在文件写入期间中断，不能仅靠季集纠偏继续；请先核验文件和 qB 状态",
            code="precondition_failed",
        )
    if (task.season_override, task.episode_override) == (season, episode):
        raise AgentToolError(
            f"任务季集映射已经是 {_episode_position(season, episode)}",
            code="precondition_failed",
        )


class LocalMediaTaskActions:
    """由单个 runtime 持有的领域动作与短期上下文。"""

    def __init__(self, repository: AgentSessionContextRepository | None = None) -> None:
        self._context_store = LocalMediaAgentContextStore(repository=repository)


    def clear_local_media_agent_context(
        self,
        *, owner: str, session_id: str
    ) -> None:
        self._context_store.clear_session(owner=owner, session_id=session_id)

    def list_local_media_task_summaries(
        self,
        arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        owner = _require_owner(context)
        scope = str(arguments["scope"])
        offset = int(arguments.get("offset", 0))
        limit = int(arguments["limit"])
        matched_total: int | None = None
        scan: dict[str, Any] | None = None
        snapshots: dict[int, dict[str, Any]] = {}
        missing = 0
        if scope == "latest_scan" or arguments.get("scan_ref"):
            try:
                scan = local_media_scan_runs.resolve_local_media_scan(
                    str(arguments.get("scan_ref") or ""), owner=common._WORKSPACE_OWNER
                )
            except ValueError as exc:
                raise AgentToolError("本地扫描引用无效") from exc
            except LookupError:
                self._context_store.capture_tasks(owner=owner, session_id=context.session_id, tasks=[])
                return ToolResult(
                    True,
                    "not_recorded",
                    "未记录到对应本地扫描的任务明细，不能以其他历史任务替代此次通知。",
                    data={
                        "scope": scope,
                        "scan_ref": str(arguments.get("scan_ref") or ""),
                        "scan_recorded": False,
                        "total": 0,
                        "offset": offset,
                        "limit": limit,
                        "has_more": False,
                        "next_offset": None,
                        "matched_total": 0,
                        "tasks": [],
                    },
                )
            tasks, snapshots, missing, matched_total = task_queries._scan_tasks(
                scan, scope, limit, offset
            )
            has_more = offset + len(tasks) < matched_total
        else:
            page = task_queries._scope_tasks(scope, limit + 1, offset)
            has_more = len(page) > limit
            tasks = page[:limit]
        self._context_store.capture_tasks(owner=owner, session_id=context.session_id, tasks=tasks)
        items = []
        for index, task in enumerate(tasks, start=1):
            snapshot = snapshots.get(int(task.id))
            item = task_queries._task_public(task, index, outcome=snapshot)
            if scan:
                item["outcome_basis"] = (
                    "scan_snapshot" if snapshot else "current_task_state"
                )
                if snapshot and snapshot.get("status") in LOCAL_TASK_STATUSES:
                    item["current_status"] = item["status"]
                    item["status"] = snapshot["status"]
                    item["reason_code"] = task_queries._status_reason(item["status"])
            items.append(item)
        attention = sum(item["status"] in common._RETRYABLE for item in items)
        scan_data = (
            {
                "scan_ref": scan.get("scan_ref"),
                "scan_started_at": scan.get("started_at", ""),
                "scan_finished_at": scan.get("finished_at", ""),
                "scan_reported_at": str(
                    (scan.get("summary") or {}).get("reported_at") or ""
                ),
                "scan_recorded": True,
                "scan_task_count": len(scan.get("task_ids", [])),
                "missing_task_count": missing,
                "missing_tasks_excluded": bool(missing),
            }
            if scan
            else {}
        )
        return ToolResult(
            True,
            "attention" if attention or missing else "completed",
            f"已列出本页 {len(items)} 个本地媒体任务" if items else "本页没有本地媒体任务",
            data={
                "scope": scope,
                "total": len(items),
                "total_kind": "page_count",
                "offset": offset,
                "limit": limit,
                "has_more": has_more,
                "next_offset": offset + len(items) if has_more else None,
                **({"matched_total": matched_total} if matched_total is not None else {}),
                "attention": attention,
                "tasks": items,
                **scan_data,
                "task_number_kind": "session_scoped_ordinal_not_database_id",
                "pagination_note": "total 仅为本页条数；翻页请保留 scan_ref/scope 并使用 next_offset。每次列表重建当前会话的短期任务序号，后续动作使用本会话最新一页的 task_number。",
                "outcome_note": "任务完成不等于文件已归档；冲突跳过必须按 skipped_video_count/file_outcome 解释。扫描快照是当次事实，可用动作基于任务当前状态。",
                "expires_in_seconds": self._context_store.ttl_seconds,
            },
            evidence=[
                Evidence(
                    "sqlite:local_media_tasks",
                    "返回本地任务的短期公开序号、安全原文件名、时间与文件级结果；没有数据库 ID、目录路径、哈希或错误正文。指定扫描时仅查询该次扫描成员，不以其他历史任务补位。",
                    common._now(),
                )
            ],
            suggestions=["可继续说：检查本地媒体任务 1，或重试本地媒体任务 1。"]
            if items
            else [],
        )

    def _current_task(self, owner: str, session_id: str, task_number: int) -> tuple[_TaskRef, Any]:
        ref = self._context_store.task(owner=owner, session_id=session_id, number=task_number)
        if ref is None:
            raise AgentToolError(
                "任务序号不存在或已过期，请先重新列出本地媒体任务",
                code="precondition_failed",
            )
        task = db.get_local_media_task(ref.task_id, owner=common._WORKSPACE_OWNER)
        if task is None or (
            (task.version) != ref.version
            or str(task.status) != ref.status
            or (task.source_id) != ref.source_id
        ):
            raise AgentToolError(
                "本地媒体任务状态已变化，请重新列出任务", code="precondition_failed"
            )
        return ref, task

    def inspect_local_media_task(
        self,
        arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        owner = _require_owner(context)
        task_number = int(arguments["task_number"])
        ref, task = self._current_task(owner, context.session_id, task_number)
        if task.status != "requires_manual":
            raise AgentToolError("只有待人工确认的任务可以检查", code="precondition_failed")
        try:
            raw = get_local_media_service().inspect_task(common._WORKSPACE_OWNER, task.id)
        except (LocalMediaServiceError, ValueError, OSError):
            raise AgentToolError(
                "暂时无法检查该任务，请确认源文件仍可访问", code="precondition_failed"
            ) from None
        files = raw.get("files") if isinstance(raw.get("files"), list) else []
        roles = {"video": 0, "subtitle": 0, "other": 0}
        names: list[str] = []
        for item in files[:50]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "other")
            roles[role if role in roles else "other"] += 1
            name = sanitize_public_text(item.get("name"), limit=120)
            if name and "/" not in name and "\\" not in name and len(names) < 8:
                names.append(name)
        inspection_number = self._context_store.capture_inspection(
            owner=owner,
            session_id=context.session_id,
            task=ref,
            inspection_id=str(raw.get("inspection_id") or ""),
            digest=str(raw.get("digest") or ""),
        )
        if not inspection_number:
            raise AgentToolError("无法创建安全检查句柄，请重试", code="unavailable")
        return ToolResult(
            True,
            "completed",
            f"本地媒体任务 {task_number} 的只读检查已完成",
            data={
                "task_number": task_number,
                "inspection_number": inspection_number,
                "title": task_queries._safe_title(raw.get("task_title") or task.title),
                "year": sanitize_public_text(raw.get("task_year"), limit=4),
                "media_type": str(
                    raw.get("task_media_type") or task.media_type or "unknown"
                ),
                "file_count": len(files),
                "video_count": roles["video"],
                "subtitle_count": roles["subtitle"],
                "file_name_sample": names,
                "suggested_query": task_queries._safe_title(raw.get("suggested_query")),
                "parsed_season": raw.get("parsed_season")
                if isinstance(raw.get("parsed_season"), int)
                else None,
                "parsed_episode": raw.get("parsed_episode")
                if isinstance(raw.get("parsed_episode"), int)
                else None,
                "cloud_write": False,
                "expires_in_seconds": self._context_store.ttl_seconds,
            },
            evidence=[
                Evidence(
                    "local_media_inspection",
                    "在已配置来源边界内只读扫描任务文件并生成当前登录与会话绑定的短期检查序号；未返回绝对/相对路径、源摘要、任务错误或内部 inspection ID。",
                    common._now(),
                )
            ],
            suggestions=[
                f"如需查看整理匹配预览，请说：预览本地媒体检查 {inspection_number}。"
            ],
        )

    def _current_inspection(self, owner: str, session_id: str, number: int) -> tuple[_InspectionRef, Any]:
        ref = self._context_store.inspection(owner=owner, session_id=session_id, number=number)
        if ref is None:
            raise AgentToolError(
                "检查序号不存在或已过期，请重新检查任务", code="precondition_failed"
            )
        task = db.get_local_media_task(ref.task.task_id, owner=common._WORKSPACE_OWNER)
        if task is None or (
            (task.version) != ref.task.version
            or str(task.status) != ref.task.status
            or (
                bool(_task_source_digest(task))
                and _task_source_digest(task) != (ref.digest or "")
            )
        ):
            raise AgentToolError(
                "任务或源文件状态已变化，请重新检查", code="precondition_failed"
            )
        return ref, task

    def preview_local_media_task(
        self,
        arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        owner = _require_owner(context)
        number = int(arguments["inspection_number"])
        ref, task = self._current_inspection(owner, context.session_id, number)
        service = get_local_media_service()

        def preview(inspection_id: str) -> dict[str, Any]:
            return service.preview(
                common._WORKSPACE_OWNER,
                inspection_id,
                tmdb_id=str(task.tmdb_id or ""),
                media_type=str(task.media_type or ""),
                automatic=False,
                season_override=task.season_override,
                episode_override=task.episode_override,
                numbering_mode=task.numbering_mode,
            )

        try:
            raw = preview(ref.inspection_id)
        except LocalMediaServiceError:
            # inspection 的完整文件快照只保存在执行 Worker 内存中。跨 Worker 或
            # 进程重启后按任务边界重新只读检查，并用持久化摘要证明仍是同一快照。
            try:
                rebuilt = service.inspect_task(common._WORKSPACE_OWNER, task.id)
                rebuilt_digest = str(rebuilt.get("digest") or "")
                if not rebuilt_digest or not secrets.compare_digest(
                    rebuilt_digest, (ref.digest or "")
                ):
                    raise AgentToolError(
                        "源文件在检查后发生变化，请重新检查任务",
                        code="precondition_failed",
                    )
                rebuilt_id = str(rebuilt.get("inspection_id") or "")
                if not rebuilt_id:
                    raise AgentToolError(
                        "整理预览已失效，请重新检查任务", code="precondition_failed"
                    )
                raw = preview(rebuilt_id)
                self._context_store.replace_inspection(
                    owner=owner,
                    session_id=context.session_id,
                    number=number,
                    inspection_id=rebuilt_id,
                    digest=rebuilt_digest,
                )
            except AgentToolError:
                raise
            except (LocalMediaServiceError, ValueError, OSError):
                raise AgentToolError(
                    "整理预览已失效或条件不足，请重新检查任务",
                    code="precondition_failed",
                ) from None
        except (ValueError, OSError):
            raise AgentToolError(
                "整理预览已失效或条件不足，请重新检查任务",
                code="precondition_failed",
            ) from None
        candidates: list[dict[str, Any]] = []
        for item in (
            raw.get("candidates") if isinstance(raw.get("candidates"), list) else []
        )[:5]:
            if not isinstance(item, dict):
                continue
            candidates.append(
                {
                    "title": task_queries._safe_title(item.get("title")),
                    "year": sanitize_public_text(item.get("year"), limit=4),
                    "media_type": str(item.get("media_type") or "unknown"),
                    "confidence": str(item.get("confidence") or "unknown"),
                }
            )
        plans: list[dict[str, Any]] = []
        for item in (raw.get("plans") if isinstance(raw.get("plans"), list) else [])[:12]:
            if not isinstance(item, dict):
                continue
            plans.append(
                {
                    "role": str(item.get("role") or "other"),
                    "action": str(item.get("action") or "move"),
                    "target_name": sanitize_public_text(item.get("target_name"), limit=140),
                }
            )
        status = str(raw.get("status") or "inconclusive")
        return ToolResult(
            status == "planned",
            "completed" if status == "planned" else "attention",
            "整理匹配预览已生成" if status == "planned" else "整理匹配仍需要人工选择",
            data={
                "inspection_number": number,
                "preview_status": status
                if status in {"planned", "requires_manual"}
                else "inconclusive",
                "candidate_count": len(candidates),
                "candidates": candidates,
                "plan_count": len(plans),
                "plans": plans,
                "cloud_write": False,
            },
            evidence=[
                Evidence(
                    "local_media_preview",
                    "重新核验文件摘要后只生成整理匹配和文件动作预览；未移动、覆盖、删除文件，也未返回路径、TMDB ID、规则快照或内部检查句柄。",
                    common._now(),
                )
            ],
            suggestions=["当前仅为只读预览；实际整理仍需使用既有人工确认流程。"],
            error="匹配结果尚不足以自动整理。" if status != "planned" else "",
        )

    def _task_snapshot(self, owner: str, session_id: str, task_number: int) -> tuple[dict[str, Any], Any]:
        _ref, task = self._current_task(owner, session_id, task_number)
        return {
            "task_id": int(task.id),
            "version": int(task.version),
            "status": str(task.status),
            "source_id": int(task.source_id),
            "snapshot_digest": str(task.snapshot_digest or ""),
            "updated_at": str(task.updated_at or ""),
            "interrupted_write": db.is_interrupted_local_media_write_error(task.error),
        }, task

    def prepare_retry_local_media_task(
        self,
        arguments: dict[str, Any], context: ToolContext
    ) -> tuple[ToolResult, str]:
        owner = _require_owner(context)
        task_number = int(arguments["task_number"])
        snapshot, task = self._task_snapshot(owner, context.session_id, task_number)
        if task.status not in common._RETRYABLE:
            raise AgentToolError(
                "只有失败或待人工确认的任务可以重试", code="precondition_failed"
            )
        season = int(arguments["season"]) if "season" in arguments else None
        episode = int(arguments["episode"]) if "episode" in arguments else None
        if season is not None and episode is not None:
            current_position = _episode_position(
                task.season_override, task.episode_override
            )
            target_position = _episode_position(season, episode)
            summary = (
                f"确认后将把本地媒体任务 {task_number} 从 {current_position} "
                f"修正为 {target_position} 并重新整理"
            )
            data = {
                "task_number": task_number,
                "current_status": str(task.status),
                "title": task_queries._safe_title(task.title),
                "current_position": current_position,
                "target_position": target_position,
                "interrupted_write": False,
                "effects": [
                    f"把本任务的明确季集映射改为 {target_position}。",
                    "任务会使用现有 TMDB 身份、整理规则、冲突策略和来源运行模式重新排队。",
                    "只纠偏已确认的源文件；确认时复核源快照与计划，变化后须重新预检，本次预检不修改文件。",
                    "后续结果可在本地媒体任务状态查看；任务启用通知且非静默时，按现有配置发送完成或失败通知。",
                ],
            }
            evidence_description = (
                "已核验任务仍处于待确认状态、具备剧集 TMDB 身份且未发生写入中断；"
                "未返回路径、哈希或内部任务 ID。"
            )
        else:
            summary = f"确认后将重试本地媒体任务 {task_number}"
            data = {
                "task_number": task_number,
                "current_status": str(task.status),
                "title": task_queries._safe_title(task.title),
                "interrupted_write": snapshot["interrupted_write"],
                "effects": [
                    *([
                        (
                            "上次任务在文件写入期间中断：请先核验文件及 qB 状态；"
                            "继续确认表示已完成核验，允许清除旧计划后重新处理。"
                        ),
                    ] if snapshot["interrupted_write"] else []),
                    "任务会回到等待执行阶段，并由调度器立即重新检查和处理。",
                    "会生成新的操作幂等标识；不会复用上一次中断的文件步骤。",
                    "本次确认不会直接移动、覆盖或删除媒体文件。",
                ],
            }
            evidence_description = (
                "已核验任务当前版本和可重试状态；未返回路径、哈希、错误正文或操作标识。"
            )
        confirmation_snapshot = _retry_confirmation_snapshot(
            snapshot, task, season=season, episode=episode,
        )
        return ToolResult(
            True,
            "confirmation_required",
            summary,
            data=data,
            evidence=[
                Evidence("sqlite:local_media_tasks", evidence_description, common._now())
            ],
        ), common._fingerprint(confirmation_snapshot)

    def retry_local_media_task_confirmed(
        self,
        arguments: dict[str, Any], expected_context: str, context: ToolContext
    ) -> ToolResult:
        owner = _require_owner(context)
        task_number = int(arguments["task_number"])
        season = int(arguments["season"]) if "season" in arguments else None
        episode = int(arguments["episode"]) if "episode" in arguments else None
        try:
            snapshot, task = self._task_snapshot(owner, context.session_id, task_number)
            confirmation_snapshot = _retry_confirmation_snapshot(
                snapshot, task, season=season, episode=episode,
            )
        except AgentToolError as exc:
            raise AgentToolError(exc.safe_message, code="confirmation_stale") from None
        if not secrets.compare_digest(
            common._fingerprint(confirmation_snapshot), (expected_context or "")
        ):
            raise AgentToolError("任务状态已变化，请重新预检", code="confirmation_stale")
        retry_fields: dict[str, Any] = {
            "expected_version": task.version,
            "expected_status": task.status,
            "confirm_interrupted_write": snapshot["interrupted_write"],
        }
        if season is not None and episode is not None:
            retry_fields.update(
                season_override=season,
                episode_override=episode,
                confirm_interrupted_write=False,
            )
        if "preview_digest" in confirmation_snapshot:
            retry_fields.update(
                preview_digest=confirmation_snapshot["preview_digest"],
                rules_snapshot=confirmation_snapshot["rules_snapshot"],
            )
        if not db.reset_local_media_task_if_current(
            task.id, owner=common._WORKSPACE_OWNER, **retry_fields,
        ):
            raise AgentToolError("任务状态已变化，请重新预检", code="confirmation_stale")
        get_local_media_scheduler().reload()
        remapped = season is not None and episode is not None
        target_position = _episode_position(season, episode) if remapped else ""
        data = {
            "operation": "remap_episode" if remapped else "retry",
            "task_number": task_number,
            **({"season": season, "episode": episode} if remapped else {}),
            "affected": 1,
            "runtime_refreshed": True,
        }
        return ToolResult(
            True,
            "accepted",
            (
                f"本地媒体任务 {task_number} 已修正为 {target_position} 并重新排队"
                if remapped
                else f"本地媒体任务 {task_number} 已重新排队"
            ),
            data=data,
            evidence=[
                Evidence(
                    "sqlite:local_media_tasks",
                    (
                        "使用一次性确认票据和任务版本条件原子写入明确季集映射，并通过统一重试事务重新排队。"
                        if remapped
                        else "使用一次性确认票据、任务版本和原子状态条件重新排队，并生成新的内部操作标识。"
                    ),
                    common._now(),
                )
            ],
            suggestions=[
                "任务已按当前来源模式重新排队；可稍后查看本地媒体任务状态，通知是否发送以现有配置为准。"
                if remapped
                else "可稍后重新列出本地媒体任务查看进度。"
            ],
            effect_metadata={
                "completion": {
                    "kind": "local_media_task",
                    "task_id": int(task.id),
                    "task_number": task_number,
                    "operation": "remap_episode" if remapped else "retry",
                }
            },
        )

    def _refresh_snapshot(
        self,
        owner: str, session_id: str, task_number: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        base, task = self._task_snapshot(owner, session_id, task_number)
        scope = task_library._bound_task_scope(task)
        return {**base, "binding": scope["binding"]}, scope

    def prepare_refresh_local_media_task_library(
        self,
        arguments: dict[str, Any], context: ToolContext
    ) -> tuple[ToolResult, str]:
        owner = _require_owner(context)
        task_number = int(arguments["task_number"])
        snapshot, scope = self._refresh_snapshot(owner, context.session_id, task_number)
        try:
            return ToolResult(
                True,
                "confirmation_required",
                f"确认后将精准刷新本地媒体任务 {task_number} 的绑定媒体库",
                data={
                    "task_number": task_number,
                    "server": scope["server_label"],
                    "library": sanitize_public_text(
                        scope["binding"]["library_name"], limit=80
                    ),
                    "path_count": len(scope["paths"]),
                    "effects": [
                        "只刷新任务已绑定且重新校验通过的一个媒体服务器和媒体库。",
                        "不会接受或使用用户提供的 URL、路径、服务器 ID 或媒体库内部 ID。",
                        "无法唯一定位时会安全停止，绝不退化为全库刷新。",
                    ],
                },
                evidence=[
                    Evidence(
                        "media_server_binding",
                        "已从任务和服务端配置重新解析唯一绑定媒体库；内部路径和媒体库 ID 仅参与确认指纹。",
                        common._now(),
                    )
                ],
            ), common._fingerprint(snapshot)
        finally:
            close_media_server_client(scope["client"])

    def refresh_local_media_task_library_confirmed(
        self,
        arguments: dict[str, Any], expected_context: str, context: ToolContext
    ) -> ToolResult:
        owner = _require_owner(context)
        task_number = int(arguments["task_number"])
        try:
            snapshot, scope = self._refresh_snapshot(owner, context.session_id, task_number)
        except AgentToolError as exc:
            raise AgentToolError(exc.safe_message, code="confirmation_stale") from None
        try:
            if not secrets.compare_digest(
                common._fingerprint(snapshot), (expected_context or "")
            ):
                raise AgentToolError(
                    "任务或媒体库绑定已变化，请重新预检", code="confirmation_stale"
                )
            lease_key = common._fingerprint({
                        "operation": "local_media.precise_refresh",
                        "task_id": int(scope["task"].id),
                        "task_version": int(scope["task"].version),
                        "server_config_digest": scope["binding"]["server_config_digest"],
                        "library_id": scope["binding"]["library_id"],
                        "paths_digest": scope["binding"]["paths_digest"],
                    })
            if db.claim_agent_action_lease(lease_key, ttl_seconds=90) is None:
                return ToolResult(
                    False,
                    "conflict",
                    "相同的绑定媒体库刷新请求刚刚已提交",
                    data={
                        "operation": "precise_refresh",
                        "task_number": task_number,
                        "refreshed": 0,
                    },
                    error="为避免重复刷新，请稍后核验媒体库可见状态。",
                    suggestions=["可直接核验该任务是否已在绑定媒体库中可见。"],
                )
            try:
                outcome = scope["client"].refresh_for_paths(
                    scope["paths"],
                    allowed_library_ids=(scope["binding"]["library_id"],),
                    allow_library_fallback=False,
                )
            except Exception:
                outcome = None
            if (
                not isinstance(outcome, dict)
                or not outcome.get("ok")
                or outcome.get("skipped")
                or outcome.get("scope") != "item"
            ):
                return ToolResult(
                    False,
                    "unavailable",
                    "绑定媒体库未完成精准刷新",
                    data={
                        "operation": "precise_refresh",
                        "task_number": task_number,
                        "refreshed": 0,
                    },
                    error="媒体服务器未接受精准刷新请求。",
                )
            return ToolResult(
                True,
                "completed",
                "绑定媒体库已完成精准刷新请求",
                data={
                    "operation": "precise_refresh",
                    "task_number": task_number,
                    "refreshed": 1,
                    "matched_paths": max(0, int(outcome.get("matched") or 0)),
                },
                evidence=[
                    Evidence(
                        "media_server_refresh",
                        "仅向重新校验通过的绑定媒体库提交受限路径刷新；未执行全库刷新。",
                        common._now(),
                    )
                ],
                suggestions=["刷新请求不等于媒体已可见，可继续核验该任务的入库可见状态。"],
            )
        finally:
            close_media_server_client(scope["client"])

    def verify_local_media_task_library_visibility(
        self,
        arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        owner = _require_owner(context)
        task_number = int(arguments["task_number"])
        _snapshot, scope = self._refresh_snapshot(owner, context.session_id, task_number)
        try:
            task = scope["task"]
            tmdb_id = str(task.tmdb_id or "").strip()
            media_type = str(task.media_type or "").strip().lower()
            index_status = "inconclusive"
            reason_code = "identity_unavailable"
            if tmdb_id.isascii() and tmdb_id.isdigit() and media_type in {"movie", "tv"}:
                try:
                    if media_type == "movie":
                        visible = scope["client"].has_tmdb_media(
                            tmdb_id,
                            "movie",
                            parent_id=scope["binding"]["library_id"],
                        )
                        index_status = "visible" if visible else "missing"
                        reason_code = "movie_indexed" if visible else "movie_not_indexed"
                    else:
                        search = scope["client"].find_series_candidates_by_tmdb(
                            tmdb_id,
                            limit=20,
                            parent_id=scope["binding"]["library_id"],
                        )
                        candidates = list(search.candidates)
                        if len(candidates) == 1 and not search.truncated:
                            season = task.season_override
                            episode = task.episode_override
                            if season is not None and episode is not None:
                                inventory = scope["client"].list_series_episode_inventory(
                                    candidates[0].id,
                                    include_specials=int(season) == 0,
                                )
                                visible = (int(season), int(episode)) in set(
                                    inventory.episodes
                                )
                                index_status = "visible" if visible else "missing"
                                reason_code = (
                                    "episode_indexed" if visible else "episode_not_indexed"
                                )
                            else:
                                index_status = "inconclusive"
                                reason_code = "series_indexed_episode_unverified"
                        elif not candidates and not search.truncated:
                            index_status = "missing"
                            reason_code = "series_not_indexed"
                        else:
                            reason_code = "series_mapping_ambiguous"
                except Exception:
                    index_status = "inconclusive"
                    reason_code = "media_server_unavailable"
            summary = {
                "visible": "媒体已在绑定媒体库中可见",
                "missing": "绑定媒体库中尚未看到该媒体",
                "inconclusive": "暂时无法确认媒体库可见状态",
            }[index_status]
            return ToolResult(
                index_status == "visible",
                index_status,
                summary,
                data={
                    "task_number": task_number,
                    "title": task_queries._safe_title(task.title),
                    "index_status": index_status,
                    "reason_code": reason_code,
                    "server": scope["server_label"],
                    "library": sanitize_public_text(
                        scope["binding"]["library_name"], limit=80
                    ),
                    "playback_status": "not_checked",
                    "playback_claim": "not_probed",
                },
                evidence=[
                    Evidence(
                        "bound_media_library",
                        "按任务保存的媒体身份在唯一绑定媒体库中查询索引；未读取播放历史，也未发起真实播放探测。",
                        common._now(),
                    )
                ],
                suggestions=[
                    "“库中可见”仅表示媒体服务器已经索引；本次没有证明文件可实际解码或播放。"
                ],
                error="媒体库尚未可见或本次核验不确定。"
                if index_status != "visible"
                else "",
            )
        finally:
            close_media_server_client(scope["client"])
