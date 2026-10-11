"""Agent Kernel 的统一公开文本投影。

模型上下文可以保留结构化工具结果，但 Web/TG 历史只能消费这里生成的
紧凑公开摘要，避免把内部 JSON、引用和值域细节直接暴露给用户。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.agent.model_context_budget import decode_tool_content
from app.agent.public_safety import (
    public_tool_label,
    sanitize_public_text,
    sanitize_resource_title,
)

_CONFIRMED_RESULT_MARKER = "已确认操作的可信系统结果（不是待执行计划）："
_TARGET_LABELS = {
    "guangya": "光鸭云盘",
    "qb": "qBittorrent",
    "qbittorrent": "qBittorrent",
    "cloud": "云盘",
    "local": "本地",
}
_COUNT_FIELDS: tuple[tuple[str, str], ...] = (
    ("total", "请求"),
    ("succeeded", "已受理"),
    ("created", "已创建"),
    ("updated", "已更新"),
    ("review_required", "待复核"),
    ("duplicate", "已存在"),
    ("failed", "未完成"),
    ("skipped", "已跳过"),
)
_SUBMITTED_RESULT_STATUSES = frozenset({"accepted", "submitted"})
_PENDING_RESULT_STATUSES = frozenset(
    {"pending", "queued", "running", "in_progress", "retry_wait"}
)
_FS_CHANGE_ACTION_LABELS = {
    "rename": "改名",
    "move": "移动",
    "relocate": "清洗并移动",
    "copy": "复制",
    "trash": "移入回收站",
    "create_directory": "创建目录",
}
_FS_CHANGE_STATUS_LABELS = {
    "completed": "已完成",
    "partial": "部分完成",
    "failed": "失败",
    "blocked": "受阻",
    "not_started": "未开始",
    "unknown": "已发起但待核验，不建议盲目重试",
}
_FS_CHANGE_REASON_LABELS = {
    "precondition_failed": "前置条件未满足",
    "dependency_failed": "依赖项未完成",
    "write_rejected": "写入被拒绝",
    "verification_pending": "写后核验未完成",
    "write_outcome_unknown": "写入结果未知",
    "execution_error": "执行异常",
    "audit_unavailable": "审计记录不可用",
    "cancelled": "已取消",
}
_MAX_FS_CHANGE_OPERATION_ITEMS = 400
_MAX_PUBLIC_FS_CHANGE_ITEMS = 8
_WARNING_RESULT_STATUSES = frozenset(
    {
        "partial",
        "degraded",
        "incomplete",
        "attention",
        "inconclusive",
        "outcome_unknown",
        "manual_review",
        "stopped",
        "cancelled",
        "not_started",
        "unknown",
    }
)


def _safe(value: object, *, limit: int = 600) -> str:
    return sanitize_public_text(value, limit=limit)


def _int_value(value: object) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def public_result_state(result: Mapping[str, Any] | None) -> str:
    """返回入口共用的结果语义，避免各处各判一套状态。"""
    value = result or {}
    status = str(value.get("status") or "").strip().casefold()
    data = value.get("data")
    background = data.get("background_job") if isinstance(data, Mapping) else None
    background_status = (
        str(background.get("status") or "").strip().casefold()
        if isinstance(background, Mapping)
        else ""
    )
    statuses = {status, background_status} - {""}
    if statuses & _WARNING_RESULT_STATUSES:
        return "warning"
    if value.get("ok") is False or status in {"failed", "error"}:
        return "failed"
    if statuses & _PENDING_RESULT_STATUSES:
        return "pending"
    if statuses & _SUBMITTED_RESULT_STATUSES:
        return "submitted"
    return "success"


def _result_icon(result: Mapping[str, Any]) -> str:
    return {
        "submitted": "📤",
        "pending": "⏳",
        "warning": "⚠️",
        "failed": "❌",
        "success": "✅",
    }[public_result_state(result)]


def _failed_item_errors(data: Mapping[str, Any]) -> list[str]:
    items = data.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return []
    errors: list[str] = []
    for item in items:
        if not isinstance(item, Mapping) or item.get("ok") is not False:
            continue
        error = _safe(item.get("error"), limit=180)
        if error and error not in errors:
            errors.append(error)
        if len(errors) >= 3:
            break
    return errors


def candidate_result_lines(data: Mapping[str, Any]) -> list[str]:
    if data.get("source_type") != "resource_candidates" or not isinstance(data.get("items"), list):
        return []
    labels = {"submitted": "已提交", "duplicate": "已存在，未重复添加", "failed": "提交失败",
              "partial": "部分目标成功", "manual_review": "结果未知，请先核验"}
    lines = []
    for item in data["items"][:12]:
        if not isinstance(item, Mapping):
            continue
        position = item.get("position")
        prefix = f"#{position} · " if type(position) is int else ""
        title = sanitize_resource_title(item.get("title"), limit=160) or "资源"
        status = "duplicate" if item.get("duplicate") else str(item.get("status") or "")
        text = f"{prefix}{title}：{labels.get(status, '结果未知，请先核验')}"
        target = _TARGET_LABELS.get(str(item.get("target") or ""), "两个目标" if item.get("target") == "both" else "")
        if target:
            text += f" · {target}"
        request_id = item.get("request_id")
        if type(request_id) is int and request_id > 0:
            text += f" · 下载请求 #{request_id}"
        for key, label in (("succeeded", "已提交"), ("failed", "失败目标")):
            if isinstance(item.get(key), list) and item[key]:
                names = [_TARGET_LABELS[name] for name in item[key] if isinstance(name, str) and name in _TARGET_LABELS]
                if names:
                    text += f" · {label}：{'、'.join(names)}"
        lines.append(text)
    return lines


def _operation_item_status(item: Mapping[str, Any]) -> str:
    status = str(item.get("status") or "")
    raw_actions = item.get("completed_actions")
    operation = item.get("operation")
    relocate_partial_actions = (
        status == "partial"
        and operation == "relocate"
        and raw_actions in (["move"], ["rename"])
    )
    if relocate_partial_actions and raw_actions == ["move"]:
        detail = "已移动；改名未完成"
    elif relocate_partial_actions:
        detail = "已改名；移动未完成"
    else:
        detail = _FS_CHANGE_STATUS_LABELS.get(status, "结果未知")
    if status != "completed" and not relocate_partial_actions:
        if isinstance(raw_actions, list):
            actions = dict.fromkeys(
                action for action in raw_actions
                if isinstance(action, str) and action in _FS_CHANGE_ACTION_LABELS
            )
            action_text = "、".join(_FS_CHANGE_ACTION_LABELS[action] for action in actions)
            action_text = action_text or ("无" if not raw_actions else "未知")
        else:
            action_text = "未知"
        detail += f"；已确认动作：{action_text}"
    reason = _FS_CHANGE_REASON_LABELS.get(str(item.get("reason") or ""))
    return f"{detail}；原因：{reason}" if reason else detail


def _operation_items_lines(
    data: Mapping[str, Any], *, result_status: str = ""
) -> list[str] | None:
    """渲染通用文件变更回执；None 表示旧结果没有该 DTO。"""
    if "operation_items" not in data:
        return None
    raw_items = data.get("operation_items")
    if not isinstance(raw_items, list):
        return ["- 计划结果：未知（逐项结果格式无效，不能确认完成情况）"]
    stats = data.get("stats")
    total = _int_value(data.get("total"))
    if total is None and isinstance(stats, Mapping):
        total = _int_value(stats.get("total"))
    items = raw_items[:_MAX_FS_CHANGE_OPERATION_ITEMS]
    completed = sum(
        isinstance(item, Mapping) and item.get("status") == "completed"
        for item in items
    )
    count = len(raw_items)
    if total is None:
        summary = f"- 计划结果：已确认完成 {completed} 项；计划总数未知，整体状态未知。"
    elif count == total and count <= _MAX_FS_CHANGE_OPERATION_ITEMS:
        summary = f"- 计划结果：已完成 {completed}/总 {total} 项"
    elif count < total:
        summary = (
            f"- 计划结果：已确认完成 {completed}/总 {total} 项；另有 "
            f"{total - len(items)} 项结果未核对，状态未知。"
        )
    else:
        summary = (
            f"- 计划结果：已确认完成 {completed}/总 {total} 项；"
            "逐项结果数量与计划不一致，汇总未知。"
        )

    remaining = [
        item for item in items
        if not isinstance(item, Mapping) or item.get("status") != "completed"
    ]
    all_completed = total is not None and count == total and completed == total
    completed_items = [
        item for item in items
        if isinstance(item, Mapping) and item.get("status") == "completed"
    ]
    prioritized = remaining
    if result_status == "partial" and total is not None and total <= _MAX_PUBLIC_FS_CHANGE_ITEMS and not all_completed:
        prioritized = [*remaining, *completed_items]
    shown = prioritized[:_MAX_PUBLIC_FS_CHANGE_ITEMS]
    lines = [summary]
    for raw_item in shown:
        item = raw_item if isinstance(raw_item, Mapping) else {}
        position = item.get("position")
        position = f"#{position}" if type(position) is int and position > 0 else "#?"
        label = sanitize_resource_title(item.get("label"), limit=180) or "文件变更项"
        lines.append(f"- {position} · {label}：{_operation_item_status(item)}")

    omitted = len(prioritized) - len(shown) + max(0, len(raw_items) - len(items))
    if omitted:
        lines.append(f"- 另有 {omitted} 项剩余或未知结果未展示。")
    return lines


def format_public_result(
    value: Mapping[str, Any] | None,
    *,
    fallback: str = "操作已结束。",
) -> str:
    """把公开 ToolResult 压缩为适合 Web/TG 展示的 Markdown。"""
    result = dict(value or {})
    summary = (_safe(result.get("summary"), limit=700)
               or _safe(result.get("message"), limit=700)
               or _safe(fallback, limit=700) or "操作已结束。")
    state = public_result_state(result)
    lines = [f"{_result_icon(result)} {summary}"]
    if state == "submitted":
        lines.append("- 状态：请求已提交，后台任务尚未完成；可以继续查询进度。")
    elif state == "pending":
        lines.append("- 状态：后台任务尚未完成")
    data = result.get("data")
    if isinstance(data, Mapping):
        operation_item_lines = _operation_items_lines(
            data, result_status=str(result.get("status") or "")
        )
        target = _safe(data.get("target"), limit=40).lower()
        if target:
            lines.append(f"- 目标：{_TARGET_LABELS.get(target, target)}")
        for key, label in _COUNT_FIELDS:
            if key == "total" and operation_item_lines is not None:
                continue
            count = _int_value(data.get(key))
            if count is not None:
                lines.append(f"- {label}：{count} 项")
        operation_ref = str(data.get("operation_ref") or "").strip().upper()
        if re.fullmatch(r"GY-(?:[0-9A-F]{4}-){7}[0-9A-F]{4}", operation_ref):
            lines.append(f"- 操作编号：{operation_ref}")
        stats = data.get("stats")
        if isinstance(stats, Mapping):
            counts = [f"{label} {count} 项" for key, label in (("renamed", "改名"), ("moved", "移动"), ("relocated", "清洗并移动"), ("copied", "复制"), ("created", "创建"), ("trashed", "回收"), ("skipped", "跳过"), ("failed", "失败")) if (count := _int_value(stats.get(key))) is not None and count > 0]
            if counts:
                lines.append("- 变更统计：" + "；".join(counts))
            if stats.get("strm_scope_unknown"):
                lines.append("- 提示：同步范围未能确认，本次未触发 STRM 联动；请核对同步目录。")
        scope_note = _safe(data.get("scope_note"), limit=300)
        if scope_note:
            lines.append(f"- 执行范围：{scope_note}")
        if operation_item_lines is not None:
            lines.extend(operation_item_lines)
        else:
            lines.extend(f"- {line}" for line in candidate_result_lines(data))
            for error in _failed_item_errors(data):
                lines.append(f"- 失败原因：{error}")

    error = _safe(result.get("error"), limit=300)
    if error and error != summary and error not in "\n".join(lines):
        lines.append(f"- 说明：{error}")
    return "\n".join(lines)


def _file_change_analysis(
    content: object, result: Mapping[str, Any] | None,
    results: Sequence[tuple[str, Mapping[str, Any]]],
) -> str:
    """Preserve only exact, bounded file-change sentences proved by one DTO.

    This is not a general inference checker. Labels, summaries and model markers
    are never evidence; ambiguous multiple snapshots are deliberately unsupported.
    """
    snapshots = []
    for value in ([result] if result is not None else []) + [v for _, v in results]:
        data = value.get("data")
        if not isinstance(data, Mapping):
            continue
        task = data.get("task")
        dto = data if "operation_items" in data else task
        if not isinstance(dto, Mapping) or "operation_items" not in dto:
            continue
        if any(v.get("truncated") or v.get("has_more") for v in (value, data, dto)):
            return ""
        if dto not in snapshots:
            snapshots.append(dto)
    if len(snapshots) != 1:
        return ""
    dto = snapshots[0]
    rows = dto.get("operation_items")
    stats = dto.get("stats")
    total = dto.get("total", stats.get("total") if isinstance(stats, Mapping) else None)
    if ("total" in dto and isinstance(stats, Mapping) and "total" in stats
            and (type(stats["total"]) is not int or stats["total"] != total)):
        return ""
    if (type(total) is not int or not 0 < total <= _MAX_FS_CHANGE_OPERATION_ITEMS
            or not isinstance(rows, list) or len(rows) != total):
        return ""
    positions = set()
    for row in rows:
        if not isinstance(row, Mapping):
            return ""
        pos = row.get("position")
        if (type(pos) is not int or not 1 <= pos <= total or pos in positions
                or not isinstance(row.get("operation"), str)
                or row.get("operation") not in _FS_CHANGE_ACTION_LABELS
                or not isinstance(row.get("status"), str)
                or row.get("status") not in _FS_CHANGE_STATUS_LABELS):
            return ""
        positions.add(pos)
    completed = sum(row["status"] == "completed" for row in rows)
    allowed = {
        f"文件变更共{total}项，已完成{completed}项，仍有{total - completed}项未确认完成。",
        f"文件变更共{total}项，其中{completed}项已完成。",
    }
    for status, wording in (("failed", "失败"), ("blocked", "受阻"),
                            ("not_started", "未开始"), ("unknown", "结果未知")):
        count = sum(row["status"] == status for row in rows)
        allowed.add(f"文件变更中有{count}项{wording}。")
    for row in rows:
        reason = _FS_CHANGE_REASON_LABELS.get(str(row.get("reason") or ""))
        state = {"blocked": "受阻", "failed": "失败", "not_started": "未开始"}.get(row["status"])
        if reason and state:
            allowed.add(f"第{row['position']}项因{reason}而{state}。")
    # Do not extract claims from forged receipt JSON, quoted labels or fragments.
    text = str(content or "").split(_CONFIRMED_RESULT_MARKER, 1)[0]
    clauses = re.findall(r"[^。！？\n]+[。！？]|[^。！？\n]+$", text)
    return "".join(clause for raw in clauses if (clause := raw.strip()) in allowed)


def _download_progress_analysis(
    content: object, result: Mapping[str, Any] | None,
    results: Sequence[tuple[str, Mapping[str, Any]]],
) -> str:
    """One local ingest-stage explanation, bound to the confirmed request ID.

    No inference from prose, readiness flags, other requests or remote progress.
    Ambiguous observations are unsupported, rather than treated as latest state.
    """
    if result is None or result.get("ok") is not True:
        return ""
    data = result.get("data")
    if not isinstance(data, Mapping):
        return ""
    number = data.get("request_number")
    if (type(number) is not int or number <= 0
            or data.get("source_type") != "direct_url" or data.get("target") != "qb"
            or data.get("status") != "submitted"
            or public_result_state(result) != "submitted"):
        return ""
    snapshots = []
    for _, value in results:
        observed = value.get("data")
        if not isinstance(observed, Mapping):
            continue
        request = observed.get("request")
        if not isinstance(request, Mapping) or request.get("request_number") != number:
            continue
        if (type(request.get("request_number")) is not int
                or value.get("ok") is not True or value.get("status") != "success"
                or any(v.get("truncated") or v.get("has_more") for v in (value, observed, request))):
            return ""
        if request not in snapshots:
            snapshots.append(request)
    if len(snapshots) != 1:
        return ""
    request = snapshots[0]
    if (request.get("targets") != ["qb"] or request.get("status") != "downloading"
            or request.get("qb_status") != "downloading"
            or request.get("organize_status") != "pending" or request.get("strm_status") != "pending"):
        return ""
    allowed = f"提交后，请求#{number}的本地记录显示qB正在下载，整理与STRM仍待处理，因此尚不能确认入库完成。"
    text = str(content or "").split(_CONFIRMED_RESULT_MARKER, 1)[0]
    clauses = re.findall(r"[^。！？\n]+[。！？]|[^。！？\n]+$", text)
    return "".join(clause for raw in clauses if (clause := raw.strip()) == allowed)


def format_partial_progress(
    reason: str, results: Sequence[tuple[str, Mapping[str, Any]]], *,
    confirmed_result: Mapping[str, Any] | None = None, write_attempted: bool = False,
    current_errors: Iterable[object] | None = None, analysis_text: str = "",
) -> str:
    """统一降级投影：查询交付已有内容，实际涉及写入才显示执行警示。"""
    lines = ["部分完成：" + _safe(reason, limit=300)]
    analysis = _file_change_analysis(analysis_text, confirmed_result, results) + _download_progress_analysis(analysis_text, confirmed_result, results)
    if analysis:
        lines.extend(("", analysis))
    if write_attempted or confirmed_result is not None:
        if confirmed_result is not None:
            lines.extend(("", format_public_result(confirmed_result)))
        if results:
            lines.append("\n本轮工具核对结果（不代表后续操作已完成）：")
            for label, result in results[-8:]:
                lines.append(f"• {_safe(label, limit=100)}：\n{format_public_result(result)}")
            if len(results) > 8:
                lines.append(f"以上为最近 8 项；本轮共记录 {len(results)} 项工具结果。")
        if current_errors is not None:
            errors = list(dict.fromkeys(
                text for error in current_errors if (text := _safe(error, limit=200))
            ))
            if errors:
                lines.append("\n未完成的查询：" + "；".join(errors))
        lines.append("\n后续写操作尚未执行；已确认操作以以上回执为准。" if confirmed_result is not None
                     else "\n本轮未执行新的写操作；检查或搜索成功不等于变更已提交。")
        lines.append("可稍后继续处理未完成部分；新的写操作仍需确认，不会自动重放。")
        return "\n".join(lines)
    items: dict[tuple[str, str, str], str] = {}
    summaries = []
    # 投影调用方 ledger 的当前错误；历史抑制只信任其可靠清错类别，不在这里推断恢复。
    errors = list(dict.fromkeys(
        text for error in current_errors if (text := _safe(error, limit=200))
    )) if current_errors is not None else []
    for _label, result in results:
        if result.get("ok") is False or result.get("status") in {"rate_limited", "failed", "error"}:
            # 标记不含按工具名清除的参数错误及未登记失败；它们与确认/写入分支仍展示历史。
            if current_errors is not None and result.get("_recovery_tracked"):
                continue
            error = _safe(result.get("error") or result.get("summary"), limit=200)
            if error and error not in errors:
                errors.append(error)
            continue
        data = result.get("data")
        rows = data.get("items", []) if isinstance(data, Mapping) else []
        for item in rows if isinstance(rows, (list, tuple)) else ():
            if not isinstance(item, Mapping):
                continue
            title = _safe(item.get("title") or item.get("series_name") or item.get("name"), limit=160)
            if not title:
                continue
            year = _safe(item.get("year"), limit=8)
            media_type = {"movie": "电影", "Movie": "电影", "tv": "剧集", "Series": "剧集", "Episode": "单集"}.get(item.get("media_type") or item.get("type"), "")
            if media_type == "单集":
                season, episode = _int_value(item.get("season_number")), _int_value(item.get("episode_number"))
                position = ((f"S{season:02}" if season is not None and season >= 0 else "") + f"E{episode:02}"
                            if episode is not None and episode >= 0 else _safe(item.get("name"), limit=160))
                if position and position != title:
                    title += " · " + position
            key = (title, year, media_type)
            overview = _safe(item.get("overview"), limit=180)
            suffix = " · ".join(value for value in (year, media_type) if value)
            items.setdefault(key, title + (f"（{suffix}）" if suffix else "") + (f"：{overview}" if overview else ""))
        summary = _safe(result.get("summary"), limit=300)
        if summary and summary not in summaries:
            summaries.append(summary)
    if items:
        lines.append("\n已取得的部分内容（尚未完成全部筛选/核对）：")
        lines.extend(f"{index}. {text}" for index, text in enumerate(list(items.values())[:12], 1))
        if len(items) > 12:
            lines.append(f"本轮共取得 {len(items)} 项去重内容，此处展示前 12 项。")
    else:
        lines.append("\n已完成的查询：" if summaries else "\n本轮尚未取得可展示的查询结果。")
        lines.extend(f"• {text}" for text in summaries[-8:])
    if errors:
        lines.append("\n未完成的查询：" + "；".join(errors[-3:]))
    lines.append("\n以上仅代表已成功取得的结果；未完成的查询不能据此判断没有内容。")
    return "\n".join(lines)


def sanitize_confirmed_answer(
    content: object, result: Mapping[str, Any] | None = None, *,
    results: Sequence[tuple[str, Mapping[str, Any]]] = (),
) -> str:
    """锁定操作回执，并从调用方的工具 ledger 保留新核验与解释。

    results 只能来自已执行工具的公开结果，不从模型正文/标记中认证事实。
    已完成正文仅保留领域 DTO 可证明的有限句型，其余说明使用公开投影。
    """
    text = str(content or "").replace("\x00", "").strip()
    prefix, marker, suffix = text.partition(_CONFIRMED_RESULT_MARKER)
    end = 0
    if marker:
        try:
            _, end = json.JSONDecoder().raw_decode(suffix.lstrip())
        except ValueError:
            pass
    source = result
    if source is None:
        return "⚠️ 执行结果尚未确认，请先查询实际业务状态，勿直接重复提交。" if marker else text
    receipt = format_public_result(source)
    analysis = _file_change_analysis(text, source, results) + _download_progress_analysis(text, source, results)
    if public_result_state(source) != "success" or (marker and not end):
        if analysis:
            receipt += "\n\n" + analysis
        if results:
            return receipt + "\n\n后续核验（不改变上述操作状态）：\n" + "\n".join(
                f"• {_safe(label, limit=100)}：\n{format_public_result(value)}"
                for label, value in results[-8:]
            )
        return receipt
    if marker:
        text = "\n\n".join(part for part in (prefix.strip(), suffix.lstrip()[end:].strip()) if part)
    return text or receipt


def public_conversation_messages(
    conversation: Sequence[Mapping[str, Any] | object],
    *, candidate_view: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """投影可公开恢复的对话，过滤工具中间回合和空助手消息。"""
    messages: list[dict[str, Any]] = []
    pending_tools: list[str] = []
    pending_candidate = False
    # 只有同一确认计划的最终答复才能取代中间回执；不能按相似文本跨任务折叠。
    final_effects = {
        str(item["effect_plan_id"]): index
        for index, item in enumerate(conversation)
        if isinstance(item, Mapping) and item.get("role") == "assistant"
        and item.get("effect_plan_id") and not item.get("tool_name")
        and not item.get("tool_calls") and str(item.get("content") or "").strip()
    }
    # 旧会话尚未保存plan关联：仅迁移明确的内部确认回执及同用户回合后续答复。
    # 普通工具摘要/跨用户消息绝不靠文字相似度合并，原始审计记录不修改。
    legacy_finals: dict[int, tuple[int, Mapping[str, Any]]] = {}
    legacy_receipt = None
    for index, item in enumerate(conversation):
        if not isinstance(item, Mapping):
            continue
        if item.get("role") == "user":
            legacy_receipt = None
        elif item.get("role") == "assistant" and not item.get("tool_calls"):
            if item.get("tool_name"):
                legacy_receipt = None
                _, marker, payload = str(item.get("content") or "").partition(_CONFIRMED_RESULT_MARKER)
                if marker and not item.get("effect_plan_id"):
                    try:
                        result, _ = json.JSONDecoder().raw_decode(payload.lstrip())
                        if isinstance(result, Mapping):
                            legacy_receipt = (index, result)
                    except ValueError:
                        pass
            elif legacy_receipt and str(item.get("content") or "").strip():
                legacy_finals[index] = legacy_receipt
                legacy_receipt = None
    legacy_receipt_indexes = {index for index, _ in legacy_finals.values()}
    # 候选卡只展示最后一次确认结果。同一批候选可以分次提交，不能按 ref
    # 隐藏全部历史回执；旧历史没有 plan ID，以最后一条同引用、同公开文本
    # 的回执定位卡片实际替代的消息，相同文本也只能折叠一次。
    folded_result_index = -1
    last_result = candidate_view.get("last_result") if candidate_view else None
    last_text = last_result.get("text") if isinstance(last_result, Mapping) else None
    if isinstance(last_text, str) and last_text.strip():
        for index in range(len(conversation) - 1, -1, -1):
            item = conversation[index]
            if (
                isinstance(item, Mapping)
                and item.get("role") == "assistant"
                and item.get("candidate_result_ref") == candidate_view.get("ref")
                and str(item.get("public_content") or "").strip() == last_text.strip()
            ):
                folded_result_index = index
                break

    candidate_followup = ""
    if folded_result_index >= 0:
        row = conversation[folded_result_index]
        final_index = final_effects.get(str(row.get("effect_plan_id") or ""), -1)
        if final_index < 0:
            final_index = next((i for i, (receipt_index, _) in legacy_finals.items() if receipt_index == folded_result_index), -1)
        if final_index >= 0:
            content = str(conversation[final_index].get("content") or "").strip()
            if final_index in legacy_finals:
                content = sanitize_confirmed_answer(content, legacy_finals[final_index][1])
            receipt = last_text.strip()
            if content == receipt or content.startswith(receipt + "\n\n"):
                folded_result_index = final_index
                candidate_followup = content[len(receipt):].strip()

    def remember_tool(value: object) -> None:
        name = str(value or "").strip()
        if name and name not in pending_tools:
            pending_tools.append(name)

    for index, item in enumerate(conversation):
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "").strip()
        if role == "user":
            if pending_candidate and candidate_view:
                messages.append({"role": "assistant", "content": "资源候选", "candidate_view": dict(candidate_view)})
            pending_candidate = False
            pending_tools.clear()
            content = str(item.get("content") or "").strip()
            if content:
                messages.append({"role": "user", "content": content})
            continue
        if role == "tool":
            if candidate_view:
                refs = decode_tool_content(str(item.get("content") or "")).get("opaque_refs", [])
                if isinstance(refs, list) and any(
                    isinstance(ref, dict) and ref.get("kind") == "resource_candidates"
                    and ref.get("ref") == candidate_view.get("ref") for ref in refs
                ):
                    pending_candidate = True
            remember_tool(item.get("tool_name"))
            continue
        if role != "assistant":
            continue
        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, Sequence) and not isinstance(
            tool_calls, (str, bytes, bytearray)
        ):
            for raw_call in tool_calls:
                if isinstance(raw_call, Mapping):
                    remember_tool(raw_call.get("name"))
            continue
        tool_name = str(item.get("tool_name") or "").strip()
        if tool_name:
            remember_tool(tool_name)
            if index in legacy_receipt_indexes or final_effects.get(str(item.get("effect_plan_id") or ""), -1) > index:
                continue
            content = str(item.get("public_content") or "").strip()
            if not content:
                # tool_name is persisted server metadata; only this receipt row
                # may migrate legacy embedded JSON, never arbitrary model text.
                raw = str(item.get("content") or "")
                _, marker, payload = raw.partition(_CONFIRMED_RESULT_MARKER)
                result = None
                if marker:
                    try:
                        result, _ = json.JSONDecoder().raw_decode(payload.lstrip())
                    except ValueError:
                        pass
                content = sanitize_confirmed_answer(raw, result if isinstance(result, Mapping) else None)
        else:
            content = str(item.get("content") or "").strip()
            if index in legacy_finals:
                content = sanitize_confirmed_answer(content, legacy_finals[index][1])
        if content:
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if pending_tools:
                message["tools"] = list(pending_tools)
                message["tool_labels"] = [
                    public_tool_label(tool_name) for tool_name in pending_tools
                ]
            if candidate_view and index == folded_result_index:
                message["candidate_result_ref"] = candidate_view["ref"]
                if candidate_followup:
                    message["candidate_followup"] = candidate_followup
            if pending_candidate and candidate_view:
                message["candidate_view"] = dict(candidate_view)
                pending_candidate = False
            messages.append(message)
            pending_tools.clear()
    if pending_candidate and candidate_view:
        messages.append({"role": "assistant", "content": "资源候选", "candidate_view": dict(candidate_view)})
    return messages
