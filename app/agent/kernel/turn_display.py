"""根据活动快照和有界事件尾部恢复公开回合状态，不读取会话执行状态。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .events import AgentEventType, terminal_turn_status

_ACTIVE_PHASE_DETAILS = {
    "turn.started": "正在启动本轮任务",
    "capabilities.selected": "正在准备可用能力",
    "model.started": "正在思考",
    "model.delta": "正在整理回复",
    "model.tool_call": "正在规划下一步",
    "tool.started": "正在执行工具",
    "tool.progress": "正在处理任务",
    "tool.completed": "正在整理工具结果",
    "tool.failed": "正在处理工具异常",
    "effect.preview_started": "正在准备操作预览",
    "effect.approval_required": "正在准备确认信息",
    "effect.completed": "正在整理操作结果",
    "effect.failed": "正在整理操作结果",
    "turn.completed": "正在完成本轮任务",
    "turn.failed": "正在结束本轮任务",
    "turn.cancelled": "正在停止本轮任务",
}
_TERMINAL_MESSAGES = {
    "completed": "本轮已完成",
    "failed": "本轮未能完成",
    "cancelled": "本轮已取消",
    "waiting": "本轮已交接，等待任务继续。",
}
_KNOWN_TURN_EVENTS = frozenset(AgentEventType)


def _matches_turn(event: Any, turn: dict[str, Any]) -> bool:
    return bool(
        isinstance(event, dict)
        and event.get("turn_id") == turn.get("turn_id")
        and event.get("request_id") == turn.get("request_id")
    )


def active_turn_view(
    activity: Any, events: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not isinstance(activity, dict) or activity.get("status") not in {"running", "cancelling"}:
        return None
    turn = {
        "request_id": activity.get("request_id", ""),
        "turn_id": activity.get("turn_id", ""),
        "generation": activity.get("generation"),
        "protected": bool(activity.get("protected")),
        "status": activity["status"],
    }
    latest_type = next(
        (str(event.get("type") or "") for event in reversed(events) if _matches_turn(event, turn)),
        "",
    )
    turn["detail"] = (
        "正在停止本轮任务"
        if turn["status"] == "cancelling"
        else _ACTIVE_PHASE_DETAILS.get(latest_type, "正在处理本轮任务")
    )
    return turn


def last_turn_view(
    events: list[dict[str, Any]], active_turn: dict[str, Any] | None
) -> dict[str, str] | None:
    if not events:
        return None
    event = events[-1]
    if not isinstance(event, dict):
        return None
    event_type = str(event.get("type") or "")
    request_id = event.get("request_id")
    turn_id = event.get("turn_id")
    if not isinstance(request_id, str) or not isinstance(turn_id, str):
        return None
    payload = event.get("payload")
    status = terminal_turn_status(event_type, payload if isinstance(payload, Mapping) else {})
    if status is not None:
        if event_type == AgentEventType.TURN_COMPLETED and status != "waiting":
            status = "completed"
        message = _TERMINAL_MESSAGES[status]
    elif event_type in _KNOWN_TURN_EVENTS:
        if active_turn is not None and _matches_turn(event, active_turn):
            return None
        # 长回复的started事件可能已超出有界尾部；不能因此把未确认终态当完成。
        status, message = "interrupted", "本轮执行状态未确认，请核对已保存的结果；不会自动重放。"
    else:
        return None
    return {
        "request_id": request_id,
        "turn_id": turn_id,
        "status": status,
        "message": message,
    }
