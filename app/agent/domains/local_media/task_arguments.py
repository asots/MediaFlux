"""本地媒体任务和检查句柄的公开参数校验。"""

from __future__ import annotations

from typing import Any

from app.agent.errors import AgentToolError
from app.modules.local_media_models import LOCAL_TASK_STATUSES

from . import common

_ALLOWED_SCOPES = frozenset({"all", "attention", "active", "history", "skipped", "latest_scan"})


def _strict_number(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= common._MAX_TASK_NUMBER
    ):
        raise AgentToolError(f"{label}必须是 1 到 {common._MAX_TASK_NUMBER} 的整数")
    return value


def local_media_task_summaries_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict) or not set(arguments).issubset(
        {"scope", "limit", "scan_ref", "offset"}
    ):
        raise AgentToolError("本地媒体任务列表参数无效")
    scope = str(arguments.get("scope") or "all").strip().lower()
    if scope not in _ALLOWED_SCOPES and scope not in LOCAL_TASK_STATUSES:
        raise AgentToolError("不支持的本地媒体任务范围")
    limit = arguments.get("limit", 12)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise AgentToolError("limit 必须是 1 到 20 的整数")
    offset = arguments.get("offset", 0)
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or not 0 <= offset <= 2_147_483_647
    ):
        raise AgentToolError("offset 必须是非负整数，且不能超过 2147483647")
    result = {"scope": scope, "limit": limit, "offset": offset}
    if "scan_ref" in arguments:
        scan_ref = arguments["scan_ref"]
        if scan_ref == "LM-UNRECORDED":
            result["scan_ref"] = scan_ref
            return result
        if (
            not isinstance(scan_ref, str)
            or not scan_ref.startswith("LM")
            or not scan_ref[2:].isascii()
            or not scan_ref[2:].isdigit()
            or scan_ref[2:3] == "0"
            or len(scan_ref) > 20
        ):
            raise AgentToolError("scan_ref 必须是本地扫描通知中的 LM 编号")
        result["scan_ref"] = scan_ref
    return result


def local_media_task_number_arguments(arguments: dict[str, Any]) -> dict[str, int]:
    if not isinstance(arguments, dict) or set(arguments) != {"task_number"}:
        raise AgentToolError("本地媒体任务参数无效")
    return {"task_number": _strict_number(arguments.get("task_number"), "task_number")}


def local_media_retry_arguments(arguments: dict[str, Any]) -> dict[str, int]:
    if not isinstance(arguments, dict) or "task_number" not in arguments or not set(
        arguments
    ).issubset({"task_number", "season", "episode"}):
        raise AgentToolError("本地媒体任务重试参数无效")
    has_season = "season" in arguments
    has_episode = "episode" in arguments
    if has_season != has_episode:
        raise AgentToolError("season 和 episode 必须同时提供")
    result = {
        "task_number": _strict_number(arguments.get("task_number"), "task_number")
    }
    if not has_season:
        return result
    season = arguments.get("season")
    episode = arguments.get("episode")
    if isinstance(season, bool) or not isinstance(season, int) or not 0 <= season <= 99:
        raise AgentToolError("season 必须是 0 到 99 的整数")
    if (
        isinstance(episode, bool)
        or not isinstance(episode, int)
        or not 1 <= episode <= 999
    ):
        raise AgentToolError("episode 必须是 1 到 999 的整数")
    result.update({"season": season, "episode": episode})
    return result


def local_media_inspection_arguments(arguments: dict[str, Any]) -> dict[str, int]:
    if not isinstance(arguments, dict) or set(arguments) != {"inspection_number"}:
        raise AgentToolError("本地媒体检查参数无效")
    value = arguments.get("inspection_number")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= common._MAX_INSPECTION_NUMBER
    ):
        raise AgentToolError("inspection_number 必须是正整数")
    return {"inspection_number": value}
