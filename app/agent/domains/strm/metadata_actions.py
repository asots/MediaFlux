"""Agent 对伴随元数据状态和冻结队列操作的薄适配。"""

from __future__ import annotations

from typing import Any

from app.agent.errors import AgentToolError
from app.agent.models import ToolContext, ToolResult
from app.modules import strm_metadata_management as service


def no_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict) or arguments:
        raise AgentToolError("该操作不接受额外参数")
    return {}


def policy_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {"enabled"}
        or type(arguments["enabled"]) is not bool
    ):
        raise AgentToolError("需要布尔参数 enabled")
    return dict(arguments)


def get_metadata_status(arguments: dict[str, Any]) -> ToolResult:
    no_arguments(arguments)
    data = service.metadata_status()
    return ToolResult(
        True, str(data["state"]), "已读取伴随元数据队列与消费者实时状态", data=data
    )


def prepare_cancel(
    arguments: dict[str, Any], context: ToolContext
) -> tuple[ToolResult, str]:
    no_arguments(arguments)
    try:
        data, token = service.prepare_backlog_cancel(context.owner)
    except ValueError as exc:
        raise AgentToolError(str(exc), code="precondition_failed") from exc
    return ToolResult(
        True,
        "confirmation_required",
        f"确认取消 {data['count']} 项伴随元数据待办，不删除文件",
        data=data,
    ), token


def cancel_confirmed(
    arguments: dict[str, Any], token: str, context: ToolContext
) -> ToolResult:
    no_arguments(arguments)
    try:
        data = service.cancel_backlog_confirmed(token, context.owner)
    except ValueError as exc:
        raise AgentToolError(str(exc), code="confirmation_stale") from exc
    return ToolResult(True, "completed", str(data["summary"]), data=data)


def prepare_policy(
    arguments: dict[str, Any], context: ToolContext
) -> tuple[ToolResult, str]:
    args = policy_arguments(arguments)
    try:
        data, token = service.prepare_policy(args["enabled"], context.owner)
    except ValueError as exc:
        raise AgentToolError(str(exc), code="precondition_failed") from exc
    return ToolResult(
        True,
        "confirmation_required",
        "确认调整伴随元数据同步开关，保留队列和文件",
        data=data,
    ), token


def policy_confirmed(
    arguments: dict[str, Any], token: str, context: ToolContext
) -> ToolResult:
    policy_arguments(arguments)
    try:
        data = service.set_policy_confirmed(token, context.owner)
    except ValueError as exc:
        raise AgentToolError(str(exc), code="confirmation_stale") from exc
    except (OSError, service.config.AtomicPublishError) as exc:
        raise AgentToolError(
            "同步开关未能可靠保存，请检查配置文件状态", code="unavailable"
        ) from exc
    verified = bool(data["verified"])
    return ToolResult(
        verified,
        "completed" if verified else "outcome_unknown",
        str(data["summary"]),
        data=data,
    )
