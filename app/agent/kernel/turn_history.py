"""回合历史检查点与完整事实保存。"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from app.agent.errors import ToolPipelineError
from app.agent.kernel.state_types import (
    HistoryCapacityError,
    StalePublicationError,
    StateUpdate,
    _ExecutionCapacity,
)
from app.agent.public_safety import public_tool_label, sanitize_public_text

from . import error_repair
from .capabilities import ToolEffect
from .continuation import _ContinuationAction
from .conversation_codec import pending_content
from .conversation_context import persisted_conversation, restore_messages
from .discovery import DISCOVERY_TOOL
from .model import (
    ModelMessage,
)
from .task_state import TaskPatch

if TYPE_CHECKING:
    from .turn import AgentTurn

logger = logging.getLogger(__name__)

# 未取得查询结果的服务错误；not_found/loading/disabled等业务观察不是执行异常。
_READ_FAILURE_STATUSES = frozenset({
    "unavailable", "timeout", "authentication", "invalid_response", "unreadable",
    "budget_exhausted", "configuration_missing", "provider_error", "provider_timeout",
    "http_error", "security_error", "response_too_large", "indexer_error",
    "validation_error", "query_rejected", "challenge_required", "result_expired", "result_not_found",
    "provider_unavailable", "upstream_timeout", "provider_auth_failed", "provider_not_configured",
    "provider_user_required", "provider_version_unsupported",
})

def record_result(turn: AgentTurn,
    name: str, arguments: Mapping[str, Any], result: Mapping[str, Any],
    *, error: ToolPipelineError | None = None, recovered_failures: tuple[str, ...] = (), rejected_call: str = "",
) -> None:
    tool = turn.session.catalog.get(name) if turn.session.catalog.has(name) else None
    if tool is not None:
        name = tool.name
    if name == DISCOVERY_TOOL:
        return
    # 显式参数重试按对象归并；无参工具依赖隐式上下文，切换后不能冒充同一对象恢复。
    if name != turn.last_business_tool:
        turn.context_revision += 1
    turn.last_business_tool = name
    attempt = (name, 0 if arguments else turn.context_revision,
               "unparsed:" + rejected_call if rejected_call else json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True, default=str))
    # 聚合 DTO 可同时提供部分结果和失败事实；保留已取得的结果，再登记未完成部分。
    turn.has_tool_result = turn.has_tool_result or (error is None and result.get("ok") is not False)
    status = result.get("status")
    if error is None and isinstance(status, str) and (
        status == "rate_limited" or (tool is not None and tool.effect is ToolEffect.READ
            and (status == "partial" or (result.get("ok") is False and status in _READ_FAILURE_STATUSES)))
    ):
        error = ToolPipelineError(
            sanitize_public_text(result.get("error") or result.get("summary") or "工具查询未取得结果"), code=status,
        )
    turn.progress_results.append((public_tool_label(name), {**result, "_recovery_tracked": error is not None}))
    if error is not None:
        turn.unresolved_errors[attempt] = error
    else:
        if result.get("ok") is not False:
            turn.unresolved_errors.pop(attempt, None)
        read_completed = tool is not None and tool.effect is ToolEffect.READ and error_repair.completed_result(tool, result)
        for previous, failure_error in list(turn.unresolved_errors.items()):
            if ((failure_error.repair_ticket and failure_error.repair_ticket in recovered_failures)
                    or (read_completed and previous[0] == name and failure_error.code == "invalid_arguments"
                        and not failure_error.repair_ticket)):
                # 只读形状错误保留在历史中，但成功重试后不再阻断回答；它不是对象核验/写授权。
                turn.unresolved_errors.pop(previous)


async def persist_conversation(turn: AgentTurn,
    *, close_pending: bool = False, preserve_capacity: bool = False,
    task_patch: TaskPatch | None = None,
    staged_messages: list[ModelMessage] | None = None,
    staged_completed: set[str] | None = None,
    wait_action: _ContinuationAction | None = None,
) -> None:
    lease = turn.lease
    assert lease is not None
    async with turn.publication_lock:
        checkpoint_messages = list(turn.messages if staged_messages is None else staged_messages)
        completed = turn.completed_call_ids if staged_completed is None else staged_completed
        if close_pending:
            for call in turn.active_calls:
                if call.call_id in completed:
                    continue
                started = call.call_id in turn.started_call_ids
                checkpoint_messages.append(
                    ModelMessage(
                        role="tool", content=pending_content(started),
                        tool_call_id=call.call_id, tool_name=call.name,
                    )
                )
        stored = persisted_conversation(
            checkpoint_messages,
            current_user_index=turn.current_user_index,
            original_message=turn.agent_input.message,
            reply_context=turn.agent_input.reply_context,
            prior_conversation=turn.state.conversation,
        )
        # The store enforces retention after receipt merge in its transaction.
        start = turn.current_user_index if turn.current_user_index is not None else next(
            (i for i in range(len(stored) - 1, -1, -1) if stored[i].get("role") == "user"), len(stored),
        )
        try:
            extra_updates = ()
            if wait_action is not None:
                extra_updates = (wait_action,)
            elif task_patch is not None and task_patch.operation == "create" and turn.session.continuation_binding is not None:
                extra_updates = (_ContinuationAction("", {
                    "operation": "seed", "model_limit": turn.session.limits.max_model_rounds,
                    "tool_limit": turn.session.limits.max_tool_calls,
                    "model_spent": turn._budget_model_attempts - turn._budget_paid_models,
                    "tool_spent": turn._budget_tool_attempts - turn._budget_paid_tools,
                }, binding=turn.session.continuation_binding),)
            # 失败确认后的诊断仍属于同一执行组，不能在续接前撤掉已有的有界容量。
            # 下一用户回合仍按正常规则释放；保留此阶段并不会提高容量上限。
            turn.state = await turn.session.state_store.commit(
                lease,
                updates=(_ExecutionCapacity(
                    "", turn.state.execution_capacity if preserve_capacity or turn.diagnosing_effect else close_pending and any(
                        call.call_id not in completed for call in turn.active_calls
                    ),
                    current_start=start if any(row.get("tool_calls") or row.get("reply_context", {}).get("reference_handle") for row in stored[start:]) else None,
                ),) + extra_updates,
                conversation=stored,
                **({"task_patch": task_patch} if task_patch is not None else {}),
            )
            if task_patch is not None and not lease.continuation_id:
                if task_patch.operation == "create" and turn.session.continuation_binding is not None:
                    turn._budget_task_id = turn.state.metadata["task"]["task_id"]
                    turn._budget_paid_models = turn._budget_model_attempts
                    turn._budget_paid_tools = turn._budget_tool_attempts
                elif task_patch.operation == "cancel":
                    turn._budget_task_id = ""
        except HistoryCapacityError as exc:
            raise ToolPipelineError(
                "当前结果超过历史容量，已保留此前检查点；已开始的调用结果可能未知，请勿重放写操作。",
                code="history_capacity_exceeded",
            ) from exc


async def remember_result(turn: AgentTurn,
    *,
    tool_name: str,
    content: str,
    public_content: str,
    candidate_result: Mapping[str, Any] | None = None,
    receipt_plan_id: str | None = None,
) -> bool:
    """先持久化真实执行结果，再沿原始任务继续，不能靠模型改写事实。"""

    lease = turn.tool_context.lease
    receipt_plan_id = receipt_plan_id or turn.plan_id
    safe_content = (content or "").strip()
    if not safe_content:
        return False
    conversation = [dict(item) for item in turn.state.conversation]
    # 完成原调用的结果，而非仅追加一条模型自述；旧预览仍保留在事件审计中。
    # 不新增虚构调用或孤立tool消息，保持各Provider的一问一答配对。
    for row in reversed(conversation):
        if row.get("role") == "tool" and row.get("effect_plan_id") == receipt_plan_id:
            row["content"] = safe_content
            break
    item = ModelMessage(
        role="assistant",
        content=(
            "已确认操作的可信系统结果（不是待执行计划）：\n"
            + safe_content
        ),
        tool_name=tool_name,
        effect_plan_id=receipt_plan_id or "",
        completion_receipt_id=receipt_plan_id or "",
    ).to_dict()
    safe_public_content = (public_content or "").strip()
    if safe_public_content:
        item["public_content"] = safe_public_content
    updates: tuple[StateUpdate, ...] = ()
    if candidate_result:
        item["candidate_result_ref"] = candidate_result["ref"]
        updates = (
            StateUpdate("metadata.ux_candidate_result", dict(candidate_result)),
        )
    conversation.append(item)
    try:
        from app.agent.kernel.completion_receipts import (
            acknowledge_effect_receipt,
            remember_effect_receipt,
        )

        await remember_effect_receipt(turn.session.state_store, owner=lease.owner,
            session_id=lease.session_id, plan_id=receipt_plan_id, message=item)
        turn.state = await turn.session.state_store.commit(
            lease,
            conversation=conversation,
            updates=updates,
        )
        await acknowledge_effect_receipt(turn.session.state_store, owner=lease.owner,
            session_id=lease.session_id, plan_id=receipt_plan_id)
        turn.messages = restore_messages(turn.state)
        return True
    except StalePublicationError:
        raise
    except Exception as exc:  # noqa: BLE001 - 已执行副作用不得被状态写回遮蔽
        logger.warning(
            "Agent 确认结果写回会话失败 type=%s", type(exc).__name__
        )
        return False


async def preserve_checkpoint(turn: AgentTurn) -> None:
    if not turn.checkpoint_ready:
        return
    try:
        await persist_conversation(turn, close_pending=True)
    except StalePublicationError:
        return
    except Exception as exc:  # noqa: BLE001 - 故障收尾不能遮蔽原始错误
        logger.warning(
            "Agent 故障检查点写回失败 type=%s", type(exc).__name__
        )
