"""模型调用批次准入、顺序结算与有界并发。"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from app.agent.errors import ToolPipelineError, _ReadPermitRejected
from app.agent.kernel.conversation_state import retain_conversation
from app.agent.kernel.state_types import StalePublicationError, _CancelledReadInline
from app.agent.kernel.tool_calls import PipelineResult, ToolCallContext
from app.agent.model_context_budget import bounded_model_messages
from app.agent.public_view import (
    public_result_state,
)

from . import error_repair
from .capabilities import ToolEffect
from .continuation import _ContinuationAction
from .conversation_codec import pending_content
from .conversation_context import persisted_conversation, tool_error_message
from .discovery import DISCOVERY_TOOL
from .events import AgentEventType
from .model import (
    ModelMessage,
    ModelToolCall,
)
from .projection import _arguments_safe
from .session_guard import session_scope_guard
from .task_state import reduce_task, task_projection

if TYPE_CHECKING:
    from .turn import AgentTurn
from .turn_history import persist_conversation, record_result

if TYPE_CHECKING:
    from .model_rounds import _ModelRound
from .tool_calls import independent_read

_MAX_PARALLEL_READS = 4

logger = logging.getLogger(__name__)

def admit_calls(turn: AgentTurn, reply: _ModelRound) -> None:
    calls, assistant_text, provider_state = reply.calls, reply.text, reply.provider_state
    try:
        for call in calls:
            if call.argument_error is None:
                _arguments_safe(call.arguments)
    except ValueError as exc:
        raise ToolPipelineError(
            "当前工具参数无法完整安全保存，本批调用未执行；请缩小或修正查询。",
            code="invalid_arguments",
        ) from exc
    # Admission covers the request we already have, not future output.
    # Do not activate a batch whose exact safe arguments/native group
    # cannot survive the ordinary storage or current model budget.
    proposed = [*turn.messages, ModelMessage(
        role="assistant", content=assistant_text,
        tool_calls=tuple(calls), provider_state=provider_state,
    ), *(ModelMessage(
        role="tool", content=pending_content(False),
        tool_call_id=call.call_id, tool_name=call.name,
    ) for call in calls)]
    stored = persisted_conversation(
        proposed, current_user_index=turn.current_user_index,
        original_message=turn.agent_input.message,
        reply_context=turn.agent_input.reply_context,
        prior_conversation=turn.state.conversation,
    )[turn.history_end:]
    try:
        retained = retain_conversation(stored)
    except ValueError as exc:
        raise ToolPipelineError(
            "当前调用参数及已有执行事实超过历史容量，本批调用未执行；请缩小查询。",
            code="history_capacity_exceeded",
        ) from exc
    if [row for row in stored if row.get("tool_calls")] != [
        row for row in retained if row.get("tool_calls")
    ]:
        raise ToolPipelineError(
            "当前调用参数及已有执行事实超过历史容量，本批调用未执行；请缩小查询。",
            code="history_capacity_exceeded",
        )
    bounded_model_messages(
        proposed, history_end=turn.history_end,
        tool_definitions=reply.tools, system_prompt=reply.system_prompt,
        context_window_tokens=turn.context_window,
        output_tokens=turn.output_tokens,
        conservative=turn.conservative_budget,
    )
    turn.active_calls = tuple(calls)
    turn.completed_call_ids.clear()
    turn.started_call_ids.clear()
    turn.returned_reads.clear()
    turn.parallel_call_ids.clear()
    turn.messages.append(
        ModelMessage(
            role="assistant",
            content=assistant_text,
            tool_calls=tuple(calls),
            provider_state=provider_state,
        )
    )


async def execute_calls(turn: AgentTurn, reply: _ModelRound, round_task: Mapping[str, Any] | None,
                        business_batch_calls: int, discovery_batch_calls: int) -> bool:
    calls, resuming_read = reply.calls, reply.resuming_read
    await turn.reserve_budget(tools=len(calls))
    turn.total_tool_calls += len(calls)
    turn.business_tool_calls += business_batch_calls
    turn.discovery_tool_calls += discovery_batch_calls
    # 同一模型批次只对领域明确声明独立的READ做有界并发。
    # 发现、修参、任务变更、混合写以及恢复单个READ均沿同一执行链逐项运行。
    parallel_batch = (len(calls) > 1 and not turn.confirming
        and not turn.tool_context.diagnostic_only and not turn.tool_context.selection_arguments
        and all(call.argument_error is None
            and call.name in turn.available_names
            and independent_read(turn.session.catalog.get(call.name), turn.state.metadata.get("task")) for call in calls))
    ready: dict[str, tuple[ToolCallContext, PipelineResult | BaseException]] = {}

    for call_index, call in enumerate(calls):
        if call.name not in turn.available_names:
            error = ToolPipelineError(
                "该工具不在本轮候选能力中；先调用 agent.capabilities 指定 tool_names 加载工具，下一轮按返回的 Schema 调用。",
                code="tool_not_available",
            )
            await turn.close_failed_call(call, error)
            continue
        tool = turn.session.catalog.get(call.name)
        if turn.agent_input.initial_read_tool and tool.effect is not ToolEffect.READ:
            await turn.close_failed_call(call, ToolPipelineError(
                "工作区行动接力仅允许只读能力。", code="read_scope_required"))
            continue
        if call.argument_error is not None:
            error = ToolPipelineError(
                "工具参数不是有效 JSON 对象；请按 Schema 重新提交完整参数。",
                code="invalid_arguments",
            )
            error.repair_ticket = error_repair.issue(tool, None, turn.tool_context)
            # No object identity can be inferred from undecodable JSON.
            # Track a parameterless syntax failure, never execute it as {}.
            record_result(turn, tool.name, {}, {"ok": False, "status": error.code, "summary": str(error)}, error=error, rejected_call=call.call_id)
            await turn.close_failed_call(call, error, tool_name=tool.name, input_schema=tool.input_schema)
            continue
        canonical_call = ModelToolCall(
            call_id=call.call_id,
            name=tool.name,
            arguments=call.arguments,
        )
        previous = turn.tool_context.confirmed_effect
        workflow = tool.metadata.get("workflow")
        if (previous and workflow and workflow == previous.tool.metadata.get("workflow")
                and tool.name != previous.tool.name
                and tool.metadata.get("workflow_stage", 0) < previous.tool.metadata.get("workflow_stage", 0)):
            # An earlier workflow stage may replace a frozen implicit plan.
            # The old receipt remains evidence, not authorization/reuse for that new plan.
            turn.tool_context = replace(turn.tool_context, confirmed_effect=None)
        if tool.effect is not ToolEffect.READ:
            turn.attempted_write = True
        discovery_checkpoint = turn.discovery.checkpoint()
        if tool.name == DISCOVERY_TOOL:
            current_state = await turn.session.state_store.load(
                owner=turn.agent_input.owner, session_id=turn.agent_input.session_id,
            )
            turn.discovery.context["reference_kinds"] = tuple(current_state.ref_kinds)
        try:
            if call.call_id not in ready:
                ready.update(await execute_group(turn, calls, call_index, round_task, parallel_batch))
            call_context, outcome = ready.pop(call.call_id)
            if call_context.schema_replacements is not turn.tool_context.schema_replacements:
                turn.tool_context.schema_replacements.update(call_context.schema_replacements)
            if isinstance(outcome, BaseException):
                raise outcome
            result = outcome
        except ToolPipelineError as exc:
            # 接力未携带持久化 scope，不能借旧任务额度停泊成无 scope 的恢复回合。
            if (isinstance(exc, _ReadPermitRejected) and not turn.agent_input.initial_read_tool
                    and turn.session.continuation_binding is not None
                    and (turn.state.metadata.get("read_wait") or turn.lease.continuation_kind == "effect") and not turn.confirming and not turn.explaining
                    and (not turn.attempted_write or (
                        turn.lease.continuation_kind == "effect" and turn.lease.continuation_id
                        and turn.confirmed_result is not None and public_result_state(turn.confirmed_result) == "failed"
                    )) and not turn.state.pending_effect_plan_id
                    and all(c.call_id in turn.completed_call_ids for c in calls if c is not call)):
                staged = [*turn.messages, tool_error_message(call, exc)]
                done = turn.completed_call_ids | {call.call_id}
                try:
                    await persist_conversation(turn, staged_messages=staged, staged_completed=done,
                        wait_action=_ContinuationAction("", {
                            "operation": "park", "permit": exc, "call_id": call.call_id,
                            "channel": turn.agent_input.channel,
                            "delivery_route": turn.agent_input.delivery_route,
                            "model_policy": turn.selected_model,
                        }, binding=turn.session.continuation_binding))
                except ValueError:
                    pass  # Invalid retry/capacity/legacy state is never advertised as waiting.
                else:
                    turn.messages, turn.completed_call_ids, turn.active_calls = staged, done, ()
                    await turn.publish_tool(AgentEventType.TOOL_FAILED, call, code=exc.code, message=str(exc))
                    await turn.publish(AgentEventType.TURN_COMPLETED, {
                        "status": "waiting", "answer": "查询受到本地额度限制，已保存等待；尚未完成。",
                        "retry_at_epoch": (turn.state.metadata["effect_waits"][turn.lease.continuation_id]["continuation"] if turn.lease.continuation_kind == "effect" else turn.state.metadata["read_wait"])["due_at"],
                        "model_calls": turn.model_calls, "tool_calls": turn.total_tool_calls,
                    })
                    return True
            if tool.name == DISCOVERY_TOOL:
                # 仅撤销本次失败的发现，不能丢掉同批之前成功的结果。
                turn.discovery.restore(discovery_checkpoint)
            if exc.task_patch is not None:
                staged = [*turn.messages, tool_error_message(call, exc, input_schema=tool.input_schema)]
                done = turn.completed_call_ids | {call.call_id}
                await persist_conversation(turn, close_pending=True, task_patch=exc.task_patch,
                                           staged_messages=staged, staged_completed=done)
                turn.messages, turn.completed_call_ids = staged, done
                record_result(turn, tool.name, call.arguments, {"ok": False, "summary": str(exc)}, error=exc)
                await turn.publish_tool(AgentEventType.TOOL_FAILED, call, code=exc.code, message=str(exc))
            else:
                record_result(turn, tool.name, call.arguments, {"ok": False, "status": exc.code, "summary": str(exc)}, error=exc)
                await turn.close_failed_call(
                    call, exc,
                    tool_name=tool.name,
                    input_schema=tool.input_schema,
                )
            if resuming_read:
                # Only trusted acquire rejection above may repark.
                # Preserve the closed error/TaskPatch, never retry a started READ.
                raise
            continue
        if (turn.tool_context.diagnostic_only and tool.effect is ToolEffect.READ
                and not tool.name.startswith("agent.")
                and public_result_state(result.outcome.public_content) == "success"):
            # An observed business result permits planning another proposal,
            # not replaying a write or upgrading the original failed receipt.
            turn.tool_context = replace(turn.tool_context, diagnostic_only=False)
        if tool.name == DISCOVERY_TOOL and result.outcome.public_content.get("ok") is False:
            turn.discovery.restore(discovery_checkpoint)
        if result.outcome.public_content.get("result_handle"):
            turn.discovery.context["reference_kinds"] = tuple(set(turn.discovery.context.get("reference_kinds", ())) | {"tool_result"})
        try:
            if result.task_patch is not None:
                preview = reduce_task(turn.state.metadata.get("task"), result.task_patch,
                                      owner=turn.lease.owner, session_id=turn.lease.session_id)
                task = task_projection(preview)
                content = {**json.loads(result.outcome.model_content), "task": task}
                result = replace(result, outcome=replace(result.outcome,
                    public_content={**dict(result.outcome.public_content), "task": task},
                    model_content=json.dumps(content, ensure_ascii=False, separators=(",", ":"))))
            staged = [*turn.messages, ModelMessage(role="tool", content=result.outcome.model_message(),
                      tool_call_id=call.call_id, tool_name=call.name,
                      effect_plan_id=result.effect_plan.plan_id if result.effect_plan else "")]
            done = turn.completed_call_ids | {call.call_id}
            # 回执与任务补丁共同提交成功后，才发布事实、清算错误和计入观察。
            await persist_conversation(turn, close_pending=True, task_patch=result.task_patch,
                                       staged_messages=staged, staged_completed=done)
        except ValueError:
            if result.task_patch is None:
                raise
            error = ToolPipelineError("任务更新被拒绝；此前检查点保持不变。", code="task_conflict")
            staged = [*turn.messages, tool_error_message(call, error)]
            done = turn.completed_call_ids | {call.call_id}
            await persist_conversation(turn, close_pending=True, staged_messages=staged, staged_completed=done)
            turn.messages, turn.completed_call_ids = staged, done
            record_result(turn, tool.name, call.arguments, {"ok": False, "summary": str(error)}, error=error)
            await turn.publish_tool(AgentEventType.TOOL_FAILED, call, code=error.code, message=str(error))
            if resuming_read:
                raise error
            continue
        turn.messages, turn.completed_call_ids = staged, done
        if result.task_patch is not None:
            turn.discovery.context["has_task"] = True
        if result.effect_plan is not None:
            plan = result.effect_plan
            if await turn.session.pipeline.scoped_trust.allows(plan, context=turn.tool_context):
                turn.scope.enter_context(session_scope_guard(turn.lease.owner, turn.lease.session_id, kind="effect"))
                if await turn.session.pipeline.scoped_trust.allows(plan, context=turn.tool_context):
                    await turn.session.coordinator.protect(turn.lease, turn.token)
                    turn.plan_id, turn.confirming = plan.plan_id, True
                    turn.last_user = turn.state.metadata.get("pending_effect_goal") or {"role": "user", "content": turn.agent_input.message}
                    if not await turn.confirm_effect(scoped_trust=True):
                        return True
                    continue
                turn.scope.close()
            await turn.publish_tool_result(canonical_call, result)
            # Provider 已被要求禁止并行工具，但兼容服务仍可能
            # 违规一次返回多个调用。写操作在此暂停等待人工确认，
            # 后续调用必须明确闭合为“未执行”，不能留下缺少
            # tool result 的无效协议历史。
            for deferred_call in calls[call_index + 1 :]:
                deferred_error = ToolPipelineError(
                    "前序写操作需要人工确认，本调用未执行",
                    code="not_executed_after_approval",
                )
                await turn.close_failed_call(deferred_call, deferred_error)
            turn.active_calls = ()
            if call_index + 1 < len(calls):
                await persist_conversation(turn)
            await turn.publish(
                AgentEventType.TURN_COMPLETED,
                {
                    "status": "approval_required",
                    "plan_id": plan.plan_id,
                    "usage": turn.total_usage,
                    "model_calls": turn.model_calls,
                    "tool_calls": turn.total_tool_calls,
                },
            )
            return True
        record_result(turn, tool.name, call.arguments, result.outcome.audit_result(), recovered_failures=result.recovered_failures)
        await turn.publish_tool_result(canonical_call, result)
        if (result.tool.effect is ToolEffect.READ and not result.tool.metadata.get("kernel_utility")
                and public_result_state(result.outcome.public_content) == "success"):
            turn.observed_reads.add((result.tool.name, json.dumps(dict(result.arguments), sort_keys=True, ensure_ascii=False)))
    if resuming_read:
        turn.resume_read_done = True
    turn.active_calls = ()
    await persist_conversation(turn)
    additions = turn.discovery.consume()
    next_tools = turn.discovery.window(turn.selected_tools, additions)
    if next_tools != turn.selected_tools:
        # 只在完整工具批次之后更新Schema；同一模型批次不能猜新工具名绕过初选。
        turn.token.raise_if_cancelled()
        turn.selected_tools = next_tools
        turn.available_names = {name for tool in turn.selected_tools for name in (tool.name, tool.model_name)}
        turn.tool_definitions = tuple(tool.model_definition() for tool in turn.selected_tools)
        await turn.publish(AgentEventType.CAPABILITIES_SELECTED, {
            "tools": [tool.name for tool in turn.selected_tools],
            "count": len(turn.selected_tools), "reason": "discovery" if additions else "result_snapshot", "round": turn.round_index + 2,
        })
    return False


async def execute_group(turn: AgentTurn, calls: tuple[ModelToolCall, ...], start: int,
                        round_task: Mapping[str, Any] | None,
                        parallel: bool) -> dict[str, tuple[ToolCallContext, PipelineResult | BaseException]]:
    """准入保持原生顺序；只对独立READ并发I/O，结果与其调用上下文一起结算。"""
    width = _MAX_PARALLEL_READS if parallel and not turn.tool_context.schema_replacements else 1
    group = calls[start:start + width]
    concurrent = len(group) > 1
    jobs: list[tuple[str, ToolCallContext, asyncio.Task[PipelineResult]]] = []
    try:
        for offset, item in enumerate(group):
            turn.token.raise_if_cancelled()
            spec = turn.session.catalog.get(item.name)
            canonical = ModelToolCall(item.call_id, spec.name, item.arguments)
            turn.started_call_ids[item.call_id] = None
            if concurrent:
                turn.parallel_call_ids.add(item.call_id)
            if spec.effect is not ToolEffect.READ:
                await turn.publish_tool(AgentEventType.EFFECT_PREVIEW_STARTED, canonical)
            await turn.publish_tool(AgentEventType.TOOL_STARTED, canonical, effect=spec.effect.value)
            await persist_conversation(turn, close_pending=True)
            options = {}
            if spec.effect is ToolEffect.READ:
                receipt = _CancelledReadInline.capture(turn.lease, turn.state.conversation, start + offset, tool=spec)
                turn.returned_reads[item.call_id] = receipt
                options["_returned_read"] = receipt
            context = replace(turn.tool_context,
                current_task=round_task if spec.metadata.get("kernel_utility") else turn.state.metadata.get("task"),
                call_id=item.call_id,
                schema_replacements={} if concurrent else turn.tool_context.schema_replacements)
            admitted = asyncio.Event()
            if concurrent:
                options["_read_admitted"] = admitted
                options["_read_publication_lock"] = turn.publication_lock
            invocation = turn.session.pipeline.execute(spec.name, item.arguments, context=context, **options)
            if not concurrent:
                # 顺序路径不创建子任务，保持既有调用上下文与持锁身份。
                try:
                    return {item.call_id: (context, await invocation)}
                except ToolPipelineError as exc:
                    return {item.call_id: (context, exc)}
            job = asyncio.create_task(invocation)
            jobs.append((item.call_id, context, job))
            # 先完成本项权限/引用/配额准入，才让后项handler进入。
            waiter = asyncio.create_task(admitted.wait())
            try:
                await asyncio.wait((job, waiter), return_when=asyncio.FIRST_COMPLETED)
            finally:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
            if job.done() and not admitted.is_set():
                break  # 准入失败先按原顺序交付，不提前扩大修参范围。
        outcomes = await asyncio.gather(*(job for _, _, job in jobs), return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, (asyncio.CancelledError, StalePublicationError)):
                raise outcome
        return {call_id: (context, outcome)
                for (call_id, context, _), outcome in zip(jobs, outcomes, strict=True)}
    finally:
        for _, _, job in jobs:
            if not job.done():
                job.cancel()
        if jobs:
            await asyncio.gather(*(job for _, _, job in jobs), return_exceptions=True)
