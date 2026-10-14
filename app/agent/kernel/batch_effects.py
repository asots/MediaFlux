"""显式同步组合：一张冻结卡、一张票据，复用单操作的授权/执行/验证边界。

只有无后台任务、无预占资源且可重做只读预检的工具才能声明 batch_confirmation。
组合不是分布式事务；运行中失败保留已完成项，后续项不执行、不自动重放。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from app.agent.confirmation_contract import build_confirmation_contract
from app.agent.models import RiskLevel, ToolResult
from app.agent.public_view import public_result_state
from app.agent.public_safety import sanitize_public_text

from .capabilities import KernelToolSpec, ToolEffect
from .effects import PreparedEffect

if TYPE_CHECKING:
    from .pipeline import ToolPipeline, ToolCallContext


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def batch_tool(pipeline: ToolPipeline) -> KernelToolSpec | None:
    from .pipeline import ToolPipelineError, _invoke

    tools = [t for t in pipeline.catalog.visible() if t.metadata.get("batch_confirmation")
             and t.effect is ToolEffect.WRITE]
    if len(tools) < 2:
        return None
    allowed = {t.name for t in tools}

    def selected(name: str) -> KernelToolSpec:
        if name not in allowed:
            raise ToolPipelineError("此工具不支持同步组合确认", code="invalid_arguments")
        tool = pipeline.catalog.get(name)
        if tool.effect is not ToolEffect.WRITE or not tool.metadata.get("batch_confirmation"):
            raise ToolPipelineError("组合能力已变化，请重新预检", code="confirmation_stale")
        return tool

    async def prepare(arguments: dict, context: ToolCallContext) -> PreparedEffect:
        steps = arguments["steps"]
        if len({step["tool"] for step in steps}) != len(steps):
            raise ToolPipelineError("同一工具请合并参数，不能在组合中重复写入", code="invalid_arguments")
        frozen, previews = [], []
        for step in steps:
            tool = selected(step["tool"])
            result = await pipeline.execute(tool.name, step["arguments"], context=context, _prepare_only=True)
            prepared = result.prepared_effect
            if prepared is None:
                raise ToolPipelineError("组合中有无需变更的操作，请移除后重新预检", code="precondition_failed")
            frozen.append({"tool": tool.name, "arguments": dict(prepared.arguments or result.arguments),
                           "snapshot": prepared.snapshot_fingerprint, "metadata": dict(prepared.metadata),
                           "preview": dict(prepared.preview)})
            previews.append({"tool": tool.name, "result": dict(result.outcome.public_content)})
        effects = []
        for item in previews:
            result = item["result"]
            data = result.get("data") or {}
            changes = "；".join(item for item in data.get("effects", []) if isinstance(item, str))
            effects.append(sanitize_public_text(str(result.get("summary") or item["tool"]) + ("；" + changes if changes else ""), limit=600))
        preview = ToolResult(True, "confirmation_required", "确认后依次执行本卡列明的组合变更",
            data={"steps": previews, "effects": effects, "total": len(previews)})
        risk = (RiskLevel.LOW_WRITE if all(step["metadata"].get("risk") == "low_write" for step in frozen)
                else RiskLevel.WRITE)
        contract = build_confirmation_contract(tool_name="agent.batch", risk=risk, preview=preview)
        contract.update(action="组合变更", object=f"本卡列明的 {len(frozen)} 项操作",
                        impact="全部快照复核后按顺序执行；失败时停止后续项。不授权任何新增操作。",
                        reversibility="不是整体事务；已完成项不会因后续失败自动回滚，请按各项回执核对。")
        arguments = {"steps": frozen}
        return PreparedEffect(preview=preview.to_dict(), arguments=arguments,
                              snapshot_fingerprint=_fingerprint(arguments),
                              metadata={"risk": risk.value, "confirmation": contract})

    async def execute(arguments: dict, expected: str, context: ToolCallContext) -> ToolResult:
        if not secrets.compare_digest(_fingerprint(arguments), expected) or context.active_effect_plan is None:
            raise ToolPipelineError("组合确认内容已变化", code="confirmation_invalid")
        frozen = arguments["steps"]
        # 在第一次写入前复核全部前置状态。执行时各领域仍保留自己的原子版本检查。
        for step in frozen:
            tool = selected(step["tool"])
            resolved = await pipeline._authorize(tool, dict(step["arguments"]), context, confirmation=True)
            observed = await _invoke(tool.prepare, resolved, context)
            if not isinstance(observed, PreparedEffect) or observed.snapshot_fingerprint != step["snapshot"]:
                raise ToolPipelineError("组合中有对象已变化，本次没有执行任何写操作；请重新预检", code="confirmation_stale")
        results = []
        for index, step in enumerate(frozen):
            tool = selected(step["tool"])
            if context.cancellation.cancelled or (results and public_result_state(results[-1]["result"]) != "success"):
                results.append({"tool": tool.name, "result": {"ok": False, "status": "not_started", "summary": "未执行"}})
                continue
            parent = context.active_effect_plan
            # 后缀仅作子操作审计标识；没有子票据，也不能单独确认或重放。
            child = replace(parent, plan_id=f"{parent.plan_id}_{index + 1}", tool_name=tool.name,
                            arguments=step["arguments"], snapshot_fingerprint=step["snapshot"],
                            preview=step["preview"], metadata=step["metadata"],
                            confirmation_contract={"audit_contract": step["metadata"].get("confirmation", {})})
            started = time.monotonic()
            try:
                await asyncio.to_thread(pipeline.effect_lifecycle.substep_started, plan=child)
                value = await pipeline._run_effect(child, context=replace(context, completion_scope=None))
            except ToolPipelineError as exc:
                pipeline._notify_effect("failed", plan=child, code=exc.code, elapsed_ms=int((time.monotonic() - started) * 1000))
                value = {"ok": False, "status": "failed", "summary": str(exc), "error": exc.code}
            except Exception:
                pipeline._notify_effect("interrupted", plan=child)
                value = {"ok": False, "status": "outcome_unknown", "summary": "该项执行结果未核验，已停止后续操作；请先检查实际状态"}
            else:
                pipeline._notify_effect("completed", plan=child, value=value, elapsed_ms=int((time.monotonic() - started) * 1000))
            outcome = await pipeline._materialize_refs(pipeline.projector.project(value), context=context)
            await pipeline._commit_updates(context.lease, outcome.state_updates)
            results.append({"tool": tool.name, "result": dict(outcome.public_content)})
        done = sum(public_result_state(item["result"]) == "success" for item in results)
        return ToolResult(done == len(results), "completed" if done == len(results) else "partial",
                          f"组合变更：{done}/{len(results)} 项完成\n" + "\n".join(
                              f"{i}. {item['result'].get('summary', '请核对执行结果')}" for i, item in enumerate(results, 1)), data={"steps": results, "completed": done, "total": len(results)})

    return KernelToolSpec(
        name="agent.batch", domain="agent", effect=ToolEffect.WRITE,
        description="把本工具Schema列出的不同同步写操作合并为一张确认卡，例如加入探索收藏并保存媒体偏好。只支持已知独立目标；不要拆成逐项确认。所有项先预检，确认后逐项执行；失败停止且不自动回滚。不支持云盘后台任务、依赖前项新建对象的操作、其他未列出工具。",
        examples=("收藏这部电影并设置以后优先4K，一次确认", "组合变更 批量确认 同时收藏保存偏好"),
        input_schema={"type": "object", "required": ["steps"], "additionalProperties": False, "properties": {
            "steps": {"type": "array", "minItems": 2, "maxItems": min(4, len(tools)), "items": {
                "oneOf": [{"type": "object", "required": ["tool", "arguments"], "additionalProperties": False,
                           "properties": {"tool": {"type": "string", "enum": [t.name]}, "arguments": t.input_schema}}
                          for t in tools]}}}},
        prepare=prepare, execute_confirmed=execute,
        metadata={"source_kind": "system_state", "risk": "write"},
    )
