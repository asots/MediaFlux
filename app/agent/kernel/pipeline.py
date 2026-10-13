"""统一工具生命周期与副作用闸门。"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from contextlib import nullcontext
from typing import Any, Protocol

from app.agent.model_context_budget import RESULT_PATH_MAX, compact_tool_content, decode_tool_content
from app.agent.public_safety import sanitize_public_text
from app.concurrency import CrossLoopAsyncLock

from .capabilities import KernelToolSpec, ToolCatalog, ToolEffect
from . import error_repair
from .task_state import TaskPatch, task_tool, observation_patch
from .effects import (
    ConfirmationEffectPlanStore,
    EffectPlan,
    EffectPlanError,
    EffectPlanStore,
    PreparedEffect,
)
from .projection import DefaultProjector, ReferenceValue, ToolOutcome
from .references import InMemoryReferenceStore, ReferenceError, ReferenceStore
from .session_guard import guarded_state_call, session_scope_guard
from .state import (
    CancellationToken,
    PendingEffectGuard,
    PublicationLease,
    SessionState,
    SessionBusyError,
    SessionStateStore,
    StalePublicationError,
    StateUpdate,
    _PendingEffectTransition,
)
from .ux_selection import (
    CANDIDATE_VIEW_KEY,
    RESOURCE_KIND,
    candidate_result,
    issue_candidate_view,
    resource_candidate_items,
    resource_model_content,
)


class ToolPipelineError(RuntimeError):
    def __init__(self, message: str, *, code: str = "tool_failed", task_patch: TaskPatch | None = None, invalid_fields: tuple[str, ...] = (),
                 argument_schemas: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.task_patch = task_patch
        self.repair_ticket = ""
        self.invalid_fields = invalid_fields
        self.argument_schemas = argument_schemas


class _ReadPermitRejected(ToolPipelineError):
    """Proven pre-handler denial, issued only at the acquire boundary."""
    def __init__(self, error, tool_name):
        ToolPipelineError.__init__(self, str(error), code=error.code)
        self._rate_limit = error.rate_limit
        self.tool_name = tool_name

    @property
    def rate_limit(self):
        return self._rate_limit


class ConfirmationClaimError(ToolPipelineError):
    """未认领票据的请求错误，不代表一次已确认操作的业务终态。"""

    def __init__(self) -> None:
        super().__init__("确认计划无效、已过期或已被使用", code="confirmation_invalid")


class _SchemaCollisionError(ToolPipelineError):
    """A root-level, schema-proven collision between exactly two keys."""

    def __init__(self, message: str, keys: frozenset[str]) -> None:
        super().__init__(message, code="invalid_arguments")
        self.keys = keys


ProgressSink = Callable[[Mapping[str, Any]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ToolCallContext:
    owner: str
    session_id: str
    request_id: str
    turn_id: str
    lease: PublicationLease
    cancellation: CancellationToken
    report_progress: ProgressSink
    wait_for_completion: bool = False
    capability_search: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    selection_arguments: Mapping[str, Any] | None = None
    resource_candidate_ref: str = ""
    channel: str = "api"
    completion_scope: Any = None
    continuation_binding: Any = None
    continuation_policy: Mapping[str, Any] | None = None
    diagnostic_only: bool = False
    delivery_route: Mapping[str, Any] = field(default_factory=dict)
    confirmed_effect: PipelineResult | None = None
    current_task: dict | None = None
    user_source: str = ""
    user_history: tuple[str, ...] = ()
    user_reply_context: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = ""
    schema_replacements: dict[str, error_repair.SchemaReplacement] = field(default_factory=dict)
    reobservation_source: Mapping[str, Any] | None = field(default=None, repr=False)

    def policy_context(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "turn_id": self.turn_id,
            "generation": self.lease.generation,
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    tool: KernelToolSpec
    arguments: Mapping[str, Any]
    outcome: ToolOutcome
    effect_plan: EffectPlan | None = None
    elapsed_ms: int = 0
    task_patch: TaskPatch | None = None
    recovered_failures: tuple[str, ...] = ()


class AuthorizationPolicy(Protocol):
    async def authorize(
        self,
        tool: KernelToolSpec,
        arguments: Mapping[str, Any],
        context: ToolCallContext,
    ) -> None: ...


class RateLimiter(Protocol):
    async def acquire(
        self, *, owner: str, tool_name: str, cost: float, arguments: Mapping[str, Any]
    ) -> None: ...


class EffectLifecycle(Protocol):
    """EffectPlan 的确定性领域后置动作、审计与运行代次生命周期。"""

    def prepared(
        self,
        *,
        tool: KernelToolSpec,
        arguments: Mapping[str, Any],
        prepared: PreparedEffect,
        context: ToolCallContext,
    ) -> PreparedEffect:
        del tool, arguments, context
        return prepared

    def prepare_failed(
        self, *, prepared: PreparedEffect, context: ToolCallContext
    ) -> None:
        del prepared, context

    def completed(self, *, plan: EffectPlan, value: Any, elapsed_ms: int) -> None:
        del plan, value, elapsed_ms

    def failed(self, *, plan: EffectPlan, code: str, elapsed_ms: int) -> None:
        del plan, code, elapsed_ms

    def interrupted(self, *, plan: EffectPlan) -> None:
        del plan

    def cancelled(self, *, plan: EffectPlan) -> None:
        del plan


class NoopEffectLifecycle(EffectLifecycle):
    pass


class DefaultAuthorizationPolicy:
    async def authorize(
        self,
        tool: KernelToolSpec,
        arguments: Mapping[str, Any],
        context: ToolCallContext,
    ) -> None:
        if tool.effect is not ToolEffect.READ and not context.owner:
            raise ToolPipelineError(
                "写操作需要已登录身份", code="authorization_required"
            )
        if tool.authorize is not None:
            try:
                allowed = bool(
                    tool.authorize(
                        {
                            **context.policy_context(),
                            "arguments": dict(arguments),
                            "effect": tool.effect.value,
                        }
                    )
                )
            except Exception as exc:
                raise ToolPipelineError(
                    "工具授权检查失败", code="authorization_failed"
                ) from exc
            if not allowed:
                raise ToolPipelineError(
                    "当前身份无权使用该工具", code="authorization_denied"
                )


class InMemoryRateLimiter:
    """按 owner/tool 的轻量滑动窗口限流；生产可替换而不改变 Pipeline。"""

    def __init__(self, *, limit: int = 30, window_seconds: float = 60.0) -> None:
        self.limit = max(1, int(limit))
        self.window_seconds = max(1.0, float(window_seconds))
        self._lock = CrossLoopAsyncLock()
        self._events: dict[tuple[str, str], list[tuple[float, float]]] = {}

    async def acquire(
        self, *, owner: str, tool_name: str, cost: float, arguments: Mapping[str, Any]
    ) -> None:
        del arguments
        now = time.monotonic()
        key = (owner, tool_name)
        weighted_cost = max(0.1, float(cost or 1.0))
        async with self._lock:
            cutoff = now - self.window_seconds
            events = [
                (stamp, weight)
                for stamp, weight in self._events.get(key, ())
                if stamp > cutoff
            ]
            if sum(weight for _stamp, weight in events) + weighted_cost > self.limit:
                raise ToolPipelineError(
                    "工具调用过于频繁，请稍后重试", code="rate_limited"
                )
            events.append((now, weighted_cost))
            self._events[key] = events


_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list, tuple),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


def _json_unique_key(value: Any) -> Any:
    """Hash JSON values without conflating booleans and numbers or object order."""
    if isinstance(value, dict):
        return ("object", frozenset((key, _json_unique_key(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return ("array", tuple(_json_unique_key(item) for item in value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("number", value)
    return (type(value), value)


def _validate_json_schema(
    value: Any, schema: Mapping[str, Any], *, path: str = "arguments"
) -> None:
    def invalid(message: str) -> None:
        raise ToolPipelineError(message, code="invalid_arguments")

    schema_type = schema.get("type")
    types = [schema_type] if isinstance(schema_type, str) else schema_type
    if isinstance(types, list) and not any(
        isinstance(value, _TYPE_CHECKS.get(kind, ()))
        and not (kind in {"integer", "number"} and isinstance(value, bool))
        for kind in types
    ):
        invalid(f"{path} 类型无效")
    if "enum" in schema and not any(
        _json_unique_key(value) == _json_unique_key(option)
        for option in schema["enum"]
    ):
        invalid(f"{path} 不在允许范围内")
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0) or 0):
            invalid(f"{path} 太短")
        maximum = schema.get("maxLength")
        if maximum is not None and len(value) > int(maximum):
            invalid(f"{path} 太长")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and not re.search(pattern, value):
            invalid(f"{path} 格式无效")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if schema.get("minimum") is not None and value < schema["minimum"]:
            invalid(f"{path} 小于允许值")
        if schema.get("maximum") is not None and value > schema["maximum"]:
            invalid(f"{path} 超过允许值")
    if isinstance(value, dict):
        required = schema.get("required") or ()
        for key in required:
            if key not in value:
                invalid(f"缺少必需参数：{key}")
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            unexpected = set(value).difference(properties)
            if unexpected:
                invalid(f"包含未知参数：{min(unexpected)}")
        for key, item in value.items():
            child = properties.get(key)
            if isinstance(child, Mapping):
                _validate_json_schema(item, child, path=f"{path}.{key}")
        if schema.get("minProperties") is not None and len(value) < schema["minProperties"]:
            invalid(f"{path} 参数数量少于允许值 {schema['minProperties']}")
        if schema.get("maxProperties") is not None and len(value) > schema["maxProperties"]:
            if (path == "arguments" and schema["maxProperties"] == 1 and len(value) == 2
                    and all(key in properties and item is not None for key, item in value.items())):
                # Equivalent to the existing two-key presence exclusion: allow
                # omission repair, but retain required keys and older value locks.
                raise _SchemaCollisionError(
                    f"{path}：" + "、".join(sorted(value)) + " 只能提供其中一个", frozenset(value),
                )
            invalid(f"{path} 参数数量超过允许值 {schema['maxProperties']}")
    if isinstance(value, (list, tuple)):
        if isinstance(schema.get("items"), Mapping):
            for index, item in enumerate(value):
                _validate_json_schema(item, schema["items"], path=f"{path}[{index}]")
        if schema.get("minItems") is not None and len(value) < schema["minItems"]:
            invalid(f"{path} 项数少于允许值 {schema['minItems']}")
        if schema.get("maxItems") is not None and len(value) > schema["maxItems"]:
            invalid(f"{path} 项数超过允许值 {schema['maxItems']}")
        if schema.get("uniqueItems") is True:
            seen = set()
            for index, item in enumerate(value):
                key = _json_unique_key(item)
                if key in seen:
                    invalid(f"{path}[{index}] 与此前项重复")
                seen.add(key)

    def matches(branch: Mapping[str, Any]) -> bool:
        try:
            _validate_json_schema(value, branch, path=path)
        except ToolPipelineError:
            return False
        return True

    def invalid_combination(message: str, groups: Any) -> None:
        # Only pure two-key presence exclusions are omission-repairable. Nested
        # and value-dependent branches remain fail-closed, not inferred from text.
        if path == "arguments" and isinstance(value, dict) and isinstance(groups, list):
            if len(groups) == 2 and all(
                isinstance(branch, Mapping) and set(branch) == {"required"}
                and isinstance(branch["required"], list) and len(branch["required"]) == 1
                and isinstance(branch["required"][0], str) for branch in groups
            ):
                keys = frozenset(branch["required"][0] for branch in groups)
                if len(keys) == 2 and all(value.get(key) is not None for key in keys):
                    raise _SchemaCollisionError(
                        message + "：" + "、".join(sorted(keys)) + " 只能提供其中一个", keys,
                    )
        invalid(message)

    for keyword in ("oneOf", "anyOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            matched = sum(matches(branch) for branch in branches)
            if (keyword == "oneOf" and matched != 1) or (keyword == "anyOf" and not matched):
                invalid_combination(f"{path} 不符合 {keyword} 参数组合约束",
                                    branches if keyword == "oneOf" else None)
    forbidden = schema.get("not")
    if isinstance(forbidden, Mapping) and matches(forbidden):
        required = forbidden.get("required")
        groups = ([{"required": [key]} for key in required]
                  if set(forbidden) == {"required"} and isinstance(required, list) else None)
        if groups is None and path == "arguments" and isinstance(value, dict):
            def involved_fields(branch):
                fields = set(branch.get("required", ())) | set(branch.get("properties", {}))
                for keyword in ("allOf", "anyOf", "oneOf"):
                    for child in branch.get(keyword, ()):
                        if matches(child):
                            fields.update(involved_fields(child))
                return fields & value.keys() & schema.get("properties", {}).keys()
            fields = tuple(sorted(involved_fields(forbidden)))
            raise ToolPipelineError(f"{path} 不允许此参数组合（not）；相关字段：" + "、".join(fields),
                                    code="invalid_arguments", invalid_fields=fields)
        invalid_combination(f"{path} 不允许此参数组合（not）", groups)


async def _invoke(
    handler: Callable[..., Any], *args: Any, protection: ToolCallContext | None = None,
) -> Any:
    try:
        if inspect.iscoroutinefunction(handler):
            return await handler(*args)
        value = (
            await guarded_state_call(
                protection.owner, protection.session_id, handler, *args,
                kind="effect", complete_on_cancel=True,
            )
            if protection is not None else await asyncio.to_thread(handler, *args)
        )
        if inspect.isawaitable(value):
            return await value
        return value
    except ToolPipelineError:
        raise
    except Exception as exc:
        raise ToolPipelineError("工具执行失败", code="tool_execution_failed") from exc


class ToolPipeline:
    """所有 READ、WRITE 与 DANGER 工具唯一允许经过的生命周期。"""

    def __init__(
        self,
        *,
        catalog: ToolCatalog,
        state_store: SessionStateStore,
        reference_store: ReferenceStore | None = None,
        effect_store: EffectPlanStore | None = None,
        projector: DefaultProjector | None = None,
        authorization: AuthorizationPolicy | None = None,
        rate_limiter: RateLimiter | None = None,
        effect_lifecycle: EffectLifecycle | None = None,
    ) -> None:
        self.catalog = catalog
        self.state_store = state_store
        self.reference_store = reference_store or InMemoryReferenceStore()
        self.effect_store = effect_store or ConfirmationEffectPlanStore()
        self.projector = projector or DefaultProjector()
        self.authorization = authorization or DefaultAuthorizationPolicy()
        self.rate_limiter = rate_limiter or InMemoryRateLimiter()
        self.effect_lifecycle = effect_lifecycle or NoopEffectLifecycle()
        if not catalog.has(error_repair.REPAIR_TOOL):
            catalog.register(KernelToolSpec(
                name=error_repair.REPAIR_TOOL, domain="agent", effect=ToolEffect.READ,
                description="明确替代本轮指定的参数拒绝调用，不是声称旧目标已知。使用失败返回的一次性ticket，提供原工具完整修正arguments；只修改错误涉及的字段，保留其他有效参数。仍需原权限与写入确认，预览/提交不代表业务完成。",
                input_schema={"type": "object", "required": ["ticket", "arguments"], "properties": {
                    "ticket": {"type": "string"}, "arguments": {"type": "object"}}, "additionalProperties": False},
                read=self._replace_schema_failure, availability=lambda context: False,
                metadata={"kernel_utility": True},
            ))
        if not catalog.has("agent.task"):
            catalog.register(task_tool())
        if not catalog.has("agent.read_result"):
            catalog.register(KernelToolSpec(
                name="agent.read_result", domain="agent", effect=ToolEffect.READ,
                description="续取同一用户会话的冻结脱敏快照，不重新执行业务。handle=result_handle；path=truncation.path（原快照JSON Pointer，空串为根）；offset=next_offset。total为原节点长度，returned仅计当前直接项/字符，不保证子项完整。scope=subtree表示祖先覆盖所有省略，须从其next_offset重读，用更小limit并按字典key或数组原索引下钻；否则续接前缀。页data.next_offset仅翻同级页；有truncation必须另取子项，即使data.has_more=false。字符串偏移按Unicode字符。元数据或键无法安全分页时报context_budget_exceeded，不应原参数重试。",
                input_schema={"type": "object", "required": ["handle", "path"], "properties": {
                    "handle": {"type": "string", "minLength": 1, "maxLength": 200},
                    "path": {"type": "string", "pattern": "^(/.*)?$", "maxLength": RESULT_PATH_MAX},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                }, "additionalProperties": False},
                read=self._read_result, metadata={"kernel_utility": True},
            ))

    async def _replace_schema_failure(self, arguments: dict[str, Any], context: ToolCallContext) -> PipelineResult:
        entry = context.schema_replacements.get(arguments["ticket"])
        if entry is None or not entry.offered or entry.scope != error_repair.scope(context):
            raise ToolPipelineError("修复入口无效或不属于本轮", code="repair_invalid")
        replacement = arguments["arguments"]
        if not error_repair.preserves(entry, replacement):
            raise ToolPipelineError("明确修复不能更改原有效参数", code="repair_conflict")
        try:
            error_repair.require_presence((entry,), replacement, ToolPipelineError)
        except ToolPipelineError as exc:
            exc.repair_ticket = arguments["ticket"]
            raise
        context.schema_replacements.pop(arguments["ticket"])
        lineage = (*entry.predecessors, arguments["ticket"])
        try:
            result = await self.execute(entry.tool, replacement, context=context)
        except ToolPipelineError as exc:
            next_entry = context.schema_replacements.get(exc.repair_ticket)
            if next_entry is not None:
                context.schema_replacements[exc.repair_ticket] = replace(next_entry, predecessors=lineage)
            raise
        content = result.outcome.public_content
        if result.effect_plan is None and error_repair.completed_result(result.tool, content):
            return replace(result, recovered_failures=tuple(dict.fromkeys((*lineage, *result.recovered_failures))))
        return result

    async def _read_result(self, arguments: dict[str, Any], context: ToolCallContext) -> ToolOutcome:
        from . import telegram_reference
        handle = arguments["handle"]
        reference = None
        try:
            manifest = await self.reference_store.resolve(handle, owner=context.owner, session_id=context.session_id, expected_kind="tool_result")
            parts = [await self.reference_store.resolve(part, owner=context.owner, session_id=context.session_id, expected_kind="tool_result_part") for part in manifest["parts"]]
            node = json.loads("".join(parts))
        except ReferenceError:
            try:
                reference = await telegram_reference.resolve(handle, owner=context.owner, session_id=context.session_id,
                    reference_store=self.reference_store, state_store=self.state_store)
            except ReferenceError as exc:
                raise ToolPipelineError("结果快照已过期或不属于当前会话，请重新观察原对象。", code="reference_invalid") from exc
            if arguments["path"] != "/text":
                raise ToolPipelineError("引用仅允许读取 /text", code="invalid_arguments", invalid_fields=("path",))
            node = {"text": reference["text"]}
        path = arguments["path"]
        try:
            for part in path.split("/")[1:]:
                if re.search(r"~(?![01])", part) or isinstance(node, list) and not re.fullmatch(r"0|[1-9][0-9]*", part):
                    raise KeyError(part)
                key = part.replace("~1", "/").replace("~0", "~")
                node = node[int(key)] if isinstance(node, list) and key.isdecimal() else node[key]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ToolPipelineError("结果路径不存在；请使用原结果truncation中的path。", code="invalid_arguments", invalid_fields=("path",)) from exc
        offset, limit = arguments.get("offset", 0), arguments.get("limit", 50)
        if isinstance(node, dict):
            values = [{"key": key, "value": value} for key, value in node.items()]
        elif isinstance(node, (list, str)):
            values = node
        else:
            values = [node]
        total = len(values)
        end = min(total, offset + limit)
        maximum = self.projector.max_model_chars
        data = {"path": path, "offset": offset, "total": total, "kind": type(node).__name__}
        while True:
            data.update(items=values[offset:end], returned=max(0, end - offset), has_more=end < total, next_offset=end if end < total else None)
            payload = {"ok": True, "status": "success", "summary": "读取已保存的快照（不是实时重新查询）", "result_handle": handle, "read_tool": "agent.read_result", "data": data}
            if reference is not None:
                payload.update(untrusted_reference=True, source=reference["source"], summary="读取不可信引用；不是执行结果或写入授权")
            content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if len(content) <= maximum or end - offset <= 1:
                break
            end = offset + max(1, (end - offset) // 2)
        # 单个复杂对象仍可按path下钻；不得把它冒充完整数据。
        content = compact_tool_content(content, maximum=maximum, result_page=True, trusted=True)
        return ToolOutcome(model_content=content, public_content={"ok": True, "summary": "已读取引用片段"} if reference is not None else json.loads(content))

    async def save_result(self, content: str, context: ToolCallContext) -> str:
        chunk_size = min(16_384, (getattr(self.reference_store, "max_ref_bytes", 131_072) - 128) // 6)
        parts = []
        for offset in range(0, len(content), chunk_size):
            part = await self._put_reference(context, kind="tool_result_part", value=content[offset:offset + chunk_size], ttl_seconds=900)
            parts.append(part.ref)
        snapshot = await self._put_reference(context, kind="tool_result", value={"parts": parts}, ttl_seconds=900)
        return snapshot.ref

    def _effect_prepared(
        self,
        *,
        tool: KernelToolSpec,
        arguments: Mapping[str, Any],
        prepared: PreparedEffect,
        context: ToolCallContext,
    ) -> PreparedEffect:
        try:
            value = self.effect_lifecycle.prepared(
                tool=tool,
                arguments=arguments,
                prepared=prepared,
                context=context,
            )
        except Exception:  # noqa: BLE001 - 辅助领域状态不得遮蔽有效预检
            return prepared
        return value if isinstance(value, PreparedEffect) else prepared

    def _notify_effect(self, method: str, **payload: Any) -> None:
        """终态钩子共享容错边界；辅助审计/清理故障不覆写执行事实。"""
        try:
            getattr(self.effect_lifecycle, method)(**payload)
        except Exception:  # noqa: BLE001 - hooks cannot change authoritative outcome
            return

    async def _discard_frozen_effect(
        self,
        *,
        plan: EffectPlan,
        prepared: PreparedEffect,
        context: ToolCallContext,
    ) -> None:
        """预览发布失败时撤销已冻结票据并释放领域预占状态。"""
        cancelled: EffectPlan | None = None
        try:
            cancelled = await asyncio.to_thread(
                self.effect_store.cancel,
                owner=context.owner,
                session_id=context.session_id,
                generation=context.lease.generation,
                plan_id=plan.plan_id,
            )
        except Exception:  # noqa: BLE001 - 原始发布失败仍是主错误
            cancelled = None
        if cancelled is not None:
            self._notify_effect("cancelled", plan=cancelled)
        else:
            self._notify_effect("prepare_failed", prepared=prepared, context=context)

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any] | None,
        *,
        context: ToolCallContext,
        _returned_read=None,
        _read_admitted: asyncio.Event | None = None,
        _read_publication_lock: asyncio.Lock | None = None,
    ) -> PipelineResult:
        started = time.monotonic()
        context.cancellation.raise_if_cancelled()
        try:
            tool = self.catalog.get(name)
        except KeyError as exc:
            raise ToolPipelineError("未知工具", code="tool_not_found") from exc
        if context.diagnostic_only and tool.effect is not ToolEffect.READ:
            raise ToolPipelineError("前序写操作未能完整核验；请先用只读业务工具核对真实状态，再为后续动作重新预检。新计划仍需确认，不能重放旧写入。", code="effect_diagnostic_only")
        raw_arguments = dict(arguments or {})
        if tool.name == error_repair.REPAIR_TOOL:
            _validate_json_schema(raw_arguments, tool.input_schema)
            return await self._replace_schema_failure(raw_arguments, context)
        if (
            context.resource_candidate_ref and tool.name == "ingest.submit"
            and raw_arguments.get("source_type") == RESOURCE_KIND
            and raw_arguments.get("resource_candidates_ref") != context.resource_candidate_ref
        ):
            raise ToolPipelineError("请从回复消息所属的候选批次选择，不可替换为其他搜索编号", code="candidate_scope_mismatch")
        if context.selection_arguments is not None:
            expected = dict(context.selection_arguments)
            # 卡片选择限定为既有提交工具的预检；模型不能换资源、扩选或另选目标。
            actual = {**raw_arguments, "target": raw_arguments.get("target", "preferred")}
            positions = raw_arguments.get("positions")
            if (
                tool.name != "ingest.submit"
                or tool.effect is ToolEffect.READ
                or actual != expected
                or not isinstance(positions, list)
                or any(type(position) is not int for position in positions)
            ):
                raise ToolPipelineError("只能预览本次已验证的候选选择", code="selection_mismatch")
        pending_repairs = error_repair.pending(tool, raw_arguments, context)
        error_repair.require_presence(pending_repairs.values(), raw_arguments, ToolPipelineError)
        try:
            _validate_json_schema(raw_arguments, tool.input_schema)
        except ToolPipelineError as exc:
            collision = (exc.keys if isinstance(exc, _SchemaCollisionError) else
                         frozenset(exc.invalid_fields) if tool.effect is ToolEffect.READ else frozenset())
            exc.repair_ticket = error_repair.issue(tool, raw_arguments, context, _validate_json_schema,
                                                  ToolPipelineError, collisions=collision, preserve_fields=True)
            raise
        try:
            normalized = tool.validator(raw_arguments)
        except ToolPipelineError as exc:
            if exc.code == "invalid_arguments":
                exc.repair_ticket = error_repair.issue(tool, raw_arguments, context, _validate_json_schema, ToolPipelineError, invalid_fields=exc.invalid_fields, argument_schemas=exc.argument_schemas)
            raise
        except Exception as exc:
            safe_message = sanitize_public_text(
                getattr(exc, "safe_message", ""),
                limit=500,
            )
            error = ToolPipelineError(safe_message or "工具参数无效", code="invalid_arguments")
            raise error from exc
        if not isinstance(normalized, dict):
            raise ToolPipelineError(
                "工具参数校验器返回无效结果", code="invalid_arguments"
            )
        error_repair.validated(pending_repairs, context)
        try:
            async with (_read_publication_lock if _read_publication_lock is not None else nullcontext()):
                resolved = await self._authorize(tool, normalized, context)
        except ToolPipelineError as exc:
            if exc.code == "invalid_arguments":
                exc.repair_ticket = error_repair.issue(tool, raw_arguments, context, _validate_json_schema,
                    ToolPipelineError, invalid_fields=exc.invalid_fields, argument_schemas=exc.argument_schemas)
            raise
        completed = context.confirmed_effect
        plan = completed.effect_plan if completed else None
        if (
            completed and plan and tool.effect is not ToolEffect.READ
            and (plan.owner, plan.session_id) == (context.owner, context.session_id)
            and tool.name == completed.tool.name and normalized == completed.arguments
            and completed.outcome.public_content.get("ok") is not False
        ):
            authoritative = await self.state_store.load(owner=context.owner, session_id=context.session_id)
            if not await self.state_store.is_current(context.lease):
                raise StalePublicationError("turn lost publication authority")
            if authoritative.metadata.get("confirmed_publication") == {
                "generation": context.lease.generation, "turn_id": context.lease.turn_id, "plan_id": plan.plan_id,
            }:
                # 只回读本次确认的事实，不重放写操作、票据、引用或状态更新。
                outcome = replace(completed.outcome, refs=(), state_updates=(), effect_plan=None,
                                  model_content="本轮该操作已确认执行；以下为原回执，不是再次执行。\n" + completed.outcome.model_message())
                return PipelineResult(tool=tool, arguments=normalized, outcome=outcome)

        if tool.effect is ToolEffect.READ:
            if tool.read is None:  # pragma: no cover - ToolSpec 已校验
                raise ToolPipelineError("工具不可执行", code="tool_not_executable")
            resource_presentation = tool.metadata.get("source_kind") == "resource_presentation"
            resource_search = resource_presentation or tool.metadata.get("source_kind") == "resource_index"
            if resource_search:
                # 同一模型回合内重搜也使旧卡失效，包括空结果和读取失败。
                await self._commit_updates(context.lease, (
                    StateUpdate(f"metadata.{CANDIDATE_VIEW_KEY}", None),
                ))
                await context.report_progress({"tool": tool.name, "candidate_view": None})
            if _read_admitted is not None:
                _read_admitted.set()
            try:
                value = await _invoke(tool.read, resolved, context)
            except ToolPipelineError as exc:
                if exc.code == "invalid_arguments":
                    exc.repair_ticket = error_repair.issue(tool, raw_arguments, context, _validate_json_schema, ToolPipelineError, invalid_fields=exc.invalid_fields, argument_schemas=exc.argument_schemas)
                if tool.observe is not None:
                    exc.task_patch = observation_patch(context.current_task, owner=context.owner,
                        session_id=context.session_id, tool=tool.name, call_id=context.call_id,
                        observations=tool.observe(None, normalized))
                raise
            patch = value if isinstance(value, TaskPatch) else None
            if tool.observe is not None and context.current_task is not None:
                observations = tool.observe(value, normalized)
                patch = observation_patch(context.current_task, owner=context.owner,
                    session_id=context.session_id, tool=tool.name, call_id=context.call_id,
                    observations=observations)
                if patch is not None and any(fact is None for fact in patch.payload["observations"].values()):
                    raise ToolPipelineError("配置观察结果不符合证据合同。", code="invalid_observation", task_patch=patch)
            if isinstance(value, TaskPatch):
                value = {"ok": True, "summary": "Task proposal"}
            projected = self.projector.project(value)
            if _returned_read is not None:
                from .state import _CancelledReadInline
                from .projection import _model_safe
                # Direct compact ToolOutcome envelopes do not prove a full source.
                if type(_returned_read) is _CancelledReadInline and _returned_read.tool is tool:
                    if patch is None and (not isinstance(value, ToolOutcome) or projected.full_model_content):
                        full = projected.full_model_content or projected.model_content
                        _returned_read.put_once(json.dumps(_model_safe(json.loads(full)), ensure_ascii=False))
                    else:
                        _returned_read.reject_incomplete()
            # Later Session/history pressure can reduce a result to 400 chars.
            # Freeze this already-safe source before that loss, using the same
            # scoped TTL store. Reader pages already point at their source:
            # never snapshot them again or replace their original coordinates.
            snapshot_only = tool.name != "agent.read_result" and not projected.full_model_content and len(projected.model_content) > 400
            if snapshot_only:
                projected = replace(projected, full_model_content=projected.model_content)
            if resource_search and not any(item.kind == RESOURCE_KIND for item in projected.refs):
                public = candidate_result(projected.public_content, None)
                projected = replace(
                    projected, public_content=public,
                    model_content=resource_model_content(
                        projected.model_content, maximum=self.projector.max_model_chars,
                    ),
                    full_model_content=resource_model_content(projected.full_model_content, maximum=max(len(projected.full_model_content) * 2, self.projector.max_model_chars)) if projected.full_model_content else "",
                )
            # 只串行化同一回合的引用/状态发布，不持锁等待远端只读I/O。
            async with (_read_publication_lock if _read_publication_lock is not None else nullcontext()):
                outcome = await self._materialize_refs(
                    projected, context=context,
                    selected_positions=normalized["positions"] if resource_presentation else None,
                    preserve_native=snapshot_only,
                )
                await self._commit_updates(context.lease, outcome.state_updates)
            return PipelineResult(
                tool=tool,
                arguments=normalized,
                outcome=outcome,
                elapsed_ms=max(0, int((time.monotonic() - started) * 1000)),
                task_patch=patch,
                recovered_failures=error_repair.completed(tool, raw_arguments, context, outcome.public_content),
            )

        if tool.prepare is None:  # pragma: no cover - ToolSpec 已校验
            raise ToolPipelineError("工具不支持预检", code="confirmation_not_supported")
        prepared_value = await _invoke(tool.prepare, resolved, context)
        if isinstance(prepared_value, ToolOutcome):
            if prepared_value.public_content.get("ok") is not True or prepared_value.public_content.get("status") != "no_changes":
                raise ToolPipelineError("预检只能直接返回无需变更的结果", code="invalid_effect_preview")
            outcome = await self._materialize_refs(self.projector.project(prepared_value), context=context)
            await self._commit_updates(context.lease, outcome.state_updates)
            return PipelineResult(tool=tool, arguments=normalized, outcome=outcome,
                                  elapsed_ms=max(0, int((time.monotonic() - started) * 1000)),
                                  recovered_failures=error_repair.completed(tool, raw_arguments, context, outcome.public_content))
        if not isinstance(prepared_value, PreparedEffect):
            raise ToolPipelineError(
                "工具预检返回无效结果", code="invalid_effect_preview"
            )
        if not prepared_value.snapshot_fingerprint:
            raise ToolPipelineError(
                "工具预检缺少快照指纹", code="invalid_effect_preview"
            )
        prepared_value = self._effect_prepared(
            tool=tool,
            arguments=normalized,
            prepared=prepared_value,
            context=context,
        )
        try:
            with session_scope_guard(context.owner, context.session_id):
                if not await self.state_store.is_current(context.lease):
                    self._notify_effect("prepare_failed", prepared=prepared_value, context=context)
                    raise StalePublicationError("turn lost publication authority")
                try:
                    preview_outcome = await self._materialize_refs(
                        self.projector.project(prepared_value.preview), context=context
                    )
                    plan = await asyncio.to_thread(
                        self.effect_store.freeze,
                        owner=context.owner,
                        session_id=context.session_id,
                        generation=context.lease.generation,
                        tool_name=tool.name,
                        effect=tool.effect,
                        arguments=normalized,
                        prepared=prepared_value,
                        public_result=preview_outcome.public_content,
                    )
                except BaseException:
                    self._notify_effect("prepare_failed", prepared=prepared_value, context=context)
                    raise
                try:
                    updates = tuple(preview_outcome.state_updates) + (
                        _PendingEffectTransition(
                            "pending_effect_plan_id", plan.plan_id, plan_generation=plan.generation,
                            goal=({"role": "user", "content": context.user_source,
                                   **({"reply_context": dict(context.user_reply_context)} if context.user_reply_context else {})}
                                  if context.user_source else {}),
                        ),
                    )
                    if tool.name == "ingest.submit" and raw_arguments.get("source_type") == RESOURCE_KIND:
                        updates += (StateUpdate("metadata.ux_candidate_plan", {
                            "plan_id": plan.plan_id, "ref": raw_arguments.get("resource_candidates_ref"),
                        }),)
                    outcome = replace(preview_outcome, state_updates=updates, effect_plan=plan)
                    await self._commit_updates(context.lease, updates)
                except BaseException:
                    await self._discard_frozen_effect(
                        plan=plan,
                        prepared=prepared_value,
                        context=context,
                    )
                    raise
                return PipelineResult(
                    tool=tool,
                    arguments=normalized,
                    outcome=outcome,
                    effect_plan=plan,
                    elapsed_ms=max(0, int((time.monotonic() - started) * 1000)),
                )
        except SessionBusyError:
            self._notify_effect("prepare_failed", prepared=prepared_value, context=context)
            raise

    @staticmethod
    def _pending_generation(state: SessionState) -> int | None:
        binding = state.metadata.get("pending_effect_binding")
        if binding is None:
            return state.generation
        if (isinstance(binding, Mapping) and set(binding) == {"plan_id", "plan_generation"}
                and binding.get("plan_id") == state.pending_effect_plan_id
                and type(binding.get("plan_generation")) is int
                and 0 < binding["plan_generation"] <= state.generation):
            return binding["plan_generation"]
        return None

    async def get_pending_plan(self, *, owner: str, session_id: str, scoped: bool = False) -> EffectPlan | None:
        with nullcontext() if scoped else session_scope_guard(owner, session_id):
            state = await self.state_store.load(owner=owner, session_id=session_id)
            generation = self._pending_generation(state)
            if not state.pending_effect_plan_id or generation is None:
                return None
            return await asyncio.to_thread(self.effect_store.get_active_plan,
                owner=owner, session_id=session_id, generation=generation, plan_id=state.pending_effect_plan_id)

    def pending_guard(self, state: SessionState, plan: EffectPlan) -> PendingEffectGuard:
        clock = getattr(getattr(self.effect_store, "store", None), "_clock", time.time)
        return PendingEffectGuard(state.generation, plan.plan_id, plan.generation, plan.expires_at, clock)

    async def execute_confirmed(
        self, plan_id: str, *, context: ToolCallContext,
    ) -> PipelineResult:
        with session_scope_guard(context.owner, context.session_id, kind="effect"):
            return await self._execute_confirmed_scoped(plan_id, context=context)

    async def _execute_confirmed_scoped(
        self,
        plan_id: str,
        *,
        context: ToolCallContext,
    ) -> PipelineResult:
        started = time.monotonic()
        context.cancellation.raise_if_cancelled()
        current = await self.state_store.load(owner=context.owner, session_id=context.session_id)
        if current.generation != context.lease.generation:
            raise StalePublicationError("turn lost publication authority")
        pending = await self.get_pending_plan(owner=context.owner, session_id=context.session_id, scoped=True)
        if pending is None or pending.plan_id != plan_id:
            await self._commit_updates(context.lease, (
                _PendingEffectTransition("pending_effect_plan_id", plan_id, mode="clear_if_equals"),
            ))
            raise ConfirmationClaimError()
        try:
            plan = await asyncio.to_thread(
                self.effect_store.claim,
                owner=context.owner,
                session_id=context.session_id,
                generation=pending.generation,
                plan_id=plan_id,
            )
        except ToolPipelineError:
            raise
        except Exception as exc:
            # 只有确定失效的当前计划才清理；存储故障不能撤销仍可确认的计划。
            if isinstance(exc, EffectPlanError) and current.pending_effect_plan_id == plan_id:
                await self._commit_updates(context.lease, (
                    _PendingEffectTransition("pending_effect_plan_id", plan_id, mode="clear_if_equals"),
                ))
            raise ConfirmationClaimError() from exc

        from app.agent.effect_completion import EffectCompletionScope, register_terminal_failure
        from .continuation import effect_origin

        completion_scope = None

        async def remember_failure(result):
            if completion_scope is None:
                return
            try:
                await register_terminal_failure(completion_scope, tool=plan.tool_name, result=result)
            except Exception as exc:  # 登记故障不能覆盖已经成立的原始失败事实/code。
                from app.logger import get_logger
                get_logger(__name__).warning("确认失败续行登记未完成 type=%s", type(exc).__name__)

        try:
            # 只有真实领到票据才接管同 generation 的发布权。该标记在原状态
            # JSON 内持久化，使锁释放/进程结束后旧读回合仍无法覆盖确认回执。
            await self.state_store.commit(
                context.lease,
                updates=(
                    StateUpdate(
                        "metadata.confirmed_publication",
                        {
                            "generation": context.lease.generation,
                            "turn_id": context.lease.turn_id,
                            "plan_id": plan.plan_id,
                        },
                    ),
                    _PendingEffectTransition(
                        "pending_effect_plan_id",
                        plan.plan_id,
                        mode="clear_if_equals",
                    ),
                ),
            )
            await context.report_progress({
                "plan_id": plan_id, "kind": "confirmed_effect", "tool": plan.tool_name,
            })
            try:
                tool = self.catalog.get(plan.tool_name)
            except KeyError as exc:
                raise ToolPipelineError(
                    "确认计划对应工具不存在", code="tool_not_found"
                ) from exc
            if tool.effect is ToolEffect.READ or tool.effect is not plan.effect:
                raise ToolPipelineError(
                    "确认计划风险类型不匹配", code="confirmation_invalid"
                )
            if tool.execute_confirmed is None:  # pragma: no cover - ToolSpec 已校验
                raise ToolPipelineError(
                    "工具不支持确认执行", code="confirmation_not_supported"
                )
            completion_scope = EffectCompletionScope(
                store=self.state_store, lease=context.lease,
                plan_id=plan.plan_id, channel=context.channel,
                origin=effect_origin(current, context, plan),
            )
            context = replace(context, completion_scope=completion_scope)
            resolved_arguments = await self._authorize(tool, dict(plan.arguments), context, confirmation=True)
            value = await _invoke(
                tool.execute_confirmed,
                resolved_arguments,
                plan.snapshot_fingerprint,
                context,
                protection=context,
            )
            if tool.verify is not None:
                verified = await _invoke(
                    tool.verify,
                    resolved_arguments,
                    value,
                    context,
                    protection=context,
                )
                if verified is False:
                    raise ToolPipelineError(
                        "写后验证失败", code="post_write_verification_failed"
                    )
                if verified is not True and verified is not None:
                    value = verified
        except ToolPipelineError as exc:
            self._notify_effect(
                "failed", plan=plan, code=exc.code,
                elapsed_ms=max(0, int((time.monotonic() - started) * 1000)),
            )
            await remember_failure({"ok": False, "status": exc.code, "summary": str(exc)})
            raise
        except BaseException:
            self._notify_effect("interrupted", plan=plan)
            raise

        elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
        # 写操作与写后验证已取得可信终态。先收束审计并推进全局运行代次；
        # 后续 DTO 投影、引用或会话提交失败，不得把已发生的副作用伪装成未执行。
        self._notify_effect("completed", plan=plan, value=value, elapsed_ms=elapsed_ms)
        outcome = await self._materialize_refs(
            self.projector.project(value), context=context
        )
        updates = tuple(outcome.state_updates)
        outcome = replace(outcome, effect_plan=plan)
        await self._commit_updates(context.lease, updates)
        await remember_failure(outcome.public_content)
        return PipelineResult(
            tool=tool,
            arguments=dict(plan.arguments),
            outcome=outcome,
            effect_plan=plan,
            elapsed_ms=elapsed_ms,
        )

    async def cancel_effect(
        self, plan_id: str, *, lease: PublicationLease,
    ) -> bool:
        with session_scope_guard(lease.owner, lease.session_id, kind="effect"):
            return await self._cancel_effect_scoped(plan_id, lease=lease)

    async def _cancel_effect_scoped(self, plan_id: str, *, lease: PublicationLease) -> bool:
        current = await self.state_store.load(owner=lease.owner, session_id=lease.session_id)
        if current.generation != lease.generation:
            raise StalePublicationError("turn lost publication authority")
        if current.pending_effect_plan_id != plan_id:
            return False
        generation = self._pending_generation(current)
        cancelled_plan = await asyncio.to_thread(
            self.effect_store.cancel, owner=lease.owner, session_id=lease.session_id,
            generation=generation if generation is not None else current.generation, plan_id=plan_id,
        )
        if cancelled_plan is not None:
            self._notify_effect("cancelled", plan=cancelled_plan)
        await self._commit_updates(lease, (
            _PendingEffectTransition("pending_effect_plan_id", plan_id, mode="clear_if_equals"),
        ))
        return cancelled_plan is not None

    async def _authorize(
        self, tool: KernelToolSpec, arguments: dict[str, Any], context: ToolCallContext, *, confirmation: bool = False,
    ) -> dict[str, Any]:
        try:
            resolved = await self._resolve_references(arguments, context=context)
        except ReferenceError as exc:
            raise ToolPipelineError(
                "确认计划引用无效、已过期或不属于当前会话" if confirmation else "工具引用无效、已过期或不属于当前会话",
                code="confirmation_stale" if confirmation else "reference_invalid",
            ) from exc
        await self.authorization.authorize(tool, resolved, context)
        from .ports.mediaflux_policy import ToolRateLimitError
        try:
            await self.rate_limiter.acquire(
                owner=context.owner, tool_name=f"confirm:{tool.name}" if confirmation else tool.name,
                cost=max(1.0, tool.cost) if confirmation else tool.cost, arguments=resolved,
            )
        except ToolRateLimitError as exc:
            if not confirmation and tool.effect is ToolEffect.READ and not tool.name.startswith("agent."):
                class RejectedReadPermit(_ReadPermitRejected, ToolRateLimitError):
                    pass
                raise RejectedReadPermit(exc, tool.name) from exc
            raise
        context.cancellation.raise_if_cancelled()
        # A preview cannot reenter a held confirmation, even with an old lease.
        # Reject busy before stale so the confirmed receipt can finish settling.
        scope = (session_scope_guard(context.owner, context.session_id)
                 if not confirmation and tool.effect is not ToolEffect.READ else nullcontext())
        with scope:
            if context.lease.continuation_id:
                # Same-connection fence check linearizes handler start after acquire.
                await self.state_store.commit(context.lease)
            elif not await self.state_store.is_current(context.lease):
                raise StalePublicationError("turn revoked before handler")
        return resolved

    async def _resolve_references(
        self,
        value: Any,
        *,
        context: ToolCallContext,
    ) -> Any:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for child_key, child_value in value.items():
                if (
                    child_key.endswith("_ref")
                    and isinstance(child_value, str)
                    and child_value.startswith("ref_")
                ):
                    expected_kind = child_key[:-4].strip("_").casefold()
                    result[
                        child_key[:-4] or child_key
                    ] = await self.reference_store.resolve(
                        child_value,
                        owner=context.owner,
                        session_id=context.session_id,
                        expected_kind=expected_kind,
                    )
                else:
                    result[child_key] = await self._resolve_references(child_value, context=context)
            return result
        if isinstance(value, list):
            return [
                await self._resolve_references(item, context=context)
                for item in value
            ]
        return value


    async def _put_reference(self, context, **kwargs):
        put = getattr(self.reference_store, "put_for_turn", None)
        if put is not None:
            return await put(context.lease, owner=context.owner, session_id=context.session_id, **kwargs)
        if not await self.state_store.is_current(context.lease):
            raise StalePublicationError("turn revoked before reference")
        return await self.reference_store.put(owner=context.owner, session_id=context.session_id, **kwargs)

    async def _materialize_refs(
        self,
        outcome: ToolOutcome,
        *,
        context: ToolCallContext,
        selected_positions: list[int] | None = None,
        preserve_native: bool = False,
    ) -> ToolOutcome:
        if not outcome.refs and not outcome.full_model_content:
            return outcome
        exposed: list[dict[str, str]] = []
        reference_arguments: dict[str, str] = {}
        candidate_view = None
        candidate_items: list[dict[str, Any]] = []
        has_candidates = False
        for item in outcome.refs:
            if not isinstance(item, ReferenceValue):
                raise ToolPipelineError(
                    "工具返回了无效引用", code="invalid_tool_result"
                )
            reference = await self._put_reference(
                context,
                kind=item.kind,
                value=item.value,
                ttl_seconds=item.ttl_seconds,
            )
            if reference.kind == RESOURCE_KIND:
                has_candidates = True
                candidate_items = resource_candidate_items(item.value, outcome.public_content)
                candidate_view = await issue_candidate_view(
                    store=self.reference_store, owner=context.owner, session_id=context.session_id,
                    generation=context.lease.generation, ref=reference.ref, items=candidate_items,
                    ttl_seconds=item.ttl_seconds, turn_id=context.turn_id,
                    selected_positions=selected_positions,
                )
            exposed.append({"ref": reference.ref, "kind": reference.kind})
            kind = re.sub(r"[^a-z0-9_]+", "_", reference.kind.casefold()).strip("_")
            if kind:
                reference_arguments[f"{kind}_ref"] = reference.ref
        public = (
            candidate_result(outcome.public_content, candidate_view)
            if has_candidates else dict(outcome.public_content)
        )
        public["refs"] = exposed
        if reference_arguments:
            public["reference_arguments"] = reference_arguments
        content = outcome.full_model_content or outcome.model_content
        if has_candidates:
            content = resource_model_content(content, maximum=max(len(content) * 2, self.projector.max_model_chars))
        payload = decode_tool_content(content)
        if exposed:
            payload["opaque_refs"] = exposed
            payload["reference_arguments"] = reference_arguments
        numbered_items = candidate_view["items"] if candidate_view else candidate_items
        if numbered_items:
            payload["candidate_numbers"] = [
                {key: item[key] for key in ("position", "title", "coverage", "media_title", "requested_episode", "match")}
                for item in numbered_items
            ]
            if candidate_view and candidate_view["recommended_positions"]:
                payload["recommended_ingest_arguments"] = {
                    "source_type": RESOURCE_KIND, "resource_candidates_ref": candidate_view["ref"],
                    "positions": candidate_view["recommended_positions"], "target": "preferred",
                }
        model_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        # Reference appendices can also become lossy in the later 400-char pass.
        if outcome.full_model_content or len(model_content) > 400:
            handle = await self.save_result(model_content, context)
            public.update(result_handle=handle, read_tool="agent.read_result")
            try:
                payload = decode_tool_content(compact_tool_content(model_content, maximum=self.projector.max_model_chars, trusted=not bool(outcome.full_model_content)))
                payload.update(result_handle=handle, read_tool="agent.read_result")
                model_content = compact_tool_content(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), maximum=self.projector.max_model_chars, trusted=True)
            except ToolPipelineError as exc:
                # Eager snapshots must not narrow the pre-projected ToolOutcome
                # domain: atomic status/code facts may fit native history even
                # when they cannot fit a compact continuation envelope. Keep the
                # original fact (never infer success from the public summary).
                # The complete scoped snapshot remains publicly readable; normal
                # Session/provider/state admission still applies without changes.
                if (exc.code != "context_budget_exceeded" or not preserve_native or exposed or has_candidates
                        or outcome.full_model_content != outcome.model_content):
                    raise
                model_content = outcome.model_content
        updates = tuple(outcome.state_updates) + (
            StateUpdate("recent_refs", [item["ref"] for item in exposed], mode="append"),
            StateUpdate("ref_kinds", [item["kind"] for item in exposed], mode="append"),
        )
        if public.get("result_handle"):
            updates += (StateUpdate("recent_refs", [public["result_handle"]], mode="append"), StateUpdate("ref_kinds", ["tool_result"], mode="append"))
        if has_candidates:
            updates += (StateUpdate(f"metadata.{CANDIDATE_VIEW_KEY}", candidate_view),)
        return replace(
            outcome,
            public_content=public,
            model_content=model_content,
            state_updates=updates,
            full_model_content="",
        )

    async def _commit_updates(
        self,
        lease: PublicationLease,
        updates: Sequence[StateUpdate],
    ) -> None:
        if not updates:
            if not await self.state_store.is_current(lease):
                raise StalePublicationError("turn lost publication authority")
            return
        await self.state_store.commit(lease, updates=updates)
