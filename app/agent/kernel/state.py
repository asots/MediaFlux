"""统一会话状态、publication generation 与取消协调。"""

from __future__ import annotations

import asyncio
import inspect
import json
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Protocol

from app.agent.model_context_budget import compact_tool_content
from app.concurrency import CrossLoopAsyncLock
from .task_state import TaskPatch, reduce_task

from .conversation_codec import (
    PendingText,
    pending_content,
    conversation_cost,
    conversation_payload,
    conversation_rows,
    json_text,
)


class KernelStateError(RuntimeError):
    pass


class StalePublicationError(KernelStateError):
    """当前回合已失去发布权，禁止提交迟到状态。"""


class SessionBusyError(KernelStateError):
    """已确认的副作用正在执行；新回合不能抢占。"""


class SelectionInvalidError(KernelStateError):
    """候选输入在取得新回合前失效，不得改变原会话。"""

    def __init__(self, message: str = "候选已失效或不属于当前会话，请重新搜索后选择。") -> None:
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CandidateSelectionGuard:
    generation: int
    ref: str
    expires_at: float
    batch_generation: int | None = None

    def check(self, state: SessionState) -> None:
        view = state.metadata.get("ux_candidate_view")
        if (
            state.generation != self.generation
            or self.expires_at <= time.time()
            or not isinstance(view, dict)
            or view.get("ref") != self.ref
            or view.get("generation") != (self.generation if self.batch_generation is None else self.batch_generation)
        ):
            raise SelectionInvalidError()


@dataclass(frozen=True, slots=True)
class PendingEffectGuard:
    generation: int
    plan_id: str
    plan_generation: int
    expires_at: float
    clock: Callable[[], float]

    def check(self, state: SessionState) -> None:
        binding = {"plan_id": self.plan_id, "plan_generation": self.plan_generation}
        current = state.metadata.get("pending_effect_binding")
        if (state.generation != self.generation or state.pending_effect_plan_id != self.plan_id
                or self.expires_at <= self.clock()
                or (current != binding and not (current is None and self.plan_generation == state.generation))):
            raise SelectionInvalidError("待确认计划已失效，请使用当前卡片。")
        state.metadata["pending_effect_binding"] = binding
        if current is None:
            state.metadata["pending_effect_goal"] = deepcopy(next(
                (row for row in reversed(state.conversation) if row.get("role") == "user"), {}
            ))


@dataclass(frozen=True, slots=True)
class AgentInput:
    message: str
    owner: str
    session_id: str
    request_id: str = ""
    channel: str = "api"
    reply_context: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        message = str(self.message or "").strip()
        owner = str(self.owner or "").strip()
        session_id = str(self.session_id or "").strip()
        request_id = str(self.request_id or "").strip() or secrets.token_urlsafe(12)
        channel = str(self.channel or "api").strip().lower() or "api"
        if not message:
            raise ValueError("message cannot be empty")
        if not owner:
            raise ValueError("owner cannot be empty")
        if not session_id:
            raise ValueError("session_id cannot be empty")
        if len(message) > 12_000:
            raise ValueError("message is too long")
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "reply_context", deepcopy(dict(self.reply_context)))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class PublicationLease:
    owner: str
    session_id: str
    generation: int
    turn_id: str
    request_id: str
    read_wait_id: str = ""
    read_claim_id: str = ""


def publication_matches(
    lease: PublicationLease, *, generation: int, confirmed: Any = None,
) -> bool:
    """普通回合按 generation；已认领确认还必须匹配持久化的执行 turn。

    不增加 generation，避免破坏已冻结票据。旧历史没有标记时继续兼容；
    新普通回合增加 generation 后，上一代确认标记自然失效。
    """
    if generation != lease.generation:
        return False
    if not isinstance(confirmed, Mapping) or confirmed.get("generation") != generation:
        return True
    return confirmed.get("turn_id") == lease.turn_id


def publication_commit_matches(
    lease: PublicationLease,
    state: SessionState,
    conversation: Sequence[Mapping[str, Any]] | None,
    updates: Sequence[StateUpdate],
) -> bool:
    """已原子领到当前待确认票据的下一步可接管发布权，旧读回合仍被隔离。"""
    claim = (
        conversation is None
        and len(updates) == 2
        and updates[0].mode == "set"
        and updates[0].key == "metadata.confirmed_publication"
        and updates[1].mode == "clear_if_equals"
        and updates[1].key == "pending_effect_plan_id"
        and (type(updates[1]) is _PendingEffectTransition or "pending_effect_binding" not in state.metadata)
    )
    value = updates[0].value if claim else None
    handoff = (
        isinstance(value, Mapping)
        and bool(state.pending_effect_plan_id)
        and str(updates[1].value or "") == state.pending_effect_plan_id
        and value
        == {
            "generation": lease.generation,
            "turn_id": lease.turn_id,
            "plan_id": state.pending_effect_plan_id,
        }
    )
    # 撤销/过期收束只做计划 ID 的原子条件清理，不接管已确认回执的发布权。
    pending_clear = (
        conversation is None and len(updates) == 1
        and updates[0].key == "pending_effect_plan_id"
        and updates[0].mode == "clear_if_equals"
        and (type(updates[0]) is _PendingEffectTransition or "pending_effect_binding" not in state.metadata)
    )
    confirmed = None if handoff or pending_clear or candidate_metadata_only(
        conversation, updates
    ) else state.metadata.get("confirmed_publication")
    return publication_matches(
        lease,
        generation=state.generation,
        confirmed=confirmed,
    )


@dataclass(frozen=True, slots=True)
class StateUpdate:
    key: str
    value: Any
    mode: str = "set"


@dataclass(frozen=True, slots=True)
class _PublicationFinished(StateUpdate):
    """Executor terminal transition, not a business state update."""


@dataclass(frozen=True, slots=True)
class _PendingEffectTransition(StateUpdate):
    """Internal atomic ID/binding/goal transition; never built from business JSON."""

    plan_generation: int = 0
    goal: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _ExecutionCapacity(StateUpdate):
    """Executor-only stage update; never reconstructed from business JSON."""

    current_start: int | None = None


class HistoryCapacityError(KernelStateError):
    """Retention would discard the executor's current call/result group."""


TRANSIENT_EXTRA_BYTES = 160 * 1024 + len(',"execution_capacity":1')


def state_capacity(payload: Mapping[str, Any], normal: int) -> int:
    phase = payload.get("execution_capacity", 0)
    if type(phase) is not int or phase not in (0, 1):
        raise ValueError("invalid execution capacity phase")
    return normal + TRANSIENT_EXTRA_BYTES if phase else normal


def validate_execution_state(state: SessionState) -> None:
    payload = state.to_payload()
    conversation_rows(payload["conversation"])
    if len(json_text(payload).encode()) > state_capacity(payload, 256 * 1024):
        raise ValueError("Agent Kernel 数据超过持久化上限")


def candidate_metadata_only(
    conversation: Sequence[Mapping[str, Any]] | None, updates: Sequence[StateUpdate],
) -> bool:
    # TG 候选 UI 使用专门的事务内 CAS，不是模型回合的 conversation 发布。
    # 只给这一窄契约例外，不能借 metadata 更新携带回执/计划覆盖。
    return (
        conversation is None and len(updates) == 1
        and updates[0].key == "metadata.ux_candidate_draft"
        and updates[0].mode == "compare_candidate"
    )


@dataclass(slots=True)
class SessionState:
    owner: str
    session_id: str
    generation: int = 0
    conversation: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    recent_refs: list[str] = field(default_factory=list)
    ref_kinds: set[str] = field(default_factory=set)
    pending_effect_plan_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_capacity: bool = False

    def clone(self) -> SessionState:
        return deepcopy(self)

    def to_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "conversation": conversation_payload(deepcopy(self.conversation)),
            "summary": self.summary,
            "recent_refs": list(self.recent_refs[-100:]),
            "ref_kinds": sorted(self.ref_kinds),
            "pending_effect_plan_id": self.pending_effect_plan_id,
            "metadata": deepcopy(self.metadata),
            **({"execution_capacity": 1} if self.execution_capacity else {}),
        }

    @classmethod
    def from_payload(
        cls, *, owner: str, session_id: str, generation: int, payload: Mapping[str, Any]
    ) -> SessionState:
        metadata = payload.get("metadata")
        return cls(
            owner=owner,
            session_id=session_id,
            generation=max(0, int(generation)),
            conversation=conversation_rows(payload.get("conversation", [])),
            summary=str(payload.get("summary") or "")[:8_000],
            recent_refs=[
                str(item) for item in payload.get("recent_refs", ()) if str(item)
            ][-100:],
            ref_kinds={str(item) for item in payload.get("ref_kinds", ()) if str(item)},
            pending_effect_plan_id=str(payload.get("pending_effect_plan_id") or "")[
                :200
            ],
            metadata=deepcopy(metadata) if isinstance(metadata, dict) else {},
            execution_capacity=bool(payload.get("execution_capacity", 0)),
        )

    def apply(self, updates: Sequence[StateUpdate]) -> None:
        for update in updates:
            if type(update) is _PublicationFinished:
                marker = self.metadata.get("active_publication")
                wait = self.metadata.get("read_wait") or {}
                if (isinstance(marker, dict) and marker.get("generation") == self.generation
                        and marker.get("status") == "running" and wait.get("state") not in {"pending", "claimed", "running"}):
                    marker["status"] = "finished"
                continue
            if type(update) is _PendingEffectTransition:
                if update.mode == "clear_if_equals":
                    if self.pending_effect_plan_id == update.value:
                        self.pending_effect_plan_id = ""
                        self.metadata.pop("pending_effect_binding", None)
                        self.metadata.pop("pending_effect_goal", None)
                elif (update.mode == "set" and update.plan_generation == self.generation
                      and update.value and isinstance(update.goal, Mapping)):
                    self.pending_effect_plan_id = str(update.value)
                    self.metadata["pending_effect_binding"] = {"plan_id": update.value, "plan_generation": update.plan_generation}
                    self.metadata["pending_effect_goal"] = deepcopy(dict(update.goal))
                else:
                    raise ValueError("invalid pending effect transition")
                continue
            if type(update) is _ExecutionCapacity:
                self.execution_capacity = bool(update.value)
                continue
            key = str(update.key or "").strip()
            if any(key == "metadata." + field or key.startswith("metadata." + field + ".")
                   for field in ("pending_effect_binding", "pending_effect_goal", "active_publication", "read_wait", "history_memory_version")):
                raise ValueError("pending binding requires a typed transition")
            if key == "metadata" or key == "metadata.task" or key.startswith("metadata.task."):
                raise ValueError("task namespace requires a typed patch")
            if not key:
                continue
            if key == "summary":
                note = str(update.value or "")[:8_000]
                if self.metadata.get("history_memory_version") == 1:
                    memory = history_memory(self)
                    memory["legacy_note"] = note
                    self.summary = _encode_history_memory(memory)
                else:
                    self.summary = note
            elif key == "pending_effect_plan_id":
                if "pending_effect_binding" in self.metadata:
                    raise ValueError("bound pending ID requires a typed transition")
                if update.mode == "clear_if_equals":
                    # 必须在 store.commit 的锁/事务内比较最新值；旧票据的
                    # 迟到取消或确认结果不能清掉同代次刚发布的新计划。
                    if self.pending_effect_plan_id == str(update.value or ""):
                        self.pending_effect_plan_id = ""
                        self.metadata.pop("pending_effect_binding", None)
                else:
                    self.pending_effect_plan_id = str(update.value or "")[:200]
                    self.metadata.pop("pending_effect_binding", None)
            elif key == "recent_refs":
                values = [str(item) for item in (update.value or []) if str(item)]
                if update.mode == "append":
                    self.recent_refs = (self.recent_refs + values)[-100:]
                else:
                    self.recent_refs = values[-100:]
            elif key == "ref_kinds":
                values = {str(item) for item in (update.value or []) if str(item)}
                self.ref_kinds = (
                    self.ref_kinds | values if update.mode == "append" else values
                )
            elif key == "metadata.ux_candidate_draft" and update.mode == "compare_candidate":
                value = update.value
                current = self.metadata.get("ux_candidate_draft") or {}
                view = self.metadata.get("ux_candidate_view") or {}
                if (
                    view.get("ref") != value["next"]["candidate_ref"]
                    or view.get("expires_at", 0) <= time.time()
                    or (value["expected"] is not None and current.get("handle") != value["expected"])
                ):
                    raise SelectionInvalidError("选择状态已更新，请使用当前消息中的按钮。")
                self.metadata["ux_candidate_draft"] = deepcopy(value["next"])
            elif key.startswith("metadata."):
                field_name = key.partition(".")[2]
                if field_name:
                    if update.mode == "delete":
                        self.metadata.pop(field_name, None)
                    else:
                        self.metadata[field_name] = deepcopy(update.value)


def retain_conversation(
    conversation: Sequence[Mapping[str, Any]], *, maximum_bytes: int = 160 * 1024,
    _required_from: int | None = None,
) -> list[dict[str, Any]]:
    """按实际存储预算保留完整用户回合，不因80条短消息删除原始条件。"""
    cost = conversation_cost

    original_bytes = cost(conversation)
    if original_bytes <= maximum_bytes:
        return list(conversation)
    if any(
        isinstance(row.get("content"), PendingText) and row["content"] == pending_content(True)
        for row in conversation
    ):
        raise ValueError("unresolved execution history exceeds conversation budget")
    ratio = maximum_bytes / original_bytes * 0.75
    turns: list[list[dict[str, Any]]] = []
    required_rows: set[int] = set()
    for index, item in enumerate(conversation):
        row = deepcopy(dict(item))
        if _required_from is not None and index >= _required_from and (
            row.get("tool_calls") or row.get("role") == "tool" or row.get("reply_context", {}).get("reference_handle")
        ):
            required_rows.add(id(row))
        if row.get("role") == "tool":
            content = str(row.get("content") or "")
            row["content"] = compact_tool_content(content, maximum=max(512, int(len(content) * ratio)), tool_name=row.get("tool_name", ""), trusted=True)
        elif row.get("role") == "assistant" and len(str(row.get("content") or "")) > 8000:
            row["content"] = str(row["content"])[:8000] + "\n[历史长文本已截断]"
        if row.get("role") == "user" or not turns:
            turns.append([])
        turns[-1].append(row)
    kept: list[dict[str, Any]] = []
    for turn in reversed(turns):
        if cost(turn + kept) <= maximum_bytes:
            kept = turn + kept
            continue
        if not kept:
            # 单次长任务也必须保留用户请求和已执行回执；工具批次不可拆开。
            blocks: list[list[dict[str, Any]]] = []
            pending: set[str] = set()
            for row in turn:
                if row.get("role") == "tool" and row.get("tool_call_id") in pending:
                    blocks[-1].append(row)
                    pending.discard(row["tool_call_id"])
                else:
                    blocks.append([row])
                    pending = {call.get("call_id") for call in row.get("tool_calls") or ()}
            order = sorted(range(len(blocks)), key=lambda i: (
                any(row.get("role") == "user" for row in blocks[i]),
                any(row.get("effect_plan_id") or row.get("completion_receipt_id") for row in blocks[i]), i,
            ), reverse=True)
            chosen = []
            used = 256
            for index in order:
                size = cost(blocks[index])
                if used + size <= maximum_bytes:
                    chosen.append(index)
                    used += size
            kept = [row for index in sorted(chosen) for row in blocks[index]]
        if required_rows - {id(row) for row in kept}:
            raise HistoryCapacityError("current execution group exceeds conversation budget")
        if kept:
            kept[0] = {**kept[0], "history_before_truncated": True}
        break
    return kept


def history_memory(state: SessionState) -> dict[str, Any]:
    """只有 store 在裁剪事务中提取的原始 user 行可以成为历史任务来源。"""
    if state.metadata.get("history_memory_version") != 1:
        return {"requests": [], "observations": [], "legacy_note": state.summary}
    try:
        memory = json.loads(state.summary)
        if (not isinstance(memory, dict) or not isinstance(memory["requests"], list)
                or not isinstance(memory["observations"], list)
                or not isinstance(memory.get("legacy_note", ""), str)
                or any(not isinstance(row, dict) or not isinstance(row.get("sha256"), str)
                       for row in memory["observations"])):
            raise ValueError("invalid history memory")
        for row in memory["requests"]:
            if (not isinstance(row, dict) or not isinstance(row.get("text"), str)
                    or row.get("sha256") != sha256(row["text"].encode()).hexdigest()):
                raise ValueError("invalid historical user source")
        return memory
    except (ValueError, KeyError, TypeError) as exc:
        raise HistoryCapacityError("historical user sources cannot be verified") from exc


def historical_user_sources(state: SessionState) -> tuple[str, ...]:
    return tuple(dict.fromkeys([
        *(row["text"] for row in history_memory(state)["requests"]),
        *(row["content"] for row in state.conversation
          if row.get("role") == "user" and isinstance(row.get("content"), str)),
    ]))


def _encode_history_memory(memory: dict[str, Any]) -> str:
    # 观察只作旧结果索引；用户原话/旧摘要不能为凑预算而悄悄截断。
    observations = memory.setdefault("observations", [])
    while len(json_text(memory)) > 8_000 and observations:
        observations.pop(0)
        memory["observations_omitted"] = True
    text = json_text(memory)
    if len(text) > 8_000:
        raise HistoryCapacityError("historical user constraints exceed memory capacity")
    return text


def preserve_history(
    state: SessionState, original: Sequence[Mapping[str, Any]], retained: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """落盘与模型窗口共用的来源收敛：不生成目标、授权或完成结论。"""
    memory = history_memory(state)
    texts = {row["text"] for row in memory["requests"]}
    kept_users = {row.get("content") for row in retained if row.get("role") == "user"}
    kept_tools = {json_text(row) for row in retained if row.get("role") == "tool"}
    observations = memory.setdefault("observations", [])
    seen = {row["sha256"] for row in observations}
    for row in original:
        content = row.get("content")
        if row.get("role") == "user" and isinstance(content, str) and content not in kept_users and content not in texts:
            memory["requests"].append({"text": content, "sha256": sha256(content.encode()).hexdigest()})
            texts.add(content)
        elif row.get("role") == "tool" and json_text(row) not in kept_tools:
            digest = sha256(json_text(row).encode()).hexdigest()
            if digest in seen:
                continue
            try:
                result = json.loads(content) if isinstance(content, str) else {}
            except ValueError:
                result = {}
            result = result if isinstance(result, dict) else {}
            observations.append({
                "sha256": digest, "tool": str(row.get("tool_name") or "")[:200],
                "call_id": str(row.get("tool_call_id") or "")[:200],
                "status": str(result.get("status") or "unknown")[:80],
                "summary": str(result.get("summary") or "")[:512],
                "result_handle": str(result.get("result_handle") or "")[:200],
                "details_omitted": True,
            })
            seen.add(digest)
    if len(observations) > 12:
        del observations[:-12]
        memory["observations_omitted"] = True
    if any(row.get("history_before_truncated") for row in original) and not state.metadata.get("history_memory_version"):
        memory["legacy_history_incomplete"] = True
    return memory


def commit_conversation(
    state: SessionState, conversation: Sequence[Mapping[str, Any]],
    updates: Sequence[StateUpdate] = (), *, receipt_metadata: Mapping[str, Any] | None = None,
) -> None:
    # Resolve the executor's positional boundary, not equal call IDs in old turns.
    start = next((u.current_start for u in updates if type(u) is _ExecutionCapacity), None)
    merged = merge_effect_receipts(conversation, state.metadata if receipt_metadata is None else receipt_metadata)
    if start is not None:
        user_ordinal = sum(row.get("role") == "user" for row in conversation[:start])
        users = [i for i, row in enumerate(merged) if row.get("role") == "user"]
        if start >= len(conversation) or conversation[start].get("role") != "user" or user_ordinal >= len(users):
            raise HistoryCapacityError("current execution boundary is unavailable")
        start = users[user_ordinal]
    retained = retain_conversation(merged, _required_from=start)
    if retained != merged:
        memory = preserve_history(state, merged, retained)
        state.summary = _encode_history_memory(memory)
        state.metadata["history_memory_version"] = 1
    state.conversation = retained


def merge_effect_receipts(
    conversation: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """按 plan ID 将已完成回执并入会话，不做文本近似匹配。"""
    merged = [deepcopy(dict(message)) for message in conversation]
    waits = metadata.get("effect_waits")
    if not isinstance(waits, Mapping):
        return merged

    for plan_id, record in waits.items():
        if not isinstance(plan_id, str) or not plan_id or not isinstance(record, Mapping):
            continue
        receipt = record.get("receipt_message")
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("completion_receipt_id") != plan_id
        ):
            continue

        matches = [
            index for index, message in enumerate(merged)
            if message.get("completion_receipt_id") == plan_id
        ]
        replacement = deepcopy(dict(receipt))
        if matches:
            # 确认后的模型答复可能已概述“仍在执行”。晚到终态必须位于它
            # 之后，否则公开投影会把更新后的系统回执当中间结果折叠掉。
            has_later_answer = any(
                message.get("role") == "user" or (
                    message.get("role") == "assistant"
                    and message.get("effect_plan_id") == plan_id
                    and not message.get("tool_name")
                )
                for message in merged[matches[0] + 1:]
            )
            if has_later_answer:
                for index in reversed(matches):
                    del merged[index]
                merged.append(replacement)
            else:
                merged[matches[0]] = replacement
                for index in reversed(matches[1:]):
                    del merged[index]
        else:
            merged.append(replacement)
    return merged


def _consume_delivered_effect_receipts(state: SessionState) -> None:
    waits = state.metadata.get("effect_waits")
    if not isinstance(waits, Mapping):
        state.metadata.pop("effect_waits", None)
        state.metadata.pop("effect_next_poll_at", None)
        return

    delivered = {
        plan_id: record
        for plan_id, record in waits.items()
        if isinstance(plan_id, str)
        and isinstance(record, Mapping)
        and record.get("delivered") is True
    }
    if delivered:
        commit_conversation(state, state.conversation, receipt_metadata={"effect_waits": delivered})

    remaining = {
        plan_id: deepcopy(record)
        for plan_id, record in waits.items()
        if plan_id not in delivered
    }
    if remaining:
        state.metadata["effect_waits"] = remaining
        next_polls = [
            record.get("next_poll_at")
            for record in remaining.values()
            if isinstance(record, Mapping)
            and isinstance(record.get("next_poll_at"), (int, float))
            and not isinstance(record.get("next_poll_at"), bool)
        ]
        if next_polls:
            state.metadata["effect_next_poll_at"] = min(next_polls)
        else:
            state.metadata.pop("effect_next_poll_at", None)
    else:
        state.metadata.pop("effect_waits", None)
        state.metadata.pop("effect_next_poll_at", None)


def _effect_time_is_due(value: Any, now: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value <= now
    )


def _apply_effect_state_change(
    state: SessionState, change: Callable[[SessionState], Any],
) -> tuple[SessionState, Any, bool]:
    """在副本上执行纯同步回调，并固定 publication 身份字段。"""
    updated = state.clone()
    result = change(updated)
    for reserved in ("pending_effect_binding", "pending_effect_goal", "confirmed_publication", "active_publication", "read_wait"):
        if (reserved in updated.metadata) != (reserved in state.metadata) or updated.metadata.get(reserved) != state.metadata.get(reserved):
            raise ValueError("effect callback cannot change publication binding")
    if ("task" in updated.metadata) != ("task" in state.metadata) or updated.metadata.get("task") != state.metadata.get("task"):
        raise ValueError("effect callback cannot change task")
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("effect state change must be synchronous")
    updated.owner = state.owner
    updated.session_id = state.session_id
    updated.generation = state.generation
    updated.pending_effect_plan_id = state.pending_effect_plan_id
    updated.execution_capacity = state.execution_capacity
    return updated, result, updated != state


def _begin_publication(state: SessionState, request_id: str) -> None:
    """Internal durable request identity, never copied from model metadata."""
    from .read_wait import revoke
    revoke(state, "superseded")
    state.metadata["active_publication"] = {
        "generation": state.generation, "request_id": request_id, "status": "running",
    }


def _cancel_publication(state: SessionState, request_id: str) -> bool:
    active = state.metadata.get("active_publication")
    confirmed = state.metadata.get("confirmed_publication")
    if (not isinstance(active, dict) or active.get("generation") != state.generation
            or active.get("status") != "running"
            or (request_id and request_id != active.get("request_id"))
            or (isinstance(confirmed, dict) and confirmed.get("generation") == state.generation)):
        return False
    state.metadata["active_publication"] = {**active, "status": "cancelled"}
    from .read_wait import revoke
    revoke(state, "cancelled")
    state.generation += 1
    return True


def _exact_value(value: Any) -> Any:
    """Immutable typed comparison; bool/int and native payloads stay distinct."""
    if isinstance(value, dict):
        return (type(value), tuple(sorted((key, _exact_value(item)) for key, item in value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_exact_value(item) for item in value))
    if type(value) not in (type(None), str, int, float, bool, PendingText):
        raise TypeError("checkpoint must contain immutable JSON values")
    return (type(value), value)


@dataclass(frozen=True, slots=True)
class _CancelledReadInline:
    """Invocation-private checkpoint and put-once bytes, never a handler DTO."""

    lease: PublicationLease
    expected: Any
    assistant_index: int
    call_index: int
    tool_index: int
    call: Any
    tool: Any = field(repr=False)
    _content: str | None = field(default=None, init=False)

    @classmethod
    def capture(cls, lease, conversation, assistant_index, call_index, tool_index, *, tool):
        from .capabilities import ToolEffect
        if tool.effect is not ToolEffect.READ:
            raise ValueError("only catalog READ invocations may retain returned facts")
        call = conversation[assistant_index]["tool_calls"][call_index]
        return cls(lease, _exact_value(conversation), assistant_index, call_index,
                   tool_index, _exact_value(call), tool)

    def reject_incomplete(self) -> None:
        if self._content is not None:
            raise ValueError("returned read already captured")
        object.__setattr__(self, "_content", "")

    def put_once(self, content: str) -> None:
        if self._content is not None:
            raise ValueError("returned read already captured")
        # Copy the full, already-safe JSON, never retain mutable domain aliases.
        import json
        object.__setattr__(self, "_content", json_text(json.loads(content)))


def _settle_cancelled_read_inline(state, receipt, maximum=256 * 1024):
    if type(receipt) is not _CancelledReadInline or not receipt._content:
        return None
    lease = receipt.lease
    if ((state.owner, state.session_id) != (lease.owner, lease.session_id)
            or state.generation != lease.generation + 1
            or _exact_value(state.metadata.get("active_publication")) != _exact_value({
                "generation": lease.generation, "request_id": lease.request_id, "status": "cancelled"})
            or _exact_value(state.conversation) != receipt.expected):
        return None
    try:
        assistant = state.conversation[receipt.assistant_index]
        call = assistant["tool_calls"][receipt.call_index]
        row = state.conversation[receipt.tool_index]
        if (not 0 <= receipt.assistant_index < receipt.tool_index < len(state.conversation)
                or receipt.tool_index != receipt.assistant_index + 1 + receipt.call_index
                or not 0 <= receipt.call_index < len(assistant["tool_calls"])
                or assistant.get("role") != "assistant" or _exact_value(call) != receipt.call
                or any(item.get("role") != "tool" for item in state.conversation[receipt.assistant_index + 1:receipt.tool_index])
                or row.get("role") != "tool"
                or row.get("tool_call_id") != call.get("call_id")
                or row.get("tool_name") != call.get("name")
                or _exact_value(row.get("content")) != _exact_value(pending_content(True))):
            return None
    except (KeyError, IndexError, TypeError):
        return None
    updated = state.clone()
    updated.conversation[receipt.tool_index]["content"] = receipt._content
    payload = updated.to_payload()
    if (_exact_value(conversation_rows(payload["conversation"])) != _exact_value(updated.conversation)
            or len(json_text(payload).encode()) > maximum):
        raise ValueError("cancelled read settlement exceeds lossless state capacity")
    return updated


class SessionStateStore(Protocol):
    async def settle_cancelled_read_inline(self, receipt: _CancelledReadInline) -> bool: ...

    async def cancel_publication(self, *, owner: str, session_id: str, request_id: str) -> bool: ...

    async def begin_turn(
        self, *, owner: str, session_id: str, request_id: str,
        selection_guard: CandidateSelectionGuard | None = None,
        pending_guard: PendingEffectGuard | None = None,
    ) -> tuple[PublicationLease, SessionState]: ...

    async def is_current(self, lease: PublicationLease) -> bool: ...

    async def commit(
        self,
        lease: PublicationLease,
        *,
        conversation: Sequence[Mapping[str, Any]] | None = None,
        updates: Sequence[StateUpdate] = (),
        task_patch: TaskPatch | None = None,
    ) -> SessionState: ...

    async def update_effect_state(
        self, *, owner: str, session_id: str,
        change: Callable[[SessionState], Any],
    ) -> Any: ...

    async def due_effect_waits(self, *, limit: int = 16) -> list[dict[str, Any]]: ...

    async def load(self, *, owner: str, session_id: str) -> SessionState: ...


class InMemorySessionStateStore:
    """测试与单进程运行使用的权威状态实现。"""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._lock = CrossLoopAsyncLock()
        self._states: dict[tuple[str, str], SessionState] = {}
        self._clock = clock

    async def begin_turn(
        self, *, owner: str, session_id: str, request_id: str,
        selection_guard: CandidateSelectionGuard | None = None,
        pending_guard: PendingEffectGuard | None = None,
    ) -> tuple[PublicationLease, SessionState]:
        key = (owner, session_id)
        async with self._lock:
            current = self._states.get(key)
            generation = (current.generation if current else 0) + 1
            state = (
                current.clone()
                if current
                else SessionState(owner=owner, session_id=session_id)
            )
            if selection_guard is not None:
                selection_guard.check(state)
            if pending_guard is not None:
                pending_guard.check(state)
            else:
                # Only a checked explanation transition may carry a frozen
                # plan's authority across publication generations.
                state.metadata.pop("pending_effect_binding", None)
            _consume_delivered_effect_receipts(state)
            state.generation = generation
            _begin_publication(state, request_id)
            if state.execution_capacity or pending_guard is not None:
                validate_execution_state(state)
            self._states[key] = state
            lease = PublicationLease(
                owner=owner,
                session_id=session_id,
                generation=generation,
                turn_id=secrets.token_urlsafe(12),
                request_id=request_id,
            )
            return lease, state.clone()

    async def cancel_publication(self, *, owner: str, session_id: str, request_id: str) -> bool:
        async with self._lock:
            state = self._states.get((owner, session_id))
            if state is None:
                return False
            updated = state.clone()
            if not _cancel_publication(updated, request_id):
                return False
            self._states[(owner, session_id)] = updated
            return True

    async def settle_cancelled_read_inline(self, receipt: _CancelledReadInline) -> bool:
        if type(receipt) is not _CancelledReadInline:
            return False
        key = (receipt.lease.owner, receipt.lease.session_id)
        async with self._lock:
            state = self._states.get(key)
            updated = _settle_cancelled_read_inline(state, receipt) if state is not None else None
            if updated is None:
                return False
            validate_execution_state(updated)
            self._states[key] = updated
            return True

    async def is_current(self, lease: PublicationLease) -> bool:
        async with self._lock:
            state = self._states.get((lease.owner, lease.session_id))
            if state and lease.read_wait_id:
                from .read_wait import check_claim
                try:
                    check_claim(state, lease, self._clock(), publication=True)
                except StalePublicationError:
                    return False
            return bool(state and publication_matches(
                lease, generation=state.generation,
                confirmed=state.metadata.get("confirmed_publication"),
            ))

    async def commit(
        self,
        lease: PublicationLease,
        *,
        conversation: Sequence[Mapping[str, Any]] | None = None,
        updates: Sequence[StateUpdate] = (),
        task_patch: TaskPatch | None = None,
    ) -> SessionState:
        key = (lease.owner, lease.session_id)
        async with self._lock:
            state = self._states.get(key)
            if state is None or not publication_commit_matches(lease, state, conversation, updates):
                raise StalePublicationError("turn no longer owns publication authority")
            state = state.clone()
            from .read_wait import apply_actions, check_claim, task_changed
            check_claim(state, lease, self._clock())
            if task_patch is not None:
                state.metadata["task"] = reduce_task(state.metadata.get("task"), task_patch, owner=state.owner, session_id=state.session_id)
                task_changed(state)
            if conversation is not None:
                commit_conversation(state, conversation, updates)
            apply_actions(state, lease, updates, self._clock())
            state.apply(updates)
            if task_patch is not None or state.execution_capacity or any(type(update) in (_ExecutionCapacity, _PendingEffectTransition) for update in updates):
                validate_execution_state(state)
            validated = state.clone()
            self._states[key] = validated
            return validated.clone()

    async def update_effect_state(
        self, *, owner: str, session_id: str,
        change: Callable[[SessionState], Any],
    ) -> Any:
        key = (owner, session_id)
        async with self._lock:
            current = self._states.get(key)
            if current is None:
                return None
            updated, result, changed = _apply_effect_state_change(current, change)
            if changed:
                if updated.execution_capacity:
                    validate_execution_state(updated)
                self._states[key] = updated.clone()
            return result

    async def due_read_waits(self, *, limit: int = 16) -> list[dict[str, Any]]:
        async with self._lock:
            now = self._clock()
            records = [deepcopy(s.metadata["read_wait"]) for s in self._states.values()
                       if s.metadata.get("read_wait", {}).get("state") in {"pending", "claimed"}
                       and s.metadata["read_wait"]["due_at"] <= now
                       and s.metadata["read_wait"].get("claim_until", 0) <= now]
            return sorted(records, key=lambda r: r["due_at"])[:max(0, min(16, limit))]

    async def claim_read_wait(self, candidate, binding):
        from .read_wait import claim
        async with self._lock:
            key = candidate['owner'], candidate['session_id']
            current = self._states.get(key)
            if current is None:
                return None
            state = current.clone()
            lease = claim(state, candidate, self._clock(), binding)
            self._states[key] = state
            return (lease, state.clone()) if lease else None

    async def due_effect_waits(self, *, limit: int = 16) -> list[dict[str, Any]]:
        maximum = max(0, int(limit))
        if not maximum:
            return []
        now = self._clock()
        async with self._lock:
            due: list[tuple[float, str, dict[str, Any]]] = []
            for state in self._states.values():
                if not _effect_time_is_due(state.metadata.get("effect_next_poll_at"), now):
                    continue
                waits = state.metadata.get("effect_waits")
                if not isinstance(waits, Mapping):
                    continue
                for plan_id, record in waits.items():
                    if (
                        not isinstance(record, Mapping)
                        or not _effect_time_is_due(record.get("next_poll_at"), now)
                    ):
                        continue
                    item = deepcopy(dict(record))
                    item["plan_id"] = plan_id
                    due.append((float(record["next_poll_at"]), str(plan_id), item))
            due.sort(key=lambda item: (item[0], item[1]))
            return [item[2] for item in due[:maximum]]

    async def load(self, *, owner: str, session_id: str) -> SessionState:
        async with self._lock:
            state = self._states.get((owner, session_id))
            return (
                state.clone()
                if state
                else SessionState(owner=owner, session_id=session_id)
            )


class CancellationToken:
    __slots__ = ("_event", "_reason", "_task_cancel_requested", "interruptible")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = ""
        self._task_cancel_requested = False
        self.interruptible = False

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason or "cancelled"

    def cancel(self, reason: str = "cancelled", task: asyncio.Task[Any] | None = None) -> None:
        self._reason = str(reason or "cancelled")[:200]
        self._event.set()
        if self.interruptible and task is not None and not task.done() and not self._task_cancel_requested:
            self._task_cancel_requested = True  # 重复取消不能打断状态/回执收尾。
            task.get_loop().call_soon_threadsafe(task.cancel)

    async def wait(self) -> None:
        while not self._event.is_set():
            await asyncio.sleep(0.01)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError(self.reason)


class TurnCoordinator:
    """同一 owner/session 的最新回合拥有唯一发布权。"""

    def __init__(self) -> None:
        self._lock = CrossLoopAsyncLock()
        self._active: dict[tuple[str, str], tuple[PublicationLease, CancellationToken, bool, asyncio.Task[Any] | None]] = {}

    async def begin(
        self,
        lease: PublicationLease,
        *,
        protected: bool = False,
        task: asyncio.Task[Any] | None = None,
    ) -> CancellationToken:
        key = (lease.owner, lease.session_id)
        token = CancellationToken()
        async with self._lock:
            previous = self._active.get(key)
            if previous is not None:
                if previous[2]:
                    raise SessionBusyError("confirmed effect is executing")
                previous[1].cancel("superseded", previous[3])
            self._active[key] = (lease, token, bool(protected), task)
        return token

    async def cancel(
        self, *, owner: str, session_id: str, reason: str = "cancelled", request_id: str = ""
    ) -> bool:
        async with self._lock:
            current = self._active.get((owner, session_id))
            if current is None or current[2] or (
                request_id and current[0].request_id != request_id
            ):
                return False
            current[1].cancel(reason, current[3])
            return True

    def _owned_turn(self, lease: PublicationLease, token: CancellationToken):
        current = self._active.get((lease.owner, lease.session_id))
        return current if current and current[0].turn_id == lease.turn_id and current[1] is token else None

    async def unprotect(self, lease: PublicationLease, token: CancellationToken) -> None:
        """真实写入与回执持久化后，后续规划恢复为可停止的普通回合。"""
        async with self._lock:
            if current := self._owned_turn(lease, token):
                self._active[(lease.owner, lease.session_id)] = (lease, token, False, current[3])

    async def describe(self, *, owner: str, session_id: str) -> dict[str, Any] | None:
        """只读观察现有轮次；不取得发布权或启动执行。"""
        async with self._lock:
            current = self._active.get((owner, session_id))
            if current is None:
                return None
            return {
                "request_id": current[0].request_id,
                "turn_id": current[0].turn_id,
                "generation": current[0].generation,
                "protected": current[2],
                "status": "cancelling" if current[1].cancelled else "running",
            }

    async def has_protected_turn(self, *, owner: str, session_id: str) -> bool:
        async with self._lock:
            current = self._active.get((owner, session_id))
            return bool(current and current[2])

    async def is_current(self, lease: PublicationLease, token: CancellationToken) -> bool:
        async with self._lock:
            return bool(self._owned_turn(lease, token) and not token.cancelled)

    async def finish(self, lease: PublicationLease, token: CancellationToken) -> None:
        async with self._lock:
            if self._owned_turn(lease, token):
                self._active.pop((lease.owner, lease.session_id), None)
