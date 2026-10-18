"""完成回执的加密、恢复、历史交付与确认范围；不观察业务任务或启动续行。"""
from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cryptography.fernet import Fernet

from app.agent.models import Evidence, ToolResult
from app.agent.public_view import _CONFIRMED_RESULT_MARKER, format_public_result

from .conversation_state import commit_conversation
from .model import ModelMessage
from .session_state import SessionState
from .state_ports import SessionStateStore
from .state_types import PublicationLease

EFFECT_PENDING = {"waiting", "pending", "claimed", "running"}
_EFFECT_WAITS_KEY = "effect_waits"
_EFFECT_NEXT_POLL_KEY = "effect_next_poll_at"


@dataclass(frozen=True)
class EffectCompletionScope:
    """确认管道注入的原会话范围，不来自模型参数。"""

    store: SessionStateStore
    lease: PublicationLease
    plan_id: str
    channel: str = "api"
    origin: dict[str, Any] | None = None


def effect_active(row):
    record = row.get("continuation") or {}
    return record.get("state") in EFFECT_PENDING


def _completion_cipher() -> Fernet:
    from app.modules.web_secret import get_web_secret

    secret = (get_web_secret() or "")
    if not secret:
        raise ValueError("Agent 回执持久化密钥不可用")
    key = hashlib.sha256(b"mediaflux-agent-effect-wait:v1\0" + secret.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(key))


def _next_effect_poll(state: SessionState) -> None:
    waits = state.metadata.get(_EFFECT_WAITS_KEY) or {}
    due = [float(row["next_poll_at"]) for row in waits.values() if not row.get("delivered")]
    if due:
        state.metadata[_EFFECT_NEXT_POLL_KEY] = min(due)
    else:
        state.metadata.pop(_EFFECT_NEXT_POLL_KEY, None)


def _seed_result(seed: Mapping[str, Any]) -> ToolResult:
    value = seed["result"]
    return ToolResult(
        bool(value["ok"]), str(value["status"]), str(value["summary"]),
        data=dict(value.get("data") or {}),
        evidence=[Evidence(**item) for item in value.get("evidence") or []],
        suggestions=list(value.get("suggestions") or []), error=str(value.get("error") or ""),
    )


def _late_receipt(seed: Mapping[str, Any], public_result: Mapping[str, Any]) -> dict[str, Any]:
    message = ModelMessage(
        role="assistant", tool_name=seed["tool"], effect_plan_id=seed["plan_id"],
        completion_receipt_id=seed["plan_id"],
        content=_CONFIRMED_RESULT_MARKER + "\n" + json.dumps(public_result, ensure_ascii=False),
    ).to_dict()
    message["public_content"] = format_public_result(public_result)
    return message


async def remember_effect_receipt(store, *, owner, session_id, plan_id, message):
    """先保存投影后的完整回执模板，再写会话，崩溃后仍能补回opaque refs等投影。"""
    def remember(state):
        row = (state.metadata.get(_EFFECT_WAITS_KEY) or {}).get(plan_id)
        if row and row.get("state") == "terminal":
            row["receipt_template"] = dict(message)
            if row.get("receipt_message") is not None:
                row["receipt_message"] = dict(message)
                commit_conversation(state, state.conversation)
            return True
        return False
    return await store.update_effect_state(owner=owner, session_id=session_id, change=remember)


async def acknowledge_effect_receipt(store, *, owner, session_id, plan_id):
    """仅在前台终态会话已落盘后清理等待记录；超时的占位回执不能ACK。"""
    def acknowledge(state):
        waits = state.metadata.get(_EFFECT_WAITS_KEY) or {}
        row = waits.get(plan_id)
        if row and row.get("state") == "terminal" and row.get("foreground_final"):
            if effect_active(row):
                row["delivered"] = True
            else:
                waits.pop(plan_id)
            _next_effect_poll(state)
            return True
        return False
    return await store.update_effect_state(owner=owner, session_id=session_id, change=acknowledge)
