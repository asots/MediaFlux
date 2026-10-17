"""冻结副作用计划与一次性确认存储。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import time
from typing import Any, Protocol

from app.agent.confirmation import ConfirmationExpiredError, ConfirmationStore
from app.agent.public_safety import sanitize_public_text
from app.sensitive_data import contains_sensitive_credential, is_sensitive_key

from .capabilities import KernelToolSpec, ToolEffect


_PRIVATE_EDIT_KEY = re.compile(
    r"(?:^|_)(?:id|ids|ref|refs|handle|fingerprint|token|secret|password|cookie|"
    r"credential|authorization|signature|signed|nonce|code|passcode|pin|otp|path|paths|url|uri|headers?)(?:_|$)", re.I,
)
_EDIT_SCHEMA_KEYS = frozenset({"type", "enum", "minimum", "maximum", "exclusiveMinimum",
                               "exclusiveMaximum", "multipleOf", "minLength", "maxLength",
                               "pattern", "minItems", "maxItems", "uniqueItems"})


def confirmation_editing(tool: KernelToolSpec, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """只公布原 Schema 中可直接编辑的公开标量/标量列表，不序列化内部参数。"""
    def safe_value(value: Any) -> bool:
        if isinstance(value, str):
            return (not value.startswith(("ref_", "plan_")) and not contains_sensitive_credential(value)
                    and sanitize_public_text(value, limit=max(1, len(value))) == value)
        if isinstance(value, list):
            return all(safe_value(item) for item in value)
        return value is None or type(value) in (bool, int, float)

    properties, values = {}, {}
    for key, schema in tool.input_schema.get("properties", {}).items():
        private_key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).replace("-", "_")
        if (is_sensitive_key(key) or _PRIVATE_EDIT_KEY.search(private_key)
                or not isinstance(schema, Mapping) or schema.get("readOnly") or schema.get("writeOnly")):
            continue
        kind = schema.get("type")
        items = schema.get("items", {})
        variants = schema.get("oneOf", ())
        scalar_union = bool(variants) and all(isinstance(item, Mapping) and item.get("type") in {"string", "integer", "number", "boolean"} for item in variants)
        if kind not in {"string", "integer", "number", "boolean", "array"} and not scalar_union:
            continue
        if kind == "array" and (not isinstance(items, Mapping) or items.get("type") not in {"string", "integer", "number", "boolean"}):
            continue
        public = {name: deepcopy(value) for name, value in schema.items() if name in _EDIT_SCHEMA_KEYS}
        if scalar_union:
            public["oneOf"] = [{name: deepcopy(value) for name, value in item.items() if name in _EDIT_SCHEMA_KEYS} for item in variants]
        if kind == "array":
            public["items"] = {name: deepcopy(value) for name, value in items.items() if name in _EDIT_SCHEMA_KEYS}
        if (not safe_value(public.get("enum", [])) or not safe_value(public.get("items", {}).get("enum", []))
                or any(not safe_value(item.get("enum", [])) for item in public.get("oneOf", ()))):
            continue
        if key in arguments and not safe_value(arguments[key]):
            continue
        for label in ("title", "description"):
            if isinstance(schema.get(label), str):
                public[label] = sanitize_public_text(schema[label], limit=240)
        properties[key] = public
        if key in arguments:
            values[key] = deepcopy(arguments[key])
    return {"schema": {"type": "object", "properties": properties, "additionalProperties": False}, "values": values}


class EffectPlanError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedEffect:
    preview: Mapping[str, Any]
    snapshot_fingerprint: str
    arguments: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EffectPlan:
    plan_id: str
    owner: str
    session_id: str
    generation: int
    tool_name: str
    effect: ToolEffect
    arguments: Mapping[str, Any]
    snapshot_fingerprint: str
    preview: Mapping[str, Any]
    expires_at: float
    owner_generation: int = 0
    confirmation_contract: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    public_result: Mapping[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        expires_at = (
            datetime.fromtimestamp(self.expires_at, tz=timezone.utc).isoformat(
                timespec="seconds"
            )
            if self.expires_at >= 946_684_800
            else ""
        )
        result = {
            "plan_id": self.plan_id,
            "tool_name": self.tool_name,
            "effect": self.effect.value,
            "preview": deepcopy(dict(self.preview)),
            "expires_at": expires_at,
        }
        confirmation = self.confirmation_contract.get("audit_contract")
        if isinstance(confirmation, Mapping) and confirmation:
            result["confirmation"] = deepcopy(dict(confirmation))
        editing = self.metadata.get("editing") or {}
        if editing.get("schema", {}).get("properties"):
            result["editing"] = deepcopy(editing)
        if self.metadata.get("trust_eligible"):
            result["trust_eligible"] = True
            result["trust_scope"] = deepcopy(self.metadata["trust_scope"])
        return result

    def public_approval_dict(self) -> dict[str, Any]:
        """返回恢复待确认卡片时与事件公开载荷一致的字段。"""
        result = self.public_dict()
        result.setdefault("confirmation", {})
        result["result"] = deepcopy(dict(self.public_result))
        return result


class EffectPlanExpiredError(EffectPlanError):
    """票据已注销，只携带原计划供失败回执/只读复验使用。"""

    def __init__(self, plan: EffectPlan) -> None:
        super().__init__("effect plan expired")
        self.plan = plan


class EffectPlanStore(Protocol):
    def freeze(
        self,
        *,
        owner: str,
        session_id: str,
        generation: int,
        tool_name: str,
        effect: ToolEffect,
        arguments: Mapping[str, Any],
        prepared: PreparedEffect,
        public_result: Mapping[str, Any] | None = None,
        deadline: float | None = None,
    ) -> EffectPlan: ...

    def claim(
        self,
        *,
        owner: str,
        session_id: str,
        generation: int,
        plan_id: str,
    ) -> EffectPlan: ...

    def cancel(
        self,
        *,
        owner: str,
        session_id: str,
        generation: int,
        plan_id: str,
    ) -> EffectPlan | None: ...

    def get_active_plan(
        self,
        *,
        owner: str,
        session_id: str,
        generation: int,
        plan_id: str,
    ) -> EffectPlan | None: ...

    def get_plan_evidence(
        self, *, owner: str, session_id: str, generation: int, plan_id: str,
    ) -> EffectPlan | None: ...

class ConfirmationEffectPlanStore:
    """在既有一次性 ConfirmationStore 之上提供 Kernel EffectPlan。"""

    CONTRACT_VERSION = 1

    def __init__(
        self,
        store: ConfirmationStore | None = None,
        *,
        record_actions: bool = False,
    ) -> None:
        self.store = store or ConfirmationStore()
        self.record_actions = bool(record_actions)

    @staticmethod
    def _scoped_owner(owner: str, session_id: str) -> str:
        owner_key = str(owner or "").strip()
        session_key = str(session_id or "").strip()
        if not owner_key or not session_key:
            raise EffectPlanError("effect owner and session are required")
        return f"kernel:{owner_key}\x1f{session_key}"

    def freeze(
        self,
        *,
        owner: str,
        session_id: str,
        generation: int,
        tool_name: str,
        effect: ToolEffect,
        arguments: Mapping[str, Any],
        prepared: PreparedEffect,
        public_result: Mapping[str, Any] | None = None,
        deadline: float | None = None,
    ) -> EffectPlan:
        if effect is ToolEffect.READ:
            raise EffectPlanError("READ tools cannot create effect plans")
        scoped_owner = self._scoped_owner(owner, session_id)
        effective_arguments = dict(prepared.arguments or arguments)
        normalized_public_result = dict(public_result or {})
        edit_deadline = prepared.metadata.get("confirmation_deadline")
        if type(edit_deadline) in (int, float):
            remaining = edit_deadline - self.store._clock()
            if remaining <= 0:
                raise EffectPlanError("effect plan expired during edit")
            deadline = min(deadline if deadline is not None else float("inf"), time.monotonic() + remaining)
        contract = {
            "kernel_effect_version": self.CONTRACT_VERSION,
            "session_id": session_id,
            "generation": int(generation),
            "effect": effect.value,
            "preview": deepcopy(dict(prepared.preview)),
            "public_result": deepcopy(normalized_public_result),
            "metadata": deepcopy(dict(prepared.metadata)),
            "audit_risk": str(prepared.metadata.get("risk") or effect.value),
            "audit_contract": deepcopy(
                dict(prepared.metadata.get("confirmation") or {})
            ),
        }
        ticket = self.store.issue(
            owner=scoped_owner,
            tool_name=tool_name,
            arguments=effective_arguments,
            context_fingerprint=str(prepared.snapshot_fingerprint or ""),
            confirmation_contract=contract,
            replace_active_ticket=True,
            deadline=deadline,
        )
        return EffectPlan(
            plan_id=ticket.confirmation_id,
            owner=owner,
            session_id=session_id,
            generation=int(generation),
            tool_name=tool_name,
            effect=effect,
            arguments=deepcopy(ticket.arguments),
            snapshot_fingerprint=ticket.context_fingerprint,
            preview=deepcopy(dict(prepared.preview)),
            expires_at=ticket.expires_at,
            owner_generation=ticket.owner_generation,
            confirmation_contract=deepcopy(ticket.confirmation_contract),
            metadata=deepcopy(dict(prepared.metadata)),
            public_result=deepcopy(normalized_public_result),
        )

    @staticmethod
    def _restore_plan(
        ticket: Any,
        *,
        owner: str,
        session_id: str,
        generation: int,
    ) -> EffectPlan:
        contract = ticket.confirmation_contract
        try:
            version = int(contract.get("kernel_effect_version"))
            stored_session = str(contract.get("session_id") or "")
            stored_generation = int(contract.get("generation"))
            effect = ToolEffect(str(contract.get("effect") or ""))
            preview = contract.get("preview")
            public_result = contract.get("public_result", {})
            metadata = contract.get("metadata")
        except (TypeError, ValueError) as exc:
            raise EffectPlanError("effect plan payload is invalid") from exc
        if (
            version != ConfirmationEffectPlanStore.CONTRACT_VERSION
            or stored_session != session_id
        ):
            raise EffectPlanError("effect plan scope is invalid")
        if stored_generation != int(generation):
            raise EffectPlanError("effect plan is stale")
        if (
            not isinstance(preview, dict)
            or not isinstance(public_result, dict)
            or not isinstance(metadata, dict)
        ):
            raise EffectPlanError("effect plan payload is invalid")
        return EffectPlan(
            plan_id=ticket.confirmation_id,
            owner=owner,
            session_id=session_id,
            generation=stored_generation,
            tool_name=ticket.tool_name,
            effect=effect,
            arguments=deepcopy(ticket.arguments),
            snapshot_fingerprint=ticket.context_fingerprint,
            preview=deepcopy(preview),
            expires_at=ticket.expires_at,
            owner_generation=ticket.owner_generation,
            confirmation_contract=deepcopy(contract),
            metadata=deepcopy(metadata),
            public_result=deepcopy(public_result),
        )

    def claim(
        self,
        *,
        owner: str,
        session_id: str,
        generation: int,
        plan_id: str,
    ) -> EffectPlan:
        scoped_owner = self._scoped_owner(owner, session_id)
        # 先只读校验 generation/contract，再做一次性原子领取。旧回合不能
        # 通过“先领取后失败”撤销同会话更新的有效计划。
        preview_ticket = self.store.inspect_ticket(owner=scoped_owner, confirmation_id=plan_id)
        if preview_ticket is None:
            raise EffectPlanError("effect plan is unavailable")
        self._restore_plan(
            preview_ticket,
            owner=owner,
            session_id=session_id,
            generation=generation,
        )
        try:
            ticket = self.store.claim_and_rotate_owner(
                owner=scoped_owner,
                confirmation_id=plan_id,
                record_execution=self.record_actions,
                execution_owner=owner,
            )
        except ConfirmationExpiredError as exc:
            raise EffectPlanExpiredError(self._restore_plan(
                exc.ticket, owner=owner, session_id=session_id, generation=generation,
            )) from exc
        return self._restore_plan(
            ticket,
            owner=owner,
            session_id=session_id,
            generation=generation,
        )

    def get_plan_evidence(
        self, *, owner: str, session_id: str, generation: int, plan_id: str,
    ) -> EffectPlan | None:
        """仅在原确认窗口内恢复引用失效证据；不是写入授权或普通票据续期。"""
        try:
            ticket = self.store.inspect_ticket(owner=self._scoped_owner(owner, session_id), confirmation_id=plan_id)
            if ticket is None:
                return None
            plan = self._restore_plan(ticket, owner=owner, session_id=session_id, generation=generation)
            now = self.store._clock()
            recovery_until = plan.confirmation_contract.get("reference_recovery_until")
            edit_deadline = plan.metadata.get("confirmation_deadline")
            if type(edit_deadline) in (int, float) and edit_deadline <= now:
                return None
            return plan if plan.expires_at > now or (
                type(recovery_until) in (int, float) and recovery_until > now
            ) else None
        except EffectPlanError:
            return None

    def get_active_plan(
        self,
        *,
        owner: str,
        session_id: str,
        generation: int,
        plan_id: str,
    ) -> EffectPlan | None:
        """从唯一有效的确认快照恢复卡片；不依赖可淘汰的事件日志。"""
        plan = self.get_plan_evidence(owner=owner, session_id=session_id, generation=generation, plan_id=plan_id)
        return plan if plan is not None and plan.expires_at > self.store._clock() else None

    def cancel(
        self,
        *,
        owner: str,
        session_id: str,
        generation: int,
        plan_id: str,
    ) -> EffectPlan | None:
        scoped_owner = self._scoped_owner(owner, session_id)
        ticket = self.store.inspect_ticket(owner=scoped_owner, confirmation_id=plan_id)
        if ticket is None:
            return None
        try:
            plan = self._restore_plan(
                ticket,
                owner=owner,
                session_id=session_id,
                generation=generation,
            )
        except EffectPlanError:
            plan = None
        discarded = self.store.discard(
            owner=scoped_owner,
            confirmation_id=plan_id,
        )
        return plan if discarded and plan is not None and plan.expires_at > self.store._clock() else None

    def revoke_session(self, *, owner: str, session_id: str) -> int:
        """撤销指定 Kernel 会话仍有效的全部确认计划。"""
        return self.store.revoke_owner(
            owner=self._scoped_owner(owner, session_id),
        )
