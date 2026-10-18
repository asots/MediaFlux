"""一次执行、一次投影的工具结果协议。"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.agent.model_context_budget import compact_tool_content
from app.agent.models import ToolReference
from app.sensitive_data import is_sensitive_key, redact_sensitive_text

from .state_types import StateUpdate

logger = logging.getLogger(__name__)

_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|token|secret|password|passwd|cookie|authorization|credential)",
    re.IGNORECASE,
)
_UNIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9._~:/-])/(?:[^\s/]+/)+[^\s]*")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])(?:[a-z]:[\\/]|\\\\)[^\s]+")
_PRIVATE_MODEL_KEY_RE = re.compile(
    r"(?:^|[_-])(?:absolute[_-]?path|database[_-]?id|internal[_-]?id|row[_-]?id)(?:$|[_-])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ReferenceValue:
    kind: str
    value: Any
    ttl_seconds: int = 900


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    model_content: str
    public_content: Mapping[str, Any]
    refs: tuple[ReferenceValue, ...] = ()
    state_updates: tuple[StateUpdate, ...] = ()
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    effect_plan: Any | None = None
    full_model_content: str = ""
    audit_content: Mapping[str, Any] | None = None

    def audit_result(self) -> dict[str, Any]:
        """Restore audit values without undoing later domain field removal.

        Candidate shaping and reference materialization remain authoritative;
        restoring their removed fields would revive private candidate IDs.
        """
        public = dict(self.public_content)
        materialized = {"refs", "reference_arguments", "result_handle", "read_tool",
                        "candidate_view", "task", "source_coverage"}
        return {key: self.audit_content[key]
                if self.audit_content is not None and key in self.audit_content and key not in materialized
                else value for key, value in public.items()}

    def model_message(self) -> str:
        return self.model_content


def _json_safe(value: Any, *, depth: int = 0, max_depth: int = 64) -> Any:
    if depth > max_depth:
        raise ValueError("工具结果结构超过安全保存范围，无法保留完整来源")
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise TypeError("工具结果对象键必须是字符串，不能无损转换")
            key = raw_key
            result[key] = (
                "[已隐藏]"
                if is_sensitive_key(key) or _SECRET_KEY_RE.search(key)
                else _json_safe(raw_value, depth=depth + 1, max_depth=max_depth)
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item, depth=depth + 1, max_depth=max_depth) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_safe(to_dict(), depth=depth + 1, max_depth=max_depth)
        except Exception as exc:  # noqa: BLE001 - arbitrary domain DTO boundary
            raise ValueError("工具结果无法转换为完整安全来源") from exc
    raise TypeError("工具结果包含不支持的非 JSON 值")


def _model_safe(value: Any, *, depth: int = 0, key: str = "") -> Any:
    safe = _json_safe(value) if depth == 0 else value
    # 唯一固定应用内路由，不把calendar_url变成任意文件路径/外链的放行口。
    if key == "calendar_url":
        return "/discovery/calendar" if isinstance(safe, str) and safe == "/discovery/calendar" else "[页面地址无效]"
    if _PRIVATE_MODEL_KEY_RE.search(str(key or "")):
        return "[内部标识已隐藏；请使用返回的 opaque ref]"
    if isinstance(safe, str) and (
        _UNIX_PATH_RE.search(safe) or _WINDOWS_PATH_RE.search(safe)
    ):
        return "[内部路径已隐藏；请使用返回的 opaque ref]"
    if isinstance(safe, dict):
        return {
            child_key: _model_safe(item, depth=depth + 1, key=child_key)
            for child_key, item in safe.items()
        }
    if isinstance(safe, list):
        return [_model_safe(item, depth=depth + 1, key=key) for item in safe]
    return safe


# 已提交参数保留业务值：限定 JSON 深度/容量并脱敏凭据，不套用 Provider 返回 DTO 的隐藏规则。
_ARGUMENT_MAX_DEPTH = 64
_ARGUMENT_MAX_BYTES = 256 * 1024


def _arguments_safe(value: Any) -> Any:
    remaining = _ARGUMENT_MAX_BYTES

    def validate(item: Any, depth: int) -> None:
        nonlocal remaining
        remaining -= len(item) if type(item) is str else 1
        if depth > _ARGUMENT_MAX_DEPTH or remaining < 0:
            raise ValueError("工具参数结构超过安全保存范围")
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("工具参数必须是 JSON 对象")
                remaining -= len(key)
                validate(child, depth + 1)
        elif type(item) is list:
            for child in item:
                validate(child, depth + 1)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("工具参数必须是 JSON 值")

    if not isinstance(value, Mapping):
        raise ValueError("工具参数必须是 JSON 对象")
    value = dict(value)  # Preserve the public root Mapping contract, not nested DTOs.
    validate(value, 0)
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("工具参数无法安全保存") from exc
    if len(encoded) > _ARGUMENT_MAX_BYTES:
        raise ValueError("工具参数超过安全保存容量")
    return _json_safe(value, max_depth=_ARGUMENT_MAX_DEPTH)


class DefaultProjector:
    """把领域返回值投影为模型 DTO 与公开 DTO，不做二次 LLM 润色。"""

    def __init__(self, *, max_model_chars: int = 24_000) -> None:
        self.max_model_chars = max(2_000, int(max_model_chars))

    async def before_read(self, tool, context, state_store) -> None:
        """领域可在读取开始前失效其展示快照；默认投影没有展示状态。"""

    def project_read(self, tool, value: Any) -> ToolOutcome:
        return self.project(value)

    async def present_references(self, outcome: ToolOutcome, references, *, context, reference_store) -> ToolOutcome:
        """引用已由执行层签发，领域只能补充展示与受控状态更新。"""
        return outcome

    def project(self, value: Any) -> ToolOutcome:
        if isinstance(value, ToolOutcome):
            return value
        def envelope(payload: Any) -> dict[str, Any]:
            if not isinstance(payload, dict):
                payload = {"data": payload}
            payload.setdefault("ok", True)
            payload.setdefault("status", "success")
            payload.setdefault("summary", "工具执行完成")
            return payload

        # Keep a complete, once-redacted JSON source, not a stringified DTO.
        audit = envelope(_json_safe(value))
        public = envelope(_model_safe(audit))
        to_model_dict = getattr(value, "to_model_dict", None)
        if callable(to_model_dict):
            try:
                model_payload = envelope(_model_safe(to_model_dict()))
            except Exception as exc:  # noqa: BLE001 - arbitrary domain DTO boundary
                # Never fall back to public data: the model converter can be a
                # privacy boundary, not merely a convenience serializer.
                raise ValueError("模型结果无法安全转换，未保存不完整来源") from exc
            if model_payload != public:
                coverage = {
                    "scope": "model_projection",
                    "complete": False,
                    "original_total": None,
                    "reason": "领域模型投影不同于公开来源；句柄仅续取该安全投影，不保证恢复原 DTO 的省略项",
                }
                model_payload["source_coverage"] = coverage
                public["source_coverage"] = coverage
        else:
            model_payload = public
        model_content = json.dumps(
            model_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        full_model_content = model_content
        model_content = compact_tool_content(model_content, maximum=self.max_model_chars)
        public = json.loads(compact_tool_content(
            json.dumps(public, ensure_ascii=False, separators=(",", ":")),
            maximum=self.max_model_chars * 2,
        ))
        audit = json.loads(compact_tool_content(
            json.dumps(audit, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            maximum=self.max_model_chars * 2,
        ))
        raw_references = getattr(value, "references", ())
        references: list[ReferenceValue] = []
        if isinstance(raw_references, Sequence) and not isinstance(
            raw_references, (str, bytes, bytearray)
        ):
            for item in raw_references:
                if not isinstance(item, ToolReference):
                    raise TypeError("tool result contains an invalid reference")
                references.append(
                    ReferenceValue(
                        kind=item.kind,
                        value=item.value,
                        ttl_seconds=item.ttl_seconds,
                    )
                )
        return ToolOutcome(
            model_content=model_content,
            public_content=public,
            audit_content=audit,
            refs=tuple(references),
            full_model_content=full_model_content if full_model_content != model_content else "",
        )
