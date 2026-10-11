"""Provider normalization/security fences followed by the shared schema semantics."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from app.agent.kernel.pipeline import ToolPipelineError, _validate_json_schema
from app.agent.provider_models import ProviderGatewayError

_FORBIDDEN_ARGUMENT_KEYS = frozenset({
    "url", "base_url", "host", "hostname", "port", "scheme",
    "headers", "header", "authorization", "token", "api_key",
    "cookie", "cookies", "password", "username", "method", "endpoint",
})


def _error(message: str) -> ProviderGatewayError:
    return ProviderGatewayError(message, code="invalid_arguments")


def _normalize_value(name: str, value: Any, schema: Mapping[str, Any]) -> Any:
    if value is None and schema.get("nullable") is True:
        return None
    expected = schema.get("type")
    if expected == "string" and isinstance(value, str):
        return value.strip()
    if expected == "integer" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if expected == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
        except OverflowError:
            raise _error(f"{name} 必须是有限数字") from None
        if not math.isfinite(number):
            raise _error(f"{name} 必须是有限数字")
        return number
    if expected == "boolean" and isinstance(value, bool):
        return value
    if expected == "array" and isinstance(value, list):
        items = schema.get("items")
        if not isinstance(items, Mapping):
            items = {}
        return [_normalize_value(f"{name}[{index}]", item, items) for index, item in enumerate(value)]
    if expected == "object" and isinstance(value, dict):
        return _normalize_object(schema, value)
    raise _error(f"{name} 参数类型无效或不支持")


def _normalize_object(schema: Mapping[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        properties = {}
    if set(arguments) - set(properties) or set(arguments) & _FORBIDDEN_ARGUMENT_KEYS:
        raise _error("Provider arguments 包含未开放字段")
    missing = set(schema.get("required", ())) - set(arguments)
    if missing:
        raise _error(f"缺少必要参数：{', '.join(sorted(missing))}")
    normalized: dict[str, Any] = {}
    for name, child in properties.items():
        if not isinstance(child, Mapping):
            raise _error("Provider operation schema 无效")
        if name in arguments or "default" in child:
            if name in _FORBIDDEN_ARGUMENT_KEYS:
                raise _error("Provider arguments 包含未开放字段")
            normalized[name] = _normalize_value(name, arguments.get(name, child.get("default")), child)
    return normalized


def _validation_schema(schema: Mapping[str, Any], *, normalization: bool = True) -> dict[str, Any]:
    """Adapt legacy nullable/default bounds, not a second constraint evaluator.

    Provider fields retain explicit scalar/object/array types. Composition is
    checked by the kernel on normalized data; branch-local defaults are not
    guessed. This is the kernel's limited schema dialect, not full JSON Schema.
    """
    result = dict(schema)
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        result["properties"] = {key: _validation_schema(child, normalization=normalization)
                                for key, child in properties.items() if isinstance(child, Mapping)}
    if isinstance(schema.get("items"), Mapping):
        result["items"] = _validation_schema(schema["items"], normalization=normalization)
    for key in ("oneOf", "anyOf"):
        if isinstance(schema.get(key), list):
            result[key] = [_validation_schema(branch, normalization=False) for branch in schema[key]]
    if isinstance(schema.get("not"), Mapping):
        result["not"] = _validation_schema(schema["not"], normalization=False)
    if normalization:
        if schema.get("type") == "string":
            result.setdefault("maxLength", 4096)
        elif schema.get("type") == "integer":
            result.setdefault("minimum", -(2**63))
            result.setdefault("maximum", 2**63 - 1)
        elif schema.get("type") == "array":
            result.setdefault("maxItems", 100)
    if result.pop("nullable", False) is True:
        return {"anyOf": [{"type": "null"}, result]}
    return result


def validate_provider_arguments(
    schema: Mapping[str, Any], arguments: dict[str, Any] | None
) -> dict[str, Any]:
    """Normalize Provider inputs without granting network or reference authority."""
    if not isinstance(arguments, dict):
        raise _error("Provider arguments 必须是对象")
    if str(schema.get("type") or "object") != "object":
        raise _error("Provider operation schema 无效")
    normalized = _normalize_object(schema, arguments)
    try:
        _validate_json_schema(normalized, _validation_schema(schema))
    except ToolPipelineError as exc:
        raise _error(str(exc)) from None
    return normalized
