"""Explicit cross-protocol route/model caps for main sessions only."""
from __future__ import annotations

import json
from typing import Any

from app.agent.model_catalog import normalize_provider_model_id
from app.agent.model_context_budget import conservative_tokens
from app.clients.openai_compatible import normalize_provider_location

BudgetProfiles = dict[tuple[str, str], tuple[int, int, int]]


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate field")
    return result


def parse_budget_profiles(raw: str) -> BudgetProfiles:
    try:
        return _parse_budget_profiles(raw)
    except (ValueError, TypeError, RecursionError):
        pass
    # Raise outside the handler, without a retained unsafe exception graph.
    raise ValueError("模型预算档案格式无效或超过安全上限")


def _parse_budget_profiles(raw: str) -> BudgetProfiles:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 65_536:
        raise ValueError("invalid size")
    rows = json.loads(raw, object_pairs_hook=_unique_fields) if raw.strip() else []
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError("invalid list")
    profiles: BudgetProfiles = {}
    fields = {"base_url", "model", "context_window_tokens", "max_output_tokens"}
    for row in rows:
        if not isinstance(row, dict) or not fields <= row.keys() or row.keys() - fields - {"reasoning_reserved_tokens"}:
            raise ValueError("invalid fields")
        if not isinstance(row["base_url"], str) or not isinstance(row["model"], str):
            raise ValueError("invalid identity")
        row["base_url"].encode("utf-8")
        row["model"].encode("utf-8")
        if not row["model"] or normalize_provider_model_id(row["model"]) != row["model"]:
            raise ValueError("invalid model")
        context, output = row["context_window_tokens"], row["max_output_tokens"]
        if type(context) is not int or type(output) is not int:
            raise ValueError("integer capacities required")
        if not 1_024 <= context <= 2_000_000 or not 1 <= output <= context:
            raise ValueError("invalid capacities")
        reasoning = row.get("reasoning_reserved_tokens", 0)
        if type(reasoning) is not int or not 0 <= reasoning <= context - output:
            raise ValueError("invalid reasoning reserve")
        identity = (normalize_provider_location(row["base_url"]).base_url, row["model"])
        if identity in profiles:
            raise ValueError("duplicate identity")
        profiles[identity] = (context, output, reasoning)
    return profiles


def is_window_feedback(status: int, raw: bytes) -> bool:
    """Only explicit structured rejection codes, never prose or claimed limits."""
    if status not in {400, 413, 422} or len(raw) > 65_536:
        return False
    try:
        envelope = json.loads(raw, object_pairs_hook=_unique_fields)
        error = envelope.get("error") if isinstance(envelope, dict) else None
        if not isinstance(error, dict):
            return False
        code = error.get("code") or error.get("type")
        return isinstance(code, str) and code in {
            "context_length_exceeded", "token_limit_exceeded",
        }
    except (ValueError, TypeError, RecursionError):
        return False


def reduced_output_body(body: dict[str, Any], context: int, output: int) -> dict[str, Any] | None:
    """One request-local reduction; retain the entire native body and margin.

    A rejected request has emitted nothing. Its output ceiling may be halved,
    but history/tools/private state and configured reasoning reserve cannot be
    weakened. Provider-provided numbers are deliberately not capacity authority.
    """
    keys = [key for key in ("max_tokens", "max_completion_tokens", "max_output_tokens") if key in body]
    if len(keys) != 1:
        return None
    key = keys[0]
    previous = body[key]
    if type(previous) is not int or previous <= 256:
        return None
    candidate = dict(body)
    candidate[key] = min(previous // 2, output)
    candidate[key] = min(candidate[key], context - conservative_tokens(candidate) - 512)
    if candidate[key] < 256 or candidate[key] >= previous:
        return None
    if conservative_tokens(candidate) + candidate[key] + 512 > context:
        return None
    return candidate


def serialize_budget_profiles(raw: str) -> str:
    parse_budget_profiles(raw)
    # Save validated raw spelling: route normalization is not idempotent.
    rows = json.loads(raw) if raw.strip() else []
    serialized = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    for separator in "\x85\u2028\u2029":
        serialized = serialized.replace(separator, f"\\u{ord(separator):04x}")
    parse_budget_profiles(serialized)
    return serialized
