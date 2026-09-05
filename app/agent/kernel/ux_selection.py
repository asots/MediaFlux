"""候选卡片的只读白名单投影与选择凭证；不执行领域工具。"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from app.agent.recent_resource_candidates import validate_safe_resource_snapshot
from app.sensitive_data import is_sensitive_key, redact_sensitive_text

from .projection import DefaultProjector, _model_safe
from .references import ReferenceError, ReferenceStore
from .state import CandidateSelectionGuard, SelectionInvalidError, SessionState
from .ux_display import display_text

CANDIDATE_VIEW_KEY = "ux_candidate_view"
RESOURCE_KIND = "resource_candidates"
SELECTION_KIND = "ux_resource_selection"
_REF_RE = re.compile(r"^ref_[A-Za-z0-9_-]{16,160}$")
_TAGS = {"resolution", "media", "video_codec", "effect", "audio"}



_RESOURCE_PRIVATE_KEY_RE = re.compile(
    r"^_|raw_?result|(?:download|torrent|detail)[_-]?(?:url|uri|link)|"
    r"magnet|credential|password|passwd|secret|cookie|authorization|api[_-]?key|token",
    re.IGNORECASE,
)
_RESOURCE_URL_RE = re.compile(r"(?i)\b(?:(?:https?|ftp|file|ed2k)\s*://|magnet:\?)[^\s<>]+")


def resource_model_content(content: str, *, maximum: int) -> str:
    """只净化已有模型 DTO 的私有枝叶，不用 UI 卡片覆盖业务结构或 model_data。"""
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: clean(child) for key, child in value.items()
                if key not in {"candidate_view", "selection"}
                and not is_sensitive_key(key) and not _RESOURCE_PRIVATE_KEY_RE.search(key)
            }
        if isinstance(value, list):
            return [clean(child) for child in value]
        if isinstance(value, str):
            return _model_safe(_RESOURCE_URL_RE.sub("[链接已隐藏]", redact_sensitive_text(value)))
        return value

    try:
        value = clean(json.loads(content))
    except (TypeError, ValueError):
        return str(clean(content))[:maximum]
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # 通常只会缩短已有投影；占位符增长时仍复用原 projector 的长度预算。
    return encoded if len(encoded) <= maximum else DefaultProjector(max_model_chars=maximum).project(value).model_content


def normalize_selection(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"ref", "position"}
        or not isinstance(value["ref"], str)
        or not _REF_RE.fullmatch(value["ref"])
        or type(value["position"]) is not int
        or not 1 <= value["position"] <= 12
    ):
        raise SelectionInvalidError()
    return {"ref": value["ref"], "position": value["position"]}


def _snapshot(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    # 绝不调用 restore_resource_candidate_reference：它会恢复 Provider 句柄。
    return validate_safe_resource_snapshot({
        key: value.get(key) for key in ("search_id", "search_status", "candidates")
    })


def _messages(value: Any, *, maximum: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(
        text for item in value[:maximum]
        if (text := display_text(item, limit=120))
    ))


def candidate_item(value: Mapping[str, Any], position: int) -> dict[str, Any]:
    quality = value.get("quality")
    quality = quality if isinstance(quality, dict) else value
    tags = quality.get("tags")
    return {
        "position": position,
        "title": display_text(value.get("title")) or f"候选 {position}",
        "site_name": display_text(value.get("site_name"), limit=80),
        "size_text": display_text(value.get("size_text"), limit=32),
        "tags": {
            key: text for key in sorted(_TAGS)
            if isinstance(tags, dict) and (text := display_text(tags.get(key), limit=64))
        },
        "reasons": _messages(quality.get("reasons"), maximum=6),
        "warnings": _messages(quality.get("warnings"), maximum=4),
    }


def _display_sources(public: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """仅从已有推荐 DTO 取解释，不能遍历 rawresult 或私有候选。"""
    data = public.get("data")
    if not isinstance(data, dict):
        return {}
    searches = [data]
    if isinstance(data.get("search"), dict):
        searches.append(data["search"])
    episodes = data.get("episodes")
    for episode in episodes[:3] if isinstance(episodes, list) else []:
        if isinstance(episode, dict) and isinstance(episode.get("search"), dict):
            searches.append(episode["search"])
    result: dict[str, Mapping[str, Any]] = {}
    for search in searches:
        items = search.get("items")
        candidates = list(items[:50]) if isinstance(items, list) else []
        recommendation = search.get("recommendation")
        if isinstance(recommendation, dict):
            alternatives = recommendation.get("alternatives")
            candidates += alternatives[:3] if isinstance(alternatives, list) else []
            candidates.append(recommendation.get("selected"))
        for item in candidates:
            if isinstance(item, dict) and isinstance(item.get("result_id"), str):
                result[item["result_id"]] = item
    return result


async def issue_candidate_view(
    *, store: ReferenceStore, owner: str, session_id: str, generation: int,
    ref: str, value: Any, ttl_seconds: int, public: Mapping[str, Any],
) -> dict[str, Any] | None:
    snapshot = _snapshot(value)
    if snapshot is None or not snapshot["candidates"]:
        return None
    ttl = max(1, min(int(ttl_seconds), 86_400))
    expires_at = time.time() + ttl
    sources = _display_sources(public)
    items: list[dict[str, Any]] = []
    for candidate in snapshot["candidates"]:
        position = candidate["position"]
        item = candidate_item({**candidate, **sources.get(candidate["result_id"], {})}, position)
        selection = await store.put(
            owner=owner, session_id=session_id, kind=SELECTION_KIND, ttl_seconds=ttl,
            value={"ref": ref, "position": position, "generation": generation, "expires_at": expires_at},
        )
        item["selection"] = {"ref": selection.ref, "position": position}
        items.append(item)
    return {"ref": ref, "expires_at": expires_at, "generation": generation, "items": items}


@dataclass(frozen=True, slots=True)
class ValidatedSelection:
    guard: CandidateSelectionGuard
    arguments: Mapping[str, Any]


async def validate_selection(
    value: Any, *, state: SessionState, store: ReferenceStore,
) -> ValidatedSelection:
    selection = normalize_selection(value)
    view = state.metadata.get(CANDIDATE_VIEW_KEY)
    try:
        if not isinstance(view, dict):
            raise SelectionInvalidError()
        guard = CandidateSelectionGuard(view["generation"], view["ref"], view["expires_at"])
        guard.check(state)
        items = view.get("items")
        if not isinstance(items, list) or not any(
            isinstance(item, dict) and item.get("position") == selection["position"]
            and item.get("selection") == selection for item in items
        ):
            raise SelectionInvalidError()
        bound = await store.resolve(
            selection["ref"], owner=state.owner, session_id=state.session_id,
            expected_kind=SELECTION_KIND,
        )
        if bound != {
            "ref": guard.ref, "position": selection["position"],
            "generation": guard.generation, "expires_at": guard.expires_at,
        }:
            raise SelectionInvalidError()
        resource = await store.resolve(
            guard.ref, owner=state.owner, session_id=state.session_id, expected_kind=RESOURCE_KIND,
        )
        snapshot = _snapshot(resource)
        if snapshot is None or not any(
            item["position"] == selection["position"] for item in snapshot["candidates"]
        ):
            raise SelectionInvalidError()
        guard.check(state)
        return ValidatedSelection(guard, {
            "source_type": RESOURCE_KIND, "resource_candidates_ref": guard.ref,
            "positions": [selection["position"]], "target": "preferred",
        })
    except (ReferenceError, KeyError, TypeError, ValueError) as exc:
        raise SelectionInvalidError() from exc


async def current_candidate_view(
    *, state: SessionState, store: ReferenceStore,
) -> dict[str, Any] | None:
    view = getattr(state, "metadata", {}).get(CANDIDATE_VIEW_KEY)
    if not isinstance(view, dict) or not isinstance(view.get("items"), list):
        return None
    items: list[dict[str, Any]] = []
    try:
        for raw in view["items"][:12]:
            if not isinstance(raw, dict):
                return None
            await validate_selection(raw.get("selection"), state=state, store=store)
            item = candidate_item(raw, raw["position"])
            item["selection"] = deepcopy(raw["selection"])
            items.append(item)
    except (SelectionInvalidError, KeyError, TypeError, ValueError):
        return None
    if not items:
        return None
    return {"ref": view["ref"], "expires_at": view["expires_at"], "items": items}


def candidate_result(public: Mapping[str, Any], view: dict[str, Any] | None) -> dict[str, Any]:
    """资源结果公开面整体使用白名单，不能在新卡旁边继续携带原始搜索 DTO。"""
    return {
        "ok": public.get("ok") is not False,
        "status": display_text(public.get("status"), limit=40) or "success",
        "summary": display_text(public.get("summary"), limit=600) or "资源候选已更新",
        "candidate_view": {key: deepcopy(value) for key, value in view.items() if key != "generation"}
        if view else None,
    }
