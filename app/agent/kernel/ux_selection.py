"""候选卡片的只读白名单投影与选择凭证；不执行领域工具。"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from app.agent.public_safety import sanitize_resource_title
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
    """新协议只接受批次凭证、明确位置集合和目标；旧单卡协议只读。"""
    if (
        not isinstance(value, dict) or set(value) != {"ref", "positions", "target"}
        or not isinstance(value["ref"], str) or not _REF_RE.fullmatch(value["ref"])
        or not isinstance(value["positions"], list) or not 1 <= len(value["positions"]) <= 12
        or any(type(pos) is not int or not 1 <= pos <= 12 for pos in value["positions"])
        or len(set(value["positions"])) != len(value["positions"])
        or value["target"] not in ("qb", "guangya", "both")
    ):
        raise SelectionInvalidError()
    return {"ref": value["ref"], "positions": sorted(value["positions"]), "target": value["target"]}


def _target_options(owner: str) -> dict[str, Any]:
    # 配置/登录状态来自领域现有契约；只用于展示，提交仍会再次预检。
    from app.agent.indexer_actions import download_target_readiness
    from app.agent.media_consumption_actions import explicit_preferred_download_target

    ready = download_target_readiness("both")
    preferred = explicit_preferred_download_target(owner)
    return {
        "target": preferred or "guangya",
        "target_source": "saved_preference" if preferred else "product_default",
        "targets": [
            {"value": name, "label": label, "available": available}
            for name, label, available in (
                ("qb", "qBittorrent", ready.get("qb", False)),
                ("guangya", "光鸭", ready.get("guangya", False)),
                ("both", "两个目标", all(ready.values()) and len(ready) == 2),
            )
        ],
    }


def _recommend(items: list[dict[str, Any]]) -> list[int]:
    """只推荐领域已确认覆盖目标缺集的互补资源；普通搜索不推断下载意图。"""
    covered: set[tuple[str, int, int]] = set()
    selected = []
    for item in items:
        if not (item.get("media_title") and item.get("requested_episode")
                and item.get("match") in {"exact_episode", "episode_pack"}):
            continue
        title = item["media_scope"]
        season, episode = item["requested_episode"]
        if (title, season, episode) in covered:
            continue
        selected.append(item["position"])
        coverage = item.get("coverage")
        start, end = (coverage[1], coverage[2]) if coverage and coverage[0] == season and coverage[1] <= episode <= coverage[2] else (episode, episode)
        covered.update((title, season, number) for number in range(start, end + 1))
    return selected


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
    from app.modules.episode_mapping import extract_release_episode_range

    title = sanitize_resource_title(value.get("title")) or f"候选 {position}"
    season, start, end = extract_release_episode_range(title)
    coverage = [season, start, end] if start and end and end - start < 1000 else None
    verification = value.get("_verification_context")
    verification = verification if isinstance(verification, dict) else {}
    requested = [verification.get("season"), verification.get("episode")] if verification else value.get("requested_episode")
    if not (isinstance(requested, list) and len(requested) == 2
            and type(requested[0]) is int and 1 <= requested[0] <= 100
            and type(requested[1]) is int and 1 <= requested[1] <= 1000):
        requested = None
    media_title = display_text(verification.get("title") or value.get("media_title"), limit=120)
    media_scope = (f"tmdb:{verification['tmdb_id']}" if verification.get("tmdb_id")
                   else display_text(value.get("media_scope"), limit=140) or media_title)
    quality = value.get("quality")
    quality = quality if isinstance(quality, dict) else value
    tags = quality.get("tags")
    match = quality.get("match")
    return {
        "position": position,
        "title": title,
        "coverage": coverage,
        "media_title": media_title,
        "media_scope": media_scope,
        "requested_episode": requested,
        "match": match if match in ("exact_episode", "episode_pack", "season_pack", "unknown", "conflict") else "",
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


def _candidate_summary(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """缺集无覆盖证据不挂卡，普通搜索保留手动候选；不代替查询结论。"""
    recommended = _recommend(items)
    if not items or (not recommended and all(item.get("requested_episode") for item in items)):
        return None
    return {"items": items, "recommended_positions": recommended}


async def issue_candidate_view(
    *, store: ReferenceStore, owner: str, session_id: str, generation: int,
    ref: str, value: Any, ttl_seconds: int, public: Mapping[str, Any], turn_id: str = "",
) -> dict[str, Any] | None:
    snapshot = _snapshot(value)
    if snapshot is None or not snapshot["candidates"]:
        return None
    ttl = max(1, min(int(ttl_seconds), 86_400))
    expires_at = time.time() + ttl
    sources = _display_sources(public)
    items = [candidate_item({**candidate, **sources.get(candidate["result_id"], {})}, candidate["position"])
             for candidate in snapshot["candidates"]]
    summary = _candidate_summary(items)
    if summary is None:
        return None
    selection = await store.put(
        owner=owner, session_id=session_id, kind=SELECTION_KIND, ttl_seconds=ttl,
        value={"ref": ref, "generation": generation, "expires_at": expires_at},
    )
    return {
        "ref": ref, "selection_ref": selection.ref, "expires_at": expires_at,
        "generation": generation, "turn_id": turn_id, **summary,
        **await asyncio.to_thread(_target_options, owner),
    }



@dataclass(frozen=True, slots=True)
class ValidatedSelection:
    guard: CandidateSelectionGuard
    arguments: Mapping[str, Any]


async def validate_selection(
    value: Any, *, state: SessionState, store: ReferenceStore, for_preview: bool = True,
) -> ValidatedSelection:
    selection = normalize_selection(value)
    view = state.metadata.get(CANDIDATE_VIEW_KEY)
    try:
        if not isinstance(view, dict):
            raise SelectionInvalidError()
        guard = CandidateSelectionGuard(
            state.generation, view["ref"], view["expires_at"], batch_generation=view["generation"],
        )
        guard.check(state)
        items = view.get("items")
        if (
            not isinstance(items, list) or selection["ref"] != view.get("selection_ref")
            or not set(selection["positions"]).issubset({item["position"] for item in items})
        ):
            raise SelectionInvalidError()
        bound = await store.resolve(
            selection["ref"], owner=state.owner, session_id=state.session_id,
            expected_kind=SELECTION_KIND,
        )
        if bound != {
            "ref": guard.ref, "generation": guard.batch_generation, "expires_at": guard.expires_at,
        }:
            raise SelectionInvalidError()
        resource = await store.resolve(
            guard.ref, owner=state.owner, session_id=state.session_id, expected_kind=RESOURCE_KIND,
        )
        snapshot = _snapshot(resource)
        if snapshot is None or not set(selection["positions"]).issubset(
            {item["position"] for item in snapshot["candidates"]}
        ):
            raise SelectionInvalidError()
        pending = state.metadata.get("ux_candidate_plan")
        if (
            for_preview and state.pending_effect_plan_id and isinstance(pending, dict)
            and pending.get("plan_id") == state.pending_effect_plan_id and pending.get("ref") == guard.ref
        ):
            raise SelectionInvalidError("已有这批资源的待确认计划，请先确认或取消后再改选。")
        guard.check(state)
        return ValidatedSelection(guard, {
            "source_type": RESOURCE_KIND, "resource_candidates_ref": guard.ref,
            "positions": selection["positions"], "target": selection["target"],
        })
    except (ReferenceError, KeyError, TypeError, ValueError) as exc:
        raise SelectionInvalidError() from exc


async def current_candidate_view(
    *, state: SessionState, store: ReferenceStore,
) -> dict[str, Any] | None:
    view = getattr(state, "metadata", {}).get(CANDIDATE_VIEW_KEY)
    if not isinstance(view, dict) or not isinstance(view.get("items"), list):
        return None
    try:
        await validate_selection({
            "ref": view.get("selection_ref"),
            "positions": [item["position"] for item in view["items"]], "target": "guangya",
        }, state=state, store=store, for_preview=False)
        public = {key: deepcopy(view[key]) for key in (
            "ref", "selection_ref", "expires_at", "turn_id",
        ) if key in view}
        summary = _candidate_summary([candidate_item(item, item["position"]) for item in view["items"]])
        if summary is None:
            return None
        public.update(summary)
        result = state.metadata.get("ux_candidate_result")
        if isinstance(result, dict) and result.get("ref") == view["ref"]:
            public["last_result"] = {key: deepcopy(result[key]) for key in ("text", "target", "handled_positions") if key in result}
        public.update(await asyncio.to_thread(_target_options, state.owner))
        return public
    except (SelectionInvalidError, KeyError, TypeError, ValueError):
        return None


def candidate_result(public: Mapping[str, Any], view: dict[str, Any] | None) -> dict[str, Any]:
    """资源结果公开面整体使用白名单，不能在新卡旁边继续携带原始搜索 DTO。"""
    return {
        "ok": public.get("ok") is not False,
        "status": display_text(public.get("status"), limit=40) or "success",
        "summary": display_text(public.get("summary"), limit=600) or "资源候选已更新",
        "candidate_view": {key: deepcopy(value) for key, value in view.items() if key != "generation"}
        if view else None,
    }
