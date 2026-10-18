"""发现详情、跨源 TMDB 候选与确认映射。"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app import config
from app import database as db
from app.agent.confirmation import confirmation_context_fingerprint
from app.agent.errors import AgentToolError
from app.agent.models import Evidence, ToolContext, ToolResult
from app.agent.public_safety import sanitize_public_text
from app.agent.session_context import (
    AgentContextWriteGuard,
    AgentSessionContextRepository,
)
from app.discovery.service import get_discovery_service

_ALLOWED_PROVIDERS = {"tmdb", "douban", "bangumi"}
_ALLOWED_MEDIA_TYPES = {"movie", "tv"}
_PUBLIC_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,180}$")


@dataclass
class _MappingSnapshot:
    scope_key: str
    provider: str
    external_id: str
    media_type: str
    title: str
    year: str
    candidates: list[dict[str, Any]]
    created_at: float
    generation: int = 0
    revision: int = 0


_TTL_SECONDS = 600.0
_CONTEXT_TYPE = "discovery_mapping"
logger = logging.getLogger(__name__)


def _scope_key(owner: str, session_id: str) -> str:
    owner_key, session_key = owner.strip(), session_id.strip()
    if not owner_key or not session_key:
        raise AgentToolError("映射候选需要已登录会话", code="precondition_failed")
    return hashlib.sha256(json.dumps([owner_key, session_key]).encode()).hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe(value: Any, limit: int) -> str:
    return sanitize_public_text(value, limit=limit)


def _identity(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict) or set(arguments) != {
        "provider",
        "external_id",
        "media_type",
    }:
        raise AgentToolError("必须且只能提供 provider、external_id 和 media_type")
    provider = str(arguments["provider"] or "").strip().casefold()
    external_id = str(arguments["external_id"] or "").strip()
    media_type = str(arguments["media_type"] or "").strip().casefold()
    if provider not in _ALLOWED_PROVIDERS:
        raise AgentToolError("provider 仅支持 tmdb、douban 或 bangumi")
    if media_type not in _ALLOWED_MEDIA_TYPES:
        raise AgentToolError("media_type 仅支持 movie 或 tv")
    if not _PUBLIC_ID_RE.fullmatch(external_id):
        raise AgentToolError("external_id 格式无效")
    return {"provider": provider, "external_id": external_id, "media_type": media_type}


def discovery_detail_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return _identity(arguments)


def discovery_mapping_candidates_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return _identity(arguments)


def discovery_confirm_mapping_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict) or set(arguments) != {"candidate_number"}:
        raise AgentToolError("必须且只能提供 candidate_number")
    value = arguments["candidate_number"]
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise AgentToolError("candidate_number 必须是 1 到 5 的整数")
    return {"candidate_number": value}


def _feature_disabled() -> ToolResult:
    return ToolResult(
        False, "disabled", "影视探索功能当前已关闭", error="请先启用影视探索。"
    )


def get_discovery_detail(
    arguments: dict[str, Any], _context: ToolContext
) -> ToolResult:
    if not config.get_bool("DISCOVERY_ENABLED"):
        return _feature_disabled()
    card = get_discovery_service().get_detail(
        arguments["provider"], arguments["media_type"], arguments["external_id"]
    )
    if card is None:
        return ToolResult(
            False, "not_found", "没有找到该影视详情", error="来源详情不存在。"
        )
    rating = None
    try:
        rating = (
            round(max(0.0, min(float(card.rating), 10.0)), 1)
            if card.rating is not None
            else None
        )
    except (TypeError, ValueError, OverflowError):
        rating = None
    return ToolResult(
        True,
        "completed",
        f"已读取《{_safe(card.title, 160) or '未命名条目'}》的发现详情",
        data={
            "provider": card.provider,
            "external_id": _safe(card.external_id, 180),
            "tmdb_id": _safe(card.external_id if card.provider == "tmdb" else card.tmdb_id, 32),
            "stable_id": _safe(card.stable_id, 240),
            "media_type": card.media_type,
            "title": _safe(card.title, 160) or "未命名条目",
            "original_title": _safe(card.original_title, 160),
            "year": _safe(card.year, 12),
            "release_date": _safe(card.release_date, 24),
            "rating": rating,
            "rating_source": _safe(card.rating_source, 30),
            "overview": _safe(card.overview, 500),
            "number_of_seasons": card.number_of_seasons,
            "number_of_episodes": card.number_of_episodes,
            "seasons": [
                {"season_number": item["season_number"],
                 "name": _safe(item["name"], 160),
                 "episode_count": item["episode_count"],
                 "air_date": _safe(item["air_date"], 24)}
                for item in card.seasons
            ] if card.seasons is not None else None,
            "seasons_note": (
                "TMDB 默认季序，不是发布组篇章或 Episode Groups；季号 0 为特别篇，"
                "不计入总正片季集数。总数与各季计数保留来源原值；null 表示未返回有效数据，"
                "不能当成 0，也不能用演员出场次数推断。"
                if card.provider == "tmdb" and card.media_type == "tv"
                else "季集字段为 null 时表示未返回或不适用，不能当成 0。"
            ),
            "credits_state": "not_queried",
            "credits_note": "本详情含作品概况与来源季集统计，未查询演职员；未返回演员字段不表示源站为空或官方未公布。",
            "mapping_confirmed": bool(
                card.provider == "tmdb"
                or (
                    (
                        row := db.get_media_external_id(
                            card.provider, card.external_id, card.media_type
                        )
                    )
                    and bool(row["confirmed"])
                )
            ),
        },
        evidence=[
            Evidence(
                "discovery_detail",
                "只读取来源身份、概况、默认季集统计与映射确认状态，未查询演员表；未保存映射、收藏、订阅或下载任务。",
                _now(),
            )
        ],
    )


def _snapshot_payload(snapshot: _MappingSnapshot) -> dict[str, Any]:
    return {
        "provider": snapshot.provider,
        "external_id": snapshot.external_id,
        "media_type": snapshot.media_type,
        "title": snapshot.title,
        "year": snapshot.year,
        "candidates": [dict(item) for item in snapshot.candidates[:5]],
    }


def _snapshot_from_payload(
    scope_key: str,
    payload: dict[str, Any],
    *,
    generation: int = 0,
    revision: int = 0,
) -> _MappingSnapshot | None:
    try:
        provider = str(payload.get("provider") or "").strip().casefold()
        external_id = str(payload.get("external_id") or "").strip()
        media_type = str(payload.get("media_type") or "").strip().casefold()
        raw_candidates = payload.get("candidates")
        if (
            provider not in _ALLOWED_PROVIDERS
            or media_type not in _ALLOWED_MEDIA_TYPES
            or not _PUBLIC_ID_RE.fullmatch(external_id)
            or not isinstance(raw_candidates, list)
        ):
            return None
        candidates: list[dict[str, Any]] = []
        for raw in raw_candidates[:5]:
            if not isinstance(raw, dict):
                return None
            number = raw.get("candidate_number")
            tmdb_id = str(raw.get("tmdb_id") or "").strip()
            candidate_type = str(raw.get("media_type") or "").strip().casefold()
            if (
                isinstance(number, bool)
                or not isinstance(number, int)
                or number != len(candidates) + 1
                or not tmdb_id.isdigit()
                or candidate_type not in _ALLOWED_MEDIA_TYPES
            ):
                return None
            score = float(raw.get("score") or 0)
            if not 0 <= score <= 1:
                return None
            candidates.append(
                {
                    "candidate_number": number,
                    "tmdb_id": tmdb_id,
                    "title": _safe(raw.get("title"), 160) or "未命名候选",
                    "year": _safe(raw.get("year"), 12),
                    "media_type": candidate_type,
                    "score": round(score, 3),
                }
            )
        return _MappingSnapshot(
            scope_key=scope_key,
            provider=provider,
            external_id=external_id,
            media_type=media_type,
            title=_safe(payload.get("title"), 160),
            year=_safe(payload.get("year"), 12),
            candidates=candidates,
            created_at=time.monotonic(),
            generation=max(0, (generation or 0)),
            revision=max(0, (revision or 0)),
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _current_mapping_state(snapshot: _MappingSnapshot) -> dict[str, Any] | None:
    existing = db.get_media_external_id(
        snapshot.provider, snapshot.external_id, snapshot.media_type
    )
    return (
        None
        if existing is None
        else {
            "tmdb_id": str(existing["tmdb_id"] or ""),
            "confirmed": bool(existing["confirmed"]),
            "version": max(0, int(existing["version"] or 0)),
            "updated_at": str(existing["updated_at"] or ""),
        }
    )


def _mapping_context(
    snapshot: _MappingSnapshot,
    candidate: dict[str, Any],
    *,
    existing_state: dict[str, Any] | None = None,
    state_loaded: bool = False,
) -> str:
    if not state_loaded:
        existing_state = _current_mapping_state(snapshot)
    return confirmation_context_fingerprint(
        {
            "source": [
                snapshot.provider,
                snapshot.external_id,
                snapshot.media_type,
                snapshot.title,
                snapshot.year,
            ],
            "candidate": candidate,
            "existing": existing_state,
        },
        domain="discovery-confirm-mapping",
    )


class DiscoveryMappingActions:
    """由单个 runtime 持有的领域动作与短期上下文。"""

    def __init__(self, repository: AgentSessionContextRepository | None = None) -> None:
        self._repository = repository
        self._lock = threading.RLock()
        self._recent: dict[str, _MappingSnapshot] = {}
        self._generation = 0
        self._memory_guards: dict[str, tuple[AgentContextWriteGuard, float]] = {}

    def clear_session(self, *, owner: str, session_id: str) -> None:
        scope_key = _scope_key(owner, session_id)
        with self._lock:
            self._recent.pop(scope_key, None)
            self._memory_guards.pop(scope_key, None)
            if self._repository is not None:
                self._repository.invalidate_context(
                    owner=scope_key, context_type=_CONTEXT_TYPE,
                )

    def _prune_locked(self) -> None:
        now = time.monotonic()
        expired = {key for key, snapshot in self._recent.items()
                   if now - snapshot.created_at > _TTL_SECONDS}
        expired.update(key for key, (_, expires_at) in self._memory_guards.items()
                       if expires_at <= now)
        for key in expired:
            self._recent.pop(key, None)
            self._memory_guards.pop(key, None)

    def _begin_snapshot(self, scope_key: str) -> AgentContextWriteGuard:
        with self._lock:
            self._prune_locked()
            self._recent.pop(scope_key, None)
            if self._repository is not None:
                return self._repository.begin_context(owner=scope_key, context_type=_CONTEXT_TYPE)
            self._generation += 1
            guard = AgentContextWriteGuard(generation=self._generation, revision=0)
            self._memory_guards[scope_key] = guard, time.monotonic() + _TTL_SECONDS
            return guard

    def _store_snapshot(self, snapshot: _MappingSnapshot) -> bool:
        with self._lock:
            self._prune_locked()
            guard = AgentContextWriteGuard(snapshot.generation, snapshot.revision)
            if self._repository is not None:
                try:
                    persisted = self._repository.replace_latest_guarded(
                        owner=snapshot.scope_key,
                        context_type=_CONTEXT_TYPE,
                        payload=_snapshot_payload(snapshot),
                        expires_at=time.time() + _TTL_SECONDS,
                        guard=guard,
                    )
                except Exception as exc:
                    logger.warning("Agent 发现映射上下文持久化失败 type=%s", type(exc).__name__)
                    persisted = None
                if persisted is None:
                    self._recent.pop(snapshot.scope_key, None)
                    return False
                snapshot.generation, snapshot.revision = persisted.generation, persisted.revision
            else:
                current = self._memory_guards.get(snapshot.scope_key)
                if current is None or current[0] != guard:
                    return False
                snapshot.revision += 1
                self._memory_guards[snapshot.scope_key] = (
                    AgentContextWriteGuard(snapshot.generation, snapshot.revision),
                    time.monotonic() + _TTL_SECONDS,
                )
            snapshot.created_at = time.monotonic()
            self._recent[snapshot.scope_key] = deepcopy(snapshot)
            return True

    def _consume_snapshot(self, snapshot: _MappingSnapshot) -> bool:
        with self._lock:
            self._prune_locked()
            guard = AgentContextWriteGuard(snapshot.generation, snapshot.revision)
            if self._repository is not None:
                try:
                    consumed = self._repository.consume_latest_guarded(
                        owner=snapshot.scope_key, context_type=_CONTEXT_TYPE, guard=guard,
                    )
                except Exception as exc:
                    logger.warning("Agent 发现映射上下文消费失败 type=%s", type(exc).__name__)
                    return False
                if not consumed:
                    return False
            else:
                current = self._memory_guards.get(snapshot.scope_key)
                if current is None or current[0] != guard:
                    return False
                self._memory_guards.pop(snapshot.scope_key, None)
            self._recent.pop(snapshot.scope_key, None)
            return True

    def _get_snapshot(self, scope_key: str) -> _MappingSnapshot | None:
        with self._lock:
            self._prune_locked()
            if self._repository is not None:
                try:
                    persisted = self._repository.get_latest(
                        owner=scope_key, context_type=_CONTEXT_TYPE, now=time.time(),
                    )
                except Exception as exc:
                    logger.warning("Agent 发现映射上下文恢复失败 type=%s", type(exc).__name__)
                    persisted = None
                restored = (
                    _snapshot_from_payload(scope_key, persisted.payload,
                        generation=persisted.generation, revision=persisted.revision)
                    if persisted is not None and persisted.generation > 0 and persisted.revision > 0
                    else None
                )
                if restored is None:
                    self._recent.pop(scope_key, None)
                    return None
                self._recent[scope_key] = restored
            return deepcopy(self._recent.get(scope_key))

    def get_discovery_mapping_candidates(
        self,
        arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        scope_key = _scope_key(context.owner, context.session_id)
        if not config.get_bool("DISCOVERY_ENABLED"):
            return _feature_disabled()
        try:
            guard = self._begin_snapshot(scope_key)
        except Exception as exc:
            logger.warning("Agent 发现映射上下文初始化失败 type=%s", type(exc).__name__)
            raise AgentToolError(
                "映射候选上下文暂时不可用", code="precondition_failed"
            ) from exc
        service = get_discovery_service()
        card = service.get_detail(
            arguments["provider"], arguments["media_type"], arguments["external_id"]
        )
        if card is None:
            return ToolResult(
                False, "not_found", "没有找到该影视详情", error="来源详情不存在。"
            )
        result = service.lookup_tmdb_mapping(
            card.provider, card.external_id, card.media_type, card.title, card.year
        )
        raw_candidates = (
            result.get("candidates") if isinstance(result.get("candidates"), list) else []
        )
        candidates: list[dict[str, Any]] = []
        for raw in raw_candidates[:5]:
            if not isinstance(raw, dict):
                continue
            tmdb_id = str(raw.get("tmdb_id") or "").strip()
            if not tmdb_id.isdigit():
                continue
            try:
                score = round(max(0.0, min(float(raw.get("score") or 0), 1.0)), 3)
            except (TypeError, ValueError, OverflowError):
                score = 0.0
            candidates.append(
                {
                    "candidate_number": len(candidates) + 1,
                    "tmdb_id": tmdb_id,
                    "title": _safe(raw.get("title"), 160) or "未命名候选",
                    "year": _safe(raw.get("year"), 12),
                    "media_type": str(raw.get("media_type") or card.media_type)
                    if str(raw.get("media_type") or card.media_type) in _ALLOWED_MEDIA_TYPES
                    else card.media_type,
                    "score": score,
                }
            )
        stored = self._store_snapshot(
            _MappingSnapshot(
                scope_key=scope_key,
                provider=card.provider,
                external_id=card.external_id,
                media_type=card.media_type,
                title=_safe(card.title, 160),
                year=_safe(card.year, 12),
                candidates=candidates,
                created_at=time.monotonic(),
                generation=guard.generation,
                revision=guard.revision,
            )
        )
        if not stored:
            raise AgentToolError(
                "本次映射候选已被更新请求取代，请重新查询", code="precondition_failed"
            )
        confirmed = bool(result.get("confirmed"))
        return ToolResult(
            True,
            "completed" if confirmed or candidates else "empty",
            (
                "该条目已有确认过的 TMDB 映射"
                if confirmed
                else f"找到 {len(candidates)} 个 TMDB 映射候选"
            ),
            data={
                "source_title": _safe(card.title, 160) or "未命名条目",
                "provider": card.provider,
                "media_type": card.media_type,
                "mapping_confirmed": confirmed,
                "candidates": [
                    {k: v for k, v in item.items() if k != "tmdb_id"} for item in candidates
                ],
                "candidate_count": len(candidates),
                "limits": {"max_candidates": 5, "ttl_seconds": int(_TTL_SECONDS)},
            },
            evidence=[
                Evidence(
                    "discovery_mapping",
                    "只读查询映射与 TMDB 候选；未写入 media_external_ids。候选内部 TMDB ID 仅保存在当前会话短期上下文。",
                    _now(),
                )
            ],
            suggestions=(
                []
                if confirmed
                else ["如需保存，请回复：确认第 1 个映射。"]
                if candidates
                else []
            ),
        )

    def prepare_confirm_discovery_mapping(
        self,
        arguments: dict[str, Any], context: ToolContext
    ) -> tuple[ToolResult, str]:
        snapshot = self._get_snapshot(_scope_key(context.owner, context.session_id))
        if snapshot is None:
            raise AgentToolError(
                "最近映射候选不存在或已过期，请重新查询", code="precondition_failed"
            )
        candidate = next(
            (
                item
                for item in snapshot.candidates
                if item["candidate_number"] == arguments["candidate_number"]
            ),
            None,
        )
        if candidate is None:
            raise AgentToolError("映射候选序号不存在", code="precondition_failed")
        service = get_discovery_service()
        try:
            match = service.verify_tmdb_mapping_candidate(
                candidate["tmdb_id"], snapshot.media_type
            )
        except (TypeError, ValueError):
            raise AgentToolError("所选 TMDB 候选当前无法核验", code="precondition_failed")
        verified = dict(candidate)
        verified["title"] = _safe(match.get("title"), 160) or candidate["title"]
        verified["year"] = _safe(match.get("year"), 12) or candidate["year"]
        snapshot.candidates = [
            verified if item["candidate_number"] == arguments["candidate_number"] else item
            for item in snapshot.candidates
        ]
        if not self._store_snapshot(snapshot):
            raise AgentToolError("映射候选已变化，请重新查询", code="confirmation_stale")
        fingerprint = _mapping_context(snapshot, verified)
        return ToolResult(
            True,
            "confirmation_required",
            f"确认后将《{snapshot.title or '该条目'}》映射到《{verified['title']}》",
            data={
                "source_title": snapshot.title or "未命名条目",
                "candidate_number": verified["candidate_number"],
                "candidate_title": verified["title"],
                "candidate_year": verified["year"],
                "effects": [
                    "会保存一条确认过的来源到 TMDB 身份映射。",
                    "不会加入收藏、创建订阅、搜索资源或下载。",
                ],
            },
            evidence=[
                Evidence("tmdb_detail", "已重新读取并核验所选 TMDB 候选身份。", _now())
            ],
        ), fingerprint

    def confirm_discovery_mapping_confirmed(
        self,
        arguments: dict[str, Any], expected_context: str, context: ToolContext
    ) -> ToolResult:
        snapshot = self._get_snapshot(_scope_key(context.owner, context.session_id))
        if snapshot is None:
            raise AgentToolError("映射候选已过期，请重新查询", code="confirmation_stale")
        candidate = next(
            (
                item
                for item in snapshot.candidates
                if item["candidate_number"] == arguments["candidate_number"]
            ),
            None,
        )
        existing_state = _current_mapping_state(snapshot)
        if candidate is None or _mapping_context(
            snapshot, candidate, existing_state=existing_state, state_loaded=True
        ) != (expected_context or ""):
            raise AgentToolError(
                "来源、候选或现有映射已变化，请重新预检", code="confirmation_stale"
            )
        if not self._consume_snapshot(snapshot):
            raise AgentToolError(
                "映射候选已变化或已被消费，请重新查询", code="confirmation_stale"
            )
        saved = get_discovery_service().confirm_tmdb_mapping_if_unchanged(
            snapshot.provider,
            snapshot.external_id,
            snapshot.media_type,
            candidate["tmdb_id"],
            existing_state,
        )
        if saved is None:
            raise AgentToolError(
                "现有映射在保存时发生变化，请重新预检", code="confirmation_stale"
            )
        return ToolResult(
            True,
            "completed",
            f"已确认《{snapshot.title or '该条目'}》的 TMDB 映射",
            data={
                "affected": 1,
                "provider": snapshot.provider,
                "media_type": snapshot.media_type,
                "candidate_number": candidate["candidate_number"],
                "candidate_title": candidate["title"],
                "mapping_confirmed": True,
            },
            evidence=[
                Evidence(
                    "media_external_ids", "已保存经过 TMDB 详情核验的确认映射。", _now()
                )
            ],
        )
