"""媒体服务器聚合搜索及其短时只读缓存。"""

from __future__ import annotations

import threading
import time
import unicodedata
from typing import Any

from app.agent.episode_audit import reset_episode_audit_cache_for_tests
from app.agent.errors import AgentToolError
from app.agent.media_links import media_open_url_resolver
from app.agent.models import Evidence, ToolResult
from app.agent.provider_actions import reset_provider_gateway_for_tests
from app.agent.workspace_actions import (
    _contains_sensitive_text,
    _safe_status,
    _safe_title,
    _safe_year,
)
from app.clients.base import normalize_playback_progress
from app.services import search_media_servers

from .shared import _now, _reject_extra, _search_arguments

_SEARCH_CACHE_TTL_SECONDS = 15
_SEARCH_CACHE_MAX_ENTRIES = 128
_search_cache: dict[tuple[str, int, int, str, str, bool], tuple[float, list[dict[str, Any]]]] = {}
_search_cache_lock = threading.Lock()


def library_search_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    _reject_extra(arguments, {"query", "limit", "offset", "search_by", "media_type", "count_only"})
    normalized = _search_arguments({key: arguments[key] for key in ("query", "limit") if key in arguments})
    if "offset" in arguments and (type(arguments["offset"]) is not int or arguments["offset"] < 0):
        raise AgentToolError("offset 必须是非负整数")
    for key, choices in (("search_by", ("title", "actor")), ("media_type", ("all", "movie", "series", "episode"))):
        if key in arguments and arguments[key] not in choices:
            raise AgentToolError(f"{key} 必须是 {' / '.join(choices)}")
    if "count_only" in arguments and type(arguments["count_only"]) is not bool:
        raise AgentToolError("count_only 必须是布尔值")
    return {**normalized, **{key: arguments[key] for key in ("offset", "search_by", "media_type", "count_only") if key in arguments}}


def _search_sources(
    query: str, limit: int, *, offset: int = 0, search_by: str = "title",
    media_type: str = "all", count_only: bool = False,
) -> list[dict[str, Any]]:
    cache_key = (query.casefold(), limit, offset, search_by, media_type, count_only)
    now = time.monotonic()
    with _search_cache_lock:
        cached = _search_cache.get(cache_key)
        if cached and now - cached[0] < _SEARCH_CACHE_TTL_SECONDS:
            return cached[1]
    sources = search_media_servers(
        query, limit=limit, offset=offset, search_by=search_by,
        media_type=media_type, count_only=count_only,
    )
    with _search_cache_lock:
        expired = [
            key
            for key, value in _search_cache.items()
            if now - value[0] >= _SEARCH_CACHE_TTL_SECONDS
        ]
        for key in expired:
            _search_cache.pop(key, None)
        while len(_search_cache) >= _SEARCH_CACHE_MAX_ENTRIES:
            oldest = min(_search_cache, key=lambda key: _search_cache[key][0])
            _search_cache.pop(oldest, None)
        _search_cache[cache_key] = (now, sources)
    return sources


def reset_agent_tool_caches_for_tests() -> None:
    with _search_cache_lock:
        _search_cache.clear()
    reset_episode_audit_cache_for_tests()
    reset_provider_gateway_for_tests()


def _safe_optional_index(value: Any, *, allow_zero: bool) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    minimum = 0 if allow_zero else 1
    return value if minimum <= value <= 10_000 else None


def _safe_runtime_minutes(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        runtime = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return runtime if 0 <= runtime <= 525_600 else 0


def _safe_overview(value: Any) -> str:
    overview = unicodedata.normalize("NFKC", str(value or "")).strip()
    overview = " ".join(overview.split())
    if not overview or _contains_sensitive_text(overview):
        return ""
    return overview[:500]


def search_library(arguments: dict[str, Any]) -> ToolResult:
    query = arguments["query"]
    match_label = "演员元数据" if arguments.get("search_by") == "actor" else "标题"
    scope_label = f"媒体库按{match_label}检索“{query}”"
    options = {key: arguments[key] for key in ("offset", "search_by", "media_type", "count_only") if key in arguments}
    sources = _search_sources(query, arguments["limit"], **options)
    search_metadata = {
        "query": query,
        "offset": arguments.get("offset", 0),
        "limit": arguments["limit"],
        "count_only": arguments.get("count_only", False),
        "search_by": arguments.get("search_by", "title"),
        "media_type": arguments.get("media_type", "all"),
        "match_basis": "actor_metadata" if arguments.get("search_by") == "actor" else "title_search",
        "count_scope": "server_records_sum_not_deduplicated",
        "pagination_scope": "per_server",
    }
    serialized_sources: list[dict[str, Any]] = []
    known_total = 0
    known_sources = 0
    returned = 0
    unavailable = 0
    for source in sources:
        items: list[dict[str, Any]] = []
        source_unavailable = bool(source.get("error"))
        if source_unavailable:
            unavailable += 1
        else:
            server_type = _safe_status(source.get("server_type"), {"jellyfin", "emby"})
            resolve_link = media_open_url_resolver(
                server_type=server_type,
                server_url=source.get("web_url"),
            )
            if server_type == "unknown":
                server_type = "media_server"
            for item in source.get("items", []):
                title = _safe_title(item.name or item.display_name, "媒体条目")
                display_name = _safe_title(item.display_name, title)
                series_name = (
                    _safe_title(item.series_name, "") if item.series_name else ""
                )
                serialized_item = {
                    "title": title,
                    "display_name": display_name,
                    "media_type": _safe_status(
                        item.type,
                        {"movie", "series", "episode", "season", "video"},
                    ),
                    "year": _safe_year(item.year),
                    "series_name": series_name,
                    "season": _safe_optional_index(
                        item.season_number, allow_zero=True
                    ),
                    "episode": _safe_optional_index(
                        item.episode_number, allow_zero=False
                    ),
                    "runtime_minutes": _safe_runtime_minutes(item.runtime),
                    "playback_progress_percent": normalize_playback_progress(
                        item.progress
                    ),
                    "overview": _safe_overview(item.overview),
                }
                open_url = resolve_link(item.web_url or item.series_web_url)
                if open_url:
                    serialized_item["open_url"] = open_url
                items.append(serialized_item)
        returned += len(items)
        source_total = source.get("total")
        if source_unavailable or type(source_total) is not int or source_total < 0:
            source_total = None
        else:
            known_total += source_total
            known_sources += 1
        source_has_more = source.get("has_more") if not source_unavailable else None
        server_type = _safe_status(source.get("server_type"), {"jellyfin", "emby"})
        if server_type == "unknown":
            server_type = "media_server"
        source_match = "found" if items or (source_total or 0) > 0 else "not_found" if source_total == 0 else "unknown"
        serialized_sources.append(
            {
                "server_type": server_type,
                "server_name": _safe_title(source.get("server_name"), "媒体服务器"),
                "status": "unavailable" if source_unavailable else "ready",
                "match_status": source_match,
                "total": source_total,
                "returned": len(items),
                "has_more": source_has_more,
                "offset": source.get("offset", arguments.get("offset", 0)),
                "limit": source.get("limit", arguments["limit"]),
                "next_offset": source.get("next_offset") if not source_unavailable else None,
                "total_source": source.get("total_source", "unknown") if not source_unavailable else "unknown",
                "person_match": source.get("person_match", ""),
                "included_item_types": source.get("included_item_types", []),
                "items": items,
            }
        )
    if not sources:
        return ToolResult(
            ok=False,
            status="not_configured",
            summary="没有可搜索的媒体服务器",
            data={
                **search_metadata,
                "total": None,
                "known_total": 0,
                "returned": 0,
                "has_more": None,
                "next_offset": None,
                "match_status": "not_configured",
                "sources": [],
            },
            evidence=[
                Evidence(
                    "media_servers",
                    "检查已启用且凭据完整的 Jellyfin / Emby 配置。",
                    _now(),
                )
            ],
            suggestions=["请先在设置中完整配置并启用 Jellyfin 或 Emby。"],
        )
    available = len(sources) - unavailable
    total = known_total if known_sources == len(sources) else None
    has_more = (
        True if any(source["has_more"] is True for source in serialized_sources)
        else False if all(source["has_more"] is False for source in serialized_sources)
        else None
    )
    next_offsets = [source["next_offset"] for source in serialized_sources if type(source["next_offset"]) is int]
    if unavailable == len(sources):
        status = "unavailable"
        match_status = "indeterminate"
        summary = "媒体服务器暂时不可用，无法判断是否存在匹配内容"
        suggestions = ["请检查媒体服务器连通性后重试。"]
    elif returned or known_total:
        status = "success" if total is not None else "partial"
        match_status = "found"
        summary = (
            f"{scope_label}匹配 {total} 条记录，本次返回 {returned} 条"
            if total is not None else f"本次返回 {returned} 条媒体记录，完整匹配总数未知"
        )
        suggestions = (
            [f"有 {unavailable} 个媒体服务器暂时不可用。"] if unavailable else []
        )
    else:
        status = "empty" if total == 0 else "partial"
        match_status = "not_found" if total == 0 else "indeterminate"
        summary = (
            f"已查询 {available} 个媒体服务器，另有 {unavailable} 个暂时不可用；可用来源未找到匹配内容"
            if unavailable
            else "媒体库中没有找到匹配内容" if total == 0 else "本次未返回媒体条目，匹配总数无法确认"
        )
        suggestions = ["请检查不可用的媒体服务器后重试。"] if unavailable else []
        if arguments.get("search_by", "title") == "title":
            suggestions.extend([
                "可尝试中文名、原名或去掉季集编号后重新搜索。",
                f"搜索《{query}》的资源。",
                f"在网上找《{query}》。",
            ])
    if total is None and available:
        suggestions.append("总数未知：不能把 returned 当作 total，也不要按片名分片累加猜测。")
    if arguments.get("search_by") == "actor":
        suggestions.append("仅统计库内 Actor 人物关联；缺失或别名元数据不代表演员没有作品，电影/剧集记录也不是全球全集。")
        if any(source["person_match"] in {"ambiguous", "incomplete"} for source in serialized_sources):
            suggestions.append("演员身份歧义或人物候选不完整，需核对媒体服务器人物元数据后再统计。")
        if any(source["person_match"] == "not_found" for source in serialized_sources):
            suggestions.append("未找到精确姓名的演员元数据；请核对库内姓名或别名，不能据此回答零作品。")
    data = {
        **search_metadata, "total": total, "known_total": known_total,
        "returned": returned, "has_more": has_more, "next_offset": min(next_offsets, default=None),
        "match_status": match_status, "sources": serialized_sources,
    }
    if arguments.get("count_only"):
        # 统计无明细页；不把 returned=0/has_more=true 等分页字段交给模型解释。
        page_fields = {"returned", "has_more", "offset", "limit", "next_offset", "pagination_scope", "items"}
        data = {key: value for key, value in data.items() if key not in page_fields}
        data["sources"] = [{key: value for key, value in source.items() if key not in page_fields} for source in serialized_sources]
        if total is not None and total > 0:
            summary = f"{scope_label}匹配 {total} 条记录"
    return ToolResult(
        ok=available > 0,
        status=status,
        summary=summary,
        data=data,
        evidence=[
            Evidence(
                "media_servers", f"查询 {len(sources)} 个已配置媒体服务器。", _now()
            )
        ],
        suggestions=suggestions,
    )
