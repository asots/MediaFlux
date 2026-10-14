"""指定剧集的本地集数查询：只读取 Jellyfin / Emby，不做 TMDB 缺集判断。"""

from __future__ import annotations

import unicodedata
from datetime import datetime
from itertools import groupby
from typing import Any

from app.agent.errors import AgentToolError
from app.agent.models import Evidence, ToolResult
from app.services import inspect_series_episode_sources

_MAX_EPISODES = 2000
_FOUND_STATUSES = {"ready", "unmapped"}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip(" ，。！？?、:：")
    if not normalized:
        raise AgentToolError("query 不能为空")
    if len(normalized) > 120:
        raise AgentToolError("query 最长 120 个字符")
    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        raise AgentToolError("query 包含不允许的控制字符")
    return normalized


def _normalize_library_name(value: object) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        raise AgentToolError("library_name 必须是字符串")
    normalized = unicodedata.normalize("NFKC", value).strip(" ，。！？?、:：")
    if not normalized or len(normalized) > 80:
        raise AgentToolError("library_name 必须是 1 到 80 个字符")
    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        raise AgentToolError("library_name 包含不允许的控制字符")
    return normalized


def count_series_episodes_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    extra = set(arguments) - {"query", "tmdb_id", "library_name"}
    if extra:
        raise AgentToolError(f"不支持的参数: {', '.join(sorted(extra))}")
    raw_query = arguments.get("query")
    if not isinstance(raw_query, str):
        raise AgentToolError("query 必须是字符串")
    query = _normalize_query(raw_query)

    tmdb_id = arguments.get("tmdb_id", "")
    if not isinstance(tmdb_id, str):
        raise AgentToolError("tmdb_id 必须是字符串")
    tmdb_id = tmdb_id.strip()
    if tmdb_id and (
        not tmdb_id.isascii() or not tmdb_id.isdigit() or not 1 <= len(tmdb_id) <= 10
    ):
        raise AgentToolError("tmdb_id 必须是 1 到 10 位数字")
    normalized = {"query": query, "tmdb_id": tmdb_id}
    library_name = _normalize_library_name(arguments.get("library_name", ""))
    if library_name:
        normalized["library_name"] = library_name
    return normalized


def _safe_candidates(sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        for candidate in source.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            name = str(candidate.get("name") or "").strip()
            year = str(candidate.get("year") or "").strip()
            if not name or (name, year) in seen:
                continue
            seen.add((name, year))
            candidates.append({"name": name, "year": year})
            if len(candidates) >= 8:
                return candidates
    return candidates


def _source_inventory(source: dict[str, Any]) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    episodes = {
        (int(item[0]), int(item[1]))
        for item in source.get("episodes") or []
        if isinstance(item, (list, tuple))
        and len(item) == 2
        and str(item[0]).isdigit()
        and str(item[1]).isdigit()
        and int(item[0]) > 0
        and int(item[1]) > 0
    }
    selected = (
        source.get("selected") if isinstance(source.get("selected"), dict) else {}
    )
    return {
        "server_type": str(source.get("server_type") or ""),
        "server_name": str(source.get("server_name") or "媒体服务器"),
        "library_name": str(source.get("library_name") or ""),
        "status": str(source.get("status") or "unavailable"),
        "title": str(selected.get("name") or ""),
        "year": str(selected.get("year") or ""),
        "local_episode_count": len(episodes),
        "season_count": len({season for season, _episode in episodes}),
        "truncated": bool(source.get("truncated")),
        "ignored_specials": max(0, int(source.get("ignored_specials", 0) or 0)),
        "ignored_unknown": max(0, int(source.get("ignored_unknown", 0) or 0)),
    }, episodes


def _season_breakdown(episodes: set[tuple[int, int]]) -> list[dict[str, Any]]:
    seasons: dict[int, list[int]] = {}
    for season, episode in sorted(episodes):
        seasons.setdefault(season, []).append(episode)
    return [
        {
            "season": season,
            "count": len(numbers),
            "first_episode": min(numbers),
            "last_episode": max(numbers),
            # 首末集只表示边界，断号必须保留；每个范围都是实际观察到的闭区间。
            "episode_ranges": [
                [run[0][1], run[-1][1]]
                for _, group in groupby(enumerate(numbers), lambda item: item[1] - item[0])
                if (run := list(group))
            ],
        }
        for season, numbers in seasons.items()
    ]


def _result(
    *,
    ok: bool,
    status: str,
    summary: str,
    data: dict[str, Any],
    suggestions: list[str] | None = None,
    error: str = "",
) -> ToolResult:
    return ToolResult(
        ok=ok,
        status=status,
        summary=summary,
        data=data,
        evidence=[
            Evidence(
                "media_servers",
                "直接读取已配置 Jellyfin / Emby 的本地 Series 与普通集号；不访问 TMDB，也不据此判断是否缺集。",
                _now(),
            )
        ],
        suggestions=suggestions or [],
        error=error,
    )


def count_series_episodes(arguments: dict[str, Any]) -> ToolResult:
    query = str(arguments.get("query") or "")
    tmdb_id = str(arguments.get("tmdb_id") or "")
    library_name = str(arguments.get("library_name") or "")
    inspect_kwargs: dict[str, Any] = {
        "tmdb_id": tmdb_id,
        "max_episodes": _MAX_EPISODES,
        "include_specials": False,
    }
    if library_name:
        inspect_kwargs["library_name"] = library_name
    sources = inspect_series_episode_sources(query, **inspect_kwargs)
    if not sources:
        return _result(
            ok=False,
            status="not_configured",
            summary="尚未配置可查询的 Jellyfin 或 Emby 媒体服务器",
            data={
                "query": query,
                "library_name": library_name,
                "local_episode_count": None,
                "inventory_complete": False,
                "matched_source_count": 0,
                "source_count": 0,
                "sources": [],
            },
            suggestions=["请先在设置中启用并配置 Jellyfin 或 Emby，然后重试。"],
            error="media_server_not_configured",
        )

    inventories = [_source_inventory(source) for source in sources]
    safe_sources = [source for source, _episodes in inventories]
    matched = [(source, episodes) for source, episodes in inventories if source["status"] in _FOUND_STATUSES]
    if not matched:
        statuses = {str(source.get("status") or "unavailable") for source in sources}
        candidates = _safe_candidates(sources)
        if statuses == {"library_not_found"}:
            summary = f"没有找到名为「{library_name}」的媒体库"
            status = "library_not_found"
            suggestions = ["请核对媒体库名称，或先列出媒体服务器中的媒体库。"]
        elif "library_ambiguous" in statuses:
            summary = f"有多个媒体库与「{library_name}」相似，暂时无法确定范围"
            status = "library_ambiguous"
            suggestions = ["请使用媒体服务器中显示的完整媒体库名称重试。"]
        elif statuses == {"not_found"}:
            scope_label = (
                f"「{library_name}」媒体库" if library_name else "已配置的媒体库"
            )
            summary = f"没有在{scope_label}中找到《{query}》"
            status = "not_found"
            suggestions = ["请尝试官方名称、原始名称，或提供 TMDB ID。"]
        elif "ambiguous" in statuses or "conflict" in statuses:
            summary = f"找到多个可能的《{query}》，暂时无法确定要统计哪一部"
            status = "ambiguous"
            suggestions = ["请补充首播年份、完整剧名或 TMDB ID。"]
        else:
            summary = "媒体服务器当前不可用，暂时无法读取本地集数"
            status = "unavailable"
            suggestions = ["请检查 Jellyfin / Emby 连接状态后重试。"]
        return _result(
            ok=False,
            status=status,
            summary=summary,
            data={
                "query": query,
                "library_name": library_name,
                "local_episode_count": None,
                "inventory_complete": False,
                "matched_source_count": 0,
                "source_count": len(sources),
                "candidates": candidates,
                "sources": safe_sources,
            },
            suggestions=suggestions,
            error=status,
        )

    episode_union: set[tuple[int, int]] = set()
    ignored_specials = 0
    ignored_unknown = 0
    truncated = False
    title = query
    year = ""
    for source, episodes in matched:
        title = source["title"] or title
        year = source["year"] or year
        ignored_specials += source["ignored_specials"]
        ignored_unknown += source["ignored_unknown"]
        truncated = truncated or source["truncated"]
        episode_union.update(episodes)

    seasons = _season_breakdown(episode_union)
    count = len(episode_union)
    incomplete = any(
        source["truncated"] or source["ignored_unknown"]
        or source["status"] not in _FOUND_STATUSES | {"not_found"}
        for source in safe_sources
    )
    status = "partial" if incomplete else "success"
    ok = True
    source_label = matched[0][0]["server_name"]
    if len(matched) > 1:
        source_label = f"{len(matched)} 个媒体服务器"
    summary = f"{source_label} 中《{title}》本地收录 {count} 集"
    if incomplete:
        summary += "，但部分来源不可用、有歧义、未完整编号或读取被截断"

    suggestions: list[str] = []
    if incomplete:
        suggestions.append(
            "当前数量是已确认的本地普通集下限；范围仅代表已确认收录，未列出的季集不能据此认定不存在，请核对来源状态。"
        )
    elif ignored_specials:
        suggestions.append(
            f"默认未计入特别篇（第 0 季），共忽略 {ignored_specials} 项。"
        )
    suggestions.append("指定集是否在库以 seasons.episode_ranges 闭区间为准，不能仅凭首末集推断；inventory_complete=false 时只能确认已观察到的集。检查已播缺集再用 library.audit_episodes。")

    return _result(
        ok=ok,
        status=status,
        summary=summary,
        data={
            "query": query,
            "library_name": library_name,
            "title": title,
            "year": year,
            "local_episode_count": count,
            "inventory_complete": not incomplete,
            "season_count": len(seasons),
            "seasons": seasons,
            "matched_source_count": len(matched),
            "source_count": len(sources),
            "ignored_specials": ignored_specials,
            "ignored_unknown": ignored_unknown,
            "truncated": truncated,
            "sources": safe_sources,
            "count_definition": "Jellyfin / Emby 中已识别季号和集号的本地普通集；跨媒体服务器重复季集号只计一次。",
        },
        suggestions=suggestions,
    )
