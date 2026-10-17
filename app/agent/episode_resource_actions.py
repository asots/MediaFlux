"""已播缺集的定向资源搜索：先审计，再复用多站安全搜索。"""

from __future__ import annotations

import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from threading import Lock
from typing import Any

from app.agent.episode_audit import audit_series_episodes
from app.agent.errors import AgentToolError
from app.agent.indexer_actions import (
    normalize_search_sites,
    search_resources,
    validate_enabled_search_sites,
)
from app.agent.indexer_actions import search_arguments as indexer_search_arguments
from app.agent.media_preference_policy import validate_resource_preference_overrides
from app.agent.models import Evidence, ToolContext, ToolResult
from app.agent.recent_resource_candidates import (
    MAX_RESOURCE_CANDIDATES,
    attach_resource_candidate_reference,
    merge_resource_candidate_references,
)
from app.agent.resource_recommendation import (
    episode_search_model_data,
    matched_episode_items,
    rank_episode_search,
)
from app.indexers.models import IndexerItem, IndexerMediaSearchRequest
from app.indexers.ranking import rank_item
from app.indexers.runtime import get_indexer_service

DEFAULT_MISSING_SEASON_EPISODES = 20
MAX_MISSING_SEASON_EPISODES = 100


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _reject_extra(arguments: dict[str, Any], allowed: set[str]) -> None:
    extra = set(arguments) - allowed
    if extra:
        raise AgentToolError(f"不支持的工具参数：{', '.join(sorted(extra))}")


def _visible_text(value: Any, *, name: str, maximum: int = 120) -> str:
    if not isinstance(value, str):
        raise AgentToolError(f"{name} 必须是字符串")
    text = unicodedata.normalize("NFKC", value).strip()
    if (
        not text
        or len(text) > maximum
        or any(unicodedata.category(char).startswith("C") for char in text)
    ):
        raise AgentToolError(f"{name} 必须是 1 到 {maximum} 个可见字符")
    return text


def _positive_int(value: Any, *, name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise AgentToolError(f"{name} 必须是 1 到 {maximum} 的整数")
    return value


def _optional_visible_text(value: Any, *, name: str, maximum: int) -> str:
    if value in (None, ""):
        return ""
    return _visible_text(value, name=name, maximum=maximum)


def missing_episode_resource_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    _reject_extra(
        arguments,
        {
            "query",
            "tmdb_id",
            "season",
            "episode",
            "as_of",
            "sites",
            "limit",
            "library_name",
            "preference_overrides",
        },
    )
    query = _visible_text(arguments.get("query"), name="query")
    library_name = _optional_visible_text(
        arguments.get("library_name", ""),
        name="library_name",
        maximum=80,
    )

    tmdb_id = arguments.get("tmdb_id", "")
    if not isinstance(tmdb_id, str):
        raise AgentToolError("tmdb_id 必须是字符串")
    tmdb_id = tmdb_id.strip()
    if tmdb_id and (
        not tmdb_id.isascii() or not tmdb_id.isdigit() or len(tmdb_id) > 10
    ):
        raise AgentToolError("tmdb_id 必须是 1 到 10 位数字")

    season = _positive_int(arguments.get("season"), name="season", maximum=100)
    episode = _positive_int(arguments.get("episode"), name="episode", maximum=1000)

    today = datetime.now().astimezone().date()
    as_of = arguments.get("as_of", today.isoformat())
    if not isinstance(as_of, str):
        raise AgentToolError("as_of 必须是 YYYY-MM-DD 日期")
    try:
        parsed_as_of = date.fromisoformat(as_of.strip())
    except ValueError as exc:
        raise AgentToolError("as_of 必须是 YYYY-MM-DD 日期") from exc
    if parsed_as_of > today:
        raise AgentToolError("as_of 不能晚于今天")

    sites = normalize_search_sites(arguments.get("sites", []))
    validate_enabled_search_sites(sites)

    limit = arguments.get("limit", 20)
    _positive_int(limit, name="limit", maximum=MAX_RESOURCE_CANDIDATES)
    normalized = {
        "query": query,
        "tmdb_id": tmdb_id,
        "season": season,
        "episode": episode,
        "as_of": parsed_as_of.isoformat(),
        "sites": sites,
        "limit": limit,
    }
    if library_name:
        normalized["library_name"] = library_name
    if "preference_overrides" in arguments:
        normalized["preference_overrides"] = validate_resource_preference_overrides(
            arguments["preference_overrides"],
        )
    return normalized


def missing_season_resource_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if "items" in arguments:
        _reject_extra(arguments, {"items", "as_of", "sites", "max_episodes", "limit_per_episode", "preference_overrides"})
        raw = arguments["items"]
        if not isinstance(raw, list) or not 1 <= len(raw) <= 12:
            raise AgentToolError("items 必须包含 1 到 12 部/季")
        shared = {key: value for key, value in arguments.items() if key != "items"}
        items = {}
        for item in raw:
            if not isinstance(item, dict):
                raise AgentToolError("items 每项必须是作品对象")
            _reject_extra(item, {"query", "tmdb_id", "season", "library_name", "episodes"})
            normalized = missing_season_resource_arguments({**shared, **item})
            key = (normalized["query"].casefold(), normalized.get("tmdb_id"), normalized["season"], normalized.get("library_name"), tuple(normalized.get("episodes", [])))
            items.setdefault(key, normalized)
        return {"items": list(items.values())}
    _reject_extra(
        arguments,
        {
            "query",
            "tmdb_id",
            "season",
            "episodes",
            "as_of",
            "sites",
            "max_episodes",
            "limit_per_episode",
            "library_name",
            "preference_overrides",
        },
    )
    query = _visible_text(arguments.get("query"), name="query")
    library_name = _optional_visible_text(
        arguments.get("library_name", ""),
        name="library_name",
        maximum=80,
    )

    tmdb_id = arguments.get("tmdb_id", "")
    if not isinstance(tmdb_id, str):
        raise AgentToolError("tmdb_id 必须是字符串")
    tmdb_id = tmdb_id.strip()
    if "tmdb_id" in arguments and not tmdb_id:
        raise AgentToolError("tmdb_id 必须是 1 到 10 位数字")
    if tmdb_id and (
        not tmdb_id.isascii() or not tmdb_id.isdigit() or len(tmdb_id) > 10
    ):
        raise AgentToolError("tmdb_id 必须是 1 到 10 位数字")

    season = _positive_int(arguments.get("season"), name="season", maximum=100)
    today = datetime.now().astimezone().date()
    as_of = arguments.get("as_of", today.isoformat())
    if not isinstance(as_of, str):
        raise AgentToolError("as_of 必须是 YYYY-MM-DD 日期")
    try:
        parsed_as_of = date.fromisoformat(as_of.strip())
    except ValueError as exc:
        raise AgentToolError("as_of 必须是 YYYY-MM-DD 日期") from exc
    if parsed_as_of > today:
        raise AgentToolError("as_of 不能晚于今天")

    sites = normalize_search_sites(arguments.get("sites", []))
    validate_enabled_search_sites(sites)
    episodes = arguments.get("episodes")
    if episodes is not None:
        if not isinstance(episodes, list) or not 1 <= len(episodes) <= MAX_MISSING_SEASON_EPISODES:
            raise AgentToolError(f"episodes 必须包含 1 到 {MAX_MISSING_SEASON_EPISODES} 个指定集号")
        episodes = sorted({_positive_int(value, name="episodes", maximum=1000) for value in episodes})
    elif "episodes" in arguments:
        raise AgentToolError("episodes 必须是指定集号数组")
    max_episodes = arguments.get("max_episodes", max(DEFAULT_MISSING_SEASON_EPISODES, len(episodes or [])))
    limit_per_episode = arguments.get("limit_per_episode", 8)
    _positive_int(max_episodes, name="max_episodes", maximum=MAX_MISSING_SEASON_EPISODES)
    _positive_int(limit_per_episode, name="limit_per_episode", maximum=10)
    normalized = {
        "query": query,
        "tmdb_id": tmdb_id,
        "season": season,
        "as_of": parsed_as_of.isoformat(),
        "sites": sites,
        "max_episodes": max_episodes,
        "limit_per_episode": limit_per_episode,
    }
    if episodes is not None:
        normalized["episodes"] = episodes
    if library_name:
        normalized["library_name"] = library_name
    if "preference_overrides" in arguments:
        normalized["preference_overrides"] = validate_resource_preference_overrides(
            arguments["preference_overrides"],
        )
    return normalized


def _bounded_count(value: Any, maximum: int = 100_000) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return max(0, min(int(value), maximum))
    except (TypeError, ValueError, OverflowError):
        return 0


def _verification(
    arguments: dict[str, Any], audit: ToolResult, *, verified: bool
) -> dict[str, Any]:
    data = audit.data if isinstance(audit.data, dict) else {}
    title = " ".join(str(data.get("title") or arguments["query"]).split())[:120]
    tmdb_id = str(data.get("tmdb_id") or arguments.get("tmdb_id") or "")
    if not tmdb_id.isascii() or not tmdb_id.isdigit():
        tmdb_id = ""
    return {
        "title": title,
        "tmdb_id": tmdb_id[:10],
        "season": arguments["season"],
        "episode": arguments["episode"],
        "as_of": arguments["as_of"],
        "audit_status": audit.status,
        "library_name": str(
            data.get("library_name") or arguments.get("library_name") or ""
        )[:80],
        "missing_count": _bounded_count(data.get("missing_count")),
        "verified_missing": verified,
        "resource_search_attempted": False,
    }


def _season_verification(
    arguments: dict[str, Any], audit: ToolResult
) -> dict[str, Any]:
    data = audit.data if isinstance(audit.data, dict) else {}
    title = " ".join(str(data.get("title") or arguments["query"]).split())[:120]
    tmdb_id = str(data.get("tmdb_id") or arguments.get("tmdb_id") or "")
    if not tmdb_id.isascii() or not tmdb_id.isdigit():
        tmdb_id = ""
    return {
        "title": title,
        "tmdb_id": tmdb_id[:10],
        "season": arguments["season"],
        "as_of": arguments["as_of"],
        "audit_status": audit.status,
        "library_name": str(
            data.get("library_name") or arguments.get("library_name") or ""
        )[:80],
        "missing_count": _bounded_count(data.get("missing_count")),
        "verified_missing": False,
        "resource_search_attempted": False,
    }


def _not_missing_result(verification: dict[str, Any], audit: ToolResult) -> ToolResult:
    target = f"第 {verification['season']} 季" + (
        f"第 {verification['episode']} 集" if "episode" in verification else "指定集"
    )
    return ToolResult(
        False,
        "not_missing",
        target + "未确认缺失，未执行资源站检索；不能据此判断没有资源",
        data={"verification": verification, "episodes": []},
        evidence=list(audit.evidence),
        suggestions=[
            "若用户要找指定资源而非补缺，继续用 indexer.search_resources 按片名、season、episode 检索；无需让用户再次授权只读查询。若只问是否有缺集更新，则到此结束，不展示资源卡。"
        ],
    )


def _episode_search_arguments(arguments: dict[str, Any], title: str) -> dict[str, Any]:
    base = " ".join(str(title or arguments["query"]).split()) or arguments["query"]
    aliases = [] if arguments["query"].casefold() == base.casefold() else [arguments["query"]]
    return indexer_search_arguments(
        {
            "title": base,
            "aliases": aliases,
            "media_type": "tv",
            "season": arguments["season"],
            "episode": arguments["episode"],
            "sites": arguments["sites"],
            "limit": arguments.get("limit", arguments.get("limit_per_episode", 20)),
        }
    )


def search_missing_episode_resources(
    arguments: dict[str, Any], *, preferences: dict[str, Any] | None = None,
) -> ToolResult:
    audit_arguments: dict[str, Any] = {
        "query": arguments["query"],
        "tmdb_id": arguments["tmdb_id"],
        "season": arguments["season"],
        "target_episode": arguments["episode"],
        "as_of": arguments["as_of"],
    }
    if arguments.get("library_name"):
        audit_arguments["library_name"] = arguments["library_name"]
    audit = audit_series_episodes(audit_arguments, refresh=True)
    verification = _verification(arguments, audit, verified=False)

    if not audit.ok or audit.status != "updates_available":
        if audit.status == "up_to_date":
            return _not_missing_result(verification, audit)
        return ToolResult(
            False,
            audit.status,
            "缺集状态尚未可靠确认，因此未搜索资源",
            data={"verification": verification},
            evidence=list(audit.evidence),
            suggestions=list(audit.suggestions),
            error=audit.error,
        )

    audit_data = audit.data if isinstance(audit.data, dict) else {}
    target_missing = audit_data.get("target_missing")
    if not isinstance(target_missing, bool):
        # 展示样本可能截断，不能成为第二套业务判定。仅消费正式审计的精确结论。
        return ToolResult(
            False,
            "inconclusive",
            "缺少指定集的精确审计结论，因此未搜索资源",
            data={"verification": verification},
            evidence=list(audit.evidence),
            suggestions=["请重新审计指定季集后再搜索。"],
        )
    if not target_missing:
        return _not_missing_result(verification, audit)

    verification["verified_missing"] = True
    search_args = _episode_search_arguments(arguments, verification["title"])
    searched = search_resources(search_args)
    verification["resource_search_attempted"] = True
    search_data = searched.data if isinstance(searched.data, dict) else {}
    ranked_search = rank_episode_search(
        search_data,
        season=arguments["season"],
        episode=arguments["episode"],
        preferences=preferences,
        preference_overrides=arguments.get("preference_overrides"),
        mapping_context=audit.effect_metadata.get("episode_mapping_context"),
    )
    data = {
        "verification": verification,
        "search": ranked_search,
    }
    if searched.ok:
        recommendation = ranked_search["recommendation"]
        summary = f"已确认 S{arguments['season']:02d}E{arguments['episode']:02d} 缺失；{recommendation['reason']}"
        if search_data.get("partial"):
            summary += "部分来源未完成。"
    else:
        summary = f"已确认指定集缺失，但{searched.summary}"
    return attach_resource_candidate_reference(
        ToolResult(
            searched.ok,
            searched.status,
            summary,
            data=data,
            model_data={**data, "search": episode_search_model_data(ranked_search)},
            evidence=list(audit.evidence)
            + list(searched.evidence)
            + [
                Evidence(
                    "agent_verification",
                    "资源站搜索仅在媒体库与 TMDB 审计确认目标为已播缺集后执行；未自动提交下载。",
                    _now(),
                )
            ],
            suggestions=(list(searched.suggestions) if not searched.ok else [
                "只报告匹配目标集的结果；没有匹配就直接说明本轮未找到，不列旧集或未知映射让用户猜。标题匹配不代表已核验文件清单或下载完成。"
            ]),
            error=searched.error,
        ),
        result_store=get_indexer_service().result_store,
    )


_MISSING_EPISODE_SEARCH_TIMEOUT_SECONDS = 30.0
_MISSING_SEASON_SEARCH_TIMEOUT_SECONDS = 45.0
_MISSING_SEASON_SEARCH_MAX_REQUESTS = 6


class _MissingResourceBudget:
    """多部并发也共用绝对截止和请求配额，不按缺集数延长。"""

    def __init__(self, max_requests: int = _MISSING_SEASON_SEARCH_MAX_REQUESTS):
        self.deadline = time.monotonic() + _MISSING_SEASON_SEARCH_TIMEOUT_SECONDS
        self.remaining_requests = max_requests
        self.lock = Lock()

    def reserve(self, context: ToolContext | None, requests: int) -> tuple[float, str]:
        if context is not None and context.cancelled():
            return 0.0, "cancelled"
        with self.lock:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                return 0.0, "time_budget"
            if not self.remaining_requests or requests >= _MISSING_SEASON_SEARCH_MAX_REQUESTS:
                return 0.0, "request_budget"
            self.remaining_requests -= 1
        return min(remaining, _MISSING_EPISODE_SEARCH_TIMEOUT_SECONDS), ""


def _attach_season_resource_reference(result: ToolResult) -> ToolResult:
    """逐集复用既有核验快照并统一编号，避免展示样本上限裁掉后续集。"""
    selections = []
    seen = set()
    store = get_indexer_service().result_store
    for episode in result.data["episodes"]:
        snapshot = attach_resource_candidate_reference(ToolResult(
            episode["ok"], episode["status"], episode["summary"], data={
                "verification": {**result.data["verification"], "episode": episode["episode"]},
                "search": episode["search"],
            }), result_store=store)
        for reference in snapshot.references:
            selections.extend((reference, candidate) for candidate in reference.value["candidates"])
    selections.sort(key=lambda selection: selection[1]["match"] not in {"exact_episode", "episode_pack"})
    unique = []
    for reference, candidate in selections:
        if candidate["result_id"] not in seen:
            seen.add(candidate["result_id"])
            unique.append((reference, candidate))
        if len(unique) >= MAX_RESOURCE_CANDIDATES:
            break
    if unique:
        result.references.append(merge_resource_candidate_references(unique, status=result.status))
    return result


def search_missing_season_resources(
    arguments: dict[str, Any], *, preferences: dict[str, Any] | None = None, context: ToolContext | None = None,
    _budget: _MissingResourceBudget | None = None,
) -> ToolResult:
    if "items" in arguments:
        return _search_missing_resource_batch(arguments["items"], preferences=preferences, context=context)
    if context is not None and context.cancelled():
        return ToolResult(False, "cancelled", "本项已停止，未执行资源核对")
    budget = _budget or _MissingResourceBudget()
    audit_arguments: dict[str, Any] = {
        "query": arguments["query"],
        "tmdb_id": arguments["tmdb_id"],
        "season": arguments["season"],
        "as_of": arguments["as_of"],
    }
    if arguments.get("library_name"):
        audit_arguments["library_name"] = arguments["library_name"]
    audit = audit_series_episodes(audit_arguments, refresh=True)
    verification = _season_verification(arguments, audit)
    audit_data = audit.data if isinstance(audit.data, dict) else {}
    requested_episodes = arguments.get("episodes")
    if requested_episodes:
        verification["requested_episodes"] = requested_episodes

    if not audit.ok or audit.status != "updates_available":
        if audit.status == "up_to_date":
            return _not_missing_result(verification, audit)
        return ToolResult(
            False,
            audit.status,
            "该季缺集状态尚未可靠确认，因此未搜索资源",
            data={"verification": verification, "episodes": []},
            evidence=list(audit.evidence),
            suggestions=list(audit.suggestions),
            error=audit.error,
        )

    if bool(audit_data.get("missing_sample_truncated")):
        return ToolResult(
            False,
            "inconclusive",
            "该季缺集清单已截断，因此未执行批量资源搜索",
            data={"verification": verification, "episodes": []},
            evidence=list(audit.evidence),
            suggestions=["请缩小审计范围或逐集搜索明确的季集编号。"],
        )

    missing: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in audit_data.get("missing_sample", []):
        if not isinstance(item, dict):
            continue
        season = item.get("season")
        episode = item.get("episode")
        if (
            isinstance(season, bool)
            or not isinstance(season, int)
            or season != arguments["season"]
            or isinstance(episode, bool)
            or not isinstance(episode, int)
            or not 1 <= episode <= 1000
            or (requested_episodes is not None and episode not in requested_episodes)
            or (season, episode) in seen
        ):
            continue
        seen.add((season, episode))
        missing.append({"season": season, "episode": episode})
    missing.sort(key=lambda item: item["episode"])
    if not missing:
        return _not_missing_result(verification, audit)

    verification["verified_missing"] = True
    verification["missing_count"] = len(missing)
    selected = missing[: arguments["max_episodes"]]
    by_episode: dict[int, dict[str, Any]] = {}
    evidence = list(audit.evidence)
    suggestions: list[str] = []
    requests = 0
    stop_reason = ""
    pool: dict[str, Any] = {"items": []}
    shared = len(selected) > 1
    mapping_context = audit.effect_metadata.get("episode_mapping_context")
    detail = (mapping_context or {}).get("detail") or {}
    identity = IndexerMediaSearchRequest.create(
        title=verification["title"], aliases=[arguments["query"]],
        original_title=" ".join(str(detail.get("original_name") or "").split())[:120],
    )
    partial_sources = False

    def search(episode: int | None) -> bool:
        nonlocal requests, stop_reason, pool, partial_sources
        timeout, stop_reason = budget.reserve(context, requests)
        if not timeout:
            return False
        per_episode_arguments = {
            **arguments,
            "episode": episode,
            "limit": MAX_RESOURCE_CANDIDATES if shared else arguments["limit_per_episode"],
        }
        search_args = _episode_search_arguments(per_episode_arguments, verification["title"])
        if shared:
            search_args["original_title"] = identity.original_title
        searched = search_resources(search_args, timeout_seconds=timeout)
        requests += 1
        verification["resource_search_attempted"] = True
        evidence.extend(searched.evidence)
        raw = searched.data if isinstance(searched.data, dict) else {}
        partial_sources |= bool(raw.get("partial") or searched.status == "partial")
        candidates = {}
        for item in [*raw.get("items", []), *pool["items"]]:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            if shared:
                ranked = rank_item(IndexerItem(
                    site_id=str(item.get("site_id") or "unknown"), site_name="indexer",
                    title=item["title"],
                ), media=identity, fallback_query=identity.title)
                if not set(ranked.match_reasons) & {"title_exact", "title_contains"}:
                    continue
            candidates.setdefault(item.get("result_id"), item)
        pool = {
            **raw, "items": list(candidates.values())[:MAX_RESOURCE_CANDIDATES],
            "sites_attempted": sorted(set(pool.get("sites_attempted", [])) | set(raw.get("sites_attempted", []))),
            "sites_succeeded": sorted(set(pool.get("sites_succeeded", [])) | set(raw.get("sites_succeeded", []))),
            "errors": [*pool.get("errors", []), *raw.get("errors", [])][:16],
            "partial": partial_sources,
        }
        for target in selected:
            number = target["episode"]
            if number in by_episode:
                continue
            search_data = rank_episode_search(
                {**pool, "search_scope": "episode_exact" if number == episode else "season_shared",
                 "exact_search_attempted": number == episode},
                season=target["season"], episode=number, preferences=preferences,
                preference_overrides=arguments.get("preference_overrides"), mapping_context=mapping_context,
            )
            if number != episode and not matched_episode_items(search_data):
                continue
            search_data["items"] = search_data["items"][:arguments["limit_per_episode"]]
            recommendation = search_data["recommendation"]
            recommendation["candidate_count"] = sum(item["quality"]["eligible"] for item in search_data["items"])
            recommendation["alternatives"] = recommendation["alternatives"][:arguments["limit_per_episode"] - 1]
            by_episode[number] = {
                **target, "episode_label": f"S{target['season']:02d}E{number:02d}",
                "ok": bool(searched.ok), "status": searched.status,
                "summary": search_data["recommendation"]["reason"] if number != episode else searched.summary[:160],
                "search": search_data,
            }
        return True

    if not shared or search(None):
        for target in selected:
            if target["episode"] not in by_episode and not search(target["episode"]):
                break
    episodes = [by_episode[number] for number in sorted(by_episode)]
    total_candidates = sum(len(matched_episode_items(item["search"])) for item in episodes)
    failed = sum(not item["ok"] for item in episodes)
    completed = len(episodes) - failed
    remaining_episodes = [target for target in missing if target["episode"] not in by_episode]
    remaining = len(remaining_episodes)
    if stop_reason == "time_budget":
        suggestions.append("批量搜索已达到本次耗时上限。")
    elif stop_reason == "request_budget":
        suggestions.append("批量搜索已达到本次请求数上限。")
    elif stop_reason == "cancelled":
        suggestions.append("本轮已停止，尚未完成的缺集资源不再检索。")
    if remaining:
        stop_reason = stop_reason or "max_episodes"
    if failed or remaining or partial_sources:
        status = "partial"
    elif total_candidates:
        status = "success"
    else:
        status = "empty"
    ok = bool(completed)
    summary = f"已核验第 {arguments['season']} 季 {len(missing)} 个缺集，并搜索其中 {len(episodes)} 集"
    if total_candidates:
        summary += f"，找到 {total_candidates} 项季集标题匹配候选"
    elif completed:
        summary += "，本轮未找到匹配目标集的资源"
    if remaining:
        summary += f"；其余 {remaining} 集未完成精确检索"
    if failed:
        summary += f"；其中 {failed} 集查询未完成"

    if total_candidates:
        suggestions.append("仅报告匹配目标集的候选；标题匹配不代表已核验文件清单，不推荐旧集或未知映射。")
    if remaining:
        suggestions.append(f"本批剩余 {remaining} 集未完成精确检索；具体范围见 remaining_episodes。共享查询无匹配不能视为逐集搜索已穷尽。")
    if failed:
        suggestions.append("部分集的资源站搜索未完成；这不代表资源站中没有相关资源。")
    if partial_sources:
        suggestions.append("部分站点未完成；保留成功来源的匹配，不据此判断全网没有资源。")
    data = {
        "verification": verification,
        "missing_total": len(missing),
        "processed": len(episodes),
        "remaining": remaining,
        "remaining_episodes": remaining_episodes,
        "failed": failed,
        "failed_episodes": [{"season": item["season"], "episode": item["episode"]} for item in episodes if not item["ok"]],
        "search_requests": requests,
        "stop_reason": stop_reason,
        "truncated": bool(remaining),
        "episodes": episodes,
    }
    return _attach_season_resource_reference(
        ToolResult(
            ok,
            status,
            summary,
            data=data,
            model_data={**data, "episodes": [
                {**item, "search": episode_search_model_data(item["search"])}
                for item in episodes
            ]},
            evidence=evidence[:16]
            + [
                Evidence(
                    "agent_verification",
                    "批量资源站搜索仅处理本次媒体库与 TMDB 审计确认的已播缺集；未自动提交下载。",
                    _now(),
                )
            ],
            suggestions=suggestions,
            error="" if ok else "该批次资源站搜索未完成。",
        ),
    )


def _search_missing_resource_batch(
    items: list[dict[str, Any]],
    *,
    preferences: dict[str, Any] | None,
    context: ToolContext | None = None,
) -> ToolResult:
    """复用单季快照；匹配候选可推荐，未核验候选仅保留供明确选择。"""
    budget = _MissingResourceBudget(max_requests=min(24, _MISSING_SEASON_SEARCH_MAX_REQUESTS * len(items)))

    def search(item: dict[str, Any]) -> ToolResult:
        try:
            if time.monotonic() >= budget.deadline:
                pending = [{"season": item["season"], "episode": episode} for episode in item.get("episodes", [])]
                return ToolResult(False, "timeout", "批量耗时上限已到，本部未执行审计或资源检索", data={
                    "verification": {"resource_search_attempted": False},
                    "remaining": len(pending), "remaining_episodes": pending, "stop_reason": "time_budget",
                })
            return search_missing_season_resources(
                item, preferences=preferences, context=context, _budget=budget
            )
        except Exception:  # noqa: BLE001 - 批量隔离单部外部检索失败
            return ToolResult(
                False,
                "unavailable",
                "本部资源检索未完成，不能判断是否有资源",
            )

    with ThreadPoolExecutor(
        max_workers=min(3, len(items)), thread_name_prefix="missing-resources"
    ) as pool:
        results = list(pool.map(search, items))

    presented: list[tuple[Any, dict[str, Any]]] = []
    public_items: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    by_id: dict[str, tuple[int, str]] = {}
    recommended_positions: list[int] = []
    review_positions: list[int] = []

    for arguments, result in zip(items, results):
        data = result.data if isinstance(result.data, dict) else {}
        reference = next(
            (ref for ref in result.references if ref.kind == "resource_candidates"),
            None,
        )
        candidates = (
            reference.value.get("candidates", []) if reference is not None else []
        )
        candidates_by_id = {candidate["result_id"]: candidate for candidate in candidates}
        verification = data.get("verification") or {}
        identity = str(
            verification.get("tmdb_id")
            or arguments.get("tmdb_id")
            or arguments["query"]
        )
        group = {
            "query": arguments["query"],
            "season": arguments["season"],
            "ok": result.ok,
            "status": result.status,
            "summary": result.summary[:160],
            "resource_search_attempted": verification.get("resource_search_attempted"),
            "suggestions": list(result.suggestions),
            "positions": [],
            "review_positions": [],
            "uncovered_episodes": [],
            "missing_total": data.get("missing_total"),
            "remaining": data.get("remaining", 0),
            "remaining_episodes": data.get("remaining_episodes", []),
            "failed": data.get("failed", 0),
            "sites_succeeded": [],
            "errors": [],
        }
        for episode in data.get("episodes", []):
            target = (episode["season"], episode["episode"])
            searched = episode.get("search") or {}
            group["sites_succeeded"] = sorted(
                set(group["sites_succeeded"]) | set(searched.get("sites_succeeded", []))
            )
            group["errors"].extend(searched.get("errors", [])[:3])
            ranked_items = [
                item
                for item in searched.get("items", [])
                if isinstance(item, dict)
                and (item.get("quality") or {}).get("eligible")
            ]
            matched = [candidates_by_id[item["result_id"]] for item in ranked_items
                       if item.get("result_id") in candidates_by_id]
            proved = next((candidate for candidate in matched
                           if candidate["match"] in {"exact_episode", "episode_pack"}
                           and candidate["confidence"] in {"high", "medium"}), None)
            review = None if proved else next((candidate for candidate in matched
                                               if candidate["match"] in {"unknown", "season_pack"}), None)
            chosen = proved or review
            prior = by_id.get(chosen["result_id"]) if chosen else None
            reason = "no_match"
            if chosen is not None and prior and prior[1] != identity:
                chosen = None
                prior = None
                reason = "identity_conflict"
            elif chosen is not None and not prior and len(presented) >= MAX_RESOURCE_CANDIDATES:
                chosen = None
                reason = "candidate_limit"
            elif review is not None:
                reason = "needs_review"
            elif not episode.get("ok"):
                reason = "unavailable"
            elif ranked_items:
                reason = "needs_review"

            if chosen is None:
                group["uncovered_episodes"].append(
                    {"season": target[0], "episode": target[1], "reason": reason}
                )
                continue
            if prior:
                position = prior[0]
            else:
                position = len(presented) + 1
                by_id[chosen["result_id"]] = (position, identity)
                presented.append((reference, chosen))
                public = {
                    key: value
                    for key, value in chosen.items()
                    if not key.startswith("_")
                }
                public.update(position=position, media_title=arguments["query"])
                public_items.append(public)

            target_positions = (
                group["positions"] if proved is not None else group["review_positions"]
            )
            global_positions = (
                recommended_positions if proved is not None else review_positions
            )
            if position not in target_positions:
                target_positions.append(position)
            if position not in global_positions:
                global_positions.append(position)
            if review is not None:
                group["uncovered_episodes"].append(
                    {
                        "season": target[0],
                        "episode": target[1],
                        "reason": "needs_review",
                        "position": position,
                    }
                )

        group["errors"] = group["errors"][:3]
        if group["uncovered_episodes"] or group["remaining"]:
            group["status"] = (
                "partial"
                if group["positions"]
                else "needs_review"
                if group["review_positions"]
                else result.status
            )
        groups.append(group)

    partial = any(
        not group["ok"] or group["status"] == "partial" or group["uncovered_episodes"] or group["remaining"]
        for group in groups
    )
    status = "partial" if partial else "success" if presented else "empty"
    proved_groups = sum(bool(group["positions"]) for group in groups)
    summary = f"已核对 {len(groups)} 部/季，{proved_groups} 部找到匹配目标集的资源"
    unsearched = sum(group["resource_search_attempted"] is False for group in groups)
    if unsearched:
        summary += f"；{unsearched} 部未执行资源站检索，不能据此判断没有资源"
    unfinished = sum(group["failed"] + group["remaining"] for group in groups)
    if unfinished:
        summary += f"；{unfinished} 集查询未完成"
    result = ToolResult(
        any(item.ok for item in results),
        status,
        summary,
        data={
            "groups": groups,
            "items": public_items,
            "recommended_positions": recommended_positions,
            "review_positions": review_positions,
        },
        suggestions=[
            "推荐位置仅表示季集标题匹配，未核验合集文件清单。只用简表列目标和检索结论；未执行或失败的集标为查询未完成。不复述旧集/未知映射，不推断最新发布进度；仅当用户主动查看或明确选择未核验候选时使用待核对位置。",
            "站点失败时简要说明未完成来源，不把检索不完整说成全网无资源；本工具不提交下载。",
        ],
    )
    result.model_data = {
        "groups": [{
            **{key: value for key, value in group.items() if key not in {"review_positions", "status", "uncovered_episodes"}},
            "uncovered_episodes": [
                {**{key: value for key, value in item.items() if key != "position"},
                 "reason": "no_match" if item["reason"] == "needs_review" else item["reason"]}
                for item in group["uncovered_episodes"]
            ],
        } for group in groups],
        "items": [item for item in public_items if item["position"] in recommended_positions],
        "recommended_positions": recommended_positions,
    }
    if presented:
        result.references.append(
            merge_resource_candidate_references(presented, status=status)
        )
    return result
