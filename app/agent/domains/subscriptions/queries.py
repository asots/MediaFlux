"""媒体订阅的安全摘要与实时只读更新核对。"""

from __future__ import annotations

import asyncio
from typing import Any

from app import database as db
from app.agent.async_bridge import AsyncBridgeUnavailable, ensure_sync_bridge_available
from app.agent.domains.workspace.actions import _safe_title
from app.agent.errors import AgentToolError
from app.agent.models import Evidence, ToolResult
from app.indexers.runtime import run_indexer_awaitable_sync
from app.logger import get_logger
from app.modules.media_subscriptions import (
    MediaSubscriptionError,
    get_media_subscription_service,
)

from . import common

logger = get_logger(__name__)
_MAX_SUMMARIES = 16
_MAX_UPDATE_SUBSCRIPTIONS = 8
_MAX_UPDATE_CANDIDATES = 12
_UPDATE_PREVIEW_TIMEOUT_SECONDS = 35.0
_UPDATE_ITEM_TIMEOUT_SECONDS = 25.0


def media_subscription_summaries_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise AgentToolError("工具参数必须是 JSON 对象")
    if arguments:
        raise AgentToolError("media.subscription_summaries 不接受参数")
    return {}


def media_subscription_updates_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise AgentToolError("工具参数必须是 JSON 对象")
    if arguments:
        raise AgentToolError("media.subscription_updates 不接受参数")
    return {}


def _safe_row(row: Any) -> dict[str, Any]:
    enabled = bool(row["enabled"])
    status = {
        "new": "等待检查",
        "checking": "检查中",
        "satisfied": "当前完整",
        "missing": "发现缺失",
        "inconclusive": "暂无法确认",
        "error": "检查异常",
        "paused": "已暂停",
    }.get(str(row["status"] or "").strip().lower(), "状态未知") if enabled else "已暂停"
    return {
        "subscription_number": int(row["id"]),
        "title": _safe_title(row["title"], fallback="未命名媒体"),
        "media_type": common._media_type_label(row["media_type"]),
        "enabled": enabled,
        "status": status,
        "missing_count": max(0, int(row["missing_count"] or 0)),
    }


def list_media_subscription_summaries(_arguments: dict[str, Any]) -> ToolResult:
    try:
        total = db.count_media_subscriptions()
        rows = db.list_media_subscriptions(limit=_MAX_SUMMARIES)
    except Exception as exc:  # noqa: BLE001 - 数据库读取失败转为工具不可用结果
        logger.warning("Agent 媒体追更摘要读取失败 type=%s", type(exc).__name__)
        return ToolResult(
            ok=False,
            status="unavailable",
            summary="暂时无法读取媒体追更订阅",
            data={"total": 0, "returned": 0, "truncated": False, "items": []},
            evidence=[
                Evidence(
                    "media_subscription_database",
                    "尝试读取本地媒体追更安全摘要；未读取站点、下载目标、路径或凭据。",
                    common._now(),
                )
            ],
            suggestions=["请检查本地数据库状态后重试。"],
            error="媒体追更订阅当前不可用。",
        )
    items = [_safe_row(row) for row in rows[:_MAX_SUMMARIES]]
    returned = len(items)
    if total:
        status = "completed"
        summary = f"已读取 {returned} 个媒体追更订阅"
        suggestions = ["如需暂停或恢复，请明确提供一个订阅编号。"]
    else:
        status = "not_configured"
        summary = "尚未创建媒体追更订阅"
        suggestions = ["可直接说：订阅《片名》，再从搜索结果中选择准确条目。"]
    return ToolResult(
        ok=True,
        status=status,
        summary=summary,
        data={
            "total": max(0, int(total or 0)),
            "returned": returned,
            "truncated": int(total or 0) > returned,
            "items": items,
        },
        evidence=[
            Evidence(
                "media_subscription_database",
                "只读取本地媒体追更编号、标题、类型、启用状态和缺失数量。",
                common._now(),
            )
        ],
        suggestions=suggestions,
    )


def _preview_failure(row: Any, *, status: str, summary: str) -> dict[str, Any]:
    return {
        "subscription_number": int(row["id"]),
        "title": _safe_title(row["title"], fallback="未命名媒体"),
        "media_type": str(row["media_type"] or ""),
        "enabled": bool(row["enabled"]),
        "status": status,
        "summary": summary,
        "expected_count": 0,
        "local_count": 0,
        "missing_count": 0,
        "missing": [],
        "inventory_complete": False,
        "sources": [],
        "resource_search": {
            "status": "not_run",
            "attempted_count": 0,
            "candidate_count": 0,
            "truncated": False,
            "items": [],
        },
        "delivery": {
            "state": "unavailable",
            "summary": "本次未形成下载提交建议",
        },
        "checked_at": common._now(),
    }


async def _preview_media_subscription_rows(rows: list[Any]) -> list[dict[str, Any]]:
    service = get_media_subscription_service()
    semaphore = asyncio.Semaphore(3)

    async def _one(row: Any) -> dict[str, Any]:
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    service.preview_subscription_updates(
                        int(row["id"]),
                        max_search_episodes=1,
                        limit_per_media=3,
                    ),
                    timeout=_UPDATE_ITEM_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                return _preview_failure(
                    row,
                    status="timeout",
                    summary="实时核对达到单条订阅耗时上限",
                )
            except MediaSubscriptionError as exc:
                return _preview_failure(
                    row,
                    status=str(exc.code or "unavailable")[:40],
                    summary=str(exc)[:160],
                )
            except Exception as exc:  # noqa: BLE001 - 单项检查失败不丢弃其他订阅结果
                logger.warning(
                    "Agent 媒体订阅实时核对失败 subscription=%s type=%s",
                    row["id"],
                    type(exc).__name__,
                )
                return _preview_failure(
                    row,
                    status="unavailable",
                    summary="实时核对暂时不可用",
                )

    return list(await asyncio.gather(*(_one(row) for row in rows)))


def inspect_media_subscription_updates(_arguments: dict[str, Any]) -> ToolResult:
    """实时串联订阅、媒体库和资源站；只读，不提交下载或改变调度。"""
    try:
        total = db.count_media_subscriptions()
        rows = db.list_media_subscriptions(limit=_MAX_UPDATE_SUBSCRIPTIONS)
    except Exception as exc:  # noqa: BLE001 - 列表读取失败转为工具不可用结果
        logger.warning("Agent 媒体订阅实时列表读取失败 type=%s", type(exc).__name__)
        return ToolResult(
            ok=False,
            status="unavailable",
            summary="暂时无法读取媒体追更订阅",
            data={
                "total": 0,
                "returned": 0,
                "truncated": False,
                "subscriptions": [],
                "items": [],
            },
            error="媒体追更订阅当前不可用。",
        )
    if not total:
        return ToolResult(
            ok=True,
            status="not_configured",
            summary="尚未创建媒体追更订阅",
            data={
                "total": 0,
                "returned": 0,
                "truncated": False,
                "subscriptions": [],
                "items": [],
            },
            suggestions=["可直接说：订阅《片名》，再从搜索结果中选择准确条目。"],
        )

    try:
        ensure_sync_bridge_available()
        subscriptions = run_indexer_awaitable_sync(
            _preview_media_subscription_rows(rows),
            timeout_seconds=_UPDATE_PREVIEW_TIMEOUT_SECONDS,
        )
    except (AsyncBridgeUnavailable, TimeoutError, RuntimeError) as exc:
        logger.warning("Agent 媒体订阅实时组合核对不可用 type=%s", type(exc).__name__)
        return ToolResult(
            ok=False,
            status="unavailable",
            summary="媒体订阅实时核对当前不可用",
            data={
                "total": max(0, int(total or 0)),
                "returned": 0,
                "truncated": int(total or 0) > len(rows),
                "subscriptions": [],
                "items": [],
            },
            suggestions=["请稍后重试；查询失败不会触发下载或改变后台调度。"],
            error="实时核对暂时不可用。",
        )

    candidate_items: list[dict[str, Any]] = []
    for subscription in subscriptions:
        search = subscription.get("resource_search")
        if not isinstance(search, dict):
            continue
        for searched in search.get("items", []):
            if not isinstance(searched, dict):
                continue
            label = str(searched.get("label") or "")[:24]
            for candidate in searched.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                candidate_items.append(
                    {
                        **candidate,
                        "media_title": str(subscription.get("title") or "")[:120],
                        "episode_label": label,
                        "subscription_number": int(
                            subscription.get("subscription_number") or 0
                        ),
                    }
                )
                if len(candidate_items) >= _MAX_UPDATE_CANDIDATES:
                    break
            if len(candidate_items) >= _MAX_UPDATE_CANDIDATES:
                break
        if len(candidate_items) >= _MAX_UPDATE_CANDIDATES:
            break

    updates = sum(1 for item in subscriptions if item.get("status") == "missing")
    current = sum(1 for item in subscriptions if item.get("status") == "satisfied")
    paused = sum(1 for item in subscriptions if item.get("status") == "paused")
    uncertain = len(subscriptions) - updates - current - paused
    partial_searches = sum(
        1
        for item in subscriptions
        if isinstance(item.get("resource_search"), dict)
        and item["resource_search"].get("status") == "partial"
    )
    if updates and (uncertain or partial_searches):
        status = "partial"
        ok = True
        summary = (
            f"已实时核对 {len(subscriptions)} 个媒体追更订阅："
            f"{updates} 个有已播缺失，另有 {uncertain or partial_searches} 个检查未完整完成"
        )
    elif updates:
        status = "updates_available"
        ok = True
        summary = (
            f"已实时核对 {len(subscriptions)} 个媒体追更订阅："
            f"{updates} 个有已播缺失，{current} 个当前完整"
        )
    elif uncertain and not current and not paused:
        status = "unavailable"
        ok = False
        summary = "本次未能可靠核对任何媒体追更订阅"
    elif uncertain:
        status = "partial"
        ok = True
        summary = (
            f"已核对 {len(subscriptions)} 个媒体追更订阅，{uncertain} 个暂无法确认"
        )
    else:
        status = "up_to_date"
        ok = True
        summary = (
            f"已实时核对 {len(subscriptions)} 个媒体追更订阅，当前未发现新的已播缺失"
        )

    suggestions: list[str] = []
    if candidate_items:
        suggestions.append("可回复“第 1 个到光鸭”或“第 1 个到 qB”，进入下载提交确认。")
    if any(
        isinstance(item.get("resource_search"), dict)
        and item["resource_search"].get("truncated")
        for item in subscriptions
    ):
        suggestions.append("有订阅包含多个缺失项；本次只搜索每个订阅最靠前的一项。")
    if uncertain:
        suggestions.append("暂无法确认的订阅应先检查 TMDB、媒体服务器或资源站连通性。")
    elif partial_searches:
        suggestions.append(
            "部分资源站查询未完成；已返回候选仍可查看，未返回不代表没有资源。"
        )
    return ToolResult(
        ok=ok,
        status=status,
        summary=summary,
        data={
            "total": max(0, int(total or 0)),
            "returned": len(subscriptions),
            "truncated": int(total or 0) > len(subscriptions),
            "updates_available_count": updates,
            "up_to_date_count": current,
            "paused_count": paused,
            "inconclusive_count": uncertain,
            "partial_search_count": partial_searches,
            "candidate_count": len(candidate_items),
            "subscriptions": subscriptions,
            # 顶层 items 供会话绑定的“第 N 个到 qB/光鸭”安全续接使用。
            "items": candidate_items,
            "check_definition": "subscription_tmdb_vs_media_servers_with_bounded_indexer_search",
            "read_only": True,
        },
        evidence=[
            Evidence(
                "media_subscription_database",
                "读取当前媒体追更订阅及其安全策略摘要。",
                common._now(),
            ),
            Evidence(
                "media_servers+tmdb",
                "实时比较 TMDB 已播清单与 Jellyfin/Emby 本地库存。",
                common._now(),
            ),
            Evidence(
                "indexer_service",
                "仅对确认缺失项执行有界多站搜索；未提交下载。",
                common._now(),
            ),
        ],
        suggestions=suggestions,
        error="部分订阅暂无法确认。" if uncertain and not ok else "",
    )


def get_media_subscription_summary(arguments: dict[str, int]) -> ToolResult:
    try:
        row = db.get_media_subscription(arguments["subscription_id"])
    except Exception as exc:  # noqa: BLE001 - 详情读取失败转为工具不可用结果
        logger.warning("Agent 媒体追更详情读取失败 type=%s", type(exc).__name__)
        return ToolResult(
            ok=False,
            status="unavailable",
            summary="暂时无法读取媒体追更订阅",
            error="媒体追更订阅当前不可用。",
        )
    if row is None:
        raise AgentToolError("未找到指定的媒体追更订阅", code="precondition_failed")
    item = _safe_row(row)
    return ToolResult(
        ok=True,
        status="completed",
        summary=f"媒体追更订阅 {item['subscription_number']}：{item['status']}",
        data=item,
        evidence=[
            Evidence(
                "media_subscription_database",
                "只读取指定订阅的安全摘要；未读取站点、下载目标、路径或凭据。",
                common._now(),
            )
        ],
        suggestions=["可明确说“暂停媒体订阅编号”或“恢复媒体订阅编号”。"],
    )
