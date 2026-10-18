"""按公开 TMDB 身份批量核对 Jellyfin / Emby 在库状态。"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any

from app.agent.errors import AgentToolError
from app.agent.models import Evidence, ToolResult
from app.services import inspect_media_identity_batch, summarize_media_identity_batch

_ALLOWED_ARGUMENTS = {"items"}
_ALLOWED_ITEM_FIELDS = {"tmdb_id", "media_type", "title", "year"}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _visible_text(value: object, *, limit: int) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip()
    if any(unicodedata.category(char).startswith("C") for char in normalized):
        return ""
    return " ".join(normalized.split())[:limit]


def batch_presence_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise AgentToolError("工具参数必须是 JSON 对象")
    extra = set(arguments) - _ALLOWED_ARGUMENTS
    if extra:
        raise AgentToolError(f"不支持的工具参数：{', '.join(sorted(extra))}")
    raw_items = arguments.get("items")
    if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= 50:
        raise AgentToolError("items 必须包含 1 到 50 个媒体身份")

    items: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_items:
        if not isinstance(raw, dict) or set(raw) - _ALLOWED_ITEM_FIELDS:
            raise AgentToolError("媒体身份字段无效")
        tmdb_id = str(raw.get("tmdb_id") or "").strip()
        if not re.fullmatch(r"[1-9][0-9]{0,9}", tmdb_id):
            raise AgentToolError("tmdb_id 必须是 1 到 10 位正整数")
        media_type = str(raw.get("media_type") or "").strip().casefold()
        if media_type not in {"movie", "tv"}:
            raise AgentToolError("media_type 仅支持 movie 或 tv")
        title = _visible_text(raw.get("title"), limit=240)
        if not title:
            raise AgentToolError("title 必须是可见文本")
        year = str(raw.get("year") or "").strip()
        if year and not re.fullmatch(r"(?:19|20)[0-9]{2}", year):
            raise AgentToolError("year 必须是 1900 到 2099 的四位年份")
        key = (media_type, tmdb_id)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "tmdb_id": tmdb_id,
                "media_type": media_type,
                "title": title,
                "year": year,
            }
        )
    if not items:
        raise AgentToolError("items 中没有可核对的唯一媒体身份")
    return {"items": items}


def batch_library_presence(arguments: dict[str, Any]) -> ToolResult:
    normalized = batch_presence_arguments(arguments)
    requested = normalized["items"]
    sources = inspect_media_identity_batch(requested)
    if not sources:
        return ToolResult(
            ok=False,
            status="not_configured",
            summary="没有可核对的媒体服务器",
            data={"total": len(requested), "items": [], "sources": []},
            evidence=[Evidence("media_servers", "检查已配置媒体服务器。", _now())],
            suggestions=["请先在设置中启用并配置 Jellyfin 或 Emby。"],
            error="没有可核对的媒体服务器。",
        )

    data = summarize_media_identity_batch(requested, sources)
    counts = data["counts"]
    unavailable = sum(
        1 for source in sources if str(source.get("status") or "") != "ready"
    )
    status = (
        "partial"
        if unavailable or counts["possible"] or counts["indeterminate"]
        else "success"
    )
    return ToolResult(
        ok=True,
        status=status,
        summary=(
            f"已批量核对 {data['total']} 部媒体：在库 {counts['present']}、"
            f"缺失 {counts['missing']}、待确认 {counts['possible']}、"
            f"无法判断 {counts['indeterminate']}"
        ),
        data=data,
        evidence=[
            Evidence(
                "media_server_provider_ids",
                "一次枚举已配置媒体服务器并按 TMDB Provider ID 批量核对。",
                _now(),
            )
        ],
        suggestions=[
            "present 为精确 TMDB 身份命中；possible 表示同名同年但本地条目缺少 TMDB 映射。"
        ],
    )
