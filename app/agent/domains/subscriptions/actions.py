"""媒体订阅创建、删除和启停的预检与确认执行。"""

from __future__ import annotations

import json
import re
import secrets
import unicodedata
from typing import Any

from app import database as db
from app.agent.async_bridge import AsyncBridgeUnavailable, ensure_sync_bridge_available
from app.agent.domains.workspace.actions import _safe_title
from app.agent.errors import AgentToolError
from app.agent.models import Evidence, ToolResult
from app.discovery.models import MediaCard, ProviderError
from app.discovery.service import get_discovery_service
from app.indexers.runtime import run_indexer_awaitable_sync
from app.logger import get_logger
from app.modules.media_subscriptions import (
    MediaSubscriptionError,
    get_media_subscription_service,
)
from app.repositories.media_subscriptions import normalize_subscription_interval

from . import common

logger = get_logger(__name__)
_CREATE_TIMEOUT_SECONDS = 35.0
_ALLOWED_PROVIDERS = frozenset({"tmdb", "douban", "bangumi"})
_ALLOWED_MEDIA_TYPES = frozenset({"movie", "tv"})
_PUBLIC_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,180}$")


def media_subscription_enabled_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise AgentToolError("工具参数必须是 JSON 对象")
    if set(arguments) != {"subscription_id", "enabled"}:
        raise AgentToolError(
            "media.set_subscription_enabled 只接受 subscription_id 和 enabled 参数"
        )
    enabled = arguments.get("enabled")
    if not isinstance(enabled, bool):
        raise AgentToolError("enabled 必须是布尔值")
    return {
        "subscription_id": common._strict_subscription_id(arguments.get("subscription_id")),
        "enabled": enabled,
    }


def _subscription_identity(
    provider: Any, external_id: Any, media_type: Any
) -> tuple[str, str, str]:
    normalized_provider = (
        unicodedata.normalize("NFKC", str(provider or "")).strip().casefold()
    )
    normalized_external_id = unicodedata.normalize(
        "NFKC", str(external_id or "")
    ).strip()
    normalized_media_type = (
        unicodedata.normalize("NFKC", str(media_type or "")).strip().casefold()
    )
    if normalized_provider not in _ALLOWED_PROVIDERS:
        raise AgentToolError("provider 仅支持 tmdb、douban 或 bangumi")
    if not _PUBLIC_ID_RE.fullmatch(normalized_external_id):
        raise AgentToolError("external_id 格式无效")
    if normalized_media_type not in _ALLOWED_MEDIA_TYPES:
        raise AgentToolError("media_type 仅支持 movie 或 tv")
    return normalized_provider, normalized_external_id, normalized_media_type


def media_subscription_create_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise AgentToolError("工具参数必须是 JSON 对象")
    if not set(arguments).issubset(
        {
            "provider",
            "external_id",
            "media_type",
            "season",
            "check_interval_minutes",
        }
    ):
        raise AgentToolError(
            "media.create_subscription 只接受 provider、external_id、media_type、"
            "可选 season 和 check_interval_minutes 参数"
        )
    if not {"provider", "external_id", "media_type"}.issubset(arguments):
        raise AgentToolError("创建媒体订阅需要精确来源、媒体 ID 和媒体类型")
    provider, external_id, media_type = _subscription_identity(
        arguments.get("provider"),
        arguments.get("external_id"),
        arguments.get("media_type"),
    )
    season = arguments.get("season")
    if season is not None:
        if (
            isinstance(season, bool)
            or not isinstance(season, int)
            or not 1 <= season <= 100
        ):
            raise AgentToolError("season 必须是 1 到 100 的整数")
        if media_type != "tv":
            raise AgentToolError("电影订阅不支持季度筛选")
    check_interval_minutes = arguments.get("check_interval_minutes")
    if check_interval_minutes is not None:
        if isinstance(check_interval_minutes, bool) or not isinstance(check_interval_minutes, int):
            raise AgentToolError("check_interval_minutes 必须是整数")
        try:
            check_interval_minutes = normalize_subscription_interval(check_interval_minutes)
        except MediaSubscriptionError as exc:
            raise AgentToolError(str(exc)) from exc
    normalized: dict[str, Any] = {
        "provider": provider,
        "external_id": external_id,
        "media_type": media_type,
    }
    if season is not None:
        normalized["season"] = int(season)
    if check_interval_minutes is not None:
        normalized["check_interval_minutes"] = check_interval_minutes
    return normalized


def media_subscription_delete_arguments(arguments: dict[str, Any]) -> dict[str, int]:
    if not isinstance(arguments, dict):
        raise AgentToolError("工具参数必须是 JSON 对象")
    if set(arguments) != {"subscription_id"}:
        raise AgentToolError("media.delete_subscription 只接受 subscription_id 参数")
    return {
        "subscription_id": common._strict_subscription_id(arguments.get("subscription_id"))
    }


def _resolved_tmdb_id(provider: str, external_id: str, media_type: str) -> str:
    if provider == "tmdb":
        if (
            not external_id.isascii()
            or not external_id.isdigit()
            or int(external_id) <= 0
        ):
            raise AgentToolError("TMDB 媒体 ID 无效", code="precondition_failed")
        return external_id
    mapping = db.get_media_external_id(provider, external_id, media_type)
    tmdb_id = str(mapping["tmdb_id"] or "").strip() if mapping is not None else ""
    if (
        mapping is None
        or not bool(mapping["confirmed"])
        or not tmdb_id.isascii()
        or not tmdb_id.isdigit()
        or int(tmdb_id) <= 0
    ):
        raise AgentToolError(
            "该来源尚未确认 TMDB 映射，请先在探索页确认媒体身份",
            code="precondition_failed",
        )
    return tmdb_id


def _decode_create_context(value: str) -> dict[str, Any]:
    try:
        payload = json.loads((value or ""))
    except (TypeError, ValueError) as exc:
        raise AgentToolError("确认上下文无效", code="confirmation_invalid") from exc
    if not isinstance(payload, dict):
        raise AgentToolError("确认上下文无效", code="confirmation_invalid")
    return payload


def prepare_create_media_subscription(
    arguments: dict[str, Any],
) -> tuple[ToolResult, str]:
    provider = str(arguments["provider"])
    external_id = str(arguments["external_id"])
    media_type = str(arguments["media_type"])
    season = arguments.get("season")
    check_interval_minutes = int(arguments.get("check_interval_minutes") or 4320)
    interval_label = {4320: "每 3 天", 10080: "每 7 天"}.get(check_interval_minutes, f"每 {check_interval_minutes} 分钟")
    tmdb_id = _resolved_tmdb_id(provider, external_id, media_type)
    with db.get_conn() as conn:
        snapshot = db.media_subscription_identity_snapshot(conn, tmdb_id, media_type)
    if snapshot.get("exists") and not snapshot.get("deleted_at"):
        raise AgentToolError("该媒体已经在追更订阅中", code="precondition_failed")

    try:
        card = get_discovery_service().get_detail(provider, media_type, external_id)
    except ProviderError as exc:
        raise AgentToolError(
            exc.safe_message or "暂时无法核对该影视条目",
            code="precondition_failed",
        ) from exc
    except Exception as exc:
        logger.warning("Agent 媒体订阅创建预检失败 type=%s", type(exc).__name__)
        raise AgentToolError(
            "暂时无法核对该影视条目", code="confirmation_unavailable"
        ) from exc
    if not isinstance(card, MediaCard) or (
        card.provider,
        card.external_id,
        card.media_type,
    ) != (provider, external_id, media_type):
        raise AgentToolError(
            "影视数据源返回的条目标识不一致", code="precondition_failed"
        )
    title = _safe_title(card.title, fallback="该媒体")
    year = (card.year or "").strip()[:12]
    season_label = f"第 {int(season)} 季" if season is not None else "全部缺失内容"
    context = {
        "operation": "create",
        "provider": provider,
        "external_id": external_id,
        "media_type": media_type,
        "tmdb_id": tmdb_id,
        "season": int(season) if season is not None else None,
        "check_interval_minutes": check_interval_minutes,
        "snapshot": snapshot,
    }
    return ToolResult(
        ok=True,
        status="confirmation_required",
        summary=f"确认后将为《{title}》创建媒体追更订阅",
        data={
            "operation": "create",
            "title": title,
            "year": year,
            "media_type": common._media_type_label(media_type),
            "monitor_scope": season_label,
            "season": int(season) if season is not None else None,
            "check_interval_minutes": check_interval_minutes,
            "check_interval": interval_label,
            "affected": 1,
            "effects": [
                "确认后只会创建或恢复本地追更订阅，不会立即下载资源。",
                (
                    f"新订阅将按{interval_label}"
                    "检查缺失内容，候选下载仍需按策略确认。"
                ),
            ],
        },
        evidence=[
            Evidence(
                "discovery_provider",
                "预检时核对精确影视条目及已确认 TMDB 身份；未搜索资源或提交下载。",
                common._now(),
            )
        ],
        suggestions=["确认票据只可使用一次；确认前请核对片名、类型和季度范围。"],
    ), json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def create_media_subscription_confirmed(
    arguments: dict[str, Any], expected_context: str
) -> ToolResult:
    context = _decode_create_context(expected_context)
    identity = (
        str(arguments["provider"]),
        str(arguments["external_id"]),
        str(arguments["media_type"]),
    )
    if (
        context.get("operation") != "create"
        or identity
        != (
            str(context.get("provider") or ""),
            str(context.get("external_id") or ""),
            str(context.get("media_type") or ""),
        )
        or arguments.get("season") != context.get("season")
        or int(arguments.get("check_interval_minutes") or 4320)
        != int(context.get("check_interval_minutes") or 4320)
    ):
        raise AgentToolError("确认上下文与目标不一致", code="confirmation_invalid")
    tmdb_id = _resolved_tmdb_id(*identity)
    if tmdb_id != str(context.get("tmdb_id") or ""):
        return ToolResult(
            ok=False,
            status="conflict",
            summary="媒体身份映射已变化，请重新预检",
            error="确认快照已失效。",
        )
    with db.get_conn() as conn:
        snapshot = db.media_subscription_identity_snapshot(conn, tmdb_id, identity[2])
    if snapshot != context.get("snapshot"):
        return ToolResult(
            ok=False,
            status="conflict",
            summary="媒体追更订阅状态已变化，请重新预检",
            error="确认快照已失效。",
        )
    season = arguments.get("season")
    check_interval_minutes = int(arguments.get("check_interval_minutes") or 4320)
    interval_label = {4320: "每 3 天", 10080: "每 7 天"}.get(check_interval_minutes, f"每 {check_interval_minutes} 分钟")
    payload: dict[str, Any] = {
        "provider": identity[0],
        "external_id": identity[1],
        "tmdb_id": tmdb_id,
        "media_type": identity[2],
        "monitor_mode": "selected" if season is not None else "missing",
        "seasons": [int(season)] if season is not None else [],
        "include_specials": False,
        "action": "confirm",
        "download_target": "guangya",
        "check_interval_minutes": check_interval_minutes,
        "enabled": True,
    }
    try:
        ensure_sync_bridge_available()
        created_result = run_indexer_awaitable_sync(
            get_media_subscription_service().create_subscription(
                payload,
                identity_confirmed=identity[0] != "tmdb",
                expected_snapshot=context["snapshot"],
            ),
            timeout_seconds=_CREATE_TIMEOUT_SECONDS,
        )
    except MediaSubscriptionError as exc:
        return ToolResult(
            ok=False,
            status=(exc.code or "unavailable")[:40],
            summary="媒体追更订阅创建失败",
            error=str(exc)[:240],
        )
    except (AsyncBridgeUnavailable, TimeoutError, RuntimeError) as exc:
        logger.warning("Agent 媒体订阅创建执行失败 type=%s", type(exc).__name__)
        return ToolResult(
            ok=False,
            status="unavailable",
            summary="媒体追更订阅暂时无法创建",
            error="订阅服务暂时不可用。",
        )
    subscription = (
        created_result.get("subscription") if isinstance(created_result, dict) else None
    )
    subscription = subscription if isinstance(subscription, dict) else {}
    subscription_id = int(subscription.get("id") or 0)
    created = (
        bool(created_result.get("created"))
        if isinstance(created_result, dict)
        else False
    )
    runtime_refreshed = common._reload_scheduler()
    return ToolResult(
        ok=True,
        status="completed",
        summary="媒体追更订阅已创建" if created else "媒体追更订阅已恢复",
        data={
            "operation": "create" if created else "restore",
            "subscription_number": subscription_id,
            "title": _safe_title(subscription.get("title"), fallback="该媒体"),
            "media_type": common._media_type_label(identity[2]),
            "season": int(season) if season is not None else None,
            "check_interval_minutes": check_interval_minutes,
            "check_interval": interval_label,
            "affected": 1,
            "created": created,
            "runtime_refreshed": runtime_refreshed,
        },
        evidence=[
            Evidence(
                "media_subscription_database",
                "已使用一次性确认票据创建本地追更订阅；未立即搜索或下载资源。",
                common._now(),
            )
        ],
        suggestions=(
            ["订阅已进入后台调度，可继续查看媒体订阅更新。"]
            if runtime_refreshed
            else ["订阅已保存；请重启 MediaFlux 使当前进程重新加载调度。"]
        ),
    )


def prepare_delete_media_subscription(
    arguments: dict[str, Any],
) -> tuple[ToolResult, str]:
    subscription_id = int(arguments["subscription_id"])
    with db.get_conn() as conn:
        state = db.media_subscription_mutation_snapshot(conn, subscription_id)
    if not state.get("exists"):
        raise AgentToolError("未找到指定的媒体追更订阅", code="precondition_failed")
    row = db.get_media_subscription(subscription_id)
    title = _safe_title(row["title"] if row is not None else "", fallback="该媒体")
    return ToolResult(
        ok=True,
        status="confirmation_required",
        summary=f"确认后将删除《{title}》的媒体追更订阅",
        data={
            "operation": "delete",
            "subscription_number": subscription_id,
            "title": title,
            "affected": 1,
            "available_candidates": int(state.get("available_candidates") or 0),
            "claimed_admissions": int(state.get("claimed_admissions") or 0),
            "running_checks": int(state.get("running_checks") or 0),
            "effects": [
                "订阅会被软删除并停止后续检查。",
                "未提交候选、待处理准入和运行中检查会失效；已提交下载和媒体文件不会删除。",
            ],
        },
        evidence=[
            Evidence(
                "media_subscription_database",
                "只核对目标订阅及未完成工作数量；不会删除下载任务或媒体文件。",
                common._now(),
            )
        ],
        suggestions=["删除属于高风险操作；确认票据只可使用一次。"],
    ), common._fingerprint(state)


def delete_media_subscription_confirmed(
    arguments: dict[str, Any], expected_context: str
) -> ToolResult:
    subscription_id = int(arguments["subscription_id"])
    with db.get_conn() as conn:
        state = db.media_subscription_mutation_snapshot(conn, subscription_id)
    if (
        not state.get("exists")
        or not secrets.compare_digest(common._fingerprint(state), (expected_context or ""))
        or not get_media_subscription_service().delete_subscription(
            subscription_id, expected_snapshot=state
        )
    ):
        return ToolResult(
            ok=False,
            status="conflict",
            summary="媒体追更订阅状态已变化，请重新预检",
            error="确认快照已失效。",
        )
    runtime_refreshed = common._reload_scheduler()
    return ToolResult(
        ok=True,
        status="completed",
        summary="媒体追更订阅已删除",
        data={
            "operation": "delete",
            "subscription_number": subscription_id,
            "affected": 1,
            "expired_candidates": int(state.get("available_candidates") or 0),
            "cancelled_admissions": int(state.get("claimed_admissions") or 0),
            "cancelled_runs": int(state.get("running_checks") or 0),
            "runtime_refreshed": runtime_refreshed,
        },
        evidence=[
            Evidence(
                "media_subscription_database",
                "已使用一次性确认票据软删除订阅；已提交下载与媒体文件未被删除。",
                common._now(),
            )
        ],
        suggestions=(
            ["订阅已删除，后续不会再安排检查。"]
            if runtime_refreshed
            else ["订阅已删除；请重启 MediaFlux 使当前进程重新加载调度。"]
        ),
    )


def prepare_set_media_subscription_enabled(
    arguments: dict[str, Any],
) -> tuple[ToolResult, str]:
    subscription_id = int(arguments["subscription_id"])
    with db.get_conn() as conn:
        state = db.media_subscription_mutation_snapshot(conn, subscription_id)
    if not state.get("exists"):
        raise AgentToolError("未找到指定的媒体追更订阅", code="precondition_failed")
    requested = bool(arguments["enabled"])
    if bool(state["enabled"]) == requested:
        label = "启用" if requested else "暂停"
        raise AgentToolError(f"该媒体追更订阅已经{label}", code="precondition_failed")
    row = db.get_media_subscription(subscription_id)
    title = _safe_title(row["title"] if row is not None else "", fallback="该媒体")
    effects = (
        ["恢复后会重新进入检查队列，并在下一次调度时核对媒体库。"]
        if requested
        else [
            "暂停后不会安排新的检查。",
            "尚未提交的候选资源和检查任务会失效；已提交的下载任务与文件不受影响。",
        ]
    )
    preview = ToolResult(
        ok=True,
        status="confirmation_required",
        summary=f"确认后将{'恢复' if requested else '暂停'}《{title}》的媒体追更",
        data={
            "operation": "enable" if requested else "disable",
            "subscription_number": subscription_id,
            "title": title,
            "enabled": requested,
            "affected": 1,
            "effects": effects,
        },
        evidence=[
            Evidence(
                "media_subscription_database",
                "仅核对本地订阅状态和待失效记录数量；未访问资源网站或下载器。",
                common._now(),
            )
        ],
        suggestions=["确认票据只可使用一次；订阅状态变化后需要重新预检。"],
    )
    return preview, common._fingerprint(state)


def set_media_subscription_enabled_confirmed(
    arguments: dict[str, Any], expected_context: str
) -> ToolResult:
    subscription_id = int(arguments["subscription_id"])
    requested = bool(arguments["enabled"])
    with db.get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        state = db.media_subscription_mutation_snapshot(conn, subscription_id)
        if not state.get("exists") or not secrets.compare_digest(
            common._fingerprint(state), (expected_context or "")
        ):
            return ToolResult(
                ok=False,
                status="conflict",
                summary="媒体追更订阅状态已变化，请重新预检",
                error="确认快照已失效。",
            )
        changed = db.update_media_subscription_config(
            subscription_id, connection=conn, enabled=int(requested), invalidate_related=not requested,
            reason="订阅已暂停", run_summary="订阅已暂停，当前检查已取消",
        )
        assert changed is not None  # 同一事务已核对存在与确认快照。

    runtime_refreshed = common._reload_scheduler()
    return ToolResult(
        ok=True,
        status="completed",
        summary="媒体追更已恢复" if requested else "媒体追更已暂停",
        data={
            "operation": "enable" if requested else "disable",
            "subscription_number": subscription_id,
            "enabled": requested,
            "affected": 1,
            "expired_candidates": changed["expired_candidates"],
            "cancelled_admissions": changed["cancelled_admissions"],
            "cancelled_runs": changed["cancelled_runs"],
            "runtime_refreshed": runtime_refreshed,
        },
        evidence=[
            Evidence(
                "media_subscription_database",
                "已使用一次性确认票据原子更新订阅；未操作已提交下载任务或媒体文件。",
                common._now(),
            )
        ],
        suggestions=(
            ["追更已恢复，下一次调度会重新检查媒体库。"]
            if requested and runtime_refreshed
            else ["追更已暂停；已提交下载任务和文件不会受影响。"]
            if not requested
            else ["配置已保存；请重启 MediaFlux 使调度器在当前进程重新读取配置。"]
        ),
    )
