"""媒体订阅策略的安全投影、有效性校验与原子确认更新。"""

from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any

from app import database as db
from app.agent.domains.workspace.actions import _safe_title
from app.agent.errors import AgentToolError
from app.agent.models import Evidence, ToolResult
from app.repositories.media_subscriptions import (
    MediaSubscriptionError,
    normalize_subscription_interval,
    normalize_subscription_seasons,
    validate_subscription_scope,
)

from . import common

_ALLOWED_MONITOR_MODES = frozenset({"missing", "future", "selected"})
_ALLOWED_SUBSCRIPTION_ACTIONS = frozenset({"notify", "confirm", "auto"})
_ALLOWED_DOWNLOAD_TARGETS = frozenset({"qb", "guangya", "both"})


def media_subscription_policy_update_arguments(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise AgentToolError("工具参数必须是 JSON 对象")
    allowed = {
        "subscription_id",
        "monitor_mode",
        "seasons",
        "include_specials",
        "action",
        "download_target",
        "check_interval_minutes",
    }
    if not set(arguments).issubset(allowed) or "subscription_id" not in arguments:
        raise AgentToolError("媒体追更策略参数无效")
    normalized: dict[str, Any] = {
        "subscription_id": common._strict_subscription_id(arguments.get("subscription_id"))
    }
    if len(arguments) == 1:
        raise AgentToolError("至少需要提供一个要修改的追更策略字段")
    if "monitor_mode" in arguments:
        mode = str(arguments.get("monitor_mode") or "").strip().lower()
        if mode not in _ALLOWED_MONITOR_MODES:
            raise AgentToolError("monitor_mode 仅支持 missing、future 或 selected")
        normalized["monitor_mode"] = mode
    if "seasons" in arguments:
        raw_seasons = arguments.get("seasons")
        if not isinstance(raw_seasons, list) or len(raw_seasons) > 20:
            raise AgentToolError("seasons 必须是最多 20 项的季度数组")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in raw_seasons):
            raise AgentToolError("季度编号必须是 1 到 100 的整数")
        try:
            normalized["seasons"] = normalize_subscription_seasons(raw_seasons)
        except MediaSubscriptionError as exc:
            raise AgentToolError(str(exc)) from exc
    if "include_specials" in arguments:
        value = arguments.get("include_specials")
        if not isinstance(value, bool):
            raise AgentToolError("include_specials 必须是布尔值")
        normalized["include_specials"] = value
    if "action" in arguments:
        action = str(arguments.get("action") or "").strip().lower()
        if action not in _ALLOWED_SUBSCRIPTION_ACTIONS:
            raise AgentToolError("action 仅支持 notify、confirm 或 auto")
        normalized["action"] = action
    if "download_target" in arguments:
        target = str(arguments.get("download_target") or "").strip().lower()
        if target not in _ALLOWED_DOWNLOAD_TARGETS:
            raise AgentToolError("download_target 仅支持 qb、guangya 或 both")
        normalized["download_target"] = target
    if "check_interval_minutes" in arguments:
        interval = arguments.get("check_interval_minutes")
        if isinstance(interval, bool) or not isinstance(interval, int):
            raise AgentToolError("check_interval_minutes 必须是 5 到 10080 的整数")
        try:
            normalized["check_interval_minutes"] = normalize_subscription_interval(interval)
        except MediaSubscriptionError as exc:
            raise AgentToolError(str(exc)) from exc
    return normalized


def _json_int_list(value: Any) -> list[int]:
    try:
        return normalize_subscription_seasons(json.loads(str(value or "[]")))
    except (TypeError, ValueError):
        return []


def _policy_projection(row: Any) -> dict[str, Any]:
    return {
        "subscription_number": int(row["id"]),
        "title": _safe_title(row["title"], fallback="未命名媒体"),
        "media_type": str(row["media_type"] or ""),
        "enabled": bool(row["enabled"]),
        "monitor_mode": str(row["monitor_mode"] or "missing"),
        "seasons": _json_int_list(row["seasons_json"]),
        "include_specials": bool(row["include_specials"]),
        "action": str(row["action"] or "confirm"),
        "download_target": str(row["download_target"] or "guangya"),
        "check_interval_minutes": max(
            5, min(int(row["check_interval_minutes"] or 4320), 10080)
        ),
        "site_count": len(_json_string_list(row["sites_json"])),
    }


def _json_string_list(value: Any) -> list[str]:
    try:
        raw = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    return [item[:32] for item in raw[:16] if isinstance(item, str) and item]


def get_media_subscription_policy(arguments: dict[str, int]) -> ToolResult:
    row = db.get_media_subscription(arguments["subscription_id"])
    if row is None:
        raise AgentToolError("未找到指定的媒体追更订阅", code="precondition_failed")
    policy = _policy_projection(row)
    return ToolResult(
        True,
        "completed",
        f"已读取《{policy['title']}》的追更策略",
        data=policy,
        evidence=[
            Evidence(
                "media_subscription_database",
                "只读取追更范围、动作模式、下载目标和检查周期；不返回站点明细或凭据。",
                common._now(),
            )
        ],
        suggestions=[
            f"如需修改，请明确说媒体订阅 {policy['subscription_number']} 要改哪些策略。"
        ],
    )


def _policy_snapshot(conn: Any, subscription_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT id,title,media_type,enabled,monitor_mode,seasons_json,include_specials,"
        "action,download_target,sites_json,check_interval_minutes,status,revision,updated_at "
        "FROM media_subscriptions WHERE id=? AND deleted_at IS NULL",
        (subscription_id,),
    ).fetchone()
    if row is None:
        return {"exists": False, "subscription_id": subscription_id}
    return {
        "exists": True,
        **_policy_projection(row),
        "status": str(row["status"] or ""),
        "revision": int(row["revision"] or 0),
        "updated_at": str(row["updated_at"] or ""),
    }


def _validate_effective_subscription_policy(
    snapshot: dict[str, Any], normalized: dict[str, Any]
) -> dict[str, Any]:
    effective = {
        "monitor_mode": normalized.get(
            "monitor_mode", snapshot.get("monitor_mode", "missing")
        ),
        "seasons": list(normalized.get("seasons", snapshot.get("seasons", [])) or []),
        "include_specials": bool(
            normalized.get("include_specials", snapshot.get("include_specials", False))
        ),
        "action": normalized.get("action", snapshot.get("action", "confirm")),
        "download_target": normalized.get(
            "download_target", snapshot.get("download_target", "guangya")
        ),
        "check_interval_minutes": normalized.get(
            "check_interval_minutes", snapshot.get("check_interval_minutes", 4320)
        ),
    }
    try:
        validate_subscription_scope(
            str(snapshot.get("media_type") or ""), effective["monitor_mode"],
            effective["seasons"], effective["include_specials"],
        )
    except MediaSubscriptionError as exc:
        raise AgentToolError(str(exc), code="precondition_failed") from exc
    return effective


def _active_admission_state(conn: Any, subscription_id: int) -> dict[str, Any]:
    """生成策略确认窗口内 active admission 的有界公开计数与内部状态摘要。"""
    rows = conn.execute(
        "SELECT id,subscription_revision,status,updated_at "
        "FROM media_download_admissions WHERE subscription_id=? "
        "AND status IN ('claimed','dispatching') ORDER BY id",
        (subscription_id,),
    ).fetchall()
    digest = hashlib.sha256()
    counts = {"claimed": 0, "dispatching": 0}
    for row in rows:
        status = str(row["status"] or "")
        if status in counts:
            counts[status] += 1
        digest.update(
            (
                f"{int(row['id'])}:{int(row['subscription_revision'] or 0)}:"
                f"{status}:{row['updated_at'] or ''!s}\n"
            ).encode()
        )
    return {
        **counts,
        "total": len(rows),
        "state_digest": digest.hexdigest(),
    }


def _policy_confirmation_snapshot(conn: Any, subscription_id: int) -> dict[str, Any]:
    snapshot = _policy_snapshot(conn, subscription_id)
    if snapshot.get("exists"):
        snapshot["_active_admissions"] = _active_admission_state(conn, subscription_id)
    return snapshot


def _policy_preview_effects(
    *, snapshot: dict[str, Any], effective: dict[str, Any], dispatching: int
) -> list[str]:
    effects = [
        "保存后会使旧候选和未提交准入失效",
        "本操作不会主动检查更新或创建新的下载提交",
    ]
    if effective["action"] == "auto":
        target = {
            "qb": "qBittorrent",
            "guangya": "光鸭",
            "both": "qBittorrent 和光鸭",
        }.get(str(effective["download_target"]), "所选下载目标")
        effects.append(f"后续调度确认匹配后会无需再次人工确认，自动提交到{target}")
    else:
        effects.append("后续匹配资源仍会按通知或人工确认模式处理")
    if snapshot.get("action") == "auto" or dispatching:
        effects.append(
            f"预检时有 {dispatching} 个任务已进入提交阶段；这些任务无法由本次策略修改撤回"
        )
    return effects


def prepare_set_media_subscription_policy(
    arguments: dict[str, Any],
) -> tuple[ToolResult, str]:
    normalized = media_subscription_policy_update_arguments(arguments)
    subscription_id = normalized["subscription_id"]
    with db.get_conn() as conn:
        snapshot = _policy_confirmation_snapshot(conn, subscription_id)
        active_admissions = snapshot.get("_active_admissions", {})
        dispatching = int(active_admissions.get("dispatching") or 0)
        claimed = int(active_admissions.get("claimed") or 0)
    if not snapshot.get("exists"):
        raise AgentToolError("未找到指定的媒体追更订阅", code="precondition_failed")
    effective = _validate_effective_subscription_policy(snapshot, normalized)
    requested = {
        key: value for key, value in normalized.items() if key != "subscription_id"
    }
    preview = ToolResult(
        True,
        "confirmation_required",
        f"确认后将修改《{snapshot['title']}》的追更策略",
        data={
            "subscription_number": subscription_id,
            "title": snapshot["title"],
            "current": {key: snapshot.get(key) for key in requested},
            "requested": requested,
            "effective": effective,
            "in_flight_dispatches_at_preflight": dispatching,
            "pending_admissions_at_preflight": claimed,
            "effects": _policy_preview_effects(
                snapshot=snapshot,
                effective=effective,
                dispatching=dispatching,
            ),
        },
        evidence=[
            Evidence(
                "media_subscription_database",
                "已读取订阅修订号和当前安全策略，尚未修改数据库。",
                common._now(),
            )
        ],
        suggestions=["确认票据只可使用一次；策略变化后需要重新预检。"],
    )
    return preview, common._fingerprint(snapshot)


def set_media_subscription_policy_confirmed(
    arguments: dict[str, Any], expected_context: str
) -> ToolResult:
    normalized = media_subscription_policy_update_arguments(arguments)
    subscription_id = normalized.pop("subscription_id")
    with db.get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        snapshot = _policy_confirmation_snapshot(conn, subscription_id)
        if not snapshot.get("exists") or not secrets.compare_digest(
            common._fingerprint(snapshot), (expected_context or "")
        ):
            return ToolResult(
                False,
                "conflict",
                "追更策略已变化，请重新预检",
                error="确认快照已失效。",
            )
        try:
            effective = _validate_effective_subscription_policy(snapshot, normalized)
        except AgentToolError as exc:
            return ToolResult(
                False, "conflict", str(exc), error=str(exc)
            )
        dispatching = int(
            snapshot.get("_active_admissions", {}).get("dispatching") or 0
        )
        fields: dict[str, Any] = {}
        for key, value in normalized.items():
            fields["seasons_json" if key == "seasons" else key] = (
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if key == "seasons"
                else int(value)
                if key == "include_specials"
                else value
            )
        changed = db.update_media_subscription_config(
            subscription_id, connection=conn, reason="订阅策略已变更",
            run_summary="订阅策略已变更，旧检查已取消", **fields,
        )
        assert changed is not None  # 同一事务已核对存在与确认快照。
    refreshed = common._reload_scheduler()
    return ToolResult(
        True,
        "completed",
        "媒体追更策略已更新",
        data={
            "subscription_number": subscription_id,
            "updated_fields": sorted(normalized),
            "expired_candidates": changed["expired_candidates"],
            "in_flight_dispatches": dispatching,
            "effective_action": effective["action"],
            "runtime_refreshed": refreshed,
        },
        evidence=[
            Evidence(
                "media_subscription_database",
                "使用一次性确认票据和修订号原子更新策略；未立即检查或下载。",
                common._now(),
            )
        ],
        suggestions=[
            "策略已保存，下一次调度会按新策略重新检查。",
            *(
                [f"仍有 {dispatching} 个已进入提交阶段的旧任务无法撤回。"]
                if dispatching
                else []
            ),
        ],
    )
