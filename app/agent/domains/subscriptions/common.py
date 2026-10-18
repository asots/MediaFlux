"""媒体订阅共享的参数、确认指纹与调度刷新。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from app.agent.errors import AgentToolError
from app.logger import get_logger

logger = get_logger(__name__)
_MAX_SAFE_ID = 2_147_483_647


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _strict_subscription_id(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > _MAX_SAFE_ID
    ):
        raise AgentToolError("subscription_id 必须是正整数")
    return value


def media_subscription_summary_arguments(arguments: dict[str, Any]) -> dict[str, int]:
    if not isinstance(arguments, dict):
        raise AgentToolError("工具参数必须是 JSON 对象")
    if set(arguments) != {"subscription_id"}:
        raise AgentToolError(
            "media.get_subscription_summary 只接受 subscription_id 参数"
        )
    return {
        "subscription_id": _strict_subscription_id(arguments.get("subscription_id"))
    }


def _media_type_label(value: Any) -> str:
    return "电影" if str(value or "").strip().lower() == "movie" else "剧集"


def _fingerprint(state: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _reload_scheduler() -> bool:
    try:
        from app.modules.media_subscription_scheduler import (
            get_media_subscription_scheduler,
        )

        get_media_subscription_scheduler().reload()
        return True
    except Exception as exc:  # noqa: BLE001 - 配置已保存，刷新失败不回滚持久结果
        logger.warning(
            "Agent 媒体追更配置已保存但调度器刷新失败 type=%s", type(exc).__name__
        )
        return False
