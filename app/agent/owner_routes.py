"""Agent owner 的低层、安全通知路由解析。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app import config

_TELEGRAM_OWNER_RE = re.compile(r"^tg:v1:(-?[1-9][0-9]*)\x1f([1-9][0-9]*)$")

_TELEGRAM_CHAT_RE = re.compile(r"^-?[1-9][0-9]*$")
_TELEGRAM_THREAD_RE = re.compile(r"^[1-9][0-9]{0,18}$")


class TransportInputError(ValueError):
    """对外安全的入口参数错误。"""


def normalize_delivery_route(
    value: Mapping[str, Any], *, channel: str = "telegram", owner: str = "",
) -> dict[str, Any]:
    """校验服务端物理投递地址；HTTP 和模型上下文不是地址来源。"""
    if not isinstance(value, Mapping):
        raise TransportInputError("delivery_route 无效")
    if not value:
        return {}
    if channel != "telegram" or set(value) - {"chat_id", "message_thread_id"}:
        raise TransportInputError("delivery_route 仅允许 Telegram 服务端设置")
    chat = value.get("chat_id")
    if isinstance(chat, bool) or not _TELEGRAM_CHAT_RE.fullmatch(str(chat)):
        raise TransportInputError("delivery_route.chat_id 无效")
    target = str(int(str(chat)))
    if owner:
        identity = _TELEGRAM_OWNER_RE.fullmatch(owner)
        if identity is None or identity.group(1) != target:
            raise TransportInputError("delivery_route 不属于当前 Telegram owner")
    route: dict[str, Any] = {"chat_id": target}
    thread = value.get("message_thread_id")
    if thread is not None:
        if isinstance(thread, bool) or not _TELEGRAM_THREAD_RE.fullmatch(str(thread)):
            raise TransportInputError("delivery_route.message_thread_id 无效")
        route["message_thread_id"] = int(thread)
    return route

def web_kernel_owner(login_identity: object) -> str:
    """由已认证登录身份派生跨浏览器会话稳定、不可逆的 Web principal。"""
    normalized_owner = str(login_identity or "").strip()
    if not normalized_owner or len(normalized_owner) > 512:
        raise ValueError("Web Agent owner 无效")
    payload = b"mediaflux-agent-web-kernel:v1\0" + normalized_owner.encode("utf-8")
    return f"webk:v1:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class TelegramOwnerRoute:
    owner: str
    chat_id: str
    user_id: str


def parse_telegram_owner_route(value: object) -> TelegramOwnerRoute | None:
    """仅接受规范 Telegram owner；其它 owner 不得回退到全局 chat。"""
    owner = str(value or "")
    match = _TELEGRAM_OWNER_RE.fullmatch(owner)
    if match is None:
        return None
    chat_id, user_id = match.groups()
    return TelegramOwnerRoute(owner=owner, chat_id=chat_id, user_id=user_id)


def configured_telegram_user_ids() -> set[str]:
    """TG 输入、Kernel 与历史回执共用的实时白名单语法。"""
    return {
        part
        for part in re.split(
            r"[,;，；\s]+",
            (config.get("TG_AGENT_ALLOWED_USER_IDS", "") or "").strip(),
        )
        if re.fullmatch(r"[1-9][0-9]*", part)
    }


def telegram_owner_route_is_currently_authorized(
    value: object,
    *,
    chat_id: object = "",
) -> bool:
    """按当前 Agent/TG 配置重新授权持久化 owner，撤权后立即失效。"""
    route = (
        value
        if isinstance(value, TelegramOwnerRoute)
        else parse_telegram_owner_route(value)
    )
    if route is None:
        return False
    persisted_chat = str(chat_id or "").strip()
    if persisted_chat and persisted_chat != route.chat_id:
        return False
    if not config.get_bool("AGENT_ENABLED", False):
        return False
    if not config.get_bool("TG_AGENT_ENABLED", False):
        return False
    if (config.get("TG_CHAT_ID", "") or "").strip() != route.chat_id:
        return False
    return route.user_id in configured_telegram_user_ids()
