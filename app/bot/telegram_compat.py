"""Telegram 参数兼容与安全诊断；只复用现有单次投递，不承担重试。"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from app.notifier import (
    TelegramSendResult,
    call_telegram_delivery,
    telegram_edit_fallback_allowed,
)
from app.sensitive_data import redact_sensitive_text

_UNCHANGED_DESCRIPTIONS = frozenset({
    "bad request: message is not modified",
    (
        "bad request: message is not modified: specified new message content and "
        "reply markup are exactly the same as a current content and reply markup of the message"
    ),
})


def _accepts_keyword(sender: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(sender).parameters
    except (TypeError, ValueError):
        return False
    parameter = parameters.get(name)
    return bool(
        parameter is not None
        and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
        }
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    )


def telegram_message_options(
    sender: Callable[..., Any],
    telebot_module: Any = None,
    *,
    reply_to_message_id: object = None,
) -> dict[str, Any]:
    """发送前选择参数；绝不捕获一次发送的 TypeError 后改参数再次发送。

    4.15 已提供现代类型，但旧封装和轻量 fake 可能只接受旧签名。未传模块
    的现有消息适配器使用已安装 SDK；显式传入 fake 时不混入真实 SDK 类型。
    """
    if telebot_module is None:
        import telebot as telebot_module
    types = getattr(telebot_module, "types", None)
    options: dict[str, Any] = {}
    preview_type = getattr(types, "LinkPreviewOptions", None)
    if callable(preview_type) and _accepts_keyword(sender, "link_preview_options"):
        options["link_preview_options"] = preview_type(is_disabled=True)
    else:
        options["disable_web_page_preview"] = True
    if reply_to_message_id is not None:
        reply_type = getattr(types, "ReplyParameters", None)
        if callable(reply_type) and _accepts_keyword(sender, "reply_parameters"):
            options["reply_parameters"] = reply_type(message_id=int(reply_to_message_id))
        else:
            options["reply_to_message_id"] = reply_to_message_id
    return options


def call_telegram_edit(
    operation: Callable[[], Any], *, message_id: int,
) -> tuple[TelegramSendResult, Any | None]:
    """复用单次投递并收紧 no-op：只接受 Telegram 400 的两种规范描述。

    不采用异常字符串包含匹配，避免把正文回显、403/429 或网络异常误判
    为编辑成功；仍保留原错误供现有 fallback/retry/outcome_unknown 判定。
    """
    result, value = call_telegram_delivery(operation)
    if (
        not result.ok and result.status_code == 400
        and result.error.strip().casefold() in _UNCHANGED_DESCRIPTIONS
    ):
        return TelegramSendResult(ok=True, message_id=message_id), value
    return replace(result, message_id=result.message_id or message_id), value


def telegram_error_summary(result: TelegramSendResult) -> str:
    """只输出固定诊断白名单；通用凭据脱敏仍不能保护正文/地址/个人信息。"""
    description = redact_sensitive_text(result.error).casefold()
    status = result.status_code
    if status == 429:
        category, detail = "rate_limited", "请求限流，等待服务端指定间隔"
    elif status == 408:
        category, detail = "read_timeout", "响应读取超时，投递结果未知"
    elif description == "sslerror":
        category, detail = "tls_error", "TLS/SSL 连接异常，投递结果未知"
    elif description == "connecttimeout":
        category, detail = "connect_timeout", "建立连接超时"
    elif status == 401:
        category, detail = "unauthorized", "Telegram 身份验证失败"
    elif status == 403:
        category, detail = "forbidden", "Telegram 拒绝访问目标会话"
    elif status == 400 and description.startswith("bad request: can't parse entities"):
        category, detail = "invalid_format", "Telegram 无法解析消息格式"
    elif telegram_edit_fallback_allowed(result):
        category, detail = "edit_rejected", "目标消息不存在或不可编辑"
    elif status >= 500:
        category, detail = "server_error", "Telegram 服务暂不可用"
    elif status >= 400:
        category, detail = "api_rejected", "Telegram API 拒绝请求"
    else:
        category, detail = "outcome_unknown", "传输失败，投递结果未知"
    return (
        f"status={status or '-'} category={category} "
        f"retry_after={result.retry_after_seconds or '-'} detail={detail}"
    )
