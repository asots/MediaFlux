"""光鸭账号容量与连接资料的最小公开投影。"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import httpx

from app.agent.errors import AgentToolError
from app.agent.models import Evidence, ToolResult
from app.agent.public_safety import sanitize_public_text
from app.clients.guangya import GuangYaClient, close_guangya_client

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def guangya_account_status_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict) or arguments:
        raise AgentToolError("光鸭账号状态查询不接受参数")
    return {}


def _payloads(value: object) -> Iterable[dict[str, Any]]:
    if not isinstance(value, dict):
        return ()
    rows = [value]
    for key in ("data", "user", "userInfo", "user_info", "storage", "space"):
        nested = value.get(key)
        if isinstance(nested, dict):
            rows.append(nested)
            for child_key in ("user", "userInfo", "storage", "space"):
                child = nested.get(child_key)
                if isinstance(child, dict):
                    rows.append(child)
    return rows


def _first(payloads: Iterable[dict[str, Any]], *keys: str) -> object:
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if value not in (None, ""):
                return value
    return ""


def _bytes(payloads: list[dict[str, Any]], *keys: str) -> int | None:
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                continue
            # 字节数必须是非负整数；字符串直接转 int，避免 float 丢失大整数精度。
            if isinstance(value, float) and not value.is_integer():
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            # 只接受有意义的 64 位字节计数，异常巨数不能让利用率计算溢出。
            if 0 <= parsed <= (1 << 63) - 1:
                return parsed
    return None


def _mask_phone(value: object) -> str:
    raw = "".join(character for character in str(value or "") if character.isdigit())
    if len(raw) < 7:
        return ""
    return raw[:3] + "****" + raw[-4:]


def _mask_email(value: object) -> str:
    text = str(value or "").strip()
    local, separator, domain = text.partition("@")
    if not separator or not local or not domain:
        return ""
    return local[:1] + "***@" + domain[:120]


def get_guangya_account_status(_arguments: dict[str, Any]) -> ToolResult:
    client: GuangYaClient | None = None
    payloads: list[dict[str, Any]] = []
    storage_payloads: list[dict[str, Any]] = []
    profile_available = False
    storage_available = False
    try:
        client = GuangYaClient()
        if not client.logged_in:
            raise AgentToolError("光鸭账号尚未连接", code="precondition_failed")
        # 容量与身份是两个独立接口。身份读取失败不应丢弃已取得的真实容量。
        try:
            storage_payloads = list(_payloads(client.account_storage_info()))
            storage_available = True
        except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("Agent 光鸭容量读取失败 type=%s", type(exc).__name__)
        try:
            payloads = list(_payloads(client.account_info()))
            profile_available = True
        except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("Agent 光鸭账号资料读取失败 type=%s", type(exc).__name__)
        if not storage_available and not profile_available:
            raise AgentToolError("光鸭账号及容量信息读取失败，当前连接状态无法确认", code="unavailable")
    except AgentToolError:
        raise
    except Exception as exc:
        logger.warning("Agent 光鸭账号资料读取失败 type=%s", type(exc).__name__)
        raise AgentToolError("光鸭账号状态当前不可用", code="unavailable") from exc
    finally:
        close_guangya_client(client)

    display_name = sanitize_public_text(
        _first(payloads, "nickname", "nickName", "username", "userName", "name"),
        limit=80,
    )
    phone = _mask_phone(
        _first(payloads, "phone_number", "phone", "phoneNumber", "mobile", "mobilePhone")
    )
    email = _mask_email(_first(payloads, "email", "emailAddress"))
    total = _bytes(
        storage_payloads,
        "totalSpaceSize",
        "totalSpace",
        "totalSize",
        "storageTotal",
        "total_capacity",
        "capacity",
        "quota",
    )
    used = _bytes(
        storage_payloads,
        "usedSpaceSize",
        "usedSpace",
        "usedSize",
        "storageUsed",
        "used_capacity",
        "useSpace",
    )
    available = _bytes(
        storage_payloads,
        "availableSpace",
        "freeSpace",
        "storageFree",
        "available_capacity",
        "remainSpace",
    )
    if available is None and total is not None and used is not None:
        available = max(0, total - used)
    if used is None and total is not None and available is not None and available <= total:
        used = total - available
    utilization = (
        round(used / total, 4)
        if total and used is not None
        else None
    )
    reported = any(value is not None for value in (total, used, available))
    complete = all(value is not None for value in (total, used, available))
    if not storage_available:
        storage_status = "unavailable"
        summary = "光鸭账号已连接，但容量接口读取失败，暂时无法确认剩余空间"
    elif not reported:
        storage_status = "not_reported"
        summary = "光鸭容量接口已响应，但本次未取得有效容量字段"
    else:
        storage_status = "ok" if complete else "partial"
        summary = "光鸭账号已连接，并已读取容量信息" if complete else "光鸭账号已连接，仅取得部分容量信息"
    data = {
        "connected": True,
        "profile_available": profile_available,
        "display_name": display_name,
        "masked_phone": phone,
        "masked_email": email,
        "storage": {
            "total_bytes": total,
            "used_bytes": used,
            "available_bytes": available,
            "utilization": utilization,
            "reported": reported,
            "status": storage_status,
        },
    }
    return ToolResult(
        True,
        "ok" if complete else "partial",
        summary,
        data=data,
        model_data={
            "connected": True,
            "storage": data["storage"],
            "identity_available": bool(display_name or phone or email),
        },
        evidence=[
            Evidence(
                "guangya_account",
                "容量读取自光鸭 assets 接口，单位为字节；账号资料独立读取。仅投影白名单字段，未返回用户 ID、Token 或原始响应。",
                _now(),
            )
        ],
    )
