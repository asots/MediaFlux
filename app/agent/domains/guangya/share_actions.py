"""光鸭自有分享的安全查询、创建与撤销动作。"""

from __future__ import annotations

import logging
import re
import time
from copy import deepcopy
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app.agent.confirmation import confirmation_context_fingerprint
from app.agent.domains.guangya.workspace_actions import (
    GuangyaWorkspaceActions,
    bind_observation_source,
    reobserved_selection_note,
    validated_reobservation_source,
)
from app.agent.errors import AgentToolError
from app.agent.models import Evidence, ToolContext, ToolReference, ToolResult
from app.agent.public_safety import (
    sanitize_public_text,
    sanitize_resource_title,
    sanitize_untrusted_filename,
)
from app.clients.guangya import (
    GuangYaClient,
    GuangYaFile,
    GuangYaWriteRejected,
    close_guangya_client,
)
from app.modules.guangya_workspace import (
    GuangYaWorkspaceError,
    GuangYaWorkspaceStale,
    capture_observation_source,
    load_directory_observation,
    observation_entry_map,
    valid_object_handle,
    valid_observation_ref,
)

logger = logging.getLogger(__name__)
_MAX_SHARES = 2_000
_MAX_SHARE_OBJECTS = 100
_CODE_RE = re.compile(r"^[A-Za-z0-9]{4,16}$")
_SHARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,256}$")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _bounded_int(value: Any, *, maximum: int = 1 << 63) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, min(int(float(value or 0)), maximum))
    except (TypeError, ValueError, OverflowError):
        return 0


def _open_client() -> GuangYaClient:
    client = GuangYaClient()
    if not client.logged_in:
        client.close()
        raise AgentToolError("光鸭账号尚未连接", code="precondition_failed")
    return client


def _public_error(exc: Exception, *, fallback: str) -> AgentToolError:
    if isinstance(exc, AgentToolError):
        return exc
    if isinstance(exc, (GuangYaWorkspaceStale,)):
        return AgentToolError(str(exc), code="confirmation_stale")
    if isinstance(exc, (GuangYaWorkspaceError, ValueError)):
        return AgentToolError(str(exc), code="precondition_failed")
    if isinstance(exc, GuangYaWriteRejected):
        return AgentToolError(
            exc.public_message or fallback,
            code="provider_rejected",
        )
    logger.warning("Agent 光鸭分享操作失败 type=%s", type(exc).__name__)
    return AgentToolError(fallback, code="unavailable")


def _first(raw: dict[str, Any], *keys: str) -> object:
    payloads = [raw]
    data = raw.get("data")
    if isinstance(data, dict):
        payloads.append(data)
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if value not in (None, ""):
                return value
    return ""


def _share_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    # 列表 id 是撤销接口的管理记录 ID；shareId 只用于公开分享链接。
    share_id = str(_first(raw, "id") or "").strip()
    return {
        "share_id": share_id,
        "title": str(_first(raw, "title", "shareName", "name") or "").strip(),
        "status": str(_first(raw, "status", "shareStatus", "state") or "").strip(),
        "created_at": str(
            _first(raw, "createTime", "createdAt", "created_at", "shareTime") or ""
        ).strip(),
        "expires_at": str(
            _first(raw, "expireTime", "expiresAt", "expiredAt", "invalidTime") or ""
        ).strip(),
        "file_count": _bounded_int(
            _first(raw, "fileCount", "resCount", "count", "resourceCount"),
            maximum=1_000_000,
        ),
        "updated_at": str(
            _first(raw, "updateTime", "updatedAt", "updated_at") or ""
        ).strip(),
    }


def _public_share(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    title = sanitize_resource_title(item.get("title"), limit=160)
    return {
        "index": index,
        "title": title or "未命名分享",
        "status": sanitize_public_text(item.get("status"), limit=40) or "unknown",
        "created_at": sanitize_public_text(item.get("created_at"), limit=60),
        "expires_at": sanitize_public_text(item.get("expires_at"), limit=60),
        "file_count": _bounded_int(item.get("file_count"), maximum=1_000_000),
    }


def _load_share_snapshots(client: GuangYaClient) -> list[dict[str, Any]]:
    snapshots = [
        _share_snapshot(item)
        for item in client.list_user_shares(max_items=_MAX_SHARES)
        if isinstance(item, dict)
    ]
    if any(not item["share_id"] for item in snapshots):
        raise AgentToolError("光鸭分享列表缺少对象标识，无法确认完整快照", code="unavailable")
    return snapshots


def guangya_share_list_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict) or set(arguments) - {"operation", "page", "page_size"}:
        raise AgentToolError("光鸭分享列表参数无效")
    operation = arguments.get("operation", "list")
    if not isinstance(operation, str) or operation not in {"list", "reobserve"}:
        raise AgentToolError("operation 只能是 list 或 reobserve")
    if operation == "reobserve":
        if set(arguments) != {"operation"}:
            raise AgentToolError("重新观察由原选中分享的稳定标识定位，不能附加旧页号或序号")
        return {"operation": "reobserve"}
    page = arguments.get("page", 1)
    page_size = arguments.get("page_size", 50)
    if type(page) is not int or not 1 <= page <= 200:
        raise AgentToolError("page 必须在 1 到 200 之间")
    if type(page_size) is not int or not 1 <= page_size <= 100:
        raise AgentToolError("page_size 必须在 1 到 100 之间")
    return {"page": page, "page_size": page_size}


def list_guangya_user_shares(
    arguments: dict[str, Any], _context: ToolContext
) -> ToolResult:
    source = (validated_reobservation_source(_context, kind="guangya_shares", tool="guangya.share.list")
              if arguments.get("operation") == "reobserve" else None)
    client: GuangYaClient | None = None
    try:
        client = _open_client()
        if source is not None and (client.credential_generation) != source["credential_generation"]:
            raise AgentToolError("光鸭账号凭据已变化，不能沿用旧分享选择", code="observation_scope_changed")
        snapshots = _load_share_snapshots(client)
        generation = (client.credential_generation)
        if source is not None and generation != source["credential_generation"]:
            raise AgentToolError("读取期间账号凭据已变化，未发布新的分享引用", code="observation_scope_changed")
    except Exception as exc:
        raise _public_error(exc, fallback="光鸭分享列表当前不可用") from exc
    finally:
        if client is not None:
            close_guangya_client(client)

    comparison = None
    if source is not None:
        by_id: dict[str, list[dict[str, Any]]] = {}
        for item in snapshots:
            by_id.setdefault(item["share_id"], []).append(item)
        selected, compared = [], []
        for old in source["baseline"]["selected"]:
            matches = by_id.get(str(old.get("share_id") or ""), [])
            row = {"previous_index": old["old_index"], "title": sanitize_resource_title(old.get("title"), limit=160)}
            if len(matches) == 1:
                current = matches[0]
                selected.append(current)
                row.update(state="same_identity", index=len(selected),
                           title=sanitize_resource_title(current.get("title"), limit=160),
                           changed_fields=[key for key in ("title", "status", "updated_at", "expires_at", "file_count")
                                           if old.get(key) != current.get(key)])
            else:
                row["state"] = "ambiguous" if len(matches) > 1 else "not_observed"
            compared.append(row)
        comparison = {"basis": "provider_identity", "items": compared,
                      "complete": all(row["state"] == "same_identity" for row in compared),
                      "read_limit": _MAX_SHARES, "possible_truncation": len(snapshots) >= _MAX_SHARES,
                      "note": "只对应原选中的分享标识；未找到的项不按同名或旧序号替换，新撤销仍须新确认。"}
        page, page_size, start = 1, max(1, len(selected)), 0
        total, has_more = len(selected), False
    else:
        page = int(arguments["page"])
        page_size = int(arguments["page_size"])
        start = (page - 1) * page_size
        selected = snapshots[start : start + page_size]
        total, has_more = len(snapshots), start + len(selected) < len(snapshots)
    public_items = [
        _public_share(item, index=start + index)
        for index, item in enumerate(selected, 1)
    ]
    return ToolResult(
        True,
        "found" if selected else "empty",
        (f"已重新核对原选中分享，{len(selected)} 条取得稳定标识对应" if source is not None
         else f"当前账号共有 {len(snapshots)} 条光鸭分享，本页显示 {len(selected)} 条"),
        data={
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_more": has_more,
            "items": public_items,
            **({"reobservation": comparison} if comparison is not None else {}),
        },
        model_data={
            "page": page,
            "total": total,
            "has_more": has_more,
            "items": public_items,
            **({"reobservation": comparison} if comparison is not None else {}),
        },
        references=[
            ToolReference(
                kind="guangya_shares",
                value={
                    "credential_generation": generation,
                    "observed_at": _now(),
                    "page": page,
                    "page_size": page_size,
                    "items": selected,
                },
                ttl_seconds=15 * 60,
            )
        ] if selected else [],
        evidence=[
            Evidence(
                "guangya_share_list",
                "仅返回分享标题、状态和时间摘要；分享 ID、访问码及底层响应保存在会话私有引用中。",
                _now(),
            )
        ],
        suggestions=["撤销分享需使用本次返回的分享引用和 index，并经过人工确认。"],
    )


def guangya_share_create_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise AgentToolError("创建光鸭分享参数必须是对象")
    allowed = {
        "observation_ref",
        "object_refs",
        "title",
        "expires_days",
        "access_code",
        "auto_access_code",
        "max_restore_count",
        "allow_download",
    }
    if set(arguments) - allowed:
        raise AgentToolError("创建光鸭分享包含不支持的参数")
    observation_ref = str(arguments.get("observation_ref") or "").strip().upper()
    if observation_ref and not valid_observation_ref(observation_ref):
        raise AgentToolError("observation_ref 格式无效")
    raw_refs = arguments.get("object_refs")
    if not isinstance(raw_refs, list) or not 1 <= len(raw_refs) <= _MAX_SHARE_OBJECTS:
        raise AgentToolError(f"object_refs 必须包含 1 到 {_MAX_SHARE_OBJECTS} 个对象")
    object_refs: list[str] = []
    for value in raw_refs:
        ref = str(value or "").strip().upper()
        if not valid_object_handle(ref):
            raise AgentToolError("object_refs 包含无效对象引用")
        if ref not in object_refs:
            object_refs.append(ref)
    title = str(arguments.get("title") or "").strip()
    if len(title) > 180:
        raise AgentToolError("分享标题不能超过 180 个字符")
    expires_days = arguments.get("expires_days", 0)
    if type(expires_days) is not int or not 0 <= expires_days <= 3650:
        raise AgentToolError("expires_days 必须在 0 到 3650 之间")
    access_code = str(arguments.get("access_code") or "").strip()
    if access_code and not _CODE_RE.fullmatch(access_code):
        raise AgentToolError("access_code 必须是 4 到 16 位字母或数字")
    auto_access_code = arguments.get("auto_access_code", not bool(access_code))
    if type(auto_access_code) is not bool:
        raise AgentToolError("auto_access_code 必须是布尔值")
    if access_code:
        auto_access_code = False
    max_restore_count = arguments.get("max_restore_count", 0)
    if type(max_restore_count) is not int or not 0 <= max_restore_count <= 1_000_000:
        raise AgentToolError("max_restore_count 必须在 0 到 1000000 之间")
    allow_download = arguments.get("allow_download", True)
    if type(allow_download) is not bool:
        raise AgentToolError("allow_download 必须是布尔值")
    return {
        "observation_ref": observation_ref,
        "object_refs": object_refs,
        "title": title,
        "expires_days": expires_days,
        "access_code": access_code,
        "auto_access_code": auto_access_code,
        "max_restore_count": max_restore_count,
        "allow_download": allow_download,
    }


def _workspace_selection(
    arguments: dict[str, Any], *, owner: str, session_id: str, verify_live: bool, workspace_actions: GuangyaWorkspaceActions
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if not owner.strip() or not session_id.strip():
        raise AgentToolError("光鸭分享需要已登录会话", code="precondition_failed")
    observation_ref = str(arguments.get("observation_ref") or "").strip().upper()
    if not observation_ref:
        observation_ref = workspace_actions.latest_guangya_observation_ref(owner, session_id)
    if not observation_ref:
        raise AgentToolError("请先读取要分享的光鸭目录", code="precondition_failed")
    observation = load_directory_observation(observation_ref, owner=owner, session_id=session_id)
    entries = observation_entry_map(observation)
    selected: list[dict[str, Any]] = []
    for ref in arguments.get("object_refs") or ():
        item = entries.get(str(ref).strip().upper())
        if item is None:
            raise AgentToolError(
                "分享对象不属于当前目录快照，请重新查询",
                code="confirmation_stale",
            )
        selected.append(item)
    client: GuangYaClient | None = None
    try:
        client = _open_client()
        expected_generation = _bounded_int(observation.get("credential_generation"))
        if (client.credential_generation) != expected_generation:
            raise AgentToolError(
                "光鸭登录凭据已变化，请重新读取目录",
                code="confirmation_stale",
            )
        if verify_live:
            for expected in selected:
                current = client.file_info(str(expected.get("file_id") or ""))
                if not _file_matches(current, expected):
                    raise AgentToolError(
                        "分享对象已变化，请重新读取目录",
                        code="confirmation_stale",
                    )
    finally:
        if client is not None:
            close_guangya_client(client)
    return observation, selected, {
        "observation_ref": observation_ref,
        "credential_generation": _bounded_int(observation.get("credential_generation")),
    }


def _file_matches(current: GuangYaFile | None, expected: dict[str, Any]) -> bool:
    if current is None:
        return False
    if (
        (current.file_id) != str(expected.get("file_id") or "")
        or (current.name) != str(expected.get("name") or "")
        or (current.is_dir) != bool(expected.get("is_dir"))
        or (current.parent_id) != str(expected.get("parent_id") or "0")
        or _bounded_int(current.size) != _bounded_int(expected.get("size"))
    ):
        return False
    etag = str(expected.get("etag") or "")
    return not etag or (current.etag or "") in {"", etag}


def _source_fields(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    # 私有来源也只保留比较/定位白名单，不整份复制可能含访问码或链接的 payload。
    return {key: deepcopy(value[key]) for key in keys if key in value}


def _share_reobservation_source(
    arguments: dict[str, Any], context: ToolContext,
    collection: dict[str, Any], selected: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = _source_fields(collection, (
        "page", "page_size", "observed_at", "created_at", "created_at_epoch",
        "expires_at", "expires_at_epoch",
    ))
    baseline.update(
        # Kernel resolve 会移除 *_ref；原句柄仍在同一 EffectPlan.arguments 中。
        # 不绕读仓储或伪造 ref，父冻结层可按 ref_argument 补齐该已有参数。
        ref=arguments.get("guangya_shares_ref"), ref_argument="guangya_shares_ref",
        selected=[{
            **_source_fields(item, (
                "share_id", "title", "status", "created_at", "expires_at", "updated_at", "file_count",
            )),
            "old_index": index,
        } for index, item in zip(arguments.get("indices") or (), selected)],
    )
    read = {"tool": "guangya.share.list", "arguments": {
        "page": collection["page"], "page_size": collection["page_size"],
    }} if collection.get("page") and collection.get("page_size") else None
    return {
        "version": 1, "kind": "guangya_shares", "owner": context.owner,
        "session_id": context.session_id,
        "credential_generation": _bounded_int(collection.get("credential_generation")),
        "available": read is not None, "read": read, "baseline": baseline,
    }


def _create_snapshot(
    arguments: dict[str, Any], *, owner: str, session_id: str, verify_live: bool, workspace_actions: GuangyaWorkspaceActions
) -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    observation, selected, identity = _workspace_selection(
        arguments, owner=owner, session_id=session_id, verify_live=verify_live, workspace_actions=workspace_actions
    )
    title = str(arguments.get("title") or "").strip()
    if not title:
        title = str(selected[0].get("name") or "MediaFlux 分享")
        if len(selected) > 1:
            title = f"{title} 等 {len(selected)} 项"
    title = sanitize_untrusted_filename(title, limit=180) or "MediaFlux 分享"
    safe = {
        # 只用于实际写客户端校验；公开预览和结果会移除该内部字段。
        "credential_generation": identity["credential_generation"],
        "count": len(selected),
        "title": title,
        "expires_days": int(arguments.get("expires_days") or 0),
        "access_policy": (
            "custom_code"
            if arguments.get("access_code")
            else "auto_code"
            if arguments.get("auto_access_code")
            else "no_code"
        ),
        "max_restore_count": int(arguments.get("max_restore_count") or 0),
        "allow_download": bool(arguments.get("allow_download")),
        "samples": [
            sanitize_untrusted_filename(item.get("name")) or "未命名对象"
            for item in selected[:6]
        ],
    }
    fingerprint = confirmation_context_fingerprint(
        {
            **identity,
            "objects": [
                {
                    key: item.get(key)
                    for key in (
                        "file_id",
                        "parent_id",
                        "name",
                        "is_dir",
                        "size",
                        "etag",
                        "updated_at",
                    )
                }
                for item in selected
            ],
            "policy": {
                "title": title,
                "expires_days": safe["expires_days"],
                "access_code": str(arguments.get("access_code") or ""),
                "auto_access_code": bool(arguments.get("auto_access_code")),
                "max_restore_count": safe["max_restore_count"],
                "allow_download": safe["allow_download"],
            },
        },
        domain="guangya-share-create",
    )
    return safe, fingerprint, selected, observation, identity


def prepare_create_guangya_share(
    arguments: dict[str, Any], context: ToolContext, *, workspace_actions: GuangyaWorkspaceActions
) -> tuple[ToolResult, str]:
    try:
        safe, fingerprint, selected, observation, identity = _create_snapshot(
            arguments, owner=context.owner, session_id=context.session_id, verify_live=True, workspace_actions=workspace_actions
        )
        note = reobserved_selection_note(context, kind="guangya_files", generation=identity["credential_generation"],
                                         selected=selected, arguments=arguments)
        safe.pop("credential_generation")
    except Exception as exc:
        raise _public_error(exc, fallback="光鸭分享创建预检失败") from exc
    return ToolResult(
        True,
        "confirmation_required",
        f"确认后将为 {safe['count']} 个光鸭对象创建分享",
        effect_metadata={"reobservation_source": bind_observation_source(
            capture_observation_source(observation, arguments["object_refs"]), context,
        )},
        data={
            **safe,
            "effects": [
                "只分享本次目录快照中选定的对象。",
                "确认时会重新校验对象、登录凭据和分享策略。",
                "创建后会向当前登录用户返回分享链接；链接不会写入模型上下文。",
                *([note] if note else []),
            ],
        },
        evidence=[
            Evidence(
                "guangya_share_create_preview",
                "已冻结分享对象和策略；访问码与 Provider 对象 ID 不进入公开预览。",
                _now(),
            )
        ],
    ), fingerprint


def _extract_created_share(raw: dict[str, Any]) -> dict[str, str]:
    candidate_id = str(
        _first(raw, "shareId", "shareID", "share_id", "id") or ""
    ).strip()
    share_id = candidate_id if _SHARE_ID_RE.fullmatch(candidate_id) else ""
    candidate_url = str(
        _first(raw, "shareUrl", "shareURL", "share_url", "url", "link") or ""
    ).strip()
    # 顶层 ``code`` 常是 HTTP 业务成功码（0/200），不能误当成分享访问码。
    data = raw.get("data")
    candidate_code = str(
        (data.get("code") if isinstance(data, dict) else "") or _first(
            raw,
            "shareCode",
            "share_code",
            "accessCode",
            "access_code",
            "extractCode",
            "extract_code",
            "pwd",
        )
        or ""
    ).strip()
    code = candidate_code if _CODE_RE.fullmatch(candidate_code) else ""
    share_url = ""
    if candidate_url and len(candidate_url) <= 2048:
        try:
            parsed = urlsplit(candidate_url)
            hostname = (parsed.hostname or "").casefold()
            if (
                parsed.scheme.casefold() == "https"
                and not parsed.username
                and not parsed.password
                and (
                    hostname == "guangyapan.com"
                    or hostname.endswith(".guangyapan.com")
                )
                and parsed.path.startswith("/s/")
            ):
                share_url = urlunsplit(parsed)
        except ValueError:
            share_url = ""
    if not share_url and share_id:
        share_url = f"https://www.guangyapan.com/s/{share_id}#/share"
    return {"share_id": share_id, "share_url": share_url, "access_code": code}


def execute_create_guangya_share(
    arguments: dict[str, Any], expected_context: str, context: ToolContext, *, workspace_actions: GuangyaWorkspaceActions
) -> ToolResult:
    try:
        safe, fingerprint, selected, _observation, _identity = _create_snapshot(
            arguments, owner=context.owner, session_id=context.session_id, verify_live=True, workspace_actions=workspace_actions
        )
        if fingerprint != (expected_context or ""):
            raise AgentToolError("光鸭分享计划已变化，请重新预检", code="confirmation_stale")
        expected_generation = safe.pop("credential_generation")
        client = _open_client()
        try:
            if (client.credential_generation) != expected_generation:
                raise AgentToolError(
                    "光鸭登录凭据已变化，请重新读取目录",
                    code="confirmation_stale",
                )
            raw = client.create_user_share(
                [str(item.get("file_id") or "") for item in selected],
                title=safe["title"],
                validate_duration=safe["expires_days"],
                code=str(arguments.get("access_code") or ""),
                auto_fill_code=bool(arguments.get("auto_access_code")),
                max_restore_count=safe["max_restore_count"],
                allow_download=safe["allow_download"],
            )
            created = _extract_created_share(raw)
        finally:
            close_guangya_client(client)
    except Exception as exc:
        raise _public_error(exc, fallback="创建光鸭分享失败") from exc

    verified = bool(created["share_url"])
    public_data = {
        "title": safe["title"],
        "count": safe["count"],
        "expires_days": safe["expires_days"],
        "allow_download": safe["allow_download"],
        "share_url": created["share_url"],
        "access_code": created["access_code"],
        "verified": verified,
        "verification_pending": not verified,
    }
    return ToolResult(
        True,
        "completed" if verified else "accepted",
        (
            f"光鸭分享已创建，共包含 {safe['count']} 个对象"
            if verified
            else f"光鸭已受理分享创建请求，共包含 {safe['count']} 个对象"
        ),
        data=public_data,
        model_data={
            "title": safe["title"],
            "count": safe["count"],
            "expires_days": safe["expires_days"],
            "share_created": verified,
            "verification_pending": not verified,
            "share_link_available_to_user": bool(created["share_url"]),
            "access_code_available_to_user": bool(created["access_code"]),
        },
        evidence=[
            Evidence(
                "guangya_share_create",
                (
                    "Provider 已返回可验证的光鸭分享链接；链接和访问码仅出现在当前用户公开结果中。"
                    if verified
                    else "Provider 已受理创建请求，但未返回可验证的分享标识；未将不可信链接或访问码公开。"
                ),
                _now(),
            )
        ],
        suggestions=(
            []
            if verified
            else ["可稍后查询光鸭分享列表，核对新分享是否已可见。"]
        ),
    )


def guangya_share_revoke_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict) or set(arguments) != {
        "guangya_shares_ref",
        "indices",
    }:
        raise AgentToolError("撤销光鸭分享需要分享引用和 indices")
    ref = str(arguments.get("guangya_shares_ref") or "").strip()
    if not ref.startswith("ref_") or len(ref) > 200:
        raise AgentToolError("分享引用无效")
    raw_indices = arguments.get("indices")
    if not isinstance(raw_indices, list) or not 1 <= len(raw_indices) <= 100:
        raise AgentToolError("indices 必须包含 1 到 100 个序号")
    indices: list[int] = []
    for value in raw_indices:
        if type(value) is not int or not 1 <= value <= _MAX_SHARES:
            raise AgentToolError("分享序号无效")
        if value not in indices:
            indices.append(value)
    return {"guangya_shares_ref": ref, "indices": indices}


def _selected_shares(arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    collection = arguments.get("guangya_shares")
    if not isinstance(collection, dict) or not isinstance(collection.get("items"), list):
        raise AgentToolError("分享引用已失效，请重新查询", code="confirmation_stale")
    page = _bounded_int(collection.get("page"))
    page_size = _bounded_int(collection.get("page_size"))
    by_index = {
        (page - 1) * page_size + offset: item
        for offset, item in enumerate(collection["items"], 1)
        if isinstance(item, dict)
    }
    selected: list[dict[str, Any]] = []
    for index in arguments.get("indices") or ():
        item = by_index.get(int(index))
        if item is None:
            raise AgentToolError(
                "选择的分享序号不属于当前快照，请重新查询",
                code="confirmation_stale",
            )
        selected.append(item)
    return collection, selected


def _revoke_snapshot(
    arguments: dict[str, Any], *, verify_live: bool
) -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, Any]]:
    collection, selected = _selected_shares(arguments)
    client: GuangYaClient | None = None
    try:
        client = _open_client()
        if (client.credential_generation) != _bounded_int(
            collection.get("credential_generation")
        ):
            raise AgentToolError(
                "光鸭登录凭据已变化，请重新查询分享列表",
                code="confirmation_stale",
            )
        if verify_live:
            current = {
                item["share_id"]: item for item in _load_share_snapshots(client)
            }
            for item in selected:
                live = current.get(str(item.get("share_id") or ""))
                if live is None or any(
                    str(live.get(key) or "") != str(item.get(key) or "")
                    for key in ("share_id", "title", "status", "updated_at")
                ):
                    raise AgentToolError(
                        "分享列表已变化，请重新查询后再确认",
                        code="confirmation_stale",
                    )
    finally:
        if client is not None:
            close_guangya_client(client)
    safe = {
        # 来自已复核的会话私有引用，不接受模型提供的世代覆盖。
        "credential_generation": _bounded_int(collection.get("credential_generation")),
        "count": len(selected),
        "samples": [
            sanitize_resource_title(item.get("title"), limit=160) or "未命名分享"
            for item in selected[:6]
        ],
    }
    fingerprint = confirmation_context_fingerprint(
        {
            "credential_generation": _bounded_int(
                collection.get("credential_generation")
            ),
            "shares": selected,
        },
        domain="guangya-share-revoke",
    )
    return safe, fingerprint, selected, collection


def prepare_revoke_guangya_shares(
    arguments: dict[str, Any], context: ToolContext
) -> tuple[ToolResult, str]:
    try:
        safe, fingerprint, selected, collection = _revoke_snapshot(arguments, verify_live=True)
        note = reobserved_selection_note(context, kind="guangya_shares", generation=safe["credential_generation"],
                                         selected=selected, arguments=arguments)
        safe.pop("credential_generation")
    except Exception as exc:
        raise _public_error(exc, fallback="光鸭分享撤销预检失败") from exc
    return ToolResult(
        True,
        "confirmation_required",
        f"确认后将撤销 {safe['count']} 条光鸭分享",
        effect_metadata={"reobservation_source": _share_reobservation_source(
            arguments, context, collection, selected,
        )},
        data={
            **safe,
            "effects": [
                "只撤销本次分享列表快照中选定的分享。",
                "撤销后原分享链接将不可继续访问。",
                "分享中的原始云盘文件不会被删除或移动。",
                *([note] if note else []),
            ],
        },
        evidence=[
            Evidence(
                "guangya_share_revoke_preview",
                "已冻结分享列表快照；未公开分享 ID 或访问凭据。",
                _now(),
            )
        ],
    ), fingerprint


def execute_revoke_guangya_shares(
    arguments: dict[str, Any], expected_context: str, _context: ToolContext
) -> ToolResult:
    try:
        safe, fingerprint, selected, _collection = _revoke_snapshot(arguments, verify_live=True)
        if fingerprint != (expected_context or ""):
            raise AgentToolError("光鸭分享撤销计划已变化，请重新预检", code="confirmation_stale")
        expected_generation = safe.pop("credential_generation")
        client = _open_client()
        try:
            if (client.credential_generation) != expected_generation:
                raise AgentToolError(
                    "光鸭登录凭据已变化，请重新查询分享列表",
                    code="confirmation_stale",
                )
            client.delete_user_shares(
                [str(item.get("share_id") or "") for item in selected]
            )
            remaining = {str(item.get("share_id") or "") for item in selected}
            for _attempt in range(20):
                try:
                    current = {item["share_id"] for item in _load_share_snapshots(client)}
                except Exception as exc:  # noqa: BLE001 -- 撤销已受理，复核失败不能丢失受理结果
                    logger.warning("光鸭撤销分享后复核暂不可用 type=%s", type(exc).__name__)
                    break
                remaining &= current
                if not remaining:
                    break
                time.sleep(0.5)
        finally:
            close_guangya_client(client)
    except Exception as exc:
        raise _public_error(exc, fallback="撤销光鸭分享失败") from exc

    verified = not remaining
    return ToolResult(
        True,
        "completed" if verified else "accepted",
        (
            f"已撤销 {safe['count']} 条光鸭分享"
            if verified
            else f"光鸭已受理 {safe['count']} 条分享撤销请求"
        ),
        data={**safe, "verified": verified, "verification_pending": not verified},
        model_data={
            "count": safe["count"],
            "verified": verified,
            "verification_pending": not verified,
        },
        evidence=[
            Evidence(
                "guangya_share_revoke",
                "Provider 已接受撤销请求；同步可见时已重新读取分享列表验证。",
                _now(),
            )
        ],
    )
