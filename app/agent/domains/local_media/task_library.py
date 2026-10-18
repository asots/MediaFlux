"""从任务归档事实解析唯一媒体库绑定与配置指纹。"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

from app import database as db
from app.agent.errors import AgentToolError
from app.clients.base import close_media_server_client
from app.clients.emby import EmbyClient
from app.clients.jellyfin import JellyfinClient
from app.modules.media_server_path_mapping import (
    MediaServerPathMapping,
    configured_media_server_refresh_options,
)
from app.modules.media_server_profiles import list_configured_profiles
from app.modules.web_secret import get_web_secret

from . import common


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def _client_for_provider(
    provider: str, profile: Any, *, refresh_options: dict[str, object]
) -> Any:
    if provider == "jellyfin":
        return JellyfinClient(profile.url, profile.credential, **refresh_options)
    if provider == "emby":
        return EmbyClient(profile.url, profile.credential, **refresh_options)
    raise AgentToolError("绑定的媒体服务器类型不受支持", code="precondition_failed")


def _server_config_digest(
    provider: str, profile: Any, refresh_options: dict[str, object]
) -> str:
    mappings = refresh_options.get("path_mappings") or ()
    canonical = {
        "provider": provider,
        "url": str(profile.url or "").strip().rstrip("/"),
        "credential": str(profile.credential or ""),
        "path_mappings": [
            {
                "local_prefix": str(getattr(item, "local_prefix", "")),
                "server_prefix": str(getattr(item, "server_prefix", "")),
            }
            for item in mappings
        ],
        "allow_global_refresh_fallback": bool(
            refresh_options.get("allow_global_refresh_fallback", False)
        ),
    }
    try:
        secret = get_web_secret().encode("utf-8")
    except Exception:
        raise AgentToolError(
            "媒体服务器配置指纹暂时不可用", code="precondition_failed"
        ) from None
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(
        secret,
        b"mediaflux-agent-media-server-config:v1\0" + payload,
        hashlib.sha256,
    ).hexdigest()


def _bound_task_scope(task: Any) -> dict[str, Any]:
    if task.status != "completed":
        raise AgentToolError(
            "只有已完成整理的任务可以刷新或核验媒体库", code="precondition_failed"
        )
    items = [
        item
        for item in db.list_local_media_task_items(task.id, owner=common._WORKSPACE_OWNER)
        if str(item["role"] or "") == "video" and str(item["target_path"] or "").strip()
    ]
    targets = db.list_local_library_targets(task.source_id, owner=common._WORKSPACE_OWNER)
    if not items or not targets:
        raise AgentToolError(
            "任务没有可验证的已绑定媒体库目标", code="precondition_failed"
        )
    resolved: list[tuple[Any, Path]] = []
    for item in items:
        target_path = Path(str(item["target_path"])).expanduser().resolve(strict=False)
        matches = [
            target for target in targets if _path_within(target_path, Path(target.path))
        ]
        if not matches:
            raise AgentToolError(
                "任务目标未绑定到明确媒体库", code="precondition_failed"
            )
        longest = max(
            len(str(Path(target.path).resolve(strict=False))) for target in matches
        )
        winners = [
            target
            for target in matches
            if len(str(Path(target.path).resolve(strict=False))) == longest
        ]
        if len(winners) != 1:
            raise AgentToolError(
                "任务目标媒体库绑定存在歧义", code="precondition_failed"
            )
        resolved.append((winners[0], target_path.parent))
    keys = {
        (str(target.provider).lower(), str(target.library_id), str(target.library_name))
        for target, _path in resolved
    }
    if len(keys) != 1:
        raise AgentToolError(
            "一个任务涉及多个媒体库，Agent 不会批量猜测刷新范围",
            code="precondition_failed",
        )
    provider, configured_library_id, configured_library_name = next(iter(keys))
    if provider not in {"jellyfin", "emby"} or not (
        configured_library_id or configured_library_name
    ):
        raise AgentToolError(
            "任务目标没有有效的媒体服务器与媒体库绑定", code="precondition_failed"
        )
    profiles = {
        item.server_type: item
        for item in list_configured_profiles()
        if item.enabled and item.configured
    }
    profile = profiles.get(provider)
    if profile is None:
        raise AgentToolError(
            "绑定的媒体服务器当前未启用或未配置", code="precondition_failed"
        )
    refresh_options = configured_media_server_refresh_options(provider)
    client = _client_for_provider(provider, profile, refresh_options=refresh_options)
    try:
        try:
            folders = client.list_virtual_folders()
        except Exception:
            raise AgentToolError(
                "暂时无法读取绑定媒体库", code="precondition_failed"
            ) from None
        if configured_library_id:
            matches = [
                item
                for item in folders
                if str(item.get("id") or "").strip() == configured_library_id
            ]
            if len(matches) != 1:
                raise AgentToolError(
                    "媒体库绑定已变化，请重新绑定", code="precondition_failed"
                )
            folder = matches[0]
            actual_name = str(folder.get("name") or "").strip()
            if (
                configured_library_name
                and actual_name.casefold() != configured_library_name.casefold()
            ):
                raise AgentToolError(
                    "媒体库名称与绑定不一致，请重新绑定", code="precondition_failed"
                )
        else:
            matches = [
                item
                for item in folders
                if str(item.get("name") or "").strip().casefold()
                == configured_library_name.casefold()
            ]
            if len(matches) != 1:
                raise AgentToolError(
                    "媒体库名称无法唯一解析，请重新绑定", code="precondition_failed"
                )
            folder = matches[0]
        library_id = str(folder.get("id") or "").strip()
        if not library_id:
            raise AgentToolError("媒体库绑定缺少可用标识", code="precondition_failed")
        try:
            paths = sorted(
                {
                    MediaServerPathMapping(target.path, target.server_path).apply(
                        str(path)
                    )
                    if str(getattr(target, "server_path", "") or "").strip()
                    else str(path)
                    for target, path in resolved
                }
            )
        except ValueError:
            raise AgentToolError(
                "媒体库服务端路径映射无效，请重新绑定", code="precondition_failed"
            ) from None
        server_config_digest = _server_config_digest(provider, profile, refresh_options)
        binding = {
            "provider": provider,
            "server_config_digest": server_config_digest,
            "library_id": library_id,
            "library_name": str(
                folder.get("name") or configured_library_name or ""
            ).strip(),
            "paths_digest": hashlib.sha256(
                "\n".join(paths).encode("utf-8")
            ).hexdigest(),
        }
        return {
            "task": task,
            "client": client,
            "paths": paths,
            "binding": binding,
            "server_label": str(profile.label),
        }
    except Exception:
        close_media_server_client(client)
        raise
