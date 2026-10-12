"""光鸭声明式工作区的私有只读观察快照。"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
import time
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from app.clients.guangya import GuangYaClient, GuangYaFile
from app.config import PATHS
from app.modules.organize import (
    DEFAULT_ORGANIZE_METADATA_EXTS,
    DEFAULT_ORGANIZE_VIDEO_EXTS,
)
from app.modules.organize_postprocess import SUBTITLE_EXTS
from app.modules.web_secret import get_web_secret
from app.private_files import protect_private_file
from app.repositories.organize_operation_jobs import organize_operation_owner_digest

_OBSERVATION_VERSION = 1
_OBSERVATION_TTL_SECONDS = 10 * 60
_MAX_SCANNED_ITEMS = 2_000
_MAX_SCANNED_DIRS = 500
_MAX_DEPTH = 12
_MAX_PLAN_BYTES = 8 * 1024 * 1024
_MAX_STORAGE_BYTES = 64 * 1024 * 1024
_MAX_OBSERVATIONS = 32
_MAX_OBSERVATIONS_PER_OWNER = 4
_MAX_PAGE_SIZE = 50
_SAFE_PLAN_ID = re.compile(r"^[0-9a-f]{32}$")
_SAFE_OBSERVATION_REF = re.compile(r"^OBS[0-9A-F]{32}$")
_SAFE_HANDLE = re.compile(r"^OBJ[0-9A-F]{24}$")
_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "gif", "bmp", "avif"}
_MEDIA_KINDS = frozenset(
    {"directory", "video", "subtitle", "image", "metadata", "other"}
)


class GuangYaWorkspaceError(RuntimeError):
    """可安全映射给 Agent 的光鸭工作区错误。"""


class GuangYaWorkspaceStale(GuangYaWorkspaceError):
    """观察快照已过期或不再属于当前会话。"""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _directory() -> Path:
    return Path(PATHS.data_dir) / "agent-guangya-workspace"


def _ensure_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        if path.is_symlink() or not path.is_dir():
            raise GuangYaWorkspaceError("光鸭观察快照目录不可用")
        if os.name == "posix":
            path.chmod(0o700)
    except OSError as exc:
        raise GuangYaWorkspaceError("光鸭观察快照目录不可用") from exc


def _plan_path(plan_id: str) -> Path:
    if not _SAFE_PLAN_ID.fullmatch(str(plan_id or "")):
        raise GuangYaWorkspaceError("光鸭观察编号无效")
    return _directory() / f"{plan_id}.json"


def observation_ref(plan_id: str) -> str:
    if not _SAFE_PLAN_ID.fullmatch(str(plan_id or "")):
        raise GuangYaWorkspaceError("光鸭观察编号无效")
    return "OBS" + str(plan_id).upper()


def observation_plan_id(value: object) -> str:
    ref = str(value or "").strip().upper()
    if not _SAFE_OBSERVATION_REF.fullmatch(ref):
        raise GuangYaWorkspaceError("光鸭观察编号无效")
    return ref[3:].casefold()


def valid_observation_ref(value: object) -> bool:
    return bool(_SAFE_OBSERVATION_REF.fullmatch(str(value or "").strip().upper()))


def valid_object_handle(value: object) -> bool:
    return bool(_SAFE_HANDLE.fullmatch(str(value or "").strip().upper()))


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        {key: value for key, value in payload.items() if key != "auth"},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _auth(payload: dict[str, Any]) -> str:
    secret = str(get_web_secret() or "").encode("utf-8")
    if not secret:
        raise GuangYaWorkspaceError("光鸭观察签名密钥不可用")
    return hmac.new(
        secret,
        b"mediaflux-guangya-workspace:v1\0" + _canonical(payload),
        hashlib.sha256,
    ).hexdigest()


def _atomic_write(payload: dict[str, Any]) -> None:
    path = _plan_path(str(payload.get("plan_id") or ""))
    _ensure_directory(path.parent)
    stored = dict(payload)
    stored["auth"] = _auth(stored)
    encoded = (
        json.dumps(stored, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_PLAN_BYTES:
        raise GuangYaWorkspaceError("光鸭观察范围过大，请缩小目录或关闭递归")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        protect_private_file(Path(temporary))
        os.replace(temporary, path)
        protect_private_file(path)
    finally:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass


def _read(plan_id: str) -> dict[str, Any]:
    path = _plan_path(plan_id)
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > _MAX_PLAN_BYTES
        ):
            raise GuangYaWorkspaceError("光鸭观察快照不可用")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise GuangYaWorkspaceError("光鸭观察快照不可用") from exc
    if (
        not isinstance(payload, dict)
        or int(payload.get("version") or 0) != _OBSERVATION_VERSION
    ):
        raise GuangYaWorkspaceError("光鸭观察快照版本无效")
    actual = str(payload.get("auth") or "")
    if not actual or not hmac.compare_digest(actual, _auth(payload)):
        raise GuangYaWorkspaceError("光鸭观察快照校验失败")
    return payload


def discard_observation(value: object) -> None:
    try:
        plan_id = (
            observation_plan_id(value)
            if str(value or "").upper().startswith("OBS")
            else str(value or "")
        )
        path = _plan_path(plan_id)
        if path.is_file() and not path.is_symlink():
            path.unlink()
    except (OSError, GuangYaWorkspaceError):
        return


def discard_owner_observations(owner: str) -> int:
    """删除指定 Agent owner 的全部短期观察，阻止旧引用在 reset 后复用。"""
    owner_key = str(owner or "").strip()
    if not owner_key:
        return 0
    owner_digest = organize_operation_owner_digest(owner_key)
    directory = _directory()
    if not directory.exists() or directory.is_symlink():
        return 0
    removed = 0
    for path in tuple(directory.glob("*.json")):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            payload = _read(path.stem)
            if hmac.compare_digest(
                str(payload.get("owner_digest") or ""), owner_digest
            ):
                path.unlink()
                removed += 1
        except (OSError, GuangYaWorkspaceError):
            continue
    return removed


def maintain_workspace_observations(
    *,
    preserve_plan_id: str = "",
) -> dict[str, int]:
    directory = _directory()
    if not directory.exists() or directory.is_symlink():
        return {"removed": 0, "remaining": 0, "bytes": 0}
    preserved = str(preserve_plan_id or "").strip().casefold()
    current = time.time()
    rows: list[tuple[float, str, int, str]] = []
    removed = 0
    total_bytes = 0
    for path in directory.glob("*.json"):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
            size = max(0, int(stat.st_size))
            try:
                payload = _read(path.stem)
                expiry = float(payload.get("expires_at_epoch") or 0)
                created = float(payload.get("created_at_epoch") or stat.st_mtime)
                owner_digest = str(payload.get("owner_digest") or "")
            except GuangYaWorkspaceError:
                expiry = 0
                created = float(stat.st_mtime)
                owner_digest = ""
            if expiry <= current:
                path.unlink()
                removed += 1
                continue
            rows.append((created, path.stem, size, owner_digest))
            total_bytes += size
        except OSError:
            continue
    rows.sort()

    def remove_row(row: tuple[float, str, int, str]) -> None:
        nonlocal removed, total_bytes
        _created, plan_id, size, _owner_digest = row
        discard_observation(plan_id)
        try:
            rows.remove(row)
        except ValueError:
            return
        total_bytes = max(0, total_bytes - size)
        removed += 1

    owner_digests = {row[3] for row in rows if row[3]}
    for owner_digest in owner_digests:
        owner_rows = [row for row in rows if row[3] == owner_digest]
        while len(owner_rows) > _MAX_OBSERVATIONS_PER_OWNER:
            victim = next(
                (row for row in owner_rows if row[1] != preserved),
                None,
            )
            if victim is None:
                break
            remove_row(victim)
            owner_rows.remove(victim)

    while rows and (len(rows) > _MAX_OBSERVATIONS or total_bytes > _MAX_STORAGE_BYTES):
        victim = next((row for row in rows if row[1] != preserved), None)
        if victim is None:
            break
        remove_row(victim)
    return {"removed": removed, "remaining": len(rows), "bytes": total_bytes}


def _normalize_path(value: object, *, allow_root: bool = False) -> str:
    path = str(value or "").strip().replace("\\", "/")
    if not path.startswith("/") or len(path) > 2048:
        raise GuangYaWorkspaceError("光鸭目录路径必须是绝对路径")
    parts = [part for part in path.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        raise GuangYaWorkspaceError("光鸭目录路径不能包含相对组件")
    if not parts:
        if allow_root:
            return "/"
        raise GuangYaWorkspaceError("该操作不能以光鸭根目录为对象")
    return "/" + "/".join(parts)


def _resolve_path(client: GuangYaClient, path: str) -> tuple[GuangYaFile, str]:
    components = _normalize_path(path).strip("/").split("/")
    parent_id = "0"
    parent_path = "/"
    for index, component in enumerate(components):
        items = client.list_dir(parent_id)
        exact = [item for item in items if item.name == component]
        matches = exact or [
            item for item in items if item.name.casefold() == component.casefold()
        ]
        if len(matches) != 1:
            raise GuangYaWorkspaceError(
                f"路径不存在或名称不唯一：/{'/'.join(components[: index + 1])}"
            )
        current = matches[0]
        if index < len(components) - 1 and not current.is_dir:
            raise GuangYaWorkspaceError("光鸭路径的中间组件不是目录")
        parent_path = "/" + "/".join(components[:index]) if index else "/"
        parent_id = str(current.file_id)
    return current, parent_path


def resolve_workspace_path(client: GuangYaClient, path: object) -> GuangYaFile:
    """只在服务端解析精确光鸭绝对路径；调用方不得向外返回对象 ID。"""
    target, _parent_path = _resolve_path(client, _normalize_path(path))
    return target


def _extension(item: GuangYaFile) -> str:
    declared = str(item.extension or "").strip().lower().lstrip(".")
    if declared:
        return declared
    match = re.search(r"[.。．]([A-Za-z0-9]{1,10})$", item.name)
    return match.group(1).lower() if match else ""


def _media_kind(item: GuangYaFile) -> str:
    if item.is_dir:
        return "directory"
    ext = _extension(item)
    if ext in DEFAULT_ORGANIZE_VIDEO_EXTS:
        return "video"
    if ext in SUBTITLE_EXTS:
        return "subtitle"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in DEFAULT_ORGANIZE_METADATA_EXTS:
        return "metadata"
    return "other"


def _snapshot(
    item: GuangYaFile, *, parent_path: str, relative_parent: str
) -> dict[str, Any]:
    return {
        "file_id": str(item.file_id),
        "parent_id": str(item.parent_id),
        "name": str(item.name),
        "is_dir": bool(item.is_dir),
        "size": max(0, int(item.size or 0)),
        "etag": str(item.etag or ""),
        "updated_at": max(0, int(item.updated_at or 0)),
        "extension": _extension(item),
        "media_kind": _media_kind(item),
        "parent_path": parent_path,
        "relative_parent": relative_parent,
    }


def _handle(plan_id: str, item: dict[str, Any]) -> str:
    secret = str(get_web_secret() or "").encode("utf-8")
    if not secret:
        raise GuangYaWorkspaceError("光鸭观察签名密钥不可用")
    digest = (
        hmac.new(
            secret,
            b"mediaflux-guangya-object-handle:v1\0"
            + plan_id.encode("ascii")
            + b"\0"
            + str(item.get("file_id") or "").encode("utf-8")
            + b"\0"
            + str(item.get("parent_id") or "").encode("utf-8"),
            hashlib.sha256,
        )
        .hexdigest()[:24]
        .upper()
    )
    return "OBJ" + digest


_SOURCE_FIELDS = (
    "scope_path", "scope_name", "scope_names", "operation", "query", "kinds",
    "max_depth", "recursive", "max_items", "truncated", "scanned_dirs", "scanned_items",
    "observed_at", "created_at", "created_at_epoch", "expires_at", "expires_at_epoch",
)
_ROOT_FIELDS = ("file_id", "parent_id", "path", "name", "is_dir")


def observation_sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """只保留实际查询来源；合并快照的展示根和过滤条件不能代替各原查询。"""
    sources = {}
    for original in payload.get("source_observations") or [payload]:
        source = {key: deepcopy(original[key]) for key in _SOURCE_FIELDS if key in original}
        source["observation_ref"] = original.get("observation_ref") or observation_ref(str(original.get("plan_id") or ""))
        roots = original.get("roots")
        source["roots"] = ([{key: deepcopy(root[key]) for key in _ROOT_FIELDS if key in root}
                            for root in roots] if isinstance(roots, list) and all(isinstance(root, dict) for root in roots) else [])
        sources[source["observation_ref"]] = source
    if len(sources) > _MAX_OBSERVATIONS:
        raise GuangYaWorkspaceError("观察来源过多，请缩小范围后重新读取")
    return list(sources.values())


def observation_query_arguments(source: dict[str, Any]) -> dict[str, Any] | None:
    roots = source.get("roots")
    if not roots or not all(root.get("file_id") and root.get("path") for root in roots):
        return None
    operation = source.get("operation")
    if operation == "stat" and len(roots) == 1:
        return {"operation": "stat", "path": roots[0]["path"]}
    if operation not in {"list", "tree", "search"} or not all(
        key in source for key in ("query", "kinds", "max_depth", "max_items")
    ):
        return None
    paths = [root["path"] for root in roots]
    return {"operation": operation, **({"path": paths[0]} if len(paths) == 1 else {"paths": paths}),
            **{key: deepcopy(source[key]) for key in ("kinds", "max_depth", "max_items")},
            **({"query": source["query"]} if source.get("query") else {})}


def capture_observation_source(payload: dict[str, Any], object_refs: Sequence[str]) -> dict[str, Any]:
    """在观察仍有效时捕获私有来源与选中身份；它不是可复用的写权限。"""
    sources = observation_sources(payload)
    ref = observation_ref(str(payload.get("plan_id") or ""))
    entries = observation_entry_map(payload)
    selected = []
    for handle in dict.fromkeys(str(value).strip().upper() for value in object_refs):
        if handle not in entries:
            raise GuangYaWorkspaceStale("所选对象不属于原观察，请重新读取")
        entry = entries[handle]
        item = {key: deepcopy(entry[key]) for key in (
            "file_id", "parent_id", "name", "is_dir", "size", "etag", "updated_at",
            "parent_path", "relative_parent", "observed_by",
        ) if key in entry}
        if isinstance(entry.get("source_root"), dict):
            item["source_root"] = {key: deepcopy(entry["source_root"][key]) for key in _ROOT_FIELDS if key in entry["source_root"]}
        item.update(old_object_ref=handle, source_observation_ref=entry.get("source_observation_ref") or ref)
        selected.append(item)
    queries = [observation_query_arguments(source) for source in sources]
    baseline = {key: deepcopy(payload[key]) for key in _SOURCE_FIELDS if key in payload}
    baseline.update(observation_ref=ref,
                    roots=[{key: deepcopy(root[key]) for key in _ROOT_FIELDS if key in root}
                           for root in payload.get("roots") or () if isinstance(root, dict)], selected=selected,
                    source_observations=sources)
    available = bool(queries) and all(query is not None for query in queries)
    return {"credential_generation": payload.get("credential_generation"), "available": available,
            "read": ({"tool": "guangya.fs.query", "arguments": queries[0] if len(queries) == 1 else {"operation": "reobserve"}}
                     if available else None), "baseline": baseline}


def _store_observation(
    client: GuangYaClient,
    *,
    owner: str,
    scope_path: str,
    scope_name: str,
    roots: Sequence[dict[str, Any]],
    max_items: int,
    operation: str,
    recursive: bool,
    query: str,
    kinds: Sequence[str] = (),
    scope_names: Sequence[str] = (),
    max_depth: int = _MAX_DEPTH,
    truncated: bool,
    scanned_dirs: int,
    scanned_items: int,
    entries: list[dict[str, Any]],
    source_observations: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    entries.sort(
        key=lambda row: (
            str(row.get("relative_parent") or "").casefold(),
            not bool(row.get("is_dir")),
            str(row.get("name") or "").casefold(),
            str(row.get("file_id") or ""),
        )
    )
    plan_id = uuid.uuid4().hex
    for entry in entries:
        entry["handle"] = _handle(plan_id, entry)
        entry.setdefault("source_observation_ref", observation_ref(plan_id))
    current = time.time()
    payload = {
        "version": _OBSERVATION_VERSION,
        "plan_id": plan_id,
        "owner_digest": organize_operation_owner_digest(owner),
        "credential_generation": int(client.credential_generation),
        "created_at": _now_iso(),
        "created_at_epoch": current,
        "expires_at_epoch": current + _OBSERVATION_TTL_SECONDS,
        "scope_path": scope_path,
        "scope_name": str(scope_name),
        "scope_names": [str(item) for item in scope_names if str(item)],
        # 私有查询来源；展示 scope 不携带完整根身份，不能用于还原查询范围。
        "roots": [dict(root) for root in roots],
        "max_items": max_items,
        "operation": operation,
        "query": query,
        "kinds": sorted({str(item) for item in kinds if str(item)}),
        "max_depth": max(0, min(int(max_depth), _MAX_DEPTH)),
        "recursive": bool(recursive),
        "truncated": bool(truncated),
        "scanned_dirs": max(0, int(scanned_dirs)),
        "scanned_items": max(0, int(scanned_items)),
        "entries": entries,
        **({"source_observations": deepcopy(list(source_observations))} if source_observations else {}),
    }
    _atomic_write(payload)
    maintain_workspace_observations(preserve_plan_id=plan_id)
    if not _plan_path(plan_id).exists():
        raise GuangYaWorkspaceError("光鸭观察快照容量已满，请稍后重试")
    return payload


def _search_match(entry: dict[str, Any], query: str, kinds: frozenset[str]) -> bool:
    if kinds and str(entry.get("media_kind") or "") not in kinds:
        return False
    needle = str(query or "").strip().casefold()
    if not needle:
        return True
    haystack = " ".join(
        (
            str(entry.get("name") or ""),
            str(entry.get("relative_parent") or ""),
            str(entry.get("extension") or ""),
            str(entry.get("media_kind") or ""),
        )
    ).casefold()
    return needle in haystack


def create_multi_directory_observation(
    client: GuangYaClient,
    *,
    owner: str,
    paths: Sequence[str],
    recursive: bool = False,
    max_items: int = 500,
    query: str = "",
    kinds: Sequence[str] = (),
    max_depth: int = _MAX_DEPTH,
    operation: str = "",
    expected_roots: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_paths = tuple(
        dict.fromkeys(_normalize_path(path, allow_root=True) for path in paths)
    )
    if not normalized_paths or len(normalized_paths) > 32:
        raise GuangYaWorkspaceError("光鸭批量观察必须包含 1 到 32 个精确目录")
    safe_max = max(1, min(int(max_items), _MAX_SCANNED_ITEMS))
    safe_depth = max(0, min(int(max_depth), _MAX_DEPTH))
    normalized_query = str(query or "").strip()
    if len(normalized_query) > 160:
        raise GuangYaWorkspaceError("光鸭搜索关键词过长")
    normalized_kinds = frozenset(str(item or "").strip().casefold() for item in kinds)
    if normalized_kinds - _MEDIA_KINDS:
        raise GuangYaWorkspaceError("光鸭对象类型过滤值无效")
    normalized_operation = (
        str(operation or ("tree" if recursive else "list")).strip().casefold()
    )
    if normalized_operation not in {"list", "tree", "search"}:
        raise GuangYaWorkspaceError("不支持的光鸭目录查询方式")
    if normalized_operation == "search" and not normalized_query:
        raise GuangYaWorkspaceError("光鸭目录搜索必须提供关键词")
    if normalized_operation == "list" and recursive:
        raise GuangYaWorkspaceError("目录列表不能启用递归")
    recursive = normalized_operation in {"tree", "search"} or bool(recursive)

    roots: list[dict[str, Any]] = []
    scope_names: list[str] = []
    seen_root_ids: set[str] = set()
    for target_path in normalized_paths:
        if target_path == "/":
            target_id = "0"
            target_name = "根目录"
            root = {"file_id": target_id, "path": target_path, "is_dir": True}
        else:
            target, _parent_path = _resolve_path(client, target_path)
            if not target.is_dir:
                raise GuangYaWorkspaceError("通用目录观察只支持精确目录路径")
            target_id = str(target.file_id)
            target_name = str(target.name)
            root = {
                "file_id": target_id,
                "path": target_path,
                "is_dir": bool(target.is_dir),
                "parent_id": str(target.parent_id),
            }
        if expected_roots is not None:
            expected = next((item for item in expected_roots if item.get("path") == target_path), None)
            if expected is None or any(root.get(key) != expected.get(key)
                                       for key in ("file_id", "is_dir", "parent_id")):
                raise GuangYaWorkspaceStale("原观察根对象已变化，未读取同名替代目录")
        if target_id in seen_root_ids:
            continue
        seen_root_ids.add(target_id)
        roots.append(root)
        scope_names.append(target_name)

    multiple = len(roots) > 1
    queue = deque(
        (root["file_id"], root["path"], name if multiple else "", 0, root)
        for root, name in zip(roots, scope_names)
    )
    entries: list[dict[str, Any]] = []
    seen_entries: set[tuple[str, str]] = set()
    scanned_dirs = 0
    scanned_items = 0
    truncated = False
    while queue:
        directory_id, directory_path, relative_parent, depth, source_root = queue.popleft()
        scanned_dirs += 1
        if scanned_dirs > _MAX_SCANNED_DIRS:
            truncated = True
            break
        for item in client.list_dir(directory_id):
            scanned_items += 1
            if scanned_items > _MAX_SCANNED_ITEMS:
                truncated = True
                queue.clear()
                break
            entry = _snapshot(
                item,
                parent_path=directory_path,
                relative_parent=relative_parent or "当前目录",
            )
            entry_key = (str(item.file_id), str(item.parent_id))
            if entry_key not in seen_entries and _search_match(
                entry, normalized_query, normalized_kinds
            ):
                if len(entries) >= safe_max:
                    truncated = True
                    queue.clear()
                    break
                seen_entries.add(entry_key)
                entry["source_root"] = dict(source_root)
                entries.append(entry)
            if item.is_dir and recursive and depth < safe_depth:
                child_path = directory_path.rstrip("/") + "/" + item.name
                child_relative = (
                    item.name
                    if not relative_parent
                    else f"{relative_parent} › {item.name}"
                )
                queue.append(
                    (str(item.file_id), child_path, child_relative, depth + 1, source_root)
                )
            elif item.is_dir and recursive and depth >= safe_depth:
                # Reaching the requested depth is expected and is not observation truncation.
                continue

    return _store_observation(
        client,
        owner=owner,
        scope_path=normalized_paths[0] if len(normalized_paths) == 1 else "/",
        scope_name=(
            scope_names[0] if len(scope_names) == 1 else f"{len(scope_names)} 个目录"
        ),
        scope_names=scope_names if multiple else (),
        roots=roots,
        max_items=safe_max,
        operation=normalized_operation,
        recursive=recursive,
        query=normalized_query,
        kinds=normalized_kinds,
        max_depth=safe_depth,
        truncated=truncated,
        scanned_dirs=scanned_dirs,
        scanned_items=scanned_items,
        entries=entries,
    )


def create_directory_observation(
    client: GuangYaClient,
    *,
    owner: str,
    path: str,
    recursive: bool = False,
    max_items: int = 500,
    query: str = "",
    kinds: Sequence[str] = (),
    max_depth: int = _MAX_DEPTH,
    operation: str = "",
    expected_roots: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return create_multi_directory_observation(
        client,
        owner=owner,
        paths=(path,),
        recursive=recursive,
        max_items=max_items,
        query=query,
        kinds=kinds,
        max_depth=max_depth,
        operation=operation,
        expected_roots=expected_roots,
    )


def create_path_observation(
    client: GuangYaClient,
    *,
    owner: str,
    path: str,
    expected_roots: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """观察单个精确对象；根目录不会被包装成可写对象引用。"""
    target_path = _normalize_path(path)
    target, parent_path = _resolve_path(client, target_path)
    if expected_roots is not None:
        if len(expected_roots) != 1 or any(
            actual != expected_roots[0].get(key) for key, actual in (
                ("file_id", str(target.file_id)), ("is_dir", bool(target.is_dir)),
                ("parent_id", str(target.parent_id)), ("path", target_path),
            )
        ):
            raise GuangYaWorkspaceStale("原观察对象已变化，未读取同名替代对象")
    entry = _snapshot(target, parent_path=parent_path, relative_parent="当前对象")
    root = {
        "file_id": str(target.file_id),
        "path": target_path,
        "is_dir": bool(target.is_dir),
        "parent_id": str(target.parent_id),
    }
    entry["source_root"] = dict(root)
    return _store_observation(
        client,
        owner=owner,
        scope_path=target_path,
        scope_name=str(target.name),
        roots=[root],
        max_items=1,
        operation="stat",
        recursive=False,
        query="",
        kinds=(),
        truncated=False,
        scanned_dirs=0,
        scanned_items=1,
        entries=[entry],
    )


def load_directory_observation(
    value: object,
    *,
    owner: str,
    require_fresh: bool = True,
) -> dict[str, Any]:
    plan_id = observation_plan_id(value)
    payload = _read(plan_id)
    if not hmac.compare_digest(
        str(payload.get("owner_digest") or ""), organize_operation_owner_digest(owner)
    ):
        raise GuangYaWorkspaceError("光鸭观察快照不属于当前会话")
    if require_fresh and float(payload.get("expires_at_epoch") or 0) <= time.time():
        discard_observation(plan_id)
        raise GuangYaWorkspaceStale("光鸭目录观察已过期，请重新读取")
    return payload


def load_directory_observation_set(
    value: object,
    *,
    owner: str,
    object_refs: Sequence[str] = (),
) -> dict[str, Any]:
    """加载主快照，并按 owner 合并近期快照中实际需要的不透明对象引用。"""
    primary = load_directory_observation(value, owner=owner)
    required = {
        str(item or "").strip().upper()
        for item in object_refs
        if valid_object_handle(item)
    }
    if not required:
        return primary
    entries = observation_entry_map(primary)
    missing = required - entries.keys()
    if not missing:
        return primary

    primary_generation = primary.get("credential_generation")
    directory = _directory()
    if not directory.exists() or directory.is_symlink():
        return primary
    candidates: list[tuple[float, str]] = []
    primary_id = str(primary.get("plan_id") or "")
    try:
        paths = tuple(directory.glob("*.json"))
    except OSError:
        return primary
    for path in paths:
        if path.stem == primary_id or not _SAFE_PLAN_ID.fullmatch(path.stem):
            continue
        try:
            candidates.append((path.stat().st_mtime, observation_ref(path.stem)))
        except (OSError, GuangYaWorkspaceError):
            continue
    candidates.sort(reverse=True)
    merged_entries = list(primary.get("entries") or ())
    sources = {item["observation_ref"]: item for item in observation_sources(primary)}
    for _modified_at, candidate_ref in candidates:
        try:
            candidate = load_directory_observation(candidate_ref, owner=owner)
        except GuangYaWorkspaceError:
            continue
        if candidate.get("credential_generation") != primary_generation:
            continue
        candidate_entries = observation_entry_map(candidate)
        for handle in tuple(missing):
            observed = candidate_entries.get(handle)
            if observed is not None:
                # 保留签名快照中该 entry 自己的 source_root；legacy 缺失时仍未知，
                # 不能套用 primary.roots 或从展示 scope 推断为全盘范围。
                observed.setdefault("source_observation_ref", candidate_ref)
                sources.update({item["observation_ref"]: item for item in observation_sources(candidate)})
                entries[handle] = observed
                merged_entries.append(observed)
                missing.discard(handle)
        if not missing:
            break
    if len(merged_entries) == len(primary.get("entries") or ()):
        return primary
    merged = dict(primary)
    merged["entries"] = merged_entries
    merged["source_observations"] = list(sources.values())
    return merged


def locate_observed_object(client: GuangYaClient, expected: dict[str, Any], roots: Sequence[dict[str, Any]],
                           *, cache: dict[str, GuangYaFile | None], check_cancelled: Callable[[], None]) -> dict[str, Any] | None:
    """只追踪原选中 ID，以有界父链证明仍在原根；不搜索其它对象或全盘。"""
    def read(file_id: str) -> GuangYaFile | None:
        check_cancelled()
        if file_id not in cache:
            cache[file_id] = client.file_info(file_id)
        check_cancelled()
        return cache[file_id]

    file_id = str(expected.get("file_id") or "")
    current = read(file_id)
    if current is None or str(current.file_id) != file_id or bool(current.is_dir) != bool(expected.get("is_dir")):
        return None
    by_id = {str(root["file_id"]): root for root in roots if root.get("is_dir") is True}
    parent_id = str(current.parent_id or "0")
    names = [str(current.name)]
    visited = {file_id}
    for _ in range(_MAX_DEPTH + 1):
        check_cancelled()
        if parent_id in by_id:
            root = by_id[parent_id]
            path = str(root["path"]).rstrip("/") + "/" + "/".join(reversed(names))
            path = _normalize_path(path)
            entry = _snapshot(current, parent_path=path.rsplit("/", 1)[0] or "/",
                              relative_parent=" › ".join(reversed(names[1:])) or "当前目录")
            entry.update(source_root=deepcopy(root), observed_by="identity")
            return entry
        if parent_id == "0" or parent_id in visited:
            return None
        visited.add(parent_id)
        parent = read(parent_id)
        if parent is None or str(parent.file_id) != parent_id or not parent.is_dir:
            return None
        names.append(str(parent.name))
        parent_id = str(parent.parent_id or "0")
    return None


def merge_directory_observations(client: GuangYaClient, *, owner: str,
                                 payloads: Sequence[dict[str, Any]], force_new: bool = False,
                                 preferred_ids: frozenset[str] = frozenset()) -> dict[str, Any]:
    """新查询结果合并成自包含的签名快照，不依赖中间 OBS 的剩余 TTL。"""
    if not payloads:
        raise GuangYaWorkspaceStale("没有成功重新读取的观察")
    if len(payloads) == 1 and not force_new:
        return payloads[0]
    entries = {}
    roots = {}
    sources = {}
    scope_names = [f"来源 {index} · {payload.get('scope_name') or '当前目录'}" for index, payload in enumerate(payloads, 1)]
    for index, payload in enumerate(payloads):
        if (payload.get("owner_digest") != organize_operation_owner_digest(owner)
                or payload.get("credential_generation") != int(client.credential_generation)):
            raise GuangYaWorkspaceStale("合并观察的会话或账号已变化")
        for item in payload.get("entries") or ():
            key = (str(item["file_id"]), str(item.get("parent_id") or "0"), bool(item["is_dir"]))
            previous = entries.get(key)
            if previous is not None and any(previous.get(field) != item.get(field)
                    for field in ("name", "size", "etag", "updated_at")):
                raise GuangYaWorkspaceStale("重新读取期间同一对象发生变化，请重新核对")
            entry = deepcopy(item)
            relative = str(entry.get("relative_parent") or "")
            entry["relative_parent"] = scope_names[index] + (f" › {relative}" if relative not in {"", "当前目录", "当前对象"} else "")
            entries[key] = entry
        for root in payload.get("roots") or ():
            roots[(str(root["file_id"]), str(root["path"]))] = root
        sources.update({item["observation_ref"]: item for item in observation_sources(payload)})
    return _store_observation(client, owner=owner, scope_path="/", scope_name="重新观察的范围",
        scope_names=scope_names,
        roots=list(roots.values()), max_items=_MAX_SCANNED_ITEMS, operation="reobserve", recursive=False, query="",
        truncated=len(entries) > _MAX_SCANNED_ITEMS or any(payload.get("truncated") for payload in payloads),
        scanned_dirs=sum(int(payload.get("scanned_dirs") or 0) for payload in payloads),
        scanned_items=sum(int(payload.get("scanned_items") or 0) for payload in payloads),
        entries=sorted(entries.values(), key=lambda item: str(item["file_id"]) not in preferred_ids)[:_MAX_SCANNED_ITEMS],
        source_observations=list(sources.values()))


def observation_page(
    payload: dict[str, Any], *, page: int = 1, page_size: int = _MAX_PAGE_SIZE
) -> dict[str, Any]:
    safe_size = max(1, min(int(page_size), _MAX_PAGE_SIZE))
    safe_page = max(1, int(page))
    entries = list(payload.get("entries") or [])
    start = (safe_page - 1) * safe_size
    selected = entries[start : start + safe_size]
    public_entries = [
        {
            "object_ref": str(item.get("handle") or ""),
            "object_name": str(item.get("name") or "")[:255],
            "kind": str(item.get("media_kind") or "other"),
            "extension": str(item.get("extension") or ""),
            "size": max(0, int(item.get("size") or 0)),
            "location": str(item.get("relative_parent") or "当前目录")[:240],
        }
        for item in selected
    ]
    return {
        "observation_ref": observation_ref(str(payload.get("plan_id") or "")),
        "scope": str(payload.get("scope_name") or "当前目录")[:120],
        "scopes": [str(item)[:120] for item in payload.get("scope_names") or ()],
        "operation": str(
            payload.get("operation") or ("tree" if payload.get("recursive") else "list")
        ),
        "query": str(payload.get("query") or "")[:160],
        "kinds": [str(item) for item in payload.get("kinds") or ()],
        "max_depth": max(0, int(payload.get("max_depth") or 0)),
        "recursive": bool(payload.get("recursive")),
        "page": safe_page,
        "page_size": safe_size,
        "total": len(entries),
        "scanned_items": max(0, int(payload.get("scanned_items") or len(entries))),
        "has_more": start + len(selected) < len(entries),
        "truncated": bool(payload.get("truncated")),
        "entries": public_entries,
        "cloud_write": False,
    }


def observation_entry_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in payload.get("entries") or []:
        if not isinstance(raw, dict):
            continue
        handle = str(raw.get("handle") or "").strip().upper()
        if valid_object_handle(handle) and handle not in result:
            result[handle] = dict(raw)
    return result
