"""Jellyfin 主动删除：索引确定身份，统一 Writer 内校验并移入回收站。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from app import config
from app.clients.guangya import GuangYaClient, close_guangya_client
from app.logger import get_logger
from app.modules.directory_scrape_errors import (
    DirectoryScrapeConflictError,
    DirectoryScrapePublicError,
    DirectoryScrapeRequestError,
    public_error_message,
)
from app.modules.download_staging_cleanup import assert_guangya_download_delete_allowed
from app.modules.media_server_path_mapping import (
    parse_media_server_path_mappings,
    reverse_media_server_path_mapping,
)
from app.modules.organize_delete_audit import (
    DeleteCandidate,
    execute_recycle_bin_delete,
    record_blocked_delete,
)
from app.modules.strm import (
    STRM_SUBDIR,
    _fingerprint_matches,
    configured_strm_source_plans,
    trigger_cloud_changes,
)
from app.repositories.strm import list_strm_file_owners, list_strm_path_owners

logger = get_logger(__name__)
MAX_DELETE_PATHS = 1000


@dataclass(frozen=True)
class DeleteTarget:
    path: str
    source_id: str
    file_id: str
    name: str
    etag: str
    size: int
    fingerprint: str


def _indexed_target(path: str, row) -> DeleteTarget:
    return DeleteTarget(
        path, str(row["source"]).removeprefix("guangya:"), str(row["file_id"]),
        str(row["filename"]), str(row["etag"]), int(row["size"]), str(row["content_fingerprint"]),
    )


def prepare_delete(payload: dict) -> tuple[DeleteTarget, ...]:
    """只接受有当前索引、内容指纹及来源的 STRM，不接受目录或裸云盘 ID。"""
    paths = payload.get("paths")
    if not isinstance(paths, list) or not 1 <= len(paths) <= MAX_DELETE_PATHS:
        raise DirectoryScrapeRequestError(f"请选择 1–{MAX_DELETE_PATHS} 个 STRM 文件")
    if any(not isinstance(path, str) or not path.strip() or len(path) > 4096 for path in paths):
        raise DirectoryScrapeRequestError("STRM 路径无效")
    fingerprints = payload.get("fingerprints")
    if not isinstance(fingerprints, dict) or any(
        not isinstance(fingerprints.get(path), str) or len(fingerprints[path]) != 71
        for path in paths
    ):
        raise DirectoryScrapeRequestError("请提供 Jellyfin 实际 STRM 文件的 SHA256 指纹")
    mode = payload.get("mapping_mode", "mf")
    if mode == "mf":
        raw_mappings = config.get("JELLYFIN_PATH_MAPPINGS", "")
    elif mode == "manual" and isinstance(payload.get("mappings"), list):
        raw_mappings = json.dumps(payload["mappings"], ensure_ascii=False)
    else:
        raise DirectoryScrapeRequestError("请选择 MF 映射或手动映射")
    try:
        mappings = parse_media_server_path_mappings(raw_mappings)
        local_paths = {}
        for path in paths:
            local_path = reverse_media_server_path_mapping(path, mappings)[0]
            fingerprint = fingerprints[path].lower()
            if local_path in local_paths and local_paths[local_path] != fingerprint:
                raise DirectoryScrapeRequestError("同一 MF 路径对应的 STRM 指纹不一致")
            local_paths[local_path] = fingerprint
    except ValueError as exc:
        raise DirectoryScrapeRequestError(str(exc)) from exc

    root_text = str(config.get("STRM_ROOT", "") or "").strip()
    if not root_text:
        raise DirectoryScrapeConflictError("尚未配置 STRM 输出目录")
    root = (Path(root_text).expanduser() / STRM_SUBDIR).resolve()
    sources, source_error = configured_strm_source_plans()
    if source_error:
        raise DirectoryScrapeConflictError("STRM 来源配置无效，请先检查 MF 配置")
    source_ids = {str(source["id"]) for source in sources}
    targets = []
    for text in sorted(local_paths):
        path = Path(text)
        if not path.is_absolute() or path.suffix.lower() != ".strm":
            raise DirectoryScrapeRequestError("只能删除已纳管的 STRM 文件，不能提交目录")
        # 不沿符号链接删除目标身份；MF/Jellyfin 必须明确指向同一纳管文件。
        if path.resolve() != path or not path.is_relative_to(root):
            raise DirectoryScrapeConflictError("STRM 路径越界或经过符号链接")
        rows = list_strm_path_owners(str(path))
        if len(rows) != 1:
            raise DirectoryScrapeConflictError("STRM 索引缺失或存在多个所有者，请先同步核对")
        row = dict(rows[0])
        source = str(row["source"])
        source_id = source.removeprefix("guangya:")
        if not source.startswith("guangya:") or source_id not in source_ids:
            raise DirectoryScrapeConflictError("文件不属于当前配置的光鸭 STRM 来源")
        if not row["file_id"] or str(row["file_id"]) == "0" or not row["etag"]:
            raise DirectoryScrapeConflictError("STRM 索引缺少云端文件身份，请先同步")
        if str(row["content_fingerprint"]) != local_paths[text]:
            raise DirectoryScrapeConflictError("Jellyfin 与 MF 的 STRM 内容不一致，请检查路径映射")
        if not _fingerprint_matches(path, row["content_fingerprint"]):
            raise DirectoryScrapeConflictError("STRM 内容已变化或文件不存在，请先同步核对")
        targets.append(_indexed_target(str(path), row))
    for file_id in {target.file_id for target in targets}:
        if any(str(row["strm_path"]) not in local_paths for row in list_strm_file_owners(file_id)):
            raise DirectoryScrapeConflictError("云端文件还被其他 STRM 引用，不能单独联动删除")
    return tuple(targets)


def delete_key(targets: tuple[DeleteTarget, ...]) -> str:
    identities = [(item.path, item.file_id, item.etag, item.fingerprint) for item in targets]
    return "jellyfin-delete:" + hashlib.sha256(json.dumps(identities).encode()).hexdigest()


def _snapshot(item) -> tuple:
    return (item.file_id, item.name, item.parent_id, item.is_dir) + (
        () if item.is_dir else (item.size, item.etag)
    )


def _verify_cloud_file(client, target: DeleteTarget, directories: dict, parents: dict):
    item = client.file_info(target.file_id)
    if item is None or item.is_dir or str(item.file_id) != target.file_id:
        raise DirectoryScrapeConflictError("云端文件已不存在或类型变化；不会重试回收站删除")
    if (item.name, str(item.etag), int(item.size)) != (target.name, target.etag, target.size):
        raise DirectoryScrapeConflictError("云端文件与 STRM 索引不一致，请先同步核对")
    current = item
    ancestors = {"0"}
    visited = {target.file_id}
    while True:
        parent_id = str(current.parent_id)
        if not parent_id or parent_id in visited or len(visited) > 256:
            raise DirectoryScrapeConflictError("无法确认云端文件的正常目录归属")
        # file_info 可能仍能读到回收站对象；必须在正常祖先链的目录列表中确认。
        if parent_id not in directories:
            directories[parent_id] = {str(child.file_id): child for child in client.list_dir(parent_id)}
        listed = directories[parent_id].get(str(current.file_id))
        if listed is None or _snapshot(listed) != _snapshot(current):
            raise DirectoryScrapeConflictError("云端文件或目录已变化，或已在回收站中；已取消删除")
        ancestors.add(parent_id)
        if parent_id == "0":
            break
        visited.add(parent_id)
        if parent_id not in parents:
            parents[parent_id] = client.file_info(parent_id)
        current = parents[parent_id]
        if current is None or not current.is_dir or str(current.file_id) != parent_id:
            raise DirectoryScrapeConflictError("无法确认云端文件的父目录")
    if target.source_id not in ancestors:
        raise DirectoryScrapeConflictError("云端文件已离开已配置 STRM 来源，已取消删除")
    candidate = DeleteCandidate(item.file_id, item.name, item.parent_id, item.size, item.etag)
    try:
        assert_guangya_download_delete_allowed(client, item)
    except DirectoryScrapeConflictError as exc:
        record_blocked_delete(trigger="jellyfin_delete", reason=public_error_message(exc), candidate=candidate)
        raise
    return candidate


def execute_delete(payload: dict, expected: tuple[DeleteTarget, ...]) -> dict:
    """调用方必须持有统一 Writer；先检查整批，再逐文件执行，部分失败如实返回。"""
    if prepare_delete(payload) != expected:
        raise DirectoryScrapeConflictError("映射或 STRM 索引已变化，请重新发起删除")
    client = GuangYaClient()
    completed = []
    attempted_sources: set[str] = set()
    error = ""
    stats: dict[str, int] = {}
    try:
        if not client.logged_in:
            raise DirectoryScrapeConflictError("光鸭未登录，请先在 MF 登录")
        directories, parents = {}, {}
        references: dict[str, list[DeleteTarget]] = {}
        candidates = {}
        # 任何可预先发现的错误都应在第一笔云端写入之前拒绝。
        for target in expected:
            candidates[target.file_id] = _verify_cloud_file(client, target, directories, parents)
            references.setdefault(target.file_id, []).append(target)
        if prepare_delete(payload) != expected:
            raise DirectoryScrapeConflictError("删除期间映射或 STRM 内容发生变化，已停止后续删除")
        selected_paths = {target.path for target in expected}
        for file_id, targets in references.items():
            if any(str(row["strm_path"]) not in selected_paths for row in list_strm_file_owners(file_id)):
                raise DirectoryScrapeConflictError("云端文件出现新的 STRM 引用，已停止后续删除")
            # 每项仅检查自身；不能在循环中重新扫描整批形成 O(n²) 索引/文件读取。
            directories, parents = {}, {}
            for target in targets:
                rows = list_strm_path_owners(target.path)
                if (Path(target.path).resolve() != Path(target.path)
                    or len(rows) != 1
                    or _indexed_target(target.path, rows[0]) != target
                    or not _fingerprint_matches(Path(target.path), target.fingerprint)):
                    raise DirectoryScrapeConflictError("执行前 STRM 身份发生变化，已停止后续删除")
                candidate = _verify_cloud_file(client, target, directories, parents)
                if candidate != candidates[file_id]:
                    raise DirectoryScrapeConflictError("云端对象在执行前发生变化，已停止后续删除")
            attempted_sources.update(target.source_id for target in targets)
            audit = execute_recycle_bin_delete(
                client, trigger="jellyfin_delete", reason="Jellyfin 用户主动确认删除对应节目",
                candidate=candidate, safe_failure_message="云端删除结果未能确认，请在 MF 核对删除审计，不要盲目重试",
            )
            completed.append({"file_id": file_id, "audit_id": audit["audit_id"]})
    except Exception as exc:  # noqa: BLE001 - 写结果未知也必须返回部分回执并触发来源校准。
        error = public_error_message(exc) if isinstance(exc, DirectoryScrapePublicError) else (
            "云端删除结果未能确认，请在 MF 核对删除审计，不要盲目重试"
            if attempted_sources else "删除前校验失败，未执行云端删除，请检查光鸭连接或文件状态"
        )
        logger.warning("Jellyfin 联动删除未完成 type=%s completed=%s", type(exc).__name__, len(completed))
    finally:
        close_guangya_client(client)
        if attempted_sources:
            # 仅复用既有来源校准；含不确定写结果时也要重新观察，不直接清理本地索引。
            operations = [{"op": "trash", "source": {"parent_id": source_id}} for source_id in attempted_sources]
            stats = trigger_cloud_changes(operations, sources={source_id: "/" for source_id in attempted_sources})
    if stats.get("strm_trigger_failed"):
        error = error or "云端文件已移入回收站，但 STRM 联动未启动，请在 MF 手动同步"
    return {
        "ok": not error, "completed": len(completed), "error": error, "items": completed,
        "stats": {"deleted": len(completed), **stats},
    }
