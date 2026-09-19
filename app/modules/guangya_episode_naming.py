"""把紧凑的剧集分季映射编译为通用光鸭文件变更操作。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.modules.scraper import parse_release_position

_MAX_MEDIA_OPERATIONS = 200
_MAX_CREATE_DIRECTORY_OPERATIONS = 32


class GuangYaEpisodeNamingError(ValueError):
    """声明式剧集命名方案无法安全编译。"""


def _normalize_path(value: object, *, field: str) -> str:
    path = str(value or "").strip().replace("\\", "/")
    if not path:
        raise GuangYaEpisodeNamingError(f"{field} 不能为空")
    if not path.startswith("/"):
        path = "/" + path
    parts = [part for part in path.split("/") if part]
    if len(path) > 2048 or any(part in {".", ".."} for part in parts):
        raise GuangYaEpisodeNamingError(f"{field} 必须是精确光鸭绝对路径")
    return "/" + "/".join(parts) if parts else "/"


def _full_path(parent: str, name: str) -> str:
    return f"/{name}" if parent == "/" else f"{parent.rstrip('/')}/{name}"


def _desired_name(title: str, season: int, episode: int, entry: dict[str, Any]) -> str:
    suffix = Path(str(entry.get("name") or "")).suffix
    if not suffix:
        extension = str(entry.get("extension") or "").strip().lstrip(".")
        suffix = f".{extension}" if extension else ""
    return f"{title} - S{season:02d}E{episode:02d}{suffix}"


def _episode_position(entry: dict[str, Any]) -> tuple[int | None, int | None]:
    parsed = parse_release_position(
        str(entry.get("name") or ""), tv_episode_mapping_context=True
    )
    season = parsed.get("season")
    episode = parsed.get("episode")
    return (
        int(season) if isinstance(season, int) else None,
        int(episode) if isinstance(episode, int) else None,
    )



def _compress_episode_numbers(values: list[int]) -> str:
    numbers = sorted(set(values))
    if not numbers:
        return ""
    chunks: list[str] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        chunks.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = number
    chunks.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(chunks)


def summarize_episode_naming_observation(
    observation: dict[str, Any], *, target_root: str
) -> dict[str, Any]:
    """把完整目录快照压缩为供模型判断篇章映射的小型 DTO。"""

    if bool(observation.get("truncated")):
        raise GuangYaEpisodeNamingError("光鸭目录观察不完整，请扩大 max_items 后重新读取")
    normalized_root = _normalize_path(target_root, field="target_root")
    grouped: dict[str, list[dict[str, Any]]] = {}
    entries = [item for item in observation.get("entries") or () if isinstance(item, dict)]
    for entry in entries:
        if bool(entry.get("is_dir")) or str(entry.get("media_kind") or "") != "video":
            continue
        parent_path = _normalize_path(entry.get("parent_path"), field="parent_path")
        if parent_path != normalized_root and not parent_path.startswith(
            normalized_root.rstrip("/") + "/"
        ):
            continue
        grouped.setdefault(parent_path, []).append(entry)

    summaries: list[dict[str, Any]] = []
    total_videos = 0
    total_unparsed = 0
    for parent_path, items in sorted(
        grouped.items(), key=lambda pair: (pair[0] != normalized_root, pair[0].casefold())
    ):
        positions: dict[int | None, list[int]] = {}
        unparsed = 0
        small_videos = 0
        samples: list[str] = []
        for entry in sorted(items, key=lambda item: str(item.get("name") or "").casefold()):
            name = str(entry.get("name") or "")
            if len(samples) < 3:
                samples.append(name)
            size = entry.get("size")
            if isinstance(size, int) and 0 <= size < 5 * 1024 * 1024:
                small_videos += 1
            season, episode = _episode_position(entry)
            if episode is None:
                unparsed += 1
            else:
                positions.setdefault(season, []).append(episode)
        total_videos += len(items)
        total_unparsed += unparsed
        summaries.append(
            {
                "source_path": parent_path,
                "directory_name": "(共同父目录)"
                if parent_path == normalized_root
                else Path(parent_path).name,
                "video_count": len(items),
                "parsed_count": len(items) - unparsed,
                "unparsed_count": unparsed,
                "small_video_count": small_videos,
                "positions": [
                    {
                        "source_season": season,
                        "episodes": _compress_episode_numbers(episodes),
                        "count": len(set(episodes)),
                    }
                    for season, episodes in sorted(
                        positions.items(), key=lambda pair: (-1 if pair[0] is None else pair[0])
                    )
                ],
                "samples": samples,
            }
        )
    if not summaries:
        raise GuangYaEpisodeNamingError("目标目录中没有可用于剧集命名盘点的视频")
    return {
        "target_root": normalized_root,
        "video_count": total_videos,
        "source_group_count": len(summaries),
        "unparsed_count": total_unparsed,
        "groups": summaries,
    }

def compile_episode_naming_operations(
    observation: dict[str, Any],
    *,
    title: str,
    target_root: str,
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """按目录和集号区间确定性展开 rename/relocate/create_directory。

    输入只描述篇章到 TMDB 季集的映射；对象引用、扩展名、目标名称和目录
    创建动作均从 owner-bound 观察快照中生成，避免模型逐文件拼装大 JSON。
    """

    if bool(observation.get("truncated")):
        raise GuangYaEpisodeNamingError("光鸭目录观察不完整，请扩大 max_items 后重新读取")
    normalized_title = str(title or "").strip()
    if not 1 <= len(normalized_title) <= 180:
        raise GuangYaEpisodeNamingError("title 长度必须在 1 到 180 之间")
    normalized_root = _normalize_path(target_root, field="target_root")
    if not isinstance(groups, list) or not 1 <= len(groups) <= 32:
        raise GuangYaEpisodeNamingError("groups 必须包含 1 到 32 个篇章映射")

    entries = [item for item in observation.get("entries") or () if isinstance(item, dict)]
    existing_directories = {
        _normalize_path(item.get("parent_path"), field="parent_path")
        for item in entries
        if str(item.get("parent_path") or "").strip()
    }
    existing_directories.update(
        _full_path(
            _normalize_path(item.get("parent_path"), field="parent_path"),
            str(item.get("name") or ""),
        )
        for item in entries
        if bool(item.get("is_dir")) and str(item.get("name") or "").strip()
    )
    used_handles: set[str] = set()
    target_names: set[tuple[str, str]] = set()
    create_paths: set[str] = set()
    create_operations: list[dict[str, Any]] = []
    change_operations: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    skipped_noop = 0

    for index, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            raise GuangYaEpisodeNamingError(f"第 {index} 个篇章映射格式无效")
        raw_source_path = str(group.get("source_path") or "").strip()
        directory_contains = str(group.get("source_directory_contains") or "").strip()
        if bool(raw_source_path) == bool(directory_contains):
            raise GuangYaEpisodeNamingError(
                f"第 {index} 个篇章映射必须且只能提供 source_path 或 source_directory_contains"
            )
        if raw_source_path:
            source_path = _normalize_path(raw_source_path, field="source_path")
        else:
            candidates = {
                _normalize_path(entry.get("parent_path"), field="parent_path")
                for entry in entries
                if not bool(entry.get("is_dir"))
                and str(entry.get("media_kind") or "") == "video"
                and directory_contains.casefold()
                in Path(str(entry.get("parent_path") or "")).name.casefold()
            }
            candidates = {
                path
                for path in candidates
                if path == normalized_root or path.startswith(normalized_root.rstrip("/") + "/")
            }
            if len(candidates) != 1:
                raise GuangYaEpisodeNamingError(
                    f"第 {index} 个篇章目录特征匹配到 {len(candidates)} 个目录，请提供更精确特征"
                )
            source_path = next(iter(candidates))
        target_season = int(group["target_season"])
        source_start = int(group["source_episode_start"])
        source_end = int(group["source_episode_end"])
        target_start = int(group.get("target_episode_start", 1))
        source_season = group.get("source_season")
        if source_season is not None:
            source_season = int(source_season)
        name_contains = str(group.get("name_contains") or "").strip()
        expected_count = group.get("expected_count")
        if expected_count is not None:
            expected_count = int(expected_count)

        selected: list[tuple[int, dict[str, Any]]] = []
        unparsed = 0
        for entry in entries:
            if bool(entry.get("is_dir")) or str(entry.get("media_kind") or "") != "video":
                continue
            if _normalize_path(entry.get("parent_path"), field="parent_path") != source_path:
                continue
            if name_contains and name_contains.casefold() not in str(entry.get("name") or "").casefold():
                continue
            parsed_season, parsed_episode = _episode_position(entry)
            if parsed_episode is None:
                unparsed += 1
                continue
            if source_season is not None and parsed_season != source_season:
                continue
            if source_season is None and target_season != 0 and parsed_season == 0:
                # 未明确源季时，非特别篇目标季不能默默吸收 S00/番外。
                continue
            if source_start <= parsed_episode <= source_end:
                selected.append((parsed_episode, entry))

        selected.sort(key=lambda pair: (pair[0], str(pair[1].get("name") or "").casefold()))
        if not selected:
            detail = "，且存在无法识别集号的文件" if unparsed else ""
            raise GuangYaEpisodeNamingError(f"第 {index} 个篇章映射没有匹配到正片{detail}")
        if expected_count is not None and len(selected) != expected_count:
            raise GuangYaEpisodeNamingError(
                f"第 {index} 个篇章映射预期 {expected_count} 集，实际匹配 {len(selected)} 集"
            )

        episodes: set[int] = set()
        relocate_items: list[dict[str, Any]] = []
        rename_items: list[dict[str, Any]] = []
        season_directory = f"Season {target_season:02d}"
        target_path = _full_path(normalized_root, season_directory)
        for source_episode, entry in selected:
            handle = str(entry.get("handle") or "").strip().upper()
            if not handle or handle in used_handles:
                raise GuangYaEpisodeNamingError("篇章映射包含重复或无效对象")
            if source_episode in episodes:
                raise GuangYaEpisodeNamingError(
                    f"第 {index} 个篇章映射存在重复源集号 E{source_episode:02d}"
                )
            target_episode = target_start + source_episode - source_start
            if not 1 <= target_episode <= 9999:
                raise GuangYaEpisodeNamingError("映射后的目标集号超出 1 到 9999")
            desired = _desired_name(normalized_title, target_season, target_episode, entry)
            target_key = (target_path.casefold(), desired.casefold())
            if target_key in target_names:
                raise GuangYaEpisodeNamingError(
                    f"多个篇章映射会生成同一目标：S{target_season:02d}E{target_episode:02d}"
                )
            target_names.add(target_key)
            used_handles.add(handle)
            episodes.add(source_episode)
            current_parent = _normalize_path(entry.get("parent_path"), field="parent_path")
            current_name = str(entry.get("name") or "")
            if current_parent == target_path:
                if current_name == desired:
                    skipped_noop += 1
                else:
                    rename_items.append(
                        {"op": "rename", "object_ref": handle, "new_name": desired}
                    )
            else:
                relocate_items.append({"object_ref": handle, "episode": target_episode})

        if (relocate_items or rename_items) and target_path not in existing_directories and target_path not in create_paths:
            create_paths.add(target_path)
            create_operations.append(
                {"op": "create_directory", "parent_path": normalized_root, "name": season_directory}
            )
        change_operations.extend(rename_items)
        if relocate_items:
            change_operations.append(
                {
                    "op": "batch_relocate",
                    "items": relocate_items,
                    "target_path": target_path,
                    "title": normalized_title,
                    "naming": "season_episode",
                    "season": target_season,
                    "episode_padding": 2,
                }
            )
        group_summaries.append(
            {
                "source_directory": "(共同父目录)"
                if source_path == normalized_root
                else Path(source_path).name,
                "season": target_season,
                "matched": len(selected),
                "renamed_in_place": len(rename_items),
                "relocated": len(relocate_items),
                "source_episode_start": selected[0][0],
                "source_episode_end": selected[-1][0],
                "target_episode_start": target_start,
                "target_episode_end": target_start + selected[-1][0] - source_start,
            }
        )

    operations = [*create_operations, *change_operations]
    effective_total = len(create_operations) + sum(
        len(item.get("items") or ()) if item.get("op") == "batch_relocate" else 1
        for item in change_operations
    )
    if effective_total == 0:
        raise GuangYaEpisodeNamingError("所选文件已经符合目标分季命名，无需变更")
    if len(used_handles) > _MAX_MEDIA_OPERATIONS:
        raise GuangYaEpisodeNamingError(
            f"完整方案需要变更 {len(used_handles)} 个媒体文件，单个冻结计划最多 "
            f"{_MAX_MEDIA_OPERATIONS} 个；请按完整季拆分 groups，不能截断同一季"
        )
    if len(create_operations) > _MAX_CREATE_DIRECTORY_OPERATIONS:
        raise GuangYaEpisodeNamingError(
            f"完整方案需要创建 {len(create_operations)} 个目录，单个冻结计划最多 "
            f"{_MAX_CREATE_DIRECTORY_OPERATIONS} 个"
        )
    return {
        "operations": operations,
        "effective_total": effective_total,
        "selected_files": len(used_handles),
        "created_directories": len(create_operations),
        "skipped_noop": skipped_noop,
        "groups": group_summaries,
    }
