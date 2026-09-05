"""作品演职员的有界只读查询；缺字段、空表与请求失败不混为一谈。"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from app import config
from app.agent.errors import AgentToolError
from app.agent.models import Evidence, ToolResult
from app.agent.public_safety import sanitize_public_text
from app.clients.tmdb import TMDBClient, close_tmdb_client
from app.discovery.models import ProviderError, ProviderNotConfigured

_ALLOWED = {"tmdb_id", "media_type", "season_number", "cast_limit", "crew_limit"}


def media_credits_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict) or set(arguments) - _ALLOWED:
        raise AgentToolError("演职员查询参数无效")
    tmdb_id = arguments.get("tmdb_id")
    if (
        not isinstance(tmdb_id, str)
        or not tmdb_id.isascii()
        or not tmdb_id.isdigit()
        or not 1 <= len(tmdb_id) <= 10
        or int(tmdb_id) <= 0
    ):
        raise AgentToolError("tmdb_id 必须是已查询到的 1 到 10 位正整数 ID 字符串")
    media_type = arguments.get("media_type")
    if not isinstance(media_type, str) or media_type not in {"movie", "tv"}:
        raise AgentToolError("media_type 仅支持 movie 或 tv")
    clean = {"tmdb_id": tmdb_id, "media_type": media_type}
    for field, default in (("cast_limit", 20), ("crew_limit", 15)):
        value = arguments.get(field, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 50
        ):
            raise AgentToolError(f"{field} 必须是 1 到 50 的整数")
        clean[field] = value
    if "season_number" in arguments:
        season = arguments["season_number"]
        if (
            media_type != "tv"
            or isinstance(season, bool)
            or not isinstance(season, int)
            or not 0 <= season <= 100
        ):
            raise AgentToolError("season_number 仅适用于剧集，必须是 0 到 100 的整数")
        clean["season_number"] = season
    return clean


def _text(value: object, limit: int = 120) -> str:
    if not isinstance(value, str):
        return ""
    return sanitize_public_text(value, limit=limit)


def _count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _person(raw: Any, *, cast: bool) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not (name := _text(raw.get("name"))):
        return None
    person_id = _count(raw.get("id"))
    item: dict[str, Any] = {
        "name": name,
        "original_name": _text(raw.get("original_name")),
        "tmdb_person_id": str(person_id) if person_id else "",
    }
    # 聚合表的 roles/jobs 与电影的 character/job 都来自真实结构化字段。
    list_key, value_key = ("roles", "character") if cast else ("jobs", "job")
    raw_roles = raw.get(list_key)
    if raw_roles is None:
        value = _text(raw.get(value_key))
        roles = [{value_key: value}] if value else []
        incomplete = not bool(value)
        total = len(roles)
    elif isinstance(raw_roles, list):
        roles = [
            {
                value_key: _text(role.get(value_key)),
                "episode_count": _count(role.get("episode_count")),
            }
            for role in raw_roles
            if isinstance(role, dict) and _text(role.get(value_key))
        ]
        total = len(raw_roles)
        incomplete = not roles or len(roles) != total
    else:
        roles, total, incomplete = [], 0, True
    item[list_key] = roles[:20]
    item["roles_total"] = total
    item["roles_truncated"] = len(roles) > 20
    item["role_details_incomplete"] = incomplete
    item["episode_count"] = _count(raw.get("total_episode_count"))
    if cast:
        item["order"] = _count(raw.get("order"))
    else:
        item["department"] = _text(raw.get("department"), 80)
    return item


def _project_people(
    rows: list[Any], *, cast: bool, limit: int
) -> tuple[list[dict], dict]:
    people = [person for row in rows if (person := _person(row, cast=cast))]
    if cast:
        people.sort(key=lambda row: row["order"] if row["order"] is not None else 10**9)
    else:
        # 只对结构化职务排序，优先展示导演及编剧，不凭名字或作品类型猜角色。
        people.sort(
            key=lambda row: (
                0
                if any(job["job"] == "Director" for job in row["jobs"])
                else 1
                if row["department"] == "Writing"
                else 2
            )
        )
    selected = people[:limit]
    discarded = len(rows) - len(people)
    truncated = len(people) > limit or any(item["roles_truncated"] for item in selected)
    incomplete = bool(discarded) or any(
        item["role_details_incomplete"] for item in selected
    )
    return selected, {
        "state": "empty" if not rows else "partial" if incomplete else "available",
        "returned": len(selected),
        "total": len(rows),
        "discarded_invalid": discarded,
        "truncated": truncated,
        "complete": not incomplete and not truncated,
    }


def get_media_credits(arguments: dict[str, Any]) -> ToolResult:
    clean = media_credits_arguments(arguments)
    scope = (
        "season_aggregate"
        if "season_number" in clean
        else "series_aggregate"
        if clean["media_type"] == "tv"
        else "movie"
    )
    base = {
        "tmdb_id": clean["tmdb_id"],
        "media_type": clean["media_type"],
        "season_number": clean.get("season_number"),
        "scope": scope,
        "source": "tmdb",
        "credits_state": "not_queried",
        "complete": False,
    }
    if not config.get_bool("DISCOVERY_ENABLED"):
        return ToolResult(
            False, "disabled", "影视探索功能当前已关闭，尚未查询演职员", data=base
        )

    client = None
    try:
        client = TMDBClient()
        payload = client.media_credits(
            clean["tmdb_id"],
            clean["media_type"],
            season_number=clean.get("season_number"),
            deadline_at=time.monotonic() + 20.0,
            retries=0,
        )
    except ProviderError as exc:
        base["credits_state"] = (
            "not_queried" if isinstance(exc, ProviderNotConfigured) else "failed"
        )
        return ToolResult(
            False,
            exc.code,
            "TMDB 演职员查询未完成",
            data=base,
            error=_text(exc.safe_message, 200),
            evidence=[
                Evidence(
                    "tmdb_credits",
                    "本次查询未取得有效演职员表，不代表源站为空或官方未公布。",
                    datetime.now().astimezone().isoformat(timespec="seconds"),
                )
            ],
            suggestions=[
                "可用 web.search 查询官方平台演员表，并用 web.read 核实来源。"
            ],
        )
    finally:
        close_tmdb_client(client)

    cast, cast_info = _project_people(
        payload["cast"], cast=True, limit=clean["cast_limit"]
    )
    crew, crew_info = _project_people(
        payload["crew"], cast=False, limit=clean["crew_limit"]
    )
    collected_at = datetime.now().astimezone().isoformat(timespec="seconds")
    complete = cast_info["complete"] and crew_info["complete"]
    empty = cast_info["total"] == crew_info["total"] == 0
    state = "empty" if empty else "available" if complete else "partial"
    return ToolResult(
        True,
        "partial" if not complete else "completed",
        "TMDB 本次返回空演职员表"
        if empty
        else f"已读取 TMDB 演职员：演员 {len(cast)} 位、主创 {len(crew)} 项",
        data={
            **base,
            "credits_state": state,
            "cast": cast,
            "crew": crew,
            "cast_coverage": cast_info,
            "crew_coverage": crew_info,
            "complete": complete,
            "truncated": cast_info["truncated"] or crew_info["truncated"],
            "collected_at": collected_at,
            "coverage_note": "完整性仅指本次 TMDB 返回数据的展示；空表不代表官方未官宣，剧集聚合表可能涵盖不同季的演员。",
        },
        evidence=[
            Evidence(
                "tmdb_credits",
                f"读取精确作品身份的 {scope} 演职员表；未查询用户媒体库或执行写操作。",
                collected_at,
            )
        ],
        suggestions=[
            "演员表为空或与已知事实冲突时，可使用 web.search / web.read 核对官方平台。"
        ]
        if empty or not complete
        else [],
    )
