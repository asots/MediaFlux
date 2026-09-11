"""季集研究的确定性证据边界；模型不能提供任意源集到目标集的改写。"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import unicodedata
from collections import defaultdict
from typing import Any

from app.clients.tmdb import TMDBClient
from app.modules.scraper import extract_recognition_context, parse_release_position
from app.sensitive_data import contains_sensitive_credential

POLICY_VERSION = 1
MAX_FILES = 80
MAX_CANDIDATES = 3
MAX_GROUPS = 6
MAX_GROUP_EPISODES = 1500
_GROUP_ID = re.compile(r"[a-fA-F0-9]{24}\Z")
_NUMERIC_ID = re.compile(r"[0-9]{1,12}\Z")


class EpisodeResearchError(ValueError):
    def __init__(self, code: str, message: str = "季集研究缺少可核验证据"):
        super().__init__(message)
        self.code = code


def _integer(value: object, *, minimum: int = 0, maximum: int = 9999) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise EpisodeResearchError("invalid_integer", "元数据中的编号必须是有效整数")
    return value


def _identity(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise EpisodeResearchError("invalid_identity")
    text = str(value).strip()
    if not _NUMERIC_ID.fullmatch(text) or int(text) <= 0:
        raise EpisodeResearchError("invalid_identity")
    return str(int(text))


def _text(value: object, *, limit: int = 300) -> str:
    if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 for c in value):
        raise EpisodeResearchError("invalid_text")
    text = value.strip()
    if contains_sensitive_credential(text):
        raise EpisodeResearchError("unsafe_input", "研究材料疑似包含凭据，已保留人工确认")
    return text


def _title_key(value: str) -> str:
    return "".join(c.casefold() for c in unicodedata.normalize("NFKC", value) if c.isalnum())


def _source_title(value: str, *, context_title: bool = False) -> str:
    context = extract_recognition_context(value)
    title = _text(context.filename_title or context.normalized_title or "", limit=500)
    key = _title_key(title)
    if not key:
        return ""
    # “诛仙”“V”等短标题仍是身份约束，不能被目录覆盖；24.S01E11 中的
    # 24也可能是真实作品名。仅真正无作品前缀的纯集号文件允许上下文兜底。
    if (not context_title and re.fullmatch(r"(?:episode|ep|e)?[0-9]+", key)
            and not re.search(r"(?i)S[0-9]{1,2}E[0-9]{1,4}", value)):
        return ""
    return title


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def normalize_case(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise EpisodeResearchError("invalid_case")
    raw_files = payload.get("files")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_files, list) or not 1 <= len(raw_files) <= MAX_FILES:
        raise EpisodeResearchError("case_file_limit")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise EpisodeResearchError("no_candidate")
    candidates = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_candidates[:MAX_CANDIDATES]):
        if not isinstance(item, dict):
            raise EpisodeResearchError("invalid_candidate")
        if item.get("media_type") != "tv" or str(item.get("provider") or "tmdb").lower() != "tmdb":
            continue
        identity = _identity(item.get("tmdb_id"))
        if identity in seen_ids:
            raise EpisodeResearchError("duplicate_candidate")
        seen_ids.add(identity)
        title = _text(item.get("title") or "", limit=300)
        if not title:
            raise EpisodeResearchError("candidate_title_missing")
        year = _text(str(item.get("year") or ""), limit=4)
        if year and not re.fullmatch(r"[0-9]{4}", year):
            raise EpisodeResearchError("invalid_candidate_year")
        candidates.append({"index": index, "tmdb_id": identity, "title": title, "year": year, "media_type": "tv"})
    if not candidates:
        raise EpisodeResearchError("no_tv_candidate")
    files = []
    positions: set[tuple[int, int]] = set()
    for index, item in enumerate(raw_files):
        if not isinstance(item, dict):
            raise EpisodeResearchError("invalid_file")
        raw_name = _text(item.get("name") or "", limit=1000)
        name = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
        if not name:
            raise EpisodeResearchError("source_name_missing")
        parsed = parse_release_position(name)
        if parsed.get("episode_end") is not None:
            raise EpisodeResearchError("multi_episode_file", "多集合并文件不能自动逐集重命名")
        season = parsed.get("season") if parsed.get("season") is not None else item.get("season")
        episode = parsed.get("episode") if parsed.get("episode") is not None else item.get("episode")
        if season is None or episode is None:
            raise EpisodeResearchError("source_position_incomplete")
        season = _integer(season, maximum=99)
        episode = _integer(episode, minimum=1)
        if (season, episode) in positions:
            raise EpisodeResearchError("duplicate_source_position")
        positions.add((season, episode))
        size = item.get("size", 0)
        if size is None:
            size = 0
        size = _integer(size, maximum=2**63-1)
        files.append({"index": index, "name": name, "source_season": season, "source_episode": episode, "size": size, "source_title": _source_title(name)})
    context_titles = []
    for value in (payload.get("identity"), str(payload.get("directory") or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]):
        if isinstance(value, str) and value.strip():
            title = _source_title(_text(value, limit=500), context_title=True)
            if title:
                context_titles.append(title)
    value = {"files": files, "candidates": candidates, "context_titles": context_titles, "version": POLICY_VERSION}
    return {**value, "case_key": _digest(value)}


def _candidate(case: dict, index: int) -> dict:
    _integer(index, maximum=MAX_CANDIDATES-1)
    for item in case["candidates"]:
        if item["index"] == index:
            return item
    raise EpisodeResearchError("candidate_not_frozen")


def _group_positions(data: dict) -> dict[int, list[dict]]:
    groups = data.get("groups")
    if not isinstance(groups, list) or not 1 <= len(groups) <= 100:
        raise EpisodeResearchError("invalid_group_data")
    positions: dict[int, list[dict]] = {}
    total = 0
    for group in groups:
        if not isinstance(group, dict):
            raise EpisodeResearchError("invalid_group_data")
        season = _integer(group.get("order"), maximum=99)
        if season in positions:
            raise EpisodeResearchError("duplicate_group_order")
        rows = group.get("episodes")
        if not isinstance(rows, list) or not rows:
            raise EpisodeResearchError("invalid_group_data")
        total += len(rows)
        if total > MAX_GROUP_EPISODES:
            raise EpisodeResearchError("group_episode_limit")
        clean = []
        seen_ids: set[str] = set()
        seen_targets: set[tuple[int, int]] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise EpisodeResearchError("invalid_group_data")
            identity = _identity(row.get("id"))
            order = _integer(row.get("order"), maximum=MAX_GROUP_EPISODES)
            target_season = _integer(row.get("season_number"), maximum=99)
            target_episode = _integer(row.get("episode_number"), minimum=1)
            if identity in seen_ids or (target_season, target_episode) in seen_targets:
                raise EpisodeResearchError("duplicate_group_episode")
            seen_ids.add(identity)
            seen_targets.add((target_season, target_episode))
            clean.append({"order": order, "target_season": target_season, "target_episode": target_episode, "episode_id": int(identity)})
        clean.sort(key=lambda x: x["order"])
        if [x["order"] for x in clean] != list(range(len(clean))):
            raise EpisodeResearchError("non_contiguous_group_order")
        positions[season] = clean
    return positions


def _map_group(case: dict, data: dict) -> list[dict] | None:
    groups = _group_positions(data)
    by_season: dict[int, list[int]] = defaultdict(list)
    for item in case["files"]:
        by_season[item["source_season"]].append(item["source_episode"])
    for season, numbers in by_season.items():
        rows = groups.get(season)
        if not rows:
            return None
        numbers.sort()
        if numbers != list(range(numbers[0], numbers[-1]+1)):
            return None
        # 完整包或同一顺序的完整尾包；孤立/缺集/尚在增长的目录不得猜偏移。
        if numbers[-1] != len(rows) or (numbers[0] > 1 and len(numbers) < 3):
            return None
    mappings = []
    targets: set[tuple[int, int]] = set()
    for item in case["files"]:
        row = groups[item["source_season"]][item["source_episode"]-1]
        target = (row["target_season"], row["target_episode"])
        if target in targets:
            raise EpisodeResearchError("duplicate_target_position")
        targets.add(target)
        mappings.append({"file_index": item["index"], "source_season": item["source_season"],
                         "source_episode": item["source_episode"], "target_season": target[0],
                         "target_episode": target[1], "episode_id": row["episode_id"]})
    return mappings


class EpisodeEvidenceReader:
    """固定 TMDB 路径、有限请求、独立生命周期的服务端证据读取器。"""
    def __init__(self, case: dict, *, client=None, max_requests: int = 16, timeout_seconds: float = 90):
        self.case = copy.deepcopy(case)
        self.client = client if client is not None else TMDBClient(timeout=10, retries=0)
        self._owns_client = client is None
        self._deadline = time.monotonic() + min(120, max(0.001, timeout_seconds))
        self._max_requests = _integer(max_requests, minimum=1, maximum=24)
        self._requests = 0
        self._closed = False
        self._resources_closed = False
        self._responses: dict[str, dict] = {}
        self._listed: dict[int, list[dict]] = {}
        self._details: dict[int, dict] = {}

    @property
    def request_count(self) -> int:
        return self._requests

    def close(self) -> bool:
        self._closed = True
        if self._resources_closed:
            return True
        if self._owns_client and self.client.close() is False:
            return False
        self._resources_closed = True
        return True

    def limit_deadline(self, deadline_at: float) -> None:
        self._deadline = min(self._deadline, float(deadline_at))
        self._check()

    def _check(self) -> None:
        if self._closed:
            raise EpisodeResearchError("reader_closed")
        if time.monotonic() >= self._deadline:
            raise EpisodeResearchError("research_timeout")

    def _get(self, path: str) -> dict:
        self._check()
        if path in self._responses:
            return copy.deepcopy(self._responses[path])
        if self._requests >= self._max_requests:
            raise EpisodeResearchError("tmdb_request_budget")
        self._requests += 1
        try:
            data = self.client.get(path, deadline_at=self._deadline, retries=0)
        except Exception as exc:
            raise EpisodeResearchError("tmdb_unavailable", "TMDB 证据读取失败") from exc
        self._check()
        if not isinstance(data, dict) or not data:
            raise EpisodeResearchError("tmdb_invalid_response")
        try:
            if len(json.dumps(data, ensure_ascii=False, allow_nan=False).encode()) > 2*1024*1024:
                raise EpisodeResearchError("tmdb_response_limit")
        except (TypeError, ValueError) as exc:
            if isinstance(exc, EpisodeResearchError):
                raise
            raise EpisodeResearchError("tmdb_invalid_response") from exc
        self._responses[path] = copy.deepcopy(data)
        return data

    def inspect_candidate(self, index: int) -> dict:
        item = _candidate(self.case, index)
        data = self._get(f"/tv/{item['tmdb_id']}")
        if _identity(data.get("id")) != item["tmdb_id"]:
            raise EpisodeResearchError("candidate_identity_mismatch")
        title = _text(data.get("name") or data.get("original_name") or "")
        if not title:
            raise EpisodeResearchError("candidate_title_missing")
        # 冻结候选不能用别的年份/同名翻拍悄悄替换；空年份不当作强证据。
        year = str(data.get("first_air_date") or "")[:4]
        if item["year"] and year and item["year"] != year:
            raise EpisodeResearchError("candidate_year_mismatch")
        aliases = {_title_key(title), _title_key(_text(data.get("original_name") or ""))} - {""}
        candidate_title = _title_key(item["title"])
        source_titles = [_title_key(row["source_title"]) for row in self.case["files"] if row.get("source_title")]
        context_titles = {_title_key(value) for value in self.case.get("context_titles", [])}
        if candidate_title not in aliases or any(value not in aliases for value in source_titles) or (not source_titles and not aliases.intersection(context_titles)):
            alternatives = self._get(f"/tv/{item['tmdb_id']}/alternative_titles")
            if _identity(alternatives.get("id")) != item["tmdb_id"] or not isinstance(alternatives.get("results"), list):
                raise EpisodeResearchError("candidate_alias_invalid")
            for row in alternatives["results"][:100]:
                if isinstance(row, dict):
                    aliases.add(_title_key(_text(row.get("title") or "")))
        if candidate_title not in aliases:
            raise EpisodeResearchError("candidate_title_mismatch")
        if any(value not in aliases for value in source_titles) or (not source_titles and not aliases.intersection(context_titles)):
            raise EpisodeResearchError("source_identity_unproven", "原发布名与候选作品缺少一致的官方标题/别名证据")
        self._details[index] = data
        return {"candidate_index": index, "tmdb_id": item["tmdb_id"], "title": title,
                "original_title": _text(data.get("original_name") or ""), "year": year,
                "seasons": [{"season_number": row.get("season_number"), "episode_count": row.get("episode_count")}
                            for row in list(data.get("seasons") or [])[:100] if isinstance(row, dict)]}

    def list_groups(self, index: int) -> dict:
        item = _candidate(self.case, index)
        if index not in self._details:
            self.inspect_candidate(index)
        data = self._get(f"/tv/{item['tmdb_id']}/episode_groups")
        if _identity(data.get("id")) != item["tmdb_id"]:
            raise EpisodeResearchError("group_candidate_mismatch")
        rows = data.get("results")
        if not isinstance(rows, list) or len(rows) > MAX_GROUPS:
            raise EpisodeResearchError("episode_group_limit")
        groups = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise EpisodeResearchError("invalid_group_index")
            group_id = row.get("id")
            if not isinstance(group_id, str) or not _GROUP_ID.fullmatch(group_id) or group_id in seen:
                raise EpisodeResearchError("invalid_group_identity")
            seen.add(group_id)
            groups.append({"id": group_id, "name": _text(row.get("name") or "", limit=200),
                           "group_count": _integer(row.get("group_count"), minimum=1, maximum=100),
                           "episode_count": _integer(row.get("episode_count"), minimum=1, maximum=MAX_GROUP_EPISODES)})
        self._listed[index] = groups
        return {"candidate_index": index, "tmdb_id": item["tmdb_id"], "groups": copy.deepcopy(groups)}

    def _load_group(self, index: int, group_id: str) -> dict:
        self._check()
        rows = self._listed.get(index)
        if rows is None:
            raise EpisodeResearchError("group_index_not_read")
        listed = next((row for row in rows if row["id"] == group_id), None)
        if listed is None:
            raise EpisodeResearchError("group_not_listed")
        data = self._get(f"/tv/episode_group/{group_id}")
        if data.get("id") != group_id:
            raise EpisodeResearchError("group_identity_mismatch")
        positions = _group_positions(data)
        if len(positions) != listed["group_count"] or sum(len(rows) for rows in positions.values()) != listed["episode_count"]:
            raise EpisodeResearchError("group_index_changed")
        return data

    def read_group(self, index: int, group_id: str) -> dict:
        _candidate(self.case, index)
        data = self._load_group(index, group_id)
        positions = _group_positions(data)
        source_seasons = {row["source_season"] for row in self.case["files"]}
        return {"candidate_index": index, "group_id": group_id, "name": _text(data.get("name") or "", limit=200),
                "groups": [{"source_season": season, "episode_count": len(rows),
                            "episodes": [{"source_episode": row["order"]+1, **{key: row[key] for key in ("target_season", "target_episode", "episode_id")}}
                                         for row in rows if any(f["source_season"] == season and f["source_episode"] == row["order"]+1 for f in self.case["files"])]}
                           for season, rows in positions.items() if season in source_seasons]}

    def _season_episodes(self, index: int, season: int) -> dict[int, int]:
        item = _candidate(self.case, index)
        details = self._get(f"/tv/{item['tmdb_id']}/season/{season}")
        if _integer(details.get("season_number"), maximum=99) != season:
            raise EpisodeResearchError("target_season_mismatch")
        rows = details.get("episodes")
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_GROUP_EPISODES:
            raise EpisodeResearchError("target_season_invalid")
        episodes: dict[int, int] = {}
        seen_ids: set[int] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise EpisodeResearchError("target_episode_invalid")
            number = _integer(row.get("episode_number"), minimum=1)
            identity = int(_identity(row.get("id")))
            if _integer(row.get("season_number", season), maximum=99) != season or number in episodes or identity in seen_ids:
                raise EpisodeResearchError("target_episode_ambiguous")
            episodes[number] = identity
            seen_ids.add(identity)
        return episodes

    def validate(self, index: int, group_id: str, *, web_evidence=()) -> dict:
        self._check()
        item = _candidate(self.case, index)
        if index not in self._details or index not in self._listed:
            raise EpisodeResearchError("evidence_not_read")
        data = self._load_group(index, group_id)
        mapping = _map_group(self.case, data)
        if not mapping:
            raise EpisodeResearchError("release_order_not_proven", "来源不是剧集组的完整编号包或完整尾包")
        for other in self._listed[index]:
            if other["id"] == group_id:
                continue
            competing = _map_group(self.case, self._load_group(index, other["id"]))
            if competing is not None and competing != mapping:
                raise EpisodeResearchError("ambiguous_episode_groups", "存在多个相互冲突的发布顺序，需人工确认")
        # 标准季集也是竞争解释，即使TMDB没有把它重复登记成episode group。
        # 对已存在的SxxExx改成另一个ID/位置，不能因“只有一个DVD组”就自动通过。
        declared = self._details[index].get("seasons")
        if not isinstance(declared, list):
            raise EpisodeResearchError("standard_season_index_missing")
        declared_seasons: set[int] = set()
        for row in declared:
            if not isinstance(row, dict):
                raise EpisodeResearchError("standard_season_index_invalid")
            season = _integer(row.get("season_number"), maximum=99)
            if season in declared_seasons:
                raise EpisodeResearchError("standard_season_index_invalid")
            declared_seasons.add(season)
        seasons: dict[int, dict[int, int]] = {}
        for season in sorted({row["source_season"] for row in mapping} & declared_seasons):
            seasons[season] = self._season_episodes(index, season)
        for row in mapping:
            standard_id = seasons.get(row["source_season"], {}).get(row["source_episode"])
            if standard_id is not None and (row["source_season"], row["source_episode"], standard_id) != (row["target_season"], row["target_episode"], row["episode_id"]):
                raise EpisodeResearchError("standard_order_conflict", "标准季集仍是有效解释，不能自动采用不同的替代顺序")
        for season in sorted({row["target_season"] for row in mapping} - seasons.keys()):
            seasons[season] = self._season_episodes(index, season)
        for row in mapping:
            if seasons[row["target_season"]].get(row["target_episode"]) != row["episode_id"]:
                raise EpisodeResearchError("target_episode_identity_changed")
        evidence = [{"provider": "tmdb", "url": "https://api.themoviedb.org/3"+path, "sha256": _digest(value)}
                    for path, value in sorted(self._responses.items())]
        for row in list(web_evidence)[:4]:
            if (isinstance(row, dict) and isinstance(row.get("url"), str) and row["url"].startswith("https://")
                    and isinstance(row.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                    and not contains_sensitive_credential(row["url"])):
                evidence.append({"provider": "web", "url": row["url"][:2000], "sha256": row["sha256"]})
        return {"version": POLICY_VERSION, "status": "verified", "reason_code": "episode_group_proven",
                "case_key": self.case["case_key"], "candidate_index": index, "tmdb_id": item["tmdb_id"],
                "group_id": group_id, "group_fingerprint": _digest(_group_positions(data)), "mappings": mapping, "evidence": evidence}
