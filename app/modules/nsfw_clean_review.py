"""光鸭清洗候选的只读证据门；不识别元数据，不执行文件操作。"""

from __future__ import annotations

from typing import Any

from app import config
from app.modules.nsfw import (
    build_clean_title_candidate,
    clean_nsfw_release_text,
    extract_nsfw_identifier,
    extract_nsfw_multipart,
    normalize_code,
)


def nsfw_clean_review_enabled() -> bool:
    """仅返回子授权；调用方还必须校验 Agent 与主动复核父开关。"""
    return config.get_bool("AGENT_NSFW_CLEAN_REVIEW_ENABLED", False)


def _codes(name: str, strip_domains: str) -> set[str]:
    # 复用同一番号解析器，逐次移除已识别片段，避免首个番号掩盖混合文件名。
    remaining = clean_nsfw_release_text(name, strip_domains)
    result: set[str] = set()
    for _ in range(12):
        identifier = extract_nsfw_identifier(remaining)
        if identifier is None:
            return result
        result.add(normalize_code(identifier.code))
        raw = identifier.matched_text
        if not raw or raw not in remaining:
            return result | {"ambiguous"}
        remaining = remaining.replace(raw, " ", 1)
    return result | {"ambiguous"}


def inspect_nsfw_clean_candidate(payload: dict, candidate: dict) -> dict[str, Any]:
    """验证冻结文件的清洗变换；名称来自服务器规则而非模型新造。"""

    def reject(code: str, summary: str) -> dict[str, Any]:
        return {
            "ok": False,
            "status": "manual_required",
            "reason_code": code,
            "summary": summary,
            "data": {"metadata_verified": False},
        }

    if str(payload.get("kind") or "guangya") != "guangya":
        return reject("not_guangya", "本地媒体不支持 NSFW 自动清洗确认")
    rules = payload.get("rules")
    if not isinstance(rules, dict) or rules.get("nsfw_exclusive") is not True:
        return reject("not_nsfw_source", "仅允许已配置的光鸭成人专用来源")
    if (
        candidate.get("provider") != "clean_title"
        or candidate.get("media_type") != "movie"
        or candidate.get("tmdb_id")
    ):
        return reject(
            "not_clean_candidate", "只复核无元数据的清洗候选，不自动确认 MetaTube"
        )
    code = normalize_code(str(candidate.get("external_id") or ""))
    if not code or candidate.get("year"):
        return reject("invalid_identity", "清洗候选必须有明确番号，不能补造年份")
    if str(payload.get("multipart_strategy") or ""):
        return reject("ambiguous_multipart", "需要人为指定分段顺序，保留人工确认")
    files = payload.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 80:
        return reject("invalid_files", "冻结文件范围无效或过大")
    strip_domains = str(rules.get("nsfw_strip_domains") or "")
    rows = []
    parts = []
    ids: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            return reject("invalid_files", "冻结文件范围无效")
        name = str(item.get("name") or "")
        file_id = str(item.get("file_id") or "")
        if (
            not name
            or len(name) > 500
            or "/" in name
            or "\\" in name
            or not file_id
            or file_id in ids
        ):
            return reject("invalid_files", "冻结文件名或标识不完整，保留人工确认")
        ids.add(file_id)
        if _codes(name, strip_domains) != {code}:
            return reject("identity_mismatch", "原文件番号缺失、混合或与候选不同")
        clean = build_clean_title_candidate(name, strip_domains)
        if not clean or normalize_code(str(clean["external_id"])) != code:
            return reject("identity_mismatch", "清洗结果未保留原始番号")
        part = extract_nsfw_multipart(name, strip_domains)
        if part is not None and (part.ambiguous or part.part_index is None):
            return reject("ambiguous_multipart", "分段先后不明确，保留人工确认")
        parts.append(part.part_index if part is not None else None)
        rows.append(
            {
                "original_name": name,
                "clean_title": str(clean["title"]),
                "part_index": parts[-1],
            }
        )
    if len(files) > 1 and (None in parts or len(set(parts)) != len(parts)):
        return reject("ambiguous_multipart", "多个文件未给出唯一明确的数字分段")
    if str(candidate.get("title") or "") != rows[0]["clean_title"]:
        return reject("title_mismatch", "冻结候选标题与服务端清洗预览不一致")
    if len({row["clean_title"] for row in rows}) != 1:
        return reject("mixed_titles", "同组文件的清洗标题不一致，保留人工确认")
    for item in payload.get("companions") or []:
        if not isinstance(item, dict):
            return reject("invalid_companion", "伴随文件快照无效")
        name = str(item.get("name") or "")
        companion_codes = _codes(name, strip_domains)
        if companion_codes and companion_codes != {code}:
            return reject("companion_mismatch", "伴随文件指向其他番号，保留人工确认")
    return {
        "ok": True,
        "status": "verified_transform",
        "reason_code": "clean_transform_verified",
        "summary": "已核对原文件番号与清洗变换；未匹配完整元数据，尚未执行整理",
        "data": {
            "number": code,
            "files": rows,
            "metadata_verified": False,
            "entry_mode": "clean_title",
            "may_invent_metadata": False,
            "target_conflicts": "执行器仍须核对目标；不授权覆盖、替换或回收旧文件",
        },
    }
