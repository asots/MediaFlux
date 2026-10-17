#!/usr/bin/env python3
"""MediaFlux Agent 验收：默认离线回归，可评判隔离真实运行的事件记录。

provenance 字段可选，提供即校验；未标记的旧语料归入 legacy_unverified。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import sys
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.domain_catalog import build_tool_specs
from app.agent.kernel.adapters import consume_events
from app.agent.kernel.capabilities import (
    CapabilityRetriever,
    KernelToolSpec,
    ToolCatalog,
    ToolEffect,
)
from app.agent.kernel.effects import PreparedEffect
from app.agent.kernel.model import (
    ModelEvent,
    ModelEventType,
    ModelRequest,
    ModelToolCall,
)
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports import catalog_from_tool_specs
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import AgentInput, InMemorySessionStateStore, SessionState
from app.agent.public_view import public_result_state

DEFAULT_FIXTURE = Path("tests/fixtures/agent_kernel_capability_cases.jsonl")
DEFAULT_LIFECYCLE_FIXTURE = Path("tests/fixtures/agent_kernel_lifecycle_cases.jsonl")
DEFAULT_LIVE_FIXTURE = Path("tests/fixtures/agent_live_task_cases.jsonl")
DEFAULT_MATRIX_FIXTURE = Path("tests/fixtures/agent_live_task_matrix.jsonl")
DEFAULT_MATRIX_CASES = Path("tests/fixtures/agent_live_matrix_cases.jsonl")
PROVENANCE_KINDS = ("legacy_unverified", "synthetic", "simulated_user_real_model")


def load_live_cases(path: Path = DEFAULT_LIVE_FIXTURE) -> list[dict[str, Any]]:
    """版本化业务判据独立于旧首窗召回语料；holdout 不参与召回调参。"""
    parameters = {
        "actor_count": {"actor"},
        "actor_over_fifty": {"actor"},
        "year_identity": {"title", "year"},
        "weekly_calendar": {"source"},
        "recommendation": {"count", "min_year"},
        "favorites_search": set(),
        "scope_aba": {"scopes"},
        "preview_cancel": {"count"},
        "read_recovery": {"actor", "fault"},
        "constraint_memory": {"title", "season", "episode"},
    }
    cases, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if (
            not isinstance(case, dict)
            or set(case) != {"id", "version", "split", "messages", "rubric"}
            or not isinstance(case["id"], str)
            or not case["id"]
            or case["id"] in seen
            or type(case["version"]) is not int
            or case["version"] != 1
            or not isinstance(case["split"], str)
            or case["split"] not in {"regression", "holdout"}
            or not isinstance(case["messages"], list)
            or not case["messages"]
            or any(
                not isinstance(message, str) or not message.strip()
                for message in case["messages"]
            )
            or not isinstance(case["rubric"], dict)
            or not isinstance(case["rubric"].get("kind"), str)
            or case["rubric"]["kind"] not in parameters
        ):
            raise ValueError("真实任务语料字段、版本或判据无效")
        rubric = case["rubric"]
        if (
            set(rubric) != {"kind", *parameters[rubric["kind"]]}
            or any(
                type(rubric[key]) is not int or rubric[key] < 1
                for key in ("count", "min_year", "season", "episode")
                if key in rubric
            )
            or any(
                not isinstance(rubric[key], str) or not rubric[key].strip()
                for key in ("actor", "source", "title")
                if key in rubric
            )
            or (
                rubric["kind"] == "year_identity"
                and (not isinstance(rubric["year"], str) or not re.fullmatch(r"(?:19|20)\d{2}", rubric["year"]))
            )
            or (
                rubric["kind"] == "read_recovery"
                and (not isinstance(rubric["fault"], str) or rubric["fault"] not in {"missing_query", "invalid_enum"})
            )
            or (
                rubric["kind"] == "constraint_memory"
                and len(case["messages"]) != 8
            )
            or (
                rubric["kind"] == "scope_aba"
                and (
                    rubric.get("scopes") != ["电视剧", "动漫", "电视剧"]
                    or len(case["messages"]) != 3
                )
            )
        ):
            raise ValueError("真实任务判据参数无效")
        seen.add(case["id"])
        cases.append(case)
    if not cases:
        raise ValueError("真实任务语料为空")
    return cases


def export_live_event(
    path: Path, kind: str, payload: dict[str, Any], *, redactions: Sequence[str] = ()
) -> None:
    """逐条落盘公开事件；不导出 Provider 私有块、headers 或凭据。"""
    private = {
        "raw",
        "provider_state",
        "headers",
        "authorization",
        "api_key",
        "password",
        "secret",
        "cookie",
        "signature",
    }

    def public(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: public(item)
                for key, item in value.items()
                if key.lower() not in private
            }
        if isinstance(value, (list, tuple)):
            return [public(item) for item in value]
        return value

    text = json.dumps({"kind": kind, "payload": public(payload)}, ensure_ascii=False)
    for secret in redactions:
        if secret:
            text = text.replace(
                json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]"
            )
    with path.open("a", encoding="utf-8") as target:
        target.write(text + "\n")


def _assess_eight_constraints(
    case: dict[str, Any], trace: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    """T13 v1 的逐轮可观察证据；引用候选不等于推荐，文本语义留人工审查。"""
    from app.indexers.release import parse_indexer_release_position
    from app.modules.media_probe import infer_dynamic_ranges

    turns, rubric = trace.get("turns", []), case["rubric"]
    # 第 4/5 轮只排除 HDR；第 6 轮才新增必须明确 SDR。
    constraints = [
        (1, "1080p", False, False), (2, "1080p", False, False),
        (2, "2160p", False, False), (2, "2160p", False, True),
        (3, "2160p", False, True), (3, "2160p", True, True),
        (3, "1080p", True, True), (3, "1080p", True, True),
    ]
    checks = {"eight_actual_user_turns": len(turns) == 8 and all(
        turn.get("message") == message
        for turn, message in zip(turns, case["messages"])
    )}
    snapshots, candidates, support, reports = {}, {}, {}, []

    def model_receipt(row):
        content = row.get("model_content")
        if content is None:
            return row.get("public", {})
        try:
            value = json.loads(content) if isinstance(content, str) else content
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    for number, (season, resolution, sdr, exclude_hdr) in enumerate(constraints, 1):
        key = (season, resolution, sdr, exclude_hdr)
        expected = {"title": rubric["title"], "season": season, "episode": 1,
                    "resolution": resolution, "dynamic_range": "SDR" if sdr else None,
                    "exclude_hdr": exclude_hdr, "read_only": True, "download": False, "save_preferences": False}

        def scope_matches(arguments):
            return (arguments.get("title") in {rubric["title"], "Black Mirror"}
                    and arguments.get("season") == season and arguments.get("episode") == 1)

        def quality_matches(arguments):
            return (arguments.get("resolution") == resolution
                    and (not sdr or arguments.get("dynamic_range") == "SDR")
                    and (not exclude_hdr or arguments.get("exclude_hdr") is True
                         or arguments.get("dynamic_range") == "SDR"))

        def matches(item):
            title, technical = item.get("title", ""), item.get("technical_evidence") or {}
            position = parse_indexer_release_position(title)
            ranges = set(technical.get("dynamic_ranges") or [])
            dynamic_range = technical.get("dynamic_range")
            hdr = bool({"HDR", "HDR10", "HDR10+", "DoVi", "HLG"} & (
                ranges | {dynamic_range} | set(infer_dynamic_ranges(title))))
            # 未标 HDR 不能证明 SDR；明确的混合 HDR/SDR 声明同样不能通过。
            return bool(item.get("result_id")) and (
                "black mirror" in title.casefold().replace(".", " ") or "黑镜" in title
            ) and position.get("season") == season and position.get("episode") == 1 and (
                technical.get("resolution") == resolution
            ) and (not exclude_hdr or not hdr) and (not sdr or (
                dynamic_range == "SDR" and ranges <= {"SDR"}
            ))

        searches, deliveries = [], []
        selection_valid = True
        for index, row in enumerate(trace.get("tools", [])):
            if row.get("turn") != number or row.get("public", {}).get("ok") is not True:
                continue
            document = model_receipt(row)
            payload = document.get("data") or {}
            arguments = row.get("arguments") or {}
            effective_query = payload.get("effective_query")
            if row.get("tool") == "indexer.search_resources":
                query = effective_query if isinstance(effective_query, dict) else arguments
                numbers = {item["position"]: item for item in document.get("candidate_numbers", [])
                           if item.get("position") is not None}
                batch = []
                for item in payload.get("items", []):
                    spec = numbers.get(item.get("position"), {})
                    observed = {**item, "source_turn": number, "tool_index": index,
                                "technical_evidence": item.get("technical_evidence") or (
                                    spec.get("technical_evidence") if spec.get("title") == item.get("title") else None
                                ) or {}}
                    batch.append(observed)
                    if item.get("result_id"):
                        candidates[item["result_id"]] = observed
                ref = document.get("reference_arguments", {}).get("resource_candidates_ref")
                observed_receipt = document.get("ok") is True and isinstance(payload.get("items"), list)
                if ref and observed_receipt:
                    snapshots[ref] = {"arguments": query, "candidates": batch,
                                      "query_observed": isinstance(effective_query, dict)}
                searches.append({"tool_index": index, "source_turn": number, "arguments": arguments,
                                 "effective_query": effective_query,
                                 "receipt_observed": observed_receipt,
                                 "candidate_count": len(batch), "matching_candidate_count": sum(matches(item) for item in batch),
                                 "partial": payload.get("partial", False),
                                 "technical_filter": payload.get("technical_filter")})
                if observed_receipt and scope_matches(query) and quality_matches(query):
                    support.setdefault(key, []).append(index)
            elif row.get("tool") == "indexer.present_candidates":
                snapshot = snapshots.get(arguments.get("resource_candidates_ref"))
                source_query = (snapshot or {}).get("arguments", {})
                query = effective_query if isinstance(effective_query, dict) else source_query
                query_observed = isinstance(effective_query, dict) or bool(snapshot and snapshot.get("query_observed"))
                # 展示只覆盖规格；同一快照的父/子 ref 各自保存有效条件。
                query_consistent = (
                    not (isinstance(effective_query, dict) and snapshot and snapshot.get("query_observed"))
                    or query == {**source_query, **{
                        key: arguments[key] for key in ("resolution", "dynamic_range", "exclude_hdr") if key in arguments
                    }}
                )
                positions = payload.get("positions")  # 实际过滤结果，不是 arguments.positions。
                by_position = {item.get("position"): item for item in (snapshot or {}).get("candidates", [])}
                valid = document.get("ok") is True and snapshot is not None and isinstance(positions, list)
                valid = valid and all(position in by_position and position in arguments.get("positions", []) for position in positions)
                numbers = {item["position"]: item for item in document.get("candidate_numbers", [])}
                if "candidate_numbers" in document and valid:
                    valid = set(numbers) == set(positions) and all(
                        numbers[position].get("title") == by_position[position].get("title") for position in positions
                    )
                selected = [{**by_position[position], "technical_evidence": numbers.get(position, {}).get(
                    "technical_evidence", by_position[position]["technical_evidence"])} for position in positions] if valid else []
                scoped = valid and scope_matches(source_query) and scope_matches(query)
                delivered_valid = (scoped and query_consistent and (not query_observed or quality_matches(query))
                                   and all(matches(item) for item in selected))
                selection_valid = selection_valid and delivered_valid
                # 非空交付逐项核验；空交付需回执证明过滤了当前规格，不能仅靠请求声称。
                requested = (payload.get("technical_filter") or {}).get("requested", {})
                if delivered_valid and (selected or quality_matches(requested)):
                    support.setdefault(key, []).append(index)
                returned_ref = document.get("reference_arguments", {}).get("resource_candidates_ref")
                if valid and query_consistent and returned_ref:
                    snapshots[returned_ref] = {**snapshot, "arguments": query, "query_observed": query_observed}
                deliveries.append({"tool_index": index, "requested_positions": arguments.get("positions"),
                                   "effective_query": query if query_observed else None,
                                   "effective_query_consistent": query_consistent,
                                   "positions": positions, "reference_resolved": valid, "scope_matches": scoped,
                                   "candidate_ids": [item.get("result_id") for item in selected],
                                   "matches_constraints": delivered_valid, "technical_filter": payload.get("technical_filter"),
                                   "scope": payload.get("scope")})
        checks[f"turn_{number}_constraints_observed"] = bool(support.get(key))
        checks[f"turn_{number}_no_invalid_structured_delivery"] = selection_valid

        answer = turns[number - 1].get("answer", "") if number <= len(turns) else ""
        references = []
        # 仅识别完整 ID/发布标题；允许删除尾部站点标签，但缩写必须唯一。
        aliases: dict[str, set[str]] = {}
        for candidate_id, item in candidates.items():
            for alias in {candidate_id, item.get("title", ""),
                          re.sub(r"\[[^\]]*\]$", "", item.get("title", "")).strip()}:
                if alias:
                    aliases.setdefault(alias, set()).add(candidate_id)
        referenced_ids = {next(iter(ids)) for alias, ids in aliases.items() if len(ids) == 1 and alias in answer}
        for candidate_id, item in candidates.items():
            if candidate_id in referenced_ids:
                references.append({"result_id": candidate_id, "title": item.get("title"),
                                   "source_turn": item["source_turn"], "tool_index": item["tool_index"],
                                   "position": item.get("position"), "technical_evidence": item["technical_evidence"],
                                   "matches_constraints": matches(item)})
        reports.append({"turn": number, "constraints": expected, "searches": searches,
                        "supporting_tool_indexes": list(support.get(key, [])),
                        "structured_deliveries": deliveries, "referenced_candidates": references,
                        "language_review": "required: mentions/endorsement/empty-result scope cannot be inferred"})
    checks["no_preference_write_or_download"] = not any(
        row.get("tool") in {"media.set_preferences", "ingest.submit"} for row in trace.get("tools", [])
    )
    # 不用标题提及、空列表或 success 状态替代八轮自由文本语义签核。
    checks["answer_semantics_verified"] = False
    return checks, {"turns": reports, "language_review_required": True,
                    "final_referenced_candidate_ids": [item["result_id"] for item in reports[-1]["referenced_candidates"]]}


def assess_live_task(case: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """用业务返回核验目标；turn.success 或模型自述不能单独证明通过。"""
    if (
        trace.get("case_id") != case["id"]
        or trace.get("case_version") != case["version"]
    ):
        raise ValueError("真实记录与任务版本不匹配")
    rubric, turns = case["rubric"], trace.get("turns", [])
    tools = [
        row for row in trace.get("tools", []) if row.get("public", {}).get("ok") is True
    ]
    kind = rubric["kind"]
    answers = "\n".join(row.get("answer", "") for row in turns)
    checks = {
        "no_confirmed_execution": trace.get("confirmed_executions") == 0,
        "all_turns_finished": len(turns) == len(case["messages"])
        and all(
            row.get("status")
            in (
                {"success", "approval_required"}
                if kind == "preview_cancel"
                else {"success"}
            )
            for row in turns
        ),
    }

    def selected(name):
        return [row for row in tools if row["tool"] == name]

    def data(row):
        return row["public"].get("data") or {}

    outcome, evidence = "failed", {}
    if kind in {"actor_count", "actor_over_fifty", "read_recovery"}:
        rows = selected("library.search")
        checks["actor_exact_total"] = any(
            row["arguments"].get("search_by") == "actor"
            and row["arguments"].get("query") == rubric["actor"]
            and row["arguments"].get("count_only") is True
            and row["arguments"].get("media_type") == "movie"
            and data(row).get("match_basis") == "actor_metadata"
            and bool(data(row).get("sources"))
            and type(data(row).get("total")) is int
            and re.search(rf"(?<!\d){data(row)['total']}(?!\d)", answers) is not None
            for row in rows
        )
        if kind == "actor_over_fifty":
            counted = [row for row in rows if data(row).get("match_basis") == "actor_metadata"
                       and row["arguments"].get("query") == rubric["actor"]
                       and row["arguments"].get("search_by") == "actor"
                       and row["arguments"].get("media_type") == "movie"
                       and row["arguments"].get("count_only") is True]
            totals = [data(row).get("total") for row in counted]
            checks["observed_total_over_fifty"] = any(
                type(total) is int and total > 50 for total in totals
            )
            checks["server_totals_verified"] = any(
                type(data(row).get("total")) is int and data(row)["total"] > 50
                and re.search(rf"(?<!\d){data(row)['total']}(?!\d)", answers) is not None
                and all(source.get("status") == "ready" and type(source.get("total")) is int
                        and source.get("total_source") not in {None, "", "unknown", "returned_items"}
                        for source in data(row).get("sources", []))
                and bool(data(row).get("sources"))
                and sum(source["total"] for source in data(row)["sources"]) == data(row)["total"]
                for row in counted
            )
            evidence["observed_actor_totals"] = totals
            if not checks["observed_total_over_fifty"]:
                outcome = "not_exercised"
        if kind == "read_recovery":
            applied = trace.get("matrix", {}).get("applied", [])
            checks["injected_once_and_validated"] = (
                len(applied) == 1
                and applied[0].get("tool") == "library.search"
                and applied[0].get("variant") == rubric["fault"]
                and applied[0].get("error_code") == "invalid_arguments"
                and (
                    "query" not in applied[0].get("after", {})
                    if rubric["fault"] == "missing_query"
                    else applied[0].get("after", {}).get("media_type")
                    == "matrix_invalid_enum"
                )
            )
            attempts = [row for row in trace.get("tools", []) if row["tool"] == "library.search"]
            checks["bounded_recovery"] = 2 <= len(attempts) <= 3
            checks["validation_feedback_recorded"] = any(
                event.get("type") == "tool.failed"
                and event.get("payload", {}).get("code") == "invalid_arguments"
                for turn in turns for event in turn.get("events", [])
            )
            evidence["fault_applied"] = len(applied)
            if not applied:
                outcome = "not_exercised"
    elif kind == "year_identity":
        items = [
            (source.get("server_type"), item)
            for row in selected("library.search")
            for source in data(row).get("sources", []) if source.get("status") == "ready"
            for item in source.get("items", [])
        ]
        related = [
            (source, item) for source, item in items
            if item.get("media_type") == "movie"
            and (rubric["title"] in str(item.get("title", "")) or str(item.get("title", "")).casefold() == "dune")
        ]
        target = [(source, item) for source, item in related if str(item.get("year")) == rubric["year"]]
        checks["receipt_entity_has_requested_year"] = bool(target)
        checks["version_identity_delivered"] = any(
            re.search(re.escape(item["title"]) + r"[^\n]{0,48}" + rubric["year"], answers)
            or re.search(rubric["year"] + r"[^\n]{0,48}" + re.escape(item["title"]), answers)
            for _, item in target
        )
        evidence["matched_entities"] = [
            {"source": source, "title": item["title"], "year": item["year"], "media_type": item["media_type"]}
            for source, item in target
        ]
        evidence["other_entity_years"] = sorted({str(item["year"]) for _, item in related if item.get("year") and str(item["year"]) != rubric["year"]})
        evidence["observed_entity_count"] = len(items)
        if not items:
            outcome = "not_exercised"
    elif kind == "weekly_calendar":
        rows = [
            row
            for row in trace.get("tools", [])
            if row["tool"] == "discovery.anime_calendar"
            if row["arguments"].get("day") == "week"
            and row["arguments"].get("source") == rubric["source"]
        ]
        pages = [data(row) for row in rows]
        items = {
            json.dumps(
                {
                    key: item.get(key)
                    for key in (
                        "stable_id",
                        "date",
                        "update_time",
                        "audience",
                        "schedule",
                    )
                },
                sort_keys=True,
            ): item
            for page in pages
            for item in page.get("items", [])
        }
        checks["complete_week_evidence"] = (
            bool(pages)
            and all(
                len(page.get("requested_dates", [])) == 7
                and bool(page.get("sources"))
                and all(
                    source.get("status") == "ok" for source in page.get("sources", [])
                )
                for page in pages
            )
            and not pages[-1].get("has_more", True)
            and len(items) == pages[-1].get("total")
        )
        try:
            start = date.fromisoformat(pages[0]["week_start"])
            dates = [(start + timedelta(days=i)).isoformat() for i in range(7)]
            checks["correct_week_dates"] = start.weekday() == 0 and all(
                page.get("requested_dates") == dates
                and page.get("week_end") == dates[-1]
                for page in pages
            )
        except (IndexError, KeyError, TypeError, ValueError):
            checks["correct_week_dates"] = False
        checks["all_titles_delivered"] = bool(items) and all(
            item.get("title") and item["title"] in answers for item in items.values()
        )
        if trace.get("safety_blocks"):
            outcome = "infrastructure_interference"
            checks["uninterfered_calendar_read"] = False
    elif kind == "recommendation":
        rows = selected("discovery.recommend")
        items = {
            item.get("tmdb_id"): item
            for row in rows
            # 库存核对回执也含同次筛选候选，不能把真实证据误判为模型编造。
            for item in data(row).get("items", [])
            + data(row).get("library_check", {}).get("items", [])
            if item.get("tmdb_id")
            and str(item.get("year", "")).isdigit()
            and int(item["year"]) >= rubric["min_year"]
            and item.get("library_status") == "missing"
            and row["arguments"].get("media_type") == "tv"
            and row["arguments"].get("exclude_library") is True
            and str(row["arguments"].get("region", "")).casefold()
            in {"cn", "中国", "中国大陆", "大陆"}
            and str(row["arguments"].get("genre", "")).casefold()
            in {"animation", "动画", "动漫"}
        }
        matched = {
            key
            for key, item in items.items()
            if item.get("title")
            and re.search(
                # 只认同一行完整的列表/表格标题和年份，不让短标题冒名。
                r"(?:^|[,，;；|])[ \t]*(?:(?:\d+[.)、]|[-+*])[ \t]+)?"
                r"[*_`《「『]*"
                rf"(?:\[{re.escape(item['title'])}\]\([^\r\n)]*\)|{re.escape(item['title'])})"
                r"[*_`》」』]*(?:[ \t]*[（(|][ \t]*|[ \t]+)"
                rf"[*_`]*{re.escape(str(item['year']))}(?!\d)",
                answers,
                re.MULTILINE,
            )
        }
        recommended = {
            item.get("tmdb_id") for row in rows for item in data(row).get("items", [])
        }
        evidence = {
            "recommended_items_matched": len(matched & recommended),
            "same_batch_verified_matches": len(matched),
        }
        # 库存核对只能补充证据，不能放宽“交付足量推荐 items”的既定门槛。
        checks["enough_verified_absent_titles"] = (
            evidence["recommended_items_matched"] >= rubric["count"]
        )
    elif kind == "favorites_search":
        rows = selected("discovery.watchlist_summaries")
        empty = bool(rows) and all(
            data(row).get("count", data(row).get("total")) == 0 for row in rows
        )
        if empty:
            outcome = "not_configured"
            checks["watchlist_configured"] = False
            checks["empty_watchlist_disclosed"] = bool(
                re.search(r"(?:没有|暂无|未.*(?:添加|收藏)|为空|空的|0.*项)", answers)
            )
        else:
            checks["updates_then_resource_search"] = (
                bool(rows)
                and bool(selected("library.check_updates"))
                and bool(
                    selected("indexer.search_resources")
                    or selected("library.search_missing_season_resources")
                )
            )
    elif kind == "scope_aba":
        rows = selected("media.recently_added")
        first = [row for row in rows if row.get("turn") == 1]
        second = [row for row in rows if row.get("turn") == 2]
        checks["explicit_a_b_scopes"] = (
            bool(first and second)
            and all(
                rubric["scopes"][0] in str(row["arguments"].get("library_name", ""))
                for row in first
            )
            and all(
                rubric["scopes"][1] in str(row["arguments"].get("library_name", ""))
                for row in second
            )
        )
        checks["returned_a_b_scopes"] = all(
            data(row).get("scope", {}).get("library", {}).get("name") == scope
            and data(row).get("granularity") == "work"
            for group, scope in (
                (first, rubric["scopes"][0]),
                (second, rubric["scopes"][1]),
            )
            for row in group
        )
        titles = [
            item.get("series_name") or item.get("title") or item.get("name")
            for row in first
            for item in data(row).get("items", [])
        ]
        anime = [
            item.get("series_name") or item.get("title") or item.get("name")
            for row in second
            for item in data(row).get("items", [])
        ]
        checks["returns_to_original_titles"] = (
            len(turns) == 3
            and len(set(titles)) == len(set(anime)) == 5
            and all(titles)
            and all(anime)
            and all(
                title in turns[0].get("answer", "")
                and title in turns[2].get("answer", "")
                for title in titles
            )
            and all(title in turns[1].get("answer", "") for title in anime)
            and not any(
                title in turns[2].get("answer", "")
                for title in set(anime) - set(titles)
            )
            and not any(row.get("turn") == 3 for row in trace.get("tools", []))
        )
    elif kind == "preview_cancel":
        checks["twenty_frozen_preview"] = any(
            row["public"].get("status") == "confirmation_required"
            and row["arguments"].get("positions") == list(range(1, rubric["count"] + 1))
            and data(row).get("count")
            == len(data(row).get("resources", []))
            == rubric["count"]
            and [item.get("position") for item in data(row).get("resources", [])]
            == row["arguments"].get("positions")
            for row in selected("ingest.submit")
        )
        checks["frozen_approval_plan"] = any(
            event.get("type") == "effect.approval_required"
            and event.get("payload", {}).get("plan", {}).get("plan_id")
            for turn in turns
            for event in turn.get("events", [])
        )
        checks["preview_cancelled"] = trace.get("cancelled_plan") is True
    elif kind == "constraint_memory":
        constraint_checks, evidence = _assess_eight_constraints(case, trace)
        checks.update(constraint_checks)
    approvals = sum(
        event.get("type") == "effect.approval_required"
        for turn in turns
        for event in turn.get("events", [])
    )
    if kind in {"actor_over_fifty", "year_identity", "read_recovery", "constraint_memory"}:
        checks["no_unnecessary_structured_intervention"] = approvals == 0 and not any(
            turn.get("status") == "waiting" for turn in turns
        )
    checks["no_runtime_failure"] = not trace.get("failure")
    if kind == "constraint_memory" and all(
        valid for key, valid in checks.items() if key != "answer_semantics_verified"
    ):
        outcome = "manual_review"
    if trace.get("classification") == "infrastructure_interference":
        checks["uninterfered_harness"] = False
        outcome = "infrastructure_interference"
    calls = trace.get("model_calls", [])
    observed: dict[str, int] = {}
    unknown = 0
    for call in calls:
        if type((call.get("usage") or {}).get("total_tokens")) is not int:
            unknown += 1
        for key, value in (call.get("usage") or {}).items():
            if type(value) is int and value >= 0:
                observed[key] = observed.get(key, 0) + value
    passed = all(checks.values())
    return {
        "case_id": case["id"],
        "version": case["version"],
        "split": case["split"],
        "source": trace.get("source"),
        "source_commit": trace.get("source_commit"),
        "execution_mode": (
            "hybrid" if trace.get("matrix", {}).get("scenario") in {
                "read_missing_query", "read_invalid_enum", "hybrid_guoman_front_owned"
            } else "deterministic" if trace.get("source") == "synthetic_test"
            else trace.get("execution_mode", "production" if trace.get("environment", {}).get("validation_layer") == "real_model_real_readonly_services" else "unclassified")
        ),
        "passed": passed,
        "outcome": "passed" if passed else outcome,
        "checks": checks,
        "evidence": evidence,
        "failure_reasons": [key for key, valid in checks.items() if not valid],
        "user_turns": len(turns),
        "model_rounds": len(calls),
        "http_calls": sum(call.get("http_attempts", 0) for call in calls),
        "necessary_confirmations": approvals if kind == "preview_cancel" else 0,
        "preview_cancellations": int(trace.get("cancelled_plan") is True),
        "confirmed_executions": trace.get("confirmed_executions"),
        "invalid_interventions": (0 if kind == "preview_cancel" else approvals)
        + sum(turn.get("status") == "waiting" for turn in turns),
        "usage": observed or None,
        "usage_complete": unknown == 0 and bool(calls),
        "usage_unknown_calls": unknown,
        "runtime_failure": trace.get("failure"),
        "tool_failures": [
            {
                "tool": row["tool"],
                "error": row.get("error"),
                "status": row.get("public", {}).get("status"),
            }
            for row in trace.get("tools", [])
            if row.get("error") or row.get("public", {}).get("ok") is False
        ],
    }


def summarize_live_tasks(
    cases: list[dict[str, Any]], traces: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {case["id"]: case for case in cases}
    if any(trace.get("case_id") not in by_id for trace in traces):
        raise ValueError("真实记录引用了未知任务")
    outcomes = [
        assess_live_task(by_id[trace["case_id"]], trace)
        for trace in traces
        if trace.get("classification") != "infrastructure_failure_not_sent"
    ]
    passed = sum(row["passed"] for row in outcomes)
    usage: dict[str, int] = {}
    for outcome in outcomes:
        for key, value in (outcome["usage"] or {}).items():
            usage[key] = usage.get(key, 0) + value
    return {
        "ok": bool(outcomes) and passed == len(outcomes),
        "scope": "versioned_task_trace_assessment",
        "summary": {
            "cases": len(outcomes),
            "passed": passed,
            "success_rate": passed / len(outcomes) if outcomes else None,
            "not_configured": sum(
                row["outcome"] == "not_configured" for row in outcomes
            ),
            "infrastructure_interference": sum(
                row["outcome"] == "infrastructure_interference" for row in outcomes
            ),
            "infrastructure_not_sent": sum(
                trace.get("classification") == "infrastructure_failure_not_sent"
                for trace in traces
            ),
            "model_rounds": sum(row["model_rounds"] for row in outcomes),
            "http_calls": sum(row["http_calls"] for row in outcomes),
            "necessary_confirmations": sum(
                row["necessary_confirmations"] for row in outcomes
            ),
            "invalid_interventions": sum(
                row["invalid_interventions"] for row in outcomes
            ),
            "preview_cancellations": sum(
                row["preview_cancellations"] for row in outcomes
            ),
            "usage": usage or None,
            "usage_complete": bool(outcomes)
            and all(row["usage_complete"] for row in outcomes),
            "usage_unknown_calls": sum(row["usage_unknown_calls"] for row in outcomes),
            "execution_modes": {
                mode: sum(row["execution_mode"] == mode for row in outcomes)
                for mode in sorted({row["execution_mode"] for row in outcomes})
            },
        },
        "outcomes": outcomes,
    }


def load_task_matrix(path: Path = DEFAULT_MATRIX_FIXTURE) -> list[dict[str, Any]]:
    """26项是执行清单，不把pytest或7个query映射成真实覆盖。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (
        len(rows) != 26
        or {row.get("id") for row in rows} != {f"T{index:02}" for index in range(1, 27)}
        or any(
            row.get("version") != 1 or type(row.get("version")) is not int
            or not row.get("messages") or not all(isinstance(message, str) and message.strip() for message in row["messages"])
            or not row.get("acceptance") or not isinstance(row.get("runner_cases"), list)
            for row in rows
        )
    ):
        raise ValueError("26任务矩阵ID、版本或执行清单无效")
    return rows


def live_supplementary_evidence(case: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """补充观察不改变任何冻结v1判据。"""
    kind = case["rubric"]["kind"]
    rows = [row for row in trace.get("tools", []) if row.get("public", {}).get("ok") is True]
    if kind == "recommendation":
        items = [item for row in rows if row["tool"] == "discovery.recommend" for item in row["public"].get("data", {}).get("items", [])]
        source = {str(item["id"]): item for page in trace.get("source_evidence", []) for item in page.get("items", []) if item.get("id")}
        verified, rejected = [], []
        for item in items:
            raw = source.get(str(item.get("tmdb_id")))
            if raw is not None:
                (verified if "CN" in (raw.get("origin_country") or []) and 16 in (raw.get("genre_ids") or []) and raw.get("adult") is False else rejected).append(str(item["tmdb_id"]))
        return {"strict_item_ids": sorted({str(item.get("tmdb_id")) for item in items}),
                "cn_animation_nonadult_ids": sorted(set(verified)),
                "source_metadata_rejected_ids": sorted(set(rejected)),
                "source_metadata_missing_ids": sorted({str(item.get("tmdb_id")) for item in items} - source.keys())}
    if kind == "weekly_calendar":
        pages = [row["public"].get("data", {}) for row in rows if row["tool"] == "discovery.anime_calendar"]
        slots = {(item.get("title"), item.get("date"), item.get("update_time"))
                 for page in pages for item in page.get("items", [])}
        return {"returned_events": sum(len(page.get("items", [])) for page in pages),
                "distinct_title_date_time_slots": len(slots),
                "requested_dates": sorted({day for page in pages for day in page.get("requested_dates", [])}),
                "provider_statuses": sorted({source.get("status", "unknown") for page in pages for source in page.get("sources", [])}),
                "last_has_more": pages[-1].get("has_more") if pages else None,
                "note": "事件还含audience/schedule；合并同日同时间作品不等于逐项披露所有事件字段，补充证据不覆盖v1结果。"}
    return {}


def assess_pagination_replay(trace: dict[str, Any]) -> dict[str, bool]:
    """仅验真实执行的固定页/库存回放，不把计划注入当已触发。"""
    matrix = trace.get("matrix", {})
    replay = matrix.get("replay", {})
    pages, inventory = replay.get("pages", []), replay.get("inventory_reads", [])
    owned = {str(key) for item in matrix.get("applied", []) for key in item.get("owned_ids", [])}
    first_two = [page for page in pages if page.get("page") in {1, 2}]
    groups = {(row.get("server"), row.get("media_type")) for row in inventory}
    return {
        "front_two_pages_observed_owned": {page.get("page") for page in first_two} == {1, 2}
        and all(page.get("ids") and set(page["ids"]) <= owned for page in first_two),
        "later_page_observed": any(page.get("page", 0) >= 3 for page in pages),
        "one_inventory_per_request_source_type": bool(inventory) and len(groups) == len(inventory),
        "declared_contract_replay": replay.get("provenance", {}).get("kind") == "synthetic_contract_replay",
    }


def summarize_task_matrix(
    traces: list[dict[str, Any]], *, matrix_path: Path = DEFAULT_MATRIX_FIXTURE,
    cases_path: Path = DEFAULT_MATRIX_CASES,
) -> dict[str, Any]:
    cases = {row["id"]: row for row in load_live_cases() + load_live_cases(cases_path)}
    unknown = {trace.get("case_id") for trace in traces} - cases.keys()
    if unknown:
        raise ValueError("矩阵记录引用了未知case")
    tasks = []
    for task in load_task_matrix(matrix_path):
        runs = []
        for trace in traces:
            if trace["case_id"] not in task["runner_cases"] + task["frozen_cases"]:
                continue
            result = assess_live_task(cases[trace["case_id"]], trace)
            result["supplementary"] = live_supplementary_evidence(cases[trace["case_id"]], trace)
            # 只有有领域判据和必要现场的任务可自动覆盖；其余明确保留缺口。
            if result["outcome"] == "infrastructure_interference":
                status = "infrastructure_interference"
            elif task["id"] == "T01":
                status = ("automated_passed" if result["passed"] else result["outcome"]) if trace["case_id"] == "actor_count_over_fifty" else "not_exercised"
            elif task["id"] in {"T03", "T12"}:
                # 正常周历/ABA没有首次loading或超长历史现场，冻结结果仍独立保留。
                status = "not_exercised"
            elif task["id"] == "T11":
                status = ("automated_passed" if result["evidence"]["other_entity_years"] else "partial") if result["passed"] else result["outcome"]
            elif task["id"] in {"T13", "T14"} and trace["case_id"] in task["runner_cases"]:
                status = "automated_passed" if result["passed"] else result["outcome"]
            elif task["id"] == "T09":
                extra = result["supplementary"]
                verified = bool(extra["strict_item_ids"]) and set(extra["strict_item_ids"]) <= set(extra["cn_animation_nonadult_ids"])
                status = ("automated_passed" if verified else "partial") if result["passed"] else result["outcome"]
            elif task["id"] == "T10" and trace["case_id"] == "hybrid_guoman_front_owned":
                result["scenario_checks"] = assess_pagination_replay(trace)
                status = "automated_passed" if result["passed"] and all(result["scenario_checks"].values()) else "not_exercised" if not trace.get("matrix", {}).get("applied") else "failed"
            else:
                status = "partial" if result["passed"] else result["outcome"]
            result["matrix_status"] = status
            result["language_review"] = "required: coverage/error_feedback/unnecessary_intervention"
            runs.append(result)
        status = "not_exercised" if not runs else "automated_passed" if any(row["matrix_status"] == "automated_passed" for row in runs) else runs[-1]["matrix_status"]
        if task["id"] == "T14" and runs:
            # 两个独立故障都要恢复；漏参通过不能掩盖非法枚举失败。
            scenario_runs = [row for row in runs if row["case_id"] in task["runner_cases"]]
            if all(any(row["case_id"] == case_id and row["matrix_status"] == "automated_passed" for row in scenario_runs) for case_id in task["runner_cases"]):
                status = "automated_passed"
            elif any(row["matrix_status"] == "failed" for row in scenario_runs):
                status = "failed"
            elif any(row["matrix_status"] == "automated_passed" for row in scenario_runs):
                status = "partial"
            else:
                status = scenario_runs[-1]["matrix_status"] if scenario_runs else "not_exercised"
        tasks.append({"task_id": task["id"], "title": task["title"],
                      "status": status,
                      "external_conditions": task["external_conditions"], "runner_status": task["runner_status"], "runs": runs})
    return {"ok": False, "scope": "26_task_matrix_partial_evidence_not_release_acceptance",
            "summary": {"tasks": 26, "tasks_with_runs": sum(bool(row["runs"]) for row in tasks),
                        "automated_passed": sum(row["status"] == "automated_passed" for row in tasks),
                        "fully_reviewed_passed": 0,
                        "note": "自动数据判据不替代错误反馈/多余介入的逐段审查；未跑的任务和异端现场不补数。"}, "tasks": tasks}


def load_live_traces(paths: Sequence[Path], interference: Sequence[Path] = ()) -> list[dict[str, Any]]:
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for path in interference:
        annotation = json.loads(path.read_text(encoding="utf-8"))
        if annotation.get("classification") != "infrastructure_interference" or not annotation.get("cause"):
            raise ValueError("设施干扰记录必须有明确分类及原因")
        for trace in traces:
            if annotation.get("affected_run") == f"{trace.get('source')}/{trace.get('case_id')}":
                trace["classification"] = "infrastructure_interference"
                trace["interference_evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return traces


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    message: str
    required: tuple[str, ...]
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EvalOutcome:
    case_id: str
    passed: bool
    selected: tuple[str, ...]
    missing: tuple[str, ...]
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "selected": list(self.selected),
            "missing": list(self.missing),
            "provenance": self.provenance or {"kind": "legacy_unverified"},
        }



@dataclass(frozen=True, slots=True)
class LifecycleCase:
    case_id: str
    message: str
    result: dict[str, Any]
    followup: str
    expected: dict[str, Any]


class _ReplayModel:
    def __init__(self, case: LifecycleCase) -> None:
        self.requests: list[ModelRequest] = []
        self._rounds = [
            [
                ModelEvent(
                    ModelEventType.TOOL_CALL_COMPLETED,
                    tool_call=ModelToolCall("write", "cloud.lifecycle_write", {}),
                ),
                ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls"),
            ],
            [
                ModelEvent(ModelEventType.TEXT_DELTA, text=case.followup),
                ModelEvent(ModelEventType.FINISH, finish_reason="stop"),
            ],
        ]

    async def stream(
        self, request: ModelRequest, *, cancellation: Any
    ) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        if not self._rounds:
            raise AssertionError("生命周期回放出现了意外模型轮次")
        for event in self._rounds.pop(0):
            cancellation.raise_if_cancelled()
            await asyncio.sleep(0)
            yield event


def load_lifecycle_cases(path: Path) -> list[LifecycleCase]:
    cases: list[LifecycleCase] = []
    seen: set[str] = set()
    expected_fields = {
        "final_status",
        "result_state",
        "model_calls",
        "confirm_events",
        "answer_contains",
        "answer_excludes",
    }
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"生命周期语料第 {line_number} 行不是有效 JSON") from exc
        if not isinstance(raw, dict) or set(raw) != {
            "id", "message", "result", "followup", "expected"
        }:
            raise ValueError(f"生命周期语料第 {line_number} 行字段无效")
        case_id = str(raw["id"] or "").strip()
        message = str(raw["message"] or "").strip()
        followup = str(raw["followup"] or "").strip()
        result = raw["result"]
        expected = raw["expected"]
        if (
            not case_id
            or case_id in seen
            or not message
            or not followup
            or not isinstance(result, dict)
            or set(result) - {"ok", "status", "summary", "data", "error"}
            or type(result.get("ok")) is not bool
            or not str(result.get("status") or "").strip()
            or not str(result.get("summary") or "").strip()
            or not isinstance(expected, dict)
            or set(expected) != expected_fields
            or type(expected.get("model_calls")) is not int
            or expected["model_calls"] < 1
        ):
            raise ValueError(f"生命周期语料第 {line_number} 行案例无效")
        for key in ("confirm_events", "answer_contains", "answer_excludes"):
            values = expected.get(key)
            if not isinstance(values, list) or any(
                not isinstance(item, str) or not item for item in values
            ):
                raise ValueError(f"生命周期语料第 {line_number} 行 {key} 无效")
        seen.add(case_id)
        cases.append(LifecycleCase(case_id, message, dict(result), followup, dict(expected)))
    if not cases:
        raise ValueError("生命周期评测语料为空")
    return cases


async def _replay_lifecycle_case(case: LifecycleCase) -> dict[str, Any]:
    executions: list[dict[str, Any]] = []
    tool = KernelToolSpec(
        name="cloud.lifecycle_write",
        domain="cloud",
        description="离线验收确认生命周期",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        effect=ToolEffect.WRITE,
        prepare=lambda _arguments, _context: PreparedEffect(
            preview={"summary": "确认后执行生命周期验收写操作"},
            snapshot_fingerprint="lifecycle-replay",
        ),
        execute_confirmed=lambda arguments, _snapshot, _context: (
            executions.append(dict(arguments)) or dict(case.result)
        ),
    )
    catalog = ToolCatalog([tool])
    state = InMemorySessionStateStore()
    model = _ReplayModel(case)
    session = AgentSession(
        model=model,
        catalog=catalog,
        retriever=CapabilityRetriever(minimum=1, maximum=1),
        pipeline=ToolPipeline(catalog=catalog, state_store=state),
        state_store=state,
    )
    owner = f"eval:{case.case_id}"
    session_id = f"lifecycle-{case.case_id}"
    preview = await consume_events(session.run(AgentInput(
        message=case.message, owner=owner, session_id=session_id, channel="eval",
    )))
    if preview.approval is None:
        return {"case_id": case.case_id, "passed": False, "errors": ["未生成确认计划"]}
    confirm_events = [
        event async for event in session.confirm(
            owner=owner, session_id=session_id, plan_id=preview.approval.plan_id, channel="eval",
        )
    ]

    async def stream() -> AsyncIterator[Any]:
        for event in confirm_events:
            yield event

    final = await consume_events(stream())
    current = await state.load(owner=owner, session_id=session_id)
    actual_events = [event.type.value for event in confirm_events]
    public_dump = json.dumps([event.to_dict() for event in confirm_events], ensure_ascii=False)
    expected = case.expected
    errors: list[str] = []
    checks = (
        (preview.status == "approval_required", "预览终态不是 approval_required"),
        (final.status == expected["final_status"], f"最终状态为 {final.status}"),
        (public_result_state(final.effect_result) == expected["result_state"], "结果语义不匹配"),
        (len(model.requests) == expected["model_calls"], f"模型调用 {len(model.requests)} 次"),
        (actual_events == expected["confirm_events"], f"确认事件序列为 {actual_events}"),
        (len(executions) == 1, f"副作用执行 {len(executions)} 次"),
        (not current.pending_effect_plan_id, "确认计划仍处于待处理状态"),
    )
    errors.extend(message for passed, message in checks if not passed)
    for text in expected["answer_contains"]:
        if text not in final.answer:
            errors.append(f"回答缺少：{text}")
    for text in expected["answer_excludes"]:
        if text in final.answer or text in public_dump:
            errors.append(f"公开结果泄漏或误报：{text}")
    return {
        "case_id": case.case_id,
        "passed": not errors,
        "errors": errors,
        "events": actual_events,
        "final_status": final.status,
        "result_state": public_result_state(final.effect_result),
        "model_calls": len(model.requests),
    }


def evaluate_lifecycle(path: Path = DEFAULT_LIFECYCLE_FIXTURE) -> dict[str, Any]:
    cases = load_lifecycle_cases(path)

    async def run() -> list[dict[str, Any]]:
        return [await _replay_lifecycle_case(case) for case in cases]

    outcomes = asyncio.run(run())
    passed = sum(bool(item["passed"]) for item in outcomes)
    return {
        "ok": passed == len(outcomes),
        "summary": {"cases": len(outcomes), "passed": passed, "failed": len(outcomes) - passed},
        "outcomes": outcomes,
    }

def _valid_provenance(
    provenance: dict[str, Any], message: str, required: tuple[str, ...]
) -> bool:
    """仅校验字段与摘录契约；原件哈希供追溯，不读取或认证私有原件。"""
    if provenance["kind"] != "simulated_user_real_model":
        return set(provenance) == {"kind"}
    if set(provenance) != {
        "kind", "source_commit", "evidence_sha256", "excerpt"
    } or not isinstance(provenance.get("source_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", provenance["source_commit"]
    ):
        return False
    hashes = provenance["evidence_sha256"]
    excerpt = provenance["excerpt"]
    if not isinstance(hashes, dict) or not {
        "model-request-1.json", "HTTP-response-1.json", "model-events-1.json",
        "query-events.json", "case-summary.json",
    } <= set(hashes) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+\.json", name)
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for name, digest in hashes.items()
    ):
        return False
    if not isinstance(excerpt, dict) or set(excerpt) != {
        "message", "initial_tools", "initial_missing", "http_status",
        "completed_tool_calls", "final_status", "business_pass",
    }:
        return False
    tools = excerpt["initial_tools"]
    calls = excerpt["completed_tool_calls"]
    if (
        excerpt["message"] != message
        or not isinstance(tools, list)
        or not tools
        or any(not isinstance(name, str) or not name.strip() for name in tools)
        or excerpt["initial_missing"] != [name for name in required if name not in tools]
        or type(excerpt["http_status"]) is not int
        or not 100 <= excerpt["http_status"] <= 599
        or type(excerpt["business_pass"]) is not bool
        or not isinstance(excerpt["final_status"], str)
        or not excerpt["final_status"].strip()
        or not isinstance(calls, list)
        or any(
            not isinstance(call, dict)
            or set(call) != {"model_index", "name", "arguments"}
            or type(call["model_index"]) is not int
            or call["model_index"] < 1
            or not isinstance(call["name"], str)
            or not call["name"].strip()
            or not isinstance(call["arguments"], dict)
            for call in calls
        )
    ):
        return False
    return True


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_number} 行不是有效 JSON") from exc
        if (
            not isinstance(raw, dict)
            or not {"id", "message", "required"} <= set(raw)
            or set(raw) - {"id", "message", "required", "provenance"}
        ):
            raise ValueError(f"第 {line_number} 行字段无效")
        case_id = str(raw["id"] or "").strip()
        message = str(raw["message"] or "").strip()
        required_raw = raw["required"]
        if (
            not case_id
            or case_id in seen
            or not message
            or not isinstance(required_raw, list)
            or not required_raw
        ):
            raise ValueError(f"第 {line_number} 行案例无效")
        required = tuple(
            dict.fromkeys(str(item or "").strip() for item in required_raw)
        )
        if any(not item for item in required):
            raise ValueError(f"第 {line_number} 行 required 无效")
        provenance = raw.get("provenance", {"kind": "legacy_unverified"})
        if (
            not isinstance(provenance, dict)
            or provenance.get("kind") not in PROVENANCE_KINDS
            or not _valid_provenance(provenance, message, required)
        ):
            raise ValueError(f"第 {line_number} 行 provenance 无效")
        seen.add(case_id)
        cases.append(EvalCase(case_id, message, required, provenance))
    if not cases:
        raise ValueError("评测语料为空")
    return cases


def evaluate(
    path: Path = DEFAULT_FIXTURE,
    lifecycle_path: Path = DEFAULT_LIFECYCLE_FIXTURE,
) -> dict[str, Any]:
    catalog = catalog_from_tool_specs(build_tool_specs())
    retriever = CapabilityRetriever()
    outcomes: list[EvalOutcome] = []
    for case in load_cases(path):
        # 与 Session 的首窗一致：当前用户行已经落入历史，而不是只评测裸 query。
        state = SessionState(owner="offline-eval", session_id="offline-eval-session",
                             conversation=[{"role": "user", "content": case.message}])
        selection = retriever.retrieve(
            case.message,
            catalog,
            context={
                "owner": state.owner,
                "session_id": state.session_id,
                "channel": "eval",
                "reference_kinds": (),
                **AgentSession._capability_retrieval_context(state),
            },
        )
        selected = selection.names
        missing = tuple(name for name in case.required if name not in selected)
        outcomes.append(EvalOutcome(
            case.case_id, not missing, selected, missing, case.provenance
        ))

    tools = catalog.visible({})
    invalid_effect_tools: list[str] = []
    for tool in tools:
        if tool.effect is ToolEffect.READ:
            valid = (
                tool.read is not None
                and tool.prepare is None
                and tool.execute_confirmed is None
            )
        else:
            valid = (
                tool.read is None
                and tool.prepare is not None
                and tool.execute_confirmed is not None
            )
        if not valid:
            invalid_effect_tools.append(tool.name)

    candidate_counts = [len(item.selected) for item in outcomes]
    passed = sum(item.passed for item in outcomes)
    safety_ok = not invalid_effect_tools
    lifecycle = evaluate_lifecycle(lifecycle_path)
    cohorts = {}
    for kind in PROVENANCE_KINDS:
        items = [item for item in outcomes if item.provenance["kind"] == kind]
        count = len(items)
        hits = sum(item.passed for item in items)
        cohorts[kind] = {
            "cases": count, "passed": hits, "failed": count - hits,
            "retrieval_recall": round(hits / count, 4) if count else None,
            "missing": {item.case_id: list(item.missing) for item in items if item.missing},
        }
    return {
        "recall_scope": "initial_window_corpus_only",
        "cohorts": cohorts,
        "ok": passed == len(outcomes) and safety_ok and lifecycle["ok"],
        "summary": {
            "cases": len(outcomes),
            "passed": passed,
            "failed": len(outcomes) - passed,
            "retrieval_recall": round(passed / len(outcomes), 4),
            "candidate_count_min": min(candidate_counts),
            "candidate_count_max": max(candidate_counts),
            "candidate_count_mean": round(statistics.fmean(candidate_counts), 2),
            "catalog_tools": len(tools),
            "read_tools": sum(tool.effect is ToolEffect.READ for tool in tools),
            "effect_tools": sum(tool.effect is not ToolEffect.READ for tool in tools),
            "effect_gate_valid": safety_ok,
            "lifecycle_cases": lifecycle["summary"]["cases"],
            "lifecycle_passed": lifecycle["summary"]["passed"],
            "lifecycle_failed": lifecycle["summary"]["failed"],
        },
        "invalid_effect_tools": invalid_effect_tools,
        "outcomes": [item.to_dict() for item in outcomes],
        "lifecycle_outcomes": lifecycle["outcomes"],
    }


def _text_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "MediaFlux Agent Kernel 离线验收",
        f"语料首窗：{summary['passed']}/{summary['cases']} 通过，样本召回率 {summary['retrieval_recall']:.2%}",
        (
            "候选工具："
            f"{summary['candidate_count_min']}-{summary['candidate_count_max']} 项，"
            f"平均 {summary['candidate_count_mean']} 项"
        ),
        (
            "能力目录："
            f"{summary['catalog_tools']} 项（READ {summary['read_tools']} / "
            f"Effect {summary['effect_tools']}）"
        ),
        "Effect Gate：通过" if summary["effect_gate_valid"] else "Effect Gate：失败",
        (
            "生命周期回放："
            f"{summary['lifecycle_passed']}/{summary['lifecycle_cases']} 通过"
        ),
    ]
    for kind, cohort in result["cohorts"].items():
        recall = cohort["retrieval_recall"]
        rate = f"{recall:.2%}" if recall is not None else "N/A（无样本）"
        lines.append(f"{kind}：{cohort['passed']}/{cohort['cases']}，首窗样本召回率 {rate}")
    lines.append(
        "simulated_user_real_model 是模拟用户输入的真实模型证据，非生产用户自然故障；"
        "仅代表本语料样本，不代表全局真实用户召回率；"
        "首窗 FAIL 不等于最终任务失败。"
    )
    for outcome in result["outcomes"]:
        if not outcome["passed"]:
            lines.append(
                f"[FAIL] {outcome['case_id']} ({outcome['provenance']['kind']}) 首窗缺少：{', '.join(outcome['missing'])}"
            )
        provenance = outcome["provenance"]
        excerpt = provenance.get("excerpt")
        if provenance["kind"] == "simulated_user_real_model" and isinstance(excerpt, dict):
            lines.append(
                f"[EVIDENCE] {outcome['case_id']} 来源 commit={provenance.get('source_commit')}；"
                f"历史首窗缺少={excerpt.get('initial_missing')}；"
                f"证据最终状态={excerpt.get('final_status')}，business_pass={excerpt.get('business_pass')}"
            )
    for outcome in result["lifecycle_outcomes"]:
        if not outcome["passed"]:
            lines.append(
                f"[FAIL] {outcome['case_id']}：{'；'.join(outcome['errors'])}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--lifecycle-fixture", type=Path, default=DEFAULT_LIFECYCLE_FIXTURE
    )
    parser.add_argument("--live-results", type=Path, nargs="+", help="评判已有真实运行 JSON；不会发起付费调用")
    parser.add_argument("--live-fixture", type=Path, default=DEFAULT_LIVE_FIXTURE)
    parser.add_argument("--matrix", action="store_true", help="输出26项执行清单；不运行模型/服务")
    parser.add_argument("--matrix-results", type=Path, nargs="+", help="离线映射已有运行；缺少现场保持未覆盖")
    parser.add_argument("--matrix-fixture", type=Path, default=DEFAULT_MATRIX_FIXTURE)
    parser.add_argument("--matrix-cases", type=Path, default=DEFAULT_MATRIX_CASES)
    parser.add_argument("--interference-report", type=Path, nargs="*", default=[], help="引用已有设施干扰证据，不修改原始run")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.matrix_results:
            result = summarize_task_matrix(load_live_traces(args.matrix_results, args.interference_report), matrix_path=args.matrix_fixture, cases_path=args.matrix_cases)
        elif args.matrix:
            result = {"ok": True, "scope": "execution_manifest_only_no_runs", "tasks": load_task_matrix(args.matrix_fixture)}
        elif args.live_results:
            result = summarize_live_tasks(load_live_cases(args.live_fixture), load_live_traces(args.live_results, args.interference_report))
        else:
            result = evaluate(args.fixture, args.lifecycle_fixture)
    except (OSError, ValueError) as exc:
        result = {"ok": False, "error": str(exc)}
    if args.json or args.live_results or args.matrix or args.matrix_results:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    else:
        print(
            _text_report(result)
            if "summary" in result
            else f"Agent Kernel 验收失败：{result['error']}"
        )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
