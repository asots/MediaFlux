"""真实任务判据的离线对照；伪造轨迹只用于测验收器，不计真实成功率。"""

from __future__ import annotations
from typing import Any

import copy
import json

import pytest

from tools import eval_agent


@pytest.fixture
def cases():
    return {row["id"]: row for row in eval_agent.load_live_cases()}


def trace(case) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "case_version": case["version"],
        "source": "synthetic_test",
        "confirmed_executions": 0,
        "failure": None,
        "tools": [],
        "turns": [
            {"status": "success", "answer": "", "events": []} for _ in case["messages"]
        ],
        "model_calls": [
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
                "http_attempts": 1,
            }
        ],
    }


def receipt(name, arguments, data, **extra) -> dict[str, Any]:
    return {
        "tool": name,
        "arguments": arguments,
        "public": {"ok": True, "data": data},
        **extra,
    }


def test_cases_are_versioned_and_separate_from_frozen_old_corpus(cases):
    assert len(cases) == 7
    assert sum(row["split"] == "holdout" for row in cases.values()) == 3
    assert all(row["version"] == 1 for row in cases.values())
    assert eval_agent.DEFAULT_LIVE_FIXTURE != eval_agent.DEFAULT_FIXTURE


@pytest.mark.parametrize(
    "mutation",
    [
        {"version": True},
        {"version": 2},
        {"split": "unknown"},
        {"messages": []},
        {"rubric": {"kind": "unknown"}},
        {"rubric": {"kind": "recommendation", "count": 0}},
        {"rubric": {"kind": "actor_count"}},
        {"rubric": {"kind": "actor_count", "actor": ""}},
        {"rubric": {"kind": "recommendation", "count": 10}},
    ],
)
def test_invalid_case_contracts_fail_closed(tmp_path, cases, mutation):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({**cases["actor_count"], **mutation}))
    with pytest.raises(ValueError):
        eval_agent.load_live_cases(path)


def test_actor_requires_actor_metadata_total_and_answer_agreement(cases):
    case = cases["actor_count"]
    row = trace(case)
    row["turns"][0]["answer"] = "媒体库周星驰参演的电影共有35部。"
    row["tools"] = [
        receipt(
            "library.search",
            {
                "query": "周星驰",
                "search_by": "actor",
                "count_only": True,
                "media_type": "movie",
            },
            {
                "total": 35,
                "match_basis": "actor_metadata",
                "sources": [{"status": "ready"}],
            },
        )
    ]
    assert eval_agent.assess_live_task(case, row)["passed"]
    for changes in ({"total": None}, {"total": 3}):
        changed = copy.deepcopy(row)
        changed["tools"][0]["public"]["data"].update(changes)
        assert not eval_agent.assess_live_task(case, changed)["passed"]
    row["tools"][0]["arguments"]["search_by"] = "title"
    assert not eval_agent.assess_live_task(case, row)["passed"]


def test_week_requires_all_pages_fresh_source_and_all_delivered_titles(cases):
    case = cases["tencent_week"]
    row = trace(case)
    row["turns"][0]["answer"] = "周一甲，周二乙，周三丙。"
    data = {
        "week_start": "2026-10-12",
        "week_end": "2026-10-18",
        "requested_dates": [f"2026-10-{day}" for day in range(12, 19)],
        "sources": [{"status": "ok"}],
        "total": 3,
    }
    items = [
        {"stable_id": str(index), "date": str(index), "title": title}
        for index, title in enumerate("甲乙丙")
    ]
    row["tools"] = [
        receipt(
            "discovery.anime_calendar",
            {"day": "week", "source": "tencent"},
            {**data, "items": items[:2], "has_more": True},
        ),
        receipt(
            "discovery.anime_calendar",
            {"day": "week", "source": "tencent"},
            {**data, "items": items[2:], "has_more": False},
        ),
    ]
    assert eval_agent.assess_live_task(case, row)["passed"]
    row["tools"].pop()
    assert not eval_agent.assess_live_task(case, row)["passed"]
    row["tools"][0]["public"]["data"]["sources"][0]["status"] = "stale"
    assert not eval_agent.assess_live_task(case, row)["passed"]


def test_recommendation_requires_ten_unique_verified_absent_titles(cases):
    case = cases["guoman_ten"]
    row = trace(case)
    items = [
        {
            "tmdb_id": str(index),
            "title": f"动画{index}部",
            "year": "2016",
            "library_status": "missing",
        }
        for index in range(10)
    ]
    row["turns"][0]["answer"] = ",".join(item["title"] + "(2016)" for item in items)
    row["tools"] = [
        receipt(
            "discovery.recommend",
            {
                "exclude_library": True,
                "region": "CN",
                "genre": "动画",
                "media_type": "tv",
            },
            {"items": items},
        )
    ]
    assert eval_agent.assess_live_task(case, row)["passed"]
    for key, value in (
        ("library_status", "unknown"),
        ("year", "2015"),
        ("tmdb_id", "1"),
    ):
        changed = copy.deepcopy(row)
        changed["tools"][0]["public"]["data"]["items"][0][key] = value
        assert not eval_agent.assess_live_task(case, changed)["passed"]


def test_fixed_twenty_task_cannot_pass_with_only_ten_or_unknown_library_state(cases):
    case = cases["guoman_twenty"]
    assert case["messages"] == [
        "推荐20部2016年起的中国大陆动画剧集，排除库里已有，自动翻页，简明列标题年份即可，只读"
    ]
    row = trace(case)
    items = [
        {
            "tmdb_id": str(i),
            "title": f"动画{i}部",
            "year": "2016",
            "library_status": "missing",
        }
        for i in range(20)
    ]
    row["turns"][0]["answer"] = ",".join(item["title"] + "(2016)" for item in items)
    arguments = {
        "exclude_library": True,
        "region": "CN",
        "genre": "动画",
        "media_type": "tv",
    }
    row["tools"] = [receipt("discovery.recommend", arguments, {"items": items[:10]})]
    assert not eval_agent.assess_live_task(case, row)["passed"]
    row["tools"].append(
        receipt("discovery.recommend", arguments, {"items": items[10:]})
    )
    assert eval_agent.assess_live_task(case, row)["passed"]
    row["tools"][1]["public"]["data"]["items"][0]["library_status"] = "indeterminate"
    assert not eval_agent.assess_live_task(case, row)["passed"]


@pytest.mark.parametrize(
    ("title", "answer", "matched"),
    [
        ("修仙", "20. **凡人修仙传**（2020）", False),
        ("修仙", "20. **修仙传**（2020）", False),
        ("Black Mirror", "20. **The Black Mirror** (2020)", False),
        ("Black Mirror", "20. **Black Mirror Chronicles** (2020)", False),
        ("Mirror", "20. DarkMirror (2020)", False),
        ("Mirror", "20. Mirrorverse (2020)", False),
        ("Mirror", "20. Dark-Mirror (2020)", False),
        ("Mirror", "20. Mirror-verse (2020)", False),
        ("修仙", "| 20 | **凡人修仙传** | 2020 |", False),
        ("修仙", "20. **修仙**\n2020", False),
        ("修仙", "| 20 | **修仙** |\n| 2020 |", False),
        ("修仙", "20. **修仙**（20200）", False),
        ("修仙", "20. **修仙**（12020）", False),
        ("修仙", "20. **修仙**（2020）", True),
        ("修仙", "20. __修仙__ (2020)", True),
        ("修仙", "- *修仙* (2020)", True),
        ("修仙", "+ `修仙` (2020)", True),
        ("修仙", "20. **《修仙》**（2020）", True),
        ("修仙", "修仙 2020", True),
        ("修仙", "另一部(2019),修仙(2020)", True),
        ("修仙", "| 20 | **修仙** | **2020** |", True),
        ("Black Mirror", "| 20 | Black Mirror | 2020 |", True),
        ("修仙", "20. [修仙](/media/288233)（2020）", True),
        ("修仙", "20. [凡人修仙传](/media/106449)（2020）", False),
        ("C++: [修仙](续)", "20. **C++: [修仙](续)**（2020）", True),
    ],
)
def test_recommendation_counts_complete_titles_and_same_line_years(
    cases, title, answer, matched
):
    case = cases["guoman_twenty"]
    assert case["rubric"]["count"] == 20
    row = trace(case)
    items = [
        {
            "tmdb_id": str(index),
            "title": f"国漫{index}部",
            "year": "2016",
            "library_status": "missing",
        }
        for index in range(19)
    ] + [
        {
            "tmdb_id": "288233",
            "title": title,
            "year": "2020",
            "library_status": "missing",
        }
    ]
    row["turns"][0]["answer"] = "\n".join(
        [f"{item['title']} (2016)" for item in items[:19]] + [answer]
    )
    row["tools"] = [
        receipt(
            "discovery.recommend",
            {
                "exclude_library": True,
                "region": "CN",
                "genre": "动画",
                "media_type": "tv",
            },
            {
                "items": items,
                "library_check": {
                    "items": [
                        {
                            "tmdb_id": "106449",
                            "title": "凡人修仙传",
                            "year": "2020",
                            "library_status": "present",
                        }
                    ]
                },
            },
        )
    ]
    result = eval_agent.assess_live_task(case, row)
    assert result["evidence"]["recommended_items_matched"] == 19 + int(matched)
    assert result["checks"]["enough_verified_absent_titles"] is matched
    assert result["passed"] is matched


def test_empty_isolated_watchlist_is_not_a_success(cases):
    case = cases["watchlist_updates"]
    row = trace(case)
    row["turns"][0]["answer"] = "当前隔离环境没有收藏，无法核对更新。"
    row["tools"] = [
        receipt("discovery.watchlist_summaries", {}, {"count": 0, "items": []})
    ]
    result = eval_agent.assess_live_task(case, row)
    assert result["outcome"] == "not_configured" and not result["passed"]
    assert result["checks"]["empty_watchlist_disclosed"]


def test_inventory_receipt_is_supplementary_not_a_relaxed_success_threshold(cases):
    case = cases["guoman_twenty"]
    row = trace(case)
    items = [
        {
            "tmdb_id": str(i),
            "title": f"国漫{i}部",
            "year": "2016",
            "library_status": "missing",
        }
        for i in range(20)
    ]
    row["turns"][0]["answer"] = "\n".join(f"{item['title']} (2016)" for item in items)
    row["tools"] = [
        receipt(
            "discovery.recommend",
            {
                "exclude_library": True,
                "region": "CN",
                "genre": "动画",
                "media_type": "tv",
            },
            {"items": items[:16], "library_check": {"items": items}},
        )
    ]
    result = eval_agent.assess_live_task(case, row)
    assert not result["passed"]
    assert result["evidence"] == {
        "recommended_items_matched": 16,
        "same_batch_verified_matches": 20,
    }
    for key, value in (("library_status", "indeterminate"), ("year", "2017")):
        changed = copy.deepcopy(row)
        changed["tools"][0]["public"]["data"]["library_check"]["items"][-1][key] = value
        result = eval_agent.assess_live_task(case, changed)
        assert not result["passed"]
        assert result["evidence"]["same_batch_verified_matches"] == 19


def test_scope_aba_requires_scope_arguments_and_original_titles(cases):
    case = cases["scope_aba"]
    row = trace(case)
    row["turns"][0]["answer"] = "**甲**、**乙**、**丙**、**丁**、**戊**"
    row["turns"][1]["answer"] = "动漫1、动漫2、动漫3、动漫4、动漫5"
    row["turns"][2]["answer"] = "甲、乙、丙、丁、戊"
    row["tools"] = [
        receipt(
            "media.recently_added",
            {"library_name": "电视剧"},
            {
                "scope": {"library": {"name": "电视剧"}},
                "granularity": "work",
                "items": [
                    {"series_name": title, "name": "单集标题"} for title in "甲乙丙丁戊"
                ],
            },
            turn=1,
        ),
        receipt(
            "media.recently_added",
            {"library_name": "动漫"},
            {
                "scope": {"library": {"name": "动漫"}},
                "granularity": "work",
                "items": [{"series_name": f"动漫{i}"} for i in range(1, 6)],
            },
            turn=2,
        ),
    ]
    assert eval_agent.assess_live_task(case, row)["passed"]
    row["turns"][2]["answer"] = "动漫1、动漫2、动漫3、动漫4、动漫5"
    assert not eval_agent.assess_live_task(case, row)["passed"]


def test_twenty_preview_needs_full_frozen_batch_cancel_and_zero_execution(cases):
    case = cases["preview_twenty_cancel"]
    row = trace(case)
    row["turns"][0].update(
        status="approval_required",
        events=[
            {
                "type": "effect.approval_required",
                "payload": {"plan": {"plan_id": "frozen-plan"}},
            }
        ],
    )
    row["tools"] = [
        receipt(
            "ingest.submit",
            {"positions": list(range(1, 21))},
            {"count": 20, "resources": [{"position": i} for i in range(1, 21)]},
            preview=False,
        )
    ]
    row["tools"][0]["public"]["status"] = "confirmation_required"
    row["cancelled_plan"] = True
    result = eval_agent.assess_live_task(case, row)
    assert result["passed"] and result["necessary_confirmations"] == 1
    assert result["invalid_interventions"] == 0
    for key, value in (("cancelled_plan", False), ("confirmed_executions", 1)):
        assert not eval_agent.assess_live_task(case, {**row, key: value})["passed"]
    row["tools"][0]["arguments"]["positions"].pop()
    assert not eval_agent.assess_live_task(case, row)["passed"]


def test_missing_usage_stays_unknown_and_unsent_infrastructure_is_separate(cases):
    row = trace(cases["actor_count"])
    row["model_calls"][0]["usage"] = None
    result = eval_agent.summarize_live_tasks(
        list(cases.values()),
        [row, {**row, "classification": "infrastructure_failure_not_sent"}],
    )
    assert result["summary"]["cases"] == 1
    assert result["summary"]["infrastructure_not_sent"] == 1
    assert result["summary"]["usage"] is None
    assert result["summary"]["usage_unknown_calls"] == 1
    assert result["summary"]["success_rate"] == 0
    assert not result["summary"]["usage_complete"]
    with pytest.raises(ValueError):
        eval_agent.assess_live_task(cases["actor_count"], {**row, "case_version": 2})


def test_partial_usage_is_not_complete_and_guard_interference_is_not_model_failure(
    cases,
):
    row = trace(cases["tencent_week"])
    row["model_calls"][0]["usage"] = {"prompt_tokens": 10}
    row["safety_blocks"] = [{"kind": "business_http_write", "method": "POST"}]
    result = eval_agent.assess_live_task(cases["tencent_week"], row)
    assert result["outcome"] == "infrastructure_interference"
    assert not result["passed"] and not result["usage_complete"]
    assert result["usage"] == {"prompt_tokens": 10}
    assert result["usage_unknown_calls"] == 1


def test_live_cli_only_reads_saved_results(tmp_path, cases, monkeypatch, capsys):
    row = trace(cases["watchlist_updates"])
    row["tools"] = [receipt("discovery.watchlist_summaries", {}, {"count": 0})]
    row["turns"][0]["answer"] = "当前收藏为空"
    path = tmp_path / "result.json"
    path.write_text(json.dumps(row), encoding="utf-8")
    monkeypatch.setattr(
        eval_agent,
        "evaluate",
        lambda *args: pytest.fail("live CLI must not run offline runtime"),
    )
    assert eval_agent.main(["--live-results", str(path)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["cases"] == result["summary"]["not_configured"] == 1
    path.write_text(json.dumps({**row, "case_id": "unknown"}), encoding="utf-8")
    assert eval_agent.main(["--live-results", str(path)]) == 1
    assert "未知任务" in json.loads(capsys.readouterr().out)["error"]


def test_event_export_appends_public_events_and_redacts_secrets(tmp_path):
    path = tmp_path / "events.jsonl"
    payload = {
        "text": 'credential-value\nquoted"secret',
        "usage": {"total_tokens": 12},
        "raw": "private",
        "nested": {"provider_state": "opaque", "headers": {"Authorization": "private"}},
    }
    for _ in range(2):
        eval_agent.export_live_event(
            path, "model", payload, redactions=['credential-value\nquoted"secret']
        )
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["payload"]["usage"] == {"total_tokens": 12}
    assert (
        "credential-value" not in path.read_text()
        and "private" not in path.read_text()
        and "opaque" not in path.read_text()
    )
