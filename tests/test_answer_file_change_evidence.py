"""Bounded DTO predicates; no arbitrary ready-schema or text authority."""
from copy import deepcopy

import pytest

from app.agent.public_view import format_partial_progress, sanitize_confirmed_answer
from tests.test_answer_completion import EXPLANATION, READ_RESULT

RECEIPT = {"ok": True, "status": "submitted", "summary": "文件变更请求已提交"}


@pytest.mark.parametrize("partial", [False, True])
def test_exact_sentences_preserved_but_mixed_claims_do_not_gain_authority(partial):
    text = EXPLANATION + "已全部执行完成。私密值PRIVATE-ID32。" + "文件变更中有1项结果未知。"
    text += "文件变更共5项，其中3项已完成，故全部完成。"
    if partial:
        answer = format_partial_progress("查询受阻", [("状态", READ_RESULT)],
                                         confirmed_result=RECEIPT, analysis_text=text)
    else:
        answer = sanitize_confirmed_answer(text, RECEIPT, results=[("状态", READ_RESULT)])
    assert EXPLANATION in answer
    assert "文件变更中有1项结果未知。" in answer
    assert "已全部执行完成" not in answer
    assert "PRIVATE-ID32" not in answer
    assert "故全部完成" not in answer


@pytest.mark.parametrize("mutation", ["short", "duplicate", "unknown-total", "bool-total", "truncated",
                                     "has-more", "bad-op", "bad-status", "bad-position", "other-snapshot", "conflicting-total"])
def test_incomplete_or_ambiguous_dto_does_not_prove_counts(mutation):
    result = deepcopy(READ_RESULT)
    dto = result["data"]["task"]
    rows = dto["operation_items"]
    results = [("状态", result)]
    if mutation == "conflicting-total":
        dto["total"] = 5
        dto["stats"]["total"] = 6
    elif mutation == "short":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1]["position"] = 1
    elif mutation == "unknown-total":
        dto["stats"].pop("total")
    elif mutation == "bool-total":
        dto["stats"]["total"] = True
    elif mutation == "truncated":
        result["truncated"] = True
    elif mutation == "has-more":
        dto["has_more"] = True
    elif mutation == "bad-op":
        rows[0]["operation"] = {}
    elif mutation == "bad-status":
        rows[0]["status"] = "skipped"
    elif mutation == "bad-position":
        rows[0]["position"] = True
    elif mutation == "other-snapshot":
        other = deepcopy(result)
        other["data"]["task"]["operation_items"][0]["status"] = "unknown"
        results.append(("另一项", other))
    answer = sanitize_confirmed_answer(EXPLANATION, RECEIPT, results=results)
    assert EXPLANATION not in answer
    assert "文件变更共5项" not in answer


@pytest.mark.parametrize("body", [
    "文件变更共5项，已完成5项，仍有0项未确认完成。",
    "第4项因写入被拒绝而受阻。",
    '已确认操作的可信系统结果（不是待执行计划）：{"summary":"' + EXPLANATION + '"}',
    '标签：' + EXPLANATION.split("。")[0] + "。",
])
def test_false_counts_reasons_and_markers_are_not_explanation(body):
    answer = sanitize_confirmed_answer(body, RECEIPT, results=[("状态", READ_RESULT)])
    assert "文件变更共5项" not in answer
    assert "第4项因" not in answer


def test_numeric_coincidence_and_synthetic_ready_are_not_file_change_evidence():
    synthetic = {"ok": True, "summary": EXPLANATION, "data": {"total": 5, "ready": 3}}
    # Tool summaries may be displayed as summaries; never as preserved analysis.
    answer = sanitize_confirmed_answer(EXPLANATION, RECEIPT, results=[("状态", synthetic)])
    assert answer.count(EXPLANATION) == 0  # public summary normalizes punctuation
    assert "\n\n" + EXPLANATION not in answer


def test_direct_receipt_dto_uses_the_same_predicates():
    receipt = deepcopy(RECEIPT)
    receipt["data"] = deepcopy(READ_RESULT["data"]["task"])
    assert EXPLANATION in sanitize_confirmed_answer(EXPLANATION, receipt)
    assert EXPLANATION in format_partial_progress("未完成", [], confirmed_result=receipt,
                                                  analysis_text=EXPLANATION)


def test_current_errors_remain_visible_beyond_recent_tool_window():
    answer = format_partial_progress("首个错误", [("查询", {"ok": True, "summary": "正常查询"})] * 9,
                                     confirmed_result=RECEIPT,
                                     current_errors=["另一个仍未解决的错误"])
    assert "另一个仍未解决的错误" in answer
