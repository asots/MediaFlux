"""矩阵与领域判据的离线反例；这些轨迹绝不算真实验收结果。"""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from tests.test_agent_live_eval import receipt, trace
from tools import eval_agent


@pytest.fixture
def matrix_cases():
    return {row["id"]: row for row in eval_agent.load_live_cases(eval_agent.DEFAULT_MATRIX_CASES)}


def actor_receipt(actor="成龙", total=71):
    return receipt("library.search", {"query": actor, "search_by": "actor", "media_type": "movie", "count_only": True},
                   {"match_basis": "actor_metadata", "total": total,
                    "sources": [{"server_type": "jellyfin", "status": "ready", "total": total, "total_source": "TotalRecordCount"}]})


def test_manifest_has_26_tasks_and_six_contextual_runner_cases_without_rule_dsl(matrix_cases):
    matrix = eval_agent.load_task_matrix()
    assert len(matrix) == 26
    assert {row["id"] for row in matrix} == {f"T{index:02}" for index in range(1, 27)}
    assert {key for row in matrix for key in row["runner_cases"]} == matrix_cases.keys()
    assert len(matrix_cases["constraints_eight"]["messages"]) == 8
    assert all(row["acceptance"] and row["external_conditions"] and row["steps"] for row in matrix)
    assert all("checks" not in row and "observations" not in row for row in matrix)
    frozen = eval_agent.DEFAULT_LIVE_FIXTURE.read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == "568476cc5dde773059a0dbe8f831d83c10202fb6c62fc9f937a275b33c92432b"


@pytest.mark.parametrize("total,exercised", [(0, False), (36, False), (50, False), (51, True), (71, True)])
def test_actor_over_fifty_requires_actual_actor_server_total(matrix_cases, total, exercised):
    case = matrix_cases["actor_count_over_fifty"]
    row = trace(case)
    row["tools"] = [actor_receipt(total=total)]
    row["turns"][0]["answer"] = f"成龙参演电影总数为{total}，按演员元数据和Jellyfin统计。"
    result = eval_agent.assess_live_task(case, row)
    assert result["passed"] is exercised
    if not exercised:
        assert result["outcome"] == "not_exercised"
    row["tools"][0]["public"]["data"]["match_basis"] = "title_search"
    assert not eval_agent.assess_live_task(case, row)["passed"]


def test_actor_over_fifty_keeps_server_counts_and_delivery_on_same_receipt(matrix_cases):
    case = matrix_cases["actor_count_over_fifty"]
    row = trace(case)
    row["tools"] = [actor_receipt(total=71), actor_receipt(total=36)]
    row["turns"][0]["answer"] = "成龙参演电影总数71部，另一记录36部。"
    row["tools"][0]["public"]["data"]["sources"][0]["total"] = 1
    assert not eval_agent.assess_live_task(case, row)["checks"]["server_totals_verified"]
    row["tools"] = [actor_receipt(total=71)]
    row["turns"][0]["answer"] = "成龙参演电影总数50部。"
    result = eval_agent.assess_live_task(case, row)
    assert not result["passed"] and result["outcome"] == "failed"
    assert result["checks"]["observed_total_over_fifty"]


def test_dune_needs_actual_movie_entity_year_not_answer_keyword(matrix_cases):
    case = matrix_cases["dune_year"]
    row = trace(case)
    row["turns"][0]["answer"] = "媒体库已收录2021年《沙丘》。"
    row["tools"] = [receipt("library.search", {"query": "沙丘", "media_type": "movie"}, {
        "sources": [{"server_type": "jellyfin", "status": "ready", "items": [
            {"title": "沙丘", "year": "1984", "media_type": "movie"}]}]})]
    assert not eval_agent.assess_live_task(case, row)["passed"]
    row["tools"][0]["public"]["data"]["sources"][0]["items"].append(
        {"title": "沙丘", "year": "2021", "media_type": "movie"})
    result = eval_agent.assess_live_task(case, row)
    assert result["passed"] and result["evidence"]["other_entity_years"] == ["1984"]
    row["tools"][0]["public"]["data"]["sources"][0]["items"][-1]["media_type"] = "series"
    assert not eval_agent.assess_live_task(case, row)["passed"]


def test_empty_library_does_not_exercise_dune_identity_branch(matrix_cases):
    case = matrix_cases["dune_year"]
    row = trace(case)
    row["tools"] = [receipt("library.search", {"query": "沙丘"}, {"sources": [{"status": "ready", "items": []}]})]
    row["turns"][0]["answer"] = "查询未找到2021沙丘。"
    result = eval_agent.assess_live_task(case, row)
    assert not result["passed"] and result["outcome"] == "not_exercised"


@pytest.mark.parametrize("case_id", ["read_missing_query", "read_invalid_enum"])
def test_read_fault_not_triggered_is_not_exercised_even_when_actor_answer_correct(matrix_cases, case_id):
    case = matrix_cases[case_id]
    row = trace(case)
    row["tools"] = [actor_receipt("周星驰")]
    row["turns"][0]["answer"] = "周星驰参演电影总计71部。"
    result = eval_agent.assess_live_task(case, row)
    assert not result["passed"] and result["outcome"] == "not_exercised"


def recovered_trace(case):
    row = trace(case)
    row["matrix"] = {"scenario": case["id"], "applied": [{"tool": "library.search", "variant": case["rubric"]["fault"],
                      "after": {} if case["rubric"]["fault"] == "missing_query" else {"media_type": "matrix_invalid_enum"},
                      "error_code": "invalid_arguments"}]}
    row["tools"] = [{"tool": "library.search", "arguments": {}, "error": "invalid_arguments"}, actor_receipt("周星驰")]
    row["turns"][0].update(answer="周星驰参演电影总计71部。", events=[{"type": "tool.failed", "payload": {"code": "invalid_arguments"}}])
    return row


@pytest.mark.parametrize("case_id", ["read_missing_query", "read_invalid_enum"])
def test_read_recovery_requires_error_feedback_once_bounded_and_correct_answer(matrix_cases, case_id):
    case = matrix_cases[case_id]
    row = recovered_trace(case)
    result = eval_agent.assess_live_task(case, row)
    assert result["passed"] and result["execution_mode"] == "hybrid"
    for mutation in ("no_error_event", "duplicate_injection", "four_attempts", "wrong_total"):
        bad = copy.deepcopy(row)
        if mutation == "no_error_event":
            bad["turns"][0]["events"] = []
        elif mutation == "duplicate_injection":
            bad["matrix"]["applied"] *= 2
        elif mutation == "four_attempts":
            bad["tools"] += copy.deepcopy(bad["tools"])
        else:
            bad["turns"][0]["answer"] = "电影总计50部。"
        assert not eval_agent.assess_live_task(case, bad)["passed"], mutation


def eight_constraints_trace(matrix_cases):
    case = matrix_cases["constraints_eight"]
    row = trace(case)
    for number, (turn, message, season, resolution) in enumerate(zip(
        row["turns"], case["messages"], (1, 2, 2, 2, 3, 3, 3, 3),
        ("1080p", "1080p", "2160p", "2160p", "2160p", "2160p", "1080p", "1080p"),
    ), 1):
        turn["message"] = message
        if number == 8:
            turn["answer"] = row["turns"][6]["answer"]
            continue
        title = f"Black.Mirror.S{season:02}E01.{resolution}.SDR"
        item = {"position": 1, "result_id": f"synthetic-turn-{number}", "title": title}
        arguments = {"title": "黑镜", "season": season, "episode": 1, "resolution": resolution}
        if number >= 6:
            arguments["dynamic_range"] = "SDR"
        reference = f"synthetic-ref-{number}"
        document = {"ok": True, "data": {"items": [item]},
                    "reference_arguments": {"resource_candidates_ref": reference},
                    "candidate_numbers": [{"position": 1, "title": title, "technical_evidence": {
                        "resolution": resolution, "dynamic_range": "SDR", "dynamic_ranges": ["SDR"],
                        "basis": "release_title_only", "stream_verified": False}}]}
        row["tools"].append({"turn": number, "tool": "indexer.search_resources", "arguments": arguments,
                             "public": {"ok": True}, "model_content": json.dumps(document)})
        row["tools"].append({"turn": number, "tool": "indexer.present_candidates",
                             "arguments": {"resource_candidates_ref": reference, "positions": [1]},
                             "public": {"ok": True}, "model_content": json.dumps({"ok": True, "data": {"positions": [1]}})})
        turn["answer"] = title
    return row


def effective_query_trace(matrix_cases):
    """完整有效条件只在回执中；续查请求只保留 ref 与本轮覆盖。"""
    row = eight_constraints_trace(matrix_cases)
    query = {"title": "黑镜", "original_title": "", "english_title": "", "aliases": ["Black Mirror"],
             "year": 2011, "media_type": "tv", "season": 1, "episode": 1, "page": 1,
             "sort_mode": "relevance_desc", "sites": ["nyaa"], "limit": 20,
             "resolution": "1080p", "dynamic_range": "", "exclude_hdr": False}
    previous_ref = None
    for search, present in zip(row["tools"][::2], row["tools"][1::2]):
        number = search["turn"]
        effective = {**query, **search["arguments"], "exclude_hdr": number >= 4}
        search["arguments"] = ({"resource_candidates_ref": previous_ref, **{
            key: value for key, value in effective.items() if value != query[key]
        }} if previous_ref else dict(effective))
        for tool in (search, present):
            document = json.loads(tool["model_content"])
            document["data"]["effective_query"] = dict(effective)
            if tool is present:
                previous_ref = f"presented-ref-{number}"
                document["reference_arguments"] = {"resource_candidates_ref": previous_ref}
            elif number in (4, 5):
                item, spec = document["data"]["items"][0], document["candidate_numbers"][0]
                item["title"] = spec["title"] = item["title"].replace(".SDR", ".WEB-DL")
                spec["technical_evidence"].update(dynamic_range=None, dynamic_ranges=[])
                row["turns"][number - 1]["answer"] = item["title"]
            tool["model_content"] = json.dumps(document)
        query = effective
    return row


def test_effective_query_receipts_inherit_omitted_patches_and_require_manual_review(matrix_cases):
    row = effective_query_trace(matrix_cases)
    assert row["tools"][2]["arguments"] == {"resource_candidates_ref": "presented-ref-1", "season": 2}
    assert row["tools"][4]["arguments"] == {"resource_candidates_ref": "presented-ref-2", "resolution": "2160p"}
    # 无新有效条件回执时，也能按返回的新 ref 读取各自冻结的查询。
    row["tools"].append({"turn": 8, "tool": "indexer.present_candidates", "public": {"ok": True},
                         "arguments": {"resource_candidates_ref": "presented-ref-7", "positions": [1]},
                         "model_content": json.dumps({"ok": True, "data": {"positions": [1]}})})
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert result["outcome"] == "manual_review" and not result["passed"]
    assert result["failure_reasons"] == ["answer_semantics_verified"]
    for report in result["evidence"]["turns"]:
        delivery = report["structured_deliveries"][0]
        assert delivery["matches_constraints"] and delivery["scope_matches"]
        assert delivery["effective_query"]["season"] == report["constraints"]["season"]
        if report["searches"]:
            assert report["searches"][0]["effective_query"] == delivery["effective_query"]
    for report in result["evidence"]["turns"][3:5]:
        candidate = report["referenced_candidates"][0]
        assert report["constraints"]["dynamic_range"] is None
        assert candidate["technical_evidence"]["dynamic_range"] is None
        assert candidate["matches_constraints"]  # 排除 HDR 不把 unknown 改成 SDR。


def test_effective_query_clears_preserve_parent_ref_and_same_candidate_snapshot(matrix_cases):
    row = effective_query_trace(matrix_cases)
    search, present = row["tools"][4:6]
    clears = {"year": None, "aliases": [], "sites": []}
    search["arguments"].update(clears)
    for tool in row["tools"]:
        if tool["turn"] >= 3:
            document = json.loads(tool["model_content"])
            document["data"]["effective_query"].update(clears)
            tool["model_content"] = json.dumps(document)
    # 同一候选快照先明确 SDR/排除 HDR，再派生清除条件的子引用。
    narrowed = json.loads(present["model_content"])
    narrowed["data"]["effective_query"].update(dynamic_range="SDR", exclude_hdr=True)
    present["arguments"].update(dynamic_range="SDR", exclude_hdr=True)
    present["model_content"] = json.dumps(narrowed)
    for source, returned, patch in (
        ("presented-ref-3", "cleared-ref-3", {"dynamic_range": "", "exclude_hdr": False}),
        ("cleared-ref-3", None, {}), ("presented-ref-3", None, {}),
    ):
        document = {"ok": True, "data": {"positions": [1]}}
        if returned:
            document["data"]["effective_query"] = {**narrowed["data"]["effective_query"], **patch}
            document["reference_arguments"] = {"resource_candidates_ref": returned}
        row["tools"].append({"turn": 3, "tool": "indexer.present_candidates", "public": {"ok": True},
                             "arguments": {"resource_candidates_ref": source, "positions": [1], **patch},
                             "model_content": json.dumps(document)})
    row["tools"][6]["arguments"].update(dynamic_range="")
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert result["outcome"] == "manual_review" and not result["passed"]
    deliveries = result["evidence"]["turns"][2]["structured_deliveries"]
    assert [item["effective_query"]["dynamic_range"] for item in deliveries] == ["SDR", "", "", "SDR"]
    assert [item["effective_query"]["exclude_hdr"] for item in deliveries] == [True, False, False, True]
    assert all(item["candidate_ids"] == ["synthetic-turn-3"] for item in deliveries)
    for item in deliveries:
        assert all(item["effective_query"][key] == value for key, value in clears.items())


@pytest.mark.parametrize("positions", [[], [1]])
def test_empty_search_reference_remains_bound_without_inventing_candidates(matrix_cases, positions):
    row = effective_query_trace(matrix_cases)
    search, present = row["tools"][6:8]
    document = json.loads(search["model_content"])
    document["data"]["items"], document["candidate_numbers"] = [], []
    search["model_content"] = json.dumps(document)
    document = json.loads(present["model_content"])
    document["data"].update(positions=positions, technical_filter={"requested": {"resolution": "2160p", "exclude_hdr": True}})
    present["arguments"]["positions"] = positions
    present["model_content"] = json.dumps(document)
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    report = result["evidence"]["turns"][3]
    assert report["searches"][0]["candidate_count"] == 0
    assert report["structured_deliveries"][0]["candidate_ids"] == []
    assert report["structured_deliveries"][0]["reference_resolved"] is (not positions)
    assert result["outcome"] == ("failed" if positions else "manual_review")
    assert not result["passed"]


@pytest.mark.parametrize("reference,scoped", [("presented-ref-7", True), ("presented-ref-1", False), ("unknown-ref", False)])
def test_effective_query_is_bound_to_each_reference_not_latest(matrix_cases, reference, scoped):
    row = effective_query_trace(matrix_cases)
    row["tools"].append({"turn": 8, "tool": "indexer.present_candidates", "public": {"ok": True},
                         "arguments": {"resource_candidates_ref": reference, "positions": [1]},
                         "model_content": json.dumps({"ok": True, "data": {"positions": [1]}})})
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    delivery = result["evidence"]["turns"][-1]["structured_deliveries"][0]
    assert delivery["scope_matches"] is scoped and delivery["matches_constraints"] is scoped
    assert result["outcome"] == ("manual_review" if scoped else "failed")
    assert not result["passed"]


@pytest.mark.parametrize("tool_offset", [-2, -1])
@pytest.mark.parametrize("field,value", [("season", 2), ("resolution", "2160p"), ("dynamic_range", "HDR10"), ("exclude_hdr", False)])
def test_wrong_effective_query_cannot_fall_back_to_correct_arguments_or_candidates(matrix_cases, tool_offset, field, value):
    row = effective_query_trace(matrix_cases)
    tool = row["tools"][tool_offset]
    document = json.loads(tool["model_content"])
    document["data"]["effective_query"][field] = value
    tool["model_content"] = json.dumps(document)
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert result["outcome"] == "failed" and not result["passed"]
    assert not result["checks"]["turn_7_no_invalid_structured_delivery"]


@pytest.mark.parametrize("number", [4, 5, 6])
def test_exclude_hdr_unknown_is_allowed_until_explicit_sdr_is_required(matrix_cases, number):
    row = effective_query_trace(matrix_cases)
    search = row["tools"][(number - 1) * 2]
    document = json.loads(search["model_content"])
    item, spec = document["data"]["items"][0], document["candidate_numbers"][0]
    item["title"] = spec["title"] = item["title"].replace(".SDR", ".WEB-DL")
    spec["technical_evidence"].update(dynamic_range=None, dynamic_ranges=[])
    search["model_content"] = json.dumps(document)
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert result["checks"][f"turn_{number}_no_invalid_structured_delivery"] is (number < 6)
    assert result["outcome"] == ("manual_review" if number < 6 else "failed")
    assert not result["passed"]


@pytest.mark.parametrize("hdr", ["HDR10", "DoVi", "HLG"])
@pytest.mark.parametrize("evidence", ["unknown", "SDR"])
def test_exclude_hdr_rejects_known_hdr_title_even_with_correct_effective_query(matrix_cases, hdr, evidence):
    row = effective_query_trace(matrix_cases)
    search = row["tools"][6]
    document = json.loads(search["model_content"])
    item, spec = document["data"]["items"][0], document["candidate_numbers"][0]
    item["title"] = spec["title"] = item["title"].replace(".WEB-DL", f".{hdr}")
    spec["technical_evidence"].update(dynamic_range=None if evidence == "unknown" else "SDR",
                                      dynamic_ranges=[] if evidence == "unknown" else ["SDR"])
    search["model_content"] = json.dumps(document)
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert result["outcome"] == "failed" and not result["passed"]
    assert not result["checks"]["turn_4_no_invalid_structured_delivery"]


def test_eight_turns_reuses_observed_model_receipts_but_requires_language_review(matrix_cases):
    case = matrix_cases["constraints_eight"]
    row = eight_constraints_trace(matrix_cases)
    result = eval_agent.assess_live_task(case, row)
    assert not result["passed"] and result["outcome"] == "manual_review"
    assert result["failure_reasons"] == ["answer_semantics_verified"]
    assert len(result["evidence"]["turns"]) == 8
    assert not any(tool["turn"] == 8 for tool in row["tools"])
    assert result["evidence"]["final_referenced_candidate_ids"] == ["synthetic-turn-7"]
    assert result["evidence"]["turns"][-1]["referenced_candidates"][0]["source_turn"] == 7
    assert result["evidence"]["turns"][-1]["referenced_candidates"][0]["matches_constraints"]
    assert all(result["evidence"]["turns"][index]["constraints"]["dynamic_range"] is None for index in (3, 4))
    matrix = eval_agent.summarize_task_matrix([row])
    assert next(task for task in matrix["tasks"] if task["task_id"] == "T13")["status"] == "manual_review"


@pytest.mark.parametrize("wrong_resolution", [False, True])
def test_inline_candidate_evidence_keeps_eight_turn_acceptance_strict(matrix_cases, wrong_resolution):
    row = effective_query_trace(matrix_cases)
    for tool in row["tools"]:
        document = json.loads(tool["model_content"])
        for candidate in document.pop("candidate_numbers", []):
            item = next(item for item in document["data"]["items"]
                        if item["position"] == candidate["position"] and item["title"] == candidate["title"])
            item["technical_evidence"] = candidate["technical_evidence"]
            if wrong_resolution and tool["turn"] == 7:
                item["title"] = item["title"].replace("1080p", "2160p")
                item["technical_evidence"]["resolution"] = "2160p"
        tool["model_content"] = json.dumps(document)
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert not result["passed"]
    if wrong_resolution:
        assert result["outcome"] == "failed"
        assert any(reason != "answer_semantics_verified" for reason in result["failure_reasons"])
    else:
        assert result["outcome"] == "manual_review"
        assert result["failure_reasons"] == ["answer_semantics_verified"]
        assert all(report["structured_deliveries"][0]["matches_constraints"]
                   for report in result["evidence"]["turns"] if report["structured_deliveries"])


def test_last_turn_candidate_cannot_cover_missing_first_seven_turns(matrix_cases):
    case = matrix_cases["constraints_eight"]
    row = eight_constraints_trace(matrix_cases)
    row["tools"] = row["tools"][-2:]
    for tool in row["tools"]:
        tool["turn"] = 8
    result = eval_agent.assess_live_task(case, row)
    assert result["outcome"] == "failed"
    assert all(not result["checks"][f"turn_{number}_constraints_observed"] for number in range(1, 8))
    assert result["checks"]["turn_8_constraints_observed"]


@pytest.mark.parametrize("number,field,value", [
    (1, "season", 2), (2, "episode", 2), (3, "resolution", "1080p"),
    (4, "dynamic_range", "HDR10"), (5, "season", 2),
    (6, "dynamic_range", None), (7, "resolution", "2160p"), (8, "season", 2),
])
def test_each_turn_checks_its_own_scope_and_delivered_quality(matrix_cases, number, field, value):
    row = eight_constraints_trace(matrix_cases)
    if number == 8:
        present = {**copy.deepcopy(row["tools"][-1]), "turn": 8}
        present["arguments"]["resource_candidates_ref"] = "synthetic-ref-1"
        row["tools"].append(present)
    else:
        tool = next(tool for tool in row["tools"] if tool["turn"] == number and tool["tool"] == "indexer.search_resources")
        tool["arguments"][field] = value
        if field in {"resolution", "dynamic_range"}:
            document = json.loads(tool["model_content"])
            document["candidate_numbers"][0]["technical_evidence"][field] = value
            tool["model_content"] = json.dumps(document)
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert result["outcome"] == "failed"
    assert not result["checks"][f"turn_{number}_no_invalid_structured_delivery"]


@pytest.mark.parametrize("mutation", ["wrong_season", "wrong_show", "resolution", "unknown", "mixed_hdr", "missing_id", "unbound_selection", "receipt_identity"])
def test_eight_turns_checks_structured_delivery_identity_and_explicit_specs(matrix_cases, mutation):
    row = eight_constraints_trace(matrix_cases)
    search, present = row["tools"][-2:]
    document = json.loads(search["model_content"])
    item, spec = document["data"]["items"][0], document["candidate_numbers"][0]
    if mutation == "wrong_season":
        item["title"] = spec["title"] = item["title"].replace("S03", "S02")
    elif mutation == "wrong_show":
        item["title"] = spec["title"] = item["title"].replace("Black.Mirror", "Other.Show")
    elif mutation == "resolution":
        spec["technical_evidence"]["resolution"] = "2160p"
    elif mutation == "unknown":
        spec["technical_evidence"].update(dynamic_range=None, dynamic_ranges=[], video_codec="H.265", bit_depth="10-bit")
    elif mutation == "mixed_hdr":
        spec["technical_evidence"]["dynamic_ranges"] = ["SDR", "HDR10"]
    elif mutation == "missing_id":
        del item["result_id"]
    elif mutation == "unbound_selection":
        present["arguments"]["resource_candidates_ref"] = "unobserved-reference"
    else:
        present["model_content"] = json.dumps({"ok": True, "data": {"positions": [1]},
                                               "candidate_numbers": [{**spec, "title": "Other.Show.S03E01.1080p.SDR"}]})
    search["model_content"] = json.dumps(document)
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert result["outcome"] == "failed"
    assert not result["checks"]["turn_7_no_invalid_structured_delivery"]


def test_unknown_free_text_reference_is_not_automatically_endorsed_or_sdr(matrix_cases):
    row = eight_constraints_trace(matrix_cases)
    row["tools"].pop()  # 第 7/8 轮仅自由文本，没有结构化交付。
    search = row["tools"][-1]
    document = json.loads(search["model_content"])
    document["candidate_numbers"][0]["technical_evidence"].update(dynamic_range=None, dynamic_ranges=[])
    document["data"]["items"][0]["title"] += "[rarbg]"
    document["candidate_numbers"][0]["title"] += "[rarbg]"
    search["model_content"] = json.dumps(document)
    row["turns"][-1]["answer"] = "这项动态范围未知，不应作为 SDR 推荐：" + row["turns"][-1]["answer"]
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert result["outcome"] == "manual_review" and not result["passed"]
    referenced = result["evidence"]["turns"][-1]["referenced_candidates"]
    assert len(referenced) == 1 and referenced[0]["source_turn"] == 7
    assert referenced[0]["technical_evidence"]["dynamic_range"] is None
    assert not referenced[0]["matches_constraints"]
    row["turns"][-1]["answer"] = "这些都是 SDR，放心使用。"  # 无可靠候选身份的肯定句也不能通过。
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert not result["passed"] and result["outcome"] == "manual_review"
    assert result["evidence"]["final_referenced_candidate_ids"] == []


def test_empty_partial_search_and_explicit_empty_display_need_scope_review(matrix_cases):
    row = eight_constraints_trace(matrix_cases)
    search, present = row["tools"][-2:]
    document = json.loads(search["model_content"])
    document.update(candidate_numbers=[])
    document["data"].update(items=[], partial=True)
    search["model_content"] = json.dumps(document)
    present["arguments"]["positions"] = []
    present["model_content"] = json.dumps({"ok": True, "data": {"positions": []}})
    row["turns"][-1]["answer"] = "本轮部分来源未完成，暂无可证实符合条件的候选。"
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert result["outcome"] == "manual_review" and not result["passed"]
    assert result["checks"]["turn_8_constraints_observed"]
    assert result["checks"]["turn_7_no_invalid_structured_delivery"]
    assert result["evidence"]["turns"][6]["searches"][0]["partial"]
    assert result["evidence"]["final_referenced_candidate_ids"] == []


def test_hidden_full_receipt_cannot_replace_missing_observed_model_evidence(matrix_cases):
    row = eight_constraints_trace(matrix_cases)
    search = row["tools"][-2]
    search["full_model_content"] = search.pop("model_content")
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert result["outcome"] == "failed" and not result["passed"]
    assert not result["checks"]["turn_7_constraints_observed"]
    assert result["evidence"]["final_referenced_candidate_ids"] == []


@pytest.mark.parametrize("returned_positions", [[2], []])
def test_cross_turn_refinement_uses_actual_returned_positions_and_bound_scope(matrix_cases, returned_positions):
    row = eight_constraints_trace(matrix_cases)
    # 第 5 轮同一 S03E01 snapshot 包含未知 2160p 与明确 SDR 1080p。
    search = next(tool for tool in row["tools"] if tool["turn"] == 5 and tool["tool"] == "indexer.search_resources")
    document = json.loads(search["model_content"])
    unknown = document["candidate_numbers"][0]
    unknown["technical_evidence"].update(dynamic_range=None, dynamic_ranges=[])
    second = {"position": 2, "result_id": "cached-1080p-sdr", "title": "Black.Mirror.S03E01.1080p.SDR"}
    second_spec = {"position": 2, "title": second["title"], "technical_evidence": {
        "resolution": "1080p", "dynamic_range": "SDR", "dynamic_ranges": ["SDR"]}}
    document["data"]["items"].append(second)
    document["candidate_numbers"].append(second_spec)
    search["model_content"] = json.dumps(document)
    # 第 5 轮明确不展示未知项，第 7/8 轮不重搜。
    present = next(tool for tool in row["tools"] if tool["turn"] == 5 and tool["tool"] == "indexer.present_candidates")
    present["arguments"].update(resolution="2160p", dynamic_range="SDR")
    present["model_content"] = json.dumps({"ok": True, "data": {"positions": [],
        "technical_filter": {"requested": {"resolution": "2160p", "dynamic_range": "SDR"}}}})
    row["tools"] = [tool for tool in row["tools"] if tool["turn"] < 7]
    refinement = {"turn": 7, "tool": "indexer.present_candidates", "public": {"ok": True},
                  "arguments": {"resource_candidates_ref": "synthetic-ref-5", "positions": [1, 2],
                                "resolution": "1080p", "dynamic_range": "SDR"},
                  "model_content": json.dumps({"ok": True, "data": {"positions": returned_positions,
                      "technical_filter": {"requested": {"resolution": "1080p", "dynamic_range": "SDR"}}},
                      "candidate_numbers": [second_spec] if returned_positions else [],
                      "reference_arguments": {"resource_candidates_ref": "refined-same-snapshot"}})}
    row["tools"].append(refinement)
    # 后续引用使用回执返回的新 ref，不按标题猜测快照身份。
    row["tools"].append({**copy.deepcopy(refinement), "turn": 8,
                         "arguments": {**refinement["arguments"], "resource_candidates_ref": "refined-same-snapshot"}})
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert result["outcome"] == "manual_review" and not result["passed"]
    assert result["checks"]["turn_7_constraints_observed"] and result["checks"]["turn_8_constraints_observed"]
    for report in result["evidence"]["turns"][-2:]:
        assert not report["searches"]
        delivered = report["structured_deliveries"][0]
        assert delivered["positions"] == returned_positions
        assert delivered["requested_positions"] == [1, 2]
        assert delivered["candidate_ids"] == (["cached-1080p-sdr"] if returned_positions else [])
        assert delivered["matches_constraints"] and delivered["scope_matches"]


def test_refinement_subset_does_not_imply_incomplete_target_season(matrix_cases):
    row = eight_constraints_trace(matrix_cases)
    search, present = row["tools"][-2:]
    document = json.loads(search["model_content"])
    document["data"]["items"], document["candidate_numbers"] = [], []
    for position in range(1, 21):
        season = 3 if position <= 12 else 4
        dynamic_range = None if position < 12 else "HDR" if position == 12 else "SDR"
        title = f"Black.Mirror.S{season:02}E01.1080p.{dynamic_range or 'WEB-DL'}.group{position}"
        document["data"]["items"].append({"position": position, "title": title, "result_id": f"subset-{position}"})
        document["candidate_numbers"].append({"position": position, "title": title, "technical_evidence": {
            "resolution": "1080p", "dynamic_range": dynamic_range, "dynamic_ranges": [dynamic_range] if dynamic_range else []}})
    search["arguments"].pop("dynamic_range")
    search["model_content"] = json.dumps(document)
    present["arguments"].update(positions=list(range(1, 13)), resolution="1080p", dynamic_range="SDR")
    scope = {"candidate_count": 20, "considered_count": 12, "complete": False}
    present["model_content"] = json.dumps({"ok": True, "candidate_numbers": [], "data": {
        "positions": [], "scope": scope, "technical_filter": {
            "requested": {"resolution": "1080p", "dynamic_range": "SDR"}, "unknown_excluded": 11, "mismatched_excluded": 1}}})
    result = eval_agent.assess_live_task(matrix_cases["constraints_eight"], row)
    assert result["outcome"] == "manual_review" and not result["passed"]
    assert result["checks"]["turn_7_constraints_observed"] and result["checks"]["turn_8_constraints_observed"]
    delivery = result["evidence"]["turns"][6]["structured_deliveries"][0]
    assert delivery["scope"] == scope and delivery["positions"] == delivery["candidate_ids"] == []
    assert delivery["matches_constraints"] and len(delivery["requested_positions"]) == 12


def test_pagination_requires_actual_front_pages_and_inventory_counts():
    row = {"matrix": {"scenario": "hybrid_guoman_front_owned", "applied": []}}
    assert not all(eval_agent.assess_pagination_replay(row).values())
    row["matrix"].update(applied=[{"owned_ids": ["1", "2"]}], replay={
        "provenance": {"kind": "synthetic_contract_replay"},
        "pages": [{"page": 1, "ids": ["1"]}, {"page": 2, "ids": ["2"]}, {"page": 3, "ids": ["3"]}],
        "inventory_reads": [{"tool_index": 0, "server": "jellyfin", "media_type": "tv"}]})
    assert all(eval_agent.assess_pagination_replay(row).values())
    row["matrix"]["replay"]["inventory_reads"].append({"tool_index": 1, "server": "jellyfin", "media_type": "tv"})
    assert not eval_agent.assess_pagination_replay(row)["one_inventory_per_request_source_type"]


def test_seven_query_cases_and_pytest_cannot_become_26_matrix_passes():
    assert eval_agent.summarize_task_matrix([])["summary"]["tasks_with_runs"] == 0
    cases = eval_agent.load_live_cases()
    rows = [trace(case) for case in cases]
    result = eval_agent.summarize_task_matrix(rows)
    assert not result["ok"]
    assert result["summary"]["fully_reviewed_passed"] == 0
    for task in result["tasks"]:
        if task["task_id"] in {"T17", "T24", "T25"}:
            assert task["status"] == "not_exercised" and not task["runs"]
        if task["task_id"] in {"T03", "T12"}:
            assert task["status"] == "not_exercised"


def test_matrix_requires_both_read_faults_and_two_year_entities_for_full_task(matrix_cases):
    missing = recovered_trace(matrix_cases["read_missing_query"])
    invalid = recovered_trace(matrix_cases["read_invalid_enum"])
    invalid["tools"][1]["arguments"].pop("media_type")
    dune = trace(matrix_cases["dune_year"])
    dune["tools"] = [receipt("library.search", {"query": "沙丘"}, {"sources": [{"status": "ready", "items": [
        {"title": "沙丘", "year": "2021", "media_type": "movie"}]}]})]
    dune["turns"][0]["answer"] = "媒体库有《沙丘》（2021）。"
    report = eval_agent.summarize_task_matrix([missing, invalid, dune])
    assert next(task for task in report["tasks"] if task["task_id"] == "T14")["status"] == "failed"
    assert next(task for task in report["tasks"] if task["task_id"] == "T11")["status"] == "partial"
    report = eval_agent.summarize_task_matrix([missing])
    assert next(task for task in report["tasks"] if task["task_id"] == "T14")["status"] == "partial"


def test_harness_interference_annotation_does_not_rewrite_original_trace(tmp_path, matrix_cases):
    case = matrix_cases["actor_count_over_fifty"]
    row = trace(case)
    row.update(source="affected-run", tools=[actor_receipt()])
    row["turns"][0]["answer"] = "成龙电影总计71部。"
    result_path, note_path = tmp_path / "result.json", tmp_path / "note.json"
    result_path.write_text(json.dumps(row))
    note_path.write_text(json.dumps({"affected_run": "affected-run/actor_count_over_fifty", "classification": "infrastructure_interference", "cause": "reproduced temporary rename race"}))
    loaded = eval_agent.load_live_traces([result_path], [note_path])
    result = eval_agent.assess_live_task(case, loaded[0])
    assert not result["passed"] and result["outcome"] == "infrastructure_interference"
    assert "classification" not in json.loads(result_path.read_text())


def test_country_genre_adult_are_supplementary_and_frozen_threshold_unchanged():
    case = next(row for row in eval_agent.load_live_cases() if row["id"] == "guoman_twenty")
    row = trace(case)
    row["tools"] = [receipt("discovery.recommend", {}, {"items": [{"tmdb_id": "1", "title": "动画", "year": "2026"}]})]
    row["source_evidence"] = [{"items": [{"id": 1, "origin_country": ["JP"], "genre_ids": [16], "adult": False}]}]
    extra = eval_agent.live_supplementary_evidence(case, row)
    assert extra["source_metadata_rejected_ids"] == ["1"]
    assert not eval_agent.assess_live_task(case, row)["passed"]


def test_partial_calendar_does_not_get_promoted_by_supplementary_event_count():
    case = next(row for row in eval_agent.load_live_cases() if row["id"] == "tencent_week")
    row = trace(case)
    row["tools"] = [receipt("discovery.anime_calendar", {"day": "week", "source": "tencent"}, {
        "items": [{"title": "动画", "date": "2026-10-12", "update_time": "10:00", "audience": who} for who in ("member", "free")],
        "sources": [{"status": "partial"}], "has_more": False})]
    extra = eval_agent.live_supplementary_evidence(case, row)
    assert extra["returned_events"] == 2 and extra["distinct_title_date_time_slots"] == 1
    assert not eval_agent.assess_live_task(case, row)["passed"]


def test_matrix_cli_only_reads_manifest_or_recordings(monkeypatch, capsys, tmp_path, matrix_cases):
    monkeypatch.setattr(eval_agent, "evaluate", lambda *a: pytest.fail("no runtime may be started"))
    assert eval_agent.main(["--matrix"]) == 0
    assert len(json.loads(capsys.readouterr().out)["tasks"]) == 26
    row = trace(matrix_cases["hybrid_guoman_front_owned"])
    path = tmp_path / "result.json"
    path.write_text(json.dumps(row))
    assert eval_agent.main(["--matrix-results", str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    t10 = next(task for task in report["tasks"] if task["task_id"] == "T10")
    assert t10["status"] == "not_exercised"
