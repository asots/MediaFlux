from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import eval_agent


class AgentKernelEvalTests(unittest.TestCase):
    def fixture_rows(self) -> list[dict]:
        return [json.loads(line) for line in eval_agent.DEFAULT_FIXTURE.read_text().splitlines()]

    def write_cases(self, rows: list[dict]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "cases.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        return path

    def test_all_corpus_and_effect_gate_pass(self) -> None:
        result = eval_agent.evaluate()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["summary"]["cases"], 21)
        self.assertEqual(result["summary"]["failed"], 0)
        legacy = result["cohorts"]["legacy_unverified"]
        self.assertEqual((legacy["cases"], legacy["passed"], legacy["failed"]), (20, 20, 0))
        self.assertEqual(result["cohorts"]["simulated_user_real_model"]["passed"], 1)
        self.assertGreaterEqual(result["summary"]["candidate_count_min"], 6)
        self.assertLessEqual(result["summary"]["candidate_count_max"], 12)
        self.assertTrue(result["summary"]["effect_gate_valid"])
        self.assertEqual(result["summary"]["lifecycle_failed"], 0)
        self.assertGreaterEqual(result["summary"]["lifecycle_cases"], 4)
        self.assertEqual(result["invalid_effect_tools"], [])

    def test_eval_uses_native_first_turn_history_context(self) -> None:
        captured = []
        retrieve = eval_agent.CapabilityRetriever.retrieve

        def observe(retriever, message, catalog, *, context):
            if context.get("owner") == "offline-eval":
                captured.append((message, context))
            return retrieve(retriever, message, catalog, context=context)

        with patch.object(eval_agent.CapabilityRetriever, "retrieve", observe):
            result = eval_agent.evaluate()
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(captured), 21)
        for message, context in captured:
            self.assertEqual(context["recent_user_messages"], (message,))
            self.assertEqual(context["recent_tool_names"], ())
            self.assertEqual(context["reference_kinds"], ())

    def test_original_twenty_contracts_are_unchanged(self) -> None:
        rows = self.fixture_rows()
        self.assertEqual(len(rows), 21)
        self.assertTrue(all(set(row) == {"id", "message", "required"} for row in rows[:20]))
        digest = hashlib.sha256(json.dumps(
            rows[:20], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        self.assertEqual(digest, "5575a8f13071cdb27a41de14d34ea7f3769ec0d1f5272cb2ea51d22c5310c926")

    def test_optional_provenance_and_legacy_normalization(self) -> None:
        row = {"id": " x ", "message": " query ", "required": ["library.search", " library.search "]}
        legacy = eval_agent.load_cases(self.write_cases([row]))[0]
        self.assertEqual((legacy.case_id, legacy.message, legacy.required), ("x", "query", ("library.search",)))
        self.assertEqual(legacy.provenance, {"kind": "legacy_unverified"})
        row["provenance"] = {"kind": "synthetic"}
        self.assertEqual(eval_agent.load_cases(self.write_cases([row]))[0].provenance, {"kind": "synthetic"})
        for provenance in (None, {}, {"kind": "production_user_real_model"}, {"kind": "synthetic", "extra": True}):
            with self.subTest(provenance=provenance):
                row["provenance"] = provenance
                with self.assertRaisesRegex(ValueError, "provenance"):
                    eval_agent.load_cases(self.write_cases([row]))

    def test_real_model_evidence_is_self_contained_and_not_final_failure(self) -> None:
        row = self.fixture_rows()[-1]
        case = eval_agent.load_cases(self.write_cases([row]))[0]
        self.assertEqual(case.required, ("library.search", "library.count_series_episodes"))
        provenance = case.provenance
        self.assertEqual(provenance["kind"], "simulated_user_real_model")
        self.assertEqual(provenance["source_commit"], "abb81acda678b26048e6c85e9f813aeb8f8020e2")
        excerpt = provenance["excerpt"]
        self.assertEqual(case.message, "查一下 Jellyfin 里是否有《9号秘事》，再统计一下《黑镜》实际收录了多少集。只查询现有媒体库，不找资源、不刷新、不播放、不修改。")
        self.assertEqual(excerpt["initial_missing"], ["library.search"])
        self.assertEqual(excerpt["http_status"], 200)
        self.assertEqual((excerpt["final_status"], excerpt["business_pass"]), ("success", True))
        self.assertEqual(excerpt["completed_tool_calls"], [
            {"model_index": 1, "name": "library__count_series_episodes", "arguments": {"query": "黑镜"}},
            {"model_index": 1, "name": "agent__capabilities", "arguments": {"tool_names": ["library.search"]}},
            {"model_index": 2, "name": "library__search", "arguments": {"query": "9号秘事"}},
        ])
        public = json.dumps(row, ensure_ascii=False)
        for private_marker in (".superpowers", "http://", "https://", "Authorization", "Bearer ", "session_id", "user.env"):
            self.assertNotIn(private_marker, public)

    def test_provided_evidence_is_always_validated(self) -> None:
        row = self.fixture_rows()[-1]
        edits = [
            (("kind",), "unknown"),
            (("source_commit",), "main"),
            (("source_commit",), 1),
            (("extra",), True),
            (("evidence_sha256",), {}),
            (("evidence_sha256", "model-request-1.json"), "bad-hash"),
            (("evidence_sha256", "../private.json"), "a" * 64),
            (("excerpt",), {}),
            (("excerpt", "message"), "different user request"),
            (("excerpt", "initial_tools"), [None]),
            (("excerpt", "initial_missing"), []),
            (("excerpt", "http_status"), True),
            (("excerpt", "business_pass"), "true"),
            (("excerpt", "completed_tool_calls"), [{"name": "library__search"}]),
        ]
        for keys, value in edits:
            with self.subTest(keys=keys, value=value):
                candidate = copy.deepcopy(row)
                target = candidate["provenance"]
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaisesRegex(ValueError, "provenance"):
                    eval_agent.load_cases(self.write_cases([candidate]))
        with self.assertRaisesRegex(ValueError, "provenance"):
            eval_agent.load_cases(self.write_cases([{**row, "provenance": {"kind": "simulated_user_real_model"}}]))

    def test_cohort_report_keeps_initial_miss_and_successful_evidence_separate(self) -> None:
        rows = self.fixture_rows()
        # 只替换召回结果测试报告契约，不把旧首窗缺失改成绿色 required。
        required = {row["message"]: tuple(row["required"]) for row in rows}
        real = rows[-1]
        required[real["message"]] = ("library.count_series_episodes",)
        with patch.object(eval_agent.CapabilityRetriever, "retrieve", side_effect=lambda message, *args, **kwargs: SimpleNamespace(names=required[message])):
            output = io.StringIO()
            with redirect_stdout(output):
                status = eval_agent.main(["--json"])
        result = json.loads(output.getvalue())
        self.assertEqual(status, 1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["recall_scope"], "initial_window_corpus_only")
        self.assertEqual(result["summary"]["retrieval_recall"], round(20 / 21, 4))
        self.assertEqual(result["cohorts"]["legacy_unverified"]["passed"], 20)
        self.assertIsNone(result["cohorts"]["synthetic"]["retrieval_recall"])
        cohort = result["cohorts"]["simulated_user_real_model"]
        self.assertEqual((cohort["cases"], cohort["passed"], cohort["failed"]), (1, 0, 1))
        self.assertEqual(cohort["missing"], {real["id"]: ["library.search"]})
        outcome = result["outcomes"][-1]
        self.assertEqual(outcome["missing"], ["library.search"])
        self.assertTrue(outcome["provenance"]["excerpt"]["business_pass"])
        report = eval_agent._text_report(result)
        self.assertIn("不代表全局真实用户召回率", report)
        self.assertIn("非生产用户自然故障", report)
        self.assertIn("首窗 FAIL 不等于最终任务失败", report)
        self.assertIn("证据最终状态=success", report)

    def test_invalid_fixture_fails_closed(self) -> None:
        row = {"id": "x", "message": "test", "required": ["agent.capabilities"]}
        for rows in ([{**row, "required": []}], [{**row, "unknown": 1}], [row, row]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    eval_agent.evaluate(self.write_cases(rows))

    def test_invalid_lifecycle_fixture_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-lifecycle.jsonl"
            path.write_text(
                '{"id":"x","message":"test","result":{},"followup":"done","expected":{}}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                eval_agent.evaluate_lifecycle(path)

    def test_json_cli_is_stable(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = eval_agent.main(["--json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["summary"]["failed"], 0)
        self.assertIn("summary", payload)
        self.assertEqual(payload["cohorts"]["legacy_unverified"]["failed"], 0)
        self.assertEqual(payload["cohorts"]["simulated_user_real_model"]["cases"], 1)


if __name__ == "__main__":
    unittest.main()
