"""Media Agent 媒体更新核对的语义、路由与 API 安全测试。"""

from __future__ import annotations

import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agent.domain_catalog.shared import _library_update_arguments
from app.agent.errors import AgentToolError
from app.agent.kernel.projection import DefaultProjector
from app.agent.models import Evidence, ToolResult
from app.agent.update_actions import check_library_updates


def _identity(arguments):
    return dict(arguments)


def _audit_result(*, status: str = "updates_available") -> ToolResult:
    return ToolResult(
        ok=status in {"updates_available", "up_to_date"},
        status=status,
        summary="发现 1 集已播但本地尚未收录"
        if status == "updates_available"
        else "未找到",
        data={"query": "黑镜", "missing_count": 1},
        evidence=[
            Evidence("media_servers+tmdb", "安全审计", "2026-08-01T10:00:00+08:00")
        ],
    )


def _query_audit_result(query: str, *, status: str) -> ToolResult:
    """为批量核对提供完整、可扁平化的单项审计 DTO。"""
    missing = (
        [{"season": 1, "episode": 2}]
        if status == "updates_available"
        else []
    )
    return ToolResult(
        ok=status in {"updates_available", "up_to_date", "comparison_unavailable"},
        status=status,
        summary={
            "updates_available": "发现 1 集已播但本地尚未收录",
            "up_to_date": "截至指定日期，已播普通剧集均已收录",
            "ambiguous": "媒体库命中多个同名剧集，暂不做猜测",
            "unavailable": "媒体服务器暂时不可用，无法完成剧集审计",
            "comparison_unavailable": "本地条目存在，但版本比较不可用",
        }[status],
        data={
            "query": query,
            "title": query,
            "tmdb_id": "12345" if status in {"updates_available", "up_to_date"} else "",
            "local_episode_count": 2 if status != "unavailable" else None,
            "expected_aired": 3 if status != "unavailable" else None,
            "missing_count": len(missing) if status != "unavailable" else None,
            "latest_local": {"season": 1, "episode": 3}
            if status != "unavailable"
            else None,
            "latest_aired": {"season": 1, "episode": 3}
            if status != "unavailable"
            else None,
            "missing_sample": missing,
            "missing_sample_truncated": False,
        },
    )


class LibraryUpdateArgumentTests(unittest.TestCase):
    def test_single_or_bounded_batch_queries_are_normalized_without_changing_order(self):
        single = _library_update_arguments(
            {
                "query": "  黑镜  ",
                "media_type": "tv",
                "as_of": "2026-08-01",
            }
        )
        self.assertEqual(single["query"], "黑镜")
        self.assertNotIn("queries", single)
        self.assertEqual(single["media_type"], "tv")
        self.assertEqual(single["as_of"], "2026-08-01")
        self.assertTrue(single["refresh"])

        queries = ["  合成剧一  ", "合成剧二", "合成剧一", "合成剧三"]
        batch = _library_update_arguments(
            {
                "queries": queries,
                "media_type": "auto",
                "as_of": "2026-08-01",
            }
        )
        self.assertEqual(batch["queries"], ["合成剧一", "合成剧二", "合成剧三"])
        self.assertNotIn("query", batch)
        self.assertEqual(batch["media_type"], "auto")
        self.assertEqual(batch["as_of"], "2026-08-01")
        self.assertTrue(batch["refresh"])

        twenty = [f"边界剧集 {index:02d}" for index in range(20)]
        normalized = _library_update_arguments(
            {"queries": twenty, "media_type": "tv", "as_of": "2026-08-01"}
        )
        self.assertEqual(normalized["queries"], twenty)

    def test_update_argument_boundaries_types_refresh_and_mutual_exclusion(self):
        self.assertFalse(
            _library_update_arguments(
                {
                    "query": "黑镜",
                    "media_type": "tv",
                    "as_of": "2026-08-01",
                    "refresh": False,
                }
            )["refresh"]
        )

        invalid_arguments = [
            {"queries": [f"剧集 {index:02d}" for index in range(21)]},
            {"queries": []},
            {"queries": "不是列表"},
            {"queries": ["有效", 2]},
            {"queries": [""]},
            {"query": ""},
            {"query": ["不是字符串"]},
            {"query": "单项", "queries": ["批量"]},
            {"queries": ["批量"], "tmdb_id": "12345"},
            {"queries": ["批量"], "season": 1},
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(AgentToolError):
                    _library_update_arguments(arguments)

        for invalid_refresh in (0, 1, "true", "false", None, []):
            with self.subTest(refresh=invalid_refresh):
                with self.assertRaises(AgentToolError):
                    _library_update_arguments(
                        {"query": "黑镜", "refresh": invalid_refresh}
                    )


class LibraryUpdateActionTests(unittest.TestCase):
    def test_movie_checks_exact_library_presence_and_offers_safe_resource_followup(
        self,
    ):
        arguments = {
            "query": "沙丘2",
            "media_type": "movie",
            "tmdb_id": "693134",
            "season": None,
            "as_of": "2026-08-01",
        }
        sources = [
            {
                "server_type": "jellyfin",
                "server_name": "客厅媒体库",
                "web_url": "http://private.invalid",
                "items": [
                    SimpleNamespace(
                        type="Movie", name="沙丘 2", display_name="沙丘 2", year="2024"
                    ),
                    SimpleNamespace(
                        type="Episode", name="沙丘2", display_name="沙丘2", year="2024"
                    ),
                    SimpleNamespace(
                        type="Movie",
                        name="沙丘：第二部",
                        display_name="沙丘：第二部",
                        year="2024",
                    ),
                ],
                "error": "",
            }
        ]
        with (
            patch("app.agent.update_actions.audit_series_episodes") as audit,
            patch(
                "app.agent.update_actions.search_media_servers", return_value=sources
            ) as search,
        ):
            result = check_library_updates(arguments)
        audit.assert_not_called()
        search.assert_called_once_with("沙丘2", limit=50)
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "comparison_unavailable")
        self.assertEqual(result.data["media_type"], "movie")
        self.assertEqual(result.data["local_match_status"], "found")
        self.assertEqual(result.data["exact_match_count"], 1)
        self.assertEqual(result.data["possible_match_count"], 1)
        self.assertNotIn("web_url", result.data["sources"][0])
        self.assertEqual(
            result.data["resource_followups"],
            [
                {
                    "tool": "indexer.search_resources",
                    "label": "搜索《沙丘2》资源候选",
                    "arguments": {
                        "title": "沙丘2",
                        "media_type": "movie",
                        "year": "2024",
                    },
                }
            ],
        )
        self.assertFalse(result.data["comparison"]["available"])
        self.assertNotIn("693134", result.summary)

    def test_movie_presence_is_honest_for_possible_missing_and_unavailable_sources(
        self,
    ):
        possible = [
            {
                "server_type": "emby",
                "server_name": "Emby",
                "items": [
                    SimpleNamespace(
                        type="Movie",
                        name="Dune Part Two",
                        display_name="Dune Part Two",
                        year="2024",
                    )
                ],
                "error": "",
            }
        ]
        with patch(
            "app.agent.update_actions.search_media_servers", return_value=possible
        ):
            result = check_library_updates(
                {
                    "query": "沙丘2",
                    "media_type": "movie",
                    "tmdb_id": "",
                    "season": None,
                    "as_of": "2026-08-01",
                }
            )
        self.assertEqual(result.status, "review_required")
        self.assertEqual(result.data["local_match_status"], "possible")
        self.assertEqual(result.data["exact_match_count"], 0)
        with patch(
            "app.agent.update_actions.search_media_servers",
            return_value=[
                {
                    "server_type": "jellyfin",
                    "server_name": "Jellyfin",
                    "items": [],
                    "error": "",
                }
            ],
        ):
            result = check_library_updates(
                {
                    "query": "沙丘2",
                    "media_type": "movie",
                    "tmdb_id": "",
                    "season": None,
                    "as_of": "2026-08-01",
                }
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "not_found")
        self.assertEqual(result.data["local_match_status"], "not_found")
        with patch(
            "app.agent.update_actions.search_media_servers",
            side_effect=RuntimeError("secret"),
        ):
            result = check_library_updates(
                {
                    "query": "沙丘2",
                    "media_type": "movie",
                    "tmdb_id": "",
                    "season": None,
                    "as_of": "2026-08-01",
                }
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "unavailable")
        self.assertNotIn("secret", result.summary)

    def test_movie_presence_caps_public_match_rows_across_all_servers(self):
        sources = [
            {
                "server_type": "jellyfin",
                "server_name": "Jellyfin",
                "items": [
                    SimpleNamespace(
                        type="Movie",
                        name=f"候选 {index}",
                        display_name=f"候选 {index}",
                        year="2024",
                    )
                    for index in range(8)
                ],
                "error": "",
            },
            {
                "server_type": "emby",
                "server_name": "Emby",
                "items": [
                    SimpleNamespace(
                        type="Movie",
                        name="额外候选",
                        display_name="额外候选",
                        year="2025",
                    )
                ],
                "error": "",
            },
        ]
        with patch(
            "app.agent.update_actions.search_media_servers", return_value=sources
        ):
            result = check_library_updates(
                {
                    "query": "不存在的精确片名",
                    "media_type": "movie",
                    "tmdb_id": "",
                    "season": None,
                    "as_of": "2026-08-01",
                }
            )
        self.assertEqual(result.data["possible_match_count"], 9)
        self.assertEqual(result.data["matches_truncated"], 1)
        self.assertEqual(
            sum(source["returned"] for source in result.data["sources"]), 8
        )
        self.assertEqual(result.data["sources"][1]["returned"], 0)

    def test_movie_presence_does_not_claim_not_found_when_source_hits_search_cap(self):
        sources = [
            {
                "server_type": "jellyfin",
                "server_name": "Jellyfin",
                "items": [
                    SimpleNamespace(
                        type="Episode",
                        name=f"同名剧集 {index}",
                        display_name=f"同名剧集 {index}",
                        year="2024",
                    )
                    for index in range(50)
                ],
                "error": "",
            }
        ]
        with patch(
            "app.agent.update_actions.search_media_servers", return_value=sources
        ):
            result = check_library_updates(
                {
                    "query": "目标电影",
                    "media_type": "movie",
                    "tmdb_id": "",
                    "season": None,
                    "as_of": "2026-08-01",
                }
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "review_required")
        self.assertEqual(result.data["local_match_status"], "indeterminate")
        self.assertEqual(result.data["search_truncated_server_count"], 1)
        self.assertNotIn("未找到同名电影", result.summary)

    def test_movie_presence_scans_later_servers_before_capping_public_rows(self):
        sources = [
            {
                "server_type": "jellyfin",
                "server_name": "Jellyfin",
                "items": [
                    SimpleNamespace(
                        type="Movie",
                        name=f"候选 {index}",
                        display_name=f"候选 {index}",
                        year="2024",
                    )
                    for index in range(8)
                ],
                "error": "",
            },
            {
                "server_type": "emby",
                "server_name": "Emby",
                "items": [
                    SimpleNamespace(
                        type="Movie", name="沙丘 2", display_name="沙丘 2", year="2024"
                    )
                ],
                "error": "",
            },
        ]
        with patch(
            "app.agent.update_actions.search_media_servers", return_value=sources
        ):
            result = check_library_updates(
                {
                    "query": "沙丘2",
                    "media_type": "movie",
                    "tmdb_id": "",
                    "season": None,
                    "as_of": "2026-08-01",
                }
            )
        self.assertEqual(result.status, "comparison_unavailable")
        self.assertEqual(result.data["local_match_status"], "found")
        self.assertEqual(result.data["exact_match_count"], 1)
        self.assertEqual(result.data["possible_match_count"], 8)
        self.assertEqual(
            sum(source["returned"] for source in result.data["sources"]), 8
        )
        self.assertEqual(result.data["sources"][1]["items"][0]["match"], "exact_title")

    def test_tv_delegates_to_episode_audit_and_labels_definition(self):
        with patch(
            "app.agent.update_actions.audit_series_episodes",
            return_value=_audit_result(),
        ) as audit:
            result = check_library_updates(
                {
                    "query": "黑镜",
                    "media_type": "tv",
                    "tmdb_id": "42009",
                    "season": 7,
                    "as_of": "2026-08-01",
                }
            )
        audit.assert_called_once_with(
            {"query": "黑镜", "tmdb_id": "42009", "season": 7, "as_of": "2026-08-01"}
        )
        self.assertEqual(result.status, "updates_available")
        self.assertEqual(result.data["media_type"], "tv")
        self.assertEqual(
            result.data["check_definition"],
            "aired_normal_episodes_missing_from_enabled_media_servers",
        )

    def test_auto_not_found_does_not_claim_movie_or_series_is_current(self):
        with patch(
            "app.agent.update_actions.audit_series_episodes",
            return_value=_audit_result(status="not_found"),
        ):
            result = check_library_updates(
                {
                    "query": "未知标题",
                    "media_type": "auto",
                    "tmdb_id": "",
                    "season": None,
                    "as_of": "2026-08-01",
                }
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "cannot_determine")
        self.assertIn("无法可靠判断", result.summary)

    def test_batch_checks_ten_queries_in_order_with_bounded_concurrency_and_safe_errors(self):
        titles = [f"合成剧集 {index:02d}" for index in range(1, 11)]
        status_by_query = {
            titles[0]: "updates_available",
            titles[1]: "updates_available",
            titles[2]: "updates_available",
            titles[3]: "up_to_date",
            titles[4]: "up_to_date",
            titles[5]: "up_to_date",
            titles[6]: "ambiguous",
            titles[7]: "ambiguous",
            titles[8]: "unavailable",
        }
        state_lock = threading.Lock()
        seen_queries: list[str] = []
        active = 0
        max_active = 0

        def audit_side_effect(arguments):
            nonlocal active, max_active
            query = arguments["query"]
            with state_lock:
                seen_queries.append(query)
                active += 1
                max_active = max(max_active, active)
            try:
                # 给并发窗口留出确定的观测时间；不访问真实媒体库或资源站。
                time.sleep(0.01)
                if query == titles[-1]:
                    raise RuntimeError("secret token at http://user:pass@private.invalid")
                return _query_audit_result(query, status=status_by_query[query])
            finally:
                with state_lock:
                    active -= 1

        with patch(
            "app.agent.update_actions.audit_series_episodes",
            side_effect=audit_side_effect,
        ) as audit:
            result = check_library_updates(
                {
                    "queries": titles,
                    "media_type": "tv",
                    "as_of": "2026-08-01",
                    "refresh": False,
                }
            )

        self.assertEqual(audit.call_count, 10)
        self.assertCountEqual(seen_queries, titles)
        self.assertGreaterEqual(max_active, 2)
        self.assertLessEqual(max_active, 3)
        self.assertTrue(result.ok, "只要至少一个单项可判定，批量 ok 应取 any(single.ok)")
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.data["as_of"], "2026-08-01")
        self.assertIsInstance(result.data["checked_at"], str)
        self.assertTrue(result.data["checked_at"])
        self.assertEqual(
            result.data["check_definition"],
            "library_inventory_vs_tmdb_not_resource_release_availability",
        )
        self.assertEqual(
            result.data["counts"],
            {
                "requested": 10,
                "updates_available": 3,
                "up_to_date": 3,
                "uncertain": 4,
            },
        )

        items = result.data["items"]
        self.assertEqual([item["query"] for item in items], titles)
        self.assertEqual(
            [item["status"] for item in items[:-1]],
            [status_by_query[query] for query in titles[:-1]],
        )
        required_item_keys = {
            "query",
            "title",
            "status",
            "ok",
            "summary",
            "tmdb_id",
            "local_episode_count",
            "expected_aired",
            "missing_count",
            "latest_local",
            "latest_aired",
            "missing_sample",
        }
        for item in items:
            with self.subTest(query=item["query"]):
                self.assertTrue(required_item_keys <= item.keys())
        failed = items[-1]
        self.assertFalse(failed["ok"])
        self.assertIsNone(failed["missing_count"])
        self.assertIsNone(failed["local_episode_count"])
        self.assertNotIn(failed["status"], {"updates_available", "up_to_date"})
        public_result = str(result.to_dict())
        self.assertNotIn("secret token", public_result)
        self.assertNotIn("private.invalid", public_result)
        self.assertNotIn("user:pass", public_result)

    def test_batch_status_is_success_when_every_item_is_determinable(self):
        titles = ["可判定剧一", "可判定剧二"]
        with patch(
            "app.agent.update_actions.audit_series_episodes",
            side_effect=lambda arguments: _query_audit_result(
                arguments["query"], status="up_to_date"
            ),
        ):
            result = check_library_updates(
                {
                    "queries": titles,
                    "media_type": "tv",
                    "as_of": "2026-08-01",
                    "refresh": False,
                }
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.data["counts"]["uncertain"], 0)
        self.assertEqual(
            [item["status"] for item in result.data["items"]],
            ["up_to_date", "up_to_date"],
        )

    def test_batch_limits_missing_sample_and_default_projector_keeps_twenty_items(self):
        titles = [f"投影剧集 {index:02d}" for index in range(1, 21)]
        all_missing = [{"season": 1, "episode": episode} for episode in range(1, 8)]

        def audit_side_effect(arguments):
            result = _query_audit_result(
                arguments["query"], status="updates_available"
            )
            result.data.update(
                {
                    "local_episode_count": 0,
                    "expected_aired": 7,
                    "missing_count": 7,
                    "latest_local": None,
                    "latest_aired": {"season": 1, "episode": 7},
                    "missing_sample": list(all_missing),
                    "missing_sample_truncated": True,
                }
            )
            return result

        with patch(
            "app.agent.update_actions.audit_series_episodes",
            side_effect=audit_side_effect,
        ):
            result = check_library_updates(
                {
                    "queries": titles,
                    "media_type": "tv",
                    "as_of": "2026-08-01",
                    "refresh": False,
                }
            )

        items = result.data["items"]
        self.assertEqual(len(items), 20)
        self.assertEqual([item["query"] for item in items], titles)
        for item in items:
            with self.subTest(query=item["query"]):
                self.assertLessEqual(len(item["missing_sample"]), 5)
                self.assertTrue(item["missing_sample_truncated"])

        projected = json.loads(DefaultProjector().project(result).model_content)
        self.assertEqual(len(projected["data"]["items"]), 20)
        self.assertEqual([item["query"] for item in projected["data"]["items"]], titles)

    def test_twenty_long_titles_keep_every_fact_under_model_projection_budget(self):
        titles = [f"第{i:02d}部" + "长标题样本" * 23 for i in range(20)]
        def audit(arguments):
            result = _query_audit_result(arguments["query"], status="updates_available")
            result.summary = "存在已播缺集，请结合实际媒体库版本及季集编号核对。" * 6
            result.data["sources"] = [{"server_type": "jellyfin", "server_name": f"服务器{i}" + "媒体资料" * 16,
                                       "status": "ready", "truncated": False} for i in range(3)]
            result.data["missing_sample"] = [{"season": 1, "episode": n} for n in range(3, 8)]
            return result
        with patch("app.agent.update_actions.audit_series_episodes", side_effect=audit):
            result = check_library_updates(_library_update_arguments({"queries": titles, "as_of": "2026-08-01"}))
        self.assertGreater(len(json.dumps(result.to_dict(), ensure_ascii=False)), 24_000)
        projected = DefaultProjector().project(result).model_content
        model = json.loads(projected)
        self.assertLess(len(projected), 24_000)
        self.assertEqual([item["query"] for item in model["data"]["items"]], titles)
        self.assertTrue(all(item["latest_local"] for item in model["data"]["items"]))
        self.assertFalse(model.get("truncated"))
        self.assertEqual(len(result.data["items"][0]["sources"]), 3)

    def test_batch_preserves_unknown_metadata_counts_in_public_and_model_evidence(self):
        scalar = _query_audit_result("示例剧", status="up_to_date")
        scalar.data.update(unknown_air_date_count=1, ignored_unknown_local=2)
        with patch("app.agent.update_actions.audit_series_episodes", return_value=scalar):
            result = check_library_updates(_library_update_arguments({"queries": ["示例剧"], "as_of": "2026-08-01"}))
        model = json.loads(DefaultProjector().project(result).model_content)
        for data in (result.data, model["data"]):
            self.assertEqual(data["items"][0]["unknown_air_date_count"], 1)
            self.assertEqual(data["items"][0]["ignored_unknown_local"], 2)
            self.assertEqual(data["items"][0]["status"], "up_to_date")
        self.assertIn("不能宣称全部最新", " ".join(result.suggestions))

    def test_movie_comparison_unavailable_counts_as_uncertain_in_batch(self):
        titles = ["电影甲", "电影乙"]

        def movie_sources(query, *, limit):
            self.assertEqual(limit, 50)
            return [
                {
                    "server_type": "jellyfin",
                    "server_name": "测试媒体库",
                    "items": [
                        SimpleNamespace(
                            type="Movie",
                            name=query,
                            display_name=query,
                            year="2024",
                        )
                    ],
                    "error": "",
                }
            ]

        with patch(
            "app.agent.update_actions.search_media_servers",
            side_effect=movie_sources,
        ):
            result = check_library_updates(
                {
                    "queries": titles,
                    "media_type": "movie",
                    "as_of": "2026-08-01",
                    "refresh": False,
                }
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.data["counts"]["uncertain"], 2)
        self.assertEqual(
            [item["status"] for item in result.data["items"]],
            ["comparison_unavailable", "comparison_unavailable"],
        )
