"""Agent 探索收藏、最近探索续句与确认边界回归。"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import Mock, patch

from app import database as db
from app.agent.discovery_watchlist_actions import (
    list_watchlist_summaries,
    remove_watchlist_arguments,
    watchlist_summaries_arguments,
    watchlist_summary_arguments,
)
from app.agent.errors import AgentToolError
from app.agent.rate_limit import agent_rate_limiter
from app.discovery.models import MediaCard
from tests.agent_kernel_test_harness import (
    get_kernel_test_service as get_agent_service,
)
from tests.agent_kernel_test_harness import (
    reset_kernel_test_service as reset_agent_service_for_tests,
)
from tests.support import IsolatedDatabaseTestCase


def _identity(arguments):
    return dict(arguments)


def _card(index: int = 1) -> MediaCard:
    return MediaCard(
        provider="tmdb",
        external_id=str(8800 + index),
        media_type="movie",
        title=f"候选影片 {index}",
        original_title=f"PRIVATE ORIGINAL {index}",
        year="2026",
        overview="PRIVATE OVERVIEW",
        poster_key=f"https://private.example/{index}?api_key=secret",
    )


class AgentDiscoveryWatchlistTests(IsolatedDatabaseTestCase):
    def setUp(self) -> None:
        with db.get_conn() as conn:
            conn.execute("DELETE FROM media_download_admissions")
            conn.execute("DELETE FROM media_subscription_candidates")
            conn.execute("DELETE FROM media_subscription_runs")
            conn.execute("DELETE FROM media_subscriptions")
            conn.execute("DELETE FROM media_watchlist")
            conn.execute("DELETE FROM agent_action_history")
        reset_agent_service_for_tests()
        agent_rate_limiter.reset()

    def tearDown(self) -> None:
        reset_agent_service_for_tests()
        agent_rate_limiter.reset()

    @staticmethod
    def _insert(
        *, external_id: str = "100", title: str = "收藏影片", media_type: str = "movie"
    ) -> int:
        db.add_media_watchlist(
            "tmdb", external_id, media_type, title, "2026", "PRIVATE_POSTER_KEY"
        )
        row = db.get_media_watchlist("tmdb", external_id, media_type)
        return int(row["id"])

    def test_read_summaries_only_expose_public_fields(self) -> None:
        number = self._insert()
        with patch(
            "app.agent.discovery_watchlist_actions.config.get_bool", return_value=True
        ):
            listed = list_watchlist_summaries({})
            single = list_watchlist_summaries({"watchlist_number": number})
        self.assertTrue(listed.ok)
        self.assertTrue(single.ok)
        self.assertEqual(single.data, listed.data)
        self.assertEqual(single.data["items"][0]["watchlist_number"], number)
        self.assertEqual(set(single.data), {"count", "items", "has_more", "next_before_number"})
        self.assertEqual(
            set(single.data["items"][0]),
            {"watchlist_number", "provider", "media_type", "title", "year"},
        )
        serialized = json.dumps(
            {"listed": listed.to_dict(), "single": single.to_dict()}, ensure_ascii=False
        )
        self.assertNotIn("PRIVATE_POSTER_KEY", serialized)
        self.assertNotIn("external_id", serialized)
        self.assertNotIn("created_at", serialized)

    def test_watchlist_pagination_reads_all_rows_without_false_total(self) -> None:
        expected = [self._insert(external_id=str(n), title=f"收藏 {n}") for n in range(45)]
        received = []
        arguments = {}
        with patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=True):
            for _ in range(3):
                result = list_watchlist_summaries(arguments)
                self.assertLessEqual(len(result.data["items"]), 20)
                self.assertEqual(result.data["count"], len(result.data["items"]))
                self.assertNotIn("共有", result.summary)
                page_numbers = [item["watchlist_number"] for item in result.data["items"]]
                if "before_number" in arguments:
                    self.assertTrue(all(number < arguments["before_number"] for number in page_numbers))
                    self.assertNotIn(arguments["before_number"], page_numbers)
                received.extend(page_numbers)
                cursor = result.data["next_before_number"]
                self.assertEqual(result.data["has_more"], cursor is not None)
                if cursor is not None:
                    self.assertEqual(cursor, page_numbers[-1])
                    self.assertIn(f"编号 {cursor} 已在本页返回", result.summary)
                    self.assertNotIn("before_number", result.summary)
                    self.assertTrue(result.suggestions)
                    for hint in (result.summary, *result.suggestions):
                        self.assertIn("已在本页返回", hint)
                        self.assertIn("编号严格小于", hint)
                        self.assertIn("不含游标本身", hint)
                    arguments = {"before_number": cursor}
                else:
                    self.assertIn("已到末页", result.summary)
                    self.assertEqual(result.suggestions, [])
        self.assertFalse(result.data["has_more"])
        self.assertEqual(received, list(reversed(expected)))
        # Web默认调用仍读取原有有界列表，没有被Agent页长截断。
        self.assertEqual(len(db.list_media_watchlist()), 45)

    def test_watchlist_sparse_high_numbers_are_fully_enumerated_across_pages(self) -> None:
        expected = [1_000_000_007 + index * 104_729 for index in range(45)]
        with db.get_conn() as conn:
            conn.executemany(
                "INSERT INTO media_watchlist(id,provider,external_id,media_type,title,year,poster_key,created_at) "
                "VALUES(?,'tmdb',?,'movie',?,'2026','PRIVATE_POSTER_KEY',?)",
                [(number, f"PRIVATE_EXTERNAL_{number}", f"收藏 {number}", f"PRIVATE_TIME_{index}")
                 for index, number in enumerate(expected)],
            )
        service = get_agent_service()
        received = []
        pages = []
        arguments = {}
        with patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=True):
            for _ in range(3):
                result = service.invoke("discovery.watchlist_summaries", arguments)["result"]
                self.assertTrue(result["ok"])
                data = result["data"]
                self.assertEqual(set(data), {"count", "items", "has_more", "next_before_number"})
                numbers = [item["watchlist_number"] for item in data["items"]]
                self.assertEqual(data["count"], len(numbers))
                self.assertLessEqual(data["count"], 20)
                self.assertEqual(numbers, sorted(numbers, reverse=True))
                if "before_number" in arguments:
                    self.assertTrue(all(number < arguments["before_number"] for number in numbers))
                pages.append(data["count"])
                received.extend(numbers)
                cursor = data["next_before_number"]
                self.assertEqual(data["has_more"], cursor is not None)
                if not data["has_more"]:
                    break
                self.assertEqual(cursor, numbers[-1])
                arguments = {"before_number": cursor}
        self.assertEqual(pages, [20, 20, 5])
        self.assertFalse(data["has_more"])
        self.assertIsNone(data["next_before_number"])
        self.assertEqual(received, list(reversed(expected)))
        self.assertEqual(len(set(received)), len(expected))

    def test_watchlist_exact_number_reads_outside_first_page_and_missing_is_empty(self) -> None:
        ids = [self._insert(external_id=str(index), title="  Ａ\t收藏\x00  ") for index in range(25)]
        number = ids[0]
        service = get_agent_service()
        with patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=True):
            first = service.invoke("discovery.watchlist_summaries", {})["result"]
            single = service.invoke(
                "discovery.watchlist_summaries", {"watchlist_number": number}
            )["result"]
            direct = list_watchlist_summaries({"watchlist_number": number})
            missing = service.invoke(
                "discovery.watchlist_summaries", {"watchlist_number": ids[-1] + 1}
            )["result"]
        self.assertNotIn(number, [item["watchlist_number"] for item in first["data"]["items"]])
        self.assertTrue(single["ok"])
        self.assertEqual(single["status"], "success")
        self.assertEqual(single["data"], {
            "count": 1,
            "items": [{"watchlist_number": number, "provider": "tmdb", "media_type": "movie",
                       "title": "A 收藏", "year": "2026"}],
            "has_more": False,
            "next_before_number": None,
        })
        self.assertEqual(direct.data, single["data"])
        self.assertEqual(single["suggestions"], [])
        self.assertTrue(missing["ok"])
        self.assertEqual(missing["status"], "empty")
        self.assertEqual(missing["data"], {
            "count": 0, "items": [], "has_more": False, "next_before_number": None,
        })
        self.assertEqual(missing["suggestions"], [])
        serialized = json.dumps([single, missing], ensure_ascii=False)
        for forbidden in ("PRIVATE_POSTER_KEY", "external_id", "created_at", "poster_key"):
            self.assertNotIn(forbidden, serialized)

    def test_watchlist_all_read_variants_use_read_only_local_queries(self) -> None:
        number = self._insert(external_id="movie")
        latest = self._insert(external_id="tv", media_type="tv")
        real_conn = db.get_conn
        queries = []

        @contextmanager
        def read_only():
            with real_conn() as conn:
                conn.execute("PRAGMA query_only=ON")
                conn.set_trace_callback(queries.append)
                yield conn

        with (
            patch.object(db, "get_conn", read_only),
            patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=True),
            patch("app.agent.discovery_watchlist_actions.get_discovery_service") as provider,
        ):
            for arguments in (
                {}, {"mode": "list"}, {"before_number": latest}, {"mode": "statistics"},
                {"watchlist_number": number}, {"watchlist_number": latest + 1},
            ):
                with self.subTest(arguments=arguments):
                    self.assertTrue(list_watchlist_summaries(arguments).ok)
            provider.assert_not_called()
        self.assertTrue(queries)
        self.assertTrue(all(query.lstrip().upper().startswith(("SELECT", "WITH")) for query in queries))

    def test_watchlist_cursor_does_not_shift_after_inserts_or_deletes(self) -> None:
        ids = [self._insert(external_id=str(n)) for n in range(25)]
        with patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=True):
            first = list_watchlist_summaries({})
            cursor = first.data["next_before_number"]
            self._insert(external_id="new")
            db.delete_media_watchlist("tmdb", "24", "movie")
            second = list_watchlist_summaries({"before_number": cursor})
        self.assertEqual([x["watchlist_number"] for x in second.data["items"]], list(reversed(ids[:5])))
        self.assertFalse(second.data["has_more"])

    def test_watchlist_exact_page_and_empty_continuation(self) -> None:
        ids = [self._insert(external_id=str(n)) for n in range(20)]
        with patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=True):
            result = list_watchlist_summaries({})
            empty = list_watchlist_summaries({"before_number": ids[0]})
        self.assertEqual(result.data["count"], 20)
        self.assertFalse(result.data["has_more"])
        self.assertIsNone(result.data["next_before_number"])
        self.assertEqual(empty.data["items"], [])
        self.assertNotIn("收藏当前为空", empty.summary)

    def test_watchlist_native_pipeline_keeps_pagination_and_private_fields_hidden(self) -> None:
        ids = [self._insert(external_id=str(n)) for n in range(21)]
        service = get_agent_service()
        with patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=True):
            first = service.invoke("discovery.watchlist_summaries", {})["result"]
            second = service.invoke(
                "discovery.watchlist_summaries",
                {"before_number": first["data"]["next_before_number"]},
            )["result"]
        self.assertTrue(first["ok"])
        self.assertTrue(first["data"]["has_more"])
        self.assertFalse(second["data"]["has_more"])
        cursor = first["data"]["next_before_number"]
        self.assertEqual(cursor, first["data"]["items"][-1]["watchlist_number"])
        self.assertIn(f"编号 {cursor} 已在本页返回", first["summary"])
        self.assertNotIn("before_number", first["summary"])
        self.assertTrue(first["suggestions"])
        for hint in (first["summary"], *first["suggestions"]):
            self.assertIn("已在本页返回", hint)
            self.assertIn("编号严格小于", hint)
            self.assertIn("不含游标本身", hint)
        self.assertEqual(second["data"]["items"][0]["watchlist_number"], ids[0])
        self.assertLess(ids[0], cursor)
        serialized = json.dumps([first, second], ensure_ascii=False)
        for forbidden in ("PRIVATE_POSTER_KEY", "external_id", "created_at"):
            self.assertNotIn(forbidden, serialized)

    def test_watchlist_cursor_contract_is_exposed_and_validated(self) -> None:
        service = get_agent_service()
        tool = service.pipeline.catalog.get("discovery.watchlist_summaries")
        self.assertEqual(set(tool.input_schema["properties"]), {"mode", "before_number", "watchlist_number"})
        self.assertFalse(tool.input_schema["additionalProperties"])
        self.assertEqual(tool.input_schema["properties"]["mode"]["enum"], ["list", "statistics"])
        self.assertEqual(tool.input_schema["properties"]["mode"]["default"], "list")
        self.assertIn("不需要逐页统计", tool.input_schema["properties"]["mode"]["description"])
        cursor_description = tool.input_schema["properties"]["before_number"]["description"]
        # 参数约束放在原生Schema字段说明中，不要求顶层摘要再重复一遍。
        self.assertIn("已返回", cursor_description)
        self.assertIn("编号严格小于", cursor_description)
        self.assertIn("不含游标本身", cursor_description)
        self.assertIn("count是本页数量，不是收藏总数", tool.description)
        self.assertIn("首次读取不传", cursor_description)
        for field in ("before_number", "watchlist_number"):
            self.assertEqual(tool.input_schema["properties"][field]["type"], "integer")
            self.assertEqual(tool.input_schema["properties"][field]["minimum"], 1)
            self.assertEqual(tool.input_schema["properties"][field]["maximum"], 2_147_483_647)
        self.assertEqual(watchlist_summaries_arguments({}), {})
        self.assertEqual(watchlist_summaries_arguments({"before_number": 12}), {"before_number": 12})
        for arguments in ({"before_number": True}, {"before_number": 0}, {"before_number": "12"}, {"before_number": None}, {"offset": 20}):
            with self.subTest(arguments=arguments), self.assertRaises(AgentToolError):
                watchlist_summaries_arguments(arguments)

    def test_watchlist_read_modes_and_number_arguments_match_schema(self) -> None:
        from app.agent.kernel.pipeline import ToolPipelineError, _validate_json_schema

        schema = get_agent_service().pipeline.catalog.get("discovery.watchlist_summaries").input_schema
        for arguments in (
            {}, {"mode": "list"}, {"before_number": 12},
            {"mode": "list", "before_number": 12}, {"mode": "statistics"},
            {"before_number": 2_147_483_647}, {"watchlist_number": 1},
            {"watchlist_number": 12}, {"watchlist_number": 2_147_483_647},
        ):
            with self.subTest(arguments=arguments):
                _validate_json_schema(arguments, schema)
                self.assertEqual(watchlist_summaries_arguments(arguments), arguments)
        invalid_arguments = [
            {"mode": "statistics", "before_number": value}
            for value in (12, True, 0, "12", None)
        ] + [
            {"mode": value} for value in ("stats", " statistics ", "STATISTICS", "", True, 1, None, [], {})
        ] + [
            {"mode": "statistics", "limit": 3}, {"before_number": 2_147_483_648},
            {"before_number": True}, {"before_number": None}, {"before_number": "12"},
            {"before_number": 0}, {"offset": 20},
        ] + [
            {field: value}
            for field in ("before_number", "watchlist_number")
            for value in (False, -1, 12.0, 2_147_483_648, 10**100, None, "12", [], {})
        ] + [
            {"watchlist_number": value} for value in (True, 0)
        ] + [
            {"watchlist_number": 12, **other}
            for other in (
                {"mode": "list"}, {"mode": "statistics"}, {"before_number": 10},
                {"mode": "list", "before_number": 10},
                {"mode": "statistics", "before_number": 10},
                {"mode": None}, {"before_number": None}, {"limit": 1},
            )
        ] + [
            None, [], 12, "12", True,
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(AgentToolError):
                    watchlist_summaries_arguments(arguments)
                with self.assertRaises(ToolPipelineError) as error:
                    _validate_json_schema(arguments, schema)
                self.assertEqual(error.exception.code, "invalid_arguments")

    def test_watchlist_number_helper_keeps_exact_read_and_remove_boundaries(self) -> None:
        for number in (1, 12, 2_147_483_647):
            arguments = {"watchlist_number": number}
            with self.subTest(arguments=arguments):
                self.assertEqual(watchlist_summary_arguments(arguments), arguments)
                self.assertEqual(watchlist_summaries_arguments(arguments), arguments)
                self.assertEqual(remove_watchlist_arguments(arguments), arguments)
        invalid_arguments = [
            {"watchlist_number": value}
            for value in (True, False, 0, -1, 12.0, 2_147_483_648, 10**100, None, "12", [], {})
        ] + [
            {}, None, [], {"watchlist_number": 12, "mode": "list"},
            {"watchlist_number": 12, "mode": "statistics"},
            {"watchlist_number": 12, "before_number": 10},
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                for validator in (watchlist_summary_arguments, remove_watchlist_arguments):
                    with self.subTest(validator=validator.__name__), self.assertRaises(AgentToolError):
                        validator(arguments)

    def test_watchlist_statistics_empty_database(self) -> None:
        self.assertEqual(db.get_media_watchlist_statistics(), [])
        with patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=True):
            result = list_watchlist_summaries({"mode": "statistics"})
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "empty")
        self.assertEqual(result.data, {"total": 0, "by_media_type": []})

    def test_watchlist_statistics_single_type_counts_all_and_limits_latest(self) -> None:
        with patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=True):
            ids = [self._insert(external_id=str(index)) for index in range(2)]
            first = list_watchlist_summaries({"mode": "statistics"})
            ids.extend(self._insert(external_id=str(index)) for index in range(2, 7))
            second = list_watchlist_summaries({"mode": "statistics"})
        for result, expected_ids in ((first, ids[:2]), (second, ids)):
            with self.subTest(total=len(expected_ids)):
                self.assertEqual(result.status, "success")
                self.assertEqual(result.data["total"], len(expected_ids))
                self.assertEqual(len(result.data["by_media_type"]), 1)
                group = result.data["by_media_type"][0]
                self.assertEqual(group["media_type"], "movie")
                self.assertEqual(group["count"], len(expected_ids))
                self.assertEqual(
                    [item["watchlist_number"] for item in group["latest"]],
                    list(reversed(expected_ids))[:3],
                )

    def test_watchlist_statistics_sparse_ids_and_historical_types_are_not_omitted(self) -> None:
        rows = [
            (3, "movie"), (19, "tv"), (47, "movie"), (89, "tv"),
            (144, "movie"), (233, "movie"), (610, "episode"), (987, ""),
            (1597, "historical_type_a"), (2584, "historical_type_b"),
        ]
        with db.get_conn() as conn:
            conn.executemany(
                "INSERT INTO media_watchlist(id,provider,external_id,media_type,title,created_at) "
                "VALUES(?,'tmdb',?,?,?,?)",
                [(number, f"PRIVATE_EXTERNAL_{number}", kind, f"收藏 {number}", f"PRIVATE_TIME_{3000-number}")
                 for number, kind in rows],
            )
        with patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=True):
            result = list_watchlist_summaries({"mode": "statistics"})
        groups = result.data["by_media_type"]
        self.assertEqual(result.data["total"], len(rows))
        self.assertEqual(sum(group["count"] for group in groups), len(rows))
        self.assertEqual(len(groups), len({kind for _, kind in rows}))
        # 类型文本按既有规范截断后相同，也不能覆盖掉历史分组的计数。
        self.assertEqual(sum(group["media_type"] == "historical_t" for group in groups), 2)
        for group, kind in zip(groups, sorted({kind for _, kind in rows}), strict=True):
            expected_ids = sorted((number for number, media_type in rows if media_type == kind), reverse=True)
            self.assertEqual(group["count"], len(expected_ids))
            self.assertEqual([item["watchlist_number"] for item in group["latest"]], expected_ids[:3])

    def test_watchlist_statistics_over_repository_page_limit_uses_one_read_query(self) -> None:
        rows = [(index * 7 + 10, "tv" if index % 3 == 0 else "movie") for index in range(1207)]
        with db.get_conn() as conn:
            conn.executemany(
                "INSERT INTO media_watchlist(id,provider,external_id,media_type,title,year,poster_key,created_at) "
                "VALUES(?,'tmdb',?,?,'收藏','2026','PRIVATE_POSTER_KEY','PRIVATE_CREATED_AT')",
                [(number, f"PRIVATE_EXTERNAL_{number}", kind) for number, kind in rows],
            )
        queries = []
        real_conn = db.get_conn

        @contextmanager
        def traced():
            with real_conn() as conn:
                conn.execute("PRAGMA query_only=ON")
                conn.set_trace_callback(queries.append)
                yield conn

        with patch.object(db, "get_conn", traced), patch(
            "app.agent.discovery_watchlist_actions.config.get_bool", return_value=True
        ):
            result = list_watchlist_summaries({"mode": "statistics"})
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0].count("COUNT(*) OVER"), 2)
        self.assertIn("ROW_NUMBER() OVER", queries[0])
        self.assertNotIn("SELECT *", queries[0])
        self.assertEqual(result.data["total"], len(rows))
        for group in result.data["by_media_type"]:
            expected_ids = [number for number, kind in reversed(rows) if kind == group["media_type"]]
            self.assertEqual(group["count"], len(expected_ids))
            self.assertEqual([item["watchlist_number"] for item in group["latest"]], expected_ids[:3])
            for item in group["latest"]:
                self.assertEqual(set(item), {"watchlist_number", "provider", "media_type", "title", "year"})
        serialized = json.dumps(result.to_dict(), ensure_ascii=False)
        for forbidden in ("poster", "external_id", "created_at", "PRIVATE_", "next_before_number", "has_more"):
            self.assertNotIn(forbidden, serialized)
            self.assertNotIn(forbidden, queries[0])

    def test_watchlist_statistics_insert_update_and_delete_are_reflected(self) -> None:
        with patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=True):
            movie = self._insert(external_id="movie")
            self._insert(external_id="tv", media_type="tv")
            first = list_watchlist_summaries({"mode": "statistics"})
            self.assertEqual(first.data["total"], 2)
            self.assertTrue(db.delete_media_watchlist("tmdb", "movie", "movie"))
            remaining = list_watchlist_summaries({"mode": "statistics"})
            self.assertEqual(remaining.data["total"], 1)
            self.assertEqual(remaining.data["by_media_type"][0]["media_type"], "tv")
            replacement = self._insert(external_id="replacement", title="新收藏")
            self._insert(external_id="replacement", title="更新标题")
            latest = list_watchlist_summaries({"mode": "statistics"})
            self.assertEqual(latest.data["total"], 2)
            group = next(group for group in latest.data["by_media_type"] if group["media_type"] == "movie")
            self.assertEqual(group["count"], 1)
            self.assertEqual(group["latest"][0]["watchlist_number"], replacement)
            self.assertGreater(replacement, movie)
            self.assertEqual(group["latest"][0]["title"], "更新标题")
            db.delete_media_watchlist("tmdb", "replacement", "movie")
            db.delete_media_watchlist("tmdb", "tv", "tv")
            self.assertEqual(list_watchlist_summaries({"mode": "statistics"}).data, {"total": 0, "by_media_type": []})

    def test_watchlist_statistics_counts_and_latest_share_snapshot_during_writes(self) -> None:
        rows = [(number, "movie" if number % 2 == 0 else "tv") for number in range(1, 65)]
        with db.get_conn() as conn:
            conn.executemany(
                "INSERT INTO media_watchlist(id,provider,external_id,media_type,title,created_at) "
                "VALUES(?,'tmdb',?,?,'收藏','PRIVATE_TIME')",
                [(number, str(number), kind) for number, kind in rows],
            )
        real_conn = db.get_conn
        queries = []
        mutated = False

        def mutate_during_read():
            nonlocal mutated
            if not mutated:
                mutated = True
                with real_conn() as writer:
                    writer.execute("DELETE FROM media_watchlist WHERE id=?", (rows[-1][0],))
                    writer.executemany(
                        "INSERT INTO media_watchlist(id,provider,external_id,media_type,title,created_at) "
                        "VALUES(?,'tmdb',?,'movie','新增','PRIVATE_TIME')",
                        [(number, str(number)) for number in (200, 201)],
                    )
            return 0

        @contextmanager
        def traced():
            with real_conn() as reader:
                reader.execute("PRAGMA query_only=ON")
                reader.set_trace_callback(queries.append)
                reader.set_progress_handler(mutate_during_read, 1000)
                try:
                    yield reader
                finally:
                    reader.set_progress_handler(None, 0)

        with patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=True):
            with patch.object(db, "get_conn", traced):
                snapshot = list_watchlist_summaries({"mode": "statistics"})
            fresh = list_watchlist_summaries({"mode": "statistics"})
        self.assertTrue(mutated)
        self.assertEqual(len(queries), 1)
        self.assertEqual(snapshot.data["total"], len(rows))
        for group in snapshot.data["by_media_type"]:
            expected_ids = [number for number, kind in reversed(rows) if kind == group["media_type"]]
            self.assertEqual(group["count"], len(expected_ids))
            self.assertEqual([item["watchlist_number"] for item in group["latest"]], expected_ids[:3])
        self.assertEqual(fresh.data["total"], len(rows) + 1)
        movie_group = next(group for group in fresh.data["by_media_type"] if group["media_type"] == "movie")
        self.assertEqual(movie_group["count"], len(rows) // 2 + 1)
        self.assertEqual([item["watchlist_number"] for item in movie_group["latest"]], [201, 200, rows[-3][0]])

    def test_watchlist_statistics_native_pipeline_and_explicit_list(self) -> None:
        movie = self._insert(external_id="movie", title="  Ａ\t收藏\x00  ")
        tv = self._insert(external_id="tv", media_type="tv")
        service = get_agent_service()
        with patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=True):
            statistics = service.invoke("discovery.watchlist_summaries", {"mode": "statistics"})["result"]
            default = service.invoke("discovery.watchlist_summaries", {})["result"]
            explicit = service.invoke("discovery.watchlist_summaries", {"mode": "list"})["result"]
        self.assertTrue(statistics["ok"])
        self.assertEqual(statistics["status"], "success")
        self.assertEqual(statistics["data"]["total"], 2)
        self.assertEqual(
            [(group["media_type"], group["count"], group["latest"][0]["watchlist_number"])
             for group in statistics["data"]["by_media_type"]],
            [("movie", 1, movie), ("tv", 1, tv)],
        )
        self.assertEqual(statistics["data"]["by_media_type"][0]["latest"][0]["title"], "A 收藏")
        self.assertEqual(default["data"], explicit["data"])
        self.assertEqual(default["summary"], explicit["summary"])
        self.assertEqual(set(default["data"]), {"count", "items", "has_more", "next_before_number"})
        serialized = json.dumps(statistics, ensure_ascii=False)
        for forbidden in ("PRIVATE_", "poster_key", "external_id", "created_at"):
            self.assertNotIn(forbidden, serialized)

    def test_watchlist_statistics_keeps_discovery_feature_switch(self) -> None:
        with patch("app.agent.discovery_watchlist_actions.config.get_bool", return_value=False):
            for arguments in ({}, {"mode": "list"}, {"mode": "statistics"}, {"watchlist_number": 12}):
                result = list_watchlist_summaries(arguments)
                self.assertFalse(result.ok)
                self.assertEqual(result.status, "disabled")
                self.assertEqual(result.data, {})

    def test_add_prepare_confirm_replay_and_owner_isolation(self) -> None:
        service = get_agent_service()
        provider = Mock()
        provider.get_detail.return_value = _card(1)
        with (
            patch(
                "app.agent.discovery_watchlist_actions.config.get_bool",
                return_value=True,
            ),
            patch(
                "app.agent.discovery_watchlist_actions.get_discovery_service",
                return_value=provider,
            ),
        ):
            prepared = service.prepare(
                "discovery.add_watchlist",
                {"provider": "tmdb", "external_id": "8801", "media_type": "movie"},
                owner="owner-a",
            )
        self.assertIsNone(db.get_media_watchlist("tmdb", "8801", "movie"))
        confirmation_id = prepared["action_plan"]["plan_id"]
        with self.assertRaises(AgentToolError):
            service.confirm(confirmation_id, owner="owner-b")
        confirmed = service.confirm(confirmation_id, owner="owner-a")
        self.assertEqual(confirmed["result"]["status"], "completed")
        self.assertIsNotNone(db.get_media_watchlist("tmdb", "8801", "movie"))
        with self.assertRaises(AgentToolError):
            service.confirm(confirmation_id, owner="owner-a")
        serialized = json.dumps(
            {"prepared": prepared, "confirmed": confirmed}, ensure_ascii=False
        )
        self.assertNotIn("PRIVATE_POSTER_KEY", serialized)
        self.assertNotIn("api_key=secret", serialized)

    def test_add_confirmation_detects_concurrent_insert(self) -> None:
        service = get_agent_service()
        provider = Mock()
        provider.get_detail.return_value = _card(2)
        with (
            patch(
                "app.agent.discovery_watchlist_actions.config.get_bool",
                return_value=True,
            ),
            patch(
                "app.agent.discovery_watchlist_actions.get_discovery_service",
                return_value=provider,
            ),
        ):
            prepared = service.prepare(
                "discovery.add_watchlist",
                {"provider": "tmdb", "external_id": "8802", "media_type": "movie"},
                owner="owner",
            )
        db.add_media_watchlist("tmdb", "8802", "movie", "并发收藏", "2026", "")
        confirmed = service.confirm(prepared["action_plan"]["plan_id"], owner="owner")
        self.assertEqual(confirmed["result"]["status"], "conflict")

    def test_remove_prepare_confirm_and_stale_snapshot(self) -> None:
        number = self._insert(external_id="201", title="待移除")
        service = get_agent_service()
        with patch(
            "app.agent.discovery_watchlist_actions.config.get_bool", return_value=True
        ):
            prepared = service.prepare(
                "discovery.remove_watchlist",
                {"watchlist_number": number},
                owner="owner",
            )
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE media_watchlist SET title=? WHERE id=?", ("状态已变化", number)
            )
        stale = service.confirm(prepared["action_plan"]["plan_id"], owner="owner")
        self.assertEqual(stale["result"]["status"], "conflict")
        self.assertIsNotNone(db.get_media_watchlist_by_id(number))
        reset_agent_service_for_tests()
        service = get_agent_service()
        with patch(
            "app.agent.discovery_watchlist_actions.config.get_bool", return_value=True
        ):
            prepared = service.prepare(
                "discovery.remove_watchlist",
                {"watchlist_number": number},
                owner="owner",
            )
        confirmed = service.confirm(prepared["action_plan"]["plan_id"], owner="owner")
        self.assertEqual(confirmed["result"]["status"], "completed")
        self.assertIsNone(db.get_media_watchlist_by_id(number))
