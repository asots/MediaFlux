"""Agent 探索收藏、最近探索续句与确认边界回归。"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

from app import database as db
from app.agent.discovery_watchlist_actions import (
    get_watchlist_summary,
    list_watchlist_summaries,
    watchlist_summaries_arguments,
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
    def _insert(*, external_id: str = "100", title: str = "收藏影片") -> int:
        db.add_media_watchlist(
            "tmdb", external_id, "movie", title, "2026", "PRIVATE_POSTER_KEY"
        )
        row = db.get_media_watchlist("tmdb", external_id, "movie")
        return int(row["id"])

    def test_read_summaries_only_expose_public_fields(self) -> None:
        number = self._insert()
        with patch(
            "app.agent.discovery_watchlist_actions.config.get_bool", return_value=True
        ):
            listed = list_watchlist_summaries({})
            single = get_watchlist_summary({"watchlist_number": number})
        self.assertTrue(listed.ok)
        self.assertEqual(single.data["watchlist_number"], number)
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
                received.extend(item["watchlist_number"] for item in result.data["items"])
                cursor = result.data["next_before_number"]
                self.assertEqual(result.data["has_more"], cursor is not None)
                if cursor is not None:
                    self.assertEqual(cursor, received[-1])
                    arguments = {"before_number": cursor}
        self.assertFalse(result.data["has_more"])
        self.assertEqual(received, list(reversed(expected)))
        # Web默认调用仍读取原有有界列表，没有被Agent页长截断。
        self.assertEqual(len(db.list_media_watchlist()), 45)

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
        self.assertEqual(second["data"]["items"][0]["watchlist_number"], ids[0])
        serialized = json.dumps([first, second], ensure_ascii=False)
        for forbidden in ("PRIVATE_POSTER_KEY", "external_id", "created_at"):
            self.assertNotIn(forbidden, serialized)

    def test_watchlist_cursor_contract_is_exposed_and_validated(self) -> None:
        service = get_agent_service()
        tool = service.pipeline.catalog.get("discovery.watchlist_summaries")
        self.assertIn("before_number", tool.input_schema["properties"])
        self.assertEqual(watchlist_summaries_arguments({}), {})
        self.assertEqual(watchlist_summaries_arguments({"before_number": 12}), {"before_number": 12})
        for arguments in ({"before_number": True}, {"before_number": 0}, {"before_number": "12"}, {"before_number": None}, {"offset": 20}):
            with self.subTest(arguments=arguments), self.assertRaises(AgentToolError):
                watchlist_summaries_arguments(arguments)

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
