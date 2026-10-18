"""缺季搜索共享候选、预算和可提交快照的纵向回归。"""

from __future__ import annotations

from contextlib import ExitStack
from unittest import TestCase
from unittest.mock import Mock, patch

from app.agent.domains.library import episode_resource_actions as actions
from app.agent.errors import AgentToolError
from app.agent.models import ToolContext, ToolResult
from app.indexers.models import IndexerItem
from app.indexers.result_store import IndexerResultStore

TITLE = "时光代理人"
SEASON = 4


def targets(episodes):
    return [{"season": SEASON, "episode": episode} for episode in episodes]


def audit_result(episodes=range(1, 12), *, title=TITLE, tmdb_id="126431"):
    episodes = list(episodes)
    result = ToolResult(True, "updates_available", "已核验缺集", data={
        "title": title, "tmdb_id": tmdb_id,
        "missing_count": len(episodes), "missing_sample": targets(episodes),
        "missing_sample_truncated": False,
    })
    result.effect_metadata["episode_mapping_context"] = {
        "detail": {"name": title, "seasons": [
            {"season_number": SEASON, "episode_count": max(episodes, default=0)},
        ]},
        "season_detail": {"season_number": SEASON, "episodes": [
            {"episode_number": episode, "air_date": "2020-01-01"}
            for episode in range(1, max(episodes, default=0) + 1)
        ]},
        "as_of": "2026-10-17",
    }
    return result


class SeasonResourceSearchTests(TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.store = IndexerResultStore()
        service = Mock(enabled_site_ids=("nyaa",), result_store=self.store)
        service.site_ids_for_scope.return_value = ("nyaa",)
        self.stack.enter_context(patch("app.agent.domains.resources.indexer_actions.config.get_bool", return_value=True))
        self.stack.enter_context(patch("app.agent.domains.resources.indexer_actions.get_indexer_service", return_value=service))
        self.stack.enter_context(patch.object(actions, "get_indexer_service", return_value=service))
        self.audit = self.stack.enter_context(patch.object(actions, "audit_series_episodes", return_value=audit_result()))

    def arguments(self, **overrides):
        return actions.missing_season_resource_arguments({
            "query": TITLE, "season": SEASON, "as_of": "2026-10-17", **overrides,
        })

    def item(self, title):
        item = IndexerItem(site_id="nyaa", site_name="Nyaa", title=title,
                           detail_url=f"https://fixture.invalid/{len(title)}",
                           magnet="magnet:?xt=urn:btih:" + "a" * 40,
                           download_state="ready", download_kinds=("magnet",))
        result_id = self.store.put(item)
        return {**item.to_public_dict(), "result_id": result_id}

    def searched(self, episodes=(), *, title=TITLE, ok=True, status="success", items=None, partial=False):
        return ToolResult(ok, status, "资源站查询已返回", data={
            "items": [self.item(f"{title} S04E{episode:02d} 1080p") for episode in episodes]
            if items is None else items,
            "query": f"{title} S04", "sites_attempted": ["nyaa", "failed"] if partial else ["nyaa"],
            "sites_succeeded": ["nyaa"] if ok else [], "partial": partial,
            "errors": [{"site_id": "failed", "code": "timeout"}] if partial else [],
            "has_more": False, "truncated": False,
        })

    def test_default_twenty_and_explicit_episode_parameters_are_not_limited_to_three(self):
        self.assertEqual(self.arguments()["max_episodes"], 20)
        self.assertEqual(self.arguments(episodes=list(range(1, 22)))["max_episodes"], 21)
        self.assertEqual(self.arguments(episodes=list(range(1, 12)))["episodes"], list(range(1, 12)))
        self.assertEqual(self.arguments(episodes=[7, 4], max_episodes=1)["max_episodes"], 1)
        self.assertEqual(self.arguments(max_episodes=100)["max_episodes"], 100)
        for invalid in ({"max_episodes": 101}, {"max_episodes": True}, {"episodes": list(range(1, 102))}):
            with self.subTest(invalid=invalid), self.assertRaises(AgentToolError):
                self.arguments(**invalid)

    def test_eleven_missing_episodes_use_one_shared_search_and_one_inventory_audit(self):
        with patch.object(actions, "search_resources", return_value=self.searched(range(1, 12))) as search:
            result = actions.search_missing_season_resources(self.arguments())
        self.audit.assert_called_once()
        self.assertEqual(self.audit.call_args.kwargs, {"refresh": True})
        search.assert_called_once()
        self.assertIsNone(search.call_args.args[0].get("episode"))
        self.assertEqual(search.call_args.args[0]["limit"], 50)
        self.assertEqual((result.status, result.data["processed"], result.data["remaining"]), ("success", 11, 0))
        self.assertEqual([row["episode"] for row in result.model_data["episodes"]], list(range(1, 12)))
        self.assertTrue(all(row["search"]["recommendation"]["selected"] for row in result.data["episodes"]))
        self.assertFalse(any("继续" in suggestion for suggestion in result.suggestions))
        self.assertEqual(len(result.references), 1)
        self.assertEqual(len(result.references[0].value["candidates"]), 11)

    def test_default_covers_twenty_and_only_returns_the_actual_unselected_target(self):
        self.audit.return_value = audit_result(range(1, 22))
        with patch.object(actions, "search_resources", return_value=self.searched(range(1, 22))) as search:
            result = actions.search_missing_season_resources(self.arguments())
        search.assert_called_once()
        self.assertEqual(result.data["processed"], 20)
        self.assertEqual(result.data["remaining_episodes"], targets([21]))

    def test_explicit_eleven_targets_are_all_covered_without_another_user_turn(self):
        with patch.object(actions, "search_resources", return_value=self.searched(range(1, 12))) as search:
            result = actions.search_missing_season_resources(self.arguments(episodes=list(range(1, 12))))
        search.assert_called_once()
        self.assertEqual(result.data["remaining_episodes"], [])
        self.assertEqual(result.data["processed"], 11)

    def test_explicit_targets_beyond_default_twenty_are_not_implicitly_capped(self):
        self.audit.return_value = audit_result(range(1, 26))
        with patch.object(actions, "search_resources", return_value=self.searched(range(1, 26))) as search:
            result = actions.search_missing_season_resources(self.arguments(episodes=list(range(1, 26))))
        search.assert_called_once()
        self.assertEqual((result.data["processed"], result.data["remaining"]), (25, 0))

    def test_single_episode_search_keeps_exact_arguments_and_one_audit(self):
        self.audit.return_value.data["target_missing"] = True
        arguments = actions.missing_episode_resource_arguments({
            "query": TITLE, "season": SEASON, "episode": 11, "as_of": "2026-10-17"})
        with patch.object(actions, "search_resources", return_value=self.searched([11])) as search:
            result = actions.search_missing_episode_resources(arguments)
        self.audit.assert_called_once()
        search.assert_called_once()
        self.assertEqual(search.call_args.args[0]["episode"], 11)
        self.assertEqual(search.call_args.kwargs, {})
        self.assertEqual(result.data["search"]["recommendation"]["selected"]["match"], "exact_episode")

    def test_only_uncovered_targets_get_precise_searches(self):
        seen = []

        def search(arguments, *, timeout_seconds):
            seen.append(arguments.get("episode"))
            self.assertLessEqual(timeout_seconds, 30)
            return self.searched(range(1, 9) if arguments.get("episode") is None else [arguments["episode"]])

        with patch.object(actions, "search_resources", side_effect=search):
            result = actions.search_missing_season_resources(self.arguments())
        self.assertEqual(seen, [None, 9, 10, 11])
        self.assertEqual((result.status, result.data["processed"], result.data["remaining"]), ("success", 11, 0))

    def test_precise_pack_is_reused_for_other_uncovered_targets(self):
        with patch.object(actions, "search_resources", side_effect=[
            self.searched(range(1, 9)),
            self.searched(items=[self.item(f"{TITLE} S04E09-E11 1080p")]),
        ]) as search:
            result = actions.search_missing_season_resources(self.arguments())
        self.assertEqual([call.args[0].get("episode") for call in search.call_args_list], [None, 9])
        self.assertEqual(result.data["processed"], 11)
        self.assertEqual(result.data["remaining"], 0)

    def test_broad_no_match_is_not_reported_as_precise_exhaustion(self):
        with patch.object(actions, "search_resources", return_value=self.searched()) as search:
            result = actions.search_missing_season_resources(self.arguments())
        self.assertEqual([call.args[0].get("episode") for call in search.call_args_list], [None, 1, 2, 3, 4, 5])
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.data["remaining_episodes"], targets(range(6, 12)))
        self.assertEqual(result.model_data["remaining_episodes"], targets(range(6, 12)))
        self.assertTrue(all(row["search"]["exact_search_attempted"] for row in result.data["episodes"]))
        self.assertEqual(result.data["stop_reason"], "request_budget")

    def test_site_partial_failure_keeps_all_shared_matches_and_failure_details(self):
        with patch.object(actions, "search_resources", return_value=self.searched(range(1, 12), partial=True, status="partial")) as search:
            result = actions.search_missing_season_resources(self.arguments())
        search.assert_called_once()
        self.assertTrue(result.ok)
        self.assertEqual((result.status, result.data["processed"], result.data["failed"]), ("partial", 11, 0))
        self.assertTrue(all(row["search"]["errors"] for row in result.model_data["episodes"]))
        self.assertEqual(result.data["remaining_episodes"], [])

    def test_absolute_budget_does_not_scale_with_missing_count_and_keeps_timeout_range(self):
        clock = [0.0]
        budgets = []

        def audit(*_args, **_kwargs):
            clock[0] += 3
            return audit_result()

        def search(arguments, *, timeout_seconds):
            budgets.append(timeout_seconds)
            clock[0] += timeout_seconds
            return self.searched(range(1, 9)) if arguments.get("episode") is None else self.searched(ok=False, status="timeout")

        self.audit.side_effect = audit
        with patch.object(actions.time, "monotonic", side_effect=lambda: clock[0]), patch.object(actions, "search_resources", side_effect=search):
            result = actions.search_missing_season_resources(self.arguments())
        self.assertEqual(budgets, [30, 12])
        self.assertEqual(clock[0], 45)
        self.assertEqual(result.data["remaining_episodes"], targets([10, 11]))
        self.assertEqual((result.data["processed"], result.data["failed"]), (9, 1))
        self.assertEqual(result.data["stop_reason"], "time_budget")

    def test_cancel_during_shared_search_keeps_matches_and_exact_remaining_range(self):
        cancelled = [False]

        def search(*_args, **_kwargs):
            cancelled[0] = True
            return self.searched(range(1, 9))

        with patch.object(actions, "search_resources", side_effect=search) as searched:
            result = actions.search_missing_season_resources(self.arguments(), context=ToolContext(cancelled=lambda: cancelled[0]))
        searched.assert_called_once()
        self.assertEqual(result.data["processed"], 8)
        self.assertEqual(result.data["remaining_episodes"], targets([9, 10, 11]))
        self.assertEqual(result.data["stop_reason"], "cancelled")

    def test_audit_consumes_deadline_without_starting_any_indexer(self):
        clock = [0]

        def audit(*_args, **_kwargs):
            clock[0] = 46
            return audit_result()

        self.audit.side_effect = audit
        with patch.object(actions.time, "monotonic", side_effect=lambda: clock[0]), patch.object(actions, "search_resources") as search:
            result = actions.search_missing_season_resources(self.arguments())
        search.assert_not_called()
        self.assertEqual(result.data["remaining_episodes"], targets(range(1, 12)))

    def test_explicit_maximum_keeps_single_target_precise_and_reports_the_other_target(self):
        self.audit.return_value = audit_result([1, 4, 7])
        with patch.object(actions, "search_resources", return_value=self.searched([4])) as search:
            result = actions.search_missing_season_resources(self.arguments(episodes=[7, 4], max_episodes=1))
        search.assert_called_once()
        self.assertEqual(search.call_args.args[0]["episode"], 4)
        self.assertEqual(result.data["remaining_episodes"], targets([7]))

    def test_unknown_future_fin_and_different_work_are_never_promoted(self):
        audit = audit_result()
        audit.effect_metadata["episode_mapping_context"]["season_detail"]["episodes"].append(
            {"episode_number": 12, "air_date": "2099-01-01"})
        self.audit.return_value = audit
        searched = self.searched(items=[
            self.item(f"{TITLE} WEB-DL 1080p"),
            self.item(f"{TITLE} S04E01-E12 Fin 1080p"),
            self.item("完全不同作品 S04E01-E11 1080p"),
            self.item(f"{TITLE} S03E01-E11 1080p"),
        ])
        with patch.object(actions, "search_resources", return_value=searched):
            result = actions.search_missing_season_resources(self.arguments())
        self.assertTrue(all(row["search"]["recommendation"]["selected"] is None for row in result.data["episodes"]))
        self.assertTrue(all(not row["search"]["items"] for row in result.model_data["episodes"]))
        self.assertEqual(result.data["remaining_episodes"], targets(range(6, 12)))
        self.assertTrue(all(candidate["match"] == "unknown" for candidate in result.references[0].value["candidates"]))

    def test_batch_keeps_one_reference_and_global_numbers_for_shared_packs(self):
        names = [TITLE, "另一部剧"]

        def audit(arguments, **_kwargs):
            return audit_result(title=arguments["query"], tmdb_id=str(1000 + names.index(arguments["query"])))

        self.audit.side_effect = audit
        seen = []

        def search(arguments, **_kwargs):
            seen.append((arguments["title"], arguments.get("episode")))
            return self.searched(items=[self.item(f"{arguments['title']} S04E01-E11 1080p")])

        arguments = actions.missing_season_resource_arguments({"items": [
            {"query": title, "season": SEASON} for title in names], "as_of": "2026-10-17"})
        with patch.object(actions, "search_resources", side_effect=search):
            result = actions.search_missing_season_resources(arguments)
        self.assertCountEqual(seen, [(TITLE, None), (names[1], None)])
        self.assertEqual(self.audit.call_count, 2)
        self.assertEqual(result.data["recommended_positions"], [1, 2])
        self.assertEqual([group["positions"] for group in result.data["groups"]], [[1], [2]])
        self.assertTrue(all(group["remaining"] == 0 for group in result.model_data["groups"]))
        self.assertEqual(len(result.references), 1)
        reference = result.references[0].value
        self.assertEqual([candidate["position"] for candidate in reference["candidates"]], [1, 2])
        self.assertEqual(len(reference["_private_items"]), 2)

    def test_cancel_during_audit_starts_no_search_and_preserves_all_missing_targets(self):
        cancelled = [False]

        def audit(*_args, **_kwargs):
            cancelled[0] = True
            return audit_result()

        self.audit.side_effect = audit
        with patch.object(actions, "search_resources") as search:
            result = actions.search_missing_season_resources(self.arguments(), context=ToolContext(cancelled=lambda: cancelled[0]))
        search.assert_not_called()
        self.assertEqual(result.data["remaining_episodes"], targets(range(1, 12)))
        self.assertEqual(result.data["stop_reason"], "cancelled")

    def test_cancelled_batch_does_not_start_audits_or_indexers(self):
        arguments = actions.missing_season_resource_arguments({"items": [{"query": TITLE, "season": SEASON}]})
        with patch.object(actions, "search_resources") as search:
            result = actions.search_missing_season_resources(arguments, context=ToolContext(cancelled=lambda: True))
        self.audit.assert_not_called()
        search.assert_not_called()
        self.assertEqual(result.data["groups"][0]["status"], "cancelled")

    def test_tool_schema_matches_expanded_parameter_contract(self):
        from app.agent.domains.library.tools import register_specs

        registry = Mock()
        register_specs(registry, missing_media_runtime=None)
        spec = next(call.args[0] for call in registry.register.call_args_list
                    if call.args[0].name == "library.search_missing_season_resources")
        properties = spec.parameters["properties"]
        self.assertEqual(properties["max_episodes"]["maximum"], 100)
        self.assertEqual(properties["episodes"]["maxItems"], 100)
        self.assertEqual(properties["items"]["items"]["properties"]["episodes"]["maxItems"], 100)
        self.assertIn("20", spec.description)

    def test_shared_matches_still_apply_quality_preferences_and_per_episode_limit(self):
        rows = [self.item(f"{TITLE} S04E{episode:02d} {resolution}p")
                for episode in range(1, 12) for resolution in (1080, 2160)]
        with patch.object(actions, "search_resources", return_value=self.searched(items=rows)) as search:
            result = actions.search_missing_season_resources(self.arguments(
                limit_per_episode=1, preference_overrides={"minimum_resolution": "2160p"}))
        search.assert_called_once()
        self.assertEqual(result.data["remaining"], 0)
        self.assertTrue(all(len(row["search"]["items"]) == 1 for row in result.data["episodes"]))
        self.assertTrue(all("2160p" in candidate["title"] for candidate in result.references[0].value["candidates"]))

    def test_real_inventory_audit_and_indexer_search_cover_eleven_in_one_pass(self):
        from app.agent.domains.library import episode_audit
        from app.clients.base import (
            SeriesCandidate,
            SeriesEpisodeInventory,
            SeriesSearchResult,
        )
        from app.indexers.registry import IndexerRegistry
        from app.indexers.service import IndexerService
        from tests.test_indexer_service import FakeAdapter

        episode_audit.reset_episode_audit_cache_for_tests()
        self.addCleanup(episode_audit.reset_episode_audit_cache_for_tests)
        media = Mock()
        media.search_series_candidates.return_value = SeriesSearchResult(candidates=[
            SeriesCandidate(id="fixture", name=TITLE, year="2020", tmdb_id="126431")], total=1)
        media.find_series_candidates_by_tmdb.return_value = media.search_series_candidates.return_value
        media.list_series_episode_inventory.return_value = SeriesEpisodeInventory(episodes=[], total=0)
        tmdb = Mock()
        tmdb.detail.return_value = {"name": TITLE, "original_name": "Link Click", "first_air_date": "2020-01-01",
                                    "seasons": [{"season_number": SEASON, "episode_count": 11}]}
        tmdb.tv_season_detail.return_value = {"season_number": SEASON, "episodes": [
            {"episode_number": episode, "air_date": "2020-01-01"} for episode in range(1, 12)]}
        provider = FakeAdapter("nyaa", [IndexerItem(
            site_id="nyaa", site_name="Nyaa", title=f"Link.Click S04E{episode:02d} 1080p",
            detail_url=f"https://fixture.invalid/{episode}",
            magnet=f"magnet:?xt=urn:btih:{episode:040x}", download_state="ready", download_kinds=("magnet",),
        ) for episode in range(1, 12)])
        service = IndexerService(registry=IndexerRegistry({"nyaa": provider}), result_store=IndexerResultStore())
        with (
            patch("socket.socket.connect", side_effect=AssertionError("network forbidden")),
            patch("app.services._configured_media_sources", return_value=[("jellyfin", "Jellyfin", "https://fixture.invalid", media)]),
            patch("app.agent.domains.library.episode_audit.TMDBClient", return_value=tmdb),
            patch.object(actions, "audit_series_episodes", wraps=episode_audit.audit_series_episodes) as audit,
            patch("app.agent.domains.resources.indexer_actions.get_indexer_service", return_value=service),
            patch.object(actions, "get_indexer_service", return_value=service),
            patch.object(service, "search_media", wraps=service.search_media) as search,
        ):
            result = actions.search_missing_season_resources(self.arguments(tmdb_id="126431"))
        audit.assert_called_once()
        media.list_series_episode_inventory.assert_called_once()
        tmdb.tv_season_detail.assert_called_once()
        search.assert_called_once()
        self.assertEqual(search.call_args.args[0].original_title, "Link Click")
        self.assertEqual((result.status, result.data["processed"], result.data["remaining"]), ("success", 11, 0))
        self.assertEqual(len(result.references[0].value["candidates"]), 11)
        self.assertEqual(len(result.references[0].value["_private_items"]), 11)

    def test_batch_queued_seasons_share_absolute_deadline_and_skip_expired_audits(self):
        from concurrent.futures import ThreadPoolExecutor

        clock = [0.0]

        def audit(*_args, **_kwargs):
            clock[0] += 1
            return audit_result()

        def search(*_args, timeout_seconds):
            clock[0] += timeout_seconds
            return self.searched()

        self.audit.side_effect = audit
        arguments = actions.missing_season_resource_arguments({"items": [
            {"query": f"作品{index}", "season": SEASON, "episodes": list(range(1, 12))}
            for index in range(4)]})
        with (
            patch.object(actions.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(actions, "ThreadPoolExecutor", side_effect=lambda **_kw: ThreadPoolExecutor(max_workers=1)),
            patch.object(actions, "search_resources", side_effect=search) as searched,
        ):
            result = actions.search_missing_season_resources(arguments)
        self.assertEqual(clock[0], 45)
        self.assertEqual(searched.call_count, 2)
        self.audit.assert_called_once()
        self.assertTrue(all(group["remaining_episodes"] == targets(range(1, 12))
                            for group in result.data["groups"][1:]))

    def test_batch_has_one_total_request_budget_across_all_seasons(self):
        arguments = actions.missing_season_resource_arguments({"items": [
            {"query": f"作品{index}", "season": SEASON} for index in range(12)]})
        with patch.object(actions, "search_resources", return_value=self.searched()) as search:
            result = actions.search_missing_season_resources(arguments)
        self.assertEqual(search.call_count, 24)
        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.data["groups"]), 12)

    def test_batch_preserves_partial_site_status_even_when_all_targets_match(self):
        arguments = actions.missing_season_resource_arguments({"items": [{"query": TITLE, "season": SEASON}]})
        with patch.object(actions, "search_resources", return_value=self.searched(range(1, 12), partial=True, status="partial")):
            result = actions.search_missing_season_resources(arguments)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.data["groups"][0]["remaining"], 0)
        self.assertTrue(result.model_data["groups"][0]["errors"])

    def test_wrong_episode_and_different_work_never_become_batch_review_positions(self):
        arguments = actions.missing_season_resource_arguments({"items": [{"query": TITLE, "season": SEASON}]})
        with patch.object(actions, "search_resources", return_value=self.searched(items=[
            self.item(f"{TITLE} S03E01-E11 1080p"),
            self.item(f"{TITLE} S04E99 1080p"),
            self.item("完全不同作品 S04E01-E11 1080p"),
        ])):
            result = actions.search_missing_season_resources(arguments)
        self.assertEqual(result.data["recommended_positions"], [])
        self.assertEqual(result.data["review_positions"], [])
        self.assertEqual(result.references, [])
        self.assertEqual(result.data["groups"][0]["remaining_episodes"], targets(range(6, 12)))
