"""剧集完整性审计的映射、播出日期、截断和脱敏测试。"""
from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import cast
from unittest.mock import patch

from app.agent.domains.library.episode_audit import (
    audit_series_episodes,
    invalidate_episode_audit_cache,
    reset_episode_audit_cache_for_tests,
)
from app.agent.domains.library.update_actions import check_library_updates
from app.agent.models import ToolResult
from app.clients.base import (
    MediaServerClient,
    SeriesCandidate,
    SeriesEpisodeInventory,
    SeriesSearchResult,
)
from app.discovery.models import ProviderNotConfigured, ProviderUnavailable
from app.services import _series_source_payload


class _FakeTMDB:
    close: Callable[[], None]
    def __init__(self):
        self.details = {"name": "The Show", "first_air_date": "2026-01-01", "seasons": [{"season_number": 1}, {"season_number": 2}]}
        self.seasons = {
            1: {"episodes": [
                {"episode_number": 1, "air_date": "2026-07-01"},
                {"episode_number": 2, "air_date": "2026-07-08"},
                {"episode_number": 3, "air_date": "2026-08-08"},
                {"episode_number": 4, "air_date": None},
            ]},
            2: {"episodes": [{"episode_number": 1, "air_date": "2026-07-15"}]},
        }

    def search(self, title, year, media_type):
        return [{
            "id": 12345,
            "name": "The Show",
            "original_name": "The Show",
            "first_air_date": "2026-01-01",
        }]

    def detail(self, tmdb_id, media_type):
        return self.details

    def tv_season_detail(self, tmdb_id, season_number):
        return self.seasons[season_number]


def _ready(episodes, *, server="Jellyfin", tmdb_id="12345", truncated=False):
    return {
        "server_type": server.casefold(),
        "server_name": server,
        "status": "ready",
        "candidates": [{"name": "The Show", "year": "2026", "tmdb_id": tmdb_id}],
        "selected": {"name": "The Show", "year": "2026", "tmdb_id": tmdb_id},
        "episodes": episodes,
        "local_total": len(episodes),
        "truncated": truncated,
        "ignored_specials": 1,
        "ignored_unknown": 0,
        "error": "",
    }


class _FakeMediaClient:
    def __init__(self, title_search, *, provider_search=None, inventories=None, folders=None):
        self.title_search = title_search
        self.provider_search = provider_search or SeriesSearchResult()
        self.inventories = inventories or {}
        self.folders = list(folders or [])
        self.parent_ids: list[str] = []

    def list_virtual_folders(self):
        return list(self.folders)

    def search_series_candidates(self, query, limit=6, *, parent_id=""):
        self.parent_ids.append(parent_id)
        return self.title_search

    def find_series_candidates_by_tmdb(self, tmdb_id, limit=20, *, parent_id=""):
        self.parent_ids.append(parent_id)
        return self.provider_search

    def list_series_episode_inventory(
        self, series_id, *, max_episodes=2000, page_size=200, include_specials=False
    ):
        return self.inventories[series_id]


class SeriesSourcePayloadTests(unittest.TestCase):
    def test_unmapped_unique_local_series_still_reads_jellyfin_inventory(self):
        candidate = SeriesCandidate(id="series-1", name="The Show", year="2026")
        client = _FakeMediaClient(
            SeriesSearchResult(candidates=[candidate], total=1),
            inventories={
                "series-1": SeriesEpisodeInventory(
                    episodes=[(1, 1), (1, 2)], total=2
                )
            },
        )
        result = _series_source_payload(
            "jellyfin", "Jellyfin", cast(MediaServerClient, client), "The Show", "", 2000
        )
        self.assertEqual(result["status"], "unmapped")
        self.assertEqual(result["episodes"], [(1, 1), (1, 2)])
        self.assertEqual(result["mapping"]["status"], "unmapped")

    def test_explicit_tmdb_id_can_use_title_inventory_when_provider_id_is_missing(self):
        candidate = SeriesCandidate(id="series-1", name="The Show", year="2026")
        client = _FakeMediaClient(
            SeriesSearchResult(candidates=[candidate], total=1),
            inventories={
                "series-1": SeriesEpisodeInventory(episodes=[(1, 1)], total=1)
            },
        )
        result = _series_source_payload(
            "jellyfin", "Jellyfin", cast(MediaServerClient, client), "The Show", "12345", 2000
        )
        self.assertEqual(result["status"], "unmapped")
        self.assertEqual(result["episodes"], [(1, 1)])

    def test_named_library_scope_uses_parent_id_but_only_returns_public_name(self):
        candidate = SeriesCandidate(id="series-1", name="The Show", year="2026")
        client = _FakeMediaClient(
            SeriesSearchResult(candidates=[candidate], total=1),
            inventories={
                "series-1": SeriesEpisodeInventory(episodes=[(1, 1)], total=1)
            },
            folders=[
                {"id": "private-library-id", "name": "美女库"},
                {"id": "other-library-id", "name": "电视剧"},
            ],
        )

        result = _series_source_payload(
            "jellyfin", "Jellyfin", cast(MediaServerClient, client), "The Show", "", 2000,
            library_name="美女库",
        )

        self.assertEqual(result["status"], "unmapped")
        self.assertEqual(result["library_name"], "美女库")
        self.assertEqual(client.parent_ids, ["private-library-id"])
        self.assertNotIn("private-library-id", repr(result))

    def test_named_library_scope_fails_closed_without_searching_series(self):
        for folders, requested, expected in (
            ([], "动漫", "library_not_found"),
            ([
                {"id": "library-1", "name": "动画电影"},
                {"id": "library-2", "name": "动画剧集"},
            ], "动画", "library_ambiguous"),
        ):
            client = _FakeMediaClient(SeriesSearchResult(), folders=folders)
            with self.subTest(expected=expected):
                result = _series_source_payload(
                    "jellyfin", "Jellyfin", cast(MediaServerClient, client), "The Show", "", 2000,
                    library_name=requested,
                )
            self.assertEqual(result["status"], expected)
            self.assertEqual(result["library_name"], requested)
            self.assertEqual(client.parent_ids, [])
            self.assertNotIn("library-1", repr(result))
            self.assertNotIn("library-2", repr(result))

    def test_provider_mapped_duplicates_are_merged(self):
        first = SeriesCandidate(
            id="series-1", name="The Show", year="2026", tmdb_id="12345"
        )
        second = SeriesCandidate(
            id="series-2", name="The Show", year="2026", tmdb_id="12345"
        )
        client = _FakeMediaClient(
            SeriesSearchResult(candidates=[first], total=1),
            provider_search=SeriesSearchResult(
                candidates=[first, second], total=2, truncated=False
            ),
            inventories={
                "series-1": SeriesEpisodeInventory(episodes=[(1, 1)], total=1),
                "series-2": SeriesEpisodeInventory(episodes=[(1, 2)], total=1),
            },
        )
        result = _series_source_payload(
            "jellyfin", "Jellyfin", cast(MediaServerClient, client), "The Show", "12345", 2000
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["episodes"], [(1, 1), (1, 2)])



class EpisodeAuditTests(unittest.TestCase):
    def setUp(self):
        reset_episode_audit_cache_for_tests()
        self.arguments: dict = {
            "query": "The Show", "tmdb_id": "", "season": None, "as_of": "2026-08-01"
        }

    def _run(self, sources):
        with patch("app.agent.domains.library.episode_audit.inspect_series_episode_sources", return_value=sources), patch(
            "app.agent.domains.library.episode_audit.TMDBClient", return_value=_FakeTMDB()
        ):
            return audit_series_episodes(dict(self.arguments))

    def _large_missing_fixture(self):
        fake = _FakeTMDB()
        # TMDB 本身断号，本地还收录了后面的集；不能用首尾范围或集数猜缺集。
        fake.seasons = {
            1: {"episodes": [
                {"episode_number": number, "air_date": "2026-07-01"}
                for number in range(1, 131) if number not in {2, 17, 88}
            ]},
            2: {"episodes": [
                {"episode_number": number, "air_date": "2026-07-01"}
                for number in (1, 3, 8, 13, 21, 150)
            ]},
        }
        source = _ready([(1, 1), (1, 4), (1, 70), (1, 124), (2, 1), (2, 8), (2, 150)])
        missing = [
            {"season": 1, "episode": number}
            for number in range(1, 131) if number not in {1, 2, 4, 17, 70, 88, 124}
        ] + [{"season": 2, "episode": number} for number in (3, 13, 21)]
        return fake, source, missing

    def test_full_missing_list_survives_real_audit_to_batch_across_sparse_seasons(self):
        fake, source, missing = self._large_missing_fixture()
        with patch(
            "app.agent.domains.library.episode_audit.inspect_series_episode_sources", return_value=[source],
        ) as inspect, patch("app.agent.domains.library.episode_audit.TMDBClient", return_value=fake):
            audit = audit_series_episodes(dict(self.arguments))
            # 不 mock audit_series_episodes：批量入口必须再次经过真实审计算法。
            batch = check_library_updates({
                "queries": ["The Show"], "media_type": "tv",
                "as_of": self.arguments["as_of"], "refresh": True,
            })
        self.assertEqual(inspect.call_count, 2)
        self.assertEqual(len(missing), 126)
        self.assertTrue(audit.ok)
        self.assertEqual(audit.status, "updates_available")
        self.assertEqual(batch.status, "success")
        self.assertEqual(batch.data["counts"], {
            "requested": 1, "updates_available": 1, "up_to_date": 0, "uncertain": 0,
        })
        assert batch.model_data is not None
        for data in (audit.data, batch.data["items"][0], batch.model_data["items"][0]):
            self.assertEqual(data["expected_aired"], 133)
            self.assertEqual(data["local_episode_count"], 7)
            self.assertEqual(data["missing_count"], len(missing))
            self.assertEqual(data["missing_sample"], missing)
            self.assertFalse(data["missing_sample_truncated"])
            self.assertEqual(data["latest_local"], {"season": 2, "episode": 150})
            self.assertEqual(data["latest_aired"], {"season": 2, "episode": 150})
        followups = audit.data["resource_followups"]
        self.assertEqual(len(followups), 12)
        self.assertTrue(audit.data["resource_followups_truncated"])
        self.assertEqual(
            [item["episode_label"] for item in followups],
            [f"S{item['season']:02d}E{item['episode']:02d}" for item in missing[:12]],
        )
        for followup, item in zip(followups, missing[:12], strict=True):
            self.assertEqual(followup["tool"], "library.search_missing_episode_resources")
            self.assertEqual(followup["arguments"]["season"], item["season"])
            self.assertEqual(followup["arguments"]["episode"], item["episode"])

    def test_full_missing_list_does_not_make_incomplete_audit_conclusive(self):
        fake, source, missing = self._large_missing_fixture()
        incomplete_sources = {
            "truncated_inventory": [{**source, "truncated": True}],
            "unpositioned_inventory": [{**source, "ignored_unknown": 1}],
            "unavailable_server": [source, {
                **_ready([], server="Emby"), "status": "unavailable", "selected": None,
            }],
        }
        for reason, sources in incomplete_sources.items():
            with self.subTest(reason=reason):
                reset_episode_audit_cache_for_tests()
                with patch(
                    "app.agent.domains.library.episode_audit.inspect_series_episode_sources", return_value=sources,
                ), patch("app.agent.domains.library.episode_audit.TMDBClient", return_value=fake):
                    audit = audit_series_episodes(dict(self.arguments))
                    batch = check_library_updates({
                        "queries": ["The Show"], "media_type": "tv",
                        "as_of": self.arguments["as_of"], "refresh": True,
                    })
                self.assertFalse(audit.ok)
                self.assertEqual(audit.status, "inconclusive")
                self.assertNotIn("resource_followups", audit.data)
                self.assertFalse(batch.ok)
                self.assertEqual(batch.status, "partial")
                self.assertEqual(batch.data["counts"], {
                    "requested": 1, "updates_available": 0, "up_to_date": 0, "uncertain": 1,
                })
                assert batch.model_data is not None
                for data in (audit.data, batch.data["items"][0], batch.model_data["items"][0]):
                    self.assertEqual(data["missing_count"], len(missing))
                    self.assertEqual(data["missing_sample"], missing)
                    self.assertFalse(data["missing_sample_truncated"])
                self.assertEqual(batch.data["items"][0]["status"], "inconclusive")

    def test_reports_aired_missing_episodes_and_ignores_future_unknown_and_specials(self):
        result = self._run([_ready([(1, 1)])])
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "updates_available")
        self.assertEqual(result.data["expected_aired"], 3)
        self.assertEqual(result.data["missing_count"], 2)
        self.assertEqual(result.data["missing_sample"], [
            {"season": 1, "episode": 2}, {"season": 2, "episode": 1}
        ])
        self.assertEqual(result.data["future_episode_count"], 1)
        self.assertEqual(result.data["unknown_air_date_count"], 1)
        self.assertEqual(result.data["ignored_specials"], 1)
        self.assertEqual(result.data["resource_followups"], [
            {
                "tool": "library.search_missing_episode_resources",
                "label": "搜索 S01E02 资源",
                "episode_label": "S01E02",
                "arguments": {
                    "query": "The Show",
                    "tmdb_id": "12345",
                    "season": 1,
                    "episode": 2,
                    "as_of": "2026-08-01",
                },
            },
            {
                "tool": "library.search_missing_episode_resources",
                "label": "搜索 S02E01 资源",
                "episode_label": "S02E01",
                "arguments": {
                    "query": "The Show",
                    "tmdb_id": "12345",
                    "season": 2,
                    "episode": 1,
                    "as_of": "2026-08-01",
                },
            },
        ])
        self.assertFalse(result.data["resource_followups_truncated"])

    def test_unpositioned_local_media_blocks_confirmed_missing_followups(self):
        source = {**_ready([(1, 1)]), "ignored_unknown": 1, "local_total": 2}
        with patch("app.agent.domains.library.episode_audit.inspect_series_episode_sources", return_value=[source]), patch(
            "app.agent.domains.library.episode_audit.TMDBClient", return_value=_FakeTMDB()
        ):
            result = audit_series_episodes(dict(self.arguments))
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.data["ignored_unknown_local"], 1)
        self.assertNotIn("resource_followups", result.data)

    def test_unpositioned_extra_does_not_hide_known_complete_inventory(self):
        source = {**_ready([(1, 1), (1, 2), (2, 1)]), "ignored_unknown": 1}
        with patch("app.agent.domains.library.episode_audit.inspect_series_episode_sources", return_value=[source]), patch(
            "app.agent.domains.library.episode_audit.TMDBClient", return_value=_FakeTMDB()
        ):
            result = audit_series_episodes(dict(self.arguments))
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "up_to_date")
        self.assertEqual(result.data["missing_count"], 0)

    def test_latest_episode_numbers_are_not_local_episode_count(self):
        fake = _FakeTMDB()
        fake.details = {
            "name": "The Show",
            "first_air_date": "2026-01-01",
            "seasons": [{"season_number": 1}, {"season_number": 2}],
        }
        fake.seasons = {
            1: {
                "episodes": [
                    {"episode_number": 1, "air_date": "2026-07-01"},
                    {"episode_number": 2, "air_date": "2026-07-08"},
                ]
            },
            2: {
                "episodes": [
                    {"episode_number": 1, "air_date": "2026-07-15"}
                ]
            },
        }
        with patch(
            "app.agent.domains.library.episode_audit.inspect_series_episode_sources",
            return_value=[_ready([(1, 1), (2, 1)])],
        ), patch("app.agent.domains.library.episode_audit.TMDBClient", return_value=fake):
            result = audit_series_episodes(dict(self.arguments))

        self.assertEqual(result.status, "updates_available")
        self.assertEqual(result.data["local_episode_count"], 2)
        self.assertEqual(
            result.data["latest_local"], {"season": 2, "episode": 1}
        )
        self.assertEqual(
            result.data["latest_aired"], {"season": 2, "episode": 1}
        )
        self.assertEqual(result.data["missing_count"], 1)
        self.assertEqual(
            result.data["missing_sample"], [{"season": 1, "episode": 2}]
        )

    def test_exact_target_beyond_missing_sample_is_still_verified(self):
        fake = _FakeTMDB()
        fake.details = {
            "name": "The Show",
            "first_air_date": "2026-01-01",
            "seasons": [{"season_number": 1}],
        }
        fake.seasons = {
            1: {
                "episodes": [
                    {"episode_number": number, "air_date": "2026-07-01"}
                    for number in range(1, 151)
                ]
            }
        }
        arguments = {
            **self.arguments,
            "season": 1,
            "target_episode": 150,
            "library_name": "美女库",
        }
        source = {**_ready([(1, number) for number in range(1, 150)]), "library_name": "美女库"}
        with patch(
            "app.agent.domains.library.episode_audit.inspect_series_episode_sources",
            return_value=[source],
        ) as inspect, patch("app.agent.domains.library.episode_audit.TMDBClient", return_value=fake):
            result = audit_series_episodes(arguments)

        self.assertEqual(result.status, "updates_available")
        self.assertTrue(result.data["missing_sample_truncated"] is False)
        self.assertTrue(result.data["target_aired"] is True)
        self.assertTrue(result.data["target_local"] is False)
        self.assertTrue(result.data["target_missing"] is True)
        self.assertEqual(
            result.data["resource_followups"][0]["episode_label"], "S01E150"
        )
        self.assertEqual(
            result.data["resource_followups"][0]["arguments"]["library_name"],
            "美女库",
        )
        inspect.assert_called_once_with(
            "The Show", tmdb_id="", max_episodes=2000, library_name="美女库"
        )

    def test_resource_followups_stay_bounded_with_full_large_missing_list(self):
        fake = _FakeTMDB()
        fake.details = {"name": "The Show", "first_air_date": "2026-01-01", "seasons": [{"season_number": 1}]}
        fake.seasons = {
            1: {
                "episodes": [
                    {"episode_number": number, "air_date": "2026-07-01"}
                    for number in range(1, 151)
                ]
            }
        }
        with patch(
            "app.agent.domains.library.episode_audit.inspect_series_episode_sources",
            return_value=[_ready([])],
        ), patch("app.agent.domains.library.episode_audit.TMDBClient", return_value=fake):
            result = audit_series_episodes(dict(self.arguments))

        self.assertEqual(result.status, "updates_available")
        self.assertEqual(result.data["missing_count"], 150)
        self.assertEqual(result.data["missing_sample"], [
            {"season": 1, "episode": number} for number in range(1, 151)
        ])
        self.assertFalse(result.data["missing_sample_truncated"])
        self.assertEqual(len(result.data["resource_followups"]), 12)
        self.assertTrue(result.data["resource_followups_truncated"])
        self.assertEqual(
            [item["episode_label"] for item in result.data["resource_followups"]],
            [f"S01E{number:02d}" for number in range(1, 13)],
        )

    def test_up_to_date_and_explicit_season_scope(self):
        self.arguments["season"] = 1
        result = self._run([_ready([(1, 1), (1, 2)])])
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "up_to_date")
        self.assertEqual(result.data["expected_aired"], 2)
        self.assertEqual(result.data["missing_count"], 0)
        self.assertNotIn("resource_followups", result.data)

        self.assertEqual(result.effect_metadata["episode_mapping_context"]["detail"]["name"], "The Show")
        self.assertEqual(result.effect_metadata["episode_mapping_context"]["as_of"], result.data["as_of"])

    def test_unmapped_local_inventory_uses_strict_title_year_tmdb_fallback(self):
        source = {
            **_ready([(1, 1)]),
            "status": "unmapped",
            "selected": {"name": "The Show", "year": "2026", "tmdb_id": ""},
            "mapping": {"status": "unmapped", "tmdb_id": ""},
        }
        result = self._run([source])
        self.assertEqual(result.status, "updates_available")
        self.assertEqual(result.data["tmdb_id"], "12345")
        self.assertEqual(result.data["mapping_status"], "title_year_fallback")
        self.assertEqual(
            result.data["sources"][0]["mapping"]["status"],
            "title_year_fallback",
        )
        self.assertEqual(result.data["local_episode_count"], 1)

    def test_unmapped_title_year_fallback_refuses_ambiguous_tmdb_results(self):
        source = {
            **_ready([(1, 1)]),
            "status": "unmapped",
            "selected": {"name": "The Show", "year": "2026", "tmdb_id": ""},
        }
        fake = _FakeTMDB()
        fake.search = lambda *_args, **_kwargs: [
            {"id": 1, "name": "The Show", "first_air_date": "2026-01-01"},
            {"id": 2, "name": "The Show", "first_air_date": "2026-05-01"},
        ]
        with patch(
            "app.agent.domains.library.episode_audit.inspect_series_episode_sources",
            return_value=[source],
        ), patch("app.agent.domains.library.episode_audit.TMDBClient", return_value=fake):
            result = audit_series_episodes(dict(self.arguments))
        self.assertEqual(result.status, "ambiguous")

    def test_explicit_tmdb_id_binds_unmapped_inventory_only_when_identity_matches(self):
        self.arguments["tmdb_id"] = "12345"
        source = {
            **_ready([(1, 1)]),
            "status": "unmapped",
            "selected": {"name": "The Show", "year": "2026", "tmdb_id": ""},
        }
        result = self._run([source])
        self.assertEqual(result.status, "updates_available")
        self.assertEqual(result.data["local_episode_count"], 1)
        self.assertEqual(result.data["mapping_status"], "explicit_tmdb_id")

        reset_episode_audit_cache_for_tests()
        source["status"] = "unmapped"
        source["mapping"] = {"status": "unmapped", "tmdb_id": ""}
        source["selected"] = {"name": "Another Show", "year": "2026", "tmdb_id": ""}
        mismatch = self._run([source])
        self.assertEqual(mismatch.status, "not_found")

    def test_matching_unmapped_duplicate_is_merged_with_provider_mapped_source(self):
        mapped = _ready([(1, 1)])
        unmapped = {
            **_ready([(1, 2)], server="Emby"),
            "status": "unmapped",
            "selected": {"name": "The Show", "year": "2026", "tmdb_id": ""},
        }
        result = self._run([mapped, unmapped])
        self.assertEqual(result.data["local_episode_count"], 2)
        self.assertEqual(result.data["sources"][1]["status"], "ready")

    def test_ambiguous_unmapped_conflict_and_partial_are_not_claimed_complete(self):
        ambiguous = {**_ready([]), "status": "ambiguous", "selected": None}
        self.assertEqual(self._run([ambiguous]).status, "ambiguous")
        reset_episode_audit_cache_for_tests()
        unmapped = {**_ready([]), "status": "unmapped", "selected": {"name": "The Show", "year": "", "tmdb_id": ""}}
        self.assertEqual(self._run([unmapped]).status, "unmapped")
        reset_episode_audit_cache_for_tests()
        self.assertEqual(self._run([_ready([], tmdb_id="1"), _ready([], server="Emby", tmdb_id="2")]).status, "conflict")
        reset_episode_audit_cache_for_tests()
        unavailable = {**_ready([]), "status": "unavailable", "selected": None, "error": "secret http://private"}
        partial = self._run([_ready([(1, 1), (1, 2), (2, 1)]), unavailable])
        self.assertEqual(partial.status, "inconclusive")
        self.assertNotIn("resource_followups", partial.data)
        self.assertNotIn("private", str(partial.to_dict()))

    def test_truncated_local_inventory_is_inconclusive(self):
        result = self._run([_ready([(1, 1)], truncated=True)])
        self.assertEqual(result.status, "inconclusive")
        self.assertFalse(result.ok)

    def test_tmdb_configuration_and_upstream_errors_are_safe(self):
        for error, status in ((ProviderNotConfigured("private secret"), "not_configured"),
                              (ProviderUnavailable("private secret"), "unavailable")):
            reset_episode_audit_cache_for_tests()
            fake = _FakeTMDB()
            fake.detail = lambda *_args, error=error, **_kwargs: (_ for _ in ()).throw(error)
            with patch("app.agent.domains.library.episode_audit.inspect_series_episode_sources", return_value=[_ready([])]), patch(
                "app.agent.domains.library.episode_audit.TMDBClient", return_value=fake
            ):
                result = audit_series_episodes(dict(self.arguments))
            self.assertEqual(result.status, status)
            self.assertNotIn("private secret", str(result.to_dict()))

    def test_invalidate_cache_evicts_only_matching_normalized_key(self):
        other = {**self.arguments, "season": 2}
        with patch(
            "app.agent.domains.library.episode_audit.inspect_series_episode_sources",
            return_value=[_ready([])],
        ) as inspect, patch("app.agent.domains.library.episode_audit.TMDBClient", return_value=_FakeTMDB()):
            audit_series_episodes(dict(self.arguments))
            audit_series_episodes(dict(other))
            audit_series_episodes(dict(self.arguments))
            self.assertEqual(inspect.call_count, 2)

            invalidate_episode_audit_cache({**self.arguments, "query": "the show"})
            audit_series_episodes(dict(self.arguments))
            self.assertEqual(inspect.call_count, 3)

            audit_series_episodes(dict(other))
            self.assertEqual(inspect.call_count, 3)

    def test_cache_dimensions_include_library_and_exact_target_episode(self):
        scoped = {**self.arguments, "library_name": "动漫库", "season": 1, "target_episode": 1}
        other_episode = {**scoped, "target_episode": 2}
        other_library = {**scoped, "library_name": "美女库"}
        with patch(
            "app.agent.domains.library.episode_audit.inspect_series_episode_sources",
            return_value=[_ready([])],
        ) as inspect, patch("app.agent.domains.library.episode_audit.TMDBClient", return_value=_FakeTMDB()):
            audit_series_episodes(dict(scoped))
            audit_series_episodes(dict(other_episode))
            audit_series_episodes(dict(other_library))
            audit_series_episodes(dict(scoped))
            self.assertEqual(inspect.call_count, 3)

            invalidate_episode_audit_cache(dict(scoped))
            audit_series_episodes(dict(scoped))
            audit_series_episodes(dict(other_episode))
            audit_series_episodes(dict(other_library))
            self.assertEqual(inspect.call_count, 4)

    def test_cache_deduplicates_same_audit(self):
        with patch("app.agent.domains.library.episode_audit.inspect_series_episode_sources", return_value=[_ready([])]) as inspect, patch(
            "app.agent.domains.library.episode_audit.TMDBClient", return_value=_FakeTMDB()
        ):
            audit_series_episodes(dict(self.arguments))
            audit_series_episodes(dict(self.arguments))
        inspect.assert_called_once()

    def test_check_updates_refreshes_inventory_but_plain_audit_cache_still_reuses(self):
        inventory = [(1, 1)]

        def inspect_inventory(*_args, **_kwargs):
            return [_ready(list(inventory))]

        with patch(
            "app.agent.domains.library.episode_audit.inspect_series_episode_sources",
            side_effect=inspect_inventory,
        ) as inspect, patch(
            "app.agent.domains.library.episode_audit.TMDBClient", return_value=_FakeTMDB()
        ):
            first = audit_series_episodes(dict(self.arguments))
            self.assertEqual(first.status, "updates_available")
            self.assertEqual(first.data["local_episode_count"], 1)

            # 普通 audit 仍应命中旧缓存，而不是因为媒体库存变化自行绕过缓存。
            inventory[:] = [(1, 1), (1, 2), (2, 1)]
            cached = audit_series_episodes(dict(self.arguments))
            self.assertEqual(cached.status, "updates_available")
            self.assertEqual(cached.data["local_episode_count"], 1)
            self.assertEqual(inspect.call_count, 1)

            # check_updates 默认 refresh=True：先失效目标缓存，再读取新库存。
            refreshed = check_library_updates(
                {
                    "query": "The Show",
                    "media_type": "tv",
                    "tmdb_id": "",
                    "season": None,
                    "as_of": "2026-08-01",
                }
            )
            self.assertEqual(refreshed.status, "up_to_date")
            self.assertEqual(refreshed.data["local_episode_count"], 3)
            self.assertEqual(inspect.call_count, 2)

            # refresh=False 允许复用刚得到的缓存，即使临时库存再次变化。
            inventory[:] = [(1, 1)]
            reused = check_library_updates(
                {
                    "query": "The Show",
                    "media_type": "tv",
                    "tmdb_id": "",
                    "season": None,
                    "as_of": "2026-08-01",
                    "refresh": False,
                }
            )
            self.assertEqual(reused.status, "up_to_date")
            self.assertEqual(reused.data["local_episode_count"], 3)
            self.assertEqual(inspect.call_count, 2)

    def test_explicit_refresh_replaces_only_target_cache_and_preserves_default_reuse(self):
        other = {**self.arguments, "query": "Other Show"}
        with patch("app.agent.domains.library.episode_audit._audit_uncached", side_effect=[
            ToolResult(True, "updates_available", "old"),
            ToolResult(True, "up_to_date", "other"),
            ToolResult(True, "up_to_date", "fresh"),
        ]) as read:
            audit_series_episodes(self.arguments)
            audit_series_episodes(other)
            self.assertEqual(audit_series_episodes(self.arguments).summary, "old")
            self.assertEqual(audit_series_episodes(self.arguments, refresh=True).summary, "fresh")
            self.assertEqual(audit_series_episodes(self.arguments).summary, "fresh")
            self.assertEqual(audit_series_episodes(other).summary, "other")
            self.assertEqual(read.call_count, 3)

    def test_explicit_refresh_fences_old_flight_and_keeps_new_readers_singleflight(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from app.agent.domains.library import episode_audit as module

        for old_refresh, old_first in ((False, True), (False, False), (True, True), (True, False)):
            with self.subTest(old_refresh=old_refresh, old_first=old_first):
                reset_episode_audit_cache_for_tests()
                started = [threading.Event(), threading.Event()]
                release = [threading.Event(), threading.Event()]
                joined = threading.Event()
                calls = []

                def read(_arguments, *, calls=calls, started=started, release=release):
                    number = len(calls)
                    calls.append(number)
                    started[number].set()
                    self.assertTrue(release[number].wait(3))
                    return ToolResult(True, "up_to_date", str(number))

                with patch.object(module, "_audit_uncached", side_effect=read), ThreadPoolExecutor(max_workers=3) as pool:
                    try:
                        old = pool.submit(audit_series_episodes, self.arguments, refresh=old_refresh)
                        self.assertTrue(started[0].wait(3))
                        fresh = pool.submit(audit_series_episodes, self.arguments, refresh=True)
                        self.assertTrue(started[1].wait(3))
                        flight = next(iter(module._inflight.values()))
                        wait = flight.wait

                        def wait_for_fresh(*args, joined=joined, wait=wait, **kwargs):
                            joined.set()
                            return wait(*args, **kwargs)

                        with patch.object(flight, "wait", side_effect=wait_for_fresh):
                            reader = pool.submit(audit_series_episodes, self.arguments)
                            self.assertTrue(joined.wait(3))
                            if old_first:
                                release[0].set()
                                self.assertEqual(old.result(3).summary, "0")
                                self.assertEqual(list(module._inflight.values()), [flight])
                            release[1].set()
                            self.assertEqual(fresh.result(3).summary, "1")
                            self.assertEqual(reader.result(3).summary, "1")
                            release[0].set()
                            self.assertEqual(old.result(3).summary, "0")
                        self.assertEqual(audit_series_episodes(self.arguments).summary, "1")
                        self.assertEqual(calls, [0, 1])
                        self.assertFalse(module._inflight)
                    finally:
                        for event in release:
                            event.set()

    def test_unknown_air_dates_are_not_presented_as_definitively_current(self):
        with patch("app.agent.domains.library.episode_audit.inspect_series_episode_sources", return_value=[_ready([(1, 1), (1, 2), (2, 1)])]), patch(
            "app.agent.domains.library.episode_audit.TMDBClient", return_value=_FakeTMDB()
        ):
            result = audit_series_episodes(dict(self.arguments))
        self.assertEqual(result.status, "up_to_date")
        self.assertEqual(result.data["unknown_air_date_count"], 1)
        self.assertIn("已知播出日期", result.summary)
        self.assertIn("不能据此判断全部最新", " ".join(result.suggestions))

    def test_refresh_replaces_inflight_query_without_stale_cache_publish_or_unlock(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from app.agent.domains.library import episode_audit as module
        old_started, new_started, old_release, new_release = (threading.Event() for _ in range(4))
        calls = []
        def audit(_arguments):
            calls.append(len(calls) + 1)
            number = calls[-1]
            (old_started if number == 1 else new_started).set()
            self.assertTrue((old_release if number == 1 else new_release).wait(3))
            return ToolResult(True, "up_to_date", "模拟库存", data={"local_episode_count": number})
        with patch("app.agent.domains.library.episode_audit._audit_uncached", side_effect=audit), ThreadPoolExecutor(max_workers=2) as pool:
            try:
                old = pool.submit(audit_series_episodes, dict(self.arguments))
                self.assertTrue(old_started.wait(3))
                fresh = pool.submit(check_library_updates, dict(self.arguments))
                self.assertTrue(new_started.wait(3))
                old_release.set()
                self.assertEqual(old.result(3).data["local_episode_count"], 1)
                self.assertEqual(len(module._inflight), 1)
                new_release.set()
                self.assertEqual(fresh.result(3).data["local_episode_count"], 2)
                self.assertEqual(audit_series_episodes(dict(self.arguments)).data["local_episode_count"], 2)
                self.assertEqual(calls, [1, 2])
            finally:
                old_release.set()
                new_release.set()


class TMDBSeasonPathTests(unittest.TestCase):
    def test_tv_season_detail_uses_bounded_relative_path(self):
        from app.clients.tmdb import TMDBClient

        client = TMDBClient(api_key="key")
        with patch.object(client, "get", return_value={"episodes": []}) as get:
            client.tv_season_detail("12345", 2)
        get.assert_called_once_with("/tv/12345/season/2")
        with self.assertRaises(ValueError):
            client.tv_season_detail("../secret", 2)
        with self.assertRaises(ValueError):
            client.tv_season_detail("12345", True)




class EpisodeAuditClientLifecycleTests(unittest.TestCase):
    def test_owned_tmdb_client_is_closed_after_success(self) -> None:
        reset_episode_audit_cache_for_tests()
        client = _FakeTMDB()
        closed = False

        def close() -> None:
            nonlocal closed
            closed = True

        client.close = close
        with patch(
            "app.agent.domains.library.episode_audit.inspect_series_episode_sources",
            return_value=[_ready([(1, 1)])],
        ), patch("app.agent.domains.library.episode_audit.TMDBClient", return_value=client):
            result = audit_series_episodes({
                "query": "The Show",
                "tmdb_id": "",
                "season": None,
                "as_of": "2026-08-01",
            })

        self.assertTrue(result.ok)
        self.assertTrue(closed)

    def test_close_failure_does_not_replace_tmdb_error(self) -> None:
        reset_episode_audit_cache_for_tests()
        client = _FakeTMDB()
        client.detail = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProviderUnavailable("upstream unavailable")
        )
        client.close = lambda: (_ for _ in ()).throw(RuntimeError("close failed"))
        with patch(
            "app.agent.domains.library.episode_audit.inspect_series_episode_sources",
            return_value=[_ready([])],
        ), patch("app.agent.domains.library.episode_audit.TMDBClient", return_value=client):
            result = audit_series_episodes({
                "query": "The Show",
                "tmdb_id": "",
                "season": None,
                "as_of": "2026-08-01",
            })

        self.assertEqual(result.status, "unavailable")


if __name__ == "__main__":
    unittest.main()
