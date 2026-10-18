"""Media Agent 本地剧集库存单元测试：桩替换来源读取，不连接媒体服务器或真实外网。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pytest

from app.agent.domains.library.episode_count import (
    count_series_episodes,
    count_series_episodes_arguments,
)
from app.agent.errors import AgentToolError


class LibraryEpisodeCountArgumentsTests(unittest.TestCase):
    def test_normalizes_supported_arguments(self):
        self.assertEqual(
            count_series_episodes_arguments(
                {"query": "  师兄啊师兄  ", "tmdb_id": " 12345 "}
            ),
            {"query": "师兄啊师兄", "tmdb_id": "12345"},
        )

    def test_accepts_named_library_scope(self):
        self.assertEqual(
            count_series_episodes_arguments(
                {
                    "query": "师兄啊师兄",
                    "library_name": "  美女库  ",
                }
            ),
            {"query": "师兄啊师兄", "tmdb_id": "", "library_name": "美女库"},
        )

    def test_rejects_invalid_or_extra_arguments(self):
        invalid = (
            {},
            {"query": 1},
            {"query": "剧名\n注入"},
            {"query": "剧名", "tmdb_id": "abc"},
            {"query": "剧名", "extra": True},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(AgentToolError):
                count_series_episodes_arguments(arguments)


class LibraryEpisodeCountToolTests(unittest.TestCase):
    @patch("app.agent.domains.library.episode_count.inspect_series_episode_sources")
    def test_counts_unique_local_episode_coordinates_and_seasons(self, inspect_sources):
        inspect_sources.return_value = [
            {
                "server_type": "jellyfin",
                "server_name": "Jellyfin",
                "status": "unmapped",
                "selected": {"name": "师兄啊师兄", "year": "2023"},
                "episodes": [(1, 1), (1, 2), (2, 1)],
                "ignored_specials": 2,
                "ignored_unknown": 0,
                "truncated": False,
            },
            {
                "server_type": "emby",
                "server_name": "Emby",
                "status": "ready",
                "selected": {"name": "师兄啊师兄", "year": "2023"},
                "episodes": [(1, 1), (2, 1), (2, 2)],
                "ignored_specials": 0,
                "ignored_unknown": 0,
                "truncated": False,
            },
        ]

        result = count_series_episodes({"query": "师兄啊师兄", "tmdb_id": ""})

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.data["local_episode_count"], 4)
        self.assertEqual(result.data["season_count"], 2)
        self.assertEqual(
            result.data["seasons"],
            [
                {
                    "season": 1,
                    "count": 2,
                    "first_episode": 1,
                    "last_episode": 2,
                    "episode_ranges": [[1, 2]],
                },
                {
                    "season": 2,
                    "count": 2,
                    "first_episode": 1,
                    "last_episode": 2,
                    "episode_ranges": [[1, 2]],
                },
            ],
        )
        self.assertEqual(result.data["matched_source_count"], 2)
        self.assertIs(result.data["inventory_complete"], True)
        self.assertIn("本地收录 4 集", result.summary)
        inspect_sources.assert_called_once_with(
            "师兄啊师兄",
            tmdb_id="",
            max_episodes=2000,
            include_specials=False,
        )

    @patch("app.agent.domains.library.episode_count.inspect_series_episode_sources")
    def test_limits_count_to_named_library_without_exposing_internal_id(
        self, inspect_sources
    ):
        inspect_sources.return_value = [
            {
                "server_type": "jellyfin",
                "server_name": "Jellyfin",
                "library_name": "美女库",
                "status": "ready",
                "selected": {"name": "师兄啊师兄", "year": "2023"},
                "episodes": [(1, 1), (1, 2)],
                "truncated": False,
            }
        ]

        result = count_series_episodes(
            {
                "query": "师兄啊师兄",
                "tmdb_id": "",
                "library_name": "美女库",
            }
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.data["library_name"], "美女库")
        self.assertIs(result.data["inventory_complete"], True)
        inspect_sources.assert_called_once_with(
            "师兄啊师兄",
            tmdb_id="",
            max_episodes=2000,
            include_specials=False,
            library_name="美女库",
        )
        self.assertNotIn("library-id", repr(result.to_dict()))

    @patch("app.agent.domains.library.episode_count.inspect_series_episode_sources")
    def test_does_not_report_zero_when_media_server_is_not_configured(
        self, inspect_sources
    ):
        inspect_sources.return_value = []

        result = count_series_episodes({"query": "师兄啊师兄", "tmdb_id": ""})

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "not_configured")
        self.assertIsNone(result.data["local_episode_count"])
        self.assertIs(result.data["inventory_complete"], False)
        self.assertNotIn("0 集", result.summary)

    @patch("app.agent.domains.library.episode_count.inspect_series_episode_sources")
    def test_not_found_returns_candidates_without_false_count(self, inspect_sources):
        inspect_sources.return_value = [
            {
                "server_type": "jellyfin",
                "server_name": "Jellyfin",
                "status": "not_found",
                "candidates": [
                    {"name": "师兄啊师兄 年番", "year": "2024", "id": "private"}
                ],
                "episodes": [],
            }
        ]

        result = count_series_episodes({"query": "师兄啊师兄", "tmdb_id": ""})

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "not_found")
        self.assertIsNone(result.data["local_episode_count"])
        self.assertIs(result.data["inventory_complete"], False)
        self.assertEqual(
            result.data["candidates"], [{"name": "师兄啊师兄 年番", "year": "2024"}]
        )
        self.assertNotIn("id", result.data["candidates"][0])

    @patch("app.agent.domains.library.episode_count.inspect_series_episode_sources")
    def test_marks_unnumbered_or_truncated_inventory_as_partial(self, inspect_sources):
        inspect_sources.return_value = [
            {
                "server_type": "jellyfin",
                "server_name": "Jellyfin",
                "status": "ready",
                "selected": {"name": "测试剧", "year": "2026"},
                "episodes": [(1, 1), (1, 2)],
                "ignored_unknown": 1,
                "truncated": True,
            }
        ]

        result = count_series_episodes({"query": "测试剧", "tmdb_id": ""})

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.data["local_episode_count"], 2)
        self.assertIs(result.data["inventory_complete"], False)
        self.assertEqual(
            result.data["seasons"],
            [
                {
                    "season": 1,
                    "count": 2,
                    "first_episode": 1,
                    "last_episode": 2,
                    "episode_ranges": [[1, 2]],
                }
            ],
        )
        self.assertIn("下限", result.suggestions[0])

    @patch("app.agent.domains.library.episode_count.inspect_series_episode_sources")
    def test_preserves_normalized_legacy_input_and_serialized_output_fields(
        self, inspect_sources
    ):
        inspect_sources.return_value = [
            {
                "server_type": "jellyfin",
                "server_name": "Jellyfin",
                "library_name": "剧集库",
                "status": "ready",
                "selected": {"name": "师兄啊师兄", "year": "2023", "id": "private"},
                "episodes": [(1, 1), (1, 2)],
                "ignored_specials": 3,
                "ignored_unknown": 0,
                "truncated": False,
            }
        ]
        arguments = count_series_episodes_arguments(
            {
                "query": "  师兄啊师兄  ",
                "tmdb_id": " 12345 ",
                "library_name": " 剧集库 ",
            }
        )

        result = count_series_episodes(arguments)
        payload = result.to_dict()

        self.assertEqual(
            arguments,
            {"query": "师兄啊师兄", "tmdb_id": "12345", "library_name": "剧集库"},
        )
        inspect_sources.assert_called_once_with(
            "师兄啊师兄",
            tmdb_id="12345",
            max_episodes=2000,
            include_specials=False,
            library_name="剧集库",
        )
        legacy_data = {
            "query": "师兄啊师兄",
            "library_name": "剧集库",
            "title": "师兄啊师兄",
            "year": "2023",
            "local_episode_count": 2,
            "season_count": 1,
            "matched_source_count": 1,
            "source_count": 1,
            "ignored_specials": 3,
            "ignored_unknown": 0,
            "truncated": False,
            "count_definition": "Jellyfin / Emby 中已识别季号和集号的本地普通集；跨媒体服务器重复季集号只计一次。",
        }
        self.assertEqual(
            {key: payload["data"][key] for key in legacy_data}, legacy_data
        )
        self.assertEqual(
            set(payload),
            {"ok", "status", "summary", "data", "evidence", "suggestions", "error"},
        )
        self.assertIs(payload["ok"], True)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["error"], "")
        self.assertIn("本地收录 2 集", payload["summary"])
        self.assertEqual(len(payload["evidence"]), 1)
        self.assertEqual(payload["evidence"][0]["source"], "media_servers")
        self.assertIn("特别篇", payload["suggestions"][0])
        self.assertIs(payload["data"]["inventory_complete"], True)
        self.assertEqual(
            payload["data"]["seasons"],
            [
                {
                    "season": 1,
                    "count": 2,
                    "first_episode": 1,
                    "last_episode": 2,
                    "episode_ranges": [[1, 2]],
                }
            ],
        )
        legacy_source = {
            "server_type": "jellyfin",
            "server_name": "Jellyfin",
            "library_name": "剧集库",
            "status": "ready",
            "title": "师兄啊师兄",
            "year": "2023",
            "local_episode_count": 2,
            "season_count": 1,
            "truncated": False,
            "ignored_specials": 3,
            "ignored_unknown": 0,
        }
        self.assertEqual(len(payload["data"]["sources"]), 1)
        self.assertEqual(
            {key: payload["data"]["sources"][0][key] for key in legacy_source},
            legacy_source,
        )
        self.assertNotIn("private", repr(payload))


# 以下为相同公共契约的参数化边界用例；共享输入构造，不重复生产计算逻辑。
def inventory_source(episodes=(), **fields):
    return {
        "server_type": "jellyfin", "server_name": "Jellyfin", "status": "ready",
        "selected": {"name": "测试剧", "year": "2026"}, "episodes": list(episodes),
        **fields,
    }


@pytest.fixture
def inspect_inventory(monkeypatch):
    def inspect(sources):
        monkeypatch.setattr("app.agent.domains.library.episode_count.inspect_series_episode_sources", lambda *a, **kw: sources)
        return count_series_episodes({"query": "测试剧"})
    return inspect


@pytest.mark.parametrize("inventories, expected", [
    ([[(1, 12), (1, 4), (1, 1), (1, 2), (1, 5), (1, 12), (1, 9), (1, 11)]],
     [(1, 7, [[1, 2], [4, 5], [9, 9], [11, 12]])]),
    ([[(2, 4), (1, 5), (1, 1), (1, 2), (1, 5), (2, 1)], [(1, 2), (1, 3), (1, 5), (2, 1), (2, 3), (2, 3)]],
     [(1, 4, [[1, 3], [5, 5]]), (2, 3, [[1, 1], [3, 4]])]),
])
def test_observed_ranges_preserve_gaps_seasons_and_deduplicate(inspect_inventory, inventories, expected):
    result = inspect_inventory([inventory_source(episodes, server_name=f"source-{i}") for i, episodes in enumerate(inventories)])
    assert result.ok and result.status == "success"
    assert result.data["inventory_complete"] is True
    assert result.data["local_episode_count"] == 7
    assert result.data["season_count"] == len(expected)
    assert result.data["matched_source_count"] == len(inventories)
    assert result.data["seasons"] == [
        {"season": season, "count": count, "first_episode": ranges[0][0], "last_episode": ranges[-1][1], "episode_ranges": ranges}
        for season, count, ranges in expected
    ]
    assert [s["local_episode_count"] for s in result.data["sources"]] == ([7] if len(inventories) == 1 else [5, 5])


@pytest.mark.parametrize("status", ["ready", "unmapped"])
@pytest.mark.parametrize("defect", [{"truncated": True}, {"ignored_unknown": 1}])
@pytest.mark.parametrize("other_complete", [False, True])
def test_incomplete_inventory_keeps_ranges_and_lower_bound(inspect_inventory, status, defect, other_complete):
    sources = [inventory_source([(1, 4), (1, 1), (1, 2), (1, 4)], status=status, **defect)]
    if other_complete:
        sources.append(inventory_source([(1, 1)]))
    result = inspect_inventory(sources)
    assert result.ok and result.status == "partial"
    assert result.data["inventory_complete"] is False
    assert result.data["local_episode_count"] == 3
    assert result.data["matched_source_count"] == len(sources)
    assert result.data["truncated"] == defect.get("truncated", False)
    assert result.data["ignored_unknown"] == defect.get("ignored_unknown", 0)
    assert result.data["seasons"][0]["episode_ranges"] == [[1, 2], [4, 4]]
    assert "下限" in result.suggestions[0]


@pytest.mark.parametrize("unresolved", [
    {"status": status} for status in ("unavailable", "ambiguous", "conflict", "library_ambiguous", "library_not_found")
] + [{"status": "not_found", "truncated": True}])
@pytest.mark.parametrize("unresolved_first", [False, True])
def test_unresolved_source_cannot_prove_absence(inspect_inventory, unresolved, unresolved_first):
    sources = [inventory_source([(1, 1), (1, 3), (2, 2)]), inventory_source(**unresolved)]
    result = inspect_inventory(list(reversed(sources)) if unresolved_first else sources)
    assert result.ok and result.status == "partial"
    assert result.data["inventory_complete"] is False
    assert result.data["local_episode_count"] == 3
    assert result.data["matched_source_count"] == 1
    assert result.data["source_count"] == 2
    assert [(s["season"], s["episode_ranges"]) for s in result.data["seasons"]] == [(1, [[1, 1], [3, 3]]), (2, [[2, 2]])]


@pytest.mark.parametrize("status", ["ready", "unmapped"])
@pytest.mark.parametrize("not_found_first", [False, True])
def test_not_found_other_source_does_not_hide_complete_inventory(inspect_inventory, status, not_found_first):
    sources = [inventory_source([(1, 1), (1, 3)], status=status, ignored_specials=2), inventory_source(status="not_found")]
    result = inspect_inventory(list(reversed(sources)) if not_found_first else sources)
    assert result.ok and result.status == "success"
    assert result.data["inventory_complete"] is True
    assert result.data["local_episode_count"] == 2
    assert result.data["matched_source_count"] == 1
    assert result.data["source_count"] == 2
    assert result.data["ignored_specials"] == 2
    assert result.data["seasons"][0]["episode_ranges"] == [[1, 1], [3, 3]]


@pytest.mark.parametrize("status", ["ready", "unmapped"])
def test_selected_empty_inventory_is_not_unmatched(inspect_inventory, status):
    result = inspect_inventory([inventory_source(status=status)])
    assert result.ok and result.status == "success"
    assert result.data["inventory_complete"] is True
    assert result.data["title"] == "测试剧"
    assert result.data["local_episode_count"] == result.data["season_count"] == 0
    assert result.data["seasons"] == []
    assert result.data["matched_source_count"] == 1


@pytest.mark.parametrize("status, expected_status", [
    ("unavailable", "unavailable"), ("ambiguous", "ambiguous"), ("conflict", "ambiguous"),
    ("library_not_found", "library_not_found"), ("library_ambiguous", "library_ambiguous"),
])
def test_unmatched_inventory_is_never_complete_or_zero(inspect_inventory, status, expected_status):
    result = inspect_inventory([inventory_source(status=status)])
    assert not result.ok and result.status == expected_status
    assert result.data["inventory_complete"] is False
    assert result.data["local_episode_count"] is None
    assert result.data["matched_source_count"] == 0
    assert result.data["source_count"] == 1
