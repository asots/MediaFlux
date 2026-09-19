"""Media Agent 外部影视探索搜索测试。"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from app.agent.discovery_actions import search_arguments, search_discovery
from app.agent.discovery_mapping_actions import get_discovery_detail
from app.agent.errors import AgentToolError
from app.agent.models import ToolContext
from app.discovery.models import MediaCard
from app.discovery.search import DiscoverySearchResult

_SECRET_POSTER = "https://image.example/private/poster?api_key=secret"
_SECRET_BACKDROP = "https://image.example/private/backdrop?token=secret"


def _identity(arguments):
    return dict(arguments)


def _card(index: int = 1, **overrides) -> MediaCard:
    values = {
        "provider": "tmdb",
        "external_id": str(1000 + index),
        "media_type": "movie",
        "title": f"示例影片 {index}",
        "original_title": f"Demo Movie {index}",
        "year": "2026",
        "overview": "剧情简介" * 180,
        "poster_key": _SECRET_POSTER,
        "backdrop_key": _SECRET_BACKDROP,
        "rating": 8.2,
        "rating_source": "tmdb",
        "release_date": "2026-08-01",
        "tmdb_id": str(1000 + index),
    }
    values.update(overrides)
    return MediaCard(**values)


def _result(
    *, items=(), attempted=("tmdb",), succeeded=("tmdb",), errors=(), has_more=False
) -> DiscoverySearchResult:
    return DiscoverySearchResult(
        query="沙丘2",
        page=1,
        items=tuple(items),
        has_more=has_more,
        providers_attempted=tuple(attempted),
        providers_succeeded=tuple(succeeded),
        errors=tuple(errors),
    )


class FakeDiscoverySearchService:
    def __init__(self, result: DiscoverySearchResult):
        self.result = result
        self.calls: list[tuple[str, int, list[str] | None]] = []

    def search(self, query: str, page: int, providers):
        self.calls.append((query, page, providers))
        return self.result


class AgentDiscoverySearchTests(unittest.TestCase):
    def test_arguments_normalize_and_reject_unsafe_fields(self):
        self.assertEqual(
            search_arguments(
                {
                    "query": "  沙丘２  ",
                    "page": 2,
                    "providers": ["TMDB", "tmdb", "Bangumi"],
                    "limit": 10,
                }
            ),
            {
                "query": "沙丘2",
                "page": 2,
                "providers": ["tmdb", "bangumi"],
                "limit": 10,
            },
        )
        invalid = (
            {},
            {"query": ""},
            {"query": "x\ny"},
            {"query": "x", "page": True},
            {"query": "x", "page": 101},
            {"query": "x", "limit": 0},
            {"query": "x", "providers": []},
            {"query": "x", "providers": ["unknown"]},
            {"query": "x", "url": "https://example.invalid"},
            {"query": "x", "token": "secret"},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(AgentToolError):
                search_arguments(arguments)

    def test_disabled_feature_does_not_create_or_call_service(self):
        service = Mock()
        with (
            patch("app.agent.discovery_actions.config.get_bool", return_value=False),
            patch(
                "app.agent.discovery_actions.get_discovery_search_service",
                return_value=service,
            ) as getter,
        ):
            result = search_discovery({"query": "沙丘2"})
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "disabled")
        getter.assert_not_called()
        service.search.assert_not_called()

    def test_success_uses_safe_allowlist_and_limit(self):
        service = FakeDiscoverySearchService(
            _result(items=(_card(1), _card(2)), has_more=True)
        )
        with (
            patch("app.agent.discovery_actions.config.get_bool", return_value=True),
            patch(
                "app.agent.discovery_actions.get_discovery_search_service",
                return_value=service,
            ),
        ):
            result = search_discovery(
                {"query": "沙丘2", "page": 1, "providers": ["tmdb"], "limit": 1}
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "success")
        self.assertEqual(service.calls, [("沙丘2", 1, ["tmdb"])])
        self.assertEqual(result.data["total"], 2)
        self.assertEqual(result.data["returned"], 1)
        self.assertEqual(len(result.data["items"][0]["overview"]), 500)
        serialized = repr(result.to_dict())
        self.assertNotIn("poster_key", serialized)
        self.assertNotIn("backdrop_key", serialized)
        self.assertNotIn(_SECRET_POSTER, serialized)
        self.assertNotIn(_SECRET_BACKDROP, serialized)

    def test_error_retry_after_is_bounded(self):
        result = _result(
            attempted=("tmdb",),
            succeeded=(),
            errors=(
                {"provider": "tmdb", "code": "rate_limited", "retry_after": 999999},
            ),
        )
        service = FakeDiscoverySearchService(result)
        with (
            patch("app.agent.discovery_actions.config.get_bool", return_value=True),
            patch(
                "app.agent.discovery_actions.get_discovery_search_service",
                return_value=service,
            ),
        ):
            response = search_discovery({"query": "沙丘2"})
        self.assertEqual(response.data["errors"][0]["retry_after"], 86400)

    def test_partial_empty_and_full_failure_semantics(self):
        cases = (
            (
                _result(
                    items=(_card(),),
                    attempted=("tmdb", "douban"),
                    succeeded=("tmdb",),
                    errors=(
                        {
                            "provider": "douban",
                            "code": "unavailable",
                            "message": "token=message-should-not-leak",
                            "retry_after": 0,
                            "detail": "token=should-not-leak",
                        },
                    ),
                ),
                True,
                "partial",
            ),
            (_result(items=()), True, "empty"),
            (
                _result(
                    items=(),
                    attempted=("tmdb",),
                    succeeded=(),
                    errors=(
                        {
                            "provider": "tmdb",
                            "code": "authentication",
                            "message": "api_key=message-should-not-leak",
                            "retry_after": 0,
                            "detail": "api_key=should-not-leak",
                        },
                    ),
                ),
                False,
                "unavailable",
            ),
        )
        for raw, ok, status in cases:
            with (
                self.subTest(status=status),
                patch("app.agent.discovery_actions.config.get_bool", return_value=True),
                patch(
                    "app.agent.discovery_actions.get_discovery_search_service",
                    return_value=FakeDiscoverySearchService(raw),
                ),
            ):
                result = search_discovery({"query": "沙丘2"})
            self.assertEqual(result.ok, ok)
            self.assertEqual(result.status, status)
            self.assertNotIn("should-not-leak", repr(result.to_dict()))


class AgentDiscoveryIdentityRegressionTests(unittest.TestCase):
    def search(self, arguments, raw):
        service = FakeDiscoverySearchService(raw)
        with (
            patch("app.agent.discovery_actions.config.get_bool", return_value=True),
            patch("app.agent.discovery_actions.get_discovery_search_service", return_value=service),
        ):
            result = search_discovery(arguments)
        return result, service

    def test_structured_type_and_year_filter_candidates_without_polluting_title(self):
        for media_type in ("tv", "movie"):
            with self.subTest(media_type=media_type):
                result, service = self.search(
                    {"query": "狐妖小红娘", "media_type": media_type, "year": "2015", "providers": ["tmdb"]},
                    _result(items=(
                        _card(1, title="狐妖小红娘", media_type=media_type, year="2015"),
                        _card(2, title="狐妖小红娘", media_type=media_type, year="2024"),
                        _card(3, title="狐妖小红娘", media_type="movie" if media_type == "tv" else "tv", year="2015"),
                    )),
                )
                self.assertEqual(service.calls, [("狐妖小红娘", 1, ["tmdb"])])
                self.assertEqual([item["external_id"] for item in result.data["items"]], ["1001"])
                self.assertEqual(result.data["filters"], {"media_type": media_type, "year": "2015"})

    def test_year_only_is_exact_filter_not_a_title_suffix(self):
        result, service = self.search(
            {"query": "斗罗大陆", "year": "2018", "page": 2, "providers": ["tmdb", "douban", "bangumi"], "limit": 1},
            _result(items=(
                _card(1, provider="tmdb", media_type="tv", year="2018"),
                _card(2, provider="douban", media_type="tv", year="2021"),
                _card(3, provider="bangumi", media_type="tv", year=""),
                _card(4, provider="douban", media_type="tv", year="2018"),
            ), has_more=True),
        )
        self.assertEqual(service.calls, [("斗罗大陆", 2, ["tmdb", "douban", "bangumi"])])
        self.assertEqual(result.data["total"], 2)
        self.assertEqual(result.data["returned"], 1)
        self.assertTrue(result.data["has_more"])

    def test_region_and_genre_stay_keyword_constraints_with_explicit_semantics(self):
        result, service = self.search(
            {"query": "示例", "media_type": "tv", "year": "2026", "region": "欧美", "genre": "科幻"},
            _result(items=(_card(media_type="tv"),)),
        )
        self.assertEqual(service.calls, [("示例 欧美 科幻", 1, None)])
        self.assertEqual(result.data["filter_modes"], {
            "media_type": "exact", "year": "exact", "region": "keyword", "genre": "keyword",
        })
        self.assertEqual(result.data["filters"]["region"], "欧美")
        self.assertEqual(result.data["filters"]["genre"], "科幻")

    def test_keyword_constraints_cannot_be_silently_truncated(self):
        service = Mock()
        with (
            patch("app.agent.discovery_actions.config.get_bool", return_value=True),
            patch("app.agent.discovery_actions.get_discovery_search_service", return_value=service),
            self.assertRaisesRegex(AgentToolError, "120"),
        ):
            search_discovery({"query": "长" * 120, "region": "日本", "genre": "动画"})
        service.search.assert_not_called()

    def test_structured_filters_do_not_remove_existing_year_or_type_words_in_title(self):
        _, service = self.search(
            {"query": "电影 2026", "media_type": "movie", "year": "2026"}, _result(),
        )
        self.assertEqual(service.calls, [("电影 2026", 1, None)])

    def test_tmdb_agent_search_keeps_chinese_query_on_single_http_request(self):
        from app.clients.tmdb import TMDBClient
        from app.discovery.search import DiscoverySearchService, TMDBSearchProvider
        from tests.test_discovery_search import FakeResponse, FakeSession

        http = FakeSession([FakeResponse({"results": [
            {"id": 75787, "media_type": "tv", "name": "狐妖小红娘", "first_air_date": "2015-06-25"},
            {"id": 205599, "media_type": "tv", "name": "狐妖小红娘月红篇", "first_air_date": "2024-05-23"},
        ], "total_pages": 1})])
        client = TMDBClient(api_key="test-key", proxy_url="", base_url="https://tmdb.invalid/3", session=http)
        service = DiscoverySearchService(providers={"tmdb": TMDBSearchProvider(client=client)})
        try:
            with (
                patch("app.agent.discovery_actions.config.get_bool", return_value=True),
                patch("app.agent.discovery_actions.get_discovery_search_service", return_value=service),
            ):
                result = search_discovery({"query": "狐妖小红娘", "media_type": "tv", "year": "2015", "providers": ["tmdb"]})
            self.assertEqual(len(http.calls), 1)
            self.assertEqual(http.calls[0][2]["params"]["query"], "狐妖小红娘")
            self.assertEqual([item["external_id"] for item in result.data["items"]], ["75787"])
        finally:
            service.shutdown()


class AgentDiscoveryDetailRegressionTests(unittest.TestCase):
    def detail(self, card):
        service = Mock()
        service.get_detail.return_value = card
        with (
            patch("app.agent.discovery_mapping_actions.config.get_bool", return_value=True),
            patch("app.agent.discovery_mapping_actions.get_discovery_service", return_value=service),
            patch("app.agent.discovery_mapping_actions.db.get_media_external_id", return_value=None),
        ):
            result = get_discovery_detail(
                {"provider": card.provider, "external_id": card.external_id, "media_type": card.media_type},
                ToolContext(),
            )
        service.get_detail.assert_called_once_with(card.provider, card.media_type, card.external_id)
        return result

    def test_detail_preserves_exact_identity_and_default_season_counts(self):
        from app.discovery.providers.tmdb import TMDBProvider

        card = TMDBProvider._card({
            "id": 75787, "name": "狐妖小红娘", "first_air_date": "2015-06-25",
            "number_of_seasons": 1, "number_of_episodes": 183,
            "seasons": [
                {"season_number": 0, "name": "特别篇", "episode_count": 29},
                {"season_number": 1, "name": "全集", "episode_count": 183},
            ],
        }, "tv")
        result = self.detail(card)
        self.assertEqual(result.data["external_id"], "75787")
        self.assertEqual(result.data["tmdb_id"], "75787")
        self.assertEqual(result.data["stable_id"], "tmdb:tv:75787")
        self.assertEqual(result.data["number_of_seasons"], 1)
        self.assertEqual(result.data["number_of_episodes"], 183)
        self.assertEqual([(s["season_number"], s["episode_count"]) for s in result.data["seasons"]], [(0, 29), (1, 183)])
        self.assertIn("默认", result.data["seasons_note"])
        self.assertIn("特别篇", result.data["seasons_note"])
        self.assertEqual(result.data["credits_state"], "not_queried")
        self.assertTrue(result.data["mapping_confirmed"])

    def test_detail_unknown_counts_are_not_reported_as_zero_or_borrowed(self):
        for provider, media_type in (("tmdb", "tv"), ("tmdb", "movie"), ("douban", "tv"), ("bangumi", "tv")):
            with self.subTest(provider=provider, media_type=media_type):
                result = self.detail(MediaCard(provider=provider, external_id="42", media_type=media_type, title="同名作品"))
                self.assertEqual(result.data["external_id"], "42")
                self.assertEqual(result.data["stable_id"], f"{provider}:{media_type}:42")
                self.assertEqual(result.data["tmdb_id"], "42" if provider == "tmdb" else "")
                self.assertIsNone(result.data["number_of_seasons"])
                self.assertIsNone(result.data["number_of_episodes"])
                self.assertIsNone(result.data["seasons"])

    def test_detail_season_names_use_existing_public_sanitizer_without_mutation(self):
        from app.agent.discovery_mapping_actions import _safe
        from app.discovery.providers.tmdb import TMDBProvider

        name = "长季名" * 200
        card = TMDBProvider._card({"id": 75787, "name": "狐妖小红娘", "seasons": [
            {"season_number": 0, "name": name, "episode_count": 29},
        ]}, "tv")
        result = self.detail(card)
        self.assertEqual(result.data["seasons"][0]["name"], _safe(name, 160))
        self.assertEqual(card.seasons[0]["name"], name)
