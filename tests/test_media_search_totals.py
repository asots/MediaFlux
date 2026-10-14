"""搜索总数必须来自原生 API；页长、标题匹配不能替代演员作品统计。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app import services
from app.agent.domain_catalog import library, library_search
from app.agent.errors import AgentToolError
from app.agent.models import ToolContext
from app.agent.provider_gateway import ProviderGateway
from app.agent.provider_models import ProviderProfileView
from app.agent.provider_operations import build_provider_catalog
from app.agent.providers.media_server import MediaServerProviderTransport
from app.clients.base import MediaItem, MediaSearchResult
from app.clients.emby import EmbyClient
from app.clients.jellyfin import JellyfinClient
from app.modules.media_server_profiles import MediaServerProfile


@pytest.fixture(params=[JellyfinClient, EmbyClient], ids=["jellyfin", "emby"])
def client(request):
    instance = request.param("http://synthetic.invalid", "synthetic")
    instance._cached_user_id = "viewer"
    if isinstance(instance, EmbyClient):
        instance.product_kind = "emby"
    yield instance
    instance.close()


def payload(count=137, size=8, start=0):
    return {
        "Items": [{"Id": f"item-{i}", "Name": f"Unrelated title {i}", "Type": "Movie"} for i in range(start, start + size)],
        "TotalRecordCount": count,
    }


def source(result, server_type="jellyfin", error=""):
    return {
        "server_type": server_type, "server_name": server_type,
        "web_url": "http://synthetic.invalid", "items": result.items,
        **result.metadata(), "error": error,
    }


def library_result(monkeypatch, sources, **options):
    monkeypatch.setattr(library_search, "_search_sources", lambda *_args, **_kwargs: sources)
    return library_search.search_library({"query": "Sample", "limit": 8, **options})


def test_search_50_is_a_page_not_the_total(client):
    with patch.object(client, "_request", return_value=payload(size=50)) as request:
        result = client.search_media("Sample", limit=50)
    assert result.total == 137
    assert result.metadata()["returned"] == 50
    assert result.has_more is True
    assert result.next_offset == 50
    params = request.call_args.kwargs["params"]
    assert params["SearchTerm"] == "Sample"
    assert params["IncludeItemTypes"] == "Movie,Series,Episode"
    assert params["Limit"] == 50
    assert params["EnableTotalRecordCount"] == "true"
    assert params["StartIndex"] == 0


@pytest.mark.parametrize("offset,size,next_offset,has_more", [(50, 50, 100, True), (100, 37, None, False), (200, 0, None, False)])
def test_native_offset_and_terminal_page(client, offset, size, next_offset, has_more):
    with patch.object(client, "_request", return_value=payload(size=size, start=offset)) as request:
        result = client.search_media("Sample", limit=50, offset=offset)
    assert result.total == 137
    assert result.has_more is has_more
    assert result.next_offset == next_offset
    assert request.call_args.kwargs["params"]["StartIndex"] == offset


@pytest.mark.parametrize("count", [None, True, False, -1, 0, 1.5, "bad", "-1"])
def test_bad_or_contradictory_count_is_unknown_not_len(client, count):
    data = payload(count=count)
    if count is None:
        del data["TotalRecordCount"]
    with patch.object(client, "_request", return_value=data):
        result = client.search_media("Sample", limit=8)
    assert result.total is None
    assert result.has_more is None
    assert result.metadata()["returned"] == 8
    assert result.next_offset == 8
    assert result.total_source == "unknown"


def test_numeric_record_count_is_accepted(client):
    with patch.object(client, "_request", return_value=payload(count="137")):
        assert client.search_media("Sample").total == 137


def test_empty_page_without_total_is_not_a_confirmed_zero(client):
    with patch.object(client, "_request", return_value={"Items": []}):
        result = client.search_media("Sample")
    assert result.total is None
    assert result.has_more is None


def test_count_only_uses_same_filter_and_native_zero_item_limit(client):
    with patch.object(client, "_request", return_value=payload(size=0)) as request:
        result = client.search_media("Sample", media_type="movie", count_only=True)
    assert result.total == 137
    assert result.items == []
    assert result.next_offset is None
    assert request.call_count == 1
    assert request.call_args.kwargs["params"]["Limit"] == 0
    assert request.call_args.kwargs["params"]["IncludeItemTypes"] == "Movie"
    assert request.call_args.kwargs["params"]["SearchTerm"] == "Sample"


@pytest.mark.parametrize("count_only", [False, True])
def test_actor_uses_exact_person_role_not_title_and_excludes_episodes(client, count_only):
    people = {"Items": [{"Id": "person-1", "Name": "Sample Performer"}], "TotalRecordCount": 1}
    with patch.object(client, "_request", side_effect=[people, payload(size=0 if count_only else 8)]) as request:
        result = client.search_media("Sample Performer", search_by="actor", count_only=count_only)
    assert result.total == 137
    assert result.person_match == "exact"
    assert result.metadata()["match_basis"] == "actor_metadata"
    assert result.metadata()["included_item_types"] == ["movie", "series"]
    first, second = request.call_args_list
    assert first.args == ("/Persons",)
    assert first.kwargs["params"]["UserId"] == "viewer"
    params = second.kwargs["params"]
    assert "SearchTerm" not in params
    assert "Person" not in params
    assert params["PersonIds"] == "person-1"
    assert params["PersonTypes"] == "Actor"
    assert params["IncludeItemTypes"] == "Movie,Series"
    assert params["Limit"] == (0 if count_only else 12)
    if not count_only:
        assert all("Sample Performer" not in item.name for item in result.items)


@pytest.mark.parametrize("media_type,item_type", [("movie", "Movie"), ("series", "Series"), ("episode", "Episode")])
def test_actor_count_can_be_scoped_by_record_type(client, media_type, item_type):
    people = {"Items": [{"Id": "person-1", "Name": "Sample"}], "TotalRecordCount": 1}
    with patch.object(client, "_request", side_effect=[people, payload(size=0)]) as request:
        result = client.search_media("Sample", search_by="actor", media_type=media_type, count_only=True)
    assert result.total == 137
    assert request.call_args.kwargs["params"]["IncludeItemTypes"] == item_type


@pytest.mark.parametrize("people,expected", [
    ({"Items": [{"Id": "a", "Name": "Sample"}, {"Id": "b", "Name": "SAMPLE"}], "TotalRecordCount": 2}, "ambiguous"),
    ({"Items": [{"Id": "a", "Name": "Sample"}], "TotalRecordCount": 51}, "incomplete"),
    ({"Items": [{"Id": "a", "Name": "Sample"}]}, "incomplete"),
    ({"Items": [{"Name": "Sample"}], "TotalRecordCount": 1}, "incomplete"),
])
def test_actor_resolution_must_be_complete_and_unambiguous(client, people, expected):
    with patch.object(client, "_request", return_value=people) as request:
        result = client.search_media("Sample", search_by="actor")
    assert request.call_count == 1
    assert result.total is None
    assert result.person_match == expected
    assert result.has_more is None


def test_fuzzy_person_name_is_not_silently_selected(client):
    people = {"Items": [{"Id": "a", "Name": "Sample Other"}], "TotalRecordCount": 1}
    with patch.object(client, "_request", return_value=people) as request:
        result = client.search_media("Sample", search_by="actor")
    assert request.call_count == 1
    assert result.total is None
    assert result.person_match == "not_found"
    assert result.total_source == "unknown"


def test_bad_rows_do_not_break_backend_cursor(client):
    data = payload(count=137, size=1)
    data["Items"].insert(0, None)
    with patch.object(client, "_request", return_value=data):
        result = client.search_media("Sample")
    assert result.metadata()["returned"] == 1
    assert result.next_offset == 2
    assert result.total == 137


def test_service_transmits_counts_paging_and_closes_clients(client):
    with (
        patch("app.services._configured_media_sources", return_value=[("jellyfin", "Synthetic", client.url, client)]),
        patch.object(client, "_request", return_value=payload(size=8, start=8)) as request,
        patch.object(client, "close") as close,
    ):
        sources = services.search_media_servers("Sample", limit=8, offset=8, media_type="movie")
    assert sources[0]["total"] == 137
    assert sources[0]["returned"] == 8
    assert sources[0]["has_more"] is True
    assert sources[0]["next_offset"] == 16
    assert request.call_args.kwargs["params"]["StartIndex"] == 8
    close.assert_called_once_with()


def test_library_keeps_server_total_separate_from_returned(monkeypatch):
    result = library_result(monkeypatch, [source(MediaSearchResult(items=[MediaItem(id="one", name="One", type="Movie")], total=137, has_more=True, next_offset=1))])
    assert result.data["total"] == 137
    assert result.data["returned"] == 1
    assert result.data["has_more"] is True
    assert result.data["sources"][0]["total"] == 137
    assert "137" in result.summary


@pytest.mark.parametrize("count", [None, 137])
def test_unknown_or_partial_library_total_is_not_a_precise_total(monkeypatch, count):
    sources = [source(MediaSearchResult(items=[MediaItem(id="one", name="One", type="Movie")], total=count, has_more=True if count else None))]
    if count is not None:
        sources.append(source(MediaSearchResult(), server_type="emby", error="unavailable"))
    result = library_result(monkeypatch, sources)
    assert result.data["total"] is None
    assert result.data["known_total"] == (count or 0)
    assert result.data["returned"] == 1
    assert result.status == "partial"
    assert "总数未知" in result.summary


def test_library_multi_source_count_is_explicitly_not_deduplicated(monkeypatch):
    sources = [source(MediaSearchResult(total=137, has_more=True)), source(MediaSearchResult(total=60, has_more=True), "emby")]
    result = library_result(monkeypatch, sources, count_only=True)
    assert result.data["total"] == 197
    assert not {"returned", "has_more", "limit", "next_offset", "pagination_scope"}.intersection(result.data)
    assert result.data["count_scope"] == "server_records_sum_not_deduplicated"
    assert all(not {"returned", "items", "has_more"}.intersection(source) for source in result.data["sources"])
    assert result.data["match_status"] == "found"


@pytest.mark.parametrize("sources", [[], [source(MediaSearchResult(), error="unavailable")]])
def test_not_configured_or_unavailable_is_not_zero_results(monkeypatch, sources):
    result = library_result(monkeypatch, sources)
    assert result.data["total"] is None
    assert result.data["returned"] == 0
    assert result.data["has_more"] is None
    assert result.ok is False


def test_actor_ambiguity_is_explained_by_library(monkeypatch):
    result = library_result(monkeypatch, [source(MediaSearchResult(search_by="actor", person_match="ambiguous"))], search_by="actor", count_only=True)
    assert result.data["total"] is None
    assert result.data["match_status"] == "indeterminate"
    assert any("歧义" in suggestion for suggestion in result.suggestions)


def test_paging_filter_and_count_mode_all_partition_cache(monkeypatch):
    with library_search._search_cache_lock:
        library_search._search_cache.clear()
    search = MagicMock(return_value=[])
    monkeypatch.setattr(library_search, "search_media_servers", search)
    options = [{}, {"offset": 8}, {"search_by": "actor"}, {"media_type": "movie"}, {"count_only": True}]
    try:
        for option in options:
            library_search._search_sources("Sample", 8, **option)
            library_search._search_sources("Sample", 8, **option)
        assert search.call_count == len(options)
    finally:
        with library_search._search_cache_lock:
            library_search._search_cache.clear()


@pytest.mark.parametrize("options", [{"offset": -1}, {"offset": True}, {"offset": "8"}, {"search_by": "person"}, {"media_type": "video"}, {"count_only": 1}, {"unexpected": True}])
def test_library_rejects_invalid_search_options(options):
    with pytest.raises(AgentToolError):
        library_search.library_search_arguments({"query": "Sample", **options})


def test_library_and_provider_schemas_expose_same_search_options():
    registry = MagicMock()
    library.register_specs(registry, resource_store=None, active_ingest_store=None, ingest_actions=None)
    spec = next(call.args[0] for call in registry.register.call_args_list if call.args[0].name == "library.search")
    provider_spec = build_provider_catalog().get("media.items.search")
    for key in ("offset", "search_by", "media_type", "count_only"):
        assert spec.parameters["properties"][key] == provider_spec.parameters["properties"][key]
    assert spec.parameters["properties"]["limit"]["maximum"] == 50
    assert provider_spec.parameters["properties"]["limit"]["maximum"] == 20
    assert spec.validator({"query": "  Sample  ", "offset": 8, "search_by": "actor", "count_only": True}) == {
        "query": "Sample", "limit": 8, "offset": 8, "search_by": "actor", "count_only": True,
    }


def test_provider_search_reports_native_total_not_20_item_page(client):
    profile = MediaServerProfile("configured:jellyfin", "jellyfin", "Synthetic", client.url, "synthetic", True)
    transport = MediaServerProviderTransport()
    with (
        patch.object(transport, "_profile", return_value=profile),
        patch.object(transport, "_client", return_value=client),
        patch.object(client, "_request", return_value=payload(size=20, start=20)) as request,
        patch.object(client, "close") as close,
    ):
        result = transport.execute_read(profile.source, "media.items.search", {"query": "Sample", "limit": 20, "offset": 20})
    assert result.data["total"] == 137
    assert result.data["returned"] == result.data["count"] == 20
    assert result.data["has_more"] is True
    assert result.data["next_offset"] == 40
    assert "137" in result.summary
    assert request.call_args.kwargs["params"]["StartIndex"] == 20
    close.assert_called_once_with()


@pytest.mark.parametrize("ambiguous", [False, True])
def test_provider_gateway_keeps_actor_statistics_and_unknowns(client, ambiguous):
    profile = MediaServerProfile("configured:jellyfin", "jellyfin", "Synthetic", client.url, "synthetic", True)
    transport = MediaServerProviderTransport()
    gateway = ProviderGateway(catalog=build_provider_catalog(), transports=[transport])
    people = [{"Id": "a", "Name": "Sample"}]
    if ambiguous:
        people.append({"Id": "b", "Name": "Sample"})
    with (
        patch.object(transport, "profiles", return_value=[ProviderProfileView(profile.source, "media", "Synthetic", "online")]),
        patch.object(transport, "_profile", return_value=profile),
        patch.object(transport, "_client", return_value=client),
        patch.object(client, "_request", side_effect=[{"Items": people, "TotalRecordCount": len(people)}, payload(size=0)]),
    ):
        result = gateway.query(
            profile_ref=profile.source, operation="media.items.search",
            arguments={"query": "Sample", "search_by": "actor", "media_type": "movie", "count_only": True},
            context=ToolContext(owner="count-test", session_id="count-test"),
        )
    assert result.data["total"] == (None if ambiguous else 137)
    assert result.data["count"] == result.data["returned"] == 0
    assert result.data["person_match"] == ("ambiguous" if ambiguous else "exact")
    assert result.data["count_only"] is True
    assert result.status == ("partial" if ambiguous else "success")


@pytest.mark.parametrize("count_only", [False, True])
def test_episode_title_empty_result_is_not_series_inventory(monkeypatch, count_only):
    result = library_result(
        monkeypatch, [source(MediaSearchResult(total=0, has_more=False))],
        media_type="episode", count_only=count_only,
    )
    assert result.status == "empty"
    assert result.data["total"] == 0
    assert result.data["match_status"] == "not_found"
    assert result.data["search_scope"] == "episode_title"
    assert "单集自身标题" in result.summary
    assert any("library.count_series_episodes" in hint for hint in result.suggestions)
    assert not any("搜索《" in hint or "在网上找" in hint for hint in result.suggestions)


def test_episode_title_found_keeps_native_results_and_scope(monkeypatch):
    result = library_result(
        monkeypatch, [source(MediaSearchResult(
            items=[MediaItem(id="ep-7", name="Sample", type="Episode", series_name="Different series")],
            total=1, has_more=False,
        ))], media_type="episode",
    )
    assert result.data["search_scope"] == "episode_title"
    assert result.data["total"] == 1
    assert result.data["sources"][0]["items"][0]["series_name"] == "Different series"
    assert "单集自身标题" in result.summary


def test_episode_actor_search_does_not_become_title_search(monkeypatch):
    result = library_result(
        monkeypatch, [source(MediaSearchResult(total=2, has_more=True, search_by="actor"))],
        media_type="episode", search_by="actor", count_only=True,
    )
    assert result.data["search_scope"] == "actor_metadata"
    assert result.data["match_basis"] == "actor_metadata"
    assert result.data["total"] == 2
    assert not any("单集自身标题" in hint for hint in result.suggestions)
