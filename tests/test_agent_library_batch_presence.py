from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from app.agent.domains.library.batch_presence import (
    batch_library_presence,
    batch_presence_arguments,
)
from app.agent.errors import AgentToolError
from app.clients.base import MediaIdentityCandidate, MediaIdentityInventory
from app.services import (
    inspect_media_identity_batch,
    summarize_media_identity_batch,
)

_ITEMS = [
    {"tmdb_id": "77", "media_type": "movie", "title": "记忆碎片", "year": "2000"},
    {"tmdb_id": "157336", "media_type": "movie", "title": "星际穿越", "year": "2014"},
    {"tmdb_id": "872585", "media_type": "movie", "title": "奥本海默", "year": "2023"},
]


class _FakeMediaClient:
    def __init__(self):
        self.calls = 0

    def list_media_identity_inventory(self, media_type, **kwargs):
        self.calls += 1
        assert media_type == "movie"
        assert kwargs == {"max_items": 5000, "page_size": 200}
        return MediaIdentityInventory(
            candidates=[
                MediaIdentityCandidate(
                    name="记忆碎片", year="2000", tmdb_id="77", media_type="movie"
                ),
                MediaIdentityCandidate(
                    name="星际穿越", year="2014", tmdb_id="", media_type="movie"
                ),
            ],
            total=2,
            unmapped=1,
        )


def test_service_batch_presence_enumerates_library_once_per_media_type():
    client = _FakeMediaClient()
    with patch("app.services._configured_media_sources", return_value=[("jellyfin", "Jellyfin", "", client)]):
        result = inspect_media_identity_batch(_ITEMS)[0]

    assert client.calls == 1
    assert [item["status"] for item in result["items"]] == [
        "present",
        "possible",
        "missing",
    ]


def test_identity_snapshot_reuses_each_server_and_type_only_within_one_request():
    from app import services

    movie = MediaIdentityInventory(candidates=[], total=0)
    tv = MediaIdentityInventory(candidates=[], total=0)
    client = Mock()
    client.list_media_identity_inventory.side_effect = lambda media_type, **_: movie if media_type == "movie" else tv
    with patch("app.services._configured_media_sources", return_value=[("jellyfin", "Jellyfin", "", client)]):
        snapshot = services.load_media_identity_snapshot(("movie", "tv"))
        for number in (1, 2):
            identities = [{"tmdb_id": str(number), "media_type": kind, "title": "作品"} for kind in ("movie", "tv")]
            sources = services.inspect_media_identity_batch(identities, snapshot=snapshot)
            assert [item["status"] for item in sources[0]["items"]] == ["missing", "missing"]
        assert client.list_media_identity_inventory.call_count == 2
        movie.candidates.append(MediaIdentityCandidate(name="作品", tmdb_id="1", media_type="movie"))
        next_snapshot = services.load_media_identity_snapshot(("movie",))
        current = services.inspect_media_identity_batch([{"tmdb_id": "1", "media_type": "movie"}], snapshot=next_snapshot)
        assert current[0]["items"][0]["status"] == "present"
        assert client.list_media_identity_inventory.call_count == 3
        # 旧请求继续使用原快照，不能被下一次枚举的可变列表偷偷更新。
        original = services.inspect_media_identity_batch([{"tmdb_id": "1", "media_type": "movie"}], snapshot=snapshot)
        assert original[0]["items"][0]["status"] == "missing"


def test_identity_snapshot_enumerates_each_server_and_type_once_and_closes_clients():
    from app import services

    clients = [Mock(), Mock()]
    for client in clients:
        client.list_media_identity_inventory.return_value = MediaIdentityInventory(candidates=[], total=0)
    configured = [("emby", "Emby", "", clients[0]), ("jellyfin", "Jellyfin", "", clients[1])]
    with patch("app.services._configured_media_sources", return_value=configured):
        snapshot = services.load_media_identity_snapshot(("movie", "tv", "movie"))
    for number in (1, 2):
        identities = [{"tmdb_id": str(number), "media_type": kind} for kind in ("movie", "tv")]
        sources = services.inspect_media_identity_batch(identities, snapshot=snapshot)
        assert services.summarize_media_identity_batch(identities, sources)["counts"]["missing"] == 2
    for client in clients:
        assert [call.args[0] for call in client.list_media_identity_inventory.call_args_list] == ["movie", "tv"]
        client.close.assert_called_once_with()


def test_batch_presence_aggregates_all_sources_without_per_title_search():
    sources = [
        {
            "server_type": "jellyfin",
            "server_name": "Jellyfin",
            "status": "ready",
            "inventories": {"movie": {"total": 100, "truncated": False, "unmapped": 0}},
            "items": [
                {"tmdb_id": "77", "media_type": "movie", "status": "present", "match": "provider_id"},
                {"tmdb_id": "157336", "media_type": "movie", "status": "missing", "match": "none"},
                {"tmdb_id": "872585", "media_type": "movie", "status": "missing", "match": "none"},
            ],
        }
    ]
    with patch(
        "app.agent.domains.library.batch_presence.inspect_media_identity_batch",
        return_value=sources,
    ) as inspect, patch(
        "app.agent.domains.library.batch_presence.summarize_media_identity_batch",
        wraps=summarize_media_identity_batch,
    ) as summarize:
        result = batch_library_presence({"items": _ITEMS})

    inspect.assert_called_once_with(_ITEMS)
    summarize.assert_called_once_with(_ITEMS, sources)
    assert result.ok
    assert result.data["counts"] == {
        "present": 1,
        "possible": 0,
        "missing": 2,
        "indeterminate": 0,
    }
    assert [item["library_status"] for item in result.data["items"]] == [
        "present",
        "missing",
        "missing",
    ]


def test_batch_presence_arguments_reject_unbounded_or_invalid_identity_lists():
    with pytest.raises(AgentToolError):
        batch_presence_arguments({"items": []})
    with pytest.raises(AgentToolError):
        batch_presence_arguments(
            {"items": [{"tmdb_id": "0", "media_type": "movie", "title": "X"}]}
        )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (("present", "present"), "present"),
        (("present", "possible"), "present"),
        (("present", "indeterminate"), "present"),
        (("present", "missing"), "present"),
        (("possible", "possible"), "possible"),
        (("possible", "indeterminate"), "possible"),
        (("possible", "missing"), "possible"),
        (("indeterminate", "indeterminate"), "indeterminate"),
        (("indeterminate", "missing"), "indeterminate"),
        (("missing", "missing"), "missing"),
    ],
)
@pytest.mark.parametrize("reverse_sources", [False, True])
def test_summarize_batch_status_priority(statuses, expected, reverse_sources):
    if reverse_sources:
        statuses = statuses[::-1]
    sources = [
        {
            "server_type": "jellyfin",
            "server_name": str(index),
            "status": "ready",
            "items": [{**_ITEMS[0], "status": status}],
        }
        for index, status in enumerate(statuses)
    ]

    result = summarize_media_identity_batch(_ITEMS[:1], sources)

    assert result["counts"] == {
        "present": int(expected == "present"),
        "possible": int(expected == "possible"),
        "indeterminate": int(expected == "indeterminate"),
        "missing": int(expected == "missing"),
    }
    assert result["items"][0]["library_status"] == expected
    assert [source["status"] for source in result["items"][0]["sources"]] == list(statuses)


def test_summarize_batch_preserves_public_payload_without_fetching_or_mutating():
    identities = [
        _ITEMS[2],
        {"tmdb_id": "77", "media_type": "tv", "title": "同 ID 剧集", "year": "2000"},
        _ITEMS[0],
        {**_ITEMS[1], "note": "保留身份字段"},
    ]
    inventories = {
        "movie": {"total": 100, "truncated": False, "unmapped": 1},
        "tv": {"total": 20, "truncated": True, "unmapped": 0},
    }
    sources = [{
        "server_type": "jellyfin",
        "server_name": 123,
        "status": "ready",
        "inventories": inventories,
        "web_url": "https://synthetic.invalid",
        "error": "synthetic private detail",
        "items": [
            None,
            "ignored",
            {**_ITEMS[0], "status": "present", "match": "provider_id", "local_id": "private"},
            {**_ITEMS[1], "status": "possible", "match": "title_year"},
            {**_ITEMS[2], "status": "missing", "match": ""},
            {"tmdb_id": "77", "media_type": "tv", "status": "unknown", "match": None},
        ],
    }]
    before = deepcopy((identities, sources))
    with patch(
        "app.services.inspect_media_identity_batch", side_effect=AssertionError("不应核对")
    ), patch(
        "app.services._configured_media_sources", side_effect=AssertionError("不应扫描")
    ):
        result = summarize_media_identity_batch(identities, sources)

    assert result == {
        "total": 4,
        "counts": {"present": 1, "possible": 1, "indeterminate": 1, "missing": 1},
        "items": [
            {
                **identity,
                "library_status": status,
                "sources": [{
                    "server_type": "jellyfin",
                    "server_name": "123",
                    "status": status,
                    "match": match,
                }],
            }
            for identity, status, match in zip(
                identities,
                ("missing", "indeterminate", "present", "possible"),
                ("none", "none", "provider_id", "title_year"),
            )
        ],
        "sources": [{
            "server_type": "jellyfin",
            "server_name": "123",
            "status": "ready",
            "inventories": inventories,
        }],
    }
    assert (identities, sources) == before
    assert result["sources"][0]["inventories"] is inventories


def test_summarize_batch_keeps_last_duplicate_source_and_item_and_inventory_defaults():
    sources = [
        {"items": [{**_ITEMS[0], "status": "present"}]},
        {
            "inventories": None,
            "items": [
                {**_ITEMS[0], "status": "possible"},
                {**_ITEMS[0], "status": "missing"},
            ],
        },
    ]

    result = summarize_media_identity_batch(_ITEMS[:1], sources)

    assert result == {
        "total": 1,
        "counts": {"present": 0, "possible": 0, "indeterminate": 0, "missing": 1},
        "items": [{
            **_ITEMS[0],
            "library_status": "missing",
            "sources": [
                {"server_type": "", "server_name": "", "status": "missing", "match": "none"},
                {"server_type": "", "server_name": "", "status": "missing", "match": "none"},
            ],
        }],
        "sources": [
            {"server_type": "", "server_name": "", "status": "unavailable", "inventories": {}},
            {"server_type": "", "server_name": "", "status": "unavailable", "inventories": None},
        ],
    }


@pytest.mark.parametrize("items", [[], [None, "ignored"], [{**_ITEMS[0]}], [
    {**_ITEMS[0], "status": "unknown"},
]])
def test_summarize_batch_missing_or_unknown_items_are_indeterminate(items):
    result = summarize_media_identity_batch(_ITEMS[:1], [{"items": items}])

    assert result["counts"]["indeterminate"] == 1
    assert result["items"][0]["library_status"] == "indeterminate"
    assert result["items"][0]["sources"] == [{
        "server_type": "", "server_name": "", "status": "indeterminate", "match": "none",
    }]


@pytest.mark.parametrize("identities", [[], _ITEMS])
def test_summarize_batch_without_sources_keeps_indeterminate_identities(identities):
    assert summarize_media_identity_batch(identities, []) == {
        "total": len(identities),
        "counts": {
            "present": 0, "possible": 0, "indeterminate": len(identities), "missing": 0,
        },
        "items": [
            {**identity, "library_status": "indeterminate", "sources": []}
            for identity in identities
        ],
        "sources": [],
    }


def test_batch_presence_without_sources_keeps_existing_early_return():
    with patch(
        "app.agent.domains.library.batch_presence.inspect_media_identity_batch", return_value=[]
    ) as inspect, patch(
        "app.agent.domains.library.batch_presence.summarize_media_identity_batch",
        side_effect=AssertionError("无媒体源时不应聚合"),
    ) as summarize:
        result = batch_library_presence({"items": _ITEMS})

    inspect.assert_called_once_with(_ITEMS)
    summarize.assert_not_called()
    assert not result.ok
    assert result.status == "not_configured"
    assert result.summary == "没有可核对的媒体服务器"
    assert result.data == {"total": 3, "items": [], "sources": []}
    assert result.suggestions == ["请先在设置中启用并配置 Jellyfin 或 Emby。"]
    assert result.error == "没有可核对的媒体服务器。"


@pytest.mark.parametrize("status", ["possible", "indeterminate"])
def test_batch_presence_keeps_partial_status_for_uncertain_items(status):
    with patch(
        "app.agent.domains.library.batch_presence.inspect_media_identity_batch",
        return_value=[{"status": "ready", "items": [{**_ITEMS[0], "status": status}]}],
    ):
        result = batch_library_presence({"items": _ITEMS[:1]})

    assert result.ok
    assert result.status == "partial"
    assert result.data["counts"][status] == 1
