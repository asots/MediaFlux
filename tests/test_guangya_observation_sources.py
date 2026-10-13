"""私有观察来源回归：离线 Provider 夹具 + 真实签名文件，不是生产实测。"""
from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
import requests

from app.clients.guangya import GuangYaFile
from app.modules import guangya_workspace as workspace
from tests.support import initialized_web_runtime

OWNER = "offline-observation-owner"
FILMS = {"file_id": "private-provider-films-shared", "path": "/Films/Shared", "is_dir": True,
         "parent_id": "private-provider-films"}
SHOWS = {"file_id": "private-provider-shows-shared", "path": "/Shows/Shared", "is_dir": True,
         "parent_id": "private-provider-shows"}
PAGE_KEYS = {"observation_ref", "scope", "scopes", "operation", "query", "kinds", "max_depth", "recursive",
             "page", "page_size", "total", "scanned_items", "has_more", "truncated", "entries", "scope_objects", "cloud_write"}
ENTRY_KEYS = {"object_ref", "object_name", "kind", "extension", "size", "location"}


class OfflineProvider:
    credential_generation = 41

    def __init__(self):
        self.calls = []
        self.directories = {
            "0": [GuangYaFile("private-provider-films", "Films", True, parent_id="0"),
                  GuangYaFile("private-provider-shows", "Shows", True, parent_id="0")],
            "private-provider-films": [GuangYaFile(FILMS["file_id"], "Shared", True, parent_id=FILMS["parent_id"])],
            "private-provider-shows": [GuangYaFile(SHOWS["file_id"], "Shared", True, parent_id=SHOWS["parent_id"])],
            FILMS["file_id"]: [GuangYaFile("private-provider-movie", "Movie.mkv", False, parent_id=FILMS["file_id"], size=100),
                               GuangYaFile("private-provider-nested", "Nested", True, parent_id=FILMS["file_id"])],
            SHOWS["file_id"]: [GuangYaFile("private-provider-episode", "Episode.mkv", False, parent_id=SHOWS["file_id"], size=80)],
            "private-provider-nested": [GuangYaFile("private-provider-subtitle", "Movie.srt", False, parent_id="private-provider-nested", size=10)],
        }

    def list_dir(self, parent_id="0"):
        self.calls.append(str(parent_id))
        return deepcopy(self.directories[str(parent_id)])


@pytest.fixture
def provider(tmp_path, monkeypatch):
    def no_live_io(*_args, **_kwargs):
        raise AssertionError("离线 Provider 夹具：不是生产实测，禁止网络 IO")

    monkeypatch.setattr(requests.sessions.Session, "request", no_live_io)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_live_io)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_live_io)
    with initialized_web_runtime():
        # 只隔离落盘目录；owner digest / HMAC / _read / _atomic_write 全部使用真实实现。
        monkeypatch.setattr(workspace, "PATHS", SimpleNamespace(data_dir=tmp_path))
        yield OfflineProvider()


def ref(payload):
    return workspace.observation_ref(payload["plan_id"])


def persisted(payload):
    loaded = workspace.load_directory_observation(ref(payload), owner=OWNER)
    assert {key: value for key, value in loaded.items() if key != "auth"} == payload
    return loaded


def assert_public_unchanged(payload):
    legacy = deepcopy(payload)
    legacy.pop("roots", None)
    legacy.pop("max_items", None)
    for entry in legacy["entries"]:
        entry.pop("source_root", None)
    public = workspace.observation_page(payload, page_size=999)
    old_public = workspace.observation_page(legacy, page_size=999)
    assert {k: v for k, v in public.items() if k != "scope_objects"} == {
        k: v for k, v in old_public.items() if k != "scope_objects"}
    assert old_public["scope_objects"] == []
    assert set(public) == PAGE_KEYS and public["page_size"] == 50
    assert all(set(entry) == ENTRY_KEYS for entry in public["entries"])
    encoded = json.dumps(public)
    assert "private-provider-" not in encoded
    assert "source_root" not in encoded and "file_id" not in encoded and "parent_id" not in encoded
    assert "/Films/Shared" not in encoded and "/Shows/Shared" not in encoded
    assert public["cloud_write"] is False
    return public


@pytest.mark.parametrize("operation", ["list", "tree", "search"])
def test_single_root_persists_resolved_identity_and_actual_query_limit(provider, operation):
    payload = workspace.create_directory_observation(provider, owner=OWNER, path="/Films/Shared",
                operation=operation, query="Movie" if operation == "search" else "", max_items=17, max_depth=2)
    loaded = persisted(payload)
    assert [{k: root[k] for k in FILMS} for root in loaded["roots"]] == [FILMS]
    assert loaded["max_items"] == 17 != len(loaded["entries"])
    assert loaded["max_depth"] == 2 and loaded["operation"] == operation
    assert all(entry["source_root"] == FILMS for entry in loaded["entries"])
    assert provider.calls == ["0", FILMS["parent_id"], FILMS["file_id"]] + (
        ["private-provider-nested"] if operation != "list" else [])
    assert_public_unchanged(loaded)


def test_same_basename_roots_and_recursive_entries_keep_distinct_origins(provider):
    payload = workspace.create_multi_directory_observation(provider, owner=OWNER,
                paths=["/Films/Shared", "/Shows/Shared", "/Films/Shared"], recursive=True, max_items=23)
    loaded = persisted(payload)
    assert loaded["scope_path"] == "/" and loaded["scope_names"] == ["目录 1 · Shared", "目录 2 · Shared"]
    assert [{k: root[k] for k in FILMS} for root in loaded["roots"]] == [FILMS, SHOWS] and loaded["max_items"] == 23
    for entry in loaded["entries"]:
        expected = SHOWS if entry["file_id"] == "private-provider-episode" else FILMS
        assert entry["source_root"] == expected
    assert next(entry for entry in loaded["entries"] if entry["name"] == "Movie.srt")["source_root"] == FILMS
    assert len(loaded["entries"]) == 4
    assert provider.calls.count(FILMS["file_id"]) == provider.calls.count(SHOWS["file_id"]) == 1
    assert_public_unchanged(loaded)


@pytest.mark.parametrize("path,expected", [
    ("/Films/Shared/Movie.mkv", {"file_id": "private-provider-movie", "path": "/Films/Shared/Movie.mkv",
                                "is_dir": False, "parent_id": FILMS["file_id"]}),
    ("/Films/Shared", FILMS),
])
def test_stat_uses_same_root_shape_and_one_item_limit(provider, path, expected):
    loaded = persisted(workspace.create_path_observation(provider, owner=OWNER, path=path))
    assert loaded["roots"] == [expected] and loaded["max_items"] == 1
    assert len(loaded["entries"]) == 1 and loaded["entries"][0]["source_root"] == expected
    assert loaded["operation"] == "stat" and loaded["recursive"] is False
    assert loaded["scanned_dirs"] == 0 and loaded["scanned_items"] == 1
    assert_public_unchanged(loaded)


def test_explicit_filesystem_root_is_not_inferred_from_display_scope(provider):
    payload = workspace.create_directory_observation(provider, owner=OWNER, path="/", max_items=9)
    root = {"file_id": "0", "path": "/", "is_dir": True}
    assert persisted(payload)["roots"] == [root]
    assert all(entry["source_root"] == root for entry in payload["entries"])
    assert provider.calls == ["0"]
    assert_public_unchanged(payload)
    with pytest.raises(workspace.GuangYaWorkspaceError):
        workspace.create_path_observation(provider, owner=OWNER, path="/")


@pytest.mark.parametrize("candidate_kind", ["directory", "stat"])
def test_merge_preserves_each_entry_signed_source_not_primary_root(provider, candidate_kind):
    primary = workspace.create_directory_observation(provider, owner=OWNER, path="/Films/Shared", max_items=1)
    if candidate_kind == "directory":
        candidate = workspace.create_directory_observation(provider, owner=OWNER, path="/Shows/Shared", max_items=7)
    else:
        candidate = workspace.create_path_observation(provider, owner=OWNER, path="/Shows/Shared/Episode.mkv")
    primary_bytes = workspace._plan_path(primary["plan_id"]).read_bytes()
    candidate_bytes = workspace._plan_path(candidate["plan_id"]).read_bytes()
    calls = list(provider.calls)
    wanted = candidate["entries"][0]
    merged = workspace.load_directory_observation_set(ref(primary), owner=OWNER, object_refs=[wanted["handle"]])
    entries = workspace.observation_entry_map(merged)
    assert len(merged["entries"]) == 2
    assert len(entries) == 3  # 两个内容对象 + 主观察的目录本身
    assert entries[primary["entries"][0]["handle"]]["source_root"] == FILMS
    assert entries[wanted["handle"]] == wanted
    assert entries[wanted["handle"]]["source_root"] == {k: candidate["roots"][0][k] for k in FILMS} != FILMS
    # 合并是读取视图，不伪造一次新查询，也不续签主/候选观察 TTL。
    assert merged["roots"] == primary["roots"] and merged["max_items"] == primary["max_items"]
    assert merged["expires_at_epoch"] == primary["expires_at_epoch"]
    assert workspace._plan_path(primary["plan_id"]).read_bytes() == primary_bytes
    assert workspace._plan_path(candidate["plan_id"]).read_bytes() == candidate_bytes
    assert provider.calls == calls
    assert_public_unchanged(merged)


@pytest.mark.parametrize("field", ["roots", "max_items", "source_root"])
def test_new_private_source_metadata_is_authenticated(provider, field):
    payload = workspace.create_directory_observation(provider, owner=OWNER, path="/Films/Shared")
    assert [{k: root[k] for k in FILMS} for root in payload["roots"]] == [FILMS]
    path = workspace._plan_path(payload["plan_id"])
    tampered = json.loads(path.read_text())
    if field == "roots":
        tampered["roots"][0]["path"] = "/"
    elif field == "max_items":
        tampered["max_items"] += 1
    else:
        tampered["entries"][0]["source_root"] = SHOWS
    path.write_text(json.dumps(tampered))
    with pytest.raises(workspace.GuangYaWorkspaceError, match="校验失败"):
        workspace.load_directory_observation(ref(payload), owner=OWNER)


@pytest.mark.parametrize("boundary", ["owner", "credential_generation", "expired"])
def test_merge_does_not_bypass_existing_source_boundaries(provider, boundary):
    primary = workspace.create_directory_observation(provider, owner=OWNER, path="/Films/Shared")
    owner = "other-owner" if boundary == "owner" else OWNER
    if boundary == "credential_generation":
        provider.credential_generation += 1
    candidate = workspace.create_directory_observation(provider, owner=owner, path="/Shows/Shared")
    if boundary == "expired":
        candidate["expires_at_epoch"] = candidate["created_at_epoch"] - 1
        workspace._atomic_write(candidate)
    if boundary == "owner":
        with pytest.raises(workspace.GuangYaWorkspaceError, match="不属于当前会话"):
            workspace.load_directory_observation(ref(candidate), owner=OWNER)
    merged = workspace.load_directory_observation_set(ref(primary), owner=OWNER,
                                                     object_refs=[candidate["entries"][0]["handle"]])
    assert merged["entries"] == primary["entries"]
    assert merged["expires_at_epoch"] == primary["expires_at_epoch"]
    assert merged["expires_at_epoch"] - merged["created_at_epoch"] == 600
    assert_public_unchanged(merged)


def make_legacy(payload):
    legacy = deepcopy(payload)
    legacy.pop("roots", None)
    legacy.pop("max_items", None)
    for entry in legacy["entries"]:
        entry.pop("source_root", None)
    workspace._atomic_write(legacy)
    return legacy


def test_legacy_multiroot_payload_remains_pageable_without_inferred_whole_disk_scope(provider):
    original = workspace.create_multi_directory_observation(provider, owner=OWNER, paths=["/Films/Shared", "/Shows/Shared"])
    legacy = make_legacy(original)
    assert legacy["scope_path"] == "/"
    loaded = workspace.load_directory_observation(ref(legacy), owner=OWNER)
    assert "roots" not in loaded and "max_items" not in loaded
    assert all("source_root" not in entry for entry in loaded["entries"])
    before = workspace._plan_path(legacy["plan_id"]).read_bytes()
    old_page = workspace.observation_page(loaded, page=2, page_size=1)
    new_page = workspace.observation_page(original, page=2, page_size=1)
    assert old_page["scope_objects"] == []
    assert {k: v for k, v in old_page.items() if k != "scope_objects"} == {
        k: v for k, v in new_page.items() if k != "scope_objects"}
    assert workspace.observation_page(loaded, page=2, page_size=1)["entries"]
    assert workspace._plan_path(legacy["plan_id"]).read_bytes() == before
    assert loaded["expires_at_epoch"] == original["expires_at_epoch"]
    assert_public_unchanged(loaded)


@pytest.mark.parametrize("legacy_primary", [False, True])
def test_merge_with_legacy_keeps_missing_source_unknown(provider, legacy_primary):
    primary = workspace.create_directory_observation(provider, owner=OWNER, path="/Films/Shared")
    candidate = workspace.create_directory_observation(provider, owner=OWNER, path="/Shows/Shared")
    if legacy_primary:
        primary = make_legacy(primary)
    else:
        candidate = make_legacy(candidate)
    wanted = candidate["entries"][0]
    merged = workspace.load_directory_observation_set(ref(primary), owner=OWNER, object_refs=[wanted["handle"]])
    entries = workspace.observation_entry_map(merged)
    assert entries[wanted["handle"]] == wanted
    if legacy_primary:
        assert "roots" not in merged and "max_items" not in merged
        assert "source_root" not in entries[primary["entries"][0]["handle"]]
        assert entries[wanted["handle"]]["source_root"] == SHOWS
    else:
        assert [{k: root[k] for k in FILMS} for root in merged["roots"]] == [FILMS]
        assert "source_root" not in entries[wanted["handle"]]
    assert_public_unchanged(merged)


@pytest.mark.parametrize("requested,expected", [(-7, 1), (2, 2), (9000, 2000)])
def test_stored_max_items_matches_existing_clamp_and_scan_page_limits(provider, requested, expected):
    provider.directories[FILMS["file_id"]] = [
        GuangYaFile(f"private-provider-item-{index}", f"Movie-{index:04d}.mkv", False, parent_id=FILMS["file_id"])
        for index in range(2005)
    ]
    payload = workspace.create_directory_observation(provider, owner=OWNER, path="/Films/Shared", max_items=requested)
    loaded = persisted(payload)
    assert loaded["max_items"] == expected
    assert len(loaded["entries"]) == expected and loaded["truncated"] is True
    assert loaded["scanned_items"] == expected + 1 and loaded["scanned_dirs"] == 1
    assert loaded["expires_at_epoch"] - loaded["created_at_epoch"] == 600
    public = assert_public_unchanged(loaded)
    assert len(public["entries"]) == min(expected, 50)
    assert public["has_more"] is (expected > 50)


@pytest.mark.parametrize("requested,expected,count", [(-3, 0, 2), (99, 12, 3)])
def test_depth_clamp_keeps_recursive_entry_bound_to_original_root(provider, requested, expected, count):
    loaded = persisted(workspace.create_directory_observation(provider, owner=OWNER, path="/Films/Shared",
                                                              recursive=True, max_depth=requested))
    assert loaded["max_depth"] == expected and loaded["truncated"] is False
    assert len(loaded["entries"]) == count
    assert all(entry["source_root"] == FILMS for entry in loaded["entries"])
    assert_public_unchanged(loaded)
