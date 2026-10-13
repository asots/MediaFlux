"""多查询重新观察的合并、局部失败和自包含来源；离线 Provider，不是生产实测。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from app.agent import guangya_workspace_actions as actions
from app.agent.errors import AgentToolError
from app.modules import guangya_workspace as workspace
from tests.test_agent_reobservation_queries import assert_private, case as case
from tests.test_guangya_observation_sources import FILMS, SHOWS


def source_for_both(case, *, select_scope_roots=False, mode="deleted"):
    first = case.file_read({"operation": "tree", "path": FILMS["path"],
                           "kinds": ["video"], "max_items": 7, "max_depth": 2})
    second = case.file_read({"operation": "search", "path": SHOWS["path"], "query": "Episode",
                            "kinds": ["video"], "max_items": 9, "max_depth": 1})
    selection = "scope_objects" if select_scope_roots else "entries"
    handles = [first.data[selection][0]["object_ref"], second.data[selection][0]["object_ref"]]
    merged = workspace.load_directory_observation_set(second.data["observation_ref"],
        owner=case.context.owner, object_refs=handles)
    source = actions.bind_observation_source(workspace.capture_observation_source(merged, handles), case.context)
    for result in (first, second):
        case.invalidate(result.data["observation_ref"], mode)
    return source


def test_one_missing_root_is_unknown_while_other_original_scope_remains_usable(case):
    source = source_for_both(case)
    case.provider.live_file(FILMS["file_id"]).file_id = "private-provider-replacement-root"
    case.provider.trace.clear()
    result = case.file_read({"operation": "reobserve"}, source=source)
    compared = result.data["reobservation"]
    assert not compared["complete"]
    assert [item["state"] for item in compared["items"]] == ["inconclusive", "same_identity"]
    assert len(compared["failed_sources"]) == 1
    assert "未能核验" in result.summary and "不能判断其中对象不存在" in result.summary
    assert [item["object_name"] for item in result.data["entries"]] == ["Episode.mkv"]
    assert not any(item["method"] == "list_dir" and item["id"] in
                   {FILMS["file_id"], "private-provider-replacement-root"} for item in case.provider.trace)
    assert_private(result)
    assert not case.provider.writes


def test_merged_reobservation_does_not_depend_on_expiring_intermediate_observations(case):
    source = source_for_both(case)
    result = case.file_read({"operation": "reobserve"}, source=source)
    assert result.data["reobservation"]["complete"] is True
    payload = workspace.load_directory_observation(result.data["observation_ref"], owner=case.context.owner)
    descriptors = workspace.observation_sources(payload)
    assert {(item["operation"], item["max_items"], item["max_depth"], item["query"]) for item in descriptors} == {
        ("tree", 7, 2, ""), ("search", 9, 1, "Episode"),
    }
    assert all(not workspace._plan_path(workspace.observation_plan_id(item["observation_ref"])).exists()
               for item in descriptors), "中间观察应清理，最终签名快照必须自包含"
    handles = [item["object_ref"] for item in result.data["reobservation"]["items"]]
    next_source = actions.bind_observation_source(workspace.capture_observation_source(payload, handles), case.context)
    workspace.discard_observation(result.data["observation_ref"])
    again = case.file_read({"operation": "reobserve"}, source=next_source)
    assert again.data["reobservation"]["complete"] is True
    assert len(again.data["entries"]) == 2
    assert_private(again)


@pytest.mark.parametrize("changed", [False, True])
def test_overlapping_observations_deduplicate_only_consistent_object_versions(case, changed):
    first = case.file_read()
    old = workspace.load_directory_observation(first.data["observation_ref"], owner=case.context.owner)
    if changed:
        case.provider.live_file("private-provider-movie").size += 1
    second = case.file_read()
    new = workspace.load_directory_observation(second.data["observation_ref"], owner=case.context.owner)
    if changed:
        with pytest.raises(workspace.GuangYaWorkspaceStale, match="同一对象发生变化"):
            workspace.merge_directory_observations(case.provider, owner=case.context.owner, payloads=[old, new])
    else:
        merged = workspace.merge_directory_observations(case.provider, owner=case.context.owner, payloads=[old, new])
        assert len(merged["entries"]) == len(old["entries"])
        assert len({item["handle"] for item in merged["entries"]}) == len(merged["entries"])
    assert not case.provider.writes


def test_malformed_partial_roots_do_not_become_a_smaller_claimed_complete_source(case):
    read = case.file_read()
    payload = workspace.load_directory_observation(read.data["observation_ref"], owner=case.context.owner)
    payload["roots"] = [deepcopy(FILMS), "invalid"]
    source = workspace.capture_observation_source(payload, [read.data["entries"][0]["object_ref"]])
    assert source["available"] is False and source["read"] is None


def test_same_basename_directories_and_files_keep_distinct_public_locations(case):
    case.provider.live_file("private-provider-movie").name = "Same.mkv"
    case.provider.live_file("private-provider-episode").name = "Same.mkv"
    reads = [case.file_read({"operation": "list", "path": root["path"], "kinds": ["video"]})
             for root in (FILMS, SHOWS)]
    handles = [read.data["entries"][0]["object_ref"] for read in reads]
    payload = workspace.load_directory_observation_set(reads[-1].data["observation_ref"],
        owner=case.context.owner, object_refs=handles)
    source = actions.bind_observation_source(workspace.capture_observation_source(payload, handles), case.context)
    result = case.file_read({"operation": "reobserve"}, source=source)
    assert result.data["reobservation"]["complete"]
    assert {item["object_name"] for item in result.data["entries"]} == {"Same.mkv"}
    assert len({item["location"] for item in result.data["entries"]}) == 2
    assert len({item["at"] for item in result.model_data["entries"]}) == 2
    assert_private(result)


@pytest.mark.parametrize("stop_after", [0, 1, 2])
def test_cancellation_stops_additional_sources_and_discards_unpublished_observations(case, monkeypatch, stop_after):
    source = source_for_both(case)
    original = actions._create_observation
    called = []
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        called.append(result["plan_id"])
        return result
    monkeypatch.setattr(actions, "_create_observation", observe)
    context = replace(case.context, reobservation_source=source, cancelled=lambda: len(called) >= stop_after)
    with pytest.raises(AgentToolError) as caught:
        case.file_read({"operation": "reobserve"}, context=context)
    assert caught.value.code == "stopped"
    assert len(called) == stop_after
    assert all(not workspace._plan_path(plan_id).exists() for plan_id in called)
    assert not case.provider.writes


def test_selected_identity_survives_the_combined_observation_capacity_limit(case):
    from app.clients.guangya import GuangYaFile

    original = case.file_read({"operation": "list", "path": FILMS["path"], "kinds": ["video"], "max_items": 2000})
    source, _ = case.file_prepare(original)
    target = case.provider.live_file("private-provider-movie")
    case.provider.directories[FILMS["file_id"]] = [
        GuangYaFile(f"private-provider-new-{index}", f"Other-{index:04}.mkv", False,
                    parent_id=FILMS["file_id"], size=1) for index in range(2000)
    ] + [target]
    workspace.discard_observation(original.data["observation_ref"])
    result = case.file_read({"operation": "reobserve"}, source=source)
    assert result.data["truncated"] is True  # 不能冒称整个目录已枚举完整。
    assert result.data["reobservation"]["complete"] is True  # 但原选中对象已单独核验。
    match = result.data["reobservation"]["items"][0]
    assert match["observed_by"] == "identity"
    saved = workspace.load_directory_observation(result.data["observation_ref"], owner=case.context.owner)
    assert len(saved["entries"]) == 2000
    assert workspace.observation_entry_map(saved)[match["object_ref"]]["file_id"] == target.file_id
    assert_private(result)
    assert not case.provider.writes


@pytest.mark.parametrize("mode", ["deleted", "expired"])
@pytest.mark.parametrize("operation", ["list", "search"])
def test_selected_scope_root_reobserves_after_original_snapshot_is_unusable(case, mode, operation):
    arguments = {"operation": operation, "path": FILMS["path"], "kinds": ["video"], "max_items": 17}
    if operation == "search":
        arguments.update(query="Movie", max_depth=2)
    initial = case.file_read(arguments)
    root_ref = initial.data["scope_objects"][0]["object_ref"]
    assert root_ref not in {item["object_ref"] for item in initial.data["entries"]}
    source, fingerprint = case.file_prepare(initial, refs=[root_ref])
    assert source["baseline"]["selected"][0]["file_id"] == FILMS["file_id"]
    frozen = deepcopy(source)
    old_path = case.invalidate(initial.data["observation_ref"], mode)
    with pytest.raises(workspace.GuangYaWorkspaceError):
        workspace.load_directory_observation(initial.data["observation_ref"], owner=case.context.owner)
    case.provider.live_file(FILMS["file_id"]).name = "Shared-Renamed"
    case.provider.trace.clear()

    result = case.file_read({"operation": "reobserve"}, source=source)
    comparison = result.data["reobservation"]
    assert comparison["complete"] is True and len(comparison["items"]) == 1
    match = comparison["items"][0]
    new_ref = result.data["scope_objects"][0]["object_ref"]
    assert match["state"] == "same_identity" and "name" in match["changed_fields"]
    assert match["previous_ref"] == root_ref and match["object_ref"] == new_ref != root_ref
    assert result.data["scope_objects"][0]["object_name"] == "Shared-Renamed"
    assert result.data["total"] == 1 and all(item["kind"] == "video" for item in result.data["entries"])
    assert new_ref not in {item["object_ref"] for item in result.data["entries"]}
    saved = workspace.load_directory_observation(result.data["observation_ref"], owner=case.context.owner)
    assert workspace.observation_entry_map(saved)[new_ref]["file_id"] == FILMS["file_id"]
    assert len([row for row in case.provider.trace if row.get("method") == "file_info"
                and row.get("id") == FILMS["file_id"]]) == 1, "查询根不应遗漏后再走父链补查"
    assert not old_path.exists()
    new_source, new_fingerprint = case.file_prepare(result, refs=[new_ref], source=source)
    assert new_fingerprint != fingerprint
    assert new_source["baseline"]["selected"][0]["old_object_ref"] == new_ref
    assert source == frozen and not case.provider.writes
    assert_private(initial)
    assert_private(result)


@pytest.mark.parametrize("mode", ["deleted", "expired"])
def test_selected_scope_roots_merge_and_reobserve_again_without_intermediate_snapshots(case, mode):
    source = source_for_both(case, select_scope_roots=True, mode=mode)
    frozen = deepcopy(source)
    expected_ids = {FILMS["file_id"], SHOWS["file_id"]}
    selected = source["baseline"]["selected"]
    assert {item["file_id"] for item in selected} == expected_ids
    assert len({item["source_observation_ref"] for item in selected}) == 2
    case.provider.trace.clear()
    result = case.file_read({"operation": "reobserve"}, source=source)
    assert result.data["operation"] == result.model_data["operation"] == "reobserve"
    assert result.data["reobservation"]["complete"] is True
    new_refs = [item["object_ref"] for item in result.data["scope_objects"]]
    assert len(new_refs) == 2 and result.data["total"] == 2
    assert {item["object_ref"] for item in result.data["reobservation"]["items"]} == set(new_refs)
    assert not set(new_refs) & {item["object_ref"] for item in result.data["entries"]}
    assert set(new_refs).isdisjoint(item["old_object_ref"] for item in selected)
    saved = workspace.load_directory_observation(result.data["observation_ref"], owner=case.context.owner)
    assert {workspace.observation_entry_map(saved)[ref]["file_id"] for ref in new_refs} == expected_ids
    descriptors = workspace.observation_sources(saved)
    assert {(item["operation"], item["max_items"], item["max_depth"], item["query"]) for item in descriptors} == {
        ("tree", 7, 2, ""), ("search", 9, 1, "Episode"),
    }
    assert all(not workspace._plan_path(workspace.observation_plan_id(item["observation_ref"])).exists()
               for item in descriptors), "合并根的 OBJ 与私有来源不能依赖已删除的中间 OBS"
    next_source, _ = case.file_prepare(result, refs=new_refs, source=source)
    assert {item["source_observation_ref"] for item in next_source["baseline"]["selected"]} <= {
        item["observation_ref"] for item in descriptors
    }
    next_frozen = deepcopy(next_source)
    case.invalidate(result.data["observation_ref"], mode)
    case.provider.live_file(FILMS["file_id"]).name = "Shared-Renamed"
    again = case.file_read({"operation": "reobserve"}, source=next_source)
    assert again.data["reobservation"]["complete"] is True
    assert len(again.data["scope_objects"]) == 2 and again.data["total"] == 2
    assert {item["previous_ref"] for item in again.data["reobservation"]["items"]} == set(new_refs)
    assert source == frozen and next_source == next_frozen and not case.provider.writes
    assert_private(result)
    assert_private(again)


def test_replaced_scope_root_is_unknown_without_losing_the_other_merged_scope(case):
    source = source_for_both(case, select_scope_roots=True, mode="expired")
    frozen = deepcopy(source)
    replacement_id = "private-provider-replacement-root"
    case.provider.live_file(FILMS["file_id"]).file_id = replacement_id
    case.provider.directories[replacement_id] = []
    case.provider.trace.clear()
    result = case.file_read({"operation": "reobserve"}, source=source)
    comparison = result.data["reobservation"]
    assert comparison["complete"] is False and len(comparison["failed_sources"]) == 1
    assert [item["state"] for item in comparison["items"]] == ["inconclusive", "same_identity"]
    assert "object_ref" not in comparison["items"][0]
    assert len(result.data["scope_objects"]) == 1 and result.data["total"] == 1
    saved = workspace.load_directory_observation(result.data["observation_ref"], owner=case.context.owner)
    surviving_ref = comparison["items"][1]["object_ref"]
    assert result.data["scope_objects"][0]["object_ref"] == surviving_ref
    assert workspace.observation_entry_map(saved)[surviving_ref]["file_id"] == SHOWS["file_id"]
    assert not any(row.get("method") == "list_dir" and row.get("id") in {FILMS["file_id"], replacement_id}
                   for row in case.provider.trace)
    assert source == frozen and not case.provider.writes
    assert_private(result)


@pytest.mark.parametrize("mode", ["deleted", "expired"])
@pytest.mark.parametrize("replacement_at", ["before_reobserve", "after_identity_read"])
def test_same_name_replacement_cannot_reobserve_or_reuse_selected_scope_ref(case, monkeypatch, mode, replacement_at):
    initial = case.file_read({"operation": "list", "path": FILMS["path"], "kinds": ["video"]})
    root_ref = initial.data["scope_objects"][0]["object_ref"]
    source, _ = case.file_prepare(initial, refs=[root_ref])
    frozen = deepcopy(source)
    case.invalidate(initial.data["observation_ref"], mode)
    replacement_id = "private-provider-replacement-scope"

    def replace_root():
        case.provider.live_file(FILMS["file_id"]).file_id = replacement_id
        case.provider.directories[replacement_id] = []

    if replacement_at == "before_reobserve":
        replace_root()
    else:
        original_info = case.provider.file_info

        def file_info(file_id):
            current = original_info(file_id)
            if str(file_id) == FILMS["file_id"]:
                replace_root()
            return current

        monkeypatch.setattr(case.provider, "file_info", file_info)
    case.provider.trace.clear()
    with pytest.raises(AgentToolError) as caught:
        case.file_read({"operation": "reobserve"}, source=source)
    assert caught.value.code == ("observation_scope_changed" if replacement_at == "before_reobserve"
                                 else "precondition_failed")
    assert not any(row.get("method") == "list_dir" and row.get("id") in {FILMS["file_id"], replacement_id}
                   for row in case.provider.trace), "身份检查与路径解析间发生替换，也不能扫描同名新根"
    fresh = case.file_read({"operation": "list", "path": FILMS["path"]})
    assert fresh.data["scope_objects"][0]["object_ref"] != root_ref
    with pytest.raises(AgentToolError, match="分享对象不属于当前目录快照"):
        case.file_prepare(fresh, refs=[root_ref], source=source)
    assert source == frozen and not case.provider.writes
    assert_private(fresh)


@pytest.mark.parametrize("field", ["size", "etag", "updated_at"])
def test_scope_root_version_conflict_is_rejected_even_when_children_are_unchanged(case, field):
    first = case.file_read({"operation": "list", "path": FILMS["path"], "kinds": ["video"]})
    old = workspace.load_directory_observation(first.data["observation_ref"], owner=case.context.owner)
    root = case.provider.live_file(FILMS["file_id"])
    setattr(root, field, "changed-scope-etag" if field == "etag" else getattr(root, field) + 1)
    second = case.file_read({"operation": "list", "path": FILMS["path"], "kinds": ["video"]})
    new = workspace.load_directory_observation(second.data["observation_ref"], owner=case.context.owner)
    assert old["entries"][0]["file_id"] == new["entries"][0]["file_id"]
    with pytest.raises(workspace.GuangYaWorkspaceStale, match="同一对象发生变化"):
        workspace.merge_directory_observations(case.provider, owner=case.context.owner, payloads=[old, new])
    assert not case.provider.writes


def test_overlapping_scope_root_and_child_entry_share_one_handle_and_one_comparison(case):
    first = case.file_read({"operation": "list", "path": "/Films", "kinds": ["directory"]})
    second = case.file_read({"operation": "list", "path": FILMS["path"], "kinds": ["video"]})
    payloads = [workspace.load_directory_observation(read.data["observation_ref"], owner=case.context.owner)
                for read in (first, second)]
    merged = workspace.merge_directory_observations(case.provider, owner=case.context.owner, payloads=payloads)
    scope = next(item for item in merged["roots"] if item["file_id"] == FILMS["file_id"])
    child = next(item for item in merged["entries"] if item["file_id"] == FILMS["file_id"])
    assert scope["handle"] == child["handle"]
    assert len([item for item in workspace.observation_entry_map(merged).values()
                if item["file_id"] == FILMS["file_id"]]) == 1
    initial = case.file_read({"observation_ref": workspace.observation_ref(merged["plan_id"])})
    source, _ = case.file_prepare(initial, refs=[scope["handle"]])
    case.invalidate(initial.data["observation_ref"], "expired")
    result = case.file_read({"operation": "reobserve"}, source=source)
    assert result.data["reobservation"]["complete"] is True
    assert len(result.data["reobservation"]["items"]) == 1
    assert result.data["reobservation"]["items"][0]["state"] == "same_identity"
    assert not case.provider.writes
    assert_private(result)


@pytest.mark.parametrize("query_kind", ["paths", "merged"])
def test_same_basename_scope_objects_keep_distinct_public_and_model_locations(case, query_kind):
    if query_kind == "paths":
        result = case.file_read({"operation": "list", "paths": [FILMS["path"], SHOWS["path"]], "kinds": ["video"]})
    else:
        source = source_for_both(case, select_scope_roots=True)
        result = case.file_read({"operation": "reobserve"}, source=source)
    scopes = result.data["scope_objects"]
    assert len(scopes) == 2 and {item["object_name"] for item in scopes} == {"Shared"}
    assert len({item["object_ref"] for item in scopes}) == 2
    assert len({item["location"] for item in scopes}) == 2, "同名查询根必须能按公开位置区分，不能只剩不同 OBJ"
    assert len({item["at"] for item in result.model_data["scope_objects"]}) == 2
    assert result.model_data["operation"] == result.data["operation"]
    assert_private(result)
    assert not case.provider.writes
