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


def source_for_both(case):
    first = case.file_read({"operation": "tree", "path": FILMS["path"],
                           "kinds": ["video"], "max_items": 7, "max_depth": 2})
    second = case.file_read({"operation": "search", "path": SHOWS["path"], "query": "Episode",
                            "kinds": ["video"], "max_items": 9, "max_depth": 1})
    handles = [first.data["entries"][0]["object_ref"], second.data["entries"][0]["object_ref"]]
    merged = workspace.load_directory_observation_set(second.data["observation_ref"],
        owner=case.context.owner, object_refs=handles)
    source = actions.bind_observation_source(workspace.capture_observation_source(merged, handles), case.context)
    for result in (first, second):
        workspace.discard_observation(result.data["observation_ref"])
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
