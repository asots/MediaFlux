"""整理通知的批次、文件事实和任务序号不能被历史 RSS / 完成状态替代。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app import database as db
from app.agent.errors import AgentToolError
from app.agent.local_media_task_actions import (
    _current_task,
    list_local_media_task_summaries,
    local_media_task_summaries_arguments,
    reset_local_media_agent_context_for_tests,
)
from app.agent.models import ToolContext
from app.modules.local_media_outcomes import local_media_task_outcome
from tests.support import IsolatedDatabaseTestCase


def _item(action: str, name: str = "本集.mkv") -> dict[str, str]:
    return {
        "role": "video",
        "source_path": f"/private/downloads/{name}",
        "action": action,
    }


@pytest.mark.parametrize(
    ("status", "actions", "expected", "archived", "skipped"),
    [
        ("completed", ["skip"], "conflict_skipped", 0, 1),
        ("completed", ["move"], "archived", 1, 0),
        ("completed", ["replace", "skip"], "partial", 1, 1),
        ("completed", [], "unknown", 0, 0),
        ("planned", ["move", "skip"], "pending", 0, 0),
        ("requires_manual", ["move"], "pending", 0, 0),
        ("failed", ["move"], "failed", 0, 0),
        ("completed", ["unrecognized"], "unknown", 0, 0),
    ],
)
def test_file_outcome_is_not_task_status(status, actions, expected, archived, skipped):
    result = local_media_task_outcome(
        SimpleNamespace(status=status), [_item(action) for action in actions]
    )
    assert result["file_outcome"] == expected
    assert result["archived_video_count"] == archived
    assert result["skipped_video_count"] == skipped
    assert "/private/" not in repr(result)


def test_completed_preview_only_does_not_claim_archival():
    task = SimpleNamespace(status="completed", warning="仅预览模式：未移动文件")
    result = local_media_task_outcome(task, [_item("move")])
    assert result["file_outcome"] == "preview_only"
    assert result["archived_video_count"] == 0


def test_task_outcome_only_retains_basename_and_bounded_names():
    task = SimpleNamespace(status="completed", content_path=r"C:\private\episode.mkv")
    result = local_media_task_outcome(
        task, [_item("move", f"episode-{i}.mkv") for i in range(25)]
    )
    assert result["original_filename"] == "episode.mkv"
    assert len(result["file_names"]) == 20
    assert result["file_names_truncated"]
    assert result["archived_video_count"] == 25


class LocalMediaTaskOutcomeTests(IsolatedDatabaseTestCase):
    def setUp(self):
        reset_local_media_agent_context_for_tests()
        with db.get_conn() as conn:
            conn.execute("DELETE FROM local_media_operation_steps")
            conn.execute("DELETE FROM local_media_task_items")
            conn.execute("DELETE FROM local_media_tasks")
            conn.execute("DELETE FROM local_library_targets")
            conn.execute("DELETE FROM local_media_sources")
        self.source_id = db.create_local_media_source(
            name="本地来源",
            qb_profile="qb",
            qb_path_prefix="/downloads",
            local_root="/private/downloads",
            owner="admin",
        )

    def tearDown(self):
        reset_local_media_agent_context_for_tests()

    def _task(
        self,
        *,
        status="completed",
        name="本集.mkv",
        action="skip",
        updated_at="2026-09-12 14:22:00",
    ):
        task_id = db.create_local_media_task(
            self.source_id, "", f"/private/downloads/{name}", owner="admin"
        )
        db.update_local_media_task(
            task_id,
            owner="admin",
            status=status,
            title="本片",
            error="PRIVATE-ERROR",
            completed_at=updated_at if status == "completed" else "",
        )
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE local_media_tasks SET updated_at=? WHERE id=?",
                (updated_at, task_id),
            )
        if action:
            db.add_local_media_task_item(
                task_id,
                f"/private/downloads/{name}",
                f"/private/library/{name}",
                role="video",
                action=action,
                owner="admin",
            )
        return task_id

    def _list(self, **arguments):
        return list_local_media_task_summaries(
            local_media_task_summaries_arguments(arguments),
            ToolContext(owner="owner-a"),
        )

    def _scan(self, task_ids, outcomes=None):
        return {
            "id": 900,
            "scan_ref": "LM900",
            "owner": "admin",
            "task_ids": task_ids,
            "started_at": "2026-09-12 14:21:00",
            "finished_at": "2026-09-12 14:22:00",
            "summary": {"task_outcomes": outcomes or []},
        }

    def test_list_exposes_safe_original_name_and_conflict_result(self):
        self._task()
        result = self._list(scope="history")
        item = result.data["tasks"][0]
        assert item["original_filename"] == "本集.mkv"
        assert item["status"] == "completed"
        assert item["file_outcome"] == "conflict_skipped"
        assert item["archived_video_count"] == 0
        assert item["skipped_video_count"] == 1
        assert item["updated_at"] == "2026-09-12 14:22:00"
        assert item["completed_at"] == "2026-09-12 14:22:00"
        for secret in ("PRIVATE-HASH", "PRIVATE-ERROR", "/private/", "task_id"):
            assert secret not in repr(result.data)

    def test_latest_scan_uses_only_its_members(self):
        old = self._task(name="昨天入库.mkv", action="move")
        current = self._task(name="本次冲突.mkv")
        waiting = self._task(status="requires_manual", name="本次待确认.mkv", action="")
        with patch(
            "app.modules.local_media_scan_runs.resolve_local_media_scan",
            return_value=self._scan([current, waiting]),
        ):
            result = self._list(scope="latest_scan")
        assert len(result.data["tasks"]) == 2
        assert result.data["scan_ref"] == "LM900"
        assert {item["original_filename"] for item in result.data["tasks"]} == {
            "本次冲突.mkv",
            "本次待确认.mkv",
        }
        assert result.data["tasks"][0]["task_number"] == 1
        assert result.data["tasks"][0]["task_number"] != current
        assert old != current

    def test_missing_scan_does_not_fall_back_to_history(self):
        self._task(name="历史入库.mkv", action="move")
        with patch(
            "app.modules.local_media_scan_runs.resolve_local_media_scan",
            side_effect=LookupError,
        ):
            result = self._list(scope="latest_scan")
        assert result.status == "not_recorded"
        assert result.data["tasks"] == []
        assert result.data["scan_recorded"] is False

    def test_deleted_scan_member_is_not_replaced_with_unrelated_history(self):
        task_id = self._task()
        self._task(name="其他历史.mkv", action="move")
        with patch(
            "app.modules.local_media_scan_runs.resolve_local_media_scan",
            return_value=self._scan([task_id, 999999]),
        ):
            result = self._list(scan_ref="LM900")
        assert result.data["total"] == 1
        assert result.data["missing_task_count"] == 1
        assert result.data["missing_tasks_excluded"] is True

    def test_scan_preserves_notification_outcome_after_task_retry(self):
        task_id = self._task(status="requires_manual", action="")
        snapshot = {
            "task_id": task_id,
            "status": "completed",
            "original_filename": "当时冲突.mkv",
            "file_names": ["当时冲突.mkv"],
            "file_outcome": "conflict_skipped",
            "video_count": 1,
            "archived_video_count": 0,
            "skipped_video_count": 1,
            "completed_at": "2026-09-12 14:22:00",
        }
        with patch(
            "app.modules.local_media_scan_runs.resolve_local_media_scan",
            return_value=self._scan([task_id], [snapshot]),
        ):
            result = self._list(scope="skipped", scan_ref="LM900")
        item = result.data["tasks"][0]
        assert item["status"] == "completed"
        assert item["current_status"] == "requires_manual"
        assert item["file_outcome"] == "conflict_skipped"
        assert item["outcome_basis"] == "scan_snapshot"
        assert item["can_inspect"]

    def test_reported_waiting_task_remains_waiting_fact_after_current_failure(self):
        task_id = self._task(status="failed", action="")
        snapshot = {
            "task_id": task_id,
            "status": "requires_manual",
            "original_filename": "当时待确认.mkv",
            "file_outcome": "pending",
            "video_count": 0,
            "archived_video_count": 0,
            "skipped_video_count": 0,
            "updated_at": "2026-09-12 14:22:00",
        }
        with patch(
            "app.modules.local_media_scan_runs.resolve_local_media_scan",
            return_value=self._scan([task_id], [snapshot]),
        ):
            result = self._list(scope="latest_scan")
        item = result.data["tasks"][0]
        assert item["status"] == "requires_manual"
        assert item["current_status"] == "failed"
        assert item["reason_code"] == "manual_match_required"
        assert item["outcome_basis"] == "scan_snapshot"
        assert item["can_inspect"] is False
        assert item["can_retry"] is True

    def test_status_filter_and_updated_sort_precede_limit(self):
        self._task(
            name="老ID最近更新.mkv", updated_at="2026-09-12 14:22:00", action="move"
        )
        for index in range(105):
            self._task(
                name=f"新ID旧时间-{index}.mkv",
                updated_at="2026-09-11 14:22:00",
                status="planned",
            )
        result = self._list(scope="history", limit=1)
        assert result.data["tasks"][0]["original_filename"] == "老ID最近更新.mkv"
        all_result = self._list(scope="all", limit=1)
        assert all_result.data["tasks"][0]["original_filename"] == "老ID最近更新.mkv"

    def test_skipped_scope_uses_video_actions_not_rss_or_task_status(self):
        self._task(name="确实归档.mkv", action="move")
        self._task(name="冲突跳过.mkv", action="skip")
        self._task(name="尚未执行.mkv", status="planned", action="skip")
        result = self._list(scope="skipped")
        assert [item["original_filename"] for item in result.data["tasks"]] == [
            "冲突跳过.mkv"
        ]

    def test_scan_pagination_reaches_all_25_tasks_and_keeps_current_action_refs(self):
        task_ids = [
            self._task(
                name=f"批次-{index}.mkv",
                status="requires_manual" if index >= 20 else "completed",
            )
            for index in range(25)
        ]
        snapshots = [
            {
                "task_id": task_id,
                "status": "completed",
                "original_filename": f"批次-{index}.mkv",
                "file_outcome": "conflict_skipped",
                "video_count": 1,
                "skipped_video_count": 1,
            }
            for index, task_id in enumerate(task_ids)
        ]
        with patch(
            "app.modules.local_media_scan_runs.resolve_local_media_scan",
            return_value=self._scan([*task_ids, 999999], snapshots),
        ):
            first = self._list(scope="skipped", scan_ref="LM900", limit=20)
            second = self._list(
                scope="skipped",
                scan_ref="LM900",
                limit=20,
                offset=first.data["next_offset"],
            )
        assert first.data["total"] == 20
        assert first.data["total_kind"] == "page_count"
        assert first.data["matched_total"] == 25
        assert first.data["has_more"]
        assert first.data["next_offset"] == 20
        assert second.data["total"] == 5
        assert second.data["matched_total"] == 25
        assert second.data["missing_task_count"] == 1
        assert second.data["has_more"] is False
        assert second.data["next_offset"] is None
        names = {
            item["original_filename"]
            for item in [*first.data["tasks"], *second.data["tasks"]]
        }
        assert len(names) == 25
        item = second.data["tasks"][0]
        assert item["task_number"] == 1
        assert item["status"] == "completed"
        assert item["current_status"] == "requires_manual"
        assert item["file_outcome"] == "conflict_skipped"
        assert item["can_inspect"]
        _, selected_task = _current_task("owner-a", 1)
        assert selected_task.id == task_ids[20]
        assert selected_task.status == "requires_manual"

    def test_history_pagination_uses_limit_plus_one_without_claiming_global_total(self):
        for index in range(25):
            self._task(name=f"历史分页-{index}.mkv")
        first = self._list(scope="history", limit=20)
        second = self._list(scope="history", limit=20, offset=first.data["next_offset"])
        assert first.data["total"] == 20
        assert first.data["has_more"]
        assert "matched_total" not in first.data
        assert second.data["total"] == 5
        assert second.data["has_more"] is False
        assert second.data["next_offset"] is None
        assert (
            len(
                {
                    item["original_filename"]
                    for item in [*first.data["tasks"], *second.data["tasks"]]
                }
            )
            == 25
        )

    def test_offset_validator_rejects_invalid_offsets(self):
        for value in (True, -1, "20", 2_147_483_648):
            with pytest.raises(AgentToolError):
                local_media_task_summaries_arguments({"offset": value})

    def test_scan_ref_validator_rejects_paths(self):
        with pytest.raises(AgentToolError):
            local_media_task_summaries_arguments({"scan_ref": "/private/LM900"})
