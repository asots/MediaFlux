"""同一登录的本地任务序号必须绑定各自 Agent 会话。"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.agent.domains.local_media.task_actions import LocalMediaTaskActions
from app.agent.domains.local_media.task_context import LocalMediaAgentContextStore
from app.agent.errors import AgentToolError
from app.agent.kernel.bootstrap import build_agent_kernel_runtime
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.models import ToolContext
from app.agent.session_context import SQLiteAgentSessionContextRepository
from app.agent.state_commit import AgentStateCommitBuffer, defer_agent_state_commits
from tests.support import isolated_test_database
from tests.test_agent_kernel_core import ScriptedModel


def _task(task_id: int, title: str):
    return SimpleNamespace(
        id=task_id, title=title, version=1, status="failed", source_id=7,
        snapshot_digest="source-digest", updated_at="2026-10-18 00:00:00",
        trigger="scan", error="", season_override=None, episode_override=None,
        media_type="movie", tmdb_id="123", numbering_mode="original",
    )


def _list(actions: LocalMediaTaskActions, context: ToolContext, task):
    buffer = AgentStateCommitBuffer(owner=context.owner)
    with (
        defer_agent_state_commits(buffer),
        patch("app.agent.domains.local_media.task_queries._scope_tasks", return_value=[task]),
        patch("app.agent.domains.local_media.task_queries._task_public", return_value={
            "task_number": 1, "title": task.title, "status": task.status,
        }),
    ):
        result = actions.list_local_media_task_summaries(
            {"scope": "attention", "offset": 0, "limit": 1}, context,
        )
    assert buffer.commit() == 1
    return result


def test_task_number_keeps_its_session_after_other_session_lists_tasks():
    actions = LocalMediaTaskActions()
    first, second = _task(101, "会话甲的影片"), _task(202, "会话乙的影片")
    first_context = ToolContext(owner="same-owner", session_id="session-a")
    second_context = ToolContext(owner="same-owner", session_id="session-b")
    assert _list(actions, first_context, first).data["tasks"][0]["title"] == first.title
    assert _list(actions, second_context, second).data["tasks"][0]["title"] == second.title
    with patch("app.agent.domains.local_media.task_actions.db.get_local_media_task",
               side_effect=lambda task_id, **_kwargs: {101: first, 202: second}[task_id]):
        preview, fingerprint = actions.prepare_retry_local_media_task(
            {"task_number": 1}, first_context,
        )
        with (
            patch("app.agent.domains.local_media.task_actions.db.reset_local_media_task_if_current", return_value=True) as retry,
            patch("app.agent.domains.local_media.task_actions.get_local_media_scheduler"),
        ):
            with pytest.raises(AgentToolError) as stale:
                actions.retry_local_media_task_confirmed({"task_number": 1}, fingerprint, second_context)
            assert stale.value.code == "confirmation_stale"
            retry.assert_not_called()
            assert actions.retry_local_media_task_confirmed({"task_number": 1}, fingerprint, first_context).ok
            assert retry.call_args.args[0] == first.id
    assert preview.data["title"] == first.title


def test_inspection_numbers_and_previews_stay_in_their_session():
    actions = LocalMediaTaskActions()
    first, second = _task(101, "甲"), _task(202, "乙")
    first.status = second.status = "requires_manual"
    contexts = [ToolContext(owner="same-owner", session_id=session) for session in ("session-a", "session-b")]
    service = Mock()
    service.inspect_task.side_effect = lambda _owner, task_id: {
        "inspection_id": f"inspection-{task_id}", "digest": "source-digest", "files": [],
    }
    service.preview.return_value = {"status": "planned"}
    with (
        patch("app.agent.domains.local_media.task_actions.db.get_local_media_task",
              side_effect=lambda task_id, **_kwargs: {101: first, 202: second}[task_id]),
        patch("app.agent.domains.local_media.task_actions.get_local_media_service", return_value=service),
    ):
        _list(actions, contexts[0], first)
        first_number = actions.inspect_local_media_task({"task_number": 1}, contexts[0]).data["inspection_number"]
        with pytest.raises(AgentToolError, match="检查序号不存在"):
            actions.preview_local_media_task({"inspection_number": first_number}, contexts[1])
        _list(actions, contexts[1], second)
        second_number = actions.inspect_local_media_task({"task_number": 1}, contexts[1]).data["inspection_number"]
        assert first_number == second_number == 1
        assert actions.preview_local_media_task({"inspection_number": 1}, contexts[0]).ok
        assert service.preview.call_args.args[1] == "inspection-101"
        assert actions.preview_local_media_task({"inspection_number": 1}, contexts[1]).ok
        assert service.preview.call_args.args[1] == "inspection-202"


def test_precise_refresh_confirmation_and_visibility_keep_the_session_binding():
    actions = LocalMediaTaskActions()
    tasks = [_task(101, "甲"), _task(202, "乙")]
    contexts = [ToolContext(owner="same-owner", session_id=session) for session in ("session-a", "session-b")]
    scopes = {}
    clients: dict[int, Mock] = {}
    for task in tasks:
        client = Mock()
        clients[task.id] = client
        client.refresh_for_paths.return_value = {"ok": True, "scope": "item", "matched": 1}
        client.has_tmdb_media.return_value = True
        scopes[task.id] = {
            "task": task, "client": client, "server_label": f"server-{task.id}",
            "paths": [f"/private/library/{task.id}.mkv"],
            "binding": {"library_name": task.title, "library_id": f"library-{task.id}",
                        "server_config_digest": "server-digest", "paths_digest": f"paths-{task.id}"},
        }
    for task, context in zip(tasks, contexts, strict=True):
        _list(actions, context, task)
    with (
        patch("app.agent.domains.local_media.task_actions.db.get_local_media_task",
              side_effect=lambda task_id, **_kwargs: next(task for task in tasks if task.id == task_id)),
        patch("app.agent.domains.local_media.task_actions.task_library._bound_task_scope",
              side_effect=lambda task: scopes[task.id]),
        patch("app.agent.domains.local_media.task_actions.close_media_server_client"),
        patch("app.agent.domains.local_media.task_actions.db.claim_agent_action_lease", return_value="lease"),
    ):
        preview, fingerprint = actions.prepare_refresh_local_media_task_library({"task_number": 1}, contexts[0])
        assert preview.data["server"] == "server-101"
        with pytest.raises(AgentToolError) as stale:
            actions.refresh_local_media_task_library_confirmed({"task_number": 1}, fingerprint, contexts[1])
        assert stale.value.code == "confirmation_stale"
        assert actions.refresh_local_media_task_library_confirmed({"task_number": 1}, fingerprint, contexts[0]).ok
        clients[101].refresh_for_paths.assert_called_once_with(
            scopes[101]["paths"], allowed_library_ids=("library-101",), allow_library_fallback=False,
        )
        clients[202].refresh_for_paths.assert_not_called()
        visibility = actions.verify_local_media_task_library_visibility({"task_number": 1}, contexts[0])
        assert visibility.ok and visibility.data["title"] == "甲"
        clients[101].has_tmdb_media.assert_called_once_with("123", "movie", parent_id="library-101")


@pytest.fixture
def repository():
    with isolated_test_database():
        yield SQLiteAgentSessionContextRepository(secret_provider=lambda: "session-scope-test")


def _capture(store: LocalMediaAgentContextStore, session_id: str, task):
    refs = store.capture_tasks(owner="same-owner", session_id=session_id, tasks=[task])
    store.capture_inspection(owner="same-owner", session_id=session_id, task=refs[0],
                             inspection_id=f"inspection-{task.id}", digest="source-digest")


def _runtime(repository):
    with patch("app.agent.kernel.bootstrap.SQLiteAgentSessionContextRepository", return_value=repository):
        return build_agent_kernel_runtime(
            model=ScriptedModel([]),
            store=SQLiteKernelStore(secret_provider=lambda: "session-scope-test"),
        )


@pytest.mark.parametrize("operation", ["reset", "delete"])
def test_lifecycle_clears_only_its_session_across_warm_workers_and_late_buffers(repository, operation):
    first, second = _runtime(repository), _runtime(repository)
    warm = first.domain_actions.local_media._context_store
    task_a, task_b = _task(101, "甲"), _task(202, "乙")
    _capture(warm, "session-a", task_a)
    _capture(warm, "session-b", task_b)
    restarted = LocalMediaAgentContextStore(repository=repository)
    restored_a = restarted.task(owner="same-owner", session_id="session-a", number=1)
    restored_b = restarted.task(owner="same-owner", session_id="session-b", number=1)
    assert restored_a is not None and restored_a.task_id == task_a.id
    assert restored_b is not None and restored_b.task_id == task_b.id
    old_buffer, other_buffer = AgentStateCommitBuffer(owner="same-owner"), AgentStateCommitBuffer(owner="same-owner")
    with defer_agent_state_commits(old_buffer):
        warm.capture_tasks(owner="same-owner", session_id="session-a", tasks=[task_a])
    with defer_agent_state_commits(other_buffer):
        warm.capture_tasks(owner="same-owner", session_id="session-b", tasks=[task_b])

    async def change_session():
        await first.store.reset_session(owner="same-owner", session_id="session-a")
        await getattr(second.lifecycle, operation)(owner="same-owner", session_id="session-a")

    asyncio.run(change_session())
    assert old_buffer.commit() == 0
    assert other_buffer.commit() == 1
    assert warm.task(owner="same-owner", session_id="session-a", number=1) is None
    assert warm.inspection(owner="same-owner", session_id="session-a", number=1) is None
    assert restarted.inspection(owner="same-owner", session_id="session-a", number=1) is None
    remaining_task = warm.task(owner="same-owner", session_id="session-b", number=1)
    remaining_inspection = warm.inspection(owner="same-owner", session_id="session-b", number=1)
    assert remaining_task is not None and remaining_task.task_id == task_b.id
    assert remaining_inspection is not None and remaining_inspection.task.task_id == task_b.id


@pytest.mark.parametrize("operation", ["reset", "delete"])
def test_lifecycle_cannot_report_success_when_persistent_scope_cleanup_fails(repository, operation):
    runtime = _runtime(repository)

    async def change_session():
        with (
            patch.object(repository, "invalidate_context", side_effect=OSError("storage failed")),
            patch.object(runtime.store, operation + "_session", wraps=getattr(runtime.store, operation + "_session")) as change,
            pytest.raises(OSError, match="storage failed"),
        ):
            await getattr(runtime.lifecycle, operation)(owner="same-owner", session_id="session-a")
        change.assert_not_called()

    asyncio.run(change_session())


def test_reset_rejects_late_memory_buffer_without_invalidating_other_session():
    store = LocalMediaAgentContextStore()
    task = _task(101, "甲")
    first, second = AgentStateCommitBuffer(owner="same-owner"), AgentStateCommitBuffer(owner="same-owner")
    with defer_agent_state_commits(first):
        _capture(store, "session-a", task)
    with defer_agent_state_commits(second):
        _capture(store, "session-b", task)
    store.clear_session(owner="same-owner", session_id="session-a")
    with defer_agent_state_commits(first):
        assert store.task(owner="same-owner", session_id="session-a", number=1) is None
        assert store.capture_tasks(owner="same-owner", session_id="session-a", tasks=[task]) == ()
    assert first.commit() == 0
    assert second.commit() == 1
    assert store.task(owner="same-owner", session_id="session-a", number=1) is None
    remaining_task = store.task(owner="same-owner", session_id="session-b", number=1)
    assert remaining_task is not None and remaining_task.task_id == task.id


def test_missing_persisted_snapshot_cannot_reseed_warm_inspections_or_an_old_buffer(repository):
    warm, other = LocalMediaAgentContextStore(repository=repository), LocalMediaAgentContextStore(repository=repository)
    task = _task(101, "旧任务")
    _capture(warm, "session-a", task)
    old_buffer = AgentStateCommitBuffer(owner="same-owner")
    with defer_agent_state_commits(old_buffer):
        warm.capture_tasks(owner="same-owner", session_id="session-a", tasks=[])
    other.clear_session(owner="same-owner", session_id="session-a")
    new_buffer = AgentStateCommitBuffer(owner="same-owner")
    with defer_agent_state_commits(new_buffer):
        assert warm.task(owner="same-owner", session_id="session-a", number=1) is None
        assert warm.inspection(owner="same-owner", session_id="session-a", number=1) is None
        warm.capture_tasks(owner="same-owner", session_id="session-a", tasks=[_task(202, "新任务")])
    assert old_buffer.commit() == 0
    assert new_buffer.commit() == 1
    new_task = warm.task(owner="same-owner", session_id="session-a", number=1)
    assert new_task is not None and new_task.task_id == 202
    assert warm.inspection(owner="same-owner", session_id="session-a", number=1) is None
    other.clear_session(owner="same-owner", session_id="session-a")
    warm.capture_tasks(owner="same-owner", session_id="session-a", tasks=[task])
    assert warm.inspection(owner="same-owner", session_id="session-a", number=1) is None


def test_failed_persistent_read_does_not_authorize_retry_from_warm_cache(repository):
    actions = LocalMediaTaskActions(repository=repository)
    context = ToolContext(owner="same-owner", session_id="session-a")
    task = _task(101, "甲")
    _list(actions, context, task)
    with patch("app.agent.domains.local_media.task_actions.db.get_local_media_task", return_value=task):
        _, fingerprint = actions.prepare_retry_local_media_task({"task_number": 1}, context)
        LocalMediaAgentContextStore(repository=repository).clear_session(owner=context.owner, session_id=context.session_id)
        with (
            patch.object(repository, "get_latest", side_effect=OSError("storage failed")),
            patch("app.agent.domains.local_media.task_actions.db.reset_local_media_task_if_current") as retry,
            pytest.raises(AgentToolError) as stale,
        ):
            actions.retry_local_media_task_confirmed({"task_number": 1}, fingerprint, context)
        assert stale.value.code == "confirmation_stale"
        retry.assert_not_called()


def test_owner_only_persisted_context_is_never_used_as_session_fallback(repository):
    store = LocalMediaAgentContextStore(repository=repository)
    task = _task(101, "旧作用域")
    refs = LocalMediaAgentContextStore().capture_tasks(owner="same-owner", session_id="session-a", tasks=[task])
    repository.replace_latest(owner="same-owner", context_type="local_media_tasks",
                              payload=store._encode_payload(refs, (), 0), expires_at=time.time() + 60)
    assert store.task(owner="same-owner", session_id="session-a", number=1) is None
    assert store.inspection(owner="same-owner", session_id="session-a", number=1) is None


@pytest.mark.parametrize("context", [ToolContext(owner="same-owner"), ToolContext(owner="same-owner", session_id=" ")])
def test_task_actions_require_a_real_session(context):
    with pytest.raises(AgentToolError) as unavailable:
        LocalMediaTaskActions().list_local_media_task_summaries({"scope": "attention", "limit": 1}, context)
    assert unavailable.value.code == "precondition_failed"
