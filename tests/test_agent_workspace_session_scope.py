"""文件观察、冻结预览与会话生命周期使用同一作用域。"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import pytest

from app import database as db
from app.agent.confirmation import SQLiteConfirmationStore
from app.agent.domain_catalog import DomainActions
from app.agent.domains.guangya import episode_naming_actions, fs_change_actions, share_actions, workspace_actions
from app.agent.errors import AgentToolError
from app.agent.kernel.bootstrap import build_agent_kernel_runtime
from app.agent.kernel.effects import ConfirmationEffectPlanStore
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.models import ToolContext
from app.agent.session_context import SQLiteAgentSessionContextRepository
from app.modules import guangya_fs_change as plans
from app.modules import guangya_workspace as workspace
from tests.support import isolated_test_database
from tests.test_agent_effect_admission import _kernel_plan, _OfflineClient, _offline_runtime, _paused_manager
from tests.test_agent_kernel_core import ScriptedModel

OWNER = "workspace-scope-owner"
A = ToolContext(owner=OWNER, session_id="a")
B = ToolContext(owner=OWNER, session_id="b")


@dataclass
class Harness:
    actions: DomainActions
    repository: SQLiteAgentSessionContextRepository | None
    client: _OfflineClient
    store: SQLiteKernelStore
    root: Path
    stack: object

    def observe(self, context=A, **arguments):
        return self.actions.workspace.query_guangya_filesystem(
            workspace_actions.guangya_fs_query_arguments(arguments or {"operation": "tree", "path": "/source"}), context,
        )

    def prepare(self, context=A, *, observation=None, name="scoped.mp4"):
        observation = observation or self.observe(context)
        handle = next(row["object_ref"] for row in observation.data["entries"] if row["object_name"] == "广告-ABC.mp4")
        return self.actions.fs_change.prepare_guangya_fs_change_confirmation(
            fs_change_actions.guangya_fs_change_arguments({
                "observation_ref": observation.data["observation_ref"],
                "operations": [{"op": "rename", "object_ref": handle, "new_name": name}],
                "trigger_strm": False,
            }), context,
        )

    def controller(self):
        actions = DomainActions(self.repository) if self.repository is not None else self.actions
        with patch("app.agent.kernel.bootstrap.SQLiteAgentSessionContextRepository", return_value=self.repository), patch(
            "app.agent.kernel.bootstrap.DomainActions", return_value=actions,
        ):
            return build_agent_kernel_runtime(model=ScriptedModel([]), store=self.store)


@pytest.fixture(params=[False, True], ids=["memory", "sqlite"])
def harness(request, tmp_path):
    client = _OfflineClient()
    with isolated_test_database(), _offline_runtime(tmp_path, client) as stack, _paused_manager(), patch.object(
        workspace_actions, "GuangYaClient", return_value=client,
    ):
        repository = SQLiteAgentSessionContextRepository() if request.param else None
        stack.domain_actions = DomainActions(repository)
        yield Harness(stack.domain_actions, repository, client, SQLiteKernelStore(), tmp_path, stack)


def test_two_kernel_confirmations_keep_each_sessions_plan(harness):
    async def scenario():
        effects = ConfirmationEffectPlanStore(SQLiteConfirmationStore())
        first, context_a, card_a, _ = await _kernel_plan(
            "fs", harness.store, harness.client, harness.stack, owner=OWNER, session_id="a",
            request_id="a-preview", new_name="session-a.mp4", effect_store=effects,
        )
        flow_a = harness.actions.fs_change._flow(OWNER, "a")
        assert flow_a is not None
        frozen = plans.load_fs_change_plan(flow_a.plan_id, owner=OWNER)
        if harness.repository is not None:
            harness.stack.domain_actions = DomainActions(harness.repository)
        second, context_b, card_b, _ = await _kernel_plan(
            "fs", harness.store, harness.client, harness.stack, owner=OWNER, session_id="b",
            request_id="b-preview", new_name="session-b.mp4", effect_store=effects,
        )
        assert card_a.plan_id != card_b.plan_id
        assert plans.load_fs_change_plan(flow_a.plan_id, owner=OWNER) == frozen
        assert await first.get_pending_plan(owner=OWNER, session_id="a") == card_a
        assert await second.get_pending_plan(owner=OWNER, session_id="b") == card_b
        for pipeline, context, card in ((first, context_a, card_a), (second, context_b, card_b)):
            result = await pipeline.execute_confirmed(card.plan_id, context=context)
            assert result.outcome.public_content["ok"]
        with db.get_conn() as conn:
            jobs = conn.execute("SELECT job_id,status FROM organize_operation_jobs").fetchall()
        assert len(jobs) == 2 and all(row["status"] == "pending" for row in jobs)
        assert harness.actions.fs_change._flow(OWNER, "a") is None
        assert harness.stack.domain_actions.fs_change._flow(OWNER, "b") is None
        # 确认只入队，不能把 preview/accepted 冒充远端已改名。
        assert harness.client.file_info("rename").name == "广告-ABC.mp4"
        queued_a = plans.load_fs_change_plan(flow_a.plan_id, owner=OWNER)
        assert queued_a["status"] == "queued"
        assert queued_a["operations"][0]["new_name"] == "session-a.mp4"
    asyncio.run(scenario())


def test_latest_cursor_explicit_observation_and_object_merge_share_the_session(harness):
    primary = harness.observe(A, operation="list", path="/")
    additional = harness.observe(A)
    foreign = harness.observe(B)
    ref_a, ref_b = additional.data["observation_ref"], foreign.data["observation_ref"]
    assert harness.actions.workspace.latest_guangya_observation_ref(OWNER, "a") == ref_a
    assert harness.actions.workspace.latest_guangya_observation_ref(OWNER, "b") == ref_b
    own_object = additional.data["entries"][0]["object_ref"]
    foreign_object = foreign.data["entries"][0]["object_ref"]
    merged = workspace.load_directory_observation_set(
        primary.data["observation_ref"], owner=OWNER, session_id="a", object_refs=[own_object, foreign_object],
    )
    assert own_object in workspace.observation_entry_map(merged)
    assert foreign_object not in workspace.observation_entry_map(merged)
    foreign_payload = workspace.load_directory_observation(ref_b, owner=OWNER, session_id="b")
    with pytest.raises(workspace.GuangYaWorkspaceStale, match="会话"):
        workspace.merge_directory_observations(harness.client, owner=OWNER, session_id="a", payloads=[foreign_payload])
    with pytest.raises(AgentToolError, match="不属于当前会话"):
        harness.observe(B, observation_ref=ref_a)
    assert harness.actions.workspace.latest_guangya_observation_ref(OWNER, "b") == ref_b
    # 同会话的旧观察仍支持分页，不因 latest 变化丢失多步选择。
    assert harness.observe(A, observation_ref=primary.data["observation_ref"]).ok


def test_fs_and_share_cannot_use_another_sessions_latest_or_explicit_observation(harness):
    observed = harness.observe(A)
    handle = next(row["object_ref"] for row in observed.data["entries"] if row["object_name"] == "广告-ABC.mp4")
    change = fs_change_actions.guangya_fs_change_arguments({
        "operations": [{"op": "rename", "object_ref": handle, "new_name": "selected.mp4"}], "trigger_strm": False,
    })
    share = share_actions.guangya_share_create_arguments({"object_refs": [handle]})
    for arguments, prepare in (
        (change, lambda arguments: harness.actions.fs_change.prepare_guangya_fs_change_confirmation(arguments, B)),
        (share, lambda arguments: share_actions.prepare_create_guangya_share(arguments, B, workspace_actions=harness.actions.workspace)),
    ):
        with pytest.raises(AgentToolError, match="请先读取"):
            prepare(arguments)
        with pytest.raises(AgentToolError, match="不属于当前会话"):
            prepare({**arguments, "observation_ref": observed.data["observation_ref"]})
    with db.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM organize_operation_jobs").fetchone()[0] == 0
    assert not list((harness.root / "fs-plans").glob("*.json"))


@pytest.mark.parametrize("operation", ["reset", "delete"])
def test_lifecycle_invalidates_only_current_session_and_cold_reconstruction(harness, operation):
    async def scenario():
        observed_a, observed_b = harness.observe(A), harness.observe(B)
        harness.prepare(A, observation=observed_a)
        harness.prepare(B, observation=observed_b)
        flow_b = harness.actions.fs_change._flow(OWNER, "b")
        assert flow_b is not None
        await harness.store.load(owner=OWNER, session_id="a")
        await getattr(harness.controller().lifecycle, operation)(owner=OWNER, session_id="a")
        consumers = [harness.actions]
        if harness.repository is not None:
            consumers.append(DomainActions(harness.repository))
        for actions in consumers:
            assert actions.workspace.latest_guangya_observation_ref(OWNER, "a") == ""
            assert actions.fs_change._flow(OWNER, "a") is None
            assert actions.workspace.latest_guangya_observation_ref(OWNER, "b") == observed_b.data["observation_ref"]
            restored_b = actions.fs_change._flow(OWNER, "b")
            assert restored_b is not None
            assert replace(restored_b, updated_at=flow_b.updated_at) == flow_b
        with pytest.raises(workspace.GuangYaWorkspaceError):
            workspace.load_directory_observation(observed_a.data["observation_ref"], owner=OWNER, session_id="a")
        assert workspace.load_directory_observation(observed_b.data["observation_ref"], owner=OWNER, session_id="b")
        assert harness.observe(A).ok
        assert harness.prepare(A)[0].ok
    asyncio.run(scenario())


@pytest.mark.parametrize("intervention", ["reset", "delete", "newer"])
@pytest.mark.parametrize("point", ["query", "prepare"])
def test_late_observation_or_preview_cannot_restore_invalidated_generation(harness, intervention, point):
    async def scenario():
        observed_a, observed_b = harness.observe(A), harness.observe(B)
        harness.prepare(B, observation=observed_b)
        original_b = harness.actions.fs_change._flow(OWNER, "b")
        assert original_b is not None
        await harness.store.load(owner=OWNER, session_id="a")
        entered, release = threading.Event(), threading.Event()
        target, method = ((workspace_actions, "_create_observation") if point == "query"
                          else (fs_change_actions, "build_fs_change_plan"))
        original = getattr(target, method)
        first = True

        def blocked(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                entered.set()
                assert release.wait(5), "scope barrier timed out"
            return original(*args, **kwargs)

        call = harness.observe if point == "query" else lambda: harness.prepare(A, observation=observed_a)
        newer = None
        with patch.object(target, method, side_effect=blocked):
            pending = asyncio.create_task(asyncio.to_thread(call))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                if intervention == "newer":
                    newer = harness.observe(A, operation="list", path="/") if point == "query" else harness.prepare(A, observation=observed_a, name="newer.mp4")
                else:
                    await getattr(harness.controller().lifecycle, intervention)(owner=OWNER, session_id="a")
            finally:
                release.set()
                result, = await asyncio.gather(pending, return_exceptions=True)
        assert isinstance(result, AgentToolError), result
        assert "取代" in str(result)
        if intervention == "newer":
            assert newer is not None
            if point == "query":
                assert harness.actions.workspace.latest_guangya_observation_ref(OWNER, "a") == newer.data["observation_ref"]
            else:
                latest = harness.actions.fs_change._flow(OWNER, "a")
                assert latest is not None
                assert plans.load_fs_change_plan(latest.plan_id, owner=OWNER)["operations"][0]["new_name"] == "newer.mp4"
        else:
            assert harness.actions.workspace.latest_guangya_observation_ref(OWNER, "a") == ""
            assert harness.actions.fs_change._flow(OWNER, "a") is None
        restored_b = harness.actions.fs_change._flow(OWNER, "b")
        assert restored_b is not None
        assert replace(restored_b, updated_at=original_b.updated_at) == original_b
        assert workspace.load_directory_observation(observed_b.data["observation_ref"], owner=OWNER, session_id="b")
        with db.get_conn() as conn:
            assert conn.execute("SELECT COUNT(*) FROM organize_operation_jobs").fetchone()[0] == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("entrypoint", ["query", "prepare", "share", "episode"])
def test_missing_session_fails_before_provider_use(harness, entrypoint):
    empty = ToolContext(owner=OWNER)
    with patch.object(workspace_actions, "GuangYaClient", side_effect=AssertionError("missing session reached Provider")), patch.object(
        fs_change_actions, "GuangYaClient", side_effect=AssertionError("missing session reached Provider"),
    ), patch.object(share_actions, "GuangYaClient", side_effect=AssertionError("missing session reached Provider")), patch.object(
        episode_naming_actions, "GuangYaClient", side_effect=AssertionError("missing session reached Provider"),
    ), pytest.raises(AgentToolError, match="会话"):
        if entrypoint == "query":
            harness.observe(empty)
        elif entrypoint == "prepare":
            harness.actions.fs_change.prepare_guangya_fs_change_confirmation({}, empty)
        elif entrypoint == "share":
            share_actions.prepare_create_guangya_share({}, empty, workspace_actions=harness.actions.workspace)
        else:
            episode_naming_actions.inspect_guangya_episode_naming({"target_root": "/source"}, empty)


@pytest.mark.parametrize("harness", [True], indirect=True, ids=["sqlite"])
def test_owner_only_context_and_observation_are_never_rebound(harness):
    observed = harness.observe(A)
    harness.prepare(A, observation=observed)
    flow = harness.actions.fs_change._flow(OWNER, "a")
    assert flow is not None
    for context_type, payload in (
        ("guangya_fs_change", fs_change_actions._flow_payload(flow)),
        ("guangya_workspace", {"observation_ref": observed.data["observation_ref"]}),
    ):
        harness.repository.replace_latest(owner=OWNER, context_type=context_type, payload=payload, expires_at=time.time() + 600)
    harness.actions.clear_session(owner=OWNER, session_id="a")
    for context in (A, B):
        cold = DomainActions(harness.repository)
        assert cold.fs_change._flow(OWNER, context.session_id) is None
        assert cold.workspace.latest_guangya_observation_ref(OWNER, context.session_id) == ""
    fresh = harness.observe(A)
    payload = workspace.load_directory_observation(fresh.data["observation_ref"], owner=OWNER, session_id="a")
    payload.pop("session_scope")
    workspace._atomic_write(payload)
    with pytest.raises(workspace.GuangYaWorkspaceError, match="不属于当前会话"):
        workspace.load_directory_observation(fresh.data["observation_ref"], owner=OWNER, session_id="a")


@pytest.mark.parametrize("harness", [True], indirect=True, ids=["sqlite"])
@pytest.mark.parametrize("domain", ["workspace", "fs_change"])
def test_clear_propagates_repository_failure_without_erasing_live_context(harness, domain):
    observed = harness.observe(A)
    harness.prepare(A, observation=observed)
    with patch.object(harness.repository, "invalidate_context", side_effect=RuntimeError("context unavailable")), pytest.raises(
        RuntimeError, match="context unavailable",
    ):
        getattr(harness.actions, domain).clear_session(owner=OWNER, session_id="a")
    assert harness.actions.fs_change._flow(OWNER, "a") is not None
    assert harness.actions.workspace.latest_guangya_observation_ref(OWNER, "a") == observed.data["observation_ref"]
    assert workspace.load_directory_observation(observed.data["observation_ref"], owner=OWNER, session_id="a")
