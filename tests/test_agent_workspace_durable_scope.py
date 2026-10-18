"""会话临时观察清理不撤销已经受理的持久文件任务。"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.agent.confirmation import SQLiteConfirmationStore
from app.agent.domain_catalog import DomainActions
from app.agent.kernel.bootstrap import build_agent_kernel_runtime
from app.agent.kernel.effects import ConfirmationEffectPlanStore
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.state_types import StalePublicationError
from app.agent.session_context import SQLiteAgentSessionContextRepository
from app.modules import guangya_fs_change as fs
from app.modules import guangya_workspace as workspace
from app.repositories import organize_operation_jobs as jobs
from tests.support import isolated_test_database
from tests.test_agent_effect_admission import (
    _OfflineClient,
    _kernel_plan,
    _offline_runtime,
    _paused_manager,
    _run_real_worker,
)
from tests.test_agent_kernel_core import ScriptedModel


@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "sqlite-cold"])
@pytest.mark.parametrize("operation", ["reset", "delete"])
def test_accepted_fs_job_survives_session_cleanup_and_executes_once(tmp_path, persistent, operation):
    async def scenario():
        owner, session_id = "durable-scope-owner", "durable-scope-session"
        client = _OfflineClient()
        with isolated_test_database(), _offline_runtime(tmp_path, client) as stack, _paused_manager():
            repository = SQLiteAgentSessionContextRepository() if persistent else None
            actions = stack.domain_actions = DomainActions(repository)
            store = SQLiteKernelStore()
            pipeline, context, plan, _ = await _kernel_plan(
                "fs", store, client, stack, owner=owner, session_id=session_id,
                effect_store=ConfirmationEffectPlanStore(SQLiteConfirmationStore()),
            )
            flow = actions.fs_change._flow(owner, session_id)
            assert flow is not None
            observation_ref = plan.arguments["observation_ref"]
            accepted = await pipeline.execute_confirmed(plan.plan_id, context=context)
            assert accepted.outcome.public_content["ok"] is True
            frozen = fs.load_fs_change_plan(flow.plan_id, owner=owner)
            assert frozen["status"] == "queued"
            job_id = frozen["job_id"]
            queued = jobs.get_organize_operation_job(job_id)
            assert queued is not None and queued["status"] == "pending"

            controller_actions = DomainActions(repository) if persistent else actions
            with patch("app.agent.kernel.bootstrap.DomainActions", return_value=controller_actions):
                controller = build_agent_kernel_runtime(model=ScriptedModel([]), store=SQLiteKernelStore())
            await getattr(controller.lifecycle, operation)(owner=owner, session_id=session_id)
            assert actions.fs_change._flow(owner, session_id) is None
            with pytest.raises(workspace.GuangYaWorkspaceError):
                workspace.load_directory_observation(observation_ref, owner=owner, session_id=session_id)
            # 只读快照和对话可以消失，受理事务冻结的计划仍是 worker 唯一执行依据。
            preserved = fs.load_fs_change_plan(flow.plan_id, owner=owner)
            assert preserved["status"] == "queued"
            assert preserved["operations"] == frozen["operations"]
            assert preserved["job_id"] == job_id

            claimed = jobs.claim_organize_operation_job(job_id)
            assert claimed is not None
            with patch.object(client, "rename", wraps=client.rename) as rename:
                await asyncio.to_thread(_run_real_worker, claimed)
                done = jobs.get_organize_operation_job(job_id)
                assert done is not None and done["status"] == "completed"
                renamed = client.file_info("rename")
                assert renamed is not None and renamed.name == "admitted.mp4"
                rename.assert_called_once_with("rename", "admitted.mp4")
                assert jobs.claim_organize_operation_job(job_id) is None
                with pytest.raises(StalePublicationError):
                    await pipeline.execute_confirmed(plan.plan_id, context=context)
                assert rename.call_count == 1

    asyncio.run(scenario())
