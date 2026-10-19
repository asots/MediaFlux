"""重命名预览必须留在创建会话，生命周期清理拒绝迟到保存。"""
from __future__ import annotations

import asyncio
import json
import time
from functools import partial
from typing import cast
from unittest.mock import patch

import httpx
import pytest

from app.agent.kernel.bootstrap import build_agent_kernel_runtime
from app.agent.kernel.state_types import AgentInput
from app.agent.models import ToolContext
from app.agent.errors import AgentToolError
from app.agent.domains.guangya import rename_actions as actions
from app.agent.session_context import SQLiteAgentSessionContextRepository
from app.clients.guangya import GuangYaClient
from app.modules import guangya_rename
from app.repositories import organize_operation_jobs as jobs
from tests.support import initialized_web_runtime, isolated_test_database
from tests.test_agent_effect_admission import _offline_runtime, _paused_manager, _run_real_worker
from tests.test_agent_guangya_fs_gateway import FakeGatewayClient
from tests.test_agent_task_first_slice import NativeScript, OWNER


class Script(NativeScript):
    def __init__(self, tool, arguments):
        super().__init__(None)
        self.tool, self.arguments = tool, arguments
        self.steps = ["discover", "tool", "finish"]

    async def respond(self, request):
        self.requests.append(json.loads(request.content))
        step = self.steps.pop(0)
        if step == "finish":
            message = {"content": "预览已完成。"}
        else:
            name, arguments = (("agent.capabilities", {"tool_names": [self.tool]}) if step == "discover"
                               else (self.tool, self.arguments))
            message = {"content": "", "tool_calls": [{"id": step + str(len(self.requests)), "type": "function",
                "function": {"name": name.replace(".", "__"), "arguments": json.dumps(arguments)}}]}
        return httpx.Response(200, json={"choices": [{"message": message,
            "finish_reason": "stop" if step == "finish" else "tool_calls"}]})


class CountingClient(FakeGatewayClient):
    def __init__(self):
        super().__init__()
        self.writes = []

    def rename(self, file_id, new_name):
        self.writes.append((file_id, new_name))
        return super().rename(file_id, new_name)


async def run(runtime, session_id, request_id, message):
    return [e async for e in runtime.session.run(AgentInput(
        owner=OWNER, session_id=session_id, request_id=request_id, message=message))]


def test_owner_only_preview_is_not_rebound_to_a_new_session(tmp_path):
    with isolated_test_database(), _offline_runtime(tmp_path, CountingClient()):
        repository = SQLiteAgentSessionContextRepository()
        rename = actions.GuangyaRenameActions(repository)
        context = ToolContext(owner=OWNER, session_id="origin")
        args = actions.guangya_rename_preview_arguments({"paths": ["/source"], "mode": "replace_text",
            "find": "广告-", "replace": "review-"})
        rename.preview_guangya_rename(args, context)
        flow = rename._flow(OWNER, "origin")
        assert flow is not None
        repository.replace_latest(owner=OWNER, context_type="guangya_rename",
            payload=actions._flow_payload(flow), expires_at=time.time() + 600)
        rename.clear_session(owner=OWNER, session_id="origin")
        for sid in ("origin", "other"):
            with pytest.raises(AgentToolError, match="预览不存在"):
                rename.prepare_guangya_rename_confirmation({}, ToolContext(owner=OWNER, session_id=sid))


@pytest.mark.parametrize("sid", ["", " "])
def test_preview_requires_a_session(sid):
    with pytest.raises(AgentToolError, match="会话"):
        actions.GuangyaRenameActions().preview_guangya_rename({}, ToolContext(owner=OWNER, session_id=sid))


@pytest.mark.parametrize("operation", ["other_session", "reset", "delete"])
def test_native_rename_never_rebinds_another_session_preview(tmp_path, operation):
    async def scenario():
        client = CountingClient()
        with initialized_web_runtime(), _offline_runtime(tmp_path, client), _paused_manager():
            first = build_agent_kernel_runtime(model=Script("guangya.rename.preview", {
                "paths": ["/source"], "mode": "replace_text", "recursive": True,
                "find": "广告-", "replace": "review-",
            }).adapter)
            events = await run(first, "origin", "preview-origin", "预览 /source 下的批量名称替换。")
            assert events[-1].payload["status"] == "success"
            if operation in {"reset", "delete"}:
                await getattr(first.lifecycle, operation)(owner=OWNER, session_id="origin")
            target = "other" if operation == "other_session" else "origin"
            second = build_agent_kernel_runtime(model=Script("guangya.rename.execute", {}).adapter)
            events = await run(second, target, "execute-new", "确认执行当前会话最近的重命名预览。")
            assert events[-1].type.value == "turn.failed"
            assert events[-1].payload["code"] == "precondition_failed"
            assert await second.session.pipeline.get_pending_plan(owner=OWNER, session_id=target) is None
            assert client.writes == []
            assert jobs.claim_organize_operation_job() is None
    asyncio.run(scenario())


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("operation", ["reset", "delete"])
def test_clear_rejects_late_preview_and_preserves_other_session(tmp_path, persistent, operation):
    import threading
    from tests.test_agent_kernel_core import ScriptedModel

    async def scenario():
        client = CountingClient()
        with isolated_test_database(), _offline_runtime(tmp_path, client), _paused_manager():
            repository = SQLiteAgentSessionContextRepository() if persistent else None
            with patch("app.agent.kernel.bootstrap.SQLiteAgentSessionContextRepository", return_value=repository):
                first = build_agent_kernel_runtime(model=ScriptedModel([]))
                controller = build_agent_kernel_runtime(model=ScriptedModel([])) if persistent else first
            args = actions.guangya_rename_preview_arguments({"paths": ["/source"], "mode": "replace_text",
                "find": "广告-", "replace": "review-"})
            a, b = ToolContext(owner=OWNER, session_id="a"), ToolContext(owner=OWNER, session_id="b")
            rename = first.domain_actions.rename
            rename.preview_guangya_rename(args, b)
            flow_b = rename._flow(OWNER, "b")
            assert flow_b is not None
            entered, release = threading.Event(), threading.Event()
            build = actions.build_rename_plan
            def blocked(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return build(*args, **kwargs)
            with patch.object(actions, "build_rename_plan", side_effect=blocked):
                pending = asyncio.create_task(asyncio.to_thread(rename.preview_guangya_rename, args, a))
                try:
                    assert await asyncio.to_thread(entered.wait, 3)
                    await getattr(controller.lifecycle, operation)(owner=OWNER, session_id="a")
                finally:
                    release.set()
                    result, = await asyncio.gather(pending, return_exceptions=True)
            assert isinstance(result, AgentToolError)
            assert rename._flow(OWNER, "a") is None
            remaining_b = rename._flow(OWNER, "b")
            assert remaining_b is not None and remaining_b.plan_id == flow_b.plan_id
            # 当前会话重新预览可以正常保存，另一个会话的计划不被替换或删除。
            rename.preview_guangya_rename(args, a)
            new_a = rename._flow(OWNER, "a")
            assert new_a is not None and new_a.plan_id != flow_b.plan_id
            assert rename.prepare_guangya_rename_confirmation({}, b)[0].ok
    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["reset", "delete"])
def test_reset_invalidates_warm_and_cold_preview_but_other_session_can_execute(tmp_path, operation):
    from tests.test_agent_kernel_core import ScriptedModel

    async def scenario():
        client = CountingClient()
        with initialized_web_runtime(), _offline_runtime(tmp_path, client), _paused_manager(), patch.object(
            guangya_rename, "execute_rename_plan", partial(guangya_rename.execute_rename_plan,
                client_factory=lambda: cast(GuangYaClient, client)),
        ):
            first = build_agent_kernel_runtime(model=ScriptedModel([]))
            args = actions.guangya_rename_preview_arguments({"paths": ["/source"], "mode": "replace_text",
                "find": "广告-", "replace": "review-"})
            for sid in ("b", "a"):
                first.domain_actions.rename.preview_guangya_rename(args, ToolContext(owner=OWNER, session_id=sid))
            second = build_agent_kernel_runtime(model=Script("guangya.rename.execute", {}).adapter)
            await getattr(second.lifecycle, operation)(owner=OWNER, session_id="a")
            assert first.domain_actions.rename._flow(OWNER, "a") is None
            assert second.domain_actions.rename._flow(OWNER, "a") is None
            events = await run(second, "b", "execute-b", "执行当前会话最近的重命名预览。")
            assert events[-1].payload["status"] == "approval_required"
            plan = await second.session.pipeline.get_pending_plan(owner=OWNER, session_id="b")
            assert plan is not None and client.writes == []
            plan_id = plan.plan_id
            async def confirm():
                return [event async for event in second.session.confirm(owner=OWNER, session_id="b", plan_id=plan_id)]
            pending = asyncio.create_task(confirm())
            for _ in range(500):
                job = jobs.claim_organize_operation_job()
                if job is not None:
                    break
                if pending.done():
                    raise AssertionError(await pending)
                await asyncio.sleep(.01)
            else:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                raise AssertionError("confirmed job not admitted")
            await asyncio.to_thread(_run_real_worker, job)
            assert (await pending)[-1].payload["status"] == "success"
            assert client.writes == [("rename", "review-ABC.mp4")]
    asyncio.run(scenario())
