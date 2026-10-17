"""fs.change 原生恢复闭环：真实 Session/SQLite/签名 plan，Provider 明确离线。

不替换领域 handler、确认执行器或回执；只注入既有 FakeGatewayClient 与模型
MockTransport。领域 plan 过期而确认卡仍有效，旧确认必须先 READ 才能给新卡。
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from unittest.mock import Mock

import httpx
import requests

from app import database as db
from app.agent import guangya_fs_change_actions as changes
from app.agent import guangya_workspace_actions as files
from app.agent.confirmation import SQLiteConfirmationStore
from app.agent.domain_catalog import build_tool_specs
from app.agent.kernel.adapters import TurnViewBuilder
from app.agent.kernel.capabilities import CapabilityRetriever, ToolCatalog
from app.agent.kernel.effects import ConfirmationEffectPlanStore
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports.existing_actions import adapt_tool_spec
from app.agent.kernel.ports.mediaflux_effects import MediaFluxEffectLifecycle
from app.agent.kernel.ports.mediaflux_policy import (
    MediaFluxAuthorizationPolicy,
    MediaFluxToolRateLimiter,
    MediaFluxTurnAdmission,
)
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import AgentInput
from app.agent.session_context import SQLiteAgentSessionContextRepository
from app.clients.guangya import GuangYaClient
from app.modules import guangya_fs_change as plans
from app.modules import guangya_workspace as workspace
from app.modules import organize_tasks
from app.repositories.organize_operation_jobs import get_organize_operation_job
from tests.support import initialized_web_runtime
from tests.test_agent_guangya_fs_gateway import FakeGatewayClient
from tests.test_agent_task_first_slice import OWNER, NativeScript

READ = "guangya.fs.query"
EXECUTE = "guangya.fs.change.execute"
SESSION_ID = "offline-fs-reobservation-native"
GOAL = "把 /source/Move.mp4 改名为 Final.mp4；若计划过期或对象改名，重新读取原范围并给我新确认卡，不要自动执行。"
SECRET = "offline-fs-native-session"


class FsReobservationScript(NativeScript):
    """只从真实 HTTP tool 消息取 OBS/OBJ；不持有 Provider 或签名文件。"""

    def __init__(self):
        super().__init__("fixture")
        self.steps = ["query", "prepare", "new_prepare"]
        self.calls = []

    async def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        assert self.steps, "离线原生回归：不允许额外模型续行"
        step = self.steps.pop(0)
        if step == "finish":
            self.calls.append({"step": step})
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
                "content": "已按新确认完成文件改名。",
            }}]})
        if step in {"query", "reobserve"}:
            name = READ
            arguments = (
                {"operation": "list", "path": "/source", "max_items": 20}
                if step == "query" else {"operation": "reobserve"}
            )
        elif step in {"prepare", "new_prepare"}:
            observations = [
                value["data"]
                for row in body["messages"] if row.get("role") == "tool"
                for value in [json.loads(row["content"])]
                if value.get("data", {}).get("observation_ref")
            ]
            assert observations, "prepare 必须使用业务 READ 返回的观察"
            latest = observations[-1]
            if step == "new_prepare":
                comparison = latest["reobservation"]
                assert comparison["complete"] is True
                assert len(comparison["items"]) == 1
                matched = comparison["items"][0]
                assert matched["state"] == "same_identity"
                object_ref = matched["object_ref"]
            else:
                object_ref = next(row["ref"] for row in latest["entries"] if row["name"] == "Move.mp4")
            name = EXECUTE
            arguments = {
                "observation_ref": latest["observation_ref"],
                "operations": [{"op": "rename", "object_ref": object_ref, "new_name": "Final.mp4"}],
                "trigger_strm": False,
            }
        else:
            raise AssertionError("未定义的离线模型步骤：" + step)
        self.calls.append({"step": step, "tool": name, "arguments": arguments})
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "content": "", "tool_calls": [{"id": "native-fs-" + step, "type": "function",
                "function": {"name": name.replace(".", "__"), "arguments": json.dumps(arguments)}}],
        }}]})


def test_expired_fs_plan_confirmation_reobserves_renamed_object_and_requires_new_card(monkeypatch):
    def no_live_io(*_args, **_kwargs):
        raise AssertionError("fs.change 离线原生回归：禁止真实网络/SDK")

    monkeypatch.setattr(socket.socket, "connect", no_live_io)
    monkeypatch.setattr(socket.socket, "connect_ex", no_live_io)
    monkeypatch.setattr(requests.sessions.Session, "request", no_live_io)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_live_io)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_live_io)
    monkeypatch.setattr(GuangYaClient, "__init__", no_live_io)

    with initialized_web_runtime() as paths:
        provider = FakeGatewayClient()
        writes = {}
        for method in ("rename", "move", "copy", "delete", "create_dir"):
            writes[method] = Mock(wraps=getattr(provider, method))
            monkeypatch.setattr(provider, method, writes[method])
        monkeypatch.setattr(workspace, "PATHS", paths)
        monkeypatch.setattr(plans, "PATHS", paths)
        monkeypatch.setattr(files, "GuangYaClient", lambda: provider)
        monkeypatch.setattr(changes, "GuangYaClient", lambda: provider)
        files.reset_guangya_workspace_context_for_tests()
        changes.reset_guangya_fs_change_context_for_tests()
        repository = SQLiteAgentSessionContextRepository()
        files.configure_guangya_workspace_context(repository)
        changes.configure_guangya_fs_change_context(repository)
        # 复用 queue 测试的独立 manager/进程内锁；入队、签名校验、worker、回执均真实。
        manager = organize_tasks.OrganizeTaskManager()
        manager._lock = threading.Lock()
        monkeypatch.setattr(organize_tasks, "get_organize_manager", lambda: manager)
        monkeypatch.setattr(manager, "_wake_download_tracker", lambda: None)
        # durable 路由省略 client_factory，默认值已绑定；仅换 Provider 工厂，不换执行器。
        monkeypatch.setattr(plans.execute_fs_change_plan, "__kwdefaults__", {
            **(plans.execute_fs_change_plan.__kwdefaults__ or {}), "client_factory": lambda: provider,
        })

        async def run():
            script = FsReobservationScript()
            specs = [adapt_tool_spec(spec) for spec in build_tool_specs() if spec.name in {READ, EXECUTE}]
            assert len(specs) == 2
            catalog = ToolCatalog(specs)
            store = SQLiteKernelStore(secret_provider=lambda: SECRET)
            pipeline = ToolPipeline(
                catalog=catalog, state_store=store, reference_store=store,
                effect_store=ConfirmationEffectPlanStore(SQLiteConfirmationStore(ttl_seconds=300), record_actions=True),
                authorization=MediaFluxAuthorizationPolicy(), rate_limiter=MediaFluxToolRateLimiter(),
                effect_lifecycle=MediaFluxEffectLifecycle(),
            )
            session = AgentSession(
                model=script.adapter, catalog=catalog, retriever=CapabilityRetriever(minimum=2, maximum=2),
                pipeline=pipeline, state_store=store, journal=store, turn_admission=MediaFluxTurnAdmission(),
            )
            initial = [event async for event in session.run(AgentInput(
                owner=OWNER, session_id=SESSION_ID, message=GOAL, request_id="original-user"))]
            assert initial[-1].payload["status"] == "approval_required"
            initial_state = await store.load(owner=OWNER, session_id=SESSION_ID)
            source_users = [row for row in initial_state.conversation if row.get("role") == "user"]
            assert len(source_users) == 1 and source_users[0]["history_id"]
            assert source_users == [{"role": "user", "content": GOAL, "history_id": source_users[0]["history_id"]}]
            old = await pipeline.get_pending_plan(owner=OWNER, session_id=SESSION_ID)
            assert old is not None and old.arguments == script.calls[1]["arguments"]
            source = old.metadata["reobservation_source"]
            assert source["baseline"]["selected"][0]["file_id"] == "move"
            old_observation = source["baseline"]["observation_ref"]
            old_object = script.calls[1]["arguments"]["operations"][0]["object_ref"]
            flow = changes._flow(OWNER)
            assert flow is not None
            frozen = plans._read(flow.plan_id)
            assert frozen["operations"][0]["source"]["name"] == "Move.mp4"
            frozen["expires_at_epoch"] = time.time() - 1
            plans._atomic_write(frozen)
            # 改变离线 Provider 事实，不调用待验证写入口，不伪造失败回执。
            next(row for row in provider.directories["source"] if row.file_id == "move").name = "External.mp4"
            assert workspace.load_directory_observation(old_observation, owner=OWNER)["expires_at_epoch"] > time.time()
            assert old.expires_at > time.time()
            pending = await pipeline.get_pending_plan(owner=OWNER, session_id=SESSION_ID)
            assert pending is not None and pending.plan_id == old.plan_id

            recovered = [event async for event in session.confirm(
                owner=OWNER, session_id=SESSION_ID, plan_id=old.plan_id, request_id="confirm-old")]
            assert recovered[-1].payload["status"] == "approval_required"
            failed = [event for event in recovered if event.type is AgentEventType.EFFECT_FAILED]
            observed = [event for event in recovered if event.type is AgentEventType.TOOL_COMPLETED
                        and event.payload.get("tool") == READ]
            preparations = [event for event in recovered if event.type is AgentEventType.MODEL_TOOL_CALL
                            and event.payload.get("tool") == EXECUTE]
            approvals = [event for event in recovered if event.type is AgentEventType.EFFECT_APPROVAL_REQUIRED]
            assert len(failed) == len(observed) == len(preparations) == len(approvals) == 1
            # 第一次失败续行模型请求之前，执行器已读到稳定身份；不靠脚本先调reobserve。
            first_model = next(event for event in recovered if event.type is AgentEventType.MODEL_STARTED)
            assert recovered.index(observed[0]) < recovered.index(first_model)
            assert all(call.get("step") != "reobserve" for call in script.calls)
            assert failed[0].payload["plan_id"] == old.plan_id
            assert failed[0].payload["result"]["status"] == "confirmation_stale"
            assert "光鸭变更预览已过期" in failed[0].payload["result"]["summary"]
            assert recovered.index(failed[0]) < recovered.index(observed[0]) < recovered.index(preparations[0]) < recovered.index(approvals[0])
            assert len({event.turn_id for event in recovered}) == 1
            current = observed[0].payload["result"]["data"]
            assert current["reobservation"]["complete"] is True
            matched = current["reobservation"]["items"][0]
            assert matched["previous_ref"] == old_object and matched["state"] == "same_identity"
            assert matched["name"] == "External.mp4" and "name" in matched["changed_fields"]
            assert current["observation_ref"] != old_observation and matched["object_ref"] != old_object
            assert script.calls[2]["arguments"]["observation_ref"] == current["observation_ref"]
            assert script.calls[2]["arguments"]["operations"][0]["object_ref"] == matched["object_ref"]
            new = await pipeline.get_pending_plan(owner=OWNER, session_id=SESSION_ID)
            assert new is not None and new.arguments == script.calls[2]["arguments"]
            assert new.plan_id != old.plan_id and new.snapshot_fingerprint != old.snapshot_fingerprint
            assert new.metadata["reobservation_source"]["baseline"]["observation_ref"] == current["observation_ref"]
            assert new.metadata["reobservation_source"]["baseline"]["selected"][0]["file_id"] == "move"
            new_flow = changes._flow(OWNER)
            assert new_flow is not None and new_flow.plan_id != frozen["plan_id"]
            new_frozen = plans._read(new_flow.plan_id)
            assert new_frozen["status"] == "previewed"
            assert new_frozen["operations"][0]["source"]["name"] == "External.mp4"
            assert new_frozen["operations"][0]["object_ref"] == matched["object_ref"]
            assert new_frozen["operations"][0]["new_name"] == "Final.mp4"
            cold = SQLiteKernelStore(secret_provider=lambda: SECRET)
            state = await cold.load(owner=OWNER, session_id=SESSION_ID)
            assert [row for row in state.conversation if row.get("role") == "user"] == source_users
            assert len([row for row in state.conversation if row.get("completion_receipt_id") == old.plan_id]) == 1
            assert state.pending_effect_plan_id == new.plan_id
            assert len(script.requests) == len(script.model_requests) == 3 and not script.steps
            assert [call["tool"] for call in script.calls] == [READ, EXECUTE, EXECUTE]
            rereads = [call for row in state.conversation for call in row.get("tool_calls", [])
                       if call["name"] in {READ, READ.replace(".", "__")}
                       and call["arguments"].get("operation") == "reobserve"]
            assert len(rereads) == 1 and rereads[0]["arguments"] == {"operation": "reobserve"}
            assert any(json.loads(row["content"]).get("status") == "confirmation_stale"
                       for row in script.requests[2]["messages"] if row.get("role") == "tool")
            script.assert_user_inputs([GOAL])
            view = TurnViewBuilder()
            for event in recovered:
                view.apply(event)
            assert view.build().to_dict()["effect_result"] == failed[0].payload["result"]
            assert "reobservation_source" not in json.dumps(script.requests)
            assert provider.file_info("move").name == "External.mp4"
            for write in writes.values():
                write.assert_not_called()
            with db.get_conn() as conn:
                assert conn.execute("SELECT COUNT(*) FROM organize_operation_jobs").fetchone()[0] == 0

            # 新卡仍需另一次明确确认；真实 durable worker 只变更一次。
            script.steps.append("finish")
            completed = [event async for event in session.confirm(
                owner=OWNER, session_id=SESSION_ID, plan_id=new.plan_id, request_id="confirm-new")]
            results = [event for event in completed if event.type is AgentEventType.EFFECT_COMPLETED]
            assert len(results) == 1 and results[0].payload["result"]["ok"] is True
            assert not any(event.type is AgentEventType.EFFECT_FAILED for event in completed)
            writes["rename"].assert_called_once_with("move", "Final.mp4")
            for method, write in writes.items():
                if method != "rename":
                    write.assert_not_called()
            assert provider.file_info("move").name == "Final.mp4"
            finished_plan = plans._read(new_flow.plan_id)
            assert finished_plan["status"] == "completed"
            job = get_organize_operation_job(finished_plan["job_id"])
            assert job is not None and job["status"] == "completed"
            assert json.loads(job["result_json"])["stats"]["renamed"] == 1
            with db.get_conn() as conn:
                assert conn.execute("SELECT COUNT(*) FROM organize_operation_jobs").fetchone()[0] == 1
            final = await cold.load(owner=OWNER, session_id=SESSION_ID)
            assert not final.pending_effect_plan_id
            assert [row for row in final.conversation if row.get("role") == "user"] == source_users
            for plan_id in (old.plan_id, new.plan_id):
                assert len([row for row in final.conversation if row.get("completion_receipt_id") == plan_id]) == 1
            assert len(script.requests) == len(script.model_requests) == 4 and not script.steps
            script.assert_user_inputs([GOAL])

            public = json.dumps(script.requests + [event.to_dict() for event in initial + recovered + completed])
            for private in ("reobservation_source", "source_observations", "source_root", "credential_generation"):
                assert private not in public, f"私有来源不得进入模型或公开执行回执: {private}"

            duplicate = [event async for event in session.confirm(
                owner=OWNER, session_id=SESSION_ID, plan_id=new.plan_id, request_id="duplicate-new")]
            assert duplicate[-1].type is AgentEventType.TURN_FAILED
            assert duplicate[-1].payload["code"] == "confirmation_invalid"
            assert (await cold.load(owner=OWNER, session_id=SESSION_ID)).to_payload() == final.to_payload()
            assert len(script.requests) == len(script.model_requests) == 4
            writes["rename"].assert_called_once_with("move", "Final.mp4")
            with db.get_conn() as conn:
                assert conn.execute("SELECT COUNT(*) FROM organize_operation_jobs").fetchone()[0] == 1

        try:
            asyncio.run(run())
        finally:
            assert manager.shutdown(timeout=2.0), "离线 worker 必须在释放临时数据库前结束"
            files.reset_guangya_workspace_context_for_tests()
            changes.reset_guangya_fs_change_context_for_tests()
