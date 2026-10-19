"""后台完成身份贯穿真实准入、worker、SQLite与冷回执。"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import unittest
from contextlib import ExitStack
from typing import cast
from unittest.mock import patch

import httpx

from app import database as db
from app.agent import completion_observers as observers
from app.agent import effect_completion as effects
from app.agent.confirmation import SQLiteConfirmationStore
from app.agent.domain_catalog import DomainActions, build_tool_specs
from app.agent.effect_completion import wait_for_effect_completion
from app.agent.feature_gate import initialize_runtime_fence
from app.agent.kernel.bootstrap import _capture_creation_binding
from app.agent.kernel.capabilities import CapabilityRetriever
from app.agent.kernel.completion_receipts import _completion_cipher
from app.agent.kernel.effects import ConfirmationEffectPlanStore
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports import existing_actions
from app.agent.kernel.ports.mediaflux_effects import MediaFluxEffectLifecycle
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state_types import AgentInput
from app.modules.process_lock import CrossProcessLock
from app.modules.scheduler import STRMScheduler
from tests.support import isolated_test_database
from tests.test_agent_task_first_slice import NativeScript

class CompletionScript(NativeScript):
    def __init__(self, calls):
        super().__init__("fixture")
        self.audit_calls = list(calls)

    async def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        if self.audit_calls:
            name, args = self.audit_calls.pop(0)
            message = {"tool_calls": [{"id": f"audit-{len(self.requests)}", "type": "function",
                "function": {"name": name.replace(".", "__"), "arguments": json.dumps(args)}}]}
        else:
            message = {"content": "请以实际任务回执为准。"}
        return httpx.Response(200, json={"choices": [{"message": message,
            "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}]})


def completion_session(spec_names, calls, *, actions=None):
    store = SQLiteKernelStore()
    source = build_tool_specs(domain_actions=actions or DomainActions())
    from app.agent.kernel.ports import catalog_from_tool_specs
    catalog = catalog_from_tool_specs([s for s in source if s.name in spec_names])
    script = CompletionScript([("agent.capabilities", {"tool_names": list(spec_names)}), *calls])
    session = AgentSession(model=script.adapter, catalog=catalog,
        retriever=CapabilityRetriever(minimum=1, maximum=8), state_store=store, journal=store,
        pipeline=ToolPipeline(catalog=catalog, state_store=store, reference_store=store,
            effect_store=ConfirmationEffectPlanStore(SQLiteConfirmationStore()),
            effect_lifecycle=MediaFluxEffectLifecycle()))
    session.continuation_binding = _capture_creation_binding()
    return session, store, script


async def assert_strm_identity(later_count: int, later_source: str):
    with isolated_test_database(), ExitStack() as stack:
        initialize_runtime_fence()
        scheduler = STRMScheduler()
        scheduler._run_lock = cast(CrossProcessLock, threading.Lock())
        sources = [{"id": key, "name": key.upper(), "path": "/" + key, "namespace": key}
                   for key in ("a", "b")]
        stack.enter_context(patch("app.modules.scheduler.get_scheduler", return_value=scheduler))
        stack.enter_context(patch("app.modules.strm.configured_strm_source_plans", return_value=(sources, "")))
        stack.enter_context(patch.object(scheduler, "validate_config", return_value=""))
        stack.enter_context(patch.object(scheduler, "_source_dirs", return_value=sources))
        stack.enter_context(patch.object(scheduler, "_refresh_media_servers", return_value={}))
        stack.enter_context(patch.object(scheduler, "_notify_scoped_results"))
        stack.enter_context(patch.object(scheduler, "_notify_scoped_failure"))
        original_trigger = scheduler.trigger
        def fail_original(*args, **kwargs):
            # Actual STRMScheduler admission, worker, task_run creation and failure finalization.
            with patch.object(scheduler, "validate_config", return_value="offline provider failed"):
                accepted = original_trigger(*args, **kwargs)
                worker = scheduler._worker
                if worker:
                    worker.join(5)
                    assert not worker.is_alive()
                return accepted
        write = "strm.run_once"
        session, store, script = completion_session([write], [(write, {"source_names": ["A"]})])
        context = {"owner": "native-strm-owner", "session_id": "native-strm-session"}
        preview = [e async for e in session.run(AgentInput(**context, request_id="strm-preview",
            message="只同步 STRM 来源 A 并核验最终结果。", channel="web"))]
        assert preview[-1].payload["status"] == "approval_required", preview[-1].payload
        plan_id = (await store.load(**context)).pending_effect_plan_id
        async def transient_read(result, **options):
            # A failed first observation is an existing supported durable-handoff path.
            # It cannot create a terminal result or mark this task delivered.
            with patch.object(observers, "_poll", side_effect=RuntimeError("transient reader unavailable")):
                return await wait_for_effect_completion(result, timeout_seconds=0, **options)
        with patch.object(scheduler, "trigger", fail_original), patch.object(
                existing_actions, "wait_for_effect_completion", transient_read):
            confirmed = [e async for e in session.confirm(**context, plan_id=plan_id,
                request_id="strm-confirm", channel="web")]
        assert confirmed[-1].payload["status"] == "waiting"
        before = await SQLiteKernelStore().load(**context)
        wait = before.metadata["effect_waits"][plan_id]
        assert not wait["delivered"] and wait["state"] == "pending" and "final_result" not in wait
        seed = json.loads(_completion_cipher().decrypt(wait["sealed"].encode()))
        assert (seed["owner"], seed["session_id"], seed["plan_id"], seed["request_id"]) == (
            context["owner"], context["session_id"], plan_id, "strm-confirm")
        original = db.list_task_runs("strm_sync", limit=1)[0]
        assert original["status"] == "failed"
        for index in range(later_count):
            # Real independent manual runs; fast + no queued changes is a real successful no-op.
            accepted = original_trigger("manual", sync_mode="fast", selected_source_ids=[later_source])
            assert accepted["ok"]
            worker = scheduler._worker
            if worker:
                worker.join(5)
                assert not worker.is_alive()
            assert db.list_task_runs("strm_sync", limit=1)[0]["status"] == "success"
        future = time.time() + 11
        cold = SQLiteKernelStore(clock=lambda: future)
        with patch.object(effects.time, "time", return_value=future):
            count = await effects.poll_effect_receipts(cold)
        assert count == 1
        after = await cold.load(**context)
        final = after.metadata["effect_waits"][plan_id]["final_result"]
        native_tool = next(r for r in after.conversation if r.get("role") == "tool" and r.get("effect_plan_id") == plan_id)
        assert json.loads(native_tool["content"]) == final
        assert after.metadata["effect_waits"][plan_id]["delivered"]
        assert final["status"] == "failed"
        assert not final["ok"]
        assert seed["tracker"]["value"]["run_id"] == original["id"]


class AgentCompletionIdentityTests(unittest.TestCase):
    def test_strm_failed_run_survives_later_runs_and_cold_receipt(self):
        for later_count, later_source in ((0, "a"), (19, "a"), (20, "a"), (20, "b")):
            with self.subTest(later_count=later_count, later_source=later_source):
                asyncio.run(assert_strm_identity(later_count, later_source))
