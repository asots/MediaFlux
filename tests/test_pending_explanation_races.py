import asyncio

import pytest

from app.agent.kernel.adapters import consume_events
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.state_types import AgentInput
from tests.support import isolated_test_database
from tests.test_pending_explanation_transports import NativeRuntime


@pytest.mark.parametrize("action", ["stop", "read", "cancel_plan", "expire"])
def test_explanation_does_not_own_or_refresh_plan(action):
    async def run():
        runtime = NativeRuntime()
        owner, sid = "owner", "explain-race"
        preview = await consume_events(
            runtime.session.run(
                AgentInput(message="改名一个对象", owner=owner, session_id=sid)
            )
        )
        plan_id = preview.approval.plan_id
        frozen = await runtime.session.pipeline.get_pending_plan(
            owner=owner, session_id=sid
        )
        entered, release = asyncio.Event(), asyncio.Event()
        runtime.gate = entered, release
        task = asyncio.create_task(
            consume_events(
                runtime.session.explain(
                    owner=owner,
                    session_id=sid,
                    plan_id=plan_id,
                    message="解释影响",
                    request_id="explain",
                )
            )
        )
        await asyncio.wait_for(entered.wait(), 2)
        if action == "stop":
            assert await runtime.session.cancel(
                owner=owner, session_id=sid, request_id="explain"
            )
        elif action == "read":
            read = await consume_events(
                runtime.session.run(
                    AgentInput(message="查看别的内容", owner=owner, session_id=sid)
                )
            )
            assert read.status == "success", read
            assert read.tool_calls == () and read.approval is None
        elif action == "cancel_plan":
            assert await runtime.session.cancel_effect(
                owner=owner, session_id=sid, plan_id=plan_id
            )
        elif action == "expire":
            runtime.now = frozen.expires_at + 1
        release.set()
        await asyncio.wait_for(task, 2)
        state = await SQLiteKernelStore().load(owner=owner, session_id=sid)
        assert runtime.writes == [] and len(runtime.prepares) == 1
        if action in ("stop", "read"):
            assert state.pending_effect_plan_id == plan_id
            assert (
                await runtime.session.pipeline.get_pending_plan(
                    owner=owner, session_id=sid
                )
                == frozen
            )
        else:
            assert (
                await runtime.session.pipeline.get_pending_plan(
                    owner=owner, session_id=sid
                )
                is None
            )
        result = await consume_events(
            runtime.session.confirm(owner=owner, session_id=sid, plan_id=plan_id)
        )
        replay = await consume_events(
            runtime.session.confirm(owner=owner, session_id=sid, plan_id=plan_id)
        )
        assert replay.status == "failed" and replay.error_code == "confirmation_invalid"
        assert runtime.writes == (
            [({"step": 1}, "frozen-1")] if action in ("stop", "read") else []
        )
        assert result.status == ("success" if action in ("stop", "read") else "failed")

    with isolated_test_database("explain-race.db"):
        asyncio.run(run())
