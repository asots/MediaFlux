"""Cold cancellation must revoke publication before a late producer returns."""
import asyncio
from contextlib import suppress

import pytest

from app import database as db
from app.agent.kernel.bootstrap import build_agent_kernel_runtime
from app.agent.kernel.model import ModelEvent, ModelEventType
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.state_types import AgentInput, PublicationLease, StateUpdate
from app.agent.task_stop import stop_agent_session


@pytest.mark.parametrize("stop_session", [False, True])
def test_cold_cancel_revokes_live_request_and_late_output(stop_session):
    class HeldModel:
        def __init__(self):
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request, *, cancellation):
            self.entered.set()
            await self.release.wait()
            yield ModelEvent(ModelEventType.TEXT_DELTA, text="late answer must not survive")
            yield ModelEvent(ModelEventType.FINISH, finish_reason="stop")

    async def run():
        db.init_db()
        model = HeldModel()
        first = build_agent_kernel_runtime(model=model, store=SQLiteKernelStore())
        cold = build_agent_kernel_runtime(model=HeldModel(), store=SQLiteKernelStore())
        request = AgentInput(owner='webk:v1:' + 'a' * 64, session_id='cold-cancel',
                             request_id='source-request', message='查看媒体状态')
        async def consume():
            return [event async for event in first.session.run(request)]
        worker = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(model.entered.wait(), 5)
            active = await first.session.coordinator.describe(owner=request.owner, session_id=request.session_id)
            before = await cold.store.load(owner=request.owner, session_id=request.session_id)
            _, other = await cold.store.begin_turn(owner=request.owner, session_id="other", request_id="other-request")
            assert not await cold.session.cancel(owner=request.owner, session_id=request.session_id, request_id='wrong')
            assert (await cold.store.load(owner=request.owner, session_id=request.session_id)) == before
            if stop_session:
                result = await stop_agent_session(cold, request.owner, request.session_id)
                assert result["status"] == "stopping"
                assert result["stopped"] is False
            else:
                assert await cold.session.cancel(owner=request.owner, session_id=request.session_id, request_id=request.request_id)
            assert active is not None
            lease = PublicationLease(request.owner, request.session_id, active['generation'], active['turn_id'], request.request_id)
            assert not await cold.store.is_current(lease)
            after = await cold.store.load(owner=request.owner, session_id=request.session_id)
            assert after.generation > before.generation
            assert after.conversation == before.conversation
            assert not await cold.session.cancel(owner=request.owner, session_id=request.session_id, request_id=request.request_id)
            assert await cold.store.load(owner=request.owner, session_id="other") == other
            if stop_session:
                repeated = await stop_agent_session(cold, request.owner, request.session_id)
                assert repeated["stopped"] is True
                assert await cold.store.load(owner=request.owner, session_id=request.session_id) == after
            model.release.set()
            events = await asyncio.wait_for(worker, 5)
            assert not any(event.type.value in {'model.delta', 'turn.completed'} for event in events)
            final = await cold.store.load(owner=request.owner, session_id=request.session_id)
            assert 'late answer' not in str(final.conversation)
            assert final.generation == after.generation
        finally:
            model.release.set()
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(worker, 5)
    asyncio.run(run())


def test_finished_turn_is_not_a_cold_cancellation_target():
    class Model:
        async def stream(self, request, *, cancellation):
            yield ModelEvent(ModelEventType.TEXT_DELTA, text='done')
            yield ModelEvent(ModelEventType.FINISH, finish_reason='stop')
    async def run():
        db.init_db()
        runtime = build_agent_kernel_runtime(model=Model(), store=SQLiteKernelStore())
        request = AgentInput(owner='webk:v1:' + 'b' * 64, session_id='finished', request_id='finished-request', message='你好')
        events = [e async for e in runtime.session.run(request)]
        assert events[-1].payload.get('status') == 'success'
        before = await runtime.store.load(owner=request.owner, session_id=request.session_id)
        assert not await runtime.session.cancel(owner=request.owner, session_id=request.session_id, request_id=request.request_id)
        assert (await runtime.store.load(owner=request.owner, session_id=request.session_id)) == before
    asyncio.run(run())


def test_business_update_cannot_forge_cancellation_identity():
    async def run():
        db.init_db()
        store = SQLiteKernelStore()
        lease, _ = await store.begin_turn(owner='owner', session_id='forgery', request_id='real')
        try:
            await store.commit(lease, updates=(StateUpdate('metadata.active_publication', {'request_id': 'forged'}),))
        except ValueError:
            return
        assert False, 'business state update forged durable cancellation authority'
    asyncio.run(run())
