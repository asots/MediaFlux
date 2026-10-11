"""Native quota rejection closes its call and parks the tracked task atomically."""
import asyncio
import json
from dataclasses import replace
from datetime import datetime
from unittest.mock import patch

import httpx
import pytest
from app import database as db
from app.agent.feature_gate import initialize_runtime_fence
from app.agent.kernel.bootstrap import build_agent_kernel_runtime, _capture_creation_binding
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.state import AgentInput
from app.agent.kernel.ports.mediaflux_policy import ToolRateLimitError
from app.agent.rate_limit import RateLimitDecision
from tests.test_agent_task_first_slice import NativeScript, declaration, GOAL, OWNER


@pytest.fixture(autouse=True)
def isolated_database(tmp_path):
    previous = db.DB_PATH, bool(getattr(db, '_configured_test_mode', False))
    try:
        db.configure_database(tmp_path / 'wait.db', test_mode=True)
        yield
    finally:
        db.configure_database(previous[0], test_mode=previous[1])


class WaitScript(NativeScript):
    def __init__(self, resume=False):
        super().__init__('tmdb')
        self.steps = ['read', 'finish'] if resume else ['discover', 'create', 'read']
        self.prefix = 'resume' if resume else 'initial'

    async def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        step = self.steps.pop(0)
        if step == 'finish':
            message = {'content': '已取得真实查询结果。'}
        else:
            name, args = {
                'discover': ('agent__capabilities', {'tool_names': ['agent.task', 'web.search']}),
                'create': ('agent__task', declaration()),
                'read': ('web__search', {'query': 'quota fixture'}),
            }[step]
            message = {'content': '', 'tool_calls': [{'id': self.prefix + '-' + step, 'type': 'function',
                       'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}]}
        return httpx.Response(200, json={'choices': [{'message': message, 'finish_reason': 'stop' if step == 'finish' else 'tool_calls'}]})


class Quota:
    async def acquire(self, *, tool_name, **kwargs):
        if tool_name == 'web.search':
            raise ToolRateLimitError(RateLimitDecision(False, 6, 1, 0, retry_after_seconds=10,
                                                      retry_at_epoch=110, reason='quota'))


def test_native_quota_park_and_cold_cancel():
    async def run():
        db.init_db()
        initialize_runtime_fence()
        script = WaitScript()
        store = SQLiteKernelStore(clock=lambda: 100)
        runtime = build_agent_kernel_runtime(model=script.adapter, store=store)
        # Explicit trusted injection; ordinary model metadata cannot supply authority.
        runtime.session.read_wait_binding = _capture_creation_binding()
        runtime.session.pipeline.rate_limiter = Quota()
        events = [e async for e in runtime.session.run(AgentInput(message=GOAL, owner=OWNER,
                            session_id='wait-park', request_id='wait-source'))]
        assert events[-1].payload.get('status') == 'waiting', [(e.type.value, dict(e.payload)) for e in events[-3:]]
        cold = SQLiteKernelStore(clock=lambda: 100)
        state = await cold.load(owner=OWNER, session_id='wait-park')
        record = state.metadata['read_wait']
        assert record['state'] == 'pending'
        assert record['budget']['model_spent'] == 3
        assert record['budget']['tool_spent'] == 3
        assert record['budget']['wait_segments'] == 1
        assert record['due_at'] == 110
        assert record['expires_at'] == 400
        results = [r for r in state.conversation if r.get('tool_call_id') == 'initial-read']
        assert len(results) == 1 and 'rate_limited' in results[0]['content']
        fresh = build_agent_kernel_runtime(model=WaitScript(True).adapter, store=cold)
        assert await fresh.session.cancel(owner=OWNER, session_id='wait-park', request_id='wait-source')
        cancelled = await cold.load(owner=OWNER, session_id='wait-park')
        assert cancelled.metadata['read_wait']['state'] == 'cancelled'
        assert cancelled.metadata['read_wait']['budget'] == record['budget']
        assert cancelled.metadata['task'] == state.metadata['task']
        assert cancelled.conversation == state.conversation
    asyncio.run(run())


def test_scheduler_cold_native_resume_uses_new_id_and_one_read():
    from app.modules.agent_jobs_scheduler import AgentJobsScheduler
    timer = [100]
    async def park():
        db.init_db()
        initialize_runtime_fence()
        script = WaitScript()
        runtime = build_agent_kernel_runtime(model=script.adapter, store=SQLiteKernelStore(clock=lambda: timer[0]))
        runtime.session.read_wait_binding = _capture_creation_binding()
        runtime.session.pipeline.rate_limiter = Quota()
        events = [e async for e in runtime.session.run(AgentInput(message=GOAL, owner=OWNER,
                          session_id='wait-resume', request_id='resume-source'))]
        assert events[-1].payload.get('status') == 'waiting'
    asyncio.run(park())
    timer[0] = 110
    resumed = WaitScript(True)
    handlers = []
    def factory(candidate, binding, store):
        runtime = build_agent_kernel_runtime(model=resumed.adapter, store=store)
        runtime.session.read_wait_binding = binding
        async def read(args, context):
            handlers.append(args)
            return {'ok': True, 'summary': 'fixture READ complete'}
        tool = runtime.session.catalog.get('web.search')
        runtime.session.catalog._tools['web.search'] = replace(tool, read=read)
        return replace(runtime, creation_binding=binding)
    scheduler = AgentJobsScheduler(clock=lambda: datetime.fromtimestamp(timer[0]))
    scheduler._read_wait_runtime_factory = factory
    assert getattr(scheduler, '_run_due_read_wait_once', lambda: 0)() == 1
    assert len(handlers) == 1
    assert len(resumed.requests) == 2
    assert [t['function']['name'] for t in resumed.requests[0]['tools']] == ['web__search']
    assert not resumed.requests[1].get('tools')
    old_wire = [r for r in resumed.requests[0]['messages'] if r.get('tool_call_id') == 'initial-read']
    assert len(old_wire) == 1 and 'rate_limited' in old_wire[0]['content']
    async def verify():
        state = await SQLiteKernelStore().load(owner=OWNER, session_id='wait-resume')
        assert state.metadata['read_wait']['state'] == 'terminal'
        assert state.metadata['read_wait']['terminal_reason'] == 'completed'
        budget = state.metadata['read_wait']['budget']
        assert budget['model_spent'] == 5 and budget['tool_spent'] == 4
        assert budget['wait_seconds_spent'] == 10
        results = {r.get('tool_call_id'): r for r in state.conversation if r.get('role') == 'tool'}
        assert 'rate_limited' in results['initial-read']['content']
        assert 'fixture READ complete' in results['resume-read']['content']
        assert results['initial-read'] != results['resume-read']
    asyncio.run(verify())
    assert scheduler._run_due_read_wait_once() == 0


@pytest.mark.parametrize('bad', ['write', 'arguments', 'old_id', 'empty', 'multiple'])
def test_illegal_resume_is_terminal_without_business_io(bad):
    async def run():
        timer = [100]
        _, candidate = await parked_runtime(timer)
        timer[0] = 110
        class Invalid(WaitScript):
            async def respond(self, request):
                response = await super().respond(request)
                body = response.json()
                message = body['choices'][0]['message']
                if self.steps == ['finish']:
                    call = message['tool_calls'][0]
                    if bad == 'write':
                        call['function']['name'] = 'ingest__submit'
                    elif bad == 'arguments':
                        call['function']['arguments'] = '{"query":"different"}'
                    elif bad == 'old_id':
                        call['id'] = 'initial-read'
                    elif bad == 'multiple':
                        message['tool_calls'].append({**call, 'id': 'extra'})
                    elif bad == 'empty':
                        message.clear()
                        message['content'] = 'I am done'
                return httpx.Response(200, json=body)
        script = Invalid(True)
        runtime = build_agent_kernel_runtime(model=script.adapter, store=SQLiteKernelStore(clock=lambda: timer[0]))
        runtime.session.read_wait_binding = _capture_creation_binding()
        async def forbidden(*args, **kwargs):
            raise AssertionError('business I/O forbidden')
        with patch.object(runtime.session.pipeline, 'execute', side_effect=forbidden):
            assert await runtime.session.resume_wait(candidate=candidate)
        state = await runtime.store.load(owner=OWNER, session_id='wait-control')
        assert state.metadata['read_wait']['state'] == 'terminal'
        assert state.metadata['read_wait']['terminal_reason'] == 'read_resume_invalid'
        assert len(script.requests) == 1
    asyncio.run(run())


@pytest.mark.parametrize('advance', [180, 301, -1])
def test_cold_wait_expiry_or_clock_reversal_never_requests_model(advance):
    async def run():
        timer = [100]
        _, candidate = await parked_runtime(timer)
        timer[0] += advance
        script = WaitScript(True)
        runtime = build_agent_kernel_runtime(model=script.adapter, store=SQLiteKernelStore(clock=lambda: timer[0]))
        runtime.session.read_wait_binding = _capture_creation_binding()
        assert not await runtime.session.resume_wait(candidate=candidate)
        state = await runtime.store.load(owner=OWNER, session_id='wait-control')
        assert state.metadata['read_wait']['state'] == 'terminal'
        assert len(script.requests) == 0
    asyncio.run(run())


def test_claim_exclusion_running_never_replays_after_lease():
    from app.agent.kernel.model import ModelEvent, ModelEventType
    from contextlib import suppress
    async def run():
        timer = [100]
        _, candidate = await parked_runtime(timer)
        timer[0] = 110
        entered, release = asyncio.Event(), asyncio.Event()
        class Held:
            async def stream(self, request, *, cancellation):
                entered.set()
                await release.wait()
                yield ModelEvent(ModelEventType.TEXT_DELTA, text='not a read')
                yield ModelEvent(ModelEventType.FINISH, finish_reason='stop')
        runtime = build_agent_kernel_runtime(model=Held(), store=SQLiteKernelStore(clock=lambda: timer[0]))
        runtime.session.read_wait_binding = _capture_creation_binding()
        worker = asyncio.create_task(runtime.session.resume_wait(candidate=candidate))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            cold = SQLiteKernelStore(clock=lambda: timer[0])
            timer[0] = 151
            assert await cold.claim_read_wait(candidate, runtime.session.read_wait_binding) is None
            assert await cold.due_read_waits() == []
            state = await cold.load(owner=OWNER, session_id='wait-control')
            assert state.metadata['read_wait']['state'] == 'running'
            assert state.metadata['read_wait']['budget']['model_spent'] == 4
        finally:
            release.set()
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(worker, 5)
    asyncio.run(run())


def test_repark_segments_and_task_budgets_survive_cold_runtimes():
    async def run():
        timer = [100]
        runtime, candidate = await parked_runtime(timer)
        for index in range(3):
            timer[0] += 10
            script = WaitScript(True)
            script.prefix = f'resume-{index}'
            fresh = build_agent_kernel_runtime(model=script.adapter, store=SQLiteKernelStore(clock=lambda: timer[0]))
            fresh.session.read_wait_binding = _capture_creation_binding()
            class Again:
                async def acquire(self, **kwargs):
                    raise ToolRateLimitError(RateLimitDecision(False, 6, 1, 0, retry_after_seconds=10,
                                                              retry_at_epoch=timer[0]+10, reason='quota'))
            fresh.session.pipeline.rate_limiter = Again()
            assert await fresh.session.resume_wait(candidate=candidate)
            state = await fresh.store.load(owner=OWNER, session_id='wait-control')
            candidate = state.metadata['read_wait']
            assert candidate['budget']['wait_segments'] == min(3, index + 2)
            assert candidate['budget']['tool_spent'] == index + 4
            assert candidate['expires_at'] == 400
            assert candidate['state'] == ('terminal' if index == 2 else 'pending')
        assert candidate['budget']['wait_seconds_spent'] == 30
    asyncio.run(run())


@pytest.mark.parametrize('change', ['foreground', 'fence'])
def test_new_turn_or_durable_epoch_revokes_old_wait(change):
    from app.agent.feature_gate import begin_runtime_change, read_runtime_fence
    async def run():
        timer = [100]
        runtime, candidate = await parked_runtime(timer)
        timer[0] = 110
        script = WaitScript(True)
        fresh = build_agent_kernel_runtime(model=script.adapter, store=SQLiteKernelStore(clock=lambda: timer[0]))
        fresh.session.read_wait_binding = _capture_creation_binding()
        if change == 'foreground':
            script.steps = ['finish']
            events = [e async for e in fresh.session.run(AgentInput(message='继续', owner=OWNER,
                          session_id='wait-control', request_id='new-request'))]
            assert events[-1].payload.get('status') == 'success'
            after = await fresh.store.load(owner=OWNER, session_id='wait-control')
            assert after.metadata['read_wait']['budget']['model_spent'] == 4
            assert after.metadata['read_wait']['budget']['tool_spent'] == 3
            count = len(script.requests)
            assert not await fresh.session.resume_wait(candidate=candidate)
            assert len(script.requests) == count
        else:
            with db.get_conn() as conn:
                conn.execute('BEGIN IMMEDIATE')
                begin_runtime_change(conn, read_runtime_fence(conn))
            try:
                await fresh.session.resume_wait(candidate=candidate)
            except Exception:
                pass
            assert len(script.requests) == 0
    asyncio.run(run())


def test_scheduler_fault_branch_does_not_block_jobs_or_receipts():
    from app.modules.agent_jobs_scheduler import AgentJobsScheduler
    seen = []
    scheduler = AgentJobsScheduler()
    async def receipt(*args, **kwargs):
        seen.append('receipt')
        raise RuntimeError('fixture receipt error')
    def job():
        seen.append('job')
        return 0
    def read():
        seen.append('read')
        scheduler._stop_event.set()
        scheduler._wake_event.set()
        raise RuntimeError('fixture read error')
    with patch('app.agent.effect_completion.poll_effect_receipts', side_effect=receipt), patch.object(scheduler, 'run_once', side_effect=job), patch.object(scheduler, '_run_due_read_wait_once', side_effect=read, create=True), patch.object(scheduler._wake_event, 'wait', side_effect=lambda **kw: scheduler._stop_event.set()):
        scheduler._loop()
    assert seen == ['receipt', 'job', 'read']
    assert scheduler._thread is None


def test_cancel_during_journal_io_cannot_publish_late_tool_event():
    import threading
    from contextlib import suppress
    async def run():
        timer = [100]
        _, candidate = await parked_runtime(timer)
        timer[0] = 110
        store = SQLiteKernelStore(clock=lambda: timer[0])
        script = WaitScript(True)
        runtime = build_agent_kernel_runtime(model=script.adapter, store=store)
        runtime.session.read_wait_binding = _capture_creation_binding()
        async def read(args, context):
            return {'ok': True, 'summary': 'read-before-cancel'}
        tool = runtime.session.catalog.get('web.search')
        runtime.session.catalog._tools['web.search'] = replace(tool, read=read)
        entered, release = threading.Event(), threading.Event()
        original = store._append_event_sync
        def delayed(event, owner, *args):
            if event.type.value == 'tool.completed' and event.payload.get('call_id') == 'resume-read':
                entered.set()
                assert release.wait(5), 'journal release timeout'
            return original(event, owner, *args)
        with patch.object(store, '_append_event_sync', side_effect=delayed):
            worker = asyncio.create_task(runtime.session.resume_wait(candidate=candidate))
            try:
                assert await asyncio.to_thread(entered.wait, 5), 'journal entry timeout'
                cold = build_agent_kernel_runtime(model=WaitScript(True).adapter, store=SQLiteKernelStore(clock=lambda: timer[0]))
                assert await cold.session.cancel(owner=OWNER, session_id='wait-control', request_id='source')
                release.set()
                await asyncio.wait_for(worker, 5)
                events = await cold.store.list_events(owner=OWNER, session_id='wait-control')
                assert not any(e['type'] == 'tool.completed' and e['payload'].get('call_id') == 'resume-read' for e in events)
                assert len(script.requests) == 1
            finally:
                release.set()
                worker.cancel()
                with suppress(asyncio.CancelledError):
                    await asyncio.wait_for(worker, 5)
    asyncio.run(run())


async def parked_runtime(timer, session_id='wait-control'):
    db.init_db()
    initialize_runtime_fence()
    script = WaitScript()
    runtime = build_agent_kernel_runtime(model=script.adapter, store=SQLiteKernelStore(clock=lambda: timer[0]))
    runtime.session.read_wait_binding = _capture_creation_binding()
    runtime.session.pipeline.rate_limiter = Quota()
    events = [e async for e in runtime.session.run(AgentInput(message=GOAL, owner=OWNER,
                      session_id=session_id, request_id='source'))]
    assert events[-1].payload.get('status') == 'waiting'
    state = await runtime.store.load(owner=OWNER, session_id=session_id)
    return runtime, state.metadata['read_wait']


@pytest.mark.parametrize('phase', ['acquire', 'handler'])
def test_cold_cancel_blocks_handler_start_and_late_reference(phase):
    from contextlib import suppress
    from app.agent.kernel.projection import ToolOutcome, ReferenceValue
    async def run():
        timer = [100]
        runtime, candidate = await parked_runtime(timer)
        timer[0] = 110
        script = WaitScript(True)
        fresh = build_agent_kernel_runtime(model=script.adapter, store=SQLiteKernelStore(clock=lambda: timer[0]))
        fresh.session.read_wait_binding = _capture_creation_binding()
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        class Limiter:
            async def acquire(self, **kwargs):
                if phase == 'acquire':
                    entered.set()
                    await release.wait()
        async def read(args, context):
            calls.append(args)
            if phase == 'handler':
                entered.set()
                await release.wait()
            return ToolOutcome(model_content='late private result', public_content={'ok': True},
                               refs=(ReferenceValue(kind='late_ref', value={'late': True}),))
        fresh.session.pipeline.rate_limiter = Limiter()
        tool = fresh.session.catalog.get('web.search')
        fresh.session.catalog._tools['web.search'] = replace(tool, read=read)
        worker = asyncio.create_task(fresh.session.resume_wait(candidate=candidate))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            cold = build_agent_kernel_runtime(model=WaitScript(True).adapter, store=SQLiteKernelStore(clock=lambda: timer[0]))
            assert await cold.session.cancel(owner=OWNER, session_id='wait-control', request_id='source')
            stopped = await cold.store.load(owner=OWNER, session_id='wait-control')
            release.set()
            await asyncio.wait_for(worker, 5)
            assert len(calls) == (0 if phase == 'acquire' else 1)
            with db.get_conn() as conn:
                assert conn.execute("SELECT count(*) FROM agent_kernel_refs WHERE kind='late_ref'").fetchone()[0] == 0
            assert (await cold.store.load(owner=OWNER, session_id='wait-control')) == stopped
            assert len(script.requests) == 1
        finally:
            release.set()
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(worker, 5)
    asyncio.run(run())
