"""Cold cancellation fences executor failures at journal and delivery boundaries."""
import asyncio
import threading
from contextlib import suppress
from unittest.mock import patch

import pytest

from app.agent.kernel.bootstrap import build_agent_kernel_runtime, _capture_creation_binding
from app.agent.kernel.persistence import SQLiteKernelStore
from tests.test_agent_read_wait_lifecycle import (
    OWNER, WaitScript, parked_runtime, isolated_database as isolated_database,
)


@pytest.mark.parametrize("phase", ["before_append", "after_append", "not_cancelled"])
def test_cold_cancel_blocks_late_terminal_failure(phase):
    async def run():
        clock = [100]
        _, candidate = await parked_runtime(clock)
        clock[0] = 110
        script = WaitScript(True)
        script.prefix = "initial"  # Real adapter returns the forbidden old native ID.
        store = SQLiteKernelStore(clock=lambda: clock[0])
        runtime = build_agent_kernel_runtime(model=script.adapter, store=store)
        runtime.session.read_wait_binding = _capture_creation_binding()
        entered, release = threading.Event(), threading.Event()
        append = store._append_event_sync
        delivered = []
        background = runtime.session._run_background

        async def observe(*args, **kwargs):
            async for event in background(*args, **kwargs):
                delivered.append(event.type.value)
                yield event

        def gate(event, owner, *args):
            if event.type.value != "turn.failed":
                return append(event, owner, *args)
            if phase == "after_append":
                append(event, owner, *args)
            entered.set()
            assert release.wait(5), "failure journal release timeout"
            if phase != "after_append":
                return append(event, owner, *args)

        with patch.object(store, "_append_event_sync", side_effect=gate), patch.object(
            runtime.session, "_run_background", side_effect=observe,
        ):
            worker = asyncio.create_task(runtime.session.resume_wait(candidate=candidate))
            try:
                assert await asyncio.to_thread(entered.wait, 5), "failure journal entry timeout"
                cold = build_agent_kernel_runtime(
                    model=WaitScript(True).adapter,
                    store=SQLiteKernelStore(clock=lambda: clock[0]),
                )
                if phase != "not_cancelled":
                    assert await cold.session.cancel(owner=OWNER, session_id="wait-control", request_id="source")
                release.set()
                await asyncio.wait_for(worker, 5)
                events = await cold.store.list_events(owner=OWNER, session_id="wait-control")
                failures = [e for e in events if e["type"] == "turn.failed"]
                assert len(failures) == (0 if phase == "before_append" else 1)
                assert ("turn.failed" in delivered) is (phase == "not_cancelled")
                assert "turn.started" in delivered
            finally:
                release.set()
                worker.cancel()
                with suppress(asyncio.CancelledError):
                    await asyncio.wait_for(worker, 5)
    asyncio.run(run())
