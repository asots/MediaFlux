"""An unguarded generation change must not inherit a frozen write ticket."""
import asyncio

import pytest

from app import database as db
from app.agent.kernel.state import StalePublicationError
from tests.test_agent_kernel_session_guard import (
    OWNER, SESSION, confirmation, prepare, persistent_runtime,
    isolated_chain as isolated_chain,
)
from tests.test_agent_ux_backend import _runtime


@pytest.mark.parametrize("sqlite", [False, True])
def test_unguarded_turn_revokes_old_confirmation_authority(isolated_chain, sqlite):
    async def run():
        writer, pipeline, store = persistent_runtime() if sqlite else _runtime()
        plan, old = await prepare(writer, pipeline, store)
        other = persistent_runtime()[2] if sqlite else store
        new, _ = await other.begin_turn(
            owner=OWNER, session_id=SESSION, request_id="unguarded-new-turn",
        )
        events = await confirmation(writer, plan)
        with db.get_conn() as conn:
            rows = conn.execute("SELECT COUNT(*) FROM download_requests").fetchone()[0]
        observed = {
            "handler_calls": isolated_chain[0].call_count,
            "download_rows": rows,
            "new_current": await other.is_current(new),
            "confirmation_invalid": any(e.payload.get("code") == "confirmation_invalid" for e in events),
        }
        assert observed == {
            "handler_calls": 0, "download_rows": 0,
            "new_current": True, "confirmation_invalid": True,
        }
        with pytest.raises(StalePublicationError):
            await other.commit(old, conversation=[])
    asyncio.run(run())
