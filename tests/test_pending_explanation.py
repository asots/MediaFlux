import asyncio

import pytest

from app.agent.kernel.state import SessionState, StateUpdate, _apply_effect_state_change


def test_reserved_binding_cannot_be_written_by_business_update():
    state = SessionState(owner="owner", session_id="session")
    with pytest.raises(ValueError):
        state.apply(
            (
                StateUpdate(
                    "metadata.pending_effect_binding",
                    {"plan_id": "forged", "plan_generation": 1},
                ),
            )
        )


def test_receipt_callback_cannot_forge_binding():
    state = SessionState(
        owner="owner", session_id="session", pending_effect_plan_id="plan"
    )
    state.metadata["pending_effect_binding"] = {"plan_id": "plan", "plan_generation": 1}
    with pytest.raises(ValueError):
        _apply_effect_state_change(
            state,
            lambda value: value.metadata.update(
                pending_effect_binding={"plan_id": "other", "plan_generation": 8}
            ),
        )


@pytest.mark.parametrize("sqlite", [False, True])
def test_pending_transition_and_guard(sqlite):
    from app.agent.kernel.persistence import SQLiteKernelStore
    from app.agent.kernel.state import (
        InMemorySessionStateStore,
        PendingEffectGuard,
        SelectionInvalidError,
        _PendingEffectTransition,
    )
    from tests.support import isolated_test_database

    async def run():
        store = SQLiteKernelStore() if sqlite else InMemorySessionStateStore()
        lease, _ = await store.begin_turn(
            owner="owner", session_id="session", request_id="one"
        )
        state = await store.commit(
            lease,
            updates=(
                _PendingEffectTransition(
                    "pending_effect_plan_id", "plan", plan_generation=1, goal={}
                ),
            ),
        )
        assert state.metadata["pending_effect_binding"] == {
            "plan_id": "plan",
            "plan_generation": 1,
        }
        state = await store.commit(
            lease,
            updates=(
                _PendingEffectTransition(
                    "pending_effect_plan_id", "old", mode="clear_if_equals"
                ),
            ),
        )
        assert state.pending_effect_plan_id == "plan"
        assert state.metadata["pending_effect_binding"]["plan_id"] == "plan"
        guard = PendingEffectGuard(1, "plan", 1, 110, lambda: 100)
        _, state = await store.begin_turn(
            owner="owner",
            session_id="session",
            request_id="explain",
            pending_guard=guard,
        )
        assert state.generation == 2 and state.pending_effect_plan_id == "plan"
        with pytest.raises(SelectionInvalidError):
            await store.begin_turn(
                owner="owner",
                session_id="session",
                request_id="stale",
                pending_guard=guard,
            )
        expired = PendingEffectGuard(2, "plan", 1, 110, lambda: 110)
        with pytest.raises(SelectionInvalidError):
            await store.begin_turn(
                owner="owner",
                session_id="session",
                request_id="expired",
                pending_guard=expired,
            )
        assert await store.load(owner="owner", session_id="session") == state
        with pytest.raises(ValueError):
            await store.commit(
                _, updates=(StateUpdate("pending_effect_plan_id", "forged"),)
            )
        assert await store.load(owner="owner", session_id="session") == state

    with isolated_test_database("explain-guard.db"):
        asyncio.run(run())
