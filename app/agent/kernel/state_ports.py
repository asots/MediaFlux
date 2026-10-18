"""执行器和完成跟踪共用的会话仓储接口。"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from .session_state import CandidateSelectionGuard, PendingEffectGuard, SessionState
from .state_types import PublicationLease, StateUpdate, _CancelledReadInline
from .task_state import TaskPatch


class SessionStateStore(Protocol):
    async def settle_cancelled_reads(self, receipts: tuple[_CancelledReadInline, ...]) -> bool: ...

    async def cancel_publication(self, *, owner: str, session_id: str, request_id: str) -> bool: ...

    async def begin_turn(
        self, *, owner: str, session_id: str, request_id: str,
        selection_guard: CandidateSelectionGuard | None = None,
        pending_guard: PendingEffectGuard | None = None,
    ) -> tuple[PublicationLease, SessionState]: ...

    async def is_current(self, lease: PublicationLease) -> bool: ...

    async def commit(
        self,
        lease: PublicationLease,
        *,
        conversation: Sequence[Mapping[str, Any]] | None = None,
        updates: Sequence[StateUpdate] = (),
        task_patch: TaskPatch | None = None,
    ) -> SessionState: ...

    async def update_effect_state(
        self, *, owner: str, session_id: str,
        change: Callable[[SessionState], Any],
    ) -> Any: ...

    def update_effect_state_in_transaction(
        self, conn: Any, *, owner: str, session_id: str,
        change: Callable[[SessionState], Any],
    ) -> Any: ...

    async def request_stop_generation(self, *, owner: str, session_id: str, generation: int) -> bool: ...

    async def due_effect_waits(self, *, limit: int = 16) -> list[dict[str, Any]]: ...

    async def load(self, *, owner: str, session_id: str) -> SessionState: ...
