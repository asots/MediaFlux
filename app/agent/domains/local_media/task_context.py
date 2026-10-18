"""本地媒体任务短期上下文：owner 隔离、持久化与请求提交缓冲。"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.agent.session_context import (
    AgentContextWriteGuard,
    AgentSessionContextRepository,
    PersistedAgentContext,
)
from app.agent.state_commit import (
    AgentStateCommitBuffer,
    active_agent_state_commit_buffer,
)
from app.modules.local_media_models import LOCAL_TASK_STATUSES

from . import common

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TaskRef:
    task_id: int
    version: int
    status: str
    source_id: int

@dataclass(frozen=True)
class _InspectionRef:
    number: int
    task: _TaskRef
    inspection_id: str
    digest: str

@dataclass
class _BufferedContext:
    """单次 Agent 请求私有的本地媒体续接视图。"""

    tasks: tuple[_TaskRef, ...]
    inspections: tuple[_InspectionRef, ...]
    next_inspection: int
    guard: AgentContextWriteGuard | None = None

class LocalMediaAgentContextStore:
    """保存短 TTL 的任务序号和检查句柄，并以 owner 指纹跨 Worker 续接。"""

    _CONTEXT_TYPE = "local_media_tasks"
    _PAYLOAD_VERSION = 1

    def __init__(
        self,
        *,
        ttl_seconds: int = 1800,
        max_owners: int = 256,
        max_inspections_per_owner: int = 16,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        repository: AgentSessionContextRepository | None = None,
    ) -> None:
        self.ttl_seconds = max(1, ttl_seconds)
        self.max_owners = max(1, max_owners)
        self.max_inspections_per_owner = max(1, max_inspections_per_owner)
        self._clock = clock
        self._wall_clock = wall_clock
        self._repository = repository
        self._lock = threading.RLock()
        self._tasks: OrderedDict[str, tuple[float, tuple[_TaskRef, ...]]] = (
            OrderedDict()
        )
        self._inspections: OrderedDict[
            str, tuple[float, tuple[_InspectionRef, ...]]
        ] = OrderedDict()
        self._next_inspection: dict[str, int] = {}


    def capture_tasks(self, *, owner: str, tasks: list[Any]) -> tuple[_TaskRef, ...]:
        owner_key = (owner or "").strip()
        if not owner_key:
            return ()
        refs = tuple(
            _TaskRef(
                task_id=int(task.id),
                version=int(task.version),
                status=str(task.status),
                source_id=int(task.source_id),
            )
            for task in tasks[:common._MAX_TASK_NUMBER]
        )
        buffered = self._buffered_context(owner_key)
        if buffered is not None:
            buffer, state = buffered
            if not self._stage_buffered_commit(buffer, owner_key, state):
                return ()
            with self._lock:
                state.tasks = refs
            return refs
        if active_agent_state_commit_buffer() is not None:
            return ()
        with self._lock:
            self._prune_locked()
            updated = self._mutate_persisted_locked(
                owner_key,
                lambda current: (
                    refs,
                    current[1]
                    if current is not None
                    else (self._inspections.get(owner_key, (0.0, ()))[1]),
                    current[2]
                    if current is not None
                    else self._next_inspection.get(owner_key, 0),
                ),
            )
            if updated is not None:
                return updated[0]
            self._restore_locked(owner_key, required="inspections")
            self._tasks.pop(owner_key, None)
            self._tasks[owner_key] = (self._clock() + self.ttl_seconds, refs)
            self._trim_locked(self._tasks)
            self._persist_locked(owner_key)
        return refs

    def task(self, *, owner: str, number: int) -> _TaskRef | None:
        owner_key = (owner or "").strip()
        buffered = self._buffered_context(owner_key)
        if buffered is not None:
            _, state = buffered
            with self._lock:
                if not 1 <= number <= len(state.tasks):
                    return None
                return state.tasks[number - 1]
        if active_agent_state_commit_buffer() is not None:
            return None
        with self._lock:
            self._prune_locked()
            self._restore_locked(owner_key, required="tasks")
            entry = self._tasks.get(owner_key)
            if entry is None or not 1 <= number <= len(entry[1]):
                return None
            self._tasks.move_to_end(owner_key)
            return entry[1][number - 1]

    def capture_inspection(
        self,
        *,
        owner: str,
        task: _TaskRef,
        inspection_id: str,
        digest: str,
    ) -> int:
        owner_key = (owner or "").strip()
        if not owner_key or not inspection_id or not digest:
            return 0
        buffered = self._buffered_context(owner_key)
        if buffered is not None:
            buffer, state = buffered
            if not self._stage_buffered_commit(buffer, owner_key, state):
                return 0
            with self._lock:
                used = {item.number for item in state.inspections}
                number = (
                    1
                    if state.next_inspection >= common._MAX_INSPECTION_NUMBER
                    else state.next_inspection + 1
                )
                while number in used:
                    number = 1 if number >= common._MAX_INSPECTION_NUMBER else number + 1
                state.next_inspection = number
                ref = _InspectionRef(number, task, inspection_id, digest)
                state.inspections = (ref, *state.inspections)[
                    : self.max_inspections_per_owner
                ]
                return number
        if active_agent_state_commit_buffer() is not None:
            return 0
        with self._lock:
            self._prune_locked()
            assigned: list[int] = []

            def append_inspection(
                current: tuple[tuple[_TaskRef, ...], tuple[_InspectionRef, ...], int]
                | None,
            ) -> tuple[tuple[_TaskRef, ...], tuple[_InspectionRef, ...], int]:
                current_tasks = (
                    current[0]
                    if current is not None
                    else (self._tasks.get(owner_key, (0.0, ()))[1])
                )
                current_inspections = (
                    current[1]
                    if current is not None
                    else (self._inspections.get(owner_key, (0.0, ()))[1])
                )
                next_number = (
                    current[2]
                    if current is not None
                    else (self._next_inspection.get(owner_key, 0))
                )
                used = {item.number for item in current_inspections}
                number = 1 if next_number >= common._MAX_INSPECTION_NUMBER else next_number + 1
                while number in used:
                    number = 1 if number >= common._MAX_INSPECTION_NUMBER else number + 1
                assigned[:] = [number]
                ref = _InspectionRef(number, task, inspection_id, digest)
                return (
                    current_tasks,
                    (ref, *current_inspections)[: self.max_inspections_per_owner],
                    number,
                )

            updated = self._mutate_persisted_locked(owner_key, append_inspection)
            if updated is not None and assigned:
                return assigned[0]
            self._restore_locked(owner_key, required="inspections")
            current = self._inspections.pop(owner_key, (0.0, ()))[1]
            used = {item.number for item in current}
            number = self._next_inspection.get(owner_key, 0) + 1
            if number > common._MAX_INSPECTION_NUMBER:
                number = 1
            while number in used:
                number = 1 if number >= common._MAX_INSPECTION_NUMBER else number + 1
            self._next_inspection[owner_key] = number
            ref = _InspectionRef(number, task, inspection_id, digest)
            self._inspections[owner_key] = (
                self._clock() + self.ttl_seconds,
                (ref, *current)[: self.max_inspections_per_owner],
            )
            self._trim_locked(self._inspections)
            self._persist_locked(owner_key)
            return number

    def inspection(self, *, owner: str, number: int) -> _InspectionRef | None:
        owner_key = (owner or "").strip()
        buffered = self._buffered_context(owner_key)
        if buffered is not None:
            _, state = buffered
            with self._lock:
                return next(
                    (item for item in state.inspections if item.number == number),
                    None,
                )
        if active_agent_state_commit_buffer() is not None:
            return None
        with self._lock:
            self._prune_locked()
            self._restore_locked(owner_key, required="inspections")
            entry = self._inspections.get(owner_key)
            if entry is None:
                return None
            selected = next(
                (item for item in entry[1] if item.number == number), None
            )
            if selected is not None:
                self._inspections.move_to_end(owner_key)
            return selected

    def replace_inspection(
        self,
        *,
        owner: str,
        number: int,
        inspection_id: str,
        digest: str,
    ) -> bool:
        owner_key = (owner or "").strip()
        if not owner_key or not inspection_id or not digest:
            return False
        buffered = self._buffered_context(owner_key)
        if buffered is not None:
            buffer, state = buffered
            if not self._stage_buffered_commit(buffer, owner_key, state):
                return False
            with self._lock:
                refs: list[_InspectionRef] = []
                replaced = False
                for item in state.inspections:
                    if item.number == number:
                        refs.append(
                            _InspectionRef(
                                item.number, item.task, inspection_id, digest
                            )
                        )
                        replaced = True
                    else:
                        refs.append(item)
                if replaced:
                    state.inspections = tuple(refs)
                return replaced
        if active_agent_state_commit_buffer() is not None:
            return False
        with self._lock:
            self._prune_locked()
            replaced = [False]

            def replace(
                current: tuple[tuple[_TaskRef, ...], tuple[_InspectionRef, ...], int]
                | None,
            ) -> tuple[tuple[_TaskRef, ...], tuple[_InspectionRef, ...], int]:
                if current is None:
                    tasks = self._tasks.get(owner_key, (0.0, ()))[1]
                    inspections = self._inspections.get(owner_key, (0.0, ()))[1]
                    next_number = self._next_inspection.get(owner_key, 0)
                else:
                    tasks, inspections, next_number = current
                refs: list[_InspectionRef] = []
                for item in inspections:
                    if item.number == number:
                        refs.append(
                            _InspectionRef(
                                item.number, item.task, inspection_id, digest
                            )
                        )
                        replaced[0] = True
                    else:
                        refs.append(item)
                return tasks, tuple(refs), next_number

            updated = self._mutate_persisted_locked(owner_key, replace)
            if updated is not None:
                return replaced[0]
            self._restore_locked(owner_key, required="inspections")
            entry = self._inspections.get(owner_key)
            if entry is None:
                return False
            refs = []
            for item in entry[1]:
                if item.number == number:
                    refs.append(
                        _InspectionRef(
                            item.number, item.task, inspection_id, digest
                        )
                    )
                    replaced[0] = True
                else:
                    refs.append(item)
            if not replaced[0]:
                return False
            self._inspections[owner_key] = (entry[0], tuple(refs))
            self._inspections.move_to_end(owner_key)
            self._persist_locked(owner_key)
            return True

    def clear_owner(self, *, owner: str, invalidate_persisted: bool = True) -> None:
        owner_key = (owner or "").strip()
        if not owner_key:
            return
        with self._lock:
            self._tasks.pop(owner_key, None)
            self._inspections.pop(owner_key, None)
            self._next_inspection.pop(owner_key, None)
            repository = self._repository
        if repository is not None and invalidate_persisted:
            try:
                invalidate = getattr(repository, "invalidate_context", None)
                if callable(invalidate):
                    invalidate(
                        owner=owner_key,
                        context_type=self._CONTEXT_TYPE,
                    )
                else:
                    repository.delete_latest(
                        owner=owner_key, context_type=self._CONTEXT_TYPE
                    )
            except Exception as exc:
                logger.warning(
                    "本地媒体 Agent 上下文清理失败 type=%s", type(exc).__name__
                )

    def reset(self) -> None:
        """仅清空本进程缓存；持久化上下文用于模拟重启与跨 Worker 续接。"""
        with self._lock:
            self._tasks.clear()
            self._inspections.clear()
            self._next_inspection.clear()

    def _buffered_context(
        self, owner: str
    ) -> tuple[AgentStateCommitBuffer, _BufferedContext] | None:
        buffer = active_agent_state_commit_buffer()
        if buffer is None or not owner:
            return None
        key = self._buffered_state_key(owner)
        state = buffer.get_or_create_request_state(
            owner=owner,
            key=key,
            factory=lambda: self._snapshot_context(owner),
        )
        if not isinstance(state, _BufferedContext):
            return None
        return buffer, state

    def _snapshot_context(self, owner: str) -> _BufferedContext:
        with self._lock:
            self._prune_locked()
            repository = self._repository
            begin_update: Callable[..., tuple[PersistedAgentContext | None, AgentContextWriteGuard]] | None = (
                getattr(repository, "begin_context_update", None)
                if repository is not None
                else None
            )
            if callable(begin_update):
                persisted, guard = begin_update(
                    owner=owner, context_type=self._CONTEXT_TYPE
                )
                decoded = (
                    self._decode_payload(persisted.payload)
                    if persisted is not None
                    else None
                )
                if decoded is None:
                    self._tasks.pop(owner, None)
                    self._inspections.pop(owner, None)
                    self._next_inspection.pop(owner, None)
                    return _BufferedContext((), (), 0, guard)
                assert persisted is not None
                self._install_decoded_locked(
                    owner, decoded, persisted_expires_at=persisted.expires_at
                )
                return _BufferedContext(*decoded, guard)
            self._restore_locked(owner, required="tasks")
            self._restore_locked(owner, required="inspections")
            return _BufferedContext(
                tasks=self._tasks.get(owner, (0.0, ()))[1],
                inspections=self._inspections.get(owner, (0.0, ()))[1],
                next_inspection=self._next_inspection.get(owner, 0),
            )

    def _stage_buffered_commit(
        self,
        buffer: AgentStateCommitBuffer,
        owner: str,
        state: _BufferedContext,
    ) -> bool:
        return buffer.add_once(
            key=f"{self._buffered_state_key(owner)}:commit",
            action=lambda: self._commit_buffered_context(owner, state),
        )

    def _commit_buffered_context(self, owner: str, state: _BufferedContext) -> bool:
        with self._lock:
            decoded = (state.tasks, state.inspections, state.next_inspection)
            expires_at = self._wall_clock() + self.ttl_seconds
            repository = self._repository
            if repository is not None:
                guarded: Callable[..., PersistedAgentContext | None] | None = getattr(repository, "replace_latest_guarded", None)
                if state.guard is not None and callable(guarded):
                    persisted = guarded(
                        owner=owner,
                        context_type=self._CONTEXT_TYPE,
                        payload=self._encode_payload(*decoded),
                        expires_at=expires_at,
                        guard=state.guard,
                    )
                    if persisted is None:
                        self._tasks.pop(owner, None)
                        self._inspections.pop(owner, None)
                        self._next_inspection.pop(owner, None)
                        return False
                    expires_at = persisted.expires_at
                else:
                    repository.replace_latest(
                        owner=owner,
                        context_type=self._CONTEXT_TYPE,
                        payload=self._encode_payload(*decoded),
                        expires_at=expires_at,
                    )
            self._install_decoded_locked(
                owner, decoded, persisted_expires_at=expires_at
            )
            return True

    def _buffered_state_key(self, owner: str) -> str:
        return f"local-media:{id(self)}:{owner}"

    def _restore_locked(self, owner: str, *, required: str) -> None:
        if not owner:
            return
        if required == "tasks" and owner in self._tasks:
            return
        if required == "inspections" and owner in self._inspections:
            return
        repository = self._repository
        if repository is None:
            return
        try:
            persisted = repository.get_latest(
                owner=owner,
                context_type=self._CONTEXT_TYPE,
                now=self._wall_clock(),
            )
        except Exception as exc:
            logger.warning("本地媒体 Agent 上下文恢复失败 type=%s", type(exc).__name__)
            return
        if persisted is None:
            return
        decoded = self._decode_payload(persisted.payload)
        if decoded is None:
            try:
                repository.delete_latest(owner=owner, context_type=self._CONTEXT_TYPE)
            except Exception:
                pass
            return
        self._install_decoded_locked(
            owner, decoded, persisted_expires_at=persisted.expires_at
        )

    def _mutate_persisted_locked(
        self,
        owner: str,
        updater: Callable[
            [tuple[tuple[_TaskRef, ...], tuple[_InspectionRef, ...], int] | None],
            tuple[tuple[_TaskRef, ...], tuple[_InspectionRef, ...], int],
        ],
    ) -> tuple[tuple[_TaskRef, ...], tuple[_InspectionRef, ...], int] | None:
        repository = self._repository
        mutate: Callable[..., PersistedAgentContext] | None = (
            getattr(repository, "mutate_latest", None)
            if repository is not None
            else None
        )
        if not callable(mutate):
            return None

        def update_payload(
            current_payload: dict[str, Any] | None,
        ) -> dict[str, Any]:
            current = (
                self._decode_payload(current_payload)
                if isinstance(current_payload, dict)
                else None
            )
            return self._encode_payload(*updater(current))

        try:
            persisted = mutate(
                owner=owner,
                context_type=self._CONTEXT_TYPE,
                updater=update_payload,
                expires_at=self._wall_clock() + self.ttl_seconds,
            )
        except Exception as exc:
            logger.warning(
                "本地媒体 Agent 上下文原子更新失败 type=%s", type(exc).__name__
            )
            return None
        decoded = self._decode_payload(persisted.payload)
        if decoded is None:
            return None
        self._install_decoded_locked(
            owner, decoded, persisted_expires_at=persisted.expires_at
        )
        return decoded

    def _install_decoded_locked(
        self,
        owner: str,
        decoded: tuple[tuple[_TaskRef, ...], tuple[_InspectionRef, ...], int],
        *,
        persisted_expires_at: float,
    ) -> None:
        tasks, inspections, next_inspection = decoded
        lifetime = max(
            1.0,
            min(
                float(self.ttl_seconds),
                float(persisted_expires_at) - self._wall_clock(),
            ),
        )
        expiry = self._clock() + lifetime
        self._tasks.pop(owner, None)
        self._inspections.pop(owner, None)
        if tasks:
            self._tasks[owner] = (expiry, tasks)
            self._trim_locked(self._tasks)
        if inspections:
            self._inspections[owner] = (expiry, inspections)
            self._trim_locked(self._inspections)
        if tasks or inspections:
            self._next_inspection[owner] = next_inspection
        else:
            self._next_inspection.pop(owner, None)

    def _encode_payload(
        self,
        tasks: tuple[_TaskRef, ...],
        inspections: tuple[_InspectionRef, ...],
        next_inspection: int,
    ) -> dict[str, Any]:
        return {
            "version": self._PAYLOAD_VERSION,
            "tasks": [self._task_payload(item) for item in tasks],
            "inspections": [
                {
                    "number": item.number,
                    "task": self._task_payload(item.task),
                    "inspection_id": item.inspection_id,
                    "digest": item.digest,
                }
                for item in inspections
            ],
            "next_inspection": next_inspection,
        }

    def _persist_locked(self, owner: str) -> None:
        repository = self._repository
        if repository is None:
            return
        tasks = self._tasks.get(owner, (0.0, ()))[1]
        inspections = self._inspections.get(owner, (0.0, ()))[1]
        if not tasks and not inspections:
            return
        payload = self._encode_payload(
            tasks, inspections, self._next_inspection.get(owner, 0)
        )
        try:
            repository.replace_latest(
                owner=owner,
                context_type=self._CONTEXT_TYPE,
                payload=payload,
                expires_at=self._wall_clock() + self.ttl_seconds,
            )
        except Exception as exc:
            logger.warning(
                "本地媒体 Agent 上下文持久化失败 type=%s", type(exc).__name__
            )

    @staticmethod
    def _task_payload(item: _TaskRef) -> dict[str, Any]:
        return {
            "task_id": item.task_id,
            "version": item.version,
            "status": item.status,
            "source_id": item.source_id,
        }

    def _decode_payload(
        self, payload: dict[str, Any]
    ) -> tuple[tuple[_TaskRef, ...], tuple[_InspectionRef, ...], int] | None:
        if (
            not isinstance(payload, dict)
            or set(payload) != {"version", "tasks", "inspections", "next_inspection"}
            or payload.get("version") != self._PAYLOAD_VERSION
        ):
            return None
        raw_tasks = payload.get("tasks")
        raw_inspections = payload.get("inspections")
        if not isinstance(raw_tasks, list) or not isinstance(raw_inspections, list):
            return None
        if (
            len(raw_tasks) > common._MAX_TASK_NUMBER
            or len(raw_inspections) > self.max_inspections_per_owner
        ):
            return None
        tasks: list[_TaskRef] = []
        for raw in raw_tasks:
            ref = self._decode_task(raw)
            if ref is None:
                return None
            tasks.append(ref)
        inspections: list[_InspectionRef] = []
        for raw in raw_inspections:
            if not isinstance(raw, dict) or set(raw) != {
                "number",
                "task",
                "inspection_id",
                "digest",
            }:
                return None
            task = self._decode_task(raw.get("task"))
            inspection_id = raw.get("inspection_id")
            digest = raw.get("digest")
            number = raw.get("number")
            if (
                task is None
                or isinstance(number, bool)
                or not isinstance(number, int)
                or not 1 <= number <= common._MAX_INSPECTION_NUMBER
                or not isinstance(inspection_id, str)
                or not 1 <= len(inspection_id) <= 256
                or not isinstance(digest, str)
                or not 1 <= len(digest) <= 256
            ):
                return None
            inspections.append(_InspectionRef(number, task, inspection_id, digest))
        next_inspection = payload.get("next_inspection")
        if (
            isinstance(next_inspection, bool)
            or not isinstance(next_inspection, int)
            or not 0 <= next_inspection <= common._MAX_INSPECTION_NUMBER
        ):
            return None
        return tuple(tasks), tuple(inspections), next_inspection

    @staticmethod
    def _decode_task(raw: Any) -> _TaskRef | None:
        if not isinstance(raw, dict) or set(raw) != {
            "task_id",
            "version",
            "status",
            "source_id",
        }:
            return None
        values = (raw["task_id"], raw["version"], raw["source_id"])
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values
        ):
            return None
        status = raw.get("status")
        if not isinstance(status, str) or status not in LOCAL_TASK_STATUSES:
            return None
        return _TaskRef(values[0], values[1], status, values[2])

    def _prune_locked(self) -> None:
        now = self._clock()
        for mapping in (self._tasks, self._inspections):
            for owner, entry in list(mapping.items()):
                if entry[0] <= now:
                    mapping.pop(owner, None)
                    self._cleanup_owner_counter_locked(owner)

    def _trim_locked(self, mapping: OrderedDict[str, Any]) -> None:
        while len(mapping) > self.max_owners:
            owner, _entry = mapping.popitem(last=False)
            self._cleanup_owner_counter_locked(owner)

    def _cleanup_owner_counter_locked(self, owner: str) -> None:
        if owner not in self._tasks and owner not in self._inspections:
            self._next_inspection.pop(owner, None)
