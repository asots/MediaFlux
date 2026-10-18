"""owner/session 文件互斥：确认写持锁至回执落盘，状态事务复用同一边界。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from functools import partial
from typing import Any, Generator

from app import database as db
from app.modules.process_lock import CrossProcessLock

from .state_types import SessionBusyError


@dataclass(slots=True)
class _HeldScope:
    key: str
    kind: str
    pid: int
    task: object | None
    active: bool = True


_held_scope: ContextVar[_HeldScope | None] = ContextVar("agent_session_guard", default=None)


def _task() -> object | None:
    try:
        return asyncio.current_task()
    except RuntimeError:  # asyncio.to_thread 会携带 context，但线程内没有事件循环。
        return None


def _scope_key(owner: str, session_id: str) -> str:
    owner, session_id = (owner or "").strip(), (session_id or "").strip()
    if not owner or not session_id:
        raise ValueError("Agent 会话 scope 无效")
    return hashlib.sha256(json.dumps(
        [str(db.resolve_db_path()), owner, session_id], ensure_ascii=False,
        separators=(",", ":"),
    ).encode()).hexdigest()


@contextmanager
def continuation_execution_guard(owner: str, session_id: str) -> Generator[bool, None, None]:
    """续行执行持有独立 OS 锁；进程退出自动释放，不用 TTL 推断死亡。

    与会话状态事务锁分开，存活续行期间仍允许冷取消和前台新请求。
    返回 False 表示该会话已有执行者，调用方不能接管或收束其状态。
    """
    lock = CrossProcessLock("agent-continuation-" + _scope_key(owner, session_id))
    try:
        acquired = lock.acquire(blocking=False)
    except OSError as exc:
        raise SessionBusyError("continuation execution guard is unavailable") from exc
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()


@contextmanager
def session_scope_guard(owner: str, session_id: str, *, kind: str = "mutation") -> Generator[None, None, None]:
    """非阻塞互斥；不创建内存锁注册表，也不依赖 TTL 判定活跃执行者。

    同一操作通过 to_thread 进入状态事务时允许复用持锁权。其它 asyncio
    task、进程或已退出的 context 不能继承该权利；确认体内也不能重入 reset。
    锁文件不 unlink，避免新 inode 与仍持有旧 inode 的 worker 分裂互斥域。
    """
    key = _scope_key(owner, session_id)
    inherited = _held_scope.get()
    task = _task()
    if (
        inherited is not None and inherited.active and inherited.key == key
        and inherited.pid == os.getpid()
        and (task is None or task is inherited.task)
    ):
        if kind != inherited.kind and kind != "commit":
            raise SessionBusyError("confirmed effect is executing")
        yield
        return

    lock = CrossProcessLock("agent-session-" + key)
    try:
        acquired = lock.acquire(blocking=False)
    except OSError as exc:
        raise SessionBusyError("session execution guard is unavailable") from exc
    if not acquired:
        raise SessionBusyError("session operation is executing")
    held = _HeldScope(key, kind, os.getpid(), task)
    token = _held_scope.set(held)
    try:
        yield
    finally:
        held.active = False
        _held_scope.reset(token)
        lock.release()


async def guarded_state_call(
    owner: str, session_id: str, call: Callable[..., Any], *args: Any,
    kind: str = "mutation", complete_on_cancel: bool = False,
) -> Any:
    """先在当前 task 验证持锁权，再进入线程；取消不能先于 DB 事务放锁。"""
    with session_scope_guard(owner, session_id, kind=kind):
        return await _run_worker(call, args, complete_on_cancel or _effect_active())


def _effect_active() -> bool:
    held = _held_scope.get()
    return bool(held and held.active and held.kind == "effect"
                and held.pid == os.getpid() and held.task is _task())


async def session_io(call: Callable[..., Any], *args: Any) -> Any:
    """确认持有者的持久化收尾不可被请求取消打断；普通读请求仍可取消。"""
    if _effect_active():
        return await _run_worker(call, args, True)
    return await asyncio.to_thread(call, *args)


async def _run_worker(call: Callable[..., Any], args: tuple[Any, ...], complete_on_cancel: bool) -> Any:
    # 使用 executor Future 而非额外 Task：事件循环 shutdown 的全 Task
    # 取消不能把仍在执行的同步线程误判为结束，进而提前释放会话文件锁。
    operation = asyncio.get_running_loop().run_in_executor(
        None, partial(copy_context().run, call, *args),
    )
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(operation)
            break
        except asyncio.CancelledError:
            cancelled = True
            if operation.done():
                result = operation.result()  # 工作项自身取消/失败必须传播，不能反复 await。
                break
    if cancelled and not complete_on_cancel:
        raise asyncio.CancelledError
    return result
