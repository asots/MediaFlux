"""统一同步/异步工具调用与确认写保护。"""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.agent.errors import ToolPipelineError

from .session_guard import guarded_state_call

if TYPE_CHECKING:
    from .tool_calls import ToolCallContext


async def invoke(
    handler: Callable[..., Any], *args: Any, protection: ToolCallContext | None = None,
) -> Any:
    try:
        if inspect.iscoroutinefunction(handler):
            return await handler(*args)
        value = (
            await guarded_state_call(
                protection.owner, protection.session_id, handler, *args,
                kind="effect", complete_on_cancel=True,
            )
            if protection is not None else await asyncio.to_thread(handler, *args)
        )
        if inspect.isawaitable(value):
            return await value
        return value
    except ToolPipelineError:
        raise
    except Exception as exc:
        raise ToolPipelineError("工具执行失败", code="tool_execution_failed") from exc
