"""MediaFlux 生产身份、功能开关与共享限流策略。"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import Any

from app.agent.feature_gate import (
    agent_runtime_generation_is_current,
    current_agent_runtime_generation,
    is_agent_enabled,
)
from app.agent.owner_routes import telegram_owner_route_is_currently_authorized
from app.agent.provider_actions import get_provider_gateway
from app.agent.provider_models import ProviderGatewayError
from app.agent.rate_limit import allow_agent_tool

from ..capabilities import KernelToolSpec
from ..pipeline import ToolCallContext, ToolPipelineError
from ..state import AgentInput

_WEB_OWNER_RE = re.compile(r"^webk:v1:[0-9a-f]{64}$")


class MediaFluxTurnAdmission:
    """把功能开关代次绑定到整个只读模型/工具回合。"""

    async def begin(self, agent_input: AgentInput) -> int:
        if not is_agent_enabled():
            raise ToolPipelineError("Media Agent 当前未启用", code="agent_disabled")
        if agent_input.owner.startswith("tg:v1:") and not (
            telegram_owner_route_is_currently_authorized(agent_input.owner)
        ):
            raise ToolPipelineError(
                "当前 Telegram 身份无权使用 Media Agent",
                code="authorization_denied",
            )
        if not (
            _WEB_OWNER_RE.fullmatch(agent_input.owner)
            or agent_input.owner.startswith("tg:v1:")
        ):
            raise ToolPipelineError("Agent 身份无效", code="authorization_denied")
        return current_agent_runtime_generation()

    async def is_current(self, token: Any, agent_input: AgentInput) -> bool:
        if not isinstance(token, int) or isinstance(token, bool):
            return False
        if not is_agent_enabled() or not agent_runtime_generation_is_current(token):
            return False
        if agent_input.owner.startswith("tg:v1:"):
            return telegram_owner_route_is_currently_authorized(agent_input.owner)
        return bool(_WEB_OWNER_RE.fullmatch(agent_input.owner))


class MediaFluxAuthorizationPolicy:
    """只允许经 Web 登录或 Telegram 白名单派生的生产 principal。"""

    async def authorize(
        self,
        tool: KernelToolSpec,
        arguments: Mapping[str, Any],
        context: ToolCallContext,
    ) -> None:
        del tool, arguments
        if not is_agent_enabled():
            raise ToolPipelineError("Media Agent 当前未启用", code="agent_disabled")
        owner = str(context.owner or "")
        if _WEB_OWNER_RE.fullmatch(owner):
            return
        if owner.startswith("tg:v1:"):
            if telegram_owner_route_is_currently_authorized(owner):
                return
            raise ToolPipelineError(
                "当前 Telegram 身份无权使用 Media Agent", code="authorization_denied"
            )
        raise ToolPipelineError("Agent 身份无效", code="authorization_denied")


class MediaFluxToolRateLimiter:
    """复用现有 SQLite 共享预算，不因多 Worker 重置工具限流。"""

    async def acquire(
        self, *, owner: str, tool_name: str, cost: float, arguments: Mapping[str, Any]
    ) -> None:
        del cost
        canonical = str(tool_name or "").removeprefix("confirm:")
        # 预览不写入，不能耗尽用户点击确认时的执行额度；两阶段仍按同一身份跨 Worker 限流。
        scope_suffix = "confirmed" if tool_name.startswith("confirm:") else ""
        if canonical == "provider.query":
            # 只按静态目录的 Provider 拆分；换会话、operation 或 profile 不会刷新预算。
            try:
                spec = get_provider_gateway().catalog.get(arguments.get("operation", ""))
                scope_suffix = spec.provider
            except ProviderGatewayError as exc:
                raise ToolPipelineError(exc.safe_message, code=exc.code) from exc
        allowed = await asyncio.to_thread(
            allow_agent_tool, owner, canonical, scope_suffix=scope_suffix
        )
        if not allowed:
            raise ToolPipelineError(
                "MediaFlux 对该能力的本地调用频率已达上限，本次未访问后端；"
                "请稍后重试，不能据此判断后端故障或没有数据。",
                code="rate_limited",
            )
