from __future__ import annotations

import pytest

from app.agent.errors import AgentToolError
from app.agent.provider_actions import provider_capabilities_arguments


def test_provider_capability_limit_rejects_lossy_integer_values():
    for invalid in (True, 1.0, 1.9, "1.0", "1e3"):
        with pytest.raises(AgentToolError):
            provider_capabilities_arguments({"limit": invalid})
    assert provider_capabilities_arguments({"limit": "8"})["limit"] == 8


def test_provider_pipeline_isolates_normalized_read_budgets_without_bypassing_limits():
    """真实工具目录/归一化/管线/限流；仅只读 handler 不连接外部服务。"""
    import asyncio
    from dataclasses import replace
    from unittest.mock import patch

    from app.agent.domain_catalog import build_tool_specs
    from app.agent.kernel.capabilities import ToolCatalog
    from app.agent.kernel.pipeline import ToolCallContext, ToolPipeline, ToolPipelineError
    from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
    from app.agent.kernel.ports.mediaflux_policy import MediaFluxToolRateLimiter
    from app.agent.kernel.state import CancellationToken, InMemorySessionStateStore
    from app.agent.rate_limit import AgentRateLimiter

    calls = []
    tool = catalog_from_tool_specs(build_tool_specs()).get("provider.query")

    def read(arguments, _context):
        calls.append(dict(arguments))
        return {"ok": True, "summary": "已查询实际后端"}

    async def progress(_payload):
        pass

    async def exercise():
        state = InMemorySessionStateStore()
        pipeline = ToolPipeline(
            catalog=ToolCatalog([replace(tool, read=read)]), state_store=state,
            rate_limiter=MediaFluxToolRateLimiter(),
        )

        async def query(operation, *, session="first", profile="media:1"):
            lease, _ = await state.begin_turn(
                owner="same-user", session_id=session, request_id=str(len(calls)),
            )
            context = ToolCallContext(
                owner=lease.owner, session_id=session, request_id=lease.request_id,
                turn_id=lease.turn_id, lease=lease, cancellation=CancellationToken(),
                report_progress=progress,
            )
            return await pipeline.execute(
                tool.model_name,
                {"profile_ref": profile, "operation": operation, "arguments": {}},
                context=context,
            )

        for _ in range(8):
            await query(" MEDIA.ITEMS.COUNTS ")
        assert all(call["operation"] == "media.items.counts" for call in calls)
        with pytest.raises(ToolPipelineError) as error:
            await query("media.system.info", session="second", profile="media:2")
        assert error.value.code == "rate_limited"
        assert "未访问后端" in str(error.value)
        assert len(calls) == 8
        await query("QB.TORRENTS.INFO", session="second", profile="qb:default")
        assert calls[-1]["operation"] == "qb.torrents.info"
        assert len(calls) == 9
        with pytest.raises(ToolPipelineError) as error:
            await query("invented.items.query")
        assert error.value.code == "operation_not_allowed"
        assert len(calls) == 9

    with patch(
        "app.agent.rate_limit.agent_rate_limiter",
        AgentRateLimiter(clock=lambda: 100.0),
    ):
        asyncio.run(exercise())
