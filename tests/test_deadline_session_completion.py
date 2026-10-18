"""Session discovery budgets at the real native request boundary."""

from typing import cast
from collections.abc import Callable
from app.clients.http import FixedHostHttpClient
import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from app.agent.kernel.events import AgentEventType
from app.agent.kernel.provider_model import (
    OpenAICompatibleModelAdapter,
    ProviderSettings,
)
from app.agent.kernel.session import SessionLimits
from tests.test_agent_kernel_capability_discovery import request, session_for
from tests.test_agent_kernel_core import collect, read_tool


def run_native(batches, *, rounds=6):
    async def exercise():
        seen, executed = [], []
        session, _ = session_for([read_tool(
            "alpha.inspect", handler=lambda a, c: executed.append("read") or {"summary": "verified"},
        )], [])
        session.limits = SessionLimits(max_tool_calls=1, max_model_rounds=rounds)

        class Response:
            status_code = 200

            def __init__(self, body):
                self.body = body
                self.headers = {"content-type": "text/event-stream"}

            async def aiter_bytes(self):
                yield ("data: " + json.dumps(self.body) + "\n\ndata: [DONE]\n\n").encode()

        class Client:
            @asynccontextmanager
            async def stream_post_json(self, url, **kwargs):
                seen.append(kwargs["json"])
                index = len(seen) - 1
                calls = batches[index] if index < len(batches) else []
                delta = {"tool_calls": [
                    {"index": i, "id": f"c{index}-{i}", "type": "function", "function": {
                        "name": name, "arguments": json.dumps(args),
                    }} for i, (name, args) in enumerate(calls)
                ]} if calls else {"content": "已依据真实读取完成。"}
                yield Response({"choices": [{"delta": delta, "finish_reason": "tool_calls" if calls else "stop"}]})

            async def aclose(self):
                pass

        session.model = OpenAICompatibleModelAdapter(
            ProviderSettings(api_url="https://offline.invalid/v1", model="fixture", protocol="chat_completions"),
            client_factory=cast(Callable[..., FixedHostHttpClient], lambda **_: Client()),
        )
        events = await collect(session.run(request()))
        return seen, executed, events
    return asyncio.run(exercise())


DISCOVER = ("agent__capabilities", {"tool_names": ["alpha.inspect"]})
READ = ("alpha__inspect", {})


@pytest.mark.parametrize("attempts", [1, 3])
def test_one_business_call_after_cached_discovery_reaches_native_final(attempts):
    seen, executed, events = run_native([[DISCOVER] * attempts, [READ]])
    assert executed == ["read"]
    assert len(seen) == 3
    assert "alpha__inspect" in {t["function"]["name"] for t in seen[1]["tools"]}
    assert not seen[2].get("tools")
    prompt = seen[1]["messages"][0]["content"]
    assert "实际剩余工具调用 1 次" in prompt
    assert "业务工具调用已使用" not in prompt
    assert "能力发现尝试" not in prompt
    # First snapshot exposes read_result, changing the authorized-visible cache key.
    # The third identical attempt reuses the second selection, not a free raw call.
    discoveries = [json.loads(message["content"])["data"]
                   for message in seen[1]["messages"] if message["role"] == "tool"]
    assert len(discoveries) == attempts
    assert discoveries[-1]["discovery_cost"] == {"distinct_requests": min(attempts, 2), "limit": 8}
    assert discoveries[-1]["cache_hit"] is (attempts == 3)
    assert events[-1].type is AgentEventType.TURN_COMPLETED
    assert events[-1].payload["status"] == "success"
    assert events[-1].payload["tool_calls"] == attempts + 1
    assert events[-1].payload["business_tool_calls"] == 1
    assert events[-1].payload["discovery_tool_calls"] == attempts


@pytest.mark.parametrize("overflow", ["discovery", "business"])
def test_cross_budget_mixed_batch_is_rejected_atomically(overflow):
    # First two rounds consume 16 raw attempts but one distinct selection.
    batches = [[DISCOVER] * 8, [DISCOVER] * 8]
    batches.append([DISCOVER, READ] if overflow == "discovery" else [READ, READ])
    seen, executed, events = run_native(batches)
    assert len(seen) == 4
    assert executed == []
    failed = [e for e in events if e.type is AgentEventType.TOOL_FAILED]
    assert len(failed) == 2
    assert all(e.payload["code"] == "tool_budget_exceeded" for e in failed)
    assert events[-1].payload["tool_calls"] == 16
    assert events[-1].payload["status"] == "partial"


def test_discovery_still_uses_finite_model_rounds():
    seen, executed, events = run_native([[DISCOVER]] * 3, rounds=3)
    assert len(seen) == 3 and executed == []
    assert not seen[-1].get("tools")
    assert events[-1].payload["finish_reason"] == "model_round_budget_exceeded"
