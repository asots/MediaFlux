"""Native group identity and exact scoped storage-reprojection contracts."""
import asyncio
import json

from app.agent.kernel.capabilities import CapabilityRetriever, ToolCatalog
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.provider_model import (
    OpenAICompatibleModelAdapter,
    ProviderSettings,
)
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state_types import AgentInput
from tests.support import isolated_test_database
from tests.test_agent_kernel_core import read_tool
from tests.test_agent_provider_state import (
    SIGNATURE,
    WireClient,
    anthropic_answer,
    anthropic_tool_frames,
)
from tests.test_cancelled_read_inline import sqlite_store


def test_native_same_id_distinct_facts_real_wire_and_cold_store():
    async def run():
        calls = []
        def handler(arguments, context):
            fact = "OLD_FACT" if not calls else "CURRENT_FACT"
            calls.append(fact)
            return {"ok": True, "data": {"fact": fact}}
        catalog = ToolCatalog([read_tool("inspect", handler=handler)])
        wire = WireClient([anthropic_tool_frames(), anthropic_answer(), anthropic_tool_frames(), anthropic_answer()])
        def session():
            store = sqlite_store()
            return AgentSession(model=OpenAICompatibleModelAdapter(
                ProviderSettings(api_url="https://api.example.com/v1", model="model-one", protocol="anthropic_messages"),
                client_factory=lambda **kwargs: wire), catalog=catalog, retriever=CapabilityRetriever(),
                pipeline=ToolPipeline(catalog=catalog, state_store=store, reference_store=store), state_store=store,
                system_prompt="inspect")
        for request in ("old", "current"):
            events = [e async for e in session().run(AgentInput(owner="o", session_id="native", request_id=request, message=request))]
            assert events[-1].type == AgentEventType.TURN_COMPLETED
            assert events[-1].payload["status"] == "success"
        assert len(wire.requests) == 4
        blocks = [b for row in wire.requests[-1]["messages"] if isinstance(row.get("content"), list) for b in row["content"]]
        assert [b["id"] for b in blocks if b.get("type") == "tool_use"] == ["call-1"]
        results = [b for b in blocks if b.get("type") == "tool_result"]
        assert len(results) == 1 and results[0]["tool_use_id"] == "call-1"
        assert "CURRENT_FACT" in json.dumps(results) and "OLD_FACT" not in json.dumps(blocks)
        cold = await sqlite_store().load(owner="o", session_id="native")
        rows = [r for r in cold.conversation if r["role"] == "tool"]
        assert [json.loads(r["content"])["data"]["fact"] for r in rows] == calls == ["OLD_FACT", "CURRENT_FACT"]
        assert [r["tool_call_id"] for r in rows] == ["call-1", "call-1"]
        native = [r["provider_state"] for r in cold.conversation if r.get("provider_state") and r.get("tool_calls")]
        assert len(native) == 2 and all(SIGNATURE in r["layout_json"] for r in native)
    with isolated_test_database():
        asyncio.run(run())
