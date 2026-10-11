"""Native group identity and exact scoped storage-reprojection contracts."""
import asyncio
import json
from dataclasses import replace

import pytest

from app.agent.kernel.capabilities import CapabilityRetriever, ToolCatalog
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.model import ModelMessage
from app.agent.kernel.pipeline import ToolPipeline, ToolPipelineError
from app.agent.kernel.provider_model import OpenAICompatibleModelAdapter, ProviderSettings
from app.agent.kernel.session import AgentSession, _require_complete_inline_history
from app.agent.kernel.state import AgentInput
from app.agent.model_context_budget import compact_tool_content
from tests.support import isolated_test_database
from tests.test_agent_kernel_core import read_tool
from tests.test_agent_provider_state import WireClient, anthropic_tool_frames, anthropic_answer, SIGNATURE
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
        # The same collision exclusion must not exempt an unrelated small fact.
        from app.agent.kernel.model import ModelToolCall
        from app.agent.model_context_budget import protocol_history_baseline
        unrelated = [ModelMessage(role="user", content="unrelated"),
            ModelMessage(role="assistant", tool_calls=(ModelToolCall("unique", "inspect", {}),)),
            ModelMessage(role="tool", tool_call_id="unique", tool_name="inspect", content='{"ok":true,"fact":"SMALL"}')]
        original = unrelated + [ModelMessage.from_dict(r) for r in cold.conversation]
        boundary = max(i for i, m in enumerate(original) if m.role == "user")
        baseline = protocol_history_baseline(original, history_end=boundary)
        assert [m.tool_call_id for m in baseline if m.role == "tool"] == ["unique", "call-1"]
        store = sqlite_store()
        lease, _ = await store.begin_turn(owner="o", session_id="native", request_id="guard")
        with pytest.raises(ToolPipelineError):
            await _require_complete_inline_history([m.to_dict() for m in baseline],
                [m.to_dict() for m in baseline if m.tool_call_id != "unique"],
                code="context_budget_exceeded", wire=True,
                pipeline=ToolPipeline(catalog=catalog, state_store=store, reference_store=store), lease=lease)
    with isolated_test_database():
        asyncio.run(run())


@pytest.mark.parametrize("mutation", ["valid", "extra", "typed", "foreign", "missing_part"])
def test_scoped_reprojection_full_tree(mutation):
    async def run():
        store = sqlite_store()
        catalog = ToolCatalog([])
        pipeline = ToolPipeline(catalog=catalog, state_store=store, reference_store=store)
        lease, _ = await store.begin_turn(owner="o", session_id="projection", request_id="verify")
        # Actual existing reference store and producer envelope; never a fake handle.
        source = json.dumps({"ok": True, "data": {"text": "x" * 15000, "flag": True}}, separators=(",", ":"))
        parts = [await store.put(kind="tool_result_part", value=chunk, owner="o", session_id="projection")
                 for chunk in (source[:7000], source[7000:])]
        reference = await store.put(kind="tool_result", value={"parts": [part.ref for part in parts]}, owner="o", session_id="projection")
        handle = reference.ref
        projected = json.loads(compact_tool_content(source, maximum=pipeline.projector.max_model_chars))
        projected.update(result_handle=handle, read_tool="agent.read_result")
        producer = compact_tool_content(json.dumps(projected, separators=(",", ":")), maximum=pipeline.projector.max_model_chars, trusted=True)
        payload = json.loads(compact_tool_content(producer, maximum=8500, trusted=True))
        if mutation == "extra": payload["extra"] = "invented"
        if mutation == "typed": payload["data"]["flag"] = 1
        if mutation == "foreign": lease = replace(lease, session_id="foreign")
        if mutation == "missing_part":
            from app import database as db
            with db.get_conn() as conn:
                conn.execute("DELETE FROM agent_kernel_refs WHERE kind='tool_result_part'")
        row = {"role": "tool", "tool_name": "inspect", "tool_call_id": "one", "content": json.dumps(payload)}
        if mutation == "valid":
            await _require_complete_inline_history([row], [], code="context_budget_exceeded", pipeline=pipeline, lease=lease)
        else:
            with pytest.raises(ToolPipelineError):
                await _require_complete_inline_history([row], [], code="context_budget_exceeded", pipeline=pipeline, lease=lease)
    with isolated_test_database():
        asyncio.run(run())


def test_small_duplicate_fact_obligations_keep_multiplicity():
    async def run():
        store = sqlite_store()
        lease, _ = await store.begin_turn(owner="o", session_id="counts", request_id="check")
        pipeline = ToolPipeline(catalog=ToolCatalog([]), state_store=store, reference_store=store)
        row = {"role": "tool", "tool_name": "inspect", "tool_call_id": "one", "content": '{"ok":true,"data":{"fact":"small"}}'}
        with pytest.raises(ToolPipelineError):
            await _require_complete_inline_history([row, dict(row)], [row], code="context_budget_exceeded", pipeline=pipeline, lease=lease, wire=True)
    with isolated_test_database():
        asyncio.run(run())
