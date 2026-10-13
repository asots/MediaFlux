"""Schema-proven omission is limited to a conflicting key pair, never filters."""
from __future__ import annotations

import asyncio

import pytest

from app.agent.kernel.capabilities import KernelToolSpec, ToolCatalog, ToolEffect
from app.agent.kernel.pipeline import ToolCallContext, ToolPipeline, ToolPipelineError
from app.agent.kernel.state import CancellationToken, InMemorySessionStateStore


@pytest.mark.parametrize("composition", [
    {"not": {"required": ["left", "right"]}},
    {"oneOf": [{"required": ["left"]}, {"required": ["right"]}]},
])
@pytest.mark.parametrize("correction,allowed", [
    ({"identity": "A", "left": "x", "filter": "narrow"}, True),
    ({"identity": "A", "right": "y", "filter": "narrow"}, True),
    ({"identity": "A", "left": "x"}, False),
    ({"identity": "A", "filter": "narrow"}, False),
    ({"identity": "A", "left": 4, "filter": "narrow"}, False),
    ({"identity": "A", "left": "x", "right": "y", "filter": "narrow"}, False),
])
def test_collision_repair_preserves_branch_and_unrelated_keys(composition, correction, allowed):
    async def run():
        schema = {"type": "object", "required": ["identity"], "properties": {
            key: {"type": "string"} for key in ("identity", "left", "right", "filter")
        }, **composition}
        executed = []
        tool = KernelToolSpec(name="lookup", domain="library", description="offline",
                              input_schema=schema, effect=ToolEffect.READ, validator=dict,
                              read=lambda args, ctx: executed.append(args) or {"ok": True})
        store = InMemorySessionStateStore()
        lease, _ = await store.begin_turn(owner="owner", session_id="repair", request_id="r")

        async def progress(payload):
            pass

        context = ToolCallContext(owner="owner", session_id="repair", request_id="r",
                                  turn_id=lease.turn_id, lease=lease, cancellation=CancellationToken(),
                                  report_progress=progress)
        pipeline = ToolPipeline(catalog=ToolCatalog([tool]), state_store=store)
        initial = {"identity": "A", "left": "x", "right": "y", "filter": "narrow"}
        with pytest.raises(ToolPipelineError) as error:
            await pipeline.execute("lookup", initial, context=context)
        assert error.value.code == "invalid_arguments"
        assert executed == []
        if allowed:
            result = await pipeline.execute("lookup", correction, context=context)
            assert result.arguments == correction
            assert executed == [correction]
        else:
            with pytest.raises(ToolPipelineError) as error:
                await pipeline.execute("lookup", correction, context=context)
            assert error.value.code == "invalid_arguments"
            assert executed == []
        # A different required identity has independent repair state.
        other = {"identity": "B", "left": "z"}
        assert (await pipeline.execute("lookup", other, context=context)).arguments == other
        # Prior A constraints are not discharged by B's successful execution.
        if not allowed:
            with pytest.raises(ToolPipelineError):
                await pipeline.execute("lookup", {"identity": "A", "left": "x"}, context=context)
    asyncio.run(run())


@pytest.mark.parametrize("composition,first", [
    ({"not": {"required": ["left", "right"]}}, {"left": 3, "right": "y"}),
    ({"oneOf": [{"required": ["left"], "properties": {"left": {"enum": ["x"]}}},
                {"required": ["right"]}]}, {"left": "x", "right": "y"}),
])
def test_value_errors_and_complex_branches_do_not_unlock_omission(composition, first):
    async def run():
        schema = {"type": "object", "properties": {
            "left": {"type": "string"}, "right": {"type": "string"}}, **composition}
        tool = KernelToolSpec(name="lookup", domain="library", description="offline",
                              input_schema=schema, effect=ToolEffect.READ, validator=dict,
                              read=lambda args, ctx: pytest.fail("rejected arguments executed"))
        store = InMemorySessionStateStore()
        lease, _ = await store.begin_turn(owner="owner", session_id="closed", request_id="r")

        async def progress(payload):
            pass

        context = ToolCallContext(owner="owner", session_id="closed", request_id="r",
                                  turn_id=lease.turn_id, lease=lease, cancellation=CancellationToken(),
                                  report_progress=progress)
        pipeline = ToolPipeline(catalog=ToolCatalog([tool]), state_store=store)
        for arguments in (first, {"right": "y"}):
            with pytest.raises(ToolPipelineError) as error:
                await pipeline.execute("lookup", arguments, context=context)
            assert error.value.code == "invalid_arguments"
    asyncio.run(run())


@pytest.mark.parametrize("correction,allowed", [
    ({"operation": "list", "paths": ["/a", "/b"], "identity": "A", "page_size": 5}, True),
    ({"operation": "stat", "path": "/a", "identity": "A", "page_size": 5}, True),
    ({"operation": "list", "paths": ["/a", "/b"], "identity": "B", "page_size": 5}, False),
    ({"operation": "stat", "path": "/a", "identity": "A"}, False),
    ({"operation": "stat", "paths": ["/a", "/b"], "identity": "A", "page_size": 5}, False),
    ({"identity": "A", "page_size": 5}, False),
])
@pytest.mark.parametrize("effect", [ToolEffect.READ, ToolEffect.WRITE, ToolEffect.DANGER])
@pytest.mark.parametrize("read_status", ["success", "found", "empty"])
def test_conditional_not_repairs_only_involved_read_fields(correction, allowed, effect, read_status):
    from tests.test_agent_schema_composition import catalog_schema
    from app.agent.kernel.effects import PreparedEffect

    async def run():
        schema = catalog_schema("guangya.fs.query")
        schema = {**schema, "required": ["identity"], "properties": {
            **schema["properties"], "identity": {"type": "string"}}}
        executed = []
        def read(args, ctx):
            executed.append(args)
            return {"ok": True, "status": read_status}
        def prepare(args, ctx):
            executed.append(args)
            return PreparedEffect(preview={"summary": "preview"}, snapshot_fingerprint="fingerprint")
        tool = KernelToolSpec(name="lookup", domain="library", description="composition contract",
            input_schema=schema, effect=effect, validator=dict,
            **({"read": read} if effect is ToolEffect.READ else
               {"prepare": prepare, "execute_confirmed": lambda plan, ctx: {"ok": True}}))
        store = InMemorySessionStateStore()
        lease, _ = await store.begin_turn(owner="owner", session_id="not-repair", request_id="r")
        async def progress(payload): pass
        context = ToolCallContext(owner="owner", session_id="not-repair", request_id="r",
            turn_id=lease.turn_id, lease=lease, cancellation=CancellationToken(), report_progress=progress)
        pipeline = ToolPipeline(catalog=ToolCatalog([tool]), state_store=store)
        initial = {"operation": "stat", "paths": ["/a", "/b"], "identity": "A", "page_size": 5}
        with pytest.raises(ToolPipelineError) as caught:
            await pipeline.execute("lookup", initial, context=context)
        assert not executed
        repair = {"ticket": caught.value.repair_ticket, "arguments": correction}
        if allowed and effect is ToolEffect.READ:
            result = await pipeline.execute("agent.repair", repair, context=context)
            assert result.arguments == correction
            assert result.recovered_failures
            assert executed == [correction]
        else:
            with pytest.raises(ToolPipelineError):
                await pipeline.execute("agent.repair", repair, context=context)
            assert not executed
    asyncio.run(run())
