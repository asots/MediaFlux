"""写侧修复仍锁合法字段；READ 重试只受当前 schema 与业务门禁约束。"""
from __future__ import annotations

import asyncio

import pytest

from app.agent.errors import ToolPipelineError
from app.agent.kernel.capabilities import KernelToolSpec, ToolCatalog, ToolEffect
from app.agent.kernel.effects import PreparedEffect
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.state import InMemorySessionStateStore
from app.agent.kernel.state_types import CancellationToken
from app.agent.kernel.tool_calls import ToolCallContext


@pytest.mark.parametrize('composition', ['oneOf', 'anyOf'])
@pytest.mark.parametrize('closed_branches', [True, False])
@pytest.mark.parametrize('effect', [ToolEffect.WRITE, ToolEffect.DANGER])
def test_union_write_repair_can_remove_only_proven_unknown_fields(composition, closed_branches, effect):
    async def run():
        branches = [
            {'type': 'object', 'required': ['operation', 'arguments'],
             'additionalProperties': False if operation == 'first' else not closed_branches,
             'properties': {'operation': {'enum': [operation]},
                            'arguments': {'type': 'object'}}}
            for operation in ('first', 'second')
        ]
        schema = {'type': 'object', 'required': ['steps'], 'properties': {
            'steps': {'type': 'array', 'items': {composition: branches}}}}
        seen = []
        tool = KernelToolSpec(name='lookup', domain='library', description='union repair',
            input_schema=schema, effect=effect,
            prepare=lambda args, _: seen.append(args) or PreparedEffect(preview={'summary': 'preview'}, snapshot_fingerprint='snapshot'),
            execute_confirmed=lambda *_: pytest.fail('repair cannot authorize writes'))
        store = InMemorySessionStateStore()
        lease, _ = await store.begin_turn(owner='owner', session_id='union', request_id='r')
        async def progress(payload): pass
        context = ToolCallContext(owner=lease.owner, session_id=lease.session_id,
            request_id=lease.request_id, turn_id=lease.turn_id, lease=lease,
            cancellation=CancellationToken(), report_progress=progress)
        pipeline = ToolPipeline(catalog=ToolCatalog([tool]), state_store=store)
        with pytest.raises(ToolPipelineError) as error:
            await pipeline.execute('lookup', {'steps': [
                {'operation': 'first', 'parameters': {'limit': 2}}]}, context=context)
        ticket = error.value.repair_ticket
        assert ticket and not seen
        # Removing an invalid key must not unlock the valid operation selector.
        with pytest.raises(ToolPipelineError, match='不能更改原有效参数'):
            await pipeline.execute('agent.repair', {'ticket': ticket, 'arguments': {
                'steps': [{'operation': 'second', 'arguments': {'limit': 2}}]}}, context=context)
        fixed = {'steps': [{'operation': 'first', 'arguments': {'limit': 2}}]}
        if closed_branches:
            result = await pipeline.execute('agent.repair', {'ticket': ticket, 'arguments': fixed}, context=context)
            assert result.effect_plan and not result.recovered_failures
            assert seen == [fixed]
        else:
            # An open alternative prevents inferring that this key is unknown.
            with pytest.raises(ToolPipelineError, match='不能更改原有效参数'):
                await pipeline.execute('agent.repair', {'ticket': ticket, 'arguments': fixed}, context=context)
            assert not seen
    asyncio.run(run())


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
@pytest.mark.parametrize('effect', [ToolEffect.WRITE, ToolEffect.DANGER])
def test_write_collision_repair_preserves_branch_and_unrelated_keys(composition, correction, allowed, effect):
    async def run():
        schema = {"type": "object", "required": ["identity"], "properties": {
            key: {"type": "string"} for key in ("identity", "left", "right", "filter")
        }, **composition}
        executed = []
        tool = KernelToolSpec(name="lookup", domain="library", description="offline",
                              input_schema=schema, effect=effect, validator=dict,
                              prepare=lambda args, ctx: executed.append(args) or PreparedEffect(preview={"summary": "preview"}, snapshot_fingerprint="snapshot"),
                              execute_confirmed=lambda *_: pytest.fail("repair cannot authorize writes"))
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
@pytest.mark.parametrize('effect', [ToolEffect.WRITE, ToolEffect.DANGER])
def test_write_value_errors_and_complex_branches_do_not_unlock_omission(composition, first, effect):
    async def run():
        schema = {"type": "object", "properties": {
            "left": {"type": "string"}, "right": {"type": "string"}}, **composition}
        tool = KernelToolSpec(name="lookup", domain="library", description="offline",
                              input_schema=schema, effect=effect, validator=dict,
                              prepare=lambda args, ctx: pytest.fail("rejected arguments prepared"),
                              execute_confirmed=lambda *_: pytest.fail("repair cannot authorize writes"))
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
    ({"operation": "list", "paths": ["/a", "/b"], "identity": "B", "page_size": 5}, True),
    ({"operation": "stat", "path": "/a", "identity": "A"}, True),
    ({"operation": "stat", "paths": ["/a", "/b"], "identity": "A", "page_size": 5}, False),
    ({"identity": "A", "page_size": 5}, False),
])
@pytest.mark.parametrize("effect", [ToolEffect.READ, ToolEffect.WRITE, ToolEffect.DANGER])
@pytest.mark.parametrize("read_status", ["success", "found", "empty"])
def test_conditional_read_retry_uses_schema_while_writes_keep_locks(correction, allowed, effect, read_status):
    from app.agent.kernel.effects import PreparedEffect
    from tests.test_agent_schema_composition import catalog_schema

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
            read=read if effect is ToolEffect.READ else None,
            prepare=None if effect is ToolEffect.READ else prepare,
            execute_confirmed=None if effect is ToolEffect.READ else lambda plan, ctx: {"ok": True})
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
            assert not caught.value.repair_ticket and not context.schema_replacements
            result = await pipeline.execute("lookup", correction, context=context)
            assert result.arguments == correction
            assert not result.recovered_failures
            assert executed == [correction]
        else:
            with pytest.raises(ToolPipelineError):
                if effect is ToolEffect.READ:
                    await pipeline.execute("lookup", correction, context=context)
                else:
                    await pipeline.execute("agent.repair", repair, context=context)
            assert not executed
    asyncio.run(run())
