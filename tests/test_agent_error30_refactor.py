"""READ 直接修参；WRITE/DANGER 仍由真实 Session/Pipeline 校验修复证据。"""
import asyncio
import json

import pytest

from app.agent.kernel.capabilities import CapabilityRetriever, KernelToolSpec, ToolCatalog, ToolEffect
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
from app.agent.kernel.effects import PreparedEffect
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.projection import ToolOutcome
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import AgentInput
from tests.test_agent_invalid_argument_progress import scenario, store, X  # noqa: F401


@pytest.mark.parametrize("stage", ["schema", "validator", "admission", "handler"])
def test_read_invalid_arguments_returns_schema_without_ticket_or_field_locks(stage):
    from app.agent.kernel.pipeline import ToolCallContext, ToolPipelineError
    from app.agent.kernel.state import CancellationToken, InMemorySessionStateStore

    async def run():
        invoked = []
        schema = {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}}}

        def check(arguments):
            if arguments["query"] == "old":
                raise ToolPipelineError("query 形状无效", code="invalid_arguments", invalid_fields=("query",))
            return arguments

        def read(arguments, _context):
            if stage == "handler":
                check(arguments)
            invoked.append(arguments)
            return {"ok": True, "status": "found"}

        class Authorization:
            async def authorize(self, _tool, arguments, _context):
                if stage == "admission":
                    check(arguments)

        tool = KernelToolSpec(name="library.lookup", domain="library", description="READ retry",
            input_schema=schema, effect=ToolEffect.READ,
            validator=check if stage == "validator" else dict, read=read)
        state = InMemorySessionStateStore()
        lease, _ = await state.begin_turn(owner="owner", session_id="direct", request_id="r")
        ctx = ToolCallContext(owner=lease.owner, session_id=lease.session_id,
            request_id=lease.request_id, turn_id=lease.turn_id, lease=lease,
            cancellation=CancellationToken(), report_progress=lambda _: asyncio.sleep(0))
        pipeline = ToolPipeline(catalog=ToolCatalog([tool]), state_store=state, authorization=Authorization())
        bad = {"query": "old", "limit": "bad" if stage == "schema" else 1}
        with pytest.raises(ToolPipelineError) as caught:
            await pipeline.execute(tool.name, bad, context=ctx)
        assert caught.value.code == "invalid_arguments"
        assert caught.value.repair_ticket == "" and ctx.schema_replacements == {}
        feedback = json.loads(AgentSession._tool_error_message(
            ModelToolCall("bad", tool.name, bad), caught.value, input_schema=schema).content)
        assert feedback["expected_input_schema"] == schema
        assert "repair" not in feedback
        # READ requests are not write approvals: changing a valid selector or
        # dropping an optional field still uses the registered schema and gates.
        result = await pipeline.execute(tool.name, {"query": "new"}, context=ctx)
        assert invoked == [{"query": "new"}]
        assert not result.recovered_failures and result.effect_plan is None
        assert ctx.schema_replacements == {}

    asyncio.run(run())


def test_unparsed_read_does_not_create_server_repair_authority():
    from app.agent.kernel import error_repair
    from app.agent.kernel.pipeline import ToolCallContext
    from app.agent.kernel.state import CancellationToken, InMemorySessionStateStore

    async def run():
        state = InMemorySessionStateStore()
        lease, _ = await state.begin_turn(owner="owner", session_id="unparsed-read", request_id="r")
        context = ToolCallContext(owner=lease.owner, session_id=lease.session_id,
            request_id=lease.request_id, turn_id=lease.turn_id, lease=lease,
            cancellation=CancellationToken(), report_progress=lambda _: asyncio.sleep(0))
        tool = KernelToolSpec(name="library.lookup", domain="library", description="READ",
            input_schema={"type": "object"}, effect=ToolEffect.READ, read=lambda *_: {"ok": True})
        assert error_repair.issue(tool, None, context) == ""
        assert context.schema_replacements == {}

    asyncio.run(run())


def test_successful_read_retry_settles_syntax_not_object_evidence(store):  # noqa: F811 - imported pytest fixture
    final, _ = asyncio.run(scenario(store, {**X, "ref": "object-y", "limit": 1}, mixed=False))
    assert final["status"] == "success"


@pytest.mark.parametrize("mode", ["repair", "unrelated", "forged", "conflict", "incomplete"])
@pytest.mark.parametrize("effect", [ToolEffect.WRITE, ToolEffect.DANGER])
def test_explicit_write_repair_real_session(store, mode, effect):  # noqa: F811 - imported pytest fixture
    async def run():
        calls = []
        schema = {"type": "object", "required": ["target", "limit"], "properties": {
            "target": {"type": "string"}, "limit": {"type": "integer"}}, "additionalProperties": False}
        def prepare(arguments, context):
            calls.append(arguments)
            if mode == "incomplete":
                return PreparedEffect(preview={"summary": "需确认"}, snapshot_fingerprint="snapshot")
            return ToolOutcome(model_content="无需变更", public_content={"ok": True, "status": "no_changes", "summary": "checked target"})
        tool = KernelToolSpec(name="library.check", domain="library", description="check", input_schema=schema,
            effect=effect, prepare=prepare, execute_confirmed=lambda *_: pytest.fail("repair is not write authorization"))
        class Model:
            def __init__(self):
                self.requests = []
            async def stream(self, request, **kwargs):
                self.requests.append(request)
                index = len(self.requests)
                if index == 1:
                    args = {"target": "x", "limit": "bad"} if mode == "conflict" else {}
                    call = ModelToolCall("failed", tool.name, args)
                elif index == 2:
                    failure = json.loads(next(m.content for m in request.messages if m.role == "tool"))
                    if mode == "unrelated":
                        call = ModelToolCall("next", tool.name, {"target": "x", "limit": 1})
                    else:
                        self.advertised = "repair" in failure and any(t["name"] == "agent__repair" for t in request.tools)
                        repair = failure.get("repair", {"ticket": "missing-server-entry"})
                        call = ModelToolCall("next", "agent.repair", {
                            "ticket": "forged" if mode == "forged" else repair["ticket"],
                            "arguments": {"target": "y" if mode == "conflict" else "x", "limit": 1}})
                else:
                    yield ModelEvent(ModelEventType.TEXT_DELTA, text="核验完成。")
                    yield ModelEvent(ModelEventType.FINISH, finish_reason="stop")
                    return
                yield ModelEvent(ModelEventType.TOOL_CALL_COMPLETED, tool_call=call)
                yield ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")
        model = Model()
        catalog = ToolCatalog([tool])
        session = AgentSession(model=model, catalog=catalog, retriever=CapabilityRetriever(minimum=1, maximum=1),
                               pipeline=ToolPipeline(catalog=catalog, state_store=store), state_store=store, journal=store)
        events = [e async for e in session.run(AgentInput(message="check", owner="owner", session_id="repair"))]
        final = events[-1].payload
        if mode != "unrelated":
            assert model.advertised, "server repair entry absent from real model request"
        if mode in {"forged", "conflict"}:
            assert events[-1].type.value == "turn.failed"
        else:
            expected = "approval_required" if mode == "incomplete" else "success" if mode == "repair" else "partial"
            assert final["status"] == expected, final
        assert len(calls) == (0 if mode in {"forged", "conflict"} else 1)
        from app.agent.kernel.persistence import SQLiteKernelStore
        cold = SQLiteKernelStore(secret_provider=lambda: "offline-invalid-progress")
        saved = await cold.load(owner="owner", session_id="repair")
        if "answer" in final:
            assert saved.conversation[-1]["content"] == final["answer"]
        assert await cold.list_events(owner="owner", session_id="repair") == [e.to_dict() for e in events]
        assert "ticket" not in json.dumps(final)
    asyncio.run(run())


@pytest.mark.parametrize("field", ["owner", "session_id", "generation", "turn_id"])
@pytest.mark.parametrize("effect", [ToolEffect.WRITE, ToolEffect.DANGER])
def test_write_repair_ticket_scope_and_single_consumption(field, effect):
    from dataclasses import replace
    from app.agent.kernel.pipeline import ToolCallContext, ToolPipelineError
    from app.agent.kernel.state import CancellationToken, InMemorySessionStateStore

    async def run():
        invoked = []
        tool = KernelToolSpec(name="library.lookup", domain="library", description="lookup",
            input_schema={"type": "object", "required": ["target"], "properties": {"target": {"type": "string"}}},
            effect=effect,
            prepare=lambda args, ctx: invoked.append(args) or ToolOutcome(model_content="无需变更",
                public_content={"ok": True, "status": "no_changes"}),
            execute_confirmed=lambda *_: pytest.fail("repair cannot execute a write"))
        state = InMemorySessionStateStore()
        pipeline = ToolPipeline(catalog=ToolCatalog([tool]), state_store=state)
        lease, _ = await state.begin_turn(owner="owner", session_id="scope", request_id="r")
        context = ToolCallContext(owner="owner", session_id="scope", request_id="r", turn_id=lease.turn_id,
            lease=lease, cancellation=CancellationToken(), report_progress=lambda _: asyncio.sleep(0))
        with pytest.raises(ToolPipelineError) as failure:
            await pipeline.execute(tool.name, {}, context=context)
        ticket = failure.value.repair_ticket
        assert ticket
        other = replace(context, lease=replace(lease, generation=lease.generation+1)) if field == "generation" else replace(context, **{field: "other"})
        args = {"ticket": ticket, "arguments": {"target": "x"}}
        with pytest.raises(ToolPipelineError, match="不属于本轮"):
            await pipeline.execute("agent.repair", args, context=other)
        assert not invoked
        result = await pipeline.execute("agent.repair", args, context=context)
        assert result.recovered_failures == (ticket,)
        with pytest.raises(ToolPipelineError, match="不属于本轮"):
            await pipeline.execute("agent.repair", args, context=context)
        assert invoked == [{"target": "x"}]
    asyncio.run(run())


@pytest.mark.parametrize("effect", [ToolEffect.WRITE, ToolEffect.DANGER])
def test_repair_chain_keeps_provenance_and_actual_authorization(effect):
    from app.agent.kernel.pipeline import ToolCallContext, ToolPipelineError
    from app.agent.kernel.state import CancellationToken, InMemorySessionStateStore
    from app.agent.kernel.effects import PreparedEffect

    async def run():
        writes = []
        allowed = [True]
        tool = KernelToolSpec(name="cloud.change", domain="cloud", description="change",
            input_schema={"type": "object", "required": ["target", "count"], "properties": {
                "target": {"type": "string"}, "count": {"type": "integer"}}},
            effect=effect, authorize=lambda _: allowed[0],
            prepare=lambda args, ctx: PreparedEffect(preview={"summary": "待确认"}, snapshot_fingerprint="observed"),
            execute_confirmed=lambda args, snapshot, ctx: writes.append(args))
        state = InMemorySessionStateStore()
        pipeline = ToolPipeline(catalog=ToolCatalog([tool]), state_store=state)
        lease, _ = await state.begin_turn(owner="owner", session_id="scope", request_id="r")
        ctx = ToolCallContext(owner="owner", session_id="scope", request_id="r", turn_id=lease.turn_id,
            lease=lease, cancellation=CancellationToken(), report_progress=lambda _: asyncio.sleep(0))
        with pytest.raises(ToolPipelineError) as first:
            await pipeline.execute(tool.name, {"target": "x"}, context=ctx)
        with pytest.raises(ToolPipelineError) as second:
            await pipeline.execute("agent.repair", {"ticket": first.value.repair_ticket,
                "arguments": {"target": "x", "count": "bad"}}, context=ctx)
        ticket = second.value.repair_ticket
        assert ticket and ctx.schema_replacements[ticket].predecessors == (first.value.repair_ticket,)
        result = await pipeline.execute("agent.repair", {"ticket": ticket,
            "arguments": {"target": "x", "count": 1}}, context=ctx)
        assert result.effect_plan and result.effect_plan.tool_name == tool.name
        assert not result.recovered_failures  # frozen preview is NOT completed work
        assert not writes
        with pytest.raises(ToolPipelineError) as third:
            await pipeline.execute(tool.name, {"target": "y"}, context=ctx)
        allowed[0] = False
        with pytest.raises(ToolPipelineError):
            await pipeline.execute("agent.repair", {"ticket": third.value.repair_ticket,
                "arguments": {"target": "y", "count": 1}}, context=ctx)
        assert not writes
    asyncio.run(run())


def test_read_direct_retry_needs_no_ancestor_tickets(store):  # noqa: F811
    from tests.test_agent_kernel_core import ScriptedModel
    calls = []
    spec = KernelToolSpec(name="library.check", domain="library", description="check",
        input_schema={"type": "object", "required": ["target", "count"], "properties": {
            "target": {"type": "string"}, "count": {"type": "integer"}}},
        effect=ToolEffect.READ, read=lambda args, ctx: calls.append(args) or {"ok": True, "status": "success"})
    rounds = [[ModelEvent(ModelEventType.TOOL_CALL_COMPLETED, tool_call=ModelToolCall(cid, spec.name, args)),
               ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")]
              for cid, args in [("first", {}), ("second", {"target": "x"}), ("third", {"target": "x", "count": 1})]]
    rounds.append([ModelEvent(ModelEventType.TEXT_DELTA, text="核验完成"), ModelEvent(ModelEventType.FINISH, finish_reason="stop")])
    model = ScriptedModel(rounds)
    catalog = ToolCatalog([spec])
    session = AgentSession(model=model, catalog=catalog, retriever=CapabilityRetriever(minimum=1, maximum=1),
        pipeline=ToolPipeline(catalog=catalog, state_store=store), state_store=store)
    async def run():
        return [event async for event in session.run(AgentInput(owner="owner", session_id="chain", message="check"))]
    events = asyncio.run(run())
    assert events[-1].payload["status"] == "success", events[-1].payload
    assert calls == [{"target": "x", "count": 1}]


@pytest.mark.parametrize("effect", [ToolEffect.WRITE, ToolEffect.DANGER])
def test_write_domain_validation_only_offers_actionable_field_repairs(effect):
    from app.agent.kernel.pipeline import ToolCallContext, ToolPipelineError
    from app.agent.kernel.state import CancellationToken, InMemorySessionStateStore

    async def run():
        state = InMemorySessionStateStore()
        lease, _ = await state.begin_turn(owner="owner", session_id="domain", request_id="r")
        ctx = ToolCallContext(owner="owner", session_id="domain", request_id="r", turn_id=lease.turn_id,
            lease=lease, cancellation=CancellationToken(), report_progress=lambda _: asyncio.sleep(0))
        for annotated in (False, True):
            invoked = []
            def validator(args):
                if args["path"] != "/text":
                    raise ToolPipelineError("only /text", code="invalid_arguments",
                                            invalid_fields=("path",) if annotated else ())
                return args
            tool = KernelToolSpec(name="library.read", domain="library", description="read",
                effect=effect, input_schema={"type": "object", "required": ["path", "target"],
                    "properties": {"path": {"type": "string"}, "target": {"type": "string"}}},
                validator=validator,
                prepare=lambda args, ctx: invoked.append(args) or ToolOutcome(model_content="无需变更",
                    public_content={"ok": True, "status": "no_changes"}),
                execute_confirmed=lambda *_: pytest.fail("repair cannot execute a write"))
            pipeline = ToolPipeline(catalog=ToolCatalog([tool]), state_store=state)
            with pytest.raises(ToolPipelineError) as error:
                await pipeline.execute(tool.name, {"path": "/other", "target": "x"}, context=ctx)
            assert bool(error.value.repair_ticket) is annotated
            if annotated:
                ticket = error.value.repair_ticket
                with pytest.raises(ToolPipelineError, match="原有效参数"):
                    await pipeline.execute("agent.repair", {"ticket": ticket,
                        "arguments": {"path": "/text", "target": "y"}}, context=ctx)
                result = await pipeline.execute("agent.repair", {"ticket": ticket,
                    "arguments": {"path": "/text", "target": "x"}}, context=ctx)
                assert result.recovered_failures == (ticket,)
                assert invoked == [{"path": "/text", "target": "x"}]
            else:
                assert not invoked
    asyncio.run(run())


def test_actual_reference_reader_directly_retries_path_with_scope_checks():
    from app.agent.kernel.pipeline import ToolCallContext, ToolPipelineError
    from app.agent.kernel.state import CancellationToken, InMemorySessionStateStore
    from app.agent.kernel import telegram_reference

    async def run():
        state = InMemorySessionStateStore()
        pipeline = ToolPipeline(catalog=ToolCatalog([]), state_store=state)
        lease, _ = await state.begin_turn(owner="owner", session_id="reader", request_id="r")
        reply = await telegram_reference.prepare({"text": "keep this reference", "source": {
            "kind": "telegram", "chat_id": 1, "message_id": 2}}, lease, pipeline.reference_store)
        await state.commit(lease, conversation=[{"role": "user", "content": "read", "reply_context": reply}])
        ctx = ToolCallContext(owner="owner", session_id="reader", request_id="r", turn_id=lease.turn_id,
            lease=lease, cancellation=CancellationToken(), report_progress=lambda _: asyncio.sleep(0))
        args = {"handle": reply["reference_handle"], "path": "/other"}
        with pytest.raises(ToolPipelineError) as error:
            await pipeline.execute("agent.read_result", args, context=ctx)
        assert error.value.code == "invalid_arguments" and not error.value.repair_ticket
        assert not ctx.schema_replacements
        result = await pipeline.execute("agent.read_result", {**args, "path": "/text"}, context=ctx)
        assert not result.recovered_failures
        assert json.loads(result.outcome.model_content)["data"]["items"] == "keep this reference"
    asyncio.run(run())


@pytest.mark.parametrize("repair_both", [False, True])
@pytest.mark.parametrize("effect", [ToolEffect.WRITE, ToolEffect.DANGER])
def test_two_unparsed_writes_keep_independent_failure_identities(store, repair_both, effect):  # noqa: F811
    calls = []
    spec = KernelToolSpec(name="library.check", domain="library", description="check",
        input_schema={"type": "object", "required": ["target"], "properties": {"target": {"type": "string"}}},
        effect=effect,
        prepare=lambda args, ctx: calls.append(args) or ToolOutcome(model_content="无需变更",
            public_content={"ok": True, "status": "no_changes", "summary": "checked"}),
        execute_confirmed=lambda *_: pytest.fail("repair cannot execute a write"))
    class Model:
        def __init__(self):
            self.round = 0
        async def stream(self, request, **kwargs):
            self.round += 1
            if self.round == 1:
                for i in range(2):
                    yield ModelEvent(ModelEventType.TOOL_CALL_COMPLETED, tool_call=ModelToolCall(
                        f"unparsed-{i}", spec.name, None, argument_error="invalid_json", raw_arguments='{bad'))
            elif self.round == 2:
                entries = [json.loads(m.content)["repair"] for m in request.messages if m.role == "tool"]
                assert len(entries) == 2 and entries[0]["ticket"] != entries[1]["ticket"]
                for i in ([0, 1] if repair_both else [1]):
                    yield ModelEvent(ModelEventType.TOOL_CALL_COMPLETED, tool_call=ModelToolCall(
                        f"repair-{i}", "agent.repair", {"ticket": entries[i]["ticket"], "arguments": {"target": str(i)}}))
            else:
                yield ModelEvent(ModelEventType.TEXT_DELTA, text="已核验")
            yield ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls" if self.round < 3 else "stop")
    catalog = ToolCatalog([spec])
    session = AgentSession(model=Model(), catalog=catalog, retriever=CapabilityRetriever(minimum=1, maximum=1),
        pipeline=ToolPipeline(catalog=catalog, state_store=store), state_store=store)
    async def run():
        return [event async for event in session.run(AgentInput(owner="owner", session_id="unparsed", message="check both"))]
    events = asyncio.run(run())
    assert events[-1].payload["status"] == ("success" if repair_both else "partial"), events[-1].payload
    assert calls == ([{"target": "0"}, {"target": "1"}] if repair_both else [{"target": "1"}])
