"""Actual catalog composition and mutual-exclusion contracts at pipeline ingress."""
from __future__ import annotations

import asyncio
import json
from collections import UserDict
from types import MappingProxyType

import pytest

from app.agent.domain_catalog import build_tool_specs
from app.agent.kernel.capabilities import (
    CapabilityRetriever,
    KernelToolSpec,
    ToolCatalog,
    ToolEffect,
)
from app.agent.kernel.pipeline import ToolCallContext, ToolPipeline, ToolPipelineError
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import (
    AgentInput,
    CancellationToken,
    InMemorySessionStateStore,
)
from tests.test_agent_tool_argument_ingress import encode, wire


def catalog_schema(name):
    return next(spec.parameters for spec in build_tool_specs() if spec.name == name)


async def execute(schema, arguments, trace):
    def validate(value):
        trace.append(("validate", value))
        return value

    tool = KernelToolSpec(name="lookup", domain="library", description="offline schema contract",
                          input_schema=schema, effect=ToolEffect.READ, validator=validate,
                          read=lambda args, ctx: trace.append(("read", args)) or {"ok": True})
    store = InMemorySessionStateStore()
    lease, _ = await store.begin_turn(owner="owner", session_id="composition", request_id="r")

    async def progress(payload):
        pass

    context = ToolCallContext(owner="owner", session_id="composition", request_id="r",
                              turn_id=lease.turn_id, lease=lease, cancellation=CancellationToken(),
                              report_progress=progress)
    return await ToolPipeline(catalog=ToolCatalog([tool]), state_store=store).execute(
        "lookup", arguments, context=context)


@pytest.mark.parametrize("name,arguments", [
    ("discovery.watchlist_summaries", {"mode": "statistics", "before_number": 12}),
    ("discovery.watchlist_summaries", {"mode": "statistics", "before_number": None}),
    ("discovery.watchlist_summaries", {"mode": "stats"}),
    ("agent.capabilities", {"query": "status", "tool_names": ["library.status"]}),
    ("rss.refresh_subscriptions", {"subscription_ids": [1], "scope": "all_configured"}),
    ("library.check_updates", {}),
    ("strm.set_schedule_policy", {}),
    ("library.check_updates", {"query": "A", "queries": ["B"]}),
    ("guangya.fs.change.preview", {"operations": [{"op": "batch_relocate", "items": [], "target_path": "test"}]}),
    ("library.check_updates", {"queries": ["B"], "tmdb_id": "1"}),
    ("library.check_updates", {"queries": ["B"], "season": 1}),
    ("library.check_updates", {"query": "B", "media_type": "movie", "season": 1}),
    ("guangya.fs.query", {"path": "/a", "paths": ["/b"]}),
    ("guangya.fs.query", {"observation_ref": "OBS" + "A" * 32, "path": "/a"}),
    ("guangya.fs.query", {"operation": "stat", "paths": ["/a"]}),
    ("guangya.fs.query", {"operation": "stat", "path": "/a", "query": "A"}),
    ("guangya.fs.query", {"operation": "stat", "path": "/a", "max_depth": 0}),
    ("guangya.fs.query", {"operation": "search", "path": "/a"}),
    ("guangya.fs.query", {}),
])
def test_actual_catalog_rejects_before_domain_or_execution(name, arguments):
    trace = []
    with pytest.raises(ToolPipelineError) as error:
        asyncio.run(execute(catalog_schema(name), arguments, trace))
    assert error.value.code == "invalid_arguments"
    assert trace == []


@pytest.mark.parametrize("name,arguments", [
    ("discovery.watchlist_summaries", {}),
    ("discovery.watchlist_summaries", {"mode": "statistics"}),
    ("discovery.watchlist_summaries", {"mode": "list"}),
    ("discovery.watchlist_summaries", {"before_number": 12}),
    ("discovery.watchlist_summaries", {"mode": "list", "before_number": 12}),
    ("agent.capabilities", {}),
    ("agent.capabilities", {"query": "status"}),
    ("rss.refresh_subscriptions", {"subscription_ids": [1]}),
    ("rss.refresh_subscriptions", {"scope": "all_configured"}),
    ("library.check_updates", {"queries": ["A"], "media_type": "movie"}),
    ("library.check_updates", {"query": "A", "season": 1}),
    ("strm.set_schedule_policy", {"enabled": False}),
    ("guangya.fs.query", {"observation_ref": "OBS" + "A" * 32, "page": 2}),
    ("guangya.fs.query", {"path": "/a", "query": "A", "operation": "tree"}),
    ("guangya.fs.query", {"path": "/a", "operation": "stat"}),
    ("guangya.fs.query", {"paths": ["/a"], "query": "A", "operation": "search"}),
])
def test_actual_catalog_valid_shapes_preserve_arguments(name, arguments):
    trace = []
    assert asyncio.run(execute(catalog_schema(name), arguments, trace)).arguments == arguments
    assert trace == [("validate", arguments), ("read", arguments)]


@pytest.mark.parametrize("schema,value,valid", [
    ({"type": ["integer", "null"]}, True, False),
    ({"type": ["number", "null"]}, False, False),
    ({"type": ["integer", "null"]}, None, True),
    ({"type": ["integer", "null"]}, 1, True),
    ({"type": ["integer", "null"]}, "1", False),
    ({"enum": [1]}, True, False),
    ({"enum": [True]}, 1, False),
    ({"enum": [{"a": [True]}]}, {"a": [1]}, False),
    ({"enum": [{"a": [1], "b": None}]}, {"b": None, "a": [1.0]}, True),
    ({"oneOf": [{"type": "number"}, {"type": "integer"}]}, 1, False),
    ({"oneOf": [{"type": "number"}, {"type": "integer"}]}, 1.5, True),
    ({"anyOf": [{"type": "number"}, {"type": "integer"}]}, 1, True),
    ({"anyOf": [{"type": "number"}, {"type": "integer"}]}, "x", False),
    ({"not": {"enum": [True]}}, 1, True),
    ({"not": {"enum": [True]}}, True, False),
])
def test_composition_semantics(schema, value, valid):
    arguments = {"value": value}
    root = {"type": "object", "properties": {"value": schema}}
    trace = []
    if valid:
        assert asyncio.run(execute(root, arguments, trace)).arguments == arguments
    else:
        with pytest.raises(ToolPipelineError, match="arguments.value") as error:
            asyncio.run(execute(root, arguments, trace))
        assert error.value.code == "invalid_arguments"
        assert trace == []


@pytest.mark.parametrize("mapping", [dict, MappingProxyType, UserDict])
def test_composition_keeps_root_mapping(mapping):
    value = {"query": "A", "season": 1}
    assert asyncio.run(execute(catalog_schema("library.check_updates"), mapping(value), [])).arguments == value


@pytest.mark.parametrize("name,invalid,valid", [
    ("rss.refresh_subscriptions",
     {"subscription_ids": [1], "scope": "all_configured"}, {"subscription_ids": [1]}),
    ("guangya.fs.change.preview",
     {"operations": [{"op": "batch_relocate", "items": [], "target_path": "test"}]},
     {"operations": [{"op": "batch_relocate", "items": [
         {"object_ref": "OBJ" + "A" * 24, "episode": 1}], "target_path": "test"}]}),
])
def test_native_composed_schema_invalid_arguments_corrected_call(name, invalid, valid):
    async def run():
        executed = []
        schema = catalog_schema(name)
        tool = KernelToolSpec(name="lookup", domain="library", description="offline schema contract",
                              input_schema=schema, effect=ToolEffect.READ, validator=dict,
                              read=lambda args, ctx: executed.append(args) or {"ok": True, "summary": "read"})
        final = {"choices": [{"message": {"content": "已核验"}, "finish_reason": "stop"}]}
        from tests.error30_model_inputs import explicit_repair_wire
        transport = explicit_repair_wire("chat_completions", [wire("chat_completions", "json", encode(invalid)),
                         wire("chat_completions", "json", encode(valid)), (encode(final).encode(), "application/json")], index=1, arguments=valid)
        catalog, store = ToolCatalog([tool]), InMemorySessionStateStore()
        session = AgentSession(model=transport.adapter, catalog=catalog, state_store=store,
                               retriever=CapabilityRetriever(minimum=1, maximum=1),
                               pipeline=ToolPipeline(catalog=catalog, state_store=store))
        events = [event async for event in session.run(AgentInput(message="schema contract", owner="owner", session_id="native"))]
        assert executed == [valid], [event.payload for event in events]
        feedback = [message for message in transport.requests[1]["messages"] if message["role"] == "tool"]
        assert json.loads(feedback[-1]["content"])["code"] == "invalid_arguments"
        assert events[-1].payload.get("status") == "success"
        transport.verify(3)
    asyncio.run(run())
