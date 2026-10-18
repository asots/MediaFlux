"""Actual catalog composition and mutual-exclusion contracts at pipeline ingress."""
from __future__ import annotations

import asyncio
import json
from collections import UserDict
from itertools import product
from types import MappingProxyType

import pytest

from app.agent.domain_catalog import build_tool_specs
from app.agent.domains.guangya.fs_change_actions import guangya_fs_change_arguments
from app.agent.errors import AgentToolError, ToolPipelineError
from app.agent.kernel.capabilities import (
    CapabilityRetriever,
    KernelToolSpec,
    ToolCatalog,
    ToolEffect,
)
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import InMemorySessionStateStore
from app.agent.kernel.state_types import AgentInput, CancellationToken
from app.agent.kernel.tool_calls import ToolCallContext
from tests.test_agent_tool_argument_ingress import Wire, encode, wire


def catalog_schema(name):
    return next(spec.parameters for spec in build_tool_specs() if spec.name == name)


@pytest.mark.parametrize("fields", list(product([False, True], repeat=3)))
def test_create_directory_schema_matches_exact_domain_shapes(fields):
    from app.agent.schema import validate_json_schema

    values = {"path": "/parent/child", "parent_path": "/parent", "name": "child"}
    operation = {"op": "create_directory", **{
        key: value for (key, value), present in zip(values.items(), fields, strict=True) if present}}
    arguments = {"operations": [operation]}
    accepted = set(operation) in ({"op", "path"}, {"op", "parent_path", "name"})
    schema = catalog_schema("guangya.fs.change.execute")
    if accepted:
        validate_json_schema(arguments, schema)
        assert guangya_fs_change_arguments(arguments)["operations"] == [
            {"op": "create_directory", "parent_path": "/parent", "name": "child"}]
    else:
        with pytest.raises(AgentToolError):
            guangya_fs_change_arguments(arguments)
        with pytest.raises(ToolPipelineError):
            validate_json_schema(arguments, schema)


@pytest.mark.parametrize("protocol", ["responses", "chat_completions", "anthropic_messages"])
def test_cloud_operation_schema_survives_native_protocol_projection(protocol):
    from app.agent.kernel.ports.existing_actions import adapt_tool_spec
    from app.clients.openai_compatible import native_tool_definitions

    spec = next(spec for spec in build_tool_specs() if spec.name == "guangya.fs.change.execute")
    native, = native_tool_definitions(protocol, [adapt_tool_spec(spec).model_definition()])
    native = json.loads(json.dumps(native))
    if protocol == "chat_completions":
        native = native["function"]
    assert native["input_schema" if protocol == "anthropic_messages" else "parameters"] == spec.parameters


async def execute(schema, arguments, trace, *, validator=None):
    def validate(value):
        trace.append(("validate", value))
        return validator(value) if validator is not None else value

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
    ("guangya.fs.change.execute", {"operations": [{"op": "batch_relocate", "items": [], "target_path": "test"}]}),
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
    ("guangya.fs.change.execute",
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
        transport = Wire("chat_completions", [wire("chat_completions", "json", encode(invalid)),
                         wire("chat_completions", "json", encode(valid)), (encode(final).encode(), "application/json")])
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


@pytest.mark.parametrize("extra,normalized_extra", [
    ({}, {}),
    ({"new_name": "a"}, {"new_name": "a"}),
    ({"new_name": " 副本.mkv "}, {"new_name": "副本.mkv"}),
    ({"new_name": "副" * 255}, {"new_name": "副" * 255}),
])
def test_copy_preview_schema_and_validator_normalize_optional_name(extra, normalized_extra):
    spec = next(spec for spec in build_tool_specs() if spec.name == "guangya.fs.change.execute")
    assert spec.validator is guangya_fs_change_arguments
    arguments = {"operations": [{"op": "copy", "object_ref": "OBJ" + "a" * 24,
                                 "target_path": "backup//child\\", **extra}]}
    original = json.loads(json.dumps(arguments))
    expected = {"observation_ref": "", "operations": [
        {"op": "copy", "object_ref": "OBJ" + "A" * 24,
         "target_path": "/backup/child", **normalized_extra}], "trigger_strm": True}
    trace = []
    result = asyncio.run(execute(spec.parameters, arguments, trace, validator=spec.validator))
    assert result.arguments == expected
    assert trace == [("validate", arguments), ("read", expected)]
    assert arguments == original


@pytest.mark.parametrize("new_name", [None, True, 1, [], {}, "", "a" * 256],
                         ids=["null", "boolean", "integer", "array", "object", "empty", "too-long"])
def test_copy_preview_schema_and_validator_reject_invalid_name(new_name):
    arguments = {"operations": [{"op": "copy", "object_ref": "OBJ" + "A" * 24,
                                 "target_path": "/backup", "new_name": new_name}]}
    trace = []
    with pytest.raises(ToolPipelineError) as error:
        asyncio.run(execute(catalog_schema("guangya.fs.change.execute"), arguments, trace,
                            validator=guangya_fs_change_arguments))
    assert error.value.code == "invalid_arguments"
    assert trace == []
    with pytest.raises(AgentToolError, match="new_name 长度必须在 1 到 255 之间"):
        guangya_fs_change_arguments(arguments)


@pytest.mark.parametrize("new_name", [" ", "\t\n"])
def test_copy_preview_validator_rejects_name_empty_after_trimming(new_name):
    arguments = {"operations": [{"op": "copy", "object_ref": "OBJ" + "A" * 24,
                                 "target_path": "/backup", "new_name": new_name}]}
    trace = []
    with pytest.raises(ToolPipelineError, match="长度必须在 1 到 255 之间") as error:
        asyncio.run(execute(catalog_schema("guangya.fs.change.execute"), arguments, trace,
                            validator=guangya_fs_change_arguments))
    assert error.value.code == "invalid_arguments"
    assert trace == [("validate", arguments)]


@pytest.mark.parametrize("operation", [
    {"op": "copy", "object_ref": "OBJ" + "A" * 24, "new_name": "副本.mkv"},
    {"op": "copy", "target_path": "/backup", "new_name": "副本.mkv"},
    {"op": "copy", "object_ref": "OBJ" + "A" * 24, "target_path": "/backup",
     "new_name": "副本.mkv", "name": "其他.mkv"},
    {"op": "move", "object_ref": "OBJ" + "A" * 24, "target_path": "/backup", "new_name": "副本.mkv"},
    {"op": "trash", "object_ref": "OBJ" + "A" * 24, "new_name": "副本.mkv"},
], ids=["missing-target", "missing-object", "extra-field", "move-name", "trash-name"])
def test_copy_preview_optional_name_keeps_required_and_other_operation_fields(operation):
    arguments = {"operations": [operation]}
    trace = []
    with pytest.raises(ToolPipelineError) as error:
        asyncio.run(execute(catalog_schema("guangya.fs.change.execute"), arguments, trace))
    assert error.value.code == "invalid_arguments"
    assert trace == []
    with pytest.raises(AgentToolError, match="光鸭变更操作字段与 op 不匹配"):
        guangya_fs_change_arguments(arguments)


def test_copy_preview_catalog_uses_one_existing_operation_with_optional_name():
    spec = next(spec for spec in build_tool_specs() if spec.name == "guangya.fs.change.execute")
    branches = spec.parameters["properties"]["operations"]["items"]["oneOf"]
    copy_schema, = [branch for branch in branches if branch["properties"]["op"]["enum"] == ["copy"]]
    assert copy_schema["required"] == ["op", "object_ref", "target_path"]
    assert set(copy_schema["properties"]) == {"op", "object_ref", "target_path", "new_name"}
    assert copy_schema["additionalProperties"] is False
    for op in ("copy", "rename", "relocate"):
        branch = next(branch for branch in branches if branch["properties"]["op"]["enum"] == [op])
        assert branch["properties"]["new_name"]["type"] == "string"
        assert branch["properties"]["new_name"]["minLength"] == 1
        assert branch["properties"]["new_name"]["maxLength"] == 255
    assert {op for branch in branches for op in branch["properties"]["op"]["enum"]} == {
        "rename", "move", "copy", "relocate", "batch_relocate", "trash", "create_directory"}
    assert "同一计划" in spec.description
    assert "副本改名" in spec.description
    assert "源对象不变" in spec.description
    assert "冲突必须拒绝" in spec.description


def test_copy_preview_mixes_legacy_and_named_copies_in_one_plan():
    legacy = {"op": "copy", "object_ref": "OBJ" + "A" * 24, "target_path": "/backup"}
    named = {"op": "copy", "object_ref": "OBJ" + "B" * 24,
             "target_path": "/backup", "new_name": "副本.mkv"}
    arguments = {"observation_ref": "OBS" + "A" * 32,
                 "operations": [legacy, named], "trigger_strm": False}
    trace = []
    result = asyncio.run(execute(catalog_schema("guangya.fs.change.execute"), arguments, trace,
                                validator=guangya_fs_change_arguments))
    assert result.arguments == arguments
    assert "new_name" not in result.arguments["operations"][0]
    assert trace == [("validate", arguments), ("read", arguments)]
