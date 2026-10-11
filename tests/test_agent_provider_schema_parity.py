"""Provider discovery/admission contracts with an isolated transport boundary."""
from __future__ import annotations

import asyncio

import pytest

from app.agent.models import RiskLevel, ToolContext
from app.agent.provider_gateway import ProviderGateway
from app.agent.provider_models import ProviderGatewayError, ProviderOperationSpec, ProviderPayload, ProviderProfileView
from app.agent.provider_operations import build_provider_catalog
from tests.test_agent_schema_composition import catalog_schema, execute
from app.agent.kernel.pipeline import ToolPipelineError


class Transport:
    provider = "media"

    def __init__(self):
        self.calls = []

    def profiles(self):
        return [ProviderProfileView("configured:media", "media", "Offline", "online")]

    def profile_revision(self, profile_ref):
        return "offline-generation"

    def execute_read(self, profile_ref, operation, arguments):
        self.calls.append((operation, arguments))
        return ProviderPayload(summary="offline", data={"count": 0}, source="offline")

    def preview_write(self, profile_ref, operation, arguments, target_snapshot):
        self.calls.append((operation, arguments))
        return ProviderPayload(summary="offline preview", data={"count": 0}, source="offline")


def gateway(schema=None, *, write=False):
    catalog = build_provider_catalog()
    operation = "media.offline.probe"
    if schema is not None:
        catalog.register(ProviderOperationSpec(operation_id=operation, provider="media",
            description="offline probe", risk=RiskLevel.WRITE if write else RiskLevel.READ,
            parameters=schema, result_kind="offline"))
    transport = Transport()
    return ProviderGateway(catalog=catalog, transports=[transport]), transport


def call(gw, arguments, operation="media.offline.probe", *, write=False, context=None):
    return (gw.preview_change if write else gw.query)(profile_ref="configured:media",
        operation=operation, arguments=arguments, context=context or ToolContext(owner="parity", session_id="parity"))


def object_schema(properties, **constraints):
    return {"type": "object", "properties": properties, "additionalProperties": False, **constraints}


def test_characterization_normalization_defaults_nullable_and_nested():
    schema = object_schema({"text": {"type": "string", "default": " default "},
        "number": {"type": "number"}, "optional": {"type": "integer", "nullable": True},
        "items": {"type": "array", "items": {"type": "string"}},
        "nested": object_schema({"flag": {"type": "boolean", "default": False}})})
    gw, transport = gateway(schema)
    call(gw, {"number": 2, "optional": None, "items": [" A "], "nested": {}})
    assert transport.calls == [("media.offline.probe", {"text": "default", "number": 2.0,
        "optional": None, "items": ["A"], "nested": {"flag": False}})]
    assert type(transport.calls[0][1]["number"]) is float


@pytest.mark.parametrize("schema,args", [
    (object_schema({"x": {"type": "integer"}}), {"x": True}),
    (object_schema({"x": {"type": "number"}}), {"x": float("inf")}),
    (object_schema({"x": {"type": "number"}}), {"x": float("nan")}),
    (object_schema({"x": {"type": "boolean"}}), {"x": 1}),
    (object_schema({}), {"extra": 1}),
    (object_schema({"token": {"type": "string"}}), {"token": "synthetic"}),
    (object_schema({"nested": object_schema({"url": {"type": "string"}})}), {"nested": {"url": "synthetic"}}),
    (object_schema({"x": {"type": "array", "items": {"type": "integer"}}}), {"x": list(range(101))}),
    (object_schema({"x": {"type": "string"}}), {"x": "a" * 4097}),
    (object_schema({"x": {"type": "integer"}}), {"x": 2**63}),
])
def test_characterization_security_and_bounds(schema, args):
    gw, transport = gateway(schema)
    with pytest.raises(ProviderGatewayError) as error:
        call(gw, args)
    assert error.value.code == "invalid_arguments"
    assert transport.calls == []


def test_characterization_operation_and_reference_scope():
    gw, transport = gateway()
    with pytest.raises(ProviderGatewayError) as error:
        call(gw, {}, "media.library.refresh")
    assert error.value.code == "operation_not_allowed"
    _, data = gw.artifacts.put(owner="parity", session_id="parity", provider="media",
        profile_ref="configured:media", operation="media.libraries.list",
        data={"items": [{"__object_id": "library-a", "__object_kind": "media_library"}]})
    ref = data["items"][0]["object_ref"]
    call(gw, {"library_ref": ref}, "media.library.counts")
    assert transport.calls[-1][1] == {"library_ref": "library-a"}
    transport.calls.clear()
    with pytest.raises(ProviderGatewayError) as error:
        call(gw, {"library_ref": ref}, "media.library.counts", context=ToolContext(owner="other", session_id="parity"))
    assert error.value.code == "artifact_expired"
    assert transport.calls == []


@pytest.mark.parametrize("schema,args", [
    (object_schema({"x": {"type": "array", "maxItems": 0, "items": {"type": "integer"}}}), {"x": [1]}),
    (object_schema({"x": {"type": "array", "uniqueItems": True, "items": {"type": "string"}}}), {"x": [" a", "a "]}),
    (object_schema({"x": {"type": "integer", "enum": [720, 1080]}}), {"x": 1000}),
    (object_schema({"x": {"type": "string", "pattern": "^A$"}}), {"x": "B"}),
    (object_schema({"a": {"type": "string"}, "b": {"type": "string"}},
                   oneOf=[{"required": ["a"]}, {"required": ["b"]}]), {}),
])
def test_declared_constraints_reject_before_transport(schema, args):
    gw, transport = gateway(schema)
    assert gw.catalog.get("media.offline.probe").parameters == schema
    with pytest.raises(ProviderGatewayError) as error:
        call(gw, args)
    assert error.value.code == "invalid_arguments"
    assert transport.calls == []


def test_recent_library_collision_is_published_and_rejected():
    gw, transport = gateway()
    public = gw.capabilities(provider="media", intent="media.items.recent_added", limit=100)
    published = next(row for row in public.data["operations"] if row["operation"] == "media.items.recent_added")
    assert published["operation"] == "media.items.recent_added"
    with pytest.raises(ProviderGatewayError) as error:
        call(gw, {"library_name": "Drama", "library_ref": "PO-invalid"}, "media.items.recent_added")
    assert error.value.code == "invalid_arguments"
    assert gw.catalog.get(published["operation"]).parameters.get("not") == {"required": ["library_name", "library_ref"]}
    assert transport.calls == []
    call(gw, {"library_name": " Drama "}, "media.items.recent_added")
    assert transport.calls[-1][1] == {"library_name": "Drama", "limit": 8}


def test_playlist_unique_refs_rejected_before_preview():
    gw, transport = gateway()
    _, data = gw.artifacts.put(owner="parity", session_id="parity", provider="media",
        profile_ref="configured:media", operation="media.items.search",
        data={"items": [{"__object_id": "movie-a", "__object_kind": "media_item"}]})
    ref = data["items"][0]["object_ref"]
    with pytest.raises(ProviderGatewayError) as error:
        call(gw, {"name": "Weekend", "item_refs": [ref, ref]}, "media.playlist.create", write=True)
    assert error.value.code == "invalid_arguments"
    assert transport.calls == []


@pytest.mark.parametrize("name,args", [
    ("guangya.fs.query", {"operation": "stat", "path": "/a", "max_items": 1}),
    ("guangya.fs.query", {"operation": "stat", "path": "/a", "kinds": ["video"]}),
    ("guangya.fs.query", {"operation": "stat", "path": "/"}),
    ("guangya.rename.preview", {"paths": ["/a"], "mode": "replace_text"}),
    ("guangya.rename.preview", {"paths": ["/a"], "mode": "remove_bitrate", "replace": ""}),
    ("library.search_missing_season_resources", {"items": [{"query": "A", "season": 1}], "query": "B"}),
])
def test_domain_declared_conditions_precede_execution(name, args):
    trace = []
    with pytest.raises(ToolPipelineError) as error:
        asyncio.run(execute(catalog_schema(name), args, trace))
    assert error.value.code == "invalid_arguments"
    assert trace == []


@pytest.mark.parametrize("name,args", [
    ("guangya.fs.query", {"operation": "stat", "path": "/a", "kinds": []}),
    ("guangya.fs.query", {"operation": "tree", "path": "/a", "query": "A"}),
    ("guangya.rename.preview", {"paths": ["/a"], "mode": "replace_text", "find": "A"}),
    ("guangya.rename.preview", {"paths": ["/a"], "mode": "remove_bitrate"}),
    ("library.search_missing_season_resources", {"items": [{"query": "A", "season": 1}], "max_episodes": 1}),
])
def test_domain_success_controls(name, args):
    trace = []
    assert asyncio.run(execute(catalog_schema(name), args, trace)).arguments == args
    assert trace == [("validate", args), ("read", args)]
