"""Provider discovery/admission contracts with an isolated transport boundary."""
from __future__ import annotations

import asyncio

import pytest

from app.agent.errors import ToolPipelineError
from app.agent.models import RiskLevel, ToolContext
from app.agent.provider_gateway import ProviderGateway
from app.agent.provider_models import (
    ProviderGatewayError,
    ProviderOperationSpec,
    ProviderPayload,
    ProviderProfileView,
)
from app.agent.provider_operations import build_provider_catalog
from tests.test_agent_schema_composition import catalog_schema, execute


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


@pytest.mark.parametrize("media_type", ["all", "movie", "tv", "episode"])
def test_recent_granularity_schema_matches_domain_validation_and_provider_admission(media_type):
    from app.agent.domain_catalog import build_tool_specs

    gw, transport = gateway()
    public = gw.capabilities(provider="media", intent="media.items.recent_added", limit=100)
    published = next(row for row in public.data["operations"] if row["operation"] == "media.items.recent_added")
    domain = next(spec for spec in build_tool_specs() if spec.name == "media.recently_added")
    domain_type = domain.parameters["properties"]["media_type"]
    assert published["parameters"]["properties"]["media_type"] == domain_type
    assert domain_type["enum"] == ["all", "movie", "tv", "episode"]
    arguments = {"library_name": "Drama", "limit": 3, "media_type": media_type}
    assert domain.validator(arguments) == {"server": "auto", **arguments}
    call(gw, arguments, "media.items.recent_added")
    assert transport.calls == [("media.items.recent_added", arguments)]


def test_recent_invalid_media_type_is_rejected_before_transport():
    gw, transport = gateway()
    with pytest.raises(ProviderGatewayError) as error:
        call(gw, {"media_type": "anime"}, "media.items.recent_added")
    assert error.value.code == "invalid_arguments"
    assert transport.calls == []


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


@pytest.mark.parametrize('invalid', [
    {'nested': {'flag': True, 'extra': 'not-a-schema'}},
    {'nested': {}},
    {'nested': {'flag': 'yes'}},
])
def test_failed_provider_normalization_retains_registered_schema_without_changing_it(invalid):
    from copy import deepcopy

    from app.agent.provider_policy import validate_provider_arguments
    from app.agent.schema import validate_json_schema

    # Missing additionalProperties is still closed by Provider normalization.
    schema = {'type': 'object', 'properties': {
        'nested': {'type': 'object', 'required': ['flag'],
                   'properties': {'flag': {'type': 'boolean'}}}}, 'required': ['nested']}
    original = deepcopy(schema)
    with pytest.raises(ProviderGatewayError) as failure:
        validate_provider_arguments(schema, invalid)
    assert schema == original
    effective = failure.value.argument_schemas["arguments"]
    assert effective is not None
    assert effective['properties']['nested']['additionalProperties'] is False
    with pytest.raises(ToolPipelineError):
        validate_json_schema(invalid, effective)
    corrected = {'nested': {'flag': True}}
    validate_json_schema(corrected, effective)
    assert validate_provider_arguments(schema, corrected) == corrected
    assert 'not-a-schema' not in str(failure.value)


@pytest.mark.parametrize("bad_operation,bad_profile", [(True, True), (True, False), (False, True)])
def test_unknown_query_selectors_directly_retry_at_real_admission(bad_operation, bad_profile):
    from unittest.mock import patch

    from app.agent.domain_catalog import build_tool_specs
    from app.agent.kernel.capabilities import ToolCatalog
    from app.agent.kernel.pipeline import ToolPipeline
    from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
    from app.agent.kernel.ports.mediaflux_policy import MediaFluxToolRateLimiter
    from app.agent.kernel.state import InMemorySessionStateStore
    from app.agent.kernel.state_types import CancellationToken
    from app.agent.kernel.tool_calls import ToolCallContext
    from app.agent.rate_limit import AgentRateLimiter

    gw, transport = gateway()
    specs = catalog_from_tool_specs(build_tool_specs())
    async def exercise():
        state = InMemorySessionStateStore()
        lease, _ = await state.begin_turn(owner="parity", session_id="selector", request_id="r")
        async def progress(payload): pass
        context = ToolCallContext(owner=lease.owner, session_id=lease.session_id, request_id=lease.request_id,
            turn_id=lease.turn_id, lease=lease, cancellation=CancellationToken(), report_progress=progress)
        pipeline = ToolPipeline(catalog=ToolCatalog([specs.get("provider.query"), specs.get("provider.capabilities")]),
            state_store=state, rate_limiter=MediaFluxToolRateLimiter())
        initial = {"profile_ref": "media" if bad_profile else "configured:media",
                   "operation": "capabilities" if bad_operation else "media.items.counts", "arguments": {}}
        with pytest.raises(ToolPipelineError) as caught:
            await pipeline.execute("provider.query", initial, context=context)
        failure = caught.value
        assert failure.code == "invalid_arguments" and not failure.repair_ticket
        assert context.schema_replacements == {}
        assert failure.argument_schemas and transport.calls == []
        discovered = await pipeline.execute("provider.capabilities", {"provider": "media"}, context=context)
        assert not discovered.recovered_failures and transport.calls == []
        corrected = {"profile_ref": "configured:media", "operation": "media.items.counts", "arguments": {}}
        separate = await pipeline.execute("provider.query", corrected, context=context)
        assert not separate.recovered_failures
        assert transport.calls == [("media.items.counts", {})]
        with pytest.raises(ToolPipelineError):
            await pipeline.execute("agent.repair", {"ticket": "not-a-read-ticket", "arguments": corrected}, context=context)
        assert len(transport.calls) == 1
    with patch("app.agent.provider_actions.get_provider_gateway", return_value=gw), \
         patch("app.agent.kernel.ports.mediaflux_policy.get_provider_gateway", return_value=gw), \
         patch("app.agent.rate_limit.agent_rate_limiter", AgentRateLimiter(clock=lambda: 100.0)):
        asyncio.run(exercise())


def test_query_selector_contract_does_not_turn_permissions_into_repair_authority():
    gw, transport = gateway()
    with pytest.raises(ProviderGatewayError) as denied:
        gw.query_spec(profile_ref="invented", operation="media.library.refresh")
    assert denied.value.code == "operation_not_allowed"
    assert denied.value.argument_schemas is None
    transport.profiles = lambda: [ProviderProfileView("configured:media", "media", "Offline", "disabled")]
    assert gw.query_spec(profile_ref="configured:media", operation="media.items.counts").operation_id == "media.items.counts"
    with pytest.raises(ProviderGatewayError) as disabled:
        call(gw, {}, "media.items.counts")
    assert disabled.value.code == "provider_not_configured"
    assert disabled.value.argument_schemas is None and transport.calls == []


def test_query_retry_changes_optional_filters_but_keeps_real_profile_validation():
    from unittest.mock import patch

    from app.agent.domain_catalog import build_tool_specs
    from app.agent.kernel.capabilities import ToolCatalog
    from app.agent.kernel.pipeline import ToolPipeline
    from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
    from app.agent.kernel.ports.mediaflux_policy import MediaFluxToolRateLimiter
    from app.agent.kernel.state import InMemorySessionStateStore
    from app.agent.kernel.state_types import CancellationToken
    from app.agent.kernel.tool_calls import ToolCallContext
    from app.agent.rate_limit import AgentRateLimiter

    gw, transport = gateway()
    async def exercise():
        state = InMemorySessionStateStore()
        lease, _ = await state.begin_turn(owner="parity", session_id="locked-selector", request_id="r")
        async def progress(payload): pass
        context = ToolCallContext(owner=lease.owner, session_id=lease.session_id, request_id=lease.request_id,
            turn_id=lease.turn_id, lease=lease, cancellation=CancellationToken(), report_progress=progress)
        tool = catalog_from_tool_specs(build_tool_specs()).get("provider.query")
        pipeline = ToolPipeline(catalog=ToolCatalog([tool]), state_store=state, rate_limiter=MediaFluxToolRateLimiter())
        initial = {"profile_ref": " configured:media ", "operation": "unknown", "arguments": {"limit": 5}}
        with pytest.raises(ToolPipelineError) as caught:
            await pipeline.execute("provider.query", initial, context=context)
        assert set(caught.value.argument_schemas) == {"operation"}, "已规范化且有效的profile不能解锁"
        assert not caught.value.repair_ticket and not context.schema_replacements
        corrected = {**initial, "operation": "media.items.recent_added"}
        with pytest.raises(ToolPipelineError) as invalid_profile:
            await pipeline.execute("provider.query", {**corrected, "profile_ref": "configured:other"}, context=context)
        assert invalid_profile.value.code == "invalid_arguments" and not invalid_profile.value.repair_ticket
        assert not transport.calls
        for arguments in ({}, {"limit": 6}):
            retried = await pipeline.execute("provider.query", {**corrected, "arguments": arguments}, context=context)
            assert not retried.recovered_failures and retried.effect_plan is None
        assert len(transport.calls) == 2
        assert transport.calls[-1] == ("media.items.recent_added", {"limit": 6})
    with patch("app.agent.provider_actions.get_provider_gateway", return_value=gw), \
         patch("app.agent.kernel.ports.mediaflux_policy.get_provider_gateway", return_value=gw), \
         patch("app.agent.rate_limit.agent_rate_limiter", AgentRateLimiter(clock=lambda: 100.0)):
        asyncio.run(exercise())
