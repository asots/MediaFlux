"""Public provider schemas through real capabilities, projection and native history."""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import pytest

from app import database as db
from app.agent.domain_catalog import build_tool_specs
from app.agent.errors import ToolPipelineError
from app.agent.kernel.bootstrap import build_agent_kernel_runtime
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
from app.agent.kernel.state_types import AgentInput
from app.agent.owner_routes import web_kernel_owner
from app.agent.provider_gateway import ProviderGateway
from app.agent.provider_operations import (
    RECENT_MEDIA_PROPERTIES,
    build_provider_catalog,
)
from tests.test_agent_schema_composition import catalog_schema, execute
from tests.test_agent_tool_argument_ingress import Wire, encode


def test_every_public_schema_is_complete_and_detached():
    for spec in build_provider_catalog().operations():
        original = deepcopy(spec.parameters)
        public = spec.public_dict()
        assert public.get('parameters') == original, spec.operation_id
        assert set(public) == {'operation', 'provider', 'description', 'risk', 'input_schema', 'limits', 'parameters'}
        assert public['operation'] == spec.operation_id
        assert public['limits'] == {'max_items': spec.max_items, 'timeout_seconds': spec.timeout_seconds}
        def mutable_ids(value):
            if isinstance(value, (dict, list)):
                children = value.values() if isinstance(value, dict) else value
                return {id(value)} | set().union(*(mutable_ids(child) for child in children))
            return set()
        assert mutable_ids(public['parameters']).isdisjoint(mutable_ids(spec.parameters))
        public['parameters']['properties'].clear()
        public['parameters'].clear()
        assert spec.parameters == original
        assert spec.public_dict()['parameters'] == original


def test_each_current_selected_schema_survives_default_projection():
    from app.agent.kernel.projection import DefaultProjector

    gateway = ProviderGateway(catalog=build_provider_catalog(), transports=[])
    for spec in gateway.catalog.operations():
        result = gateway.capabilities(provider=spec.provider, intent=spec.operation_id, limit=8)
        assert any(row['operation'] == spec.operation_id for row in result.data['operations'])
        projected = json.loads(DefaultProjector().project(result).model_content)
        row = next(row for row in projected['data']['operations'] if row['operation'] == spec.operation_id)
        assert row.get('parameters') == spec.parameters, spec.operation_id


def test_old_recent_summary_fields_are_unchanged():
    public = build_provider_catalog().get('media.items.recent_added').public_dict()
    public.pop('parameters', None)
    description = public.pop('description')
    assert 'media_type=episode' in description and '默认按作品去重' in description
    assert public == {
        'operation': 'media.items.recent_added', 'provider': 'media',
        'risk': 'read', 'input_schema': 'limit:integer(default=8), library_name:string, library_ref:string, media_type:string',
        'limits': {'max_items': 20, 'timeout_seconds': 15},
    }


def test_recent_subscription_exclusion_before_domain_validation():
    schema = catalog_schema('media.recently_added')
    trace = []
    with pytest.raises(ToolPipelineError) as error:
        asyncio.run(execute(schema, {'library_name': 'Drama', 'library_ref': 'PO-invalid'}, trace))
    assert error.value.code == 'invalid_arguments'
    assert trace == []
    assert schema['not'] == {'required': ['library_name', 'library_ref']}
    assert schema['properties'] == {'server': {'type': 'string', 'enum': ['auto', 'jellyfin', 'emby']}, **RECENT_MEDIA_PROPERTIES}


@pytest.mark.parametrize('arguments', [{}, {'library_name': 'Drama'}, {'library_ref': 'PO-reference'}, {'server': 'emby', 'limit': 8}])
def test_recent_subscription_success_shapes(arguments):
    assert asyncio.run(execute(catalog_schema('media.recently_added'), arguments, [])).arguments == arguments


def response(name=None, arguments=None, call_id='capabilities'):
    message = {'content': '已读取能力。'} if name is None else {'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': encode(arguments)}}]}
    return encode({'choices': [{'message': message, 'finish_reason': 'stop' if name is None else 'tool_calls'}]}).encode(), 'application/json'


@pytest.mark.parametrize('operation', ['media.items.recent_added', 'media.playlist.create', 'media.library.quality'])
def test_capabilities_schema_reaches_next_native_request_and_cold_history(tmp_path, monkeypatch, operation):
    previous = db.DB_PATH, bool(getattr(db, '_configured_test_mode', False))
    db.configure_database(tmp_path / 'projection.db', test_mode=True)
    db.init_db()
    gateway = ProviderGateway(catalog=build_provider_catalog(), transports=[])
    monkeypatch.setattr('app.agent.provider_actions.get_provider_gateway', lambda: gateway)
    tool = catalog_from_tool_specs(build_tool_specs()).get('provider.capabilities')
    arguments = {'provider': 'media', 'intent': operation, 'limit': 1}
    selected = gateway.catalog.list(**arguments)
    assert [spec.operation_id for spec in selected] == [operation]
    expected = deepcopy(selected[0].parameters)

    async def run():
        store = SQLiteKernelStore(secret_provider=lambda: 'projection-test-only')
        wire = Wire('chat_completions', [
            response('agent__capabilities', {'tool_names': ['provider.capabilities']}, 'discover'),
            response(tool.model_name, arguments), response(), response(),
        ])
        owner = web_kernel_owner('admin')
        def session(state_store):
            return build_agent_kernel_runtime(model=wire.adapter, store=state_store).session
        events = [e async for e in session(store).run(AgentInput(message='读取 Provider 能力', owner=owner, session_id='projection-session'))]
        assert events[-1].payload.get('status') == 'success', [e.payload for e in events]
        # Complete production catalog, discovery/window, authorization and budgets.
        next_body = wire.requests[2]
        (tmp_path / 'next-native.json').write_text(json.dumps(next_body, ensure_ascii=False, indent=2))
        tools = [json.loads(m['content']) for m in next_body['messages'] if m['role'] == 'tool']
        rows = tools[-1]['data']['operations']
        assert rows[0]['operation'] == operation
        assert rows[0].get('parameters') == expected
        assert '[内容过深]' not in encode(rows[0]['parameters'])
        assert 'reference_arguments' not in rows[0]
        cold = SQLiteKernelStore(secret_provider=lambda: 'projection-test-only')
        await cold.load(owner=owner, session_id='projection-session')
        _ = [e async for e in session(cold).run(AgentInput(message='按刚才的能力继续说明', owner=owner, session_id='projection-session'))]
        replay = [json.loads(m['content']) for m in wire.requests[3]['messages'] if m['role'] == 'tool']
        assert replay[-1]['data']['operations'][0]['parameters'] == expected
        wire.verify(4)

    try:
        asyncio.run(run())
    finally:
        db.configure_database(previous[0], test_mode=previous[1])
