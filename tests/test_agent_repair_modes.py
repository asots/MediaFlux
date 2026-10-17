"""READ 按 schema 直接重试；写侧修参仍不能扩大授权或跳过确认。"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json

import pytest

from app.agent.capability_discovery_actions import CAPABILITY_PARAMETERS
from app.agent.domain_catalog import build_tool_specs
from app.agent.kernel.capabilities import CapabilityRetriever, KernelToolSpec, ToolCatalog, ToolEffect
from app.agent.kernel.effects import PreparedEffect
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import AgentInput
from tests.error30_model_inputs import explicit_repair_wire
from tests.test_agent_recovered_progress import reply
from tests.test_agent_tool_argument_ingress import Wire, offline as offline, store as store


_PROTOCOL = 'anthropic_messages'
_OWNER = 'repair-modes-owner'
_SESSION = 'repair-modes-session'
_FINAL = '已按真实工具回执完成核验。'


def _session(model, catalog, store):
    return AgentSession(model=model, catalog=catalog, retriever=CapabilityRetriever(minimum=2, maximum=2),
                        pipeline=ToolPipeline(catalog=catalog, state_store=store), state_store=store)


async def _run(catalog, store, name, arguments, *, repair=False):
    replies = [reply(_PROTOCOL, name=catalog.get(name).model_name, arguments=value, index=index)
               for index, value in enumerate(arguments)] + [reply(_PROTOCOL, text=_FINAL)]
    wire = (explicit_repair_wire(_PROTOCOL, replies, index=1, arguments=arguments[1])
            if repair else Wire(_PROTOCOL, replies))
    events = [event async for event in _session(wire.adapter, catalog, store).run(
        AgentInput(message='核验媒体库与规则能力', owner=_OWNER, session_id=_SESSION))]
    cold = SQLiteKernelStore(secret_provider=lambda: 'offline-tool-argument-test')
    saved = await cold.load(owner=_OWNER, session_id=_SESSION)
    feedback = [json.loads(row['content']) for row in saved.conversation if row.get('role') == 'tool']
    return events, feedback, saved, wire


def test_actual_capabilities_bad_catalog_does_not_block_independent_query(store):
    async def run():
        specs = tuple(spec for spec in build_tool_specs() if spec.name == 'agent.capabilities')
        catalog = catalog_from_tool_specs(specs)
        assert catalog.get('agent.capabilities').input_schema == CAPABILITY_PARAMETERS
        catalog.register(KernelToolSpec(name='library.inventory', domain='library', description='读取媒体库清单',
            input_schema={'type': 'object', 'properties': {}}, effect=ToolEffect.READ,
            read=lambda *_: {'ok': True, 'status': 'success', 'summary': '媒体库清单'}))
        bad, independent = {'catalog': {'_错误': '未知字段'}}, {'query': '媒体库清单'}
        events, feedback, saved, wire = await _run(catalog, store, 'agent.capabilities', [bad, independent])
        wire.verify(3)
        assert len(feedback) == 2 and feedback[0]['code'] == 'invalid_arguments'
        assert feedback[1]['ok'] is True, 'query is an independent schema mode; it must not retain catalog'
        assert feedback[1]['status'] == 'completed'
        assert 'library.inventory' in json.dumps(feedback[1])
        assert any(event.type is AgentEventType.TOOL_COMPLETED and event.payload['call_id'] == 'call-1'
                   for event in events)
        calls = [call for row in saved.conversation for call in row.get('tool_calls', [])]
        assert [call['arguments'] for call in calls] == [bad, independent]
        assert any(event.type is AgentEventType.TOOL_FAILED for event in events)
    asyncio.run(run())


_IDENTITY_SCHEMA = {
    'type': 'object', 'required': ['id'],
    'properties': {'id': {'type': 'string', 'minLength': 1},
                   'filter': {'type': 'string', 'minLength': 1},
                   'limit': {'type': 'integer', 'minimum': 1}},
    'additionalProperties': False,
}
_WRITE_SCHEMA = {
    'type': 'object', 'required': ['id'],
    'properties': {'id': {'type': 'string', 'minLength': 1},
                   'period': {'type': 'integer', 'minimum': 1}},
    'additionalProperties': False,
}


def _read_catalog(schema, executed):
    def read(arguments, context):
        executed.append(deepcopy(arguments))
        return {'ok': True, 'status': 'success', 'summary': '真实只读 handler 已完成', 'data': {}}
    return ToolCatalog([KernelToolSpec(name='library.lookup', domain='library', description='读取媒体库范围',
        input_schema=deepcopy(schema), effect=ToolEffect.READ, read=read)])


def _write_catalog(previews, writes, *, schema=None, authorize=None):
    def prepare(arguments, context):
        previews.append(deepcopy(arguments))
        return PreparedEffect(preview={'summary': '规则预览，尚未执行'}, snapshot_fingerprint='offline-rule-snapshot')
    def write(arguments, snapshot_fingerprint, context):
        writes.append(deepcopy(arguments))
        return {'ok': True, 'status': 'success', 'summary': '规则写入完成'}
    return ToolCatalog([KernelToolSpec(name='rule.create', domain='rule', description='创建媒体库规则',
        input_schema=deepcopy(_WRITE_SCHEMA if schema is None else schema), effect=ToolEffect.WRITE,
        validator=lambda arguments: {'period': 10080, **arguments}, authorize=authorize,
        prepare=prepare, execute_confirmed=write)])


def test_ordinary_read_max_properties_modes_are_independent_without_name_exception(store):
    async def run():
        executed = []
        schema = {'type': 'object', 'maxProperties': 1, 'additionalProperties': False, 'properties': {
            'listing': {'type': 'object', 'properties': {}, 'additionalProperties': False},
            'search': {'type': 'string', 'minLength': 1},
        }}
        catalog = _read_catalog(schema, executed)
        bad, independent = {'listing': {'_错误': '未知字段'}}, {'search': '媒体库'}
        events, feedback, _, wire = await _run(catalog, store, 'library.lookup', [bad, independent])
        wire.verify(3)
        assert executed == [independent], 'schema modes must work for ordinary READ, not a capabilities-name exemption'
        assert feedback[0]['code'] == 'invalid_arguments' and feedback[1]['status'] == 'success'
        assert events[-1].payload['status'] == 'success'
        assert 'repair' not in feedback[0]
    asyncio.run(run())


def test_read_can_drop_optional_filter_and_directly_retry(store):
    async def run():
        executed = []
        catalog = _read_catalog(_IDENTITY_SCHEMA, executed)
        bad = {'id': 'A', 'filter': 'unwatched-only', 'limit': 'invalid'}
        dropped = {'id': 'A', 'limit': 3}
        events, feedback, saved, wire = await _run(catalog, store, 'library.lookup', [bad, dropped])
        wire.verify(3)
        assert executed == [dropped]
        assert feedback[0]['code'] == 'invalid_arguments'
        assert 'repair' not in feedback[0]
        assert feedback[1]['ok'] is True
        assert events[-1].payload['status'] == 'success'
        calls = [call for row in saved.conversation for call in row.get('tool_calls', [])]
        assert calls[0]['arguments'] == bad
    asyncio.run(run())


def test_other_required_identity_is_an_independent_read(store):
    async def run():
        executed = []
        catalog = _read_catalog(_IDENTITY_SCHEMA, executed)
        bad = {'id': 'A', 'filter': 'unwatched-only', 'limit': 'invalid'}
        independent = {'id': 'B'}
        events, feedback, _, wire = await _run(catalog, store, 'library.lookup', [bad, independent])
        wire.verify(3)
        assert executed == [independent]
        assert feedback[0]['code'] == 'invalid_arguments' and feedback[1]['status'] == 'success'
        assert events[-1].payload['status'] == 'success'
        assert 'repair' not in feedback[0]
    asyncio.run(run())


@pytest.mark.parametrize('repair', [False, True], ids=['direct', 'explicit'])
@pytest.mark.parametrize('period', ['invalid', False, 0, []], ids=['string', 'false', 'zero', 'empty-list'])
def test_invalid_write_period_cannot_be_omitted_to_default(period, repair, store):
    async def run():
        previews, writes = [], []
        catalog = _write_catalog(previews, writes)
        bad, omitted = {'id': 'A', 'period': period}, {'id': 'A'}
        attempts = [bad, omitted]
        if repair and period == []:
            # Empty-list errors do not offer a ticket. Use a real earlier ticket,
            # reject [] through agent.repair, then prove omission still cannot default.
            attempts = [{'id': 'A', 'period': 'invalid'}, bad, omitted]
        events, feedback, saved, wire = await _run(catalog, store, 'rule.create', attempts, repair=repair)
        wire.verify(len(attempts) + 1)
        assert previews == [] and writes == [], 'falsy supplied values are not absence and cannot become a defaulted plan'
        assert len(feedback) == len(attempts) and feedback[0]['code'] == 'invalid_arguments'
        assert all(result['ok'] is False and result['code'] in {'invalid_arguments', 'repair_conflict'}
                   for result in feedback[1:])
        assert not saved.pending_effect_plan_id
        if not repair and period == []:
            assert 'repair' not in feedback[0], 'an unsupported structure must not invent an offered ticket'
            assert catalog.get('agent.repair').model_name not in {
                tool['name'] for tool in wire.requests[1]['tools']
            }, 'a guard-only record cannot open agent.repair in the model window'
        assert not any(event.type is AgentEventType.EFFECT_APPROVAL_REQUIRED for event in events)
        assert events[-1].payload.get('status') not in {'success', 'approval_required'}
    asyncio.run(run())


@pytest.mark.parametrize('replacement', [{}, {'period': 60}], ids=['drop-both', 'keep-one'])
def test_explicit_mutual_exclusion_repair_must_keep_a_selected_field(replacement, store):
    async def run():
        previews, writes = [], []
        schema = deepcopy(_WRITE_SCHEMA)
        schema['properties']['schedule'] = {'type': 'string', 'minLength': 1}
        schema['not'] = {'required': ['period', 'schedule']}
        catalog = _write_catalog(previews, writes, schema=schema)
        bad = {'id': 'A', 'period': 60, 'schedule': 'daily'}
        corrected = {'id': 'A', **replacement}
        events, feedback, saved, wire = await _run(catalog, store, 'rule.create', [bad, corrected], repair=True)
        assert feedback[0]['code'] == 'invalid_arguments' and writes == []
        if replacement:
            wire.verify(2)
            assert previews == [corrected]
            assert saved.pending_effect_plan_id == events[-1].payload['plan_id']
            assert events[-1].payload['status'] == 'approval_required'
            assert not any(event.type is AgentEventType.EFFECT_COMPLETED for event in events)
            assert any(event.type is AgentEventType.EFFECT_APPROVAL_REQUIRED for event in events)
        else:
            wire.verify(3)
            assert previews == [] and not saved.pending_effect_plan_id
            assert feedback[1]['ok'] is False
            assert events[-1].payload.get('status') != 'success'
    asyncio.run(run())


def test_explicit_repair_does_not_bypass_revoked_write_permission(store):
    async def run():
        previews, writes = [], []
        allowed = [True]
        catalog = _write_catalog(previews, writes, authorize=lambda context: allowed[0])
        bad, corrected = {'id': 'A', 'period': 'invalid'}, {'id': 'A', 'period': 60}
        replies = [reply(_PROTOCOL, name=catalog.get('rule.create').model_name, arguments=value, index=index)
                   for index, value in enumerate([bad, corrected])] + [reply(_PROTOCOL, text='没有权限，未执行写入。')]
        wire = explicit_repair_wire(_PROTOCOL, replies, index=1, arguments=corrected)
        original_handler = wire.handler
        async def revoke_before_repair(request):
            if len(wire.requests) == 1:
                allowed[0] = False  # Real authorization is rechecked after the model selects its repair.
            return await original_handler(request)
        wire.handler = revoke_before_repair
        events = [event async for event in _session(wire.adapter, catalog, store).run(
            AgentInput(message='创建媒体规则', owner=_OWNER, session_id=_SESSION))]
        wire.verify(3)
        saved = await SQLiteKernelStore(secret_provider=lambda: 'offline-tool-argument-test').load(
            owner=_OWNER, session_id=_SESSION)
        feedback = [json.loads(row['content']) for row in saved.conversation if row.get('role') == 'tool']
        assert feedback[0]['code'] == 'invalid_arguments'
        assert feedback[1]['code'] == 'authorization_denied'
        assert previews == [] and writes == [] and not saved.pending_effect_plan_id
        assert events[-1].payload.get('status') not in {'success', 'approval_required'}
    asyncio.run(run())



@pytest.mark.parametrize('shape', ['multiple-modes', 'single-filter-property'])
def test_empty_read_retry_uses_current_schema_not_old_fields(shape, store):
    async def run():
        executed = []
        if shape == 'multiple-modes':
            properties = {'listing': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                          'search': {'type': 'string', 'minLength': 1}}
            bad = {'listing': {'_错误': '未知字段'}}
        else:
            properties = {'filter': {'type': 'string', 'minLength': 1}}
            bad = {'filter': 'unwatched-only', '_错误': '未知字段'}
        schema = {'type': 'object', 'properties': properties, 'additionalProperties': False, 'maxProperties': 1}
        catalog = _read_catalog(schema, executed)
        events, feedback, saved, wire = await _run(catalog, store, 'library.lookup', [bad, {}])
        wire.verify(3)
        assert executed == [{}], 'the current schema permits an empty READ request'
        assert feedback[0]['code'] == 'invalid_arguments'
        assert 'repair' not in feedback[0]
        assert feedback[1]['ok'] is True
        assert events[-1].payload['status'] == 'success'
        calls = [call for row in saved.conversation for call in row.get('tool_calls', [])]
        assert [call['arguments'] for call in calls] == [bad, {}]
    asyncio.run(run())
