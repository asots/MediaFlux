"""Provider 内层 schema 错误必须由真实业务成功精确解除，而不是按工具名清错。"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json

import pytest

from app.agent import provider_actions
from app.agent.domain_catalog import build_tool_specs
from app.agent.kernel.capabilities import CapabilityRetriever
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import AgentInput
from app.agent.provider_gateway import ProviderGateway
from app.agent.provider_models import ProviderGatewayError, ProviderPayload, ProviderProfileView
from app.agent.provider_operations import build_provider_catalog
from tests.error30_model_inputs import explicit_repair_wire
from tests.test_agent_recovered_progress import reply
from tests.test_agent_tool_argument_ingress import Wire, offline as offline, store as store


_PROTOCOL = 'anthropic_messages'
_OWNER = 'provider-repair-owner'
_SESSION = 'provider-repair-session'
_PROFILE = 'configured:jellyfin'
_OPERATION = 'media.libraries.list'
_FINAL = '已根据真实 Provider 读取核验媒体库。'


class _OfflineMediaTransport:
    """仅替代网络业务边界，不接管 operation schema、Gateway 或 Pipeline 结果。"""
    provider = 'media'

    def __init__(self):
        self.reads = []
        self.outcomes = []
        self.previews = []
        self.writes = []

    def profiles(self):
        return [ProviderProfileView(ref, 'media', label, 'online') for ref, label in (
            (_PROFILE, 'Jellyfin'), ('configured:emby', 'Emby'))]

    def profile_revision(self, profile_ref):
        return 'offline-media-profile-v1'

    def execute_read(self, profile_ref, operation, arguments):
        self.reads.append((profile_ref, operation, deepcopy(arguments)))
        outcome = self.outcomes.pop(0) if self.outcomes else 'success'
        if isinstance(outcome, Exception):
            raise outcome
        data = ({'items': [{'name': '示例影片', '__object_id': 'offline-item-1',
                            '__object_kind': 'media_item'}], 'count': 1}
                if operation == 'media.items.search' else
                {'libraries': [{'name': '电影', '__object_id': 'offline-library-1',
                                '__object_kind': 'media_library'}], 'count': 1})
        return ProviderPayload(
            summary='Provider 媒体库实时核验回执', data=data,
            source='offline-media-api', status=outcome,
        )


    def preview_write(self, profile_ref, operation, arguments, target_snapshot):
        self.previews.append((profile_ref, operation, deepcopy(arguments)))
        return ProviderPayload(summary='已预检，尚未执行写操作', data={'name': arguments['name']},
                               source='offline-media-api')

    def execute_write(self, *args, **kwargs):
        self.writes.append((args, kwargs))
        raise AssertionError('a preview or argument repair is not write authorization')


@pytest.fixture
def provider_chain(monkeypatch, store):
    transport = _OfflineMediaTransport()
    gateway = ProviderGateway(catalog=build_provider_catalog(), transports=[transport])
    monkeypatch.setattr(provider_actions, 'get_provider_gateway', lambda: gateway)
    specs = tuple(spec for spec in build_tool_specs() if spec.name in {
        'provider.query', 'provider.change.preview'})
    catalog = catalog_from_tool_specs(specs)
    return gateway, transport, catalog


def _query(arguments, *, profile=_PROFILE, operation=_OPERATION):
    return {'profile_ref': profile, 'operation': operation, 'arguments': deepcopy(arguments)}


def _session(model, catalog, store):
    return AgentSession(
        model=model, catalog=catalog, retriever=CapabilityRetriever(minimum=2, maximum=2),
        pipeline=ToolPipeline(catalog=catalog, state_store=store), state_store=store,
    )


async def _run_queries(catalog, store, calls, *, protocol=_PROTOCOL, repair=False):
    replies = [reply(protocol, name=catalog.get('provider.query').model_name, arguments=arguments, index=index)
               for index, arguments in enumerate(calls)] + [reply(protocol, text=_FINAL)]
    transport = (explicit_repair_wire(protocol, replies, index=1, arguments=calls[1])
                 if repair else Wire(protocol, replies))
    events = [event async for event in _session(transport.adapter, catalog, store).run(
        AgentInput(message='核验 Jellyfin 媒体库', owner=_OWNER, session_id=_SESSION))]
    transport.verify(len(calls) + 1)
    cold = SQLiteKernelStore(secret_provider=lambda: 'offline-tool-argument-test')
    saved = await cold.load(owner=_OWNER, session_id=_SESSION)
    feedback = [json.loads(row['content']) for row in saved.conversation if row.get('role') == 'tool']
    return events, feedback, saved, transport


def test_provider_direct_retry_deleting_only_unknown_argument_recovers(provider_chain, store):
    async def run():
        gateway, media, catalog = provider_chain
        # The outer domain ToolSpec intentionally allows arguments; the real operation does not.
        assert catalog.get('provider.query').input_schema['properties']['arguments']['additionalProperties'] is True
        operation = gateway.catalog.get(_OPERATION)
        assert operation.parameters == {'type': 'object', 'properties': {}, 'additionalProperties': False}
        bad, corrected = _query({'reason': '获取当前媒体库列表'}), _query({})
        events, feedback, saved, wire = await _run_queries(catalog, store, [bad, corrected])
        assert media.reads == [(_PROFILE, _OPERATION, {})], 'the invalid call must never reach Transport'
        assert len(feedback) == 2 and feedback[0]['code'] == 'invalid_arguments'
        assert feedback[1]['ok'] is True and feedback[1]['status'] == 'success'
        assert any(event.type is AgentEventType.TOOL_FAILED for event in events), 'original failure remains audit history'
        assert any(event.type is AgentEventType.TOOL_COMPLETED for event in events)
        stored = [call for row in saved.conversation for call in row.get('tool_calls', [])]
        assert [call['arguments'] for call in stored] == [bad, corrected], 'never rewrite the failed call into a success'
        assert events[-1].type is AgentEventType.TURN_COMPLETED
        assert events[-1].payload['status'] == 'success', 'a schema-proven correction and actual success must resolve the old obligation'
        assert events[-1].payload['answer'] == _FINAL
    asyncio.run(run())



def test_provider_deleting_extra_keeps_original_valid_nested_values(provider_chain, store):
    async def run():
        _, media, catalog = provider_chain
        operation = 'media.items.search'
        bad = _query({'query': '原目标', 'limit': 3, 'reason': '额外解释'}, operation=operation)
        corrected = _query({'query': '原目标', 'limit': 3}, operation=operation)
        events, feedback, _, _ = await _run_queries(catalog, store, [bad, corrected])
        assert media.reads == [(_PROFILE, operation, {'query': '原目标', 'limit': 3})]
        assert feedback[0]['code'] == 'invalid_arguments' and feedback[1]['status'] == 'success'
        assert events[-1].payload['status'] == 'success'
    asyncio.run(run())


@pytest.mark.parametrize('case', ['different-profile', 'different-operation', 'changed-query',
                                  'extra-and-missing', 'extra-and-bad-value'])
def test_provider_non_equivalent_direct_retry_keeps_old_failure(case, provider_chain, store):
    async def run():
        _, media, catalog = provider_chain
        bad, corrected = _query({'reason': '获取媒体库'}), _query({})
        expected = (_PROFILE, _OPERATION, {})
        if case == 'different-profile':
            corrected = _query({}, profile='configured:emby')
            expected = ('configured:emby', _OPERATION, {})
        elif case == 'different-operation':
            corrected = _query({}, operation='media.items.continue_watching')
            expected = (_PROFILE, 'media.items.continue_watching', {'limit': 8})
        else:
            operation = 'media.items.search'
            bad = _query({'query': '原目标', 'limit': 3, 'reason': '额外解释'}, operation=operation)
            corrected = _query({'query': '原目标', 'limit': 3}, operation=operation)
            expected_arguments = {'query': '原目标', 'limit': 3}
            if case == 'changed-query':
                corrected['arguments']['query'] = expected_arguments['query'] = '另一个目标'
            elif case == 'extra-and-missing':
                bad['arguments'].pop('query')
            else:
                bad['arguments']['query'] = False
            expected = (_PROFILE, operation, expected_arguments)
        events, feedback, _, _ = await _run_queries(catalog, store, [bad, corrected])
        assert media.reads == [expected]
        assert feedback[0]['code'] == 'invalid_arguments'
        assert feedback[1]['ok'] is True and feedback[1]['status'] == 'success'
        assert events[-1].payload['status'] == 'partial', 'another query or a non-unique correction cannot clear the old obligation'
    asyncio.run(run())


@pytest.mark.parametrize('case', ['extra-only', 'extra-and-missing'])
def test_provider_explicit_agent_repair_uses_real_domain_success(case, provider_chain, store):
    async def run():
        _, media, catalog = provider_chain
        bad, corrected = _query({'reason': '获取媒体库'}), _query({})
        expected = (_PROFILE, _OPERATION, {})
        if case != 'extra-only':
            bad = _query({'reason': '查询目标', 'limit': 3}, operation='media.items.search')
            corrected = _query({'query': '明确补充目标', 'limit': 3}, operation='media.items.search')
            expected = (_PROFILE, 'media.items.search', {'query': '明确补充目标', 'limit': 3})
        events, feedback, saved, _ = await _run_queries(catalog, store, [bad, corrected], repair=True)
        assert media.reads == [expected]
        assert feedback[0]['code'] == 'invalid_arguments' and feedback[0]['repair']['tool'] == 'agent.repair'
        assert feedback[1]['ok'] is True and feedback[1]['status'] == 'success'
        assert events[-1].payload['status'] == 'success'
        assert sum(event.type is AgentEventType.TOOL_FAILED for event in events) == 1
        calls = [call for row in saved.conversation for call in row.get('tool_calls', [])]
        assert calls[0]['arguments'] == bad
        assert calls[1]['name'] == catalog.get('agent.repair').model_name
        assert calls[1]['arguments']['arguments'] == corrected
    asyncio.run(run())


def test_provider_explicit_repair_cannot_change_original_valid_nested_field(provider_chain, store):
    async def run():
        _, media, catalog = provider_chain
        bad = _query({'query': '原目标', 'limit': 3, 'reason': '额外解释'}, operation='media.items.search')
        changed = _query({'query': '另一个目标', 'limit': 3}, operation='media.items.search')
        events, feedback, _, _ = await _run_queries(catalog, store, [bad, changed], repair=True)
        assert media.reads == []
        assert feedback[0]['code'] == 'invalid_arguments'
        assert feedback[1]['code'] == 'repair_conflict'
        assert events[-1].payload.get('status') != 'success'
    asyncio.run(run())


@pytest.mark.parametrize('outcome', ['transport-error', 'partial'])
def test_provider_without_actual_success_does_not_clear_old_failure(outcome, provider_chain, store):
    async def run():
        _, media, catalog = provider_chain
        calls = [_query({'reason': '获取媒体库'}), _query({})]
        media.outcomes = [ProviderGatewayError('媒体读取失败', code='provider_unavailable')
                          if outcome == 'transport-error' else outcome]
        events, feedback, _, _ = await _run_queries(catalog, store, calls)
        assert media.reads == [(_PROFILE, _OPERATION, {})]
        assert feedback[0]['code'] == 'invalid_arguments'
        if outcome == 'transport-error':
            assert feedback[1]['code'] == 'provider_unavailable'
        else:
            assert feedback[1]['status'] == outcome
        assert events[-1].payload.get('status') != 'success'
        assert any(event.type is AgentEventType.TOOL_FAILED for event in events)
    asyncio.run(run())


@pytest.mark.parametrize('case', ['unknown-operation', 'domain-error-without-schema'])
def test_provider_unknown_failure_does_not_invent_repair_or_clear_by_tool_name(case, provider_chain, store):
    async def run():
        _, media, catalog = provider_chain
        if case == 'unknown-operation':
            calls = [_query({}, operation='media.unregistered.operation'), _query({})]
            expected = [(_PROFILE, _OPERATION, {})]
        else:
            calls = [_query({'query': query}, operation='media.items.search') for query in ('原目标', '另一个目标')]
            media.outcomes = [ProviderGatewayError('领域约束不满足', code='invalid_arguments'), 'success']
            expected = [(_PROFILE, 'media.items.search', {'query': query, 'limit': 12}) for query in ('原目标', '另一个目标')]
        events, feedback, _, _ = await _run_queries(catalog, store, calls)
        assert media.reads == expected
        assert 'repair' not in feedback[0], 'no registered schema evidence means no speculative repair ticket'
        assert feedback[1]['ok'] is True and feedback[1]['status'] == 'success'
        assert events[-1].payload['status'] == 'partial'
    asyncio.run(run())


@pytest.mark.parametrize('repair', [False, True], ids=['direct', 'explicit'])
def test_provider_corrected_write_preview_is_not_execution_success(repair, provider_chain, store):
    async def run():
        _, media, catalog = provider_chain
        _, seed, _, _ = await _run_queries(catalog, store, [_query({'query': '示例影片'}, operation='media.items.search')])
        item_ref = seed[0]['data']['items'][0]['object_ref']
        corrected = _query({'name': '周末片单', 'item_refs': [item_ref]}, operation='media.playlist.create')
        bad = deepcopy(corrected)
        bad['arguments']['reason'] = '额外解释'
        name = catalog.get('provider.change.preview').model_name
        replies = [reply(_PROTOCOL, name=name, arguments=arguments, index=index + 10)
                   for index, arguments in enumerate([bad, corrected])] + [reply(_PROTOCOL, text='只有预览，尚未写入。')]
        wire = (explicit_repair_wire(_PROTOCOL, replies, index=1, arguments=corrected)
                if repair else Wire(_PROTOCOL, replies))
        events = [event async for event in _session(wire.adapter, catalog, store).run(
            AgentInput(message='准备新媒体播放列表预览', owner=_OWNER, session_id=_SESSION))]
        wire.verify(3)
        assert len(media.reads) == 1 and len(media.previews) == 1 and media.writes == []
        state = await SQLiteKernelStore(secret_provider=lambda: 'offline-tool-argument-test').load(
            owner=_OWNER, session_id=_SESSION)
        feedback = [json.loads(row['content']) for row in state.conversation if row.get('role') == 'tool'][-2:]
        assert feedback[0]['code'] == 'invalid_arguments'
        assert feedback[1]['ok'] is True and feedback[1]['status'] == 'preview'
        assert feedback[1]['data']['plan_ref'].startswith('PP-')
        assert events[-1].payload['status'] == 'partial', 'a frozen preview plan is not actual execution success'
    asyncio.run(run())


def test_provider_error_feedback_uses_the_registered_operation_schema(provider_chain, store):
    async def run():
        gateway, media, catalog = provider_chain
        bad = _query({'reason': '额外解释不是 operation 参数'})
        events, feedback, _, _ = await _run_queries(catalog, store, [bad, _query({})])
        expected = feedback[0]['expected_input_schema']
        assert expected['properties']['arguments'] == gateway.catalog.get(_OPERATION).parameters
        assert expected['properties']['arguments']['additionalProperties'] is False
        assert feedback[0]['repair']['ticket']
        assert media.reads == [(_PROFILE, _OPERATION, {})]
        assert events[-1].payload['status'] == 'success'
    asyncio.run(run())
