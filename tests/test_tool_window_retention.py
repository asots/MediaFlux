"""Explicit turn-local retention through the real capability handler and native wire."""

from typing import cast
from collections.abc import Callable
from app.clients.http import FixedHostHttpClient
import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from app.agent.domains.system.capability_actions import (
    CAPABILITY_PARAMETERS,
    capability_arguments,
)
from app.agent.errors import AgentToolError, ToolPipelineError
from app.agent.kernel.capabilities import (
    CapabilityRetriever,
    CapabilitySelection,
    ToolCatalog,
)
from app.agent.kernel.discovery import CapabilityDiscovery
from app.agent.kernel.projection import DefaultProjector
from app.agent.kernel.provider_model import (
    OpenAICompatibleModelAdapter,
    ProviderSettings,
)
from tests.test_agent_kernel_capability_discovery import (
    call,
    capability_tool,
    final,
    request,
    session_for,
)
from tests.test_agent_kernel_core import collect, read_tool


class Initial(CapabilityRetriever):
    def retrieve(self, message, catalog, *, context=None):
        return CapabilitySelection((catalog.get('anchor.inspect'),), {'anchor.inspect': 1})


def test_retained_dependency_reaches_fourth_native_wire_and_second_read():
    asyncio.run(native_retention())


async def native_retention():
    seen, executed = [], []
    anchor = read_tool('anchor.inspect', handler=lambda a, c: executed.append('anchor') or {'summary': '已读取对象快照'})
    others = [read_tool(f'phase.item{i}') for i in range(10)]
    session, _ = session_for([anchor, *others], [])
    session.retriever = Initial()
    session.limits = replace(session.limits, capability_window=10)

    class Response:
        status_code = 200
        headers = {'content-type': 'text/event-stream'}
        def __init__(self, body): self.body = body
        async def aiter_bytes(self):
            yield ('data: ' + json.dumps(self.body) + '\n\ndata: [DONE]\n\n').encode()

    class Client:
        @asynccontextmanager
        async def stream_post_json(self, url, **kwargs):
            seen.append(kwargs['json'])
            n = len(seen)
            calls = {
                1: [('anchor__inspect', {}), ('agent__capabilities', {'window': {'retain': ['anchor.inspect']}})],
                2: [('agent__capabilities', {'tool_names': [t.name for t in others[:6]]})],
                3: [('agent__capabilities', {'tool_names': [t.name for t in others[6:]]})],
                4: [('anchor__inspect', {})],
            }.get(n, [])
            delta = {'tool_calls': [{'index': i, 'id': f'c{n}-{i}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}} for i, (name, args) in enumerate(calls)]} if calls else {'content': '已完成复核。'}
            yield Response({'choices': [{'delta': delta, 'finish_reason': 'tool_calls' if calls else 'stop'}]})
        async def aclose(self): pass

    session.model = OpenAICompatibleModelAdapter(ProviderSettings(api_url='https://offline.invalid/v1', model='fixture', protocol='chat_completions'), client_factory=cast(Callable[..., FixedHostHttpClient], lambda **_: Client()))
    await collect(session.run(replace(request(), message='先读取初始对象快照，保留初始读取能力，再分阶段检查十种能力；最后再次读取初始对象核对变化。')))
    names = [[t['function']['name'] for t in r.get('tools', [])] for r in seen]
    results = [json.loads(m['content']) for m in seen[-1]['messages'] if m['role'] == 'tool']
    assert session.retriever.maximum == 10
    assert 'anchor__inspect' in names[3], 'explicitly retained anchor must survive both discovery phases'
    assert executed == ['anchor', 'anchor']
    assert any(r.get('data', {}).get('retained_tools') == ['anchor.inspect'] for r in results)
    schema = next(t['function']['parameters'] for t in seen[0]['tools'] if t['function']['name'] == 'agent__capabilities')
    assert schema == CAPABILITY_PARAMETERS
    assert schema['maxProperties'] == 1
    assert schema['properties']['window']['properties']['retain']['maxItems'] == 32


def finder_for(*, authorized=None):
    tools = [read_tool(f'item.read{i}') for i in range(4)]
    if authorized is not None:
        tools[0] = replace(tools[0], authorize=lambda c: authorized[0])
    finder = CapabilityDiscovery(ToolCatalog([capability_tool(), *tools]), context={}, maximum=2)
    finder.window(tools[:2])
    return finder, tools


def change(finder, **window):
    return finder.search(capability_arguments({'window': window}))


def test_release_reclaims_capacity_and_cache_reapplies_retention():
    finder, tools = finder_for()
    args = {'retain': [tools[0].name]}
    assert change(finder, **args)['retained_tools'] == [tools[0].name]
    finder.search({'tool_names': [t.name for t in tools[2:]]})
    assert {t.name for t in finder.window(finder._current, finder.consume())} == {'agent.capabilities', tools[0].name, tools[2].name}
    change(finder, release=[tools[0].name])
    assert change(finder, **args)['retained_tools'] == [tools[0].name]
    change(finder, release=[tools[0].name])
    finder.search({'tool_names': [t.name for t in tools[2:]]})
    assert {t.name for t in finder.window(finder._current, finder.consume())} == {'agent.capabilities', tools[2].name, tools[3].name}


def test_pending_commitments_capacity_and_invalid_retain_are_atomic():
    finder, tools = finder_for()
    finder.search({'tool_names': [t.name for t in tools[2:]]})
    checkpoint = finder.checkpoint()
    with pytest.raises(ToolPipelineError):
        change(finder, retain=[tools[0].name])
    assert finder.checkpoint() == checkpoint
    assert {t.name for t in finder.consume()} == {tools[2].name, tools[3].name}
    with pytest.raises(ToolPipelineError):
        change(finder, retain=['hidden.inspect'])
    assert change(finder, retain=[t.name for t in tools[:2]])['retained_tools'] == [t.name for t in tools[:2]]
    overflow = finder.search({'tool_names': [tools[2].name]})
    assert overflow['not_loaded_tools'] == [tools[2].name]
    assert overflow['loaded_tools'] == []


def test_revocation_overrides_retention_and_does_not_resurrect_it():
    allowed = [True]
    finder, tools = finder_for(authorized=allowed)
    change(finder, retain=[tools[0].name])
    allowed[0] = False
    result = finder.search({'tool_names': [tools[0].name]})
    assert result['retained_tools'] == []
    assert tools[0].name not in {t.name for t in finder.window(finder._current, finder.consume())}
    allowed[0] = True
    assert finder.search({})['retained_tools'] == []
    fresh, _ = finder_for()
    assert fresh.search({})['retained_tools'] == []


def test_pending_can_be_retained_and_release_preserves_earlier_commitments():
    finder, tools = finder_for()
    finder.search({'tool_names': [tools[2].name]})
    assert change(finder, retain=[tools[2].name])['retained_tools'] == [tools[2].name]
    change(finder, release=[tools[2].name])
    result = finder.search({'tool_names': [tools[0].name, tools[1].name]})
    assert result['loaded_tools'] == [tools[2].name, tools[0].name]
    assert result['not_loaded_tools'] == [tools[1].name]


def test_window_array_schema_rejects_overflow_before_handler():
    from tests.test_agent_schema_arrays import execute

    trace = []
    with pytest.raises(ToolPipelineError, match=r'arguments\.window\.retain'):
        asyncio.run(execute(CAPABILITY_PARAMETERS, {'window': {'retain': ['item.read0'] * 33}}, trace))
    assert trace == []


@pytest.mark.parametrize('arguments', [
    {'window': {'retain': 'item.read0'}},
    {'window': {'retain': ['item.read0'] * 33}},
    {'window': {'retain': ['item.read0'], 'release': ['item.read0']}},
    {'window': {'retain': [' item.read0']}},
    {'window': {'retain': ['']}},
    {'window': {'other': []}},
    {'window': {}, 'query': 'inspect'},
])
def test_window_validation_rejects_invalid_arguments(arguments):
    with pytest.raises(AgentToolError):
        capability_arguments(arguments)


@pytest.mark.parametrize('operation', ['retain', 'release'])
def test_real_projection_failure_restores_retention_and_earlier_pending(operation):
    observed = []

    class FailWindow(DefaultProjector):
        def project(self, value):
            if hasattr(value, 'data') and value.data.get('window_update') == operation:
                observed.append(operation)
                raise ToolPipelineError('window projection failed', code='invalid_tool_result')
            return super().project(value)

    anchor = read_tool('anchor.inspect')
    others = [read_tool(f'phase.item{i}') for i in range(10)]
    prior = [call('agent.capabilities', {'window': {'retain': [anchor.name]}}, 'pin')] if operation == 'release' else []
    rounds = [prior or [call('anchor.inspect')],
              [call('agent.capabilities', {'tool_names': [others[0].name]}, 'earlier'),
               call('agent.capabilities', {'window': {operation: [anchor.name]}}, 'fail')],
              [call('agent.capabilities', {'tool_names': [t.name for t in others[1:7]]}, 'phase1')],
              [call('agent.capabilities', {'tool_names': [t.name for t in others[7:]]}, 'phase2')], final()]
    session, model = session_for([anchor, *others], rounds, projector=FailWindow())
    session.retriever = Initial()
    session.limits = replace(session.limits, capability_window=10)
    asyncio.run(collect(session.run(request())))
    assert observed == [operation]
    assert 'phase__item0' in {t['name'] for t in model.requests[2].tools}
    last = {t['name'] for t in model.requests[-1].tools}
    assert ('anchor__inspect' in last) == (operation == 'release')
