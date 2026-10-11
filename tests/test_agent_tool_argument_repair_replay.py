import asyncio
import json
import pytest
from tests.test_agent_tool_argument_ingress import offline as offline, store as store, Wire, wire, frames, event_wire, encode, observe
from app.agent.kernel.capabilities import CapabilityRetriever, KernelToolSpec, ToolCatalog, ToolEffect
from app.agent.kernel.model import ModelMessage, ModelRequest, ModelEventType
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.session import AgentSession, SessionLimits
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.state import AgentInput, CancellationToken
from app.clients.openai_compatible import parse_native_tool_turn
from app.agent.public_view import public_conversation_messages
from app.agent.kernel.provider_model import ModelProviderError

RAW = '  {"query":"私有原串-SENTINEL", \n'
OPAQUE = ' opaque\nPRIVATE-SENTINEL  '
PROTOCOLS = ['chat_completions', 'responses']


def payload(protocol, mode, raw=RAW, cid='call-original', name='lookup'):
    data, kind = wire(protocol, mode, raw)
    if mode == 'json':
        value = json.loads(data)
        if protocol == 'chat_completions':
            message = value['choices'][0]['message']
            call = message['tool_calls'][0]
            call['id'] = cid
            call['function']['name'] = name
            message['reasoning_content'] = OPAQUE
        else:
            call = value['output'][0]
            call['call_id'] = cid
            call['name'] = name
            value['output'].insert(0, {'type':'reasoning','id':'rs-private','encrypted_content':OPAQUE})
        return encode(value).encode(), kind
    events = frames(protocol, raw, final=raw)
    for event in events:
        if isinstance(event, dict):
            if protocol == 'chat_completions':
                delta = event['choices'][0]['delta']
                delta['reasoning_content'] = OPAQUE if delta.get('tool_calls') else ''
                for call in delta.get('tool_calls', []):
                    call['id'] = cid
                    call['function']['name'] = name
            elif isinstance(event.get('item'), dict):
                event['item']['call_id'] = cid
                event['item']['name'] = name
    if protocol == 'responses':
        events.insert(-1, {'type':'response.output_item.done','output_index':1,'item':{'type':'reasoning','id':'rs-private','encrypted_content':OPAQUE}})
    return event_wire(events)


def final(protocol):
    value = ({'choices':[{'message':{'content':'已核验'},'finish_reason':'stop'}]} if protocol == 'chat_completions' else
             {'status':'completed','output_text':'已核验','output':[]})
    return encode(value).encode(), 'application/json'


def make_session(model, store, executed, limits=None):
    def handler(arguments, context):
        executed.append(dict(arguments))
        return {'ok':True,'status':'success','summary':'已核验','data':{}}
    catalog = ToolCatalog([KernelToolSpec(name='lookup', domain='library', description='只读核验', input_schema={'type':'object','properties':{'query':{'type':'string'}},'required':['query']}, effect=ToolEffect.READ, read=handler)])
    return AgentSession(model=model, catalog=catalog, retriever=CapabilityRetriever(minimum=1,maximum=1), pipeline=ToolPipeline(catalog=catalog,state_store=store), state_store=store, **({'limits':limits} if limits else {}))


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('mode', ['json','sse'])
def test_real_cold_restore_next_wire_and_one_correction(protocol, mode, store):
    async def run():
        executed = []
        first = Wire(protocol, [payload(protocol, mode)])
        entered = asyncio.Event()
        class StopAtNextRequest:
            count = 0
            async def stream(self, request, *, cancellation):
                self.count += 1
                if self.count == 2:
                    entered.set()
                    await asyncio.Event().wait()
                async for event in first.adapter.stream(request, cancellation=cancellation):
                    yield event
        session = make_session(StopAtNextRequest(), store, executed)
        async def consume():
            return [e async for e in session.run(AgentInput(message='只读核验',owner='owner',session_id='cold'))]
        task = asyncio.create_task(consume())
        waiter = asyncio.create_task(entered.wait())
        done, _ = await asyncio.wait((task, waiter),timeout=3,return_when=asyncio.FIRST_COMPLETED)
        assert done, 'bounded model boundary handshake did not complete'
        if entered.is_set():
            assert await session.cancel(owner='owner',session_id='cold')
        else:
            waiter.cancel()
        before = await asyncio.wait_for(task,3)
        await asyncio.gather(waiter,return_exceptions=True)
        first.verify()
        cold = SQLiteKernelStore(secret_provider=lambda:'offline-tool-argument-test')
        state = await cold.load(owner='owner',session_id='cold')
        rows = [row for row in state.conversation if row.get('tool_calls')]
        assert rows, 'Malformed identifiable call must be durably repairable'
        call = rows[0]['tool_calls'][0]
        assert call['arguments'] is None and call['argument_error'] == 'invalid_json' and call['raw_arguments'] == RAW
        restored = AgentSession._restore_messages(state)
        assert any(m.tool_calls and m.tool_calls[0].raw_arguments == RAW for m in restored)
        errors = [row for row in state.conversation if row['role'] == 'tool']
        assert json.loads(errors[0]['content'])['code'] == 'invalid_arguments'
        assert 'expected_input_schema' in json.loads(errors[0]['content'])
        assert executed == []
        assert 'SENTINEL' not in json.dumps(public_conversation_messages(state.conversation),ensure_ascii=False)
        assert not any(e.type.name in ('TOOL_STARTED','EFFECT_PREVIEW_STARTED') for e in before)
        public = json.dumps([dict(e.payload) for e in before],ensure_ascii=False)
        assert 'SENTINEL' not in public
        next_wire = Wire(protocol, [payload(protocol, mode, '{"query":"example"}', 'call-corrected'), final(protocol)])
        after = [e async for e in make_session(next_wire.adapter, cold, executed).run(AgentInput(message='继续核验',owner='owner',session_id='cold'))]
        next_wire.verify(2)
        assert executed == [{'query':'example'}]
        history = next_wire.requests[0]['messages' if protocol == 'chat_completions' else 'input']
        if protocol == 'chat_completions':
            original = next(m for m in history if m.get('tool_calls'))
            assert original['tool_calls'][0]['id'] == 'call-original'
            assert original['tool_calls'][0]['function']['arguments'] == RAW
            assert original['reasoning_content'] == OPAQUE
            assert any(m.get('tool_call_id') == 'call-original' for m in history)
        else:
            original = next(m for m in history if m.get('call_id') == 'call-original' and m['type'] == 'function_call')
            assert original['arguments'] == RAW and original['id'] == 'item-0'
            assert any(m.get('encrypted_content') == OPAQUE for m in history)
            assert any(m.get('call_id') == 'call-original' and m['type'] == 'function_call_output' for m in history)
        assert 'SENTINEL' not in json.dumps([dict(e.payload) for e in after],ensure_ascii=False)
    asyncio.run(run())


def mixed_payload(protocol, bad_first, spaced):
    good_id = '  good-native  ' if spaced else 'good-native'
    items = [('bad', RAW), (good_id, '{"query":"example"}')]
    if not bad_first:
        items.reverse()
    if protocol == 'chat_completions':
        calls = [{'index':i, 'id':cid, 'function':{'name':'lookup', 'arguments':args}}
                 for i, (cid, args) in enumerate(items)]
        events = [{'choices':[{'delta':{'tool_calls':calls, 'reasoning_content':OPAQUE},
                               'finish_reason':'tool_calls'}]}, '[DONE]']
    else:
        events = []
        for i, (cid, args) in enumerate(items):
            item = {'type':'function_call', 'id':f'item-{i}', 'call_id':cid,
                    'name':'lookup', 'arguments':args}
            events.extend([{'type':'response.output_item.added', 'output_index':i, 'item':item},
                           {'type':'response.output_item.done', 'output_index':i, 'item':item}])
        events.extend([{'type':'response.output_item.done', 'output_index':2,
                        'item':{'type':'reasoning','id':'private-mixed','encrypted_content':OPAQUE}},
                       {'type':'response.completed','response':{'status':'completed'}}])
    return event_wire(events), items


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('bad_first', [True, False])
@pytest.mark.parametrize('spaced', [True, False])
def test_mixed_sse_cold_next_wire_exact_or_safe_rejection(protocol, bad_first, spaced, store):
    async def run():
        response, items = mixed_payload(protocol, bad_first, spaced)
        observed, error = await observe(protocol, response)
        if protocol == 'responses' and spaced:
            assert isinstance(error, ModelProviderError)
            assert not any(e.type is ModelEventType.FINISH for e in observed)
        else:
            assert error is None
            assert observed[-1].type is ModelEventType.FINISH
        transport = Wire(protocol, [response, final(protocol)])
        executed = []
        events = [e async for e in make_session(transport.adapter, store, executed).run(
            AgentInput(message='只读核验', owner='owner', session_id='mixed'))]
        cold = SQLiteKernelStore(secret_provider=lambda:'offline-tool-argument-test')
        state = await cold.load(owner='owner', session_id='mixed')
        # Responses may already have emitted a trimmed successful call. Reject
        # the entire turn before FINISH rather than mutate that event's identity.
        if protocol == 'responses' and spaced:
            transport.verify(1)
            assert executed == []
            assert not any(row.get('tool_calls') or row.get('role') == 'tool' for row in state.conversation)
            assert not any(e.type.name == 'TOOL_STARTED' for e in events)
            return
        transport.verify(2)
        assert executed == [{'query':'example'}]
        stored = [call for row in state.conversation for call in row.get('tool_calls', [])]
        assert sorted(c['call_id'] for c in stored) == sorted(cid for cid, _ in items)
        assert [c['name'] for c in stored] == ['lookup', 'lookup']
        assert next(c for c in stored if c['call_id'] == 'bad')['raw_arguments'] == RAW
        replay = Wire(protocol, [final(protocol)])
        _ = [e async for e in make_session(replay.adapter, cold, executed).run(
            AgentInput(message='继续核验', owner='owner', session_id='mixed'))]
        replay.verify(1)
        assert executed == [{'query':'example'}]
        for body in [transport.requests[1], replay.requests[0]]:
            history = body['messages' if protocol == 'chat_completions' else 'input']
            if protocol == 'chat_completions':
                assistant = next(row for row in history if row.get('tool_calls'))
                calls = assistant['tool_calls']
                assert [c['id'] for c in calls] == [cid for cid, _ in items]
                assert [c['function']['name'] for c in calls] == ['lookup', 'lookup']
                assert next(c for c in calls if c['id'] == 'bad')['function']['arguments'] == RAW
                assert assistant['reasoning_content'] == OPAQUE
                assert [row['tool_call_id'] for row in history if row.get('role') == 'tool'] == [cid for cid, _ in items]
            else:
                calls = [row for row in history if row.get('type') == 'function_call']
                assert [(c['call_id'], c['id'], c['name']) for c in calls] == [(cid, f'item-{i}', 'lookup') for i, (cid, _) in enumerate(items)]
                assert next(c for c in calls if c['call_id'] == 'bad')['arguments'] == RAW
                assert any(row.get('encrypted_content') == OPAQUE for row in history)
                assert sorted(row['call_id'] for row in history if row.get('type') == 'function_call_output') == sorted(cid for cid, _ in items)
    asyncio.run(run())


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('bad_first', [True, False])
@pytest.mark.parametrize('field,value', [('name', 123), ('call_id', 123), ('call_id', '\ud800')])
def test_mixed_invalid_success_identity_rejects_whole_turn(protocol, bad_first, field, value, store):
    async def run():
        response, _ = mixed_payload(protocol, bad_first, False)
        frames = [json.loads(line[6:]) if line[6:] != '[DONE]' else '[DONE]'
                  for line in response[0].decode().splitlines() if line.startswith('data: ')]
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            if protocol == 'chat_completions':
                for call in frame['choices'][0]['delta']['tool_calls']:
                    if call['id'] == 'good-native':
                        if field == 'name':
                            call['function']['name'] = value
                        else:
                            call['id'] = value
            elif frame.get('item', {}).get('call_id') == 'good-native':
                frame['item'][field] = value
        # ensure_ascii transports an invalid scalar to the actual JSON decoder.
        response = (''.join('data: ' + (frame if isinstance(frame, str) else json.dumps(frame)) + '\n\n' for frame in frames).encode(), 'text/event-stream')
        observed, error = await observe(protocol, response)
        assert isinstance(error, ModelProviderError)
        assert not any(e.type is ModelEventType.FINISH for e in observed)
        executed = []
        transport = Wire(protocol, [response])
        events = [e async for e in make_session(transport.adapter, store, executed).run(
            AgentInput(message='只读核验', owner='owner', session_id='invalid-mixed'))]
        transport.verify(1)
        assert not executed
        assert not any(e.type.name == 'TOOL_STARTED' for e in events)
        state = await SQLiteKernelStore(secret_provider=lambda:'offline-tool-argument-test').load(owner='owner', session_id='invalid-mixed')
        assert not any(row.get('tool_calls') or row.get('role') == 'tool' for row in state.conversation)
    asyncio.run(run())


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('mode', ['json','sse'])
@pytest.mark.parametrize('cid', [None, '', '   '])
def test_missing_identity_stays_provider_error(protocol, mode, cid):
    events, error = asyncio.run(observe(protocol, payload(protocol,mode,cid=cid)))
    assert error is not None
    assert not any(e.type is ModelEventType.FINISH for e in events)
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        assert 'SENTINEL' not in repr(current.args)
        pending.extend((current.__cause__, current.__context__))


@pytest.mark.parametrize('protocol', PROTOCOLS)
def test_strict_shared_parser_unchanged(protocol):
    data, _ = payload(protocol,'json')
    with pytest.raises(ValueError):
        parse_native_tool_turn(json.loads(data), protocol)


@pytest.mark.parametrize('raw', [' ' * 32768 + '{', '中' * 11000 + '{', '\ud800{'])
def test_invalid_raw_utf8_and_capacity_fail_closed(raw):
    # ASCII escaping permits the invalid scalar to reach decoded JSON validation.
    value = {'choices':[{'message':{'tool_calls':[{'type':'function','id':'real','function':{'name':'lookup','arguments':raw}}]},'finish_reason':'tool_calls'}]}
    events, error = asyncio.run(observe('chat_completions',(json.dumps(value).encode(),'application/json')))
    assert error is not None
    assert not any(e.type is ModelEventType.FINISH for e in events)


def test_anthropic_malformed_signed_input_stays_rejected():
    events = frames('anthropic_messages', {}, [RAW])
    events[:0] = [{'type':'content_block_start','index':2,'content_block':{'type':'thinking','thinking':OPAQUE,'signature':'signed-opaque'}}, {'type':'content_block_stop','index':2}]
    seen,error = asyncio.run(observe('anthropic_messages',event_wire(events)))
    assert error is not None and not any(e.type is ModelEventType.FINISH for e in seen)


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('mode', ['json', 'sse'])
def test_exact_opaque_identity_is_not_trimmed(protocol, mode):
    events, error = asyncio.run(observe(protocol, payload(protocol, mode, cid='  native-id  ')))
    assert error is None
    call = next(e.tool_call for e in events if e.tool_call)
    assert call.call_id == '  native-id  ' and call.raw_arguments == RAW
    restored = ModelMessage.from_dict(ModelMessage(role='assistant', tool_calls=(call,)).to_dict())
    assert restored.tool_calls[0] == call


@pytest.mark.parametrize('protocol', PROTOCOLS)
def test_duplicate_native_identity_and_invalid_private_reject(protocol):
    data, kind = payload(protocol, 'json')
    value = json.loads(data)
    if protocol == 'chat_completions':
        calls = value['choices'][0]['message']['tool_calls']
        calls.append(dict(calls[0]))
    else:
        value['output'].append(dict(value['output'][1]))
    events, error = asyncio.run(observe(protocol, (encode(value).encode(), kind)))
    assert error is not None and not any(e.type is ModelEventType.FINISH for e in events)
    value = json.loads(data)
    if protocol == 'chat_completions':
        value['choices'][0]['message']['reasoning_content'] = {'bad':OPAQUE}
    else:
        value['output'][0]['encrypted_content'] = {'bad':OPAQUE}
    events, error = asyncio.run(observe(protocol, (encode(value).encode(), kind)))
    assert error is not None and not any(e.type is ModelEventType.FINISH for e in events)


def test_unknown_tool_precedes_syntax_repair(store):
    async def run():
        executed=[]
        transport=Wire('chat_completions',[payload('chat_completions','json',name='not_loaded'), final('chat_completions')])
        events=[e async for e in make_session(transport.adapter,store,executed).run(AgentInput(message='只读核验',owner='owner',session_id='unknown'))]
        transport.verify(2)
        assert executed == []
        feedback=[m for m in transport.requests[1]['messages'] if m.get('role') == 'tool']
        error=json.loads(feedback[0]['content'])
        assert error['code'] == 'tool_not_available' and 'expected_input_schema' not in error
        assert not any(e.type.name == 'TOOL_STARTED' for e in events)
    asyncio.run(run())


def test_actual_raw_storage_capacity_rejects_whole_batch(store):
    async def run():
        executed=[]
        raw='{' + 'x' * 32000
        data,kind=payload('chat_completions','json',raw)
        value=json.loads(data)
        item=value['choices'][0]['message']['tool_calls'][0]
        value['choices'][0]['message']['tool_calls']=[{**item,'id':f'large-{i}'} for i in range(6)]
        transport=Wire('chat_completions',[(encode(value).encode(),kind)])
        events=[e async for e in make_session(transport.adapter,store,executed).run(AgentInput(message='只读核验',owner='owner',session_id='capacity'))]
        transport.verify()
        assert not executed and not any(e.type.name == 'TOOL_STARTED' for e in events)
        assert events[-1].payload['code'] == 'history_capacity_exceeded'
        state=await SQLiteKernelStore(secret_provider=lambda:'offline-tool-argument-test').load(owner='owner',session_id='capacity')
        assert not any(row.get('tool_calls') for row in state.conversation)
    asyncio.run(run())


@pytest.mark.parametrize('protocol', PROTOCOLS)
def test_malformed_native_binding_cannot_be_dropped(protocol):
    async def run():
        transport=Wire(protocol,[payload(protocol,'json')])
        events=[e async for e in transport.adapter.stream(ModelRequest(system_prompt='',messages=(),tools=()),cancellation=CancellationToken())]
        transport.verify()
        call=next(e.tool_call for e in events if e.tool_call)
        assert call.arguments is None
        message=ModelMessage(role='assistant',tool_calls=(call,),provider_state=events[-1].provider_state)
        replay=Wire(protocol,[])
        with pytest.raises(ModelProviderError):
            _=[e async for e in replay.adapter.stream(ModelRequest(system_prompt='',messages=(message,),tools=(),model='different-model'),cancellation=CancellationToken())]
        assert replay.requests == [] and all(c._client.is_closed for c in replay.clients)
    asyncio.run(run())


def test_second_batch_repair_does_not_replay_first_result(store):
    async def run():
        executed=[]
        transport=Wire('chat_completions',[payload('chat_completions','json','{"query":"first"}','call-first'),payload('chat_completions','json',cid='call-bad'),payload('chat_completions','json','{"query":"corrected"}','call-corrected'),final('chat_completions')])
        events=[e async for e in make_session(transport.adapter,store,executed).run(AgentInput(message='只读核验',owner='owner',session_id='second'))]
        transport.verify(4)
        assert executed == [{'query':'first'},{'query':'corrected'}]
        assert events[-1].payload['status'] == 'success'
        assert events[-1].payload['tool_calls'] == 3
    asyncio.run(run())


def test_repeated_errors_exhaust_normal_budget(store):
    async def run():
        executed=[]
        transport=Wire('chat_completions',[payload('chat_completions','json',cid=f'call-{i}') for i in range(3)])
        events=[e async for e in make_session(transport.adapter,store,executed,SessionLimits(max_model_rounds=3,max_tool_calls=2)).run(AgentInput(message='只读核验',owner='owner',session_id='budget'))]
        transport.verify(3)
        assert not executed
        assert events[-1].payload.get('finish_reason') == 'tool_budget_exceeded'
        assert events[-1].payload['tool_calls'] == 2
    asyncio.run(run())
