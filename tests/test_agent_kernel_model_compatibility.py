"""Independent positive compatibility assertions; no production edits."""
from collections import UserDict
from types import MappingProxyType

import pytest

from app.agent.kernel.conversation_context import restore_messages
from app.agent.kernel.model import ModelMessage, ModelToolCall

MISSING=object()
@pytest.mark.parametrize('value',[MISSING,None,{},[],False,0,'',[['query','example']]],ids=['missing','null','object','empty-list','false','zero','empty-string','pairs'])
def test_legacy_message_success_domain(value):
    raw={'call_id':'legacy-call','name':'lookup'}
    if value is not MISSING:
        raw['arguments']=value
    message=ModelMessage.from_dict({'role':'assistant','tool_calls':[raw]})
    expected={'query':'example'} if value == [['query','example']] else {}
    assert dict(message.tool_calls[0].arguments)==expected
    assert message.to_dict()['tool_calls'][0]['arguments']==expected

class DictChild(dict):
    pass
@pytest.mark.parametrize('factory',[dict,MappingProxyType,UserDict,DictChild])
def test_mapping_root_contract(factory):
    source=factory({'query':'example'})
    call=ModelToolCall('same','lookup',source)
    assert dict(call.arguments)=={'query':'example'}
    result=ModelMessage.from_dict({'role':'assistant','tool_calls':[{'call_id':'same','name':'lookup','arguments':source}]})
    assert result.to_dict()['tool_calls'][0]['arguments']=={'query':'example'}

def test_legacy_from_dict_takes_snapshot():
    source={'query':'before'}
    result=ModelMessage.from_dict({'role':'assistant','tool_calls':[{'call_id':'same','name':'lookup','arguments':source}]})
    source['query']='after'
    assert dict(result.tool_calls[0].arguments)=={'query':'before'}


@pytest.mark.parametrize('extra', [
    {'argument_error':'unknown', 'arguments':None, 'raw_arguments':'{broken'},
    {'argument_error':None, 'arguments':{}, 'raw_arguments':'{broken'},
    {'argument_error':'invalid_json', 'arguments':{}, 'raw_arguments':'{broken'},
    {'argument_error':'invalid_json', 'arguments':None, 'raw_arguments':None},
])
def test_serialized_error_states_remain_exclusive(extra):
    with pytest.raises(ValueError):
        ModelMessage.from_dict({'role':'assistant', 'tool_calls':[
            {'call_id':'native', 'name':'lookup', **extra}]})


def test_error_state_roundtrip_does_not_normalize_raw_or_none():
    raw = '  {broken\n'
    call = ModelToolCall('  native  ', 'lookup', None, 'invalid_json', raw)
    restored = ModelMessage.from_dict(ModelMessage('assistant', tool_calls=(call,)).to_dict())
    assert restored.tool_calls == (call,)
    assert restored.tool_calls[0].arguments is None


@pytest.mark.parametrize('field', ['call_id', 'name'])
def test_serialized_error_identity_is_not_coerced(field):
    raw = {'call_id':'native', 'name':'lookup', 'arguments':None,
           'argument_error':'invalid_json', 'raw_arguments':'{broken'}
    raw[field] = 123
    with pytest.raises(ValueError):
        ModelMessage.from_dict({'role':'assistant', 'tool_calls':[raw]})


def test_legacy_snapshot_is_shallow():
    nested = {'value':'before'}
    source = {'nested':nested}
    restored = ModelMessage.from_dict({'role':'assistant', 'tool_calls':[
        {'call_id':'native', 'name':'lookup', 'arguments':source}]})
    assert restored.tool_calls[0].arguments is not source
    assert restored.tool_calls[0].arguments['nested'] is nested

from types import SimpleNamespace


def test_legacy_restore_keeps_empty_args_native_group():
    state=SimpleNamespace(conversation=[
        {'role':'assistant','tool_calls':[{'call_id':'legacy','name':'lookup'}]},
        {'role':'tool','tool_call_id':'legacy','tool_name':'lookup','content':'{"ok":true}'}])
    result=restore_messages(state)
    assert [m.role for m in result]==['assistant','tool']
    assert dict(result[0].tool_calls[0].arguments)=={}
    assert result[0].tool_calls[0].call_id==result[1].tool_call_id
