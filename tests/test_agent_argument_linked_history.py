import pytest
from app.agent.kernel.model import ModelMessage,ModelToolCall
from app.agent.model_context_budget import bounded_model_messages


@pytest.mark.parametrize('count',[20,2000])
def test_history_groups_are_kept_whole_or_omitted_without_blocking_current(count):
    args={'objects':['OBJ'+'A'*24]*count}
    messages=[ModelMessage(role='user',content='完整读取狐妖小红娘并给方案'),ModelMessage(role='assistant',tool_calls=(ModelToolCall('large-call','library.status',args),)),ModelMessage(role='tool',tool_call_id='large-call',tool_name='library.status',content='{"ok":true,"status":"success","summary":"完整读取完成","data":"'+'x'*100000+'"}'),ModelMessage(role='assistant',content='方案 A：按 TMDB 分季规整'),ModelMessage(role='user',content='方案 A')]
    bounded=bounded_model_messages(messages,system_prompt='',context_window_tokens=16384,output_tokens=1024,history_end=4,tool_definitions=())
    if count == 2000:
        assert bounded == (messages[-1],)
        return
    assert any(m.content=='方案 A：按 TMDB 分季规整' for m in bounded), 'do not silently send dependent follow-up with its plan removed'
    assert any(c.arguments==args for m in bounded for c in m.tool_calls)
