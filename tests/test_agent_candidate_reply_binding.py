"""Signed Telegram candidate cards retain their validated resource binding."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from app.agent.kernel.model import ModelEvent, ModelEventType
from app.agent.transports import TelegramKernelTransport
from app.bot import agent_adapter, agent_candidates
from tests import test_agent_ux_backend as ux
from tests.test_agent_kernel_telegram_adapter import TELEBOT, FakeBot, Message
from tests.test_agent_resource_batch import _target_options

store = ux.store


@pytest.mark.parametrize('kind', ['empty', 'text', 'caption', 'quote'])
def test_signed_candidate_binding_reaches_current_model_message(store, monkeypatch, kind):
    monkeypatch.setattr('app.agent.domains.resources.selection._target_options', _target_options)

    class Model:
        def __init__(self):
            self.requests = []

        async def stream(self, request, *, cancellation):
            self.requests.append(request)
            yield ModelEvent(ModelEventType.TEXT_DELTA, text='仅说明候选，尚未执行。')
            yield ModelEvent(ModelEventType.FINISH, finish_reason='stop')

    model = Model()
    session, pipeline, states = ux._runtime(store, model=model)
    runtime = NS(store=store, state_store=store, telegram=TelegramKernelTransport(session))
    message = Message('解释本批候选1和2，不下载', chat_id=-876543, user_id=73, message_id=100)
    owner, sid, _ = agent_adapter._session_for_source(message.chat.id, message.from_user.id, message)

    async def seed():
        view, _ = await ux._publish_candidates(pipeline, states, owner=owner, session_id=sid)
        draft = await agent_candidates.start_draft(runtime, owner=owner, session_id=sid, view=view)
        return view, draft

    view, draft = asyncio.run(seed())
    body = '资源推荐与批选：1 Example.S01E01.2160p；2 Example.S01E02.2160p'
    message.reply_to_message = NS(
        text=body if kind in ('text', 'quote') else '',
        caption=body if kind == 'caption' else '', message_id=90, chat=message.chat,
        reply_markup=NS(keyboard=[[{'callback_data': f"agk:s:{draft['handle']}:p"}]]),
    )
    if kind == 'quote':
        quote = NS(text='选中的候选片段：2 Example.S01E02.2160p')
        setattr(message, "quote", quote)
        body = quote.text
    # Resolve the actual signed keyboard, not injected metadata or validation.
    assert asyncio.run(agent_candidates.reply_selection_ref(
        runtime, owner=owner, session_id=sid, message=message,
    )) == view['selection_ref']
    with patch.object(agent_adapter, 'get_agent_kernel_runtime', return_value=runtime):
        result = agent_adapter._execute_query(
            FakeBot(), TELEBOT, message, chat_id=str(message.chat.id),
            user_id=str(message.from_user.id), text=message.text,
        )
    assert result.status == 'success', result
    assert len(model.requests) == 1
    expected = '已校验的回复批次，仅可从该引用解析候选编号：' + view['ref']
    last_user = next(m.content for m in reversed(model.requests[0].messages) if m.role == 'user')
    assert last_user.count(expected) == 1, {'current_message': last_user, 'expected_binding': expected}
    # Reopen SQLite independently: body capability is not the resource capability.
    cold = type(store)(secret_provider=lambda: ux.SECRET)
    saved = asyncio.run(cold.load(owner=owner, session_id=sid))
    row = next(row for row in reversed(saved.conversation) if row.get('role') == 'user')
    if kind != 'empty':
        reply = row['reply_context']
        assert reply['text'] == body
        assert reply['source']['kind'] == kind
        assert reply['reference_handle'] != view['ref']
        assert reply['reference_handle'] in last_user
    else:
        assert not row.get('reply_context')


@pytest.mark.parametrize('change', ['query_only', 'search_id', 'candidates', 'private_items'])
def test_reply_bound_batch_allows_refinement_token_but_not_rebinding(store, change):
    from copy import deepcopy
    from dataclasses import replace

    from app.agent.errors import ToolPipelineError
    from tests.test_agent_resource_batch import _prepare_batch

    async def exercise():
        _, pipeline, states = ux._runtime(store)
        view, context = await ux._publish_candidates(pipeline, states)
        original = await store.resolve(view['ref'], owner=ux.OWNER, session_id=ux.SESSION, expected_kind='resource_candidates')
        original['_query_arguments'] = {'title': 'Example', 'resolution': '2160p'}
        bound = await store.put(owner=ux.OWNER, session_id=ux.SESSION, kind='resource_candidates', value=original, ttl_seconds=900)
        context = replace(context, resource_candidate_ref=bound.ref)
        refined = await pipeline.execute('indexer.present_candidates', {
            'resource_candidates_ref': bound.ref, 'positions': [1, 2], 'exclude_hdr': True,
        }, context=context)
        new_ref = refined.outcome.public_content['reference_arguments']['resource_candidates_ref']
        assert new_ref != bound.ref
        value = await store.resolve(new_ref, owner=ux.OWNER, session_id=ux.SESSION, expected_kind='resource_candidates')
        assert value['_query_arguments']['exclude_hdr'] is True
        assert value['candidates'] == original['candidates'] and value['search_id'] == original['search_id']
        if change != 'query_only':
            value = deepcopy(value)
            if change == 'search_id':
                value['search_id'] = 'rs_' + 'z' * 16
            elif change == 'candidates':
                value['candidates'][0]['title'] = 'Different resource'
            else:
                value['_private_items'] = []
            changed = await store.put(owner=ux.OWNER, session_id=ux.SESSION, kind='resource_candidates', value=value, ttl_seconds=900)
            new_ref = changed.ref
        intent = {'source_type': 'resource_candidates', 'resource_candidates_ref': new_ref, 'positions': [1, 2], 'target': 'guangya'}
        with patch('app.agent.domains.resources.candidate_actions.prepare_submit_resource_batch', side_effect=_prepare_batch) as prepare:
            if change == 'query_only':
                # 只预检真实业务链，未确认前不执行下载；冻结的仍是原批次两个真实ID。
                result = await pipeline.execute('ingest.submit', intent, context=context)
                assert result.effect_plan is not None
                assert prepare.call_args.args[0]['result_ids'] == [item['result_id'] for item in original['candidates']]
                assert result.effect_plan.arguments['resource_candidates_ref'] == new_ref
            else:
                with pytest.raises(ToolPipelineError) as error:
                    await pipeline.execute('ingest.submit', intent, context=context)
                assert error.value.code == 'candidate_scope_mismatch'
                prepare.assert_not_called()
        assert (await store.resolve(bound.ref, owner=ux.OWNER, session_id=ux.SESSION, expected_kind='resource_candidates')) == original
    asyncio.run(exercise())
