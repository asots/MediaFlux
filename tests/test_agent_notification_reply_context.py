"""通知引用应跨轮保留指代，不混入公共聊天、授权或旧话题路由。"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.agent.domain_catalog import build_tool_specs
from app.agent.kernel.capabilities import (
    CapabilityRetriever,
    KernelToolSpec,
    ToolCatalog,
    ToolEffect,
)
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelMessage
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
from app.agent.kernel.public_view import public_conversation_messages
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import AgentInput, InMemorySessionStateStore, SessionState
from app.bot.agent_adapter import _reply_context

NOTICE = (
    "⚠️ 本地下载整理部分完成\n状态：需要处理\n来源：1 已扫描 · 1 已配置\n"
    "候选：2 个\n任务：1 完成 · 1 待确认 · 0 失败\n"
    "文件：0 已归档 · 1 按冲突策略跳过\n本地待确认项目将继续发送候选卡。"
)


class RecordingModel:
    def __init__(self):
        self.requests = []

    async def stream(self, request, *, cancellation):
        cancellation.raise_if_cancelled()
        self.requests.append(request)
        yield ModelEvent(ModelEventType.TEXT_DELTA, text="请以该批次的文件明细为准。")
        yield ModelEvent(ModelEventType.FINISH, finish_reason="stop")


def _read_tool(name, description="读取状态"):
    return KernelToolSpec(
        name=name, domain="test", description=description,
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        effect=ToolEffect.READ,
        read=lambda _args, _ctx: {"summary": "读取完成"},
    )


def test_telegram_prefers_selected_quote_and_supports_external_quote():
    message = SimpleNamespace(
        quote=SimpleNamespace(text=NOTICE),
        reply_to_message=SimpleNamespace(text="旧内容" * 800 + NOTICE, caption=""),
    )
    assert _reply_context(message) == {"text": NOTICE}
    message.reply_to_message = None
    assert _reply_context(message) == {"text": NOTICE}
    message.quote.text = "字" * 5_000
    assert len(_reply_context(message)["text"]) == 2_000


def test_telegram_falls_back_to_caption_and_does_not_invent_quote():
    message = SimpleNamespace(reply_to_message=SimpleNamespace(text="", caption=NOTICE))
    assert _reply_context(message) == {"text": NOTICE}
    assert _reply_context(SimpleNamespace()) == {}


def test_reference_is_retained_across_followups_but_not_public_chat():
    async def scenario():
        catalog = ToolCatalog([_read_tool("local.read")])
        state_store = InMemorySessionStateStore()
        model = RecordingModel()
        for index, text in enumerate(("这个什么意思", "哪个已完成那个跳过了", "那一个还待确认呢")):
            # 每轮重建 Kernel，引用只能从状态恢复，不能依赖进程内临时变量。
            session = AgentSession(
                model=model, catalog=catalog, retriever=CapabilityRetriever(),
                pipeline=ToolPipeline(catalog=catalog, state_store=state_store),
                state_store=state_store,
            )
            events = [event async for event in session.run(AgentInput(
                message=text, owner="owner", session_id="session",
                reply_context={"text": NOTICE, "ignored_private_field": "do-not-store"} if index == 0 else {},
            ))]
            assert events[-1].type == AgentEventType.TURN_COMPLETED
            assert NOTICE in "\n".join(message.content for message in model.requests[-1].messages)
        stored = await state_store.load(owner="owner", session_id="session")
        first = stored.conversation[0]
        assert first["content"] == "这个什么意思"
        assert first["reply_context"] == {"text": NOTICE}
        assert "do-not-store" not in str(stored.conversation)
        public = public_conversation_messages(stored.conversation)
        assert public[0] == {"role": "user", "content": "这个什么意思"}
        assert NOTICE not in str(public)
        context = AgentSession._capability_retrieval_context(stored)
        assert any(NOTICE in hint for hint in context["recent_user_messages"])
        other = await state_store.load(owner="other", session_id="session")
        assert not other.conversation
        another_session = await state_store.load(owner="owner", session_id="different")
        assert not another_session.conversation
    asyncio.run(scenario())


def test_pasted_notification_stays_visible_and_is_usable_as_recent_context():
    messages = [ModelMessage(role="user", content=NOTICE), ModelMessage(role="assistant", content="已收到。")]
    stored = AgentSession._persisted_conversation(messages, current_user_index=0, original_message=NOTICE)
    assert stored[0]["content"] == NOTICE
    assert "reply_context" not in stored[0]
    context = AgentSession._capability_retrieval_context(SessionState(
        owner="owner", session_id="session", conversation=stored,
    ))
    assert NOTICE in context["recent_user_messages"][0]


def test_history_tool_context_decays_by_turn_not_just_last_six_tools():
    state = SessionState(owner="owner", session_id="session", conversation=[
        {"role": "user", "content": "昨天RSS跳过哪些"},
        {"role": "tool", "tool_name": "rss.entry_summaries", "content": "旧结果"},
        {"role": "user", "content": "这次本地整理呢", "reply_context": {"text": NOTICE}},
        {"role": "tool", "tool_name": "local_media.task_summaries", "content": "新结果"},
    ])
    context = AgentSession._capability_retrieval_context(state)
    weights = context["recent_tool_weights"]
    assert weights["local_media.task_summaries"] == 1.0
    assert weights["rss.entry_summaries"] < weights["local_media.task_summaries"]
    assert NOTICE in context["recent_user_messages"][0]


def test_retriever_respects_ordered_user_hints():
    catalog = ToolCatalog([
        _read_tool("a.read", "alpha reports"), _read_tool("b.read", "beta reports"),
    ])
    retriever = CapabilityRetriever(minimum=1, maximum=1)
    first = retriever.retrieve("continue", catalog, context={"recent_user_messages": ("alpha", "beta")})
    second = retriever.retrieve("continue", catalog, context={"recent_user_messages": ("beta", "alpha")})
    assert first.names == ("a.read",)
    assert second.names == ("b.read",)


def test_reference_focus_selects_original_domain_despite_rss_history():
    catalog = catalog_from_tool_specs(build_tool_specs())
    state = SessionState(owner="owner", session_id="session", conversation=[
        {"role": "user", "content": "RSS已下载过的条目为何跳过"},
        {"role": "tool", "tool_name": "rss.diagnose", "content": "旧结果"},
        {"role": "tool", "tool_name": "rss.entry_summaries", "content": "旧结果"},
        {"role": "user", "content": "这条通知呢", "reply_context": {"text": NOTICE}},
        {"role": "assistant", "content": "需要查本次文件结果。"},
    ])
    selection = CapabilityRetriever().retrieve(
        "哪个已完成那个跳过了", catalog,
        context=AgentSession._capability_retrieval_context(state),
    )
    assert "local_media.task_summaries" in selection.names
    assert selection.scores["local_media.task_summaries"] > selection.scores["rss.entry_summaries"]


def test_fast_paste_then_followup_keeps_user_notice_when_old_model_is_cancelled():
    async def scenario():
        first_started = asyncio.Event()
        class SlowFirstModel(RecordingModel):
            async def stream(self, request, *, cancellation):
                self.requests.append(request)
                if len(self.requests) == 1:
                    first_started.set()
                    while True:
                        await asyncio.sleep(0.005)
                        cancellation.raise_if_cancelled()
                yield ModelEvent(ModelEventType.TEXT_DELTA, text="需检查该批次文件明细。")
                yield ModelEvent(ModelEventType.FINISH, finish_reason="stop")
        catalog = ToolCatalog([_read_tool("local.read")])
        store = InMemorySessionStateStore()
        model = SlowFirstModel()
        session = AgentSession(
            model=model, catalog=catalog, retriever=CapabilityRetriever(),
            pipeline=ToolPipeline(catalog=catalog, state_store=store), state_store=store,
        )
        async def collect(message):
            return [event async for event in session.run(AgentInput(
                message=message, owner="owner", session_id="session",
            ))]
        first = asyncio.create_task(collect(NOTICE))
        await asyncio.wait_for(first_started.wait(), timeout=2)
        second = await collect("哪个已完成那个跳过了")
        previous = await asyncio.wait_for(first, timeout=2)
        assert second[-1].type == AgentEventType.TURN_COMPLETED
        assert previous[-1].type == AgentEventType.TURN_CANCELLED
        assert NOTICE in [message.content for message in model.requests[-1].messages]
        stored = await store.load(owner="owner", session_id="session")
        users = [item["content"] for item in stored.conversation if item["role"] == "user"]
        assert users == [NOTICE, "哪个已完成那个跳过了"]
        assert sum(item["role"] == "assistant" for item in stored.conversation) == 1
    asyncio.run(scenario())


def test_notice_is_saved_before_turn_started_journal_yields_to_next_request():
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        class BlockingJournal:
            async def append(self, event, *, owner):
                assert owner == "owner"
                if event.request_id == "notice" and event.type == AgentEventType.TURN_STARTED:
                    started.set()
                    await release.wait()
        catalog = ToolCatalog([_read_tool("local.read")])
        store = InMemorySessionStateStore()
        model = RecordingModel()
        session = AgentSession(
            model=model, catalog=catalog, retriever=CapabilityRetriever(),
            pipeline=ToolPipeline(catalog=catalog, state_store=store), state_store=store,
            journal=BlockingJournal(),
        )
        async def collect(message, request_id, session_id="session"):
            return [event async for event in session.run(AgentInput(
                message=message, owner="owner", session_id=session_id, request_id=request_id,
            ))]
        first = asyncio.create_task(collect(NOTICE, "notice"))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            assert not model.requests  # 比“模型已开始”更早：首事件尚未发布。
            saved = await store.load(owner="owner", session_id="session")
            assert saved.conversation[0]["content"] == NOTICE
            # 较慢的事件持久化不能占着全局 start lock，另一会话仍可独立完成。
            unrelated = await asyncio.wait_for(collect("你好", "other", "independent"), timeout=2)
            assert unrelated[-1].type == AgentEventType.TURN_COMPLETED
            latest = await asyncio.wait_for(collect("哪个已完成那个跳过了", "followup"), timeout=2)
            assert latest[-1].type == AgentEventType.TURN_COMPLETED
            assert NOTICE in [message.content for message in model.requests[-1].messages]
        finally:
            release.set()
            previous = await asyncio.wait_for(first, timeout=2)
        assert previous[-1].type == AgentEventType.TURN_CANCELLED
        saved = await store.load(owner="owner", session_id="session")
        users = [item["content"] for item in saved.conversation if item["role"] == "user"]
        assert users == [NOTICE, "哪个已完成那个跳过了"]
        assert sum(item["role"] == "assistant" for item in saved.conversation) == 1
    asyncio.run(scenario())


def test_sensitive_input_or_reference_is_never_presaved_or_sent_to_model():
    async def scenario():
        catalog = ToolCatalog([_read_tool("local.read")])
        store = InMemorySessionStateStore()
        model = RecordingModel()
        session = AgentSession(
            model=model, catalog=catalog, retriever=CapabilityRetriever(),
            pipeline=ToolPipeline(catalog=catalog, state_store=store), state_store=store,
        )
        secret = "api_key=sk-ThisIsAFakeCredential1234567890"
        for message, quote in ((secret, {}), ("解释这条", {"text": secret})):
            events = [event async for event in session.run(AgentInput(
                message=message, reply_context=quote, owner="owner", session_id="session",
            ))]
            assert [event.type for event in events] == [AgentEventType.TURN_STARTED, AgentEventType.TURN_FAILED]
            assert events[-1].payload["code"] == "sensitive_input"
            state = await store.load(owner="owner", session_id="session")
            assert not state.conversation
            assert secret not in str(events)
        assert not model.requests
    asyncio.run(scenario())
