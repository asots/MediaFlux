from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterator

from app.agent.kernel.capabilities import (
    CapabilityRetriever,
    KernelToolSpec,
    ToolCatalog,
    ToolEffect,
)
from app.agent.kernel.events import AgentEvent
from app.agent.kernel.model import (
    ModelEvent,
    ModelEventType,
    ModelRequest,
    ModelToolCall,
)
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import InMemorySessionStateStore
from app.agent.kernel.transports import (
    QueryEnvelope,
    TelegramKernelTransport,
    TransportInputError,
    WebKernelTransport,
)


class ReadThenAnswerModel:
    async def stream(
        self, request: ModelRequest, *, cancellation
    ) -> AsyncIterator[ModelEvent]:
        cancellation.raise_if_cancelled()
        has_result = any(message.role == "tool" for message in request.messages)
        if not has_result:
            yield ModelEvent(
                ModelEventType.TOOL_CALL_COMPLETED,
                tool_call=ModelToolCall("call-1", "library__count", {}),
            )
        else:
            yield ModelEvent(ModelEventType.TEXT_DELTA, text="媒体库共有 37 集")
            yield ModelEvent(ModelEventType.FINISH, finish_reason="stop")


def make_session() -> AgentSession:
    tool = KernelToolSpec(
        name="library.count",
        domain="library",
        description="读取媒体库剧集数量",
        examples=("媒体库有多少集",),
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        effect=ToolEffect.READ,
        read=lambda _arguments, _context: {
            "summary": "读取完成",
            "data": {"count": 37},
        },
    )
    catalog = ToolCatalog([tool])
    state = InMemorySessionStateStore()
    return AgentSession(
        model=ReadThenAnswerModel(),
        catalog=catalog,
        retriever=CapabilityRetriever(minimum=1, maximum=1),
        pipeline=ToolPipeline(catalog=catalog, state_store=state),
        state_store=state,
    )


class AgentKernelTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_and_telegram_use_the_same_kernel_event_contract(self) -> None:
        web = WebKernelTransport(make_session())
        web_request = QueryEnvelope(
            owner="owner-1",
            session_id="web-session",
            message="媒体库有多少集",
            request_id="request-web",
            channel="web",
        )
        web_events = [json.loads(chunk) async for chunk in web.query(web_request)]

        telegram_events: list[AgentEvent] = []

        async def observe(event: AgentEvent) -> None:
            telegram_events.append(event)

        telegram = TelegramKernelTransport(make_session())
        view = await telegram.query(
            QueryEnvelope(
                owner="owner-1",
                session_id="telegram-session",
                message="媒体库有多少集",
                request_id="request-telegram",
            ),
            observe=observe,
        )

        self.assertEqual(
            [event["type"] for event in web_events],
            [event.type.value for event in telegram_events],
        )
        self.assertEqual(view.status, "success")
        self.assertEqual(view.answer, "媒体库共有 37 集")
        self.assertEqual(web_events[-1]["payload"]["answer"], view.answer)

    async def test_transport_accepts_internal_telegram_owner_scope(self) -> None:
        request = QueryEnvelope(
            owner="tg:v1:-123\x1f456",
            session_id="tg-session",
            message="hello",
            channel="telegram",
        )
        self.assertEqual(request.to_agent_input().owner, "tg:v1:-123\x1f456")

    async def test_transport_rejects_invalid_scope_before_kernel(self) -> None:
        request = QueryEnvelope(
            owner="owner", session_id="bad session", message="hello"
        )
        with self.assertRaises(TransportInputError):
            request.to_agent_input()

class TelegramCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_requested_before_turn_start_is_not_lost(self):
        import asyncio
        from app.agent.kernel.state import CancellationToken
        from app.agent.kernel.events import AgentEventType

        session = make_session()
        original = session.run
        async def delayed_start(value):
            await asyncio.sleep(0.025)
            async for event in original(value):
                yield event
        session.run = delayed_start
        class SlowModel:
            async def stream(self, request, *, cancellation):
                await cancellation.wait()
                cancellation.raise_if_cancelled()
                yield
        session.model = SlowModel()
        token = CancellationToken()
        observed = []
        async def observe(event):
            observed.append(event.type)
        task = asyncio.create_task(TelegramKernelTransport(session).query(
            QueryEnvelope(owner="owner", session_id="stop-before-start", message="slow"),
            cancellation=token, observe=observe,
        ))
        token.cancel("service_stopping")
        view = await asyncio.wait_for(task, 1)
        self.assertEqual(view.status, "cancelled")
        self.assertIn(AgentEventType.TURN_STARTED, observed)
        self.assertIn(AgentEventType.TURN_CANCELLED, observed)
