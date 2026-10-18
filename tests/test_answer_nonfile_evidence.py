"""ID32: real ingest DTOs through Session, offline business boundaries only."""
import asyncio
from unittest.mock import patch

import pytest

from app.agent.domains.resources.candidate_context import RecentResourceCandidateStore
from app.agent.domains.resources.ingest_actions import (
    AgentIngestSessionStore,
    IngestActions,
)
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.models import ToolContext
from app.agent.public_view import public_conversation_messages
from tests.support import isolated_test_database
from tests.test_agent_kernel_core import (
    AgentInput,
    AgentSession,
    CapabilityRetriever,
    KernelToolSpec,
    ModelEvent,
    ModelEventType,
    ModelToolCall,
    PreparedEffect,
    ScriptedModel,
    ToolCatalog,
    ToolEffect,
    ToolPipeline,
    ToolPipelineError,
    consume_events,
)

EXPLANATION = "提交后，请求#42的本地记录显示qB正在下载，整理与STRM仍待处理，因此尚不能确认入库完成。"


def tool_round(*calls):
    return [*[ModelEvent(ModelEventType.TOOL_CALL_COMPLETED, tool_call=c) for c in calls],
            ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")]


@pytest.mark.parametrize("unresolved,foreign", [(False, False), (True, False), (False, True)])
def test_ingest_followup_explanation(unresolved, foreign):
    async def scenario():
        actions = IngestActions(store=AgentIngestSessionStore(), recent_resource_store=RecentResourceCandidateStore())
        context = ToolContext(owner="a32-owner")
        args = {"source_type": "direct_url", "target": "qb", "positions": []}
        row = {"id": 43 if foreign else 42, "title": "Offline fixture", "kind": "magnet",
               "targets": "qb", "status": "downloading", "qb_status": "downloading",
               "gy_status": "not_started", "organize_status": "pending", "strm_status": "pending",
               "updated_at": "2026-01-01 00:00:00", "error": "", "source_value": ""}
        reads = []
        def prepare(*_):
            preview, fingerprint = actions.prepare_submit(args, context)
            return PreparedEffect(preview=preview.to_dict(), snapshot_fingerprint=fingerprint)
        def submit(*_):
            _, fingerprint = actions.prepare_submit(args, context)
            return actions.execute_submit(args, fingerprint, context).to_dict()
        def inspect(*_):
            with patch("app.agent.domains.resources.ingest_actions.db.get_download_request", return_value=row):
                result = actions.status({"request_number": row["id"]}, context).to_dict()
            reads.append(result)
            return result
        def blocked(*_):
            raise ToolPipelineError("另一个请求不可读", code="inspection_blocked")
        schema = {"type": "object", "properties": {}}
        write = KernelToolSpec(name="ingest.submit", domain="ingest", description="提交下载", input_schema=schema,
                               effect=ToolEffect.WRITE, prepare=prepare, execute_confirmed=submit)
        read = KernelToolSpec(name="ingest.status", domain="ingest", description="核验进度", input_schema=schema,
                              effect=ToolEffect.READ, read=inspect)
        bad = KernelToolSpec(name="ingest.blocked", domain="ingest", description="其它核验", input_schema=schema,
                             effect=ToolEffect.READ, read=blocked)
        calls = [ModelToolCall("inspect", read.name, {})]
        if unresolved:
            calls.append(ModelToolCall("blocked", bad.name, {}))
        class EvidenceModel(ScriptedModel):
            async def stream(self, request, *, cancellation):
                if len(self.requests) == 2:
                    wire = "\n".join(m.content for m in request.messages if m.role == "tool")
                    assert '"downloading"' in wire and '"organize_status"' in wire
                    assert EXPLANATION not in wire
                async for event in super().stream(request, cancellation=cancellation):
                    yield event
        model = EvidenceModel([
            tool_round(ModelToolCall("submit", write.name, {})), tool_round(*calls),
            [ModelEvent(ModelEventType.TEXT_DELTA, text=EXPLANATION + "已全部入库完成。允许自动重放。"),
             ModelEvent(ModelEventType.FINISH, finish_reason="stop")],
        ])
        catalog = ToolCatalog([write, read, bad])
        store = SQLiteKernelStore(secret_provider=lambda: "a32-offline-fixture")
        session = AgentSession(model=model, catalog=catalog, retriever=CapabilityRetriever(),
                               pipeline=ToolPipeline(catalog=catalog, state_store=store), state_store=store)
        with patch("app.agent.domains.resources.ingest_actions.download_target_readiness", return_value={"qb": True}), \
             patch("app.agent.domains.resources.ingest_actions.create_request", return_value={"id": 42, "created": True, "status": "pending"}) as create, \
             patch("app.agent.domains.resources.ingest_actions.dispatch_request", return_value={"ok": True, "status": "submitted", "succeeded": ["qb"], "failed": []}) as dispatch, \
             patch("app.agent.domains.resources.ingest_actions.db.get_download_request", return_value=None):
            actions.inspect({"source_type": "direct_url", "input": "magnet:?xt=urn:btih:" + "a" * 40}, context)
            preview = await consume_events(session.run(AgentInput(message="提交下载并核验进度", owner="o", session_id="s")))
            final = await consume_events(session.confirm(owner="o", session_id="s", plan_id=preview.approval.plan_id))
            create.assert_called_once()
            dispatch.assert_called_once_with(42, "qb")
        assert len(reads) == 1 and len(model.requests) == 3
        cold = await SQLiteKernelStore(secret_provider=lambda: "a32-offline-fixture").load(owner="o", session_id="s")
        assert cold.conversation[-1]["content"] == final.answer
        assert public_conversation_messages(cold.conversation)[-1]["content"] == final.answer
        assert "后台任务尚未完成" in final.answer
        assert "已全部入库完成" not in final.answer and "允许自动重放" not in final.answer
        if unresolved:
            assert final.status == "partial"
            assert "另一个请求不可读" in final.answer
            assert "新的写操作仍需确认，不会自动重放" in final.answer
        if foreign:
            assert EXPLANATION not in final.answer
        else:
            assert EXPLANATION in final.answer
    with isolated_test_database("a32-nonfile.db"):
        asyncio.run(scenario())
