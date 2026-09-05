"""生产短句的完整能力窗口与搜索引用→批量 EffectPlan 回归。"""
from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from app.agent.domain_catalog import build_tool_specs
from app.agent.ingest_actions import AgentIngestSessionStore
from app.agent.kernel.bootstrap import build_agent_kernel
from app.agent.kernel.capabilities import CapabilityRetriever
from app.agent.kernel.discovery import CapabilityDiscovery
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import AgentInput, InMemorySessionStateStore
from app.agent.models import ToolReference, ToolResult
from app.agent.recent_resource_candidates import (
    RecentResourceCandidateStore,
    new_resource_search_id,
    safe_resource_snapshot,
)
from tests.test_agent_kernel_capability_corpus import NoModel
from tests.test_agent_kernel_resource_ingest import (
    SearchThenAnswerModel,
    _collect,
    _reference_from,
    _search_result,
)


class SubmitTwoFromHistoryModel:
    def __init__(self):
        self.requests = []

    async def stream(self, request, *, cancellation):
        cancellation.raise_if_cancelled()
        self.requests.append(request)
        yield ModelEvent(
            ModelEventType.TOOL_CALL_COMPLETED,
            tool_call=ModelToolCall("batch-followup", "ingest.submit", {
                "source_type": "resource_candidates", "target": "guangya", "positions": [3, 4],
                "resource_candidates_ref": _reference_from(request),
            }),
        )
        yield ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")


class AgentKernelBtFollowupTests(unittest.IsolatedAsyncioTestCase):
    def test_short_numbered_cloud_followup_keeps_submit_and_discovery_visible(self):
        session = build_agent_kernel(model=NoModel())
        context = {
            "owner": "fixture-owner", "session_id": "fixture-session", "channel": "test",
            "reference_kinds": ("resource_candidates",),
            "recent_user_messages": ("囧徒预演告别是什么片子", "搜一下资源"),
            "recent_tool_names": ("discovery.search", "web.search", "indexer.search_resources"),
        }
        for message in ("推送 3，4 到云盘", "推送3、4到光鸭"):
            with self.subTest(message=message):
                selected = session.retriever.retrieve(message, session.catalog, context=context)
                discovery = CapabilityDiscovery(session.catalog, context=context, maximum=session.retriever.maximum)
                window = discovery.window(selected.tools)
                names = [tool.name for tool in window]
                self.assertEqual(names[0], "agent.capabilities")
                self.assertIn("ingest.submit", names)
                self.assertLessEqual(len(names), session.retriever.maximum)

    async def test_real_catalog_followup_builds_batch_plan_and_only_executes_after_confirmation(self):
        result = _search_result()
        template = result.data["items"][0]
        result.data["items"] = [{**template, "result_id": f"fixture-resource-{i}"} for i in range(1, 5)]
        result.references = [ToolReference(
            "resource_candidates", safe_resource_snapshot(result, search_id=new_resource_search_id())
        )]
        specs = build_tool_specs(RecentResourceCandidateStore(), AgentIngestSessionStore())
        catalog = catalog_from_tool_specs(tuple(
            replace(spec, handler=lambda _arguments: result)
            if spec.name == "indexer.search_resources" else spec for spec in specs
        ))
        state = InMemorySessionStateStore()
        pipeline = ToolPipeline(catalog=catalog, state_store=state)
        first = AgentSession(model=SearchThenAnswerModel(), catalog=catalog, retriever=CapabilityRetriever(),
                             pipeline=pipeline, state_store=state)
        await _collect(first.run(AgentInput(message="搜一下资源", owner="fixture-owner", session_id="fixture-session")))
        model = SubmitTwoFromHistoryModel()
        followup = AgentSession(model=model, catalog=catalog, retriever=CapabilityRetriever(),
                                pipeline=pipeline, state_store=state)
        with (
            patch("app.agent.indexer_candidate_actions.prepare_submit_resource_batch", return_value=(
                ToolResult(True, "confirmation_required", "确认后提交 2 项资源"), "fixture-batch-context"
            )) as prepare,
            patch("app.agent.indexer_candidate_actions.submit_resource_batch_confirmed", return_value=(
                ToolResult(True, "accepted", "已提交到光鸭")
            )) as execute,
        ):
            events = await _collect(followup.run(AgentInput(
                message="推送 3，4 到云盘", owner="fixture-owner", session_id="fixture-session"
            )))
            self.assertFalse(any(event.type is AgentEventType.TOOL_FAILED for event in events))
            plan = next(event.payload["plan"] for event in events if event.type is AgentEventType.EFFECT_APPROVAL_REQUIRED)
            arguments = {"result_ids": ["fixture-resource-3", "fixture-resource-4"], "target": "guangya"}
            prepare.assert_called_once_with(arguments)
            execute.assert_not_called()
            self.assertEqual(model.requests[0].tools[0]["name"], "agent__capabilities")
            self.assertIn("ingest__submit", [tool["name"] for tool in model.requests[0].tools])
            confirmed = await _collect(followup.confirm(
                owner="fixture-owner", session_id="fixture-session", plan_id=plan["plan_id"]
            ))
            execute.assert_called_once_with(arguments, "fixture-batch-context")
            self.assertTrue(any(event.type is AgentEventType.EFFECT_COMPLETED for event in confirmed))
