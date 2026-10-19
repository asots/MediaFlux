"""真实 indexer domain DTO 经三种原生模型协议进入会话的聚合失败收束回归。"""
import asyncio
import json
from typing import cast
from unittest.mock import patch

import pytest

from app.agent.domain_catalog import build_tool_specs
from app.agent.domains.resources import indexer_actions as actions
from app.agent.kernel.capabilities import CapabilityRetriever, ToolCatalog
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports.existing_actions import adapt_tool_spec
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state_types import AgentInput
from app.indexers.errors import IndexerUnavailable
from app.indexers.providers.base import IndexerAdapter
from app.indexers.registry import IndexerRegistry
from app.indexers.result_store import IndexerResultStore
from app.indexers.service import IndexerService
from tests.test_indexer_service import FakeAdapter
from tests.test_agent_tool_argument_ingress import Wire
from tests.test_agent_tool_argument_ingress import offline as offline
from tests.test_agent_tool_argument_ingress import store as store

CLAIM = "所有来源均已检索完成，没有匹配资源。"

def reply(protocol, name=None):
    if protocol == "chat_completions":
        message = {"tool_calls": [{"id": "search", "type": "function", "function": {
            "name": name, "arguments": '{"title":"Demo"}'}}]} if name else {"content": CLAIM}
        body = {"choices": [{"message": message, "finish_reason": "tool_calls" if name else "stop"}]}
    elif protocol == "responses":
        item = {"type": "function_call", "id": "fc_search", "call_id": "search", "name": name,
                "arguments": '{"title":"Demo"}'} if name else {"type": "message", "content": [{"type": "output_text", "text": CLAIM}]}
        body = {"status": "completed", "output": [item]}
    else:
        item = {"type": "tool_use", "id": "search", "name": name, "input": {"title": "Demo"}} if name else {"type": "text", "text": CLAIM}
        body = {"content": [item], "stop_reason": "tool_use" if name else "end_turn"}
    return json.dumps(body).encode(), "application/json"

@pytest.mark.parametrize("protocol", ["chat_completions", "responses", "anthropic_messages"])
@pytest.mark.parametrize("mode", ["all-failed", "mixed", "successful-empty-control"])
def test_service_failure_is_not_completed(protocol, mode, store):
    async def scenario():
        failed = mode != "successful-empty-control"
        provider = FakeAdapter("nyaa", error=IndexerUnavailable() if failed else None)
        providers = {"nyaa": provider}
        if mode == "mixed":
            providers["backup"] = FakeAdapter("backup")
        service = IndexerService(registry=IndexerRegistry(cast(dict[str, IndexerAdapter], providers)), result_store=IndexerResultStore())
        spec = next(s for s in build_tool_specs() if s.name == "indexer.search_resources")
        tool = adapt_tool_spec(spec)
        catalog = ToolCatalog([tool])
        wire = Wire(protocol, [reply(protocol, tool.model_name), reply(protocol)])
        session = AgentSession(model=wire.adapter, catalog=catalog, state_store=store, journal=store,
            pipeline=ToolPipeline(catalog=catalog, state_store=store),
            retriever=CapabilityRetriever(minimum=1, maximum=1))
        original = actions.config.get_bool
        with patch.object(actions, "get_indexer_service", return_value=service), patch.object(
            actions.config, "get_bool", side_effect=lambda key, *a, **kw: True if key == "INDEXER_SEARCH_ENABLED" else original(key, *a, **kw)):
            events = [e async for e in session.run(AgentInput(owner="owner", session_id="audit-partial", message="请查完所有来源，确认是否有资源"))]
        wire.verify(posts=2)
        assert provider.calls >= 1
        state = await store.load(owner="owner", session_id="audit-partial")
        dto = json.loads(next(row["content"] for row in state.conversation if row.get("role") == "tool"))
        if mode == "all-failed":
            assert dto["ok"] is False and dto["status"] == "partial" and dto["data"]["errors"]
            assert events[-1].type.value == "turn.failed", "所有索引服务失败时不能发布success"
            assert CLAIM not in json.dumps(state.conversation, ensure_ascii=False)
        elif mode == "mixed":
            assert dto["ok"] is True and dto["status"] == "partial" and dto["data"]["errors"]
            assert events[-1].payload["status"] == "partial", "部分索引服务失败时不能发布全部完成"
            assert CLAIM not in json.dumps(state.conversation, ensure_ascii=False)
        else:
            assert dto["ok"] is True and dto["status"] == "empty"
            assert events[-1].payload["status"] == "success"
    asyncio.run(scenario())
