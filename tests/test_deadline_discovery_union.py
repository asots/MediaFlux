"""Bounded discovery contracts; real Session -> native serialized requests."""
import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from app.agent.capability_discovery_actions import capability_arguments
from app.agent.kernel.capabilities import ToolCatalog
from app.agent.kernel.discovery import CapabilityDiscovery
from app.agent.kernel.pipeline import ToolPipelineError
from app.agent.kernel.provider_model import (
    OpenAICompatibleModelAdapter,
    ProviderSettings,
)
from tests.test_agent_kernel_capability_discovery import (
    capability_tool,
    request,
    session_for,
)
from tests.test_agent_kernel_core import collect, read_tool


def test_union_pending_cache_and_overflow():
    tools = [read_tool(f"sample.item{i}") for i in range(5)]
    finder = CapabilityDiscovery(ToolCatalog([capability_tool(), *tools]), context={}, maximum=2)
    finder.window([tools[4]])
    first = finder.search({"tool_names": [tools[0].name]})
    second = finder.search({"tool_names": [tools[1].name, tools[2].name]})
    assert first["loaded_tools"] == [tools[0].name]
    assert second["loaded_tools"] == [tools[0].name, tools[1].name]
    assert second["not_loaded_tools"] == [tools[2].name]
    assert second["unloaded_tools"] == [tools[4].name]
    spent = finder.requests
    repeat = finder.search({"tool_names": [tools[1].name, tools[2].name]})
    assert finder.requests == spent
    assert repeat["cache_hit"] is True
    assert [t.name for t in finder.window(finder._current, finder.consume())] == [
        "agent.capabilities", tools[0].name, tools[1].name,
    ]


def test_catalog_paging_filters_and_does_not_load():
    tools = [read_tool(f"sample.item{i}", domain="sample") for i in range(4)]
    hidden = replace(read_tool("sample.secret", domain="sample"), authorize=lambda c: False)
    finder = CapabilityDiscovery(ToolCatalog([*tools, hidden]), context={})
    result = finder.search(capability_arguments({"catalog": {"domains": ["sample"], "limit": 2}}))
    assert [t["name"] for t in result["tools"]] == [t.name for t in tools[:2]]
    assert result["next_offset"] == 2
    assert result["loaded_tools"] == []
    result = finder.search(capability_arguments({"catalog": {"domains": ["sample"], "offset": 2, "limit": 2}}))
    assert [t["name"] for t in result["tools"]] == [t.name for t in tools[2:]]
    assert result["next_offset"] is None
    assert "secret" not in json.dumps(result)
    assert not finder.consume()


def test_later_failed_serialization_preserves_only_prior_success():
    from app.agent.kernel.projection import DefaultProjector
    from tests.test_agent_kernel_capability_discovery import call, final

    class FailSecond(DefaultProjector):
        def project(self, value):
            if hasattr(value, "data") and "metadata.bar" in value.data.get("loaded_tools", []):
                raise ToolPipelineError("serialization failed", code="invalid_tool_result")
            return super().project(value)

    session, model = session_for(
        [read_tool("metadata.foo"), read_tool("metadata.bar")],
        [[call("agent.capabilities", {"tool_names": ["metadata.foo"]}, "one"),
          call("agent.capabilities", {"tool_names": ["metadata.bar"]}, "two")], final()],
        projector=FailSecond(),
    )
    asyncio.run(collect(session.run(request())))
    names = {t["name"] for t in model.requests[1].tools}
    assert "metadata__foo" in names
    assert "metadata__bar" not in names


def test_finite_distinct_discovery_budget_and_cached_replay():
    finder = CapabilityDiscovery(ToolCatalog([read_tool("sample.inspect")]), context={})
    for index in range(8):
        finder.search({"query": f"unmatched-{index}"})
    with pytest.raises(ToolPipelineError, match="8次"):
        finder.search({"query": "ninth-distinct"})
    assert finder.search({"query": "unmatched-0"})["cache_hit"]
    assert finder.requests == 8


def test_existing_real_colloquial_corpus_directory_recovery():
    from app.agent.domain_catalog import build_tool_specs
    from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs

    cases = [json.loads(line) for line in (Path(__file__).parent / "fixtures/agent_kernel_capability_cases.jsonl").read_text().splitlines()]
    case = next(c for c in cases if c["id"] == "rss-summary-natural-variant")
    assert case["message"] == "我订阅了那些RSS？"
    catalog = catalog_from_tool_specs(build_tool_specs())
    finder = CapabilityDiscovery(catalog, context={})
    lexical = finder.search({"query": case["message"]})
    # No claim that today's lexical result fails: exercise the independent escape hatch.
    page = finder.search(capability_arguments({"catalog": {"domains": ["rss"]}}))
    entries = list(page["tools"])
    if page["next_offset"] is not None:
        page = finder.search(capability_arguments({"catalog": {"domains": ["rss"], "offset": page["next_offset"]}}))
        entries.extend(page["tools"])
    assert case["required"][0] in [t["name"] for t in entries]
    assert all("parameters" not in t for t in entries)
    loaded = finder.search({"tool_names": case["required"]})
    window = finder.window((), finder.consume())
    assert set(loaded["loaded_tools"]) <= {t.name for t in window}
    assert "rss.subscription_summaries" in loaded["loaded_tools"]
    assert lexical["discovery_cost"]["distinct_requests"] == 1


def test_union_reaches_next_native_request_and_real_handlers():
    asyncio.run(_native_union())


async def _native_union():
    seen, executed = [], []
    targets = [read_tool("alpha.inspect", description="alpha inspection", handler=lambda a,c: executed.append("alpha") or {"summary":"alpha"}),
               read_tool("beta.inspect", description="beta inspection", handler=lambda a,c: executed.append("beta") or {"summary":"beta"})]
    secret = replace(read_tool("secret.inspect"), authorize=lambda c: False)
    session, _ = session_for([*targets, read_tool("overflow.inspect"), secret], [])
    session.retriever.maximum = 2
    discovery_outcomes = []
    execute = session.pipeline.execute

    async def record_execute(name, *args, **kwargs):
        result = await execute(name, *args, **kwargs)
        if name == "agent.capabilities":
            discovery_outcomes.append(result.outcome)
        return result

    session.pipeline.execute = record_execute

    class Response:
        status_code = 200
        def __init__(self, body):
            self.body = body
            self.headers = {"content-type": "text/event-stream"}
        async def aiter_bytes(self):
            yield ("data: " + json.dumps(self.body) + "\n\ndata: [DONE]\n\n").encode()

    class Client:
        @asynccontextmanager
        async def stream_post_json(self, url, **kwargs):
            seen.append(kwargs["json"])
            index = len(seen)
            if index == 1:
                calls = [("agent__capabilities", {"queries": ["alpha inspection", "beta inspection"]}),
                         ("agent__capabilities", {"tool_names": ["overflow.inspect", "secret.inspect"]})]
            elif index == 2:
                calls = [("alpha__inspect", {}), ("beta__inspect", {})]
            else:
                calls = []
            delta = {"tool_calls": [{"index":i,"id":f"c{index}-{i}","type":"function","function":{"name":name,"arguments":json.dumps(args)}} for i,(name,args) in enumerate(calls)]} if calls else {"content":"已完成两项检查。"}
            yield Response({"choices":[{"delta":delta,"finish_reason":"tool_calls" if calls else "stop"}]})
        async def aclose(self): pass

    session.model = OpenAICompatibleModelAdapter(
        ProviderSettings(api_url="https://offline.invalid/v1", model="fixture", protocol="chat_completions"),
        client_factory=lambda **_: Client(),
    )
    events = await collect(session.run(request()))
    names = [t["function"]["name"] for t in seen[1]["tools"]]
    assert len(names) == 4
    assert set(names) == {"agent__capabilities", "agent__read_result", "alpha__inspect", "beta__inspect"}
    assert "agent__read_result" not in {t["function"]["name"] for t in seen[0]["tools"]}
    reader_schema = next(t["function"] for t in seen[1]["tools"] if t["function"]["name"] == "agent__read_result")
    assert reader_schema == session.catalog.get("agent.read_result").model_definition()
    # The additional utility comes from real discovery outcomes, not a permissive window.
    assert len(discovery_outcomes) == 2
    wire_results = [json.loads(m["content"]) for m in seen[1]["messages"] if m["role"] == "tool"]
    assert len(wire_results) == 2
    for outcome, wire_result in zip(discovery_outcomes, wire_results, strict=True):
        handle = outcome.public_content["result_handle"]
        assert handle and wire_result["result_handle"] == handle
        assert wire_result["read_tool"] == "agent.read_result"
        store = session.pipeline.reference_store
        manifest = await store.resolve(handle, owner="owner-1", session_id="session-1", expected_kind="tool_result")
        chunks = [await store.resolve(part, owner="owner-1", session_id="session-1", expected_kind="tool_result_part") for part in manifest["parts"]]
        snapshot = json.loads("".join(chunks))
        assert snapshot["data"]["loaded_tools"] == wire_result["data"]["loaded_tools"]
        assert snapshot["data"]["loaded_tools"] == ["alpha.inspect", "beta.inspect"]
    assert events[-1].payload["status"] == "success"
    final_results = [json.loads(m["content"]) for m in seen[2]["messages"] if m.get("tool_call_id") in {"c2-0", "c2-1"}]
    assert [result["summary"] for result in final_results] == ["alpha", "beta"]
    assert executed == ["alpha", "beta"]
    assert len(seen) == 3
    first_schema = next(t["function"]["parameters"] for t in seen[0]["tools"] if t["function"]["name"] == "agent__capabilities")
    assert first_schema["properties"]["queries"]["maxItems"] == 4
    assert first_schema["properties"]["catalog"]["properties"]["limit"]["maximum"] == 12
