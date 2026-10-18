"""Bounded native discovery admission; scripted model, not LLM adoption."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from app.agent.kernel.bootstrap import build_agent_kernel_runtime
from app.agent.kernel.capabilities import CapabilityRetriever, ToolCatalog
from app.agent.kernel.discovery import CapabilityDiscovery
from app.agent.kernel.state_types import AgentInput
from tests.test_agent_kernel_capability_discovery import capability_tool
from tests.test_agent_kernel_core import read_tool
from tests.test_agent_task_entry import (
    GOAL,
    proposal,
    store,
)
from tests.test_agent_task_entry import (
    isolated_db as isolated_db,  # noqa: PLC0414 - pytest共享fixture需要显式导出
)
from tests.test_agent_task_first_slice import OWNER, NativeScript


class WindowScript(NativeScript):
    def __init__(self, has_task, requested):
        super().__init__("tmdb")
        self.requested = requested
        self.steps = (["find_task", "create"] if has_task else []) + [
            "find_config",
            "finish",
        ]
        self.discovery_results = []

    async def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        for message in body["messages"]:
            if message.get("role") == "tool":
                value = json.loads(message["content"])
                data = value.get("data", value)
                if "loaded_tools" in data:
                    self.discovery_results.append(data)
        step = self.steps.pop(0)
        if step == "finish":
            message, reason = {"content": "已完成发现，未执行业务写操作。"}, "stop"
        else:
            if step == "create":
                name, args = "agent__task", proposal()
            else:
                name = "agent__capabilities"
                args = {
                    "tool_names": ["agent.task"]
                    if step == "find_task"
                    else self.requested
                }
            message = {
                "content": "",
                "tool_calls": [
                    {
                        "id": "window-" + step,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                ],
            }
            reason = "tool_calls"
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": reason}]}
        )


@pytest.mark.parametrize("has_task", [False, True])
@pytest.mark.parametrize("overflow", [False, True])
def test_native_small_window_loaded_is_in_next_request(
    isolated_db, tmp_path, has_task, overflow
):
    from app.agent.kernel.session import SessionLimits
    from app.agent.model_context_budget import estimated_tokens

    requested = ["config.explain_component"]
    if overflow:
        requested += ["config.feature_summary", "config.diagnose"]
    script = WindowScript(has_task, requested)
    runtime = build_agent_kernel_runtime(model=script.adapter, store=store())
    runtime.session.retriever = CapabilityRetriever(minimum=1, maximum=2)
    runtime.session.limits = SessionLimits(
        context_window_tokens=16_384, capability_window=2
    )

    async def run():
        return [
            e
            async for e in runtime.session.run(
                AgentInput(
                    message=GOAL,
                    owner=OWNER,
                    session_id="window-native",
                    reply_context={},
                )
            )
        ]

    events = asyncio.run(run())
    (tmp_path / "native.json").write_text(
        json.dumps(
            {
                "requests": script.requests,
                "model_requests": script.model_requests,
                "results": script.discovery_results,
                "events": [
                    {"type": e.type.value, "payload": dict(e.payload)} for e in events
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    assert script.discovery_results, "real discovery must execute"
    result = script.discovery_results[-1]
    names = [t["function"]["name"] for t in script.requests[-1]["tools"]]
    assert result["loaded_tools"] == requested[:2]
    assert {n.replace(".", "__") for n in result["loaded_tools"]} <= set(names)
    assert "config__explain_component" in names
    assert result["not_loaded_tools"] == requested[2:]
    previous = {t["function"]["name"] for t in script.requests[-2]["tools"]}
    assert {n.replace(".", "__") for n in result["unloaded_tools"]} == previous - set(
        names
    )
    assert "agent__task" in names  # 与查询相关即可见，不再要求先创建任务。
    assert bool(events[-1].payload.get("task")) is has_task  # 可见本身不能建立任务。
    assert (
        len(
            [
                n
                for n in names
                if n not in {"agent__task", "agent__capabilities", "agent__read_result"}
            ]
        )
        <= 2
    )
    assert len(names) <= 5
    for request in script.model_requests:
        cost = estimated_tokens(request["system_prompt"]) + estimated_tokens(
            request["tools"]
        )
        cost += sum(8 + estimated_tokens(message) for message in request["messages"])
        assert cost + request["max_output_tokens"] + 512 <= 16_384
    assert not [e for e in events if e.type.value == "tool.failed"]


@pytest.mark.parametrize("has_task", [False, True])
def test_business_slots_retention_overflow_and_truth(has_task):
    from app.agent.kernel.task_state import task_tool

    old = read_tool("old.read")
    one, two, three = [read_tool(n) for n in ("one.read", "two.read", "three.read")]
    reader = read_tool("agent.read_result")
    finder = CapabilityDiscovery(
        ToolCatalog([capability_tool(), task_tool(), reader, old, one, two, three]),
        context={"has_task": has_task, "reference_kinds": ["tool_result"]},
        maximum=2,
    )
    initial = finder.window([old])
    first = finder.search({"tool_names": [one.name]})
    after = finder.window(initial, finder.consume())
    assert {old.name, one.name} <= {t.name for t in after}, (
        "utility must not displace business"
    )
    assert first["unloaded_tools"] == []
    second = finder.search({"tool_names": [one.name, two.name, three.name]})
    final = finder.window(after, finder.consume())
    names = {t.name for t in final}
    assert second["loaded_tools"] == [one.name, two.name]
    assert second["not_loaded_tools"] == [three.name]
    assert second["unloaded_tools"] == [old.name]
    assert set(second["loaded_tools"]) <= names
    assert "agent.read_result" in names
    assert len(names) == 4 + int(has_task)
    assert second["window_budget"]["business_count"] == 2


def test_related_and_authorize_filter_preserve_real_selection():
    target = read_tool("metadata.find", description="quasar lookup")
    neighbor = read_tool("metadata.follow", description="unrelated continuation")
    denied = replace(
        read_tool("secret.read", description="quasar lookup"), authorize=lambda c: False
    )
    target = replace(target, metadata={"related_tools": [neighbor.name, denied.name]})
    finder = CapabilityDiscovery(
        ToolCatalog([capability_tool(), target, neighbor, denied]),
        context={},
        maximum=2,
    )
    initial = finder.window([])
    result = finder.search({"query": "quasar"})
    final = finder.window(initial, finder.consume())
    assert set(result["loaded_tools"]) == {target.name, neighbor.name}
    assert {target.name, neighbor.name} <= {t.name for t in final}
    assert denied.name not in json.dumps(result)


def test_failed_discovery_restore_does_not_change_window():
    a, b = read_tool("one.read"), read_tool("two.read")
    finder = CapabilityDiscovery(
        ToolCatalog([capability_tool(), a, b]), context={}, maximum=2
    )
    before = finder.window([a])
    checkpoint = finder.checkpoint()
    finder.search({"tool_names": [b.name]})
    finder.restore(checkpoint)
    assert finder.window(before, finder.consume()) == before


@pytest.mark.parametrize(
    "reference_kinds,authorized,visible",
    [
        ([], True, False),
        (["tool_result"], True, True),
        (["telegram_reference"], True, True),
        (["tool_result"], False, False),
    ],
)
def test_reader_visibility_is_not_granted_by_free_utility_slot(
    reference_kinds, authorized, visible
):
    reader = replace(read_tool("agent.read_result"), authorize=lambda c: authorized)
    finder = CapabilityDiscovery(
        ToolCatalog([capability_tool(), reader]),
        context={"reference_kinds": reference_kinds},
        maximum=2,
    )
    result = finder.search({"tool_names": [reader.name]})
    window = finder.window([], finder.consume())
    assert (reader.name in {t.name for t in window}) is visible
    assert bool(result["loaded_tools"]) is visible


@pytest.mark.parametrize("maximum", [2, 10, 12])
def test_window_has_hard_business_and_utility_bounds(maximum):
    from app.agent.kernel.task_state import task_tool

    business = [read_tool(f"business.tool{i}") for i in range(20)]
    utility = [capability_tool(), task_tool(), read_tool("agent.read_result")]
    finder = CapabilityDiscovery(
        ToolCatalog([*business, *utility]),
        context={"has_task": True, "reference_kinds": ["tool_result"]},
        maximum=maximum,
    )
    assert len(finder.window(business)) == maximum + 3


@pytest.mark.parametrize("opted_in", [False, True])
def test_unmatched_kernel_utility_cannot_enter_via_workflow_neighbor(opted_in):
    utility = replace(read_tool("agent.utility", description="unrelated internal helper"),
                      metadata={"kernel_utility": True, "query_discoverable": opted_in})
    target = replace(read_tool("metadata.find", description="quasar lookup"),
                     metadata={"related_tools": [utility.name]})
    finder = CapabilityDiscovery(ToolCatalog([capability_tool(), target, utility]), context={})
    result = finder.search({"query": "quasar"})
    assert result["loaded_tools"] == [target.name]
    assert utility.name not in {tool["name"] for tool in result["tools"]}
    # 明确按名称查询仍可加载；这里修的是自动补位，不是取消工具权限。
    explicit = finder.search({"tool_names": [utility.name]})
    assert utility.name in explicit["loaded_tools"]


@pytest.mark.parametrize("question", [
    "我收藏的剧集有更新吗？把收藏里的剧集都核对一下，只看媒体库和已播进度，先不找资源、不下载。",
    "我收藏的剧集，帮我核对有无更新，不找下载资源",
    "我的收藏片单都有什么？",
    "收藏编号1043是哪部？",
])
def test_watchlist_entry_is_visible_for_collection_and_known_number_queries(question):
    from app.agent.domain_catalog import build_tool_specs
    from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs

    catalog = catalog_from_tool_specs(build_tool_specs())
    selected = CapabilityRetriever().retrieve(question, catalog)
    assert "discovery.watchlist_summaries" in [tool.name for tool in selected.tools]
    assert len(selected.tools) <= 10
