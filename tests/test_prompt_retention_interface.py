"""Composed retention/guidance characterization; real native wire, fixture READs."""
import asyncio
import json
from dataclasses import replace

import httpx

from app.agent.domain_catalog import build_tool_specs
from app.agent.kernel.capabilities import CapabilityRetriever, CapabilitySelection
from tests.test_agent_kernel_capability_discovery import request, session_for
from tests.test_agent_kernel_core import collect, read_tool
from tests.test_agent_task_first_slice import NativeScript


def test_native_retention_guidance_release_then_actual_eviction(tmp_path):
    executed = []
    registered = next(s for s in build_tool_specs() if s.name == "provider.query")
    assert registered.model_name == "mf_provider_query"
    provider = replace(read_tool(
        registered.name, domain="provider",
        handler=lambda a, c: executed.append(c.owner) or {
            "summary": "已读取隔离对象快照", "data": {"observation": len(executed)}},
    ), model_name=registered.model_name)
    old = read_tool("guangya.fs.query", domain="guangya")
    others = [read_tool(f"phase.item{i}", domain="phase") for i in range(10)]
    session, _ = session_for([provider, old, *others], [])

    class Initial(CapabilityRetriever):
        def retrieve(self, message, catalog, *, context=None):
            return CapabilitySelection((catalog.get(provider.name), catalog.get(old.name)), {})

    class Script(NativeScript):
        async def respond(self, outbound):
            body = json.loads(outbound.content)
            self.requests.append(body)
            index = len(self.requests)
            calls = {
                1: [(provider.model_name, {}), ("agent__capabilities", {"window": {"retain": [provider.name]}})],
                2: [("agent__capabilities", {"tool_names": [t.name for t in others[:6]]})],
                3: [("agent__capabilities", {"tool_names": [t.name for t in others[6:]]})],
                4: [(provider.model_name, {}), ("agent__capabilities", {"window": {"release": [provider.name]}})],
                5: [("agent__capabilities", {"tool_names": [t.name for t in others[:6]]})],
                6: [("agent__capabilities", {"tool_names": [t.name for t in others[6:]]})],
            }.get(index, [])
            message = {"tool_calls": [
                {"id": f"interface-{index}-{i}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}
                for i, (name, args) in enumerate(calls)
            ]} if calls else {"content": "已完成隔离只读复核，未执行写入。"}
            return httpx.Response(200, json={"choices": [{
                "message": message, "finish_reason": "tool_calls" if calls else "stop"}]})

    script = Script("fixture")
    session.retriever = Initial()
    session.model = script.adapter
    events = asyncio.run(collect(session.run(replace(
        request(), message="读取初始对象并保留能力，分两阶段发现十种能力，再次读取；释放保留后再次分阶段发现。"))))
    # Capture before assertions so an actual failure remains inspectable.
    (tmp_path / "wire.json").write_text(json.dumps(script.requests, ensure_ascii=False, indent=2))
    (tmp_path / "execution.json").write_text(json.dumps({
        "read_owners": executed, "events": [e.type.value for e in events]}, ensure_ascii=False, indent=2))
    assert len(script.requests) == 7
    assert session.retriever.maximum == 10
    assert executed == ["owner-1", "owner-1"]
    assert not [e for e in events if e.type.value == "tool.failed"]
    names = [{t["function"]["name"] for t in body["tools"]} for body in script.requests]
    systems = [body["messages"][0]["content"] for body in script.requests]
    rules = ("不能用全库数量代替", "Series/tv类型及Genres不等于媒体库归属", "[打开媒体库](原样 open_url)")
    assert old.model_name in names[0]
    assert "先列出或递归观察父目录" in systems[0]
    assert old.model_name not in names[3]
    assert "先列出或递归观察父目录" not in systems[3]
    for index in range(6):
        assert provider.model_name in names[index], index
        for rule in rules:
            assert rule in systems[index], (index, rule)
    assert names[3] == names[4], "release alone must not unload the selected window"
    assert provider.model_name not in names[6]
    assert names[6] - {"agent__capabilities", "agent__read_result", "agent__task"} == {
        t.model_name for t in others
    }
    for rule in rules:
        assert rule not in systems[6], rule
    results = [json.loads(m["content"]) for m in script.requests[-1]["messages"] if m["role"] == "tool"]
    data = [r.get("data", {}) for r in results]
    assert any(d.get("window_update") == "retain" and d.get("retained_tools") == [provider.name] for d in data)
    assert any(d.get("window_update") == "release" and d.get("retained_tools") == [] for d in data)
    assert any(provider.name in d.get("unloaded_tools", []) for d in data)
    for system in systems:
        assert "历史工具事实" in system and "owner/session" in system
        assert "每项为一条排期事件" not in system
