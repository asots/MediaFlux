"""真实能力发现/原生模型协议/SQLite：工具目录不等于业务观察。"""

import asyncio
import json

import httpx

from app import database as db
from app.agent.kernel.bootstrap import build_agent_kernel_runtime
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.state_types import AgentInput
from tests.test_agent_read_wait_lifecycle import isolated_database as isolated_database
from tests.test_agent_task_first_slice import OWNER, NativeScript


class DiscoveryScript(NativeScript):
    def __init__(self):
        super().__init__("tmdb")

    async def respond(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        index = len(self.requests)
        if not payload.get("tools") or index > 14:
            message = {"content": "仅浏览了工具目录，没有读取业务数据。"}
            finish = "stop"
        else:
            message = {"tool_calls": [{"id": f"discover-{index}", "type": "function", "function": {
                "name": "agent__capabilities", "arguments": json.dumps({"query": f"媒体目录查询能力 {index}"}),
            }}]}
            finish = "tool_calls"
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish}]})


def test_discovery_keeps_its_own_bound_without_shortening_model_round_limit():
    async def run():
        db.init_db()
        script = DiscoveryScript()
        runtime = build_agent_kernel_runtime(model=script.adapter, store=SQLiteKernelStore())
        events = [event async for event in runtime.session.run(AgentInput(
            owner=OWNER, session_id="discovery-budget", message="查看媒体目录查询能力",
        ))]
        assert len(script.requests) == 15
        assert script.requests[-1].get("tools")  # 模型自行完成，不再在第12轮被收走工具。
        assert events[-1].payload["successful_read_queries"] == 0
        started = [event for event in events if event.type is AgentEventType.TOOL_STARTED]
        completed = [event for event in events if event.type is AgentEventType.TOOL_COMPLETED]
        assert started and completed
        assert all(event.payload["tool"] == "agent.capabilities" for event in started + completed)
        assert all(event.payload["result"]["ok"] for event in completed)
        # 保留生产的 8 次不同发现上限；成功和超额拒绝都不是业务观察。
        failures = [event for event in events if event.type is AgentEventType.TOOL_FAILED]
        assert len(completed) == 8 and len(failures) == 6
        assert all(event.payload["code"] == "capability_discovery_budget" for event in failures)
        saved = await SQLiteKernelStore().load(owner=OWNER, session_id="discovery-budget")
        calls = [call for row in saved.conversation for call in row.get("tool_calls", ())]
        results = [row for row in saved.conversation if row.get("role") == "tool"]
        assert len(calls) == len(results) == 14
        assert {call["name"] for call in calls} == {"agent__capabilities"}
    asyncio.run(run())
