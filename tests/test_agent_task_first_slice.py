"""Explicit task lifecycle through real native wire, domain tools and SQLite."""

import asyncio
import json
import socket
from dataclasses import asdict
from unittest.mock import patch
from starlette.requests import Request

import httpx
import pytest
from dataclasses import replace
from contextlib import contextmanager

from app import database as db
from app.agent.owner_routes import web_kernel_owner
from app.agent.kernel.bootstrap import build_agent_kernel_runtime
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.provider_model import (
    OpenAICompatibleModelAdapter,
    ProviderSettings,
)
from app.agent.kernel.state import AgentInput, retain_conversation
from app.agent.kernel.adapters import TurnViewBuilder
from app.indexers.http import FixedHostHttpClient
from app.routes import agent_api

GOAL = "观察 TMDB 和 qBittorrent 当前配置状态，仅观察、不修改。"
OWNER = web_kernel_owner("admin")


def declaration():
    return {
        "operation": "create",
        "expected": None,
        "source": GOAL,
        "root_goal": GOAL,
        "constraints": ["仅观察、不修改"],
        "obligations": [
            {
                "id": component,
                "description": "观察配置 " + component,
                "required": True,
                "acceptance": {"kind": "observed_component", "component": component},
            }
            for component in ("tmdb", "qbittorrent")
        ],
    }


class NativeScript:
    def __init__(self, component, create=False):
        self.component = component
        self.steps = (
            ["discover"]
            + (["create"] if create else [])
            + ["observe", "checkpoint", "finish"]
        )
        self.requests = []
        self.model_requests = []
        self.latest = None
        self.force_final_tool = False
        script = self

        class Adapter(OpenAICompatibleModelAdapter):
            async def stream(self, request, *, cancellation):
                script.model_requests.append(asdict(request))
                async for event in super().stream(request, cancellation=cancellation):
                    yield event

        self.adapter = Adapter(
            ProviderSettings(
                api_url="https://api.example.com/v1",
                model="fixture",
                protocol="chat_completions",
            ),
            client_factory=lambda **kw: FixedHostHttpClient(
                **kw,
                transport=httpx.MockTransport(self.respond),
                resolver=lambda host, port: [
                    (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
                ],
            ),
        )

    async def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        for message in body["messages"]:
            if message.get("role") == "tool":
                try:
                    value = json.loads(message["content"])
                except (ValueError, TypeError):
                    continue
                if value.get("task"):
                    self.latest = value["task"]
        step = self.steps.pop(0)
        if step == "finish" and not self.force_final_tool:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "全部任务完成。"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        if step == "discover":
            name, args = (
                "agent__capabilities",
                {"tool_names": ["agent.task", "config.explain_component"]},
            )
        elif step == "create":
            name, args = "agent__task", declaration()
        elif step == "observe":
            name, args = "config__explain_component", {"component": self.component}
        else:
            task = self.latest or {}
            name, args = (
                "agent__task",
                {
                    "operation": "checkpoint",
                    "expected": {
                        k: task.get(k) for k in ("task_id", "epoch", "revision")
                    },
                    "evidence_ids": [
                        f["evidence_id"]
                        for f in task.get("verified_facts", [])
                        if f["component"] == self.component
                    ],
                    "blockers": [],
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-" + step,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(
                                            args, ensure_ascii=False
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )


@pytest.mark.parametrize("budget", [False, True])
def test_observe_checkpoint_prune_reopen_continue(tmp_path, budget):
    previous = db.DB_PATH, bool(getattr(db, "_configured_test_mode", False))
    db.configure_database(tmp_path / "task.db", test_mode=True)
    db.init_db()

    async def run():
        store = SQLiteKernelStore(secret_provider=lambda: "task-test-only")
        first = NativeScript("tmdb", create=True)
        runtime = build_agent_kernel_runtime(model=first.adapter, store=store)
        events = [
            e
            async for e in runtime.session.run(
                AgentInput(message=GOAL, owner=OWNER, session_id="task-session-0001")
            )
        ]
        (tmp_path / "first-events.json").write_text(
            json.dumps(
                [{"type": e.type.value, "payload": dict(e.payload)} for e in events],
                ensure_ascii=False,
                indent=2,
            )
        )
        discoveries = [
            e.payload.get("result", {})
            for e in events
            if e.payload.get("tool") == "agent.capabilities"
        ]
        assert any(
            "agent.task" in json.dumps(result)
            and "agent.task" in result.get("data", result).get("loaded_tools", [])
            for result in discoveries
        ), (
            "task control must be discoverable through the real capability tool: "
            + repr([(e.type, dict(e.payload)) for e in events])
        )
        cold = await store.load(owner=OWNER, session_id="task-session-0001")
        assert cold.metadata["task"]["status"] == "partial"
        assert events[-1].payload["task"]["status"] == "partial"
        assert cold.metadata["task"]["obligations"][0]["state"] == "verified"
        retained = retain_conversation(
            cold.conversation + [{"role": "user", "content": "保留的新回合"}],
            maximum_bytes=100,
        )
        assert GOAL not in json.dumps(retained, ensure_ascii=False)
        lease, _ = await store.begin_turn(
            owner=OWNER, session_id="task-session-0001", request_id="prune"
        )
        await store.commit(lease, conversation=retained)
        fresh = SQLiteKernelStore(secret_provider=lambda: "task-test-only")
        second = NativeScript("qbittorrent")
        runtime2 = build_agent_kernel_runtime(model=second.adapter, store=fresh)
        if budget:
            second.force_final_tool = True
            runtime2.session.limits = replace(runtime2.session.limits, max_tool_calls=3)
        events2 = [
            e
            async for e in runtime2.session.run(
                AgentInput(message="继续", owner=OWNER, session_id="task-session-0001")
            )
        ]
        for projection in (second.model_requests[0], second.requests[0]):
            text = json.dumps(projection, ensure_ascii=False)
            assert GOAL in text and "仅观察、不修改" in text
            assert "verified_facts" in text and "tmdb" in text and "qbittorrent" in text
        assert (
            sum(
                e.payload.get("tool") == "config.explain_component"
                for e in events2
                if e.type.value == "tool.started"
            )
            == 1
        )
        end = await SQLiteKernelStore(secret_provider=lambda: "task-test-only").load(
            owner=OWNER, session_id="task-session-0001"
        )
        assert end.metadata["task"]["status"] == "complete"
        assert events2[-1].payload["turn_status"] == events2[-1].payload["status"]
        assert events2[-1].payload["task"]["status"] == "complete"
        assert events2[-1].payload["turn_status"] == (
            "partial" if budget else "success"
        )
        builder = TurnViewBuilder()
        for event in events2:
            builder.apply(event)
        assert builder.build().to_dict()["task"] == events2[-1].payload["task"]
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/api/agent/sessions/task-session-0001",
                "headers": [],
                "session": {"logged_in": True},
            }
        )
        with patch.object(agent_api, "get_agent_kernel_runtime", return_value=runtime2):
            response = await agent_api.get_session(request, "task-session-0001")
        assert response.status_code == 200
        api = json.loads(response.body)
        assert api["task"] == events2[-1].payload["task"]
        assert api["turn_status"] == events2[-1].payload["turn_status"]
        (tmp_path / "vertical.json").write_text(
            json.dumps(
                {
                    "first_requests": first.model_requests,
                    "first_wire": first.requests,
                    "first_terminal": dict(events[-1].payload),
                    "retained": retained,
                    "second_requests": second.model_requests,
                    "second_wire": second.requests,
                    "second_events": [
                        {"type": e.type.value, "payload": dict(e.payload)}
                        for e in events2
                    ],
                    "cold_state": end.to_payload(),
                    "api": api,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    try:
        asyncio.run(run())
    finally:
        db.configure_database(previous[0], test_mode=previous[1])


def test_no_task_qa_transaction_trace(tmp_path):
    previous = db.DB_PATH, bool(getattr(db, "_configured_test_mode", False))
    db.configure_database(tmp_path / "qa.db", test_mode=True)
    db.init_db()
    trace = []
    original = db.get_conn

    @contextmanager
    def traced():
        with original() as connection:
            connection.set_trace_callback(trace.append)
            yield connection

    async def run():
        model = NativeScript("tmdb")
        model.steps = ["finish"]
        store = SQLiteKernelStore(secret_provider=lambda: "task-test-only")
        runtime = build_agent_kernel_runtime(model=model.adapter, store=store)
        with patch.object(db, "get_conn", traced):
            events = [
                e
                async for e in runtime.session.run(
                    AgentInput(
                        message="你好", owner=OWNER, session_id="qa-session-000001"
                    )
                )
            ]
        assert len(model.requests) == 1
        assert not any(e.type.value == "tool.started" for e in events)
        assert events[-1].payload["status"] == "success"
        assert events[-1].payload.get("task") is None
        assert (
            "task"
            not in (
                await store.load(owner=OWNER, session_id="qa-session-000001")
            ).metadata
        )
        (tmp_path / "qa-trace.json").write_text(
            json.dumps(
                {
                    "transactions": [
                        s
                        for s in trace
                        if s.startswith(("BEGIN", "COMMIT", "ROLLBACK"))
                    ],
                    "model_calls": len(model.requests),
                    "tools": 0,
                },
                indent=2,
            )
        )

    try:
        asyncio.run(run())
    finally:
        db.configure_database(previous[0], test_mode=previous[1])


@pytest.mark.parametrize("failure", ["invalid", "exception", "commit"])
def test_native_failed_observation_is_atomic_and_object_scoped(tmp_path, failure):
    from app.agent import config_explain_actions as domain

    previous = db.DB_PATH, bool(getattr(db, "_configured_test_mode", False))
    db.configure_database(tmp_path / "failure.db", test_mode=True)
    db.init_db()
    original = domain.component_configuration

    class FailingStore(SQLiteKernelStore):
        async def commit(
            self, lease, *, conversation=None, updates=(), task_patch=None
        ):
            if (
                failure == "commit"
                and task_patch is not None
                and task_patch.operation == "observe"
            ):
                raise ValueError("injected completion capacity failure")
            return await super().commit(
                lease,
                conversation=conversation,
                updates=updates,
                **({"task_patch": task_patch} if task_patch is not None else {}),
            )

    def broken(component):
        if component == "tmdb":
            if failure == "exception":
                raise RuntimeError("synthetic query failure")
            if failure == "invalid":
                return {**original(component), "network_accessed": True}
        return original(component)

    async def turn(store, component, create=False, break_query=False):
        script = NativeScript(component, create)
        runtime = build_agent_kernel_runtime(model=script.adapter, store=store)
        events = []
        with patch.object(
            domain, "component_configuration", broken if break_query else original
        ):
            async for event in runtime.session.run(
                AgentInput(
                    message=GOAL if create else "继续",
                    owner=OWNER,
                    session_id="failure-session-01",
                )
            ):
                events.append(event)
                if (
                    event.type.value == "tool.failed"
                    and event.payload.get("tool") == "config.explain_component"
                ):
                    cold = await store.load(
                        owner=OWNER, session_id="failure-session-01"
                    )
                    assert not cold.metadata["task"]["verified_facts"]
                    assert any(
                        row.get("role") == "tool"
                        and row.get("tool_call_id") == "call-observe"
                        for row in cold.conversation
                    )
        return events, await store.load(owner=OWNER, session_id="failure-session-01")

    async def run():
        store = FailingStore(secret_provider=lambda: "task-test-only")
        events, first = await turn(store, "tmdb", True, True)
        assert not first.metadata["task"]["verified_facts"]
        assert not any(
            e.type.value == "tool.completed"
            and e.payload.get("tool") == "config.explain_component"
            for e in events
        )
        assert first.metadata["task"]["status"] != "complete"
        if failure == "commit":
            assert "observed_at" not in json.dumps(first.conversation)
            return
        query_blocker = {
            "origin": "query", "component": "tmdb", "epoch": 1, "text": "query:tmdb"
        }
        assert first.metadata["task"]["blockers"] == [query_blocker]
        assert events[-1].payload["task"]["status"] == "blocked"
        _, other = await turn(store, "qbittorrent")
        assert other.metadata["task"]["blockers"] == [query_blocker]
        _, recovered = await turn(store, "tmdb")
        assert recovered.metadata["task"]["blockers"] == []
        assert recovered.metadata["task"]["status"] == "complete"

    try:
        asyncio.run(run())
    finally:
        db.configure_database(previous[0], test_mode=previous[1])
