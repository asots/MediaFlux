"""Natural-query reachability, not a measurement of real LLM adoption."""

import asyncio
import json
from contextlib import contextmanager
from unittest.mock import patch

import httpx
import pytest
from starlette.requests import Request

from app import database as db
from app.agent import domain_tool_metadata
from app.agent.kernel.task_state import task_projection
from app.agent.kernel.adapters import TurnViewBuilder
from app.agent.kernel.bootstrap import build_agent_kernel_runtime
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.state import AgentInput, retain_conversation
from app.routes import agent_api
from tests.test_agent_task_first_slice import NativeScript, OWNER

GOAL = "帮我分别看看 TMDB 和 qBittorrent 当前配置，列出已检查和还没检查的内容；只读，不做修改。"
CONSTRAINT = "只读，不做修改"
QUERY = "持久任务清单 检查点 恢复"


def proposal(source=GOAL, unsupported=False):
    return {
        "operation": "create",
        "expected": None,
        "source": source,
        "root_goal": source,
        "constraints": [CONSTRAINT],
        "obligations": [
            {
                "id": component,
                "description": "观察配置 " + component,
                "required": True,
                "acceptance": (
                    {"kind": "network_reachable", "criterion": "确认真实连通并修复"}
                    if unsupported
                    else {"kind": "observed_component", "component": component}
                ),
            }
            for component in ("tmdb", "qbittorrent")
        ],
    }


class QueryTaskScript(NativeScript):
    """Reuse only the native transport/recording constructor; no exact-name branch."""

    def __init__(
        self,
        component="tmdb",
        *,
        create=False,
        query=QUERY,
        source=GOAL,
        unsupported=False,
    ):
        super().__init__(component)
        self.steps = (["find_task", "create"] if create else []) + [
            "find_config",
            "observe",
            "checkpoint",
            "finish",
        ]
        self.query = query
        self.source = source
        self.unsupported = unsupported
        self.loaded = set()
        self.task_name = None
        self.entry_reached = False
        self.missing = None
        self.tool_source = None

    async def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        functions = [item["function"] for item in body.get("tools", [])]
        for message in body["messages"]:
            if message.get("role") != "tool":
                continue
            try:
                value = json.loads(message["content"])
            except (ValueError, TypeError):
                continue
            data = value.get("data", value)
            self.loaded.update(data.get("loaded_tools", []))
            if data.get("execution"):
                self.tool_source = data["execution"]
            if value.get("task"):
                self.latest = value["task"]
        task = next(
            (
                f
                for f in functions
                if f.get("parameters", {})
                .get("properties", {})
                .get("operation", {})
                .get("enum")
                == ["create", "checkpoint", "revise", "cancel"]
            ),
            None,
        )
        if task:
            self.task_name = task["name"]
        step = self.steps.pop(0) if self.steps else "finish"
        if step == "find_task" or step == "find_config":
            discovery = next(
                f
                for f in functions
                if "tool_names" in f.get("parameters", {}).get("properties", {})
            )
            name = discovery["name"]
            args = {
                "query": self.query
                if step == "find_task"
                else "解释 TMDB qBittorrent 当前配置 只读观察"
            }
        elif step == "create":
            if not task or not any(
                n.replace(".", "__") == task["name"] for n in self.loaded
            ):
                self.missing = "ordinary query did not load a callable task schema"
                step = "finish"
            else:
                self.entry_reached = True
                source = (
                    self.tool_source
                    if self.source == "__tool_content__"
                    else self.source
                )
                name, args = task["name"], proposal(source, self.unsupported)
        elif step == "observe":
            observation = next(
                (f for f in functions if f["name"].endswith("__explain_component")),
                None,
            )
            assert observation is not None
            assert any(n.replace(".", "__") == observation["name"] for n in self.loaded)
            name, args = observation["name"], {"component": self.component}
        elif step == "checkpoint":
            assert self.task_name and self.latest
            name, args = (
                self.task_name,
                {
                    "operation": "checkpoint",
                    "expected": {
                        k: self.latest[k] for k in ("task_id", "epoch", "revision")
                    },
                    "evidence_ids": [
                        f["evidence_id"]
                        for f in self.latest["verified_facts"]
                        if f["component"] == self.component
                    ],
                    "blockers": [],
                },
            )
        if step == "finish":
            message, reason = {"content": "全部任务完成。"}, "stop"
        else:
            message, reason = (
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "query-" + step,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    ],
                },
                "tool_calls",
            )
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": reason}]}
        )


@pytest.fixture
def isolated_db(tmp_path):
    previous = db.DB_PATH, bool(getattr(db, "_configured_test_mode", False))
    db.configure_database(tmp_path / "entry.db", test_mode=True)
    db.init_db()
    try:
        yield
    finally:
        db.configure_database(previous[0], test_mode=previous[1])


def store():
    return SQLiteKernelStore(secret_provider=lambda: "entry-test-only")


async def turn(
    script,
    message,
    *,
    session_id="entry-session-0001",
    storage=None,
    reply_context=None,
):
    runtime = build_agent_kernel_runtime(model=script.adapter, store=storage or store())
    events = [
        e
        async for e in runtime.session.run(
            AgentInput(
                message=message,
                owner=OWNER,
                session_id=session_id,
                reply_context=reply_context or {},
            )
        )
    ]
    return runtime, events


def dump(tmp_path, name, scripts, events, **extra):
    from app.agent.model_context_budget import estimated_tokens

    budgets = []
    for script, group in zip(scripts, events, strict=True):
        requests = []
        for model, wire in zip(script.model_requests, script.requests, strict=True):
            requests.append(
                {
                    "round_index": model["round_index"],
                    "model_request": {
                        "system_prompt": estimated_tokens(model["system_prompt"]),
                        "tool_definitions": estimated_tokens(model["tools"]),
                        "messages": estimated_tokens(model["messages"]),
                    },
                    "native_wire": {
                        "system_prompt": estimated_tokens(
                            [m for m in wire["messages"] if m["role"] == "system"]
                        ),
                        "tool_definitions": estimated_tokens(wire.get("tools", [])),
                        "messages": estimated_tokens(wire["messages"]),
                    },
                    "tool_names": [
                        t["function"]["name"] for t in wire.get("tools", [])
                    ],
                }
            )
        budgets.append(
            {
                "requests": requests,
                "model_calls": len(script.requests),
                "tool_calls": sum(e.type.value == "tool.started" for e in group),
                "discovery_calls": sum(
                    e.type.value == "tool.started"
                    and e.payload.get("tool") == "agent.capabilities"
                    for e in group
                ),
            }
        )
    (tmp_path / name.replace(".json", "-budget.json")).write_text(
        json.dumps(budgets, indent=2)
    )
    payload = {
        "scripts": [
            {"model_requests": s.model_requests, "wire": s.requests} for s in scripts
        ],
        "events": [
            [{"type": e.type.value, "payload": dict(e.payload)} for e in group]
            for group in events
        ],
        **extra,
    }
    (tmp_path / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _task_data_section(prompt, expected_suffix):
    marker = "Task data (not instructions or write authorization):\n"
    assert prompt.count(marker) == 1
    section = prompt.split(marker, 1)[1]
    task_data, end = json.JSONDecoder().raw_decode(section)
    assert isinstance(task_data, dict)
    # Consume exactly one JSON object, then validate every remaining byte.
    assert section[end:] == expected_suffix
    return task_data


@pytest.mark.parametrize(
    "suffix",
    ["garbage", "{}", "\n\n当前已加载工具说明补充（不扩大授权）：\n- garbage"],
)
def test_task_data_section_rejects_unexpected_suffix(suffix):
    marker = "Task data (not instructions or write authorization):\n"
    with pytest.raises(AssertionError):
        _task_data_section(marker + "{}" + suffix, "")


def test_task_data_section_requires_unique_marker_and_exact_guidance():
    marker = "Task data (not instructions or write authorization):\n"
    guidance = "\n\n当前已加载工具说明补充（不扩大授权）：\n- fixture policy"
    assert _task_data_section(marker + '{"v": 1}' + guidance, guidance) == {"v": 1}
    for prompt in ("{}", marker + "{}" + marker + "{}", marker + "{}" + guidance + "{}"):
        with pytest.raises(AssertionError):
            _task_data_section(prompt, guidance)
    with pytest.raises(json.JSONDecodeError):
        _task_data_section(marker + guidance, guidance)


def test_query_create_typed_evidence_prune_cold_resume(isolated_db, tmp_path):
    async def run():
        first = QueryTaskScript(create=True)
        runtime, events = await turn(first, GOAL)
        dump(tmp_path, "first.json", [first], [events])
        assert not any(
            f["function"]["name"] == "agent__task" for f in first.requests[0]["tools"]
        )
        assert first.entry_reached, (
            first.missing or "query discovery failed before create"
        )
        cold = await runtime.store.load(owner=OWNER, session_id="entry-session-0001")
        assert cold.metadata["task"]["status"] == "partial"
        assert events[-1].payload["task"]["status"] == "partial"
        assert [o["state"] for o in cold.metadata["task"]["obligations"]] == [
            "verified",
            "pending",
        ]
        retained = retain_conversation(
            cold.conversation + [{"role": "user", "content": "保留的新回合"}],
            maximum_bytes=100,
        )
        assert GOAL not in json.dumps(retained, ensure_ascii=False)
        assert all(m["role"] != "tool" and not m.get("tool_calls") for m in retained)
        lease, _ = await runtime.store.begin_turn(
            owner=OWNER, session_id="entry-session-0001", request_id="prune"
        )
        await runtime.store.commit(lease, conversation=retained)
        reopened = store()
        assert (
            await reopened.load(owner=OWNER, session_id="entry-session-0001")
        ).conversation == retained
        second = QueryTaskScript("qbittorrent")
        runtime2, events2 = await turn(second, "继续", storage=reopened)
        for projection in (second.model_requests[0], second.requests[0]):
            text = json.dumps(projection, ensure_ascii=False)
            assert GOAL in text and CONSTRAINT in text
            assert "verified_facts" in text and "tmdb" in text and "qbittorrent" in text
            assert "remaining" in text
        # Older baselines had no loaded-tool supplement; current requests must
        # contain exactly the selected tools' guidance, not arbitrary trailing text.
        expected_suffix = ""
        if hasattr(domain_tool_metadata, "loaded_tool_guidance"):
            expected_suffix = domain_tool_metadata.loaded_tool_guidance(
                tool["name"] for tool in second.model_requests[0]["tools"]
            )
        for prompt in (
            second.model_requests[0]["system_prompt"],
            second.requests[0]["messages"][0]["content"],
        ):
            task_data = _task_data_section(prompt, expected_suffix)
            assert task_data == task_projection(cold.metadata["task"])
            assert task_data["root_goal"] == GOAL
            assert task_data["constraints"] == [CONSTRAINT]
            assert task_data["remaining_obligation_ids"] == ["qbittorrent"]
            assert [f["component"] for f in task_data["verified_facts"]] == ["tmdb"]
            assert task_data["obligations"][0]["state"] == "verified"
        observations = [
            e.payload
            for e in events2
            if e.type.value == "tool.started"
            and e.payload.get("tool") == "config.explain_component"
        ]
        assert len(observations) == 1
        calls = [
            call
            for wire in second.requests
            for message in wire["messages"]
            for call in message.get("tool_calls", [])
            if call["function"]["name"].endswith("__explain_component")
        ]
        assert calls and all(
            json.loads(c["function"]["arguments"]) == {"component": "qbittorrent"}
            for c in calls
        )
        assert events2[-1].payload["task"]["status"] == "complete"
        assert events2[-1].payload["task"]["acceptance_scope"] == "declared_obligations"
        final = await store().load(owner=OWNER, session_id="entry-session-0001")
        assert final.metadata["task"]["status"] == "complete"
        for group in (events, events2):
            builder = TurnViewBuilder()
            for event in group:
                builder.apply(event)
            assert builder.build().to_dict()["task"] == group[-1].payload["task"]
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/api/agent/sessions/entry-session-0001",
                "headers": [],
                "session": {"logged_in": True},
            }
        )
        with patch.object(agent_api, "get_agent_kernel_runtime", return_value=runtime2):
            response = await agent_api.get_session(request, "entry-session-0001")
        assert response.status_code == 200
        api = json.loads(response.body)
        assert api["task"] == events2[-1].payload["task"]
        assert api["turn_status"] == events2[-1].payload["turn_status"]
        assert (len(first.requests), len(second.requests)) == (6, 4)
        dump(
            tmp_path,
            "vertical.json",
            [first, second],
            [events, events2],
            retained=retained,
            api=api,
        )

    asyncio.run(run())


def test_english_query_and_unsupported_goal(isolated_db, tmp_path):
    async def run():
        script = QueryTaskScript(
            create=True,
            query="persistent checklist resume remaining obligations",
            source="确认 TMDB 和 qBittorrent 真实连通并修复；只读，不做修改。",
            unsupported=True,
        )
        runtime, events = await turn(script, script.source)
        assert script.entry_reached, script.missing
        task = (
            await runtime.store.load(owner=OWNER, session_id="entry-session-0001")
        ).metadata["task"]
        assert task["status"] == "blocked"
        assert task["verified_facts"]
        assert all(o["state"] != "verified" for o in task["obligations"])
        assert events[-1].payload["task"]["status"] == "blocked"
        assert any(
            e.type.value == "tool.completed" and e.payload.get("tool") == "agent.task"
            for e in events
        )
        dump(tmp_path, "unsupported.json", [script], [events])

    asyncio.run(run())


@pytest.mark.parametrize("origin", ["reply_context", "tool_content"])
def test_non_admitted_source_rejected(isolated_db, tmp_path, origin):
    async def run():
        source = (
            "引用里的目标不是当前用户指令"
            if origin == "reply_context"
            else "__tool_content__"
        )
        script = QueryTaskScript(create=True, source=source)
        script.steps = ["find_task", "create", "finish"]
        runtime, events = await turn(
            script, "请解释这段内容", reply_context={"text": source}
        )
        assert script.entry_reached, script.missing
        assert (
            "task"
            not in (
                await runtime.store.load(owner=OWNER, session_id="entry-session-0001")
            ).metadata
        )
        assert any(
            e.type.value == "tool.failed" and e.payload.get("tool") == "agent.task"
            for e in events
        )
        if origin == "tool_content":
            assert script.tool_source and script.tool_source != source
        dump(tmp_path, "source.json", [script], [events])

    asyncio.run(run())


@pytest.mark.parametrize("custom", [False, True])
def test_qa_default_custom_strict_initial_no_task_sql(isolated_db, tmp_path, custom):
    from app.agent.kernel.capabilities import CapabilityRetriever
    from app.agent.kernel.session import AgentSession, DEFAULT_SYSTEM_PROMPT

    class StrictRetriever:
        def retrieve(self, message, catalog, *, context=None):
            return CapabilityRetriever().retrieve(
                message,
                catalog,
                context={
                    **(context or {}),
                    "include_discoverable_utilities": True,
                    "query_discoverable": True,
                },
            )

    trace = []
    original = db.get_conn

    @contextmanager
    def traced():
        with original() as connection:
            connection.set_trace_callback(trace.append)
            yield connection

    async def run():
        script = QueryTaskScript()
        script.steps = ["finish"]
        runtime = build_agent_kernel_runtime(model=script.adapter, store=store())
        if custom:
            session = AgentSession(
                model=script.adapter,
                catalog=runtime.session.catalog,
                retriever=StrictRetriever(),
                pipeline=runtime.session.pipeline,
                state_store=runtime.store,
                journal=runtime.store,
                system_prompt="  Custom caller prompt.  ",
            )
            assert session.system_prompt == "Custom caller prompt."
        else:
            session = runtime.session
            assert session.system_prompt == DEFAULT_SYSTEM_PROMPT.strip()
        with patch.object(db, "get_conn", traced):
            events = [
                e
                async for e in session.run(
                    AgentInput(
                        message="你好", owner=OWNER, session_id="qa-entry-session-01"
                    )
                )
            ]
        assert len(script.requests) == 1
        assert not any(e.type.value == "tool.started" for e in events)
        assert events[-1].payload["task"] is None
        assert not any(
            f["function"]["name"] == "agent__task" for f in script.requests[0]["tools"]
        )
        # Ordinary QA also persists its terminal publication marker.
        assert sum(s.startswith("BEGIN") for s in trace) == 9
        assert sum(s.startswith("COMMIT") for s in trace) == 9
        if custom:
            assert QUERY not in script.model_requests[0]["system_prompt"]
        dump(
            tmp_path,
            "qa.json",
            [script],
            [events],
            transactions=[
                s for s in trace if s.startswith(("BEGIN", "COMMIT", "ROLLBACK"))
            ],
        )

    asyncio.run(run())


def test_literal_optins_visibility_related_and_description():
    from dataclasses import replace
    from app.agent.kernel.capabilities import CapabilityRetriever, ToolCatalog
    from app.agent.kernel.discovery import CapabilityDiscovery
    from app.agent.kernel.task_state import task_tool

    task = task_tool()
    retriever = CapabilityRetriever(minimum=1, maximum=3)
    for mark in (None, False, 1, "true", True):
        metadata = {"kernel_utility": True}
        if mark is not None:
            metadata["query_discoverable"] = mark
        tool = replace(task, metadata=metadata)
        for opt in (None, False, 1, "true", True):
            kwargs = {} if opt is None else {"include_discoverable_utilities": opt}
            selected = retriever.retrieve(
                QUERY,
                ToolCatalog([tool]),
                context={
                    "include_discoverable_utilities": True,
                    "query_discoverable": True,
                },
                **kwargs,
            )
            assert (task.name in selected.names) is (mark is True and opt is True)
    hidden = replace(
        task,
        name="agent.hidden",
        model_name="agent__hidden",
        metadata={"kernel_utility": True},
    )
    denied = replace(
        task,
        name="agent.denied",
        model_name="agent__denied",
        authorize=lambda context: False,
    )
    ordinary = replace(
        task,
        name="ordinary.check",
        model_name="ordinary__check",
        metadata={"kernel_utility": 0, "related_tools": [hidden.name, denied.name]},
    )
    catalog = ToolCatalog([ordinary, hidden, denied])
    result = CapabilityDiscovery(catalog, context={}).search({"query": QUERY})
    assert result["loaded_tools"] == [ordinary.name]
    assert denied.name not in json.dumps(result)
    assert (
        hidden.name
        not in retriever.retrieve(
            QUERY, catalog, include_discoverable_utilities=True
        ).names
    )
    # The retriever alone does not provide authorization; discovery supplies its filtered catalog.
    assert len(task.description) <= 500
    for text in (task.description[:500], task.model_definition()["description"]):
        for contract in (
            "TMDB/qBittorrent",
            "blocked",
            "No write authority",
            "create/revise",
            "checkpoint",
            "cancel",
            "exact current user source",
            "task_id/epoch/revision",
        ):
            assert contract in text


def test_small_window_inherited_exact_discovery_characterization(isolated_db, tmp_path):
    from app.agent.kernel.discovery import CapabilityDiscovery

    runtime = build_agent_kernel_runtime(model=QueryTaskScript().adapter, store=store())
    discovery = CapabilityDiscovery(
        runtime.session.catalog, context={"has_task": True}, maximum=2
    )
    result = discovery.search({"tool_names": ["config.explain_component"]})
    window = discovery.window((), discovery.consume())
    snapshot = {"loaded": result["loaded_tools"], "window": [t.name for t in window]}
    (tmp_path / "small-window.json").write_text(json.dumps(snapshot, indent=2))
    # Business slots exclude fixed utilities; a loaded tool must enter the next window.
    assert snapshot == {
        "loaded": ["config.explain_component"],
        "window": ["agent.capabilities", "agent.task", "config.explain_component"],
    }
