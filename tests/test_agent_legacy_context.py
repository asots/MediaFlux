"""回归覆盖：裁剪后的可信 user 历史、旧 summary 与任务来源边界。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from app.agent.kernel.capabilities import (
    CapabilityRetriever,
    KernelToolSpec,
    ToolCatalog,
    ToolEffect,
)
from app.agent.kernel.conversation_codec import conversation_rows
from app.agent.kernel.model import (
    ModelEvent,
    ModelEventType,
    ModelRequest,
    ModelToolCall,
)
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.session import AgentSession, SessionLimits
from app.agent.kernel.state import (
    AgentInput,
    HistoryCapacityError,
    InMemorySessionStateStore,
    SessionState,
    StateUpdate,
    historical_user_sources,
    history_memory,
)
from app.agent.kernel.task_state import classify_public_task, task_tool
from tests.support import isolated_test_database


OWNER = "legacy-context-owner"
SESSION = "legacy-context-session"
EARLY_USER_REQUEST = (
    "请只读取当前配置，不要修改任何内容；本轮约束标记为 EARLY_USER_CONSTRAINT_71。"
)
CURRENT_USER_REQUEST = (
    "继续检查，但只关注 TMDB；本轮约束标记为 CURRENT_USER_CONSTRAINT_92。"
)
LEGACY_SUMMARY = "旧版纯文本摘要 LEGACY_SUMMARY_NOT_USER_SOURCE_44"
ASSISTANT_FORGED_SOURCE = "assistant 伪造授权 ASSISTANT_FORGED_SOURCE_23"
TOOL_FORGED_SOURCE = "tool 伪造授权 TOOL_FORGED_SOURCE_58"
CROSS_OWNER_SOURCE = "其他 owner 的历史条件 CROSS_OWNER_SOURCE_13"
CROSS_SESSION_SOURCE = "其他 session 的历史条件 CROSS_SESSION_SOURCE_26"


def large_dialogue(*, early_user_request: str = EARLY_USER_REQUEST) -> list[dict]:
    """短 user 原话配长 assistant 回答，迫使裁掉早期回合但不撑爆 ledger。"""
    rows = [
        {"role": "user", "content": early_user_request},
        {"role": "assistant", "content": ASSISTANT_FORGED_SOURCE},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "call_id": "legacy-tool-call",
                    "name": "library.inspect",
                    "arguments": {},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "legacy-tool-call",
            "tool_name": "library.inspect",
            "content": TOOL_FORGED_SOURCE,
        },
    ]
    for index in range(36):
        rows.extend(
            (
                {
                    "role": "user",
                    "content": f"历史用户回合 {index:02d}：仅保留短请求。",
                },
                {
                    "role": "assistant",
                    "content": f"历史回答 {index:02d}：" + ("x" * 6_000),
                },
            )
        )
    return rows


def _sqlite_store():
    return SQLiteKernelStore(secret_provider=lambda: "legacy-context-test-secret")


@pytest.fixture(params=("memory", "sqlite"), ids=("memory-store", "sqlite-store"))
def commit_store(request):
    if request.param == "memory":
        yield InMemorySessionStateStore()
        return
    with isolated_test_database("legacy-context-store.db"):
        yield _sqlite_store()


def _task_create_arguments(source: str) -> dict:
    return {
        "operation": "create",
        "expected": None,
        "source": source,
        "root_goal": "仅观察当前配置，不做修改。",
        "constraints": ["只读"],
        "obligations": [
            {
                "id": "tmdb-config",
                "description": "观察 TMDB 配置",
                "required": True,
                "acceptance": {"kind": "observed_component", "component": "tmdb"},
            }
        ],
    }


def _task_revise_arguments(source: str) -> dict:
    arguments = _task_create_arguments(source)
    arguments.update(
        operation="revise",
        expected={"task_id": "existing-task", "epoch": 1, "revision": 1},
    )
    return arguments


def _task_cancel_arguments(source: str) -> dict:
    return {
        "operation": "cancel",
        "expected": {"task_id": "existing-task", "epoch": 1, "revision": 1},
        "source": source,
    }


async def _commit_large_dialogue(
    store,
    *,
    owner: str = OWNER,
    session_id: str = SESSION,
    early_user_request: str = EARLY_USER_REQUEST,
):
    lease, _ = await store.begin_turn(
        owner=owner, session_id=session_id, request_id="seed-large-dialogue"
    )
    return await store.commit(
        lease, conversation=large_dialogue(early_user_request=early_user_request)
    )


class ScriptedModel:
    """只记录真实 ModelRequest，并按轮次返回本地脚本事件。"""

    def __init__(self, rounds: list[list[ModelEvent]]) -> None:
        self.rounds = list(rounds)
        self.requests: list[ModelRequest] = []

    async def stream(
        self, request: ModelRequest, *, cancellation
    ) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        if not self.rounds:
            raise AssertionError("unexpected model request")
        for event in self.rounds.pop(0):
            cancellation.raise_if_cancelled()
            yield event


def _call_round(name: str, arguments: dict, call_id: str) -> list[ModelEvent]:
    return [
        ModelEvent(
            ModelEventType.TOOL_CALL_COMPLETED,
            tool_call=ModelToolCall(call_id, name, arguments),
        ),
        ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls"),
    ]


def _capabilities_tool() -> KernelToolSpec:
    def search(arguments, context):
        assert context.capability_search is not None
        return context.capability_search(arguments)

    return KernelToolSpec(
        name="agent.capabilities",
        domain="agent",
        description="发现当前会话可用的受控能力。",
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        effect=ToolEffect.READ,
        read=search,
        metadata={"kernel_utility": True},
    )


def _read_tool() -> KernelToolSpec:
    return KernelToolSpec(
        name="library.inspect",
        domain="library",
        description="只读检查库配置。",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        effect=ToolEffect.READ,
        read=lambda _arguments, _context: {"ok": True, "status": "success"},
    )


def _session(
    store, model: ScriptedModel, *, limits: SessionLimits | None = None
) -> AgentSession:
    catalog = ToolCatalog([_capabilities_tool(), _read_tool()])
    pipeline = ToolPipeline(catalog=catalog, state_store=store)
    return AgentSession(
        model=model,
        catalog=catalog,
        retriever=CapabilityRetriever(minimum=1, maximum=3),
        pipeline=pipeline,
        state_store=store,
        limits=limits,
    )


def _run(coroutine):
    return asyncio.run(coroutine)


def test_commit_migrates_cropped_user_conditions_and_never_forges_task(commit_store):
    async def exercise():
        state = await _commit_large_dialogue(commit_store)
        assert all(
            row.get("content") != EARLY_USER_REQUEST for row in state.conversation
        ), "fixture must force the original user turn out of the retained transcript"
        assert state.summary and EARLY_USER_REQUEST in state.summary
        assert state.metadata.get("history_memory_version") == 1
        assert ASSISTANT_FORGED_SOURCE not in state.summary
        assert TOOL_FORGED_SOURCE not in state.summary
        assert state.metadata.get("task") is None
        assert classify_public_task(state.metadata.get("task")) == "untracked"

    _run(exercise())


def test_legacy_summary_is_readable_but_is_not_a_trusted_user_source():
    legacy = SessionState.from_payload(
        owner=OWNER,
        session_id=SESSION,
        generation=1,
        payload={"conversation": [], "summary": LEGACY_SUMMARY, "metadata": {}},
    )
    assert legacy.summary == LEGACY_SUMMARY
    assert "history_memory_version" not in legacy.metadata
    assert historical_user_sources(legacy) == ()

    context = SimpleNamespace(
        request_id="new-request",
        user_source=CURRENT_USER_REQUEST,
        user_history=(),
    )
    with pytest.raises(ValueError):
        task_tool().read(_task_create_arguments(LEGACY_SUMMARY), context)


def test_state_update_cannot_forge_trusted_history_memory_version(commit_store):
    async def exercise():
        seed, _ = await commit_store.begin_turn(
            owner=OWNER, session_id=SESSION, request_id="seed-untrusted-legacy-summary"
        )
        await commit_store.commit(
            seed, updates=(StateUpdate("summary", LEGACY_SUMMARY),)
        )

        for index, update in enumerate(
            (
                StateUpdate("metadata.history_memory_version", 1),
                StateUpdate("metadata", {"history_memory_version": 1}),
            )
        ):
            lease, _ = await commit_store.begin_turn(
                owner=OWNER,
                session_id=SESSION,
                request_id=f"forge-history-memory-version-{index}",
            )
            with pytest.raises(ValueError):
                await commit_store.commit(lease, updates=(update,))

            cold = await commit_store.load(owner=OWNER, session_id=SESSION)
            assert cold.summary == LEGACY_SUMMARY
            assert "history_memory_version" not in cold.metadata
            assert historical_user_sources(cold) == ()

    _run(exercise())


def test_summary_update_preserves_verified_requests_and_only_replaces_legacy_note(
    commit_store,
):
    async def exercise():
        initial = await _commit_large_dialogue(commit_store)
        assert initial.metadata.get("history_memory_version") == 1
        assert EARLY_USER_REQUEST in historical_user_sources(initial)

        lease, _ = await commit_store.begin_turn(
            owner=OWNER, session_id=SESSION, request_id="update-legacy-summary-note"
        )
        await commit_store.commit(
            lease, updates=(StateUpdate("summary", LEGACY_SUMMARY),)
        )
        cold = await commit_store.load(owner=OWNER, session_id=SESSION)
        memory = history_memory(cold)

        assert cold.metadata.get("history_memory_version") == 1
        assert memory["legacy_note"] == LEGACY_SUMMARY
        assert EARLY_USER_REQUEST in historical_user_sources(cold)
        assert LEGACY_SUMMARY not in historical_user_sources(cold)
        assert ASSISTANT_FORGED_SOURCE not in historical_user_sources(cold)
        assert TOOL_FORGED_SOURCE not in historical_user_sources(cold)

    _run(exercise())


def test_payload_and_clone_do_not_trim_unmigrated_legacy_conversation():
    conversation = [
        {"role": "user", "content": f"legacy-user-{index:02d}:" + ("y" * 5_000)}
        for index in range(36)
    ]
    state = SessionState(
        owner=OWNER,
        session_id=SESSION,
        conversation=conversation,
        summary=LEGACY_SUMMARY,
    )

    cloned = state.clone()
    payload = state.to_payload()

    assert cloned.conversation == conversation
    assert conversation_rows(payload["conversation"]) == conversation
    assert payload["summary"] == LEGACY_SUMMARY
    assert payload["metadata"] == {}


@pytest.mark.parametrize("oversized_store", ("memory", "sqlite"), ids=("memory-store", "sqlite-store"))
def test_oversized_cropped_user_history_rejects_commit_without_mutating_checkpoint(
    oversized_store,
):
    async def exercise(store):
        initial, _ = await store.begin_turn(
            owner=OWNER, session_id=SESSION, request_id="before-oversized-history"
        )
        await store.commit(
            initial,
            conversation=[{"role": "user", "content": "已有检查点，保持不变。"}],
        )
        lease, before = await store.begin_turn(
            owner=OWNER, session_id=SESSION, request_id="oversized-history"
        )

        with pytest.raises(HistoryCapacityError):
            await store.commit(
                lease,
                conversation=[
                    {"role": "user", "content": "OVERSIZED_USER_SOURCE_" + ("z" * 200_000)}
                ],
            )

        after = await store.load(owner=OWNER, session_id=SESSION)
        assert after.generation == before.generation
        assert after.conversation == before.conversation
        assert after.summary == before.summary
        assert after.metadata == before.metadata

    if oversized_store == "memory":
        _run(exercise(InMemorySessionStateStore()))
        return
    with isolated_test_database("legacy-context-oversized-history.db"):
        _run(exercise(_sqlite_store()))


def test_cold_session_request_uses_migrated_memory_and_keeps_current_user_turn():
    with isolated_test_database("legacy-context-cold-session.db"):
        async def exercise():
            initial_store = _sqlite_store()
            await _commit_large_dialogue(initial_store)

            cold_store = _sqlite_store()
            cold = await cold_store.load(owner=OWNER, session_id=SESSION)
            assert cold.summary and EARLY_USER_REQUEST in cold.summary
            assert cold.metadata.get("history_memory_version") == 1
            assert cold.metadata.get("task") is None
            assert classify_public_task(cold.metadata.get("task")) == "untracked"

            model = ScriptedModel(
                [[
                    ModelEvent(ModelEventType.TEXT_DELTA, text="仅完成当前问答。"),
                    ModelEvent(ModelEventType.FINISH, finish_reason="stop"),
                ]]
            )
            session = _session(cold_store, model)
            events = [
                event
                async for event in session.run(
                    AgentInput(
                        message=CURRENT_USER_REQUEST,
                        owner=OWNER,
                        session_id=SESSION,
                        request_id="new-current-turn",
                    )
                )
            ]
            assert events[-1].payload["status"] == "success"
            request = model.requests[0]
            assert EARLY_USER_REQUEST in request.system_prompt
            assert any(
                message.role == "user" and CURRENT_USER_REQUEST in message.content
                for message in request.messages
            )
            after = await cold_store.load(owner=OWNER, session_id=SESSION)
            assert after.metadata.get("task") is None
            assert classify_public_task(after.metadata.get("task")) == "untracked"

        _run(exercise())


def test_low_model_window_injects_early_user_history_without_storage_trimming():
    async def exercise():
        store = InMemorySessionStateStore()
        dialogue = [
            {"role": "user", "content": EARLY_USER_REQUEST},
            {"role": "assistant", "content": "已记录最早的用户约束。"},
        ]
        for index in range(14):
            dialogue.extend(
                (
                    {"role": "user", "content": f"低窗口历史用户回合 {index:02d}。"},
                    {
                        "role": "assistant",
                        "content": f"低窗口历史回答 {index:02d}：" + ("m" * 5_000),
                    },
                )
            )
        lease, _ = await store.begin_turn(
            owner=OWNER, session_id=SESSION, request_id="seed-model-window"
        )
        stored = await store.commit(lease, conversation=dialogue)
        assert any(
            row.get("role") == "user" and row.get("content") == EARLY_USER_REQUEST
            for row in stored.conversation
        )
        assert stored.summary == ""
        assert "history_memory_version" not in stored.metadata

        model = ScriptedModel(
            [[
                ModelEvent(ModelEventType.TEXT_DELTA, text="已完成当前问答。"),
                ModelEvent(ModelEventType.FINISH, finish_reason="stop"),
            ]]
        )
        session = _session(
            store,
            model,
            limits=SessionLimits(context_window_tokens=16_384, max_output_tokens=128),
        )
        events = [
            event
            async for event in session.run(
                AgentInput(
                    message=CURRENT_USER_REQUEST,
                    owner=OWNER,
                    session_id=SESSION,
                    request_id="low-window-current-turn",
                )
            )
        ]

        assert events[-1].payload["status"] == "success"
        request = model.requests[0]
        assert EARLY_USER_REQUEST in request.system_prompt
        assert all(EARLY_USER_REQUEST not in message.content for message in request.messages)
        assert any(
            message.role == "user" and CURRENT_USER_REQUEST in message.content
            for message in request.messages
        )

    _run(exercise())


def test_history_memory_is_scoped_to_exact_owner_and_session():
    with isolated_test_database("legacy-context-scope-isolation.db"):
        async def exercise():
            store = _sqlite_store()
            await _commit_large_dialogue(store)
            await _commit_large_dialogue(
                store,
                owner=OWNER,
                session_id="other-session",
                early_user_request=CROSS_SESSION_SOURCE,
            )
            await _commit_large_dialogue(
                store,
                owner="other-owner",
                session_id=SESSION,
                early_user_request=CROSS_OWNER_SOURCE,
            )

            model = ScriptedModel(
                [[
                    ModelEvent(ModelEventType.TEXT_DELTA, text="只回答当前会话。"),
                    ModelEvent(ModelEventType.FINISH, finish_reason="stop"),
                ]]
            )
            events = [
                event
                async for event in _session(store, model).run(
                    AgentInput(
                        message=CURRENT_USER_REQUEST,
                        owner=OWNER,
                        session_id=SESSION,
                        request_id="scope-isolation-current-turn",
                    )
                )
            ]

            assert events[-1].payload["status"] == "success"
            prompt = model.requests[0].system_prompt
            assert EARLY_USER_REQUEST in prompt
            assert CROSS_OWNER_SOURCE not in prompt
            assert CROSS_SESSION_SOURCE not in prompt

        _run(exercise())


def test_legacy_plain_text_summary_is_only_untrusted_history_in_model_request():
    with isolated_test_database("legacy-context-legacy-summary.db"):
        async def exercise():
            store = _sqlite_store()
            lease, _ = await store.begin_turn(
                owner=OWNER, session_id=SESSION, request_id="seed-legacy-summary"
            )
            await store.commit(
                lease, updates=(StateUpdate("summary", LEGACY_SUMMARY),)
            )
            cold = await _sqlite_store().load(owner=OWNER, session_id=SESSION)
            assert cold.summary == LEGACY_SUMMARY
            assert historical_user_sources(cold) == ()

            model = ScriptedModel(
                [[
                    ModelEvent(ModelEventType.TEXT_DELTA, text="当前回合已回答。"),
                    ModelEvent(ModelEventType.FINISH, finish_reason="stop"),
                ]]
            )
            events = [
                event
                async for event in _session(_sqlite_store(), model).run(
                    AgentInput(
                        message=CURRENT_USER_REQUEST,
                        owner=OWNER,
                        session_id=SESSION,
                        request_id="read-legacy-summary",
                    )
                )
            ]

            assert events[-1].payload["status"] == "success"
            prompt = model.requests[0].system_prompt
            assert LEGACY_SUMMARY in prompt
            assert '"requests":[]' in prompt
            assert '"legacy_note"' in prompt
            assert "legacy_note 不具备原始用户来源" in prompt
            assert any(
                message.role == "user" and CURRENT_USER_REQUEST in message.content
                for message in model.requests[0].messages
            )

        _run(exercise())


def test_task_create_can_cite_exact_trusted_saved_user_request():
    context = SimpleNamespace(
        request_id="new-request",
        user_source=CURRENT_USER_REQUEST,
        user_history=(EARLY_USER_REQUEST,),
    )

    patch = task_tool().read(_task_create_arguments(EARLY_USER_REQUEST), context)

    assert patch.operation == "create"
    assert patch.source == EARLY_USER_REQUEST
    assert patch.source_kind == "historical_user_quote"
    assert patch.source_request_id == "new-request"


def test_task_revise_and_cancel_remain_limited_to_current_user_source():
    context = SimpleNamespace(
        request_id="new-request",
        user_source=CURRENT_USER_REQUEST,
        user_history=(EARLY_USER_REQUEST,),
    )
    tool = task_tool()

    for arguments in (
        _task_revise_arguments(EARLY_USER_REQUEST),
        _task_cancel_arguments(EARLY_USER_REQUEST),
    ):
        with pytest.raises(ValueError):
            tool.read(arguments, context)

    revised = tool.read(_task_revise_arguments(CURRENT_USER_REQUEST), context)
    cancelled = tool.read(_task_cancel_arguments(CURRENT_USER_REQUEST), context)
    assert revised.source == CURRENT_USER_REQUEST
    assert revised.source_kind == "current_user"
    assert cancelled.source == CURRENT_USER_REQUEST
    assert cancelled.source_kind == "current_user"


def test_task_create_rejects_assistant_and_tool_text_as_historical_sources():
    context = SimpleNamespace(
        request_id="new-request",
        user_source=CURRENT_USER_REQUEST,
        user_history=(EARLY_USER_REQUEST,),
    )
    tool = task_tool()

    for forged in (ASSISTANT_FORGED_SOURCE, TOOL_FORGED_SOURCE):
        with pytest.raises(ValueError):
            tool.read(_task_create_arguments(forged), context)


def test_real_session_task_create_receives_only_trusted_cropped_user_history():
    with isolated_test_database("legacy-context-task-proposal.db"):
        async def exercise():
            await _commit_large_dialogue(_sqlite_store())
            cold_store = _sqlite_store()
            model = ScriptedModel(
                [
                    _call_round(
                        "agent__capabilities",
                        {"tool_names": ["agent.task"]},
                        "discover-task",
                    ),
                    _call_round(
                        "agent__task",
                        _task_create_arguments(EARLY_USER_REQUEST),
                        "create-from-history",
                    ),
                    [
                        ModelEvent(ModelEventType.TEXT_DELTA, text="任务清单已保存为部分状态。"),
                        ModelEvent(ModelEventType.FINISH, finish_reason="stop"),
                    ],
                ]
            )
            session = _session(cold_store, model)
            events = [
                event
                async for event in session.run(
                    AgentInput(
                        message=CURRENT_USER_REQUEST,
                        owner=OWNER,
                        session_id=SESSION,
                        request_id="task-from-history",
                    )
                )
            ]
            saved = await cold_store.load(owner=OWNER, session_id=SESSION)
            assert EARLY_USER_REQUEST in model.requests[0].system_prompt
            assert saved.metadata["task"]["status"] == "partial"
            assert saved.metadata["task"]["source_quote"] == EARLY_USER_REQUEST
            assert saved.metadata["task"]["source_kind"] == "historical_user_quote"
            assert events[-1].payload["task"]["status"] == "partial"
            assert session.catalog.get("agent.task").effect is ToolEffect.READ
            assert all(tool.effect is ToolEffect.READ for tool in session.catalog.visible())

        _run(exercise())


@pytest.mark.parametrize("schema_error", [False, True])
def test_long_old_request_is_readable_without_freezing_unrelated_turn_or_repair(schema_error):
    from app.agent.kernel.pipeline import ToolCallContext, ToolPipelineError
    from app.agent.kernel.state import CancellationToken

    async def exercise():
        store = InMemorySessionStateStore()
        long_request = "旧任务背景" + "文" * 11_000 + "END_NO_WRITES"
        lease, _ = await store.begin_turn(owner=OWNER, session_id=SESSION, request_id="long-old")
        await store.commit(lease, conversation=[
            {"role": "user", "content": long_request},
            {"role": "assistant", "content": "旧任务未执行"},
            {"role": "user", "content": "最近独立任务：你好"},
            {"role": "assistant", "content": "你好"},
        ])
        rounds = [_call_round("library.inspect", {"extra": "invalid"}, "bad-args")] if schema_error else []
        rounds.append([
            ModelEvent(ModelEventType.TEXT_DELTA, text="你好。"),
            ModelEvent(ModelEventType.FINISH, finish_reason="stop"),
        ])
        model = ScriptedModel(rounds)
        session = _session(store, model, limits=SessionLimits(context_window_tokens=16_384, max_output_tokens=6000))
        events = [event async for event in session.run(AgentInput(
            owner=OWNER, session_id=SESSION, request_id="unrelated", message="新的普通问题：你好",
        ))]
        assert model.requests
        if not schema_error:
            assert events[-1].payload["status"] == "success"
        else:
            assert "agent__repair" in {tool["name"] for tool in model.requests[-1].tools}
        prompt = model.requests[0].system_prompt
        view = next(json.loads(line) for line in prompt.splitlines() if line.startswith('{"requests"'))
        assert view["truncated"] and view["truncation"]
        assert "agent__read_result" in {tool["name"] for tool in model.requests[0].tools}
        lease, _ = await store.begin_turn(owner=OWNER, session_id=SESSION, request_id="read-old")

        async def progress(_payload):
            pass

        context = ToolCallContext(owner=OWNER, session_id=SESSION, request_id="read-old", turn_id=lease.turn_id,
                                  lease=lease, cancellation=CancellationToken(), report_progress=progress)
        arguments = {"handle": view["result_handle"], "path": "/requests/0/text", "offset": 10_900, "limit": 1000}
        result = await session.pipeline.execute("agent.read_result", arguments, context=context)
        page = json.loads(result.outcome.model_content)["data"]
        assert page["items"] == long_request[10_900:]
        assert "END_NO_WRITES" in page["items"]
        assert page["total"] == len(long_request)
        from dataclasses import replace
        other, _ = await store.begin_turn(owner="another-owner", session_id=SESSION, request_id="other")
        with pytest.raises(ToolPipelineError) as error:
            await session.pipeline.execute("agent.read_result", arguments,
                context=replace(context, owner="another-owner", lease=other, turn_id=other.turn_id, request_id="other"))
        assert error.value.code == "reference_invalid"

    _run(exercise())


def test_large_latest_assistant_text_cannot_freeze_legacy_task_after_tool_result():
    from app.agent.kernel.model import ModelMessage
    from app.agent.model_context_budget import bounded_model_messages

    prior = [ModelMessage(role="user", content=EARLY_USER_REQUEST),
             ModelMessage(role="assistant", content="旧助手说明" * 15_000)]
    current = [ModelMessage(role="user", content=CURRENT_USER_REQUEST)]
    bounded = bounded_model_messages(prior + current, history_end=2, tool_definitions=(),
                                    system_prompt="", context_window_tokens=16_384, output_tokens=4096)
    assert bounded == tuple(current)
    # 旧调用按完整组退出窗口，不能截断参数，也不能阻塞新问题。
    guarded = [prior[0], ModelMessage(role="assistant", content=prior[1].content,
               tool_calls=(ModelToolCall("executed", "library.inspect", {}),)),
               ModelMessage(role="tool", tool_call_id="executed", tool_name="library.inspect", content='{"ok":true}'), *current]
    assert bounded_model_messages(guarded, history_end=3, tool_definitions=(),
                                  system_prompt="", context_window_tokens=16_384,
                                  output_tokens=4096) == tuple(current)
