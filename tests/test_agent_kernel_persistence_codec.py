"""Baseline-characterized canonical JSON, crypto and transaction contracts."""
from typing import cast
from cryptography.fernet import Fernet
import asyncio
import hashlib
import hmac
import json
import threading
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import pytest

from app import database as db
from app.agent.kernel.events import AgentEvent, AgentEventType
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.references import ReferenceError
from app.agent.kernel.session_guard import session_scope_guard
from app.agent.kernel.state_types import StateUpdate
from tests.support import isolated_test_database

SECRET = "r2-synthetic-test-key"
DOMAIN = b"r2:synthetic:domain"


def sign(text, domain=DOMAIN):
    return hmac.new(
        SECRET.encode(), domain + b"\0" + text.encode(), hashlib.sha256
    ).hexdigest()


def outcome(call):
    try:
        return ["ok", call()]
    except Exception as exc:
        return [
            type(exc).__name__,
            str(exc),
            type(exc.__cause__).__name__ if exc.__cause__ else None,
        ]


@pytest.fixture
def store():
    return SQLiteKernelStore(secret_provider=lambda: SECRET, clock=lambda: 100.0)


@pytest.mark.parametrize("reference", [False, True])
def test_canonical_roundtrip_and_exact_utf8_boundary(store, reference):
    value = {"中": "文", "b": [None, True, 1.5], "a": 0}
    canonical = '{"a":0,"b":[null,true,1.5],"中":"文"}'
    maximum = len(canonical.encode("utf-8"))
    store.max_ref_bytes = maximum
    if reference:
        encoded, signature = store._encode_reference(value, domain=DOMAIN)
        assert encoded.startswith("enc:v1:")
        assert canonical not in encoded
        assert store._reference_cipher().decrypt(encoded[7:].encode()).decode() == canonical
        result = store._decode_reference(encoded, signature, domain=DOMAIN)
        store.max_ref_bytes -= 1
        failed = outcome(lambda: store._encode_reference(value, domain=DOMAIN))
    else:
        encoded, signature = store._encode(value, domain=DOMAIN, maximum=maximum)
        assert encoded == canonical
        result = store._decode(encoded, signature, domain=DOMAIN, expected_type=dict)
        failed = outcome(
            lambda: store._encode(value, domain=DOMAIN, maximum=maximum - 1)
        )
    assert signature == sign(encoded)
    assert result == value
    label = "引用" if reference else "数据"
    assert failed == ["ValueError", f"Agent Kernel {label}超过持久化上限", None]


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("kind", ["nan", "inf", "object", "surrogate", "cycle"])
def test_invalid_values_exact_errors(store, reference, kind):
    cycle = []
    cycle.append(cycle)
    value = {
        "nan": float("nan"),
        "inf": float("inf"),
        "object": object(),
        "surrogate": "\ud800",
        "cycle": cycle,
    }[kind]

    def encode():
        if reference:
            return store._encode_reference(value, domain=DOMAIN)
        return store._encode(value, domain=DOMAIN, maximum=4096)

    cause = {
        "nan": "ValueError",
        "inf": "ValueError",
        "object": "TypeError",
        "surrogate": "UnicodeEncodeError",
        "cycle": "ValueError",
    }[kind]
    label = "引用" if reference else "数据"
    assert outcome(encode) == ["ValueError", f"Agent Kernel {label}无法序列化", cause]


@pytest.mark.parametrize(
    "exception", [TypeError, ValueError, OverflowError, RecursionError, UnicodeError]
)
@pytest.mark.parametrize("reference", [False, True])
def test_injected_serializer_failure_preserves_cause(store, exception, reference):
    cause = exception("synthetic")
    with patch("app.agent.kernel.persistence.json.dumps", side_effect=cause):
        with pytest.raises(ValueError) as caught:
            if reference:
                store._encode_reference({}, domain=DOMAIN)
            else:
                store._encode({}, domain=DOMAIN, maximum=4096)
    label = "引用" if reference else "数据"
    assert type(caught.value) is ValueError
    assert str(caught.value) == f"Agent Kernel {label}无法序列化"
    assert caught.value.__cause__ is cause


def test_reference_limit_is_read_after_serializer(store):
    original = json.dumps

    def changing_limit(*args, **kwargs):
        store.max_ref_bytes = 1
        return original(*args, **kwargs)

    with patch("app.agent.kernel.persistence.json.dumps", side_effect=changing_limit):
        result = outcome(lambda: store._encode_reference({"safe": 1}, domain=DOMAIN))
    assert result == ["ValueError", "Agent Kernel 引用超过持久化上限", None]


def test_decrypt_order_legacy_domain_and_invalid_types(store):
    trace = []
    real_cipher = store._reference_cipher

    def cipher():
        trace.append("cipher")
        return real_cipher()

    with patch.object(store, "_reference_cipher", cipher):
        failed = outcome(
            lambda: store._decode_reference("enc:v1:broken", "invalid", domain=DOMAIN)
        )
        assert failed == ["ValueError", "Agent Kernel 引用校验失败", None]
        assert trace == []  # Signature failure precedes decrypt/key derivation.
        failed_decrypt = outcome(
            lambda: store._decode_reference(
                "enc:v1:broken", sign("enc:v1:broken"), domain=DOMAIN
            )
        )
        assert failed_decrypt == [
            "ValueError", "Agent Kernel 引用解密失败", "InvalidToken"
        ]
        assert trace == ["cipher"]
        # Old signed plaintext intentionally has no decode-time size restriction.
        old = json.dumps({"legacy": "x" * 5000})
        store.max_ref_bytes = 4
        assert store._decode_reference(old, sign(old), domain=DOMAIN) == json.loads(old)
        assert outcome(
            lambda: store._decode_reference(old, sign(old), domain=b"other")
        ) == ["ValueError", "Agent Kernel 引用校验失败", None]
        assert outcome(
            lambda: store._decode("[]", sign("[]"), domain=DOMAIN, expected_type=dict)
        ) == ["TypeError", "Agent Kernel 持久化数据类型无效", None]
        assert outcome(
            lambda: store._decode("{", sign("{"), domain=DOMAIN, expected_type=dict)
        ) == [
            "JSONDecodeError",
            "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
            None,
        ]
        token = "enc:v1:" + real_cipher().encrypt(b'"12345"').decode()
        assert outcome(
            lambda: store._decode_reference(token, sign(token), domain=DOMAIN)
        ) == ["ValueError", "Agent Kernel 引用超过持久化上限", None]
        assert trace == ["cipher", "cipher"]


def test_secret_cipher_injection_order_is_stable():
    trace = []

    def secret():
        trace.append("secret")
        return SECRET

    class Cipher:
        def encrypt(self, value):
            assert value == b'{"synthetic":1}'
            trace.append("encrypt")
            return b"controlled-token"

    class Injected(SQLiteKernelStore):
        def _reference_cipher(self):
            trace.append("cipher")
            return cast(Fernet, Cipher())

        def _encode(self, value, *, domain, maximum):
            raise AssertionError("References must not route through overridable _encode")

    store = Injected(secret_provider=secret)
    assert store._encode_reference({"synthetic": 1}, domain=DOMAIN) == (
        "enc:v1:controlled-token", sign("enc:v1:controlled-token")
    )
    assert trace == ["cipher", "encrypt", "secret"]
    trace.clear()
    SQLiteKernelStore._encode(store, {"synthetic": 1}, domain=DOMAIN, maximum=100)
    assert trace == ["secret"]
    trace.clear()
    store.max_ref_bytes = 1
    assert outcome(lambda: store._encode_reference({}, domain=DOMAIN)) == [
        "ValueError", "Agent Kernel 引用超过持久化上限", None
    ]
    assert trace == []


def test_public_state_ref_event_roundtrip_and_transaction_rollback(store):
    async def exercise():
        with isolated_test_database("r2.db"):
            lease, _ = await store.begin_turn(owner="o", session_id="s", request_id="r")
            messages = [{"role": "user", "content": "synthetic"}]
            before = await store.commit(
                lease, conversation=messages, updates=(StateUpdate("summary", "kept"),)
            )
            with pytest.raises(ValueError, match="数据无法序列化"):
                await store.commit(
                    lease,
                    conversation=[{"role": "user", "content": "changed"}],
                    updates=(StateUpdate("metadata.bad", float("nan")),),
                )
            reopened = SQLiteKernelStore(secret_provider=lambda: SECRET, clock=lambda: 100.0)
            assert await reopened.load(owner="o", session_id="s") == before
            ref = await store.put(
                owner="o", session_id="s", kind="test_kind", value={"safe": [1]}
            )
            assert await reopened.resolve(
                ref.ref, owner="o", session_id="s", expected_kind="test_kind"
            ) == {"safe": [1]}
            for owner, sid, kind in [
                ("other", "s", "test_kind"),
                ("o", "other", "test_kind"),
                ("o", "s", "other"),
            ]:
                with pytest.raises(ReferenceError):
                    await reopened.resolve(
                        ref.ref, owner=owner, session_id=sid, expected_kind=kind
                    )
            with db.get_conn() as conn:
                before_refs = conn.execute("SELECT COUNT(*) FROM agent_kernel_refs").fetchone()[0]
            with pytest.raises(ValueError, match="引用无法序列化"):
                await store.put(owner="o", session_id="s", kind="test_kind", value=float("nan"))
            with db.get_conn() as conn:
                assert conn.execute("SELECT COUNT(*) FROM agent_kernel_refs").fetchone()[0] == before_refs
            for seq, kind in enumerate([
                AgentEventType.TOOL_STARTED,
                AgentEventType.TOOL_COMPLETED,
                AgentEventType.TURN_COMPLETED,
            ], 1):
                await store.append(AgentEvent(
                    type=kind, session_id="s", turn_id="t", request_id="r",
                    sequence=seq, payload={"stage": seq},
                ), owner="o")
            events = await reopened.list_events(owner="o", session_id="s")
            assert [e["sequence"] for e in events] == [1, 2, 3]
            assert [e["type"] for e in events] == [
                "tool.started", "tool.completed", "turn.completed"
            ]
    asyncio.run(exercise())


@pytest.mark.parametrize("use_tool", [False, True])
def test_real_session_order_requests_and_plain_qa_budget(use_tool):
    from app.agent.kernel.bootstrap import build_agent_kernel_runtime
    from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
    from app.agent.kernel.state_types import AgentInput
    from tests.test_agent_kernel_core import ScriptedModel, collect

    async def exercise():
        with isolated_test_database("r2-session.db"):
            trace, statements = [], []

            class ObservedStore(SQLiteKernelStore):
                def _write_state(self, conn, state):
                    super()._write_state(conn, state)
                    trace.append("state-written")

                async def append_for_turn(self, event, lease):
                    await super().append_for_turn(event, lease)
                    trace.append("event:" + event.type.value)

                async def append(self, event, *, owner):
                    await super().append(event, owner=owner)
                    trace.append("event:" + event.type.value)

            answer = [
                ModelEvent(ModelEventType.TEXT_DELTA, text="synthetic answer"),
                ModelEvent(ModelEventType.FINISH, finish_reason="stop"),
            ]
            tool = [
                ModelEvent(ModelEventType.TOOL_CALL_COMPLETED, tool_call=ModelToolCall(
                    "r2-call", "config.explain_component", {"component": "tmdb"}
                )),
                ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls"),
            ]
            model = ScriptedModel([tool, answer] if use_tool else [answer])
            store = ObservedStore(secret_provider=lambda: SECRET)
            runtime = build_agent_kernel_runtime(model=model, store=store)
            original = db.get_conn

            @contextmanager
            def traced():
                with original() as conn:
                    conn.set_trace_callback(statements.append)
                    try:
                        yield conn
                    finally:
                        conn.set_trace_callback(None)

            with patch.object(db, "get_conn", traced):
                events = await collect(runtime.session.run(AgentInput(
                    "config.explain_component 读取TMDB配置状态" if use_tool else "解释什么是媒体元数据",
                    "webk:v1:" + "a" * 64, "r2-session", channel="test",
                )))
            assert len(model.requests) == (2 if use_tool else 1)
            assert sum(e.type is AgentEventType.TOOL_STARTED for e in events) == int(use_tool)
            # Durable publication finish adds one write; READ snapshots add two
            # references, their state registration and capability refresh (+4).
            assert sum(q.strip().upper() == "BEGIN IMMEDIATE" for q in statements) == (21 if use_tool else 9)
            transaction_tables = []
            for statement in statements:
                if statement.strip().upper() == "BEGIN IMMEDIATE":
                    transaction_tables.append([])
                elif statement.startswith(("INSERT INTO ", "INSERT OR IGNORE INTO ")):
                    transaction_tables[-1].append(statement.split("(", 1)[0].split()[-1])
            state_tables = ["agent_kernel_sessions", "agent_kernel_session_epochs"]
            event_tables = ["agent_kernel_events"]
            expected_tables = [state_tables, state_tables] + [event_tables] * 3
            if use_tool:
                expected_tables += [event_tables, event_tables, state_tables,
                    ["agent_rate_limit_buckets"], ["agent_kernel_refs"], ["agent_kernel_refs"],
                    state_tables, state_tables, event_tables, state_tables, event_tables, event_tables]
            expected_tables += [event_tables, state_tables, event_tables, state_tables]
            assert transaction_tables == expected_tables
            expected_events = ["turn.started", "capabilities.selected", "model.started"]
            expected_trace = [
                "state-written", "state-written", "event:turn.started",
                "event:capabilities.selected", "event:model.started",
            ]
            if use_tool:
                assert sum(e.type is AgentEventType.TOOL_COMPLETED for e in events) == 1
                assert any(m.role == "tool" for m in model.requests[1].messages)
                expected_events += [
                    "model.tool_call", "tool.started", "tool.completed", "capabilities.selected", "model.started"
                ]
                # Persist started/unknown, then the real result before publishing completion.
                expected_trace += [
                    "event:model.tool_call", "event:tool.started", "state-written",
                    "state-written", "state-written", "event:tool.completed", "state-written",
                    "event:capabilities.selected", "event:model.started",
                ]
            expected_events += ["model.delta", "turn.completed"]
            expected_trace += ["event:model.delta", "state-written", "event:turn.completed", "state-written"]
            assert [e.type.value for e in events] == expected_events
            assert trace == expected_trace
            state = await store.load(owner="webk:v1:" + "a" * 64, session_id="r2-session")
            assert [row["role"] for row in state.conversation] == (
                ["user", "assistant", "tool", "assistant"] if use_tool else ["user", "assistant"]
            )
            assert "task" not in state.metadata
    asyncio.run(exercise())


@pytest.mark.parametrize("protected", [False, True])
def test_cancel_after_insert_retains_transaction_and_guard(store, protected):
    async def exercise():
        with isolated_test_database("r2-cancel.db"):
            lease, _ = await store.begin_turn(owner="o", session_id="s", request_id="r")
            entered, release = threading.Event(), threading.Event()
            trace = []
            original = store._write_epoch

            def blocked(conn, **kwargs):
                trace.append("after-state-insert")
                entered.set()
                assert release.wait(5), "commit barrier was not released"
                original(conn, **kwargs)
                trace.append("epoch-written")

            async def commit():
                with session_scope_guard("o", "s", kind="effect") if protected else nullcontext():
                    return await store.commit(lease, updates=(StateUpdate("summary", "finished"),))

            with patch.object(store, "_write_epoch", blocked):
                task = asyncio.create_task(commit())
                try:
                    try:
                        assert await asyncio.to_thread(entered.wait, 5), "commit did not enter barrier"
                        # A real SQLite reader cannot observe the uncommitted update.
                        assert (await store.load(owner="o", session_id="s")).summary == ""
                        task.cancel()
                        await asyncio.sleep(0)
                        assert not task.done()
                    finally:
                        release.set()
                    if protected:
                        assert (await task).summary == "finished"
                        trace.append("returned")
                    else:
                        with pytest.raises(asyncio.CancelledError):
                            await task
                        trace.append("cancelled")
                    assert (await store.load(owner="o", session_id="s")).summary == "finished"
                    assert trace == [
                        "after-state-insert", "epoch-written", "returned" if protected else "cancelled"
                    ]
                finally:
                    # Even assertion failures must drain the worker before restoring DB paths.
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
    asyncio.run(exercise())
