"""Public title classification: default reachable work and legacy boundaries."""
from __future__ import annotations

import asyncio
import copy
import itertools
import json
import random
import re
from time import process_time

import pytest

from app.agent import public_safety as safety
from app.agent.kernel.adapters import TurnViewBuilder, iter_ndjson
from app.agent.kernel.capabilities import CapabilityRetriever, KernelToolSpec, ToolCatalog, ToolEffect
from app.agent.kernel.events import AgentEvent, AgentEventType
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import AgentInput
from app.agent.models import ToolReference, ToolResult
from app.agent.public_view import candidate_result_lines, public_conversation_messages
from app.agent.kernel.ux_selection import candidate_item
from app.agent.recent_resource_candidates import new_resource_search_id, safe_resource_snapshot
from tests.test_agent_public_text_decoding import offline as offline, store as store
from tests.test_agent_tool_argument_ingress import Wire, encode

# Frozen short-input characterization oracle, never used for long cost cases.
LEGACY_RELATIVE = re.compile(
    r"(?<!\w)(?=[A-Za-z0-9._~:-]*[A-Za-z._~:-])[A-Za-z0-9._~:-]+[\\/][^\s]+"
)


@pytest.mark.parametrize("text,expected", [
    ("a/b", True), ("0a/b", True), ("123/b", False), ("_/b", True),
    ("./b", True), ("../b", True), ("~/b", True), ("-/b", True),
    (":/b", True), ("dir\\file", True), ("C:\\private\\file", True),
    ("a/b/", True), ("a/", False), ("a/ b", False), ("a/\t", False),
    ("中文a/b", False), ("éa/b", False), ("中.a/b", True),
    ("中./b", False), ("中.1/b", False), ("中..1/b", True),
    ("中_a/b", False), ("中_a~b/x", True), ("中_a~1/x", False),
    ("中_a~_1/x", True), ("中_a~-1/x", True), ("中_a~~/x", True),
    ("中_a~/x", False), ("中_a~9.x/x", True), ("中_a~9/x", False),
    ("中_a~9:0/x", True), ("中_a~9-0/x", True), ("中_a~9_0/x", True),
    ("中_a~9~0/x", True), ("a\u0301b/x", True), ("中a ! b/x", True),
    ("a.b_c", False), ("a b / c", False), ("a b /", False),
    ("MiB/s", True), ("/srv/private", True), ("12/34", False),
])
def test_relative_path_short_characterization(text, expected):
    assert bool(LEGACY_RELATIVE.search(text)) is expected
    assert bool(safety._RELATIVE_PATH_RE.search(text)) is expected


def test_relative_path_bounded_differential():
    alphabet = "a0_.~:-/\\ 中é\u0301!"
    samples = itertools.chain.from_iterable(itertools.product(alphabet, repeat=n) for n in range(5))
    rng = random.Random(602)
    random_samples = (rng.choices(alphabet + "AZ19\t\n²", k=rng.randrange(1, 81)) for _ in range(5000))
    for chars in itertools.chain(samples, random_samples):
        text = "".join(chars)
        assert bool(safety._RELATIVE_PATH_RE.search(text)) == bool(LEGACY_RELATIVE.search(text)), repr(text)


@pytest.mark.parametrize("text,public,title", [
    ("Film.Name_2026.mkv", "内部检查", "Film.Name_2026.mkv"),
    ("速度 12 MiB/s", "速度 12 MiB/s", "速度 12 MiB/s"),
    ("速度 12.5 MB/s", "速度 12.5 MB/s", "速度 12.5 MB/s"),
    ("a/b", "", ""), ("12/34", "12/34", "12/34"),
    ("标题 /srv/private/canary", "", ""), ("标题 C:\\private\\canary", "", ""),
    ("电影 https://private.invalid/canary", "", "电影 [网址]"),
    ("中a/b", "中a/b", "中a/b"), ("中.a/b", "", ""),
    ("~/", "~/", "~/"), ("~~/ 结束", "~~/ 结束", "~~/ 结束"),
    ("a! / 结束", "a! / 结束", "a! / 结束"),
])
def test_short_public_classification_contract(text, public, title):
    assert safety.sanitize_public_text(text) == public
    assert safety.sanitize_resource_title(text) == title


def long_title(kind, length):
    if kind == "plain":
        return "~" * length
    if kind == "trailing-slash":
        return ".-~" * (length // 3) + "/"
    if kind == "slash-space":
        return "~" * length + "/ 结束"
    if kind == "remote-slash":
        return "~" * length + " ! / 结束"
    if kind == "punctuation-runs":
        return ("~.-" * 25 + "! ") * (length // 77) + "/ 结束"
    if kind == "unicode-boundary":
        return "中" + "0" * length + "~/x"
    if kind == "unicode-word":
        return "中" + "0" * length + "a/x"
    if kind == "word":
        return "a" * length
    if kind == "underscores":
        return "_" * length
    if kind == "digits":
        return "1" * length + "/34"
    if kind == "dots":
        return "." * length + "/ "
    if kind == "relative-private":
        return "~" * length + "/Q2PRIVATECANARY"
    if kind == "late-private":
        return "~" * length + " /srv/private/Q2PRIVATECANARY"
    raise AssertionError(kind)


LONG_KINDS = ["plain", "trailing-slash", "slash-space", "remote-slash", "punctuation-runs",
              "unicode-boundary", "unicode-word", "word", "underscores", "digits", "dots", "relative-private", "late-private"]


@pytest.mark.parametrize("kind", LONG_KINDS)
def test_whole_title_and_consumers_have_bounded_cpu(kind):
    raw = long_title(kind, 40000)
    expected = "" if kind.endswith("private") or kind == "unicode-boundary" else raw[:300].rstrip()
    start = process_time()
    result = safety.sanitize_resource_title(raw)
    cpu = process_time() - start
    assert result == expected
    # Independent regression protection, not a product latency SLO. Includes all
    # decoding, credentials, identifier and adjacent path regexes, not one mock.
    assert cpu < 1.0, (kind, cpu)
    start = process_time()
    assert candidate_item({"title": raw}, 1)["title"] == (expected or "候选 1")
    lines = candidate_result_lines({"source_type": "resource_candidates", "items": [
        {"title": raw, "position": 1, "status": "submitted"},
    ]})
    assert lines == ["#1 · " + (raw[:160].rstrip() if expected else "资源") + "：已提交"]
    assert process_time() - start < 1.0


def test_default_pipeline_long_title_cost_and_reopened_consumers(store, monkeypatch):
    async def exercise():
        samples = []
        original_sanitize = safety.sanitize_public_text

        def observed(text, **kwargs):
            start = process_time()
            result = original_sanitize(text, **kwargs)
            samples.append((len(text), process_time() - start))
            return result

        monkeypatch.setattr(safety, "sanitize_public_text", observed)
        value = ToolResult(True, "success", "找到一个资源", data={"items": [{
            "result_id": "cost-resource-result-001", "title": "~" * 40000,
            "site_id": "demo", "site_name": "演示站点", "size_text": "2 GB",
            "download_state": "ready", "download_kinds": ["magnet"],
        }]})
        snapshot = safe_resource_snapshot(value, search_id=new_resource_search_id())
        assert len(snapshot["candidates"]) == 1
        value.references.append(ToolReference("resource_candidates", snapshot))
        original = copy.deepcopy(value)
        calls = []

        def read(arguments, context):
            assert context.owner == "cost-owner"
            calls.append(dict(arguments))
            return value

        tool = KernelToolSpec(
            name="indexer.search_resources", domain="indexer", description="只读资源搜索", effect=ToolEffect.READ,
            input_schema={"type": "object", "properties": {"title": {"type": "string"}},
                          "required": ["title"], "additionalProperties": False}, read=read,
        )
        catalog = ToolCatalog([tool])
        responses = [{"choices": [{"message": {"tool_calls": [{"type": "function", "id": "call-0",
            "function": {"name": tool.model_name, "arguments": encode({"title": "测试作品"})}}]},
            "finish_reason": "tool_calls"}]},
            {"choices": [{"message": {"content": "已找到资源，等待选择。"}, "finish_reason": "stop"}]}]
        wire = Wire("chat_completions", [(encode(item).encode(), "application/json") for item in responses])
        pipeline = ToolPipeline(catalog=catalog, state_store=store, reference_store=store)
        assert pipeline.projector.max_model_chars == 24000
        session = AgentSession(model=wire.adapter, catalog=catalog, pipeline=pipeline,
            retriever=CapabilityRetriever(minimum=1, maximum=1), state_store=store, journal=store)
        start = process_time()
        events = [event async for event in session.run(AgentInput(
            message="搜索测试作品", owner="cost-owner", session_id="cost-session",
        ))]
        session_cpu = process_time() - start
        assert value == original and calls == [{"title": "测试作品"}]
        wire.verify(posts=2)
        assert not wire.replies
        fresh = SQLiteKernelStore(secret_provider=lambda: "offline-public-decoding")
        journal = await fresh.list_events(owner="cost-owner", session_id="cost-session")
        saved = await fresh.load(owner="cost-owner", session_id="cost-session")
        assert journal == [event.to_dict() for event in events]
        assert events[-1].type is AgentEventType.TURN_COMPLETED
        assert events[-1].payload["status"] == "success"
        assert saved.conversation[-1]["content"] == events[-1].payload["answer"]
        assert public_conversation_messages(saved.conversation)[-1]["content"] == events[-1].payload["answer"]
        completed = [event["payload"]["result"] for event in journal if event["type"] == "tool.completed"]
        # Resource references intentionally compact the public DTO; the raw
        # returned ToolResult above must remain unchanged, not replace this DTO.
        assert len(completed) == 1
        assert completed[0]["ok"] is True and completed[0]["status"] == "success"
        assert completed[0]["summary"] == value.summary
        assert [ref["kind"] for ref in completed[0]["refs"]] == ["resource_candidates"]
        builder = TurnViewBuilder()
        restored = [AgentEvent(**{**event, "type": AgentEventType(event["type"])}) for event in journal]
        for event in restored:
            builder.apply(event)

        async def stream():
            for event in restored:
                yield event

        assert [json.loads(line) async for line in iter_ndjson(stream())] == journal
        assert builder.build().to_dict()["status"] == "success"
        full_scans = [cpu for length, cpu in samples if length == 40000]
        assert full_scans, "default pipeline must reach the complete title, not a shortened fixture"
        assert max(full_scans) < 1.0, samples
        assert session_cpu < 1.0, session_cpu
    asyncio.run(exercise())
