"""Public-shape table and stream-boundary compatibility (not verifier tests)."""
from typing import Any

import asyncio
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from app.agent.kernel.adapters import TurnView
from app.agent.kernel.events import AgentEventType, EventFactory
from app.agent.kernel.metrics import KernelMetrics, TurnMeasurement
from app.agent.kernel.task_state import classify_public_task
from app.bot.agent_adapter import _render_turn, _task_summary
from app.bot.telegram_markdown import render_telegram_markdown


def shape() -> dict[str, Any]:
    return dict(
        v=1,
        task_id="opaque",
        epoch=1,
        revision=2,
        acceptance_scope="declared_obligations",
        status="partial",
        root_goal="观察配置",
        obligations=[
            dict(
                id="qb", description="查看 qBittorrent", required=True, state="pending"
            )
        ],
        remaining_obligation_ids=["qb"],
        blockers=[],
    )


def shape_cases():
    cases = [(None, "untracked")] + [
        (v, "unknown") for v in ({}, [], False, 0, "", {"status": "complete"})
    ]
    good = shape()
    cases.append((good, "partial"))
    for key, value in [
        ("v", True),
        ("v", 2),
        ("task_id", ""),
        ("epoch", True),
        ("revision", 0),
        ("acceptance_scope", "all"),
        ("root_goal", ""),
        ("status", " complete"),
        ("status", []),
        ("obligations", []),
        ("remaining_obligation_ids", []),
        ("remaining_obligation_ids", ["qb", "qb"]),
        ("remaining_obligation_ids", [{}]),
        ("blockers", [{"text": ""}]),
        ("status", "complete"),
    ]:
        cases.append(({**good, key: value}, "unknown"))
    for key, value in [
        ("id", ""),
        ("description", ""),
        ("required", 1),
        ("state", "done"),
        ("state", {}),
    ]:
        item = copy.deepcopy(good)
        item["obligations"][0][key] = value
        cases.append((item, "unknown"))
    item = copy.deepcopy(good)
    item["obligations"] *= 2
    cases.append((item, "unknown"))
    for status in ("blocked", "cancelled"):
        cases.append(({**good, "status": status}, status))
    complete = copy.deepcopy(good)
    complete.update(status="complete", remaining_obligation_ids=[])
    complete["obligations"][0]["state"] = "verified"
    cases.append((complete, "complete"))
    cases.append(({**complete, "blockers": [{"text": "矛盾"}]}, "unknown"))
    cases.append(
        (
            {**good, "verified_facts": {"ignored": "secret"}, "constraints": None},
            "partial",
        )
    )
    return cases


def test_python_javascript_public_shape_table():
    cases = shape_cases()
    expected = [status for _, status in cases]
    assert [classify_public_task(value) for value, _ in cases] == expected
    source = (Path(__file__).parents[1] / "app/static/js/agent.js").read_text()
    classifier = source.split("    function classifyPublicTask(task) {", 1)[1].split(
        "    function clearTaskSnapshot", 1
    )[0]
    script = (
        "function classifyPublicTask(task) {"
        + classifier
        + "\nconsole.log(JSON.stringify(JSON.parse(process.argv[1]).map(classifyPublicTask)))"
    )
    node = shutil.which("node")
    assert node, "Node >=22 is required for parity"
    result = subprocess.run(
        [node, "-e", script, json.dumps([v for v, _ in cases])],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == expected


@pytest.mark.parametrize(("task", "status"), shape_cases())
def test_telegram_task_summary_only_for_present_snapshots(task, status):
    rendered = _task_summary(task)
    if status == "untracked":
        assert rendered == ""
    elif status == "unknown":
        assert rendered == "会话声明任务快照：任务验收状态未知"
    else:
        assert "会话声明任务快照" in rendered
        assert "目标：" + task["root_goal"] in rendered


@pytest.mark.parametrize("status", ["success", "partial", "effect_completed", "cancelled", "failed"])
@pytest.mark.parametrize("has_receipt", [False, True])
def test_telegram_task_footer_preserves_turn_body_and_receipts(status, has_receipt):
    view = TurnView(
        session_id="s", turn_id="t", request_id="r", status=status,
        answer="**本轮答复**", error_message="处理失败",
        effect_result={"ok": True, "summary": "前项已执行"} if has_receipt else {},
    )
    rendered = _render_turn(view)
    assert "会话声明任务快照" not in rendered
    assert "未建立任务验收清单" not in rendered
    assert rendered == rendered.rstrip()
    if has_receipt:
        assert rendered.count("前项已执行") == 1
    if status in {"success", "partial"}:
        assert "<b>本轮答复</b>" in rendered
    for task in ({}, shape()):
        assert _render_turn(replace(view, task=task)) == (
            rendered + "\n\n" + render_telegram_markdown(_task_summary(task))
        )


def test_telegram_new_text_is_sanitized_before_display_budget():
    task = shape()
    private = "token=never-show-this-secret"
    task["root_goal"] = "安全目标 " + private
    task["obligations"][0]["description"] = "查看配置 " + private
    task["blockers"] = [{"text": "等待 " + private}]
    rendered = _task_summary(task)
    assert "never-show-this-secret" not in rendered
    assert "opaque" not in rendered
    assert "会话声明任务快照" in rendered
    assert (
        _task_summary({"status": "complete", "root_goal": private})
        == "会话声明任务快照：任务验收状态未知"
    )


def test_reused_complete_snapshot_does_not_rewrite_partial_turn():
    from app.agent.kernel.adapters import TurnViewBuilder
    from app.bot.agent_adapter import _render_turn

    captured = json.loads(
        (Path(__file__).parent / "fixtures/task_complete_turn_partial.json").read_text()
    )
    event = EventFactory(session_id="s", turn_id="t", request_id="r").create(
        AgentEventType.TURN_COMPLETED, captured["payload"]
    )

    async def run():
        metrics = KernelMetrics()

        async def stream():
            yield event

        _ = [e async for e in metrics.track(stream(), channel="telegram")]
        result = metrics.snapshot()
        assert result["statuses"] == {"partial": 1}
        assert result["task_statuses_by_turn"] == {"complete": 1}

    asyncio.run(run())
    builder = TurnViewBuilder()
    builder.apply(event)
    assert builder.build().status == "partial"
    assert "已声明检查项已核验" in _render_turn(builder.build())


def test_metrics_exhaustion_exceptions_close_and_last_terminal():
    async def run():
        factory = EventFactory(session_id="s", turn_id="t", request_id="r")
        terminal = factory.create(
            AgentEventType.TURN_COMPLETED, {"status": "success", "task": shape()}
        )

        async def stream(events, error=None):
            for event in events:
                yield event
            if error:
                raise error

        for error, status in (
            (RuntimeError("upstream"), "failed"),
            (asyncio.CancelledError(), "cancelled"),
        ):
            metrics = KernelMetrics()
            tracked = metrics.track(stream([terminal], error), channel="web")
            with pytest.raises(type(error)) as raised:
                _ = [e async for e in tracked]
            assert raised.value is error
            result = metrics.snapshot()
            assert result["turns"] == 1
            assert result["statuses"] == {status: 1}
            assert result["recent"][-1]["task_status"] == "unknown"
            assert result["task_statuses_by_turn"] == {}
            await tracked.aclose()
            await tracked.aclose()
            assert metrics.snapshot() == result
        metrics = KernelMetrics()
        source = stream([terminal])
        tracked = metrics.track(source, channel="web")
        assert await anext(tracked) is terminal
        await tracked.aclose()
        result = metrics.snapshot()
        assert result["turns"] == 1
        assert result["statuses"] == {"interrupted": 1}
        assert result["recent"][-1]["task_status"] == "unknown"
        assert result["task_statuses_by_turn"] == {}
        await tracked.aclose()
        await tracked.aclose()
        assert metrics.snapshot() == result
        await source.aclose()
        _ = [e async for e in metrics.track(stream([]), channel="web")]
        assert metrics.snapshot()["recent"][-1]["task_status"] == "unknown"
        assert metrics.snapshot()["statuses"] == {"interrupted": 2}
        assert metrics.snapshot()["task_statuses_by_turn"] == {}
        last = factory.create(AgentEventType.TURN_COMPLETED, {"status": "partial"})
        for _ in range(25):
            _ = [
                e async for e in metrics.track(stream([terminal, last]), channel="web")
            ]
        result = metrics.snapshot()
        assert result["turns"] == 27
        assert result["statuses"] == {"interrupted": 2, "partial": 25}
        assert result["task_statuses_by_turn"] == {"untracked": 25}
        assert len(result["recent"]) == 20
        assert all(
            e["task_status"] == "untracked"
            and e["turn_status"] == e["status"] == "partial"
            for e in result["recent"]
        )

    asyncio.run(run())
    assert (
        TurnMeasurement("r", "web", "success", 0, None, 0, 0, 0).to_dict()[
            "task_status"
        ]
        == "unknown"
    )
