#!/usr/bin/env python3
"""MediaFlux Agent Kernel 离线验收器；不调用 LLM、网络或业务写接口。"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.domain_catalog import build_tool_specs
from app.agent.kernel.adapters import consume_events
from app.agent.kernel.capabilities import (
    CapabilityRetriever,
    KernelToolSpec,
    ToolCatalog,
    ToolEffect,
)
from app.agent.kernel.effects import PreparedEffect
from app.agent.kernel.model import (
    ModelEvent,
    ModelEventType,
    ModelRequest,
    ModelToolCall,
)
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports import catalog_from_tool_specs
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import AgentInput, InMemorySessionStateStore
from app.agent.public_view import public_result_state

DEFAULT_FIXTURE = Path("tests/fixtures/agent_kernel_capability_cases.jsonl")
DEFAULT_LIFECYCLE_FIXTURE = Path("tests/fixtures/agent_kernel_lifecycle_cases.jsonl")


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    message: str
    required: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalOutcome:
    case_id: str
    passed: bool
    selected: tuple[str, ...]
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "selected": list(self.selected),
            "missing": list(self.missing),
        }



@dataclass(frozen=True, slots=True)
class LifecycleCase:
    case_id: str
    message: str
    result: dict[str, Any]
    followup: str
    expected: dict[str, Any]


class _ReplayModel:
    def __init__(self, case: LifecycleCase) -> None:
        self.requests: list[ModelRequest] = []
        self._rounds = [
            [
                ModelEvent(
                    ModelEventType.TOOL_CALL_COMPLETED,
                    tool_call=ModelToolCall("write", "cloud.lifecycle_write", {}),
                ),
                ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls"),
            ],
            [
                ModelEvent(ModelEventType.TEXT_DELTA, text=case.followup),
                ModelEvent(ModelEventType.FINISH, finish_reason="stop"),
            ],
        ]

    async def stream(
        self, request: ModelRequest, *, cancellation: Any
    ) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        if not self._rounds:
            raise AssertionError("生命周期回放出现了意外模型轮次")
        for event in self._rounds.pop(0):
            cancellation.raise_if_cancelled()
            await asyncio.sleep(0)
            yield event


def load_lifecycle_cases(path: Path) -> list[LifecycleCase]:
    cases: list[LifecycleCase] = []
    seen: set[str] = set()
    expected_fields = {
        "final_status",
        "result_state",
        "model_calls",
        "confirm_events",
        "answer_contains",
        "answer_excludes",
    }
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"生命周期语料第 {line_number} 行不是有效 JSON") from exc
        if not isinstance(raw, dict) or set(raw) != {
            "id", "message", "result", "followup", "expected"
        }:
            raise ValueError(f"生命周期语料第 {line_number} 行字段无效")
        case_id = str(raw["id"] or "").strip()
        message = str(raw["message"] or "").strip()
        followup = str(raw["followup"] or "").strip()
        result = raw["result"]
        expected = raw["expected"]
        if (
            not case_id
            or case_id in seen
            or not message
            or not followup
            or not isinstance(result, dict)
            or set(result) - {"ok", "status", "summary", "data", "error"}
            or type(result.get("ok")) is not bool
            or not str(result.get("status") or "").strip()
            or not str(result.get("summary") or "").strip()
            or not isinstance(expected, dict)
            or set(expected) != expected_fields
            or type(expected.get("model_calls")) is not int
            or expected["model_calls"] < 1
        ):
            raise ValueError(f"生命周期语料第 {line_number} 行案例无效")
        for key in ("confirm_events", "answer_contains", "answer_excludes"):
            values = expected.get(key)
            if not isinstance(values, list) or any(
                not isinstance(item, str) or not item for item in values
            ):
                raise ValueError(f"生命周期语料第 {line_number} 行 {key} 无效")
        seen.add(case_id)
        cases.append(LifecycleCase(case_id, message, dict(result), followup, dict(expected)))
    if not cases:
        raise ValueError("生命周期评测语料为空")
    return cases


async def _replay_lifecycle_case(case: LifecycleCase) -> dict[str, Any]:
    executions: list[dict[str, Any]] = []
    tool = KernelToolSpec(
        name="cloud.lifecycle_write",
        domain="cloud",
        description="离线验收确认生命周期",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        effect=ToolEffect.WRITE,
        prepare=lambda _arguments, _context: PreparedEffect(
            preview={"summary": "确认后执行生命周期验收写操作"},
            snapshot_fingerprint="lifecycle-replay",
        ),
        execute_confirmed=lambda arguments, _snapshot, _context: (
            executions.append(dict(arguments)) or dict(case.result)
        ),
    )
    catalog = ToolCatalog([tool])
    state = InMemorySessionStateStore()
    model = _ReplayModel(case)
    session = AgentSession(
        model=model,
        catalog=catalog,
        retriever=CapabilityRetriever(minimum=1, maximum=1),
        pipeline=ToolPipeline(catalog=catalog, state_store=state),
        state_store=state,
    )
    owner = f"eval:{case.case_id}"
    session_id = f"lifecycle-{case.case_id}"
    preview = await consume_events(session.run(AgentInput(
        message=case.message, owner=owner, session_id=session_id, channel="eval",
    )))
    if preview.approval is None:
        return {"case_id": case.case_id, "passed": False, "errors": ["未生成确认计划"]}
    confirm_events = [
        event async for event in session.confirm(
            owner=owner, session_id=session_id, plan_id=preview.approval.plan_id, channel="eval",
        )
    ]

    async def stream() -> AsyncIterator[Any]:
        for event in confirm_events:
            yield event

    final = await consume_events(stream())
    current = await state.load(owner=owner, session_id=session_id)
    actual_events = [event.type.value for event in confirm_events]
    public_dump = json.dumps([event.to_dict() for event in confirm_events], ensure_ascii=False)
    expected = case.expected
    errors: list[str] = []
    checks = (
        (preview.status == "approval_required", "预览终态不是 approval_required"),
        (final.status == expected["final_status"], f"最终状态为 {final.status}"),
        (public_result_state(final.effect_result) == expected["result_state"], "结果语义不匹配"),
        (len(model.requests) == expected["model_calls"], f"模型调用 {len(model.requests)} 次"),
        (actual_events == expected["confirm_events"], f"确认事件序列为 {actual_events}"),
        (len(executions) == 1, f"副作用执行 {len(executions)} 次"),
        (not current.pending_effect_plan_id, "确认计划仍处于待处理状态"),
    )
    errors.extend(message for passed, message in checks if not passed)
    for text in expected["answer_contains"]:
        if text not in final.answer:
            errors.append(f"回答缺少：{text}")
    for text in expected["answer_excludes"]:
        if text in final.answer or text in public_dump:
            errors.append(f"公开结果泄漏或误报：{text}")
    return {
        "case_id": case.case_id,
        "passed": not errors,
        "errors": errors,
        "events": actual_events,
        "final_status": final.status,
        "result_state": public_result_state(final.effect_result),
        "model_calls": len(model.requests),
    }


def evaluate_lifecycle(path: Path = DEFAULT_LIFECYCLE_FIXTURE) -> dict[str, Any]:
    cases = load_lifecycle_cases(path)

    async def run() -> list[dict[str, Any]]:
        return [await _replay_lifecycle_case(case) for case in cases]

    outcomes = asyncio.run(run())
    passed = sum(bool(item["passed"]) for item in outcomes)
    return {
        "ok": passed == len(outcomes),
        "summary": {"cases": len(outcomes), "passed": passed, "failed": len(outcomes) - passed},
        "outcomes": outcomes,
    }

def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_number} 行不是有效 JSON") from exc
        if not isinstance(raw, dict) or set(raw) != {"id", "message", "required"}:
            raise ValueError(f"第 {line_number} 行字段无效")
        case_id = str(raw["id"] or "").strip()
        message = str(raw["message"] or "").strip()
        required_raw = raw["required"]
        if (
            not case_id
            or case_id in seen
            or not message
            or not isinstance(required_raw, list)
            or not required_raw
        ):
            raise ValueError(f"第 {line_number} 行案例无效")
        required = tuple(
            dict.fromkeys(str(item or "").strip() for item in required_raw)
        )
        if any(not item for item in required):
            raise ValueError(f"第 {line_number} 行 required 无效")
        seen.add(case_id)
        cases.append(EvalCase(case_id, message, required))
    if not cases:
        raise ValueError("评测语料为空")
    return cases


def evaluate(
    path: Path = DEFAULT_FIXTURE,
    lifecycle_path: Path = DEFAULT_LIFECYCLE_FIXTURE,
) -> dict[str, Any]:
    catalog = catalog_from_tool_specs(build_tool_specs())
    retriever = CapabilityRetriever()
    outcomes: list[EvalOutcome] = []
    for case in load_cases(path):
        selection = retriever.retrieve(
            case.message,
            catalog,
            context={
                "owner": "offline-eval",
                "session_id": "offline-eval-session",
                "channel": "eval",
                "reference_kinds": (),
            },
        )
        selected = selection.names
        missing = tuple(name for name in case.required if name not in selected)
        outcomes.append(EvalOutcome(case.case_id, not missing, selected, missing))

    tools = catalog.visible({})
    invalid_effect_tools: list[str] = []
    for tool in tools:
        if tool.effect is ToolEffect.READ:
            valid = (
                tool.read is not None
                and tool.prepare is None
                and tool.execute_confirmed is None
            )
        else:
            valid = (
                tool.read is None
                and tool.prepare is not None
                and tool.execute_confirmed is not None
            )
        if not valid:
            invalid_effect_tools.append(tool.name)

    candidate_counts = [len(item.selected) for item in outcomes]
    passed = sum(item.passed for item in outcomes)
    safety_ok = not invalid_effect_tools
    lifecycle = evaluate_lifecycle(lifecycle_path)
    return {
        "ok": passed == len(outcomes) and safety_ok and lifecycle["ok"],
        "summary": {
            "cases": len(outcomes),
            "passed": passed,
            "failed": len(outcomes) - passed,
            "retrieval_recall": round(passed / len(outcomes), 4),
            "candidate_count_min": min(candidate_counts),
            "candidate_count_max": max(candidate_counts),
            "candidate_count_mean": round(statistics.fmean(candidate_counts), 2),
            "catalog_tools": len(tools),
            "read_tools": sum(tool.effect is ToolEffect.READ for tool in tools),
            "effect_tools": sum(tool.effect is not ToolEffect.READ for tool in tools),
            "effect_gate_valid": safety_ok,
            "lifecycle_cases": lifecycle["summary"]["cases"],
            "lifecycle_passed": lifecycle["summary"]["passed"],
            "lifecycle_failed": lifecycle["summary"]["failed"],
        },
        "invalid_effect_tools": invalid_effect_tools,
        "outcomes": [item.to_dict() for item in outcomes],
        "lifecycle_outcomes": lifecycle["outcomes"],
    }


def _text_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "MediaFlux Agent Kernel 离线验收",
        f"语料：{summary['passed']}/{summary['cases']} 通过，召回率 {summary['retrieval_recall']:.2%}",
        (
            "候选工具："
            f"{summary['candidate_count_min']}-{summary['candidate_count_max']} 项，"
            f"平均 {summary['candidate_count_mean']} 项"
        ),
        (
            "能力目录："
            f"{summary['catalog_tools']} 项（READ {summary['read_tools']} / "
            f"Effect {summary['effect_tools']}）"
        ),
        "Effect Gate：通过" if summary["effect_gate_valid"] else "Effect Gate：失败",
        (
            "生命周期回放："
            f"{summary['lifecycle_passed']}/{summary['lifecycle_cases']} 通过"
        ),
    ]
    for outcome in result["outcomes"]:
        if not outcome["passed"]:
            lines.append(
                f"[FAIL] {outcome['case_id']} 缺少：{', '.join(outcome['missing'])}"
            )
    for outcome in result["lifecycle_outcomes"]:
        if not outcome["passed"]:
            lines.append(
                f"[FAIL] {outcome['case_id']}：{'；'.join(outcome['errors'])}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--lifecycle-fixture", type=Path, default=DEFAULT_LIFECYCLE_FIXTURE
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.fixture, args.lifecycle_fixture)
    except (OSError, ValueError) as exc:
        result = {"ok": False, "error": str(exc)}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    else:
        print(
            _text_report(result)
            if "summary" in result
            else f"Agent Kernel 验收失败：{result['error']}"
        )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
