"""Single signed-session task: strict proposals, exact CAS and evidence reduction.

Task declarations are data, never write authorization or a second execution loop.
"""

from __future__ import annotations

import json
import secrets
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from app.sensitive_data import contains_sensitive_credential

COMPONENTS = frozenset({"tmdb", "qbittorrent"})
IDENTITY = ("task_id", "epoch", "revision")
# 同一字段集合用于模型 Schema 和领域校验，不能再出现“模型可选、执行必填”。
_OPERATION_FIELDS = {
    "create": ("source", "root_goal", "constraints", "obligations"),
    "checkpoint": ("evidence_ids", "blockers"),
    "revise": ("source", "root_goal", "constraints", "obligations"),
    "cancel": ("source",),
}


def classify_public_task(task: Any = None) -> str:
    """Classify only the public consumer shape, not evidence or authorization."""
    if task is None:
        return "untracked"
    if not isinstance(task, dict):
        return "unknown"

    def text(value: Any) -> bool:
        return isinstance(value, str) and bool(value)

    if (
        type(task.get("v")) is not int or task["v"] != 1
        or not text(task.get("task_id"))
        or any(type(task.get(k)) is not int or task[k] <= 0 for k in ("epoch", "revision"))
        or task.get("acceptance_scope") != "declared_obligations"
        or task.get("status") not in ("partial", "blocked", "complete", "cancelled")
        or not text(task.get("root_goal"))
    ):
        return "unknown"
    obligations = task.get("obligations")
    remaining = task.get("remaining_obligation_ids")
    blockers = task.get("blockers")
    if not isinstance(obligations, list) or not obligations:
        return "unknown"
    ids: set[str] = set()
    pending: set[str] = set()
    for item in obligations:
        if (not isinstance(item, dict) or not text(item.get("id"))
            or not text(item.get("description")) or item.get("required") is not True
            or item.get("state") not in ("pending", "blocked", "verified")):
            return "unknown"
        if item["id"] in ids:
            return "unknown"
        ids.add(item["id"])
        if item["state"] != "verified":
            pending.add(item["id"])
    if (not isinstance(remaining, list) or not all(isinstance(v, str) for v in remaining)
        or len(set(remaining)) != len(remaining) or set(remaining) != pending
        or not isinstance(blockers, list)
        or any(not isinstance(b, dict) or not text(b.get("text")) for b in blockers)):
        return "unknown"
    if task["status"] == "complete" and (remaining or blockers):
        return "unknown"
    return task["status"]


class TaskConflict(ValueError):
    pass


def _object(value: Any, keys: set[str]) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("invalid task fields")
    return value


def _text(value: Any, maximum: int = 512) -> str:
    # 任务声明可引用用户的路径/链接和多行原话；不是对外工具摘要。
    # source是否来自获准用户仍由proposal核对，文字不产生文件操作授权。
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError("invalid task text")
    if (contains_sensitive_credential(value)
            or any((ord(char) < 32 and char not in "\n\r\t") or ord(char) == 127 for char in value)):
        raise ValueError("unsafe task text")
    return value


def _array(value: Any, maximum: int, minimum: int = 0) -> list:
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise ValueError("invalid task array")
    if len({json.dumps(v, sort_keys=True, ensure_ascii=False) for v in value}) != len(
        value
    ):
        raise ValueError("duplicate task item")
    return value


def identity(task: dict | None) -> dict | None:
    return {key: task[key] for key in IDENTITY} if task else None


def validate_proposal(value: dict) -> dict:
    operation = value.get("operation")
    if operation not in _OPERATION_FIELDS:
        raise ValueError("invalid task operation")
    _object(value, {"operation", "expected", *_OPERATION_FIELDS[operation]})
    expected = value["expected"]
    if expected is not None:
        _object(expected, set(IDENTITY))
        _text(expected["task_id"], 64)
        for key in ("epoch", "revision"):
            if type(expected[key]) is not int or expected[key] < 1:
                raise ValueError("invalid task version")
    elif operation != "create":
        raise ValueError("task identity required")
    if "source" in value:
        _text(value["source"], 2048)
    if operation in {"create", "revise"}:
        _text(value["root_goal"], 2048)
        for constraint in _array(value["constraints"], 8):
            _text(constraint)
        obligations = _array(value["obligations"], 16, 1)
        ids = []
        for obligation in obligations:
            _object(obligation, {"id", "description", "required", "acceptance"})
            ids.append(_text(obligation["id"], 64))
            _text(obligation["description"])
            if obligation["required"] is not True:
                raise ValueError("all M1 obligations must be required")
            acceptance = obligation["acceptance"]
            if type(acceptance) is not dict:
                raise ValueError("invalid acceptance")
            kind = _text(acceptance.get("kind"), 64)
            if kind == "observed_component":
                _object(acceptance, {"kind", "component"})
                if acceptance["component"] not in COMPONENTS:
                    raise ValueError("unsupported component verifier")
            else:
                _object(acceptance, {"kind", "criterion"})
                _text(acceptance["criterion"])
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate obligation id")
    if operation == "checkpoint":
        for evidence in _array(value["evidence_ids"], 32):
            _text(evidence, 64)
        for blocker in _array(value["blockers"], 16):
            _text(blocker)
    return deepcopy(value)


@dataclass(frozen=True, slots=True)
class TaskPatch:
    expected: dict | None
    operation: str
    payload: dict
    source_request_id: str
    source: str
    new_id: str = ""
    source_kind: str = "current_user"


def proposal(arguments: dict, context: Any) -> TaskPatch:
    args = validate_proposal(arguments)
    _text(context.request_id, 64)
    source = args.get("source", "")
    if source and source not in context.user_source:
        history = getattr(context, "user_history", ()) if args["operation"] == "create" else ()
        if not any(source in text for text in history):
            raise ValueError("task source must be current admitted user text or verified history for create")
    return TaskPatch(
        args["expected"],
        args["operation"],
        args,
        context.request_id,
        source,
        secrets.token_urlsafe(18),
        "current_user" if source in context.user_source else "historical_user_quote",
    )


def observation_patch(
    task: dict | None,
    *,
    owner: str,
    session_id: str,
    tool: str,
    call_id: str,
    observations: dict[str, dict | None] | None,
) -> TaskPatch | None:
    if not task or task["status"] == "cancelled" or not observations:
        return None
    facts = {}
    for component, fact in observations.items():
        if component not in COMPONENTS:
            continue
        facts[component] = None if fact is None else {
            **fact,
            "evidence_id": secrets.token_urlsafe(18),
            "task_id": task["task_id"], "epoch": task["epoch"],
            "tool": tool, "call_id": call_id,
            "verifier_version": "configuration-v1", "observed_at": time.time(),
            "scope": {"owner": owner, "session_id": session_id, "task_id": task["task_id"],
                      "epoch": task["epoch"], "component": component},
        }
    return TaskPatch(identity(task), "observe", {"observations": facts}, "", "") if facts else None


def valid_fact(fact: dict, task: dict, *, owner: str, session_id: str) -> bool:
    keys = {
        "component",
        "status",
        "enabled",
        "probe_mode",
        "network_accessed",
        "evidence_id",
        "task_id",
        "epoch",
        "tool",
        "call_id",
        "verifier_version",
        "observed_at",
        "scope",
    }
    if type(fact) is not dict or set(fact) != keys:
        return False
    for key in ("evidence_id", "task_id", "tool", "call_id", "verifier_version"):
        if type(fact[key]) is not str or not 1 <= len(fact[key]) <= 64:
            return False
    return (
        fact["component"] in COMPONENTS
        and fact["status"] in {"ready", "incomplete", "not_configured"}
        and type(fact["enabled"]) is bool
        and fact["network_accessed"] is False
        and fact["probe_mode"] == "configuration_only"
        and fact["verifier_version"] == "configuration-v1"
        and type(fact["observed_at"]) in (float, int)
        and type(fact["epoch"]) is int
        and fact["epoch"] == task["epoch"]
        and fact["task_id"] == task["task_id"]
        and fact["scope"]
        == {
            "owner": owner,
            "session_id": session_id,
            "task_id": task["task_id"],
            "epoch": task["epoch"],
            "component": fact["component"],
        }
    )


def reduce_task(
    current: dict | None, patch: TaskPatch, *, owner: str, session_id: str
) -> dict:
    if identity(current) != patch.expected:
        raise TaskConflict("task revision changed")
    operation, args = patch.operation, patch.payload
    checkpoint_evidence: dict[str, str] = {}
    if operation == "create":
        if current is not None and current["status"] != "cancelled":
            raise TaskConflict("cancel existing task before create")
        task = {"v": 1, "task_id": patch.new_id, "epoch": 1, "revision": 0}
    else:
        if current is None or current["status"] == "cancelled":
            raise TaskConflict("task is not active")
        task = deepcopy(current)
    if operation in {"create", "revise"}:
        validate_proposal(args)
        if operation == "revise":
            task["epoch"] += 1
        task.update(
            source_request_id=patch.source_request_id,
            source_quote=patch.source,
            source_kind=patch.source_kind,
            root_goal=args["root_goal"],
            constraints=deepcopy(args["constraints"]),
            verified_facts=[],
            blockers=[],
            status="partial",
            obligations=[
                {**deepcopy(o), "state": "pending", "evidence_ids": []}
                for o in args["obligations"]
            ],
        )
    elif operation == "cancel":
        task["status"] = "cancelled"
    elif operation == "observe":
        for component, fact in args["observations"].items():
            if component not in COMPONENTS:
                raise ValueError("unsupported observation component")
            marker = {"origin": "query", "component": component, "epoch": task["epoch"],
                      "text": "query:" + component}
            if fact is None:
                if marker not in task["blockers"]:
                    task["blockers"].append(marker)
            else:
                if fact.get("component") != component or not valid_fact(fact, task, owner=owner, session_id=session_id):
                    raise ValueError("invalid observation evidence")
                task["verified_facts"].append(deepcopy(fact))
                task["blockers"] = [b for b in task["blockers"] if not (
                    b["origin"] == "query" and b["component"] == component and b["epoch"] == task["epoch"])]
    elif operation == "checkpoint":
        validate_proposal(args)
        facts = {f["evidence_id"]: f for f in task["verified_facts"]}
        for evidence_id in args["evidence_ids"]:
            fact = facts.get(evidence_id)
            if not fact or not valid_fact(
                fact, task, owner=owner, session_id=session_id
            ):
                raise ValueError("invalid evidence scope")
        checkpoint_evidence = {
            fact["component"]: fact["evidence_id"]
            for fact in task["verified_facts"]
            if fact["evidence_id"] in args["evidence_ids"]
        }
        task["blockers"] = [
            b for b in task["blockers"] if b["origin"] == "query"
        ] + [{"origin": "reported", "text": text} for text in args["blockers"]]
    else:
        raise ValueError("invalid patch")
    if task["status"] != "cancelled":
        facts = {f["evidence_id"]: f for f in task["verified_facts"]}
        for obligation in task["obligations"]:
            for evidence_id in obligation["evidence_ids"]:
                fact = facts.get(evidence_id)
                if (
                    not fact
                    or not valid_fact(fact, task, owner=owner, session_id=session_id)
                    or obligation["acceptance"]
                    != {"kind": "observed_component", "component": fact["component"]}
                ):
                    raise ValueError("invalid linked evidence")
            # 先完整校验旧关联，再按登记顺序规范化；不能自动关联新观察。
            existing = set(obligation["evidence_ids"])
            obligation["evidence_ids"] = [eid for eid in facts if eid in existing][-1:]
            if obligation["acceptance"]["kind"] == "observed_component":
                evidence_id = checkpoint_evidence.get(obligation["acceptance"]["component"])
                if evidence_id:
                    obligation["evidence_ids"] = [evidence_id]
            obligation["state"] = (
                "blocked"
                if obligation["acceptance"]["kind"] != "observed_component"
                else "verified"
                if obligation["evidence_ids"]
                else "pending"
            )
        task["status"] = (
            "blocked"
            if task["blockers"]
            or any(o["state"] == "blocked" for o in task["obligations"])
            else "complete"
            if all(o["state"] == "verified" for o in task["obligations"])
            else "partial"
        )
        # 状态快照不是审计日志：保留所有义务引用和每个组件最新登记的观察。
        linked = {eid for o in task["obligations"] for eid in o["evidence_ids"]}
        latest = {f["component"]: f["evidence_id"] for f in task["verified_facts"]}
        task["verified_facts"] = [
            f for f in task["verified_facts"]
            if f["evidence_id"] in linked or f["evidence_id"] == latest[f["component"]]
        ]
    task["revision"] += 1
    _array(task["verified_facts"], 32)
    _array(task["blockers"], 16)
    if (
        len(json.dumps(task, ensure_ascii=False, separators=(",", ":")).encode())
        > 16 * 1024
    ):
        raise ValueError("task capacity exceeded")
    return task


def task_projection(task: dict | None) -> dict | None:
    if task is None:
        return None
    result = deepcopy(task)
    for fact in result["verified_facts"]:
        fact.pop("scope", None)
        fact.pop("call_id", None)
    result["acceptance_scope"] = "declared_obligations"
    result["remaining_obligation_ids"] = [
        o["id"] for o in result["obligations"] if o["state"] != "verified"
    ]
    linked = {eid for o in result["obligations"] for eid in o["evidence_ids"]}
    result["unassociated_evidence_ids"] = [
        f["evidence_id"]
        for f in result["verified_facts"]
        if f["evidence_id"] not in linked
    ]
    return result


def _task_input_schema() -> dict:
    text = {"type": "string", "minLength": 1}
    properties = {
        "operation": {"type": "string", "enum": list(_OPERATION_FIELDS)},
        "expected": {
            "type": ["object", "null"],
            "properties": {
                "task_id": {**text, "maxLength": 64},
                **{key: {"type": "integer", "minimum": 1} for key in ("epoch", "revision")},
            },
            "required": list(IDENTITY), "additionalProperties": False,
            "description": "null only for first create; otherwise exact committed task_id/epoch/revision.",
        },
        "source": {
            **text, "maxLength": 2048,
            "description": "Exact user quote. Create accepts original user history; revise/cancel require current user text.",
        },
        "root_goal": {**text, "maxLength": 2048},
        "constraints": {
            "type": "array", "maxItems": 8, "uniqueItems": True,
            "items": {**text, "maxLength": 512},
        },
        "obligations": {
            "type": "array", "minItems": 1, "maxItems": 16, "uniqueItems": True,
            "items": {
                "type": "object", "required": ["id", "description", "required", "acceptance"],
                "additionalProperties": False,
                "properties": {
                    "id": {**text, "maxLength": 64},
                    "description": {**text, "maxLength": 512},
                    "required": {"type": "boolean", "enum": [True]},
                    "acceptance": {
                        "type": "object",
                        "description": "Observed_component requires an observation, even incomplete/not_configured satisfies it. It does not require fixing configuration or proving connectivity. Other goals remain blocked.",
                        "oneOf": [
                            {
                                "properties": {
                                    "kind": {"enum": ["observed_component"]},
                                    "component": {"type": "string", "enum": sorted(COMPONENTS)},
                                },
                                "required": ["kind", "component"], "additionalProperties": False,
                            },
                            {
                                "properties": {
                                    "kind": {**text, "maxLength": 64, "not": {"enum": ["observed_component"]}},
                                    "criterion": {**text, "maxLength": 512},
                                },
                                "required": ["kind", "criterion"], "additionalProperties": False,
                            },
                        ],
                    },
                },
            },
        },
        "evidence_ids": {
            "type": "array", "maxItems": 32, "uniqueItems": True,
            "items": {**text, "maxLength": 64},
        },
        "blockers": {
            "description": "Unresolved work in the requested task. Discovered configuration issues do not block an observation-only goal once observed.",
            "type": "array", "maxItems": 16, "uniqueItems": True,
            "items": {**text, "maxLength": 512},
        },
    }
    return {
        "type": "object", "properties": properties,
        "required": ["operation", "expected"], "additionalProperties": False,
        "oneOf": [
            {
                "properties": {
                    **{key: {} for key in ("expected", *fields)},
                    "operation": {"enum": [operation]},
                    **({"expected": {"type": "object"}} if operation != "create" else {}),
                },
                "required": ["operation", "expected", *fields],
                "additionalProperties": False,
            }
            for operation, fields in _OPERATION_FIELDS.items()
        ],
    }


def task_tool():
    from .capabilities import KernelToolSpec, ToolEffect

    return KernelToolSpec(
        name="agent.task",
        domain="agent",
        effect=ToolEffect.READ,
        description=(
            "持久任务清单、检查点与恢复 / persistent task checklist checkpoint. "
            "TMDB/qB configuration observations from config.diagnose or config.explain_component are verifiable, "
            "not connectivity/repairs. No write authority. "
            "create accepts exact current/historical user source; revise/cancel require current user source. "
            "checkpoint uses task.verified_facts evidence_ids (not artifact refs), plus blockers. "
            "expected=null only for first create, otherwise exact task_id/epoch/revision."
        ),
        examples=("持久任务清单 检查点 剩余义务 恢复", "persistent task checklist checkpoint resume remaining obligations"),
        input_schema=_task_input_schema(),
        validator=validate_proposal,
        read=proposal,
        metadata={"kernel_utility": True, "query_discoverable": True},
    )
