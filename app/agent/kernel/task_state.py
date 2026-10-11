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

from app.agent.public_safety import sanitize_public_text

COMPONENTS = frozenset({"tmdb", "qbittorrent"})
IDENTITY = ("task_id", "epoch", "revision")


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
    if type(value) is not str or not value or len(value) > maximum:
        raise ValueError("invalid task text")
    if not sanitize_public_text(value, limit=maximum):
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
    keys = {"operation", "expected"}
    if operation in {"create", "revise"}:
        keys |= {"source", "root_goal", "constraints", "obligations"}
    elif operation == "cancel":
        keys |= {"source"}
    elif operation == "checkpoint":
        keys |= {"evidence_ids", "blockers"}
    else:
        raise ValueError("invalid task operation")
    _object(value, keys)
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
    component: str,
    fact: dict | None,
) -> TaskPatch | None:
    if not task or task["status"] == "cancelled" or component not in COMPONENTS:
        return None
    payload: dict = {"component": component, "fact": None}
    if fact is not None:
        payload["fact"] = {
            **fact,
            "evidence_id": secrets.token_urlsafe(18),
            "task_id": task["task_id"],
            "epoch": task["epoch"],
            "tool": tool,
            "call_id": call_id,
            "verifier_version": "configuration-v1",
            "observed_at": time.time(),
            "scope": {
                "owner": owner,
                "session_id": session_id,
                "task_id": task["task_id"],
                "epoch": task["epoch"],
                "component": component,
            },
        }
    return TaskPatch(identity(task), "observe", payload, "", "")


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
        component = args["component"]
        marker = {
            "origin": "query",
            "component": component,
            "epoch": task["epoch"],
            "text": "query:" + component,
        }
        if args["fact"] is None:
            if marker not in task["blockers"]:
                task["blockers"].append(marker)
        else:
            if not valid_fact(args["fact"], task, owner=owner, session_id=session_id):
                raise ValueError("invalid observation evidence")
            task["verified_facts"].append(deepcopy(args["fact"]))
            task["blockers"] = [
                b
                for b in task["blockers"]
                if not (
                    b["origin"] == "query"
                    and b["component"] == component
                    and b["epoch"] == task["epoch"]
                )
            ]
    elif operation == "checkpoint":
        validate_proposal(args)
        facts = {f["evidence_id"]: f for f in task["verified_facts"]}
        for evidence_id in args["evidence_ids"]:
            fact = facts.get(evidence_id)
            if not fact or not valid_fact(
                fact, task, owner=owner, session_id=session_id
            ):
                raise ValueError("invalid evidence scope")
            for obligation in task["obligations"]:
                if obligation["acceptance"] == {
                    "kind": "observed_component",
                    "component": fact["component"],
                }:
                    if evidence_id not in obligation["evidence_ids"]:
                        obligation["evidence_ids"].append(evidence_id)
        task["blockers"] = [
            b for b in task["blockers"] if b["origin"] == "query"
        ] + [{"origin": "reported", "text": text} for text in args["blockers"]]
    else:
        raise ValueError("invalid patch")
    if task["status"] != "cancelled":
        for obligation in task["obligations"]:
            facts = {f["evidence_id"]: f for f in task["verified_facts"]}
            for evidence_id in obligation["evidence_ids"]:
                fact = facts.get(evidence_id)
                if (
                    not fact
                    or not valid_fact(fact, task, owner=owner, session_id=session_id)
                    or obligation["acceptance"]
                    != {"kind": "observed_component", "component": fact["component"]}
                ):
                    raise ValueError("invalid linked evidence")
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


def task_tool():
    from .capabilities import KernelToolSpec, ToolEffect

    return KernelToolSpec(
        name="agent.task",
        domain="agent",
        effect=ToolEffect.READ,
        description=(
            "持久任务清单、检查点与恢复 / persistent task checklist checkpoint. "
            "Only TMDB/qBittorrent read-only configuration observations are verifiable; "
            "connectivity, repairs and other goals remain blocked. No write authority. "
            "create: exact current or historical original user source; revise: current user source only. "
            "root_goal, constraints, required obligations; "
            "checkpoint: registered evidence_ids; cancel: exact source. "
            "expected=null only for first create, otherwise exact committed task_id/epoch/revision."
        ),
        examples=("持久任务清单 检查点 剩余义务 恢复", "persistent task checklist checkpoint resume remaining obligations"),
        input_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["create", "checkpoint", "revise", "cancel"],
                },
                "expected": {
                    "description": "null for first create; otherwise {task_id:string,epoch:integer,revision:integer} exactly from committed task."
                },
                "source": {
                    "type": "string",
                    "maxLength": 2048,
                    "description": "Nonempty exact user quote; create also accepts preserved original user history, not legacy notes or tool/assistant text. Revise/cancel require current user text.",
                },
                "root_goal": {"type": "string", "maxLength": 2048},
                "constraints": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string", "maxLength": 512},
                },
                "obligations": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "required": ["id", "description", "required", "acceptance"],
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string", "maxLength": 64},
                            "description": {"type": "string", "maxLength": 512},
                            "required": {"type": "boolean", "enum": [True]},
                            "acceptance": {
                                "type": "object",
                                "description": "{kind:observed_component,component:tmdb|qbittorrent}; other goals use {kind:<name>,criterion:<text>} and remain blocked; never claim repairs or network reachability from configuration observations.",
                            },
                        },
                    },
                },
                "evidence_ids": {
                    "type": "array",
                    "maxItems": 32,
                    "items": {"type": "string", "maxLength": 64},
                },
                "blockers": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {"type": "string", "maxLength": 512},
                },
            },
            "required": ["operation", "expected"],
            "additionalProperties": False,
        },
        validator=validate_proposal,
        read=proposal,
        metadata={"kernel_utility": True, "query_discoverable": True},
    )
