"""Single signed-session READ wait reducer; no scheduler or second task ledger."""
from __future__ import annotations

import hashlib
import json
import math
import secrets
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from app.agent.feature_gate import RuntimeFence, require_runtime_fence
from .conversation_codec import conversation_payload, json_text
from .state import PublicationLease, SessionState, StateUpdate, StalePublicationError
from .task_state import identity

MAX_WAIT_SEGMENTS = 3
MAX_WAIT_SECONDS = 180
EXPIRY_SECONDS = 300
CLAIM_SECONDS = 30
ACTIVE_SECONDS = 60
SCAN_LIMIT = 16
ACTIVE = {"pending", "claimed", "running"}


@dataclass(frozen=True, slots=True)
class _ReadWaitAction(StateUpdate):
    """Only the Session executor builds these; never decoded from tool output."""
    binding: Any = None


def digest(rows):
    return hashlib.sha256(json.dumps(json.loads(json_text(conversation_payload(rows))), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def anchor_for(state, call_id, tool):
    matches = []
    for index, row in enumerate(state.conversation):
        if any(c.get("call_id") == call_id for c in row.get("tool_calls", ())):
            calls = row["tool_calls"]
            group = state.conversation[index:index + 1 + len(calls)]
            if len(group) != len(calls) + 1 or {r.get("tool_call_id") for r in group[1:]} != {c["call_id"] for c in calls}:
                raise ValueError("read wait requires a closed native group")
            user = next((r for r in reversed(state.conversation[:index]) if r.get("role") == "user"), None)
            if user is not None:
                matches.append((group, user))
    if len(matches) != 1:
        raise ValueError("read wait history unavailable")
    group, user = matches[0]
    return {"group_digest": digest(group), "user_digest": digest([user]), "call_id": call_id, "tool": tool}


def history_call(state, record):
    anchor = record["anchor"]
    if anchor_for(state, anchor["call_id"], anchor["tool"]) != anchor:
        raise ValueError("read wait history changed")
    for row in state.conversation:
        for call in row.get("tool_calls", ()):
            if call["call_id"] == anchor["call_id"]:
                return deepcopy(call)
    raise ValueError("read wait history missing")


def check_claim(state, lease, now, conn=None, *, publication=False):
    if not lease.read_wait_id:
        return
    record = state.metadata.get("read_wait") or {}
    if (record.get("wait_id") != lease.read_wait_id
            or record.get("claim_id") != lease.read_claim_id
            or record.get("resume_turn_id") != lease.turn_id
            or record.get("publication_generation") != state.generation
            or record.get("expected_task") != identity(state.metadata.get("task"))
            or record.get("state") not in ({"claimed", "running", "pending", "terminal"} if publication else {"claimed", "running"})
            or now < record.get("last_checked_at", now)
            or now >= record.get("expires_at", 0)):
        raise StalePublicationError("read wait claim revoked")
    if conn is not None:
        require_runtime_fence(conn, RuntimeFence(**record["admission_fence"]))


def revoke(state, reason):
    record = state.metadata.get("read_wait")
    if isinstance(record, dict) and record.get("state") in ACTIVE:
        record.update(state="cancelled" if reason == "cancelled" else "terminal", terminal_reason=reason)


def task_changed(state):
    record = state.metadata.get("read_wait")
    if not isinstance(record, dict):
        return
    task = state.metadata.get("task")
    if not task or record["task_id"] != task["task_id"] or task["status"] in {"cancelled", "complete"}:
        revoke(state, "task_changed")
    elif record.get("expected_task") and record["expected_task"]["epoch"] != task["epoch"]:
        revoke(state, "task_changed")
    record["expected_task"] = identity(task)


def apply_actions(state, lease, updates, now, conn=None):
    for update in updates:
        if type(update) is not _ReadWaitAction:
            continue
        value = update.value
        operation = value["operation"]
        if conn is not None:
            require_runtime_fence(conn, update.binding.fence)
        task = state.metadata.get("task")
        record = state.metadata.get("read_wait")
        if operation == "seed":
            if not task or (record and record["task_id"] == task["task_id"]):
                raise ValueError("read wait budget cannot be reseeded")
            state.metadata["read_wait"] = {
                "v": 1, "task_id": task["task_id"], "expected_task": identity(task),
                "state": "idle", "terminal_reason": "", "wait_id": "",
                "owner": state.owner, "session_id": state.session_id,
                "budget": {"model_limit": value["model_limit"], "tool_limit": value["tool_limit"],
                           "model_spent": value["model_spent"], "tool_spent": value["tool_spent"],
                           "wait_segments": 0, "wait_seconds_spent": 0},
            }
            continue
        if not record or not task or record["task_id"] != task["task_id"]:
            raise ValueError("trusted task budget unavailable")
        budget = record["budget"]
        if operation == "reserve":
            budget["model_limit"] = min(budget["model_limit"], value["model_limit"])
            budget["tool_limit"] = min(budget["tool_limit"], value["tool_limit"])
            if value.get("require_tool") and budget["tool_spent"] >= budget["tool_limit"]:
                raise ValueError("cumulative task tool budget exhausted")
            for counter, limit in (("model_spent", "model_limit"), ("tool_spent", "tool_limit")):
                amount = value.get(counter, 0)
                if type(amount) is not int or amount < 0 or budget[counter] + amount > budget[limit]:
                    raise ValueError("cumulative task budget exhausted")
                budget[counter] += amount
            if lease.read_wait_id:
                record["state"] = "running"
                record["last_checked_at"] = now
            continue
        if operation == "park":
            permit = value["permit"]
            from .pipeline import _ReadPermitRejected
            if not isinstance(permit, _ReadPermitRejected) or permit.rate_limit["reason"] != "quota":
                raise ValueError("untrusted READ rejection")
            due = permit.rate_limit["retry_at_epoch"]
            if (type(due) not in (float, int) or not math.isfinite(due) or due < now
                    or state.pending_effect_plan_id or task["status"] in {"complete", "cancelled"}
                    or budget["wait_segments"] >= MAX_WAIT_SEGMENTS
                    or budget["model_spent"] >= budget["model_limit"]
                    or budget["tool_spent"] >= budget["tool_limit"]
                    or due - now + budget["wait_seconds_spent"] > MAX_WAIT_SECONDS):
                raise ValueError("read wait budget or eligibility rejected")
            if record["state"] in ACTIVE and not lease.read_wait_id:
                raise ValueError("another read wait is active")
            anchor = anchor_for(state, value["call_id"], permit.tool_name)
            first = record.get("first_park_at", now)
            expires = record.get("expires_at", first + EXPIRY_SECONDS)
            if due >= expires:
                raise ValueError("read wait expired")
            record.update(state="pending", terminal_reason="", wait_id=record["wait_id"] if lease.read_wait_id else secrets.token_urlsafe(18),
                          anchor=anchor, admission_fence=asdict(update.binding.fence),
                          effective_digest=update.binding.effective_digest,
                          model_policy=value["model_policy"],
                          channel=value["channel"], source_request_id=lease.request_id,
                          source_turn_id=record.get("source_turn_id", lease.turn_id), publication_generation=lease.generation,
                          first_park_at=first, parked_at=now, due_at=due, expires_at=expires,
                          last_checked_at=now, claim_id=lease.read_claim_id, claim_until=0,
                          resume_turn_id=lease.turn_id if lease.read_wait_id else "",
                          expected_task=identity(task))
            budget["wait_segments"] += 1
        elif operation == "finish":
            record.update(state="terminal", terminal_reason=value["reason"])
        else:
            raise ValueError("invalid read wait operation")


def claim(state: SessionState, candidate, now, binding, conn=None):
    record = state.metadata.get("read_wait") or {}
    same_candidate = all(record.get(key) == candidate.get(key) for key in (
        "wait_id", "publication_generation", "expected_task", "claim_id", "state"))
    if (record.get("wait_id") != candidate.get("wait_id")
            or record.get("publication_generation") != state.generation
            or record.get("expected_task") != identity(state.metadata.get("task"))
            or record.get("state") not in {"pending", "claimed"}
            or (record["state"] == "claimed" and now < record["claim_until"])):
        return None
    # An expired claim may be retried from its original scan only while the
    # same native parked segment has never started. Repark changes the anchor
    # and segment count; claim/reclaim leaves that segment identity intact.
    reclaimable = (record["state"] == "claimed"
                   and candidate.get("state") in {"pending", "claimed"}
                   and all(record.get(key) == candidate.get(key) for key in (
                       "wait_id", "expected_task", "anchor"))
                   and record["budget"]["wait_segments"] == candidate.get("budget", {}).get("wait_segments"))
    if not same_candidate and not reclaimable:
        return None
    confirmed = state.metadata.get("confirmed_publication") or {}
    if state.pending_effect_plan_id or confirmed.get("generation") == state.generation:
        if same_candidate:
            revoke(state, "effect_conflict")
        return None
    if record["admission_fence"] != asdict(binding.fence) or record["effective_digest"] != binding.effective_digest:
        if same_candidate:
            revoke(state, "runtime_changed")
        return None
    if conn is not None:
        require_runtime_fence(conn, binding.fence)
    spent = record["budget"]["wait_seconds_spent"] + max(0, now - record["parked_at"])
    if now < record["last_checked_at"] or now >= record["expires_at"] or spent >= MAX_WAIT_SECONDS:
        if same_candidate:
            revoke(state, "expired")
        return None
    if now < record["due_at"]:
        return None
    try:
        history_call(state, record)
    except ValueError:
        if same_candidate:
            revoke(state, "history_unavailable")
        return None
    state.generation += 1
    record.update(state="claimed", claim_id=secrets.token_urlsafe(18), claim_until=now + CLAIM_SECONDS,
                  resume_turn_id=secrets.token_urlsafe(12), publication_generation=state.generation,
                  parked_at=now, last_checked_at=now)
    record["budget"]["wait_seconds_spent"] = spent
    state.metadata["active_publication"] = {"generation": state.generation,
        "request_id": record["source_request_id"], "status": "running"}
    return PublicationLease(state.owner, state.session_id, state.generation, record["resume_turn_id"],
                            record["source_request_id"], record["wait_id"], record["claim_id"])
