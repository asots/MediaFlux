"""签名会话续行：READ额度恢复与已确认后台结果共用领取、预算和执行租约。"""
from __future__ import annotations

import hashlib
import json
import math
import secrets
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from app.agent.errors import _ReadPermitRejected
from app.agent.feature_gate import RuntimeFence, require_runtime_fence

from .batch_progress import observe_batch
from .completion_receipts import EFFECT_PENDING, _completion_cipher, _late_receipt
from .conversation_codec import conversation_payload, json_text
from .conversation_state import commit_conversation
from .session_state import SessionState
from .state_types import PublicationLease, StalePublicationError, StateUpdate
from .task_state import identity
from .tool_calls import PipelineResult

MAX_WAIT_SEGMENTS = 3
MAX_WAIT_SECONDS = 180
EXPIRY_SECONDS = 300
CLAIM_SECONDS = 30
ACTIVE_SECONDS = 60
SCAN_LIMIT = 16
ACTIVE = {"pending", "claimed", "running"}


@dataclass(frozen=True, slots=True)
class _ContinuationAction(StateUpdate):
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
    if not lease.continuation_id:
        return
    record = continuation_record(state, lease.continuation_id, lease.continuation_kind)
    if (record.get("wait_id") != lease.continuation_id
            or record.get("claim_id") != lease.continuation_claim_id
            or record.get("resume_turn_id") != lease.turn_id
            or record.get("publication_generation") != state.generation
            or (lease.continuation_kind == "read_quota" and record.get("expected_task") != identity(state.metadata.get("task")))
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
    for row in (state.metadata.get("effect_waits") or {}).values():
        continuation = row.get("continuation") or {}
        if continuation.get("state") not in EFFECT_PENDING:
            continue
        if reason == "cancelled":
            continuation.update(state="cancelled", terminal_reason=reason)
        elif reason == "superseded" and continuation.get("state") in {"claimed", "running"}:
            continuation.update(state="pending", claim_id="", claim_until=0, resume_turn_id="")


def task_changed(state):
    record = state.metadata.get("read_wait")
    if not isinstance(record, dict):
        return
    task = state.metadata.get("task")
    # 完成业务清单不撤销当前执行租约；由同一终结事件保存最终回答/投递。
    if (not task or record["task_id"] != task["task_id"] or task["status"] == "cancelled"
            or (task["status"] == "complete" and record.get("state") != "running")):
        revoke(state, "task_changed")
    elif record.get("expected_task") and record["expected_task"]["epoch"] != task["epoch"]:
        revoke(state, "task_changed")
    record["expected_task"] = identity(task)


def interrupt_read_wait(state: SessionState, now: float) -> bool:
    """调用方已独占续行 OS 锁；开始过的 READ 只能收束，不能重放。"""
    record = state.metadata.get("read_wait") or {}
    if record.get("state") != "running":
        return False
    record.update(state="terminal", terminal_reason="interrupted", claim_id="",
                  claim_until=0, last_checked_at=max(now, record.get("last_checked_at", now)))
    marker = state.metadata.get("active_publication") or {}
    if (record.get("publication_generation") == state.generation
            and marker.get("generation") == state.generation
            and marker.get("request_id") == record.get("source_request_id")):
        marker["status"] = "finished"
    return True


def apply_actions(state, lease, updates, now, conn=None):
    for update in updates:
        if type(update) is not _ContinuationAction:
            continue
        value = update.value
        operation = value["operation"]
        if conn is not None:
            require_runtime_fence(conn, update.binding.fence)
        if operation == "finish":
            record = continuation_record(state, lease.continuation_id, lease.continuation_kind)
            record.update(state="terminal", terminal_reason=value["reason"])
            if lease.continuation_kind == "effect":
                row = state.metadata["effect_waits"][record["wait_id"]]
                if row.get("batch_progress"):
                    payload = effect_payload(state, record)
                    final = observe_batch(row, payload["origin"], payload["result"])
                    row.update(final_result=final, receipt_message=_late_receipt(payload["seed"], final))
                    commit_conversation(state, state.conversation)
            if value.get("delivery_view") is not None and record.get("channel") == "telegram":
                record.update(delivery_view=deepcopy(value["delivery_view"]), delivery_pending=True)
            marker = state.metadata.get("active_publication") or {}
            if marker.get("generation") == lease.generation and marker.get("request_id") == lease.request_id:
                marker["status"] = "finished"
            continue
        if lease.continuation_kind == "effect" and lease.continuation_id:
            apply_effect_action(state, lease, update, now)
            continue
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
            for counter, limit in (("model_spent", "model_limit"), ("tool_spent", "tool_limit")):
                amount = value.get(counter, 0)
                if type(amount) is not int or amount < 0 or budget[counter] + amount > budget[limit]:
                    raise ValueError("cumulative task budget exhausted")
                budget[counter] += amount
            if lease.continuation_id:
                record["state"] = "running"
                record["last_checked_at"] = now
            continue
        if operation == "park":
            permit = value["permit"]
            if not isinstance(permit, _ReadPermitRejected) or permit.rate_limit["reason"] != "quota":
                raise ValueError("untrusted READ rejection")
            due = permit.rate_limit["retry_at_epoch"]
            if ((type(due) is not float and type(due) is not int) or not math.isfinite(due) or due < now
                    or state.pending_effect_plan_id or task["status"] in {"complete", "cancelled"}
                    or budget["wait_segments"] >= MAX_WAIT_SEGMENTS
                    or budget["model_spent"] >= budget["model_limit"]
                    or budget["tool_spent"] >= budget["tool_limit"]
                    or due - now + budget["wait_seconds_spent"] > MAX_WAIT_SECONDS):
                raise ValueError("read wait budget or eligibility rejected")
            if record["state"] in ACTIVE and not lease.continuation_id:
                raise ValueError("another read wait is active")
            anchor = anchor_for(state, value["call_id"], permit.tool_name)
            from app.agent.owner_routes import normalize_delivery_route
            route = normalize_delivery_route(value["delivery_route"], channel=value["channel"], owner=state.owner)
            first = record.get("first_park_at", now)
            expires = record.get("expires_at", first + EXPIRY_SECONDS)
            if due >= expires:
                raise ValueError("read wait expired")
            record.update(state="pending", terminal_reason="", wait_id=record["wait_id"] if lease.continuation_id else secrets.token_urlsafe(18),
                          anchor=anchor, admission_fence=asdict(update.binding.fence),
                          effective_digest=update.binding.effective_digest,
                          model_policy=value["model_policy"],
                          channel=value["channel"], delivery_route=route, source_request_id=lease.request_id,
                          source_turn_id=record.get("source_turn_id", lease.turn_id), publication_generation=lease.generation,
                          first_park_at=first, parked_at=now, due_at=due, expires_at=expires,
                          last_checked_at=now, claim_id=lease.continuation_claim_id, claim_until=0,
                          resume_turn_id=lease.turn_id if lease.continuation_id else "",
                          expected_task=identity(task))
            budget["wait_segments"] += 1
        else:
            raise ValueError("invalid read wait operation")


def claim(state: SessionState, candidate, now, binding, conn=None):
    if candidate.get("kind") == "effect":
        return claim_effect(state, candidate, now, binding, conn)
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


# Accepted writes are never resubmitted. A frozen batch may finish its approved
# synchronous tail; other continuations can only plan new work with fresh consent.
EFFECT_ACTIVE_SECONDS = 1800


def continuation_record(state, wait_id, kind):
    if kind == "effect":
        row = (state.metadata.get("effect_waits") or {}).get(wait_id) or {}
        return row.get("continuation") or {}
    return state.metadata.get("read_wait") or {}



def effect_origin(state, context, plan):
    binding = context.continuation_binding
    policy = context.continuation_policy
    goal = state.metadata.get("pending_effect_goal")
    if (binding is None or not policy or not isinstance(goal, dict)
            or (context.channel == "telegram" and not context.delivery_route)
            or goal.get("role") != "user" or not str(goal.get("content") or "").strip()):
        return None
    return {"goal": deepcopy(goal), "arguments": dict(plan.arguments),
            **({"reobservation_source": deepcopy(plan.metadata["reobservation_source"])}
               if plan.metadata.get("reobservation_source") is not None else {}),
            "owner": state.owner, "session_id": state.session_id,
            "source_request_id": context.lease.request_id, "source_turn_id": context.lease.turn_id,
            "origin_generation": context.lease.generation, "expected_task": identity(state.metadata.get("task")),
            "admission_fence": asdict(binding.fence), "effective_digest": binding.effective_digest,
            "model_policy": policy["model_policy"], "model_limit": policy["model_limit"],
            "tool_limit": policy["tool_limit"], "channel": context.channel,
            "delivery_route": dict(context.delivery_route)}


def register_effect(row, scope, now):
    origin = scope.origin
    if not origin or "continuation" in row:
        return
    row["continuation"] = {key: deepcopy(origin[key]) for key in (
        "owner", "session_id", "source_request_id", "source_turn_id", "origin_generation",
        "expected_task", "admission_fence", "effective_digest", "model_policy", "channel")}
    row["continuation"].update(kind="effect", wait_id=scope.plan_id, state="waiting", terminal_reason="",
        due_at=now, expires_at=row["expires_at"], last_checked_at=now,
        publication_generation=scope.lease.generation, claim_id="", claim_until=0, resume_turn_id="",
        budget={"model_limit": origin["model_limit"], "tool_limit": origin["tool_limit"],
                "model_spent": 0, "tool_spent": 0, "wait_segments": 0, "wait_seconds_spent": 0})


def release_effect(row, now):
    record = row.get("continuation")
    if record and record["state"] == "waiting" and isinstance(row.get("final_result"), dict):
        record.update(state="pending", due_at=now, last_checked_at=now)


def due_records(metadata, now):
    records = []
    candidates = [metadata.get("read_wait") or {}, *((row.get("continuation") or {})
                  for row in (metadata.get("effect_waits") or {}).values())]
    for record in candidates:
        if record.get("due_at", float("inf")) > now:
            continue
        # READ running 不可重放；effect 的冻结尾部保留既有租约恢复规则。
        resumable = {"pending", "claimed", "running"} if record.get("kind") == "effect" else {"pending", "claimed"}
        if record.get("delivery_pending") is True or (record.get("state") in resumable and record.get("claim_until", 0) <= now):
            records.append(deepcopy(record))
    return sorted(records, key=lambda r: r["due_at"])


def effect_payload(state, record):
    # The encrypted admission seed, not a chat summary or model-supplied goal.
    row = (state.metadata.get("effect_waits") or {}).get(record["wait_id"])
    if not row or not isinstance(row.get("final_result"), dict):
        raise ValueError("effect completion unavailable")
    seed = json.loads(_completion_cipher().decrypt(row["sealed"].encode()))
    origin = seed.get("origin") or {}
    if (seed.get("owner"), seed.get("session_id"), seed.get("plan_id")) != (
            state.owner, state.session_id, record["wait_id"]):
        raise ValueError("effect continuation scope mismatch")
    if any(record.get(key) != origin.get(key) for key in (
            "owner", "session_id", "source_request_id", "source_turn_id", "origin_generation",
            "expected_task", "admission_fence", "effective_digest", "model_policy", "channel")):
        raise ValueError("effect continuation origin changed")
    return {"seed": seed, "origin": origin, "result": deepcopy(row["final_result"])}


def claim_effect(state, candidate, now, binding, conn=None):
    row = (state.metadata.get("effect_waits") or {}).get(candidate.get("wait_id")) or {}
    record = row.get("continuation") or {}
    if (not record or record.get("state") not in {"pending", "claimed", "running"}
            or any(record.get(key) != candidate.get(key) for key in (
                "wait_id", "claim_id", "state", "publication_generation"))
            or record.get("claim_until", 0) > now or record["due_at"] > now):
        return None
    stopped_generation = state.metadata.get("stop_requested_generation")
    if type(stopped_generation) is int and record["origin_generation"] <= stopped_generation <= state.generation:
        record.update(state="cancelled", terminal_reason="cancelled")
        return None
    if (record["admission_fence"] != asdict(binding.fence) or record["effective_digest"] != binding.effective_digest
            or now < record["last_checked_at"] or now >= record["expires_at"]):
        record.update(state="terminal", terminal_reason="expired_or_runtime_changed")
        return None
    if conn is not None:
        require_runtime_fence(conn, binding.fence)
    if state.pending_effect_plan_id:
        return None  # Retain this goal, never overwrite the user's newer confirmation.
    publication = state.metadata.get("active_publication") or {}
    if (publication.get("generation") == state.generation and publication.get("status") == "running"
            and publication.get("request_id") != record["source_request_id"]):
        return None  # A newer user turn is still working; defer without discarding the old goal.
    task = state.metadata.get("task")
    expected = record.get("expected_task")
    if task and expected and task["task_id"] == expected["task_id"] and (
            task["epoch"] != expected["epoch"] or task["status"] == "cancelled"):
        record.update(state="cancelled", terminal_reason="goal_changed")
        return None
    from cryptography.fernet import InvalidToken
    try:
        payload = effect_payload(state, record)
        # Confirmation has its own request ID within the same user generation.
        # Only a newer publication supersedes the frozen tail authorization.
        if payload["origin"].get("batch_plan") and state.generation != record["publication_generation"]:
            record.update(state="cancelled", terminal_reason="new_user_turn")
            return None
    except (InvalidToken, ValueError, KeyError, TypeError):
        record.update(state="cancelled", terminal_reason="origin_unavailable")
        return None
    state.generation += 1
    record.update(state="claimed", claim_id=secrets.token_urlsafe(18), claim_until=now + CLAIM_SECONDS,
        resume_turn_id=secrets.token_urlsafe(12), publication_generation=state.generation, last_checked_at=now)
    state.metadata["active_publication"] = {"generation": state.generation,
        "request_id": record["source_request_id"], "status": "running"}
    state.metadata["confirmed_publication"] = {"generation": state.generation,
        "turn_id": record["resume_turn_id"], "plan_id": record["wait_id"]}
    return PublicationLease(state.owner, state.session_id, state.generation, record["resume_turn_id"],
        record["source_request_id"], record["wait_id"], record["claim_id"], "effect")


def apply_effect_action(state, lease, update, now):
    record = continuation_record(state, lease.continuation_id, lease.continuation_kind)
    value, budget = update.value, record["budget"]
    if value["operation"] == "reserve":
        for counter, limit in (("model_spent", "model_limit"), ("tool_spent", "tool_limit")):
            budget[limit] = min(budget[limit], value[limit])
            amount = value.get(counter, 0)
            if type(amount) is not int or amount < 0 or budget[counter] + amount > budget[limit]:
                raise ValueError("cumulative effect continuation budget exhausted")
            budget[counter] += amount
        record.update(state="running", last_checked_at=now, claim_until=now + EFFECT_ACTIVE_SECONDS + CLAIM_SECONDS)
    elif value["operation"] == "park":
        permit = value["permit"]
        if not isinstance(permit, _ReadPermitRejected) or permit.rate_limit["reason"] != "quota":
            raise ValueError("untrusted READ rejection")
        due = permit.rate_limit["retry_at_epoch"]
        if ((type(due) is not float and type(due) is not int) or not math.isfinite(due) or not now <= due < record["expires_at"]
                or state.pending_effect_plan_id or budget["wait_segments"] >= MAX_WAIT_SEGMENTS
                or budget["model_spent"] >= budget["model_limit"] or budget["tool_spent"] >= budget["tool_limit"]
                or budget["wait_seconds_spent"] + due - now > MAX_WAIT_SECONDS):
            raise ValueError("effect continuation wait budget exhausted")
        # Only an acquire refusal is resumable; no started external call is retried here.
        anchor = anchor_for(state, value["call_id"], permit.tool_name)
        budget["wait_segments"] += 1
        budget["wait_seconds_spent"] += due - now
        record.update(state="pending", anchor=anchor, due_at=due, claim_until=0, last_checked_at=now)
    else:
        raise ValueError("unsupported effect continuation transition")


async def adopt_foreground(store, lease, plan_id):
    """完成回执不消费原目标；原前台沿同一续行租约继续，不再另开模型循环。"""
    def adopt(state):
        row = (state.metadata.get("effect_waits") or {}).get(plan_id) or {}
        record = row.get("continuation") or {}
        if not record or record.get("state") not in {"pending", "waiting"} or not isinstance(row.get("final_result"), dict):
            return None
        if state.generation != lease.generation or state.metadata.get("confirmed_publication") != {
                "generation": lease.generation, "turn_id": lease.turn_id, "plan_id": plan_id}:
            return None
        effect_payload(state, record)
        record.update(state="claimed", claim_id=secrets.token_urlsafe(18), claim_until=store._clock() + EFFECT_ACTIVE_SECONDS + CLAIM_SECONDS,
                      resume_turn_id=lease.turn_id, publication_generation=lease.generation)
        return PublicationLease(lease.owner, lease.session_id, lease.generation, lease.turn_id,
                                lease.request_id, plan_id, record["claim_id"], "effect")
    return await store.update_effect_state(owner=lease.owner, session_id=lease.session_id, change=adopt)


def confirmed_result(catalog, state, record):
    from .effects import EffectPlan
    from .projection import ToolOutcome
    payload = effect_payload(state, record)
    seed, origin, result = payload["seed"], payload["origin"], payload["result"]
    tool = catalog.get(seed["tool"]).for_arguments(origin["arguments"])
    batch = origin.get("batch_plan") or {}
    if batch and (tool.name != "agent.batch" or batch["effect"] != tool.effect.value):
        raise ValueError("frozen batch risk changed")
    plan = EffectPlan(seed["plan_id"], state.owner, state.session_id, origin["origin_generation"],
                      tool.name, tool.effect, origin["arguments"], batch.get("snapshot", ""), {}, 0,
                      owner_generation=batch.get("owner_generation", 0),
                      confirmation_contract=deepcopy(batch.get("confirmation_contract", {})),
                      metadata=deepcopy(batch["metadata"]) if batch else
                               ({"reobservation_source": deepcopy(origin["reobservation_source"])}
                                if origin.get("reobservation_source") is not None else {}))
    return PipelineResult(tool, origin["arguments"], ToolOutcome(
        model_content=json.dumps(result, ensure_ascii=False), public_content=result,
    ), effect_plan=plan)


@dataclass(frozen=True, slots=True)
class _DeliveryAcknowledged:
    record: dict
    accepted: bool
    expired: bool
    now: float

    def __call__(self, state):
        record = self.record
        current = continuation_record(state, record["wait_id"], record.get("kind"))
        if (current.get("wait_id") == record["wait_id"] and current.get("delivery_pending") is True
                and current.get("resume_turn_id") == record["resume_turn_id"]
                and current.get("delivery_view") == record["delivery_view"]):
            current["delivery_pending"] = not (self.accepted or self.expired)
            current["delivery_state"] = "queued" if self.accepted else "expired" if self.expired else "retry_wait"
            if self.accepted:
                current.pop("delivery_view", None)
            elif not self.expired:
                current["due_at"] = self.now + CLAIM_SECONDS


async def deliver_continuation_result(store, candidate):
    """只重投已保存的最终视图；进程重启不能再次调用模型或执行旧写操作。"""
    from app.bot.agent_adapter import (
        prepare_agent_continuation_delivery,
        publish_agent_continuation_delivery,
    )

    state = await store.load(owner=candidate["owner"], session_id=candidate["session_id"])
    record = continuation_record(state, candidate["wait_id"], candidate.get("kind"))
    if (record.get("wait_id") != candidate["wait_id"] or record.get("delivery_pending") is not True or record.get("resume_turn_id") != candidate.get("resume_turn_id")
            or record.get("delivery_view") != candidate.get("delivery_view")):
        return False
    now = store._clock()
    expired = now >= record["expires_at"]
    accepted = False
    if not expired:
        route = effect_payload(state, record)["origin"].get("delivery_route") if candidate.get("kind") == "effect" else record.get("delivery_route")
        payload = prepare_agent_continuation_delivery(
            record["delivery_view"], owner=state.owner, session_id=state.session_id,
            delivery_route=route or {},
            delivery_id=f'{record["wait_id"]}:{record["resume_turn_id"]}',
        )
        result = await publish_agent_continuation_delivery(payload)
        accepted = result.accepted

    await store.update_effect_state(owner=state.owner, session_id=state.session_id,
        change=_DeliveryAcknowledged(record, accepted, expired, now))
    return accepted
