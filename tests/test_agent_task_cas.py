"""Small same-generation CAS, lifecycle and strict evidence controls."""

import asyncio
import multiprocessing
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app import database as db
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.state import StateUpdate, StalePublicationError, SessionBusyError
from app.agent.kernel.task_state import (
    TaskConflict,
    identity,
    observation_patch,
    proposal,
    reduce_task,
    validate_proposal,
)
from tests.test_agent_task_first_slice import GOAL, OWNER, declaration


@pytest.fixture
def task_store(tmp_path):
    previous = db.DB_PATH, bool(getattr(db, "_configured_test_mode", False))
    db.configure_database(tmp_path / "cas.db", test_mode=True)
    db.init_db()
    yield SQLiteKernelStore(secret_provider=lambda: "task-test-only")
    db.configure_database(previous[0], test_mode=previous[1])


def control(args):
    return proposal(args, SimpleNamespace(request_id="request", user_source=GOAL))


def command(task, operation, **extra):
    return control({"operation": operation, "expected": identity(task), **extra})


def cas_worker(database, lease, task_patch, marker, ready, start, committed, results):
    db.configure_database(database, test_mode=True)
    store = SQLiteKernelStore(secret_provider=lambda: "task-test-only")
    ready.put(marker)
    if not start.wait(15):
        raise TimeoutError("CAS start barrier")

    async def attempt():
        try:
            await store.commit(
                lease,
                conversation=[{"role": "user", "content": marker}],
                task_patch=task_patch,
            )
        except StalePublicationError as exc:
            assert isinstance(exc.__cause__, SessionBusyError)
            assert committed.wait(15)
            await store.commit(
                lease,
                conversation=[{"role": "user", "content": marker}],
                task_patch=task_patch,
            )

    try:
        asyncio.run(attempt())
    except TaskConflict:
        results.put("conflict")
    else:
        committed.set()
        results.put("committed")


def test_sqlite_same_generation_cas_and_capacity_are_atomic(task_store):
    async def run():
        lease, _ = await task_store.begin_turn(
            owner=OWNER, session_id="cas", request_id="request"
        )
        initial = await task_store.commit(lease, task_patch=control(declaration()))
        task = initial.metadata["task"]
        patch = command(task, "checkpoint", evidence_ids=[], blockers=[])
        other = SQLiteKernelStore(secret_provider=lambda: "task-test-only")
        context = multiprocessing.get_context("spawn")
        ready, results, start, committed = (
            context.Queue(),
            context.Queue(),
            context.Event(),
            context.Event(),
        )
        workers = [
            context.Process(
                target=cas_worker,
                args=(
                    db.DB_PATH,
                    lease,
                    patch,
                    marker,
                    ready,
                    start,
                    committed,
                    results,
                ),
            )
            for marker in ("winner-a", "winner-b")
        ]
        try:
            for worker in workers:
                worker.start()
            assert {ready.get(timeout=20), ready.get(timeout=20)} == {
                "winner-a",
                "winner-b",
            }
            start.set()
            assert sorted([results.get(timeout=20), results.get(timeout=20)]) == [
                "committed",
                "conflict",
            ]
            for worker in workers:
                worker.join(20)
                assert worker.exitcode == 0
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(5)
        saved = await other.load(owner=OWNER, session_id="cas")
        with pytest.raises(TaskConflict):
            await task_store.commit(
                lease,
                conversation=[{"role": "user", "content": "stale"}],
                task_patch=patch,
            )
        assert (
            await other.load(owner=OWNER, session_id="cas")
        ).to_payload() == saved.to_payload()
        current = command(
            saved.metadata["task"], "checkpoint", evidence_ids=[], blockers=[]
        )
        with pytest.raises(ValueError):
            await task_store.commit(
                lease,
                task_patch=current,
                conversation=[{"role": "user", "content": "overflow"}],
                updates=[StateUpdate("metadata.pressure", "x" * 262144)],
            )
        assert (
            await other.load(owner=OWNER, session_id="cas")
        ).to_payload() == saved.to_payload()
        for update in (
            StateUpdate("metadata.task", {}),
            StateUpdate("metadata.task", None, "delete"),
            StateUpdate("metadata", {}),
        ):
            with pytest.raises(ValueError):
                await task_store.commit(lease, updates=[update])
        with pytest.raises(ValueError):
            await task_store.update_effect_state(
                owner=OWNER,
                session_id="cas",
                change=lambda state: state.metadata.pop("task"),
            )
        await task_store.update_effect_state(
            owner=OWNER,
            session_id="cas",
            change=lambda state: state.metadata.update(other="safe"),
        )
        assert (await other.load(owner=OWNER, session_id="cas")).metadata[
            "task"
        ] == saved.metadata["task"]

    asyncio.run(run())


def test_evidence_scope_lifecycle_and_component_blockers():
    def apply(task, patch):
        return reduce_task(task, patch, owner=OWNER, session_id="cas")

    task = apply(None, control(declaration()))
    observed = {
        "component": "tmdb",
        "status": "not_configured",
        "enabled": True,
        "probe_mode": "configuration_only",
        "network_accessed": False,
    }

    def observation(current, component, fact):
        return observation_patch(
            current,
            owner=OWNER,
            session_id="cas",
            tool="config.explain_component",
            call_id="call",
            component=component,
            fact=fact,
        )

    task = apply(task, observation(task, "tmdb", None))
    assert task["status"] == "blocked"
    task = apply(
        task, observation(task, "qbittorrent", {**observed, "component": "qbittorrent"})
    )
    assert task["blockers"] == [
        {"origin": "query", "component": "tmdb", "epoch": 1, "text": "query:tmdb"}
    ]
    task = apply(task, observation(task, "tmdb", observed))
    assert task["blockers"] == [] and task["status"] == "partial"
    evidence = task["verified_facts"][-1]["evidence_id"]
    with pytest.raises(ValueError):
        apply(task, command(task, "checkpoint", evidence_ids=["forged"], blockers=[]))
    for key, value in (("owner", "another"), ("session_id", "another"), ("epoch", 999)):
        corrupt = deepcopy(task)
        corrupt["verified_facts"][-1]["scope"][key] = value
        with pytest.raises(ValueError):
            apply(
                corrupt,
                command(corrupt, "checkpoint", evidence_ids=[evidence], blockers=[]),
            )
    done = apply(
        task,
        command(
            task,
            "checkpoint",
            evidence_ids=[f["evidence_id"] for f in task["verified_facts"]],
            blockers=[],
        ),
    )
    assert done["status"] == "complete"
    revised = declaration()
    revised.update(operation="revise", expected=identity(done))
    changed = apply(done, control(revised))
    assert changed["epoch"] == 2 and changed["verified_facts"] == []
    with pytest.raises(ValueError):
        apply(
            changed,
            command(changed, "checkpoint", evidence_ids=[evidence], blockers=[]),
        )
    cancelled = apply(changed, command(changed, "cancel", source=GOAL))
    with pytest.raises(TaskConflict):
        apply(cancelled, control(declaration()))
    replacement = declaration()
    replacement["expected"] = identity(cancelled)
    fresh = apply(cancelled, control(replacement))
    assert fresh["task_id"] != cancelled["task_id"] and (
        fresh["epoch"],
        fresh["revision"],
    ) == (1, 1)
    assert fresh["verified_facts"] == [] and fresh["blockers"] == []
    unsupported = declaration()
    unsupported["obligations"][1]["acceptance"] = {
        "kind": "repair",
        "criterion": "修复并联网",
    }
    assert apply(None, control(unsupported))["status"] == "blocked"


def test_strict_proposal_rejects_coercion_and_unbounded_data():
    mutations = [
        lambda a: a.update(status="complete"),
        lambda a: a.update(obligations=[]),
        lambda a: a["obligations"][0].update(required=1),
        lambda a: a["obligations"][0].update(required=False),
        lambda a: a.update(root_goal="x" * 2049),
        lambda a: a.update(constraints=[str(i) for i in range(9)]),
        lambda a: a["obligations"][1].update(id="tmdb"),
        lambda a: a["obligations"][0].update(
            acceptance={"kind": "repair", "criterion": {"nested": True}}
        ),
        lambda a: a.update(expected={"task_id": "id", "epoch": True, "revision": 1}),
        lambda a: a.update(source=""),
    ]
    for mutate in mutations:
        args = declaration()
        mutate(args)
        with pytest.raises((ValueError, TypeError)):
            validate_proposal(args)
    args = declaration()
    args["source"] = "not the current user"
    with pytest.raises(ValueError):
        control(args)


def test_blocker_origins_coexist_and_remain_bounded():
    def apply(task, patch):
        return reduce_task(task, patch, owner=OWNER, session_id="cas")

    def observe(task, fact=None):
        return observation_patch(
            task, owner=OWNER, session_id="cas", tool="config.explain_component",
            call_id="call", component="tmdb", fact=fact,
        )

    task = apply(None, control(declaration()))
    task = apply(task, observe(task))
    query = {"origin": "query", "component": "tmdb", "epoch": 1, "text": "query:tmdb"}
    reported = {"origin": "reported", "text": "query:tmdb"}
    task = apply(task, command(task, "checkpoint", evidence_ids=[], blockers=["query:tmdb"]))
    assert task["blockers"] == [query, reported] and task["status"] == "blocked"
    again = apply(task, observe(task))
    assert again["blockers"] == [query, reported]  # Repeated failure is idempotent.
    cleared = apply(task, command(task, "checkpoint", evidence_ids=[], blockers=[]))
    assert cleared["blockers"] == [query] and cleared["status"] == "blocked"
    observed = {
        "component": "tmdb", "status": "not_configured", "enabled": False,
        "probe_mode": "configuration_only", "network_accessed": False,
    }
    recovered = apply(task, observe(task, observed))
    assert recovered["blockers"] == [reported] and recovered["status"] == "blocked"
    assert len(recovered["verified_facts"]) == 1
    # Clearing is qualified by epoch, never merely matching text/component.
    other_epoch = deepcopy(task)
    other_epoch["blockers"][0]["epoch"] = 2
    assert apply(other_epoch, observe(other_epoch, observed))["blockers"] == other_epoch["blockers"]
    bounded = apply(task, command(task, "checkpoint", evidence_ids=[], blockers=[str(i) for i in range(15)]))
    assert len(bounded["blockers"]) == 16
    before = deepcopy(task)
    with pytest.raises(ValueError):
        apply(task, command(task, "checkpoint", evidence_ids=[], blockers=[str(i) for i in range(16)]))
    assert task == before
    for values in ([query], [reported], ["duplicate", "duplicate"]):
        with pytest.raises(ValueError):
            command(task, "checkpoint", evidence_ids=[], blockers=values)
    revised = declaration()
    revised.update(operation="revise", expected=identity(task))
    assert apply(task, control(revised))["blockers"] == []
