"""映射候选序号必须绑定会话，清理后不能从暖缓存或持久快照复活。"""
from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from app import config, database as db
from app.agent.domains.discovery.mapping_actions import DiscoveryMappingActions
from app.agent.errors import AgentToolError, ToolPipelineError
from app.agent.kernel.bootstrap import AgentKernelRuntime, build_agent_kernel_runtime
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.models import ToolContext
from app.agent.session_context import (
    AgentSessionContextRepository,
    SQLiteAgentSessionContextRepository,
)
from app.discovery.models import MediaCard
from app.discovery.service import DiscoveryService
from tests.support import initialized_web_runtime
from tests.test_agent_kernel_core import ScriptedModel
from tests.test_agent_kernel_existing_ports import context_for

OWNER = "webk:v1:" + "a" * 64
SECRET = "mapping-session-scope-test"


def _detail(provider: str, media_type: str, external_id: str) -> MediaCard:
    return MediaCard(provider=provider, external_id=external_id, media_type=media_type,
                     title=f"来源影片-{external_id}", year="2026")


def _lookup(provider: str, external_id: str, media_type: str, title: str, year: str) -> dict[str, Any]:
    return {"confirmed": False, "candidates": [{"tmdb_id": f"90{external_id}",
            "title": f"候选影片-{external_id}", "year": year, "media_type": media_type, "score": 0.9}]}


@pytest.fixture
def service() -> Iterator[DiscoveryService]:
    with initialized_web_runtime():
        config.set_and_save({"DISCOVERY_ENABLED": "1"})
        service = DiscoveryService()
        with (
            patch.object(service, "get_detail", side_effect=_detail),
            patch.object(service, "lookup_tmdb_mapping", side_effect=_lookup),
            patch.object(service, "verify_tmdb_mapping_candidate", side_effect=lambda tmdb_id, media_type: {
                "tmdb_id": tmdb_id, "title": f"已核验-{tmdb_id}", "year": "2026", "media_type": media_type,
            }),
            patch("app.agent.domains.discovery.mapping_actions.get_discovery_service", return_value=service),
        ):
            yield service


def _repository(persistent: bool) -> SQLiteAgentSessionContextRepository | None:
    return SQLiteAgentSessionContextRepository(secret_provider=lambda: SECRET) if persistent else None


def _runtime(repository: AgentSessionContextRepository | None) -> AgentKernelRuntime:
    with patch("app.agent.kernel.bootstrap.SQLiteAgentSessionContextRepository", return_value=repository):
        return build_agent_kernel_runtime(model=ScriptedModel([]), store=SQLiteKernelStore(secret_provider=lambda: SECRET))


async def _read(runtime: AgentKernelRuntime, session_id: str, external_id: str):
    context = await context_for(runtime.store, owner=OWNER, session=session_id)
    return await runtime.session.pipeline.execute("discovery.mapping_candidates", {
        "provider": "douban", "external_id": external_id, "media_type": "movie",
    }, context=context)


async def _prepare(runtime: AgentKernelRuntime, session_id: str):
    context = await context_for(runtime.store, owner=OWNER, session=session_id)
    preview = await runtime.session.pipeline.execute("discovery.confirm_mapping", {"candidate_number": 1}, context=context)
    return context, preview


async def _confirm(runtime: AgentKernelRuntime, session_id: str, external_id: str) -> None:
    context, preview = await _prepare(runtime, session_id)
    assert preview.outcome.public_content["data"]["source_title"] == f"来源影片-{external_id}"
    assert preview.effect_plan is not None
    result = await runtime.session.pipeline.execute_confirmed(preview.effect_plan.plan_id, context=context)
    assert result.outcome.public_content["ok"]
    row = db.get_media_external_id("douban", external_id, "movie")
    assert row is not None and row["tmdb_id"] == f"90{external_id}" and row["confirmed"]


@pytest.mark.parametrize("persistent,worker", [(False, "shared"), (True, "shared"), (True, "warm"), (True, "cold")])
def test_real_pipeline_candidate_number_stays_in_its_session(service, persistent, worker):
    repository = _repository(persistent)
    first = _runtime(repository)
    second = first if worker == "shared" else _runtime(repository)

    async def exercise():
        result = await _read(first, "session-a", "101")
        assert result.outcome.public_content["data"]["source_title"] == "来源影片-101"
        await _read(second, "session-b", "202")
        receiver = _runtime(repository) if worker == "cold" else first
        await _confirm(receiver, "session-a", "101")
        await _confirm(second, "session-b", "202")

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["reset", "delete"])
@pytest.mark.parametrize("worker", ["warm", "cold"])
def test_lifecycle_invalidates_only_its_mapping_session_across_workers(service, operation, worker):
    repository = _repository(True)
    first, second = _runtime(repository), _runtime(repository)

    async def exercise():
        await _read(first, "session-b", "202")
        context_b, plan_b = await _prepare(first, "session-b")
        assert plan_b.effect_plan is not None
        await _read(first, "session-a", "101")
        _, plan_a = await _prepare(first, "session-a")
        assert plan_a.effect_plan is not None
        await getattr(second.lifecycle, operation)(owner=OWNER, session_id="session-a")
        state = await second.store.load(owner=OWNER, session_id="session-a")
        assert not state.conversation and not state.pending_effect_plan_id
        receiver = first if worker == "warm" else _runtime(repository)
        fresh = await context_for(receiver.store, owner=OWNER, session="session-a")
        with pytest.raises(ToolPipelineError):
            await receiver.session.pipeline.execute_confirmed(plan_a.effect_plan.plan_id, context=fresh)
        with pytest.raises(ToolPipelineError, match="最近映射候选不存在或已过期"):
            await receiver.session.pipeline.execute("discovery.confirm_mapping", {"candidate_number": 1}, context=fresh)
        result = await first.session.pipeline.execute_confirmed(plan_b.effect_plan.plan_id, context=context_b)
        assert result.outcome.public_content["ok"]
        assert db.get_media_external_id("douban", "101", "movie") is None
        row = db.get_media_external_id("douban", "202", "movie")
        assert row is not None and row["confirmed"]

    asyncio.run(exercise())


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("operation", ["reset", "delete"])
def test_lifecycle_rejects_late_candidate_publication_without_harming_other_session(service, persistent, operation):
    repository = _repository(persistent)
    first = _runtime(repository)
    controller = _runtime(repository) if persistent else first
    started, release = threading.Event(), threading.Event()

    def lookup(provider, external_id, media_type, title, year):
        if external_id == "101":
            started.set()
            assert release.wait(timeout=5)
        return _lookup(provider, external_id, media_type, title, year)

    async def exercise():
        await _read(first, "session-b", "202")
        with patch.object(service, "lookup_tmdb_mapping", side_effect=lookup):
            late = asyncio.create_task(_read(first, "session-a", "101"))
            try:
                assert await asyncio.to_thread(started.wait, 3)
                await getattr(controller.lifecycle, operation)(owner=OWNER, session_id="session-a")
            finally:
                release.set()
                await asyncio.gather(late, return_exceptions=True)
        with pytest.raises(ToolPipelineError, match="最近映射候选不存在或已过期"):
            await _prepare(first, "session-a")
        await _confirm(first, "session-b", "202")
        assert db.get_media_external_id("douban", "101", "movie") is None

    asyncio.run(exercise())


def test_legacy_owner_only_snapshot_is_not_a_session_fallback(service):
    repository = _repository(True)
    assert repository is not None
    guard = repository.begin_context(owner=OWNER, context_type="discovery_mapping")
    repository.replace_latest_guarded(owner=OWNER, context_type="discovery_mapping", guard=guard,
        expires_at=time.time() + 600, payload={
            "provider": "douban", "external_id": "101", "media_type": "movie", "title": "旧快照", "year": "2026",
            "candidates": [{"candidate_number": 1, "tmdb_id": "90101", "title": "旧候选",
                            "year": "2026", "media_type": "movie", "score": 0.9}],
        })

    async def exercise():
        with pytest.raises(ToolPipelineError, match="最近映射候选不存在或已过期"):
            await _prepare(_runtime(repository), "new-session")
        assert db.get_media_external_id("douban", "101", "movie") is None

    asyncio.run(exercise())


@pytest.mark.parametrize("session_id", ["", " "])
def test_mapping_actions_require_a_real_session(service, session_id):
    actions = DiscoveryMappingActions()
    context = ToolContext(owner=OWNER, session_id=session_id)
    with pytest.raises(AgentToolError, match="会话"):
        actions.get_discovery_mapping_candidates({"provider": "douban", "external_id": "101", "media_type": "movie"}, context)


@pytest.mark.parametrize("operation", ["reset", "delete"])
def test_lifecycle_does_not_report_success_when_mapping_invalidation_fails(service, operation):
    repository = _repository(True)
    assert repository is not None
    runtime = _runtime(repository)
    original = repository.invalidate_context

    def invalidate(*, owner, context_type):
        if context_type == "discovery_mapping":
            raise OSError("mapping storage failed")
        return original(owner=owner, context_type=context_type)

    async def exercise():
        with (
            patch.object(repository, "invalidate_context", side_effect=invalidate),
            patch.object(runtime.store, operation + "_session", wraps=getattr(runtime.store, operation + "_session")) as change,
            pytest.raises(OSError, match="mapping storage failed"),
        ):
            await getattr(runtime.lifecycle, operation)(owner=OWNER, session_id="session-a")
        change.assert_not_called()

    asyncio.run(exercise())
