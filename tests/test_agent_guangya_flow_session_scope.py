"""光鸭短期预览的会话写入权与已受理持久任务相互独立。"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from app.agent.domain_catalog import DomainActions
from app.agent.domains.guangya import cleanup_actions
from app.agent.domains.guangya import directory_scrape_actions as scrape_actions
from app.agent.errors import AgentToolError
from app.agent.kernel.bootstrap import AgentKernelRuntime, build_agent_kernel_runtime
from app.agent.models import ToolContext, ToolResult
from app.agent.session_context import SQLiteAgentSessionContextRepository
from app.clients.guangya import GuangYaClient, GuangYaFile
from app.modules import guangya_residual_cleanup as cleanup
from app.modules import organize_tasks
from app.modules.directory_scrape import DirectoryScrapeService, DirectoryScrapeStore
from app.modules.organize_rules import OrganizeRules
from app.modules.scraper import TMDBScraper
from app.repositories import organize_operation_jobs as jobs
from tests.support import isolated_test_database
from tests.test_agent_effect_admission import (
    _kernel_plan,
    _offline_runtime,
    _OfflineClient,
    _paused_manager,
    _run_real_worker,
)
from tests.test_agent_kernel_core import ScriptedModel
from tests.test_guangya_directory_scrape import _DirectoryScrapeTMDB

OWNER = "guangya-flow-scope-owner"
A = ToolContext(owner=OWNER, session_id="a")
B = ToolContext(owner=OWNER, session_id="b")
PREVIEW = scrape_actions.directory_scrape_preview_arguments({"candidate_number": 1})
SEARCH = {"query": "Iron Man", "year": "2008", "media_type": "movie"}


class CountingClient(_OfflineClient):
    def __init__(self):
        super().__init__()
        self.writes: list[tuple[Any, ...]] = []
        video = self.file_info("rename")
        assert video is not None
        self.directories["source"] = [
            replace(video, name="Iron.Man.2008.1080p.mkv", extension="mkv", size=2 * 1024**3),
            GuangYaFile("trash", "空目录", True, parent_id="source", etag="empty-v1", updated_at=1),
            GuangYaFile("residual", "a", True, parent_id="source", etag="junk-v1", updated_at=1),
        ]
        self.directories["residual"] = [
            GuangYaFile("junk", "xxx.png", False, parent_id="residual", extension="png", size=10, etag="j1"),
        ]

    def rename(self, file_id, new_name):
        self.writes.append(("rename", file_id, new_name))
        return super().rename(file_id, new_name)

    def move(self, file_ids, parent_id):
        self.writes.append(("move", tuple(file_ids), parent_id))
        return super().move(file_ids, parent_id)

    def create_dir(self, name, parent_id="0"):
        self.writes.append(("mkdir", name, parent_id))
        return super().create_dir(name, parent_id)

    def delete(self, file_ids):
        self.writes.append(("delete", tuple(file_ids)))
        return super().delete(file_ids)


def new_scrape_service(client: CountingClient) -> DirectoryScrapeService:
    rules = OrganizeRules(target_dir_id="target", small_file_mb=0, region_split=False,
                          year_split=False, clean_empty=False, link_strm=False, notify_enabled=False)
    return DirectoryScrapeService(
        client=cast(GuangYaClient, client), scraper=cast(TMDBScraper, _DirectoryScrapeTMDB()),
        store=DirectoryScrapeStore(), rules_loader=lambda: rules,
    )


@dataclass
class Harness:
    actions: DomainActions
    repository: SQLiteAgentSessionContextRepository | None
    client: CountingClient
    service: DirectoryScrapeService
    stack: ExitStack

    def domain(self, name: str):
        return self.actions.directory_scrape if name == "scrape" else self.actions.cleanup

    def preview(self, name: str, context: ToolContext, *, full: bool = False):
        if name == "cleanup":
            arguments = cleanup_actions.guangya_cleanup_preview_arguments({
                "path": "/source", "scope": "all" if full else "empty_only", "max_candidates": 20,
            })
            return self.actions.cleanup.preview_guangya_cleanup(arguments, context)
        self.actions.directory_scrape.inspect_directory_scrape({"directory_id": "source"}, context)
        self.actions.directory_scrape.search_directory_scrape(SEARCH, context)
        return self.actions.directory_scrape.preview_directory_scrape(PREVIEW, context)

    def controller(self, *, cold: bool = False) -> AgentKernelRuntime:
        actions = DomainActions(self.repository) if cold and self.repository is not None else self.actions
        with patch("app.agent.kernel.bootstrap.SQLiteAgentSessionContextRepository", return_value=self.repository), patch(
            "app.agent.kernel.bootstrap.DomainActions", return_value=actions,
        ):
            return build_agent_kernel_runtime(model=ScriptedModel([]))


@pytest.fixture(params=[False, True], ids=["memory", "sqlite"])
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Harness]:
    client = CountingClient()
    with isolated_test_database(), _offline_runtime(tmp_path, client) as stack, _paused_manager():
        repository = SQLiteAgentSessionContextRepository() if request.param else None
        stack.domain_actions = DomainActions(repository)
        service = new_scrape_service(client)
        stack.enter_context(patch.object(scrape_actions, "get_directory_scrape_service", return_value=service))
        stack.enter_context(patch.object(cleanup_actions, "execute_cleanup_plan", partial(
            cleanup.execute_cleanup_plan, client_factory=lambda: cast(GuangYaClient, client),
        )))
        yield Harness(stack.domain_actions, repository, client, service, stack)


@pytest.mark.parametrize("name", ["scrape", "cleanup"])
@pytest.mark.parametrize("session_id", ["", " "])
def test_preview_requires_session_before_provider_use(harness: Harness, name: str, session_id: str):
    with (
        patch.object(harness.client, "list_dir", side_effect=AssertionError("invalid session reached provider")),
        pytest.raises(AgentToolError, match="会话"),
    ):
        harness.preview(name, ToolContext(owner=OWNER, session_id=session_id))


@pytest.mark.parametrize("name", ["scrape", "cleanup"])
@pytest.mark.parametrize("harness", [True], indirect=True, ids=["sqlite"])
def test_owner_only_row_is_never_rebound(harness: Harness, name: str):
    assert harness.repository is not None
    harness.preview(name, A)
    domain = harness.domain(name)
    flow = domain._flow(OWNER, "a")
    assert flow is not None
    payload = (scrape_actions._flow_payload(flow) if isinstance(flow, scrape_actions._Flow)
               else cleanup_actions._payload(flow))
    context_type = "directory_scrape" if name == "scrape" else "guangya_cleanup"
    harness.repository.replace_latest(owner=OWNER, context_type=context_type, payload=payload,
                                      expires_at=time.time() + 600)
    domain.clear_session(owner=OWNER, session_id="a")
    cold = DomainActions(harness.repository)
    for actions in (harness.actions, cold):
        target = actions.directory_scrape if name == "scrape" else actions.cleanup
        assert target._flow(OWNER, "a") is None
        assert target._flow(OWNER, "b") is None


@pytest.mark.parametrize("name", ["scrape", "cleanup"])
@pytest.mark.parametrize("harness", [True], indirect=True, ids=["sqlite"])
def test_clear_propagates_repository_failure(harness: Harness, name: str):
    assert harness.repository is not None
    harness.preview(name, A)
    with (
        patch.object(harness.repository, "invalidate_context", side_effect=RuntimeError("context DB failure")),
        pytest.raises(RuntimeError, match="context DB failure"),
    ):
        harness.domain(name).clear_session(owner=OWNER, session_id="a")
    assert harness.domain(name)._flow(OWNER, "a") is not None


@pytest.mark.parametrize("operation", ["reset", "delete"])
@pytest.mark.parametrize("point", ["scrape_inspect", "scrape_search", "scrape_preview", "cleanup_preview", "cleanup_classify"])
def test_lifecycle_rejects_late_result_in_memory_and_sqlite(harness: Harness, operation: str, point: str):
    async def scenario():
        name = point.split("_", 1)[0]
        harness.preview(name, B)
        harness.preview(name, A, full=point == "cleanup_classify")
        domain = harness.domain(name)
        previous_b = domain._flow(OWNER, "b")
        assert previous_b is not None
        entered, release = threading.Event(), threading.Event()
        call: Callable[[], ToolResult]
        if name == "scrape":
            method = point.removeprefix("scrape_")
            original = getattr(harness.service, method)
            target = harness.service
            call = {
                "inspect": partial(harness.actions.directory_scrape.inspect_directory_scrape, {"directory_id": "source"}, A),
                "search": partial(harness.actions.directory_scrape.search_directory_scrape, SEARCH, A),
                "preview": partial(harness.actions.directory_scrape.preview_directory_scrape, PREVIEW, A),
            }[method]
        else:
            target = cleanup_actions
            method = "revise_cleanup_plan" if point == "cleanup_classify" else "build_cleanup_plan"
            original = getattr(target, method)
            call = (partial(harness.actions.cleanup.classify_guangya_cleanup_candidates,
                            {"decisions": [{"candidate_number": 1, "action": "keep"}]}, A)
                    if point == "cleanup_classify" else partial(harness.preview, name, A))

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(5), "late-result barrier timed out"
            return original(*args, **kwargs)

        controller = harness.controller(cold=True)
        with patch.object(target, method, side_effect=blocked):
            pending = asyncio.create_task(asyncio.to_thread(call))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                await getattr(controller.lifecycle, operation)(owner=OWNER, session_id="a")
            finally:
                release.set()
                result, = await asyncio.gather(pending, return_exceptions=True)
        assert isinstance(result, AgentToolError), result
        assert domain._flow(OWNER, "a") is None
        remaining_b = domain._flow(OWNER, "b")
        assert remaining_b is not None
        assert replace(remaining_b, updated_at=previous_b.updated_at) == previous_b
        assert harness.client.writes == []
        assert jobs.claim_organize_operation_job() is None
        assert harness.preview(name, A).ok
    asyncio.run(scenario())


@pytest.mark.parametrize("intervention", ["none", "clear", "new_inspect"])
def test_queue_rejection_restores_only_the_same_generation(harness: Harness, intervention: str):
    harness.preview("scrape", B)
    harness.preview("scrape", A)
    actions = harness.actions.directory_scrape
    _preview, fingerprint = actions.prepare_run_directory_scrape({}, A)

    def rejected(*_args, **_kwargs):
        assert actions._flow(OWNER, "a") is None  # consume 已完成。
        if intervention == "clear":
            controller = DomainActions(harness.repository) if harness.repository is not None else harness.actions
            controller.clear_session(owner=OWNER, session_id="a")
        elif intervention == "new_inspect":
            actions.inspect_directory_scrape({"directory_id": "source"}, A)
        return {"ok": False}

    manager = organize_tasks.get_organize_manager()
    with (
        patch.object(manager, "start_durable_operation", side_effect=rejected),
        pytest.raises(AgentToolError) as failure,
    ):
        actions.run_directory_scrape_confirmed({}, fingerprint, A)
    assert failure.value.code == "precondition_failed"
    flow = actions._flow(OWNER, "a")
    if intervention == "none":
        assert "预览已保留" in str(failure.value)
        assert actions.prepare_run_directory_scrape({}, A)[1] == fingerprint
        with patch.object(manager, "start_durable_operation", return_value={"ok": True, "queued": True}):
            assert actions.run_directory_scrape_confirmed({}, fingerprint, A).status == "accepted"
        assert actions._flow(OWNER, "a") is None
    elif intervention == "clear":
        assert flow is None
    else:
        assert flow is not None and flow.preview_safe is None
    assert actions.prepare_run_directory_scrape({}, B)[0].ok


@pytest.mark.parametrize("operation", ["reset", "delete"])
@pytest.mark.parametrize("name", ["scrape", "cleanup"])
def test_admitted_v1_job_survives_session_clear_and_new_preview(harness: Harness, name: str, operation: str):
    async def scenario():
        runtime = harness.controller()
        pipeline, context, plan, _events = await _kernel_plan(
            name, runtime.store, harness.client, harness.stack, owner=OWNER, session_id="a",
        )
        result = await pipeline.execute_confirmed(plan.plan_id, context=context)
        assert result.outcome.public_content["status"] == "accepted"
        assert harness.client.writes == []
        row = jobs.claim_organize_operation_job()
        assert row is not None and jobs.verify_organize_operation_payload(row)
        payload = json.loads(row["payload_json"])
        assert payload["version"] == 1
        journal: bytes | None = None
        if name == "scrape":
            assert set(payload) == {"version", "execution_owner", "expected_context", "flow"}
            assert set(payload["flow"]) == {"scope_type", "scope_id", "selected_candidate", "preview_arguments", "preview_guard"}
            # 真正的冷服务不持有原 inspection/preview store。
            harness.stack.enter_context(patch.object(scrape_actions, "get_directory_scrape_service",
                                                    return_value=new_scrape_service(harness.client)))
        else:
            assert set(payload) == {"version", "plan_id", "plan_fingerprint", "owner_digest", "credential_generation"}
            cleanup._append_journal(payload["plan_id"], {"action": "accepted", "task_id": row["job_id"]})
            journal = cleanup._journal_path(payload["plan_id"]).read_bytes()
        await getattr(runtime.lifecycle, operation)(owner=OWNER, session_id="a")
        # reset/delete 只撤销会话临时数据；同一业务 owner 的新预览不销毁已确认凭据。
        assert harness.preview(name, A).ok
        if name == "cleanup":
            recovered = cleanup.load_cleanup_plan(payload["plan_id"], owner=OWNER, require_confirmed=True)
            assert recovered["fingerprint"] == payload["plan_fingerprint"]
            assert cleanup._journal_path(payload["plan_id"]).read_bytes() == journal
        retained = jobs.get_organize_operation_job(row["job_id"])
        assert retained is not None
        assert (retained["payload_json"], retained["payload_auth"]) == (row["payload_json"], row["payload_auth"])
        await asyncio.to_thread(_run_real_worker, row)
        completed = jobs.get_organize_operation_job(row["job_id"])
        assert completed is not None and completed["status"] == "completed", dict(completed or {})
        assert harness.client.writes
        assert jobs.claim_organize_operation_job() is None
        writes = list(harness.client.writes)
        await asyncio.to_thread(_run_real_worker, row)
        assert harness.client.writes == writes
    asyncio.run(scenario())
