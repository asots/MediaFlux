"""真实 Kernel 与 SQLite 中，领域预览只属于创建它的会话生命周期。"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.agent.domains.guangya import directory_scrape_actions as scrape_actions
from app.agent.errors import ToolPipelineError
from app.agent.kernel.bootstrap import build_agent_kernel_runtime
from app.agent.kernel.pipeline import InMemoryRateLimiter
from tests.support import initialized_web_runtime
from tests.test_agent_effect_admission import _OfflineClient, _offline_runtime
from tests.test_agent_kernel_core import ScriptedModel
from tests.test_agent_kernel_existing_ports import context_for

OWNER = "webk:v1:" + "a" * 64


def scrape_service():
    service = Mock()
    service.inspect.side_effect = lambda owner, directory: {
        "inspection_id": "inspection-" + directory,
        "media_type": "tv", "suggested_query": directory, "counts": {"videos": 1},
    }
    service.search.return_value = [{"provider": "tmdb", "external_id": "123", "tmdb_id": "123",
        "title": "Session preview", "year": "2026", "media_type": "tv", "score": 0.95}]
    service.preview.side_effect = lambda owner, inspection, *args, **kwargs: {
        "preview_id": "preview-" + inspection, "plans": [{"action": "move"}], "companion_plans": [],
    }
    service.store.get_preview.return_value = SimpleNamespace(claimed=False, signature=(("origin-file", "renamed"),))
    return service


def runtime_for_test():
    runtime = build_agent_kernel_runtime(model=ScriptedModel([]))
    runtime.session.pipeline.rate_limiter = InMemoryRateLimiter(limit=100)
    return runtime


async def preview(runtime, kind, session_id):
    context = await context_for(runtime.store, owner=OWNER, session=session_id)
    pipeline = runtime.session.pipeline
    if kind == "scrape":
        await pipeline.execute("guangya.directory_scrape.inspect", {"directory_id": session_id}, context=context)
        await pipeline.execute("guangya.directory_scrape.search", {"media_type": "auto"}, context=context)
        await pipeline.execute("guangya.directory_scrape.preview", {"candidate_number": 1}, context=context)
    else:
        result = await pipeline.execute("guangya.organize.cleanup.preview", {
            "path": "/source", "scope": "empty_only",
        }, context=context)
        assert result.outcome.public_content["data"]["empty_dir_count"] == 1
    return context


@pytest.mark.parametrize("kind", ["scrape", "cleanup"])
@pytest.mark.parametrize("operation", ["reset", "delete"])
@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_previews_cannot_cross_sessions_or_survive_lifecycle_change(tmp_path, kind, operation, cold):
    async def run():
        service = scrape_service()
        client = _OfflineClient()
        with initialized_web_runtime(), _offline_runtime(tmp_path, client), patch.object(
            scrape_actions, "get_directory_scrape_service", return_value=service,
        ):
            runtime = runtime_for_test()
            tool = "guangya.directory_scrape.run" if kind == "scrape" else "guangya.organize.cleanup.execute"
            origin = await preview(runtime, kind, "session-a")
            initial = await runtime.session.pipeline.execute(tool, {}, context=origin)
            assert initial.effect_plan is not None

            other = await context_for(runtime.store, owner=OWNER, session="session-b")
            with pytest.raises(ToolPipelineError) as rejected:
                await runtime.session.pipeline.execute(tool, {}, context=other)
            assert rejected.value.code == "precondition_failed"
            assert not (await runtime.store.load(owner=OWNER, session_id="session-b")).pending_effect_plan_id

            await preview(runtime, kind, "session-b")
            await getattr(runtime.lifecycle, operation)(owner=OWNER, session_id="session-a")
            if cold:
                runtime = runtime_for_test()
            after = await context_for(runtime.store, owner=OWNER, session="session-a")
            with pytest.raises(ToolPipelineError) as rejected:
                await runtime.session.pipeline.execute(tool, {}, context=after)
            assert rejected.value.code == "precondition_failed"
            assert not (await runtime.store.load(owner=OWNER, session_id="session-a")).pending_effect_plan_id

            # 清空 A 不能误清同一 owner 的 B；冷重建也必须保持这一事实。
            other = await context_for(runtime.store, owner=OWNER, session="session-b")
            valid = await runtime.session.pipeline.execute(tool, {}, context=other)
            assert valid.effect_plan is not None
            assert valid.effect_plan.session_id == "session-b"
            assert valid.effect_plan.plan_id != initial.effect_plan.plan_id
    asyncio.run(run())
