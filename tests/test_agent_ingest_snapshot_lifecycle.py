"""入库只消费真实会话引用；冷重建与生命周期变化不依赖进程缓存。"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app import database as db
from app.agent.errors import ToolPipelineError
from tests.support import initialized_web_runtime
from tests.test_agent_effect_admission import _OfflineClient, _offline_runtime
from tests.test_agent_guangya_preview_lifecycle import OWNER, runtime_for_test
from tests.test_agent_kernel_existing_ports import context_for

MAGNET = "magnet:?xt=urn:btih:" + "a" * 40
SHARE = "https://www.guangyapan.com/s/1938060615538466851_aecd7hu726g3whGl#/share"


async def inspect(runtime, session, source):
    context = await context_for(runtime.store, owner=OWNER, session=session)
    result = await runtime.session.pipeline.execute("ingest.inspect", {
        "source_type": source, "input": MAGNET if source == "direct_url" else SHARE,
    }, context=context)
    public = result.outcome.public_content
    assert MAGNET not in str(public) and SHARE not in str(public)
    return public["reference_arguments"]["ingest_snapshot_ref"]


async def submit(runtime, session, source, reference=None):
    context = await context_for(runtime.store, owner=OWNER, session=session)
    arguments = {"source_type": source, "target": "qb" if source == "direct_url" else "guangya"}
    if reference is not None:
        arguments["ingest_snapshot_ref"] = reference
    return await runtime.session.pipeline.execute("ingest.submit", arguments, context=context)


@pytest.mark.parametrize("source", ["direct_url", "guangya_share"])
@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
@pytest.mark.parametrize("operation", ["reset", "delete"])
def test_ingest_reference_is_the_only_input_across_runtime_and_session_lifecycle(tmp_path, cold, operation, source):
    async def run():
        with initialized_web_runtime(), _offline_runtime(tmp_path, _OfflineClient()), patch(
            "app.agent.domains.resources.ingest_actions.download_target_readiness", side_effect=lambda target: {target: True},
        ), patch("app.agent.domains.resources.ingest_actions.inspect_share_for_transfer", return_value={
            "preview_id": "private-preview", "share_id": "private-share",
            "files": [{"id": "private-file", "name": "Episode.mkv", "is_dir": False, "size": 10}],
            "target_id": "private-target", "target_name": "默认目录", "expires_in": 900,
        }):
            runtime = runtime_for_test()
            first = await inspect(runtime, "ingest-a", source)
            second = await inspect(runtime, "ingest-b", source)
            if cold:
                runtime = runtime_for_test()

            # 曾经 inspect 也必须带引用；不能借暖进程中的 latest 快照继续提交。
            with pytest.raises(ToolPipelineError) as missing:
                await submit(runtime, "ingest-a", source)
            assert missing.value.code == "invalid_arguments"
            with pytest.raises(ToolPipelineError) as cross_session:
                await submit(runtime, "ingest-b", source, first)
            assert cross_session.value.code == "reference_invalid"
            prepared = await submit(runtime, "ingest-a", source, first)
            assert prepared.effect_plan is not None

            await getattr(runtime.lifecycle, operation)(owner=OWNER, session_id="ingest-a")
            if cold:
                runtime = runtime_for_test()
            with pytest.raises(ToolPipelineError) as expired:
                await submit(runtime, "ingest-a", source, first)
            assert expired.value.code == "reference_invalid"
            with pytest.raises(ToolPipelineError) as missing_after_reset:
                await submit(runtime, "ingest-a", source)
            assert missing_after_reset.value.code == "invalid_arguments"
            assert not (await runtime.store.load(owner=OWNER, session_id="ingest-a")).pending_effect_plan_id

            unaffected = await submit(runtime, "ingest-b", source, second)
            assert unaffected.effect_plan is not None
            assert unaffected.effect_plan.session_id == "ingest-b"
            fresh = await inspect(runtime, "ingest-a", source)
            assert fresh != first
            assert (await submit(runtime, "ingest-a", source, fresh)).effect_plan is not None
            # 预检仅创建确认计划，不会提前创建下载请求。
            with db.get_conn() as conn:
                assert conn.execute("SELECT COUNT(*) FROM download_requests").fetchone()[0] == 0
    asyncio.run(run())
