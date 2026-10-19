"""持久确认在真实配置发布和独立冷进程中保留相同的授权边界。"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app import config, database as db
from app.agent.kernel.bootstrap import get_agent_kernel_runtime, invalidate_agent_kernel_runtime
from app.agent.kernel.model import ModelEvent, ModelEventType
from app.agent.kernel.pipeline import InMemoryRateLimiter
from app.clients.qbittorrent import TorrentAddResult
from tests.support import initialized_web_runtime
from tests.test_agent_kernel_core import ScriptedModel
from tests.test_agent_kernel_existing_ports import context_for

OWNER = "webk:v1:" + "a" * 64
OLD = {"QB_URL": "http://old-qb.invalid", "TG_QB_SAVE_PATH": "/old-target", "TG_QB_CATEGORY": "old"}
NEW = {"QB_URL": "http://new-qb.invalid", "TG_QB_SAVE_PATH": "/new-target", "TG_QB_CATEGORY": "new"}


class RecordingQB:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.url = kwargs["url"]

    def add_torrent_detailed(self, **kwargs):
        self.calls.append({"url": self.url, "path": kwargs["save_path"], "category": kwargs["category"]})
        return TorrentAddResult(ok=True)

    def close(self):
        pass


def runtime():
    current = get_agent_kernel_runtime()
    current.session.pipeline.rate_limiter = InMemoryRateLimiter(limit=100)
    current.session.model = ScriptedModel([[ModelEvent(ModelEventType.FINISH)]])
    return current


async def prepare(current, session_id):
    context = await context_for(current.store, owner=OWNER, session=session_id)
    inspected = await current.session.pipeline.execute(
        "ingest.inspect", {"input": "magnet:?xt=urn:btih:" + "b" * 40}, context=context,
    )
    reference = inspected.outcome.public_content["reference_arguments"]["ingest_snapshot_ref"]
    result = await current.session.pipeline.execute(
        "ingest.submit", {"source_type": "direct_url", "target": "qb", "ingest_snapshot_ref": reference}, context=context,
    )
    assert result.effect_plan is not None
    return result.effect_plan


async def confirm(current, session_id, plan_id):
    return [event.to_dict() async for event in current.session.confirm(owner=OWNER, session_id=session_id, plan_id=plan_id)]


def save(values):
    from app.routes import api
    from starlette.requests import Request
    request = Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(background_services_enabled=False))})
    with patch.object(api, "require_api_login"):
        assert api.save_config(request, values) == {"success": True}


def cold_confirm(paths, plan_id):
    env = {key: value for key, value in os.environ.items() if key not in config.all_items()}
    env.update({
        **config._SYSTEM_ENV_SNAPSHOT,
        "PYTHONPATH": os.getcwd(),
        "MEDIAFLUX_TEST_MODE": "0", "MEDIAFLUX_TEST_DB_PATH": "",
        **{f"MEDIAFLUX_{name.upper()}_DIR": str(getattr(paths, f"{name}_dir"))
           for name in ("data", "config", "cache", "log", "strm")},
    })
    result = subprocess.run(
        [sys.executable, __file__, plan_id],
        env=env, input=json.dumps(config._SYSTEM_ENV_SNAPSHOT), capture_output=True, text=True, timeout=30, check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("change", ["unchanged", "changed", "aba"])
def test_confirmed_destination_survives_restart_only_when_authority_is_unchanged(change):
    async def scenario():
        with initialized_web_runtime() as paths, patch("app.agent.feature_gate._runtime_generation", 0):
            config.set_and_save({**OLD, "AGENT_LLM_API_URL": "http://model.invalid/v1", "AGENT_LLM_MODEL": "test-model"})
            invalidate_agent_kernel_runtime()
            try:
                # 非零本地代次下的卡也必须在同配置重启后有效。
                save(NEW)
                plan = await prepare(runtime(), "cold-confirm")
                if change != "unchanged":
                    save(OLD)
                if change == "aba":
                    save(NEW)
                result = cold_confirm(paths, plan.plan_id)
                completed = [event for event in result["events"] if event["type"] == "effect.completed"]
                if change == "unchanged":
                    assert len(completed) == 1
                    assert result["calls"] == [{"url": NEW["QB_URL"], "path": NEW["TG_QB_SAVE_PATH"], "category": NEW["TG_QB_CATEGORY"]}]
                else:
                    assert not completed
                    assert result["calls"] == []
                    assert any(event["payload"].get("code") == "confirmation_stale" for event in result["events"])
            finally:
                invalidate_agent_kernel_runtime()
    asyncio.run(scenario())


def test_zero_generation_old_confirmation_is_rejected_after_configuration_change_and_restart():
    async def scenario():
        with initialized_web_runtime() as paths, patch("app.agent.feature_gate._runtime_generation", 0):
            config.set_and_save({**OLD, "AGENT_LLM_API_URL": "http://model.invalid/v1", "AGENT_LLM_MODEL": "test-model"})
            invalidate_agent_kernel_runtime()
            try:
                plan = await prepare(runtime(), "cold-confirm")
                save(NEW)
                result = cold_confirm(paths, plan.plan_id)
                assert result["calls"] == []
                assert any(event["payload"].get("code") == "confirmation_stale" for event in result["events"])
            finally:
                invalidate_agent_kernel_runtime()
    asyncio.run(scenario())


@pytest.mark.parametrize("tool,updates,arguments", [
    ("config.create_media_path_mapping", {"JELLYFIN_PATH_MAPPINGS": "[]"},
     {"provider": "jellyfin", "local_path": "/media/strm/movies", "server_path": "/mnt/media/movies"}),
    ("library.set_patrol_policy", {"AGENT_LIBRARY_PATROL_ENABLED": "0"}, {"enabled": True}),
])
def test_configuration_writer_verifies_and_reloads_its_committed_snapshot(tool, updates, arguments):
    from app.modules.agent_library_patrol_scheduler import AgentLibraryPatrolScheduler

    async def scenario():
        with initialized_web_runtime():
            config.set_and_save({**updates, "AGENT_LLM_API_URL": "http://model.invalid/v1", "AGENT_LLM_MODEL": "test-model"})
            invalidate_agent_kernel_runtime()
            try:
                current = runtime()
                context = await context_for(current.store, owner=OWNER, session="config-writer")
                preview = await current.session.pipeline.execute(tool, arguments, context=context)
                assert preview.effect_plan is not None
                scheduler = AgentLibraryPatrolScheduler()
                with patch("app.modules.agent_library_patrol_scheduler.get_agent_library_patrol_scheduler", return_value=scheduler):
                    events = await confirm(current, "config-writer", preview.effect_plan.plan_id)
                result = next(event["payload"]["result"] for event in events if event["type"] == "effect.completed")
                assert result["ok"] and result["status"] == "completed"
                values = config.read_env_snapshot(config.ENV_FILE)[1]
                if tool == "library.set_patrol_policy":
                    assert values["AGENT_LIBRARY_PATROL_ENABLED"] == "1"
                    assert result["data"]["policy"]["enabled"] is True
                    assert result["data"]["runtime_refreshed"] is True
                else:
                    assert len(json.loads(values["JELLYFIN_PATH_MAPPINGS"])) == 1
                    assert result["data"]["count"] == 1
            finally:
                invalidate_agent_kernel_runtime()
    asyncio.run(scenario())


async def child(plan_id):
    # tests 包的默认隔离不能把子进程重定向到另一安装的数据库。
    os.environ["MEDIAFLUX_TEST_MODE"] = "0"
    os.environ["MEDIAFLUX_TEST_DB_PATH"] = ""
    db.configure_database(config.PATHS.database_path, test_mode=False)
    # 安装夹具改了 PATHS，但父进程的启动覆盖仍来自测试包初始环境；
    # 传递同一份部署覆盖，避免临时目录差异伪装成配置变化。
    startup_overrides = json.loads(sys.stdin.read())
    with patch.object(config, "_SYSTEM_ENV_SNAPSHOT", startup_overrides), patch("app.modules.download_dispatcher.QBittorrentClient", RecordingQB), patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
        events = await confirm(runtime(), "cold-confirm", plan_id)
        print(json.dumps({"events": events, "calls": RecordingQB.calls}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(child(sys.argv[1]))
