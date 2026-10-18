"""无需变更的真实领域预检应正常结束，不签发确认票据或执行写入。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import config
from app import database as db
from app.agent.domain_catalog import build_tool_specs
from app.agent.domains.configuration.feature_actions import verify_feature_state_write
from app.agent.domains.resources.indexer_config_actions import verify_indexer_sites_write
from app.agent.kernel.capabilities import CapabilityRetriever
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import InMemorySessionStateStore
from app.agent.kernel.state_types import AgentInput
from app.agent.models import Evidence, ToolResult
from tests.support import isolated_test_database
from tests.test_agent_kernel_core import ScriptedModel, collect
from tests.test_agent_kernel_existing_ports import context_for


_WATCHLIST_ARGUMENTS = {"provider": "tmdb", "external_id": "8801", "media_type": "movie"}
_NO_CHANGES_CASES = (
    ("config.set_indexer_sites", {"site_ids": ["nyaa"]}),
    ("library.set_patrol_policy", {"enabled": False}),
    ("guangya.organize.set_schedule_policy", {"enabled": False}),
    ("guangya.organize.run_once", {}),
    ("discovery.add_watchlist", _WATCHLIST_ARGUMENTS),
    ("rss.retry_failed", {"limit": 10}),
    ("rss.submit_pending_to_qb", {"limit": 10}),
    ("strm.retry_failures", {"scope": "all"}),
    ("config.set_feature_state", {"feature": "discovery", "enabled": True}),
    ("strm.set_schedule_policy", {"enabled": False}),
    ("config.set_safe_policy", {"policy": "tmdb_match_mode", "value": "strict"}),
)


class NoChangesContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(isolated_test_database())
        directory = self.enterContext(tempfile.TemporaryDirectory())
        env_file = Path(directory) / "user.env"
        values = {
            "DISCOVERY_ENABLED": "1",
            "INDEXER_ENABLED_SITES": "nyaa",
            "INDEXER_SITE_CONFIG_VERSION": "2",
            "INDEXER_SUKEBEI_ENABLED": "0",
            "AGENT_LIBRARY_PATROL_ENABLED": "0",
            "GY_ORGANIZE_SCHEDULE_ENABLED": "0",
            "GY_ORGANIZE_SOURCE_DIRS": '[{"id":"source","name":"Source"}]',
            "GY_ORGANIZE_TARGET_DIR": "target",
            "STRM_SCHEDULE_ENABLED": "0",
            "TMDB_MATCH_MODE": "strict",
        }
        config.write_env_file(env_file, values, replace=False)
        self.enterContext(patch.dict(os.environ, {key: "" for key in values}))
        self.enterContext(patch.object(config, "ENV_FILE", env_file))
        self.enterContext(patch.object(config, "_cache", None))
        self.enterContext(patch.object(config, "_STARTUP_ENV_OVERRIDES", frozenset()))
        self.writer = self.enterContext(patch.object(config, "update_runtime_env_file"))
        self.discovery = self.enterContext(patch(
            "app.agent.domains.discovery.watchlist_actions.get_discovery_service"
        ))
        self.manager = Mock()
        self.manager.task_status.return_value = {"status": "idle"}
        self.enterContext(patch(
            "app.agent.domains.guangya.organize_actions.get_organize_manager",
            return_value=self.manager,
        ))
        self.enterContext(patch(
            "app.agent.domains.guangya.organize_actions.GuangYaClient",
            return_value=SimpleNamespace(logged_in=True, credential_generation=1, close=Mock()),
        ))
        self.organizer = Mock()
        self.organizer.organize.return_value = ([], {"total": 0})
        self.enterContext(patch(
            "app.agent.domains.guangya.organize_actions.Organizer",
            return_value=self.organizer,
        ))
        self.enterContext(patch(
            "app.modules.rss.capture_rss_qb_runtime_config",
            return_value=({"url": "http://qb.invalid", "default_method": "qb"}, ""),
        ))
        self.rss_submit = self.enterContext(patch("app.modules.rss.RSSEngine.submit_snapshot"))
        self.enterContext(patch(
            "app.modules.strm.capture_strm_retry_runtime_config",
            return_value=({"sources": [{"id": "source"}], "strm_root": "/strm"}, ""),
        ))
        self.strm_retry = self.enterContext(patch("app.modules.strm.retry_strm_failures"))
        db.add_media_watchlist("tmdb", "8801", "movie", "已收藏影片", "2026", "")
        names = {name for name, _arguments in _NO_CHANGES_CASES}
        self.catalog = catalog_from_tool_specs([
            spec for spec in build_tool_specs() if spec.name in names
        ])
        self.state = InMemorySessionStateStore()
        self.pipeline = ToolPipeline(catalog=self.catalog, state_store=self.state)
        self.freeze = self.enterContext(patch.object(
            self.pipeline.effect_store, "freeze", wraps=self.pipeline.effect_store.freeze
        ))

    def tearDown(self):
        self.freeze.assert_not_called()
        self.writer.assert_not_called()
        self.discovery.assert_not_called()
        self.manager.start.assert_not_called()
        self.rss_submit.assert_not_called()
        self.strm_retry.assert_not_called()

    async def test_all_eleven_domain_preparations_finish_without_a_confirmation(self):
        self.assertEqual(len(_NO_CHANGES_CASES), 11)
        for name, arguments in _NO_CHANGES_CASES:
            with self.subTest(tool=name):
                context = await context_for(self.state, session=name)
                result = await self.pipeline.execute(name, arguments, context=context)
                public = result.outcome.public_content
                self.assertIs(public["ok"], True)
                self.assertEqual(public["status"], "no_changes")
                self.assertEqual(public["error"], "")
                self.assertTrue(public["summary"])
                self.assertIsNone(result.effect_plan)
                state = await self.state.load(owner=context.owner, session_id=name)
                self.assertFalse(state.pending_effect_plan_id)
        self.organizer.organize.assert_called_once()
        self.assertIs(self.organizer.organize.call_args.kwargs["dry_run"], True)

    async def test_duplicate_watchlist_finishes_a_real_session_successfully(self):
        tool = self.catalog.get("discovery.add_watchlist")
        model = ScriptedModel([
            [ModelEvent(ModelEventType.TOOL_CALL_COMPLETED,
                        tool_call=ModelToolCall("already-saved", tool.model_name, _WATCHLIST_ARGUMENTS)),
             ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")],
            [ModelEvent(ModelEventType.TEXT_DELTA, text="该影视条目已在探索收藏中，无需重复添加。"),
             ModelEvent(ModelEventType.FINISH, finish_reason="stop")],
        ])
        session = AgentSession(
            model=model, catalog=self.catalog, retriever=CapabilityRetriever(),
            pipeline=self.pipeline, state_store=self.state,
        )
        events = await collect(session.run(AgentInput(
            message="把 tmdb 编号 8801 的电影加入探索收藏", owner="owner", session_id="watchlist",
        )))
        self.assertEqual(events[-1].type, AgentEventType.TURN_COMPLETED,
                         [event.to_dict() for event in events])
        self.assertEqual(events[-1].payload["status"], "success")
        self.assertEqual(len(model.requests), 2)
        results = [event.payload["result"] for event in events
                   if event.type == AgentEventType.TOOL_COMPLETED]
        self.assertEqual(len(results), 1)
        self.assertIs(results[0]["ok"], True)
        self.assertEqual(results[0]["status"], "no_changes")
        self.assertEqual(results[0]["error"], "")
        self.assertFalse(any(event.type in {
            AgentEventType.TOOL_FAILED, AgentEventType.EFFECT_APPROVAL_REQUIRED,
            AgentEventType.EFFECT_COMPLETED, AgentEventType.EFFECT_FAILED,
        } for event in events))
        facts = json.loads(next(message.content for message in model.requests[1].messages
                                if message.role == "tool"))
        self.assertEqual((facts["ok"], facts["status"]), (True, "no_changes"))
        state = await self.state.load(owner="owner", session_id="watchlist")
        self.assertFalse(state.pending_effect_plan_id)
        self.assertEqual(len(db.list_media_watchlist()), 1)

    async def test_write_verifiers_preserve_no_changes_without_reading_persistence(self):
        cases = (
            (verify_feature_state_write, {"feature": "discovery", "enabled": True}),
            (verify_indexer_sites_write, {"site_ids": ["nyaa"]}),
        )
        for verifier, arguments in cases:
            with self.subTest(verifier=verifier.__name__):
                result = ToolResult(True, "no_changes", "已经符合目标状态",
                                    evidence=[Evidence("server_configuration", "无需写入", "now")])
                with patch.object(config, "read_env_snapshot", return_value=(None, {})) as readback:
                    checked = verifier(arguments, result)
                self.assertIs(checked, result)
                readback.assert_not_called()
