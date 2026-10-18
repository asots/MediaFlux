"""SQLite 跨会话预览/确认竞态：已接受后台任务的冻结计划必须可恢复。"""

from __future__ import annotations

from threading import Event, Thread
from unittest.mock import Mock, patch

import pytest

from app.agent.domains.guangya import rename_actions as actions
from app.agent.models import ToolContext
from app.agent.session_context import SQLiteAgentSessionContextRepository
from app.modules import guangya_rename as plans
from tests.support import isolated_test_database
from tests.test_agent_guangya_rename import FakeGuangYaClient


@pytest.mark.parametrize("other_session_id", ["first-session", "second-session"])
def test_accepted_rename_survives_replacement_preview(tmp_path, other_session_id):
    with (
        isolated_test_database(),
        patch.object(plans, "_plan_directory", return_value=tmp_path),
        patch.object(plans, "get_web_secret", return_value="test-secret"),
        patch.object(actions, "GuangYaClient", FakeGuangYaClient),
    ):
        rename_actions = actions.GuangyaRenameActions(SQLiteAgentSessionContextRepository())
        entered, proceed = Event(), Event()
        accepted, failures = [], []
        thread = None
        try:
            context = ToolContext(owner="deep-rename-owner", session_id="first-session")
            other_session = ToolContext(
                owner=context.owner, session_id=other_session_id
            )
            arguments = actions.guangya_rename_preview_arguments(
                {"paths": ["/整理/动漫"], "mode": "remove_bitrate"}
            )
            assert rename_actions.preview_guangya_rename(arguments, context).ok
            prior = rename_actions._flow(context.owner, context.session_id)
            _, fingerprint = rename_actions.prepare_guangya_rename_confirmation({}, context)
            load, build = actions.load_rename_plan, actions.build_rename_plan
            manager = Mock()
            manager.start_durable_operation.return_value = {
                "ok": True,
                "queued": True,
                "task_id": "a" * 32,
            }

            def delayed_load(*args, **kwargs):
                # 确认已捕获旧 revision，暂停在加载冻结计划之前。
                entered.set()
                assert proceed.wait(5), "新预览未进入扫描"
                return load(*args, **kwargs)

            def confirm_old():
                try:
                    accepted.append(
                        rename_actions.execute_guangya_rename_confirmed(
                            {}, fingerprint, context
                        )
                    )
                except BaseException as exc:
                    failures.append(exc)

            def interleave(*args, **kwargs):
                # 另一会话开始预览时，原会话确认仍须保留独立的消费世代。
                proceed.set()
                assert thread is not None
                thread.join(5)
                assert not thread.is_alive() and not failures, failures
                return build(*args, **kwargs)

            with (
                patch.object(actions, "load_rename_plan", side_effect=delayed_load),
                patch.object(actions, "build_rename_plan", side_effect=interleave),
                patch(
                    "app.modules.organize_tasks.get_organize_manager",
                    return_value=manager,
                ),
            ):
                thread = Thread(target=confirm_old, daemon=True)
                thread.start()
                assert entered.wait(5), "旧确认未取得 flow"
                assert rename_actions.preview_guangya_rename(arguments, other_session).ok
            assert accepted[0].ok and accepted[0].status == "accepted"
            # 另一会话预览不会改变原会话的消费世代。
            assert accepted[0].data["requires_manual"] is (other_session_id == context.session_id)
            manager.start_durable_operation.assert_called_once()
            assert prior is not None
            queued = plans.load_rename_plan(
                prior.plan_id,
                expected_fingerprint=prior.fingerprint,
                require_confirmed=True,
            )
            assert len(queued["entries"]) == 2
        finally:
            proceed.set()
            if thread is not None:
                thread.join(5)
