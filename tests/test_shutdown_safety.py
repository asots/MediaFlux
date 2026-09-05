"""后台协调线程必须纳入统一关停判定。"""
from contextlib import ExitStack
from unittest.mock import MagicMock, patch
import pytest
from app import main

getters = (
    'app.modules.organize_tasks.get_organize_manager',
    'app.modules.agent_jobs_scheduler.get_agent_jobs_scheduler',
    'app.modules.agent_library_patrol_scheduler.get_agent_library_patrol_scheduler',
    'app.modules.agent_download_verification_scheduler.get_download_library_verification_scheduler',
    'app.modules.download_tracker.get_download_tracker',
    'app.modules.rss_scheduler.get_rss_scheduler',
    'app.modules.media_subscription_scheduler.get_media_subscription_scheduler',
    'app.modules.organize_scheduler.get_organize_scheduler',
    'app.modules.local_media_scheduler.get_local_media_scheduler',
    'app.modules.scheduler.get_scheduler',
    'app.modules.media_refresh_coordinator.get_media_refresh_coordinator',
)

@pytest.mark.parametrize("outcome,expected", [(True, True), (False, False), (RuntimeError("fixture failure"), False)])
def test_agent_runtime_shutdown_is_included_in_overall_safety(outcome, expected):
    with ExitStack() as stack:
        for name in getters:
            service = MagicMock()
            service.stop.return_value = True
            service.shutdown.return_value = True
            stack.enter_context(patch(name, return_value=service))
        for name in (
            'app.bot.stop_bot',
            'app.modules.organize_confirmations.stop_confirmation_dispatcher',
            'app.modules.telegram_notification_center.stop_telegram_notification_dispatcher',
        ):
            stack.enter_context(patch(name, return_value=True))
        kwargs = {'side_effect':outcome} if isinstance(outcome, Exception) else {'return_value':outcome}
        stop = stack.enter_context(patch('app.modules.agent_runtime.shutdown_agent_runtime', **kwargs))
        safe = main.stop_background_services()
        stop.assert_called_once_with()
        assert safe is expected
