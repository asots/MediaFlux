"""STRM 通知不把清理保护重复计为错误，不虚构后续重试。"""
import json
from unittest.mock import patch

import pytest

from app import database as db
from app.clients.guangya import GuangYaFile
from app.modules.scheduler import STRMScheduler
from app.modules.strm import _directory_fingerprint, _source_snapshot_directories
from app.modules.strm_notifications import (
    ROUND_CLEANUP_BLOCKED_REASON,
    cleanup_notice_lines,
)
from tests.support import isolated_test_database


def test_round_guard_reason_is_not_repeated_with_source_prefixes():
    aggregate = STRMScheduler._empty_stats()
    aggregate.update(clean_skipped=True, failed=1, error_samples=[ROUND_CLEANUP_BLOCKED_REASON])
    source = STRMScheduler._empty_stats()
    source.update(clean_skipped=True, error_samples=[
        '扫描目录 source：扫描期间目录状态发生变化：目录读取与全量快照不一致',
        ROUND_CLEANUP_BLOCKED_REASON,
    ])
    STRMScheduler._merge_source_stats(aggregate, source, '整理')
    other = STRMScheduler._empty_stats()
    other.update(clean_skipped=True, error_samples=[ROUND_CLEANUP_BLOCKED_REASON])
    STRMScheduler._merge_source_stats(aggregate, other, 'NSFW')
    assert len(aggregate['error_samples']) == 2
    assert aggregate['error_samples'].count(ROUND_CLEANUP_BLOCKED_REASON) == 1
    event = STRMScheduler._build_success_event(aggregate, {}, 1.0, 'organize', [], '/tmp/strm')
    assert sum('为避免误删' in line for line in event.lines) == 1
    errors = [line for line in event.lines if line.startswith('错误摘要：')]
    assert len(errors) == 1
    assert '整理/扫描目录 source' in errors[0]
    assert not any('已排队' in line for line in event.lines)


@pytest.mark.parametrize('queued', [False, True])
def test_followup_notice_requires_observed_queue(queued):
    notes = cleanup_notice_lines({'clean_skipped': True, 'organize_followup_queued': queued})
    assert any('本轮删除型清理' in line for line in notes)
    assert any('另有整理联动已排队' in line for line in notes) is queued
    assert not any('自动重试' in line for line in notes)
    assert cleanup_notice_lines({'organize_followup_queued': queued}) == ()


def test_snapshot_change_still_raises_with_clear_explanation():
    source = GuangYaFile('root', '整理', True)
    video = GuangYaFile('file', 'Movie.mkv', False, parent_id='root')
    with pytest.raises(RuntimeError, match='扫描期间目录状态发生变化'):
        _source_snapshot_directories(
            {'root': source, 'file': video}, 'root', {'root': _directory_fingerprint([])},
        )


@pytest.mark.parametrize('queued,stopping', [(False, False), (True, False), (True, True)])
def test_terminal_result_records_actual_pending_queue_without_starting_it(queued, stopping, tmp_path):
    with isolated_test_database():
        scheduler = STRMScheduler()
        pending = {'organize_changes': [], 'trigger_type': 'organize'} if queued else None
        scheduler._pending_organize_options = pending
        stats = scheduler._empty_stats()
        stats.update(clean_skipped=True, failed=1, error_samples=[ROUND_CLEANUP_BLOCKED_REASON])
        sources = [{'id': 'source', 'name': '整理', 'rel_prefix': '整理'}]
        values = {'GY_STRM_BASE_URL': 'http://media.invalid', 'STRM_ROOT': str(tmp_path)}

        def refresh():
            if stopping:
                scheduler._stop_event.set()
            return {}

        with (
            patch.object(scheduler, 'validate_config', return_value=''),
            patch.object(scheduler, '_source_dirs', return_value=sources),
            patch.object(scheduler, '_run_full_sources', return_value=(stats, [], False)),
            patch.object(scheduler, '_refresh_media_servers', side_effect=lambda **_kwargs: refresh()),
            patch.object(scheduler, '_notify_scoped_results') as notify,
            patch('app.modules.scheduler._publish_linked_notification_threads'),
            patch('app.modules.scheduler.get', side_effect=lambda key, default='': values.get(key, default)),
            patch('app.modules.scheduler.get_bool', return_value=False),
            patch('app.modules.strm_metadata_worker.get_strm_metadata_worker'),
        ):
            result = scheduler.run_blocking('organize', force_full=True)
        assert result['ok'] is True
        assert result['partial'] is True
        assert result['stats']['organize_followup_queued'] is (queued and not stopping)
        assert scheduler._pending_organize_options is pending
        assert scheduler._pending_thread is None
        assert scheduler._run_lock.acquire(blocking=False)
        scheduler._run_lock.release()
        notify.assert_called_once()
        saved = json.loads(db.get_last_task_run('strm_sync')['result'])
        assert saved['stats']['organize_followup_queued'] is (queued and not stopping)
        assert db.get_last_task_run('strm_sync')['status'] == 'partial'
