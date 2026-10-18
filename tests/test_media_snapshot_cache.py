"""媒体概览快照的缓存、失效与非阻塞页面契约；不访问网络。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from app import services
from app.clients.base import DashboardData, MediaItem


class _ObservedLock:
    """确定第二个调用已经排队，避免用 sleep 猜测并发时序。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.guard = threading.Lock()
        self.waiting = threading.Event()
        self.calls = 0

    def __enter__(self):
        with self.guard:
            self.calls += 1
            if self.calls == 2:
                self.waiting.set()
        self.lock.acquire()
        return self

    def __exit__(self, *_args):
        self.lock.release()


class MediaSnapshotCacheTests(unittest.TestCase):
    def setUp(self):
        services.clear_dashboard_cache()
        self.addCleanup(services.clear_dashboard_cache)
        self.now = 100.0
        self.values = {
            "JELLYFIN_ENABLED": "1", "JELLYFIN_URL": "http://media.test",
            "JELLYFIN_API_KEY": "fixture", "JELLYFIN_USER_ID": "user",
        }
        self.enterContext(patch("app.services.get", side_effect=lambda key, default="": self.values.get(key, default)))
        self.enterContext(patch("app.services.get_bool", side_effect=lambda key, default=False: key == "JELLYFIN_ENABLED"))
        self.enterContext(patch("app.services.get_int", return_value=30))
        self.enterContext(patch("app.services.time.monotonic", side_effect=lambda: self.now))
        self.dashboard = self.enterContext(patch("app.services._fetch_dashboards", side_effect=lambda **kw: [self.board()]))
        self.recent = self.enterContext(patch("app.services._fetch_recent_media", side_effect=lambda **kw: [self.source()]))

    @staticmethod
    def board(**changes):
        return replace(DashboardData(
            server_type="jellyfin", server_name="Test", online=True,
            total_items=42, movie_count=5,
            recent_added=[MediaItem(id="first", name="First", type="Movie")],
        ), **changes)

    @staticmethod
    def source(**changes):
        return {
            "server_type": "jellyfin", "server_name": "Test", "web_url": "http://media.test",
            "error": "", "items": [MediaItem(id=str(i), name=f"Movie {i}", type="Movie") for i in range(72)],
            **changes,
        }

    def test_cold_page_readers_never_load_remote_data(self):
        boards, loaded = services.get_cached_dashboards_or_stubs()
        sources, recent_loaded = services.get_cached_recent_media_or_stubs()
        self.assertFalse(loaded)
        self.assertFalse(recent_loaded)
        self.assertEqual(boards[0].server_type, "jellyfin")
        self.assertEqual(sources[0]["items"], [])
        self.dashboard.assert_not_called()
        self.recent.assert_not_called()

    def test_snapshot_timestamps_follow_process_timezone(self):
        # 在独立进程设置TZ，避免tzset污染并行测试或服务线程。
        script = """
import json
from app.clients.base import DashboardData
from app.services import _merge_dashboards, _merge_recent_media
boards, _, _ = _merge_dashboards([DashboardData(server_name='Test', server_type='jellyfin', online=True)], [])
sources, _, _ = _merge_recent_media([{'server_type': 'jellyfin', 'error': ''}], [])
print(json.dumps([boards[0].updated_at, sources[0]['updated_at']]))
"""
        for zone in ("Asia/Shanghai", "Asia/Kolkata", "America/New_York", "UTC"):
            with self.subTest(timezone=zone):
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "TZ": zone},
                    capture_output=True, text=True, check=False, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                for stamp in json.loads(result.stdout):
                    value = datetime.fromisoformat(stamp)
                    local = value.astimezone(ZoneInfo(zone))
                    self.assertEqual(value.utcoffset(), local.utcoffset())
                    self.assertEqual(stamp, local.isoformat(timespec="seconds"))

    def test_hot_and_expired_snapshots_are_reused_without_remote_reads(self):
        services.build_dashboards()
        services.build_recent_media()
        for now, stale in ((110, False), (131, True)):
            with self.subTest(now=now):
                self.now = now
                boards, loaded = services.get_cached_dashboards_or_stubs()
                sources, recent_loaded = services.get_cached_recent_media_or_stubs()
                self.assertTrue(loaded and recent_loaded)
                self.assertEqual(boards[0].total_items, 42)
                self.assertEqual(len(sources[0]["items"]), 72)
                self.assertEqual(boards[0].stale, stale)
                self.assertEqual(sources[0]["stale"], stale)
        self.dashboard.assert_called_once()
        self.recent.assert_called_once()

    def test_failed_reads_preserve_data_and_do_not_extend_stale_lifetime(self):
        old = services.build_dashboards()[0]
        recent = services.build_recent_media()[0]
        self.dashboard.side_effect = lambda **kw: [self.board(online=False, error="unavailable", total_items=0)]
        self.recent.side_effect = lambda **kw: [self.source(error="unavailable", items=[])]
        for now in (131, 150, 399):
            self.now = now
            board = services.build_dashboards()[0]
            source = services.build_recent_media()[0]
            self.assertEqual(board.total_items, 42)
            self.assertEqual(board.updated_at, old.updated_at)
            self.assertFalse(board.online)
            self.assertTrue(board.stale and board.refresh_error)
            self.assertEqual(source["items"], recent["items"])
            self.assertEqual(source["updated_at"], recent["updated_at"])
            self.assertTrue(source["stale"])
        self.now = 400
        self.assertFalse(services.get_cached_dashboards_or_stubs()[1])
        self.assertFalse(services.get_cached_recent_media_or_stubs()[1])
        self.assertEqual(services.build_dashboards()[0].total_items, 0)
        self.assertEqual(services.build_recent_media()[0]["items"], [])

    def test_partial_failure_retains_only_failed_fields(self):
        old = services.build_dashboards()[0]
        self.now = 131
        self.dashboard.side_effect = lambda **kw: [self.board(total_items=90, movie_count=0, partial_errors=["media_counts"])]
        board = services.build_dashboards()[0]
        self.assertEqual(board.total_items, 90)
        self.assertEqual(board.movie_count, 5)
        self.assertEqual(board.updated_at, old.updated_at)
        self.assertTrue(board.online and board.stale)
        self.assertEqual(board.partial_errors, ["media_counts"])

    def test_aggregate_exception_retains_snapshot_but_not_after_expiry(self):
        services.build_dashboards()
        self.now = 131
        self.dashboard.side_effect = RuntimeError("fixture unavailable")
        board = services.build_dashboards()[0]
        self.assertEqual(board.total_items, 42)
        self.assertTrue(board.stale)
        self.now = 401
        with self.assertRaises(RuntimeError):
            services.build_dashboards()

    def test_successfully_empty_results_replace_previous_items(self):
        services.build_dashboards()
        services.build_recent_media()
        self.now = 131
        self.dashboard.side_effect = lambda **kw: [self.board(total_items=0, recent_added=[])]
        self.recent.side_effect = lambda **kw: [self.source(items=[])]
        board = services.build_dashboards()[0]
        source = services.build_recent_media()[0]
        self.assertEqual(board.total_items, 0)
        self.assertEqual(board.recent_added, [])
        self.assertFalse(board.stale)
        self.assertEqual(source["items"], [])
        self.assertFalse(source["stale"])
        self.assertTrue(source["updated_at"])

    def test_soft_invalidation_preserves_both_snapshots_and_hard_clear_removes_them(self):
        services.build_dashboards()
        services.build_recent_media()
        services.clear_dashboard_cache(keep_stale=True)
        self.assertTrue(services.get_cached_dashboards_or_stubs()[0][0].stale)
        self.assertTrue(services.get_cached_recent_media_or_stubs()[0][0]["stale"])
        services.clear_dashboard_cache()
        self.assertFalse(services.get_cached_dashboards_or_stubs()[1])
        self.assertFalse(services.get_cached_recent_media_or_stubs()[1])

    def test_changed_server_token_user_or_enabled_flag_never_reuses_snapshots(self):
        for key in self.values:
            with self.subTest(key=key):
                services.clear_dashboard_cache()
                services.build_dashboards()
                services.build_recent_media()
                original = self.values[key]
                self.values[key] = original + "changed"
                self.assertFalse(services.get_cached_dashboards_or_stubs()[1])
                self.assertFalse(services.get_cached_recent_media_or_stubs()[1])
                self.values[key] = original

    def test_recent_limit_is_part_of_cache_identity(self):
        services.build_recent_media(limit=8)
        self.assertFalse(services.get_cached_recent_media_or_stubs(limit=72)[1])
        services.build_recent_media(limit=72)
        services.build_recent_media(limit=72)
        self.assertEqual(self.recent.call_count, 2)
        self.recent.assert_called_with(limit=72)

    def test_dashboard_reuses_fresh_large_recent_snapshot_and_closes_client(self):
        source = services.build_recent_media()[0]
        client = Mock()
        client.get_dashboard.return_value = self.board()
        services._load_dashboard(client, server_type="jellyfin", web_url="http://media.test")
        client.get_dashboard.assert_called_once_with(recent_added=source["items"][:8])
        client.close.assert_called_once()
        self.now = 131
        client.reset_mock()
        services._load_dashboard(client, server_type="jellyfin", web_url="http://media.test")
        client.get_dashboard.assert_called_once_with()
        client.close.assert_called_once()

    def test_clear_invalidates_recent_before_any_dashboard_can_reuse_it(self):
        services.build_recent_media()
        original = services._recent_media_cache.invalidate
        def between_caches(**kwargs):
            original(**kwargs)
            self.assertIsNone(services._recent_media_cache.peek((*services._dashboard_config_key(), "72")))
            services.build_dashboards()
        with patch.object(services._recent_media_cache, "invalidate", side_effect=between_caches):
            services.clear_dashboard_cache(keep_stale=True)
        self.assertTrue(services.get_cached_dashboards_or_stubs()[0][0].stale)

    def test_parallel_reads_merge_but_forced_reads_do_not_reuse_normal_refresh(self):
        for first_force, second_force, expected in ((False, False, 1), (True, True, 1), (False, True, 2)):
            with self.subTest(first=first_force, second=second_force):
                services.clear_dashboard_cache()
                started, release = threading.Event(), threading.Event()
                gate = _ObservedLock()
                calls, errors, results = [], [], []
                def loader(calls=calls, started=started, release=release, **kwargs):
                    calls.append(kwargs)
                    if len(calls) == 1:
                        started.set()
                        if not release.wait(3):
                            raise AssertionError("refresh not released")
                    return [self.board(total_items=len(calls))]
                def invoke(force, results=results, errors=errors):
                    try:
                        results.append(services.build_dashboards(force=force))
                    except (RuntimeError, AssertionError) as exc:
                        errors.append(exc)
                with patch.object(services._dashboard_cache, "_refresh_lock", gate):
                    self.dashboard.side_effect = loader
                    threads = [threading.Thread(target=invoke, args=(flag,)) for flag in (first_force, second_force)]
                    threads[0].start()
                    try:
                        self.assertTrue(started.wait(3))
                        threads[1].start()
                        self.assertTrue(gate.waiting.wait(3))
                    finally:
                        release.set()
                        for thread in threads:
                            if thread.ident is not None:
                                thread.join(3)
                    self.assertFalse(any(thread.is_alive() for thread in threads))
                self.assertEqual(errors, [])
                self.assertEqual(len(results), 2)
                self.assertEqual(len(calls), expected)
                if second_force:
                    self.assertEqual(calls[-1], {"reuse_recent": False})

    def test_invalidated_inflight_read_cannot_repopulate_cache(self):
        started, release = threading.Event(), threading.Event()
        errors = []
        def loader(**kwargs):
            started.set()
            if not release.wait(3):
                raise AssertionError("refresh not released")
            return [self.board()]
        def invoke():
            try:
                services.build_dashboards()
            except RuntimeError as exc:
                errors.append(str(exc))
        self.dashboard.side_effect = loader
        thread = threading.Thread(target=invoke)
        thread.start()
        try:
            self.assertTrue(started.wait(3))
            services.clear_dashboard_cache()
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["媒体配置或内容已更新，请重试"])
        self.assertFalse(services.get_cached_dashboards_or_stubs()[1])
