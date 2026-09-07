"""刷新消费者的启停交接不能丢取消信号或覆盖尚未退出的 worker。"""

from __future__ import annotations

import threading
from unittest import TestCase
from unittest.mock import patch

from app.modules.media_refresh_coordinator import MediaRefreshCoordinator

# 保留真实测试控制线程；生产 Thread 工厂在单测中替换为确定性 worker。
_ControlThread = threading.Thread


class _Worker:
    def __init__(self, *, target, name, daemon):
        self.alive = False
        self.finish_on_join = True

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        if self.finish_on_join:
            self.alive = False


class MediaRefreshLifecycleAuditTests(TestCase):
    def test_start_waits_for_stop_handoff_instead_of_clearing_its_signal(self):
        coordinator = MediaRefreshCoordinator()
        stop_signalled = threading.Event()
        release_stop = threading.Event()
        start_attempted = threading.Event()
        start_returned = threading.Event()
        stopped = []
        errors = []
        original_set = coordinator._stop_event.set

        def pause_stop():
            original_set()
            stop_signalled.set()
            if not release_stop.wait(3):
                raise AssertionError("stop gate timed out")

        def stop():
            try:
                stopped.append(coordinator.stop(timeout=0.1))
            except BaseException as exc:
                errors.append(exc)

        def start():
            start_attempted.set()
            try:
                coordinator.start()
            except BaseException as exc:
                errors.append(exc)
            finally:
                start_returned.set()

        stopper = _ControlThread(target=stop, daemon=True)
        starter = _ControlThread(target=start, daemon=True)
        with (
            patch(
                "app.modules.media_refresh_coordinator.threading.Thread",
                side_effect=_Worker,
            ) as factory,
            patch.object(coordinator._stop_event, "set", side_effect=pause_stop),
        ):
            try:
                stopper.start()
                self.assertTrue(stop_signalled.wait(2))
                starter.start()
                self.assertTrue(start_attempted.wait(2))
                self.assertFalse(start_returned.wait(0.1))
                self.assertTrue(coordinator._stop_event.is_set())
                self.assertEqual(factory.call_count, 0)
            finally:
                release_stop.set()
                stopper.join(3)
                if starter.ident is not None:
                    starter.join(3)
        try:
            self.assertFalse(stopper.is_alive())
            self.assertFalse(starter.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(stopped, [True])
            self.assertTrue(start_returned.is_set())
            self.assertEqual(factory.call_count, 1)
            self.assertTrue(coordinator._thread.is_alive())
        finally:
            coordinator.stop()

    def test_timeout_keeps_live_worker_and_cancellation_until_it_exits(self):
        coordinator = MediaRefreshCoordinator()
        with patch(
            "app.modules.media_refresh_coordinator.threading.Thread",
            side_effect=_Worker,
        ) as factory:
            coordinator.start()
            first = coordinator._thread
            first.finish_on_join = False
            self.assertFalse(coordinator.stop(timeout=0.1))
            self.assertIs(coordinator._thread, first)
            coordinator.start()
            self.assertEqual(factory.call_count, 1)
            self.assertTrue(coordinator._stop_event.is_set())
            first.alive = False
            coordinator.start()
            self.assertEqual(factory.call_count, 2)
            self.assertIsNot(coordinator._thread, first)
            self.assertFalse(coordinator._stop_event.is_set())
            self.assertTrue(coordinator.stop())
            self.assertIsNone(coordinator._thread)

    def test_repeated_start_stop_and_thread_start_failure_are_recoverable(self):
        coordinator = MediaRefreshCoordinator()
        with patch(
            "app.modules.media_refresh_coordinator.threading.Thread",
            side_effect=_Worker,
        ) as factory:
            with patch.object(
                _Worker,
                "start",
                side_effect=RuntimeError("synthetic thread start failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "thread start failure"):
                    coordinator.start()
            self.assertFalse(coordinator._thread.is_alive())
            coordinator.start()
            for _ in range(10):
                coordinator.start()
            self.assertEqual(factory.call_count, 2)
            self.assertTrue(coordinator.stop())
            self.assertTrue(coordinator.stop())
