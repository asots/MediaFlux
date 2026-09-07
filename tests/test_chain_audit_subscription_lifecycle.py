"""订阅调度器关闭超时或并发重启不能撤销旧 worker 的取消信号。"""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from app.modules.media_subscription_scheduler import MediaSubscriptionScheduler


class _Thread:
    def __init__(self, **_kwargs):
        self.alive = False

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass


class SubscriptionSchedulerLifecycleTests(unittest.TestCase):
    def test_start_after_stop_timeout_keeps_old_worker_cancelled(self):
        scheduler = MediaSubscriptionScheduler()
        worker = _Thread()
        worker.alive = True
        scheduler._workers = {1: worker}
        self.assertFalse(scheduler.stop(timeout=0))
        with patch(
            "app.modules.media_subscription_scheduler.threading.Thread", _Thread
        ):
            scheduler.start()
        self.assertTrue(scheduler._stop_event.is_set())
        self.assertFalse(scheduler._accepting)
        self.assertIsNone(scheduler._thread)
        self.assertIs(scheduler._workers[1], worker)

    def test_restart_after_old_workers_finish_reopens_one_scheduler(self):
        scheduler = MediaSubscriptionScheduler()
        worker = _Thread()
        worker.alive = True
        scheduler._workers = {1: worker}
        self.assertFalse(scheduler.stop(timeout=0))
        worker.alive = False
        with patch(
            "app.modules.media_subscription_scheduler.threading.Thread", _Thread
        ):
            scheduler.start()
            active = scheduler._thread
            scheduler.start()
        self.assertIs(scheduler._thread, active)
        self.assertTrue(scheduler._accepting)
        self.assertFalse(scheduler._stop_event.is_set())

    def test_start_during_join_waits_and_new_thread_reference_survives(self):
        scheduler = MediaSubscriptionScheduler()
        joined = threading.Event()
        release = threading.Event()
        attempted = threading.Event()
        restarted = threading.Event()
        results = []

        class JoiningThread(_Thread):
            def join(self, timeout=None):
                self.alive = False
                joined.set()
                release.wait(2)

        old = JoiningThread()
        old.alive = True
        scheduler._thread = old

        def restart():
            attempted.set()
            scheduler.start()
            restarted.set()

        stopping = threading.Thread(
            target=lambda: results.append(scheduler.stop(timeout=1))
        )
        starting = threading.Thread(target=restart)
        with patch(
            "app.modules.media_subscription_scheduler.threading.Thread", _Thread
        ):
            try:
                stopping.start()
                self.assertTrue(joined.wait(1))
                starting.start()
                self.assertTrue(attempted.wait(1))
                self.assertFalse(restarted.wait(0.05))
            finally:
                release.set()
                stopping.join(2)
                if starting.ident is not None:
                    starting.join(2)
        self.assertFalse(stopping.is_alive())
        self.assertFalse(starting.is_alive())
        self.assertEqual(results, [True])
        self.assertIsNotNone(scheduler._thread)
        self.assertIsNot(scheduler._thread, old)
        self.assertTrue(scheduler._thread.is_alive())

    def test_real_worker_keeps_cancellation_across_four_stop_restart_cycles(self):
        import asyncio
        from types import SimpleNamespace

        scheduler = MediaSubscriptionScheduler()
        observed = []
        with (
            patch.object(scheduler, "_loop", side_effect=scheduler._stop_event.wait),
            patch(
                "app.modules.media_subscription_scheduler.db.recover_stale_media_subscription_checks",
                return_value=0,
            ),
            patch(
                "app.modules.media_subscription_scheduler.db.list_due_media_subscriptions",
                return_value=[{"id": 1}],
            ),
        ):
            for cycle in range(4):
                entered = threading.Event()
                release = threading.Event()

                async def check(_subscription_id, *, trigger, cancel_event):
                    entered.set()
                    await asyncio.to_thread(release.wait, 2)
                    observed.append((trigger, cancel_event.is_set()))

                with (
                    self.subTest(cycle=cycle),
                    patch(
                        "app.modules.media_subscription_scheduler.get_media_subscription_service",
                        return_value=SimpleNamespace(check_subscription=check),
                    ),
                ):
                    scheduler.start()
                    control = scheduler._thread
                    self.assertEqual(scheduler.run_due(), 1)
                    self.assertTrue(entered.wait(1))
                    worker = scheduler._workers[1]
                    try:
                        self.assertFalse(scheduler.stop(timeout=0))
                        control.join(1)
                        self.assertFalse(control.is_alive())
                        scheduler.start()
                        self.assertTrue(scheduler._stop_event.is_set())
                        self.assertFalse(scheduler._accepting)
                    finally:
                        release.set()
                        worker.join(2)
                        self.assertTrue(scheduler.stop(timeout=1))
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(scheduler._workers, {})
        self.assertEqual(observed, [("scheduler", True)] * 4)
