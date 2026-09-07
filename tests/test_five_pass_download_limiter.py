"""反复 asyncio.run/取消后释放索引下载并发门，不保留已关闭事件循环。"""
from __future__ import annotations

import asyncio
import gc
import unittest
import weakref

from app.indexers import downloads


class DownloadLimiterLifecycleTests(unittest.TestCase):
    @staticmethod
    async def _contended_batch(*, cancel_waiter=False):
        loop_ref = weakref.ref(asyncio.get_running_loop())
        limiter = downloads._download_limiter()
        limiter_ref = weakref.ref(limiter)
        active = peak = completed = 0
        full = asyncio.Event()
        release = asyncio.Event()

        async def worker():
            nonlocal active, peak, completed
            async with downloads._download_limiter():
                active += 1
                peak = max(peak, active)
                if active == downloads._DOWNLOAD_LIMIT:
                    full.set()
                try:
                    await release.wait()
                    completed += 1
                finally:
                    active -= 1

        tasks = [asyncio.create_task(worker()) for _ in range(9)]
        await asyncio.wait_for(full.wait(), 2)
        if cancel_waiter:
            tasks[-1].cancel()
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        return loop_ref, limiter_ref, peak, completed

    def test_contended_closed_loops_are_not_retained_across_repeated_runs(self):
        refs = [asyncio.run(self._contended_batch()) for _ in range(12)]
        gc.collect()
        self.assertEqual([row[2:] for row in refs], [(downloads._DOWNLOAD_LIMIT, 9)] * 12)
        self.assertTrue(all(loop() is None for loop, _limiter, _peak, _done in refs))
        self.assertTrue(all(limiter() is None for _loop, limiter, _peak, _done in refs))

    def test_cancelled_waiter_does_not_leak_loop_or_reduce_next_run_capacity(self):
        refs = [asyncio.run(self._contended_batch(cancel_waiter=True)) for _ in range(4)]
        refs.append(asyncio.run(self._contended_batch()))
        self.assertEqual([row[2:] for row in refs], [(3, 8)] * 4 + [(3, 9)])
        gc.collect()
        self.assertTrue(all(loop() is None for loop, _limiter, _peak, _done in refs))

    def test_live_callers_on_same_loop_share_one_limiter(self):
        async def exercise():
            first = downloads._download_limiter()
            self.assertIs(first, downloads._download_limiter())
            async with first:
                await asyncio.sleep(0)
                self.assertIs(first, downloads._download_limiter())
        asyncio.run(exercise())
