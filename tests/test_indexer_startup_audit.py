"""独立索引运行时启动审计：用真实线程/loop 固定启动竞争窗口。"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
import unittest
from unittest.mock import patch

import tests  # noqa: F401 - 任何应用导入前先隔离运行目录。
from app.indexers import runtime


class IndexerStartupAuditTests(unittest.TestCase):
    def setUp(self):
        runtime.unbind_indexer_event_loop()
        self.assertIsNone(runtime._standalone_thread)
        self.real_new_loop = asyncio.new_event_loop
        self.loops = []
        self.workers = []
        self.callers = []
        self.gates = []
        self.awaitables = []
        self.timeout_patch = patch.object(
            runtime,
            "_STANDALONE_STARTUP_TIMEOUT_SECONDS",
            1.0,
            create=True,
        )
        self.timeout_patch.start()

    def tearDown(self):
        # RED 版本可能根本没有登记线程；清理不能只依赖被测全局句柄。
        for gate in self.gates:
            gate.set()
        for caller in self.callers:
            caller.join(timeout=7.0)
        deadline = time.monotonic() + 3.0
        while True:
            runtime._stop_standalone_runtime(timeout_seconds=0.01)
            # 创建屏障释放后 loop 可能稍晚才出现，反复接管 RED 遗留对象。
            for loop in list(self.loops):
                if not loop.is_closed():
                    try:
                        loop.call_soon_threadsafe(loop.stop)
                    except RuntimeError:
                        pass
            for worker in self.workers:
                worker.join(timeout=0.01)
            if not any(worker.is_alive() for worker in self.workers):
                break
            if time.monotonic() >= deadline:
                break
        runtime._stop_standalone_runtime(timeout_seconds=0.1)
        for loop in self.loops:
            if not loop.is_closed() and not loop.is_running():
                loop.close()
        runtime.unbind_indexer_event_loop()
        for awaitable in self.awaitables:
            awaitable.close()
        self.timeout_patch.stop()
        self.assertFalse(any(worker.is_alive() for worker in self.workers))
        self.assertTrue(all(loop.is_closed() for loop in self.loops))
        self.assertFalse(any(caller.is_alive() for caller in self.callers))

    def new_awaitable(self):
        async def capture_loop():
            return asyncio.get_running_loop()

        awaitable = capture_loop()
        self.awaitables.append(awaitable)
        return awaitable

    def new_loop(self):
        worker = threading.current_thread()
        if worker not in self.workers:
            self.workers.append(worker)
        loop = self.real_new_loop()
        self.loops.append(loop)
        return loop

    def new_gate(self):
        gate = threading.Event()
        self.gates.append(gate)
        return gate

    def gated_run_factory(self, entered, gate):
        def factory():
            loop = self.new_loop()
            original_run_forever = loop.run_forever

            def first_run_forever():
                loop.run_forever = original_run_forever
                entered.set()
                if not gate.wait(timeout=10.0):
                    raise RuntimeError("test startup barrier timed out")
                original_run_forever()

            loop.run_forever = first_run_forever
            return loop

        return factory

    def start_caller(self):
        done = threading.Event()
        outcome = {}
        awaitable = self.new_awaitable()

        def invoke():
            try:
                outcome["loop"] = runtime.run_indexer_awaitable_sync(
                    awaitable,
                    timeout_seconds=1.0,
                )
            except Exception as exc:
                outcome["error"] = exc
            finally:
                done.set()

        caller = threading.Thread(target=invoke, name="indexer-startup-test-caller")
        self.callers.append(caller)
        caller.start()
        return done, outcome

    def test_concurrent_cold_calls_wait_until_loop_really_runs_and_share_one_runtime(
        self,
    ):
        entered = threading.Event()
        gate = self.new_gate()
        with patch.object(
            runtime.asyncio,
            "new_event_loop",
            self.gated_run_factory(entered, gate),
        ):
            first_done, first = self.start_caller()
            self.assertTrue(entered.wait(timeout=1.0))
            second_done, second = self.start_caller()
            self.assertFalse(
                first_done.wait(timeout=0.05), "loop 未运行时不可报告启动失败或成功"
            )
            self.assertFalse(second_done.is_set())
            self.assertEqual(len(self.loops), 1)
            gate.set()
            self.assertTrue(first_done.wait(timeout=2.0))
            self.assertTrue(second_done.wait(timeout=2.0))
            self.assertNotIn("error", first)
            self.assertNotIn("error", second)
            self.assertIs(first["loop"], second["loop"])
            self.assertIs(first["loop"], runtime._standalone_loop)
            self.assertEqual(len(self.loops), 1)

    def test_loop_creation_error_is_reported_promptly_and_retry_succeeds(self):
        error = OSError("isolated loop creation failure")

        def fail_create():
            self.workers.append(threading.current_thread())
            raise error

        awaitable = self.new_awaitable()
        started = time.monotonic()
        with (
            patch.object(runtime.asyncio, "new_event_loop", fail_create),
            patch(
                "threading.excepthook",
            ) as uncaught,
        ):
            with self.assertRaisesRegex(RuntimeError, "启动失败") as caught:
                runtime.run_indexer_awaitable_sync(awaitable, timeout_seconds=1.0)
            self.assertIs(caught.exception.__cause__, error)
            self.assertLess(time.monotonic() - started, 1.0)
            uncaught.assert_not_called()
        self.assertEqual(inspect.getcoroutinestate(awaitable), inspect.CORO_CLOSED)
        self.assertIsNone(runtime._standalone_thread)
        self.assertFalse(any(worker.is_alive() for worker in self.workers))
        with patch.object(runtime.asyncio, "new_event_loop", self.new_loop):
            observed = runtime.run_indexer_awaitable_sync(
                self.new_awaitable(), timeout_seconds=1.0
            )
        self.assertIs(observed, runtime._standalone_loop)

    def test_initialization_error_closes_created_loop_and_unsubmitted_coroutine(self):
        error = RuntimeError("isolated set_event_loop failure")
        real_set_loop = asyncio.set_event_loop

        def fail_set_loop(loop):
            if loop is not None:
                raise error
            real_set_loop(None)

        awaitable = self.new_awaitable()
        with (
            patch.object(runtime.asyncio, "new_event_loop", self.new_loop),
            patch.object(
                runtime.asyncio,
                "set_event_loop",
                fail_set_loop,
            ),
            patch("threading.excepthook") as uncaught,
        ):
            with self.assertRaisesRegex(RuntimeError, "启动失败") as caught:
                runtime.run_indexer_awaitable_sync(awaitable, timeout_seconds=1.0)
            self.assertIs(caught.exception.__cause__, error)
            uncaught.assert_not_called()
        self.assertEqual(len(self.loops), 1)
        self.assertTrue(self.loops[0].is_closed())
        self.assertEqual(inspect.getcoroutinestate(awaitable), inspect.CORO_CLOSED)
        self.assertIsNone(runtime._standalone_thread)

    def test_timeout_during_loop_creation_retains_thread_until_it_exits_and_blocks_retry(
        self,
    ):
        entered = threading.Event()
        gate = self.new_gate()
        ran = threading.Event()

        def blocked_create():
            self.workers.append(threading.current_thread())
            entered.set()
            if not gate.wait(timeout=10.0):
                raise RuntimeError("test creation barrier timed out")
            loop = self.new_loop()
            original_run_forever = loop.run_forever

            def observe_run_forever():
                ran.set()
                original_run_forever()

            loop.run_forever = observe_run_forever
            return loop

        awaitable = self.new_awaitable()
        with (
            patch.object(runtime, "_STANDALONE_STARTUP_TIMEOUT_SECONDS", 0.05),
            patch.object(
                runtime.asyncio,
                "new_event_loop",
                blocked_create,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "启动超时"):
                runtime.run_indexer_awaitable_sync(awaitable, timeout_seconds=1.0)
            self.assertTrue(entered.is_set())
            worker = self.workers[0]
            self.assertIs(runtime._standalone_thread, worker)
            self.assertEqual(inspect.getcoroutinestate(awaitable), inspect.CORO_CLOSED)
            with self.assertRaises(RuntimeError):
                runtime.run_indexer_awaitable_sync(
                    self.new_awaitable(), timeout_seconds=1.0
                )
            self.assertEqual(len(self.workers), 1, "失败线程未退出前不得启动第二个线程")
            self.assertFalse(runtime._stop_standalone_runtime(timeout_seconds=0.01))
            gate.set()
            worker.join(timeout=1.0)
            self.assertFalse(worker.is_alive())
            self.assertTrue(self.loops[0].is_closed())
            self.assertFalse(ran.is_set(), "启动已取消时迟到的 loop 不应再运行")
        with patch.object(runtime.asyncio, "new_event_loop", self.new_loop):
            first = runtime.run_indexer_awaitable_sync(
                self.new_awaitable(), timeout_seconds=1.0
            )
            second = runtime.run_indexer_awaitable_sync(
                self.new_awaitable(), timeout_seconds=1.0
            )
        self.assertIs(first, second)
        self.assertEqual(len(self.loops), 2)

    def test_timeout_before_run_forever_stops_late_loop_without_publishing_runtime(
        self,
    ):
        entered = threading.Event()
        gate = self.new_gate()
        awaitable = self.new_awaitable()
        with (
            patch.object(runtime, "_STANDALONE_STARTUP_TIMEOUT_SECONDS", 0.05),
            patch.object(
                runtime.asyncio,
                "new_event_loop",
                self.gated_run_factory(entered, gate),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "启动超时"):
                runtime.run_indexer_awaitable_sync(awaitable, timeout_seconds=1.0)
            self.assertTrue(entered.is_set())
            self.assertIsNone(runtime._runtime_loop)
            self.assertIs(runtime._standalone_thread, self.workers[0])
            self.assertEqual(inspect.getcoroutinestate(awaitable), inspect.CORO_CLOSED)
            with self.assertRaises(RuntimeError):
                runtime.run_indexer_awaitable_sync(
                    self.new_awaitable(), timeout_seconds=1.0
                )
            self.assertEqual(len(self.loops), 1)
            gate.set()
            self.workers[0].join(timeout=1.0)
            self.assertTrue(self.loops[0].is_closed())
            self.assertTrue(runtime._stop_standalone_runtime(timeout_seconds=0.1))
            self.assertIsNone(runtime._standalone_thread)

    def test_async_bridge_closes_unsubmitted_coroutine_on_startup_failure(self):
        awaitable = self.new_awaitable()
        with patch.object(
            runtime, "_ensure_standalone_runtime", side_effect=RuntimeError("启动失败")
        ):
            with self.assertRaisesRegex(RuntimeError, "启动失败"):
                asyncio.run(
                    runtime.run_indexer_awaitable(awaitable, timeout_seconds=1.0)
                )
        self.assertEqual(inspect.getcoroutinestate(awaitable), inspect.CORO_CLOSED)

    def test_thread_start_failure_clears_handle_and_closes_unsubmitted_coroutine(self):
        awaitable = self.new_awaitable()
        with patch(
            "threading.Thread.start",
            side_effect=RuntimeError("isolated thread start failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "thread start failure"):
                runtime.run_indexer_awaitable_sync(awaitable, timeout_seconds=1.0)
        self.assertIsNone(runtime._standalone_thread)
        self.assertEqual(inspect.getcoroutinestate(awaitable), inspect.CORO_CLOSED)
