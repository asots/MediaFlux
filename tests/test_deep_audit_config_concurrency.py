"""深审：普通保存和Agent CAS策略保存必须遵循同一锁顺序。"""
from __future__ import annotations

import multiprocessing
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock

from app import config
from app.modules import backup


def simultaneous_configuration_writers(directory, output):
    """用独立进程隔离旧实现的真实死锁，失败不能污染后续pytest全局锁。"""
    # 排除首次导入模块的锁，明确检验应用自己的RLock/文件锁顺序。
    from app.modules.process_lock import CrossProcessLock

    warmup = CrossProcessLock("config-snapshot", directory=Path(directory))
    warmup.acquire()
    warmup.release()
    target = Path(directory) / "user.env"
    simple_ready, policy_locked = threading.Event(), threading.Event()
    outcomes = {}
    original_update = config.update_runtime_env_file
    original_guard = backup.config_snapshot_guard

    def delayed_simple_update(*args, **kwargs):
        if threading.current_thread().name == "simple-config":
            simple_ready.set()
            if not policy_locked.wait(5):
                raise RuntimeError("策略写入未进入文件锁")
        return original_update(*args, **kwargs)

    @contextmanager
    def observed_guard(paths):
        with original_guard(paths):
            if threading.current_thread().name == "policy-config":
                policy_locked.set()
            yield

    with mock.patch.object(config, "ENV_FILE", target), \
            mock.patch.object(config, "PATHS", replace(config.PATHS, data_dir=Path(directory), config_dir=Path(directory))), \
            mock.patch.object(config, "_cache", None), \
            mock.patch.object(config, "update_runtime_env_file", side_effect=delayed_simple_update), \
            mock.patch.object(backup, "config_snapshot_guard", side_effect=observed_guard), \
            mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SIMPLE_KEY", None)
        os.environ.pop("POLICY_KEY", None)
        config.write_env_file(target, {"HISTORICAL_KEY": "preserved"}, replace=False)
        expected, _ = config.read_env_snapshot(target)

        def simple_writer():
            try:
                config.set_and_save({"SIMPLE_KEY": "simple"})
                outcomes["simple"] = "saved"
            except Exception as exc:
                outcomes["simple"] = type(exc).__name__

        def policy_writer():
            try:
                original_update(target, {"POLICY_KEY": "policy"}, expected=expected)
                outcomes["policy"] = "saved"
            except Exception as exc:
                outcomes["policy"] = type(exc).__name__

        simple = threading.Thread(target=simple_writer, name="simple-config", daemon=True)
        policy = threading.Thread(target=policy_writer, name="policy-config", daemon=True)
        simple.start()
        if not simple_ready.wait(5):
            raise RuntimeError("普通写入未取得快照")
        policy.start()
        simple.join(2)
        policy.join(2)
        alive = [thread.name for thread in (simple, policy) if thread.is_alive()]
        output.put({
            "alive": alive, "outcomes": outcomes,
            "file": target.read_text(),
            "policy_runtime": os.environ.get("POLICY_KEY"),
            "simple_runtime": os.environ.get("SIMPLE_KEY"),
            "cache": dict(config._cache or {}),
        })


class ConfigLockOrderBusinessTests(unittest.TestCase):
    def test_concurrent_plain_and_policy_writes_finish_with_safe_cas_conflict(self):
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            process = context.Process(target=simultaneous_configuration_writers, args=(directory, output))
            process.start()
            try:
                result = output.get(timeout=12)
                process.join(5)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)
                self.assertEqual(result["alive"], [], f"配置写入未完成：{result['outcomes']}")
                self.assertEqual(result["outcomes"], {"simple": "ConcurrentConfigUpdateError", "policy": "saved"})
                self.assertEqual(result["policy_runtime"], "policy")
                self.assertIsNone(result["simple_runtime"])
                self.assertEqual(result["cache"], {"HISTORICAL_KEY": "preserved", "POLICY_KEY": "policy"})
                self.assertIn("HISTORICAL_KEY", result["file"])
                self.assertIn("POLICY_KEY", result["file"])
                self.assertNotIn("SIMPLE_KEY", result["file"])
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                output.close()
                output.join_thread()
