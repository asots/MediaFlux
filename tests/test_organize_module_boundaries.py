"""整理拆分后的类型同一性、规则入口与共享运行态契约。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import inspect
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from app.modules import organize, organize_identity, organize_models, organize_results, organize_rules
from app.modules.organize_runtime import OrganizeTaskRuntime
from app.modules.scraper import MatchResult


class OrganizeModuleBoundaryTests(unittest.TestCase):
    def test_models_rules_and_identity_are_direct_exports_not_parallel_implementations(self):
        for module in (organize_models, organize_rules, organize_identity):
            for name, value in inspect.getmembers(module):
                if not (inspect.isfunction(value) or inspect.isclass(value)):
                    continue
                if value.__module__ != module.__name__:
                    continue
                with self.subTest(module=module.__name__, name=name):
                    self.assertIs(getattr(organize, name), value)
        for name in ("_format_scan_summary", "_format_phase_timing"):
            self.assertIs(getattr(organize, name), getattr(organize_results, name))
        for name in ("DEFAULT_ORGANIZE_VIDEO_EXTS", "DEFAULT_ORGANIZE_METADATA_EXTS"):
            self.assertIs(getattr(organize, name), getattr(organize_rules, name))

    def test_leaf_modules_can_load_before_the_coordinator(self):
        orders = (
            ("organize_models", "organize_rules", "organize_identity"),
            ("organize_identity", "organize_rules", "organize_models"),
        )
        for order in orders:
            with self.subTest(order=order):
                code = f'''
from importlib import import_module
import sys
from unittest.mock import patch
with patch("sqlite3.connect", side_effect=AssertionError("import opened database")):
    for name in {order!r}:
        import_module("app.modules." + name)
    assert "app.modules.organize" not in sys.modules
    from app.modules import organize, organize_models, organize_rules
assert organize.OrganizeContext is organize_models.OrganizeContext
assert organize.OrganizeRules is organize_rules.OrganizeRules
'''
                completed = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "MEDIAFLUX_DISABLE_FILE_LOGGING": "1"},
                    capture_output=True, text=True, timeout=30, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_context_replacement_preserves_shared_runtime_lock_executor_and_cancellation(self):
        runtime = OrganizeTaskRuntime()
        event = threading.Event()
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=1) as executor:
            context = organize_models.OrganizeContext(
                source_dir_id="source-a", cancel_event=event, execution_lock=lock,
                planning_executor=executor, task_runtime=runtime,
            )
            other = replace(context, source_dir_id="source-b", post_actions=False)
            self.assertIs(other.task_runtime, runtime)
            self.assertIs(other.execution_lock, lock)
            self.assertIs(other.planning_executor, executor)
            self.assertTrue(other.probe_cache_only)
            self.assertFalse(replace(other, dry_run=False).probe_cache_only)
            self.assertFalse(other.cancelled())
            event.set()
            self.assertTrue(context.cancelled())
            self.assertTrue(other.cancelled())
            self.assertEqual(executor.submit(lambda: "still-owned-by-caller").result(), "still-owned-by-caller")

    def test_plan_defaults_and_positional_fields_remain_shared_across_entrypoints(self):
        plan = organize.OrganizePlan("video-1", "Movie.mkv", "/source/Movie.mkv", "source", 42)
        other = organize_models.OrganizePlan("video-2", "Other.mkv", "/source/Other.mkv")
        self.assertIsInstance(plan, organize_models.OrganizePlan)
        self.assertEqual((plan.original_parent_id, plan.size), ("source", 42))
        self.assertEqual(plan.variant, other.variant)
        self.assertIsNot(plan.variant, other.variant)
        self.assertEqual(other.action, "move")
        self.assertEqual(other.source_group_id, "")

    def test_config_rules_and_snapshots_use_one_policy_without_persisting_server_secret(self):
        with patch.object(organize_rules, "get", side_effect=lambda key, default="": default), patch.object(
            organize_rules, "get_bool", side_effect=lambda key, default=False: default,
        ), patch.object(organize_rules, "get_int", side_effect=lambda key, default=0: default):
            rules = organize.OrganizeRules.from_config("target-dir")
        self.assertEqual(rules.target_dir_id, "target-dir")
        trusted = replace(rules, nsfw_metatube_token="fixture-current-secret")
        snapshot = organize.organize_rules_snapshot(trusted)
        self.assertNotIn("nsfw_metatube_token", snapshot)
        restored = organize_rules.restore_organize_rules_snapshot(
            {**snapshot, "nsfw_metatube_token": "fixture-stale-secret", "rename_enabled": False},
            trusted_rules=trusted,
        )
        self.assertEqual(restored.nsfw_metatube_token, "fixture-current-secret")
        self.assertTrue(restored.rename_enabled)
        self.assertTrue(organize.organize_rules_snapshot_matches(snapshot, restored))
        self.assertEqual(organize.organize_rules_snapshot(restored), snapshot)

    def test_automatic_confirmation_and_result_formatting_keep_their_contract(self):
        matched = MatchResult(status="matched", confidence=0.95, need_confirm=False)
        self.assertFalse(organize_rules.automatic_match_requires_confirmation(matched))
        self.assertTrue(organize.automatic_match_requires_confirmation(replace(matched, need_confirm=True)))
        self.assertTrue(organize.automatic_match_requires_confirmation(None))
        stats = {
            "total": 2, "matched": 1, "failed": 1,
            "scan_complete": False, "scan_limit_kind": "depth",
            "source_groups": [{"name": "Example"}],
            "scan_elapsed_seconds": 1.25, "total_elapsed_seconds": 2.5,
            "scan_list_dir_calls": 3,
        }
        self.assertEqual(
            organize._format_scan_summary(stats),
            "[Example] 共 2 个视频 · 已识别 1 · 失败 1 · 扫描不完整(depth) | 扫描=1.25s",
        )
        self.assertEqual(
            organize._format_phase_timing(stats),
            "scan=1.25s total=2.50s list_dir=3",
        )
