"""候选版本的升级、回退与 compare 声明须对应实际源码合同。"""

from __future__ import annotations

import re
from pathlib import Path
import unittest

from app import __version__
from app.database import SCHEMA_VERSION
from tests.test_release_metadata import B


class ReleaseUpgradeNotesTests(unittest.TestCase):
    def setUp(self):
        self.changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
        self.section = B._changelog_section(self.changelog, __version__)
        self.assertTrue(self.section, "当前源码版本缺少有效的候选发布说明")

    def upgrade_span(self):
        span = re.search(r"\bschema\s*(\d+)\s*→\s*(\d+)", self.section)
        self.assertIsNotNone(span, "升级说明必须声明完整 schema 跨度")
        return int(span.group(1)), int(span.group(2))

    def test_candidate_upgrade_target_matches_supported_schema(self):
        previous, target = self.upgrade_span()
        self.assertEqual(target, SCHEMA_VERSION)
        self.assertLessEqual(previous, target)

    def test_rollback_warning_names_the_actual_new_schema(self):
        previous, _target = self.upgrade_span()
        self.assertRegex(
            self.section,
            rf"schema{previous}\s*程序[^\n]*schema{SCHEMA_VERSION}\s*数据库",
        )
        self.assertIn("离线恢复", self.section)
        self.assertIn("停止服务", self.section)

    def test_compare_links_start_from_previous_formal_version(self):
        versions = re.findall(
            r"^## \[([^\]]+)\] - \d{4}-\d{2}-\d{2}$", self.changelog, re.MULTILINE
        )
        position = versions.index(__version__)
        previous = versions[position + 1]
        links = dict(
            re.findall(r"^\[([^\]]+)\]:\s*(\S+)$", self.changelog, re.MULTILINE)
        )
        self.assertTrue(links["Unreleased"].endswith(f"/compare/v{__version__}...HEAD"))
        self.assertTrue(
            links[__version__].endswith(f"/compare/v{previous}...v{__version__}")
        )
