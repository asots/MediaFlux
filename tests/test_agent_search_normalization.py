"""目录工具与工作区工具共享一套标题规范化，避免入口分叉。"""

from __future__ import annotations

import unittest

from app.agent.domain_catalog import shared
from app.agent.errors import AgentToolError
from app.agent import workspace_actions


class AgentSearchNormalizationTests(unittest.TestCase):
    def test_catalog_and_workspace_use_the_same_normalizer(self):
        self.assertIs(
            shared._normalize_search_query, workspace_actions._normalize_query
        )

    def test_both_entrypoints_apply_identical_title_contract(self):
        for raw, expected in (
            ("  沙丘２  ", "沙丘2"),
            ("Ｄｕｎｅ", "Dune"),
            ("a" * 120, "a" * 120),
        ):
            with self.subTest(raw=raw):
                arguments = {"query": raw}
                self.assertEqual(shared._search_arguments(arguments)["query"], expected)
                self.assertEqual(
                    workspace_actions.workspace_search_arguments(arguments)["query"],
                    expected,
                )
        for raw in (
            " ",
            "a" * 121,
            "Dune\u200b",
            "https://example.com/movie",
            "/media/Movie.mkv",
        ):
            for validator in (
                shared._search_arguments,
                workspace_actions.workspace_search_arguments,
            ):
                with (
                    self.subTest(raw=raw, validator=validator.__name__),
                    self.assertRaises(AgentToolError),
                ):
                    validator({"query": raw})
