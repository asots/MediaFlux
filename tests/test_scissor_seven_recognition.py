"""Scissor Seven 的 NF/Mys 发布名回归；仅运行离线清洗与季集解析。"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests.support import isolated_test_database


class ScissorSevenRecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # tests 包先隔离配置和日志；本类另建 DB，并禁止任何真实网络请求。
        for target in (
            "socket.socket.connect",
            "socket.socket.connect_ex",
            "socket.create_connection",
            "requests.sessions.Session.request",
        ):
            cls.enterClassContext(
                patch(target, side_effect=AssertionError("识别回归禁止联网"))
            )
        cls.enterClassContext(isolated_test_database("scissor-seven.db"))
        from app.modules import scraper

        cls.scraper = scraper
        cls.parser = scraper.TMDBScraper(
            client=SimpleNamespace(api_key="", base_url="")
        )

    def assert_release(self, episode: int, episode_title_number: int):
        filename = (
            f"Scissor.Seven.S01E{episode:02d}.Episode.{episode_title_number}."
            "1080p.NF.WEB-DL.DDP2.0.H.264-Mys.mkv"
        )
        expected_position = (1, episode)
        context = self.scraper.extract_recognition_context(filename)
        parsed = self.parser.parse_media(filename)

        self.assertEqual(self.parser.clean_title(filename), "Scissor Seven")
        self.assertEqual(context.normalized_title, "Scissor Seven")
        self.assertEqual(context.filename_title, "Scissor Seven")
        # 约束主搜索词，不把当前保留的 Episode N 附加候选固化成清洗规则。
        self.assertEqual(
            self.scraper.generate_query_variants(context)[0], "Scissor Seven"
        )
        self.assertEqual((context.season, context.episode), expected_position)
        self.assertEqual(
            self.scraper.parse_release_position(filename),
            {"season": 1, "episode": episode, "episode_end": None},
        )
        self.assertEqual(parsed.title, "Scissor Seven")
        self.assertEqual(parsed.media_type, "tv")
        self.assertEqual(
            (parsed.source_season, parsed.source_episode), expected_position
        )
        # 不提供 MatchResult；只验证隔离默认规则下的本地 parser 输出。
        self.assertEqual(
            (parsed.effective_season, parsed.effective_episode), expected_position
        )
        self.assertEqual(parsed.preprocess_rules, ())

    def test_nf_mys_episode_11_to_14_keep_title_and_explicit_position(self):
        for episode in range(11, 15):
            with self.subTest(episode=episode):
                self.assert_release(episode, episode)

    def test_episode_title_number_does_not_override_explicit_episode(self):
        for episode, episode_title_number in ((11, 1), (1, 11)):
            with self.subTest(
                episode=episode, episode_title_number=episode_title_number
            ):
                self.assert_release(episode, episode_title_number)
