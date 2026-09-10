"""NAS 待确认样本揭示的目录范围保护；只用脱敏合成数据离线验证。"""
from __future__ import annotations

import unittest

from app.modules.episode_mapping import build_directory_episode_evidence, infer_episode_mapping


class ProductionNumberingGuardTests(unittest.TestCase):
    @staticmethod
    def _evidence(end=40):
        return build_directory_episode_evidence([
            ("show", "The Degenerate-Drawing Jianghu S2", 2, number)
            for number in range(19, end + 1)
        ])["show"]

    def test_same_title_shorter_series_cannot_partially_rebase_an_overlong_directory(self):
        # NAS 来源是 S02E19–40；2022 同名条目的两季均18集。
        # 即使 E19 单独减18会落到 E01，也无法证明整个目录属于这份编号方案。
        detail = {"id": 216820, "seasons": [
            {"season_number": 1, "episode_count": 18},
            {"season_number": 2, "episode_count": 18},
        ]}
        for number in range(19, 41):
            with self.subTest(number=number):
                plan = infer_episode_mapping(
                    source_season=2, source_episode=number, detail=detail,
                    directory_evidence=self._evidence(), mode="auto",
                )
                self.assertFalse(plan.changed)
                self.assertEqual((plan.target_season, plan.target_episode), (2, number))

    def test_matching_2014_series_preserves_valid_episodes_without_cleaning_offset(self):
        detail = {"id": 83463, "seasons": [
            {"season_number": 1, "episode_count": 54},
            {"season_number": 2, "episode_count": 40},
        ]}
        for number in (19, 36, 40):
            with self.subTest(number=number):
                plan = infer_episode_mapping(
                    source_season=2, source_episode=number, detail=detail,
                    directory_evidence=self._evidence(), mode="auto",
                )
                self.assertFalse(plan.changed)

    def test_fully_supported_continuous_directory_still_maps(self):
        detail = {"seasons": [
            {"season_number": 1, "episode_count": 18},
            {"season_number": 2, "episode_count": 18},
        ]}
        for number in (19, 36):
            with self.subTest(number=number):
                plan = infer_episode_mapping(
                    source_season=2, source_episode=number, detail=detail,
                    directory_evidence=self._evidence(end=36), mode="auto",
                )
                self.assertEqual((plan.target_season, plan.target_episode), (2, number - 18))
