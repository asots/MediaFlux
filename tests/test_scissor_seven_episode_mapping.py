"""刺客伍六七发布编号的纯函数回归；不是实网 TMDB 测试。

作品 ID、四个特别篇 ID 及发布顺序来自主线提供的已核验事实：episode
组 5e6c0d87396e9700138b38d2 第一组的 order 10–13 对应发布 E11–E14。
其余标题、常规季逐集日期和 S02 均是最小合成夹具，不是 TMDB 响应快照。
日期保留正片 2018-04-25 至 06-20、特别篇 08-01/08/15/22 的间隔，
不能通过放宽通用 21 天关联窗口使本测试通过。
"""
from __future__ import annotations

import unittest
from datetime import date, timedelta

from app.modules.episode_mapping import (
    build_directory_episode_evidence,
    infer_episode_mapping,
    infer_overflow_tmdb_special_mapping,
)


_SPECIAL_IDS = (1535061, 1543201, 1549531, 1556661)
_TAIL = (11, 12, 13, 14)
_FULL_PACK = tuple(range(1, 15))


def _detail(*, tmdb_id=79141, regular_count=10):
    return {
        "id": tmdb_id,
        "name": "Scissor Seven",
        "seasons": [
            {"season_number": 0, "episode_count": 4},
            {"season_number": 1, "episode_count": regular_count},
            # 合成 S02，使不受约束的 absolute 路径确实能够误投下一季。
            {"season_number": 2, "episode_count": 10},
        ],
    }


def _source_season_detail(*, count=10, season=1):
    return {
        "season_number": season,
        "episodes": [
            {
                "season_number": season,
                "episode_number": index + 1,
                "air_date": (
                    date(2018, 4, 25)
                    + timedelta(days=index * 56 // (count - 1))
                ).isoformat(),
            }
            for index in range(count)
        ],
    }


def _special_season_detail():
    return {
        "season_number": 0,
        "episodes": [
            {
                "id": episode_id,
                "season_number": 0,
                "episode_number": index + 1,
                "name": f"合成特别篇 {index + 1}",
                "air_date": (date(2018, 8, 1) + timedelta(days=index * 7)).isoformat(),
            }
            for index, episode_id in enumerate(_SPECIAL_IDS)
        ],
    }


def _directory_evidence(episodes, *, season=1):
    return build_directory_episode_evidence(
        [("synthetic-pack", "Scissor Seven", season, episode) for episode in episodes],
        minimum_episodes=1,
    ).get("synthetic-pack")


def _overflow_kwargs(episodes=_TAIL, *, source_episode=11, source_season=1):
    return {
        "source_season": source_season,
        "source_episode": source_episode,
        "detail": _detail(),
        "source_season_detail": _source_season_detail(season=source_season),
        "special_season_detail": _special_season_detail(),
        "directory_evidence": _directory_evidence(episodes, season=source_season),
        "directory_member_count": len(episodes),
    }


class ScissorSevenOverflowSpecialMappingTests(unittest.TestCase):
    def test_complete_full_pack_and_tail_map_all_four_verified_specials(self):
        for episodes in (_FULL_PACK, _TAIL):
            for source_episode in _TAIL:
                with self.subTest(pack=episodes, source_episode=source_episode):
                    mapping = infer_overflow_tmdb_special_mapping(
                        **_overflow_kwargs(episodes, source_episode=source_episode)
                    )
                    self.assertIsNotNone(mapping)
                    self.assertEqual(
                        (mapping.source_season, mapping.source_episode),
                        (1, source_episode),
                    )
                    self.assertEqual(
                        (mapping.target_season, mapping.target_episode),
                        (0, source_episode - 10),
                    )
                    self.assertTrue(mapping.changed)
                    self.assertEqual(mapping.mode, "tmdb_special")
                    self.assertEqual(mapping.confidence, 1.0)

    def test_stable_ids_follow_current_s00_numbers_not_list_or_number_order(self):
        # 合成 TMDB 重编号并打乱返回顺序，不能把发布 11–14 硬编码成 S00E1–4。
        current_numbers = (4, 2, 1, 3)
        for source_episode, target_episode in zip(_TAIL, current_numbers):
            with self.subTest(source_episode=source_episode):
                kwargs = _overflow_kwargs(source_episode=source_episode)
                rows = kwargs["special_season_detail"]["episodes"]
                for row, number in zip(rows, current_numbers):
                    row["episode_number"] = number
                rows[:] = [rows[2], rows[0], rows[3], rows[1]]
                mapping = infer_overflow_tmdb_special_mapping(**kwargs)
                self.assertIsNotNone(mapping)
                self.assertEqual(
                    (mapping.target_season, mapping.target_episode), (0, target_episode)
                )

    def test_incomplete_extra_or_duplicate_directory_members_are_rejected(self):
        cases = {
            "single_file": ((11,), 11),
            "missing_middle": ((11, 13, 14), 11),
            "missing_last": ((11, 12, 13), 11),
            "missing_first": ((12, 13, 14), 12),
            "partial_regular_pack": (tuple(range(2, 15)), 11),
            "regular_episode_in_tail": ((10, 11, 12, 13, 14), 11),
            "extra_episode": ((11, 12, 13, 14, 15), 11),
            # 证据构造器会去重，真实成员数仍为 5；helper 必须拒绝。
            "duplicate_member": ((11, 12, 13, 14, 14), 11),
        }
        for name, (episodes, source_episode) in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(infer_overflow_tmdb_special_mapping(
                    **_overflow_kwargs(episodes, source_episode=source_episode)
                ))

    def test_missing_evidence_or_mismatched_physical_member_count_is_rejected(self):
        kwargs = _overflow_kwargs()
        kwargs["directory_evidence"] = None
        self.assertIsNone(infer_overflow_tmdb_special_mapping(**kwargs))
        for count in (0, 3, 5):
            with self.subTest(directory_member_count=count):
                kwargs = _overflow_kwargs()
                kwargs["directory_member_count"] = count
                self.assertIsNone(infer_overflow_tmdb_special_mapping(**kwargs))

    def test_wrong_identity_source_season_or_regular_count_is_rejected(self):
        for tmdb_id in (None, 79142):
            with self.subTest(tmdb_id=tmdb_id):
                kwargs = _overflow_kwargs()
                kwargs["detail"] = _detail(tmdb_id=tmdb_id)
                self.assertIsNone(infer_overflow_tmdb_special_mapping(**kwargs))
        self.assertIsNone(infer_overflow_tmdb_special_mapping(
            **_overflow_kwargs(source_season=2)
        ))
        for count in (9, 11):
            with self.subTest(regular_count=count):
                # 连续尾包与错误集数也完全自洽，必须由已核验的 count=10 拒绝。
                episodes = tuple(range(count + 1, count + 5))
                kwargs = _overflow_kwargs(episodes, source_episode=max(11, count + 1))
                kwargs["detail"] = _detail(regular_count=count)
                kwargs["source_season_detail"] = _source_season_detail(count=count)
                self.assertIsNone(infer_overflow_tmdb_special_mapping(**kwargs))

    def test_missing_or_duplicated_stable_episode_id_is_rejected(self):
        for case in ("missing_id", "duplicate_id", "identical_duplicate_row"):
            with self.subTest(case=case):
                kwargs = _overflow_kwargs()
                rows = kwargs["special_season_detail"]["episodes"]
                if case == "missing_id":
                    rows[1]["id"] = 99999999
                elif case == "duplicate_id":
                    rows.append({**rows[0], "episode_number": 5})
                else:
                    rows.append(dict(rows[0]))
                self.assertIsNone(infer_overflow_tmdb_special_mapping(**kwargs))

    def test_non_special_season_container_or_episode_is_rejected(self):
        for case in ("container", "episode"):
            with self.subTest(case=case):
                kwargs = _overflow_kwargs()
                special = kwargs["special_season_detail"]
                if case == "container":
                    special["season_number"] = 1
                else:
                    special["episodes"][0]["season_number"] = 1
                self.assertIsNone(infer_overflow_tmdb_special_mapping(**kwargs))

    def test_other_titles_do_not_gain_a_wider_generic_special_date_window(self):
        kwargs = _overflow_kwargs()
        kwargs["detail"]["id"] = 79142
        for index, row in enumerate(kwargs["special_season_detail"]["episodes"], 1):
            # 满足旧通用规则的小数标题，但日期距正片结束仍为 42–63 天。
            row["name"] = f"第10.{index}话 合成回顾"
        self.assertIsNone(infer_overflow_tmdb_special_mapping(**kwargs))


class ScissorSevenAutomaticMappingGuardTests(unittest.TestCase):
    def assert_verified_mapping_required(self, source_episode, evidence):
        mapping = infer_episode_mapping(
            source_season=1,
            source_episode=source_episode,
            detail=_detail(),
            mode="auto",
            directory_evidence=evidence,
        )
        self.assertEqual(
            (mapping.target_season, mapping.target_episode), (1, source_episode)
        )
        self.assertFalse(mapping.changed)
        self.assertEqual(mapping.reason, "verified_special_mapping_required")
        self.assertEqual(mapping.confidence, 0.0)

    def test_auto_full_pack_and_tail_do_not_roll_verified_specials_into_s02(self):
        for episodes in (_FULL_PACK, _TAIL):
            for source_episode in _TAIL:
                with self.subTest(pack=episodes, source_episode=source_episode):
                    self.assert_verified_mapping_required(
                        source_episode, _directory_evidence(episodes)
                    )

    def test_auto_without_directory_evidence_still_requires_verified_special_mapping(self):
        for source_episode in _TAIL:
            with self.subTest(source_episode=source_episode):
                self.assert_verified_mapping_required(source_episode, None)

    def test_standard_and_user_requested_absolute_keep_existing_semantics(self):
        for mode in ("standard", "absolute"):
            for evidence in (None, _directory_evidence(_TAIL)):
                for source_episode in _TAIL:
                    with self.subTest(mode=mode, evidence=evidence, episode=source_episode):
                        mapping = infer_episode_mapping(
                            source_season=1,
                            source_episode=source_episode,
                            detail=_detail(),
                            mode=mode,
                            directory_evidence=evidence,
                        )
                        expected = (1, source_episode) if mode == "standard" else (2, source_episode - 10)
                        self.assertEqual(
                            (mapping.target_season, mapping.target_episode), expected
                        )
                        self.assertEqual(mapping.mode, mode)
                        self.assertEqual(
                            mapping.reason,
                            "identity" if mode == "standard"
                            else "absolute_numbering_rolled_over_tmdb_seasons",
                        )

    def test_auto_guard_does_not_expand_to_other_titles_counts_or_episodes(self):
        cases = (
            (79142, 10, 11, (2, 1)),
            (79141, 9, 11, (2, 2)),
            (79141, 10, 10, (1, 10)),
            (79141, 10, 15, (2, 5)),
        )
        for tmdb_id, count, episode, expected in cases:
            with self.subTest(tmdb_id=tmdb_id, count=count, episode=episode):
                mapping = infer_episode_mapping(
                    source_season=1,
                    source_episode=episode,
                    detail=_detail(tmdb_id=tmdb_id, regular_count=count),
                    mode="auto",
                    directory_evidence=_directory_evidence(tuple(range(1, 21))),
                )
                self.assertEqual(
                    (mapping.target_season, mapping.target_episode), expected
                )
                self.assertNotEqual(mapping.reason, "verified_special_mapping_required")
