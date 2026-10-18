"""日志 8885：下载隔离目录不能参与作品身份或绕过未消歧的人工确认。"""
from __future__ import annotations

from app.modules.scraper import RecognitionResult

from typing import cast
from app.clients.tmdb import TMDBClient

import unittest
from contextlib import ExitStack
from copy import deepcopy
from itertools import product
from unittest.mock import patch

from app.modules import scraper
from tests.support import isolated_test_database
from tests.test_recognition_pipeline import _DeterministicClient

FILENAME = "[ANi] 魔法騎士雷阿斯 - 02 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4"
RELEASE = (
    "[ANi] Majikku Naito Reiāsu _ 魔法骑士雷阿斯 - 02 "
    "[1080P][Baha][WEB-DL][AAC AVC][CHT][MP4]"
)
PARENT = "MF-368-" + RELEASE
# 真实候选仅保留身份字段与本例需要的中英文、繁简别名；不读取内部快照。
CANDIDATES = tuple({
    "id": identity, "name": name, "original_name": "魔法騎士レイアース",
    "first_air_date": aired, "media_type": "tv",
    "aliases": ["Majikku Naito Reiāsu", "Magic Knight Rayearth", "魔法骑士雷阿斯", "魔法騎士雷阿斯"],
} for identity, name, aired in (
    (34158, "魔法骑士", "1994-10-17"),
    (303272, "魔法骑士雷阿斯", "2026-10-07"),
))


class _OfflineClient(_DeterministicClient):
    api_key = "offline-fixture"

    def get(self, path, params=None):
        assert path.startswith("/tv/") and "/season/" in path, path
        return {}  # 没有逐集证据时不得虚构合并季或跨季映射。

    def tv_season_detail(self, tmdb_id, season):
        return self.get(f"/tv/{tmdb_id}/season/{season}")


def _details(candidates, seasons=(1,)):
    # 合成位置容量只服务于门禁测试，不声称两版具有相同真实季数。
    return {str(item["id"]): {
        **item, "seasons": [{"season_number": season, "episode_count": 12} for season in seasons],
    } for item in candidates}


class RecognitionStagingIdentityTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(isolated_test_database("recognition-staging-identity.db"))
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("离线测试禁止联网")))

    def _recognize(self, filename=FILENAME, parent=PARENT, candidates=CANDIDATES, *, details=None, full=False):
        available = deepcopy(candidates)
        client = _OfflineClient(available, deepcopy(details) if details is not None else _details(available))
        parser = scraper.TMDBScraper(client=cast(TMDBClient, client))
        parser.match_mode = "strict"
        self.addCleanup(parser.close)
        if not full:
            return parser.deterministic_recognize(filename, parent)
        # 保留实际 deterministic、预处理与 DB 链路，仅封死辅助源。
        with ExitStack() as stack:
            fallbacks = [stack.enter_context(patch.object(
                parser, name, side_effect=AssertionError(f"不应调用 {name}"),
            )) for name in ("_external_hint_fallback", "_release_group_fallback", "_ai_fallback")]
            result = parser.match(filename, parent)
            for fallback in fallbacks:
                fallback.assert_not_called()
        return result

    def _assert_manual(self, result, reason="ambiguous_exact_title"):
        self.assertEqual(result.status, "low_confidence")
        self.assertTrue(result.need_confirm)
        self.assertFalse(result.locked)
        self.assertEqual(result.threshold, 0.9)
        self.assertFalse(result.threshold_decision["passed"])
        self.assertEqual(result.threshold_decision["reason"], reason)
        self.assertIn(reason, result.rejected_constraints)
        self.assertIn("人工确认", result.error)
        self.assertEqual({item.tmdb_id: item.score for item in result.candidates}, {"34158": 1.0, "303272": 1.0})
        for candidate in result.candidates:
            self.assertEqual(candidate.score_breakdown.rejected_constraints, [])

    def _assert_matched(self, result, identity):
        self.assertEqual(result.status, "matched")
        self.assertFalse(result.need_confirm)
        self.assertEqual(result.tmdb_id, str(identity))
        self.assertEqual(result.threshold, 0.9)
        self.assertGreaterEqual(result.confidence, result.threshold)

    def test_yearless_versions_score_equally_for_scripts_prefixes_and_candidate_order(self):
        for simplified, prefix, reverse in product((False, True), ("MF-368-", "MF-download-"), (False, True)):
            with self.subTest(simplified=simplified, prefix=prefix, reverse=reverse):
                filename = FILENAME.replace("魔法騎士", "魔法骑士") if simplified else FILENAME
                parent = prefix + RELEASE.replace("魔法骑士", "魔法騎士") if not simplified else prefix + RELEASE
                result = self._recognize(filename, parent, CANDIDATES[::-1] if reverse else CANDIDATES)
                self._assert_manual(result)

    def test_staging_context_equals_unwrapped_context_without_changing_raw_paths(self):
        for prefix in ("MF-368-", "MF-download-", "MF-2026-"):
            with self.subTest(prefix=prefix):
                parent = "/downloads/" + prefix + RELEASE
                actual = scraper.extract_recognition_context(FILENAME, parent)
                plain = scraper.extract_recognition_context(FILENAME, "/downloads/" + RELEASE)
                self.assertEqual((actual.filename, actual.parent_path), (FILENAME, parent))
                for field in ("normalized_title", "filename_title", "folder_title", "filename_year", "folder_year", "season", "episode"):
                    self.assertEqual(getattr(actual, field), getattr(plain, field), field)
                self.assertEqual(scraper.generate_query_variants(actual), scraper.generate_query_variants(plain))
                self.assertEqual((actual.filename_year, actual.folder_year), ("", ""))
                self._assert_manual(self._recognize(parent=parent))

    def test_explicit_filename_or_folder_year_selects_the_corresponding_version(self):
        for year, source, reverse in product(("1994", "2026"), ("filename", "folder"), (False, True)):
            with self.subTest(year=year, source=source, reverse=reverse):
                filename, parent = FILENAME, "MF-2026-" + RELEASE
                if source == "filename":
                    filename = filename.replace(" - 02", f" ({year}) - 02")
                else:
                    parent = parent.replace(" - 02", f" ({year}) - 02")
                result = self._recognize(filename, parent, CANDIDATES[::-1] if reverse else CANDIDATES, full=True)
                self._assert_matched(result, 34158 if year == "1994" else 303272)
                self.assertEqual(result.year, year)
                assert isinstance(result, RecognitionResult)
                self.assertEqual(getattr(result.context, "filename_year" if source == "filename" else "folder_year"), year)
                assert result.context is not None
                self.assertEqual(result.context.parent_path, parent)

    def test_full_match_unresolved_exact_or_romaji_identity_never_calls_assisted_sources(self):
        for romaji, prefix, reverse in product((False, True), ("MF-368-", "MF-download-", "MF-2026-"), (False, True)):
            with self.subTest(romaji=romaji, prefix=prefix, reverse=reverse):
                filename = "Magic Knight Rayearth - 02.mp4" if romaji else FILENAME
                result = self._recognize(filename, prefix + RELEASE, CANDIDATES[::-1] if reverse else CANDIDATES, full=True)
                self._assert_manual(result, "ambiguous_romaji_alias" if romaji else "ambiguous_exact_title")
                assert isinstance(result, RecognitionResult)
                assert result.context is not None
                self.assertEqual((result.context.filename_year, result.context.folder_year), ("", ""))
                self.assertNotIn("ai_revalidation", result.metadata)

    def test_full_match_single_candidate_still_matches_either_version(self):
        for candidate, prefix in product(CANDIDATES, ("MF-368-", "MF-download-")):
            with self.subTest(identity=candidate["id"], prefix=prefix):
                result = self._recognize(parent=prefix + RELEASE, candidates=(candidate,), full=True)
                self._assert_matched(result, candidate["id"])
                assert result.candidates[0].score_breakdown is not None
                self.assertEqual(result.candidates[0].score_breakdown.rejected_constraints, [])

    def test_nested_season_and_low_information_episode_use_directory_title(self):
        for prefix, season_name, filename in product(("MF-368-", "MF-download-"), ("Season02", "Season 02"), (FILENAME, "E02.mp4")):
            with self.subTest(prefix=prefix, season_name=season_name, filename=filename):
                parent = "/downloads/" + prefix + RELEASE + "/" + season_name
                result = self._recognize(filename, parent, details=_details(CANDIDATES, (1, 2)), full=True)
                self._assert_manual(result)
                assert isinstance(result, RecognitionResult)
                assert result.context is not None
                self.assertEqual((result.context.season, result.context.episode), (2, 2))
                self.assertEqual((result.effective_season, result.effective_episode), (2, 2))
                assert result.context is not None
                self.assertEqual(result.context.parent_path, parent)
                if filename == "E02.mp4":
                    assert isinstance(result, RecognitionResult)
                    assert result.context is not None
                    self.assertEqual(result.context.normalized_title, result.context.folder_title)
                    assert isinstance(result.context.normalized_title, str)
                    self.assertIn("魔法骑士雷阿斯", result.context.normalized_title)

    def test_full_match_unique_season_position_still_resolves_the_ambiguity(self):
        for winner, reverse in product(CANDIDATES, (False, True)):
            with self.subTest(identity=winner["id"], reverse=reverse):
                details = _details(CANDIDATES, (2,))
                for candidate in CANDIDATES:
                    if candidate["id"] != winner["id"]:
                        details[str(candidate["id"])]["seasons"][0]["episode_count"] = 1
                result = self._recognize(parent=PARENT + "/Season02", candidates=CANDIDATES[::-1] if reverse else CANDIDATES, details=details, full=True)
                self._assert_matched(result, winner["id"])
                self.assertEqual((result.effective_season, result.effective_episode), (2, 2))
                self.assertIn("alias_position_resolution", result.metadata)

    def test_genuine_distinctive_and_short_suffix_conflicts_remain_protected(self):
        cases = (
            ("[ANi] 魔法騎士雷阿斯 Totally Different - 02 [1080p].mp4",
             PARENT.replace(" - 02", " Totally Different - 02"), CANDIDATES[1], "distinctive_title_tokens_missing"),
            ("Kamen Rider X.S01E02.mkv", "MF-368-Kamen Rider X",
             {"id": 1, "name": "Kamen Rider", "first_air_date": "2024-01-01", "media_type": "tv"}, "protected_title_suffix_missing"),
        )
        for filename, parent, candidate, reason in cases:
            with self.subTest(reason=reason):
                result = self._recognize(filename, parent, (candidate,))
                self.assertTrue(result.need_confirm)
                self.assertEqual(result.status, "low_confidence")
                assert result.candidates[0].score_breakdown is not None
                self.assertIn(reason, result.candidates[0].score_breakdown.rejected_constraints)
                assert isinstance(result, RecognitionResult)
                self.assertEqual(result.threshold_decision["reason"], reason)

    def test_real_filename_with_mf_prefix_is_not_trimmed(self):
        for prefix, retained in (("MF-368-", "MF 368"), ("MF-download-", "MF download")):
            with self.subTest(prefix=prefix):
                filename = prefix + FILENAME
                context = scraper.extract_recognition_context(filename, PARENT)
                self.assertEqual(context.filename, filename)
                self.assertIn(retained, context.filename_title)
                self.assertIn(retained, context.normalized_title)
                self.assertTrue(any(retained in query for query in scraper.generate_query_variants(context)))

    def test_non_system_and_non_leading_folder_prefixes_are_not_trimmed(self):
        for prefix in ("MF-other-", "MF-downloaded-", "mf-368-", "Studio-MF-368-"):
            with self.subTest(prefix=prefix):
                parent = prefix + RELEASE
                context = scraper.extract_recognition_context(FILENAME, parent)
                self.assertEqual(context.parent_path, parent)
                self.assertIn(prefix.rstrip("-").replace("-", " "), context.folder_title)
