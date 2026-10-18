"""目录展示不携带NSFW内部标记，计划仍按真实身份防止混入。"""
from __future__ import annotations

from typing import cast

import unittest
from unittest.mock import Mock, patch

from app import database as db
from app.clients.guangya import GuangYaFile
from app.modules.naming import MOVIE_DIR_DEFAULT, NamingContext, render_template
from app.modules.organize import OrganizePlan, Organizer, OrganizeRules
from app.modules.scraper import MatchResult
from tests.support import isolated_test_database


class NsfwDirectoryNamingTests(unittest.TestCase):
    def setUp(self):
        self.organizer = Organizer(client=Mock(), scraper=Mock())
        cast(Mock, self.organizer.scraper.get_detail).return_value = {}
        self.rules = OrganizeRules(nsfw_enabled=True, nsfw_category_name="NSFW")
        self.enterContext(patch("app.modules.media_probe.probe_media_profile", return_value=None))

    def match(self, provider="clean_title", external_id="SODAH-003", *, year="", media_type="movie"):
        return MatchResult(
            provider=provider, external_id=external_id, title="SODAH-003", year=year,
            media_type=media_type, confidence=1.0,
        )

    def plan(self, match, file_id="video"):
        file = GuangYaFile(file_id, "SODAH-003.mp4", False, 100, "etag")
        plan = OrganizePlan(
            file_id=file_id, original_name=file.name, original_path="source",
            match=match, action="move",
        )
        self.organizer._populate_move_plan(
            plan, file, self.rules, match, {}, {}, parsed_season=None,
            automatic=False, media_probe_cache_only=True,
            media_probe_cached_payload="", media_probe_cache_prefetched=True,
        )
        return plan

    def test_nsfw_directory_has_only_title_and_optional_year(self):
        for provider in ("metatube", "clean_title"):
            for media_type in ("movie", "tv"):
                for year in ("", "2024"):
                    with self.subTest(provider=provider, media_type=media_type, year=year):
                        match = self.match(provider, year=year, media_type=media_type)
                        expected = "SODAH-003 (2024)" if year else "SODAH-003"
                        self.assertEqual(self.organizer.build_media_dir(match, self.rules), expected)
                        self.assertEqual(self.organizer.build_show_dir(match, self.rules), expected)
                        self.assertEqual(self.organizer._match_identity_key(match), f"{provider}:SODAH-003")

    def test_ordinary_tmdb_directory_keeps_identity_and_omits_empty_year(self):
        for media_type in ("movie", "tv"):
            for year in ("", "2024"):
                with self.subTest(media_type=media_type, year=year):
                    match = MatchResult(tmdb_id="123", title="普通影视", year=year, media_type=media_type)
                    expected = f"普通影视{' (2024)' if year else ''} {{tmdb-123}}"
                    self.assertEqual(self.organizer.build_media_dir(match, self.rules), expected)
                    self.assertFalse(self.plan(match).identity_guard_required)

    def test_year_suffix_does_not_strip_parentheses_from_the_title(self):
        for year in ("", "2024"):
            with self.subTest(year=year):
                context = NamingContext(title="标题 (特别版)", year=year, identity_tag="{tmdb-123}")
                self.assertEqual(
                    render_template(MOVIE_DIR_DEFAULT, context),
                    f"标题 (特别版){' (2024)' if year else ''} {{tmdb-123}}",
                )
        long_name = render_template(
            MOVIE_DIR_DEFAULT, NamingContext(title="中文长标题" * 100, year="2024", identity_tag="{tmdb-123}"),
        )
        self.assertLessEqual(len(long_name.encode("utf-8")), 240)
        self.assertTrue(long_name.endswith(" (2024) {tmdb-123}"))

    def test_actual_unmarked_target_enables_identity_guard(self):
        for provider in ("metatube", "clean_title"):
            with self.subTest(provider=provider):
                plan = self.plan(self.match(provider))
                self.assertEqual(plan.media_root_path, "NSFW/SODAH-003")
                self.assertEqual(plan.target_path, "NSFW/SODAH-003")
                self.assertTrue(plan.identity_guard_required)
                self.assertEqual(self.organizer._plan_identity(plan), ("movie", f"{provider}:SODAH-003"))

    def test_same_directory_rejects_different_identities_in_one_batch(self):
        first = self.plan(self.match("metatube", "javbus:first"), "first")
        second = self.plan(self.match("metatube", "javbus:second"), "second")
        self.organizer._apply_identity_guards([first, second], identity_history_loader=lambda _plan: set())
        self.assertEqual(first.action, "move")
        self.assertEqual(second.action, "conflict")
        self.assertEqual(second.conflict_decision, "identity_blocked")
        cast(Mock, self.organizer.client.create_dir).assert_not_called()

    def test_history_identity_is_retained_without_visible_marker(self):
        with isolated_test_database():
            db.add_organize_log(
                "guangya", "source", "NSFW/SODAH-003", "archived", "success", "",
                provider="clean_title", external_id="SODAH-003",
                original_parent_id="source", original_name="SODAH-003.mp4", media_type="movie",
            )
            same = self.plan(self.match())
            other = self.plan(self.match("metatube", "javbus:other"))
            self.organizer._apply_identity_guards([same])
            self.organizer._apply_identity_guards([other])
            self.assertEqual(same.action, "move")
            self.assertEqual(other.action, "conflict")
            self.assertEqual(other.conflict_decision, "identity_blocked")
            self.assertEqual(
                self.organizer._historical_root_identities("NSFW/SODAH-003"),
                {("movie", "clean_title:SODAH-003")},
            )

    def test_old_provider_markers_are_still_recognized(self):
        for tag in ("{clean_title-SODAH-003}", "{metatube-javbus-sodah003}"):
            with self.subTest(tag=tag):
                self.assertEqual(self.organizer._identity_marker(f"NSFW/SODAH-003 () {tag}"), tag.lower())
