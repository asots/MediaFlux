"""刮削设置两列等高及单列回退的真实Chromium几何回归。"""
from __future__ import annotations

import unittest

from tests import test_agent_nsfw_clean_review_settings_browser as fixture


@unittest.skipUnless(fixture.sync_playwright is not None, "未安装 Playwright")
class MetadataSettingsAlignmentBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.NsfwCleanReviewSettingsBrowserTests.setUpClass()
        cls.browser = fixture.NsfwCleanReviewSettingsBrowserTests.browser

    @classmethod
    def tearDownClass(cls):
        fixture.NsfwCleanReviewSettingsBrowserTests.tearDownClass()

    _page = fixture.NsfwCleanReviewSettingsBrowserTests._page
    _resolve_config = staticmethod(fixture.NsfwCleanReviewSettingsBrowserTests._resolve_config)
    _rect = staticmethod(fixture.NsfwCleanReviewSettingsBrowserTests._rect)
    _assert_same_rect = fixture.NsfwCleanReviewSettingsBrowserTests._assert_same_rect

    def _open(self, width):
        page, errors = self._page(width, {fixture.PARENT_KEY: "1"})
        styles = fixture.TEMPLATE.split("<style>", 1)[1].split("</style>", 1)[0]
        page.add_style_tag(content=styles)
        return page, errors

    def _cards(self, page):
        return [page.locator(f'[aria-labelledby="metadata-{name}-heading"]') for name in ("tmdb", "tavily", "ai")]

    def test_desktop_card_edges_align_and_loading_preserves_geometry(self):
        for width in (1280, 1440, 1920):
            with self.subTest(width=width):
                page, errors = self._open(width)
                cards = self._cards(page)
                before = [self._rect(card) for card in cards]
                self._resolve_config(page)
                tmdb, tavily, ai = [self._rect(card) for card in cards]
                for old, new in zip(before, (tmdb, tavily, ai)):
                    self._assert_same_rect(old, new)
                self.assertAlmostEqual(tmdb["y"], ai["y"], delta=.5)
                self.assertAlmostEqual(tavily["y"] + tavily["height"], ai["y"] + ai["height"], delta=.5)
                self.assertAlmostEqual(tmdb["width"], ai["width"], delta=.5)
                self.assertAlmostEqual(tavily["y"] - tmdb["y"] - tmdb["height"], 28, delta=.5)
                for card in cards:
                    self.assertGreaterEqual(card.locator(".metadata-card-body").bounding_box()["height"], 100)
                for field in ("tmdbApiUrlInput", "tmdbApiKeyInput"):
                    self.assertAlmostEqual(page.locator(f"#{field}").bounding_box()["height"], 42, delta=.5)
                self.assertTrue(page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
                self.assertEqual(errors, [])

    def test_single_column_keeps_natural_card_heights_without_desktop_stretch(self):
        for width in (390, 768, 1260):
            with self.subTest(width=width):
                page, errors = self._open(width)
                self._resolve_config(page)
                cards = self._cards(page)
                tmdb, tavily, ai = [self._rect(card) for card in cards]
                self.assertAlmostEqual(tmdb["x"], ai["x"], delta=.5)
                self.assertGreater(ai["y"], tavily["y"] + tavily["height"])
                self.assertLess(tavily["height"], tmdb["height"])
                for card in cards:
                    self.assertEqual(card.evaluate("e=>getComputedStyle(e).flexGrow"), "0")
                    self.assertLessEqual(self._rect(card)["width"], width)
                self.assertTrue(page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
                self.assertEqual(errors, [])

    def test_resizing_between_columns_recomputes_edges_without_fixed_heights(self):
        page, errors = self._open(1920)
        self._resolve_config(page)
        original = [self._rect(card) for card in self._cards(page)]
        page.set_viewport_size({"width": 390, "height": 900})
        page.evaluate("new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))")
        tmdb, tavily, ai = [self._rect(card) for card in self._cards(page)]
        self.assertGreater(ai["y"], tavily["y"] + tavily["height"])
        page.set_viewport_size({"width": 1920, "height": 900})
        page.evaluate("new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))")
        for before, card in zip(original, self._cards(page)):
            self._assert_same_rect(before, self._rect(card))
        self.assertEqual(errors, [])
