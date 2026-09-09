"""BTBtla 实时零结果摘要的最小契约；空结果不能误报为站点不可用。"""
from __future__ import annotations

from pathlib import Path
import unittest

from app.indexers.errors import IndexerUnavailable
from app.indexers.models import IndexerSearchRequest
from app.indexers.providers.btbtla import BTBtlaAdapter
from tests.test_indexer_providers import BTBTLA_DETAIL_HTML, BTBTLA_SEARCH_HTML, FakeHttpClient

_EMPTY = (Path(__file__).with_name("fixtures") / "indexers" / "btbtla-empty-search.html").read_bytes()


class BTBtlaEmptySearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_zero_total_summary_is_empty_without_trying_mirror(self):
        http = FakeHttpClient(_EMPTY)
        page = await BTBtlaAdapter(http=http).search(IndexerSearchRequest.create("Big Buck Bunny"))
        self.assertEqual(page.items, [])
        self.assertEqual(page.page, 1)
        self.assertFalse(page.has_more)
        self.assertTrue(page.pagination_supported)
        self.assertEqual(len(http.calls), 1)
        self.assertTrue(http.calls[0]["url"].startswith("https://www.btbtlb.com/search/"))

    async def test_zero_total_with_whitespace_is_still_a_valid_empty_page(self):
        body = _EMPTY.replace(b'>0</strong>', b'> 0 \n</strong>')
        http = FakeHttpClient(body)
        page = await BTBtlaAdapter(http=http).search(IndexerSearchRequest.create("Missing"))
        self.assertEqual(page.items, [])
        self.assertEqual(len(http.calls), 1)

    async def test_missing_structure_nonzero_count_and_challenge_are_not_empty_results(self):
        for name, body in (
            ("blank", b'<html><body></body></html>'),
            ("unrelated_counter", b'<html><h2><strong class="mac_total">0</strong></h2></html>'),
            ("nonzero_results_missing", _EMPTY.replace(b'>0</strong>', b'>12</strong>')),
            ("verification", _EMPTY + b'<div id="cf-chl-widget">Just a moment...</div>'),
        ):
            with self.subTest(page=name):
                http = FakeHttpClient(body)
                with self.assertRaises(IndexerUnavailable):
                    await BTBtlaAdapter(http=http).search(IndexerSearchRequest.create("Missing"))

    async def test_actual_candidates_take_priority_over_stale_zero_summary(self):
        http = FakeHttpClient(BTBTLA_DETAIL_HTML)
        http.responses = [_EMPTY + BTBTLA_SEARCH_HTML, BTBTLA_DETAIL_HTML]
        page = await BTBtlaAdapter(http=http).search(IndexerSearchRequest.create("Frieren"))
        self.assertTrue(page.items)
