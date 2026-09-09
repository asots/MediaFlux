"""Mikan 的网络故障应尝试已注册备站，不把传输异常误当成停止条件。"""
from __future__ import annotations

import asyncio
import unittest

import httpx

from app.indexers.errors import IndexerSecurityError, IndexerTimeout, IndexerUnavailable
from app.indexers.http import IndexerHttpResponse
from app.indexers.models import IndexerSearchRequest
from app.indexers.providers.mikan import MikanAdapter


_HTML = b'''<table><tbody><tr class="js-search-results-row">
<td><input data-magnet="magnet:?xt=urn:btih:0123456789abcdef0123456789abcdef01234567"></td>
<td><a href="/Home/Episode/fixture">Fixture Episode</a></td>
</tr></tbody></table>'''


class _Transport:
    def __init__(self, primary_error, mirror_error=None):
        self.primary_error = primary_error
        self.mirror_error = mirror_error
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append(url)
        error = self.primary_error if len(self.calls) == 1 else self.mirror_error
        if error is not None:
            raise error
        return IndexerHttpResponse(
            url=url, status_code=200, headers={"content-type": "text/html; charset=utf-8"}, body=_HTML,
        )


class MikanNetworkFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_or_connection_failure_tries_registered_mirror(self):
        for error in (httpx.ReadTimeout("fixture timeout"), httpx.ConnectError("fixture connection failure")):
            with self.subTest(error=type(error).__name__):
                http = _Transport(error)
                page = await MikanAdapter(http=http).search(IndexerSearchRequest.create("Fixture"))
                self.assertEqual(len(page.items), 1)
                self.assertEqual(http.calls, [
                    "https://mikanani.me/Home/Search", "https://mikanime.tv/Home/Search",
                ])
                self.assertEqual(page.items[0].detail_url, "https://mikanime.tv/Home/Episode/fixture")

    async def test_exhausted_network_failures_use_typed_provider_errors(self):
        for error_type, expected in ((httpx.ReadTimeout, IndexerTimeout), (httpx.ConnectError, IndexerUnavailable)):
            with self.subTest(error=error_type.__name__):
                http = _Transport(error_type("fixture primary"), error_type("fixture mirror"))
                with self.assertRaises(expected):
                    await MikanAdapter(http=http).search(IndexerSearchRequest.create("Fixture"))
                self.assertEqual(len(http.calls), 2)

    async def test_caller_cancellation_does_not_start_another_network_request(self):
        http = _Transport(asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await MikanAdapter(http=http).search(IndexerSearchRequest.create("Fixture"))
        self.assertEqual(http.calls, ["https://mikanani.me/Home/Search"])

    async def test_security_failure_remains_fail_closed(self):
        http = _Transport(IndexerSecurityError("upstream resolved to a non-public address"))
        with self.assertRaises(IndexerSecurityError):
            await MikanAdapter(http=http).search(IndexerSearchRequest.create("Fixture"))
        self.assertEqual(http.calls, ["https://mikanani.me/Home/Search"])
