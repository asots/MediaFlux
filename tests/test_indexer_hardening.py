from __future__ import annotations

import asyncio
import base64
import socket
import threading
import unittest

import httpx

from app.indexers.errors import IndexerInvalidResponse
from app.indexers.http import FixedHostHttpClient, IndexerHttpResponse
from app.indexers.models import IndexerCapabilities, IndexerItem, IndexerPage
from app.indexers.providers.base import IndexerAdapter, magnet_infohash
from app.indexers.providers.mikan import MikanAdapter
from app.indexers.providers.nyaa import NyaaAdapter
from app.indexers.registry import IndexerRegistry
from app.indexers.result_store import IndexerResultStore
from app.indexers.service import IndexerService


PUBLIC_RESOLVER = lambda host, port: [
    (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
]


class FakeHttp:
    def __init__(self, body: bytes, content_type: str = "text/html; charset=utf-8"):
        self.body = body
        self.content_type = content_type

    async def get(self, url, **kwargs):
        return IndexerHttpResponse(
            url=url,
            status_code=200,
            headers={"content-type": self.content_type},
            body=self.body,
        )


class TrackingAdapter(IndexerAdapter):
    capabilities = IndexerCapabilities(True, ("magnet",))
    default_enabled = True

    def __init__(self, site_id: str, tracker: dict):
        self.site_id = site_id
        self.site_name = site_id
        self.base_url = f"https://{site_id}.example/"
        self.tracker = tracker
        self.calls = 0

    async def search(self, request):
        self.calls += 1
        self.tracker["active"] += 1
        self.tracker["max"] = max(self.tracker["max"], self.tracker["active"])
        try:
            await asyncio.sleep(0.03)
        finally:
            self.tracker["active"] -= 1
        seed = (request.query.encode("utf-8").hex() + "0" * 40)[:40]
        return IndexerPage(items=[IndexerItem(
            site_id=self.site_id,
            site_name=self.site_name,
            title=request.query,
            download_state="ready",
            download_kinds=("magnet",),
            magnet=f"magnet:?xt=urn:btih:{seed}",
        )], page=request.page, has_more=False, pagination_supported=True)

    async def resolve(self, stored_result):
        raise NotImplementedError


class IndexerHardeningTests(unittest.IsolatedAsyncioTestCase):
    async def test_dns_validation_does_not_block_event_loop_timeout(self):
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        started = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()
        resolver_threads = []

        def slow_resolver(host, port):
            resolver_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(started.set)
            # 错误地内联执行时不要卡住事件循环；下面的线程断言会明确失败。
            if threading.get_ident() != loop_thread:
                if not release.wait(10):
                    raise RuntimeError("test resolver cleanup timed out")
            finished.set()
            return PUBLIC_RESOLVER(host, port)

        transport = httpx.MockTransport(lambda request: httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"<html></html>", request=request
        ))
        client = FixedHostHttpClient(
            allowed_hosts={"nyaa.si"}, resolver=slow_resolver, transport=transport
        )
        request = asyncio.create_task(client.get("https://nyaa.si/"))
        try:
            # 先确认DNS已进入后台线程，再检查超时；不比较两个短sleep的墙钟先后。
            await asyncio.wait_for(started.wait(), timeout=5)
            self.assertNotEqual(resolver_threads[0], loop_thread)
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(request, timeout=0.05)
            self.assertFalse(finished.is_set(), "DNS仍被测试闸门阻塞时事件循环必须已经超时")
        finally:
            release.set()
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            await client.aclose()

    async def test_concurrency_limit_is_shared_across_search_calls(self):
        tracker = {"active": 0, "max": 0}
        adapters = {name: TrackingAdapter(name, tracker) for name in ("a", "b")}
        service = IndexerService(
            registry=IndexerRegistry(adapters),
            result_store=IndexerResultStore(),
            max_concurrency=2,
            enabled_site_ids=adapters,
        )
        await asyncio.gather(service.search("first"), service.search("second"))
        self.assertLessEqual(tracker["max"], 2)

    async def test_cached_results_receive_fresh_valid_tokens_and_cache_is_bounded(self):
        tracker = {"active": 0, "max": 0}
        adapter = TrackingAdapter("a", tracker)
        store = IndexerResultStore(max_entries=1)
        service = IndexerService(
            registry=IndexerRegistry({"a": adapter}),
            result_store=store,
            enabled_site_ids=("a",),
            cache_ttl_seconds=120,
            max_cache_entries=2,
        )
        first = await service.search("first")
        await service.search("second")
        cached = await service.search("first")
        self.assertTrue(cached.cached)
        self.assertNotEqual(first.items[0].result_id, cached.items[0].result_id)
        self.assertEqual(store.get(cached.items[0].result_id).title, "first")
        await service.search("third")
        self.assertLessEqual(len(service._cache), 2)

    async def test_explicit_empty_enabled_sites_remains_empty(self):
        tracker = {"active": 0, "max": 0}
        adapter = TrackingAdapter("a", tracker)
        service = IndexerService(
            registry=IndexerRegistry({"a": adapter}),
            result_store=IndexerResultStore(),
            enabled_site_ids=(),
        )
        self.assertEqual(service.enabled_site_ids, frozenset())
        with self.assertRaises(Exception):
            await service.search("query")

    async def test_wrong_mime_and_block_page_are_invalid_responses(self):
        nyaa_html = b"<html><table class='table'><tr><th>x</th></tr></table></html>"
        nyaa = NyaaAdapter(
            site_id="nyaa", site_name="Nyaa", base_url="https://nyaa.si/",
            http=FakeHttp(nyaa_html, "application/octet-stream"), default_enabled=True,
        )
        with self.assertRaises(IndexerInvalidResponse):
            await nyaa.search(type("R", (), {"query": "x", "page": 1})())

        mikan = MikanAdapter(http=FakeHttp(b"<html><title>blocked</title></html>"))
        with self.assertRaises(IndexerInvalidResponse):
            await mikan.search(type("R", (), {"query": "x", "page": 1})())

    def test_hex_and_base32_btih_normalize_to_same_key(self):
        raw = bytes.fromhex("0123456789abcdef0123456789abcdef01234567")
        hex_hash = raw.hex()
        base32_hash = base64.b32encode(raw).decode("ascii")
        self.assertEqual(
            magnet_infohash(f"magnet:?xt=urn:btih:{hex_hash}"),
            magnet_infohash(f"magnet:?xt=urn:btih:{base32_hash}"),
        )

    def test_btmh_sha256_normalizes_to_qb_torrent_id(self):
        full_v2_hash = "12" * 32
        self.assertEqual(
            magnet_infohash(f"magnet:?xt=urn:btmh:1220{full_v2_hash}"),
            full_v2_hash[:40],
        )


if __name__ == "__main__":
    unittest.main()
