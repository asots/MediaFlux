"""可选 TMDB 读取的绝对截止时间、响应大小和资源关闭；全程 MockTransport。"""
from __future__ import annotations

import asyncio
import socket
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import tests  # noqa: F401  # 必须先隔离运行配置。
import httpx

from tests.support import isolated_test_database
from app.discovery.cache import DiscoveryCache
from app.discovery.calendar.metadata import CalendarMetadata
from app.discovery.calendar.models import CalendarEntry, SourceUnavailable
from app.discovery.calendar.tmdb_http import CalendarTMDBClient


class SlowBody(httpx.AsyncByteStream):
    closed = False

    async def __aiter__(self):
        while True:
            # 每片段间隔低于普通read timeout，但总响应永不结束。
            await asyncio.sleep(0.01)
            yield b" "

    async def aclose(self):
        self.closed = True


class CalendarTMDBHttpTests(unittest.IsolatedAsyncioTestCase):
    def client(self, handler):
        self.settings = SimpleNamespace(
            api_key="offline-fixture-secret", base_url="https://api.themoviedb.org/3", language="zh-CN",
            config_error="", session=SimpleNamespace(proxies={}), close=Mock(return_value=True),
        )
        client = CalendarTMDBClient(settings_factory=lambda: self.settings, transport=httpx.MockTransport(handler))
        self.settings.close.assert_called_once_with()
        self.addAsyncCleanup(client.aclose)
        return client

    async def test_configuration_and_json_use_existing_tmdb_contract(self):
        def handle(request):
            self.assertEqual(request.url.path, "/3/search/tv")
            self.assertEqual(request.url.params["api_key"], self.settings.api_key)
            self.assertEqual(request.url.params["language"], "zh-CN")
            self.assertEqual(request.headers["accept-encoding"], "identity")
            return httpx.Response(200, json={"results": [], "page": 1, "total_pages": 0, "total_results": 0})
        client = self.client(handle)
        result = await client.get("/search/tv", {"query": "离线测试"}, deadline_at=time.monotonic() + 1)
        self.assertEqual(result["results"], [])
        self.assertNotIn("api_key", result)

    async def test_slow_drip_response_is_cancelled_and_closed_at_total_deadline(self):
        body = SlowBody()
        client = self.client(lambda request: httpx.Response(200, stream=body))
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            await client.get("/search/tv", {}, deadline_at=started + 0.04)
        self.assertLess(time.monotonic() - started, 1)
        self.assertTrue(body.closed)

    async def test_redirect_and_failure_are_not_followed_or_retried(self):
        for status in (302, 401, 429, 503):
            with self.subTest(status=status):
                calls = []
                def handle(request):
                    calls.append(request.url)
                    return httpx.Response(status, headers={"Location": "https://untrusted.invalid/"})
                client = self.client(handle)
                with self.assertRaisesRegex(SourceUnavailable, "暂不可用"):
                    await client.get("/search/tv", {}, deadline_at=time.monotonic() + 1)
                self.assertEqual(len(calls), 1)

    async def test_declared_size_actual_size_and_unrequested_compression_are_rejected(self):
        cases = (
            ({"Content-Length": str(2 * 1024 * 1024 + 1)}, b"{}"),
            ({"Content-Length": "-1"}, b"{}"),
            ({}, b" " * (2 * 1024 * 1024 + 1)),
            ({"Content-Encoding": "br"}, b"{}"),
        )
        for headers, content in cases:
            with self.subTest(headers=headers, length=len(content)):
                client = self.client(lambda request: httpx.Response(200, headers=headers, content=content))
                with self.assertRaises((SourceUnavailable, httpx.DecodingError)):
                    await client.get("/search/tv", {}, deadline_at=time.monotonic() + 1)

    async def test_no_arbitrary_path_or_retries(self):
        client = self.client(lambda request: self.fail("unexpected request"))
        for path, retries in (("/tv/1", 0), ("https://evil.invalid/", 0), ("/search/tv", 1)):
            with self.subTest(path=path, retries=retries), self.assertRaises(SourceUnavailable):
                await client.get(path, {}, deadline_at=time.monotonic() + 1, retries=retries)


class CalendarTMDBDNSRoundTests(unittest.TestCase):
    """真实 HTTPX 解析链、合成 resolver；不使用真实 DNS/连接或部署配置。"""

    def setUp(self):
        self.enterContext(isolated_test_database())
        self.cache = DiscoveryCache()
        self.attempts = []

        def forbidden(*args, **kwargs):
            self.attempts.append(True)
            raise AssertionError("测试禁止真实网络/DNS")

        for name in ("getaddrinfo", "create_connection", "socket.connect", "socket.connect_ex"):
            self.enterContext(patch("socket." + name, side_effect=forbidden))
        self.show = CalendarEntry(
            "tencent", "dns_fixture", "离线 DNS 动漫", "animation",
            "https://v.qq.com/x/cover/dns_fixture.html", free_progress="免费至第3集", evidence="离线事实",
        )

    def tearDown(self):
        self.assertEqual(self.attempts, [])

    @staticmethod
    def settings(*, base_url="https://api.themoviedb.org/3", proxy=""):
        return SimpleNamespace(
            api_key="offline-fixture-secret", base_url=base_url, language="zh-CN", config_error="",
            session=SimpleNamespace(proxies={"https": proxy} if proxy else {}), close=Mock(return_value=True),
        )

    def test_cancelled_native_dns_is_bounded_across_loops_and_capacity_recovers(self):
        from app.discovery.calendar import http as module
        entered, release, returned = (threading.Event() for _ in range(3))
        calls = []

        class Slots(threading.BoundedSemaphore):
            def release(self, n=1):
                super().release(n)
                returned.set()

        def blocked(host, port, *args, **kwargs):
            calls.append((host, port))
            entered.set()
            if not release.wait(5):
                raise AssertionError("测试 resolver 必须释放")
            raise socket.gaierror("合成 DNS 故障")

        clients = [CalendarTMDBClient(settings_factory=self.settings) for _ in range(4)]
        factory = iter(clients)
        mapper = CalendarMetadata(self.cache, client_factory=lambda: next(factory), budget_seconds=0.3)
        try:
            with patch.object(module, "_DNS_SLOTS", Slots(1)), patch("socket.getaddrinfo", side_effect=blocked):
                for _ in range(3):
                    card = mapper.enrich((self.show,))[0]
                    self.assertEqual(card["free_progress"], self.show.free_progress)
                self.assertTrue(entered.is_set())
                self.assertEqual(calls, [(b"api.themoviedb.org", 443)])
                self.assertFalse(returned.is_set())
                self.assertTrue(all(client._client.is_closed for client in clients[:3]))
                release.set()
                self.assertTrue(returned.wait(2))
                mapper.enrich((self.show,))
                self.assertEqual(calls, [(b"api.themoviedb.org", 443)] * 2)
        finally:
            release.set()
            for client in clients:
                asyncio.run(client.aclose())

    def test_owned_loop_preserves_custom_base_url_and_native_proxy_dns_target(self):
        cases = (
            ("http://metadata.internal:8123/custom/v3", "", "metadata.internal", 8123),
            ("https://metadata.internal:8443/custom/v3", "http://proxy-user:proxy-pass@proxy.internal:8888",
             "proxy.internal", 8888),
            ("https://metadata.internal:8443/custom/v3", "https://proxy-user:proxy-pass@proxy.internal:9443",
             "proxy.internal", 9443),
        )
        for base_url, proxy, host, port in cases:
            with self.subTest(base_url=base_url, proxy_scheme=proxy.split(":", 1)[0]):
                settings = self.settings(base_url=base_url, proxy=proxy)
                client = CalendarTMDBClient(settings_factory=lambda settings=settings: settings)
                mapper = CalendarMetadata(self.cache, client_factory=lambda client=client: client, budget_seconds=1)
                with patch("socket.getaddrinfo", side_effect=socket.gaierror("合成 DNS 故障")) as resolver:
                    card = mapper.enrich((self.show,))[0]
                resolver.assert_called_once()
                # AnyIO 原生链会把主机编码成 ASCII bytes；有界层不得改写。
                self.assertEqual(resolver.call_args.args[:2], (host.encode("ascii"), port))
                self.assertEqual(card["free_progress"], self.show.free_progress)
                self.assertTrue(client._client.is_closed)
                self.assertEqual(client.base_url, base_url)
                settings.close.assert_called_once_with()
