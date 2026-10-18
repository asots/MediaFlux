from __future__ import annotations

import asyncio
import base64
import gzip
import threading
import unittest
import zlib
from typing import cast
from unittest.mock import PropertyMock, patch

import httpx
from httpx import AsyncClient

import tests  # noqa: F401 -- 应用导入前隔离生产配置和 DB。
from app.clients.http import (
    BrowserImpersonatingHttpClient,
    FixedHostHttpClient,
    HttpInvalidResponse,
    HttpResponseTooLarge,
    HttpSecurityError,
)
from app.indexers.providers.btbtla import BTBtlaAdapter

PUBLIC_DNS = lambda host, port: [(2, 1, 6, "", ("93.184.216.34", port))]
PRIVATE_DNS = lambda host, port: [(2, 1, 6, "", ("127.0.0.1", port))]




class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False
        self.started = False
        self.read_chunks = 0
        self.read_bytes = 0

    async def __aiter__(self):
        self.started = True
        for chunk in self.chunks:
            self.read_chunks += 1
            self.read_bytes += len(chunk)
            yield chunk

    async def aclose(self):
        self.closed = True


class FakeCurlResponse:
    def __init__(self, status_code=200, content=b"ok", headers=None, url="https://example.com/"):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"content-type": "text/html"}
        self.url = url


class FakeCurlSession:
    curl_options: dict
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False
        self.cookies = {}

    def get(self, url, **kwargs):
        recorded = dict(kwargs)
        recorded["curl_options"] = dict(getattr(self, "curl_options", {}))
        self.calls.append((url, recorded))
        response = self.responses.pop(0)
        callback = kwargs.get("content_callback")
        if callback is not None:
            callback(response.content)
        return response

    def close(self):
        self.closed = True


class IndexerHttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        client = getattr(self, "client", None)
        if client is not None:
            await client.aclose()

    async def test_close_tolerates_only_asyncio_closed_loop_provenance(self):
        client = self.make_client(
            lambda request: httpx.Response(200, content=b"ok")
        )
        original = client._client
        await original.aclose()
        closed_loop = asyncio.new_event_loop()
        closed_loop.close()

        class ClosedLoopClient:
            async def aclose(self):
                closed_loop.call_soon(lambda: None)

        client._client = cast(AsyncClient, ClosedLoopClient())
        await client.aclose()

        class SameTextClient:
            async def aclose(self):
                raise RuntimeError("Event loop is closed")

        client._client = cast(AsyncClient, SameTextClient())
        with self.assertRaisesRegex(RuntimeError, "Event loop is closed"):
            await client.aclose()

        class OtherTextClient:
            async def aclose(self):
                raise RuntimeError("transport shutdown failed")

        client._client = cast(AsyncClient, OtherTextClient())
        with self.assertRaisesRegex(RuntimeError, "transport shutdown failed"):
            await client.aclose()

        class ClosedLoopSubclass(RuntimeError):
            pass

        class SubclassClient:
            async def aclose(self):
                try:
                    closed_loop.call_soon(lambda: None)
                except RuntimeError as exc:
                    raise ClosedLoopSubclass(*exc.args) from exc

        client._client = cast(AsyncClient, SubclassClient())
        with self.assertRaises(ClosedLoopSubclass):
            await client.aclose()
        self.client = None

    def make_client(self, handler, *, resolver=PUBLIC_DNS, max_response_bytes=1024):
        self.client = FixedHostHttpClient(
            allowed_hosts={"nyaa.si"},
            transport=httpx.MockTransport(handler),
            resolver=resolver,
            max_response_bytes=max_response_bytes,
            timeout_seconds=1,
        )
        return self.client

    async def test_get_retry_budget_and_post_idle_budget_are_explicit(self):
        for retries, configured, expected in ((1, 10, 5), (0, 30, 30), (0, 120, 120)):
            with self.subTest(retries=retries, configured=configured):
                requests = []

                def timed_out(request, requests=requests):
                    requests.append(request)
                    raise httpx.ReadTimeout("offline timeout", request=request)

                client = FixedHostHttpClient(
                    allowed_hosts={"nyaa.si"}, resolver=PUBLIC_DNS,
                    transport=httpx.MockTransport(timed_out),
                    timeout_seconds=configured, get_timeout_retries=retries,
                )
                self.addAsyncCleanup(client.aclose)
                with self.assertRaises(httpx.ReadTimeout):
                    await client.get("https://nyaa.si/search")
                self.assertEqual(len(requests), retries + 1)
                for request in requests:
                    self.assertEqual(request.extensions["timeout"], {
                        "connect": min(4.0, expected), "read": expected,
                        "write": expected, "pool": expected,
                    })
                requests.clear()
                with self.assertRaises(httpx.ReadTimeout):
                    await client.post_json("https://nyaa.si/search", json={"query": "fixture"})
                self.assertEqual(len(requests), 1)

    async def test_pinned_single_host_keeps_pool_but_multi_host_disables_reuse(self):
        single = FixedHostHttpClient(
            allowed_hosts={"nyaa.si"},
            resolver=PUBLIC_DNS,
            pin_resolved_address=True,
        )
        multi = FixedHostHttpClient(
            allowed_hosts={"nyaa.si", "sukebei.nyaa.si"},
            resolver=PUBLIC_DNS,
            pin_resolved_address=True,
        )
        try:
            assert isinstance(single._client._transport, httpx.AsyncHTTPTransport)
            assert isinstance(multi._client._transport, httpx.AsyncHTTPTransport)
            from httpcore import AsyncConnectionPool
            assert isinstance(single._client._transport._pool, AsyncConnectionPool)
            assert isinstance(multi._client._transport._pool, AsyncConnectionPool)
            self.assertGreater(
                single._client._transport._pool._max_keepalive_connections, 0
            )
            self.assertEqual(
                multi._client._transport._pool._max_keepalive_connections, 0
            )
        finally:
            await single.aclose()
            await multi.aclose()

    async def test_rejects_non_https_off_host_credentials_and_private_dns(self):
        client = self.make_client(lambda request: httpx.Response(200, content=b"ok"))
        rejected = (
            "http://nyaa.si/",
            "https://example.com/",
            "https://user:pass@nyaa.si/",
            "https://localhost/",
        )
        for url in rejected:
            with self.subTest(url=url):
                with self.assertRaises(HttpSecurityError):
                    await client.get(url)

        await client.aclose()
        self.client = FixedHostHttpClient(
            allowed_hosts={"nyaa.si"},
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"ok")),
            resolver=PRIVATE_DNS,
        )
        with self.assertRaises(HttpSecurityError):
            await self.client.get("https://nyaa.si/")

    async def test_follows_relative_redirect_but_rejects_redirect_to_unregistered_host(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            if request.url.path == "/start":
                return httpx.Response(302, headers={"Location": "/final"})
            return httpx.Response(200, headers={"Content-Type": "text/html"}, content=b"done")

        client = self.make_client(handler)
        response = await client.get("https://nyaa.si/start")
        self.assertEqual(response.body, b"done")
        self.assertEqual([httpx.URL(url).path for url in seen], ["/start", "/final"])

        await client.aclose()
        client = self.make_client(
            lambda request: httpx.Response(302, headers={"Location": "https://evil.example/file"})
        )
        with self.assertRaises(HttpSecurityError):
            await client.get("https://nyaa.si/start")

    async def test_search_form_redirects_to_get_without_replaying_body(self):
        seen = []

        def handler(request):
            seen.append((request.method, request.content, request.headers.get("content-type")))
            if request.url.path == "/search":
                return httpx.Response(302, headers={"Location": "/results"})
            return httpx.Response(200, content=b"matches")

        client = self.make_client(handler)
        result = await client.post_form("https://nyaa.si/search", content=b"keyboard=%C8%FD%CC%E5")
        self.assertEqual(result.body, b"matches")
        self.assertEqual(seen, [("POST", b"keyboard=%C8%FD%CC%E5", "application/x-www-form-urlencoded"), ("GET", b"", None)])

    async def test_search_form_rejects_cross_host_redirect_and_json_stays_strict(self):
        client = self.make_client(lambda request: httpx.Response(302, headers={"Location": "https://evil.example/search"}))
        with self.assertRaises(HttpSecurityError):
            await client.post_form("https://nyaa.si/search", content=b"keyboard=test")
        await client.aclose()
        client = self.make_client(lambda request: httpx.Response(302, headers={"Location": "/result"}))
        with self.assertRaises(HttpSecurityError):
            await client.post_json("https://nyaa.si/search", json={})

    async def test_enforces_declared_and_streamed_response_size_limits(self):
        client = self.make_client(
            lambda request: httpx.Response(200, headers={"Content-Length": "11"}, content=b"01234567890"),
            max_response_bytes=10,
        )
        with self.assertRaises(HttpResponseTooLarge):
            await client.get("https://nyaa.si/")

        await client.aclose()
        client = self.make_client(
            lambda request: httpx.Response(200, content=b"01234567890"),
            max_response_bytes=10,
        )
        with self.assertRaises(HttpResponseTooLarge):
            await client.get("https://nyaa.si/")

    async def test_browser_client_rewrites_sni_keeps_host_and_impersonates_chrome(self):
        session = FakeCurlSession([FakeCurlResponse(url="https://btbtlb.com/search/demo")])
        client = BrowserImpersonatingHttpClient(
            allowed_hosts={"www.btbtlb.com", "btbtlb.com"},
            resolver=PUBLIC_DNS,
            session_factory=lambda: session,
            sni_host="btbtlb.com",
        )
        self.client = client

        response = await client.get("https://www.btbtlb.com/search/demo")

        url, options = session.calls[0]
        self.assertEqual(url, "https://btbtlb.com/search/demo")
        self.assertEqual(options["headers"]["Host"], "www.btbtlb.com")
        self.assertEqual(options["impersonate"], "chrome")
        self.assertFalse(options["allow_redirects"] )
        from curl_cffi.const import CurlOpt
        self.assertEqual(
            options["curl_options"][CurlOpt.RESOLVE],
            ["btbtlb.com:443:93.184.216.34"],
        )
        self.assertNotIn(CurlOpt.RESOLVE, session.curl_options)
        self.assertEqual(response.body, b"ok")

    async def test_btbtla_browser_session_keeps_start_intervals_after_slow_request_backlog(self):
        for name in ("getaddrinfo", "create_connection", "socket.connect", "socket.connect_ex"):
            self.enterContext(patch("socket." + name, side_effect=AssertionError("禁止真实网络/DNS")))

        interval = 1 / 64  # 可精确表示的虚拟间隔，不依赖墙钟精度。
        clock = [0.0]
        starts = []
        first_started = asyncio.Event()
        release_first = threading.Event()
        loop = asyncio.get_running_loop()

        class SlowFirstSession(FakeCurlSession):
            def get(self, url, **kwargs):
                # 记录真实线程中的 session 入口，而不是 Pacer 的放行时刻。
                starts.append((url, clock[0]))
                if len(starts) == 1:
                    loop.call_soon_threadsafe(first_started.set)
                    if not release_first.wait(timeout=2):
                        raise TimeoutError("慢请求未被释放")
                return super().get(url, **kwargs)

        urls = [f"https://www.btbtlb.com/search/{name}" for name in ("slow", "one", "two", "three")]
        session = SlowFirstSession([FakeCurlResponse() for _ in urls])
        self.client = BrowserImpersonatingHttpClient(
            allowed_hosts={"www.btbtlb.com", "btbtlb.com"},
            resolver=PUBLIC_DNS,
            session_factory=lambda: session,
            sni_host="btbtlb.com",
        )

        async def sleep(delay):
            clock[0] += delay

        adapter = BTBtlaAdapter(
            http=self.client, min_interval_seconds=interval,
            monotonic=lambda: clock[0], sleeper=sleep,
        )
        queued = asyncio.Event()
        queued_count = 0

        async def queued_request(url):
            nonlocal queued_count
            queued_count += 1
            if queued_count == len(urls) - 1:
                queued.set()
            return await adapter._get(url)

        tasks = [asyncio.create_task(adapter._get(urls[0]))]
        try:
            await asyncio.wait_for(first_started.wait(), 2)
            tasks.extend(asyncio.create_task(queued_request(url)) for url in urls[1:])
            await asyncio.wait_for(queued.wait(), 2)
            self.assertEqual(len(starts), 1)
            self.assertTrue(all(not task.done() for task in tasks))

            # 首项耗时超过所有积压项的间隔总和；提前放行会导致后续项同时启动。
            clock[0] = interval * 8
            release_first.set()
            responses = await asyncio.wait_for(asyncio.gather(*tasks), 2)
        finally:
            release_first.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        self.assertEqual([response.body for response in responses], [b"ok"] * len(urls))
        self.assertEqual(len(starts), len(urls))
        self.assertCountEqual(
            [url for url, _options in session.calls],
            [str(httpx.URL(url).copy_with(host="btbtlb.com")) for url in urls],
        )
        for (previous_url, previous_start), (url, start) in zip(starts, starts[1:]):
            self.assertGreaterEqual(start - previous_start, interval, f"{previous_url} -> {url}")

    async def test_browser_client_stops_oversized_body_during_receive(self):
        session = FakeCurlSession([FakeCurlResponse(content=b"12345")])
        client = BrowserImpersonatingHttpClient(
            allowed_hosts={"www.example.com"},
            resolver=PUBLIC_DNS,
            session_factory=lambda: session,
            max_response_bytes=4,
        )
        self.client = client

        with self.assertRaises(HttpResponseTooLarge):
            await client.get("https://www.example.com/search/demo")

    async def test_browser_client_rejects_unregistered_sni_host(self):
        with self.assertRaisesRegex(ValueError, "sni_host"):
            BrowserImpersonatingHttpClient(
                allowed_hosts={"www.btbtlb.com"},
                resolver=PUBLIC_DNS,
                session_factory=lambda: FakeCurlSession([]),
                sni_host="btbtlb.com",
            )

    async def test_browser_client_validates_actual_sni_host_dns(self):
        session = FakeCurlSession([])

        def resolver(host, port):
            address = "127.0.0.1" if host == "btbtlb.com" else "93.184.216.34"
            return [(2, 1, 6, "", (address, port))]

        client = BrowserImpersonatingHttpClient(
            allowed_hosts={"www.btbtlb.com", "btbtlb.com"},
            resolver=resolver,
            session_factory=lambda: session,
            sni_host="btbtlb.com",
        )
        self.client = client

        with self.assertRaises(HttpSecurityError):
            await client.get("https://www.btbtlb.com/search/demo")
        self.assertEqual(session.calls, [])

    async def test_browser_client_warms_up_once_and_reuses_cookie_session(self):
        session = FakeCurlSession([
            FakeCurlResponse(url="https://www.example.com/"),
            FakeCurlResponse(content=b"first", url="https://www.example.com/search/one"),
            FakeCurlResponse(content=b"second", url="https://www.example.com/search/two"),
        ])
        client = BrowserImpersonatingHttpClient(
            allowed_hosts={"www.example.com"},
            resolver=PUBLIC_DNS,
            session_factory=lambda: session,
            warmup_url="https://www.example.com/",
        )
        self.client = client

        first = await client.get("https://www.example.com/search/one")
        second = await client.get("https://www.example.com/search/two")

        self.assertEqual(first.body, b"first")
        self.assertEqual(second.body, b"second")
        self.assertEqual([call[0] for call in session.calls], [
            "https://www.example.com/",
            "https://www.example.com/search/one",
            "https://www.example.com/search/two",
        ])

    async def test_browser_client_close_failure_retains_session_for_retry(self):
        class FailOnceSession(FakeCurlSession):
            def __init__(self):
                super().__init__([FakeCurlResponse(url="https://www.example.com/")])
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                if self.close_calls == 1:
                    raise RuntimeError("simulated close failure")
                super().close()

        session = FailOnceSession()
        client = BrowserImpersonatingHttpClient(
            allowed_hosts={"www.example.com"},
            resolver=PUBLIC_DNS,
            session_factory=lambda: session,
        )
        self.client = client
        await client.get("https://www.example.com/")

        with self.assertRaisesRegex(RuntimeError, "simulated close failure"):
            await client.aclose()
        self.assertIs(client._session, session)
        self.assertEqual(session.close_calls, 1)

        await client.aclose()
        self.assertIsNone(client._session)
        self.assertTrue(session.closed)
        self.assertEqual(session.close_calls, 2)
        self.client = None

    async def test_browser_client_injects_configured_cookies_into_private_session(self):
        session = FakeCurlSession([FakeCurlResponse(url="https://www.example.com/")])
        client = BrowserImpersonatingHttpClient(
            allowed_hosts={"www.example.com"},
            resolver=PUBLIC_DNS,
            session_factory=lambda: session,
            cookies={"cf_clearance": "secret-value"},
        )
        self.client = client

        await client.get("https://www.example.com/")

        self.assertEqual(session.cookies, {"cf_clearance": "secret-value"})
        self.assertNotIn("secret-value", repr(session.calls))

    async def test_browser_client_rejects_off_host_before_session_request(self):
        session = FakeCurlSession([])
        client = BrowserImpersonatingHttpClient(
            allowed_hosts={"www.example.com"},
            resolver=PUBLIC_DNS,
            session_factory=lambda: session,
        )
        self.client = client

        with self.assertRaises(HttpSecurityError):
            await client.get("https://evil.example/search")
        self.assertEqual(session.calls, [])

    async def test_query_params_are_encoded_by_httpx(self):
        observed: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            observed["query"] = request.url.params.get("q")
            return httpx.Response(200, content=b"ok")

        client = self.make_client(handler)
        await client.get("https://nyaa.si/", params={"q": "葬送 / test"})
        self.assertEqual(observed["query"], "葬送 / test")


    async def test_pinned_request_uses_validated_ip_with_original_host_and_sni(self):
        observed: dict = {}
        resolution_calls = []

        def resolver(host, port):
            resolution_calls.append((host, port))
            return PUBLIC_DNS(host, port)

        def handler(request: httpx.Request) -> httpx.Response:
            observed["url_host"] = request.url.host
            observed["host_header"] = request.headers.get("host")
            observed["sni_hostname"] = request.extensions.get("sni_hostname")
            return httpx.Response(200, content=b"ok")

        self.client = FixedHostHttpClient(
            allowed_hosts={"nyaa.si"},
            transport=httpx.MockTransport(handler),
            resolver=resolver,
            pin_resolved_address=True,
        )
        response = await self.client.get("https://nyaa.si/path")

        self.assertEqual(resolution_calls, [("nyaa.si", 443)])
        self.assertEqual(observed["url_host"], "93.184.216.34")
        self.assertEqual(observed["host_header"], "nyaa.si")
        self.assertEqual(observed["sni_hostname"], "nyaa.si")
        self.assertEqual(response.url, "https://nyaa.si/path")


if __name__ == "__main__":
    unittest.main()

class FixedHostPostJsonTests(unittest.IsolatedAsyncioTestCase):
    async def test_post_json_preserves_method_body_and_headers(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["body"] = request.content
            seen["authorization"] = request.headers.get("authorization")
            return httpx.Response(200, json={"results": []})

        client = FixedHostHttpClient(
            allowed_hosts={"api.tavily.com"},
            transport=httpx.MockTransport(handler),
            resolver=PUBLIC_DNS,
        )
        try:
            response = await client.post_json(
                "https://api.tavily.com/search",
                json={"query": "MediaFlux"},
                headers={"Authorization": "Bearer secret"},
                max_redirects=0,
            )
        finally:
            await client.aclose()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen["method"], "POST")
        self.assertIn(b'MediaFlux', seen["body"])
        self.assertEqual(seen["authorization"], "Bearer secret")

    async def test_stream_post_json_preserves_security_context_and_bounds_chunks(self):
        seen: dict = {}
        stream = ChunkStream([b"data: one\n\n", b"data: two\n\n"])

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["url_host"] = request.url.host
            seen["host_header"] = request.headers.get("host")
            seen["sni_hostname"] = request.extensions.get("sni_hostname")
            seen["accept"] = request.headers.get("accept")
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                stream=stream,
            )

        client = FixedHostHttpClient(
            allowed_hosts={"api.tavily.com"},
            transport=httpx.MockTransport(handler),
            resolver=PUBLIC_DNS,
            pin_resolved_address=True,
            max_response_bytes=64,
        )
        try:
            async with client.stream_post_json(
                "https://api.tavily.com/v1/responses",
                json={"stream": True},
                headers={"Accept": "text/event-stream"},
                max_redirects=0,
            ) as response:
                chunks = [chunk async for chunk in response.aiter_bytes()]
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "text/event-stream")
        finally:
            await client.aclose()

        self.assertEqual(chunks, [b"data: one\n\n", b"data: two\n\n"])
        self.assertEqual(seen["method"], "POST")
        self.assertEqual(seen["url_host"], "93.184.216.34")
        self.assertEqual(seen["host_header"], "api.tavily.com")
        self.assertEqual(seen["sni_hostname"], "api.tavily.com")
        self.assertEqual(seen["accept"], "text/event-stream")
        self.assertTrue(stream.closed)

    async def test_stream_post_json_enforces_cumulative_size_limit(self):
        stream = ChunkStream([b"12345", b"67890", b"x"])
        client = FixedHostHttpClient(
            allowed_hosts={"api.tavily.com"},
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream"},
                    stream=stream,
                )
            ),
            resolver=PUBLIC_DNS,
            max_response_bytes=10,
        )
        try:
            with self.assertRaises(HttpResponseTooLarge):
                async with client.stream_post_json(
                    "https://api.tavily.com/v1/responses", json={"stream": True}
                ) as response:
                    _ = [chunk async for chunk in response.aiter_bytes()]
        finally:
            await client.aclose()
        self.assertTrue(stream.closed)

    async def test_post_json_rejects_redirect_and_off_host(self):
        client = FixedHostHttpClient(
            allowed_hosts={"api.tavily.com"},
            transport=httpx.MockTransport(
                lambda request: httpx.Response(307, headers={"Location": "https://evil.example/"})
            ),
            resolver=PUBLIC_DNS,
        )
        try:
            with self.assertRaises(HttpSecurityError):
                await client.post_json("https://api.tavily.com/search", json={}, max_redirects=0)
            with self.assertRaises(HttpSecurityError):
                await client.post_json("https://evil.example/search", json={})
        finally:
            await client.aclose()


class FixedHostIdentityEncodingTests(unittest.IsolatedAsyncioTestCase):
    """编码校验必须早于 raw/decoded 迭代，三种入口共用同一边界。"""

    METHODS = ("get", "post", "stream")
    LIMIT = 2 * 1024 * 1024
    CHUNK = 64 * 1024
    PNG = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg=="
    )

    def setUp(self):
        self.network_attempts = []
        self.stream_entries = 0

        def forbidden(*args, **kwargs):
            self.network_attempts.append(True)
            raise AssertionError("HTTP 边界回归禁止真实网络/DNS")

        for name in ("getaddrinfo", "create_connection", "socket.connect", "socket.connect_ex"):
            self.enterContext(patch("socket." + name, side_effect=forbidden))

    def tearDown(self):
        self.assertEqual(self.network_attempts, [])

    def make_client(self, stream=None, *, headers=(), strict=True, maximum=LIMIT, response=None):
        if response is None:
            response = httpx.Response(200, headers=headers, stream=stream)
        calls = []

        def handle(request):
            calls.append(request)
            return response

        options: dict = {} if strict is None else {"require_identity_encoding": strict}
        client = FixedHostHttpClient(
            allowed_hosts={"nyaa.si"}, resolver=PUBLIC_DNS,
            transport=httpx.MockTransport(handle), max_response_bytes=maximum, **options,
        )
        self.addAsyncCleanup(client.aclose)
        return client, response, calls

    async def read(self, client, method):
        if method == "get":
            return (await client.get("https://nyaa.si/", max_redirects=0)).body
        if method == "post":
            return (await client.post_json("https://nyaa.si/", json={}, max_redirects=0)).body
        async with client.stream_post_json("https://nyaa.si/", json={}, max_redirects=0) as response:
            self.stream_entries += 1
            return b"".join([chunk async for chunk in response.aiter_bytes()])

    async def assert_rejected_before_read(self, method, headers, chunks, *, error=HttpInvalidResponse):
        stream = ChunkStream(chunks)
        client, response, calls = self.make_client(stream, headers=headers)
        entries = self.stream_entries
        with (patch.object(response, "aiter_bytes", side_effect=AssertionError("decoded iterator entered")),
              patch.object(response, "aiter_raw", side_effect=AssertionError("raw iterator entered")),
              self.assertRaises(error)):
            await self.read(client, method)
        self.assertFalse(stream.started)
        self.assertEqual((stream.read_chunks, stream.read_bytes), (0, 0))
        self.assertTrue(stream.closed)
        self.assertTrue(response.is_closed)
        self.assertEqual(len(calls), 1)  # 编码/大小拒绝不触发 GET timeout 重试。
        self.assertEqual(self.stream_entries, entries)  # stream POST 必须在 yield 前拒绝。

    async def test_strict_rejects_encodings_and_multiple_headers_before_all_body_reads(self):
        headers = [[("Content-Encoding", value)] for value in (
            "gzip", "br", "deflate", "compress", "gzip, br", "identity, gzip",
            "identity, identity", "identity;foo=bar",
        )]
        headers += [
            [("Content-Encoding", "identity"), ("Content-Encoding", "gzip")],
            [("Content-Encoding", "identity"), ("Content-Encoding", "identity")],
            [("Content-Encoding", ""), ("Content-Encoding", "")],
        ]
        for method in self.METHODS:
            for header in headers:
                with self.subTest(method=method, header=header):
                    await self.assert_rejected_before_read(method, header, [self.PNG])

    async def test_strict_rejects_gzip_extra_header_and_bomb_without_consuming_chunks(self):
        # 合法 gzip FCOMMENT 超过 3MiB，但解压内容仍是完整 1×1 PNG。
        base = gzip.compress(self.PNG)
        extra = base[:3] + bytes([base[3] | 16]) + base[4:10] + b"x" * (3 * 1024 * 1024) + b"\0" + base[10:]
        self.assertGreater(len(extra), self.LIMIT)
        self.assertEqual(gzip.decompress(extra), self.PNG)
        bomb = gzip.compress(self.PNG + b"x" * (16 * 1024 * 1024 - len(self.PNG)))
        self.assertLess(len(bomb), self.LIMIT)
        for method in self.METHODS:
            for name, wire in (("extra-header", extra), ("16MiB-bomb", bomb)):
                headers = {"Content-Type": "image/png", "Content-Encoding": "gzip"}
                if name == "16MiB-bomb":
                    headers["Content-Length"] = str(len(wire))
                with self.subTest(method=method, name=name):
                    await self.assert_rejected_before_read(method, headers,
                        [wire[i:i + self.CHUNK] for i in range(0, len(wire), self.CHUNK)])

    async def test_strict_accepts_identity_png_at_exact_limit_using_only_raw_chunks(self):
        for method in self.METHODS:
            for encoding in (None, "", "identity", "  IdEnTiTy\t"):
                with self.subTest(method=method, encoding=encoding):
                    stream = ChunkStream([self.PNG[:12], self.PNG[12:]])
                    headers = {"Content-Type": "image/png", "Content-Length": str(len(self.PNG))}
                    if encoding is not None:
                        headers["Content-Encoding"] = encoding
                    client, response, calls = self.make_client(stream, headers=headers, maximum=len(self.PNG))
                    with (patch.object(response, "aiter_raw", wraps=response.aiter_raw) as raw,
                          patch.object(response, "aiter_bytes", side_effect=AssertionError("automatic decoder used"))):
                        self.assertEqual(await self.read(client, method), self.PNG)
                    raw.assert_called_once_with(chunk_size=self.CHUNK)
                    self.assertEqual(len(calls), 1)
                    self.assertTrue(stream.closed)
                    self.assertTrue(response.is_closed)

    async def test_strict_enforces_declared_length_before_entering_body_or_stream_context(self):
        for method in self.METHODS:
            for length, error in ((str(self.LIMIT + 1), HttpResponseTooLarge),
                                  ("-1", HttpInvalidResponse), ("invalid", HttpInvalidResponse),
                                  ("1, 2", HttpInvalidResponse)):
                with self.subTest(method=method, length=length):
                    await self.assert_rejected_before_read(method, {"Content-Length": length}, [self.PNG], error=error)

    async def test_strict_limits_raw_bytes_with_missing_or_understated_content_length(self):
        for method in self.METHODS:
            for headers in ({}, {"Content-Length": "1"}):
                with self.subTest(method=method, headers=headers):
                    count = self.LIMIT // self.CHUNK
                    stream = ChunkStream([b"x" * self.CHUNK] * (count + 1) + [b"must not be read"])
                    client, response, calls = self.make_client(stream, headers=headers)
                    with self.assertRaises(HttpResponseTooLarge):
                        await self.read(client, method)
                    self.assertEqual(stream.read_chunks, count + 1)
                    self.assertEqual(stream.read_bytes, self.LIMIT + self.CHUNK)
                    self.assertTrue(stream.closed)
                    self.assertTrue(response.is_closed)
                    self.assertEqual(len(calls), 1)

    async def test_strict_stream_is_single_use_and_closes_without_body_consumption(self):
        for consume in (False, True):
            with self.subTest(consume=consume):
                stream = ChunkStream([self.PNG])
                client, response, _ = self.make_client(stream)
                async with client.stream_post_json("https://nyaa.si/", json={}) as bounded:
                    if consume:
                        self.assertEqual(b"".join([chunk async for chunk in bounded.aiter_bytes()]), self.PNG)
                        with self.assertRaisesRegex(RuntimeError, "already been consumed"):
                            _ = [chunk async for chunk in bounded.aiter_bytes()]
                self.assertEqual(stream.started, consume)
                self.assertTrue(stream.closed)
                self.assertTrue(response.is_closed)

    async def test_strict_cancellation_closes_response_for_every_entrypoint(self):
        for method in self.METHODS:
            with self.subTest(method=method):
                waiting = asyncio.Event()

                class WaitingStream(ChunkStream):
                    async def __aiter__(self):
                        async for chunk in super().__aiter__():
                            yield chunk
                        waiting.set()
                        await asyncio.Event().wait()

                stream = WaitingStream([b"x" * self.CHUNK])
                client, response, _ = self.make_client(stream)
                task = asyncio.create_task(self.read(client, method))
                try:
                    await asyncio.wait_for(waiting.wait(), 1)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertTrue(stream.closed)
                    self.assertTrue(response.is_closed)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_default_and_explicit_false_preserve_gzip_and_deflate_decoding(self):
        for method in self.METHODS:
            for strict in (None, False):
                for encoding, compress in (("gzip", gzip.compress), ("deflate", zlib.compress)):
                    with self.subTest(method=method, strict=strict, encoding=encoding):
                        wire = compress(self.PNG)
                        stream = ChunkStream([wire[:5], wire[5:]])
                        client, response, _ = self.make_client(stream, strict=strict,
                            headers={"Content-Encoding": encoding, "Content-Length": str(len(wire))})
                        self.assertEqual(await self.read(client, method), self.PNG)
                        self.assertTrue(stream.closed)
                        self.assertTrue(response.is_closed)

    async def test_default_decoded_size_limit_is_preserved(self):
        wire = gzip.compress(b"x" * 1025)
        self.assertLess(len(wire), 1024)
        for method in self.METHODS:
            with self.subTest(method=method):
                stream = ChunkStream([wire])
                client, response, _ = self.make_client(stream, strict=None, maximum=1024,
                    headers={"Content-Encoding": "gzip", "Content-Length": str(len(wire))})
                with self.assertRaises(HttpResponseTooLarge):
                    await self.read(client, method)
                self.assertTrue(stream.closed)
                self.assertTrue(response.is_closed)

    async def test_encoding_option_does_not_change_get_timeout_retry_or_post_no_retry(self):
        for strict in (None, True):
            for method in self.METHODS:
                with self.subTest(strict=strict, method=method):
                    calls = []

                    def timeout(request):
                        calls.append(request)
                        raise httpx.ReadTimeout("offline fixture", request=request)

                    options: dict = {} if strict is None else {"require_identity_encoding": strict}
                    client = FixedHostHttpClient(allowed_hosts={"nyaa.si"}, resolver=PUBLIC_DNS,
                        transport=httpx.MockTransport(timeout), **options)
                    self.addAsyncCleanup(client.aclose)
                    with self.assertRaises(httpx.ReadTimeout):
                        await self.read(client, method)
                    self.assertEqual(len(calls), 2 if method == "get" else 1)


    async def test_strict_prebuffered_content_json_text_remain_compatible_without_decoding(self):
        for method in self.METHODS:
            for kind in ("content", "json", "text"):
                with self.subTest(method=method, kind=kind):
                    value = {"content": self.PNG, "json": {"ok": True}, "text": "离线响应"}[kind]
                    body_options: dict = {kind: value}
                    response = httpx.Response(200, **body_options)
                    expected = response.content
                    self.assertTrue(response.is_stream_consumed)
                    client, _, _ = self.make_client(response=response, maximum=len(expected))
                    with (patch.object(response, "aiter_bytes", side_effect=AssertionError("buffer decoded again")),
                          patch.object(response, "aiter_raw", side_effect=AssertionError("consumed stream reopened"))):
                        self.assertEqual(await self.read(client, method), expected)
                    self.assertTrue(response.is_closed)

    async def test_strict_checks_buffered_length_without_trusting_missing_or_short_content_length(self):
        for method in self.METHODS:
            for declared in (None, "1"):
                with self.subTest(method=method, declared=declared):
                    response = httpx.Response(200, content=self.PNG)
                    response.headers.pop("Content-Length")
                    if declared is not None:
                        response.headers["Content-Length"] = declared
                    client, _, _ = self.make_client(response=response, maximum=len(self.PNG) - 1)
                    with (patch.object(response, "aiter_bytes", side_effect=AssertionError("buffer decoded again")),
                          patch.object(response, "aiter_raw", side_effect=AssertionError("consumed stream reopened")),
                          self.assertRaises(HttpResponseTooLarge)):
                        await self.read(client, method)
                    self.assertTrue(response.is_closed)

    async def test_strict_rejects_buffered_encoding_before_accessing_content(self):
        for method in self.METHODS:
            with self.subTest(method=method):
                response = httpx.Response(200, content=self.PNG)
                # 模拟显式预读 transport；编码仍须拒绝，不将已有 buffer 重新解码。
                response.headers["Content-Encoding"] = "gzip"
                client, _, calls = self.make_client(response=response)
                entries = self.stream_entries
                with (patch.object(httpx.Response, "content", new_callable=PropertyMock,
                                   side_effect=AssertionError("buffer accessed before encoding validation")),
                      self.assertRaisesRegex(HttpInvalidResponse, "Content-Encoding")):
                    await self.read(client, method)
                self.assertTrue(response.is_closed)
                self.assertEqual(self.stream_entries, entries)
                self.assertEqual(len(calls), 1)

    async def test_strict_consumed_transport_without_a_buffer_fails_closed(self):
        for method in self.METHODS:
            with self.subTest(method=method):
                stream = ChunkStream([self.PNG])
                response = httpx.Response(200, stream=stream)
                _ = [chunk async for chunk in response.aiter_raw()]
                self.assertTrue(response.is_stream_consumed)
                client, _, _ = self.make_client(response=response)
                with self.assertRaisesRegex(HttpInvalidResponse, "no buffered body"):
                    await self.read(client, method)
                self.assertEqual(stream.read_chunks, 1)  # 不重读已被 transport 消费的流。
                self.assertTrue(stream.closed)
                self.assertTrue(response.is_closed)


class BuiltinIndexerAddressTests(unittest.TestCase):
    def test_builtin_hosts_are_encoded_in_runtime_sources(self):
        from pathlib import Path
        from urllib.parse import urlsplit

        from app.indexers import config as endpoints
        from app.indexers.providers.kpkuang import KPKUANG_HOST_CONFIG

        urls = [value for name, value in vars(endpoints).items() if name.endswith('_URL')]
        urls.extend(KPKUANG_HOST_CONFIG[key] for key in ('base_url', 'api_search_url'))
        self.assertGreaterEqual(len(urls), 15)
        hosts = {urlsplit(url).hostname for url in urls}
        self.assertTrue(all(url.startswith('https://') for url in urls))
        self.assertNotIn(None, hosts)
        root = Path(__file__).resolve().parents[1] / 'app'
        for path in root.rglob('*'):
            if path.suffix not in {'.py', '.js', '.html'}:
                continue
            source = path.read_text(encoding='utf-8')
            for host in hosts:
                self.assertNotIn(host, source, f'{path.relative_to(root)} contains a plaintext indexer host')
