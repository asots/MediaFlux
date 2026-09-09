"""平台原图：独立签名/鉴权/收藏身份、固定公网 HTTPS 与有界图片校验。"""
from __future__ import annotations

import asyncio
import copy
import socket
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

import tests  # noqa: F401 -- 导入项目代码前先隔离生产配置和 DB。
from app.discovery.calendar.models import CalendarEntry, CalendarEvent
from app.discovery.calendar.poster_http import MAX_PLATFORM_IMAGE_BYTES, fetch_platform_poster
from app.discovery.calendar.posters import canonical_platform_poster_key, platform_poster_key
from app.indexers.errors import IndexerSecurityError
from app.modules.image_payload import ImagePayloadError
from app.routes.discovery_image import decode_poster_token, encode_poster_token
from tests.test_discovery_api import _BaseClientTests
from tests.test_free_calendar_api import SNAPSHOT

_HOST = "pic0.iqiyipic.com"
_KEY = _HOST + "/image/20250416/14/a8/a_100570241_m_601_m4.webp"
_IMAGE = b"RIFF\x08\x00\x00\x00WEBPtest"


def _resolver(host, port):
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", port))]


class PlatformPosterKeyTests(unittest.TestCase):
    def test_known_asset_normalization_never_forwards_scheme_credentials_or_queries(self):
        for prefix in ("https://", "http://", "//"):
            with self.subTest(prefix=prefix):
                self.assertEqual(platform_poster_key("iqiyi", prefix + _KEY), _KEY)
        self.assertEqual(platform_poster_key("iqiyi", "https://" + _KEY + "?quality=90"), _KEY)
        self.assertEqual(canonical_platform_poster_key("iqiyi", _KEY), _KEY)

    def test_exact_three_platform_hosts_and_asset_namespaces_do_not_expand_by_suffix(self):
        assets = {
            "tencent": "vcover-hz-pic.puui.qpic.cn/vcover_hz_pic/0/mzc0020082u0tna1788947534562/750",
            "iqiyi": _KEY,
            "youku": "liangcang-material.alicdn.com/prod/upload/981696b79f914f5497f78fb8b9c8d14e.webp.jpg",
        }
        for source, key in assets.items():
            with self.subTest(source=source):
                self.assertEqual(platform_poster_key(source, "http://" + key), key)
                for other in assets.keys() - {source}:
                    self.assertEqual(canonical_platform_poster_key(other, key), "")
        key = "m.ykimg.com/05810000677F92A513FAB41363CE6B02"
        self.assertEqual(canonical_platform_poster_key("youku", key), key)
        for source, key in (
            ("iqiyi", _KEY.replace("pic0.", "pic10.")),
            ("tencent", assets["tencent"].replace("vcover-hz-pic.puui.", "puui.")),
            ("youku", key.replace("m.ykimg.com", "liangcang-material.alicdn.com")),
            ("youku", assets["youku"].replace("liangcang-material.alicdn.com", "m.ykimg.com")),
        ):
            with self.subTest(source=source, key=key):
                self.assertEqual(canonical_platform_poster_key(source, key), "")
        for source in (None, [], {}, "evil"):
            self.assertEqual(platform_poster_key(source, "https://" + _KEY), "")
            self.assertEqual(canonical_platform_poster_key(source, _KEY), "")

    def test_untrusted_image_values_fail_closed(self):
        for value in (None, [], {}, 123, "", "https://127.0.0.1/image/x.webp", "https://evil.invalid/image/x.webp",
                      "https://" + _HOST + ".evil.invalid/image/x.webp", "https://" + _HOST + ":443/image/x.webp",
                      "https://" + _HOST + "/redirect/test.webp", "https://u:p@" + _KEY, "https://" + _KEY + "#x", "https://" + _HOST + "/../image/x.webp",
                      "https://" + _HOST + "/%2e%2e/image/x.webp", "https://" + _HOST + "/image//x.webp",
                      "https://" + _KEY + "\n", "https://" + _HOST + "/image\\x.webp",
                      "data:image/png,x", "javascript:alert(1)", "https://" + _HOST + "/image/" + "x" * 2100):
            with self.subTest(value=str(value)[:80]):
                self.assertEqual(platform_poster_key("iqiyi", value), "")
        self.assertEqual(platform_poster_key("tencent", "https://" + _KEY), "")
        for value in ("/" + _KEY, _KEY + "?x=y", "https://" + _KEY, _KEY + "#x", " " + _KEY):
            with self.subTest(key=value):
                self.assertEqual(canonical_platform_poster_key("iqiyi", value), "")

    def test_bad_optional_image_does_not_drop_programme_facts(self):
        for key in (_KEY, "evil.invalid/image/x.webp", None, {"url": _KEY}):
            with self.subTest(key=key):
                entry = CalendarEntry("iqiyi", "123", "未匹配的动漫", "animation", "https://www.iqiyi.com/a_123.html",
                                      events=(CalendarEvent("2026-09-09", "10:00", "每周三更新"),),
                                      evidence="官方日期排期", platform_poster_key=key)
                self.assertEqual(entry.platform_poster_key, _KEY if key == _KEY else "")
                self.assertEqual(entry.stable_id, "iqiyi:123")
                self.assertEqual(len(entry.events), 1)

    def test_platform_token_is_not_a_metadata_or_other_source_token(self):
        token = encode_poster_token("calendar-iqiyi", _KEY)
        self.assertEqual(decode_poster_token("calendar-iqiyi", token), _KEY)
        from fastapi import HTTPException
        for provider in ("tmdb", "douban", "calendar-tencent"):
            with self.subTest(provider=provider), self.assertRaises(HTTPException):
                decode_poster_token(provider, token)


class PlatformPosterHTTPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = []
        self.attempts = []

        def forbidden(*args, **kwargs):
            self.attempts.append(True)
            raise AssertionError("回归测试禁止真实网络/DNS")

        for name in ("getaddrinfo", "create_connection", "socket.connect", "socket.connect_ex"):
            self.enterContext(patch("socket." + name, side_effect=forbidden))

    def tearDown(self):
        self.assertEqual(self.attempts, [])

    async def fetch(self, response, *, resolver=_resolver):
        def handler(request):
            self.calls.append(request)
            if isinstance(response, BaseException):
                raise response
            return response
        return await fetch_platform_poster("iqiyi", _KEY, transport=httpx.MockTransport(handler), resolver=resolver)

    async def test_cancelled_dns_retains_shared_capacity_until_real_return_and_recovers(self):
        from app.discovery.calendar import http as module
        entered, release, returned = (threading.Event() for _ in range(3))

        class Slots(threading.BoundedSemaphore):
            def release(self, n=1):
                super().release(n)
                returned.set()

        def blocked(host, port):
            entered.set()
            if not release.wait(5):
                raise AssertionError("测试 resolver 必须释放")
            return _resolver(host, port)

        fast = Mock(side_effect=_resolver)
        transport = httpx.MockTransport(lambda request: httpx.Response(
            200, content=_IMAGE, headers={"Content-Type": "image/webp"},
        ))
        with patch.object(module, "_DNS_SLOTS", Slots(1)):
            public = module.CalendarHttp({"v.qq.com"}, resolver=fast,
                                         transport=transport, min_interval=0)
            task = asyncio.create_task(fetch_platform_poster(
                "iqiyi", _KEY, resolver=blocked, transport=transport,
            ))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(returned.is_set())
                with self.assertRaises(IndexerSecurityError):
                    await fetch_platform_poster("iqiyi", _KEY, resolver=fast, transport=transport)
                # 海报不能另建自己的上限；与公开排期共用进程级容量。
                with self.assertRaises(IndexerSecurityError):
                    await public.get_json("https://v.qq.com/public")
                fast.assert_not_called()
                release.set()
                self.assertTrue(await asyncio.to_thread(returned.wait, 2))
                self.assertEqual(await fetch_platform_poster(
                    "iqiyi", _KEY, resolver=fast, transport=transport,
                ), (_IMAGE, "image/webp"))
                fast.assert_called_once_with(_HOST, 443)
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                await public.aclose()

    async def test_single_get_pins_public_address_with_original_host_sni_and_no_credentials(self):
        body, mime = await self.fetch(httpx.Response(200, content=_IMAGE, headers={"Content-Type": "image/webp"}))
        self.assertEqual((body, mime), (_IMAGE, "image/webp"))
        self.assertEqual(len(self.calls), 1)
        request = self.calls[0]
        self.assertEqual(str(request.url), "https://93.184.216.34/" + _KEY.split("/", 1)[1])
        self.assertEqual(request.method, "GET")
        self.assertEqual(request.headers["Host"], _HOST)
        self.assertEqual(request.extensions["sni_hostname"], _HOST)
        self.assertEqual(request.headers["Accept-Encoding"], "identity")
        self.assertEqual(request.headers["User-Agent"], "Mozilla/5.0 (compatible; MediaFluxCalendar/1.0)")
        for header in ("Cookie", "Authorization", "Proxy-Authorization", "Referer"):
            self.assertNotIn(header, request.headers)

    async def test_private_mixed_dns_and_redirects_are_rejected_without_followup(self):
        for address in ("127.0.0.1", "10.0.0.8", "::1", "169.254.169.254"):
            with self.subTest(address=address), self.assertRaises(IndexerSecurityError):
                await self.fetch(httpx.Response(200, content=_IMAGE), resolver=lambda h, p: [
                    (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, p)),
                    *_resolver(h, p),
                ])
        self.assertEqual(self.calls, [])
        with self.assertRaises(IndexerSecurityError):
            await self.fetch(httpx.Response(302, headers={"Location": "https://127.0.0.1/private"}))
        self.assertEqual(len(self.calls), 1)

    async def test_failures_never_retry_and_response_is_closed(self):
        for status in (403, 429, 500, 502, 503):
            response = httpx.Response(status, content=b"challenge")
            with self.subTest(status=status), self.assertRaises(ImagePayloadError):
                await self.fetch(response)
            self.assertTrue(response.is_closed)
        self.assertEqual(len(self.calls), 5)
        with self.assertRaises(httpx.ReadTimeout):
            await self.fetch(httpx.ReadTimeout("offline timeout"))
        self.assertEqual(len(self.calls), 6)

    async def test_compressed_upstream_is_rejected_before_any_body_or_decoder_work(self):
        from app.indexers.errors import IndexerInvalidResponse
        class Wire(httpx.AsyncByteStream):
            started = False
            closed = False
            async def __aiter__(self):
                self.started = True
                yield b"must not reach an automatic decoder"
            async def aclose(self):
                self.closed = True

        for encoding in ("gzip", "br", "deflate", "identity, gzip"):
            wire = Wire()
            with self.subTest(encoding=encoding), self.assertRaises(IndexerInvalidResponse):
                await self.fetch(httpx.Response(200, stream=wire, headers={
                    "Content-Type": "image/png", "Content-Encoding": encoding,
                }))
            self.assertFalse(wire.started)
            self.assertTrue(wire.closed)

    async def test_mime_magic_length_and_stream_limits(self):
        from app.indexers.errors import IndexerResponseTooLarge
        cases = [
            ({"Content-Type": "text/html"}, b"<html>challenge</html>", ImagePayloadError),
            ({"Content-Type": "image/svg+xml"}, b"<svg></svg>", ImagePayloadError),
            ({"Content-Type": "image/webp"}, b"<html>challenge</html>", ImagePayloadError),
            ({"Content-Type": "image/png"}, _IMAGE, ImagePayloadError),
            ({"Content-Type": "image/webp", "Content-Length": str(MAX_PLATFORM_IMAGE_BYTES + 1)}, _IMAGE, IndexerResponseTooLarge),
            ({"Content-Type": "image/webp"}, _IMAGE + b"x" * MAX_PLATFORM_IMAGE_BYTES, IndexerResponseTooLarge),
        ]
        for headers, body, error in cases:
            response = httpx.Response(200, headers=headers, content=body)
            with self.subTest(headers=headers, size=len(body)), self.assertRaises(error):
                await self.fetch(response)
            self.assertTrue(response.is_closed)
        self.assertEqual(len(self.calls), len(cases))


class PlatformPosterAPITests(_BaseClientTests):
    def setUp(self):
        super().setUp()
        from app import database
        database.init_db()
        self.image = self.enterContext(patch("app.routes.discovery_image.fetch_platform_poster", new=AsyncMock(return_value=(_IMAGE, "image/webp"))))
        self.token = encode_poster_token("calendar-iqiyi", _KEY)
        self.path = "/discovery-calendar-poster/iqiyi/" + self.token

    def test_platform_proxy_auth_feature_flag_and_safe_headers(self):
        self.assertEqual(self.client.get(self.path).status_code, 401)
        self.image.assert_not_called()
        self.authenticate()
        response = self.client.get(self.path)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, _IMAGE)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["Cache-Control"], "private, max-age=86400")
        self.image.assert_awaited_once_with("iqiyi", _KEY)
        self.image.reset_mock()
        with patch("app.routes.discovery_image.config.get_bool", return_value=False):
            self.assertEqual(self.client.get(self.path).status_code, 404)
        self.image.assert_not_called()

    def test_invalid_cross_provider_and_generic_paths_cannot_use_unpinned_proxy(self):
        self.authenticate()
        for path in (self.path + "bad", self.path + "?url=https://evil.invalid/", "/discovery-calendar-poster/tencent/" + self.token,
                     "/discovery-calendar-poster/evil/" + self.token,
                     "/discovery-poster/calendar-iqiyi/" + self.token):
            with self.subTest(path=path):
                self.assertIn(self.client.get(path).status_code, {400, 404})
        self.image.assert_not_called()

    def test_proxy_failure_response_does_not_leak_remote_url_or_error_details(self):
        self.authenticate()
        self.image.side_effect = ImagePayloadError("secret https://evil.invalid/private")
        response = self.client.get(self.path)
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret", response.text)
        self.assertNotIn("evil.invalid", response.text)

    def snapshot(self, **changes):
        snapshot = copy.deepcopy(SNAPSHOT)
        raw = snapshot["days"][0]["items"][0]
        raw.update(source="iqiyi", source_id="123", stable_id="iqiyi:123", platform_poster_key=_KEY)
        raw.update(changes)
        service = SimpleNamespace(get_week=lambda: snapshot)
        with patch("app.routes.discovery_api.get_calendar_service", return_value=service):
            response = self.client.get("/api/discovery/calendar")
        self.assertEqual(response.status_code, 200)
        return response.json()["days"][0]["items"][0]

    def test_unmatched_card_gets_original_poster_but_no_fake_collection_or_detail_identity(self):
        self.authenticate()
        card = self.snapshot(tmdb_id="", douban_id="", poster_key="", mapping_status="unmatched")
        self.assertEqual(card["poster_urls"], [self.path])
        self.assertEqual(card["poster_url"], self.path)
        self.assertEqual(card["poster_provider"], "calendar-iqiyi")
        self.assertEqual(card["mapping_status"], "unmatched")
        self.assertIsNone(card["watchlist"])
        self.assertEqual(card["detail_url"], "")
        self.assertNotIn("platform_poster_key", card)

    def test_original_poster_is_last_fallback_and_never_a_watchlist_poster_token(self):
        headers = self.authenticate()
        card = self.snapshot(douban_id="77", douban_poster_key="img1.doubanio.com/view/photo/test.jpg")
        self.assertEqual(len(card["poster_urls"]), 3)
        self.assertTrue(card["poster_urls"][0].startswith("/discovery-poster/tmdb/"))
        self.assertTrue(card["poster_urls"][1].startswith("/discovery-poster/douban/"))
        self.assertEqual(card["poster_urls"][-1], self.path)
        self.assertEqual(card["watchlist"]["provider"], "tmdb")
        self.assertEqual(decode_poster_token("tmdb", card["watchlist"]["poster_token"]), "poster.jpg")
        image_only = self.snapshot(tmdb_id="42", poster_key="")
        self.assertEqual(image_only["poster_url"], self.path)
        self.assertEqual(image_only["watchlist"]["poster_token"], "")
        for provider in ("tmdb", "calendar-iqiyi"):
            response = self.client.post("/api/discovery/watchlist", headers=headers, json={
                "provider": provider, "media_type": "tv", "external_id": "42", "title": "测试动漫",
                "poster_token": self.token,
            })
            self.assertEqual(response.status_code, 400)

    def test_invalid_optional_key_or_cross_source_key_is_discarded_not_signed(self):
        self.authenticate()
        for changes in ({"platform_poster_key": "evil.invalid/image/x.webp"}, {"source": "tencent"},
                        {"platform_poster_key": "https://" + _KEY}, {"platform_poster_key": [_KEY]}):
            with self.subTest(changes=changes):
                card = self.snapshot(**changes)
                self.assertEqual(len(card["poster_urls"]), 1)
                self.assertNotIn("platform_poster_key", card)
