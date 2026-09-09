"""优酷正常匿名协议全离线回归：真实CalendarHttp、MockTransport、假DNS、禁止socket。"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import socket
import traceback
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import tests  # noqa: F401 -- 应用导入前隔离生产配置/DB。

# isort: split
import httpx

from app.discovery.calendar import youku_http
from app.discovery.calendar.http import CALENDAR_USER_AGENT, CalendarHttp
from app.discovery.calendar.models import SourceUnavailable
from app.discovery.calendar.youku_http import fetch_youku_calendar

_URL = "https://acs.youku.com/h5/mtop.youku.columbus.home.query/1.0/"
_API = "mtop.youku.columbus.home.query"
_PATH = "/h5/mtop.youku.columbus.home.query/1.0/"
_NOW = datetime(2026, 9, 10, tzinfo=timezone.utc).timestamp()
_TK = "offlinefixture_1999999999999"
_ENC = "offlinefixturecompanion"
_GRANT = f"_m_h5_tk={_TK}; Domain=youku.com; Path=/, _m_h5_tk_enc={_ENC}; Domain=.youku.com; Path=/"
_FUTURE = "Wed, 10 Sep 2036 00:00:00 GMT"
_PAST = "Wed, 10 Sep 2025 00:00:00 GMT"


def _success(label="offline"):
    # 只测试transport的MTOP envelope，不用本例冒充真实排期/日期fixture。
    return {"api": _API, "v": "1.0", "ret": ["SUCCESS::调用成功"],
            "data": {"2019061000": {"label": label}}}


def _token(code="FAIL_SYS_TOKEN_EMPTY"):
    return {"api": _API, "v": "1.0", "ret": [code + "::offline"], "data": {}}


def _response(payload, *, cookie=None, status=200, mime="application/json;charset=UTF-8"):
    headers = [("Content-Type", mime)]
    if cookie is not None:
        headers.extend(("Set-Cookie", value) for value in (cookie if isinstance(cookie, list) else [cookie]))
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode()
    return httpx.Response(status, headers=headers, stream=httpx.ByteStream(body))


class RecordingHttp:
    """仅观察被测函数持有的引用；I/O仍原样委托真实预算factory。"""
    def __init__(self, http):
        self.http = http
        self.references = []
        self.snapshots = []
        self.jar_sizes = []

    async def get_response(self, url, *, params, headers):
        self.references.append((params, headers))
        self.snapshots.append((url, params.copy(), headers.copy()))
        result = await self.http.get_response(url, params=params, headers=headers)
        self.jar_sizes.append(len(self.http._client._client.cookies))
        return result


class YoukuCalendarHttpTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection", "socket.getaddrinfo"):
            guard = patch(target, side_effect=AssertionError("real network forbidden"))
            guard.start()
            self.addCleanup(guard.stop)
        self.mock_clock = Mock(return_value=_NOW)
        # 只替换被测模块的clock，不影响HTTPX CookieJar内部读取标准time模块。
        self.clock = patch.object(youku_http, "time", SimpleNamespace(time=self.mock_clock))
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def client(self, handler, *, max_requests=8, hosts=None, resolver=None, min_interval=0):
        calls = []

        async def handle(request):
            calls.append(request)
            response = handler(request)
            return await response if inspect.isawaitable(response) else response

        def fake_resolver(host, port):
            self.assertIn(host, {"acs.youku.com", "www.youku.com"})
            self.assertEqual(port, 443)
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 443))]

        http = CalendarHttp(
            hosts or {"acs.youku.com"}, transport=httpx.MockTransport(handle),
            resolver=resolver or fake_resolver, max_requests=max_requests, min_interval=min_interval,
        )
        self.addAsyncCleanup(http.aclose)
        return http, calls

    async def unavailable(self, http):
        with self.assertRaises(SourceUnavailable) as caught:
            await fetch_youku_calendar(http)
        error = caught.exception
        self.assertEqual(str(error), "优酷公开排期暂不可用，未继续请求")
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)
        self.assertNotIn(_TK, "".join(traceback.format_exception(error)))
        return error

    def assert_clean(self, recording):
        for params, headers in recording.references:
            self.assertEqual(params, {})
            self.assertEqual(headers, {})
        self.assertEqual(len(recording.http._client._client.cookies), 0)

    def assert_request(self, request, token_prefix):
        self.assertEqual(request.method, "GET")
        self.assertEqual(request.url.scheme, "https")
        self.assertEqual(request.url.host, "8.8.8.8")
        self.assertEqual(request.url.path, _PATH)
        self.assertEqual(request.headers["host"], "acs.youku.com")
        self.assertEqual(request.extensions["sni_hostname"], "acs.youku.com")
        self.assertEqual(request.headers["accept"], "application/json")
        self.assertEqual(request.headers["content-type"], "application/x-www-form-urlencoded")
        self.assertEqual(request.headers["origin"], "https://www.youku.com")
        self.assertEqual(request.headers["referer"], "https://www.youku.com/ku/webcomic")
        self.assertEqual(request.headers["accept-encoding"], "identity")
        self.assertEqual(request.headers["user-agent"], CALENDAR_USER_AGENT)
        self.assertNotIn("authorization", request.headers)
        query = request.url.params
        self.assertEqual(set(query), {"jsv", "appKey", "t", "sign", "api", "v", "dataType", "type", "jsonpIncPrefix", "data"})
        self.assertEqual((query["jsv"], query["appKey"], query["api"], query["v"]), ("2.7.4", "24679788", _API, "1.0"))
        self.assertEqual((query["dataType"], query["type"]), ("json", "originaljson"))
        self.assertEqual(query["jsonpIncPrefix"], query["t"])
        self.assertEqual(query["sign"], hashlib.md5(
            (token_prefix + "&" + query["t"] + "&24679788&" + query["data"]).encode("ascii"),
            usedforsecurity=False,
        ).hexdigest())
        self.assertEqual(request.content, b"")

    async def test_cold_success_is_one_get_with_official_ascii_data_and_no_cookies(self):
        http, calls = self.client(lambda request: _response(_success(), cookie="account=ignored; Path=/"))
        http._client._client.cookies.set("account", "never-send", domain="8.8.8.8")
        recording = RecordingHttp(http)
        self.assertEqual(await fetch_youku_calendar(recording), _success())
        self.assertEqual(len(calls), 1)
        self.assert_request(calls[0], "undefined")
        self.assertNotIn("cookie", calls[0].headers)
        raw = calls[0].url.params["data"]
        data = json.loads(raw)
        self.assertTrue(raw.isascii())
        self.assertEqual(raw, json.dumps(data, ensure_ascii=True, separators=(",", ":")))
        self.assertEqual(set(data), {"ms_codes", "params", "system_info"})
        self.assertEqual(data["ms_codes"], "2019061000")
        self.assertIsInstance(data["params"], str)
        self.assertIsInstance(data["system_info"], str)
        params = json.loads(data["params"])
        system = json.loads(data["system_info"])
        self.assertEqual(params, {
            "debug": 0, "utdid": "empty_cna", "appPackageKey": "com.youku.pcweb", "appPackageId": "com.youku.pcweb",
            "ip": "127.0.0.1", "reqSubNode": 0, "gray": "0", "pageNo": 1, "bizKey": "kuflix_pc_home",
            "showNodeList": 0, "nodeKey": "WEBCOMIC", "appKey": "24679788", "bizContext": "{}",
        })
        self.assertEqual(system["userAgent"], CALENDAR_USER_AGENT)
        self.assertEqual(system["guid"], "1590141704165YXe")
        self.assertEqual(system["disableUserRec"], "0")
        self.assertFalse({"session", "userId", "cna", "CNA", "sid", "needLogin"} & (set(params) | set(system)))
        self.assert_clean(recording)

    async def test_logical_origin_grant_works_despite_empty_pinned_ip_cookie_jar(self):
        def handler(request):
            return _response(_success()) if "cookie" in request.headers else _response(_token(), cookie=_GRANT)
        http, calls = self.client(handler)
        recording = RecordingHttp(http)
        self.assertEqual(await fetch_youku_calendar(recording), _success())
        self.assertEqual(recording.jar_sizes, [0, 0])
        self.assertEqual(len(calls), 2)
        self.assert_request(calls[0], "undefined")
        self.assert_request(calls[1], "offlinefixture")
        self.assertEqual(calls[1].headers["cookie"], f"_m_h5_tk={_TK}; _m_h5_tk_enc={_ENC}")
        self.assertNotEqual(calls[0].url.params["sign"], calls[1].url.params["sign"])
        self.assertEqual(http._requests, 2)
        self.assert_clean(recording)

    async def test_token_exoired_uses_one_followup_with_new_timestamp(self):
        self.mock_clock.side_effect = [_NOW, _NOW + 1, _NOW + 2]
        http, calls = self.client(lambda request: _response(_success()) if "cookie" in request.headers else _response(_token("FAIL_SYS_TOKEN_EXOIRED"), cookie=_GRANT))
        await fetch_youku_calendar(http)
        self.assertEqual(len(calls), 2)
        self.assertEqual(int(calls[1].url.params["t"]) - int(calls[0].url.params["t"]), 2000)
        self.assert_request(calls[1], "offlinefixture")

    async def test_valid_cookie_scopes_expires_comma_flags_and_optional_companion(self):
        valid = [
            f"_m_h5_tk={_TK}",
            f"_m_h5_tk={_TK}; Domain=ACS.YOUKU.COM; Path=/h5; HttpOnly; Secure; SameSite=Lax",
            f"_m_h5_tk={_TK}; Domain=.acs.youku.com; Path={_PATH}",
            f"_m_h5_tk={_TK}; Path={_PATH[:-1]}",
            f'_m_h5_tk="{_TK}"; Domain=.youku.com; Path=/',
            f"_m_h5_tk={_TK}; Expires={_FUTURE}; Domain=youku.com; Path=/, _m_h5_tk_enc={_ENC}; Path=/; Expires={_FUTURE}",
            [f"_m_h5_tk_enc={_ENC}; Path=/", f"_m_h5_tk={_TK}; Max-Age=60; Path=/"],
            f"_m_h5_tk={_TK}; Max-Age=60; Expires={_PAST}; Path=/",
            f"account=discard; Domain=youku.com; Path=/, x5sec=discard; Path=/, {_GRANT}",
        ]
        for cookie in valid:
            with self.subTest(cookie=cookie):
                http, calls = self.client(lambda request, cookie=cookie: _response(_success()) if "cookie" in request.headers else _response(_token(), cookie=cookie))
                await fetch_youku_calendar(http)
                self.assertEqual(len(calls), 2)
                self.assert_request(calls[1], "offlinefixture")
                self.assertTrue(calls[1].headers["cookie"].startswith("_m_h5_tk="))
                self.assertNotIn("discard", calls[1].headers["cookie"])

    async def test_cookie_grant_rejects_unsafe_scope_value_expiry_and_conflicts(self):
        invalid = [
            "", "account=only", f"_m_h5_tk_enc={_ENC}; Path=/",
            *[f"_m_h5_tk={_TK}; Domain={domain}; Path=/" for domain in ("evil.invalid", "youku.com.evil.invalid", "www.youku.com", "com", "8.8.8.8", "..youku.com", "youku.com.")],
            *[f"_m_h5_tk={_TK}; Path={path}" for path in ("/h", "/h5x", "/h5/other/", "/h5/../", "relative", "/h5/%2e%2e/")],
            *[f"_m_h5_tk={_TK}; Max-Age={age}; Path=/" for age in ("0", "-1", "+10", "1e3", "1.5", "2147483648", "999999999999999999999")],
            *[f"_m_h5_tk={_TK}; Expires={expiry}; Path=/" for expiry in (_PAST, "not-a-date", "Wed, 10 Sep 2036 00:00:00")],
            f"_m_h5_tk={_TK}; Path=/; Domain=youku.com; domain=youku.com",
            f"_m_h5_tk={_TK}; Path=/; Max-Age=60; max-age=0",
            f"_m_h5_tk={_TK}; Path=/; other=injected",
            f"_m_h5_tk={_TK}; Path=/; Secure=false",
            f"_m_h5_tk={_TK}; Path=/; SameSite=unknown",
            *[f"_m_h5_tk={_TK}; {attribute}=" for attribute in ("Domain", "Path", "Expires", "Max-Age", "SameSite")],
            f"_m_h5_tk ={_TK}; Path=/",
            f"_m_h5_tk={_TK}; Path=/, _m_h5_tk=other_999; Path=/",
            f"_m_h5_tk={_TK}; Path=/, _m_h5_tk={_TK}; Domain=youku.com; Path=/",
            f"{_GRANT}, _m_h5_tk_enc=conflict; Path=/",
            f"{_GRANT}, _m_h5_tk_enc={_ENC}; Domain=evil.invalid; Path=/",
            '_m_h5_tk="space value_999"; Path=/',
            '_m_h5_tk="comma,value_999"; Path=/',
            '_m_h5_tk="escaped\\value_999"; Path=/',
            "_m_h5_tk=_999; Path=/",
            "_m_h5_tk=" + "a" * 1024 + "_9; Path=/",
            f"_m_h5_tk={_TK}; Path=/\r\nCookie: stolen=1",
            f"_m_h5_tk={_TK}; Path=/\t", f"_m_h5_tk={_TK}; Path=/\x00", f"_m_h5_tk={_TK}; Path=/\x7f",
            ", ".join(["unrelated=x"] * 33 + [_GRANT]), "unrelated=" + "x" * 16385,
        ]
        for cookie in invalid:
            with self.subTest(cookie=cookie[:140]):
                http, calls = self.client(lambda request, cookie=cookie: _response(_token(), cookie=cookie))
                recording = RecordingHttp(http)
                await self.unavailable(recording)
                self.assertEqual(len(calls), 1)
                self.assert_clean(recording)

    async def test_second_token_failure_never_obtains_a_third_request(self):
        http, calls = self.client(lambda request: _response(_token(), cookie=_GRANT))
        recording = RecordingHttp(http)
        await self.unavailable(recording)
        self.assertEqual(len(calls), 2)
        self.assert_clean(recording)

    async def test_grant_that_expires_before_followup_is_not_sent(self):
        self.mock_clock.side_effect = [_NOW, _NOW, _NOW + 2]
        http, calls = self.client(lambda request: _response(_token(), cookie=f"_m_h5_tk={_TK}; Max-Age=1; Path=/"))
        await self.unavailable(http)
        self.assertEqual(len(calls), 1)

    async def test_denied_unknown_login_mixed_or_missing_ret_never_bootstraps(self):
        ret_values = [
            ["FAIL_SYS_ILLEGAL_ACCESS::denied"], ["FAIL_SYS_TOKEN_EXPIRED::not-SDK-spelling"],
            ["FAIL_SYS_SESSION_EXPIRED"], ["FAIL_SYS_SID_INVALID"], ["FAIL_SYS_AUTH_REJECT"],
            ["FAIL_SYS_NEED_LOGIN"], ["FAIL_SYS_USER_VALIDATE"], ["RGV587_ERROR"], ["CHECKJS_FLAG"],
            ["FAIL_SYS_SIGN_ERROR"], ["UNKNOWN::SUCCESS"], ["SUCCESS", "FAIL_SYS_TOKEN_EMPTY"],
            ["FAIL_SYS_TOKEN_EMPTY", "SUCCESS"], ["SUCCESS", "SUCCESS"], [], [None], "SUCCESS", None,
            ["FAIL_SYS_TOKEN_EMPTY", "FAIL_SYS_TOKEN_EMPTY"], ["FAIL_SYS_TOKEN_EMPTY" + "x" * 513],
        ]
        for ret in ret_values:
            with self.subTest(ret=ret):
                http, calls = self.client(lambda request, ret=ret: _response({"ret": ret, "data": {}}, cookie=_GRANT))
                await self.unavailable(http)
                self.assertEqual(len(calls), 1)

    async def test_token_ret_with_challenge_data_or_wrong_api_does_not_retry(self):
        payloads = [
            {**_token(), "data": {"url": "https://verify.invalid/"}},
            {**_token(), "api": "mtop.other.query"}, {**_success(), "v": "2.0"},
            {**_success(), "data": []}, {"ret": ["SUCCESS"]}, [], None,
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                http, calls = self.client(lambda request, payload=payload: _response(payload, cookie=_GRANT))
                await self.unavailable(http)
                self.assertEqual(len(calls), 1)

    async def test_http_errors_and_redirects_stop_with_no_retries_or_cross_origin(self):
        for status in (301, 302, 307, 308, 400, 401, 403, 429, 500, 503):
            with self.subTest(status=status):
                http, calls = self.client(lambda request, status=status: httpx.Response(status, headers={"Location": "https://www.youku.com/ku/webcomic", "Set-Cookie": _GRANT}, stream=httpx.ByteStream(b"")), hosts={"acs.youku.com", "www.youku.com"})
                await self.unavailable(http)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].headers["host"], "acs.youku.com")

    async def test_html_jsonp_invalid_json_and_duplicate_keys_stop(self):
        invalid = [
            b'<!doctype html><html>captcha verify you are human</html>',
            b'mtopjsonp1({"ret":["SUCCESS"],"data":{}});', b'{', b'\xff',
            b'{"ret":["SUCCESS"],"ret":["FAIL_SYS_TOKEN_EMPTY"],"data":{}}',
            b'{"ret":["SUCCESS"],"data":{"x":NaN}}', b'{"ret":["SUCCESS"],"data":{"x":Infinity}}',
            b'{"ret":["SUCCESS"],"data":{"x":1e999}}', b'{"ret":["SUCCESS"],"data":{}} trailing',
        ]
        for body in invalid:
            with self.subTest(body=body[:80]):
                http, calls = self.client(lambda request, body=body: httpx.Response(200, headers={"Content-Type": "application/json", "Set-Cookie": _GRANT}, stream=httpx.ByteStream(body)))
                await self.unavailable(http)
                self.assertEqual(len(calls), 1)
        for mime in ("text/html", "application/javascript", "application/octet-stream", ""):
            with self.subTest(mime=mime):
                http, calls = self.client(lambda request, mime=mime: _response(_success(), mime=mime))
                await self.unavailable(http)
                self.assertEqual(len(calls), 1)

    async def test_transport_timeout_failure_and_dns_do_not_retry_or_expose_exception(self):
        for exception in (httpx.ReadTimeout, httpx.ConnectError, RuntimeError):
            with self.subTest(exception=exception):
                def failed(request, error_type=exception):
                    raise error_type(f"{_URL}?sign=private-sign cookie={_TK}")
                http, calls = self.client(failed)
                error = await self.unavailable(http)
                self.assertNotIn("private-sign", str(error))
                self.assertEqual(len(calls), 1)
        http, calls = self.client(lambda request: self.fail("non-public DNS must not connect"), resolver=lambda h, p: [(2, 1, 6, "", ("127.0.0.1", 443))])
        await self.unavailable(http)
        self.assertEqual(calls, [])
        self.assertEqual(http._requests, 1)

    async def test_shared_calendar_budget_covers_bootstrap_and_followup(self):
        http, calls = self.client(lambda request: _response(_token(), cookie=_GRANT), max_requests=2)
        await http.get_json(_URL)  # 主线已有请求占用同一个factory预算。
        recording = RecordingHttp(http)
        await self.unavailable(recording)
        self.assertEqual(http._requests, 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(recording.snapshots), 2)  # 第二次调用在factory预算门前被拒。
        self.assertNotIn("cookie", calls[1].headers)
        self.assert_clean(recording)

    async def test_existing_size_and_identity_bounds_remain_in_effect(self):
        cases = [
            ({"Content-Length": str(2 * 1024 * 1024 + 1)}, b"{}"),
            ({}, b" " * (2 * 1024 * 1024 + 1)),
            ({"Content-Encoding": "gzip"}, b"{}"),
        ]
        for headers, body in cases:
            with self.subTest(headers=headers, size=len(body)):
                http, calls = self.client(lambda request, headers=headers, body=body: httpx.Response(200, headers={"Content-Type": "application/json", **headers}, stream=httpx.ByteStream(body)))
                await self.unavailable(http)
                self.assertEqual(len(calls), 1)

    async def test_separate_invocations_are_cold_and_never_cache_anonymous_state(self):
        http, calls = self.client(lambda request: _response(_success()) if "cookie" in request.headers else _response(_token(), cookie=_GRANT), hosts={"acs.youku.com", "www.youku.com"})
        for _ in range(2):
            await fetch_youku_calendar(http)
        await http.get_json("https://www.youku.com/ku/webcomic")
        self.assertEqual(len(calls), 5)
        self.assertEqual(["cookie" in request.headers for request in calls], [False, True, False, True, False])
        self.assert_request(calls[2], "undefined")
        self.assertEqual(len(http._client._client.cookies), 0)

    async def test_concurrent_calls_keep_grants_separate(self):
        issued = 0
        async def handler(request):
            nonlocal issued
            await asyncio.sleep(0)
            if "cookie" not in request.headers:
                issued += 1
                return _response(_token(), cookie=f"_m_h5_tk=session{issued}_99999; Path=/")
            prefix = request.headers["cookie"].split("=", 1)[1].split("_", 1)[0]
            self.assert_request(request, prefix)
            return _response(_success(prefix))
        http, calls = self.client(handler)
        results = await asyncio.gather(fetch_youku_calendar(http), fetch_youku_calendar(http))
        self.assertEqual({p["data"]["2019061000"]["label"] for p in results}, {"session1", "session2"})
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(http._client._client.cookies), 0)

    async def test_cancellation_cleans_local_state_and_propagates_without_retry(self):
        def handler(request):
            if "cookie" in request.headers:
                raise asyncio.CancelledError
            return _response(_token(), cookie=_GRANT)
        http, calls = self.client(handler)
        recording = RecordingHttp(http)
        with self.assertRaises(asyncio.CancelledError):
            await fetch_youku_calendar(recording)
        self.assertEqual(len(calls), 2)
        self.assert_clean(recording)

    async def test_no_external_endpoint_data_or_credentials_arguments(self):
        self.assertEqual(list(inspect.signature(fetch_youku_calendar).parameters), ["http"])
        http, calls = self.client(lambda request: self.fail("must not call transport"))
        with self.assertRaises(TypeError):
            await fetch_youku_calendar(http, api="mtop.other", credentials="not-accepted")
        self.assertEqual(calls, [])

    async def test_cookie_prefix_is_raw_without_url_decoding_or_invented_suffix_rules(self):
        for value, prefix in (("barefixture", "barefixture"), ("fixture_", "fixture"), ("fixture%2Fprefix_999", "fixture%2Fprefix")):
            with self.subTest(value=value):
                http, calls = self.client(lambda request, value=value: _response(_success()) if "cookie" in request.headers else _response(_token(), cookie=f"_m_h5_tk={value}; Path=/"))
                await fetch_youku_calendar(http)
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[1].headers["cookie"], "_m_h5_tk=" + value)
                self.assert_request(calls[1], prefix)

    async def test_second_get_failure_discards_grant_before_another_invocation(self):
        number = 0
        def handler(request):
            nonlocal number
            number += 1
            if number == 1:
                return _response(_token(), cookie=_GRANT)
            if number == 2:
                raise httpx.ReadTimeout(f"Cookie={_TK}; sign=offline-sensitive; url={_URL}")
            return _response(_success())
        http, calls = self.client(handler)
        recording = RecordingHttp(http)
        await self.unavailable(recording)
        self.assert_clean(recording)
        await fetch_youku_calendar(http)
        self.assertEqual(len(calls), 3)
        self.assertNotIn("cookie", calls[2].headers)
        self.assert_request(calls[2], "undefined")

    async def test_factory_total_deadline_cancels_slow_body_without_a_retry(self):
        class SlowBody(httpx.AsyncByteStream):
            closed = False
            async def __aiter__(self):
                while True:
                    await asyncio.sleep(0.01)
                    yield b" "
            async def aclose(self):
                self.closed = True
        body = SlowBody()
        http, calls = self.client(lambda request: httpx.Response(200, headers={"Content-Type": "application/json"}, stream=body))
        original_timeout = asyncio.timeout
        with patch("app.discovery.calendar.http.asyncio.timeout", side_effect=lambda seconds: original_timeout(0.03)) as timeout:
            await self.unavailable(http)
        timeout.assert_called_once_with(12)
        self.assertEqual(len(calls), 1)
        self.assertTrue(body.closed)

    async def test_actual_minimum_interval_is_shared_by_both_gets(self):
        observed = []
        def handler(request):
            observed.append(asyncio.get_running_loop().time())
            return _response(_success()) if "cookie" in request.headers else _response(_token(), cookie=_GRANT)
        http, calls = self.client(handler, min_interval=0.02)
        await fetch_youku_calendar(http)
        self.assertEqual(len(calls), 2)
        self.assertGreaterEqual(observed[1] - observed[0], 0.015)


if __name__ == "__main__":
    unittest.main()
