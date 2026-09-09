"""实采匿名动态周历回放；网络由MockTransport隔离，会话数据全部为合成fixture。"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx

import tests  # noqa: F401 -- 先隔离DB/config，再导入应用。
from app.discovery.calendar.http import CalendarHttp
from app.discovery.calendar.providers.youku import YoukuCalendarProvider, _dynamic_calendar_data, _parse_calendar_data
from app.discovery.calendar.youku_challenge import has_access_challenge

FIXTURES = Path(__file__).with_name("fixtures") / "calendar" / "youku"
DYNAMIC = json.loads((FIXTURES / "dynamic_week_20260910.json").read_text())
STALE_HTML = (FIXTURES / "anime_20260910_completion_stale_webcomic.html").read_text()
CURRENT_HTML = (FIXTURES / "anime_20260910_webcomic.html").read_text()
API = "mtop.youku.columbus.home.query"
GRANT = "_m_h5_tk=fixturetoken_4102444800000; Domain=.youku.com; Path=/; Secure"


def root_of(payload):
    return payload["data"]["2019061000"]["data"]["nodes"][0]


def calendar_of(payload):
    return root_of(payload)["nodes"][0]["nodes"][0]


class YoukuDynamicTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.network_attempts = []
        def forbidden(*args, **kwargs):
            self.network_attempts.append(True)
            raise AssertionError("动态周历回归禁止真实网络")
        for name in ("getaddrinfo", "create_connection", "socket.connect", "socket.connect_ex"):
            self.enterContext(patch("socket." + name, side_effect=forbidden))
        self.provider = YoukuCalendarProvider(clock=lambda: datetime(2026, 9, 10, 3, 7))
        self.calls = []
        self.html = STALE_HTML
        self.payload = copy.deepcopy(DYNAMIC)
        self.responses = None

    def tearDown(self):
        self.assertEqual(self.network_attempts, [])

    def handle(self, request):
        self.calls.append(request)
        self.assertEqual(request.method, "GET")
        self.assertEqual(request.headers["accept-encoding"], "identity")
        self.assertEqual(request.extensions["sni_hostname"], request.headers["host"])
        if request.headers["host"] == "www.youku.com":
            self.assertEqual(request.url.path, "/ku/webcomic")
            self.assertNotIn("cookie", request.headers)
            if isinstance(self.html, httpx.Response):
                return self.html
            return httpx.Response(200, text=self.html)
        self.assertEqual(request.headers["host"], "acs.youku.com")
        self.assertEqual(request.url.path, "/h5/" + API + "/1.0/")
        self.assertNotIn("callback", request.url.params)
        self.assertEqual(json.loads(json.loads(request.url.params["data"])["params"])["nodeKey"], "WEBCOMIC")
        if self.responses is not None:
            result = self.responses.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        if len(self.calls) == 2:
            self.assertNotIn("cookie", request.headers)
            return httpx.Response(200, json={"api": API, "v": "1.0", "ret": ["FAIL_SYS_TOKEN_EMPTY::token missing"], "data": {}},
                                  headers={"Set-Cookie": GRANT})
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(request.headers["cookie"], "_m_h5_tk=fixturetoken_4102444800000")
        return httpx.Response(200, json=self.payload)

    async def fetch(self, *, max_requests=3):
        client = CalendarHttp(self.provider.allowed_hosts, transport=httpx.MockTransport(self.handle),
                              resolver=lambda h, p: [(2, 1, 6, "", ("93.184.216.34", p))],
                              min_interval=0, max_requests=max_requests)
        try:
            return await self.provider.fetch(client)
        finally:
            self.assertFalse(list(client._client._client.cookies.jar))
            await client.aclose()

    async def test_stale_ssr_then_real_dynamic_week_uses_exactly_three_bounded_gets(self):
        result = await self.fetch()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual((result.status, len(result.entries), result.sampled, result.ignored), ("partial", 47, 47, 0))
        self.assertIn("动态", result.message)
        events = [event for entry in result.entries for event in entry.events]
        self.assertEqual((len(events), sum(bool(event.update_time) for event in events)), (92, 91))
        self.assertEqual(sorted({event.date for event in events}), [f"2026-09-{day:02}" for day in range(7, 14)])
        self.assertTrue(all(entry.category == "animation" and not entry.free_progress for entry in result.entries))
        self.assertTrue(all(entry.platform_poster_key and entry.url.startswith("https://v.youku.com/video?s=")
                            for entry in result.entries))
        self.assertEqual(result.entries, _parse_calendar_data(_dynamic_calendar_data(DYNAMIC), date(2026, 9, 10)).entries)

    async def test_valid_current_ssr_remains_one_get_and_does_not_start_anonymous_session(self):
        self.html = CURRENT_HTML
        result = await self.fetch()
        self.assertEqual((result.status, len(result.entries), len(self.calls)), ("partial", 47, 1))
        self.assertNotIn("动态", result.message)

    async def test_denied_challenged_or_unrecognized_ssr_stops_before_dynamic(self):
        for page in (httpx.Response(403), httpx.Response(429), httpx.Response(302, headers={"Location": "/captcha"}),
                     "<title>安全验证</title>", "<script>window.__INITIAL_DATA__ = {};</script>",
                     '<script>window.__INITIAL_DATA__ = {"moduleList": []};</script>'):
            with self.subTest(page=type(page).__name__):
                self.calls = []
                self.html = page
                self.assertEqual((await self.fetch()).status, "unavailable")
                self.assertEqual(len(self.calls), 1)

    async def test_dynamic_denial_or_missing_grant_stops_without_third_get(self):
        for response in (httpx.Response(401), httpx.Response(403), httpx.Response(429),
                         httpx.Response(302, headers={"Location": "https://login.youku.com/"}),
                         httpx.Response(200, text="<title>captcha</title>"),
                         httpx.Response(200, json={"data": {}, "ret": ["FAIL_SYS_TOKEN_EMPTY::missing"]}),
                         httpx.Response(200, json={"data": {}, "ret": ["FAIL_SYS_ILLEGAL_ACCESS::denied"]}),
                         RuntimeError("upstream-private-query-must-not-escape")):
            with self.subTest(response=type(response).__name__):
                self.calls = []
                self.responses = [response]
                result = await self.fetch()
                self.assertEqual((result.status, len(self.calls)), ("unavailable", 2))
                self.assertIn("08.31", result.message)
                self.assertNotIn("private", result.message)

    async def test_api_cancellation_propagates_and_client_jar_is_empty(self):
        self.responses = [asyncio.CancelledError()]
        with self.assertRaises(asyncio.CancelledError):
            await self.fetch()
        self.assertEqual(len(self.calls), 2)

    async def test_dynamic_initialization_shares_ssr_budget(self):
        result = await self.fetch(max_requests=2)
        self.assertEqual((result.status, len(self.calls)), ("unavailable", 2))

    async def test_old_dynamic_dates_and_wrong_channel_cannot_turn_into_current_week(self):
        for invalid in ("old", "partial", "weekday", "channel", "schema", "empty", "no_api", "mixed_ret"):
            with self.subTest(invalid=invalid):
                self.calls = []
                self.payload = copy.deepcopy(DYNAMIC)
                component = calendar_of(self.payload)
                if invalid == "old":
                    for day, value in zip(component["nodes"], ("08.31", "09.01", "09.02", "09.03", "09.04", "09.05", "09.06")):
                        day["data"]["date"] = value
                elif invalid == "partial":
                    component["nodes"].pop()
                elif invalid == "weekday":
                    component["nodes"][0]["data"]["title"] = "二"
                elif invalid == "channel":
                    root_of(self.payload)["data"]["nodeKey"] = "TV"
                elif invalid == "schema":
                    component["id"] = 36
                elif invalid == "empty":
                    for day in component["nodes"]:
                        day["nodes"] = []
                elif invalid == "no_api":
                    self.payload.pop("api")
                else:
                    self.payload["ret"].append("FAIL_SYS_TOKEN_EMPTY::invalid")
                result = await self.fetch()
                self.assertEqual((result.status, result.entries, len(self.calls)), ("unavailable", (), 3))

    def test_real_projection_is_pure_bounded_and_does_not_expose_tracking_context(self):
        original = copy.deepcopy(DYNAMIC)
        raw = copy.deepcopy(DYNAMIC)
        raw["session"] = "synthetic-untrusted-context"
        calendar_of(raw)["nodes"][0]["nodes"][0]["data"]["trackInfo"] = {"secret": "not-a-profile"}
        raw["recommendation"] = original
        result = _dynamic_calendar_data(raw)
        self.assertEqual(result, _dynamic_calendar_data(DYNAMIC))
        self.assertEqual(DYNAMIC, original)
        self.assertNotIn("synthetic-untrusted-context", json.dumps(result))
        self.assertNotIn("not-a-profile", json.dumps(result))
        for wrong_level in (True, "0", 1, None):
            with self.subTest(level=wrong_level):
                root_of(raw)["level"] = wrong_level
                self.assertIsNone(_dynamic_calendar_data(raw))
        self.assertEqual(_parse_calendar_data(result, date(2026, 9, 17)).status, "unavailable")

    def test_real_fixture_provenance_and_member_labels_remain_independent_of_free_facts(self):
        provenance = json.loads((FIXTURES / "dynamic_week_20260910_provenance.json").read_text())
        self.assertEqual(hashlib.sha256((FIXTURES / provenance["fixture"]).read_bytes()).hexdigest(), provenance["fixture_sha256"])
        self.assertFalse(provenance["header_or_token_values_retained"])
        raw = copy.deepcopy(DYNAMIC)
        for day in calendar_of(raw)["nodes"]:
            for card in day["nodes"]:
                card["data"]["reason"]["text"]["title"] = "10:00 第5集"
                card["data"]["mark"] = {"type": "SIMPLE", "data": {"text": "VIP"}}
        entries = _parse_calendar_data(_dynamic_calendar_data(raw), date(2026, 9, 10)).entries
        self.assertTrue(entries)
        self.assertTrue(all(event.audience == "member" for entry in entries for event in entry.events))
        self.assertTrue(all(not entry.free_progress for entry in entries))

    async def test_independent_complete_runtime_capture_rotates_posters_not_schedule_or_identity(self):
        proof = json.loads((FIXTURES / "dynamic_runtime_20260910_provenance.json").read_text())
        delta = json.loads((FIXTURES / proof["poster_delta_fixture"]).read_text())
        self.assertEqual(hashlib.sha256((FIXTURES / proof["base_fixture"]).read_bytes()).hexdigest(), proof["base_fixture_sha256"])
        self.assertEqual(hashlib.sha256((FIXTURES / proof["poster_delta_fixture"]).read_bytes()).hexdigest(), proof["poster_delta_sha256"])
        self.assertFalse(proof["response_header_or_cookie_values_retained"])
        for day in calendar_of(self.payload)["nodes"]:
            for node in day["nodes"]:
                card = node["data"]
                source_id = card["action"]["value"]
                if source_id in delta["img_by_source_id"]:
                    card["img"] = delta["img_by_source_id"][source_id]
        result = await self.fetch()
        baseline = _parse_calendar_data(_dynamic_calendar_data(DYNAMIC), date(2026, 9, 10))
        self.assertEqual(len(self.calls), 3)
        self.assertEqual([replace(entry, platform_poster_key="") for entry in result.entries],
                         [replace(entry, platform_poster_key="") for entry in baseline.entries])
        self.assertEqual(sum(a.platform_poster_key != b.platform_poster_key for a, b in zip(result.entries, baseline.entries)), 9)
        self.assertEqual((len(result.entries), sum(len(entry.events) for entry in result.entries)),
                         (proof["runtime_summary"]["programmes"], proof["runtime_summary"]["events"]))

    async def test_inert_sdk_error_checks_do_not_suppress_current_or_dynamic_calendars(self):
        for code in ("FAIL_SYS_ILLEGAL_ACCESS", "FAIL_SYS_USER_VALIDATE", "FAIL_SYS_TOKEN_EMPTY", "FAIL_SYS_TOKEN_EXOIRED"):
            for html, count in ((CURRENT_HTML, 1), (STALE_HTML, 3)):
                with self.subTest(code=code, count=count):
                    self.calls = []
                    self.html = '<script>function onPublicError(e){return e.ret[0].indexOf("' + code + '")>=0}</script>' + html
                    result = await self.fetch()
                    self.assertEqual((result.status, len(result.entries), len(self.calls)), ("partial", 47, count))

    async def test_actual_visible_challenge_error_envelope_or_navigation_stops_even_with_ssr(self):
        for challenge in (
            '<title>安全验证</title><body>请完成安全验证</body>',
            '<body>FAIL_SYS_USER_VALIDATE</body>',
            '<body>&#70;AIL_SYS_ILLEGAL_ACCESS</body>',
            '<script>window.result={"ret":["FAIL_SYS_USER_VALIDATE::denied"]};</script>',
            '<script>result.ret = ["FAIL_SYS_ILLEGAL_ACCESS"];</script>',
            '<script>window.location.href="/login";</script>',
            '<script>window.location.replace("https://www.youku.com/captcha");</script>',
            '<script>window.rgv587_flag=true;</script>',
        ):
            for html in (CURRENT_HTML, STALE_HTML):
                with self.subTest(challenge=challenge, current=html is CURRENT_HTML):
                    self.calls = []
                    self.html = challenge + html
                    result = await self.fetch()
                    self.assertEqual((result.status, result.entries, len(self.calls)), ("unavailable", (), 1))
                    self.assertIn("验证", result.message)

    async def test_inert_factories_examples_and_callbacks_are_not_page_error_responses(self):
        for script in (
            'function buildSdkError(){return {ret:["FAIL_SYS_ILLEGAL_ACCESS"]};}',
            "const sdkExample='" + '{"ret":["FAIL_SYS_ILLEGAL_ACCESS"]}' + "';",
            'function showLoginOnClick(){window.location.href="/login";}',
            'const onClick = () => {window.location.href="/login";};',
            'const onClick = function(){window.location.replace("/captcha");};',
            '// ret: ["FAIL_SYS_USER_VALIDATE"]\nconst note="ordinary";',
            'function f(){return "braces } ; ( in a string";} const next="ordinary";',
            # 普通正则字面量中的引号不能当成页面拒访。
            "const quote = /'/;",
            'const quote = /"/;',
            'const quote = /`/;',
            "const re = /[\"']/g;",
            r"const re = /[\"\']/g;",
        ):
            for html, count in ((CURRENT_HTML, 1), (STALE_HTML, 3)):
                with self.subTest(script=script, count=count):
                    self.calls = []
                    self.html = "<script>" + script + "</script>" + html
                    result = await self.fetch()
                    self.assertEqual((result.status, len(result.entries), len(self.calls)), ("partial", 47, count))

    async def test_decoded_visible_denial_mixed_ret_and_top_level_navigation_variants_stop(self):
        for challenge in (
            '<title>安全验&#x8bc1;</title><body>请完成安全验&#x8bc1;</body>',
            '<body>请完成<span>安全验证</span></body>',
            '<script>window.result={"ret":["SUCCESS","FAIL_SYS_USER_VALIDATE::denied"]};</script>',
            '<script>const unrelated = 1; window.result={"ret":["SUCCESS","FAIL_SYS_ILLEGAL_ACCESS"]};</script>',
            '<script>function onClick(){} window.result={"ret":["FAIL_SYS_USER_VALIDATE"]};</script>',
            '<script>result["ret"]=["SUCCESS","FAIL_SYS_USER_VALIDATE"];</script>',
            '<script>window.location.replace ("/captcha");</script>',
            '<script>const x = "semicolon; brace}"; window.location["href"] = "/captcha";</script>',
            '<script>/* normal comment */ window["location"]["replace"] ("/captcha");</script>',
            '<script>window.location = "/cap%74cha";</script>',
            '<meta http-equiv="refresh" content="0;url=/captcha">',
            '<script>const quote = /"/;</script><script>window.result={"ret":["FAIL_SYS_ILLEGAL_ACCESS"]};</script>',
        ):
            for html in (CURRENT_HTML, STALE_HTML):
                with self.subTest(challenge=challenge, current=html is CURRENT_HTML):
                    self.calls = []
                    self.html = challenge + html
                    result = await self.fetch()
                    self.assertEqual((result.status, result.entries, len(self.calls)), ("unavailable", (), 1))

    def test_regex_literals_and_division_do_not_hide_adjacent_real_challenges(self):
        # 前两个是主线对照旧版 False 的确切兼容案例，均不执行 JS/正则。
        ordinary = (
            "const re = /[\"']/g;",
            r"const re = /[\"\']/g;",
            "const re = /[\"'/]/g;",
            r"const re = /[\]\"'\/]+/g;",
            "const re = /[;{}()\"'/]+/g;",
            "if (ready) /[\"']/.test(value);",
            "function escape(value) { return /[\"']/g; }",
            "const re = /* ordinary comment */ /[\"']/g;",
            "const result = 12 / 3 / 2;",
            "const result = obj.return / 2;",
            "const result = obj.if(12) / 2;",
            "const re = /FAIL_SYS_ILLEGAL_ACCESS|captcha/g;",
        )
        denials = (
            'window.result={"ret":["FAIL_SYS_ILLEGAL_ACCESS"]};',
            'window.result={"ret":["SUCCESS","FAIL_SYS_USER_VALIDATE"]};',
            'window.location.href="/captcha";',
        )
        for script in ordinary:
            with self.subTest(script=script):
                self.assertFalse(has_access_challenge("<script>" + script + "</script>"))
                for denial in denials:
                    for combined in (script + denial, denial + script):
                        self.assertTrue(has_access_challenge("<script>" + combined + "</script>"), combined)
