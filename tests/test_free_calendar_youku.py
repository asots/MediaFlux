"""真实七日模块回归；边界变异不代表额外现场采集。"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

import tests  # noqa: F401 -- 必须先隔离配置/DB，禁止读取生产运行环境。
from app.discovery.calendar.http import CalendarHttp
from app.discovery.calendar.models import SourceUnavailable
from app.discovery.calendar.providers.youku import YoukuCalendarProvider, _initial_data
from app.discovery.calendar.youku_challenge import script_tokens
from app.indexers.errors import IndexerSecurityError

_FIXTURES = Path(__file__).with_name("fixtures") / "calendar" / "youku"
_HTML = (_FIXTURES / "webcomic.html").read_text(encoding="utf-8")
_DATA = json.loads(_HTML.split("window.__INITIAL_DATA__ =", 1)[1].split(";</script>", 1)[0])
_COMPLETION = json.loads((_FIXTURES / "completion_20260910_observations.json").read_text(encoding="utf-8"))
_POSTERS = json.loads((_FIXTURES / "poster_provenance.json").read_text(encoding="utf-8"))
_IMAGE_SAMPLE = _POSTERS["samples"][0]
_IMAGE_URL = _IMAGE_SAMPLE["raw_url"]
_IMAGE_KEY = _IMAGE_SAMPLE["normalized_host"] + _IMAGE_SAMPLE["normalized_path"]
_OTHER_IMAGE = _POSTERS["samples"][2]["raw_url"]
_URL = "https://www.youku.com/ku/webcomic"
_NOW = datetime(2026, 9, 9, 21)  # noqa: DTZ001 -- 刻意验证注入上海 naive datetime。


def _component():
    return copy.deepcopy(_DATA["moduleList"][2]["components"][0])


def _card():
    return copy.deepcopy(_DATA["moduleList"][2]["components"][0]["itemList"][3][1])


def _page(component):
    return '<script>window.__INITIAL_DATA__ =' + json.dumps(
        {"moduleList": [{"components": [component]}]}, ensure_ascii=False,
    ) + ';</script>'


def _single(*cards):
    component = _component()
    component["itemList"] = [list(cards), [], [], [], [], [], []]
    return component


class FakeHttp:
    def __init__(self, response=_HTML):
        self.response = response
        self.calls = []

    async def get_text(self, url, *, params=None):
        self.calls.append((url, params))
        if len(self.calls) != 1:
            raise AssertionError("周历不得追加请求")
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    async def get_json(self, *args, **kwargs):
        raise AssertionError("不能猜测签名接口")


class YoukuCalendarTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.network_attempts = []

        def forbidden(*args, **kwargs):
            self.network_attempts.append(True)
            raise AssertionError("Calendar regression forbids live sockets/DNS")

        for target in ("socket.getaddrinfo", "socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex"):
            self.enterContext(patch(target, side_effect=forbidden))

    def tearDown(self):
        self.assertEqual(self.network_attempts, [])

    async def fetch_component(self, component, *, now=_NOW):
        return await YoukuCalendarProvider(clock=lambda: now).fetch(FakeHttp(_page(component)))

    async def test_real_weekly_fixture_returns_47_programmes_92_events(self):
        http = FakeHttp()
        result = await YoukuCalendarProvider(clock=lambda: _NOW).fetch(http)
        self.assertEqual((result.status, result.sampled, result.ignored), ("partial", 47, 0))
        self.assertEqual(len(result.entries), 47)
        self.assertEqual(len({entry.source_id for entry in result.entries}), 47)
        events = [event for entry in result.entries for event in entry.events]
        self.assertEqual(len(events), 92)
        self.assertEqual(sum(bool(event.update_time) for event in events), 91)
        self.assertEqual(sum(event.audience == "member" for event in events), 44)
        self.assertEqual(sum(event.audience == "free" for event in events), 0)
        self.assertEqual({event.weekday for event in events}, set(range(1, 8)))
        self.assertEqual(http.calls, [(_URL, None)])
        self.assertIn("追漫日历", result.message)
        for entry in result.entries:
            self.assertEqual((entry.source, entry.category), ("youku", "animation"))
            self.assertEqual(entry.url, "https://v.youku.com/video?s=" + entry.source_id)
            self.assertEqual((entry.free_progress, entry.free_weekdays, entry.free_schedule), ("", (), ""))
            self.assertTrue(entry.evidence)
            self.assertLessEqual(len(entry.events), 50)

    async def test_live_rows_and_original_schedule_text_are_preserved(self):
        result = await self.fetch_component(_component())
        actual = {(entry.source_id, event.date, event.schedule) for entry in result.entries for event in entry.events}
        expected = set()
        for i, row in enumerate(_component()["itemList"]):
            for card in row:
                expected.add((card["action"]["value"], f"2026-09-{7+i:02d}", card["reason"]["text"]["title"]))
        self.assertEqual(actual, expected)

    async def test_svip_and_ordinary_days_remain_distinct_for_one_programme(self):
        result = await self.fetch_component(_component())
        entry = next(entry for entry in result.entries if entry.source_id == "badbb5792f934ddb82fd")
        self.assertEqual(entry.title, "师兄啊师兄")
        self.assertEqual([(e.date, e.update_time, e.audience, e.schedule) for e in entry.events], [
            ("2026-09-09", "10:00", "member", "10:00 SVIP更新1话"),
            ("2026-09-10", "10:00", "unknown", "10:00更新1话"),
        ])

    async def test_vip_programme_is_not_filtered_or_misreported_as_free(self):
        result = await self.fetch_component(_component())
        entry = next(entry for entry in result.entries if entry.title == "名侦探柯南")
        self.assertEqual([(e.weekday, e.update_time, e.audience) for e in entry.events], [(6, "19:30", "member")])
        self.assertEqual(entry.free_progress, "")

    async def test_unknown_time_does_not_borrow_other_day_or_episode_count(self):
        result = await self.fetch_component(_component())
        entry = next(entry for entry in result.entries if entry.source_id == "bcce2a04465a472bb38a")
        saturday = next(e for e in entry.events if e.weekday == 6)
        self.assertEqual(saturday.update_time, "")
        self.assertEqual(saturday.schedule, "草根厨神守护美食真心")
        self.assertTrue(any(e.update_time == "10:00" for e in entry.events))

    async def test_exact_two_host_allowlist_and_unrecognized_ssr_has_no_fixture_fallback(self):
        provider = YoukuCalendarProvider(clock=lambda: _NOW)
        self.assertEqual(provider.allowed_hosts, frozenset({"www.youku.com", "acs.youku.com"}))
        self.assertIsInstance(provider.allowed_hosts, frozenset)
        http = FakeHttp('<script>window.__INITIAL_DATA__ ={"moduleList":[]};</script>')
        result = await provider.fetch(http)
        self.assertEqual(result.entries, ())
        self.assertEqual(http.calls, [(_URL, None)])

    async def test_calendar_is_found_by_schema_not_array_position(self):
        result = await self.fetch_component(_component())  # 从真实索引2移到0。
        self.assertEqual(len(result.entries), 47)

    async def test_recommendations_and_free_ranges_are_not_calendar_dates(self):
        card = _card()
        card["mark"] = {"text": "第1-3话免费"}
        for html in (
            _page({"typeName": "KU_FLIX_V_SCROLL_COMPONENT", "title": "正在热播", "itemList": [card]}),
            _page({"typeName": "KU_FLIX_V_SCROLL_COMPONENT", "title": "限免推荐", "itemList": [card]}),
        ):
            with self.subTest(html=html[:50]):
                result = await YoukuCalendarProvider(clock=lambda: _NOW).fetch(FakeHttp(html))
                self.assertEqual(result.entries, ())
                self.assertEqual(result.status, "unavailable")

    async def test_last_week_and_out_of_week_dates_are_not_reassigned(self):
        result = await self.fetch_component(_component(), now=date(2026, 9, 14))
        self.assertEqual(result.entries, ())
        self.assertIn("不使用旧周", result.message)
        for value in ("09.06", "09.14", "2026.09.07", "09.99", None):
            with self.subTest(value=value):
                component = _component()
                component["tabList"][0]["date"] = value
                self.assertEqual((await self.fetch_component(component)).entries, ())

    async def test_real_stale_week_is_unavailable_instead_of_zero_success(self):
        for fixture in ("anime_20260910_stale_webcomic.html", _COMPLETION["fixture"]):
            with self.subTest(fixture=fixture):
                html = (_FIXTURES / fixture).read_text(encoding="utf-8")
                http = FakeHttp(html)
                result = await YoukuCalendarProvider(clock=lambda: date(2026, 9, 10)).fetch(http)
                self.assertEqual((result.status, result.entries), ("unavailable", ()))
                self.assertIn("08.31–09.06", result.message)
                self.assertIn("09.07–09.13", result.message)
                self.assertIn("不能据此判断", result.message)
                self.assertEqual(http.calls, [(_URL, None)])

    async def test_completion_capture_is_real_old_week_data_and_scripts_are_not_executed(self):
        fixture = (_FIXTURES / _COMPLETION["fixture"]).read_bytes()
        self.assertEqual(hashlib.sha256(fixture).hexdigest(), _COMPLETION["fixture_sha256"])
        html = fixture.decode("utf-8")
        script = _COMPLETION["requests"][3]["url"]
        self.assertIn(f'<script src="{script}"></script>', html)
        # 只在其原始日期周回放，证明旧周排期本身并非空模块；不改日期来通过当周校验。
        http = FakeHttp(html)
        result = await YoukuCalendarProvider(clock=lambda: date(2026, 9, 3)).fetch(http)
        self.assertEqual((result.status, len(result.entries)), ("partial", 42))
        events = [event for entry in result.entries for event in entry.events]
        self.assertEqual(len(events), 86)
        self.assertEqual({event.date for event in events}, {
            (date(2026, 8, 31) + timedelta(days=i)).isoformat() for i in range(7)
        })
        self.assertEqual(http.calls, [(_URL, None)])

    async def test_real_img_fields_replay_only_in_their_original_week(self):
        html = (_FIXTURES / _COMPLETION["fixture"]).read_text(encoding="utf-8")
        data = json.loads(html.split("window.__INITIAL_DATA__ =", 1)[1].split(";</script>", 1)[0])
        component = data["moduleList"][0]["components"][0]
        expected = {}
        # 同次实采脱敏时剥离的图片字段按节目ID回接；不改写任何真实日期。
        for sample in _POSTERS["samples"]:
            card = next(card for row in component["itemList"] for card in row
                        if card["action"]["value"] == sample["source_id"])
            self.assertEqual(card["title"], sample["title"])
            card[sample["field"]] = sample["raw_url"]
            if sample["field"] == "img":
                expected[sample["source_id"]] = sample["normalized_host"] + sample["normalized_path"]
        result = await self.fetch_component(component, now=date(2026, 9, 3))
        self.assertEqual((len(result.entries), sum(len(e.events) for e in result.entries)), (42, 86))
        self.assertEqual({entry.source_id: entry.platform_poster_key for entry in result.entries
                          if entry.platform_poster_key}, expected)
        current = await self.fetch_component(component, now=date(2026, 9, 10))
        self.assertEqual((current.status, current.entries), ("unavailable", ()))
        self.assertTrue(_POSTERS["image_request"]["mime_magic_valid"])
        self.assertTrue(_POSTERS["samples"][0]["https_url_verified"])
        self.assertFalse(_POSTERS["samples"][2]["https_url_verified"])

    async def test_img_is_used_without_himg_fallback_or_identity_changes(self):
        # 以下图片/当周卡片组合均是合成边界，不代表本轮当周现场排期。
        card = _card()
        baseline = (await self.fetch_component(_single(card))).entries[0]
        card.update(img=_IMAGE_URL, hImg=_OTHER_IMAGE)
        http = FakeHttp(_page(_single(card)))
        result = await YoukuCalendarProvider(clock=lambda: _NOW).fetch(http)
        entry = result.entries[0]
        self.assertEqual(entry.platform_poster_key, _IMAGE_KEY)
        self.assertEqual({k: v for k, v in entry.to_dict().items() if k != "platform_poster_key"},
                         {k: v for k, v in baseline.to_dict().items() if k != "platform_poster_key"})
        self.assertEqual(http.calls, [(_URL, None)])
        card.pop("img")
        no_img = (await self.fetch_component(_single(card))).entries[0]
        self.assertEqual(no_img.platform_poster_key, "")
        self.assertEqual(no_img.events, baseline.events)

    async def test_conflicting_img_keys_clear_only_image_and_cannot_recover_later(self):
        for urls in ((_IMAGE_URL, _OTHER_IMAGE, _IMAGE_URL), (_OTHER_IMAGE, _IMAGE_URL, _OTHER_IMAGE)):
            with self.subTest(first=urls[0]):
                cards = []
                for i, url in enumerate(urls):
                    card = _card()
                    card["img"] = url
                    card["reason"]["text"]["title"] = f"10:0{i}VIP更新1话"
                    cards.append(card)
                component = _single()
                component["itemList"][:3] = [[card] for card in cards]
                result = await self.fetch_component(component)
                self.assertEqual((result.status, result.sampled, result.ignored), ("partial", 1, 0))
                self.assertEqual(len(result.entries), 1)
                entry = result.entries[0]
                self.assertEqual(entry.platform_poster_key, "")
                self.assertEqual((entry.source_id, entry.title), (_card()["action"]["value"], _card()["title"]))
                self.assertEqual({(event.date, event.update_time) for event in entry.events}, {
                    (f"2026-09-{7+i:02d}", f"10:0{i}") for i in range(3)
                })
                self.assertEqual((entry.free_progress, entry.free_weekdays, entry.free_schedule), ("", (), ""))

    async def test_equivalent_img_urls_do_not_create_conflicts_or_forward_queries(self):
        cards = []
        for url in (_IMAGE_URL, _IMAGE_URL.replace("http:", "https:", 1),
                    _IMAGE_URL.removeprefix("http:"), _IMAGE_URL + "?x-oss-process=image/resize,w_240"):
            card = _card()
            card["img"] = url
            cards.append(card)
        result = await self.fetch_component(_single(*cards))
        entry = result.entries[0]
        self.assertEqual(entry.platform_poster_key, _IMAGE_KEY)
        self.assertEqual(len(entry.events), 1)
        self.assertNotIn("x-oss-process", json.dumps(entry.to_dict()))

    async def test_invalid_img_fields_only_drop_image_not_schedule(self):
        path = _IMAGE_SAMPLE["normalized_path"]
        for value in (None, "", 1, True, [], {"url": _IMAGE_URL}, "x" * 2049,
                      "javascript:alert(1)", "data:image/svg+xml,svg", path,
                      "https://127.0.0.1" + path,
                      "https://liangcang-material.alicdn.com.evil.example" + path,
                      "https://user:password@liangcang-material.alicdn.com" + path,
                      "https://liangcang-material.alicdn.com:443" + path,
                      "https://liangcang-material.alicdn.com/private.jpg",
                      "https://liangcang-material.alicdn.com" + _POSTERS["samples"][2]["normalized_path"],
                      "https://m.ykimg.com" + path, _IMAGE_URL + "#fragment"):
            with self.subTest(value=repr(value)[:100]):
                card = _card()
                card.update(img=value, hImg=_IMAGE_URL)
                result = await self.fetch_component(_single(card))
                self.assertEqual((result.status, len(result.entries), result.ignored), ("partial", 1, 0))
                self.assertEqual(result.entries[0].platform_poster_key, "")
                self.assertEqual(len(result.entries[0].events), 1)

    async def test_missing_and_rejected_img_values_do_not_poison_a_verified_key(self):
        for urls in ((_IMAGE_URL, None, "https://evil.example/x.jpg"),
                     ("https://evil.example/x.jpg", None, _IMAGE_URL)):
            cards = []
            for url in urls:
                card = _card()
                card["img"] = url
                cards.append(card)
            result = await self.fetch_component(_single(*cards))
            self.assertEqual(result.entries[0].platform_poster_key, _IMAGE_KEY)
            self.assertEqual(len(result.entries[0].events), 1)

    async def test_shared_img_asset_never_merges_programme_identity(self):
        first, second = _card(), _card()
        first["img"] = second["img"] = _IMAGE_URL
        second["action"]["value"] = "f" * 20
        result = await self.fetch_component(_single(first, second))
        self.assertEqual(len(result.entries), 2)
        self.assertEqual({entry.platform_poster_key for entry in result.entries}, {_IMAGE_KEY})
        self.assertEqual(len({entry.stable_id for entry in result.entries}), 2)

    async def test_observed_official_alias_redirects_are_not_automatically_followed(self):
        # 实采仅保留状态/Location，未消费302原文；空body是明确的离线重建。
        for record in _COMPLETION["requests"][1:3]:
            with self.subTest(url=record["url"]):
                calls = []

                def handle(request):
                    calls.append((request.method, request.headers["host"], request.url.path))
                    self.assertFalse(request.headers.get("cookie"))
                    self.assertFalse(request.headers.get("authorization"))
                    return httpx.Response(record["http_status"], headers={
                        "Content-Type": record["mime"],
                        "Location": record["redirect_target_without_query"],
                    }, content=b"")

                http = CalendarHttp({record["host"]}, transport=httpx.MockTransport(handle),
                                    resolver=lambda host, port: [(2, 1, 6, "", ("93.184.216.34", port))],
                                    max_requests=1, min_interval=0)
                try:
                    with self.assertRaises(IndexerSecurityError):
                        await http.get_text(record["url"])
                finally:
                    await http.aclose()
                self.assertEqual(calls, [("GET", record["host"], record["path"])])

    async def test_http_failures_do_not_probe_aliases_scripts_or_return_historical_fixture(self):
        # 合成HTTP边界，不宣称这些状态是本轮webcomic实采结果。
        for status in (302, 401, 403, 429, 503):
            with self.subTest(status=status):
                calls = []

                def handle(request):
                    calls.append((request.method, request.headers["host"], request.url.path))
                    return httpx.Response(status, headers={
                        "Location": "https://www.youku.com/channel/webcomic",
                    }, text="cookie=private; FAIL_SYS_USER_VALIDATE")

                provider = YoukuCalendarProvider(clock=lambda: date(2026, 9, 10))
                http = CalendarHttp(provider.allowed_hosts, transport=httpx.MockTransport(handle),
                                    resolver=lambda host, port: [(2, 1, 6, "", ("93.184.216.34", port))],
                                    max_requests=6, min_interval=0)
                try:
                    result = await provider.fetch(http)
                finally:
                    await http.aclose()
                self.assertEqual((result.status, result.entries), ("unavailable", ()))
                self.assertEqual(calls, [("GET", "www.youku.com", "/ku/webcomic")])
                self.assertNotIn("private", result.message)

    async def test_cross_year_week_uses_both_correct_years(self):
        component = _component()
        start = date(2026, 12, 28)
        names = ("一", "二", "三", "四", "五", "六", "日")
        for i, tab in enumerate(component["tabList"]):
            tab.update(date=(start + timedelta(days=i)).strftime("%m.%d"), title=names[i])
        result = await self.fetch_component(component, now=date(2027, 1, 1))
        dates = {e.date for entry in result.entries for e in entry.events}
        self.assertEqual(dates, {(start + timedelta(days=i)).isoformat() for i in range(7)})
        self.assertNotIn("2027-12-28", dates)
        self.assertNotIn("2026-01-01", dates)

    async def test_shanghai_week_not_utc_or_host_timezone(self):
        # UTC周日17:00已经是上海周一01:00。
        result = await self.fetch_component(_component(), now=datetime(2026, 9, 6, 17, tzinfo=timezone.utc))
        self.assertEqual(len(result.entries), 47)
        result = await self.fetch_component(_component(), now=datetime(2026, 9, 13, 17, tzinfo=timezone.utc))
        self.assertEqual(result.entries, ())

    async def test_today_label_does_not_override_explicit_date(self):
        component = _single(_card())
        component["tabList"][0]["title"] = "今"
        component["selectedIndex"] = 0
        result = await self.fetch_component(component)
        self.assertEqual(result.entries[0].events[0].date, "2026-09-07")

    async def test_week_structure_or_weekday_conflicts_are_rejected(self):
        for field in ("tabList", "itemList"):
            component = _component()
            component[field].pop()
            self.assertEqual((await self.fetch_component(component)).entries, ())
        component = _component()
        component["tabList"][0]["title"] = "日"
        self.assertEqual((await self.fetch_component(component)).entries, ())
        component = _component()
        component["tabList"][1]["date"] = "09.07"
        self.assertEqual((await self.fetch_component(component)).entries, ())

    async def test_wrong_component_type_or_title_does_not_create_dates(self):
        for field, value in (("type", 10000), ("typeName", "RECOMMEND"), ("title", "热门推荐")):
            component = _component()
            component[field] = value
            self.assertEqual((await self.fetch_component(component)).entries, ())

    async def test_programme_id_and_explicit_category_required(self):
        for field, values in (("value", (None, "", "XNjU0NTk3OTUwMA==", "https://evil.invalid/", [])),
                              ("category", (None, "", "电视剧", "tv", "animation", "电影", "少儿", [], {}))):
            for value in values:
                with self.subTest(field=field, value=value):
                    card = _card()
                    if field == "category":
                        card["action"]["extra"][field] = value
                    else:
                        card["action"][field] = value
                    self.assertEqual((await self.fetch_component(_single(card))).entries, ())

    async def test_single_episode_vid_never_becomes_programme_identity(self):
        card = _card()
        card["action"]["type"] = "JUMP_TO_VIDEO"
        card["previewInfo"] = {"showId": "badbb5792f934ddb82fd"}
        self.assertEqual((await self.fetch_component(_single(card))).entries, ())

    async def test_same_title_not_identity_and_duplicate_cards_are_idempotent(self):
        card = _card()
        other = copy.deepcopy(card)
        other["action"]["value"] = "0123456789abcdefabcd"
        result = await self.fetch_component(_single(card, card, other))
        self.assertEqual(len(result.entries), 2)
        self.assertEqual(sum(len(entry.events) for entry in result.entries), 2)

    async def test_conflicting_title_for_same_id_is_ignored(self):
        card = _card()
        other = copy.deepcopy(card)
        other["title"] = "另一节目"
        result = await self.fetch_component(_single(card, other, card))
        self.assertEqual(result.entries, ())
        self.assertIn("冲突", result.message)

    async def test_explicit_audience_is_independent_of_programme_badge(self):
        for text, expected in (("10:00 SVIP更新1话", "member"), ("10:00VIP更新1话", "member"),
                               ("10:00非会员更新1话", "free"), ("10:00免费更新1话", "free")):
            with self.subTest(text=text):
                card = _card()
                card["mark"] = {"text": "VIP"}
                card["reason"]["text"]["title"] = text
                result = await self.fetch_component(_single(card))
                event = result.entries[0].events[0]
                self.assertEqual((event.audience, event.update_time, event.schedule), (expected, "10:00", text))
                self.assertEqual(result.entries[0].free_progress, "")

    async def test_free_or_ordinary_progress_is_never_inferred_from_badges(self):
        for label in ("逐集限免", "限免中", "免费", "独播", ""):
            card = _card()
            card["mark"] = {"text": label}
            card["lbTexts"] = "更新至99话"
            card["tags"] = [{"text": {"title": "更新至99话"}}]
            result = await self.fetch_component(_single(card))
            self.assertEqual(result.entries[0].events[0].audience, "unknown")
            self.assertEqual(result.entries[0].free_progress, "")

    async def test_unknown_reason_does_not_borrow_generic_dates_or_times(self):
        for reason in ("剧情简介", "更新至999话", "25:00更新1话", "每周六18:00更新", None):
            card = _card()
            card["reason"]["text"]["title"] = reason
            card["lbTexts"] = "18:00更新1话"
            result = await self.fetch_component(_single(card))
            event = result.entries[0].events[0]
            self.assertEqual((event.date, event.update_time), ("2026-09-07", ""))
            self.assertEqual(event.schedule, reason or "每日更新")

    async def test_same_day_distinct_member_tiers_and_times_are_retained(self):
        cards = []
        for reason in ("10:00SVIP更新1话", "10:00VIP更新1话", "12:00更新1话"):
            card = _card()
            card["reason"]["text"]["title"] = reason
            cards.append(card)
        result = await self.fetch_component(_single(*cards))
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(len(result.entries[0].events), 3)

    async def test_programme_and_event_caps(self):
        cards = []
        for i in range(110):
            card = _card()
            card["action"]["value"] = f"{i:020x}"
            cards.append(card)
        result = await self.fetch_component(_single(*cards))
        self.assertEqual((len(result.entries), result.sampled, result.ignored), (100, 110, 10))
        self.assertIn("上限", result.message)
        cards = []
        for i in range(60):
            card = _card()
            card["reason"]["text"]["title"] = f"10:{i:02d}更新1话"
            cards.append(card)
        result = await self.fetch_component(_single(*cards))
        self.assertEqual(len(result.entries[0].events), 50)
        self.assertIn("上限", result.message)

    async def test_network_or_http_challenge_stops_without_retry_or_raw_errors(self):
        for error in (SourceUnavailable("要求验证 token=private"), TimeoutError("cookie=private")):
            http = FakeHttp(error)
            result = await YoukuCalendarProvider(clock=lambda: _NOW).fetch(http)
            self.assertEqual((result.status, result.entries), ("unavailable", ()))
            self.assertEqual(len(http.calls), 1)
            self.assertNotIn("private", result.message)

    async def test_challenge_body_never_falls_back_to_fixture(self):
        for html in ("<title>安全验证</title>", "FAIL_SYS_USER_VALIDATE", "FAIL_SYS_ILLEGAL_ACCESS"):
            http = FakeHttp(html)
            result = await YoukuCalendarProvider(clock=lambda: _NOW).fetch(http)
            self.assertEqual((result.status, result.entries, len(http.calls)), ("unavailable", (), 1))

    async def test_cancellation_propagates(self):
        with self.assertRaises(asyncio.CancelledError):
            await YoukuCalendarProvider().fetch(FakeHttp(asyncio.CancelledError()))

    async def test_undefined_safe_json_parse_and_no_script_execution(self):
        html = _page(_single(_card())).replace('"moduleList":', '"unused":undefined,"moduleList":', 1)
        result = await YoukuCalendarProvider(clock=lambda: _NOW).fetch(FakeHttp(html))
        self.assertEqual(len(result.entries), 1)
        html = html.replace(';</script>', ';alert("not executed");</script>')
        result = await YoukuCalendarProvider(clock=lambda: _NOW).fetch(FakeHttp(html))
        self.assertEqual(result.entries, ())

    async def test_unterminated_escape_runs_and_comments_never_start_dynamic_requests(self):
        for quote in ('"', "'", "`"):
            with self.subTest(quote=quote):
                html = "<script>window.__INITIAL_DATA__ = " + quote + ("\\" + quote) * 4096 + "</script>"
                http = FakeHttp(html)
                result = await YoukuCalendarProvider(clock=lambda: _NOW).fetch(http)
                self.assertEqual((result.status, result.entries), ("unavailable", ()))
                self.assertEqual(http.calls, [(_URL, None)])
                self.assertIsNone(_initial_data(html))
        http = FakeHttp("<script>window.__INITIAL_DATA__ = /*" + "\\\"" * 4096 + "</script>")
        result = await YoukuCalendarProvider(clock=lambda: _NOW).fetch(http)
        self.assertEqual((result.status, result.entries, len(http.calls)), ("unavailable", (), 1))

    def test_script_lexer_consumes_escaped_quotes_once_and_preserves_token_ranges(self):
        class CountedText(str):
            reads = 0
            def __getitem__(self, index):
                self.reads += 1
                return super().__getitem__(index)
        value = '"' + '\\"' * 20000 + '"'
        text = CountedText(value + ';/* comment */[]')
        tokens = list(script_tokens(text))
        self.assertEqual(tokens[0], ("string", 0, len(value)))
        self.assertEqual([kind for kind, _, _ in tokens],
                         ["string", "punctuation", "comment", "punctuation", "punctuation"])
        self.assertLess(text.reads, 2 * len(text))
        unclosed = CountedText(value[:-1])
        with self.assertRaises(ValueError):
            list(script_tokens(unclosed))
        self.assertLess(unclosed.reads, 2 * len(unclosed))
        regex = CountedText('/["\'/' + "\\]" * 20000 + ']/g;')
        regex_tokens = list(script_tokens(regex))
        self.assertEqual(regex_tokens[0], ("regex", 0, len(regex) - 1))
        self.assertLess(regex.reads, 3 * len(regex))
        with self.assertRaises(ValueError):
            list(script_tokens("(" * 257))
        with self.assertRaises(ValueError):
            list(script_tokens("x" * (2 * 1024 * 1024 + 1)))

    async def test_html_script_extraction_and_undefined_keep_escaped_literal_bytes(self):
        literal = r'undefined; // braces } \"' * 1024
        data = {"literal": literal, "moduleList": [{"components": [_single(_card())]}]}
        payload = json.dumps(data).replace('"moduleList":', '"unused":undefined,"moduleList":', 1)
        html = '<script data-note="a > b">window.__INITIAL_DATA__ =' + payload + ';</script>'
        parsed = _initial_data(html)
        self.assertEqual(parsed["literal"], literal)
        self.assertIsNone(parsed["unused"])
        result = await YoukuCalendarProvider(clock=lambda: _NOW).fetch(FakeHttp(html))
        self.assertEqual(len(result.entries), 1)
        # HTML 注释/属性内的伪 script 不构成 SSR；不执行任何内容。
        self.assertIsNone(_initial_data('<!--' + html + '-->'))
        self.assertIsNone(_initial_data("<div data-example='" + _page(_single(_card())) + "'></div>"))

    async def test_malformed_structures_and_cards_do_not_crash(self):
        for html in (None, '<html>token=private</html>', '<script>window.__INITIAL_DATA__ =[];</script>'):
            result = await YoukuCalendarProvider(clock=lambda: _NOW).fetch(FakeHttp(html))
            self.assertEqual((result.status, result.entries), ("unavailable", ()))
            self.assertNotIn("private", result.message)
        result = await self.fetch_component(_single(None, [], {}, {"action": []}, _card()))
        self.assertEqual(len(result.entries), 1)


    async def test_non_anime_rows_do_not_poison_verified_anime_with_the_same_id(self):
        card = _card()
        card["img"] = _IMAGE_URL
        other = copy.deepcopy(card)
        other["img"] = _OTHER_IMAGE
        other["action"]["extra"]["category"] = "电视剧"
        other["title"] = "非动漫"
        for cards in ((card, other), (other, card)):
            with self.subTest(first=cards[0]["title"]):
                result = await self.fetch_component(_single(*cards))
                self.assertEqual([(entry.source_id, entry.title, entry.category) for entry in result.entries],
                                 [(card["action"]["value"], card["title"], "animation")])
                self.assertEqual(len(result.entries[0].events), 1)
                self.assertEqual(result.entries[0].platform_poster_key, _IMAGE_KEY)

    async def test_same_time_member_and_non_member_events_stay_independent(self):
        member, free = _card(), _card()
        member["reason"]["text"]["title"] = "10:00VIP更新1话"
        free["reason"]["text"]["title"] = "10:00非会员更新1话"
        result = await self.fetch_component(_single(member, free, member))
        self.assertEqual(len(result.entries), 1)
        entry = result.entries[0]
        self.assertEqual({(event.date, event.update_time, event.audience) for event in entry.events},
                         {("2026-09-07", "10:00", "member"), ("2026-09-07", "10:00", "free")})
        self.assertEqual((entry.free_progress, entry.free_weekdays, entry.free_schedule), ("", (), ""))

    async def test_clock_accepts_date_and_preserves_original_naive_and_aware_semantics(self):
        for value in (_NOW, date(2026, 9, 9), datetime(2026, 9, 8, 18, tzinfo=timezone.utc)):
            with self.subTest(value=value):
                provider = YoukuCalendarProvider(clock=lambda value=value: value)
                self.assertEqual(provider._today(), date(2026, 9, 9))
                http = FakeHttp()
                result = await provider.fetch(http)
                self.assertEqual((len(result.entries), sum(len(entry.events) for entry in result.entries)), (47, 92))
                self.assertEqual(http.calls, [(_URL, None)])

    async def test_invalid_additional_calendar_preserves_verified_component_without_more_requests(self):
        valid, invalid = _component(), _component()
        invalid["tabList"][0]["date"] = "09.14"
        data = {"moduleList": [{"components": [valid, invalid]}]}
        http = FakeHttp('<script>window.__INITIAL_DATA__ =' + json.dumps(data) + ';</script>')
        result = await YoukuCalendarProvider(clock=lambda: _NOW).fetch(http)
        self.assertEqual((len(result.entries), sum(len(entry.events) for entry in result.entries)), (47, 92))
        self.assertEqual(result.status, "partial")
        self.assertEqual(http.calls, [(_URL, None)])

    async def test_shared_http_needs_only_one_anime_get_and_no_detail_capability(self):
        requests = []

        def handle(request):
            requests.append(request)
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.headers["host"], "www.youku.com")
            self.assertEqual(request.url.path, "/ku/webcomic")
            self.assertFalse(request.url.query)
            self.assertFalse(request.headers.get("cookie"))
            self.assertFalse(request.headers.get("authorization"))
            return httpx.Response(200, text=_HTML)

        provider = YoukuCalendarProvider(clock=lambda: _NOW)
        http = CalendarHttp(provider.allowed_hosts, transport=httpx.MockTransport(handle),
                            resolver=lambda host, port: [(2, 1, 6, "", ("93.184.216.34", port))],
                            max_requests=1, min_interval=0)
        try:
            result = await provider.fetch(http)
        finally:
            await http.aclose()
        self.assertEqual((len(requests), len(result.entries), sum(len(entry.events) for entry in result.entries)), (1, 47, 92))
        self.assertEqual(result.status, "partial")

if __name__ == "__main__":
    unittest.main()
