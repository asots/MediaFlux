"""真实官方追番表回放；clock 固定，不依赖运行日期，不请求外网。"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import socket
import unittest
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

import tests  # noqa: F401 -- 应用导入前隔离 config/DB。
from app.discovery.calendar.http import CalendarHttp
from app.discovery.calendar.models import SourceUnavailable
from app.discovery.calendar.providers.iqiyi import IqiyiCalendarProvider
from app.indexers.errors import IndexerSecurityError

_FIXTURES = Path(__file__).parent / "fixtures" / "calendar" / "iqiyi"
_URL = "https://mesh.if.iqiyi.com/portal/lw/v7/channel/page/tracking"
_PARAMS = {"channelId": "4", "mode": "page", "page": "1", "v": "17.091.26283"}
_NOW = datetime(2026, 9, 9, 12)  # noqa: DTZ001 -- 刻意验证注入上海 naive datetime。


def _fixture():
    return json.loads((_FIXTURES / "weekly_tracking.json").read_text("utf-8"))


def _http(payload=None):
    async def get_json(url, *, params=None):
        if url != _URL or params != _PARAMS:
            raise AssertionError("只能请求固定动漫 tracking 入口与参数")
        return _fixture() if payload is None else payload

    return SimpleNamespace(
        get_json=AsyncMock(side_effect=get_json),
        get_text=AsyncMock(side_effect=AssertionError("不得补抓节目详情")),
    )


def _provider(clock=lambda: _NOW):
    return IqiyiCalendarProvider(clock=clock)


class IqiyiWeeklyCalendarTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.network_attempts = []

        def forbidden(*args, **kwargs):
            self.network_attempts.append(True)
            raise AssertionError("Calendar regression forbids live sockets/DNS")

        for target in ("socket.getaddrinfo", "socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex"):
            self.enterContext(patch(target, side_effect=forbidden))

    def tearDown(self):
        self.assertEqual(self.network_attempts, [])

    async def test_real_weekly_fixture_produces_programmes_and_events_without_free_progress(self):
        http = _http()
        result = await _provider().fetch(http)
        self.assertEqual(result.status, "partial")
        self.assertEqual((len(result.entries), sum(len(e.events) for e in result.entries)), (40, 72))
        self.assertEqual((result.sampled, result.ignored), (41, 1))
        self.assertEqual(sum(bool(event.update_time) for e in result.entries for event in e.events), 8)
        http.get_json.assert_awaited_once_with(_URL, params=_PARAMS)
        http.get_text.assert_not_awaited()
        self.assertEqual(IqiyiCalendarProvider.allowed_hosts, frozenset({"mesh.if.iqiyi.com"}))
        self.assertEqual(len({entry.source_id for entry in result.entries}), len(result.entries))
        for entry in result.entries:
            self.assertEqual(entry.category, "animation")
            self.assertTrue(entry.evidence)
            self.assertEqual((entry.free_progress, entry.free_weekdays, entry.free_schedule), ("", (), ""))
            self.assertLessEqual(len(entry.events), 50)
            self.assertEqual(len({e.date for e in entry.events}), len(entry.events))
            for event in entry.events:
                self.assertGreaterEqual(event.date, "2026-09-07")
                self.assertLessEqual(event.date, "2026-09-13")
                self.assertEqual(event.audience, "unknown")
                self.assertTrue(event.schedule)

    async def test_real_thursday_clock_and_programme_identity(self):
        result = await _provider().fetch(_http())
        entry = next(e for e in result.entries if e.source_id == "8837962497879001")
        self.assertEqual(entry.title, "逆天邪神年番")
        event, = entry.events
        self.assertEqual((event.date, event.weekday, event.update_time), ("2026-09-10", 4, "09:00"))
        self.assertIn("明日09:00更新", event.schedule)
        self.assertEqual(entry.free_progress, "")
        self.assertEqual(event.audience, "unknown")
        self.assertIn("VIP", entry.evidence)
        self.assertIn("不构成免费承诺", entry.evidence)

    async def test_real_next_update_time_is_not_copied_to_other_weekdays(self):
        result = await _provider().fetch(_http())
        rows = [entry for entry in result.entries if entry.title == "原来我早就无敌了 动态漫画"]
        self.assertEqual(len(rows), 1)
        entry = rows[0]
        self.assertGreater(len(entry.events), 1)
        for event in entry.events:
            self.assertEqual(event.update_time, "10:00" if event.date == "2026-09-10" else "")
        self.assertNotIn("2026-09-06", [e.date for e in entry.events])
        self.assertNotIn("2026-09-13", [e.date for e in entry.events])

    async def test_coming_preview_is_not_regular_programme_schedule(self):
        result = await _provider().fetch(_http())
        # 真实 fixture 的周四/即将上线组均含此 PREVUE，不能当正片更新。
        self.assertNotIn("5466459440004201", {e.source_id for e in result.entries})

    async def test_clock_accepts_date_naive_and_aware_shanghai_conversion(self):
        clocks = (lambda: date(2026, 9, 9), lambda: _NOW,
                  lambda: datetime(2026, 9, 8, 16, tzinfo=timezone.utc))
        for clock in clocks:
            with self.subTest(clock=clock):
                result = await _provider(clock).fetch(_http())
                self.assertEqual(sum(len(e.events) for e in result.entries), 72)
        # 已存 09-06..09-12 的响应不能改造成下一周的数据。
        result = await _provider(lambda: date(2026, 9, 16)).fetch(_http())
        self.assertEqual(result.entries, ())

    async def test_same_week_stale_response_uses_verified_source_reference_day(self):
        payload = _fixture()
        original = copy.deepcopy(payload)
        baseline = await _provider(lambda: date(2026, 9, 9)).fetch(_http(payload))
        http = _http(payload)
        result = await _provider(lambda: date(2026, 9, 10)).fetch(http)
        self.assertEqual(result.entries, baseline.entries)
        timed = [event for entry in result.entries for event in entry.events if event.update_time]
        self.assertEqual(len(timed), 8)
        self.assertEqual({event.date for event in timed}, {"2026-09-10"})
        self.assertIn("源今天为2026-09-09", result.message)
        self.assertEqual(http.get_json.await_count, 1)
        self.assertEqual(payload, original)  # 没有重写 fixture 日期或源字段。

    async def test_relative_times_without_unique_verified_reference_are_unknown(self):
        for invalid in ("missing", "conflicting", "wrong_weekday", "future"):
            payload = _fixture()
            groups = payload["items"][0]["video"]
            today = next(group for group in groups if group.get("title") == "今天")
            if invalid == "missing":
                today["title"] = "周三"
            elif invalid == "conflicting":
                groups[4]["title"] = "今天"  # 合成冲突标签，不造新日期。
            elif invalid == "wrong_weekday":
                today["block_id"] = "jmd_Sun"
            http = _http(payload)
            now = date(2026, 9, 8) if invalid == "future" else date(2026, 9, 9)
            with self.subTest(invalid=invalid):
                result = await _provider(lambda: now).fetch(http)
                self.assertTrue(result.entries)
                self.assertTrue(all(not event.update_time for entry in result.entries for event in entry.events))
                self.assertEqual(http.get_json.await_count, 1)

    async def test_unrelated_channel_today_label_cannot_poison_reference(self):
        payload = _fixture()
        other = copy.deepcopy(payload["items"][0])
        other["channel"] = "2"
        other["video"][4]["title"] = "今天"
        payload["items"].append(other)
        result = await _provider().fetch(_http(payload))
        baseline = await _provider().fetch(_http())
        self.assertEqual(result, baseline)

    async def test_non_ascii_time_label_drops_only_bad_time_not_source(self):
        baseline = await _provider().fetch(_http())
        source_id = "8837962497879001"
        for label in ("明日09:0\u0660更新", "明日0\u0669:00更新", "明日09:0９更新"):
            with self.subTest(label=label):
                payload = _fixture()
                payload["items"][0]["video"][4]["data"][0]["tag3lines"] = [{"text": label}]
                result = await _provider().fetch(_http(payload))
                self.assertEqual((len(result.entries), sum(len(e.events) for e in result.entries)), (40, 72))
                self.assertEqual(tuple(e for e in result.entries if e.source_id != source_id),
                                 tuple(e for e in baseline.entries if e.source_id != source_id))
                entry = next(e for e in result.entries if e.source_id == source_id)
                self.assertEqual([(e.date, e.update_time) for e in entry.events], [("2026-09-10", "")])

    async def test_wrong_calendar_category_or_dates_fail_closed(self):
        for field, value in (("channel", "2"), ("date", "09-11"), ("weekday", "jmd_Fri")):
            with self.subTest(field=field):
                payload = _fixture()
                block = payload["items"][0]
                block["video"] = [block["video"][4]]
                if field == "channel":
                    block["channel"] = value
                elif field == "date":
                    block["video"][0]["sub_title"] = value
                else:
                    block["video"][0]["block_id"] = value
                result = await _provider().fetch(_http(payload))
                self.assertEqual((result.status, result.entries), ("partial", ()))
        for payload in ({}, {"code": "A00000", "data": {"list": []}},
                        {"code": 0, "cname": "电视剧-v7", "items": []}):
            self.assertEqual((await _provider().fetch(_http(payload))).entries, ())

    async def test_real_payload_duplicates_are_merged_and_untrusted_urls_not_followed(self):
        payload = _fixture()
        payload["items"] *= 2
        result = await _provider().fetch(_http(payload))
        self.assertEqual((len(result.entries), sum(len(e.events) for e in result.entries)), (40, 72))
        for url in ("http://127.0.0.1/", "https://www.iqiyi.com.evil.invalid/v_test.html"):
            with self.subTest(url=url):
                payload = _fixture()
                block = payload["items"][0]
                group = block["video"][4]
                group["data"] = [group["data"][0]]
                group["data"][0]["page_url"] = url
                block["video"] = [group]
                http = _http(payload)
                self.assertEqual((await _provider().fetch(http)).entries, ())
                http.get_json.assert_awaited_once_with(_URL, params=_PARAMS)
                http.get_text.assert_not_awaited()

    async def test_limits_do_not_add_requests_and_conflicting_time_is_unknown(self):
        payload = _fixture()
        block = payload["items"][0]
        group = block["video"][4]
        original = group["data"][0]
        group["data"] = []
        for i in range(110):
            row = copy.deepcopy(original)
            row["album_id"] = 100000 + i
            group["data"].append(row)
        block["video"] = [group]
        http = _http(payload)
        result = await _provider().fetch(http)
        self.assertEqual(len(result.entries), 100)
        self.assertEqual(result.ignored, 10)
        self.assertEqual(http.get_json.await_count, 1)
        group["data"] = [original]
        original["tag3lines"].append({"text": "明日11:00更新"})
        result = await _provider().fetch(_http(payload))
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].events[0].update_time, "")

    async def test_network_challenge_and_cancellation_stop_without_retries(self):
        for error in (SourceUnavailable("captcha token=secret"), httpx.ConnectError("secret"),
                      TimeoutError("secret"), IndexerSecurityError("secret")):
            with self.subTest(error=type(error).__name__):
                http = _http()
                http.get_json.side_effect = error
                result = await _provider().fetch(http)
                self.assertEqual((result.status, result.entries), ("unavailable", ()))
                self.assertEqual(http.get_json.await_count, 1)
                self.assertNotIn("secret", result.message)
        http = _http({"code": "P00111", "message": "captcha token=secret"})
        result = await _provider().fetch(http)
        self.assertEqual((result.status, result.entries), ("unavailable", ()))
        self.assertEqual(http.get_json.await_count, 1)
        self.assertNotIn("secret", result.message)
        http = _http()
        http.get_json.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await _provider().fetch(http)

    async def test_real_fixture_through_shared_http_mock_transport(self):
        calls = []
        def handler(request):
            calls.append(request)
            self.assertEqual(request.headers["host"], "mesh.if.iqiyi.com")
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.url.path, "/portal/lw/v7/channel/page/tracking")
            self.assertEqual(dict(request.url.params), _PARAMS)
            self.assertFalse(request.headers.get("cookie"))
            self.assertFalse(request.headers.get("authorization"))
            return httpx.Response(200, json=_fixture())
        http = CalendarHttp(IqiyiCalendarProvider.allowed_hosts,
            transport=httpx.MockTransport(handler),
            resolver=lambda host, port: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port))],
            min_interval=0)
        try:
            result = await _provider().fetch(http)
        finally:
            await http.aclose()
        self.assertEqual((len(calls), len(result.entries)), (1, 40))


    async def test_non_anime_rows_cannot_create_entries_or_poison_verified_anime_identity(self):
        payload = _fixture()
        block = payload["items"][0]
        group = block["video"][4]
        original = group["data"][0]
        non_anime = copy.deepcopy(original)
        non_anime.update(channel_id=2, title="<script>not-anime</script>", page_url="http://127.0.0.1/private")
        other = copy.deepcopy(non_anime)
        other["album_id"] = "999999999999"
        block["video"] = [group]
        for rows in ([non_anime, original, other], [original, other, non_anime]):
            with self.subTest(first=rows[0]["channel_id"]):
                group["data"] = rows
                http = _http(payload)
                result = await _provider().fetch(http)
                self.assertEqual([(entry.source_id, entry.category) for entry in result.entries],
                                 [(str(original["album_id"]), "animation")])
                http.get_json.assert_awaited_once_with(_URL, params=_PARAMS)
                http.get_text.assert_not_awaited()

    async def test_malformed_date_group_keeps_other_verified_dates_with_one_request(self):
        baseline = await _provider().fetch(_http())
        payload = _fixture()
        group = payload["items"][0]["video"][4]
        self.assertEqual(group["sub_title"], "09-10")
        group["data"] = None
        http = _http(payload)
        result = await _provider().fetch(http)
        expected = {(entry.source_id, event) for entry in baseline.entries for event in entry.events
                    if event.date != "2026-09-10"}
        self.assertTrue(expected)
        self.assertEqual({(entry.source_id, event) for entry in result.entries for event in entry.events}, expected)
        self.assertEqual(result.status, "partial")
        http.get_json.assert_awaited_once_with(_URL, params=_PARAMS)
        http.get_text.assert_not_awaited()

    @staticmethod
    def poster_snapshot():
        return json.loads((_FIXTURES / "platform_images_20260910.json").read_text("utf-8"))

    @staticmethod
    def poster_rows(payload):
        return [row for block in payload["items"] for group in block.get("video", []) for row in group["data"]]

    @staticmethod
    def without_posters(result):
        return replace(result, entries=tuple(replace(entry, platform_poster_key="") for entry in result.entries))

    async def fetch_poster_snapshot(self, payload=None):
        http = _http(self.poster_snapshot() if payload is None else payload)
        result = await _provider(clock=lambda: date(2026, 9, 10)).fetch(http)
        http.get_json.assert_awaited_once_with(_URL, params=_PARAMS)
        http.get_text.assert_not_awaited()
        return result

    async def test_real_image_cover_preserves_album_identity_all_dates_and_free_progress(self):
        payload = self.poster_snapshot()
        expected = {str(row["album_id"]): row["image_cover"][8:] for row in self.poster_rows(payload)
                    if row.get("channel_id") == 4 and row.get("content_type") == "FEATURE_FILM" and row.get("is_episode") is False}
        result = await self.fetch_poster_snapshot(payload)
        for row in self.poster_rows(payload):
            row.pop("image_cover", None)
        baseline = await self.fetch_poster_snapshot(payload)
        self.assertEqual(self.without_posters(result), baseline)
        self.assertEqual({e.source_id: e.platform_poster_key for e in result.entries}, expected)
        self.assertEqual((len(result.entries), sum(len(e.events) for e in result.entries)), (44, 84))
        for entry in result.entries:
            self.assertEqual(entry.stable_id, f"iqiyi:{entry.source_id}")
            self.assertEqual((entry.free_progress, entry.free_weekdays, entry.free_schedule), ("", (), ""))
        historical = await _provider().fetch(_http())
        self.assertTrue(all(not e.platform_poster_key for e in historical.entries))
        self.assertEqual((len(historical.entries), sum(len(e.events) for e in historical.entries)), (40, 72))

    async def test_missing_bad_original_cover_never_uses_back_image_or_hover_or_drops_programme(self):
        template = self.poster_snapshot()
        for row in self.poster_rows(template):
            row.pop("image_cover", None)
        baseline = await self.fetch_poster_snapshot(template)
        first_cover = self.poster_snapshot()["items"][0]["video"][0]["data"][0]["image_cover"]
        bad_images = (None, "", False, 7, {}, [], "x" * 2049, "https://[broken",
            "data:image/png;base64,AAAA", "javascript:alert(1)", "http://127.0.0.1/private",
            "https://pic0.iqiyipic.com.evil.invalid/image/a.webp",
            "https://user:password@pic0.iqiyipic.com/image/a.webp",
            "https://pic0.iqiyipic.com/../private", first_cover + "#fragment")
        for image in bad_images:
            with self.subTest(image=image):
                payload = copy.deepcopy(template)
                for row in self.poster_rows(payload):
                    row.update(image_cover=image, back_image=first_cover, image_url_normal=first_cover,
                               album_image_url_hover=first_cover)
                self.assertEqual(await self.fetch_poster_snapshot(payload), baseline)

    async def test_same_album_different_valid_covers_clear_only_poster_independent_of_order(self):
        template = self.poster_snapshot()
        original = template["items"][0]["video"][0]["data"][0]
        conflict = copy.deepcopy(original)
        conflict["image_cover"] = template["items"][0]["video"][0]["data"][1]["image_cover"]
        baseline = self.without_posters(await self.fetch_poster_snapshot(template))
        for first in (True, False):
            with self.subTest(conflict_first=first):
                payload = copy.deepcopy(template)
                rows = payload["items"][0]["video"][0]["data"]
                rows.insert(0, conflict) if first else rows.append(conflict)
                result = await self.fetch_poster_snapshot(payload)
                self.assertEqual(self.without_posters(result), baseline)
                entry = next(e for e in result.entries if e.source_id == str(original["album_id"]))
                self.assertEqual(entry.platform_poster_key, "")
                self.assertEqual(sum(bool(e.platform_poster_key) for e in result.entries), 43)

    async def test_duplicate_missing_bad_or_equivalent_cover_keeps_one_verified_key(self):
        template = self.poster_snapshot()
        original = template["items"][0]["video"][0]["data"][0]
        url = original["image_cover"]
        baseline = await self.fetch_poster_snapshot(template)
        for image in (None, "https://127.0.0.1/private", url, "http:" + url[6:], url[6:]):
            for first in (True, False):
                with self.subTest(image=image, duplicate_first=first):
                    payload = copy.deepcopy(template)
                    duplicate = copy.deepcopy(original)
                    duplicate["image_cover"] = image
                    rows = payload["items"][0]["video"][0]["data"]
                    rows.insert(0, duplicate) if first else rows.append(duplicate)
                    self.assertEqual(await self.fetch_poster_snapshot(payload), baseline)

    async def test_title_conflict_keeps_existing_identity_rejection_even_with_valid_image(self):
        template = self.poster_snapshot()
        original = template["items"][0]["video"][0]["data"][0]
        baseline = await self.fetch_poster_snapshot(template)
        for first in (True, False):
            payload = copy.deepcopy(template)
            conflict = copy.deepcopy(original)
            conflict["title"] = "同专辑ID但不同节目名（合成冲突）"
            rows = payload["items"][0]["video"][0]["data"]
            rows.insert(0, conflict) if first else rows.append(conflict)
            result = await self.fetch_poster_snapshot(payload)
            self.assertEqual(result.entries, tuple(e for e in baseline.entries if e.source_id != str(original["album_id"])))

    async def test_non_anime_or_unverified_date_group_cannot_lend_or_poison_cover(self):
        template = self.poster_snapshot()
        group = template["items"][0]["video"][0]
        original = group["data"][0]
        baseline = await self.fetch_poster_snapshot(template)
        for fields in ({"channel_id": 2}, {"content_type": "PREVUE"}, {"is_episode": True}, {"isAd": True}):
            payload = copy.deepcopy(template)
            duplicate = copy.deepcopy(original)
            duplicate.update(fields, image_cover=group["data"][1]["image_cover"])
            payload["items"][0]["video"][0]["data"].insert(0, duplicate)
            self.assertEqual(await self.fetch_poster_snapshot(payload), baseline)
        for subtitle in ("09-06", "invalid"):
            payload = copy.deepcopy(template)
            other = copy.deepcopy(group)
            other.update(block_id="jmd_Sun", title="周日", sub_title=subtitle)
            other["data"] = [copy.deepcopy(original)]
            other["data"][0]["image_cover"] = group["data"][1]["image_cover"]
            payload["items"][0]["video"].insert(0, other)
            self.assertEqual(await self.fetch_poster_snapshot(payload), baseline)

    def test_new_cover_fixture_provenance_has_exact_observed_hosts_and_one_verified_sample(self):
        provenance = json.loads((_FIXTURES / "platform_images_20260910_provenance.json").read_text("utf-8"))
        self.assertEqual(provenance["kind"], "sanitized_live_platform_poster_fields")
        self.assertEqual(provenance["image_field"], "image_cover")
        self.assertEqual(provenance["observed_image_hosts"], {
            "pic0.iqiyipic.com": 8, "pic1.iqiyipic.com": 8, "pic2.iqiyipic.com": 13,
            "pic3.iqiyipic.com": 4, "pic4.iqiyipic.com": 5, "pic5.iqiyipic.com": 20,
            "pic6.iqiyipic.com": 7, "pic7.iqiyipic.com": 6, "pic8.iqiyipic.com": 7,
            "pic9.iqiyipic.com": 6})
        data = (_FIXTURES / provenance["fixture"]["fixture"]).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), provenance["fixture"]["sha256"])
        self.assertEqual(len(data), provenance["fixture"]["bytes"])
        for field in (b"session", b"pingback", b"authorization", b"back_image", b"image_url_normal", b"album_image_url_hover"):
            self.assertNotIn(field, data)
        for name in ("schedule", "image_sample"):
            evidence = provenance[name]
            self.assertIsNotNone(datetime.fromisoformat(evidence["started_at"]).tzinfo)
            self.assertEqual((evidence["status"], evidence["retries"], evidence["max_redirects"]), (200, 0, 0))
            self.assertLess(evidence["bytes"], 2 * 1024 * 1024)
            self.assertEqual(len(evidence["sha256"]), 64)
            self.assertTrue(evidence["pin_resolved_address"])
            self.assertFalse(evidence["credentials"])
        self.assertEqual(provenance["schedule"]["params"], _PARAMS)
        self.assertEqual(provenance["image_sample"]["mime"], "image/webp")
        self.assertTrue(provenance["image_sample"]["mime_magic_match"])
        self.assertEqual(provenance["queries"], [""])


if __name__ == "__main__":
    unittest.main()
