"""三平台追漫日历Agent只读动作；真实结构回放/缓存与公共投影均隔离网络及生产DB。"""
from __future__ import annotations

import copy
from datetime import date, datetime, timedelta
from pathlib import Path
import json
import unittest
from unittest.mock import Mock, patch

import tests  # noqa: F401 -- 必须先隔离环境/数据库，再导入应用。
from tests.support import isolated_test_database
from app.agent.calendar_actions import anime_calendar, anime_calendar_arguments
from app.agent.errors import AgentToolError
from app.agent.kernel.projection import DefaultProjector
from app.discovery.cache import DiscoveryCache
from app.discovery.calendar.models import SOURCE_NAMES, SourceUnavailable
from app.discovery.calendar.providers.youku import _dynamic_calendar_data, _parse_calendar_data
from app.discovery.calendar.service import CalendarService

FIXTURES = Path(__file__).with_name("fixtures") / "calendar" / "youku"


def snapshot(*, today="2026-09-10", start="2026-09-07", status="partial"):
    monday = date.fromisoformat(start)
    return {"timezone": "Asia/Shanghai", "today": today, "week_start": start,
            "days": [{"date": (monday + timedelta(days=i)).isoformat(), "items": []} for i in range(7)],
            "sources": [{"id": source, "name": name, "status": status,
                         "fetched_at": "2026-09-10T04:24:05+08:00", "message": "真实公开来源，有限覆盖"}
                        for source, name in SOURCE_NAMES.items()],
            "refreshing": False}


def card(index=1, *, source="youku", day="2026-09-10", **changes):
    value = {"source": source, "source_id": f"show_{index}", "stable_id": f"{source}:show_{index}",
             "title": f"测试动漫{index}", "category": "animation", "free_progress": "", "stale": False,
             "events": [{"date": day, "weekday": date.fromisoformat(day).isoweekday(),
                         "update_time": "10:00", "schedule": "10:00 VIP更新1话", "audience": "member"}],
             "url": "https://v.youku.com/private?token=should-not-be-returned",
             "platform_poster_key": "private-poster-key", "poster_urls": ["private-token"],
             "tmdb_id": "1234", "watchlist": {"provider": "tmdb", "external_id": "1234"}}
    value.update(changes)
    return value


class AgentAnimeCalendarTests(unittest.TestCase):
    def setUp(self):
        self.data = snapshot()
        self.service = Mock(get_week=Mock(side_effect=lambda: copy.deepcopy(self.data)))
        self.getter = self.enterContext(patch("app.agent.calendar_actions.get_calendar_service", return_value=self.service))
        self.enabled = self.enterContext(patch("app.agent.calendar_actions.config.get_bool", return_value=True))
        self.network_attempts = []
        def forbidden(*args, **kwargs):
            self.network_attempts.append(True)
            raise AssertionError("日历Agent回归不得发出真实网络")
        for name in ("getaddrinfo", "create_connection", "socket.connect", "socket.connect_ex"):
            self.enterContext(patch("socket." + name, side_effect=forbidden))

    def tearDown(self):
        self.assertEqual(self.network_attempts, [])

    def test_defaults_and_strict_parameters_reject_unknown_privileged_fields(self):
        self.assertEqual(anime_calendar_arguments({}), {"day": "today", "source": "all", "query": "", "page": 1, "limit": 20})
        self.assertEqual(anime_calendar_arguments({"day": "monday", "source": "youku", "query": "  测试 ", "page": 2, "limit": 10})["query"], "测试")
        for arguments in (None, [], {"source": None}, {"source": []}, {"source": "bangumi"}, {"source": "优酷"},
                          {"day": True}, {"day": "2026-09-31"}, {"day": "2026-9-10"}, {"day": "next_week"},
                          {"query": "x" * 81}, {"query": None}, {"query": "line\nbreak"},
                          {"query": "https://evil.invalid/secret"}, {"query": "cookie=private-session"},
                          {"query": "/home/private/file"}, {"page": 0}, {"page": True}, {"page": 101},
                          {"limit": 21}, {"limit": "1"}, {"refresh": True}, {"force": True}, {"url": "secret"},
                          {1: "wrong key"}, {"cookie": "private"}, {"auth": "private"}):
            with self.subTest(arguments=arguments), self.assertRaises(AgentToolError):
                anime_calendar(arguments)
        self.getter.assert_not_called()

    def test_disabled_feature_does_not_create_calendar_service(self):
        self.enabled.return_value = False
        result = anime_calendar({})
        self.assertEqual((result.ok, result.status), (False, "disabled"))
        self.getter.assert_not_called()

    def test_today_source_keyword_filter_is_case_insensitive_readonly_and_allowlisted(self):
        self.data["days"][3]["items"] = [card(1, title="DR. Stone"), card(2), card(3, source="iqiyi")]
        result = anime_calendar({"source": "youku", "query": "dr. stone"})
        self.assertEqual((result.ok, result.status, result.data["returned"]), (True, "partial", 1))
        self.assertEqual(result.data["requested_dates"], ["2026-09-10"])
        self.service.get_week.assert_called_once_with()
        self.assertEqual([source["id"] for source in result.data["sources"]], ["youku"])
        item = result.data["items"][0]
        self.assertEqual((item["title"], item["audience"], item["free_progress"]), ("DR. Stone", "member", ""))
        self.assertEqual(item["date"], "2026-09-10")
        serialized = json.dumps(result.to_dict())
        for forbidden in ("private", "token", "poster", "tmdb_id", "watchlist", "url=", "trackInfo", "https://"):
            self.assertNotIn(forbidden, serialized)
        self.assertTrue(result.evidence)
        self.assertEqual(result.evidence[0].source, "anime_calendar")

    def test_shanghai_snapshot_not_host_clock_selects_today_tomorrow_and_weekdays(self):
        for day, expected in (("today", "2026-09-10"), ("tomorrow", "2026-09-11"), ("monday", "2026-09-07"),
                              ("sunday", "2026-09-13"), ("2026-09-09", "2026-09-09")):
            with self.subTest(day=day):
                self.assertEqual(anime_calendar({"day": day}).data["requested_dates"], [expected])
        result = anime_calendar({"day": "week"})
        self.assertEqual(result.data["requested_dates"], [f"2026-09-{day:02}" for day in range(7, 14)])
        self.assertEqual((result.data["timezone"], result.data["week_end"]), ("Asia/Shanghai", "2026-09-13"))

    def test_cross_year_week_is_preserved_and_tomorrow_out_of_week_is_not_rebased(self):
        self.data = snapshot(today="2027-01-01", start="2026-12-28")
        result = anime_calendar({"day": "week"})
        self.assertEqual((result.data["requested_dates"][0], result.data["requested_dates"][-1]), ("2026-12-28", "2027-01-03"))
        self.data["today"] = "2027-01-03"
        result = anime_calendar({"day": "tomorrow"})
        self.assertEqual((result.ok, result.status, result.data["items"]), (False, "unsupported_range", []))
        self.assertEqual(result.data["requested_dates"], ["2027-01-04"])
        for day in ("2026-12-21", "2027-01-04", "2030-01-01"):
            self.assertEqual(anime_calendar({"day": day}).status, "unsupported_range")

    def test_explicit_events_keep_member_free_unknown_and_missing_time_independent(self):
        base = card()
        member = base["events"][0]
        free = dict(member, update_time="", audience="free", schedule="非会员更新，时刻未注明")
        unknown = dict(member, update_time="14:00", audience="unknown", schedule="14:00更新1话")
        base.update(events=[member, free, unknown, member], free_progress="已核验免费至第3集")
        self.data["days"][3]["items"] = [base, copy.deepcopy(base)]
        result = anime_calendar({})
        self.assertEqual((result.data["total"], result.data["total_programmes"]), (3, 1))
        by_audience = {row["audience"]: row for row in result.data["items"]}
        self.assertEqual(by_audience["free"]["update_time"], "")
        self.assertEqual(by_audience["member"]["update_time"], "10:00")
        self.assertEqual(by_audience["unknown"]["update_time"], "14:00")
        self.assertTrue(all(row["free_progress"] == "已核验免费至第3集" for row in by_audience.values()))

    def test_page_limit_and_has_more_describe_events_not_full_site(self):
        self.data["days"][3]["items"] = [card(i) for i in range(25, 0, -1)]
        first = anime_calendar({"limit": 10})
        second = anime_calendar({"limit": 10, "page": 2})
        last = anime_calendar({"limit": 10, "page": 3})
        beyond = anime_calendar({"limit": 10, "page": 4})
        self.assertEqual((first.data["total"], first.data["total_programmes"], first.data["returned"]), (25, 25, 10))
        self.assertTrue(first.data["has_more"])
        self.assertFalse(last.data["has_more"])
        self.assertEqual(last.data["returned"], 5)
        self.assertEqual((beyond.ok, beyond.status), (True, "empty"))
        self.assertIn("本页", beyond.summary)
        self.assertTrue({row["stable_id"] for row in first.data["items"]}.isdisjoint({row["stable_id"] for row in second.data["items"]}))
        old_order = [row["stable_id"] for row in first.data["items"]]
        self.data["days"][3]["items"].reverse()
        self.assertEqual(old_order, [row["stable_id"] for row in anime_calendar({"limit": 10}).data["items"]])

    def test_loading_unavailable_empty_and_stale_are_distinguished_without_retry(self):
        for status, expected, ok in (("loading", "loading", False), ("unavailable", "unavailable", False),
                                      ("stale", "unavailable", False), ("partial", "empty", True), ("ok", "empty", True)):
            with self.subTest(status=status):
                self.service.get_week.reset_mock()
                self.data["sources"][2]["status"] = status
                result = anime_calendar({"source": "youku"})
                self.assertEqual((result.ok, result.status), (ok, expected))
                self.service.get_week.assert_called_once_with()
                self.assertIn("不代表", result.data["scope_note"])
                if status == "loading":
                    self.assertEqual(result.data["retry_after"], 5)
        self.data["sources"][2]["status"] = "stale"
        self.data["days"][3]["items"] = [card()]
        result = anime_calendar({"source": "youku"})
        self.assertTrue(result.data["items"][0]["stale"])
        self.assertEqual(result.status, "partial")

    def test_selected_healthy_source_is_not_failed_by_other_unavailable_sources(self):
        self.data["sources"][0]["status"] = "unavailable"
        self.data["days"][3]["items"] = [card(1, source="tencent"), card(2, source="youku")]
        result = anime_calendar({"source": "youku"})
        self.assertEqual((result.status, result.data["total"]), ("partial", 1))
        self.assertEqual(result.data["sources"][0]["id"], "youku")
        self.assertEqual(anime_calendar({}) .data["total"], 1)

    def test_source_failure_and_optional_metadata_refresh_do_not_erase_valid_facts(self):
        self.data["refreshing"] = True
        self.data["days"][3]["items"] = [card()]
        self.data["sources"][0]["status"] = "unavailable"
        result = anime_calendar({})
        self.assertTrue(result.ok)
        self.assertTrue(result.data["refreshing"])
        self.assertEqual(result.data["returned"], 1)
        self.assertEqual(result.data["sources"][0]["status"], "unavailable")

    def test_malformed_cross_date_non_anime_and_sensitive_fields_do_not_reach_agent(self):
        good = card()
        invalid = [card(2, category="tv"), card(3, source="unknown"), card(4, title="cookie=private"),
                   card(5, source_id="/home/private"), card(6, day="2026-09-03"), card(7, events=None)]
        for change in ({"audience": "VIP"}, {"update_time": "25:00"}, {"update_time": False}, {"date": "2026-09-11"}):
            event = dict(good["events"][0], **change)
            invalid.append(card(8, events=[event]))
        self.data["days"][3]["items"] = [good, *invalid]
        self.data["sources"][2].update(message="cookie=secret; https://evil.invalid/private", fetched_at="private")
        result = anime_calendar({"source": "youku"})
        self.assertEqual(result.data["total"], 1)
        self.assertEqual(result.data["sources"][0]["fetched_at"], "")
        self.assertNotIn("private", json.dumps(result.to_dict()))
        self.assertNotIn("secret", json.dumps(result.to_dict()))

    def test_bad_snapshot_or_exception_is_unavailable_without_raw_error_text(self):
        base = snapshot()
        variants = [None, [], {}, dict(base, timezone="UTC"), dict(base, today="2026-09-14"),
                    dict(base, week_start="2026-09-08"), dict(base, days=base["days"][:6])]
        wrong_day = copy.deepcopy(base)
        wrong_day["days"][3]["date"] = "2026-09-03"
        variants.append(wrong_day)
        for value in variants:
            with self.subTest(value=type(value).__name__):
                self.data = value
                result = anime_calendar({})
                self.assertEqual((result.ok, result.status), (False, "unavailable"))
        self.service.get_week.side_effect = SourceUnavailable("cookie=private; https://evil.invalid/secret")
        result = anime_calendar({})
        self.assertEqual(result.status, "unavailable")
        self.assertNotIn("private", json.dumps(result.to_model_dict()))
        self.assertNotIn("evil", json.dumps(result.to_model_dict()))

    def test_verified_legacy_free_rule_has_dates_but_free_progress_alone_does_not(self):
        known = card(1, events=[], free_progress="免费观看第3集", free_weekdays=(4,),
                     free_schedule="非会员每周四更新", free_update_time="")
        no_rule = card(2, events=[], free_progress="免费观看第3集")
        wrong_day = card(3, events=[], free_weekdays=(3,), free_schedule="非会员每周三更新")
        self.data["days"][3]["items"] = [known, no_rule, wrong_day]
        result = anime_calendar({})
        self.assertEqual(result.data["total"], 1)
        row = result.data["items"][0]
        self.assertEqual((row["date"], row["update_time"], row["audience"]), ("2026-09-10", "", "free"))

    def test_exact_calendar_link_survives_model_projection_without_loosening_private_paths(self):
        result = anime_calendar({})
        projected = DefaultProjector().project(result)
        self.assertEqual(projected.public_content["data"]["calendar_url"], "/discovery/calendar")
        self.assertEqual(json.loads(projected.model_content)["data"]["calendar_url"], "/discovery/calendar")
        for target in ("/discovery/calendar?x=1", "/discovery/calendar#x", "/home/secret.txt", "//host/path",
                       "https://example.com/discovery/calendar", "/discovery/calendar/", " /discovery/calendar", None,
                       ["/discovery/calendar"], {"nested": "/discovery/calendar"}):
            with self.subTest(target=target):
                model = json.loads(DefaultProjector().project({"data": {"calendar_url": target}}).model_content)
                self.assertEqual(model["data"]["calendar_url"], "[页面地址无效]")
        model = json.loads(DefaultProjector().project({"data": {"path": "/discovery/calendar", "cookie": "never-public"}}).model_content)
        self.assertIn("隐藏", model["data"]["path"])
        self.assertNotIn("never-public", json.dumps(model))

    def test_maximum_sized_page_keeps_facts_under_default_projector_context_limit(self):
        self.data["days"][3]["items"] = [card(i, source_id="s" * 120 + str(i), title="题" * 200,
            free_progress="免" * 120, events=[{"date": "2026-09-10", "update_time": "10:00",
                "schedule": "排" * 400, "audience": "member"}]) for i in range(20)]
        for source in self.data["sources"]:
            source["message"] = "范" * 300
        result = anime_calendar({"query": "题" * 80})
        projected = DefaultProjector().project(result)
        model = json.loads(projected.model_content)
        self.assertFalse(model.get("truncated", False))
        self.assertLess(len(projected.model_content), 24000)
        self.assertEqual(len(model["data"]["items"]), 20)

    def test_json_escaping_compacts_text_only_and_never_drops_events_or_pagination(self):
        for mode in ("schedule", "title", "all"):
            with self.subTest(mode=mode):
                title = '"' * 200 if mode in {"title", "all"} else "题" * 200
                progress = '"' * 120 if mode == "all" else "免" * 120
                schedule = '"更新"' * 80 if mode != "title" else "排" * 400
                self.data["days"][3]["items"] = [card(i, source="tencent", source_id="s" * 126 + f"{i:02}",
                    title=title, free_progress=progress,
                    events=[{"date": "2026-09-10", "update_time": "10:00", "schedule": schedule, "audience": "member"}])
                    for i in range(21)]
                for source in self.data["sources"]:
                    source["message"] = '"' * 300 if mode == "all" else "范" * 300
                result = anime_calendar({"day": "week"})
                public = copy.deepcopy(result.data)
                projected = DefaultProjector().project(result)
                model = json.loads(projected.model_content)
                self.assertFalse(model.get("truncated", False))
                self.assertEqual(len(model["data"]["items"]), 20)
                self.assertEqual(model["data"]["calendar_url"], "/discovery/calendar")
                self.assertTrue(model["data"]["has_more"])
                self.assertEqual((model["data"]["page"], model["data"]["total"]), (1, 21))
                self.assertLess(len(projected.model_content), 24_000)
                self.assertEqual(result.data, public)
                self.assertTrue(all("schedule" in row and "source_id" in row for row in result.data["items"]))
                for original, row in zip(public["items"], model["data"]["items"]):
                    for key in ("stable_id", "source", "title", "date", "update_time", "audience", "free_progress", "stale"):
                        self.assertEqual(row[key], original[key])
                self.assertEqual([(s["id"], s["status"], s["fetched_at"]) for s in model["data"]["sources"]],
                                 [(s["id"], s["status"], s["fetched_at"]) for s in public["sources"]])
                self.assertIn("model_omitted_fields", model["data"])
                if "items.schedule" in model["data"]["model_omitted_fields"]:
                    self.assertIn("不得据此推断集数", model["data"]["model_view_note"])

    def test_last_permitted_page_marks_limit_without_recommending_invalid_next_page(self):
        self.data["days"][3]["items"] = [card(i) for i in range(101)]
        result = anime_calendar({"page": 100, "limit": 1})
        self.assertEqual((result.data["total"], result.data["returned"]), (101, 1))
        self.assertTrue(result.data["pagination_limited"])
        self.assertFalse(result.data["has_more"])
        self.assertIn("收窄", "".join(result.suggestions))
        self.assertIn("第100页", "".join(result.suggestions))

    def test_original_source_snapshot_is_not_mutated(self):
        self.data["days"][3]["items"] = [card()]
        before = copy.deepcopy(self.data)
        self.service.get_week.side_effect = lambda: self.data
        anime_calendar({"query": "测试"})
        self.assertEqual(self.data, before)

    def test_real_source_fixture_to_shared_service_cache_to_agent_replays_all_92_events(self):
        self.enterContext(isolated_test_database())
        now = datetime(2026, 9, 10, 12)
        raw = json.loads((FIXTURES / "dynamic_week_20260910.json").read_text())
        result = _parse_calendar_data(_dynamic_calendar_data(raw), now.date())
        class Source:
            source = "youku"
            allowed_hosts = frozenset({"www.youku.com", "acs.youku.com"})
            calls = 0
            async def fetch(self, http):
                self.calls += 1
                return result
        class NoNetwork:
            def __init__(self, hosts):
                self.hosts = hosts
            async def aclose(self):
                pass
        class Metadata:
            def enrich(self, entries):
                return [entry.to_dict() for entry in entries]
        source = Source()
        service = CalendarService(providers={"youku": source}, cache=DiscoveryCache(clock=lambda: now),
                                  metadata=Metadata(), clock=lambda: now, monotonic=lambda: 100.,
                                  submit=lambda fn: fn(), http_factory=NoNetwork)
        self.addCleanup(service.shutdown)
        self.getter.return_value = service
        rows = []
        for page in range(1, 6):
            response = anime_calendar({"day": "week", "source": "youku", "page": page})
            self.assertEqual((response.status, response.data["total"], response.data["total_programmes"]), ("partial", 92, 47))
            rows.extend(response.data["items"])
        self.assertEqual(source.calls, 1)
        self.assertEqual((len(rows), sum(bool(row["update_time"]) for row in rows)), (92, 91))
        self.assertEqual({row["date"] for row in rows}, {f"2026-09-{day:02}" for day in range(7, 14)})
        self.assertTrue(all(not row["free_progress"] for row in rows))
        serialized = json.dumps(rows)
        for forbidden in ("platform_poster_key", "trackInfo", "_m_h5_tk", "poster_urls", "watchlist", "tmdb_id"):
            self.assertNotIn(forbidden, serialized)
