"""日历API沿用探索鉴权/开关/CSRF，海报只经过签名同源代理。"""
from __future__ import annotations

import copy
from unittest.mock import Mock, patch

from tests.test_discovery_api import _BaseClientTests
from app.discovery.calendar.service import CalendarService
from app.routes.discovery_image import decode_poster_token


SNAPSHOT = {
    "timezone": "Asia/Shanghai", "today": "2026-09-09", "week_start": "2026-09-07",
    "days": [{"date": "2026-09-07", "weekday": 1, "label": "周一", "is_today": False, "items": [{
        "stable_id": "tencent:show", "source": "tencent", "source_id": "show", "title": "免费动漫", "category": "animation",
        "poster_key": "poster.jpg", "tmdb_id": "42", "free_progress": "免费至第2集",
    }]}], "unscheduled": [], "sources": [], "refreshing": False, "updated_at": "", "items_count": 1,
}


class FreeCalendarAPITests(_BaseClientTests):
    def setUp(self):
        super().setUp()
        from app import database
        database.init_db()  # _BaseClientTests 已隔离 DB_PATH。
        self.service = Mock(spec=CalendarService)
        self.service.get_week.side_effect = lambda **kw: copy.deepcopy(SNAPSHOT)
        self.enterContext(patch("app.routes.discovery_api.get_calendar_service", return_value=self.service))

    def test_api_rejects_legacy_non_animation_cards_before_identity_or_poster_rendering(self):
        self.authenticate()
        data = copy.deepcopy(SNAPSHOT)
        data["days"][0]["items"].append({**data["days"][0]["items"][0], "stable_id": "tencent:legacy_tv",
                                        "category": "tv", "title": "旧电视剧"})
        data["items_count"] = 2
        self.service.get_week.side_effect = lambda **kw: data
        result = self.client.get("/api/discovery/calendar").json()
        self.assertEqual(result["items_count"], 1)
        self.assertEqual([card["category"] for card in result["days"][0]["items"]], ["animation"])

    def test_api_and_page_require_login_before_any_source_read(self):
        self.assertEqual(self.client.get("/api/discovery/calendar").status_code, 401)
        self.assertEqual(self.client.get("/discovery/calendar", follow_redirects=False).status_code, 302)
        self.service.get_week.assert_not_called()

    def test_read_encodes_poster_without_exposing_remote_url(self):
        self.authenticate()
        response = self.client.get("/api/discovery/calendar")
        self.assertEqual(response.status_code, 200)
        card = response.json()["days"][0]["items"][0]
        self.assertNotIn("poster_key", card)
        self.assertTrue(card["poster_url"].startswith("/discovery-poster/tmdb/"))
        token = card["poster_url"].rsplit("/", 1)[1]
        self.assertEqual(decode_poster_token("tmdb", token), "poster.jpg")

    def test_refresh_requires_csrf_and_rejects_custom_source_urls(self):
        headers = self.authenticate()
        self.assertEqual(self.client.post("/api/discovery/calendar/refresh", json={}).status_code, 403)
        self.assertEqual(self.client.post("/api/discovery/calendar/refresh", json={}, headers=headers).status_code, 200)
        self.service.get_week.assert_called_once_with(force=True)
        self.assertEqual(self.client.post("/api/discovery/calendar/refresh", json={"url": "https://evil.invalid"}, headers=headers).status_code, 400)
        self.assertEqual(self.client.get("/api/discovery/calendar?source=evil").status_code, 400)

    def test_calendar_respects_discovery_feature_gate(self):
        self.authenticate()
        from app import config
        original = config.get_bool
        with patch.object(config, "get_bool", side_effect=lambda key, *args: False if key == "DISCOVERY_ENABLED" else original(key, *args)):
            self.assertEqual(self.client.get("/api/discovery/calendar").status_code, 404)
            self.assertEqual(self.client.get("/discovery/calendar").status_code, 404)
        self.service.get_week.assert_not_called()

    def test_cached_bad_image_and_tmdb_id_do_not_become_external_links(self):
        self.authenticate()
        data = copy.deepcopy(SNAPSHOT)
        data["days"][0]["items"][0].update(poster_key="https://127.0.0.1/private", tmdb_id="javascript:alert(1)")
        self.service.get_week.side_effect = lambda **kw: data
        card = self.client.get("/api/discovery/calendar").json()["days"][0]["items"][0]
        self.assertEqual(card["poster_url"], "")
        self.assertEqual(card["tmdb_id"], "")

    def test_mixed_covers_keep_matching_identity_and_signed_provider_tokens(self):
        self.authenticate()
        data = copy.deepcopy(SNAPSHOT)
        raw = data["days"][0]["items"][0]
        raw.update(douban_id="1234", poster_provider="douban", poster_key="img1.doubanio.com/view/photo/l/public/p1.jpg",
                   tmdb_poster_key="poster.jpg", douban_poster_key="img1.doubanio.com/view/photo/l/public/p1.jpg")
        self.service.get_week.side_effect = lambda **kw: data
        card = self.client.get("/api/discovery/calendar").json()["days"][0]["items"][0]
        self.assertTrue(card["poster_urls"][0].startswith("/discovery-poster/douban/"))
        self.assertTrue(card["poster_urls"][1].startswith("/discovery-poster/tmdb/"))
        self.assertNotIn("douban_poster_key", card)
        self.assertEqual(card["watchlist"]["provider"], "tmdb")
        self.assertEqual(decode_poster_token("tmdb", card["watchlist"]["poster_token"]), "poster.jpg")
        self.assertEqual(card["detail_url"], "/discovery?detail_provider=tmdb&detail_type=tv&detail_id=42")

    def test_douban_fallback_can_be_saved_using_existing_discovery_watchlist(self):
        headers = self.authenticate()
        data = copy.deepcopy(SNAPSHOT)
        data["days"][0]["items"][0].update(tmdb_id="", douban_id="1234", poster_provider="douban",
                poster_key="img1.doubanio.com/view/photo/l/public/p1.jpg")
        self.service.get_week.side_effect = lambda **kw: data
        card = self.client.get("/api/discovery/calendar").json()["days"][0]["items"][0]
        identity = card["watchlist"]
        self.assertFalse(identity["in_watchlist"])
        response = self.client.post("/api/discovery/watchlist", headers=headers, json={
            **{key: identity[key] for key in ("provider", "external_id", "media_type", "poster_token")},
            "title": card["title"], "year": "2026",
        })
        self.assertEqual(response.status_code, 200)
        updated = self.client.get("/api/discovery/calendar").json()["days"][0]["items"][0]
        self.assertTrue(updated["watchlist"]["in_watchlist"])
        self.assertEqual(self.client.delete("/api/discovery/watchlist/douban/tv/1234", headers=headers).status_code, 200)
        self.assertFalse(self.client.get("/api/discovery/calendar").json()["days"][0]["items"][0]["watchlist"]["in_watchlist"])

    def test_no_trusted_media_identity_disables_watchlist_without_source_id_impersonation(self):
        self.authenticate()
        data = copy.deepcopy(SNAPSHOT)
        data["days"][0]["items"][0].update(tmdb_id="", douban_id="", source_id="42", poster_key="")
        self.service.get_week.side_effect = lambda **kw: data
        card = self.client.get("/api/discovery/calendar").json()["days"][0]["items"][0]
        self.assertIsNone(card["watchlist"])
        self.assertEqual(card["detail_url"], "")
