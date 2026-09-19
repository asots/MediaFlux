from __future__ import annotations

import json
import unittest
from urllib.parse import urlsplit

import requests

from app.clients.douban_authenticated import (
    DoubanAuthenticatedClient,
    normalize_dbcl2,
)
from app.discovery.models import ProviderInvalidResponse, ProviderNotConfigured


class CapturingSession(requests.Session):
    def __init__(self, payload):
        super().__init__()
        self.payload = payload
        self.prepared_requests = []

    def send(self, request, **kwargs):
        del kwargs
        self.prepared_requests.append(request)
        response = requests.Response()
        response.status_code = 200
        response.headers = {"Content-Type": "application/json; charset=utf-8"}
        response.url = request.url
        response.request = request
        response._content = json.dumps(self.payload, ensure_ascii=False).encode("utf-8")
        response._content_consumed = True
        return response


class AuthStatusSession(requests.Session):
    def __init__(self, *, status_code=200, body="", headers=None, error=None):
        super().__init__()
        self.status_code = status_code
        self.body = body.encode("utf-8")
        self.response_headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.error = error
        self.prepared_requests = []

    def send(self, request, **kwargs):
        del kwargs
        self.prepared_requests.append(request)
        if self.error is not None:
            raise self.error
        response = requests.Response()
        response.status_code = self.status_code
        response.headers = self.response_headers
        response.url = request.url
        response.request = request
        response._content = self.body
        response._content_consumed = True
        return response


class DoubanDbcl2NormalizationTests(unittest.TestCase):
    def test_accepts_raw_quoted_and_full_cookie_but_retains_only_value(self):
        expected = "123456789:test-dbcl2-value"
        cases = (
            expected,
            f'dbcl2="{expected}"',
            f'll="118281"; bid=abc; dbcl2="{expected}"; ck=xyz',
        )

        for value in cases:
            with self.subTest(value=value):
                self.assertEqual(normalize_dbcl2(value), expected)

    def test_rejects_missing_dbcl2_controls_and_oversized_values(self):
        invalid = (
            "bid=abc; ck=xyz",
            "dbcl2=value\r\nX-Test: injected",
            "dbcl2=value\x00suffix",
            "x" * 513,
        )

        for value in invalid:
            with self.subTest(value=value[:40]):
                with self.assertRaises(ValueError):
                    normalize_dbcl2(value)

    def test_empty_value_is_valid_as_disabled_optional_fallback(self):
        self.assertEqual(normalize_dbcl2("   "), "")
        self.assertEqual(normalize_dbcl2("dbcl2=; ck=value"), "")


class DoubanAuthenticatedClientTests(unittest.TestCase):
    DBCL2 = "123456789:test-dbcl2-value"

    def test_dedicated_session_scopes_only_dbcl2_to_fixed_movie_host(self):
        session = CapturingSession(
            {
                "subjects": [
                    {
                        "id": "1292052",
                        "title": "肖申克的救赎",
                        "cover": "https://img1.doubanio.com/view/photo/test.webp",
                        "rate": "9.7",
                    }
                ]
            }
        )
        session.headers.update({"Authorization": "Bearer stale", "Cookie": "sid=stale"})
        session.cookies.set("sid", "stale", domain="attacker.invalid", path="/")
        client = DoubanAuthenticatedClient(
            dbcl2=self.DBCL2,
            session=session,
            page_size=20,
        )

        page = client.list_items("movie_hot", "movie", 1, {})

        self.assertEqual(len(page.items), 1)
        prepared = session.prepared_requests[0]
        parsed = urlsplit(prepared.url)
        self.assertEqual((parsed.scheme, parsed.hostname, parsed.path), (
            "https", "movie.douban.com", "/j/search_subjects",
        ))
        self.assertEqual(prepared.headers.get("Cookie"), f"dbcl2={self.DBCL2}")
        self.assertNotIn("Authorization", prepared.headers)
        self.assertFalse(session.trust_env)

    def test_blank_dbcl2_never_performs_network_request(self):
        session = CapturingSession({"subjects": []})
        client = DoubanAuthenticatedClient(dbcl2="", session=session)

        with self.assertRaises(ProviderNotConfigured):
            client.list_items("movie_hot", "movie", 1, {})

        self.assertEqual(session.prepared_requests, [])

    def test_invalid_local_request_is_rejected_before_network(self):
        session = CapturingSession({"subjects": []})
        client = DoubanAuthenticatedClient(dbcl2=self.DBCL2, session=session)

        with self.assertRaises(ProviderInvalidResponse):
            client.get_detail("../../admin", "movie")

        self.assertEqual(session.prepared_requests, [])

    def test_cookie_value_is_not_written_to_logs_or_errors(self):
        session = CapturingSession({"unexpected": []})
        client = DoubanAuthenticatedClient(dbcl2=self.DBCL2, session=session)

        with self.assertLogs("app.clients.douban_authenticated", level="WARNING") as captured:
            with self.assertRaises(ProviderInvalidResponse) as raised:
                client.list_items("movie_hot", "movie", 1, {})

        rendered = "\n".join(captured.output + [str(raised.exception), repr(raised.exception)])
        self.assertNotIn(self.DBCL2, rendered)
        self.assertNotIn("dbcl2=", rendered.lower())

    def test_authentication_status_requires_authenticated_marker_on_private_page(self):
        cases = (
            (
                "valid",
                '<div class="top-nav-info"><span class="nav-user-name">测试用户</span></div>',
                200,
                {},
            ),
            (
                "invalid",
                '<a class="nav-login" href="https://accounts.douban.com/passport/login">登录/注册</a>',
                200,
                {},
            ),
            (
                "invalid",
                "",
                302,
                {"Location": "https://accounts.douban.com/passport/login?source=movie"},
            ),
            (
                "unknown",
                "",
                302,
                {"Location": "https://sec.douban.com/aegis/verify?source=movie"},
            ),
            (
                "unknown",
                '<title>没有访问权限</title><a class="nav-login">登录/注册</a>',
                403,
                {},
            ),
            (
                "unknown",
                "<html><body>公共页面</body></html>",
                200,
                {},
            ),
        )
        for expected, body, status_code, headers in cases:
            with self.subTest(expected=expected, status_code=status_code):
                response_headers = {"Content-Type": "text/html; charset=utf-8"}
                response_headers.update(headers)
                session = AuthStatusSession(
                    status_code=status_code,
                    body=body,
                    headers=response_headers,
                )
                client = DoubanAuthenticatedClient(dbcl2=self.DBCL2, session=session)

                self.assertEqual(client.check_authentication(), expected)
                self.assertEqual(
                    urlsplit(session.prepared_requests[0].url).path,
                    "/mine",
                )

    def test_authentication_status_does_not_use_public_search_page_or_leak_cookie(self):
        session = AuthStatusSession(
            body='{"subjects": []}',
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        client = DoubanAuthenticatedClient(dbcl2=self.DBCL2, session=session)

        self.assertEqual(client.check_authentication(), "unknown")
        self.assertNotIn("search_subjects", session.prepared_requests[0].url)
        self.assertEqual(
            session.prepared_requests[0].headers.get("Cookie"),
            f"dbcl2={self.DBCL2}",
        )

    def test_unconfigured_authentication_status_never_performs_network_request(self):
        session = AuthStatusSession()
        client = DoubanAuthenticatedClient(dbcl2="", session=session)

        self.assertEqual(client.check_authentication(), "unconfigured")
        self.assertEqual(session.prepared_requests, [])

    def test_authentication_status_keeps_timeout_inconclusive(self):
        session = AuthStatusSession(error=requests.Timeout("simulated timeout"))
        client = DoubanAuthenticatedClient(dbcl2=self.DBCL2, session=session)

        self.assertEqual(client.check_authentication(), "unknown")


if __name__ == "__main__":
    unittest.main()
