"""RSS 种子准备失败专项回归：真实 prepare/repo/RSS，仅替换网络与下载提交边界。"""
from __future__ import annotations

import hashlib
import json
from unittest.mock import patch

import httpx

from app import database as db
from app.indexers import downloads
from app.modules import download_dispatcher as dispatcher
from app.modules import offline, rss
from app.repositories import rss as rss_repo
from tests.support import IsolatedDatabaseTestCase, isolated_test_database

_INFO = b"d6:lengthi10485760e4:name9:Movie.mkve"
_TORRENT = b"d4:info" + _INFO + b"e"
_INFOHASH = hashlib.sha1(_INFO).hexdigest()
_TOKEN = "rss-preparation-test-token"
_URL = f"https://feed.invalid/download/{_INFOHASH}.torrent?passkey={_TOKEN}"
_HTTP_CASES = (
    (408, True), (425, True), (429, True),
    (500, True), (502, True), (503, True), (504, True), (599, True),
    (400, False), (401, False), (403, False), (404, False), (410, False),
)


def _http_error(status):
    request = httpx.Request("GET", _URL)
    return httpx.HTTPStatusError(
        f"upstream token={_TOKEN}", request=request,
        response=httpx.Response(status, request=request),
    )


class RSSTorrentPreparationFailureTests(IsolatedDatabaseTestCase):
    def setUp(self):
        super().setUp()
        # 每个用例都使用独立临时库，表驱动子例使用不同订阅/条目。
        self.enterContext(isolated_test_database())
        self.enterContext(patch(
            "socket.socket.connect", side_effect=AssertionError("real network forbidden"),
        ))
        self.enterContext(patch(
            "socket.getaddrinfo", side_effect=AssertionError("real DNS forbidden"),
        ))
        rules = offline.OfflineRules(
            True, True, False, "target", "测试目录", False, "0", "", (), (), 0, ("mkv",),
        )
        self.enterContext(patch.object(offline.OfflineRules, "from_config", return_value=rules))
        self.fetch = self.enterContext(patch.object(
            rss, "_fetch_rss_payload",
            return_value=(_TORRENT, {"content-type": "application/x-bittorrent"}),
        ))
        self.guangya = self.enterContext(patch.object(
            dispatcher, "_submit_guangya",
            return_value={"ok": True, "task_id": "gy-task", "task_ids": ["gy-task"]},
        ))
        self.qb = self.enterContext(patch.object(
            dispatcher, "_submit_qb", return_value={"ok": True, "task_id": _INFOHASH},
        ))
        self.engine = rss.RSSEngine()
        self.addCleanup(self.engine.close)
        self.sequence = 0

    def entry(self, url=_URL, method="guangya"):
        self.sequence += 1
        sid = db.add_rss_subscription(
            f"准备失败 {self.sequence}", f"https://feed.invalid/{self.sequence}.xml",
            download_method=method, gy_target_dir="target", gy_target_dir_name="测试目录",
        )
        eid = db.add_rss_entry_with_media(
            sid, "Movie", f"preparation-{self.sequence}",
            payload=json.dumps({"torrent_url": url, "content_type": "application/x-bittorrent"}),
        )["id"]
        self.assertIsNotNone(eid)
        return int(eid)

    def logs(self, eid):
        with db.get_conn() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM download_log WHERE rss_item_id=? ORDER BY id",
                (db.get_rss_entry(eid)["rss_item_id"],),
            )]

    def assert_no_requests(self):
        self.guangya.assert_not_called()
        self.qb.assert_not_called()
        with db.get_conn() as conn:
            for table in ("download_requests", "download_request_keys"):
                self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)

    def failure(self, eid, code, retryable=False, *, review=False, public_fields=True):
        with self.assertLogs("app", level="WARNING") as captured:
            result = self.engine.download(eid)
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["status"], "failed")
        if public_fields:
            self.assertEqual(result["failure_code"], code)
            self.assertEqual(result["retryable"], retryable)
        self.assertEqual(bool(result.get("review_required")), review)
        row = db.get_rss_entry(eid)
        self.assertEqual((row["status"], row["processed"]), ("failed", 0))
        self.assertEqual((row["failure_code"], row["failure_retryable"]), (code, int(retryable)))
        self.assertTrue(row["failed_at"])
        logs = self.logs(eid)
        self.assertTrue(logs)
        self.assertTrue(all(not log["backend_task_id"] for log in logs))
        self.assertNotIn(_TOKEN, json.dumps([result, logs, captured.output], default=str))
        return result, captured.output

    def age_failure(self, eid, seconds=61):
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE rss_entries SET failed_at=datetime('now','localtime',?) WHERE id=?",
                (f"-{seconds} seconds", eid),
            )

    def retryable_ids(self):
        return {int(row["id"]) for row in db.get_retryable_failed_rss_snapshot("guangya")}

    def test_preparation_error_public_constructor_and_defaults(self):
        error = offline.TorrentPreparationError(
            "安全错误", failure_code="torrent_invalid", stage="validate",
        )
        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error), "安全错误")
        self.assertEqual((error.failure_code, error.stage, error.retryable, error.http_status),
                         ("torrent_invalid", "validate", False, None))
        error = offline.TorrentPreparationError(
            "HTTP 503", failure_code="torrent_http_error", stage="fetch",
            retryable=True, http_status=503,
        )
        self.assertEqual((error.failure_code, error.stage, error.retryable, error.http_status),
                         ("torrent_http_error", "fetch", True, 503))

    def test_real_fetch_classification_survives_prepare_download_input(self):
        cases = [
            (httpx.ReadTimeout(_URL), "torrent_fetch_timeout", True, None),
            (httpx.ConnectTimeout(_URL), "torrent_fetch_timeout", True, None),
            (TimeoutError(_URL), "torrent_fetch_timeout", True, None),
            (httpx.ConnectError(_URL), "torrent_fetch_failed", True, None),
            (httpx.ReadError(_URL), "torrent_fetch_failed", True, None),
            (httpx.RemoteProtocolError(_URL), "torrent_fetch_failed", True, None),
            (OSError(_URL), "torrent_fetch_failed", True, None),
            *[(_http_error(status), "torrent_http_error", retryable, status)
              for status, retryable in _HTTP_CASES],
        ]
        for source_error, code, retryable, status in cases:
            for target in ("offline", "guangya", "both"):
                with self.subTest(error=type(source_error).__name__, status=status, target=target):
                    self.fetch.side_effect = source_error
                    with self.assertRaises(offline.TorrentPreparationError) as caught:
                        if target == "offline":
                            offline._prepare_offline_resource(_URL)
                        else:
                            dispatcher.prepare_download_input(
                                dispatcher.normalize_download_url(_URL), target,
                            )
                    error = caught.exception
                    self.assertEqual((error.failure_code, error.stage, error.retryable, error.http_status),
                                     (code, "fetch", retryable, status))
                    self.assertIs(error.__cause__, source_error)
                    self.assertNotIn(_TOKEN, str(error))
        self.assert_no_requests()

    def test_validation_classifies_bad_bencode_and_mismatched_hash(self):
        cases = (
            (b"", _URL, "torrent_invalid"),
            (b"de", _URL, "torrent_invalid"),
            (f"<html>token={_TOKEN}</html>".encode(), _URL, "torrent_invalid"),
            (_TORRENT, _URL.replace(_INFOHASH, "f" * 40), "torrent_identity_mismatch"),
        )
        for payload, url, code in cases:
            for supplied in (False, True):
                with self.subTest(code=code, payload=payload[:12], supplied=supplied):
                    self.fetch.return_value = (payload, {})
                    self.fetch.reset_mock()
                    with self.assertRaises(offline.TorrentPreparationError) as caught:
                        offline._prepare_offline_resource(url, payload if supplied else None)
                    error = caught.exception
                    self.assertEqual((error.failure_code, error.stage, error.retryable, error.http_status),
                                     (code, "validate", False, None))
                    self.assertNotIn(_TOKEN, str(error))
                    self.assertEqual(self.fetch.call_count, int(not supplied))
        self.assert_no_requests()

    def test_rss_initial_timeout_is_unbound_without_backend_or_request(self):
        eid = self.entry()
        self.fetch.side_effect = httpx.ReadTimeout(_URL)
        result, output = self.failure(eid, "torrent_fetch_timeout", True)
        self.assertEqual(result["infohash"], _INFOHASH)
        self.assertEqual(result["request_id"], 0)
        row = db.get_rss_entry(eid)
        self.assertEqual((row["download_request_key"], row["download_backend"], row["retry_count"]),
                         ("", "", 0))
        self.assertIn("stage=fetch", "\n".join(output))
        self.fetch.assert_called_once_with(
            _URL, user_agent="MediaFlux/1.0", timeout_seconds=20, allow_http=True,
        )
        self.assert_no_requests()

    def test_cooldown_rejects_retry_then_expired_timeout_submits_once(self):
        eid = self.entry()
        self.fetch.side_effect = httpx.ReadTimeout(_URL)
        self.failure(eid, "torrent_fetch_timeout", True)
        before = dict(db.get_rss_entry(eid))
        self.assertNotIn(eid, self.retryable_ids())
        self.assertFalse(self.engine.download(eid)["ok"])
        self.assertEqual(dict(db.get_rss_entry(eid)), before)
        self.assertEqual(self.fetch.call_count, 1)
        self.assert_no_requests()
        self.age_failure(eid)
        self.assertIn(eid, self.retryable_ids())
        self.fetch.side_effect = None
        result = self.engine.download(eid)
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["existing"])
        self.assertEqual(self.fetch.call_count, 2)
        self.guangya.assert_called_once()
        row = db.get_rss_entry(eid)
        self.assertEqual((row["status"], row["processed"], row["retry_count"]), ("downloaded", 1, 1))
        self.assertEqual((row["failure_code"], row["failure_retryable"]), ("", 0))
        request = db.get_download_request(result["request_id"])
        self.assertEqual(request["torrent_data"], _TORRENT)
        self.assertEqual(request["gy_task_id"], "gy-task")
        self.assertEqual(row["download_request_key"], request["request_key"])
        repeated = self.engine.download(eid)
        self.assertTrue(repeated["ok"])
        self.assertTrue(repeated["existing"])
        self.assertEqual(self.fetch.call_count, 2)
        self.guangya.assert_called_once()
        with db.get_conn() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM download_requests").fetchone()[0], 1)

    def test_all_read_codes_share_cooldown_and_five_retry_limit(self):
        for error, code in (
            (httpx.ReadTimeout(_URL), "torrent_fetch_timeout"),
            (httpx.ConnectError(_URL), "torrent_fetch_failed"),
            (_http_error(503), "torrent_http_error"),
        ):
            with self.subTest(code=code):
                eid = self.entry()
                self.fetch.reset_mock()
                self.fetch.side_effect = error
                self.failure(eid, code, True)
                self.age_failure(eid, 30)
                before = dict(db.get_rss_entry(eid))
                self.assertNotIn(eid, self.retryable_ids())
                self.assertFalse(self.engine.download(eid)["ok"])
                self.assertEqual(dict(db.get_rss_entry(eid)), before)
                self.assertEqual(self.fetch.call_count, 1)
                for retry in range(1, 6):
                    self.age_failure(eid)
                    self.assertIn(eid, self.retryable_ids())
                    self.failure(eid, code, True)
                    self.assertEqual(db.get_rss_entry(eid)["retry_count"], retry)
                self.age_failure(eid)
                before = dict(db.get_rss_entry(eid))
                self.assertNotIn(eid, self.retryable_ids())
                self.assertFalse(self.engine.download(eid)["ok"])
                self.assertFalse(db.claim_rss_entry(eid))
                self.assertEqual(dict(db.get_rss_entry(eid)), before)
                self.assertEqual(self.fetch.call_count, 6)
                self.assert_no_requests()

    def test_rss_http_statuses_preserve_source_classification_and_safe_logs(self):
        for status, retryable in _HTTP_CASES:
            with self.subTest(status=status):
                eid = self.entry()
                self.fetch.side_effect = _http_error(status)
                result, output = self.failure(eid, "torrent_http_error", retryable)
                self.assertIn(str(status), result["error"])
                self.assertIn(f"http_status={status}", "\n".join(output))
                self.assertEqual(db.get_rss_entry(eid)["download_request_key"], "")
                self.age_failure(eid)
                self.assertEqual(eid in self.retryable_ids(), retryable)
                self.assert_no_requests()

    def test_rss_network_errors_are_fetch_failures_not_invalid_urls(self):
        for error in (httpx.ConnectError(_URL), httpx.ReadError(_URL), httpx.RemoteProtocolError(_URL)):
            with self.subTest(error=type(error).__name__):
                eid = self.entry()
                self.fetch.side_effect = error
                self.failure(eid, "torrent_fetch_failed", True)
                self.assertEqual(db.get_rss_entry(eid)["download_request_key"], "")
                self.assert_no_requests()

    def test_rss_bad_content_and_hash_mismatch_never_become_retryable(self):
        for payload, url, code in (
            (f"token={_TOKEN}".encode(), _URL, "torrent_invalid"),
            (b"de", _URL, "torrent_invalid"),
            (_TORRENT, _URL.replace(_INFOHASH, "f" * 40), "torrent_identity_mismatch"),
        ):
            with self.subTest(code=code, payload=payload[:12]):
                eid = self.entry(url)
                self.fetch.return_value = (payload, {})
                _, output = self.failure(eid, code)
                self.assertIn("stage=validate", "\n".join(output))
                self.age_failure(eid)
                before = dict(db.get_rss_entry(eid))
                calls = self.fetch.call_count
                self.assertNotIn(eid, self.retryable_ids())
                self.assertFalse(self.engine.download(eid)["ok"])
                self.assertEqual(dict(db.get_rss_entry(eid)), before)
                self.assertEqual(self.fetch.call_count, calls)
                self.assertEqual(before["download_request_key"], "")
                self.assert_no_requests()

    def test_truly_invalid_url_is_normalize_failure_without_fetch(self):
        for url in (
            f"https:///movie.torrent?passkey={_TOKEN}",
            f"magnet:?dn=Movie&token={_TOKEN}",
            f"ftp://feed.invalid/movie.torrent?token={_TOKEN}",
        ):
            with self.subTest(url=url.split("?")[0]):
                _, output = self.failure(self.entry(url), "invalid_payload")
                self.assertIn("stage=normalize", "\n".join(output))
                self.fetch.assert_not_called()
                self.assert_no_requests()

    def test_real_binding_conflict_does_not_overwrite_winner_or_record_bad_link(self):
        self.assertTrue(issubclass(rss_repo.RSSEntryStateChangedError, ValueError))
        for status, processed in (("skipped", 1), ("failed", 0), ("downloaded", 1)):
            with self.subTest(status=status):
                eid = self.entry()
                winner = {}

                def fetch_and_change_state(
                    *_args, status=status, processed=processed, eid=eid, winner=winner, **_kwargs,
                ):
                    with db.get_conn() as conn:
                        conn.execute(
                            "UPDATE rss_entries SET status=?,processed=?,failure_code='guangya_auth_failed',"
                            "failure_retryable=0,retry_count=3,download_request_key='winner-key',"
                            "download_backend='guangya',failed_at='2026-10-01 00:00:00' WHERE id=?",
                            (status, processed, eid),
                        )
                    winner.update(dict(db.get_rss_entry(eid)))
                    return _TORRENT, {}

                self.fetch.side_effect = fetch_and_change_state
                with patch.object(db, "record_rss_entry_failure", wraps=db.record_rss_entry_failure) as record, \
                        patch.object(rss_repo, "bind_rss_entry_download", wraps=rss_repo.bind_rss_entry_download) as bind, \
                        self.assertLogs("app", level="WARNING") as captured:
                    result = self.engine.download(eid)
                self.assertFalse(result["ok"])
                self.assertEqual(result["status"], "conflict")
                self.assertTrue(result["error"])
                self.assertNotEqual(result.get("failure_code"), "invalid_payload")
                bind.assert_called_once()
                record.assert_not_called()
                self.assertEqual(dict(db.get_rss_entry(eid)), winner)
                self.assertEqual(self.logs(eid), [])
                self.assertNotIn(_TOKEN, json.dumps([result, captured.output]))
                self.assert_no_requests()

    def test_backend_valueerror_is_unknown_nonretryable_and_never_a_bad_url(self):
        for method in ("guangya", "qb"):
            # 内容身份相同，必须隔离子例，避免光鸭 unknown 防重挡住 qB 提交。
            with self.subTest(method=method), isolated_test_database():
                eid = self.entry(method=method)
                backend = self.guangya if method == "guangya" else self.qb
                backend.side_effect = ValueError(f"receipt uncertain token={_TOKEN} {_URL}")
                result, _ = self.failure(
                    eid, f"{method}_outcome_unknown", review=True, public_fields=False,
                )
                backend.assert_called_once()
                request = db.get_download_request(result["request_id"])
                backend_status = "gy_status" if method == "guangya" else "qb_status"
                self.assertEqual(request[backend_status], "outcome_unknown")
                self.assertTrue(db.get_rss_entry(eid)["download_request_key"])
                before_calls = backend.call_count
                before = dict(db.get_rss_entry(eid))
                repeated = self.engine.download(eid)
                self.assertFalse(repeated["ok"])
                self.assertTrue(rss.rss_entry_delivery(db.get_rss_entry(eid))["review_required"])
                self.assertEqual(backend.call_count, before_calls)
                self.assertEqual(dict(db.get_rss_entry(eid)), before)
                self.assertEqual(db.update_rss_entries_processed([eid], False), 0)
                self.assertNotIn(_TOKEN, json.dumps(repeated))

    def test_other_submission_valueerror_cannot_masquerade_as_invalid_payload(self):
        for method in ("guangya", "qb"):
            with self.subTest(method=method):
                eid = self.entry(method=method)
                with patch.object(downloads, "create_request", side_effect=ValueError(f"commit token={_TOKEN}")):
                    result, output = self.failure(eid, f"{method}_outcome_unknown", review=True)
                self.assertIn("stage=submit", "\n".join(output))
                self.assertEqual(result["request_id"], 0)
                self.assertTrue(db.get_rss_entry(eid)["download_request_key"])
                self.assert_no_requests()

    def test_qb_http_torrent_is_not_prefetched_and_only_submitted_once(self):
        eid = self.entry(method="qb")
        self.fetch.side_effect = AssertionError("qB HTTP torrent must not be prefetched")
        result = self.engine.download(eid)
        self.assertTrue(result["ok"], result)
        self.qb.assert_called_once()
        self.guangya.assert_not_called()
        self.fetch.assert_not_called()
        request = db.get_download_request(result["request_id"])
        self.assertEqual((request["kind"], request["source_value"], request["torrent_data"]),
                         ("http", _URL, None))
        repeated = self.engine.download(eid)
        self.assertTrue(repeated["ok"])
        self.assertTrue(repeated["existing"])
        self.qb.assert_called_once()
        self.fetch.assert_not_called()
        self.assertNotIn(_TOKEN, json.dumps([result, repeated, self.logs(eid)]))

    def test_legacy_invalid_payload_requires_explicit_reset_before_retry(self):
        eid = self.entry()
        self.assertTrue(db.claim_rss_entry(eid))
        db.record_rss_entry_failure(eid, "invalid_payload", False)
        self.age_failure(eid)
        before = dict(db.get_rss_entry(eid))
        self.assertNotIn(eid, self.retryable_ids())
        self.assertFalse(self.engine.download(eid)["ok"])
        self.assertFalse(db.claim_rss_entry(eid))
        self.assertEqual(dict(db.get_rss_entry(eid)), before)
        self.fetch.assert_not_called()
        self.assert_no_requests()
        self.assertEqual(db.update_rss_entries_processed([eid], False), 1)
        self.assertEqual(db.get_rss_entry(eid)["status"], "pending")
        self.assertEqual(db.get_rss_entry(eid)["failure_code"], "")
        result = self.engine.download(eid)
        self.assertTrue(result["ok"], result)
        self.guangya.assert_called_once()
        self.fetch.assert_called_once()
        self.assertTrue(self.engine.download(eid)["existing"])
        self.guangya.assert_called_once()
