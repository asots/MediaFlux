"""动态HTTP种子的来源别名不能替代内容身份，重试不得换种子。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app import database as db
from app.indexers.downloads import submit_download_input
from app.modules import download_dispatcher as dispatcher, offline
from tests.support import isolated_test_database

TORRENT_A = b"d4:infod6:lengthi10485760e4:name9:Movie.mkvee"
TORRENT_B = TORRENT_A.replace(b"Movie.mkv", b"Other.mkv")
MIME = "application/x-bittorrent"


class HttpTorrentIdentityLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(isolated_test_database())
        self.payload = TORRENT_A
        self.cloud_accepts = True
        self.sent = []
        self.raw = replace(
            dispatcher.normalize_download_url("https://fixture.invalid/download.php?id=1"),
            content_type=MIME,
        )
        self.fetch = self.enterContext(patch(
            "app.modules.rss._fetch_rss_payload",
            side_effect=lambda *_args, **_kwargs: (self.payload, {"content-type": MIME}),
        ))
        self.enterContext(patch.object(
            offline.OfflineRules, "from_config", return_value=offline.OfflineRules(
                True, True, False, "target", "Fixture", False, "0", "", (), (), 0, ("mkv",),
            ),
        ))
        self.enterContext(patch.object(dispatcher, "submit_offline", side_effect=self.cloud))
        self.enterContext(patch.object(
            dispatcher, "get", side_effect=lambda name, default="": (
                "https://qb.invalid" if name == "QB_URL" else default
            ),
        ))
        qb = SimpleNamespace(add_torrent_detailed=self.qb, close=lambda: None)
        self.enterContext(patch.object(dispatcher, "QBittorrentClient", return_value=qb))

    def cloud(self, _url, **kwargs):
        self.sent.append(("guangya", kwargs["torrent_data"]))
        if not self.cloud_accepts:
            return {"ok": False, "error": "fixture: backend rejected submission"}
        return {"ok": True, "task_ids": [f"cloud-{len(self.sent)}"], "selected_count": 1}

    def qb(self, **kwargs):
        self.sent.append(("qb", kwargs["torrents"]))
        return SimpleNamespace(
            ok=True, task_ids=[dispatcher.parse_torrent_metadata(kwargs["torrents"] or self.payload)[1]],
            failure_code="", retryable=False,
        )

    def key(self, payload):
        return dispatcher.request_key(dispatcher.torrent_download_input("fixture.torrent", payload))

    def submit(self, target="guangya"):
        return submit_download_input(self.raw, target, origin="rss:identity-fixture")

    def fail_and_purge(self):
        self.cloud_accepts = False
        first = self.submit()
        request_id = first["request_id"]
        self.assertEqual(db.get_download_request(request_id)["status"], "failed")
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE download_requests SET updated_at=datetime('now','localtime','-31 days'),"
                "completed_at=datetime('now','localtime','-31 days') WHERE id=?", (request_id,),
            )
        self.assertEqual(db.purge_expired_download_request_torrent_data(30), 1)
        self.assertIsNone(db.get_download_request(request_id)["torrent_data"])
        self.cloud_accepts = True
        return request_id

    def assert_changed_content_is_independent(self, target):
        first = self.submit()
        self.payload = TORRENT_B
        second = self.submit(target)
        self.assertTrue(second["dispatch"]["ok"], second)
        self.assertNotEqual(first["request_id"], second["request_id"])
        expected = [("guangya", TORRENT_A)]
        if target == "both":
            expected.append(("qb", TORRENT_B))
        expected.append(("guangya", TORRENT_B))
        self.assertCountEqual(self.sent, expected)
        for payload, result in ((TORRENT_A, first), (TORRENT_B, second)):
            owner = db.get_download_request_by_request_key(self.key(payload))
            self.assertEqual(owner["id"], result["request_id"])
            self.assertEqual(owner["torrent_data"], payload)

    def test_changed_source_is_not_suppressed_as_old_cloud_task(self):
        self.assert_changed_content_is_independent("guangya")

    def test_changed_source_does_not_append_old_bytes_to_new_backend(self):
        self.assert_changed_content_is_independent("both")

    def test_concurrent_creation_never_moves_another_contents_keys(self):
        items = [replace(self.raw, torrent_data=payload) for payload in (TORRENT_A, TORRENT_B)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda item: dispatcher.create_request(item, "", "fixture"), items))
        self.assertEqual(len({result["id"] for result in results}), 2)
        for payload, result in zip((TORRENT_A, TORRENT_B), results):
            self.assertEqual(db.get_download_request_by_request_key(self.key(payload))["id"], result["id"])

    def test_purged_retry_rejects_changed_content_before_archiving_original(self):
        request_id = self.fail_and_purge()
        before = dict(db.get_download_request(request_id))
        self.payload = TORRENT_B
        self.sent.clear()
        retried = dispatcher.resubmit_download_request(request_id, "guangya")
        self.assertFalse(retried["ok"], retried)
        self.assertEqual(self.sent, [])
        self.assertEqual(dict(db.get_download_request(request_id)), before)
        self.assertEqual(db.get_download_request_by_request_key(self.key(TORRENT_A))["id"], request_id)
        self.assertIsNone(db.get_download_request_by_request_key(self.key(TORRENT_B)))
        with db.get_conn() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM download_requests").fetchone()[0], 1)

    def test_new_content_after_retention_does_not_archive_the_previous_failure(self):
        request_id = self.fail_and_purge()
        before = dict(db.get_download_request(request_id))
        self.payload = TORRENT_B
        result = self.submit()
        self.assertTrue(result["dispatch"]["ok"], result)
        self.assertNotEqual(request_id, result["request_id"])
        self.assertEqual(dict(db.get_download_request(request_id)), before)
        self.assertEqual(db.get_download_request_by_request_key(self.key(TORRENT_A))["id"], request_id)

    def test_purged_retry_rejects_source_archived_while_fetching(self):
        request_id = self.fail_and_purge()
        successor = {}

        def fetch_and_supersede(*_args, **_kwargs):
            successor.update(dispatcher.create_request(replace(self.raw, torrent_data=TORRENT_A), "", "winner"))
            return TORRENT_B, {"content-type": MIME}

        self.fetch.side_effect = fetch_and_supersede
        self.sent.clear()
        result = dispatcher.resubmit_download_request(request_id, "guangya")
        self.assertFalse(result["ok"], result)
        self.assertTrue(successor["created"])
        self.assertEqual(self.sent, [])
        self.assertEqual(db.get_download_request_by_request_key(self.key(TORRENT_A))["id"], successor["id"])
        self.assertIsNone(db.get_download_request_by_request_key(self.key(TORRENT_B)))

    def test_purged_qb_retry_also_preserves_previously_verified_identity(self):
        request_id = self.fail_and_purge()
        self.payload = TORRENT_B
        self.sent.clear()
        retried = dispatcher.resubmit_download_request(request_id, "qb")
        self.assertFalse(retried["ok"], retried)
        self.assertEqual(self.sent, [])
        self.assertEqual(db.get_download_request_by_request_key(self.key(TORRENT_A))["id"], request_id)

    def test_purged_retry_with_same_content_keeps_canonical_identity(self):
        request_id = self.fail_and_purge()
        self.sent.clear()
        result = dispatcher.resubmit_download_request(request_id, "guangya")
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.sent, [("guangya", TORRENT_A)])
        owner = db.get_download_request_by_request_key(self.key(TORRENT_A))
        self.assertEqual(owner["id"], result["request_id"])
        self.assertEqual(owner["request_key"], self.key(TORRENT_A))

    def test_same_url_and_content_still_deduplicate(self):
        first, second = self.submit(), self.submit()
        self.assertEqual(first["request_id"], second["request_id"])
        self.assertEqual(self.sent, [("guangya", TORRENT_A)])

    def test_cached_retry_uses_original_bytes_without_refetching(self):
        self.cloud_accepts = False
        first = self.submit()
        self.cloud_accepts = True
        self.payload = TORRENT_B
        self.sent.clear()
        self.fetch.reset_mock()
        result = dispatcher.resubmit_download_request(first["request_id"], "guangya")
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.sent, [("guangya", TORRENT_A)])
        self.fetch.assert_not_called()

    def start_untyped_qb_request(self):
        first = submit_download_input(
            replace(self.raw, content_type=""), "qb", origin="telegram:legacy-http",
        )
        self.assertTrue(first["dispatch"]["ok"], first)
        row = db.get_download_request(first["request_id"])
        self.assertIsNone(row["torrent_data"])
        self.assertEqual(row["content_type"], "")
        return first["request_id"]

    def test_active_qb_append_preserves_prepared_dynamic_torrent(self):
        request_id = self.start_untyped_qb_request()
        original_task = db.get_download_request(request_id)["qb_task_id"]
        result = self.submit("guangya")
        self.assertTrue(result["dispatch"]["ok"], result)
        self.assertEqual(result["request_id"], request_id)
        self.assertEqual(self.sent, [("qb", None), ("guangya", TORRENT_A)])
        self.assertEqual(self.fetch.call_count, 1)
        row = db.get_download_request(request_id)
        self.assertEqual(row["qb_task_id"], original_task)
        self.assertEqual(row["qb_status"], "submitted")
        self.assertEqual(row["gy_status"], "submitted")
        self.assertEqual(row["torrent_data"], TORRENT_A)
        self.assertEqual(row["content_type"], MIME)

    def test_active_qb_both_target_does_not_resubmit_qb_or_refetch_torrent(self):
        request_id = self.start_untyped_qb_request()
        self.fetch.side_effect = [(TORRENT_A, {"content-type": MIME}), AssertionError("must reuse verified payload")]
        result = self.submit("both")
        self.assertTrue(result["dispatch"]["ok"], result)
        self.assertEqual(result["request_id"], request_id)
        self.assertEqual(self.sent, [("qb", None), ("guangya", TORRENT_A)])
        self.assertEqual(self.fetch.call_count, 1)

    def test_active_qb_append_rejects_changed_content_before_claiming_cloud(self):
        request_id = self.start_untyped_qb_request()
        before = dict(db.get_download_request(request_id))
        self.payload = TORRENT_B
        result = self.submit("guangya")
        self.assertFalse(result["dispatch"]["ok"], result)
        self.assertEqual(self.sent, [("qb", None)])
        self.assertEqual(dict(db.get_download_request(request_id)), before)
        self.assertIsNone(db.get_download_request_by_request_key(self.key(TORRENT_B)))

    def test_active_qb_append_does_not_store_payload_after_cancelled_claim(self):
        request_id = self.start_untyped_qb_request()
        claim = db.claim_download_request_targets

        def cancel_before_claim(current_id, targets):
            db.update_download_request(current_id, status="cancelled")
            return claim(current_id, targets)

        with patch.object(db, "claim_download_request_targets", side_effect=cancel_before_claim):
            self.submit("guangya")
        row = db.get_download_request(request_id)
        self.assertEqual(row["status"], "cancelled")
        self.assertIsNone(row["torrent_data"])
        self.assertEqual(row["content_type"], "")
        self.assertEqual(self.sent, [("qb", None)])

    def test_active_qb_concurrent_appends_only_submit_cloud_once(self):
        request_id = self.start_untyped_qb_request()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.submit("guangya"), range(2)))
        self.assertTrue(any(result["dispatch"]["ok"] for result in results), results)
        self.assertTrue(all(result["request_id"] == request_id for result in results))
        self.assertEqual(self.sent, [("qb", None), ("guangya", TORRENT_A)])
        self.assertEqual(db.get_download_request(request_id)["torrent_data"], TORRENT_A)

    def test_active_qb_append_rejects_prepared_source_mismatch_before_claim(self):
        request_id = self.start_untyped_qb_request()
        before = dict(db.get_download_request(request_id))
        result = dispatcher.dispatch_missing_targets(
            request_id, "guangya", prepared_input=replace(
                self.raw, source_value="https://other.invalid/resource", torrent_data=TORRENT_A,
            ),
        )
        self.assertFalse(result["ok"], result)
        self.assertEqual(dict(db.get_download_request(request_id)), before)
        self.assertEqual(self.sent, [("qb", None)])
