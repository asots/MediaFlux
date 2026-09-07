"""失败下载目标重试：方向对称、保留另一端、认领竞态与 HTTP 身份。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app import database as db
from app.modules import download_dispatcher as dispatcher
from tests.support import isolated_test_database


class DownloadRetrySymmetryAuditTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(isolated_test_database())
        self.enterContext(
            patch.object(
                dispatcher,
                "get",
                side_effect=lambda key, default="": (
                    "http://qb.invalid" if key == "QB_URL" else default
                ),
            )
        )
        self.enterContext(
            patch.object(
                dispatcher,
                "analyze_offline_url",
                return_value=SimpleNamespace(allowed=True, reason=""),
            )
        )
        self.serial = 0

    def request(self, target="qb", peer_status="downloading", *, kind="magnet"):
        self.serial += 1
        item = dispatcher.DownloadInput(
            kind=kind,
            title="重试目标",
            source_value=(
                f"https://downloads.invalid/{self.serial}.torrent"
                if kind == "http"
                else f"magnet:?xt=urn:btih:{self.serial:040x}"
            ),
        )
        created = dispatcher.create_request(item, "", "audit")
        self.assertTrue(created["created"])
        request_id = int(created["id"])
        peer = "guangya" if target == "qb" else "qb"
        fields = {
            "status": "submitted",
            "targets": "both",
            "qb_task_id": "a" * 40,
            "gy_task_id": "gy-kept",
            "gy_task_ids": '["gy-kept"]',
            "gy_target_dir": "stage-kept",
            "gy_staging_name": "untouched",
            "local_import_target": "local-kept",
        }
        fields[("gy" if target == "guangya" else target) + "_status"] = "failed"
        fields[("gy" if peer == "guangya" else peer) + "_status"] = peer_status
        db.update_download_request(request_id, **fields)
        return request_id

    def test_either_failed_target_retries_in_place_while_peer_is_tracked(self):
        for target in ("qb", "guangya"):
            for peer_status in (
                "submitted",
                "downloading",
                "completed",
                "outcome_unknown",
            ):
                with self.subTest(target=target, peer_status=peer_status):
                    request_id = self.request(target, peer_status)
                    before = dict(db.get_download_request(request_id))
                    selected = "_submit_qb" if target == "qb" else "_submit_guangya"
                    other = "_submit_guangya" if target == "qb" else "_submit_qb"
                    with (
                        patch.object(
                            dispatcher,
                            selected,
                            return_value={"ok": True, "task_id": "retry"},
                        ) as submit,
                        patch.object(dispatcher, other) as untouched,
                    ):
                        result = dispatcher.resubmit_download_request(
                            request_id, target
                        )
                    self.assertTrue(result["ok"], result)
                    self.assertFalse(result["created"])
                    self.assertEqual(result["request_id"], request_id)
                    submit.assert_called_once()
                    untouched.assert_not_called()
                    after = dict(db.get_download_request(request_id))
                    self.assertEqual(after["request_key"], before["request_key"])
                    preserved = (
                        (
                            "gy_status",
                            "gy_task_id",
                            "gy_task_ids",
                            "gy_target_dir",
                            "gy_staging_name",
                        )
                        if target == "qb"
                        else ("qb_status", "qb_task_id", "local_import_target")
                    )
                    for field in preserved:
                        self.assertEqual(after[field], before[field], field)
                    with db.get_conn() as conn:
                        self.assertEqual(
                            conn.execute(
                                "SELECT count(*) FROM download_requests WHERE id=?",
                                (request_id,),
                            ).fetchone()[0],
                            1,
                        )

    def test_failed_retry_preserves_attention_and_other_backend(self):
        for backend_result in (
            {"ok": False, "error": "拒绝"},
            {"ok": False, "failure_code": "qb_outcome_unknown", "error": "结果未知"},
        ):
            with self.subTest(result=backend_result):
                request_id = self.request()
                with patch.object(
                    dispatcher, "_submit_qb", return_value=backend_result
                ) as submit:
                    result = dispatcher.resubmit_download_request(request_id, "qb")
                submit.assert_called_once()
                self.assertFalse(result["ok"])
                self.assertTrue(result["source_attention_preserved"])
                self.assertEqual(result["request_id"], request_id)
                self.assertEqual(
                    db.get_download_request(request_id)["gy_status"], "downloading"
                )
                expected = (
                    "outcome_unknown"
                    if backend_result.get("failure_code")
                    else "failed"
                )
                self.assertEqual(
                    db.get_download_request(request_id)["qb_status"], expected
                )

    def test_reentrant_retry_cannot_submit_twice(self):
        request_id = self.request()
        nested = []

        def submit(_row, **_kwargs):
            nested.append(dispatcher.resubmit_download_request(request_id, "qb"))
            return {"ok": True, "task_id": "b" * 40}

        with patch.object(dispatcher, "_submit_qb", side_effect=submit) as backend:
            result = dispatcher.resubmit_download_request(request_id, "qb")
        self.assertTrue(result["ok"], result)
        backend.assert_called_once()
        self.assertFalse(nested[0]["ok"])

    def test_cancellation_after_capability_check_blocks_backend_submission(self):
        request_id = self.request()
        original = dispatcher.download_resubmit_capabilities

        def capture_then_cancel(row, **kwargs):
            result = original(row, **kwargs)
            db.update_download_request(
                request_id, status="cancelled", qb_status="cancelled"
            )
            return result

        with (
            patch.object(
                dispatcher,
                "download_resubmit_capabilities",
                side_effect=capture_then_cancel,
            ),
            patch.object(dispatcher, "_submit_qb") as backend,
        ):
            result = dispatcher.resubmit_download_request(request_id, "qb")
        self.assertFalse(result["ok"])
        backend.assert_not_called()
        self.assertEqual(db.get_download_request(request_id)["status"], "cancelled")

    def test_http_retry_keeps_previously_resolved_torrent_identity(self):
        request_id = self.request(kind="http")
        client = SimpleNamespace(
            add_torrent_detailed=lambda **_kwargs: SimpleNamespace(
                ok=True, failure_code="", task_ids=[], retryable=False
            ),
        )
        with (
            patch.object(dispatcher, "QBittorrentClient", return_value=client),
            patch.object(dispatcher, "close_qbittorrent_client"),
            patch.object(dispatcher, "_submit_guangya") as untouched,
        ):
            result = dispatcher.resubmit_download_request(request_id, "qb")
        self.assertTrue(result["ok"], result)
        self.assertEqual(db.get_download_request(request_id)["qb_task_id"], "a" * 40)
        untouched.assert_not_called()
