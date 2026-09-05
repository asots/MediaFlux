"""真实 SDK 的 BT/JSON 请求编码回归；所有 HTTP 只进入内存 transport。"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from unittest.mock import patch

import httpx
from guangyaclient import GuangyaClient as RawClient

from app.clients.guangya import GuangYaClient


class GuangYaTorrentTransportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.client = GuangYaClient(token_file=Path(temp.name) / "absent-token.json")
        self.addCleanup(self.client.close)
        network = patch("socket.socket.connect", side_effect=AssertionError("external network forbidden"))
        network.start()
        self.addCleanup(network.stop)

    def raw_client(self, handler):
        raw = RawClient(access_token="fixture-access", device_id="fixture-device")
        headers = dict(raw._client.headers)
        raw.close()
        raw._client = httpx.Client(headers=headers, transport=httpx.MockTransport(handler))
        self.addCleanup(raw.close)
        self.client._install_request_retry_policy(raw)
        return raw

    def assert_torrent_request(self, request, payload):
        content_type = request.headers["content-type"]
        self.assertTrue(content_type.startswith("multipart/form-data; boundary="), content_type)
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + request.read()
        )
        parts = list(message.iter_parts())
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].get_param("name", header="content-disposition"), "torrent")
        self.assertEqual(parts[0].get_filename(), "file.torrent")
        self.assertEqual(parts[0].get_payload(decode=True), payload)
        self.assertEqual(request.headers["authorization"], "Bearer fixture-access")
        self.assertIn("traceparent", request.headers)

    def test_sdk_torrent_upload_has_matching_multipart_boundary_and_bytes(self):
        payload = b"d4:infod4:name8:test.mkvee\x00\xff"
        captured = []
        raw = self.raw_client(lambda req: captured.append(req) or httpx.Response(200, json={"data": {"files": []}}))
        raw.cloud_resolve_torrent(payload)
        self.assert_torrent_request(captured[0], payload)

    def test_json_and_explicit_binary_content_headers_are_preserved(self):
        captured = []
        raw = self.raw_client(lambda req: captured.append(req) or httpx.Response(200, json={}))
        raw.cloud_resolve_url("magnet:?xt=urn:btih:" + "a" * 40)
        self.assertEqual(captured[0].headers["content-type"], "application/json")
        self.assertIn("magnet:?xt=", json.loads(captured[0].read())["url"])
        raw.request("https://example.invalid/object", "PUT", content=b"binary",
                    headers={"Content-Type": "application/octet-stream", "X-Fixture": "kept"})
        self.assertEqual(captured[1].headers["content-type"], "application/octet-stream")
        self.assertEqual(captured[1].headers["x-fixture"], "kept")
        self.assertEqual(captured[1].read(), b"binary")

    def test_concurrent_json_and_multipart_requests_do_not_switch_shared_defaults(self):
        barrier = threading.Barrier(2)
        captured = []
        defaults = []

        def handler(request):
            defaults.append(dict(raw._client.headers))
            barrier.wait(timeout=5)
            captured.append(request)
            return httpx.Response(200, json={})

        raw = self.raw_client(handler)
        expected_headers = dict(raw._client.headers)
        # 反复安装不能再改写一个已经投入使用的客户端。
        self.client._install_request_retry_policy(raw)
        with ThreadPoolExecutor(max_workers=2) as executor:
            torrent = executor.submit(raw.cloud_resolve_torrent, b"fixture-torrent")
            metadata = executor.submit(raw.cloud_resolve_url, "fixture-url")
            torrent.result(timeout=8)
            metadata.result(timeout=8)
        for request in captured:
            if request.url.path.endswith("resolve_torrent"):
                self.assert_torrent_request(request, b"fixture-torrent")
            else:
                self.assertEqual(request.headers["content-type"], "application/json")
        self.assertEqual(defaults, [expected_headers, expected_headers])
        self.assertEqual(dict(raw._client.headers), expected_headers)

    def test_separate_clients_and_write_401_never_trigger_sdk_replay(self):
        captured = []
        raw = self.raw_client(lambda req: captured.append(req) or httpx.Response(401))
        raw.refresh_token_value = "fixture-refresh"
        other = self.raw_client(lambda req: httpx.Response(200, json={}))
        with patch.object(raw, "refresh_token") as refresh:
            with self.assertRaises(httpx.HTTPStatusError):
                raw.request("https://example.invalid/write", "POST", json={"operation": "create"})
        self.assertEqual(len(captured), 1)
        refresh.assert_not_called()
        self.assertEqual(raw.refresh_token_value, "fixture-refresh")
        other.cloud_resolve_torrent(b"another-torrent")
        self.assertEqual(len(captured), 1)

    def prepare_wrapper(self, raw):
        self.client._raw = raw
        for method in ("_invalidate_if_stale", "_ensure_fresh_token"):
            patcher = patch.object(self.client, method, return_value=False)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_automatic_offline_flow_resolves_bt_then_sends_filtered_json_selection(self):
        from app.modules import offline
        from tests.test_guangya_offline_selection import RESOLVE_SUBFILES_FIXTURE

        captured = []

        def handler(request):
            captured.append(request)
            if request.url.path.endswith("resolve_torrent"):
                self.assert_torrent_request(request, b"fixture-torrent")
                return httpx.Response(200, json=RESOLVE_SUBFILES_FIXTURE)
            self.assertTrue(request.url.path.endswith("create_task"))
            self.assertEqual(request.headers["content-type"], "application/json")
            self.assertEqual(json.loads(request.read())["fileIndexes"], [0])
            return httpx.Response(200, json={"code": 0, "data": {"taskId": "fixture-task"}})

        self.prepare_wrapper(self.raw_client(handler))
        rules = offline.OfflineRules.from_mapping({"exclude_keywords": "sample", "min_file_mb": 0})
        with patch.object(offline.OfflineRules, "from_config", return_value=rules):
            result = offline.submit_offline(
                "magnet:?xt=urn:btih:" + "a" * 40, client=self.client,
                torrent_data=b"fixture-torrent",
            )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["task_ids"], ["fixture-task"])
        self.assertEqual(len(captured), 2)

    def test_real_read_policy_has_bounded_retries_without_creating_tasks(self):
        from app.modules import offline

        for status, attempts in ((401, 2), (403, 1), (415, 1), (429, 2), (503, 2)):
            with self.subTest(status=status):
                captured = []
                raw = self.raw_client(lambda req: captured.append(req) or httpx.Response(status))
                self.prepare_wrapper(raw)
                with (
                    patch.object(self.client, "_refresh_after_unauthorized") as refresh,
                    patch("app.clients.guangya.sleep"),
                ):
                    result = offline.submit_offline(
                        "magnet:?xt=urn:btih:" + "a" * 40, client=self.client,
                        torrent_data=b"fixture-torrent", isolate_task=True,
                    )
                self.assertFalse(result["ok"])
                self.assertEqual(result["resolve_http_status"], status)
                self.assertEqual(len(captured), attempts)
                self.assertEqual(refresh.call_count, int(status == 401))
                self.assertTrue(all(req.url.path.endswith("resolve_torrent") for req in captured))
                for request in captured:
                    self.assert_torrent_request(request, b"fixture-torrent")
