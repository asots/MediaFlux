"""官网 assets 容量契约、真实请求边界与 Agent 最小公开投影。"""
from __future__ import annotations

import json
import unittest
from unittest import mock

import httpx

from app.agent import guangya_account_actions as actions
from app.agent.domain_catalog import build_tool_specs
from app.agent.errors import AgentToolError
from app.agent.kernel.capabilities import CapabilityRetriever, ToolEffect
from app.agent.kernel.pipeline import ToolCallContext, ToolPipeline
from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
from app.agent.kernel.state import CancellationToken, InMemorySessionStateStore
from app.clients.guangya import GuangYaClient

PROFILE = {"sub": "private-account", "name": "Alice", "phone_number": "13800138000"}
# 2026-09-07 官网 get_assets 结构；仅保留公开字段名，数值使用测试数据。
ASSETS = {
    "code": 0,
    "msg": "success",
    "data": {
        "totalSpaceSize": 500 * 1024**4,
        "usedSpaceSize": 40 * 1024**4,
        "freeDirectLinkTraffic": 123,
        "totalDirectLinkTraffic": 456,
        "freeShareGuestTraffic": 789,
        "vipStatus": 2,
        "systemTime": 1_234_567_890,
    },
}


class AccountClient(GuangYaClient):
    """真实包装方法 + mock HTTP 传输，不读取账号凭据。"""

    def __init__(self, *, assets=None, profile=None, status_code=200):
        self.calls = []
        self.closed = False
        self.assets = ASSETS if assets is None else assets
        self.profile = PROFILE if profile is None else profile
        self.status_code = status_code
        self._raw = mock.Mock()
        self._raw._account_headers.return_value = {"x-client-id": "test-client"}
        self._raw.request.side_effect = self._request

    @property
    def raw(self):
        return self._raw

    @property
    def logged_in(self):
        return True

    def _request(self, url, method="GET", **kwargs):
        self.calls.append((url, method, kwargs))
        if url == "https://api.guangyapan.com/assets/v1/get_assets":
            payload = self.assets
        elif url == "https://account.guangyapan.com/v1/user/me":
            payload = self.profile
        else:
            raise AssertionError(f"unexpected endpoint: {url}")
        if isinstance(payload, Exception):
            raise payload
        response = httpx.Response(
            self.status_code, json=payload, request=httpx.Request(method, url)
        )
        response.raise_for_status()
        return response

    def close(self):
        self.closed = True
        return True


def read_status(client):
    with mock.patch.object(actions, "GuangYaClient", return_value=client):
        return actions.get_guangya_account_status({})


class GuangYaAccountCapacityTests(unittest.TestCase):
    def test_actual_assets_schema_is_read_from_separate_endpoint(self):
        client = AccountClient()
        result = read_status(client)
        storage = result.data["storage"]
        self.assertEqual(storage["total_bytes"], 500 * 1024**4)
        self.assertEqual(storage["used_bytes"], 40 * 1024**4)
        self.assertEqual(storage["available_bytes"], 460 * 1024**4)
        self.assertEqual(storage["utilization"], 0.08)
        self.assertTrue(storage["reported"])
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0][0], "https://api.guangyapan.com/assets/v1/get_assets")
        for _url, method, kwargs in client.calls:
            self.assertEqual(method, "GET" if "/v1/user/me" in _url else "POST")
            self.assertEqual(kwargs["timeout"], 15)
        self.assertEqual(client.calls[0][2]["json"], {})
        self.assertEqual(client.calls[1][2]["headers"], {"x-client-id": "test-client"})
        client._raw.user_info.assert_not_called()
        self.assertTrue(client.closed)

    def test_byte_values_are_exact_and_zero_is_reported(self):
        for total, used, expected_total, expected_used in (
            ("9007199254740993", "1", 9007199254740993, 1),
            ("1000", "0", 1000, 0),
            (0, 0, 0, 0),
            (1000, 1250, 1000, 1250),
            (1000.0, 250.0, 1000, 250),
        ):
            with self.subTest(total=total, used=used):
                result = read_status(AccountClient(assets={"code": 0, "data": {
                    "totalSpaceSize": total, "usedSpaceSize": used,
                }}))
                storage = result.data["storage"]
                self.assertEqual(storage["total_bytes"], expected_total)
                self.assertEqual(storage["used_bytes"], expected_used)
                self.assertEqual(storage["available_bytes"], max(0, expected_total - expected_used))
                self.assertTrue(storage["reported"])
                if expected_total and expected_used > expected_total:
                    self.assertGreater(storage["utilization"], 1)

    def test_invalid_capacity_does_not_become_zero_or_a_completed_overview(self):
        for value in (True, False, -1, "-1", "not-a-size", "2TB", "1.2", 1.2, 10**400, "9" * 400, 1 << 63, float("inf"), float("nan"), [], {}):
            with self.subTest(value=value):
                # 直接测试解析器，JSON 不允许 NaN/Infinity。
                self.assertIsNone(actions._bytes([{"totalSpaceSize": value}], "totalSpaceSize"))
        result = read_status(AccountClient(assets={"code": 0, "data": {
            "totalSpaceSize": -1, "usedSpaceSize": True,
            "totalDirectLinkTraffic": 500, "freeShareGuestTraffic": 250,
        }}))
        self.assertFalse(result.data["storage"]["reported"])
        self.assertEqual(result.data["storage"]["status"], "not_reported")
        self.assertEqual(result.status, "partial")

    def test_capacity_does_not_mix_account_profile_or_traffic_counters(self):
        client = AccountClient(
            profile={"data": {"nickname": "Alice", "totalSpace": 999, "usedSpace": 111}},
            assets={"code": 0, "data": {"totalDirectLinkTraffic": 500, "freeShareGuestTraffic": 250}},
        )
        self.assertFalse(read_status(client).data["storage"]["reported"])

    def test_known_legacy_capacity_names_and_incomplete_values(self):
        for data, expected in (
            ({"storage": {"totalSpace": 1000, "usedSpace": 250}}, (1000, 250, 750, "ok")),
            ({"totalSpaceSize": "bad", "totalSpace": 1000, "usedSpaceSize": 0}, (1000, 0, 1000, "ok")),
            ({"totalSpaceSize": 1000}, (1000, None, None, "partial")),
            ({"totalSpace": 1000, "freeSpace": 800}, (1000, 200, 800, "ok")),
            ({"totalSpace": 1000, "freeSpace": 1200}, (1000, None, 1200, "partial")),
        ):
            with self.subTest(data=data):
                storage = read_status(AccountClient(assets={"code": 0, "data": data})).data["storage"]
                self.assertEqual(tuple(storage[key] for key in ("total_bytes", "used_bytes", "available_bytes", "status")), expected)

    def test_profile_failure_cannot_discard_valid_capacity(self):
        for profile in (
            {"error": "private-token", "error_code": 401, "error_description": "private-info"},
            {"code": 500, "data": {}},
            httpx.ReadTimeout("private-request"),
        ):
            with self.subTest(profile=type(profile).__name__):
                result = read_status(AccountClient(profile=profile))
                self.assertTrue(result.data["connected"])
                self.assertFalse(result.data["profile_available"])
                self.assertTrue(result.data["storage"]["reported"])
                self.assertNotIn("private-", json.dumps(result.to_dict()))

    def test_assets_failure_is_not_described_as_provider_omitting_fields(self):
        for assets in (
            {"error": "private-error"},
            {"code": 157, "msg": "private-error", "data": ASSETS["data"]},
            {"code": 0, "data": {"code": 401, **ASSETS["data"]}},
            {"code": 0, "success": False, "data": ASSETS["data"]},
            {"code": True, "data": ASSETS["data"]},
            {"code": 0, "data": None},
            {"code": 0, "data": []},
            {"code": 0, "data": "private-value"},
            "private-body", [], {}, {"data": {}}, {"data": ASSETS["data"]},
            {"code": 0, "success": 0, "data": {}},
            {"code": 0, "success": "false", "data": {}},
            httpx.ReadTimeout("private-request"),
        ):
            with self.subTest(assets=type(assets).__name__):
                client = AccountClient(assets=assets)
                result = read_status(client)
                self.assertEqual(result.status, "partial")
                self.assertEqual(result.data["storage"]["status"], "unavailable")
                self.assertFalse(result.data["storage"]["reported"])
                self.assertIn("读取失败", result.summary)
                self.assertNotIn("服务端本次未返回", result.summary)
                self.assertNotIn("private-", json.dumps(result.to_dict()))
                self.assertTrue(client.closed)

    def test_no_remote_success_never_claims_connected(self):
        for profile in ({"error": "invalid_token"}, {"data": {}}, {"success": 0, "data": {}}, {"name": ""}):
            client = AccountClient(assets={"code": 401}, profile=profile)
            with self.subTest(profile=profile), self.assertRaises(AgentToolError) as caught:
                read_status(client)
            self.assertEqual(caught.exception.code, "unavailable")
            self.assertTrue(client.closed)

    def test_http_errors_use_existing_bounded_retry_and_no_raw_message(self):
        for status in (401, 403, 500):
            for method in ("account_info", "account_storage_info"):
                client = AccountClient(status_code=status)
                with (
                    self.subTest(status=status, method=method),
                    mock.patch.object(client, "_refresh_after_unauthorized") as refresh,
                    mock.patch("app.clients.guangya.sleep"),
                    self.assertRaises(httpx.HTTPStatusError),
                ):
                    getattr(client, method)()
                self.assertEqual(len(client.calls), 1 if status == 403 else 2)
                self.assertEqual(refresh.call_count, int(status == 401))

    def test_identity_is_masked_and_never_in_model_projection(self):
        result = read_status(AccountClient())
        self.assertEqual(result.data["masked_phone"], "138****8000")
        public = json.dumps(result.to_dict())
        model = json.dumps(result.to_model_dict())
        for sensitive in ("private-account", "13800138000", "totalDirectLinkTraffic", "vipStatus"):
            self.assertNotIn(sensitive, public)
            self.assertNotIn(sensitive, model)
        self.assertNotIn("Alice", model)

    def test_disconnected_and_close_failure(self):
        client = AccountClient()
        with (
            mock.patch.object(AccountClient, "logged_in", new=False),
            self.assertRaises(AgentToolError) as caught,
        ):
            read_status(client)
        self.assertEqual(caught.exception.code, "precondition_failed")
        self.assertEqual(client.calls, [])
        self.assertTrue(client.closed)
        with mock.patch.object(client, "close", side_effect=RuntimeError("close-failed")):
            self.assertTrue(read_status(client).data["storage"]["reported"])


class GuangYaAccountKernelTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_capacity_query_reaches_same_read_port_and_projected_result(self):
        catalog = catalog_from_tool_specs(build_tool_specs())
        selection = CapabilityRetriever().retrieve("云盘空间概览", catalog)
        self.assertIn("guangya.account.status", selection.names)
        self.assertLessEqual(len(selection.tools), 12)
        state = InMemorySessionStateStore()
        pipeline = ToolPipeline(catalog=catalog, state_store=state)
        lease, _ = await state.begin_turn(owner="test-owner", session_id="test-session", request_id="test")

        async def progress(_payload):
            pass

        context = ToolCallContext(
            owner="test-owner", session_id="test-session", request_id="test",
            turn_id=lease.turn_id, lease=lease, cancellation=CancellationToken(), report_progress=progress,
        )
        with mock.patch.object(actions, "GuangYaClient", return_value=AccountClient()):
            result = await pipeline.execute("guangya.account.status", {}, context=context)
        self.assertEqual(result.tool.effect, ToolEffect.READ)
        self.assertTrue(result.outcome.public_content["data"]["storage"]["reported"])
        self.assertTrue(json.loads(result.outcome.model_content)["data"]["storage"]["reported"])
        self.assertNotIn("private-account", json.dumps(result.outcome.public_content))
        self.assertIsNone(result.outcome.effect_plan)


if __name__ == "__main__":
    unittest.main()
