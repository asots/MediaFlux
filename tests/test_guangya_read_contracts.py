"""光鸭真盘响应契约：失败非空、游标翻页、任务终态与写后复核。"""
from __future__ import annotations

import json
import unittest
from copy import deepcopy
from unittest import mock

import httpx

from app.agent import guangya_recycle_actions as recycle
from app.agent import guangya_share_actions as shares
from app.agent import guangya_workspace_actions as workspace
from app.agent.errors import AgentToolError
from app.agent.models import ToolContext
from app.clients.guangya import GuangYaClient, IncompleteOfflineTaskListError
from tests.test_agent_guangya_sdk_capabilities import _RecycleClient, _ShareClient


class ReadClient(GuangYaClient):
    def __init__(self, payload):
        self._raw = mock.Mock()
        for name in ("fs_files", "fs_detail", "fs_recycle_files", "share_user_list", "cloud_task_list", "get_task_status"):
            getattr(self._raw, name).return_value = payload

    @property
    def raw(self):
        return self._raw


class GuangYaReadContractTests(unittest.TestCase):
    def test_failed_and_unknown_lists_are_not_empty_success(self):
        for payload in (
            None, {}, "unexpected", {"code": 500, "msg": "private-provider-error"},
            {"code": 0, "error": "private-token", "data": {"list": []}},
            {"data": {"code": 401, "list": []}}, {"data": None},
            {"data": {"total": 5}}, {"data": {"list": [None]}},
        ):
            for method in ("list_dir", "list_recycle", "list_user_shares", "list_offline_tasks"):
                with self.subTest(payload=payload, method=method):
                    with self.assertRaises(RuntimeError) as caught:
                        getattr(ReadClient(payload), method)()
                    self.assertNotIn("private-", str(caught.exception))

    def test_error_details_and_task_responses_are_not_missing_objects(self):
        for payload in (None, {}, {"code": 500}, {"error": "private-error"}):
            for method in ("file_info", "task_status"):
                with self.subTest(method=method, payload=payload), self.assertRaises(RuntimeError):
                    getattr(ReadClient(payload), method)("private-id")

    def test_file_absence_needs_a_successful_empty_detail_response(self):
        for payload in ({"data": {}}, {"data": "invalid"}, {"data": {"fileInfo": {}}}):
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                ReadClient(payload).file_info("f")
        self.assertIsNone(ReadClient({"msg": "success", "data": {}}).file_info("f"))
        self.assertIsNone(ReadClient({"code": 0, "data": {"fileInfo": None}}).file_info("f"))

    def test_actual_empty_share_and_nested_list_shapes(self):
        self.assertEqual(GuangYaClient._extract_list({"msg": "success", "data": {}}), [])
        self.assertEqual(GuangYaClient._extract_list({"data": {"total": 0, "list": []}}), [])
        item = {"fileId": "f", "fileName": "sample", "resType": 2}
        for key in ("file_list", "fileList", "files", "list", "res_list"):
            with self.subTest(key=key):
                self.assertEqual(GuangYaClient._extract_list({"data": {key: [item]}}), [item])

    def test_short_pages_with_remaining_total_are_not_truncated(self):
        for method, sdk in (("list_dir", "fs_files"), ("list_recycle", "fs_recycle_files"), ("list_user_shares", "share_user_list")):
            client = ReadClient(None)
            records = [
                {"fileId": "1", "fileName": "one", "shareId": "share-1"},
                {"fileId": "2", "fileName": "two", "shareId": "share-2"},
            ]
            getattr(client.raw, sdk).side_effect = [
                {"msg": "success", "data": {"total": 2, "list": [row]}}
                for row in records
            ]
            with self.subTest(method=method):
                self.assertEqual(len(getattr(client, method)()), 2)
                self.assertEqual(getattr(client.raw, sdk).call_count, 2)

    def test_error_on_second_page_does_not_return_first_page_as_complete(self):
        for method, sdk in (("list_dir", "fs_files"), ("list_recycle", "fs_recycle_files"), ("list_user_shares", "share_user_list")):
            client = ReadClient(None)
            getattr(client.raw, sdk).side_effect = [
                {"data": {"total": 2, "list": [{"fileId": "1", "fileName": "a", "shareId": "s"}]}},
                {"code": 500},
            ]
            with self.subTest(method=method), self.assertRaises(RuntimeError):
                getattr(client, method)()

    def test_offline_cursor_protocol_instead_of_ignored_page_number(self):
        client = ReadClient({"msg": "success", "data": {
            "total": 2, "list": [{"taskId": "a", "fileName": "A", "status": 2}],
            "cursor": "private-cursor", "hasMore": True,
        }})
        client.raw.request.return_value = httpx.Response(200, json={"msg": "success", "data": {
            "total": 2, "list": [{"taskId": "b", "fileName": "B", "status": 2}],
            "cursor": "next-cursor", "hasMore": False,
        }})
        result = client.list_offline_tasks()
        self.assertEqual([x["id"] for x in result], ["a", "b"])
        client.raw.cloud_task_list.assert_called_once()
        request = client.raw.request.call_args
        self.assertEqual(request.kwargs["json"]["cursor"], "private-cursor")
        self.assertNotIn("page", request.kwargs["json"])

    def test_offline_cursor_end_omits_list_but_retains_total(self):
        client = ReadClient({"msg": "success", "data": {
            "total": 1, "list": [{"taskId": "a", "fileName": "A", "status": 2}],
            "cursor": "cursor", "hasMore": True,
        }})
        client.raw.request.return_value = httpx.Response(200, json={"msg": "success", "data": {
            "total": 1, "cursor": "cursor", "statusCounts": [],
        }})
        self.assertEqual(len(client.list_offline_tasks()), 1)
        client.raw.request.assert_called_once()
        client.raw.request.return_value = httpx.Response(200, json={"msg": "success", "data": {
            "total": 2, "cursor": "cursor", "statusCounts": [],
        }})
        with self.assertRaises(IncompleteOfflineTaskListError):
            client.list_offline_tasks()

    def test_offline_stuck_cursor_and_repeated_short_page_fail_closed(self):
        first = {"msg": "success", "data": {
            "total": 3, "hasMore": True, "cursor": "cursor",
            "list": [{"taskId": "a", "fileName": "A", "status": 2}],
        }}
        for second_id in ("a", "b"):
            client = ReadClient(first)
            second = deepcopy(first)
            second["data"]["list"][0]["taskId"] = second_id
            client.raw.request.return_value = httpx.Response(200, json=second)
            with self.subTest(second_id=second_id), self.assertRaises(IncompleteOfflineTaskListError):
                client.list_offline_tasks()

    def test_shared_file_cursor_and_budget_are_explicit(self):
        raw = mock.Mock()
        raw.share_access_token.return_value = {"data": {"accessToken": "private-token"}}
        raw.share_files_list.return_value = {"msg": "success", "data": {
            "list": [{"fileId": "f1", "fileName": "one"}], "total": 2, "cursor": "next",
        }}
        raw._public_post.return_value = {"msg": "success", "data": {
            "list": [{"fileId": "f2", "fileName": "two"}], "total": 2, "cursor": "end",
        }}
        with mock.patch("app.clients.guangya._load_raw", return_value=raw):
            result = ReadClient(None).list_share_files("https://www.guangyapan.com/s/public_test", page_size=10)
            self.assertEqual(result["count"], 2)
            self.assertNotIn("private-token", json.dumps(result))
            self.assertEqual(raw._public_post.call_args.args[1]["cursor"], "next")
            with self.assertRaisesRegex(RuntimeError, "分页上限"):
                ReadClient(None).list_share_files("https://www.guangyapan.com/s/public_test", page_size=10, max_pages=1)

    def test_full_share_list_rejects_legacy_full_page_at_budget(self):
        raw = mock.Mock()
        raw.share_access_token.return_value = {"data": {"accessToken": "private-token"}}
        raw.share_files_list.return_value = {"data": {"list": [
            {"fileId": "f1", "fileName": "one"},
            {"fileId": "f2", "fileName": "two"},
        ]}}
        with mock.patch("app.clients.guangya._load_raw", return_value=raw), self.assertRaisesRegex(
            RuntimeError, "分页上限",
        ):
            ReadClient(None).list_share_files(
                "https://www.guangyapan.com/s/public_test", page_size=2, max_pages=1,
            )
        raw.share_files_list.assert_called_once()
        raw._public_post.assert_not_called()

    def test_share_preview_keeps_one_page_when_more_files_exist(self):
        first = {"fileId": "f1", "fileName": "one"}
        cases = (
            {"list": [first], "total": 2, "cursor": "next"},
            {"list": [{"fileId": f"f{i}", "fileName": str(i)} for i in range(200)]},
        )
        for payload in cases:
            raw = mock.Mock()
            raw.share_access_token.return_value = {"data": {"accessToken": "private-token"}}
            raw.share_files_list.return_value = {"msg": "success", "data": payload}
            with self.subTest(count=len(payload["list"])), mock.patch(
                "app.clients.guangya._load_raw", return_value=raw,
            ):
                result = ReadClient(None).inspect_share("https://www.guangyapan.com/s/public_test")
                self.assertEqual(result["count"], len(payload["list"]))
                self.assertEqual(result["access_token"], "private-token")
                self.assertTrue(result["has_more"])
                raw.share_files_list.assert_called_once()
                raw._public_post.assert_not_called()

    def test_share_preview_marks_a_successful_short_page_as_final(self):
        raw = mock.Mock()
        raw.share_access_token.return_value = {"data": {"accessToken": "private-token"}}
        raw.share_files_list.return_value = {"msg": "success", "data": {"list": [
            {"fileId": "f1", "fileName": "one"},
        ]}}
        with mock.patch("app.clients.guangya._load_raw", return_value=raw):
            result = ReadClient(None).inspect_share("https://www.guangyapan.com/s/public_test")
        self.assertEqual(result["count"], 1)
        self.assertFalse(result["has_more"])
        raw.share_files_list.assert_called_once()
        raw._public_post.assert_not_called()

    def test_share_preview_does_not_hide_provider_or_item_errors(self):
        for response in ({"code": 500, "msg": "private-error"}, {"data": {"list": [None]}}):
            raw = mock.Mock()
            raw.share_access_token.return_value = {"data": {"accessToken": "private-token"}}
            raw.share_files_list.return_value = response
            with self.subTest(response=response), mock.patch(
                "app.clients.guangya._load_raw", return_value=raw,
            ), self.assertRaises(RuntimeError):
                ReadClient(None).inspect_share("https://www.guangyapan.com/s/public_test")
            raw._public_post.assert_not_called()


class GuangYaAgentReadCompletionTests(unittest.TestCase):
    def test_unknown_task_and_terminal_error_detail_are_not_success(self):
        client = _RecycleClient()
        for payload, expected, ok in (
            ({"status": "unknown"}, "unknown", False), ({}, "unknown", False),
            ({"status": 0}, "running", True), ({"status": 1}, "running", True),
            ({"status": 2, "detail": {"code": 0}}, "completed", True),
            ({"status": 2, "detail": {"code": 157, "msg": "private-error"}}, "failed", False),
            ({"status": 3}, "failed", False),
            ({"status": 99}, "unknown", False),
        ):
            with (
                self.subTest(payload=payload),
                mock.patch.object(recycle, "GuangYaClient", return_value=client),
                mock.patch.object(client, "task_status", return_value={"data": payload}),
            ):
                result = recycle.query_guangya_task_status({"guangya_task": {"task_id": "task", "operation": "copy"}}, ToolContext())
                self.assertEqual(result.status, expected)
                self.assertEqual(result.ok, ok)
                self.assertNotIn("private-error", json.dumps(result.to_dict()))
                if expected == "completed":
                    self.assertEqual(result.data["progress"], 1.0)

    def test_restore_and_clear_acceptance_survives_failed_verification(self):
        for operation in ("restore", "clear"):
            client = _RecycleClient()
            context = ToolContext(owner="owner", session_id="session")
            original = client.list_recycle

            def read_after_write(*, client=client, original=original, **kwargs):
                if not client.items:
                    raise RuntimeError("provider-private-error")
                return original(**kwargs)

            with mock.patch.object(recycle, "GuangYaClient", return_value=client):
                if operation == "restore":
                    listed = recycle.list_guangya_recycle({"page": 1, "page_size": 50}, context)
                    args = {"guangya_recycle_items": listed.references[0].value, "indices": [1]}
                    _preview, fingerprint = recycle.prepare_restore_guangya_recycle(args, context)
                    execute = recycle.execute_restore_guangya_recycle
                else:
                    args = {}
                    _preview, fingerprint = recycle.prepare_clear_guangya_recycle(args, context)
                    execute = recycle.execute_clear_guangya_recycle
                with mock.patch.object(client, "list_recycle", side_effect=read_after_write):
                    result = execute(args, fingerprint, context)
                self.assertEqual(result.status, "accepted")
                self.assertTrue(result.data["verification_pending"])
                self.assertFalse(result.data["verified"])
                self.assertEqual(result.references[0].kind, "guangya_task")
                self.assertNotIn("provider-private-error", json.dumps(result.to_dict()))

    def test_revoke_acceptance_survives_failed_verification(self):
        client = _ShareClient()
        context = ToolContext(owner="owner", session_id="session")
        original = client.list_user_shares

        def read_after_write(**kwargs):
            if not client.shares:
                raise RuntimeError("private-error")
            return original(**kwargs)

        with mock.patch.object(shares, "GuangYaClient", return_value=client):
            listed = shares.list_guangya_user_shares({"page": 1, "page_size": 50}, context)
            args = {"guangya_shares": listed.references[0].value, "indices": [1]}
            _preview, fingerprint = shares.prepare_revoke_guangya_shares(args, context)
            with mock.patch.object(client, "list_user_shares", side_effect=read_after_write):
                result = shares.execute_revoke_guangya_shares(args, fingerprint, context)
            self.assertEqual(result.status, "accepted")
            self.assertTrue(result.data["verification_pending"])
            self.assertFalse(result.data["verified"])

    def test_share_without_identity_is_not_silently_dropped(self):
        client = _ShareClient()
        client.shares = [{"title": "nameless-id"}]
        with (
            mock.patch.object(shares, "GuangYaClient", return_value=client),
            self.assertRaises(AgentToolError),
        ):
            shares.list_guangya_user_shares({"page": 1, "page_size": 50}, ToolContext())

    def test_model_keeps_type_size_extension_without_provider_identifiers(self):
        page = {
            "observation_ref": "OBS" + "A" * 32, "scope": "scope", "scopes": [],
            "page": 1, "total": 1, "has_more": False, "truncated": False,
            "entries": [{"object_ref": "opaque-object", "object_name": "episode.mkv",
                         "location": "scope", "kind": "video", "size": 456, "extension": "mkv"}],
        }
        with mock.patch.object(workspace, "_read_observation_page", return_value=page):
            result = workspace.query_guangya_filesystem({}, ToolContext())
        entry = result.model_data["entries"][0]
        self.assertEqual((entry["kind"], entry["size"], entry["extension"]), ("video", 456, "mkv"))
        self.assertNotIn("file_id", entry)


if __name__ == "__main__":
    unittest.main()
