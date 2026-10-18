"""光鸭只读工作区快照的持久化与公开投影测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.agent.domain_catalog import DomainActions
from app.agent.domains.guangya import workspace_actions
from app.agent.domains.guangya.workspace_actions import GuangyaWorkspaceActions
from app.agent.models import ToolContext
from app.clients.guangya import GuangYaFile
from app.modules import guangya_workspace


class FakeWorkspaceClient:
    def __init__(self):
        self.logged_in = True
        self.credential_generation = 21
        self.closed = False
        self.list_calls: list[str] = []
        self.directories = {
            "0": [GuangYaFile("a", "a", True, parent_id="0", etag="root")],
            "a": [
                GuangYaFile(
                    "spam",
                    "[最新地址]ABC-123.mp4",
                    False,
                    parent_id="a",
                    size=100,
                    etag="spam-etag",
                    extension="mp4",
                ),
                GuangYaFile(
                    "clean",
                    "DEF-456.mp4",
                    False,
                    parent_id="a",
                    size=90,
                    etag="clean-etag",
                    extension="mp4",
                ),
                GuangYaFile(
                    "nested",
                    "spam.example.com",
                    True,
                    parent_id="a",
                    etag="nested-etag",
                    updated_at=3,
                ),
            ],
            "nested": [
                GuangYaFile(
                    "subtitle",
                    "广告词-GHI-789.CHT.srt",
                    False,
                    parent_id="nested",
                    size=10,
                    etag="sub-etag",
                    extension="srt",
                )
            ],
        }

    def list_dir(self, parent_id="0"):
        self.list_calls.append(str(parent_id))
        return [deepcopy(item) for item in self.directories.get(str(parent_id), [])]

    def file_info(self, file_id):
        for items in self.directories.values():
            for item in items:
                if item.file_id == str(file_id):
                    return deepcopy(item)
        return None

    def close(self):
        self.closed = True


class GuangYaWorkspaceAgentTests(unittest.TestCase):
    def setUp(self):
        self._domain_actions = DomainActions()
        self.temp = tempfile.TemporaryDirectory()
        self.obs_dir = Path(self.temp.name) / "observations"
        self.patches = [
            mock.patch.object(
                guangya_workspace, "_directory", return_value=self.obs_dir
            ),
            mock.patch.object(
                guangya_workspace, "get_web_secret", return_value="test-secret"
            ),
            mock.patch.object(
                guangya_workspace,
                "organize_operation_owner_digest",
                side_effect=lambda owner: f"digest:{owner}",
            ),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temp.cleanup()

    def _query(self, client, **overrides):
        arguments = workspace_actions.guangya_fs_query_arguments(
            {
                "operation": "tree",
                "path": "/a",
                "page": 1,
                "page_size": 10,
                "max_items": 100,
                **overrides,
            }
        )
        with mock.patch.object(workspace_actions, "GuangYaClient", return_value=client):
            return self._domain_actions.workspace.query_guangya_filesystem(
                arguments, ToolContext(owner="owner", session_id="session")
            )

    def _create_observation(self, owner="owner"):
        payload = guangya_workspace.create_directory_observation(
            FakeWorkspaceClient(),
            owner=owner,
            path="/a",
            recursive=False,
            max_items=10,
        )
        return (
            guangya_workspace.observation_ref(payload["plan_id"]),
            guangya_workspace._plan_path(payload["plan_id"]),
        )

    def test_directory_observation_returns_names_and_opaque_refs_without_ids(self):
        result = self._query(FakeWorkspaceClient())
        self.assertTrue(result.ok)
        self.assertRegex(result.data["observation_ref"], "^OBS[0-9A-F]{32}$")
        self.assertEqual(result.data["total"], 4)
        names = {item["object_name"] for item in result.data["entries"]}
        self.assertIn("[最新地址]ABC-123.mp4", names)
        self.assertIn("广告词-GHI-789.CHT.srt", names)
        for item in result.data["entries"]:
            self.assertRegex(item["object_ref"], "^OBJ[0-9A-F]{24}$")
            self.assertNotIn("file_id", item)
            self.assertNotIn("parent_id", item)
            self.assertNotIn("path", item)

    def test_directory_itself_is_selectable_without_changing_content_pagination(self):
        client = FakeWorkspaceClient()
        result = self._query(client, operation="list", page_size=1)
        self.assertEqual(result.model_data["operation"], "list")
        self.assertEqual(result.data["total"], 3)
        self.assertTrue(result.data["has_more"])
        self.assertEqual(len(result.data["entries"]), 1)
        root = result.model_data["scope_objects"][0]
        self.assertEqual((root["name"], root["kind"]), ("a", "directory"))
        self.assertNotIn(root["ref"], [item["ref"] for item in result.model_data["entries"]])
        payload = guangya_workspace.load_directory_observation(result.data["observation_ref"], owner="owner")
        selected = guangya_workspace.observation_entry_map(payload)[root["ref"]]
        self.assertEqual((selected["file_id"], selected["parent_id"], selected["parent_path"]), ("a", "0", "/"))
        source = guangya_workspace.capture_observation_source(payload, [root["ref"]])
        self.assertTrue(source["available"])
        self.assertEqual(source["baseline"]["selected"][0]["file_id"], "a")
        calls = list(client.list_calls)
        following = self._domain_actions.workspace.query_guangya_filesystem(
            workspace_actions.guangya_fs_query_arguments({"observation_ref": result.data["observation_ref"], "page": 2, "page_size": 1}),
            ToolContext(owner="owner", session_id="session"))
        self.assertEqual(following.data["scope_objects"], result.data["scope_objects"])
        self.assertEqual(client.list_calls, calls)  # 不为目录引用增加云端读取
        self.assertEqual(following.data["total"], 3)

    def test_empty_or_filtered_directory_is_not_a_matching_child(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                client = FakeWorkspaceClient()
                if empty:
                    client.directories["a"] = []
                result = self._query(client, operation="search", query="no-such-video", kinds=["video"])
                self.assertEqual(result.data["total"], 0)
                self.assertEqual(result.data["entries"], [])
                self.assertEqual(result.status, "empty")
                self.assertEqual(result.model_data["scope_objects"][0]["kind"], "directory")
                self.assertFalse(result.data["has_more"])

    def test_account_root_and_legacy_roots_do_not_gain_writable_handles(self):
        result = self._query(FakeWorkspaceClient(), operation="list", path="/")
        self.assertEqual(result.data["scope_objects"], [])
        payload = guangya_workspace.load_directory_observation(result.data["observation_ref"], owner="owner")
        self.assertNotIn("0", [entry["file_id"] for entry in guangya_workspace.observation_entry_map(payload).values()])
        # 已保存的旧快照只有来源身份，不推测或拼凑可写对象。
        result = self._query(FakeWorkspaceClient(), operation="list")
        payload = guangya_workspace.load_directory_observation(result.data["observation_ref"], owner="owner")
        payload["roots"] = [{key: root[key] for key in ("file_id", "parent_id", "path", "is_dir")} for root in payload["roots"]]
        guangya_workspace._atomic_write(payload)
        old = guangya_workspace.load_directory_observation(result.data["observation_ref"], owner="owner")
        self.assertEqual(guangya_workspace.observation_page(old)["scope_objects"], [])
        self.assertEqual(len(guangya_workspace.observation_entry_map(old)), len(old["entries"]))

    def test_overlapping_root_and_child_share_one_identity(self):
        with mock.patch.object(workspace_actions, "GuangYaClient", return_value=FakeWorkspaceClient()):
            result = self._domain_actions.workspace.query_guangya_filesystem(
                workspace_actions.guangya_fs_query_arguments({"paths": ["/a", "/a/spam.example.com"], "operation": "tree"}),
                ToolContext(owner="owner", session_id="session"))
        self.assertEqual(result.data["total"], 4)
        roots = result.model_data["scope_objects"]
        self.assertEqual([root["name"] for root in roots], ["a", "spam.example.com"])
        child = next(entry for entry in result.model_data["entries"] if entry["name"] == "spam.example.com")
        self.assertEqual(roots[1]["ref"], child["ref"])
        payload = guangya_workspace.load_directory_observation(result.data["observation_ref"], owner="owner")
        self.assertEqual(len(guangya_workspace.observation_entry_map(payload)), 5)

    def test_default_page_size_supports_large_model_observations(self):
        client = FakeWorkspaceClient()
        client.directories["a"] = [
            GuangYaFile(
                f"item-{index}",
                f"Episode-{index:03d}.mkv",
                False,
                parent_id="a",
                size=index,
                extension="mkv",
            )
            for index in range(60)
        ]
        arguments = workspace_actions.guangya_fs_query_arguments(
            {"operation": "list", "path": "/a", "max_items": 100}
        )
        with mock.patch.object(workspace_actions, "GuangYaClient", return_value=client):
            result = self._domain_actions.workspace.query_guangya_filesystem(
                arguments, ToolContext(owner="owner", session_id="session")
            )
        self.assertEqual(result.data["page_size"], 50)
        self.assertEqual(len(result.data["entries"]), 50)
        self.assertTrue(result.data["has_more"])

    def test_query_normalizes_tree_with_query_and_filters_to_video(self):
        arguments = workspace_actions.guangya_fs_query_arguments(
            {
                "operation": "tree",
                "path": "/a",
                "query": "ABC",
                "kinds": ["video"],
            }
        )
        self.assertEqual(arguments["operation"], "search")
        self.assertEqual(arguments["kinds"], ("video",))
        with mock.patch.object(
            workspace_actions, "GuangYaClient", return_value=FakeWorkspaceClient()
        ):
            result = self._domain_actions.workspace.query_guangya_filesystem(
                arguments, ToolContext(owner="owner", session_id="session")
            )
        self.assertEqual(result.data["operation"], "search")
        self.assertEqual(result.data["kinds"], ["video"])
        self.assertEqual(
            [item["object_name"] for item in result.data["entries"]],
            ["[最新地址]ABC-123.mp4"],
        )

    def test_query_can_merge_multiple_exact_directories_into_one_snapshot(self):
        client = FakeWorkspaceClient()
        client.directories["0"].append(
            GuangYaFile("b", "b", True, parent_id="0", etag="root-b")
        )
        client.directories["b"] = [
            GuangYaFile(
                "episode-b",
                "Episode-B.mkv",
                False,
                parent_id="b",
                size=80,
                extension="mkv",
            )
        ]
        arguments = workspace_actions.guangya_fs_query_arguments(
            {
                "operation": "list",
                "paths": ["/a", "/b"],
                "kinds": ["video"],
            }
        )
        with mock.patch.object(workspace_actions, "GuangYaClient", return_value=client):
            result = self._domain_actions.workspace.query_guangya_filesystem(
                arguments, ToolContext(owner="owner", session_id="session")
            )
        self.assertEqual(result.data["scope"], "2 个目录")
        self.assertEqual(result.data["scopes"], ["a", "b"])
        self.assertEqual(result.data["max_depth"], 0)
        self.assertEqual(
            {item["object_name"] for item in result.data["entries"]},
            {"[最新地址]ABC-123.mp4", "DEF-456.mp4", "Episode-B.mkv"},
        )
        self.assertEqual(
            {item["location"] for item in result.data["entries"]}, {"a", "b"}
        )

    def test_query_rejects_unknown_kind_filter(self):
        with self.assertRaisesRegex(
            workspace_actions.AgentToolError, "不支持的对象类型"
        ):
            workspace_actions.guangya_fs_query_arguments(
                {"operation": "tree", "path": "/a", "kinds": ["archive"]}
            )

    def test_missing_exact_path_returns_actionable_safe_error(self):
        arguments = workspace_actions.guangya_fs_query_arguments(
            {"operation": "list", "path": "/a/not-there"}
        )
        with (
            mock.patch.object(
                workspace_actions, "GuangYaClient", return_value=FakeWorkspaceClient()
            ),
            self.assertRaisesRegex(workspace_actions.AgentToolError, "请先列出父目录"),
        ):
            self._domain_actions.workspace.query_guangya_filesystem(
                arguments, ToolContext(owner="owner", session_id="session")
            )

    def test_observation_can_continue_by_ref_and_is_owner_bound(self):
        first = self._query(FakeWorkspaceClient(), page_size=2)
        self.assertEqual(
            self._domain_actions.workspace.latest_guangya_observation_cursor("owner"),
            {
                "observation_ref": first.data["observation_ref"],
                "page": 1,
                "page_size": 2,
                "has_more": True,
            },
        )
        arguments = workspace_actions.guangya_fs_query_arguments(
            {
                "observation_ref": first.data["observation_ref"],
                "page": 2,
                "page_size": 2,
            }
        )
        second = self._domain_actions.workspace.query_guangya_filesystem(
            arguments, ToolContext(owner="owner", session_id="session")
        )
        self.assertEqual(second.data["page"], 2)
        self.assertEqual(len(second.data["entries"]), 2)
        self.assertEqual(
            self._domain_actions.workspace.latest_guangya_observation_cursor("owner")["has_more"],
            False,
        )
        with self.assertRaisesRegex(Exception, "不属于当前会话"):
            guangya_workspace.load_directory_observation(
                first.data["observation_ref"], owner="other-owner"
            )

    def test_expired_observation_is_removed_by_maintenance(self):
        with mock.patch.object(
            guangya_workspace.time, "time", return_value=1000.0
        ) as clock:
            observations = [self._create_observation(), self._create_observation()]
            stored = {path: path.read_bytes() for _ref, path in observations}
            expiry = 1000.0 + 10 * 60  # 固定 10 分钟 TTL，维护不能续期。
            clock.return_value = expiry - 1
            result = guangya_workspace.maintain_workspace_observations(
                preserve_plan_id=observations[-1][1].stem
            )
            self.assertEqual(result["removed"], 0)
            for ref, path in observations:
                payload = guangya_workspace.load_directory_observation(
                    ref, owner="owner"
                )
                self.assertEqual(payload["expires_at_epoch"], expiry)
                self.assertEqual(path.read_bytes(), stored[path])

            clock.return_value = expiry
            ref, path = observations[0]
            with self.assertRaisesRegex(
                guangya_workspace.GuangYaWorkspaceStale, "已过期"
            ):
                guangya_workspace.load_directory_observation(ref, owner="owner")
            self.assertFalse(path.exists())
            # 保留标记只豁免容量淘汰，不能延长 TTL。
            result = guangya_workspace.maintain_workspace_observations(
                preserve_plan_id=observations[-1][1].stem
            )
            self.assertEqual(result, {"removed": 1, "remaining": 0, "bytes": 0})
            self.assertFalse(observations[-1][1].exists())

    def test_observation_capacity_is_fair_per_owner(self):
        with mock.patch.object(
            guangya_workspace.time, "time", return_value=1000.0
        ) as clock:
            owner_b_ref, owner_b_path = self._create_observation("owner-b")
            clock.return_value = 1001.0
            old_a_ref, old_a_path = self._create_observation("owner-a")
            clock.return_value = 1002.0
            new_a_ref, new_a_path = self._create_observation("owner-a")
            owner_budget = max(old_a_path.stat().st_size, new_a_path.stat().st_size)
            global_budget = sum(
                path.stat().st_size for path in (owner_b_path, old_a_path, new_a_path)
            )
            with (
                mock.patch.object(
                    guangya_workspace, "_MAX_OWNER_STORAGE_BYTES", owner_budget
                ),
                mock.patch.object(
                    guangya_workspace, "_MAX_STORAGE_BYTES", global_budget
                ),
            ):
                result = guangya_workspace.maintain_workspace_observations()

            self.assertEqual(
                result,
                {
                    "removed": 1,
                    "remaining": 2,
                    "bytes": owner_b_path.stat().st_size + new_a_path.stat().st_size,
                },
            )
            with self.assertRaises(guangya_workspace.GuangYaWorkspaceError):
                guangya_workspace.load_directory_observation(old_a_ref, owner="owner-a")
            # owner-b 的观察更老，但仅 owner-a 超出字节预算，不能牵连其它 owner。
            for ref, path, owner in (
                (owner_b_ref, owner_b_path, "owner-b"),
                (new_a_ref, new_a_path, "owner-a"),
            ):
                self.assertEqual(
                    guangya_workspace.load_directory_observation(ref, owner=owner)[
                        "plan_id"
                    ],
                    path.stem,
                )
            self.assertLessEqual(new_a_path.stat().st_size, owner_budget)

    def test_global_byte_budget_evicts_oldest_observation_across_owners(self):
        with mock.patch.object(
            guangya_workspace.time, "time", return_value=1000.0
        ) as clock:
            observations = []
            for index, owner in enumerate(("owner-a", "owner-b", "owner-c")):
                clock.return_value = 1000.0 + index
                observations.append((owner, *self._create_observation(owner)))
            total_bytes = sum(
                path.stat().st_size for _owner, _ref, path in observations
            )
            budget = sum(path.stat().st_size for _owner, _ref, path in observations[1:])
            with (
                mock.patch.object(
                    guangya_workspace, "_MAX_OWNER_STORAGE_BYTES", total_bytes
                ),
                mock.patch.object(guangya_workspace, "_MAX_STORAGE_BYTES", budget),
            ):
                result = guangya_workspace.maintain_workspace_observations()
                self.assertEqual(
                    result, {"removed": 1, "remaining": 2, "bytes": budget}
                )
                self.assertEqual(
                    guangya_workspace.maintain_workspace_observations(),
                    {"removed": 0, "remaining": 2, "bytes": budget},
                )
            owner, ref, path = observations[0]
            self.assertFalse(path.exists())
            with self.assertRaises(guangya_workspace.GuangYaWorkspaceError):
                guangya_workspace.load_directory_observation(ref, owner=owner)
            for owner, ref, path in observations[1:]:
                self.assertEqual(
                    guangya_workspace.load_directory_observation(ref, owner=owner)[
                        "plan_id"
                    ],
                    path.stem,
                )

    def test_new_observation_is_preserved_under_byte_pressure(self):
        for limit in ("_MAX_OWNER_STORAGE_BYTES", "_MAX_STORAGE_BYTES"):
            with (
                self.subTest(limit=limit),
                mock.patch.object(
                    guangya_workspace, "_directory", return_value=self.obs_dir / limit
                ),
                mock.patch.object(
                    guangya_workspace.time, "time", return_value=1001.0
                ) as clock,
            ):
                _old_ref, old_path = self._create_observation()
                budget = old_path.stat().st_size
                # 模拟时钟回拨：刚写入的观察排序更老，仍必须被 preserve 保护。
                clock.return_value = 1000.0
                with mock.patch.object(guangya_workspace, limit, budget):
                    new_ref, new_path = self._create_observation()
                self.assertFalse(old_path.exists())
                self.assertEqual(list(new_path.parent.glob("*.json")), [new_path])
                self.assertEqual(
                    guangya_workspace.load_directory_observation(
                        new_ref, owner="owner"
                    )["plan_id"],
                    new_path.stem,
                )

    def test_maintenance_keeps_valid_signatures_and_removes_tampered_observations(self):
        with mock.patch.object(guangya_workspace.time, "time", return_value=1000.0):
            valid_ref, valid_path = self._create_observation()
            tampered_ref, tampered_path = self._create_observation()
            original = valid_path.read_bytes()
            tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
            tampered["entries"][0]["name"] = "tampered.mp4"
            # 不重新签名，确保维护后的读取仍校验内容完整性。
            tampered_path.write_text(
                json.dumps(tampered, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                guangya_workspace.GuangYaWorkspaceError, "校验失败"
            ):
                guangya_workspace.load_directory_observation(
                    tampered_ref, owner="owner"
                )
            result = guangya_workspace.maintain_workspace_observations(
                preserve_plan_id=tampered_path.stem
            )
            self.assertEqual(
                result, {"removed": 1, "remaining": 1, "bytes": len(original)}
            )
            self.assertFalse(tampered_path.exists())
            self.assertEqual(valid_path.read_bytes(), original)
            self.assertEqual(
                guangya_workspace.load_directory_observation(valid_ref, owner="owner")[
                    "plan_id"
                ],
                valid_path.stem,
            )

    def test_stale_observation_save_discards_only_new_snapshot(self):
        client = FakeWorkspaceClient()
        previous = guangya_workspace.create_directory_observation(
            client, owner="owner", path="/a", recursive=False, max_items=10
        )
        previous_ref = guangya_workspace.observation_ref(previous["plan_id"])

        class StaleRepository:
            def __init__(self):
                self.begin_called = False

            def begin_context_update(self, *, owner, context_type):
                self.begin_called = True
                return (
                    SimpleNamespace(
                        payload={"observation_ref": previous_ref},
                        generation=2,
                        revision=7,
                    ),
                    workspace_actions.AgentContextWriteGuard(2, 7),
                )

            def replace_latest_guarded(self, **_kwargs):
                return None

        repository = StaleRepository()
        self._domain_actions.workspace = GuangyaWorkspaceActions(repository)
        original_list_dir = client.list_dir

        def guarded_list_dir(parent_id="0"):
            self.assertTrue(repository.begin_called)
            return original_list_dir(parent_id)

        client.list_dir = guarded_list_dir
        with (
            mock.patch.object(workspace_actions, "GuangYaClient", return_value=client),
            self.assertRaisesRegex(Exception, "更新请求取代"),
        ):
            self._domain_actions.workspace.query_guangya_filesystem(
                workspace_actions.guangya_fs_query_arguments(
                    {
                        "operation": "list",
                        "path": "/a",
                        "page": 1,
                        "page_size": 10,
                        "max_items": 10,
                    }
                ),
                ToolContext(owner="owner", session_id="session"),
            )
        self.assertTrue(guangya_workspace._plan_path(previous["plan_id"]).is_file())
        self.assertEqual(
            [path.stem for path in self.obs_dir.glob("*.json")], [previous["plan_id"]]
        )

    def test_new_observation_keeps_previous_snapshot_for_multi_step_planning(self):
        client = FakeWorkspaceClient()
        context = ToolContext(owner="owner", session_id="session")
        with mock.patch.object(workspace_actions, "GuangYaClient", return_value=client):
            first = self._domain_actions.workspace.query_guangya_filesystem(
                workspace_actions.guangya_fs_query_arguments(
                    {"operation": "list", "path": "/a"}
                ),
                context,
            )
            self._domain_actions.workspace.query_guangya_filesystem(
                workspace_actions.guangya_fs_query_arguments(
                    {"operation": "stat", "path": "/a/DEF-456.mp4"}
                ),
                context,
            )

        preserved = guangya_workspace.load_directory_observation(
            first.data["observation_ref"], owner="owner"
        )
        self.assertEqual(
            preserved["plan_id"], first.data["observation_ref"][3:].lower()
        )
