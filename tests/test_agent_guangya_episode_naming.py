"""光鸭声明式剧集分季命名与统一确认链路测试。"""

from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from app.agent import guangya_episode_naming_actions as episode_actions
from app.agent import guangya_fs_change_actions as change_actions
from app.agent import guangya_workspace_actions as workspace_actions
from app.agent.errors import AgentToolError
from app.agent.models import ToolContext
from app.clients.guangya import GuangYaFile
from app.modules import guangya_fs_change, guangya_workspace
from app.modules.guangya_episode_naming import (
    GuangYaEpisodeNamingError,
    compile_episode_naming_operations,
)
from tests.support import isolated_test_database


class EpisodeNamingClient:
    def __init__(self, *, target_directories: bool = True, episodes_per_group: int = 100):
        self.logged_in = True
        self.credential_generation = 44
        self.closed = False
        self.directories: dict[str, list[GuangYaFile]] = {
            "0": [GuangYaFile("show", "狐妖小红娘", True, parent_id="0", etag="show")],
            "show": [
                GuangYaFile("release-a", "发布组A", True, parent_id="show", etag="a"),
                GuangYaFile("release-b", "发布组B", True, parent_id="show", etag="b"),
            ],
            "release-a": [],
            "release-b": [],
        }
        for episode in range(1, episodes_per_group + 1):
            self.directories["release-a"].append(
                GuangYaFile(
                    f"a-{episode}",
                    f"[A] Fox Spirit Matchmaker.S01E{episode:03d}.1080p.mkv",
                    False,
                    parent_id="release-a",
                    size=1000 + episode,
                    etag=f"a-{episode}",
                    extension="mkv",
                )
            )
            self.directories["release-b"].append(
                GuangYaFile(
                    f"b-{episode}",
                    f"[B] Fox Spirit Matchmaker.S02E{episode:03d}.1080p.mp4",
                    False,
                    parent_id="release-b",
                    size=2000 + episode,
                    etag=f"b-{episode}",
                    extension="mp4",
                )
            )
        if target_directories:
            self.directories["show"].extend(
                [
                    GuangYaFile("season-1", "Season 01", True, parent_id="show", etag="s1"),
                    GuangYaFile("season-2", "Season 02", True, parent_id="show", etag="s2"),
                ]
            )
            self.directories["season-1"] = [
                GuangYaFile(
                    "season-1-note",
                    "README.txt",
                    False,
                    parent_id="season-1",
                    size=1,
                    etag="note1",
                    extension="txt",
                )
            ]
            self.directories["season-2"] = [
                GuangYaFile(
                    "season-2-note",
                    "README.txt",
                    False,
                    parent_id="season-2",
                    size=1,
                    etag="note2",
                    extension="txt",
                )
            ]

    def list_dir(self, parent_id="0"):
        return [deepcopy(item) for item in self.directories.get(str(parent_id), [])]

    def file_info(self, file_id):
        for items in self.directories.values():
            for item in items:
                if item.file_id == str(file_id):
                    return deepcopy(item)
        return None

    def close(self):
        self.closed = True
        return True


class GuangYaEpisodeNamingTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(isolated_test_database())
        self.enterContext(mock.patch("socket.socket.connect", side_effect=AssertionError("禁止外联")))
        workspace_actions.reset_guangya_workspace_context_for_tests()
        change_actions.reset_guangya_fs_change_context_for_tests()
        self.temp = tempfile.TemporaryDirectory()
        self.obs_dir = Path(self.temp.name) / "observations"
        self.plan_dir = Path(self.temp.name) / "changes"
        self.patches = [
            mock.patch.object(guangya_workspace, "_directory", return_value=self.obs_dir),
            mock.patch.object(guangya_workspace, "get_web_secret", return_value="test-secret"),
            mock.patch.object(
                guangya_workspace,
                "organize_operation_owner_digest",
                side_effect=lambda owner: f"digest:{owner}",
            ),
            mock.patch.object(guangya_fs_change, "_directory", return_value=self.plan_dir),
            mock.patch.object(guangya_fs_change, "get_web_secret", return_value="test-secret"),
            mock.patch.object(
                guangya_fs_change, "_owner_digest", side_effect=lambda owner: f"digest:{owner}"
            ),
            mock.patch.object(
                change_actions,
                "organize_operation_owner_digest",
                side_effect=lambda owner: f"digest:{owner}",
            ),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        workspace_actions.reset_guangya_workspace_context_for_tests()
        change_actions.reset_guangya_fs_change_context_for_tests()
        self.temp.cleanup()

    @staticmethod
    def _arguments() -> dict:
        return episode_actions.guangya_episode_naming_plan_arguments(
            {
                "title": "狐妖小红娘",
                "target_root": "/狐妖小红娘",
                "groups": [
                    {
                        "source_path": "/狐妖小红娘/发布组A",
                        "source_season": 1,
                        "source_episode_start": 1,
                        "source_episode_end": 100,
                        "target_season": 1,
                        "expected_count": 100,
                    },
                    {
                        "source_path": "/狐妖小红娘/发布组B",
                        "source_season": 2,
                        "source_episode_start": 1,
                        "source_episode_end": 100,
                        "target_season": 2,
                        "expected_count": 100,
                    },
                ],
                "trigger_strm": False,
            }
        )

    @staticmethod
    def _plain_suffix_observation(*names: str) -> dict:
        return {
            "plan_id": "plain-suffix-audit",
            "truncated": False,
            "entries": [
                {
                    "handle": f"plain-{index}",
                    "name": name,
                    "is_dir": False,
                    "media_kind": "video",
                    "parent_path": "/Fox Spirit Matchmaker",
                    "size": 1000 + index,
                }
                for index, name in enumerate(names, start=1)
            ],
        }

    def _observe(self, client: EpisodeNamingClient) -> str:
        arguments = workspace_actions.guangya_fs_query_arguments(
            {
                "operation": "tree",
                "path": "/狐妖小红娘",
                "page": 1,
                "page_size": 50,
                "max_items": 500,
                "max_depth": 2,
            }
        )
        with mock.patch.object(workspace_actions, "GuangYaClient", return_value=client):
            result = workspace_actions.query_guangya_filesystem(
                arguments, ToolContext(owner="owner", session_id="session")
            )
        self.assertEqual(result.data["total"], 206)
        self.assertTrue(result.data["has_more"])
        return str(result.data["observation_ref"])

    def test_inspect_returns_compact_complete_source_groups_without_plan(self):
        client = EpisodeNamingClient(episodes_per_group=3)
        arguments = episode_actions.guangya_episode_naming_inspect_arguments(
            {"target_root": "/狐妖小红娘"}
        )
        with mock.patch.object(episode_actions, "GuangYaClient", return_value=client):
            result = episode_actions.inspect_guangya_episode_naming(
                arguments, ToolContext(owner="owner", session_id="session")
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.data["video_count"], 6)
        self.assertEqual(result.data["source_group_count"], 2)
        self.assertEqual(result.data["groups"][0]["positions"][0]["episodes"], "1-3")
        self.assertEqual(result.data["groups"][1]["positions"][0]["source_season"], 2)
        self.assertEqual(list(self.obs_dir.glob("*.json")), [])
        self.assertEqual(list(self.plan_dir.glob("*.json")), [])

    def test_inspect_parses_plain_english_trailing_episodes_in_tv_mapping_context(self):
        observation = self._plain_suffix_observation(
            "Fox Spirit Matchmaker 001.mkv",
            "Fox Spirit Matchmaker 014.mkv",
        )
        arguments = episode_actions.guangya_episode_naming_inspect_arguments(
            {"target_root": "/Fox Spirit Matchmaker"}
        )
        with (
            mock.patch.object(episode_actions, "_fresh_observation", return_value=observation),
            mock.patch.object(episode_actions, "discard_observation"),
        ):
            result = episode_actions.inspect_guangya_episode_naming(
                arguments, ToolContext(owner="owner", session_id="session")
            )

        self.assertEqual(result.data["video_count"], 2)
        self.assertEqual(result.data["unparsed_count"], 0)
        self.assertEqual(result.data["groups"][0]["positions"][0]["episodes"], "1,14")

    def test_plain_suffix_mapping_excludes_specials_without_source_season(self):
        observation = self._plain_suffix_observation(
            "Fox Spirit Matchmaker 014.mkv",
            "Fox Spirit Matchmaker Special 014.mkv",
        )
        group = {
            "source_path": "/Fox Spirit Matchmaker",
            "source_episode_start": 14,
            "source_episode_end": 14,
            "target_season": 1,
            "expected_count": 1,
        }
        compiled = compile_episode_naming_operations(
            observation,
            title="Fox Spirit Matchmaker",
            target_root="/Fox Spirit Matchmaker",
            groups=[group],
        )

        self.assertEqual(compiled["selected_files"], 1)
        self.assertEqual(compiled["groups"][0]["matched"], 1)
        relocate = next(item for item in compiled["operations"] if item["op"] == "batch_relocate")
        self.assertEqual([item["object_ref"] for item in relocate["items"]], ["PLAIN-1"])

        mismatch = dict(group, expected_count=2)
        with self.assertRaisesRegex(GuangYaEpisodeNamingError, "预期 2 集，实际匹配 1 集"):
            compile_episode_naming_operations(
                observation,
                title="Fox Spirit Matchmaker",
                target_root="/Fox Spirit Matchmaker",
                groups=[mismatch],
            )

    def test_compact_plan_freezes_two_hundred_files_once(self):
        client = EpisodeNamingClient()
        arguments = self._arguments()
        with (
            mock.patch.object(episode_actions, "GuangYaClient", return_value=client),
            mock.patch.object(change_actions, "GuangYaClient", return_value=client),
        ):
            confirmation, fingerprint = episode_actions.prepare_guangya_episode_naming_confirmation(
                arguments, ToolContext(owner="owner", session_id="session")
            )

        self.assertEqual(confirmation.status, "confirmation_required")
        self.assertEqual(confirmation.data["total"], 200)
        self.assertEqual(confirmation.data["relocate_count"], 200)
        self.assertEqual(confirmation.data["create_directory_count"], 0)
        self.assertEqual(confirmation.data["episode_naming"]["selected_files"], 200)
        self.assertEqual(len(confirmation.data["episode_naming"]["groups"]), 2)
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(len(list(self.plan_dir.glob("*.json"))), 1)

        flow = change_actions._flow("owner")
        self.assertIsNotNone(flow)
        plan = guangya_fs_change.load_fs_change_plan(
            flow.plan_id, owner="owner", expected_fingerprint=flow.fingerprint
        )
        self.assertEqual(len(plan["operations"]), 200)
        self.assertTrue(all(item["op"] == "relocate" for item in plan["operations"]))
        self.assertNotIn("batch_relocate", str(plan["operations"]))
        self.assertEqual(plan["operations"][0]["new_name"], "狐妖小红娘 - S01E01.mkv")
        self.assertEqual(plan["operations"][-1]["new_name"], "狐妖小红娘 - S02E100.mp4")

    def test_two_hundred_files_and_missing_season_directories_stay_one_plan(self):
        client = EpisodeNamingClient(target_directories=False)
        arguments = self._arguments()
        with (
            mock.patch.object(episode_actions, "GuangYaClient", return_value=client),
            mock.patch.object(change_actions, "GuangYaClient", return_value=client),
        ):
            confirmation, fingerprint = episode_actions.prepare_guangya_episode_naming_confirmation(
                arguments, ToolContext(owner="owner", session_id="session")
            )

        self.assertEqual(confirmation.status, "confirmation_required")
        self.assertEqual(confirmation.data["total"], 202)
        self.assertEqual(confirmation.data["relocate_count"], 200)
        self.assertEqual(confirmation.data["create_directory_count"], 2)
        self.assertEqual(confirmation.data["episode_naming"]["selected_files"], 200)
        self.assertEqual(confirmation.data["episode_naming"]["created_directories"], 2)
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(len(list(self.plan_dir.glob("*.json"))), 1)

        flow = change_actions._flow("owner")
        self.assertIsNotNone(flow)
        plan = guangya_fs_change.load_fs_change_plan(
            flow.plan_id, owner="owner", expected_fingerprint=flow.fingerprint
        )
        self.assertEqual(len(plan["operations"]), 202)
        self.assertEqual(
            sum(item["op"] == "create_directory" for item in plan["operations"]), 2
        )
        self.assertEqual(sum(item["op"] == "relocate" for item in plan["operations"]), 200)

    def test_more_than_two_hundred_media_files_requires_complete_season_split(self):
        client = EpisodeNamingClient(episodes_per_group=101)
        payload = guangya_workspace.create_directory_observation(
            client,
            owner="owner",
            path="/狐妖小红娘",
            recursive=True,
            max_items=500,
            max_depth=2,
        )
        groups = self._arguments()["groups"]
        for group in groups:
            group["source_episode_end"] = 101
            group["expected_count"] = 101
        with self.assertRaisesRegex(GuangYaEpisodeNamingError, "202 个媒体文件"):
            compile_episode_naming_operations(
                payload,
                title="狐妖小红娘",
                target_root="/狐妖小红娘",
                groups=groups,
            )

    def test_expected_count_mismatch_rejects_before_freezing(self):
        client = EpisodeNamingClient(episodes_per_group=3)
        arguments = self._arguments()
        arguments["groups"][0]["source_episode_end"] = 3
        arguments["groups"][0]["expected_count"] = 4
        with (
            mock.patch.object(episode_actions, "GuangYaClient", return_value=client),
            mock.patch.object(change_actions, "GuangYaClient", return_value=client),
            self.assertRaises(AgentToolError) as raised,
        ):
            episode_actions.prepare_guangya_episode_naming_confirmation(
                arguments, ToolContext(owner="owner", session_id="session")
            )
        self.assertEqual(raised.exception.code, "precondition_failed")
        self.assertIn("预期 4 集，实际匹配 3 集", str(raised.exception))
        self.assertEqual(list(self.plan_dir.glob("*.json")), [])

    def test_directory_name_fragment_resolves_unique_release_group(self):
        client = EpisodeNamingClient(episodes_per_group=3)
        payload = guangya_workspace.create_directory_observation(
            client,
            owner="owner",
            path="/狐妖小红娘",
            recursive=True,
            max_items=100,
            max_depth=2,
        )
        compiled = compile_episode_naming_operations(
            payload,
            title="狐妖小红娘",
            target_root="/狐妖小红娘",
            groups=[
                {
                    "source_directory_contains": "发布组A",
                    "source_season": 1,
                    "source_episode_start": 1,
                    "source_episode_end": 3,
                    "target_season": 1,
                    "expected_count": 3,
                }
            ],
        )
        self.assertEqual(compiled["effective_total"], 3)
        self.assertEqual(compiled["groups"][0]["matched"], 3)

    def test_validator_rejects_ambiguous_or_unsupported_group_fields(self):
        with self.assertRaisesRegex(AgentToolError, "不支持的参数"):
            episode_actions.guangya_episode_naming_plan_arguments(
                {
                    "title": "狐妖小红娘",
                    "target_root": "/狐妖小红娘",
                    "groups": [
                        {
                            "source_path": "/狐妖小红娘/发布组A",
                            "source_episode_start": 1,
                            "source_episode_end": 10,
                            "target_season": 1,
                            "regex": ".*",
                        }
                    ],
                }
            )
