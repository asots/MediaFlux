"""Agent 统一资源接入链路与传统 Telegram 隔离测试。"""

from __future__ import annotations

import os
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from app import database as db
from app.agent.domains.resources.candidate_context import (
    RecentResourceCandidateStore,
    new_resource_search_id,
    safe_resource_snapshot,
)
from app.agent.domains.resources.ingest_actions import (
    AgentIngestSessionStore,
    IngestActions,
    ingest_inspect_arguments,
    ingest_status_arguments,
    ingest_submit_arguments,
)
from app.agent.errors import AgentToolError
from app.agent.kernel.adapters import TurnView
from app.agent.kernel.projection import DefaultProjector
from app.agent.models import ToolContext, ToolResult
from app.agent.public_view import format_public_result, public_result_state
from app.agent.state_commit import (
    AgentStateCommitBuffer,
    defer_agent_state_commits,
    stage_agent_resource_candidates,
)
from app.bot.agent_adapter import _render_turn
from app.bot.telegram_markdown import render_telegram_markdown
from app.modules import download_dispatcher as dispatcher
from tests.support import isolated_test_database

_MAGNET = "magnet:?xt=urn:btih:" + "a" * 40 + "&dn=Demo"
_SHARE = "https://www.guangyapan.com/s/1938060615538466851_aecd7hu726g3whGl#/share"


def _resource_result(
    result_id: str, title: str, *additional: tuple[str, str]
) -> ToolResult:
    return ToolResult(
        True,
        "success",
        "searched",
        data={
            "items": [
                {
                    "result_id": candidate_id,
                    "title": candidate_title,
                    "site_id": "mikan",
                    "site_name": "Mikan",
                    "size_text": "1.2 GiB",
                    "download_state": "ready",
                    "download_kinds": ["magnet"],
                }
                for candidate_id, candidate_title in ((result_id, title), *additional)
            ]
        },
    )


class AgentIngestArgumentTests(unittest.TestCase):
    def test_submit_arguments_are_public_coordinates_only(self):
        self.assertEqual(
            ingest_submit_arguments(
                {
                    "source_type": "resource_candidates",
                    "positions": [2, 1, 2],
                    "target": "both",
                }
            ),
            {
                "source_type": "resource_candidates",
                "positions": [2, 1],
                "target": "both",
            },
        )
        with self.assertRaises(AgentToolError):
            ingest_submit_arguments(
                {"source_type": "direct_url", "target": "qb", "input": _MAGNET}
            )

    def test_resource_search_id_is_optional_but_strict_when_present(self):
        search_id = "rs_1234567890abcdef"
        self.assertEqual(
            ingest_submit_arguments(
                {
                    "source_type": "resource_candidates",
                    "positions": [1],
                    "target": "qb",
                    "search_id": search_id,
                }
            )["search_id"],
            search_id,
        )
        with self.assertRaises(AgentToolError):
            ingest_submit_arguments(
                {
                    "source_type": "resource_candidates",
                    "positions": [1],
                    "target": "qb",
                    "search_id": "latest",
                }
            )
        with self.assertRaises(AgentToolError):
            ingest_submit_arguments(
                {"source_type": "direct_url", "target": "qb", "search_id": search_id}
            )

    def test_resource_candidate_reference_is_private_coordinate_only(self):
        reference = "ref_1234567890abcdefghijkl"
        self.assertEqual(
            ingest_submit_arguments(
                {
                    "source_type": "resource_candidates",
                    "positions": [1],
                    "target": "guangya",
                    "resource_candidates_ref": reference,
                }
            )["resource_candidates_ref"],
            reference,
        )
        self.assertEqual(
            ingest_inspect_arguments(
                {
                    "source_type": "resource_candidates",
                    "resource_candidates_ref": reference,
                }
            )["resource_candidates_ref"],
            reference,
        )
        with self.assertRaises(AgentToolError):
            ingest_submit_arguments(
                {
                    "source_type": "resource_candidates",
                    "positions": [1],
                    "target": "guangya",
                    "resource_candidates_ref": "latest",
                }
            )

    def test_direct_and_share_submit_accept_only_typed_ingest_snapshot_reference(self):
        reference = "ref_1234567890abcdefghijkl"
        direct = ingest_submit_arguments(
            {
                "source_type": "direct_url",
                "target": "qb",
                "ingest_snapshot_ref": reference,
            }
        )
        share = ingest_submit_arguments(
            {
                "source_type": "guangya_share",
                "ingest_snapshot_ref": reference,
            }
        )
        self.assertEqual(direct["ingest_snapshot_ref"], reference)
        self.assertEqual(share["ingest_snapshot_ref"], reference)
        with self.assertRaises(AgentToolError):
            ingest_submit_arguments(
                {
                    "source_type": "direct_url",
                    "target": "qb",
                    "ingest_snapshot_ref": "latest",
                }
            )
        with self.assertRaises(AgentToolError):
            ingest_submit_arguments(
                {
                    "source_type": "direct_url",
                    "target": "qb",
                    "resource_candidates_ref": reference,
                }
            )

    def test_share_defaults_to_guangya_and_all_items(self):
        self.assertEqual(
            ingest_submit_arguments({"source_type": "guangya_share"}),
            {"source_type": "guangya_share", "target": "guangya", "positions": []},
        )
        self.assertEqual(
            ingest_submit_arguments({"source_type": "guangya_share", "positions": []}),
            {"source_type": "guangya_share", "target": "guangya", "positions": []},
        )

    def test_resource_inspection_rejects_input(self):
        with self.assertRaises(AgentToolError):
            ingest_inspect_arguments(
                {"source_type": "resource_candidates", "input": "secret"}
            )

    def test_resource_inspection_without_positions_keeps_legacy_arguments(self):
        self.assertEqual(
            ingest_inspect_arguments({"source_type": "resource_candidates"}),
            {"source_type": "resource_candidates", "input": ""},
        )

    def test_resource_inspection_positions_follow_submit_ordered_deduplication(self):
        reference = "ref_1234567890abcdefghijkl"
        for positions in ([1], [12], [3, 1, 3, 2, 1], [12, 1] * 7):
            with self.subTest(positions=positions):
                arguments = {
                    "source_type": "resource_candidates",
                    "positions": positions,
                    "resource_candidates_ref": reference,
                }
                original_positions = list(positions)
                inspected = ingest_inspect_arguments(arguments)
                submitted = ingest_submit_arguments({**arguments, "target": "qb"})
                self.assertEqual(
                    inspected,
                    {
                        "source_type": "resource_candidates",
                        "input": "",
                        "positions": list(dict.fromkeys(positions)),
                        "resource_candidates_ref": reference,
                    },
                )
                self.assertEqual(inspected["positions"], submitted["positions"])
                self.assertEqual(arguments["positions"], original_positions)

    def test_resource_inspection_rejects_explicit_invalid_positions(self):
        invalid_positions = (
            [], None, True, False, 1, "1", {}, (1,),
            [True], [False], [1, True], [0], [-1], [51], [1, 51],
            [1.0], ["1"], [None],
        )
        for positions in invalid_positions:
            with self.subTest(positions=positions), self.assertRaises(AgentToolError):
                ingest_inspect_arguments(
                    {"source_type": "resource_candidates", "positions": positions}
                )

    def test_non_resource_inspection_rejects_positions_even_when_empty_or_null(self):
        sources = (
            {"source_type": "auto", "input": _MAGNET},
            {"source_type": "direct_url", "input": _MAGNET},
            {"source_type": "guangya_share", "input": _SHARE},
            {"input": _MAGNET},
        )
        for source in sources:
            for positions in ([1], [], None, True):
                with (
                    self.subTest(source=source, positions=positions),
                    self.assertRaises(AgentToolError),
                ):
                    ingest_inspect_arguments({**source, "positions": positions})

    def test_status_requires_public_positive_request_number(self):
        self.assertEqual(
            ingest_status_arguments({"request_number": 12}), {"request_number": 12}
        )
        with self.assertRaises(AgentToolError):
            ingest_status_arguments({"request_id": 12})


class AgentIngestActionTests(unittest.TestCase):
    def setUp(self):
        self.store = AgentIngestSessionStore()
        self.resources = RecentResourceCandidateStore()
        self.actions = IngestActions(
            store=self.store, recent_resource_store=self.resources
        )
        self.context = ToolContext(owner="owner-a")
        self.resource_result = _resource_result(
            "resource-first-001", "First",
            ("resource-second-01", "Second"),
            ("resource-third-001", "Third"),
        )

    def test_resource_inspection_without_positions_returns_entire_snapshot(self):
        search_id = self.resources.capture(owner="owner-a", result=self.resource_result)

        result = self.actions.inspect(
            ingest_inspect_arguments({"source_type": "resource_candidates"}),
            self.context,
        )

        self.assertTrue(result.ok)
        self.assertTrue(
            {"source_type", "search_id", "count", "items"} <= result.data.keys()
        )
        self.assertEqual(result.data["source_type"], "resource_candidates")
        self.assertEqual(result.data["search_id"], search_id)
        self.assertEqual(result.data["count"], 3)
        self.assertEqual(
            [(item["position"], item["title"]) for item in result.data["items"]],
            [(1, "First"), (2, "Second"), (3, "Third")],
        )

    def test_resource_inspection_returns_only_selected_original_positions_in_order(self):
        search_id = self.resources.capture(owner="owner-a", result=self.resource_result)
        titles = {1: "First", 2: "Second", 3: "Third"}
        for positions in ([3, 1, 3, 1], [2], [3, 2, 1]):
            with self.subTest(positions=positions):
                expected = list(dict.fromkeys(positions))
                result = self.actions.inspect(
                    ingest_inspect_arguments(
                        {"source_type": "resource_candidates", "positions": positions}
                    ),
                    self.context,
                )

                self.assertTrue(result.ok)
                self.assertEqual(result.data["search_id"], search_id)
                self.assertEqual(result.data["count"], len(expected))
                self.assertEqual(result.data["total"], 3)
                self.assertRegex(result.summary, rf"已检查\s*{len(expected)}\s*项")
                self.assertRegex(result.summary, r"快照共\s*3\s*项")
                self.assertEqual(
                    result.data["items"],
                    [
                        {
                            "position": position,
                            "title": titles[position],
                            "site_name": "Mikan",
                            "size_text": "1.2 GiB",
                            "download_state": "ready",
                        }
                        for position in expected
                    ],
                )
                rendered = repr(result.to_dict())
                for position, title in titles.items():
                    if position not in expected:
                        self.assertNotIn(title, rendered)
                for item in self.resource_result.data["items"]:
                    self.assertNotIn(item["result_id"], rendered)
                self.assertNotIn("result_id", rendered)

    def test_resource_inspection_accepts_twelfth_position_without_renumbering(self):
        resources = _resource_result(
            "resource-boundary-01", "Candidate 1",
            *[
                (f"resource-boundary-{position:02d}", f"Candidate {position}")
                for position in range(2, 13)
            ],
        )
        self.resources.capture(owner="owner-a", result=resources)
        result = self.actions.inspect(
            ingest_inspect_arguments(
                {"source_type": "resource_candidates", "positions": [12, 1, 12]}
            ),
            self.context,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.data["count"], 2)
        self.assertEqual(result.data["total"], 12)
        self.assertRegex(result.summary, r"已检查\s*2\s*项")
        self.assertRegex(result.summary, r"快照共\s*12\s*项")
        self.assertEqual(
            [(item["position"], item["title"]) for item in result.data["items"]],
            [(12, "Candidate 12"), (1, "Candidate 1")],
        )

    def test_resource_inspection_fails_whole_request_if_any_position_is_missing(self):
        self.resources.capture(owner="owner-a", result=self.resource_result)
        for positions in ([4], [2, 4], [4, 2]):
            with self.subTest(positions=positions):
                with self.assertRaises(AgentToolError) as raised:
                    self.actions.inspect(
                        ingest_inspect_arguments(
                            {"source_type": "resource_candidates", "positions": positions}
                        ),
                        self.context,
                    )
                self.assertEqual(raised.exception.code, "precondition_failed")

    def test_resource_inspection_preserves_owner_and_session_binding_boundaries(self):
        self.resources.capture(owner="owner-a", result=self.resource_result)
        contexts = (
            ToolContext(owner="owner-b"),
            ToolContext(owner="owner-a", session_id="session-a"),
            ToolContext(owner="owner-a", session_id="session-b"),
        )
        for context in contexts:
            for selection in ({}, {"positions": [1]}):
                with self.subTest(context=context, selection=selection):
                    with self.assertRaises(AgentToolError) as raised:
                        self.actions.inspect(
                            ingest_inspect_arguments(
                                {"source_type": "resource_candidates", **selection}
                            ),
                            context,
                        )
                    self.assertEqual(raised.exception.code, "precondition_failed")
                    if context.session_id:
                        self.assertIn("资源候选引用", raised.exception.safe_message)

    def test_resource_inspection_uses_bound_snapshot_not_newer_latest(self):
        search_id = new_resource_search_id()
        snapshot = safe_resource_snapshot(self.resource_result, search_id=search_id)
        latest_id = self.resources.capture(
            owner="owner-a", result=_resource_result("resource-newer-001", "Newer")
        )
        arguments = ingest_inspect_arguments(
            {"source_type": "resource_candidates", "positions": [3, 1]}
        )
        arguments["resource_candidates"] = snapshot

        result = self.actions.inspect(
            arguments, ToolContext(owner="owner-a", session_id="session-a")
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.data["search_id"], search_id)
        self.assertEqual(result.data["count"], 2)
        self.assertEqual(result.data["total"], 3)
        self.assertEqual(
            [(item["position"], item["title"]) for item in result.data["items"]],
            [(3, "Third"), (1, "First")],
        )
        self.assertNotIn("Newer", repr(result.to_dict()))
        self.assertEqual(self.resources.get(owner="owner-a")["search_id"], latest_id)

    def test_resource_inspection_does_not_fill_missing_bound_position_from_latest(self):
        snapshot = safe_resource_snapshot(
            _resource_result("resource-bound-001", "Bound"),
            search_id=new_resource_search_id(),
        )
        self.resources.capture(owner="owner-a", result=self.resource_result)
        for positions in ([1, 3], [3, 1]):
            with self.subTest(positions=positions):
                arguments = ingest_inspect_arguments(
                    {"source_type": "resource_candidates", "positions": positions}
                )
                arguments["resource_candidates"] = snapshot
                with self.assertRaises(AgentToolError) as raised:
                    self.actions.inspect(
                        arguments, ToolContext(owner="owner-a", session_id="session-a")
                    )
                self.assertEqual(raised.exception.code, "precondition_failed")

    def test_resource_inspection_is_read_only_and_selection_does_not_persist(self):
        self.resources.capture(owner="owner-a", result=self.resource_result)
        before = self.resources.get(owner="owner-a")
        mutations = (
            "app.agent.domains.resources.candidate_actions.prepare_submit_resource",
            "app.agent.domains.resources.candidate_actions.prepare_submit_resource_batch",
            "app.agent.domains.resources.candidate_actions.submit_resource_confirmed",
            "app.agent.domains.resources.candidate_actions.submit_resource_batch_confirmed",
            "app.agent.domains.resources.ingest_actions.create_request",
            "app.agent.domains.resources.ingest_actions.dispatch_request",
            "app.agent.domains.resources.ingest_actions.create_share_request",
        )
        with ExitStack() as stack:
            mocked_mutations = [stack.enter_context(patch(name)) for name in mutations]
            capture = stack.enter_context(patch.object(self.store, "capture"))
            for selection, count in (({"positions": [3]}, 1), ({}, 3)):
                with self.subTest(selection=selection):
                    result = self.actions.inspect(
                        ingest_inspect_arguments(
                            {"source_type": "resource_candidates", **selection}
                        ),
                        self.context,
                    )
                    self.assertTrue(result.ok)
                    self.assertEqual(result.status, "success")
                    self.assertEqual(result.data["count"], count)
                    self.assertEqual(result.references, [])
                    self.assertEqual(result.effect_metadata, {})
                    self.assertNotIn("plan_ref", result.data)
                    self.assertNotIn("request_number", result.data)
            for mocked_mutation in mocked_mutations:
                mocked_mutation.assert_not_called()
            capture.assert_not_called()
        self.assertEqual(self.resources.get(owner="owner-a"), before)
        self.assertIsNone(self.store.get(owner="owner-a"))

    def test_direct_link_snapshot_never_returns_link_or_hash(self):
        result = self.actions.inspect(
            {"source_type": "direct_url", "input": _MAGNET}, self.context
        )
        self.assertTrue(result.ok)
        rendered = repr(result.to_dict())
        self.assertNotIn("magnet:?", rendered)
        self.assertNotIn("a" * 40, rendered)
        self.assertEqual(result.data["source_type"], "direct_url")
        self.assertIsNone(self.store.get(owner="owner-b"))

    def test_ingest_memory_snapshots_are_isolated_by_conversation_session(self):
        first_context = ToolContext(owner="owner-a", session_id="session-a")
        second_context = ToolContext(owner="owner-a", session_id="session-b")
        self.actions.inspect(
            {"source_type": "direct_url", "input": _MAGNET}, first_context
        )
        self.actions.inspect(
            {"source_type": "direct_url", "input": _MAGNET}, second_context
        )

        first = self.store.get(
            owner="owner-a",
            source_type="direct_url",
            conversation_session_id="session-a",
        )
        second = self.store.get(
            owner="owner-a",
            source_type="direct_url",
            conversation_session_id="session-b",
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first.session_id, second.session_id)

    @patch(
        "app.agent.domains.resources.ingest_actions.download_target_readiness",
        return_value={"qb": True},
    )
    def test_direct_reference_rebuilds_preview_after_memory_store_restart(
        self, _readiness
    ):
        context = ToolContext(owner="owner-a", session_id="session-a")
        inspected = self.actions.inspect(
            {"source_type": "direct_url", "input": _MAGNET}, context
        )
        reference_value = inspected.references[0].value
        restarted = IngestActions(
            store=AgentIngestSessionStore(),
            recent_resource_store=RecentResourceCandidateStore(),
        )
        resolved_arguments = {
            "source_type": "direct_url",
            "target": "qb",
            "positions": [],
            "ingest_snapshot": reference_value,
        }

        preview, fingerprint = restarted.prepare_submit(resolved_arguments, context)

        self.assertTrue(preview.ok)
        self.assertTrue(fingerprint)
        self.assertEqual(preview.data["source_type"], "direct_url")
        self.assertIsNone(
            restarted.store.get(
                owner="owner-a",
                source_type="direct_url",
                conversation_session_id="session-a",
            )
        )

    def test_ordinary_web_page_is_not_ingested(self):
        with self.assertRaises(AgentToolError) as raised:
            self.actions.inspect(
                {
                    "source_type": "auto",
                    "input": "http://192.168.0.195:1258/guangya/offline",
                },
                self.context,
            )
        self.assertEqual(raised.exception.code, "precondition_failed")
        self.assertIsNone(self.store.get(owner="owner-a"))

    def test_malformed_download_link_returns_agent_input_error(self):
        with self.assertRaises(AgentToolError) as raised:
            self.actions.inspect(
                {"source_type": "direct_url", "input": "magnet:?dn=missing-hash"},
                self.context,
            )
        self.assertEqual(raised.exception.code, "precondition_failed")
        self.assertIsNone(self.store.get(owner="owner-a"))

    @patch("app.agent.domains.resources.ingest_actions.db.get_download_request", return_value=None)
    @patch(
        "app.agent.domains.resources.ingest_actions.dispatch_request",
        return_value={
            "ok": True,
            "status": "submitted",
            "succeeded": ["qb"],
            "failed": [],
        },
    )
    @patch(
        "app.agent.domains.resources.ingest_actions.create_request",
        return_value={"id": 42, "created": True, "status": "pending"},
    )
    @patch(
        "app.agent.domains.resources.ingest_actions.download_target_readiness", return_value={"qb": True}
    )
    def test_direct_link_requires_confirmation_and_executes_via_dispatcher(
        self, _readiness, create_request_mock, dispatch_mock, _get_request
    ):
        self.actions.inspect(
            {"source_type": "direct_url", "input": _MAGNET}, self.context
        )
        arguments = {"source_type": "direct_url", "target": "qb", "positions": []}
        preview, fingerprint = self.actions.prepare_submit(arguments, self.context)
        self.assertTrue(preview.ok)
        self.assertNotIn("magnet:?", repr(preview.to_dict()))
        result = self.actions.execute_submit(arguments, fingerprint, self.context)
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.data["request_number"], 42)
        create_request_mock.assert_called_once()
        dispatch_mock.assert_called_once_with(42, "qb")

    @patch(
        "app.agent.domains.resources.ingest_actions.inspect_share_for_transfer",
        return_value={
            "preview_id": "preview-secret",
            "share_id": "share-secret",
            "files": [
                {"id": "file-secret-1", "name": "Episode 1.mkv", "size": 10},
                {"id": "file-secret-2", "name": "Episode 2.mkv", "size": 20},
            ],
            "selected_ids": ["file-secret-1", "file-secret-2"],
            "target_id": "target-secret",
            "target_name": "默认目录",
            "expires_in": 900,
        },
    )
    def test_share_inspection_projects_positions_only(self, _inspect):
        result = self.actions.inspect(
            {"source_type": "guangya_share", "input": _SHARE}, self.context
        )
        rendered = repr(result.to_dict())
        self.assertTrue(result.ok)
        self.assertNotIn(_SHARE, rendered)
        self.assertNotIn("preview-secret", rendered)
        self.assertNotIn("file-secret", rendered)
        self.assertEqual([item["position"] for item in result.data["items"]], [1, 2])

    @patch(
        "app.agent.domains.resources.ingest_actions.download_target_readiness",
        return_value={"guangya": True},
    )
    @patch(
        "app.agent.domains.resources.ingest_actions.inspect_share_for_transfer",
        return_value={
            "preview_id": "preview-secret",
            "share_id": "share-secret",
            "files": [
                {
                    "id": "file-secret-1",
                    "name": "Episode 1.mkv",
                    "is_dir": False,
                    "size": 10,
                },
                {
                    "id": "file-secret-2",
                    "name": "Episode 2.mkv",
                    "is_dir": False,
                    "size": 20,
                },
            ],
            "selected_ids": ["file-secret-1", "file-secret-2"],
            "target_id": "target-secret",
            "target_name": "默认目录",
            "expires_in": 900,
        },
    )
    def test_share_reference_revalidates_after_memory_store_restart(
        self, inspect_share, _readiness
    ):
        context = ToolContext(owner="owner-a", session_id="session-a")
        inspected = self.actions.inspect(
            {"source_type": "guangya_share", "input": _SHARE}, context
        )
        reference_value = inspected.references[0].value
        restarted = IngestActions(
            store=AgentIngestSessionStore(),
            recent_resource_store=RecentResourceCandidateStore(),
        )

        preview, fingerprint = restarted.prepare_submit(
            {
                "source_type": "guangya_share",
                "target": "guangya",
                "positions": [1, 2],
                "ingest_snapshot": reference_value,
            },
            context,
        )

        self.assertTrue(preview.ok)
        self.assertTrue(fingerprint)
        self.assertEqual(preview.data["count"], 2)
        self.assertEqual(inspect_share.call_count, 2)

    @patch(
        "app.agent.domains.resources.ingest_actions.inspect_share_for_transfer",
        return_value={
            "preview_id": "preview-secret",
            "share_id": "share-secret",
            "files": [
                {"name": "invalid", "size": "not-a-number"},
                {"id": "file-secret-2", "name": "Episode 2.mkv", "size": "bad"},
            ],
            "target_id": "0",
            "target_name": "根目录",
            "expires_in": "invalid",
        },
    )
    def test_share_projection_keeps_positions_aligned_with_private_ids(self, _inspect):
        result = self.actions.inspect(
            {"source_type": "guangya_share", "input": _SHARE}, self.context
        )
        self.assertEqual(result.data["count"], 1)
        self.assertEqual(result.data["items"][0]["position"], 1)
        self.assertEqual(result.data["items"][0]["size_bytes"], 0)
        snapshot = self.store.get(owner="owner-a", source_type="guangya_share")
        self.assertEqual(snapshot.private["file_ids"], ["file-secret-2"])

    @patch("app.agent.domains.resources.candidate_actions.prepare_submit_resource")
    def test_staged_candidates_override_older_persisted_latest(self, prepare_resource):
        prepare_resource.side_effect = lambda arguments: (
            ToolResult(True, "confirmation_required", "preview", data={"resource": {}}),
            f"{arguments['result_id']}:{arguments['target']}",
        )
        self.resources.capture(
            owner="owner-a", result=_resource_result("resource-old-0001", "Old")
        )
        staged_search_id = new_resource_search_id()
        staged_snapshot = safe_resource_snapshot(
            _resource_result("resource-new-0001", "New"), search_id=staged_search_id
        )
        buffer = AgentStateCommitBuffer(owner="owner-a")
        arguments = ingest_submit_arguments(
            {"source_type": "resource_candidates", "positions": [1], "target": "qb"}
        )
        with defer_agent_state_commits(buffer):
            self.assertTrue(
                stage_agent_resource_candidates(
                    owner="owner-a", snapshot=staged_snapshot
                )
            )
            self.actions.prepare_submit(arguments, self.context)
        self.assertEqual(arguments["search_id"], staged_search_id)
        prepare_resource.assert_called_once_with(
            {"result_id": "resource-new-0001", "target": "qb"}
        )
        buffer.discard()

    def test_kernel_session_does_not_fall_back_to_another_sessions_latest_candidates(
        self,
    ):
        self.resources.capture(
            owner="owner-a", result=_resource_result("resource-other-001", "Other")
        )
        with self.assertRaises(AgentToolError) as raised:
            self.actions.prepare_submit(
                ingest_submit_arguments(
                    {
                        "source_type": "resource_candidates",
                        "positions": [1],
                        "target": "qb",
                    }
                ),
                ToolContext(owner="owner-a", session_id="session-a"),
            )
        self.assertEqual(raised.exception.code, "precondition_failed")
        self.assertIn("资源候选引用", raised.exception.safe_message)

    @patch("app.agent.domains.resources.candidate_actions.submit_resource_confirmed")
    @patch("app.agent.domains.resources.candidate_actions.prepare_submit_resource")
    def test_confirm_uses_frozen_snapshot_after_a_newer_search(
        self, prepare_resource, submit_resource
    ):
        prepare_resource.side_effect = lambda arguments: (
            ToolResult(True, "confirmation_required", "preview", data={"resource": {}}),
            f"{arguments['result_id']}:{arguments['target']}",
        )
        submit_resource.side_effect = lambda arguments, _expected: ToolResult(
            True, "accepted", "submitted", data={"result_id": arguments["result_id"]}
        )
        first_search_id = self.resources.capture(
            owner="owner-a", result=_resource_result("resource-first-001", "First")
        )
        arguments = ingest_submit_arguments(
            {"source_type": "resource_candidates", "positions": [1], "target": "qb"}
        )
        _preview, expected_context = self.actions.prepare_submit(
            arguments, self.context
        )
        self.assertEqual(arguments["search_id"], first_search_id)
        self.resources.capture(
            owner="owner-a", result=_resource_result("resource-second-01", "Second")
        )
        result = self.actions.execute_submit(arguments, expected_context, self.context)
        self.assertTrue(result.ok)
        self.assertEqual(result.data["result_id"], "resource-first-001")
        self.assertEqual(
            [call.args[0]["result_id"] for call in prepare_resource.call_args_list],
            ["resource-first-001", "resource-first-001"],
        )
        submit_resource.assert_called_once_with(
            {"result_id": "resource-first-001", "target": "qb"}, "resource-first-001:qb"
        )

    @patch("app.agent.domains.resources.candidate_actions.submit_resource_confirmed")
    @patch("app.agent.domains.resources.candidate_actions.prepare_submit_resource")
    def test_kernel_reference_snapshot_survives_preview_and_confirmation(
        self, prepare_resource, submit_resource
    ):
        prepare_resource.side_effect = lambda arguments: (
            ToolResult(True, "confirmation_required", "preview", data={"resource": {}}),
            f"{arguments['result_id']}:{arguments['target']}",
        )
        submit_resource.side_effect = lambda arguments, _expected: ToolResult(
            True, "accepted", "submitted", data={"result_id": arguments["result_id"]}
        )
        search_id = new_resource_search_id()
        snapshot = safe_resource_snapshot(
            _resource_result("resource-kernel-0001", "Kernel"),
            search_id=search_id,
        )
        arguments = {
            "source_type": "resource_candidates",
            "positions": [1],
            "target": "guangya",
            "resource_candidates": snapshot,
        }

        preview, expected_context = self.actions.prepare_submit(arguments, self.context)
        self.assertTrue(preview.ok)
        self.assertEqual(arguments["search_id"], search_id)
        self.assertIsNone(self.resources.get(owner="owner-a"))

        result = self.actions.execute_submit(arguments, expected_context, self.context)
        self.assertTrue(result.ok)
        submit_resource.assert_called_once_with(
            {"result_id": "resource-kernel-0001", "target": "guangya"},
            "resource-kernel-0001:guangya",
        )

    @patch("app.agent.domains.resources.candidate_actions.submit_resource_confirmed")
    @patch("app.agent.domains.resources.candidate_actions.prepare_submit_resource")
    def test_confirm_fails_closed_when_frozen_snapshot_was_evicted(
        self, prepare_resource, submit_resource
    ):
        resources = RecentResourceCandidateStore(max_snapshots_per_owner=1)
        actions = IngestActions(store=self.store, recent_resource_store=resources)
        prepare_resource.side_effect = lambda arguments: (
            ToolResult(True, "confirmation_required", "preview", data={"resource": {}}),
            f"{arguments['result_id']}:{arguments['target']}",
        )
        resources.capture(
            owner="owner-a", result=_resource_result("resource-first-001", "First")
        )
        arguments = ingest_submit_arguments(
            {"source_type": "resource_candidates", "positions": [1], "target": "qb"}
        )
        _preview, expected_context = actions.prepare_submit(arguments, self.context)
        resources.capture(
            owner="owner-a", result=_resource_result("resource-second-01", "Second")
        )
        with self.assertRaises(AgentToolError) as raised:
            actions.execute_submit(arguments, expected_context, self.context)
        self.assertEqual(raised.exception.code, "confirmation_stale")
        submit_resource.assert_not_called()

    def _status_row(self, **overrides):
        row = {
            "id": 7,
            "title": "Safe title",
            "kind": "magnet",
            "targets": "both",
            "status": "submitted",
            "qb_status": "submitted",
            "gy_status": "failed",
            "organize_status": "pending",
            "strm_status": "pending",
            "updated_at": "2026-09-01 12:00:00",
            "error": "guangya: 文件违规; url=https://private.example/?token=secret-token",
            "source_value": _MAGNET,
        }
        row.update(overrides)
        return row

    @patch("app.agent.domains.resources.ingest_actions.db.get_download_request")
    def test_status_exposes_safe_partial_error_without_overwriting_stage_status(
        self, get_request
    ):
        get_request.return_value = self._status_row()

        result = self.actions.status({"request_number": 7}, self.context)

        request = result.data["request"]
        self.assertEqual(request["status"], "submitted")
        self.assertEqual(request["qb_status"], "submitted")
        self.assertEqual(request["guangya_status"], "failed")
        self.assertEqual(request["error"], "光鸭返回：文件违规")
        self.assertNotIn("private.example", repr(result.to_dict()))
        self.assertNotIn("secret-token", repr(result.to_dict()))

    @patch("app.agent.domains.resources.ingest_actions.db.get_download_request")
    def test_status_exposes_safe_manual_review_reason(self, get_request):
        get_request.return_value = self._status_row(
            status="manual_review",
            qb_status="outcome_unknown",
            gy_status="failed",
            error="qb: task_id=private-qb-task; guangya: 文件违规; url=https://private.example/x",
        )

        result = self.actions.status({"request_number": 7}, self.context)

        request = result.data["request"]
        self.assertEqual(request["status"], "manual_review")
        self.assertEqual(request["qb_status"], "outcome_unknown")
        self.assertEqual(request["guangya_status"], "failed")
        self.assertEqual(
            request["error"],
            "下载后端提交结果未知，请先核对下载器，勿直接重复提交",
        )
        self.assertNotIn("private-qb-task", repr(result.to_dict()))
        self.assertNotIn("private.example", repr(result.to_dict()))

    @patch("app.agent.domains.resources.ingest_actions.db.get_download_request")
    def test_status_exposes_safe_outcome_unknown_reason(self, get_request):
        get_request.return_value = self._status_row(
            status="submitted",
            qb_status="outcome_unknown",
            gy_status="not_started",
            error="qb: provider accepted but task_id=private-qb-task is unknown",
        )

        result = self.actions.status({"request_number": 7}, self.context)

        request = result.data["request"]
        self.assertEqual(request["status"], "submitted")
        self.assertEqual(request["qb_status"], "outcome_unknown")
        self.assertEqual(request["guangya_status"], "not_started")
        self.assertEqual(
            request["error"],
            "下载后端提交结果未知，请先核对下载器，勿直接重复提交",
        )
        self.assertNotIn("private-qb-task", repr(result.to_dict()))

    @patch("app.agent.domains.resources.ingest_actions.db.get_download_request")
    def test_status_does_not_invent_failure_for_waiting_or_skipped_rows(self, get_request):
        cases = (
            {"status": "pending", "qb_status": "pending", "gy_status": "pending"},
            {"status": "pending", "qb_status": "not_started", "gy_status": "not_started"},
            {"status": "cancelled", "qb_status": "cancelled", "gy_status": "cancelled"},
            {"status": "skipped", "qb_status": "skipped", "gy_status": "skipped"},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                get_request.return_value = self._status_row(error="", **overrides)
                result = self.actions.status({"request_number": 7}, self.context)
                self.assertEqual(result.data["request"]["error"], "")

    @patch("app.agent.domains.resources.ingest_actions.db.get_download_request")
    def test_status_returns_only_public_request_projection(self, get_request):
        get_request.return_value = {
            "id": 7,
            "title": "Safe title",
            "kind": "magnet",
            "targets": "qb",
            "status": "downloading",
            "qb_status": "downloading",
            "gy_status": "not_started",
            "organize_status": "pending",
            "strm_status": "pending",
            "updated_at": "2026-09-01 12:00:00",
            "source_value": _MAGNET,
        }
        result = self.actions.status({"request_number": 7}, self.context)
        self.assertTrue(result.ok)
        self.assertEqual(result.data["request"]["request_number"], 7)
        self.assertNotIn("source_value", result.data["request"])
        self.assertNotIn("magnet:?", repr(result.to_dict()))


@pytest.mark.parametrize(
    "qb_ok,gy_ok,repeat,tool_status,dispatch_status,public_state,summary",
    [
        pytest.param(True, False, False, "partial", "partial", "warning", "仅部分目标已提交", id="qb-only"),
        pytest.param(False, True, False, "partial", "partial", "warning", "仅部分目标已提交", id="guangya-only"),
        pytest.param(True, True, False, "accepted", "submitted", "submitted", "已提交", id="both-success"),
        pytest.param(False, False, False, "unavailable", "failed", "failed", "未完成提交", id="both-failed"),
        pytest.param(True, True, True, "conflict", "duplicate", "failed", "未完成提交", id="duplicate-both"),
        pytest.param(True, False, True, "conflict", "duplicate", "failed", "未完成提交", id="duplicate-qb-only"),
        pytest.param(False, True, True, "conflict", "duplicate", "failed", "未完成提交", id="duplicate-guangya-only"),
    ],
)
def test_direct_submission_receipt_matches_real_dispatch(
    qb_ok, gy_ok, repeat, tool_status, dispatch_status, public_state, summary,
):
    """真实落盘与分发；只替换后端边界，确认结果由 Web/TG 共用生产投影。"""
    with (
        isolated_test_database(),
        patch.dict(os.environ, {"QB_URL": "http://qb.invalid"}),
        patch("socket.socket.connect", side_effect=AssertionError("禁止外联")),
        patch("app.agent.domains.resources.indexer_actions.GuangYaClient") as cloud_client,
        patch.object(dispatcher, "_submit_qb", return_value={"ok": qb_ok}) as qb,
        patch.object(dispatcher, "_submit_guangya", return_value={"ok": gy_ok}) as gy,
    ):
        cloud_client.return_value.logged_in = True
        actions = IngestActions(
            store=AgentIngestSessionStore(),
            recent_resource_store=RecentResourceCandidateStore(),
        )
        context = ToolContext(owner="owner-a", session_id="direct-receipt")
        inspected = actions.inspect(
            ingest_inspect_arguments({"source_type": "direct_url", "input": _MAGNET}),
            context,
        )
        arguments = ingest_submit_arguments({"source_type": "direct_url", "target": "both"})
        arguments["ingest_snapshot"] = inspected.references[0].value
        preview, fingerprint = actions.prepare_submit(arguments, context)
        assert preview.ok and preview.data["backends"] == {"qb": True, "guangya": True}
        with db.get_conn() as conn:
            assert conn.execute("SELECT COUNT(*) FROM download_requests").fetchone()[0] == 0
        qb.assert_not_called()
        gy.assert_not_called()

        first = actions.execute_submit(arguments, fingerprint, context)
        result = actions.execute_submit(arguments, fingerprint, context) if repeat else first
        request_number = result.data["request_number"]
        assert request_number == first.data["request_number"]
        assert result.data["created"] is (not repeat)
        assert result.data["duplicate"] is repeat
        qb.assert_called_once()
        gy.assert_called_once()
        row = db.get_download_request(request_number)
        assert row["qb_status"] == ("submitted" if qb_ok else "failed")
        assert row["gy_status"] == ("submitted" if gy_ok else "failed")
        with db.get_conn() as conn:
            assert conn.execute("SELECT COUNT(*) FROM download_requests").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM download_log").fetchone()[0] == 2
        assert result.data["status"] == dispatch_status
        assert result.ok is (not repeat and (qb_ok or gy_ok))
        assert result.data["succeeded"] == ([] if repeat else [
            name for name, ok in (("qb", qb_ok), ("guangya", gy_ok)) if ok
        ])
        assert result.data["failed"] == ([] if repeat else [
            name for name, ok in (("qb", qb_ok), ("guangya", gy_ok)) if not ok
        ])
        assert (result.status, result.summary) == (tool_status, f"下载请求 #{request_number} {summary}")

        public = dict(DefaultProjector().project(result).public_content)
        receipt = format_public_result(public)
        assert public_result_state(public) == public_state
        assert result.summary in receipt
        view = TurnView(
            session_id=context.session_id, turn_id="direct-receipt", request_id="direct-receipt",
            status="effect_completed" if result.ok else "failed", effect_result=public,
        )
        web = view.to_dict()["effect_result"]
        assert web == public
        assert public_result_state(web) == public_state
        assert _render_turn(view) == render_telegram_markdown(receipt)
        assert ("⚠️" in receipt) is (public_state == "warning")
