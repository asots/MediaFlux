"""只冻结 share 私有来源；不运行 refresh/新旧对照，不延长旧引用 TTL。"""
from __future__ import annotations
from typing import Any

import json
import tempfile
import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

from app.agent.confirmation import SQLiteConfirmationStore
from app.agent.domain_catalog.catalog import DomainActions, ToolSpecCollector
from app.agent.domains.guangya import share_actions as actions
from app.agent.domains.guangya.sdk_tools import register_specs
from app.agent.domains.guangya.workspace_actions import guangya_fs_query_arguments
from app.agent.errors import AgentToolError
from app.agent.kernel.capabilities import ToolCatalog
from app.agent.kernel.effects import ConfirmationEffectPlanStore
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.pipeline import ToolPipeline
from app.agent.kernel.ports.existing_actions import adapt_tool_spec
from app.agent.kernel.projection import DefaultProjector
from app.agent.kernel.state_types import CancellationToken
from app.agent.kernel.tool_calls import ToolCallContext
from app.agent.models import ToolContext
from app.clients.guangya import GuangYaFile
from app.modules import guangya_workspace as workspace
from tests.support import isolated_test_database


class ShareReobservationSourceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(isolated_test_database())
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(workspace, "_directory", return_value=directory))
        self.enterContext(patch.object(workspace, "get_web_secret", return_value="source-test-secret"))
        self.enterContext(patch("app.repositories.agent_jobs.get_web_secret", return_value="source-test-secret"))
        self.enterContext(patch("app.modules.web_secret.get_web_secret", return_value="source-test-secret"))
        self.enterContext(patch.object(actions, "_now", return_value="2026-10-12T00:00:00+08:00"))
        self._domain_actions = DomainActions()
        self.context = ToolContext(owner="source-owner", session_id="source-session")
        self.ref = "OBS" + "A" * 32
        self.handles = ["OBJ" + "B" * 24, "OBJ" + "C" * 24]
        self.roots = [
            {"file_id": "root-one", "path": "/One", "is_dir": True, "parent_id": "0"},
            {"file_id": "empty-root", "path": "/Empty", "is_dir": True, "parent_id": "0"},
        ]
        self.entries: list[dict[str, Any]] = [{
            "handle": ref, "file_id": fid, "parent_id": "root-one", "name": name,
            "is_dir": False, "size": 8, "etag": "etag-" + fid, "updated_at": 17,
            "parent_path": "/One", "relative_parent": "One",
            "source_root": {**self.roots[0], "token": "ROOT-TOKEN-SECRET"},
            "access_code": "ENTRY-CODE-SECRET", "share_url": "https://private.invalid/entry",
        } for ref, fid, name in zip(self.handles, ("file-a", "file-b"), ("a.txt", "b.txt"))]
        now = time.time()
        self.observation = {
            "version": 1, "plan_id": "a" * 32,
            "owner_digest": workspace.organize_operation_owner_digest(self.context.owner),
            "session_scope": workspace.workspace_session_scope(self.context.owner, self.context.session_id),
            "credential_generation": 7, "created_at": "2026-10-12T00:00:00+08:00",
            "created_at_epoch": now, "expires_at_epoch": now + 600,
            "scope_path": "/", "scope_name": "2 个目录", "scope_names": ["One", "Empty"],
            "roots": deepcopy(self.roots), "operation": "list", "query": "", "kinds": [],
            "max_depth": 0, "recursive": False, "max_items": 25, "truncated": False,
            "scanned_dirs": 2, "scanned_items": 2, "entries": deepcopy(self.entries),
            "token": "OBS-TOKEN-SECRET", "code": "OBS-CODE-SECRET",
            "share_url": "https://private.invalid/observation",
        }
        workspace._atomic_write(self.observation)
        self.raw_shares = [{
            "id": f"share-{i}", "title": f"title-{i}", "status": "active", "updateTime": "t1",
            "fileCount": 1, "code": "SHARE-CODE-SECRET", "shareUrl": "https://private.invalid/share",
        } for i in range(1, 5)]
        self.collection = {
            "credential_generation": 7, "page": 2, "page_size": 2,
            "items": [actions._share_snapshot(v) for v in self.raw_shares[2:]],
            "access_code": "COLLECTION-CODE-SECRET", "token": "COLLECTION-TOKEN-SECRET",
        }
        self.collection["items"][1].update(code="ITEM-CODE-SECRET", share_url="https://private.invalid/item")
        self.client = Mock(logged_in=True, credential_generation=7)
        self.client.file_info.side_effect = lambda fid: GuangYaFile(**{
            k: next(e for e in self.entries if e["file_id"] == fid)[k]
            for k in ("file_id", "parent_id", "name", "is_dir", "size", "etag", "updated_at")
        })
        self.client.list_user_shares.return_value = deepcopy(self.raw_shares)
        self.factory = self.enterContext(patch.object(actions, "GuangYaClient", return_value=self.client))

    def create_arguments(self):
        return actions.guangya_share_create_arguments({
            "observation_ref": self.ref, "object_refs": [self.handles[1]],
            "expires_days": 1, "access_code": "Ab12Cd34",
        })

    def revoke_arguments(self, *, with_ref=True):
        args: dict[str, Any] = {"indices": [4], "guangya_shares": self.collection}
        if with_ref:
            args["guangya_shares_ref"] = "ref_original_inventory"
        return args

    def assert_private(self, preview, source):
        serialized = json.dumps(source, ensure_ascii=False)
        for secret in ("Ab12Cd34", "TOKEN-SECRET", "CODE-SECRET", "private.invalid"):
            self.assertNotIn(secret, serialized)
        for public in (preview.to_dict(), preview.to_model_dict(), DefaultProjector().project(preview).public_content):
            text = json.dumps(public, ensure_ascii=False)
            for private in ("reobservation_source", "credential_generation", "source-owner", "file-b", "share-4", "Ab12Cd34", "ref_original_inventory"):
                self.assertNotIn(private, text)
        self.assertNotIn("reobservation_source", DefaultProjector().project(preview).model_content)

    def test_create_preserves_all_roots_but_only_selected_identity_without_extra_reads(self):
        with patch.object(actions, "load_directory_observation", wraps=actions.load_directory_observation) as load:
            preview, _ = actions.prepare_create_guangya_share(self.create_arguments(), self.context, workspace_actions=self._domain_actions.workspace)
        source = preview.effect_metadata["reobservation_source"]
        self.assertEqual((source["version"], source["kind"], source["owner"], source["session_id"], source["credential_generation"]),
                         (1, "guangya_files", self.context.owner, self.context.session_id, 7))
        self.assertTrue(source["available"])
        self.assertEqual(source["read"], {"tool": "guangya.fs.query", "arguments": {
            "operation": "list", "paths": ["/One", "/Empty"], "kinds": [], "max_depth": 0, "max_items": 25,
        }})
        guangya_fs_query_arguments(source["read"]["arguments"])
        baseline = source["baseline"]
        self.assertEqual(baseline["observation_ref"], self.ref)
        self.assertEqual(baseline["roots"], self.roots)
        self.assertEqual(baseline["scope_path"], "/")
        self.assertEqual(baseline["expires_at_epoch"], self.observation["expires_at_epoch"])
        self.assertEqual(len(baseline["selected"]), 1)
        selected = baseline["selected"][0]
        self.assertEqual((selected["file_id"], selected["old_object_ref"]), ("file-b", self.handles[1]))
        self.assertEqual(selected["source_root"], self.roots[0])
        self.assertNotIn("entries", baseline)
        self.assertEqual(load.call_count, 1)
        self.client.file_info.assert_called_once_with("file-b")
        self.assertEqual(self.factory.call_count, 1)
        self.assert_private(preview, source)

    def test_old_missing_roots_keeps_preview_but_does_not_guess_slash_or_expiry(self):
        args = self.create_arguments()
        expected, expected_fingerprint = actions.prepare_create_guangya_share(args, self.context, workspace_actions=self._domain_actions.workspace)
        for key in ("roots", "max_items", "created_at", "created_at_epoch"):
            self.observation.pop(key)
        workspace._atomic_write(self.observation)
        actual, fingerprint = actions.prepare_create_guangya_share(args, self.context, workspace_actions=self._domain_actions.workspace)
        source = actual.effect_metadata["reobservation_source"]
        self.assertFalse(source["available"])
        self.assertIsNone(source["read"])
        self.assertEqual(source["baseline"]["roots"], [])
        self.assertNotIn("created_at", source["baseline"])
        self.assertNotIn("observed_at", source["baseline"])
        self.assertEqual(actual.to_dict(), expected.to_dict())
        self.assertEqual(actual.to_model_dict(), expected.to_model_dict())
        self.assertEqual(fingerprint, expected_fingerprint)

    def test_query_bounds_and_stat_locator_are_frozen_from_private_observation(self):
        for operation in ("tree", "search", "stat"):
            with self.subTest(operation=operation):
                observation = deepcopy(self.observation)
                observation.update(operation=operation, query="needle" if operation == "search" else "", max_depth=3, recursive=operation != "stat", kinds=["other"], max_items=12, truncated=True)
                if operation == "stat":
                    observation["scope_path"] = "/One/b.txt"
                    observation["roots"] = [{"file_id": "file-b", "path": "/One/b.txt", "is_dir": False, "parent_id": "root-one"}]
                workspace._atomic_write(observation)
                preview, _ = actions.prepare_create_guangya_share(self.create_arguments(), self.context, workspace_actions=self._domain_actions.workspace)
                source = preview.effect_metadata["reobservation_source"]
                self.assertTrue(source["available"])
                self.assertEqual(source["baseline"]["max_items"], 12)
                self.assertTrue(source["baseline"]["truncated"])
                read = source["read"]["arguments"]
                self.assertEqual(read["operation"], operation)
                if operation == "stat":
                    self.assertEqual(read, {"operation": "stat", "path": "/One/b.txt"})
                else:
                    self.assertEqual(read["paths"], ["/One", "/Empty"])
                    self.assertEqual(read["max_items"], 12)
                    if operation == "search":
                        self.assertEqual(read["query"], "needle")
                guangya_fs_query_arguments(read)

    def test_resolved_observation_ref_is_recorded_when_request_uses_latest(self):
        args = self.create_arguments()
        args["observation_ref"] = ""
        with patch.object(self._domain_actions.workspace, "latest_guangya_observation_ref", return_value=self.ref) as latest:
            preview, _ = actions.prepare_create_guangya_share(args, self.context, workspace_actions=self._domain_actions.workspace)
        latest.assert_called_once_with(self.context.owner, self.context.session_id)
        self.assertEqual(preview.effect_metadata["reobservation_source"]["baseline"]["observation_ref"], self.ref)

    def test_revoke_records_stable_share_id_and_original_page_index_not_whole_list(self):
        self.collection.update(observed_at="original-observation-time", expires_at=9876)
        preview, _ = actions.prepare_revoke_guangya_shares(self.revoke_arguments(), self.context)
        source = preview.effect_metadata["reobservation_source"]
        self.assertEqual(source["kind"], "guangya_shares")
        self.assertEqual((source["owner"], source["session_id"], source["credential_generation"]), (self.context.owner, self.context.session_id, 7))
        self.assertEqual(source["read"], {"tool": "guangya.share.list", "arguments": {"page": 2, "page_size": 2}})
        baseline = source["baseline"]
        self.assertEqual(baseline["ref"], "ref_original_inventory")
        self.assertEqual((baseline["page"], baseline["page_size"]), (2, 2))
        self.assertEqual((baseline["observed_at"], baseline["expires_at"]), ("original-observation-time", 9876))
        self.assertEqual([(x["share_id"], x["old_index"]) for x in baseline["selected"]], [("share-4", 4)])
        self.assertNotIn("items", baseline)
        self.assertEqual(self.factory.call_count, 1)
        self.client.list_user_shares.assert_called_once_with(max_items=actions._MAX_SHARES)
        self.assert_private(preview, source)
        self.collection["items"][1]["title"] = "changed-after-prepare"
        self.assertEqual(baseline["selected"][0]["title"], "title-4")

    def test_reordered_collection_does_not_erase_selected_share_identity(self):
        self.collection["items"].reverse()
        args = self.revoke_arguments()
        args["indices"] = [3]
        preview, _ = actions.prepare_revoke_guangya_shares(args, self.context)
        baseline = preview.effect_metadata["reobservation_source"]["baseline"]
        self.assertEqual([(x["share_id"], x["old_index"]) for x in baseline["selected"]], [("share-4", 3)])
        self.assertNotIn("observed_at", baseline)
        self.assertNotIn("expires_at", baseline)

    def test_expired_before_first_prepare_has_no_source_or_TTL_bypass(self):
        self.observation["expires_at_epoch"] = time.time() - 1
        workspace._atomic_write(self.observation)
        with self.assertRaises(AgentToolError) as error:
            actions.prepare_create_guangya_share(self.create_arguments(), self.context, workspace_actions=self._domain_actions.workspace)
        self.assertEqual(error.exception.code, "confirmation_stale")
        self.assertFalse(workspace._plan_path("a" * 32).exists())
        self.factory.assert_not_called()

    def test_owner_and_credential_gates_still_reject_before_source_is_issued(self):
        with self.assertRaises(AgentToolError):
            actions.prepare_create_guangya_share(self.create_arguments(), ToolContext(owner="another-owner", session_id=self.context.session_id), workspace_actions=self._domain_actions.workspace)
        self.factory.assert_not_called()
        self.client.credential_generation = 8
        with self.assertRaises(AgentToolError) as error:
            actions.prepare_create_guangya_share(self.create_arguments(), self.context, workspace_actions=self._domain_actions.workspace)
        self.assertEqual(error.exception.code, "confirmation_stale")
        self.client.file_info.assert_not_called()

    async def test_native_pipeline_and_sqlite_effect_plan_preserve_private_source(self):
        collector = ToolSpecCollector()
        register_specs(collector, workspace_actions=self._domain_actions.workspace)
        catalog = ToolCatalog(adapt_tool_spec(spec) for spec in collector.items)
        state_store = SQLiteKernelStore(secret_provider=lambda: "source-test-secret")
        effect_store = ConfirmationEffectPlanStore(SQLiteConfirmationStore())
        pipeline = ToolPipeline(catalog=catalog, state_store=state_store, reference_store=state_store, effect_store=effect_store)

        async def progress(_event):
            pass

        for operation in ("create", "revoke"):
            with self.subTest(operation=operation):
                session = "native-" + operation
                lease, _ = await state_store.begin_turn(owner=self.context.owner, session_id=session, request_id="prepare")
                context = ToolCallContext(owner=self.context.owner, session_id=session, request_id=lease.request_id,
                    turn_id=lease.turn_id, lease=lease, cancellation=CancellationToken(), report_progress=progress)
                reference = None
                if operation == "create":
                    workspace._atomic_write({**self.observation, "session_scope": workspace.workspace_session_scope(self.context.owner, session)})
                    args = {"observation_ref": self.ref, "object_refs": [self.handles[1]], "expires_days": 1, "access_code": "Ab12Cd34"}
                else:
                    reference = await state_store.put(owner=self.context.owner, session_id=session, kind="guangya_shares", value=self.collection)
                    args = {"guangya_shares_ref": reference.ref, "indices": [4]}
                result = await pipeline.execute("guangya.share." + operation, args, context=context)
                plan = result.effect_plan
                assert (plan) is not None
                loaded = ConfirmationEffectPlanStore(SQLiteConfirmationStore()).get_active_plan(
                    owner=self.context.owner, session_id=session, generation=lease.generation, plan_id=plan.plan_id)
                assert (loaded) is not None
                source = loaded.metadata["reobservation_source"]
                self.assertEqual(source, plan.metadata["reobservation_source"])
                self.assertEqual((source["owner"], source["session_id"]), (self.context.owner, session))
                self.assertEqual(len(source["baseline"]["selected"]), 1)
                if operation == "create":
                    self.assertEqual(source["baseline"]["selected"][0]["file_id"], "file-b")
                else:
                    self.assertEqual(source["baseline"]["selected"][0]["share_id"], "share-4")
                    # 原生 resolve 丢弃原 *_ref，原 ref 仍可从同一持久计划参数取得。
                    self.assertIsNone(source["baseline"]["ref"])
                    assert reference is not None
                    self.assertEqual(loaded.arguments[source["baseline"]["ref_argument"]], reference.ref)
                encoded = json.dumps(source, ensure_ascii=False)
                for secret in ("Ab12Cd34", "TOKEN-SECRET", "CODE-SECRET", "private.invalid"):
                    self.assertNotIn(secret, encoded)
                public = json.dumps(loaded.public_approval_dict(), ensure_ascii=False)
                for private in ("reobservation_source", "credential_generation", "file-b", "share-4", "Ab12Cd34"):
                    self.assertNotIn(private, public)
                self.assertEqual(json.loads(json.dumps(source)), source)
        self.client.create_user_share.assert_not_called()
        self.client.delete_user_shares.assert_not_called()


if __name__ == "__main__":
    unittest.main()
