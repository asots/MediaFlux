"""schema 只有正式初始化/迁移入口；Agent 热路径不能另行建表或补列。"""
from __future__ import annotations

import ast
import asyncio
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from app import database as db
from app.agent.confirmation import SQLiteConfirmationStore
from app.agent.kernel.persistence import SQLiteKernelStore
from app.agent.kernel.state import StateUpdate
from app.agent.rate_limit import AgentRateLimiter
from tests.support import IsolatedDatabaseTestCase


class SchemaSingleOwnerTests(IsolatedDatabaseTestCase):
    @contextmanager
    def _without_ddl(self):
        statements = []
        original = db.get_conn

        @contextmanager
        def trace():
            with original() as conn:
                conn.set_trace_callback(statements.append)
                yield conn

        with patch.object(db, "get_conn", trace):
            yield
        ddl = [sql for sql in statements if sql.lstrip().upper().startswith(("CREATE ", "ALTER ", "DROP "))]
        self.assertEqual(ddl, [])

    def test_kernel_session_roundtrip_does_not_create_schema(self) -> None:
        async def roundtrip():
            store = SQLiteKernelStore(secret_provider=lambda: "schema-owner-test")
            lease, _ = await store.begin_turn(owner="schema-owner", session_id="session", request_id="request")
            await store.commit(lease, updates=(StateUpdate("summary", "kept"),))
            self.assertTrue(await store.is_current(lease))
            self.assertEqual((await store.load(owner="schema-owner", session_id="session")).summary, "kept")
            self.assertTrue(await store.delete_session(owner="schema-owner", session_id="session"))

        with self._without_ddl():
            asyncio.run(roundtrip())

    def test_confirmation_issue_claim_does_not_create_schema(self) -> None:
        store = SQLiteConfirmationStore()
        with self._without_ddl():
            ticket = store.issue(owner="schema-owner", tool_name="write.test", arguments={"id": 7})
            self.assertEqual(len(store.list_active_tickets(owner="schema-owner")), 1)
            claimed = store.claim_and_rotate_owner(owner="schema-owner", confirmation_id=ticket.confirmation_id)
            self.assertEqual(claimed.arguments, {"id": 7})
            self.assertEqual(store.list_active_tickets(owner="schema-owner"), [])

    def test_shared_limiter_does_not_create_or_migrate_schema(self) -> None:
        limiter = AgentRateLimiter(shared=True)
        with self._without_ddl(), patch("app.agent.rate_limit.time.time", return_value=130.0):
            limiter.reset()
            self.assertTrue(limiter.allow("schema-owner", limit=1, window_seconds=60))
            self.assertFalse(limiter.allow("schema-owner", limit=1, window_seconds=60))
            self.assertEqual(limiter.tracked_keys(), 1)

    def test_no_alternative_runtime_table_definitions_remain(self) -> None:
        root = Path(__file__).resolve().parents[1]
        owners = {"app/database_schema.py", "app/database_migrations.py"}
        violations = []
        for path in (root / "app").rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            if relative in owners:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = " ".join(node.value.upper().split())
                    if value.startswith(("CREATE TABLE ", "CREATE INDEX ")):
                        violations.append(f"{relative}:{node.lineno}")
        self.assertEqual(violations, [])

    def test_organize_delete_audit_crud_uses_initialized_schema_only(self):
        with self._without_ddl():
            audit_id = db.add_organize_delete_audit(trigger="audit-test", file_id="audit-file", reason="test", status="pending")
            self.assertTrue(db.update_organize_delete_audit(audit_id, status="success"))
            self.assertEqual(db.get_organize_delete_audit(audit_id)["status"], "success")
            self.assertTrue(any(row["id"] == audit_id for row in db.list_organize_delete_audits()))

    def test_retired_strm_source_crud_uses_initialized_schema_only(self):
        with self._without_ddl():
            db.enqueue_strm_retired_source("audit-source", "audit", "/isolated/strm")
            self.assertTrue(any(row["source_id"] == "audit-source" for row in db.list_strm_retired_sources()))
            db.update_strm_retired_source_error("audit-source", "test")
            self.assertEqual(db.cancel_strm_retired_sources(["audit-source"]), 1)
            db.enqueue_strm_retired_source("audit-source", "audit", "/isolated/strm")
            self.assertEqual(db.delete_strm_retired_source("audit-source"), 1)

    def test_confirmed_action_audit_uses_initialized_schema_only(self):
        with self._without_ddl():
            audit_id = db.add_agent_action_history(owner_digest="a"*64, tool_name="audit.test", risk="write", status="succeeded", ok=True, summary="test")
            self.assertGreater(audit_id, 0)
