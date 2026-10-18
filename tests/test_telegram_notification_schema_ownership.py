"""通知日常消费只使用正式 schema，不在热路径隐式执行第二套建表逻辑。"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from app import database as db
from app.repositories import telegram_notifications as repository
from tests.support import IsolatedDatabaseTestCase


class TelegramNotificationSchemaOwnershipTests(IsolatedDatabaseTestCase):
    def test_publish_read_claim_complete_do_not_run_schema_ddl(self) -> None:
        statements = []
        original = db.get_conn

        @contextmanager
        def traced_connection():
            with original() as conn:
                conn.set_trace_callback(statements.append)
                yield conn

        with patch.object(db, "get_conn", traced_connection):
            row = repository.upsert_notification(
                "schema-owner", topic="system", importance="result", chat_id="test", event_json="{}",
            )
            assert (repository.get_notification("schema-owner")) is not None
            claimed = repository.claim_due_notifications(event_key="schema-owner")
            self.assertEqual(len(claimed), 1)
            assert row is not None
            self.assertTrue(repository.complete_notification(
                row["id"], lease_generation=claimed[0]["lease_generation"],
                claimed_revision=claimed[0]["revision"], message_id=7,
            ))
            get_notification_result = repository.get_notification("schema-owner")
            assert get_notification_result is not None
            self.assertEqual(get_notification_result["status"], "sent")
        ddl = [sql for sql in statements if sql.lstrip().upper().startswith(("CREATE ", "ALTER ", "DROP "))]
        self.assertEqual(ddl, [], "热路径必须由数据库初始化统一提供 schema")
