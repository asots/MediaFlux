"""活动搜索应选全域最近更新记录，不能先按各表ID截断再补排序。"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from unittest.mock import patch

from app import database as db
from app.agent.activity_actions import search_activities
from app.agent.models import ToolContext
from app.repositories import activity
from tests.support import IsolatedDatabaseTestCase


class ActivitySearchWindowAuditTests(IsolatedDatabaseTestCase):
    def setUp(self):
        with db.get_conn() as conn:
            for table in (
                "download_request_keys",
                "download_log",
                "download_requests",
                "local_media_task_items",
                "local_media_tasks",
                "local_media_sources",
                "organize_log",
            ):
                conn.execute(f"DELETE FROM {table}")
            self.source = conn.execute(
                "INSERT INTO local_media_sources(name,local_root,created_at,updated_at) "
                "VALUES('audit','/fixture','2026-01-01 00:00:00','2026-01-01 00:00:00')"
            ).lastrowid

    def _add(self, kind, title, updated, *, created="2026-01-01 00:00:00"):
        key = uuid.uuid4().hex
        with db.get_conn() as conn:
            if kind == "download":
                return conn.execute(
                    "INSERT INTO download_requests(request_key,kind,title,status,created_at,updated_at) "
                    "VALUES(?,'magnet',?,'downloading',?,?)",
                    (key, title, created, updated),
                ).lastrowid
            if kind == "organize":
                return conn.execute(
                    "INSERT INTO organize_log(source,original_path,new_path,title,status,created_at,updated_at) "
                    "VALUES('guangya','/fixture/source','/fixture/target',?,'success',?,?)",
                    (title, created, updated),
                ).lastrowid
            return conn.execute(
                "INSERT INTO local_media_tasks(source_id,content_path,operation_token,title,status,created_at,updated_at) "
                "VALUES(?,'/fixture',?,?,'waiting_stable',?,?)",
                (self.source, key, title, created, updated),
            ).lastrowid

    def test_old_ids_updated_latest_are_not_hidden_by_newer_ids(self):
        for kind in ("download", "organize", "local_media"):
            with self.subTest(kind=kind):
                wanted = self._add(kind, f"{kind}-wanted", "2026-01-09 00:00:00")
                for index in range(30):
                    self._add(kind, f"{kind}-old-{index}", "2026-01-02 00:00:00")
                rows = activity.search(query=kind, limit=1)
                self.assertEqual((rows[0]["kind"], rows[0]["id"]), (kind, wanted))
                self.assertEqual(len(rows), 2)  # 多取一条维持 has_more 合同。

    def test_three_domains_share_one_statement_and_bounded_global_window(self):
        for kind in ("download", "organize", "local_media"):
            for index in range(25):
                self._add(kind, f"fixture-{index}", f"2026-01-{index + 1:02} 00:00:00")
        original = db.get_conn
        statements = []

        @contextmanager
        def traced():
            with original() as conn:
                conn.set_trace_callback(statements.append)
                yield conn

        with patch.object(db, "get_conn", traced):
            rows = activity.search(query="fixture", limit=20)
        self.assertEqual(len(rows), 21)
        self.assertEqual(sum(sql.startswith("SELECT") for sql in statements), 1)
        self.assertEqual(
            set(rows[0]), {"kind", "id", "title", "status", "created_at", "updated_at"}
        )

    def test_source_reads_cannot_mix_a_later_cross_domain_insert(self):
        first = self._add("download", "fixture-first", "2026-01-02 00:00:00")
        original = db.get_conn
        fired = False

        class Cursor:
            def __init__(self, cursor):
                self.cursor = cursor

            def fetchall(cursor_self):
                nonlocal fired
                rows = cursor_self.cursor.fetchall()
                if not fired:
                    fired = True
                    self._add("local_media", "fixture-later", "2026-01-09 00:00:00")
                return rows

        class Connection:
            def __init__(self, conn):
                self.conn = conn

            def __getattr__(self, name):
                return getattr(self.conn, name)

            def execute(self, sql, *args):
                cursor = self.conn.execute(sql, *args)
                return (
                    Cursor(cursor) if sql.startswith("SELECT") and not fired else cursor
                )

        @contextmanager
        def hooked():
            with original() as conn:
                yield Connection(conn)

        with patch.object(db, "get_conn", hooked):
            rows = activity.search(query="fixture", limit=20)
        self.assertTrue(fired)
        self.assertEqual(
            [(row["kind"], row["id"]) for row in rows], [("download", first)]
        )
        self.assertEqual(len(activity.search(query="fixture", limit=20)), 2)

    def test_public_positions_reference_actual_newest_identity_and_has_more(self):
        wanted = self._add("download", "fixture-newest", "2026-01-09 00:00:00")
        for index in range(25):
            self._add("download", f"fixture-{index}", "2026-01-02 00:00:00")
        result = search_activities(
            {"query": "fixture", "limit": 1}, ToolContext(owner="fixture")
        )
        self.assertEqual(result.data["items"][0]["title"], "fixture-newest")
        self.assertTrue(result.data["has_more"])
        self.assertEqual(
            result.references[0].value["items"], [{"kind": "download", "id": wanted}]
        )

    def test_literal_query_empty_timestamp_and_empty_results_keep_contract(self):
        wanted = self._add(
            "organize", "100%_fixture", "", created="2026-01-09 00:00:00"
        )
        self._add("download", "other fixture", "2026-01-02 00:00:00")
        self.assertEqual(activity.search(query="%_", limit=10)[0]["id"], wanted)
        self.assertEqual(activity.search(query="missing", limit=10), [])
        rows = activity.search(query="", limit=0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], wanted)

    def test_equal_time_and_id_keep_stable_kind_order_on_repeated_reads(self):
        for kind in ("download", "organize", "local_media"):
            identifier = self._add(kind, "tie-fixture", "2026-01-09 00:00:00")
            table = {
                "download": "download_requests",
                "organize": "organize_log",
                "local_media": "local_media_tasks",
            }[kind]
            with db.get_conn() as conn:
                conn.execute(f"UPDATE {table} SET id=999999 WHERE id=?", (identifier,))
        expected = ["download", "organize", "local_media"]
        for _ in range(3):
            self.assertEqual(
                [row["kind"] for row in activity.search(query="tie-fixture", limit=3)],
                expected,
            )
