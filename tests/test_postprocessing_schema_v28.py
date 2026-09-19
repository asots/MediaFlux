"""持久整理收尾与 probe 通知身份的正式 v27 -> v28 升级契约。"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

from app import database as db
from app import database_migrations
from app.modules.backup import verify_backup
from tests.support import IsolatedDatabaseTestCase


class PostprocessingSchemaV28Tests(IsolatedDatabaseTestCase):
    @staticmethod
    def _legacy_connection() -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(
            "CREATE TABLE download_requests(id INTEGER PRIMARY KEY);"
            "CREATE TABLE organize_confirmations(id INTEGER PRIMARY KEY);"
            "CREATE TABLE organize_probe_queue("
            "id INTEGER PRIMARY KEY,organize_log_id INTEGER NOT NULL UNIQUE,"
            "rules_json TEXT NOT NULL DEFAULT '{}',"
            "pending_strm_changes_json TEXT NOT NULL DEFAULT '[]',"
            "status TEXT NOT NULL DEFAULT 'queued',attempts INTEGER NOT NULL DEFAULT 0);"
            "INSERT INTO download_requests(id) VALUES(57);"
            "INSERT INTO organize_confirmations(id) VALUES(78);"
            "INSERT INTO organize_probe_queue(id,organize_log_id,rules_json,"
            "pending_strm_changes_json,status,attempts) "
            "VALUES(1,42,'{\"notify_enabled\":true}',"
            "'[{\"file_id\":\"synthetic\",\"action\":\"upsert\"}]','running',1);"
            "PRAGMA user_version=27;"
        )
        return conn

    def test_schema_28_registers_postprocessing_upgrade(self) -> None:
        self.assertGreaterEqual(db.SCHEMA_VERSION, 28)
        self.assertIs(
            database_migrations._SCHEMA_MIGRATIONS[27], database_migrations._migrate_postprocessing_recovery_v28,
        )

    def test_fresh_schema_has_probe_identity_and_due_reconciliation(self) -> None:
        with db.get_conn() as conn:
            columns = {
                str(row["name"]): row
                for row in conn.execute("PRAGMA table_info(organize_probe_queue)")
            }
            queue_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(download_staging_reconcile)")
            }
            indexes = {
                str(row["name"])
                for row in conn.execute("PRAGMA index_list(download_staging_reconcile)")
            }
        self.assertEqual(columns["notification_context_json"]["notnull"], 1)
        self.assertEqual(columns["notification_context_json"]["dflt_value"], "'{}'")
        self.assertTrue({
            "confirmation_id", "request_id", "identity_json", "result_json", "status",
            "attempt_count", "next_attempt_at", "last_error", "created_at", "updated_at",
        }.issubset(queue_columns))
        self.assertIn("idx_download_staging_reconcile_due", indexes)

    def test_upgrade_defaults_unknown_probe_scope_without_rewriting_payload(self) -> None:
        conn = self._legacy_connection()
        try:
            before = tuple(conn.execute("SELECT * FROM organize_probe_queue").fetchone())
            database_migrations._migrate_postprocessing_recovery_v28(conn)
            row = conn.execute("SELECT * FROM organize_probe_queue").fetchone()
            self.assertEqual(tuple(row)[:-1], before)
            self.assertEqual(row["notification_context_json"], "{}")
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 27)
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM download_staging_reconcile").fetchone()[0],
                0,
            )
        finally:
            conn.close()

    def test_repeated_upgrade_preserves_context_and_reconciliation_intent(self) -> None:
        conn = self._legacy_connection()
        try:
            database_migrations._migrate_postprocessing_recovery_v28(conn)
            context = json.dumps({"version": 1, "task_id": "synthetic-parent"})
            conn.execute(
                "UPDATE organize_probe_queue SET notification_context_json=?", (context,),
            )
            conn.execute(
                "INSERT INTO download_staging_reconcile(confirmation_id,request_id,"
                "identity_json,result_json,status,attempt_count,next_attempt_at,last_error,"
                "created_at,updated_at) VALUES(78,57,'{}','{}','retry',2,'later',"
                "'ReadUnavailable','before','before')"
            )
            before = tuple(conn.execute("SELECT * FROM download_staging_reconcile").fetchone())
            database_migrations._migrate_postprocessing_recovery_v28(conn)
            self.assertEqual(
                conn.execute("SELECT notification_context_json FROM organize_probe_queue").fetchone()[0],
                context,
            )
            self.assertEqual(
                tuple(conn.execute("SELECT * FROM download_staging_reconcile").fetchone()),
                before,
            )
        finally:
            conn.close()

    def test_queue_guards_invalid_state_attempts_and_orphan_identity(self) -> None:
        conn = self._legacy_connection()
        try:
            database_migrations._migrate_postprocessing_recovery_v28(conn)
            query = (
                "INSERT INTO download_staging_reconcile(confirmation_id,request_id,"
                "identity_json,status,attempt_count,created_at,updated_at) "
                "VALUES(?,?, '{}',?,?, 'stamp','stamp')"
            )
            for values in ((78, 57, "unknown", 0), (78, 57, "pending", -1),
                           (999, 57, "pending", 0), (78, 999, "pending", 0)):
                with self.subTest(values=values), self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(query, values)
            conn.execute(query, (78, 57, "pending", 0))
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(query, (78, 57, "pending", 0))
            conn.execute("DELETE FROM organize_confirmations WHERE id=78")
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM download_staging_reconcile").fetchone()[0], 0,
            )
        finally:
            conn.close()

    def test_failed_upgrade_rolls_back_ddl_and_keeps_schema_generation(self) -> None:
        conn = self._legacy_connection()
        try:
            def interrupted(connection: sqlite3.Connection) -> None:
                database_migrations._migrate_postprocessing_recovery_v28(connection)
                raise RuntimeError("synthetic migration interruption")

            with self.assertRaisesRegex(RuntimeError, "synthetic migration interruption"):
                db._run_schema_savepoint(conn, operation=interrupted)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(organize_probe_queue)")}
            self.assertNotIn("notification_context_json", columns)
            self.assertIsNone(conn.execute(
                "SELECT name FROM sqlite_master WHERE name='download_staging_reconcile'"
            ).fetchone())
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 27)
        finally:
            conn.close()

    def test_real_v27_initialization_keeps_old_work_and_crosses_backup_gate(self) -> None:
        previous_path = db.DB_PATH
        previous_mode = db._configured_test_mode
        with tempfile.TemporaryDirectory(prefix="mediaflux-v27-postprocessing-") as root:
            path = Path(root) / "legacy.db"
            conn = sqlite3.connect(path)
            try:
                conn.executescript(db._SCHEMA)
                conn.execute("ALTER TABLE organize_probe_queue DROP COLUMN notification_context_json")
                conn.execute("DROP TABLE download_staging_reconcile")
                conn.execute(
                    "INSERT INTO download_requests(id,request_key,kind,targets,status,"
                    "gy_isolated,gy_status,organize_started,organize_status,"
                    "gy_staging_cleanup_status,gy_staging_cleanup_error,created_at,updated_at) "
                    "VALUES(57,'legacy-57','magnet','guangya','completed',1,'completed',1,"
                    "'completed','retained','legacy unrecognized item','before','before')"
                )
                conn.execute(
                    "INSERT INTO organize_log(id,source,original_path,new_path,status,created_at) "
                    "VALUES(42,'guangya','/before','/after','success','before')"
                )
                conn.execute(
                    "INSERT INTO organize_probe_queue(organize_log_id,source_id,rel_dir,"
                    "status,next_attempt_at,created_at,updated_at,pending_strm_changes_json) "
                    "VALUES(42,'synthetic-source','series','queued','later','before','before',"
                    "'[{\"file_id\":\"synthetic-file\"}]')"
                )
                conn.execute("PRAGMA user_version=27")
                conn.commit()
            finally:
                conn.close()
            db.configure_database(path, test_mode=True)
            try:
                with (
                    mock.patch.object(db, "_test_mode_enabled", return_value=False),
                    mock.patch.object(
                        db, "_create_pre_migration_backup", wraps=db._create_pre_migration_backup,
                    ) as backup,
                ):
                    db.init_db()
                self.assertEqual(backup.call_count, 1)
                self.assertEqual(backup.call_args.kwargs["current_version"], 27)
                archives = list((path.parent / "backups").glob("*.zip"))
                self.assertEqual(len(archives), 1)
                manifest = verify_backup(archives[0])
                self.assertEqual(manifest.payload["database_schema_version"], 27)
                restored_path = path.parent / "verified-preupgrade.db"
                with ZipFile(archives[0]) as archive:
                    restored_path.write_bytes(archive.read("database/mediaflux.db"))
                with sqlite3.connect(restored_path) as before_upgrade:
                    self.assertEqual(before_upgrade.execute("PRAGMA user_version").fetchone()[0], 27)
                    self.assertEqual(before_upgrade.execute("PRAGMA quick_check").fetchone()[0], "ok")
                    self.assertEqual(before_upgrade.execute(
                        "SELECT gy_staging_cleanup_status FROM download_requests WHERE id=57"
                    ).fetchone()[0], "retained")
                    columns = {row[1] for row in before_upgrade.execute("PRAGMA table_info(organize_probe_queue)")}
                    self.assertNotIn("notification_context_json", columns)
                with db.get_conn() as migrated:
                    self.assertEqual(migrated.execute("PRAGMA user_version").fetchone()[0], db.SCHEMA_VERSION)
                    row = migrated.execute(
                        "SELECT pending_strm_changes_json,notification_context_json,status "
                        "FROM organize_probe_queue WHERE organize_log_id=42"
                    ).fetchone()
                    self.assertEqual(tuple(row), ('[{"file_id":"synthetic-file"}]', "{}", "queued"))
                    row = migrated.execute(
                        "SELECT gy_staging_cleanup_status,gy_staging_cleanup_error "
                        "FROM download_requests WHERE id=57"
                    ).fetchone()
                    self.assertEqual(tuple(row), ("retained", "legacy unrecognized item"))
                    self.assertEqual(migrated.execute("PRAGMA quick_check").fetchone()[0], "ok")
                    self.assertEqual(migrated.execute("PRAGMA foreign_key_check").fetchall(), [])
                db.init_db()
                with db.get_conn() as migrated:
                    self.assertEqual(migrated.execute("PRAGMA user_version").fetchone()[0], db.SCHEMA_VERSION)
            finally:
                db.configure_database(previous_path, test_mode=previous_mode)
