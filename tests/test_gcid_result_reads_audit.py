"""GCID 历史回执只读取所需失败样本，不为计数加载完整明细。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from unittest.mock import patch

from app import database as db
from app.modules import gcid_import
from tests.support import IsolatedDatabaseTestCase


class GCIDResultReadsAuditTests(IsolatedDatabaseTestCase):
    def _task(self, count=2000):
        task_id = db.create_gcid_import_task(
            operation_token=self._testMethodName,
            manifest_digest="a" * 64,
            target_dir_id="target",
            file_count=count,
            total_size=count,
        )
        db.replace_gcid_import_items(
            task_id,
            [
                {
                    "path": f"Film/{i:05d}.mkv",
                    "size": 1,
                    "gcid": f"gcid-{i}",
                    "status": "failed",
                }
                for i in range(count)
            ],
        )
        return task_id

    @contextmanager
    def _reads(self):
        real = db.get_conn
        metrics = {"detail_rows": 0, "selects": []}

        @contextmanager
        def traced():
            with real() as conn:
                conn.set_trace_callback(
                    lambda sql: (
                        metrics["selects"].append(sql)
                        if sql.lstrip().upper().startswith("SELECT")
                        else None
                    )
                )

                def row_factory(cursor, values):
                    if "gcid" in {column[0] for column in cursor.description}:
                        metrics["detail_rows"] += 1
                    return sqlite3.Row(cursor, values)

                conn.row_factory = row_factory
                yield conn

        with patch.object(db, "get_conn", traced):
            yield metrics

    def test_failed_samples_materialize_only_three_of_two_thousand_items(self):
        task_id = self._task()
        with self._reads() as reads:
            samples = gcid_import._failed_samples(task_id)
        self.assertEqual(
            [r["path"] for r in samples], [f"Film/{i:05d}.mkv" for i in range(3)]
        )
        self.assertEqual(reads["detail_rows"], 3)
        self.assertEqual(len(reads["selects"]), 1)
        self.assertEqual(set(samples[0]), {"id", "path", "error"})

    def test_finish_task_counts_in_sql_and_materializes_only_its_samples(self):
        task_id = self._task()
        with self._reads() as reads:
            result = gcid_import._finish_task(task_id)
        self.assertEqual(
            (result["status"], result["success_count"], result["failed_count"]),
            ("failed", 0, 2000),
        )
        self.assertEqual(len(result["failed_samples"]), 3)
        self.assertEqual(reads["detail_rows"], 3)
        self.assertEqual(db.get_gcid_import_task(task_id)["failed_count"], 2000)

    def test_sample_limits_and_existing_full_detail_reads_keep_contract(self):
        task_id = self._task(8)
        for limit in (-1, 0, 1, 3, 20):
            with self.subTest(limit=limit), self._reads() as reads:
                samples = gcid_import._failed_samples(task_id, limit)
                self.assertEqual(len(samples), min(8, max(0, limit)))
                self.assertEqual(reads["detail_rows"], len(samples))
        self.assertEqual(len(db.list_gcid_import_items(task_id, "failed")), 8)
        self.assertEqual(len(db.list_gcid_import_items(task_id)), 8)

    def test_partial_success_and_empty_history_keep_summary_fields(self):
        task_id = self._task(3)
        rows = db.list_gcid_import_items(task_id)
        gcid_import._update_item(
            rows[0]["id"], status="success", remote_file_id="first"
        )
        result = gcid_import._finish_task(task_id)
        self.assertEqual(
            (result["status"], result["success_count"], result["failed_count"]),
            ("partial_success", 1, 2),
        )
        self.assertEqual(
            [r["id"] for r in result["failed_samples"]], [r["id"] for r in rows[1:]]
        )
        db.replace_gcid_import_items(task_id, [])
        empty = gcid_import._finish_task(task_id)
        self.assertEqual(
            (empty["status"], empty["success_count"], empty["failed_count"]),
            ("failed", 0, 0),
        )
        self.assertEqual(empty["failed_samples"], [])
