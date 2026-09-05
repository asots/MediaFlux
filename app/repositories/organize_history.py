"""整理操作历史与业务前像，单一事务恢复身份、位置和成员目标快照。"""
from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType


def _database() -> "ModuleType":
    from app import database

    return database


_BUSINESS_LOG_FIELDS = (
    "tmdb_id", "provider", "external_id", "media_type", "title", "year",
    "season", "episode", "new_path", "target_parent_id", "release_parse_json",
)
_FINISH_FIELDS = {"status", "operation_type", "current_parent_id", "current_name", "error"}


def list_organize_operation_steps(log_id: int, limit: int = 300) -> list[sqlite3.Row]:
    with _database().get_conn() as conn:
        return conn.execute(
            "SELECT * FROM organize_operation_steps WHERE log_id=? ORDER BY id DESC LIMIT ?",
            (log_id, max(1, min(int(limit or 300), 1000))),
        ).fetchall()


def add_organize_operation_step(
    log_id: int,
    operation_token: str,
    step_index: int,
    action: str,
    **fields,
) -> int:
    timestamp = _database().now()
    before = fields.get("state_before")
    before_json = (
        json.dumps(_validated_snapshot(before), ensure_ascii=False, separators=(",", ":"))
        if before is not None else ""
    )
    with _database().get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO organize_operation_steps(log_id,operation_token,step_index,action,file_id,"
            "from_parent_id,from_name,to_parent_id,to_name,status,error,started_at,finished_at,state_before_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                log_id,
                operation_token,
                int(step_index),
                action,
                str(fields.get("file_id") or ""),
                str(fields.get("from_parent_id") or ""),
                str(fields.get("from_name") or ""),
                str(fields.get("to_parent_id") or ""),
                str(fields.get("to_name") or ""),
                str(fields.get("status") or "pending"),
                str(fields.get("error") or ""),
                fields.get("started_at") or timestamp,
                fields.get("finished_at"),
                before_json,
            ),
        )
        return int(cur.lastrowid)


def finish_organize_operation_step(step_id: int, status: str, error: str = "") -> bool:
    with _database().get_conn() as conn:
        cur = conn.execute(
            "UPDATE organize_operation_steps SET status=?,error=?,finished_at=? WHERE id=?",
            (status, error, _database().now(), step_id),
        )
        return cur.rowcount == 1


def _validated_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("version", 1) != 1:
        raise ValueError("整理业务前像格式无效")
    log = value.get("log")
    items = value.get("items")
    if not isinstance(log, dict) or not isinstance(items, list):
        raise ValueError("整理业务前像缺少日志或成员状态")
    if any(key not in log for key in _BUSINESS_LOG_FIELDS):
        raise ValueError("整理业务前像字段不完整，不能猜测旧媒体身份")
    clean_items = []
    seen_ids = set()
    for item in items:
        if not isinstance(item, dict) or not all(key in item for key in ("id", "file_id", "target_parent_id", "target_name")):
            raise ValueError("整理成员前像字段不完整")
        item_id = int(item["id"])
        file_id = str(item["file_id"] or "")
        if item_id <= 0 or not file_id or item_id in seen_ids:
            raise ValueError("整理成员前像 ID 无效或重复")
        seen_ids.add(item_id)
        clean_items.append({"id": item_id, "file_id": file_id,
                            "target_parent_id": item["target_parent_id"], "target_name": item["target_name"]})
    return {"version": 1, "log": {key: log[key] for key in _BUSINESS_LOG_FIELDS}, "items": clean_items}


def capture_organize_business_snapshot(log_id: int) -> dict[str, Any]:
    """同一 SQLite 读快照中取得前像；不要把临时 busy 状态当成待恢复身份。"""
    with _database().get_conn() as conn:
        conn.execute("BEGIN")
        log = conn.execute(
            f"SELECT {','.join(_BUSINESS_LOG_FIELDS)} FROM organize_log WHERE id=?", (int(log_id),),
        ).fetchone()
        if log is None:
            raise LookupError("整理日志不存在")
        items = conn.execute(
            "SELECT id,file_id,target_parent_id,target_name FROM organize_log_items "
            "WHERE log_id=? ORDER BY id", (int(log_id),),
        ).fetchall()
        return {"version": 1, "log": dict(log), "items": [dict(row) for row in items]}


def restore_organize_business_snapshot(log_id: int, snapshot: dict, **log_fields) -> bool:
    """文件补偿全部成功后原子恢复业务前像；任何成员失配时整笔拒绝。"""
    state = _validated_snapshot(snapshot)
    if set(log_fields) - _FINISH_FIELDS:
        raise ValueError("不允许用收尾字段覆盖业务前像")
    database = _database()
    with database.get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT id FROM organize_log WHERE id=?", (int(log_id),)).fetchone() is None:
            raise LookupError("整理日志不存在")
        members = conn.execute(
            "SELECT id,file_id FROM organize_log_items WHERE log_id=?", (int(log_id),),
        ).fetchall()
        actual = {(int(row["id"]), str(row["file_id"])) for row in members}
        expected = {(item["id"], item["file_id"]) for item in state["items"]}
        if actual != expected:
            raise ValueError("媒体组成员已变化，业务前像不能安全应用")
        timestamp = database.now()
        values = {**state["log"], **log_fields}
        assignments = ','.join(f'{key}=?' for key in values)
        conn.execute(
            f"UPDATE organize_log SET {assignments},version=version+1,updated_at=? WHERE id=?",
            [*values.values(), timestamp, int(log_id)],
        )
        for item in state["items"]:
            conn.execute(
                "UPDATE organize_log_items SET target_parent_id=?,target_name=?,updated_at=? "
                "WHERE id=? AND log_id=? AND file_id=?",
                (item["target_parent_id"], item["target_name"], timestamp, item["id"], int(log_id), item["file_id"]),
            )
    return True


def list_latest_reversible_organize_steps(log_id: int) -> list[sqlite3.Row]:
    """回退使用完整操作集，不复用日志页最近300步的展示窗口。"""
    with _database().get_conn() as conn:
        conn.execute("BEGIN")
        latest = conn.execute(
            "SELECT operation_token FROM organize_operation_steps "
            "WHERE log_id=? AND status='success' AND action='move_rename' "
            "ORDER BY id DESC LIMIT 1", (int(log_id),),
        ).fetchone()
        if latest is None:
            return []
        return conn.execute(
            "SELECT * FROM organize_operation_steps WHERE log_id=? AND operation_token=? "
            "AND status='success' AND action='move_rename' ORDER BY id DESC",
            (int(log_id), latest["operation_token"]),
        ).fetchall()


def list_pending_organize_probe_steps(
    log_id: int, job_id: int, *, include_succeeded: bool = False,
) -> list[sqlite3.Row]:
    """完整恢复写意图；收尾可带成功步骤，作为跨进程的已提交事实。"""
    statuses = ("running", "interrupted", "success") if include_succeeded else ("running", "interrupted")
    placeholders = ",".join("?" for _ in statuses)
    with _database().get_conn() as conn:
        return conn.execute(
            "SELECT * FROM organize_operation_steps WHERE log_id=? "
            "AND operation_token GLOB ? AND action='probe_rename' "
            f"AND status IN ({placeholders}) ORDER BY id DESC",
            (int(log_id), f"probe:{int(job_id)}:*", *statuses),
        ).fetchall()
