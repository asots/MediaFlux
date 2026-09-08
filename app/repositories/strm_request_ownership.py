"""STRM 现有队列与下载请求的关联凭据；不包含独立消费者或重试循环。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

INTERRUPTED_ERROR = "上次进程在 STRM 同步或排队期间中断"
REFRESH_PENDING_ERROR = "STRM 已完成，媒体库刷新等待后台重试"


def _db():
    from app import database

    return database


def _snapshot(row) -> dict[str, Any]:
    return {
        "request_id": int(row["id"]),
        "generation": int(row["strm_generation"]),
        "organize_task_id": str(row["organize_task_id"] or ""),
    }


def _begin_request_owners(
    conn: sqlite3.Connection, request_ids: object, stamp: str
) -> list[dict[str, Any]]:
    owners = []
    for request_id in dict.fromkeys(
        int(value) for value in (request_ids or ()) if int(value) > 0
    ):
        row = conn.execute(
            "SELECT * FROM download_requests WHERE id=?", (request_id,)
        ).fetchone()
        if row is None or row["status"] in {"cancelled", "resubmitted", "failed"}:
            continue
        generation = int(row["strm_generation"]) + 1
        task_id = str(row["organize_task_id"] or "")
        # 同一整理的后续目标合并到新围栏，换整理对象则丢弃旧归属，不能串任务。
        conn.execute(
            "DELETE FROM strm_request_work WHERE request_id=? AND organize_task_id<>?",
            (request_id, task_id),
        )
        conn.execute(
            "UPDATE strm_request_work SET generation=?,failed_lease_generation=-1 WHERE request_id=?",
            (generation, request_id),
        )
        conn.execute(
            "UPDATE download_requests SET strm_generation=?,strm_status='queued',"
            "strm_error='',strm_finished_at=NULL,updated_at=? WHERE id=?",
            (generation, stamp, request_id),
        )
        owners.append(
            {
                "request_id": request_id,
                "generation": generation,
                "organize_task_id": task_id,
            }
        )
    return owners


def _current_owner_row(conn: sqlite3.Connection, owner: dict[str, Any]):
    return conn.execute(
        "SELECT * FROM download_requests WHERE id=? AND strm_generation=? "
        "AND COALESCE(organize_task_id,'')=? AND status NOT IN ('cancelled','resubmitted','failed')",
        (
            int(owner["request_id"]),
            int(owner["generation"]),
            str(owner["organize_task_id"]),
        ),
    ).fetchone()


def _attach_request_work(
    conn: sqlite3.Connection, owners: object, kind: str, work_key: str
) -> None:
    for owner in owners or ():
        if _current_owner_row(conn, owner) is None:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO strm_request_work(request_id,generation,organize_task_id,kind,work_key) "
            "VALUES(?,?,?,?,?)",
            (
                int(owner["request_id"]),
                int(owner["generation"]),
                str(owner["organize_task_id"]),
                kind,
                str(work_key),
            ),
        )


def refresh_work_key(path: str, allow_emby: bool) -> str:
    return hashlib.sha256(
        json.dumps([path, bool(allow_emby)], ensure_ascii=False).encode()
    ).hexdigest()


def current_strm_request_owners(request_ids: object) -> list[dict[str, Any]]:
    ids = list(
        dict.fromkeys(int(value) for value in (request_ids or ()) if int(value) > 0)
    )
    if not ids:
        return []
    with _db().get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM download_requests WHERE id IN ("
            + ",".join("?" for _ in ids)
            + ")",
            ids,
        ).fetchall()
    return [_snapshot(row) for row in rows]


def request_owners_for_work(
    kind: str,
    work_keys: object,
    *,
    claimed_targets: object = None,
) -> list[dict[str, Any]]:
    keys = list(dict.fromkeys(str(key) for key in (work_keys or ())))
    if not keys:
        return []
    with _db().get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if claimed_targets is not None:
            claims = {
                str(item["id"]): int(item.get("lease_generation") or 0)
                for item in claimed_targets
                if item.get("id")
            }
            keys = [
                key
                for key in keys
                if key in claims
                and conn.execute(
                    "SELECT 1 FROM strm_change_queue WHERE id=? AND state='running' AND lease_generation=?",
                    (key, claims[key]),
                ).fetchone()
            ]
            if not keys:
                return []
        placeholders = ",".join("?" for _ in keys)
        rows = conn.execute(
            "SELECT DISTINCT r.* FROM strm_request_work w JOIN download_requests r ON r.id=w.request_id "
            "AND r.strm_generation=w.generation AND COALESCE(r.organize_task_id,'')=w.organize_task_id "
            "WHERE w.kind=? AND w.work_key IN (" + placeholders + ") "
            "AND r.status NOT IN ('cancelled','resubmitted','failed')",
            [kind, *keys],
        ).fetchall()
        owners = []
        for row in rows:
            owner = _snapshot(row)
            retry = (
                kind == "change"
                and claimed_targets is not None
                and row["strm_status"] == "failed"
            )
            proof = (
                retry
                and conn.execute(
                    "SELECT 1 FROM strm_request_work w JOIN strm_change_queue q ON q.id=w.work_key "
                    "WHERE w.request_id=? AND w.generation=? AND w.kind='change' "
                    "AND w.failed_lease_generation>=0 AND q.lease_generation>w.failed_lease_generation "
                    "AND w.work_key IN (" + placeholders + ") LIMIT 1",
                    [owner["request_id"], owner["generation"], *keys],
                ).fetchone()
            )
            if proof:
                # 只允许当前工作自己的新lease恢复失败；换代后旧回调立即失效。
                owner["generation"] += 1
                conn.execute(
                    "UPDATE download_requests SET strm_generation=? WHERE id=?",
                    (owner["generation"], owner["request_id"]),
                )
                conn.execute(
                    "UPDATE strm_request_work SET generation=?,failed_lease_generation=-1 WHERE request_id=?",
                    (owner["generation"], owner["request_id"]),
                )
                _update_strm_owner_state(
                    conn,
                    owner,
                    {
                        "strm_status": "queued",
                        "strm_error": "",
                        "strm_finished_at": None,
                    },
                    _db().now(),
                    resume_failure=True,
                )
            owners.append(owner)
    return owners


def has_pending_strm_request_refresh(owners: object) -> bool:
    if not owners:
        return False
    with _db().get_conn() as conn:
        return any(
            _current_owner_row(conn, owner) is not None
            and conn.execute(
                "SELECT 1 FROM strm_request_work WHERE request_id=? AND generation=? AND kind='refresh' LIMIT 1",
                (int(owner["request_id"]), int(owner["generation"])),
            ).fetchone()
            is not None
            for owner in owners
        )


def _update_strm_owner_state(
    conn: sqlite3.Connection,
    owner: dict[str, Any],
    fields: dict[str, Any],
    stamp: str,
    *,
    resume_failure: bool = False,
) -> bool:
    row = _current_owner_row(conn, owner)
    if row is None:
        return False
    old_error = str(row["strm_error"] or "")
    if (
        row["strm_status"] == "failed"
        and old_error != INTERRUPTED_ERROR
        and not resume_failure
    ):
        return False  # 不能用旧成功/旧恢复覆盖后来真实发生的失败。
    if (
        int(owner["generation"]) > 0
        and fields.get("strm_status") == "partial"
        and fields.get("strm_error") == REFRESH_PENDING_ERROR
    ):
        pending_refresh = conn.execute(
            "SELECT 1 FROM strm_request_work WHERE request_id=? AND generation=? AND kind='refresh' LIMIT 1",
            (int(owner["request_id"]), int(owner["generation"])),
        ).fetchone()
        if not pending_refresh:
            # outbox可能已经在本轮结算前被消费者接走，迟到的pending不能倒退完成状态。
            fields = {**fields, "strm_status": "completed", "strm_error": ""}
    if fields.get("strm_status") == "completed":
        if row["strm_status"] == "partial" and old_error != REFRESH_PENDING_ERROR:
            return False
        waiting = conn.execute(
            "SELECT 1 FROM strm_request_work WHERE request_id=? AND generation=? LIMIT 1",
            (int(owner["request_id"]), int(owner["generation"])),
        ).fetchone()
        if waiting:
            return False
    allowed = {"strm_status", "strm_error", "strm_finished_at", "strm_run_id"}
    if set(fields) - allowed:
        raise ValueError("STRM 请求投影包含不支持的字段")
    from app.repositories.download_requests import _update_download_request_conn

    if not _update_download_request_conn(conn, int(owner["request_id"]), fields, stamp):
        return False
    if fields.get("strm_status") == "failed":
        # 此失败属于当前队列执行，记录失败lease；普通状态写入会撤销这份凭据。
        conn.execute(
            "UPDATE strm_request_work SET failed_lease_generation=COALESCE("
            "(SELECT lease_generation FROM strm_change_queue q WHERE q.id=work_key),-1) "
            "WHERE request_id=? AND generation=? AND kind='change'",
            (int(owner["request_id"]), int(owner["generation"])),
        )
    # 仅恢复明确属于启动中断/当前队列新lease的准入；后来独立失败不改。
    if row["strm_status"] == "failed" and (
        old_error == INTERRUPTED_ERROR or resume_failure
    ):
        from app.repositories.media_subscriptions import (
            _download_request_admission_projection,
        )

        updated = conn.execute(
            "SELECT * FROM download_requests WHERE id=?", (int(owner["request_id"]),)
        ).fetchone()
        projection = _download_request_admission_projection(updated, stamp)
        if projection is not None and projection[0] == "processing":
            conn.execute(
                "UPDATE media_download_admissions SET status='processing',error='',completed_at=NULL,updated_at=? "
                "WHERE request_id=? AND status='failed' AND error=? "
                "AND NOT EXISTS(SELECT 1 FROM media_download_admissions newer "
                "WHERE newer.media_key=media_download_admissions.media_key "
                "AND newer.id<>media_download_admissions.id "
                "AND newer.status IN ('claimed','dispatching','submitted','downloading','processing'))",
                (
                    stamp,
                    int(owner["request_id"]),
                    f"下载后处理失败（STRM 联动）：{old_error or '请在下载记录中重试'}"[
                        :500
                    ],
                ),
            )
    from app.repositories.media_subscriptions import (
        _sync_media_download_admissions_conn,
    )

    _sync_media_download_admissions_conn(conn, int(owner["request_id"]), stamp)
    return True


def update_strm_request_state(owner: dict[str, Any], **fields: Any) -> bool:
    with _db().get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        return _update_strm_owner_state(conn, owner, fields, _db().now())


def _complete_request_work(
    conn: sqlite3.Connection, kind: str, work_key: str, stamp: str
) -> None:
    owners = [
        dict(row)
        for row in conn.execute(
            "SELECT request_id,generation,organize_task_id FROM strm_request_work WHERE kind=? AND work_key=?",
            (kind, str(work_key)),
        ).fetchall()
    ]
    conn.execute(
        "DELETE FROM strm_request_work WHERE kind=? AND work_key=?",
        (kind, str(work_key)),
    )
    for owner in owners:
        _update_strm_owner_state(
            conn,
            owner,
            {
                "strm_status": "completed",
                "strm_error": "",
                "strm_finished_at": stamp,
            },
            stamp,
        )
