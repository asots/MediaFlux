"""只读活动关联：只认持久化请求/运行标识，不凭标题拼接跨域时间线。"""

from __future__ import annotations

from app import database as db

_TABLES = {
    "download": "download_requests",
    "organize": "organize_log",
    "local_media": "local_media_tasks",
}


def search(*, query: str, limit: int = 10) -> list[dict]:
    """先按同一更新时间口径选全域窗口；单条 SQL 同时保证跨模块读快照。"""
    bounded = max(1, min(int(limit), 20))
    selections = []
    params: list[object] = []
    for rank, (kind, table) in enumerate(_TABLES.items()):
        # 表名和类型只取本模块常量；instr 保留 %/_ 的文字包含语义。
        selections.append(
            f"SELECT '{kind}' AS kind,{rank} AS kind_rank,"
            f"id,title,status,created_at,updated_at FROM {table} "
            "WHERE (?='' OR instr(lower(title),lower(?))>0)"
        )
        params.extend((query, query))
    params.append(bounded + 1)
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT kind,id,title,status,created_at,updated_at FROM ("
            + " UNION ALL ".join(selections)
            + ") ORDER BY COALESCE(NULLIF(updated_at,''),created_at) DESC,"
            "id DESC,kind_rank ASC LIMIT ?",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def snapshot(kind: str, identifier: int) -> dict | None:
    table = _TABLES.get(kind)
    if table is None:
        raise ValueError("活动类型无效")
    with db.get_conn() as conn:
        conn.execute("BEGIN")
        row = conn.execute(
            f"SELECT * FROM {table} WHERE id=?", (identifier,)
        ).fetchone()
        if row is None:
            return None
        result = {"kind": kind, "record": dict(row)}
        if kind == "download":
            result["logs"] = [
                dict(item)
                for item in conn.execute(
                    "SELECT source,status,progress,error,created_at,updated_at,completed_at "
                    "FROM download_log WHERE request_id=? ORDER BY id DESC LIMIT 40",
                    (identifier,),
                )
            ]
            result["runs"] = [
                dict(item)
                for item in conn.execute(
                    "SELECT id,task_name,status,started_at,finished_at,error FROM task_runs "
                    "WHERE id IN (?,?) ORDER BY id",
                    (row["organize_run_id"], row["strm_run_id"]),
                )
            ]
            verification = conn.execute(
                "SELECT status,result,attempts,updated_at FROM agent_download_verifications WHERE request_id=?",
                (identifier,),
            ).fetchone()
            result["verification"] = dict(verification) if verification else None
            # 本地任务归属按 tracker 写入的 qB 身份确定，绝不使用媒体标题猜测。
            result["local_tasks"] = [
                dict(item)
                for item in conn.execute(
                    "SELECT id,title,status,error,updated_at,completed_at FROM local_media_tasks "
                    "WHERE qb_hash=? AND ?!='' ORDER BY id DESC LIMIT 20",
                    (row["qb_task_id"], row["qb_task_id"] or ""),
                )
            ]
            # 小列表已完整；达到展示上限才额外汇总，避免常见单任务路径多查一次。
            result["local_task_status_counts"] = (
                {
                    str(item["status"] or ""): int(item["total"])
                    for item in conn.execute(
                        "SELECT status,COUNT(*) AS total FROM local_media_tasks "
                        "WHERE qb_hash=? AND ?!='' GROUP BY status",
                        (row["qb_task_id"], row["qb_task_id"] or ""),
                    )
                }
                if len(result["local_tasks"]) == 20
                else {}
            )

        elif kind == "organize":
            result["steps"] = [
                dict(item)
                for item in conn.execute(
                    "SELECT * FROM organize_operation_steps WHERE log_id=? ORDER BY id DESC LIMIT 100",
                    (identifier,),
                )
            ]
        else:
            result["item_status_counts"] = {
                str(item["status"] or ""): int(item["total"])
                for item in conn.execute(
                    "SELECT status,COUNT(*) AS total FROM local_media_task_items "
                    "WHERE task_id=? GROUP BY status",
                    (identifier,),
                )
            }
    return result
