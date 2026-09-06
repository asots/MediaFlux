"""下载隔离目录的只读定位与身份条件更新，不改变全局活动目录保护规则。"""
from __future__ import annotations

import json
import re
from contextlib import nullcontext

from app import database as db

# 清理只读业务身份，避免列表查询载入磁力、种子内容和通知凭据。
_IDENTITY_KEYS = (
    'gy_target_dir', 'gy_staging_parent_dir', 'gy_staging_name',
    'organize_task_id', 'gy_task_id', 'gy_task_ids', 'targets',
)
_COLUMNS = (
    'id,status,targets,gy_status,gy_isolated,gy_target_dir,gy_target_name,'
    'gy_staging_parent_dir,gy_staging_name,gy_staging_cleanup_status,'
    'gy_staging_cleanup_error,organize_started,organize_status,organize_task_id,'
    'gy_task_id,gy_task_ids,attention_cleared_at'
)


def list_staging_cleanup_requests(*, source_id: str = '', limit: int = 1000, conn=None) -> list[dict]:
    """读取有界隔离请求；超限让调用方停止，而不是静默漏掉保护对象。"""
    where = "gy_isolated=1 AND TRIM(COALESCE(gy_target_dir,'')) NOT IN ('','0')"
    params: list[object] = []
    if source_id:
        where += ' AND gy_target_dir=?'
        params.append(str(source_id))
    else:
        where += " AND COALESCE(gy_staging_cleanup_status,'')!='completed'"
    params.append(max(1, min(int(limit), 1000)) + 1)
    with (nullcontext(conn) if conn is not None else db.get_conn()) as connection:
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='download_requests'").fetchone() is None:
            return []  # 与全局staging保护查询保持初始化前的兼容行为。
        rows = connection.execute(
            f'SELECT {_COLUMNS} FROM download_requests WHERE {where} ORDER BY id LIMIT ?', params
        ).fetchall()
    if len(rows) > int(params[-1]) - 1:
        raise RuntimeError('待复核下载目录超过安全上限，请缩小清理范围')
    return [dict(row) for row in rows]


def _unsafe_confirmation_stats(encoded: str | None) -> bool:
    try:
        stats = json.loads(encoded or '{}')
        return (
            not isinstance(stats, dict)
            or stats.get('scan_complete') is False
            or any(stats.get(key) for key in (
                'failed', 'need_confirm', 'stopped', 'scan_errors', 'scan_limited', 'audit_failures',
                'replacement_cleanup_failed', 'empty_dir_cleanup_failed', 'source_dir_cleanup_failed',
            ))
        )
    except (TypeError, ValueError):
        return True


def unresolved_staging_confirmations(source_id: str) -> int:
    """完成卡仍含失败也阻止删除；旧卡按source ID匹配，同指纹只看最新。"""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT c.status,c.result_json FROM organize_confirmations c WHERE "
            "NOT EXISTS (SELECT 1 FROM organize_confirmations newer "
            "WHERE newer.fingerprint=c.fingerprint AND newer.id>c.id) AND "
            "CASE WHEN json_valid(c.payload_json) THEN "
            "CAST(COALESCE(json_extract(c.payload_json,'$.source_dir_id'),'') AS TEXT)=? "
            "ELSE c.status NOT IN ('completed','cancelled','expired') END", (str(source_id),),
        ).fetchall()
    return sum(
        _unsafe_confirmation_stats(row['result_json']) if row['status'] == 'completed'
        else True
        for row in rows
    )


def is_current_staging_confirmation(confirmation_id: int, source_id: str, *, conn=None) -> bool:
    """持久收尾授权只能使用原卡；新成功卡也不能替代旧意图的授权身份。"""
    with (nullcontext(conn) if conn is not None else db.get_conn()) as connection:
        row = connection.execute(
            "SELECT c.status,c.result_json FROM organize_confirmations c WHERE c.id=? "
            "AND CASE WHEN json_valid(c.payload_json) THEN "
            "CAST(json_extract(c.payload_json,'$.source_dir_id') AS TEXT)=? ELSE 0 END "
            "AND NOT EXISTS (SELECT 1 FROM organize_confirmations newer "
            "WHERE newer.fingerprint=c.fingerprint AND newer.id>c.id)", (int(confirmation_id), str(source_id)),
        ).fetchone()
    return row is not None and row['status'] == 'completed' and not _unsafe_confirmation_stats(row['result_json'])


def staging_identity_snapshot(row: dict) -> dict:
    """只持久化清理授权需要的原始身份，不带下载凭据或用户私密载荷。"""
    return {key: row.get(key) for key in ('id', 'gy_isolated', *_IDENTITY_KEYS)}


def same_staging_identity(current: dict, expected: dict) -> bool:
    return (
        int(current.get('id') or 0) == int(expected.get('id') or 0)
        and int(current.get('gy_isolated') or 0) == 1
        and all(str(current.get(key) or '') == str(expected.get(key) or '') for key in _IDENTITY_KEYS)
    )


def policy_skip_has_evidence(expected: dict, *, conn) -> bool:
    """策略保留白名单：无失败初态，或旧告警中的每个文件都有完整成功确认。"""
    error = str(expected.get('gy_staging_cleanup_error') or '').strip()
    existing_policy = expected.get('gy_staging_cleanup_status') == 'skipped' and error.startswith(
        '按策略保留隔离目录（clean_empty 已关闭）',
    )
    legacy = re.fullmatch(r'隔离目录仍有 ([1-9][0-9]*) 项未整理或未识别：(.+)', error)
    if error and not existing_policy and legacy is None:
        return False
    names: list[str] = []
    if legacy:
        count = int(legacy.group(1))
        names = [legacy.group(2)] if count == 1 else legacy.group(2).split('、')
        # 旧代码超过五项会截断预览；不能凭不完整列表解除告警。
        if count > 5 or len(names) != count or len(set(names)) != count:
            return False
    confirmations = conn.execute(
        "SELECT c.payload_json,c.result_json FROM organize_confirmations c WHERE c.status='completed' "
        "AND CASE WHEN json_valid(c.payload_json) THEN "
        "CAST(json_extract(c.payload_json,'$.source_dir_id') AS TEXT)=? ELSE 0 END "
        "AND NOT EXISTS (SELECT 1 FROM organize_confirmations newer "
        "WHERE newer.fingerprint=c.fingerprint AND newer.id>c.id)", (str(expected.get('gy_target_dir') or ''),),
    ).fetchall()
    confirmed_names: set[str] = set()
    has_success = False
    for confirmation in confirmations:
        if _unsafe_confirmation_stats(confirmation['result_json']):
            continue
        try:
            payload = json.loads(confirmation['payload_json'])
            result = json.loads(confirmation['result_json'] or '{}')
            files = payload.get('files')
            moved = result.get('moved')
            if type(moved) is not int or moved <= 0:
                continue
            has_success = True
            if not isinstance(files, list) or not files or moved < len(files):
                continue
            confirmed_names.update(str(item['name']) for item in files
                                   if isinstance(item, dict) and item.get('file_id') and item.get('name'))
        except (TypeError, ValueError, AttributeError):
            continue
    return has_success and set(names) <= confirmed_names


def update_staging_cleanup(expected: dict, *, status: str, error: str = '') -> bool:
    """CAS保留原隔离身份/后端绑定与清理阶段，旧回调不能改写重绑定后的请求。"""
    if status not in ('retained', 'failed', 'completed', 'skipped'):
        raise ValueError('非法隔离目录清理状态')
    where = ' AND '.join(f"COALESCE({key},'')=?" for key in _IDENTITY_KEYS)
    with db.get_conn() as conn:
        if status == 'skipped':
            if (expected.get('gy_staging_cleanup_status') not in ('', None, 'pending', 'retained', 'skipped')
                    or expected.get('organize_status') != 'completed'):
                return False
            where += " AND organize_status='completed'"
            # 策略跳过不等于隐藏：必须在同一事务内确认成功证据与旧告警仍有效。
            conn.execute('BEGIN IMMEDIATE')
            if expected.get('_confirmation_id') and not is_current_staging_confirmation(
                expected['_confirmation_id'], str(expected.get('gy_target_dir') or ''), conn=conn,
            ):
                return False
            summary = staging_confirmation_summary(str(expected.get('gy_target_dir') or ''), conn=conn)
            if not summary['total'] or summary['completed'] != summary['total']:
                return False
            if not policy_skip_has_evidence(expected, conn=conn):
                return False
            owners = list_staging_cleanup_requests(source_id=str(expected.get('gy_target_dir') or ''), conn=conn)
            if len(owners) != 1 or not same_staging_identity(owners[0], expected):
                return False
            current = conn.execute(
                'SELECT gy_staging_cleanup_error FROM download_requests WHERE id=?', (int(expected['id']),),
            ).fetchone()
            if current is None or str(current['gy_staging_cleanup_error'] or '') != str(expected.get('gy_staging_cleanup_error') or ''):
                return False
            unknown = conn.execute(
                "SELECT 1 FROM organize_delete_audit WHERE trigger='download_staging_cleanup' "
                "AND status IN ('pending','failed') AND (file_id=? OR parent_id=?) LIMIT 1",
                (str(expected.get('gy_target_dir') or ''), str(expected.get('gy_target_dir') or '')),
            ).fetchone()
            if unknown:
                return False
        cur = conn.execute(
            'UPDATE download_requests SET gy_staging_cleanup_status=?,gy_staging_cleanup_error=?,updated_at=? '
            "WHERE id=? AND gy_isolated=1 AND gy_status='completed' "
            "AND status NOT IN ('cancelled','resubmitted','failed') "
            "AND COALESCE(organize_status,'') NOT IN ('resubmitted','cleared','failed','stopped') "
            "AND COALESCE(attention_cleared_at,'')='' "
            "AND COALESCE(gy_staging_cleanup_status,'')=? AND " + where,
            [status, str(error or '')[:500], db.now(), int(expected['id']),
             str(expected.get('gy_staging_cleanup_status') or ''),
             *(str(expected.get(key) or '') for key in _IDENTITY_KEYS)],
        )
    return cur.rowcount == 1


def staging_confirmation_summary(source_id: str, *, conn=None) -> dict[str, int]:
    """同指纹只看最新确认，历史失败/被替代卡不能永远冻结已完成请求。"""
    with (nullcontext(conn) if conn is not None else db.get_conn()) as connection:
        rows = connection.execute(
            "SELECT c.status,c.result_json FROM organize_confirmations c "
            "WHERE json_valid(c.payload_json) AND CAST(json_extract(c.payload_json,'$.source_dir_id') AS TEXT)=? "
            "AND NOT EXISTS (SELECT 1 FROM organize_confirmations newer "
            "WHERE newer.fingerprint=c.fingerprint AND newer.id>c.id)", (str(source_id),),
        ).fetchall()
    result = {'total': len(rows), 'completed': 0, 'pending': 0, 'failed': 0}
    for row in rows:
        if row['status'] == 'completed':
            result['failed' if _unsafe_confirmation_stats(row['result_json']) else 'completed'] += 1
        elif row['status'] in ('pending', 'queued', 'running'):
            result['pending'] += 1
        else:
            # 取消、过期或失败不等于文件已成功入库，不冒报完成。
            result['failed'] += 1
    return result


def requests_for_download_confirmation(payload: dict, *, strict: bool = False, conn=None) -> list[dict]:
    """新卡按业务关联校验，旧卡只接受唯一且精确绑定的隔离source ID。"""
    source_id = str(payload.get('source_dir_id') or '').strip()
    if source_id in ('', '0'):
        if strict and 'download_request_ids' in payload:
            raise ValueError('原下载请求的来源关联无效')
        return []
    owners = list_staging_cleanup_requests(source_id=source_id, conn=conn)
    if len(owners) != 1:
        if strict and (owners or 'download_request_ids' in payload):
            raise ValueError('原下载请求关联缺失或不唯一，请重新核对整理任务')
        return []
    row = owners[0]
    if 'download_request_ids' in payload:
        ids = payload['download_request_ids']
        if (
            not isinstance(ids, list) or not ids or len(ids) > 100
            or any(type(value) is not int or value <= 0 for value in ids)
            or int(row['id']) not in ids
        ):
            if strict:
                raise ValueError('原下载请求关联已变化，请重新核对整理任务')
            return []
    task_id = str(payload.get('organize_task_id') or '').strip()
    if task_id and task_id != str(row.get('organize_task_id') or ''):
        if strict:
            raise ValueError('原下载整理任务关联已变化，不能降级为普通整理')
        return []
    return [row]


def complete_staging_confirmation_phase(expected: dict) -> bool:
    """所有最新确认成功后，条件更新原下载整理阶段；不覆盖其它失败/新任务。"""
    where = ' AND '.join(f"COALESCE({key},'')=?" for key in _IDENTITY_KEYS)
    with db.get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        summary = staging_confirmation_summary(str(expected.get('gy_target_dir') or ''), conn=conn)
        if not summary['total'] or summary['completed'] != summary['total']:
            return False
        cur = conn.execute(
            "UPDATE download_requests SET organize_status='completed',organize_error='',"
            "organize_finished_at=?,updated_at=? WHERE id=? AND gy_isolated=1 AND gy_status='completed' "
            "AND status NOT IN ('cancelled','resubmitted','failed') "
            "AND organize_status IN ('completed','requires_manual') "
            "AND COALESCE(attention_cleared_at,'')='' AND " + where,
            [db.now(), db.now(), int(expected['id']),
             *(str(expected.get(key) or '') for key in _IDENTITY_KEYS)],
        )
    return cur.rowcount == 1


def mark_staging_waiting_confirmation(request_id: int, task_id: str, count: int) -> bool:
    with db.get_conn() as conn:
        cur = conn.execute(
            "UPDATE download_requests SET organize_status='requires_manual',organize_error=?,"
            "organize_finished_at=NULL,updated_at=? WHERE id=? AND organize_task_id=? "
            "AND gy_status='completed' AND status NOT IN ('cancelled','resubmitted','failed')",
            (f'仍有 {max(1, int(count))} 项媒体等待人工确认', db.now(), int(request_id), str(task_id)),
        )
    return cur.rowcount == 1
