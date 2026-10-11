"""重整真实成功后解除旧确认；所有目录删除均运行于 fake 云盘。"""
import json
from unittest.mock import patch

from app import database as db
from app.modules import download_staging_reconcile as worker
from app.modules.organize import OrganizeRules
from app.modules.process_lock import CrossProcessLock
from app.repositories import download_staging_reconcile as queue
from tests.support import IsolatedDatabaseTestCase
from tests.test_download_staging_cleanup_lifecycle import _CloudTree


class ReorganizedDownloadReconciliationTests(IsolatedDatabaseTestCase):
    def setUp(self):
        with db.get_conn() as conn:
            for table in ('download_staging_reconcile', 'download_request_keys', 'download_requests',
                          'organize_confirmations', 'organize_log_items', 'organize_log', 'organize_delete_audit'):
                conn.execute(f'DELETE FROM {table}')
        db.kv_set(queue.LEGACY_CURSOR_KEY, '0')
        self.request_id, _ = db.create_download_request('reorganized', 'magnet')
        db.update_download_request(
            self.request_id, targets='guangya', status='completed', gy_status='completed', gy_isolated=1,
            gy_target_dir='stage', gy_staging_parent_dir='source', gy_staging_name='MF-case',
            gy_staging_cleanup_status='retained', gy_staging_cleanup_error='仍有待确认事项',
            organize_started=1, organize_status='requires_manual', organize_error='仍有1项媒体等待确认',
            organize_task_id='task', gy_task_id='cloud', gy_task_ids='["cloud"]',
        )
        self.payload = {
            'source_dir_id': 'stage', 'source_parent_id': 'stage', 'organize_task_id': 'task',
            'download_request_ids': [self.request_id],
            'files': [{'file_id': 'video', 'parent_id': 'stage', 'size': 100, 'etag': 'hash'}],
        }
        self.confirm('card', 'failed')
        self.log_id = self.evidence('video')

    def confirm(self, token, status, *, fingerprint='fingerprint'):
        db.create_organize_confirmation(
            token=token, fingerprint=fingerprint, chat_id='test', source_name='stage', directory_path='/',
            payload=self.payload, expires_at='2099-01-01 00:00:00',
        )
        db.update_organize_confirmation(token, status=status, error='旧确认失败')

    def evidence(self, file_id):
        log_id = db.add_organize_log(
            'guangya', '/stage/video.mkv', '/target/video.mkv', file_id, 'success',
            operation_type='reorganize', source_dir_id='stage', original_parent_id='stage',
            original_name='video.mkv', current_parent_id='target', current_name='video.mkv',
            target_parent_id='target',
        )
        db.add_organize_log_items(log_id, [{
            'file_id': file_id, 'role': 'video', 'original_parent_id': 'stage', 'original_name': 'video.mkv',
            'current_parent_id': 'target', 'current_name': 'video.mkv',
            'target_parent_id': 'target', 'target_name': 'video.mkv', 'size': 100, 'etag': 'hash',
        }])
        return log_id

    def drain(self, cloud):
        lock = CrossProcessLock('guangya-organize', directory=self.test_db_path.parent)
        self.assertTrue(lock.acquire(blocking=False))
        try:
            with patch('app.modules.organize_tasks.config.get', side_effect=lambda key, default='': default):
                return worker.reconcile_with_client(
                    cloud, rules=OrganizeRules(target_dir_id='target', clean_empty=True, link_strm=False),
                    writer_lock=lock,
                )
        finally:
            lock.release()

    def test_successful_reorganize_closes_card_and_request_then_cleans_empty_root(self):
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), ['card'])
        card = db.get_organize_confirmation('card')
        stats = json.loads(card['result_json'])
        self.assertEqual(card['status'], 'completed')
        self.assertEqual(stats['previous_confirmation']['status'], 'failed')
        self.assertEqual(stats['organize_log_ids'], [self.log_id])
        cloud = _CloudTree()
        self.drain(cloud)
        request = db.get_download_request(self.request_id)
        self.assertEqual(request['organize_status'], 'completed')
        self.assertEqual(request['organize_error'], '')
        self.assertEqual(request['gy_staging_cleanup_status'], 'completed')
        self.assertEqual(db.count_download_requests_requiring_attention(), 0)
        self.assertEqual(cloud.deleted, ['stage'])
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), [])
        self.drain(cloud)
        self.assertEqual(cloud.deleted, ['stage'])

    def test_successful_manual_organize_also_supersedes_old_expired_card(self):
        db.update_organize_log(self.log_id, operation_type='organize')
        db.update_organize_confirmation('card', status='expired')
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), ['card'])

    def test_newer_failed_log_prevents_reusing_old_success_evidence(self):
        newer = self.evidence('video')
        db.update_organize_log(newer, status='failed')
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), [])

    def test_newer_failed_primary_without_items_cannot_fall_back_to_old_success(self):
        db.add_organize_log('guangya', '/stage/video.mkv', '', 'video', 'failed', source_dir_id='stage')
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), [])

    def test_incomplete_audit_is_not_success_evidence(self):
        db.update_organize_log(self.log_id, legacy_incomplete=1)
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), [])

    def test_missing_payload_companion_blocks_resolution(self):
        self.payload['companions'] = [{'file_id': 'subtitle', 'parent_id': 'stage', 'size': 100, 'etag': 'hash'}]
        with db.get_conn() as conn:
            conn.execute("UPDATE organize_confirmations SET payload_json=? WHERE token='card'", (json.dumps(self.payload),))
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), [])
        db.add_organize_log_items(self.log_id, [{
            'file_id': 'subtitle', 'role': 'subtitle', 'original_parent_id': 'stage',
            'current_parent_id': 'target', 'current_name': 'video.srt',
            'target_parent_id': 'target', 'target_name': 'video.srt', 'size': 100, 'etag': 'hash',
        }])
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), ['card'])

    def test_newer_failed_companion_without_details_blocks_old_group_success(self):
        self.payload['companions'] = [{'file_id': 'subtitle', 'parent_id': 'stage'}]
        with db.get_conn() as conn:
            conn.execute("UPDATE organize_confirmations SET payload_json=? WHERE token='card'", (json.dumps(self.payload),))
        db.add_organize_log_items(self.log_id, [{
            'file_id': 'subtitle', 'role': 'subtitle', 'original_parent_id': 'stage',
            'current_parent_id': 'target', 'current_name': 'video.srt',
            'target_parent_id': 'target', 'target_name': 'video.srt',
        }])
        db.add_organize_log('guangya', '/stage/video.srt', '', 'subtitle', 'failed', source_dir_id='stage')
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), [])

    def test_newer_failed_companion_under_other_video_log_blocks_old_group(self):
        self.payload['companions'] = [{'file_id': 'subtitle', 'parent_id': 'stage'}]
        with db.get_conn() as conn:
            conn.execute("UPDATE organize_confirmations SET payload_json=? WHERE token='card'", (json.dumps(self.payload),))
        item = {
            'file_id': 'subtitle', 'role': 'subtitle', 'original_parent_id': 'stage',
            'current_parent_id': 'target', 'current_name': 'video.srt',
            'target_parent_id': 'target', 'target_name': 'video.srt',
        }
        db.add_organize_log_items(self.log_id, [item])
        newer = self.evidence('other-video')
        db.add_organize_log_items(newer, [{**item, 'status': 'failed'}])
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), [])

    def test_history_discovery_repairs_old_reorganized_records_without_redownload(self):
        self.assertEqual(queue.discover_legacy_confirmation_cleanup(), 1)
        self.assertEqual(db.get_organize_confirmation('card')['status'], 'completed')
        self.assertEqual(len(queue.list_due_cleanup_intents()), 1)

    def test_only_latest_card_is_closed_and_terminal_history_is_preserved(self):
        self.confirm('retry', 'pending')
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), ['retry'])
        self.assertEqual(db.get_organize_confirmation('card')['status'], 'failed')
        self.assertEqual(db.get_organize_confirmation('retry')['status'], 'completed')

    def test_expired_and_cancelled_cards_can_be_resolved_but_active_execution_cannot(self):
        for status in ('expired', 'cancelled', 'queued', 'running'):
            with self.subTest(status=status):
                db.update_organize_confirmation('card', status=status)
                tokens = queue.reconcile_reorganized_confirmations(['stage'])
                self.assertEqual(tokens, ['card'] if status in ('expired', 'cancelled') else [])

    def test_all_files_in_confirmation_must_have_successful_evidence(self):
        self.payload['files'].append({'file_id': 'second', 'parent_id': 'stage', 'size': 100, 'etag': 'hash'})
        with db.get_conn() as conn:
            conn.execute("UPDATE organize_confirmations SET payload_json=? WHERE token='card'", (json.dumps(self.payload),))
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), [])
        self.evidence('second')
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), ['card'])

    def test_mismatched_identity_failed_move_and_unmoved_file_never_close_card(self):
        for column, value in (('etag', 'other'), ('size', 999), ('current_parent_id', 'stage'),
                              ('current_name', 'wrong.mkv'), ('status', 'rollback_failed')):
            with self.subTest(column=column):
                with db.get_conn() as conn:
                    old = conn.execute(f'SELECT {column} FROM organize_log_items WHERE log_id=?', (self.log_id,)).fetchone()[0]
                    conn.execute(f'UPDATE organize_log_items SET {column}=? WHERE log_id=?', (value, self.log_id))
                self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), [])
                with db.get_conn() as conn:
                    conn.execute(f'UPDATE organize_log_items SET {column}=? WHERE log_id=?', (old, self.log_id))

    def test_failed_companion_blocks_resolution(self):
        db.add_organize_log_items(self.log_id, [{
            'file_id': 'subtitle', 'role': 'subtitle', 'status': 'failed', 'original_parent_id': 'stage',
        }])
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), [])

    def test_unrelated_task_or_cancelled_download_is_not_resurrected(self):
        db.update_download_request(self.request_id, organize_task_id='new-task')
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), [])
        db.update_download_request(self.request_id, organize_task_id='task', status='cancelled')
        self.assertEqual(queue.reconcile_reorganized_confirmations(['stage']), [])

    def test_confirmation_and_cleanup_intent_roll_back_together_on_queue_failure(self):
        import sqlite3
        with db.get_conn() as conn:
            conn.execute(
                "CREATE TRIGGER reject_reorganized_queue BEFORE INSERT ON download_staging_reconcile "
                "BEGIN SELECT RAISE(ABORT,'queue-failure'); END"
            )
        try:
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'queue-failure'):
                queue.reconcile_reorganized_confirmations(['stage'])
            self.assertEqual(db.get_organize_confirmation('card')['status'], 'failed')
            self.assertEqual(queue.list_due_cleanup_intents(), [])
        finally:
            with db.get_conn() as conn:
                conn.execute('DROP TRIGGER reject_reorganized_queue')

    def test_original_cleanup_disabled_policy_is_preserved(self):
        self.payload['rules'] = {'clean_empty': False}
        with db.get_conn() as conn:
            conn.execute("UPDATE organize_confirmations SET payload_json=? WHERE token='card'", (json.dumps(self.payload),))
        queue.reconcile_reorganized_confirmations(['stage'])
        cloud = _CloudTree()
        self.drain(cloud)
        self.assertEqual(db.get_download_request(self.request_id)['organize_status'], 'completed')
        self.assertEqual(cloud.deleted, [])

    def test_remaining_file_keeps_attention_even_after_old_confirmation_is_resolved(self):
        queue.reconcile_reorganized_confirmations(['stage'])
        cloud = _CloudTree()
        cloud.add_file('unrelated.mkv', 'stage')
        self.drain(cloud)
        self.assertEqual(db.get_download_request(self.request_id)['organize_status'], 'completed')
        self.assertEqual(db.get_download_request(self.request_id)['gy_staging_cleanup_status'], 'retained')
        self.assertEqual(db.count_download_requests_requiring_attention(), 1)
        self.assertEqual(cloud.deleted, [])
