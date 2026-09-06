"""首次提交和补目标必须复用同一执行核，同时保持各自公开合同。"""
from __future__ import annotations

import ast
import inspect
import unittest
from unittest.mock import patch

from app import database as db
from app.modules import download_dispatcher as dispatcher
from tests.support import isolated_test_database


class DownloadDispatchSingleTrackTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(isolated_test_database())
        self.serial = 0

    def request(self):
        self.serial += 1
        request_id, _ = db.create_download_request('single-track-'+str(self.serial), 'magnet', title='统一分发', source_value='magnet:?xt=urn:btih:'+'a'*40)
        return request_id

    def test_entrypoints_only_claim_then_delegate_and_never_submit_directly(self):
        for function in (dispatcher.dispatch_request, dispatcher.dispatch_missing_targets):
            with self.subTest(entry=function.__name__):
                tree = ast.parse(inspect.getsource(function))
                calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
                self.assertEqual(calls.count('_dispatch_claimed_targets'), 1)
                self.assertNotIn('_safe_submit', calls)
                self.assertNotIn('_submit_qb', calls)
                self.assertNotIn('_submit_guangya', calls)

    def test_both_entrypoints_share_result_and_staging_projection(self):
        cases = (
            {'ok': True, 'task_id': 'gy-1', 'task_ids': ['gy-1', 'gy-2'], 'selected_count': 2,
             'decision': {'target_dir_id': 'stage', 'target_dir_name': '测试'},
             'staging': {'isolated': True, 'parent_id': 'root', 'name': 'stage'}},
            {'ok': False, 'partial_success': True, 'task_ids': ['gy-1'], 'error': '部分提交'},
            {'ok': False, 'error': '明确失败'},
        )
        for result in cases:
            with self.subTest(result=result):
                rows = []
                for additional in (False, True):
                    request_id = self.request()
                    if additional:
                        db.update_download_request(request_id, targets='qb', status='downloading', qb_status='completed', qb_task_id='kept-qb')
                    operation = dispatcher.dispatch_missing_targets if additional else dispatcher.dispatch_request
                    with patch.object(dispatcher, '_submit_guangya', return_value=result) as gy, patch.object(dispatcher, '_submit_qb') as qb:
                        outcome = operation(request_id, 'guangya', gy_target_dir='selected', gy_target_name='selected-name', log_path='[magnet]')
                    gy.assert_called_once()
                    qb.assert_not_called()
                    row = db.get_download_request(request_id)
                    # 用真实仓储业务结果补足旧 Mock-only 的 status 参数断言。
                    expected_status = (
                        'submitted' if result.get('ok') or result.get('partial_success')
                        else 'completed' if additional else 'failed'
                    )
                    self.assertEqual(row['status'], expected_status)
                    self.assertEqual(outcome['status'], expected_status)
                    rows.append(row)
                    self.assertEqual(outcome['results'], {'guangya': result})
                    if additional:
                        self.assertTrue(outcome['handled'])
                        self.assertFalse(outcome['duplicate'])
                        self.assertEqual((row['qb_status'], row['qb_task_id']), ('completed', 'kept-qb'))
                    else:
                        self.assertNotIn('handled', outcome)
                    logs = db.list_download_logs()
                    own = [log for log in logs if log['request_id'] == request_id]
                    self.assertEqual(len(own), 1)
                    self.assertEqual(own[0]['path'], '[magnet]')
                for field in ('gy_status','gy_task_id','gy_task_ids','gy_batch_count','gy_target_dir','gy_isolated','gy_staging_parent_dir','gy_expected_file_count'):
                    self.assertEqual(rows[0][field], rows[1][field], field)

    def test_duplicate_and_restart_do_not_repeat_backend(self):
        request_id = self.request()
        with patch.object(dispatcher, '_submit_qb', return_value={'ok': True, 'task_id': 'a'*40}) as backend:
            first = dispatcher.dispatch_request(request_id, 'qb')
            self.assertTrue(first['ok'])
            db.init_db()
            second = dispatcher.dispatch_request(request_id, 'qb')
            third = dispatcher.dispatch_missing_targets(request_id, 'qb')
        backend.assert_called_once()
        self.assertTrue(second['duplicate'])
        self.assertTrue(third['duplicate'])
        self.assertFalse(third['handled'])

    def test_late_completion_does_not_overwrite_recovered_state_in_either_entry(self):
        for additional in (False, True):
            with self.subTest(additional=additional):
                request_id = self.request()
                if additional:
                    db.update_download_request(request_id, targets='qb', status='downloading', qb_status='completed')
                def submit(_row, **_kwargs):
                    db.update_download_request(request_id, status='manual_review', gy_status='manual_review')
                    return {'ok': True, 'task_id': 'external-task'}
                operation = dispatcher.dispatch_missing_targets if additional else dispatcher.dispatch_request
                with patch.object(dispatcher, '_submit_guangya', side_effect=submit):
                    result = operation(request_id, 'guangya')
                self.assertTrue(result['stale_result'])
                self.assertTrue(result['outcome_unknown'])
                self.assertEqual(db.get_download_request(request_id)['gy_status'], 'manual_review')

    def test_same_request_can_add_only_the_missing_backend(self):
        request_id = self.request()
        with patch.object(dispatcher, '_submit_qb', return_value={'ok': True, 'task_id': 'a'*40}) as qb, patch.object(dispatcher, '_submit_guangya', return_value={'ok': True, 'task_id': 'gy-1'}) as gy:
            dispatcher.dispatch_request(request_id, 'qb')
            result = dispatcher.dispatch_missing_targets(request_id, 'both')
        qb.assert_called_once()
        gy.assert_called_once()
        self.assertEqual(result['succeeded'], ['guangya'])
        self.assertEqual(set(result['results']), {'guangya'})
        row = db.get_download_request(request_id)
        self.assertEqual((row['targets'], row['qb_status'], row['gy_status']), ('both','submitted','submitted'))

    def test_completed_request_requires_successor_instead_of_reopening_in_place(self):
        request_id = self.request()
        db.update_download_request(request_id, status='completed', targets='qb', qb_status='completed')
        with patch.object(dispatcher, '_submit_guangya') as backend:
            result = dispatcher.dispatch_missing_targets(request_id, 'guangya')
        backend.assert_not_called()
        self.assertFalse(result['handled'])
        self.assertTrue(result['duplicate'])
        self.assertEqual(db.get_download_request(request_id)['status'], 'completed')
