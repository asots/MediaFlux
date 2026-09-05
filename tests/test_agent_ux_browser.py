"""Agent体验的真实浏览器回归：等待、草稿、稳定位置与安全候选交互。"""
from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

from tests import test_agent_kernel_browser as harness

SCOPE = 'a' * 64
SESSION_A = 'session_agent_ux_000000001'
SESSION_B = 'session_agent_ux_000000002'


def candidate_view(*, expires_at: float | None = None) -> dict:
    return {
        'ref': 'ref_resource_candidates_0000001',
        'expires_at': expires_at if expires_at is not None else time.time() + 900,
        'items': [
            {'position': 1, 'title': '第一集 1080p <img src=x onerror=window.__ux_xss=1>',
             'site_name': '资源站 A', 'size_text': '1.2 GB', 'tags': {'resolution': '1080p'},
             'reasons': ['明确匹配 S01E01'], 'warnings': ['字幕需要人工核对'],
             'selection': {'ref': 'ref_selection_candidate_0000001', 'position': 1}},
            {'position': 2, 'title': '第一集 2160p', 'site_name': '资源站 B',
             'tags': {'resolution': '2160p', 'audio': 'AAC'}, 'reasons': ['存在备选版本'],
             'warnings': [], 'selection': {'ref': 'ref_selection_candidate_0000002', 'position': 2}},
        ],
    }


def events_for_candidates(view: dict) -> list[dict]:
    return [
        harness._event(1, 'turn.started'),
        harness._event(2, 'tool.completed', {'tool': 'resource.search', 'call_id': 'search1', 'result': {'candidate_view': view}}),
        harness._event(3, 'turn.completed', {'status': 'success', 'answer': '已找到候选，请先核对版本。'}),
    ]


@unittest.skipIf(harness.sync_playwright is None, '系统环境未安装 Playwright')
class AgentUXBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        harness.AgentKernelBrowserTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        harness.AgentKernelBrowserTests.tearDownClass.__func__(cls)

    make_page = harness.AgentKernelBrowserTests.make_page

    def page(self, config: dict | None = None, **kwargs):
        value = dict(config or {})
        value.setdefault('sessions', {'sessions': [], 'draft_scope': SCOPE})
        page = self.make_page(value, **kwargs)
        page.wait_for_function("() => document.querySelector('#agentSessionList').getAttribute('aria-busy') === 'false'")
        return page

    def reload_ui(self, page, config: dict):
        page.reload()
        page.add_style_tag(content=self.styles)
        page.evaluate(harness.MOCK_FETCH, config)
        page.add_script_tag(content=self.source)
        page.wait_for_function("() => document.querySelector('#agentSessionList').getAttribute('aria-busy') === 'false'")

    def snapshot(self, page, name: str):
        output = os.getenv('MEDIAFLUX_BROWSER_EVIDENCE_DIR')
        if output:
            root = Path(output)
            root.mkdir(parents=True, exist_ok=True)
            page.add_script_tag(path=str(harness.ROOT / 'app/static/js/lucide.min.js'))
            page.evaluate('() => window.lucide?.createIcons()')
            page.evaluate("""async () => { await Promise.all(document.getAnimations().filter(animation => animation.effect?.getTiming().iterations !== Infinity).map(animation => animation.finished.catch(() => {}))); }""")
            page.screenshot(path=str(root / f'{name}.png'))

    def test_next_actions_are_bounded_readonly_and_do_not_move_composer(self):
        actions = [{'id': f'item-{i}', 'title': f'检查待办 {i}', 'description': '只读本地快照', 'prompt': f'查看待办 {i}'} for i in range(5)]
        for viewport in ({'width': 1280, 'height': 800}, {'width': 390, 'height': 844}):
            with self.subTest(viewport=viewport):
                page = self.page({'nextActions': {'actions': actions, 'snapshot_status': 'attention'}, 'nextActionsDelayMs': 400}, viewport=viewport)
                before = page.locator('#agentComposer').bounding_box()
                page.wait_for_function("() => document.querySelectorAll('[data-agent-draft]').length === 3")
                after = page.locator('#agentComposer').bounding_box()
                self.assertLessEqual(abs(before['y'] - after['y']), 0.5)
                self.assertLessEqual(abs(before['height'] - after['height']), 0.5)
                console = page.locator('.agent-workbench').bounding_box()
                self.assertLessEqual(abs(after['y'] + after['height'] / 2 - console['y'] - console['height'] / 2), 0.5)
                self.assertTrue(page.locator('#agentStartActions').get_attribute('aria-busy') == 'false')
                page.locator('[data-agent-draft]').first.click()
                self.assertEqual(page.locator('#agentPrompt').input_value(), '查看待办 0')
                self.assertEqual(page.evaluate("window.__kernelCalls.filter(call => call.url === '/api/agent/query').length"), 0)
                self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'), viewport['width'])
                targets = page.locator('.agent-start-action').evaluate_all('(items) => items.map(item => item.getBoundingClientRect().height)')
                self.assertTrue(all(height >= 44 for height in targets))
                self.snapshot(page, f'start-{viewport["width"]}')

    def test_empty_resume_has_its_own_slot_and_history_focus_matches_rounded_style(self):
        for viewport in ({'width': 1280, 'height': 800}, {'width': 390, 'height': 844}, {'width': 390, 'height': 430}):
            with self.subTest(viewport=viewport):
                page = self.page({'sessions': {'draft_scope': SCOPE, 'sessions': [
                    {'session_id': SESSION_A, 'title': '媒体库排障', 'updated_at': 20, 'message_count': 16},
                ]}, 'sessionDetails': {SESSION_A: {'messages': []}}}, viewport=viewport)
                resume = page.locator('#agentStartResume #agentResumeLatestSession')
                page.wait_for_function("() => !document.querySelector('#agentResumeLatestSession').disabled")
                resume_box = resume.bounding_box()
                status_box = page.locator('#agentStartActionsStatus').bounding_box()
                composer_box = page.locator('#agentComposer').bounding_box()
                self.assertIsNotNone(resume_box)
                console = page.locator('.agent-workbench').bounding_box()
                self.assertLessEqual(abs(composer_box['y'] + composer_box['height'] / 2 - console['y'] - console['height'] / 2), 0.5)
                self.assertGreaterEqual(resume_box['y'], composer_box['y'] + composer_box['height'])
                self.assertLessEqual(resume_box['y'] + resume_box['height'] + 4, status_box['y'])
                self.assertLessEqual(composer_box['y'] + composer_box['height'], viewport['height'])
                self.snapshot(page, f'feedback-empty-with-history-{viewport["width"]}-{viewport["height"]}')
                page.locator('#toggleAgentRail').click()
                self.assertEqual(page.locator('#agentSessionCount').inner_text(), '1 条')
                self.assertEqual(page.locator('#agent-session-heading').inner_text(), '历史会话')
                self.assertEqual(page.locator('.agent-history-drawer .agent-kicker').count(), 0)
                self.assertTrue(page.locator('#agent-session-heading').evaluate('(node) => document.activeElement === node'))
                count = page.locator('#agentSessionCount').evaluate("node => {const style = getComputedStyle(node); return {border: style.borderTopWidth, background: style.backgroundColor, font: style.fontFamily};}")
                self.assertEqual(count['border'], '0px')
                self.assertEqual(count['background'], 'rgba(0, 0, 0, 0)')
                self.assertNotIn('mono', count['font'].lower())
                page.locator('#agentSessionSearch').focus()
                self.assertEqual(page.locator('#agentSessionSearch').evaluate('(node) => getComputedStyle(node).outlineStyle'), 'none')
                search_style = page.locator('.agent-session-search').evaluate("node => {const style = getComputedStyle(node); return {radius: style.borderRadius, shadow: style.boxShadow};}")
                self.assertEqual(search_style['radius'], '12px')
                self.assertNotEqual(search_style['shadow'], 'none')
                self.snapshot(page, f'feedback-history-search-{viewport["width"]}-{viewport["height"]}')
                page.locator('[data-session-rename]').click()
                editor = page.locator('.agent-session-editor input')
                style = editor.evaluate("node => {const style = getComputedStyle(node); return {outline: style.outlineStyle, radius: style.borderRadius, border: style.borderTopWidth};}")
                self.assertEqual(style, {'outline': 'none', 'radius': '10px', 'border': '1px'})
                self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'), viewport['width'])
                self.snapshot(page, f'feedback-history-rename-{viewport["width"]}-{viewport["height"]}')

    def test_next_actions_failure_leaves_chat_usable(self):
        page = self.page({'nextActionsStatus': 503})
        page.wait_for_function("() => document.querySelector('#agentStartActionsStatus').textContent.includes('暂时不可用')")
        self.assertTrue(page.locator('#agentPrompt').is_enabled())
        page.locator('#agentPrompt').fill('仍然可以提问')
        self.assertTrue(page.locator('#agentSend').is_enabled())

    def test_busy_composer_keeps_next_draft_without_queuing_or_retrying(self):
        page = self.page({'queryEvents': [harness._event(1, 'turn.started')], 'holdQueryOpen': True})
        page.locator('#agentPrompt').fill('检查下载状态')
        page.locator('#agentSend').click()
        page.wait_for_selector('#agentStop:not([hidden])')
        self.assertTrue(page.locator('#agentPrompt').is_enabled())
        page.locator('#agentPrompt').fill('另外只看失败任务')
        page.locator('#agentPrompt').press('Enter')
        self.assertEqual(page.evaluate("window.__kernelCalls.filter(call => call.url === '/api/agent/query').length"), 1)
        page.locator('#agentStop').click()
        page.wait_for_selector('.agent-retry-draft')
        self.assertEqual(page.locator('#agentPrompt').input_value(), '另外只看失败任务')
        page.locator('.agent-retry-draft').click()
        self.assertEqual(page.locator('#agentPrompt').input_value(), '另外只看失败任务')
        page.locator('#agentPrompt').fill('')
        page.locator('.agent-retry-draft').click()
        self.assertEqual(page.locator('#agentPrompt').input_value(), '检查下载状态')
        self.assertEqual(page.evaluate("window.__kernelCalls.filter(call => call.url === '/api/agent/query').length"), 1)

    def test_drafts_survive_reload_but_are_account_scoped_and_expire(self):
        page = self.page()
        page.locator('#agentPrompt').fill('尚未发送的工作区问题')
        self.reload_ui(page, {'sessions': {'sessions': [], 'draft_scope': SCOPE}})
        self.assertEqual(page.locator('#agentPrompt').input_value(), '尚未发送的工作区问题')
        self.assertEqual(page.evaluate("window.__kernelCalls.filter(call => call.url === '/api/agent/query').length"), 0)
        self.reload_ui(page, {'sessions': {'sessions': [], 'draft_scope': 'b' * 64}})
        self.assertEqual(page.locator('#agentPrompt').input_value(), '')
        page.evaluate("""scope => {
            const key = 'mediaflux.agent.drafts.v1.' + scope;
            const drafts = JSON.parse(sessionStorage.getItem(key));
            for (const draft of Object.values(drafts)) draft.updated_at = Date.now() - 7 * 60 * 60 * 1000;
            sessionStorage.setItem(key, JSON.stringify(drafts));
        }""", SCOPE)
        self.reload_ui(page, {'sessions': {'sessions': [], 'draft_scope': SCOPE}})
        self.assertEqual(page.locator('#agentPrompt').input_value(), '')
        self.assertEqual(page.evaluate("scope => Object.keys(JSON.parse(sessionStorage.getItem('mediaflux.agent.drafts.v1.' + scope))).length", SCOPE), 0)

    def test_live_account_scope_change_does_not_reuse_memory_draft(self):
        page = self.page()
        page.locator('#agentPrompt').fill('旧主体的草稿')
        page.evaluate("() => {window.__kernelConfig.sessions.draft_scope = 'b'.repeat(64);}")
        page.locator('#toggleAgentRail').click()
        page.wait_for_function("() => document.querySelector('#agentPrompt').value === ''")
        stored = page.evaluate("() => sessionStorage.getItem('mediaflux.agent.drafts.v1.' + 'b'.repeat(64))")
        self.assertNotIn('旧主体的草稿', stored or '')

    def test_history_refresh_preserves_focused_rename_editor(self):
        page = self.page({'sessions': {'draft_scope': SCOPE, 'sessions': [
            {'session_id': SESSION_A, 'title': '准备改名', 'updated_at': 20},
            {'session_id': SESSION_B, 'title': '其他会话', 'updated_at': 10},
        ]}})
        page.locator('#toggleAgentRail').click()
        page.locator(f'[data-session-rename="{SESSION_A}"]').click()
        editor = page.locator('.agent-session-editor input')
        editor.fill('还没有保存的新名称')
        page.locator('#agentSessionSearch').fill('准备')
        editor.focus()
        page.locator('#agentSessionSearch').evaluate("node => node.dispatchEvent(new Event('input', {bubbles: true}))")
        self.assertTrue(editor.evaluate('(node) => document.activeElement === node'))
        self.assertEqual(editor.input_value(), '还没有保存的新名称')

    def test_suspected_credentials_are_not_written_to_session_storage(self):
        page = self.page()
        page.locator('#agentPrompt').fill('api_key=fixture-not-a-real-secret')
        stored = page.evaluate("() => Object.keys(sessionStorage).map(key => sessionStorage.getItem(key)).join(' ')")
        self.assertNotIn('fixture-not-a-real-secret', stored)

    def test_drafts_stay_with_each_session_and_delete_clears_only_that_session(self):
        config = {
            'sessions': {'draft_scope': SCOPE, 'sessions': [
                {'session_id': SESSION_A, 'title': 'A 会话', 'updated_at': 20},
                {'session_id': SESSION_B, 'title': 'B 会话', 'updated_at': 10},
            ]},
            'sessionDetails': {SESSION_A: {'messages': []}, SESSION_B: {'messages': []}},
        }
        page = self.page(config, stored_session=SESSION_A)
        page.wait_for_function("id => window.__kernelCalls.some(call => call.url.endsWith(id))", arg=SESSION_A)
        page.locator('#agentPrompt').fill('A 的草稿')
        page.locator('#toggleAgentRail').click()
        page.locator(f'[data-session-open="{SESSION_B}"]').click()
        page.wait_for_function("() => !document.querySelector('#agentHistoryRail').open")
        self.assertEqual(page.locator('#agentPrompt').input_value(), '')
        page.locator('#agentPrompt').fill('B 的草稿')
        page.locator('#toggleAgentRail').click()
        page.locator(f'[data-session-open="{SESSION_A}"]').click()
        page.wait_for_function("() => !document.querySelector('#agentHistoryRail').open")
        self.assertEqual(page.locator('#agentPrompt').input_value(), 'A 的草稿')
        page.locator('#toggleAgentRail').click()
        page.locator(f'[data-session-delete="{SESSION_A}"]').click()
        page.wait_for_function("() => !document.querySelector('#agentHistoryRail').open")
        drafts = page.evaluate("scope => JSON.parse(sessionStorage.getItem('mediaflux.agent.drafts.v1.' + scope))", SCOPE)
        self.assertNotIn(SESSION_A, drafts)
        self.assertEqual(drafts[SESSION_B]['text'], 'B 的草稿')

    def test_completion_and_error_do_not_pull_reader_to_bottom(self):
        for outcome in ('turn.completed', 'turn.failed'):
            with self.subTest(outcome=outcome):
                long_answer = '\n\n'.join(f'第 {i} 条查询记录，需要保留阅读位置。' for i in range(80))
                events = [harness._event(1, 'turn.started'), harness._event(2, 'model.delta', {'delta': long_answer}),
                          harness._event(3, outcome, {'status': 'success' if outcome == 'turn.completed' else 'failed', 'answer': long_answer, 'message': '查询暂不可用'})]
                page = self.page({'queryEvents': events, 'queryDelayMs': 500,
                                  'sessionDetails': {SESSION_A: {'messages': [{'role': 'assistant', 'content': long_answer}]}}},
                                 stored_session=SESSION_A)
                page.wait_for_selector('.agent-narrative')
                page.locator('#agentPrompt').fill('读取长记录')
                page.locator('#agentSend').click()
                page.wait_for_function("() => document.querySelector('#agentTranscript').scrollHeight > 2000")
                page.evaluate("() => { const node = document.querySelector('#agentTranscript'); node.scrollTop = 0; node.dispatchEvent(new Event('scroll')); }")
                page.wait_for_function("() => document.querySelector('#agentStop').hidden")
                self.assertLess(page.locator('#agentTranscript').evaluate('(node) => node.scrollTop'), 10)
                self.assertTrue(page.locator('#agentNewReplies').is_visible())
                page.locator('#agentNewReplies').click()
                page.wait_for_function("() => {const node = document.querySelector('#agentTranscript'); return node.scrollHeight - node.clientHeight - node.scrollTop < 3;}")
                self.assertTrue(page.locator('#agentNewReplies').is_hidden())

    def test_history_filter_rename_and_pin_keep_existing_conversation(self):
        page = self.page({'sessions': {'draft_scope': SCOPE, 'sessions': [
            {'session_id': SESSION_A, 'title': '旧排障记录', 'updated_at': 10, 'pinned': False},
            {'session_id': SESSION_B, 'title': '最近选片', 'updated_at': 20, 'pinned': False},
        ]}})
        page.locator('#agentPrompt').fill('正在编辑的草稿')
        page.locator('#toggleAgentRail').click()
        page.locator('#agentSessionSearch').fill('排障')
        self.assertEqual(page.locator('.agent-session-item:visible').count(), 1)
        page.locator(f'[data-session-rename="{SESSION_A}"]').click()
        editor = page.locator('.agent-session-editor')
        editor.locator('input').fill('我的排障手册')
        editor.locator('button[type=submit]').click()
        page.wait_for_selector('.agent-session-editor', state='detached')
        self.assertEqual(page.locator('.agent-session-open:visible strong').inner_text(), '我的排障手册')
        page.locator('#agentSessionSearch').fill('')
        page.locator(f'[data-session-pin="{SESSION_A}"]').click()
        page.wait_for_function("id => document.querySelector('.agent-session-item').dataset.sessionId === id", arg=SESSION_A)
        self.assertEqual(page.locator(f'[data-session-pin="{SESSION_A}"]').get_attribute('aria-pressed'), 'true')
        self.assertEqual(page.locator('#agentPrompt').input_value(), '正在编辑的草稿')
        self.assertEqual(page.evaluate("window.__kernelCalls.filter(call => call.url === '/api/agent/query').length"), 0)
        page.locator('#agentSessionSearch').fill('不存在的名字')
        self.assertIn('没有匹配', page.locator('.agent-session-empty').inner_text())
        self.snapshot(page, 'history-filter')

    def test_history_update_failure_keeps_editor_and_reports_error(self):
        page = self.page({'patchStatus': 409, 'sessions': {'sessions': [{'session_id': SESSION_A, 'title': '原会话'}]}})
        page.locator('#toggleAgentRail').click()
        page.locator('[data-session-rename]').click()
        page.locator('.agent-session-editor input').fill('新名称')
        page.keyboard.press('Enter')
        page.wait_for_function("() => document.querySelector('#agentSessionStatus').textContent.includes('更新失败')")
        self.assertEqual(page.locator('.agent-session-editor input').input_value(), '新名称')
        self.assertTrue(page.locator('.agent-session-editor input').evaluate('(node) => document.activeElement === node'))
        page.keyboard.press('Escape')
        self.assertTrue(page.locator('#agentHistoryRail').is_visible())
        self.assertEqual(page.locator('.agent-session-open strong').inner_text(), '原会话')

    def test_candidate_comparison_uses_safe_fields_and_only_selects_for_preview(self):
        view = candidate_view()
        page = self.page({'queryEvents': events_for_candidates(view)})
        page.locator('#agentPrompt').fill('找第一集的版本')
        page.locator('#agentSend').click()
        page.wait_for_selector('.agent-candidate-select:not([disabled])')
        self.assertEqual(page.locator('.agent-candidate-card').count(), 2)
        self.assertEqual(page.locator('.agent-candidates img').count(), 0)
        self.assertFalse(page.evaluate('Boolean(window.__ux_xss)'))
        self.assertIn('1080p', page.locator('.agent-candidate-tags').first.inner_text())
        self.assertIn('字幕需要人工核对', page.locator('.agent-candidate-warnings').inner_text())
        page.locator('#agentPrompt').fill('下一句草稿保持原样')
        page.evaluate("() => {window.__kernelConfig.queryEvents = [{type: 'turn.started', event_id: 'new-selection', sequence: 1, payload: {}}]; window.__kernelConfig.holdQueryOpen = true;}")
        page.locator('[data-candidate-position="2"]').click()
        page.wait_for_selector('#agentStop:not([hidden])')
        last = page.evaluate("JSON.parse(window.__kernelCalls.filter(call => call.url === '/api/agent/query').at(-1).body)")
        self.assertEqual(last['selection'], view['items'][1]['selection'])
        self.assertEqual(page.locator('#agentPrompt').input_value(), '下一句草稿保持原样')
        self.assertEqual(page.evaluate("window.__kernelCalls.filter(call => call.url === '/api/agent/actions/confirm').length"), 0)
        self.assertTrue(page.locator('[data-candidate-position="1"]').is_disabled())
        self.snapshot(page, 'candidates-desktop')

    def test_same_turn_search_invalidation_disables_previous_cards(self):
        for signal in ('progress', 'empty_result'):
            with self.subTest(signal=signal):
                events = events_for_candidates(candidate_view())[:2]
                events.append(harness._event(3, 'tool.progress' if signal == 'progress' else 'tool.completed',
                    {'candidate_view': None} if signal == 'progress' else {'result': {'candidate_view': None}}))
                events.append(harness._event(4, 'turn.completed', {'status': 'success', 'answer': '新搜索没有可选择的候选。'}))
                page = self.page({'queryEvents': events})
                page.locator('#agentPrompt').fill('搜索后调整条件')
                page.locator('#agentSend').click()
                page.wait_for_selector('.agent-narrative')
                self.assertEqual(page.locator('.agent-candidate-select:not([disabled])').count(), 0)
                self.assertIn('仅供回看', page.locator('.agent-candidates-note').inner_text())

    def test_rejected_selection_keeps_existing_approval_and_never_offers_unbound_replay(self):
        approval = {'plan_id': 'plan-already-pending-00001', 'tool_name': 'ingest.submit', 'effect': 'WRITE',
                    'preview': {'summary': '已有待确认计划'}, 'result': {}, 'expires_at': time.time() + 900}
        page = self.page({'sessionDetails': {SESSION_A: {'messages': [], 'candidate_view': candidate_view(), 'pending_approval': approval}},
                          'queryEvents': [harness._event(1, 'turn.failed', {'code': 'selection_invalid', 'message': '候选已更新，请重新搜索后选择'})]},
                         stored_session=SESSION_A)
        page.wait_for_selector('.agent-candidate-select:not([disabled])')
        page.locator('.agent-candidate-select').first.click()
        page.wait_for_function("() => document.querySelector('#agentResponseStatus').textContent === '请求失败'")
        self.assertTrue(page.locator('[data-effect-confirm]').is_enabled())
        self.assertEqual(page.locator('.agent-confirmation-card.is-expired').count(), 0)
        self.assertEqual(page.locator('.agent-retry-draft').count(), 0)
        self.assertEqual(page.evaluate("window.__kernelCalls.filter(call => call.url === '/api/agent/actions/confirm').length"), 0)

    def test_expired_candidates_are_readonly_and_mobile_stays_inside_viewport(self):
        page = self.page({'queryEvents': events_for_candidates(candidate_view(expires_at=time.time() - 5))}, viewport={'width': 390, 'height': 844})
        page.locator('#agentPrompt').fill('显示旧的搜索结果')
        page.locator('#agentSend').click()
        page.wait_for_selector('.agent-candidate-card')
        page.wait_for_function("() => document.querySelector('#agentStop').hidden")
        self.assertTrue(page.locator('.agent-candidate-select').first.is_disabled())
        self.assertIn('仅供回看', page.locator('.agent-candidates-note').inner_text())
        self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'), 390)
        self.snapshot(page, 'candidates-mobile')

    def test_resume_nonempty_history_preserves_focused_resume_control(self):
        page = self.page({'sessions': {'sessions': [{'session_id': SESSION_A, 'title': '已保存的历史'}]},
                          'sessionDetails': {SESSION_A: {'messages': [{'role': 'assistant', 'content': '之前已经完成的排障记录。'}]}}})
        page.locator('#agentResumeLatestSession').click()
        page.wait_for_selector('.agent-narrative')
        self.assertTrue(page.locator('#agentResumeLatestSession').evaluate('(node) => document.activeElement === node'))
        self.assertTrue(page.locator('#agentComposer #agentResumeLatestSession').is_visible())

    def test_candidate_preview_message_never_splits_an_emoji_surrogate_pair(self):
        view = candidate_view()
        view['items'][0]['title'] = '观' * 159 + '🎬' + '测试版本'
        page = self.page({'queryEvents': events_for_candidates(view)})
        page.locator('#agentPrompt').fill('搜索一个长标题')
        page.locator('#agentSend').click()
        page.wait_for_selector('.agent-candidate-select:not([disabled])')
        page.locator('.agent-candidate-select').first.click()
        page.wait_for_function("window.__kernelCalls.filter(call => call.url === '/api/agent/query').length === 2")
        payload = page.evaluate("JSON.parse(window.__kernelCalls.filter(call => call.url === '/api/agent/query').at(-1).body)")
        payload['message'].encode('utf-8', errors='strict')
        self.assertEqual(payload['selection'], view['items'][0]['selection'])

    def test_verified_candidate_view_can_be_recovered_from_session(self):
        page = self.page({'sessionDetails': {SESSION_A: {'messages': [], 'candidate_view': candidate_view()}}}, stored_session=SESSION_A)
        page.wait_for_selector('.agent-candidate-select:not([disabled])')
        self.assertEqual(page.locator('.agent-candidate-card').count(), 2)
        self.assertEqual(page.evaluate("window.__kernelCalls.filter(call => call.url === '/api/agent/query').length"), 0)
