"""真实 Jinja/base 外壳的 Agent 布局回归；所有浏览器请求只由本地夹具处理。"""
from __future__ import annotations

import json
import mimetypes
import unittest
from urllib.parse import unquote, urlsplit

from tests import test_agent_kernel_browser as browser_support
from tests import test_agent_page as page_support
from tests.support import InitializedWebTestCase


@unittest.skipIf(browser_support.sync_playwright is None, '系统环境未安装 Playwright')
class AgentUXShellBrowserTests(InitializedWebTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from app import database as db

        db.init_db()
        case = page_support.AgentPageTests('test_authenticated_agent_renders_stable_semantic_shell')
        case.setUp()
        try:
            case._login()
            response = case.client.get('/agent', follow_redirects=False)
            if response.status_code != 200:
                raise AssertionError(f'Agent 页面夹具加载失败: {response.status_code}')
            cls.html = response.text
        finally:
            case.tearDown()
        browser_support.AgentKernelBrowserTests.setUpClass.__func__(cls)
        cls.static_root = (browser_support.ROOT / 'app/static').resolve()

    @classmethod
    def tearDownClass(cls):
        try:
            browser_support.AgentKernelBrowserTests.tearDownClass.__func__(cls)
        finally:
            super().tearDownClass()

    def make_shell(self, viewport, *, history):
        context = self.browser.new_context(viewport=viewport)
        self.addCleanup(context.close)
        unexpected = []
        errors = []
        assets = {}

        def route(request):
            url = urlsplit(request.request.url)
            path = unquote(url.path)
            if url.netloc == 'testserver' and path == '/agent':
                request.fulfill(status=200, content_type='text/html', body=self.html)
                return
            if url.netloc == 'testserver' and path.startswith('/static/'):
                asset = (self.static_root / path.removeprefix('/static/')).resolve()
                if asset.is_relative_to(self.static_root) and asset.is_file():
                    assets.setdefault(path, asset.read_bytes())
                    request.fulfill(status=200, content_type=mimetypes.guess_type(str(asset))[0] or 'application/octet-stream', body=assets[path])
                    return
            unexpected.append(request.request.url)
            request.abort()

        context.route('**/*', route)
        context.route_web_socket('**/*', lambda websocket: websocket.close())
        config = {
            'sessions': {'draft_scope': 'a' * 64, 'sessions': [
                {'session_id': 'session_shell_0000000001', 'title': '媒体库排障', 'message_count': 3},
            ] if history else []},
            'nextActionsDelayMs': 250,
            'nextActions': {'snapshot_status': 'attention', 'actions': [
                {'id': f'action-{i}', 'title': f'查看待办 {i}', 'description': '仅查看本地记录', 'prompt': f'查看待办 {i}'} for i in range(3)
            ]},
        }
        mock = browser_support.MOCK_FETCH.replace('window.renderLucideIcons = () => {};', '')
        context.add_init_script(f'({mock})({json.dumps(config, ensure_ascii=False)});')
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://testserver/agent')
        page.wait_for_function("document.querySelector('#agentSessionList').getAttribute('aria-busy') === 'false'")
        return page, unexpected, errors

    def test_real_shell_keeps_composer_centered_and_auxiliary_content_separate(self):
        for viewport in ({'width': 1280, 'height': 800}, {'width': 390, 'height': 844}, {'width': 320, 'height': 360}):
            for history in (False, True):
                with self.subTest(viewport=viewport, history=history):
                    page, unexpected, errors = self.make_shell(viewport, history=history)
                    before = page.locator('#agentComposer').bounding_box()
                    page.wait_for_function("document.querySelector('#agentStartActions').getAttribute('aria-busy') === 'false'")
                    page.evaluate('async () => { await document.fonts.ready; }')
                    after = page.locator('#agentComposer').bounding_box()
                    workspace = page.locator('.agent-workbench').bounding_box()
                    console = page.locator('.agent-console').bounding_box()
                    self.assertLessEqual(abs(console['height'] - workspace['height']), 0.5)
                    self.assertLessEqual(abs(after['y'] + after['height'] / 2 - workspace['y'] - workspace['height'] / 2), 0.5)
                    self.assertLessEqual(abs(before['y'] - after['y']), 0.5)
                    self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'), viewport['width'])
                    self.assertEqual(page.locator('#agentResponseStatus').evaluate('(node) => getComputedStyle(node).position'), 'absolute')
                    if history:
                        resume = page.locator('#agentResumeLatestSession').bounding_box()
                        status = page.locator('#agentStartActionsStatus').bounding_box()
                        self.assertLessEqual(resume['y'] + resume['height'] + 4, status['y'])
                    page.locator('#toggleAgentRail').click()
                    self.assertEqual(page.locator('#agent-session-heading').inner_text(), '历史会话')
                    self.assertTrue(page.locator('#agent-session-heading').evaluate('(node) => document.activeElement === node'))
                    self.assertEqual(page.locator('#agentSessionCount').inner_text(), f'{1 if history else 0} 条')
                    self.assertEqual(unexpected, [])
                    self.assertEqual(errors, [])
