"""教学台 Chromium 回归：交互边界隔离测试及真实 API/临时数据库联通测试。"""
from __future__ import annotations

import json
from contextlib import contextmanager
import os
import unittest
from pathlib import Path

from tests.support import InitializedWebTestCase

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - optional browser contract
    sync_playwright = None


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "app" / "templates" / "organize.html"
APP_SCRIPT = ROOT / "app" / "static" / "js" / "app.js"
RELEASE_FORMATS_SCRIPT = ROOT / "app" / "static" / "js" / "release-formats.js"
MAIN_STYLES = ROOT / "app" / "static" / "css" / "main.css"
ORGANIZE_STYLES = ROOT / "app" / "static" / "css" / "organize.css"


MOCK_FETCH = r"""
(config) => {
  window.__releaseConfig = config || {};
  window.__releaseFormatCalls = [];
  window.__releaseFormatConfirmCalls = [];
  const delay = (ms) => new Promise(resolve => setTimeout(resolve, Number(ms || 0)));
  const jsonResponse = (payload, status = 200) => new Response(JSON.stringify(payload), {
    status,
    headers: {'Content-Type': 'application/json'},
  });
  const clone = value => JSON.parse(JSON.stringify(value));
  const defaultPreview = (payload) => ({
    draft: payload.draft,
    rows: payload.filenames.map((filename, index) => ({
      filename,
      before: {title: '旧标题', season: null, episode: null},
      after: {title: payload.examples[0]?.title || '新标题', season: null, episode: index + 1},
      status: 'matched',
      reason: '教学样本提供字段证据',
    })),
    examples: payload.examples.map(example => ({...example, passed: true})),
    summary: {
      total: payload.filenames.length, matched: payload.filenames.length,
      changed: payload.filenames.length, unmatched: 0, blocked: 0,
      conflicts: 0, regressions: 0,
    },
    can_save: true,
    preview_token: 'preview-token-default',
    warnings: [],
  });
  window.fetch = async (url, options = {}) => {
    const parsed = new URL(String(url), document.baseURI);
    const path = parsed.pathname;
    const method = String(options.method || 'GET').toUpperCase();
    let body = null;
    try { body = options.body ? JSON.parse(options.body) : null; } catch (_) {}
    window.__releaseFormatCalls.push({path, method, body, headers: Object.fromEntries(new Headers(options.headers || {}))});

    if (path === '/api/tools/release-formats' && method === 'GET') {
      await delay(window.__releaseConfig.getDelayMs);
      return jsonResponse({items: clone(window.__releaseConfig.items || [])});
    }
    if (path === '/api/tools/release-formats/preview' && method === 'POST') {
      const queued = (window.__releaseConfig.previewQueue || []).shift();
      if (queued) {
        await delay(queued.delayMs);
        return jsonResponse(queued.response || defaultPreview(body), queued.status || 200);
      }
      await delay(window.__releaseConfig.previewDelayMs);
      return jsonResponse(window.__releaseConfig.previewResponse || defaultPreview(body), window.__releaseConfig.previewStatus || 200);
    }
    if (path === '/api/tools/release-formats' && method === 'POST') {
      await delay(window.__releaseConfig.saveDelayMs);
      const response = window.__releaseConfig.saveResponse || {
        item: {
          id: 91, name: body.draft.name, template: body.draft.template,
          scope: body.draft.scope, parent_path: body.draft.parent_path,
          disabled: false, revision: 1, examples: body.examples,
        },
        created: true,
      };
      const status = window.__releaseConfig.saveStatus || 201;
      if (status < 300 && response.item?.id) {
        const items = window.__releaseConfig.items || [];
        window.__releaseConfig.items = [...items.filter(item => item.id !== response.item.id), clone(response.item)];
      }
      return jsonResponse(response, status);
    }
    if (path.startsWith('/api/tools/release-formats/') && method === 'PUT') {
      const id = decodeURIComponent(path.split('/').pop());
      const items = window.__releaseConfig.items || [];
      const item = items.find(value => String(value.id) === id);
      if (!item) return jsonResponse({error: '规则不存在'}, 404);
      item.disabled = Boolean(body.disabled);
      item.revision = Number(body.revision) + 1;
      return jsonResponse({item: clone(item)});
    }
    if (path.startsWith('/api/tools/release-formats/') && method === 'DELETE') {
      const id = decodeURIComponent(path.split('/').pop());
      window.__releaseConfig.items = (window.__releaseConfig.items || []).filter(value => String(value.id) !== id);
      await delay(window.__releaseConfig.deleteDelayMs);
      return jsonResponse({deleted: true});
    }
    throw new Error(`unexpected release-format endpoint: ${method} ${path}`);
  };
  window.appConfirm = async (options = {}) => {
    window.__releaseFormatConfirmCalls.push({title: options.title || '', confirmText: options.confirmText || ''});
    return true;
  };
}
"""


def _chromium_executable(playwright) -> str | None:
    candidates = []
    configured = str(os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path(playwright.chromium.executable_path))
    cache_root = Path.home() / ".cache" / "ms-playwright"
    candidates.extend(sorted(cache_root.glob("chromium-*/chrome-linux*/chrome"), reverse=True))
    candidates.extend((Path("/usr/bin/google-chrome"), Path("/usr/bin/chromium")))
    return next((str(path) for path in candidates if path.is_file()), None)


def _modal_fragment() -> str:
    source = TEMPLATE.read_text(encoding="utf-8")
    start = source.index('<div id="releaseFormatsModal"')
    end = source.index("\n{% endif %}", start)
    return source[start:end]


def _preview_response(*, filenames: list[str], status_rows: list[str] | None = None, can_save: bool = True) -> dict:
    statuses = status_rows or ["matched"] * len(filenames)
    rows = []
    for index, filename in enumerate(filenames):
        status = statuses[index % len(statuses)]
        rows.append(
            {
                "filename": filename,
                "before": {"title": "track 原值", "season": None, "episode": None},
                "after": {"title": "星海航行", "season": None, "episode": index + 13},
                "status": status,
                "reason": {"matched": "样本字段通过", "unmatched": "未找到完整字段", "conflict": "多个格式结果冲突"}.get(status, "需要人工复核"),
            }
        )
    return {
        "draft": {"name": "Example-Team 教学示例", "template": "[Example-Team][{title}][track{episode}r{version}][{resolution}].mkv", "scope": "directory", "parent_path": "/Anime/Teaching"},
        "rows": rows,
        "examples": [
            {"filename": "[Example-Team][星海航行][track013r2][1080p].mkv", "title": "星海航行", "episode": 13, "passed": can_save},
            {"filename": "[Example-Team][星海航行][track014r2][1080p].mkv", "title": "星海航行", "episode": 14, "passed": can_save},
        ],
        "summary": {"total": len(filenames), "matched": sum(status == "matched" for status in statuses), "changed": 1, "unmatched": sum(status == "unmatched" for status in statuses), "blocked": 0, "conflicts": sum(status == "conflict" for status in statuses), "regressions": 0},
        "can_save": can_save,
        "preview_token": "preview-token-teaching" if can_save else "",
        "warnings": ["未匹配和冲突保留现有识别流程。"] if not can_save else [],
    }


@unittest.skipIf(sync_playwright is None, "系统环境未安装 Playwright")
class ReleaseFormatsBrowserTests(InitializedWebTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.playwright = sync_playwright().start()
        executable_path = _chromium_executable(cls.playwright)
        options = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage"]}
        if executable_path:
            options["executable_path"] = executable_path
        cls.browser = cls.playwright.chromium.launch(**options)
        cls.app_script = APP_SCRIPT.read_text(encoding="utf-8")
        cls.release_script = RELEASE_FORMATS_SCRIPT.read_text(encoding="utf-8")
        cls.styles = "\n".join((MAIN_STYLES.read_text(encoding="utf-8"), ORGANIZE_STYLES.read_text(encoding="utf-8")))
        cls.modal_fragment = _modal_fragment()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.browser.close()
        cls.playwright.stop()
        super().tearDownClass()

    def make_page(self, config: dict | None = None, *, viewport: dict | None = None):
        errors: list[str] = []
        page = self.browser.new_page(viewport=viewport or {"width": 1280, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.set_content(
            f"""<!doctype html>
<html lang="zh-CN" data-theme="light">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="csrf-token" content="fixture-csrf"><base href="http://mediaflux.test/organize-rules"></head>
<body class="organize-page organize-rules-page">
<main>
<div class="tmdb-regex-launch release-formats-launch">
  <div>
    <strong>发布格式教学</strong>
    <span>推荐先让 Agent 帮我识别：贴真实样本，由 Agent 推导并先预览核对。</span>
    <span>需要已配置 Agent 模型；未配置时请先到 Agent 设置。</span>
  </div>
  <div class="release-formats-launch-actions">
    <a class="jump-btn release-formats-agent-launch" href="/agent#release-format-teaching">让 Agent 帮我识别</a>
    <button type="button" class="jump-btn release-formats-manual-launch" id="openReleaseFormatsBtn">高级手动</button>
  </div>
</div>
</main>
{self.modal_fragment}
</body></html>"""
        )
        page.add_style_tag(content=self.styles)
        page.add_script_tag(content=self.app_script)
        page.evaluate(MOCK_FETCH, config or {"items": []})
        page.add_script_tag(content=self.release_script)
        page.wait_for_function("() => Boolean(window.__releaseFormatCalls)")
        return page, errors

    @staticmethod
    def open_modal(page) -> None:
        page.locator("#openReleaseFormatsBtn").click()
        page.locator("#releaseFormatsModal").wait_for(state="visible")
        page.locator("#releaseFormatsListState").wait_for()
        page.wait_for_function("() => document.querySelector('#releaseFormatsListState').textContent !== '正在读取规则…'")

    @staticmethod
    def load_teaching_example(page) -> None:
        page.locator("#loadReleaseFormatExampleBtn").click()
        page.locator("#releaseFormatName").wait_for()
        assert page.locator("#releaseFormatTemplate").input_value() == "[Example-Team][{title}][track{episode}r{version}][{resolution}].mkv"

    @staticmethod
    def call_count(page, *, method: str | None = None, path: str | None = None) -> int:
        return page.evaluate(
            """({method, path}) => window.__releaseFormatCalls.filter(call =>
                (!method || call.method === method) && (!path || call.path === path)
            ).length""",
            {"method": method, "path": path},
        )

    @contextmanager
    def real_app_page(self):
        from fastapi.testclient import TestClient
        from app.main import create_app
        from app.modules.recognition import formats
        from tests.support import isolated_test_database
        from tests.test_release_formats import ReleaseFormatApiTests

        errors = []
        with isolated_test_database(), TestClient(create_app()) as client:
            formats.invalidate_cache()
            login = client.get("/login")
            response = client.post("/login", data={
                "csrf_token": ReleaseFormatApiTests.csrf(login), "username": "admin", "password": "123456",
            }, follow_redirects=False)
            self.assertEqual(response.status_code, 302)
            page = self.browser.new_page(viewport={"width": 1280, "height": 900})
            page.on("pageerror", lambda error: errors.append(str(error)))

            def serve(route):
                request = route.request
                if not request.url.startswith("http://testserver/"):
                    route.abort()
                    return
                response = client.request(request.method, request.url, content=request.post_data_buffer,
                                          headers=request.headers)
                headers = {key: value for key, value in response.headers.items()
                           if key not in {"content-encoding", "content-length"}}
                route.fulfill(status=response.status_code, headers=headers, body=response.content)

            page.route("**/*", serve)
            try:
                yield page, client, errors
            finally:
                page.close()
                formats.invalidate_cache()

    def test_real_app_preview_save_reload_and_delete_roundtrip(self):
        from app.modules.recognition import formats
        from app.modules.scraper import _parse_release_core
        from tests.test_release_formats import filename, PARENT

        with self.real_app_page() as (page, _client, errors):
            page.goto("http://testserver/organize-rules")
            self.open_modal(page)
            self.load_teaching_example(page)
            page.locator("#previewReleaseFormatBtn").click()
            page.wait_for_function("() => !document.querySelector('#saveReleaseFormatBtn').disabled")
            self.assertFalse(page.locator("#releaseFormatPreviewEmpty").is_visible())
            self.assertEqual(formats.list_rules(), [])
            page.locator("#saveReleaseFormatBtn").click()
            page.locator(".release-format-rule-card").wait_for()
            self.assertEqual(len(formats.list_rules()), 1)
            self.assertEqual(_parse_release_core(filename(15), PARENT).context.episode, 15)
            page.reload()
            self.open_modal(page)
            self.assertEqual(page.locator(".release-format-rule-card").count(), 1)
            page.locator('[data-release-action="delete"]').click()
            page.locator("#appConfirmSubmit").click()
            page.wait_for_function("() => document.querySelectorAll('.release-format-rule-card').length === 0")
            self.assertEqual(formats.list_rules(), [])
            self.assertIsNone(_parse_release_core(filename(15), PARENT).context.episode)
            self.assertEqual(errors, [])

    def test_agent_entry_previews_and_confirms_through_real_app(self):
        from unittest.mock import patch
        from app.agent.kernel.bootstrap import build_agent_kernel_runtime
        from app.modules.recognition import formats
        from app.modules.scraper import _parse_release_core
        from tests.test_agent_release_format_kernel import TeachingModel, MESSAGE
        from tests.test_release_formats import filename, PARENT, TEMPLATE

        with self.real_app_page() as (page, _client, errors):
            model = TeachingModel()
            runtime = build_agent_kernel_runtime(model=model)
            with patch("app.routes.agent_api.get_agent_kernel_runtime", return_value=runtime):
                page.goto("http://testserver/organize-rules")
                page.locator('a[href="/agent#release-format-teaching"]').click()
                page.locator("#agentPrompt").wait_for(state="visible")
                page.wait_for_function("() => document.querySelector('#agentPrompt').value.includes('发布格式教学')")
                self.assertEqual(model.requests, [])
                self.assertEqual(formats.list_rules(), [])
                page.locator("#agentPrompt").fill(MESSAGE)
                page.locator("#agentComposer").evaluate("form => form.requestSubmit()")
                card = page.locator(".agent-confirmation-card")
                card.wait_for()
                self.assertIn("第13集", card.inner_text())
                self.assertIn("星海航行", card.inner_text())
                self.assertNotIn(TEMPLATE, card.inner_text())
                self.assertNotIn(PARENT, card.inner_text())
                self.assertEqual(formats.list_rules(), [])
                page.set_viewport_size({"width": 390, "height": 844})
                self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 390)
                card.locator("[data-effect-confirm]").click()
                page.wait_for_function("() => document.querySelector('.agent-result-card')?.innerText.includes('已保存')")
                self.assertEqual(len(formats.list_rules()), 1)
                self.assertEqual(_parse_release_core(filename(15), PARENT).context.episode, 15)
                self.assertEqual(len(model.requests), 2)
                self.assertEqual(errors, [])

    def test_teaching_example_builds_fixed_preview_and_save_contract(self):
        response = _preview_response(
            filenames=["[Example-Team][星海航行][track013r2][1080p].mkv", "[Example-Team][星海航行][track014r2][1080p].mkv"]
        )
        page, errors = self.make_page(
            {
                "items": [],
                "previewResponse": response,
                "saveResponse": {
                    "item": {
                        "id": 7,
                        "name": "Example-Team 教学示例",
                        "template": response["draft"]["template"],
                        "scope": "directory",
                        "parent_path": "/Anime/Teaching",
                        "disabled": False,
                        "revision": 1,
                        "examples": response["examples"],
                    },
                    "created": True,
                },
            }
        )
        try:
            self.open_modal(page)
            self.load_teaching_example(page)
            page.locator("#previewReleaseFormatBtn").click()
            page.locator(".release-format-preview-status").first.wait_for()
            page.wait_for_function("() => !document.querySelector('#saveReleaseFormatBtn').disabled")

            preview_call = page.evaluate("() => window.__releaseFormatCalls.find(call => call.path.endsWith('/preview'))")
            self.assertEqual(preview_call["method"], "POST")
            preview_payload = preview_call["body"]
            self.assertEqual(preview_payload["draft"], response["draft"])
            self.assertEqual(preview_payload["filenames"], response["draft"] and response["rows"] and [row["filename"] for row in response["rows"]])
            self.assertEqual(preview_payload["examples"][0]["episode"], 13)
            self.assertNotIn("season", preview_payload["examples"][0])
            self.assertNotIn("regex", json.dumps(preview_payload, ensure_ascii=False).lower())
            self.assertIn("BEFORE", page.locator(".release-formats-preview").inner_text())
            self.assertIn("AFTER", page.locator(".release-formats-preview").inner_text())

            page.locator("#saveReleaseFormatBtn").click()
            page.wait_for_function("() => window.__releaseFormatCalls.some(call => call.method === 'POST' && call.path === '/api/tools/release-formats' && call.body?.confirmed === true)")
            save_call = page.evaluate("() => window.__releaseFormatCalls.find(call => call.path === '/api/tools/release-formats' && call.method === 'POST' && call.body?.confirmed === true)")
            self.assertEqual(set(save_call["body"]), {"draft", "examples", "filenames", "preview_token", "confirmed"})
            self.assertEqual(save_call["body"]["preview_token"], "preview-token-teaching")
            self.assertTrue(save_call["body"]["confirmed"])
        finally:
            page.close()
        self.assertEqual(errors, [])

    def test_selected_template_text_is_replaced_by_limited_field_button(self):
        page, errors = self.make_page({"items": []})
        try:
            self.open_modal(page)
            page.locator("#releaseFormatTemplate").fill("TITLE - {episode}")
            page.locator("#releaseFormatTemplate").evaluate("node => { node.focus(); node.setSelectionRange(0, 5); node.dispatchEvent(new Event('select', {bubbles: true})); }")
            page.locator('[data-release-field="title"]').click()
            self.assertEqual(page.locator("#releaseFormatTemplate").input_value(), "{title} - {episode}")
            self.assertEqual(page.locator("#releaseFormatTemplate").evaluate("node => [node.selectionStart, node.selectionEnd]"), [7, 7])
            self.assertEqual(page.locator('[data-release-field]').count(), 6)
        finally:
            page.close()
        self.assertEqual(errors, [])

    def test_input_change_and_late_preview_response_cannot_restore_save(self):
        first_response = _preview_response(filenames=["first.mkv"])
        late_response = _preview_response(filenames=["late-response.mkv"])
        page, errors = self.make_page({"items": [], "previewResponse": first_response})
        try:
            self.open_modal(page)
            self.load_teaching_example(page)
            page.locator("#previewReleaseFormatBtn").click()
            page.wait_for_function("() => !document.querySelector('#saveReleaseFormatBtn').disabled")
            old_table = page.locator("#releaseFormatPreviewTable").inner_text()

            page.evaluate("(response) => { window.__releaseConfig.previewQueue = [{delayMs: 180, response}]; }", late_response)
            page.locator("#releaseFormatFilenames").fill("late-response.mkv")
            page.locator("#previewReleaseFormatBtn").click()
            page.wait_for_function("() => window.__releaseFormatCalls.filter(call => call.path.endsWith('/preview')).length === 2")
            page.locator("#releaseFormatName").fill("用户在迟到响应前修改了输入")
            page.wait_for_timeout(260)

            self.assertTrue(page.locator("#saveReleaseFormatBtn").is_disabled())
            self.assertIn("输入已变更", page.locator("#releaseFormatsFormState").inner_text())
            self.assertEqual(page.locator("#releaseFormatPreviewTable").inner_text(), old_table)
        finally:
            page.close()
        self.assertEqual(errors, [])

    def test_validation_covers_release_scope_season_and_batch_limit(self):
        page, errors = self.make_page({"items": []})
        try:
            self.open_modal(page)
            self.load_teaching_example(page)
            page.locator("#releaseFormatScope").select_option("release")
            page.locator('[data-example-field="title"]').nth(1).fill("另一部作品")
            page.locator("#releaseFormatTemplate").fill("[{title}][S{season}][E{episode}]")
            page.locator("#previewReleaseFormatBtn").click()
            self.assertIn("需要填写 season", page.locator("#releaseFormatsFormState").inner_text())
            self.assertEqual(self.call_count(page, method="POST", path="/api/tools/release-formats/preview"), 0)

            page.locator("#releaseFormatTemplate").fill("[{title}][E{episode}]")
            page.locator("#releaseFormatFilenames").fill("\n".join(f"file-{index}.mkv" for index in range(101)))
            self.assertEqual(page.locator("#releaseFormatFilenameCount").inner_text(), "101 / 100")
            page.locator("#previewReleaseFormatBtn").click()
            self.assertIn("最多 100", page.locator("#releaseFormatsFormState").inner_text())
            self.assertEqual(self.call_count(page, method="POST", path="/api/tools/release-formats/preview"), 0)
        finally:
            page.close()
        self.assertEqual(errors, [])

    def test_preview_renders_unmatched_and_conflict_without_moving_files(self):
        filenames = ["matched.mkv", "unmatched.mkv", "conflict.mkv"]
        response = _preview_response(filenames=filenames, status_rows=["matched", "unmatched", "conflict"], can_save=False)
        page, errors = self.make_page({"items": [], "previewResponse": response})
        try:
            self.open_modal(page)
            self.load_teaching_example(page)
            page.locator("#releaseFormatFilenames").fill("\n".join(filenames))
            page.locator("#previewReleaseFormatBtn").click()
            page.locator(".release-format-preview-status.is-conflict").wait_for()
            table = page.locator("#releaseFormatPreviewTable").inner_text()
            self.assertIn("未匹配", table)
            self.assertIn("冲突", table)
            self.assertIn("track 原值", table)
            self.assertIn("星海航行", table)
            self.assertTrue(page.locator("#saveReleaseFormatBtn").is_disabled())
            paths = page.evaluate("() => window.__releaseFormatCalls.map(call => call.path)")
            self.assertNotIn("/api/guangya/organize/run", paths)
        finally:
            page.close()
        self.assertEqual(errors, [])

    def test_saved_rules_support_versioned_toggle_and_delete(self):
        item = {
            "id": 13,
            "name": "停用的教学规则",
            "template": "[{title}][{episode}]",
            "scope": "directory",
            "parent_path": "/Anime/Teaching",
            "disabled": True,
            "revision": 3,
            "examples": [
                {"filename": "a.mkv", "title": "A", "episode": 1},
                {"filename": "b.mkv", "title": "A", "episode": 2},
            ],
        }
        page, errors = self.make_page({"items": [item]})
        try:
            self.open_modal(page)
            card = page.locator(".release-format-rule-card").first
            self.assertIn("停用", card.inner_text())
            self.assertIn("v3", card.inner_text())
            card.locator('[data-release-action="toggle"]').click()
            page.wait_for_function("() => window.__releaseFormatCalls.some(call => call.method === 'PUT')")
            page.wait_for_function("() => document.querySelector('.release-format-rule-card')?.innerText.includes('v4')")
            self.assertIn("启用", page.locator(".release-format-rule-card").first.inner_text())
            put_call = page.evaluate("() => window.__releaseFormatCalls.find(call => call.method === 'PUT')")
            self.assertEqual(put_call["body"], {"disabled": False, "revision": 3})

            page.locator('[data-release-action="delete"]').click()
            page.wait_for_function("() => window.__releaseFormatCalls.some(call => call.method === 'DELETE')")
            page.wait_for_function("() => document.querySelectorAll('.release-format-rule-card').length === 0")
            delete_call = page.evaluate("() => window.__releaseFormatCalls.find(call => call.method === 'DELETE')")
            self.assertEqual(delete_call["body"], {"revision": 4})
            self.assertEqual(len(page.evaluate("() => window.__releaseFormatConfirmCalls")), 2)
        finally:
            page.close()
        self.assertEqual(errors, [])

    def test_examples_only_preview_renders_empty_batch_and_remains_saveable(self):
        page, errors = self.make_page({"items": []})
        try:
            self.open_modal(page)
            self.load_teaching_example(page)
            page.locator("#releaseFormatFilenames").fill("")
            page.locator("#previewReleaseFormatBtn").click()
            page.wait_for_function("() => document.querySelector('#releaseFormatPreviewFrame').getAttribute('aria-busy') === 'false'")
            self.assertEqual(page.locator("#releaseFormatPreviewTable td").count(), 1)
            self.assertIn("本批次没有", page.locator("#releaseFormatPreviewTable").inner_text())
            self.assertFalse(page.locator("#saveReleaseFormatBtn").is_disabled())
            self.assertEqual(errors, [])
        finally:
            page.close()

    def test_populated_preview_hides_empty_state_and_scope_fields_obey_hidden(self):
        page, errors = self.make_page({"items": []})
        try:
            self.open_modal(page)
            self.load_teaching_example(page)
            page.locator("#previewReleaseFormatBtn").click()
            page.wait_for_function("() => !document.querySelector('#saveReleaseFormatBtn').disabled")
            self.assertFalse(page.locator("#releaseFormatPreviewEmpty").is_visible())
            self.assertEqual(page.locator("#releaseFormatPreviewTable tr").first.locator(":scope > td").count(), 4)
            self.assertFalse(page.locator("#releaseFormatReleaseNote").is_visible())
            page.locator("#releaseFormatScope").select_option("release")
            self.assertFalse(page.locator("#releaseFormatParentPathField").is_visible())
            self.assertTrue(page.locator("#releaseFormatReleaseNote").is_visible())
        finally:
            page.close()
        self.assertEqual(errors, [])

    def test_save_ack_updates_rule_list_without_overwriting_a_new_draft(self):
        page, errors = self.make_page({"items": [], "saveDelayMs": 180})
        try:
            self.open_modal(page)
            self.load_teaching_example(page)
            page.locator("#previewReleaseFormatBtn").click()
            page.wait_for_function("() => !document.querySelector('#saveReleaseFormatBtn').disabled")
            page.evaluate("() => { const form=document.querySelector('#releaseFormatsForm'); form.dispatchEvent(new Event('submit',{cancelable:true})); form.dispatchEvent(new Event('submit',{cancelable:true})); }")
            page.locator("#releaseFormatName").fill("正在编辑的新草稿")
            page.wait_for_timeout(350)
            self.assertEqual(self.call_count(page, method="POST", path="/api/tools/release-formats"), 1)
            self.assertEqual(page.locator(".release-format-rule-card").count(), 1)
            self.assertEqual(page.locator("#releaseFormatName").input_value(), "正在编辑的新草稿")
            self.assertTrue(page.locator("#saveReleaseFormatBtn").is_disabled())
        finally:
            page.close()
        self.assertEqual(errors, [])

    def test_rule_mutation_invalidates_current_preview_ticket(self):
        item = {"id": 13, "name": "已有格式", "template": "[Group]{title}-{episode}.mkv",
                "scope": "directory", "parent_path": "/Other", "disabled": False, "revision": 1,
                "examples": [{"filename": "a.mkv", "title": "A", "episode": 1},
                             {"filename": "b.mkv", "title": "A", "episode": 2}]}
        page, errors = self.make_page({"items": [item]})
        try:
            self.open_modal(page)
            self.load_teaching_example(page)
            page.locator("#previewReleaseFormatBtn").click()
            page.wait_for_function("() => !document.querySelector('#saveReleaseFormatBtn').disabled")
            page.locator('[data-release-action="toggle"]').click()
            page.wait_for_function("() => document.querySelector('.release-format-rule-card').innerText.includes('v2')")
            self.assertTrue(page.locator("#saveReleaseFormatBtn").is_disabled())
            self.assertIn("重新预览", page.locator("#releaseFormatsFormState").inner_text())
        finally:
            page.close()
        self.assertEqual(errors, [])

    def test_mobile_modal_has_no_page_overflow_and_escape_restores_trigger_focus(self):
        page, errors = self.make_page({"items": []}, viewport={"width": 390, "height": 844})
        try:
            self.open_modal(page)
            geometry = page.evaluate(
                """() => ({
                    documentWidth: document.documentElement.scrollWidth,
                    bodyWidth: document.body.scrollWidth,
                    viewportWidth: window.innerWidth,
                })"""
            )
            self.assertLessEqual(geometry["documentWidth"], geometry["viewportWidth"])
            self.assertLessEqual(geometry["bodyWidth"], geometry["viewportWidth"])
            self.load_teaching_example(page)
            page.locator("#previewReleaseFormatBtn").click()
            page.wait_for_function("() => !document.querySelector('#saveReleaseFormatBtn').disabled")
            self.assertGreaterEqual(page.locator("#releaseFormatName").evaluate("e => parseFloat(getComputedStyle(e).fontSize)"), 16)
            page.keyboard.press("Escape")
            page.locator("#releaseFormatsModal").wait_for(state="hidden")
            self.assertEqual(page.evaluate("() => document.activeElement?.id"), "openReleaseFormatsBtn")
        finally:
            page.close()
        self.assertEqual(errors, [])

    def test_recommended_agent_entry_keeps_manual_modal_trigger(self):
        page, errors = self.make_page({"items": []})
        try:
            agent_entry = page.locator('a[href="/agent#release-format-teaching"]')
            self.assertEqual(agent_entry.count(), 1)
            self.assertEqual(agent_entry.get_attribute("href"), "/agent#release-format-teaching")
            self.assertIn("让 Agent 帮我识别", agent_entry.inner_text())
            self.assertIn("需要已配置 Agent 模型", page.locator(".release-formats-launch").inner_text())

            manual_entry = page.locator("#openReleaseFormatsBtn")
            self.assertEqual(manual_entry.count(), 1)
            self.assertIn("高级手动", manual_entry.inner_text())
            self.open_modal(page)
            self.assertTrue(page.locator("#releaseFormatsModal").is_visible())
        finally:
            page.close()
        self.assertEqual(errors, [])

    def test_template_scopes_script_to_rules_page_and_exposes_no_regex_editor(self):
        source = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("<script src=\"{{ static_url('js/release-formats.js') }}\"></script>", source)
        scripts_block = source.split("{% block scripts %}", 1)[1].split("{% endblock %}", 1)[0]
        self.assertIn("{% if organize_view == 'rules' %}", scripts_block)
        self.assertIn('id="releaseFormatsModal"', source)
        self.assertIn('href="/agent#release-format-teaching"', source)
        self.assertIn("让 Agent 帮我识别", source)
        self.assertIn("高级手动", source)
        self.assertIn("需要已配置 Agent 模型", source)
        self.assertEqual(source.count('id="openReleaseFormatsBtn"'), 1)
        self.assertEqual(source.count('data-release-field="'), 6)
        self.assertNotIn('data-release-field="regex"', source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
