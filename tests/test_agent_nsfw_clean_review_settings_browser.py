"""真实设置模板/脚本的 NSFW 授权组件浏览器回归；不连接运行服务。"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from urllib.parse import urlparse

from jinja2 import Environment

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - 浏览器依赖可选，与现有浏览器测试一致
    sync_playwright = None

ROOT = Path(__file__).resolve().parents[1]
KEY = "AGENT_NSFW_CLEAN_REVIEW_ENABLED"
PARENT_KEY = "AGENT_RECOGNITION_REVIEW_ENABLED"
APP_SCRIPT = ROOT / "app/static/js/app.js"
SETTINGS_SCRIPT = ROOT / "app/static/js/settings.js"
TEMPLATE = (ROOT / "app/templates/settings.html").read_text("utf-8")
CONTENT = TEMPLATE.split("{% block content %}", 1)[1].split("{% endblock %}", 1)[0]
HARNESS = """<!doctype html><html lang="zh-CN" data-theme="light" data-settings-config="pending">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body class="settings-page"><main><div class="content">{content}</div></main></body></html>""".format(content=Environment(autoescape=True).from_string(CONTENT).render(
    app_version="test", resource_results_enabled=False,
))
MOCK_FETCH = r"""(config) => {
    window.__mediafluxInitialSettingsTarget = 'metadata';
    window.__settingsConfig = config;
    window.__settingsWrites = [];
    window.fetch = async (url, options = {}) => {
        if (String(url) !== '/api/config') throw new Error('未授权的浏览器测试请求: ' + url);
        if ((options.method || 'GET') === 'GET') {
            return new Promise(resolve => {
                window.__resolveSettingsConfig = () => resolve(new Response(JSON.stringify(window.__settingsConfig), {
                    status: 200, headers: {'Content-Type': 'application/json'},
                }));
            });
        }
        window.__settingsWrites.push(JSON.parse(options.body));
        return new Promise(resolve => {
            window.__resolveSettingsSave = () => resolve(new Response(JSON.stringify({success: true}), {
                status: 200, headers: {'Content-Type': 'application/json'},
            }));
        });
    };
}"""


@unittest.skipUnless(sync_playwright is not None, "未安装 Playwright")
class NsfwCleanReviewSettingsBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        candidates = [
            Path(os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or cls.playwright.chromium.executable_path),
            *sorted((Path.home() / ".cache/ms-playwright").glob("chromium-*/chrome-linux*/chrome"), reverse=True),
            Path("/usr/bin/google-chrome"), Path("/usr/bin/chromium"),
        ]
        browser_path = next((str(path) for path in candidates if path.is_file()), None)
        if not browser_path:
            cls.playwright.stop()
            raise unittest.SkipTest("未找到 Chrome/Chromium")
        try:
            cls.browser = cls.playwright.chromium.launch(
                headless=True, executable_path=browser_path, args=["--no-sandbox"],
            )
        except Exception:
            cls.playwright.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def _page(self, width, config):
        page = self.browser.new_page(viewport={"width": width, "height": 900}, reduced_motion="reduce")
        self.addCleanup(page.close)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def route_request(route):
            path = urlparse(route.request.url).path
            if path == "/settings":
                route.fulfill(status=200, content_type="text/html", body=HARNESS)
            elif path.startswith("/static/"):
                file = (ROOT / "app" / path.lstrip("/")).resolve()
                if file.is_relative_to(ROOT / "app/static") and file.is_file():
                    route.fulfill(path=str(file))
                else:
                    route.abort()
            else:
                route.abort()

        page.route("**/*", route_request)
        page.goto("http://settings.test/settings")
        for stylesheet in ("main.css", "settings-agent.css"):
            page.add_style_tag(path=str(ROOT / "app/static/css" / stylesheet))
        page.evaluate(MOCK_FETCH, {
            "TMDB_MATCH_MODE": "strict",
            "ORGANIZE_TAVILY_HINTS_DAILY_CREDIT_LIMIT": "20",
            "AI_RECOGNITION_CONFIDENCE_THRESHOLD": "0.8",
            "AI_RECOGNITION_REQUESTS_PER_MINUTE": "6",
            "AI_RECOGNITION_DAILY_REQUEST_LIMIT": "100",
            "AI_RECOGNITION_MAX_CONCURRENCY": "2",
            "AI_RECOGNITION_CIRCUIT_BREAKER_SECONDS": "60",
            **config,
        })
        page.add_script_tag(path=str(APP_SCRIPT))
        page.add_script_tag(path=str(SETTINGS_SCRIPT))
        page.wait_for_function("typeof window.__resolveSettingsConfig === 'function'")
        return page, errors

    @staticmethod
    def _resolve_config(page):
        page.evaluate("window.__resolveSettingsConfig()")
        page.wait_for_function("document.getElementById('settingsForm').getAttribute('aria-busy') === 'false'")
        page.evaluate("document.fonts.ready")
        page.locator(".metadata-nsfw-clean-review").wait_for(state="visible")

    @staticmethod
    def _rect(locator):
        return locator.evaluate("""element => {
            const rect = element.getBoundingClientRect();
            return {x: rect.x + scrollX, y: rect.y + scrollY, width: rect.width, height: rect.height};
        }""")

    def _assert_same_rect(self, before, after):
        for key in ("x", "y", "width", "height"):
            self.assertAlmostEqual(before[key], after[key], delta=0.5, msg=f"{key}: {before} → {after}")

    def test_320_390_desktop_load_toggle_and_save_preserve_component_geometry(self):
        for width in (320, 390, 1280):
            with self.subTest(width=width):
                page, errors = self._page(width, {PARENT_KEY: "1"})
                option = page.locator(".metadata-nsfw-clean-review")
                child = page.locator(f'[data-key="{KEY}"]')
                parent = page.locator(f'[data-key="{PARENT_KEY}"]')
                before_load = self._rect(option)
                self.assertTrue(child.is_disabled())
                self._resolve_config(page)
                self.assertFalse(child.is_checked())
                self.assertFalse(child.is_disabled())
                self._assert_same_rect(before_load, self._rect(option))
                self.assertLessEqual(option.bounding_box()["x"] + option.bounding_box()["width"], width + 0.5)
                self.assertGreaterEqual(option.locator("label.toggle").bounding_box()["width"], 44)
                self.assertGreaterEqual(option.locator("label.toggle").bounding_box()["height"], 44)
                self.assertTrue(page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"))

                parent.locator("..").click()
                self.assertTrue(child.is_disabled())
                self.assertTrue(option.is_visible())
                after_disable = self._rect(option)
                parent.locator("..").click()
                self.assertFalse(child.is_disabled())
                self._assert_same_rect(after_disable, self._rect(option))
                child.locator("..").click()
                self.assertTrue(child.is_checked())
                payload = page.evaluate("collectConfigFields(document.getElementById('settings-panel-metadata'))")
                self.assertEqual(payload, {KEY: "1"})

                button = page.locator('#settings-panel-metadata [data-save-settings]')
                button.scroll_into_view_if_needed()
                page.mouse.move(0, 0)
                rest_state = """() => {
                    const button = document.querySelector('#settings-panel-metadata [data-save-settings]');
                    const transform = getComputedStyle(button).transform;
                    return transform === 'none' || new DOMMatrixReadOnly(transform).isIdentity;
                }"""
                page.wait_for_function(rest_state)
                before_save = button.bounding_box()
                button.click()
                # 去掉既有 hover 的 1px transform，只比较加载/完成状态本身。
                page.mouse.move(0, 0)
                page.wait_for_function("typeof window.__resolveSettingsSave === 'function'")
                self.assertTrue(button.is_disabled())
                # mousemove 的返回不保证 hover 样式已经复位；等真实 transform
                # 回到静止态，不放宽几何容差，也不跳过 loading 状态断言。
                page.wait_for_function(rest_state)
                self._assert_same_rect(before_save, button.bounding_box())
                page.evaluate("window.__resolveSettingsSave()")
                page.wait_for_function("!document.querySelector('#settings-panel-metadata [data-save-settings]').disabled")
                self._assert_same_rect(before_save, button.bounding_box())
                self.assertEqual(page.evaluate("window.__settingsWrites"), [{KEY: "1"}])
                self.assertEqual(page.evaluate("collectConfigFields(document.getElementById('settings-panel-metadata'))"), {})
                self.assertEqual(errors, [])

                artifact_dir = os.getenv("MEDIAFLUX_SETTINGS_TEST_ARTIFACTS")
                if artifact_dir:
                    directory = Path(artifact_dir)
                    directory.mkdir(parents=True, exist_ok=True)
                    option.screenshot(path=str(directory / f"nsfw-settings-{width}.png"))
                    (directory / f"nsfw-settings-{width}.json").write_text(json.dumps({
                        "width": width, "before_load": before_load,
                        "option_rect": self._rect(option), "save_rect": before_save,
                        "page_errors": errors, "overflow": False,
                    }, ensure_ascii=False, indent=2), "utf-8")
                page.close()

    def test_parent_off_keeps_stored_child_authorization_visible_but_inactive(self):
        page, errors = self._page(390, {PARENT_KEY: "0", KEY: "1"})
        self._resolve_config(page)
        child = page.locator(f'[data-key="{KEY}"]')
        self.assertTrue(child.is_disabled())
        self.assertTrue(child.is_checked())
        self.assertTrue(page.locator(".metadata-nsfw-clean-review").is_visible())
        self.assertNotIn(KEY, page.evaluate("collectConfigFields(document.getElementById('settings-panel-metadata'))"))
        self.assertEqual(errors, [])

    def test_environment_boolean_aliases_match_runtime_permission_and_keyboard_focus(self):
        page, errors = self._page(390, {PARENT_KEY: " ON ", KEY: "y"})
        self._resolve_config(page)
        child = page.locator(f'[data-key="{KEY}"]')
        self.assertTrue(child.is_checked())
        self.assertFalse(child.is_disabled())
        child.focus()
        page.keyboard.press("Space")
        self.assertFalse(child.is_checked())
        self.assertEqual(child.locator("..").locator(".toggle-slider").evaluate(
            "element => getComputedStyle(element).outlineStyle"
        ), "solid")
        self.assertEqual(page.evaluate("collectConfigFields(document.getElementById('settings-panel-metadata'))"), {KEY: "0"})
        self.assertEqual(errors, [])

    def test_parent_toggle_cannot_unlock_environment_managed_child_permission(self):
        page, errors = self._page(390, {PARENT_KEY: "1", KEY: "0", "__managed_fields": [KEY]})
        self._resolve_config(page)
        child = page.locator(f'[data-key="{KEY}"]')
        parent = page.locator(f'[data-key="{PARENT_KEY}"]')
        self.assertTrue(child.is_disabled())
        parent.locator("..").click()
        parent.locator("..").click()
        self.assertTrue(child.is_disabled())
        self.assertFalse(child.is_checked())
        self.assertEqual(page.evaluate("collectConfigFields(document.getElementById('settings-panel-metadata'))"), {})
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
