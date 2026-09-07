"""深审：真实浏览器验证RSS刷新与用户筛选交错；不访问后端或生产环境。"""
from __future__ import annotations

import shutil
import unittest
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

ROOT = Path(__file__).resolve().parents[1]


def extract(source, start, end):
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


@unittest.skipIf(sync_playwright is None, "未安装 Playwright")
class RssFilterRefreshBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        bundled = Path(cls.playwright.chromium.executable_path)
        executable = str(bundled) if bundled.is_file() else next(
            (path for name in ("google-chrome", "chromium", "chromium-browser")
             if (path := shutil.which(name))), None,
        )
        if not executable:
            cls.playwright.stop()
            raise unittest.SkipTest("未找到 Chrome/Chromium")
        cls.browser = cls.playwright.chromium.launch(
            executable_path=executable, headless=True, args=["--no-sandbox"],
        )
        source = (ROOT / "app/templates/rss.html").read_text()
        cls.script = "\n".join((
            "let rssSubsController=null,rssSubsRequestId=0,subscriptions=[],rssSubsLoaded=false;",
            "window.pending=[]; const api=(_path, options)=>new Promise(resolve=>pending.push({resolve, options}));",
            "const updateStats=()=>{}; const appAlert=()=>{};",
            extract(source, "function lockRssListHeight(list)", "\nfunction animateRssStats"),
            extract(source, "function escapeHtml(value)", "\n// Smart parser"),
            extract(source, "function rssSkeleton(", "\nasync function updateStats"),
            extract(source, "async function loadSubs(", "\nfunction openSubForm"),
            "window.rows=[{id:1,name:'One',urls:'',enabled:true},{id:2,name:'Two',urls:'',enabled:true}];",
            "window.finish=(rows)=>pending.shift().resolve({ok:true,json:async()=>rows});",
        ))

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def page(self, width):
        context = self.browser.new_context(viewport={"width": width, "height": 900})
        self.addCleanup(context.close)
        context.route("**/*", lambda route: route.abort())
        page = context.new_page()
        page.set_content('<select id="filterSub"><option value="">全部订阅源</option></select><div id="subList"></div>')
        page.add_script_tag(content=self.script)
        page.evaluate("async()=>{const task=loadSubs();finish(rows);await task;}")
        return page

    def test_user_filter_change_survives_five_delayed_background_refreshes(self):
        for width in (375, 1280):
            with self.subTest(width=width):
                page = self.page(width)
                page.select_option("#filterSub", "1")
                for index in range(5):
                    selected = "2" if index % 2 == 0 else "1"
                    page.evaluate("window.task=loadSubs();void 0;")
                    self.assertEqual(page.locator(".rss-sub-card").count(), 2)
                    self.assertEqual(page.locator(".subscription-skeleton").count(), 0)
                    page.select_option("#filterSub", selected)
                    before = page.locator("#filterSub").bounding_box()
                    page.evaluate("async()=>{finish(rows);await task;}")
                    self.assertEqual(page.locator("#filterSub").input_value(), selected)
                    self.assertEqual(page.locator("#filterSub").bounding_box(), before)
                page.close()

    def test_superseded_response_cannot_restore_old_filter_options(self):
        page = self.page(1280)
        page.select_option("#filterSub", "2")
        page.evaluate("window.old=loadSubs();window.latest=loadSubs();void 0;")
        page.evaluate("async()=>{pending[1].resolve({ok:true,json:async()=>[rows[1]]});await latest;}")
        page.evaluate("async()=>{pending[0].resolve({ok:true,json:async()=>[rows[0]]});await old;}")
        self.assertEqual(page.locator("#filterSub").input_value(), "2")
        self.assertEqual(page.locator("#filterSub option").count(), 2)
        self.assertEqual(page.locator(".rss-sub-card").count(), 1)
