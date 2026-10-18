"""旧 FD0 用例迁为正常 Web 首启验收，不保留额外初始化入口。"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from app.env_file import read_env_bytes
from app.runtime_paths import RuntimePaths
from tests.test_setup_bootstrap import cold_run
from tests.test_setup_bootstrap_support import initialize_web_paths


def test_web_setup_persists_admin_without_initial_get_writes(tmp_path):
    paths = RuntimePaths.from_environment({
        "MEDIAFLUX_DATA_DIR": str(tmp_path),
        "MEDIAFLUX_CONFIG_DIR": str(tmp_path / "config"),
    })

    def inspect_page(page, client):
        assert page.status_code == 200
        assert 'class="login-form setup-form"' in page.text
        assert 'class="form-error setup-form-error is-empty"' in page.text
        assert 'tabindex="0" aria-label="初始化错误详情"' in page.text
        assert 'name="password_confirm"' in page.text
        assert 'name="setup_code"' not in page.text
        assert "创建管理员，前往登录" in page.text
        assert "然后重启" not in page.text
        assert client.get("/readyz").status_code == 503
        assert not paths.env_file.exists()
        assert not paths.database_path.exists()
        assert not (paths.config_dir / ".web-secret-key").exists()

    def install():
        initialize_web_paths(paths, "operator", "test-bootstrap-password", inspect_page)
        return 0

    assert cold_run(tmp_path, install) == {"result": 0}
    values = read_env_bytes(paths.env_file.read_bytes())
    assert values["ENV_WEB_PASSPORT"] == "operator"
    assert values["ENV_WEB_PASSWORD"] == "test-bootstrap-password"
    assert values["MEDIAFLUX_INITIALIZED"] == "1"
    assert len(values["WEB_SECRET_KEY"]) >= 32
    with sqlite3.connect(paths.database_path) as conn:
        fence = json.loads(conn.execute(
            "SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'"
        ).fetchone()[0])
    assert fence["phase"] == "stable"


def test_setup_error_css_reserves_scrollable_space_without_changing_login():
    root = Path(__file__).resolve().parents[1]
    css = (root / "app/static/css/core-layout.css").read_text(encoding="utf-8")

    def rule(selector):
        search_result = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", css)
        assert search_result is not None
        return search_result.group(1)

    setup = rule(".form-error.setup-form-error")
    assert "display: flex !important" in setup
    assert "height: 80px" in setup
    assert "flex-shrink: 0" in setup
    assert "overflow-y: auto" in setup
    assert "visibility: hidden" in rule(".form-error.setup-form-error.is-empty")
    assert "display: none !important" in rule(".form-error.is-empty")
    template = (root / "app/templates/login.html").read_text(encoding="utf-8")
    assert "{% if setup_mode %} setup-form-error{% endif %}" in template
