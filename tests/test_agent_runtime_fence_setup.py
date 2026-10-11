"""The actual setup lifecycle must not leave a stale empty-config fence."""
import os
from unittest.mock import patch
from tests import test_first_run as support


def test_initialized_legacy_database_can_publish_missing_session_secret():
    with patch.dict(os.environ):
        case = support.FirstRunTests()
        case.setUp()
        try:
            case.paths.ensure_writable_dirs()
            case.paths.env_file.write_text(
                'MEDIAFLUX_INITIALIZED=1\nENV_WEB_PASSPORT=admin\nENV_WEB_PASSWORD=synthetic-strong-password\n'
            )
            case._reload_runtime_modules()
            case.database.init_db()
            from app.main import create_app
            create_app(start_background=False)
            assert case.config.read_env_snapshot(case.paths.env_file)[1].get('WEB_SECRET_KEY')
        finally:
            case.tearDown()


def test_setup_then_normal_publication():
    with patch.dict(os.environ):
        case = support.FirstRunTests()
        case.setUp()
        try:
            from tests.test_setup_bootstrap_support import initialize_web_paths
            def inspect_page(page, client):
                case._setup_page(client)  # retain exact static fingerprints/form oracle
            initialize_web_paths(case.paths, 'admin', 'synthetic-correct-horse', inspect_page)
            case._reload_runtime_modules()
            with case._new_client() as client:
                page = client.get('/login')
                response = client.post('/login', data={
                    'csrf_token': case._csrf_token(page), 'username': 'admin',
                    'password': 'synthetic-correct-horse',
                }, follow_redirects=False)
                assert response.status_code == 302
                case.config.set_and_save({'AGENT_LLM_MODEL': 'synthetic-new-model'})
                assert case.config.get('AGENT_LLM_MODEL') == 'synthetic-new-model'
                from app.agent.feature_gate import read_runtime_fence
                with case.database.get_conn() as conn:
                    assert read_runtime_fence(conn).phase == 'stable'
        finally:
            os.environ.pop('AGENT_LLM_MODEL', None)
            case.tearDown()
