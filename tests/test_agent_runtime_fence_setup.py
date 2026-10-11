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
            with case._new_client() as client:
                page = case._setup_page(client)
                response = client.post('/setup', data=case._fields(page), follow_redirects=False)
                assert response.status_code == 303
                assert response.headers['location'] == '/login'
                case._login(client)
                case.config.set_and_save({'AGENT_LLM_MODEL': 'synthetic-new-model'})
                assert case.config.get('AGENT_LLM_MODEL') == 'synthetic-new-model'
                from app.agent.feature_gate import read_runtime_fence
                with case.database.get_conn() as conn:
                    fence = read_runtime_fence(conn)
                    assert fence.phase == 'stable'
                    assert fence.config_digest == case.config.capture_runtime_config_snapshot().config_digest
                assert client.get('/readyz').status_code == 200
        finally:
            os.environ.pop('AGENT_LLM_MODEL', None)
            case.tearDown()
