"""Explicit first installation, using real isolated processes and SQLite."""
import hashlib
import json
import os
import select
import signal
import sqlite3
import sys
import tempfile
import time


def cold_run(root, action):
    """Fork inherits the offline runner, but no application/config module state."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            for name in list(sys.modules):
                if name == 'app' or name.startswith('app.'):
                    del sys.modules[name]
            for key in list(os.environ):
                if key.startswith(('MEDIAFLUX_', 'ENV_WEB_', 'WEB_', 'APP_ENV', 'AGENT_')):
                    os.environ.pop(key)
            os.environ.update(MEDIAFLUX_DATA_DIR=str(root), MEDIAFLUX_CONFIG_DIR=str(root / 'config'), MEDIAFLUX_CACHE_DIR=str(root / 'cache'), MEDIAFLUX_LOG_DIR=str(root / 'logs'), MEDIAFLUX_STRM_DIR=str(root / 'strm'), MEDIAFLUX_DISABLE_FILE_LOGGING='1', AGENT_LLM_ENABLED='0')
            result = action()
            payload = json.dumps({'result': result}).encode()
        except BaseException as exc:
            payload = json.dumps({'error_type': type(exc).__name__}).encode()
        os.write(write_fd, payload)
        os.close(write_fd)
        os._exit(0)
    os.close(write_fd)
    try:
        assert select.select([read_fd], [], [], 25)[0], 'child deadline'
        data = os.read(read_fd, 65536)
        deadline = time.monotonic() + 5
        while True:
            found, status = os.waitpid(pid, os.WNOHANG)
            if found:
                assert status == 0
                break
            assert time.monotonic() < deadline, 'child reap deadline'
            time.sleep(.01)
        return json.loads(data)
    finally:
        os.close(read_fd)
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except ProcessLookupError:
            pass


def initialize(root, settings=None, expected='absent'):
    def action():
        from app.cli import main
        with tempfile.TemporaryFile() as payload:
            payload.write(json.dumps(settings or {'ENV_WEB_PASSPORT': 'operator', 'ENV_WEB_PASSWORD': 'test-bootstrap-password'}).encode())
            payload.seek(0)
            return main(['initialize', '--data-dir', str(root), '--confirm-new-installation', '--expect-env-sha256', expected, '--settings-fd', str(payload.fileno())])
    return cold_run(root, action)


def test_explicit_cli_publishes_stable_and_cold_login(tmp_path):
    assert initialize(tmp_path) == {'result': 0}
    env = tmp_path / 'config/user.env'
    assert env.is_file()
    from app.env_file import read_env_bytes
    values = read_env_bytes(env.read_bytes())
    assert values['ENV_WEB_PASSPORT'] == 'operator'
    assert len(values['WEB_SECRET_KEY']) >= 32
    with sqlite3.connect(tmp_path / 'db/mediaflux.db') as conn:
        fence = json.loads(conn.execute("SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'").fetchone()[0])
    assert fence['phase'] == 'stable'
    assert initialize(tmp_path) == {'result': 2}
    def login():
        from fastapi.testclient import TestClient
        from app.main import create_app
        with TestClient(create_app()) as client:
            response = client.get('/login')
            import re
            token = re.search(r'name="csrf_token" value="([^"]+)"', response.text).group(1)
            result = client.post('/login', data={'username': 'operator', 'password': 'test-bootstrap-password', 'csrf_token': token}, follow_redirects=False)
            return [response.status_code, result.status_code, result.headers.get('location'), client.get('/readyz').status_code]
    assert cold_run(tmp_path, login) == {'result': [200, 302, '/', 200]}


def test_ordinary_missing_storage_is_guidance_without_recovery(tmp_path):
    config = tmp_path / 'config'
    config.mkdir()
    backup = config / '.user.env.recovery.probe.bak'
    backup.write_text("ENV_WEB_PASSWORD='private-not-for-publication'\n")
    before = hashlib.sha256(backup.read_bytes()).hexdigest()
    def guidance():
        from app.cli import main
        import uvicorn
        from fastapi.testclient import TestClient
        def run(app, **kwargs):
            with TestClient(app) as client:
                assert client.get('/setup').status_code == 200
                assert 'initialize' in client.get('/setup').text
                assert client.post('/setup', data={}).status_code == 403
                assert client.get('/readyz').status_code == 503
            assert 'app.main' not in sys.modules
            # Existing lifecycle lock imports database -> config (pure import).
            assert sys.modules['app.config']._cache is None
            assert 'app.modules.first_run' not in sys.modules
        uvicorn.run = run
        return main(['start', '--data-dir', str(tmp_path)])
    assert cold_run(tmp_path, guidance) == {'result': 0}
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == before
    assert not (config / 'user.env').exists()
    assert not (config / '.web-secret-key').exists()
    assert not (tmp_path / 'db/mediaflux.db').exists()


def test_browser_live_grant_actual_form_and_single_consume(tmp_path):
    def browser():
        import re
        import uvicorn
        from fastapi.testclient import TestClient
        from app.cli import main
        observed = []
        failures = []
        def serve(server):
            try:
                with TestClient(server.config.app, base_url='http://127.0.0.1:1258') as client:
                    page = client.get('/setup')
                    assert page.status_code == 200
                    code = (tmp_path / 'grant').read_text()
                    assert code not in page.text
                    assert 'setup_code' in page.text
                    token = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
                    fields = {'username': 'operator', 'password': 'test-bootstrap-password', 'password_confirm': 'test-bootstrap-password', 'setup_code': code, 'csrf_token': token}
                    wrong = dict(fields, setup_code='incorrect')
                    observed.append(client.post('/setup', data=wrong, headers={'Origin': 'http://127.0.0.1:1258'}).status_code)
                    observed.append(client.post('/setup', data=fields, headers={'Origin': 'http://evil.invalid'}).status_code)
                    assert not (tmp_path / 'db/mediaflux.db').exists()
                    result = client.post('/setup', data=fields, headers={'Origin': 'http://127.0.0.1:1258'})
                    observed.append(result.status_code)
                    assert 'restart' in result.text and 'session' not in client.cookies
                    observed.append(client.post('/setup', data=fields, headers={'Origin': 'http://127.0.0.1:1258'}).status_code)
            except BaseException as exc:
                failures.append(type(exc).__name__)
        uvicorn.Server.run = serve
        result = main(['initialize', '--data-dir', str(tmp_path), '--confirm-new-installation', '--expect-env-sha256', 'absent', '--web', '--grant-file', str(tmp_path / 'grant')])
        return [result, observed, failures]
    assert cold_run(tmp_path, browser) == {'result': [0, [403, 403, 200, 409], []]}
    assert (tmp_path / 'grant').stat().st_mode & 0o077 == 0
    with sqlite3.connect(tmp_path / 'db/mediaflux.db') as conn:
        assert json.loads(conn.execute("SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'").fetchone()[0])['phase'] == 'stable'


def test_env_cas_preserves_bytes_on_refusal(tmp_path):
    config = tmp_path / 'config'
    config.mkdir()
    env = config / 'user.env'
    env.write_text("EXAMPLE='retained'\n")
    before = env.read_bytes()
    assert initialize(tmp_path) == {'result': 2}
    assert env.read_bytes() == before
    assert not (tmp_path / 'db/mediaflux.db').exists()
