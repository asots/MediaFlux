"""单一安装事务与普通 Web 初装；仅使用隔离进程和真实临时 SQLite。"""
import json
import os
import select
import signal
import sqlite3
import sys
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


def initialize(root, settings=None, expected=None):
    """保留跨测试名称/结果契约；唯一安装实现接收冻结的原始 env 字节。"""
    def action():
        import secrets
        from app.env_file import read_env_bytes
        from app.modules.setup_bootstrap import BootstrapError, initialize_installation
        from app.runtime_paths import get_runtime_paths
        values = read_env_bytes(expected) if expected is not None else {}
        values.setdefault('ENV_WEB_PASSPORT', 'operator')
        values.setdefault('ENV_WEB_PASSWORD', 'test-bootstrap-password')
        values.setdefault('WEB_SECRET_KEY', secrets.token_urlsafe(48))
        values.update(settings or {})
        values['MEDIAFLUX_INITIALIZED'] = '1'
        try:
            initialize_installation(get_runtime_paths(), values, expected_env=expected)
        except BootstrapError:
            return 2
        return 0
    return cold_run(root, action)


def test_single_installation_publishes_stable_and_cold_login(tmp_path):
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


def test_env_cas_preserves_bytes_on_refusal(tmp_path):
    config = tmp_path / 'config'
    config.mkdir()
    env = config / 'user.env'
    env.write_text("EXAMPLE='retained'\n")
    before = env.read_bytes()
    assert initialize(tmp_path) == {'result': 2}
    assert env.read_bytes() == before
    assert not (tmp_path / 'db/mediaflux.db').exists()


def test_recovery_residual_is_never_read_or_rebuilt_by_setup(tmp_path):
    import hashlib
    config = tmp_path / 'config'
    config.mkdir()
    backup = config / '.user.env.recovery.probe.bak'
    backup.write_text("ENV_WEB_PASSWORD='private-not-for-publication'\n")
    before = hashlib.sha256(backup.read_bytes()).hexdigest()
    def setup():
        import re
        from fastapi.testclient import TestClient
        from app.main import create_app
        with TestClient(create_app()) as client:
            page = client.get('/setup')
            assert page.status_code == 200
            assert 'name="setup_code"' not in page.text
            assert client.get('/readyz').status_code == 503
            assert not (tmp_path / 'db/mediaflux.db').exists()
            assert not (config / 'user.env').exists()
            token = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
            response = client.post('/setup', data={'username':'operator', 'password':'test-bootstrap-password',
                                                  'password_confirm':'test-bootstrap-password', 'csrf_token':token},
                                   follow_redirects=False)
            return response.status_code
    assert cold_run(tmp_path, setup) == {'result': 409}
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == before
    assert not (config / 'user.env').exists()
    assert not (config / '.web-secret-key').exists()
    assert not (tmp_path / 'db/mediaflux.db').exists()


def test_writable_shared_ancestor_does_not_prevent_installation(tmp_path):
    root = tmp_path / 'shared'
    root.mkdir(mode=0o777)
    root.chmod(0o777)
    assert initialize(root) == {'result': 0}
    assert (root / 'config/user.env').stat().st_mode & 0o077 == 0
