"""单一安装事务与普通 Web 初装；仅使用隔离进程和真实临时 SQLite。"""
import json
from contextlib import closing
import multiprocessing
import os
import socket
import sqlite3
import sys
import traceback
import warnings

import pytest


def _cold_entry(root, action, args, connection, warning_filters):
    """独立解释器中清除测试包导入的配置，并显式安装离线边界。"""
    warnings.filters = list(warning_filters)
    def deny_network(event, args):
        if event in ('socket.connect', 'socket.bind', 'socket.sendto', 'socket.sendmsg') and args[0].family != socket.AF_UNIX:
            raise AssertionError('cold runtime forbids INET')
        if event in ('socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyaddr', 'socket.getnameinfo'):
            raise AssertionError('cold runtime forbids DNS')

    sys.addaudithook(deny_network)
    with connection:
        try:
            for name in list(sys.modules):
                if name == 'app' or name.startswith('app.'):
                    del sys.modules[name]
            for key in list(os.environ):
                if key.startswith(('MEDIAFLUX_', 'ENV_WEB_', 'WEB_', 'APP_ENV', 'AGENT_')):
                    os.environ.pop(key)
            os.environ.update(MEDIAFLUX_DATA_DIR=str(root), MEDIAFLUX_CONFIG_DIR=str(root / 'config'), MEDIAFLUX_CACHE_DIR=str(root / 'cache'), MEDIAFLUX_LOG_DIR=str(root / 'logs'), MEDIAFLUX_STRM_DIR=str(root / 'strm'), MEDIAFLUX_DISABLE_FILE_LOGGING='1', AGENT_LLM_ENABLED='0')
            result = action(*args)
            connection.send({'result': result})
        except BaseException as exc:
            traceback.print_exc()
            connection.send({'error_type': type(exc).__name__})


def cold_run(root, action, *args):
    """只接收顶层场景与显式参数，不继承 pytest 的线程、锁或闭包。"""
    context = multiprocessing.get_context('spawn')
    reader, writer = context.Pipe(duplex=False)
    child = context.Process(target=_cold_entry, args=(root, action, args, writer, warnings.filters))
    try:
        child.start()
        writer.close()
        assert reader.poll(25), 'child deadline'
        result = reader.recv()
        child.join(5)
        assert child.exitcode == 0, 'child reap deadline or abnormal exit'
        return result
    finally:
        writer.close()
        reader.close()
        if child.is_alive():
            child.kill()
            child.join()
        child.close()


def _initialize_action(settings, expected):
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


def initialize(root, settings=None, expected=None):
    """保留跨测试名称/结果契约；唯一安装实现接收冻结的原始 env 字节。"""
    return cold_run(root, _initialize_action, settings, expected)


def _login_action():
    from fastapi.testclient import TestClient
    from app.main import create_app
    with TestClient(create_app()) as client:
        response = client.get('/login')
        import re
        search_result = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
        assert search_result is not None
        token = search_result.group(1)
        result = client.post('/login', data={'username': 'operator', 'password': 'test-bootstrap-password', 'csrf_token': token}, follow_redirects=False)
        return [response.status_code, result.status_code, result.headers.get('location'), client.get('/readyz').status_code]


def test_single_installation_publishes_stable_and_cold_login(tmp_path):
    assert initialize(tmp_path) == {'result': 0}
    env = tmp_path / 'config/user.env'
    assert env.is_file()
    from app.env_file import read_env_bytes
    values = read_env_bytes(env.read_bytes())
    assert values['ENV_WEB_PASSPORT'] == 'operator'
    assert len(values['WEB_SECRET_KEY']) >= 32
    with closing(sqlite3.connect(tmp_path / 'db/mediaflux.db')) as conn, conn:
        fence = json.loads(conn.execute("SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'").fetchone()[0])
    assert fence['phase'] == 'stable'
    assert initialize(tmp_path) == {'result': 2}
    assert cold_run(tmp_path, _login_action) == {'result': [200, 302, '/', 200]}


def test_env_cas_preserves_bytes_on_refusal(tmp_path):
    config = tmp_path / 'config'
    config.mkdir()
    env = config / 'user.env'
    env.write_text("EXAMPLE='retained'\n")
    before = env.read_bytes()
    assert initialize(tmp_path) == {'result': 2}
    assert env.read_bytes() == before
    assert not (tmp_path / 'db/mediaflux.db').exists()


def _recovery_setup_action(tmp_path, config):
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
        search_result_2 = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        assert search_result_2 is not None
        token = search_result_2.group(1)
        response = client.post('/setup', data={'username':'operator', 'password':'test-bootstrap-password',
                                              'password_confirm':'test-bootstrap-password', 'csrf_token':token},
                               follow_redirects=False)
        return response.status_code


def test_recovery_residual_is_never_read_or_rebuilt_by_setup(tmp_path):
    import hashlib
    config = tmp_path / 'config'
    config.mkdir()
    backup = config / '.user.env.recovery.probe.bak'
    backup.write_text("ENV_WEB_PASSWORD='private-not-for-publication'\n")
    before = hashlib.sha256(backup.read_bytes()).hexdigest()
    assert cold_run(tmp_path, _recovery_setup_action, tmp_path, config) == {'result': 409}
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


def _isolation_action(root, parent_pid):
    import threading

    assert os.getpid() != parent_pid
    assert threading.enumerate() == [threading.main_thread()]
    assert not any(name == 'app' or name.startswith('app.') for name in sys.modules)
    assert 'MEDIAFLUX_TEST_DB_PATH' not in os.environ
    assert 'ENV_WEB_PASSWORD' not in os.environ
    assert os.environ['AGENT_LLM_ENABLED'] == '0'
    from app import config
    assert config.PATHS.data_dir == root
    assert config.ENV_FILE == root / 'config/user.env'
    assert config._cache is None
    with socket.socket() as tcp, socket.socket(type=socket.SOCK_DGRAM) as udp:
        for operation in (
            lambda: socket.getaddrinfo('localhost', 80),
            lambda: tcp.connect(('127.0.0.1', 9)),
            lambda: tcp.bind(('127.0.0.1', 0)),
            lambda: udp.sendto(b'x', ('127.0.0.1', 9)),
        ):
            with pytest.raises(AssertionError, match='cold runtime forbids'):
                operation()
    left, right = socket.socketpair()
    with left, right:
        left.sendall(b'local-ipc')
        assert right.recv(9) == b'local-ipc'
    return 'isolated'


def test_cold_run_isolates_live_threads_config_and_network(tmp_path, monkeypatch):
    import threading
    from app import config

    monkeypatch.setattr(config, '_cache', {'AGENT_LLM_MODEL': 'parent-only'})
    monkeypatch.setenv('MEDIAFLUX_TEST_DB_PATH', str(tmp_path / 'poison.db'))
    monkeypatch.setenv('ENV_WEB_PASSWORD', 'parent-only')
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait)
    thread.start()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='This process.*is multi-threaded', category=DeprecationWarning)
            assert cold_run(tmp_path, _isolation_action, tmp_path, os.getpid()) == {'result': 'isolated'}
        assert config._cache == {'AGENT_LLM_MODEL': 'parent-only'}
        assert os.environ['ENV_WEB_PASSWORD'] == 'parent-only'
        assert not (tmp_path / 'poison.db').exists()
    finally:
        stop.set()
        thread.join(5)
    assert not thread.is_alive()


def _failing_action(crash):
    if crash:
        os._exit(17)
    raise ValueError('controlled child failure')


def _warning_action():
    warnings.warn('controlled child warning', UserWarning)


def test_cold_run_reports_failures_and_reaps_crashed_children(tmp_path):
    before = multiprocessing.active_children()
    assert cold_run(tmp_path, _failing_action, False) == {'error_type': 'ValueError'}
    with pytest.raises(EOFError):
        cold_run(tmp_path, _failing_action, True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        assert cold_run(tmp_path, _warning_action) == {'error_type': 'UserWarning'}
    assert multiprocessing.active_children() == before
