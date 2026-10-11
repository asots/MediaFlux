"""Exec isolation is installed locally, never assumed inherited from pytest."""
import os
from pathlib import Path
import subprocess
import sys

from tests.test_backup import _LIFECYCLE_EXEC_BOOTSTRAP


def test_lifecycle_exec_bootstrap_cleans_paths_and_denies_real_network(tmp_path):
    source = Path(__file__).resolve().parents[1]
    code = _LIFECYCLE_EXEC_BOOTSTRAP + r'''
assert not any(n == 'app' or n.startswith('app.') for n in sys.modules)
assert os.environ['MEDIAFLUX_DATA_DIR'] == str(root / 'data')
assert os.environ['MEDIAFLUX_CONFIG_DIR'] == str(root / 'config')
assert 'MEDIAFLUX_TEST_DB_PATH' not in os.environ
assert 'ENV_WEB_PASSWORD' not in os.environ
assert 'WEB_SECRET_KEY' not in os.environ
assert 'PYTHONPATH' not in os.environ
assert os.environ['AGENT_LLM_ENABLED'] == '0'
assert sys.dont_write_bytecode
with socket.socket() as tcp, socket.socket(type=socket.SOCK_DGRAM) as udp:
    actions = [
        lambda: socket.getaddrinfo('localhost', 80),
        lambda: socket.gethostbyname('unlisted.invalid'),
        lambda: tcp.connect(('127.0.0.1', 9)),
        lambda: tcp.bind(('127.0.0.1', 0)),
        lambda: udp.sendto(b'x', ('127.0.0.1', 9)),
    ]
    for action in actions:
        try:
            action()
        except AssertionError:
            pass
        else:
            raise AssertionError('exec network operation escaped')
left, right = socket.socketpair()
try:
    left.sendall(b'local-ipc')
    assert right.recv(9) == b'local-ipc'
finally:
    left.close()
    right.close()
print('ISOLATION_CONTROL_PASSED')
'''
    env = dict(os.environ, MEDIAFLUX_TEST_DB_PATH=str(tmp_path / 'poison.db'),
               ENV_WEB_PASSWORD='synthetic-inherited-password',
               WEB_SECRET_KEY='synthetic-inherited-secret',
               PYTHONPATH=str(tmp_path / 'not-source'), AGENT_LLM_ENABLED='1')
    result = subprocess.run(
        [sys.executable, '-B', '-c', code, str(tmp_path)], cwd=source,
        env=env, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'EXEC_GUARD_VERIFIED' in result.stdout
    assert 'ISOLATION_CONTROL_PASSED' in result.stdout
    assert not (tmp_path / 'poison.db').exists()
