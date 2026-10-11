"""CLI/FD0 integration companion to the retired first_run API migration.

Reuse the reviewed exec worker: no alternative bootstrap or fake admission.
"""
from tests.test_setup_bootstrap_completion import SETTINGS, fence, finish, worker


def test_explicit_fd0_initialize_then_fresh_exec_normal_login(tmp_path):
    with worker(tmp_path, 'initialize') as args:
        assert finish(*args) == 0
    from app.env_file import read_env_bytes
    assert read_env_bytes((tmp_path / 'user.env').read_bytes()) == SETTINGS
    assert fence(tmp_path)['phase'] == 'stable'
    assert (tmp_path / 'user.env').stat().st_mode & 0o077 == 0
    before = (tmp_path / 'user.env').read_bytes(), fence(tmp_path)
    with worker(tmp_path, 'login') as args:
        assert finish(*args) == 0
    with worker(tmp_path, 'initialize') as args:
        assert finish(*args) == 2
    assert before == ((tmp_path / 'user.env').read_bytes(), fence(tmp_path))
