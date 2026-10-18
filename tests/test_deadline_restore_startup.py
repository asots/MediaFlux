"""CORE-SPEC-R1: real prepared restore remains reachable from direct ASGI."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from tests.test_setup_bootstrap import cold_run, initialize


@pytest.mark.parametrize('background_factory', [False, True])
def test_prepared_restore_with_missing_canonical_is_recovered(tmp_path, background_factory):
    assert initialize(tmp_path) == {'result': 0}

    def action():
        from app.modules import backup
        from app.runtime_paths import get_runtime_paths
        paths = get_runtime_paths()
        # Factory variant starts with an already imported, initialized module.
        if background_factory:
            from app import main
            main.app.state.release_startup_lifecycle_guard()
        payload = tmp_path / 'restore-payload.db'
        with closing(sqlite3.connect(paths.database_path)) as src, closing(sqlite3.connect(payload)) as dst:
            src.backup(dst)
        real_replace = backup.os.replace
        interrupted = False

        def replace_then_interrupt(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
            nonlocal interrupted
            result = real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
            if Path(src) == paths.database_path and str(dst).endswith('.bak'):
                interrupted = True
                raise KeyboardInterrupt('controlled interruption after DB capture')
            return result

        backup.os.replace = replace_then_interrupt
        try:
            with backup._offline_restore_guard(paths):
                backup._transactional_restore_files(paths, {'database/mediaflux.db': (paths.database_path, payload)})
        except KeyboardInterrupt:
            pass
        finally:
            backup.os.replace = real_replace
        assert interrupted and not paths.database_path.exists()
        assert backup._restore_journal_path(paths).exists()
        real_recover = backup.recover_pending_restore
        calls = []

        def checked_recover(paths, *, lifecycle_lock_held=False):
            assert lifecycle_lock_held
            assert backup._runtime_lifecycle_key(paths) in backup._RUNTIME_LIFECYCLE_REGISTRY
            result = real_recover(paths, lifecycle_lock_held=True)
            calls.append(result)
            return result

        backup.recover_pending_restore = checked_recover
        try:
            if background_factory:
                from app import main
                app = main.create_app(start_background=True)
            else:
                from app import main
                app = main.app
        finally:
            backup.recover_pending_restore = real_recover
        restored = paths.database_path.exists()
        journal = backup._restore_journal_path(paths).exists()
        if hasattr(app.state, 'release_startup_lifecycle_guard'):
            app.state.release_startup_lifecycle_guard()
        assert not backup._RUNTIME_LIFECYCLE_REGISTRY
        phase = None
        if restored:
            with closing(sqlite3.connect(paths.database_path)) as conn:
                phase = json.loads(conn.execute("SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'").fetchone()[0])['phase']
        return {'restored': restored, 'journal': journal, 'phase': phase, 'recovery_calls': calls}

    assert cold_run(tmp_path, action) == {'result': {'restored': True, 'journal': False, 'phase': 'stable', 'recovery_calls': [True]}}


@pytest.mark.parametrize('start_background', [False, True])
@pytest.mark.parametrize('pending_env', [False, True])
def test_fresh_setup_form_does_not_publish_or_promote(tmp_path, start_background, pending_env):
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    pending = config_dir / '.user.env.recovery.probe.bak'
    pending_bytes = b"ENV_WEB_PASSWORD='private-not-for-publication'\n"
    if pending_env:
        pending.write_bytes(pending_bytes)

    def action():
        from fastapi.testclient import TestClient
        from app import config, main
        from app.modules import backup
        app = main.create_app(start_background=start_background)
        with TestClient(app) as client:
            assert client.get('/setup').status_code == 200
            assert 'name="username"' in client.get('/setup').text
            assert 'name="setup_code"' not in client.get('/setup').text
            assert client.get('/readyz').status_code == 503
            assert client.post('/setup', data={}).status_code == 403
        assert config._cache == {}  # 表单读取空快照，不晋升磁盘恢复片段。
        assert not backup._RUNTIME_LIFECYCLE_REGISTRY
        return True

    assert cold_run(tmp_path, action) == {'result': True}
    assert not (tmp_path / 'db/mediaflux.db').exists()
    assert not (config_dir / 'user.env').exists()
    assert not (config_dir / '.web-secret-key').exists()
    if pending_env:
        assert pending.read_bytes() == pending_bytes


def test_recovery_exception_is_not_guidance_and_releases_background_lock(tmp_path):
    def action():
        from app import main
        from app.modules import backup

        def fail(*args, **kwargs):
            raise OSError('controlled recovery failure')

        backup.recover_pending_restore = fail
        with pytest.raises(OSError, match='controlled recovery failure'):
            main.create_app(start_background=True)
        assert not backup._RUNTIME_LIFECYCLE_REGISTRY
        return True

    assert cold_run(tmp_path, action) == {'result': True}
