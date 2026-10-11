"""Storage cuts in explicit initialization; no fake stable fixtures."""
import json
import sqlite3
from contextlib import contextmanager

import pytest

from tests.test_setup_bootstrap import cold_run, initialize


@pytest.mark.parametrize('cut', ['env_directory', 'commit_before', 'commit_after'])
def test_failed_durability_never_reports_success_or_replays(tmp_path, cut):
    def action():
        import tempfile
        from app.cli import main
        from app import database
        from app.modules import setup_bootstrap as setup
        sync = setup._sync_directory
        get_conn = database.get_conn
        if cut == 'env_directory':
            def fail(path):
                if path == tmp_path / 'config' and (path / 'user.env').exists():
                    raise OSError('injected directory durability failure')
                sync(path)
            setup._sync_directory = fail
        else:
            @contextmanager
            def uncertain():
                canonical = database.resolve_db_path() == tmp_path / 'db/mediaflux.db'
                with get_conn() as conn:
                    yield conn
                    if canonical and cut == 'commit_before':
                        raise OSError('before commit')
                if canonical and cut == 'commit_after':
                    raise OSError('after actual commit')
            database.get_conn = uncertain
        with tempfile.TemporaryFile() as payload:
            payload.write(json.dumps({'ENV_WEB_PASSPORT': 'operator', 'ENV_WEB_PASSWORD': 'test-bootstrap-password'}).encode())
            payload.seek(0)
            return main(['initialize', '--data-dir', str(tmp_path), '--confirm-new-installation', '--expect-env-sha256', 'absent', '--settings-fd', str(payload.fileno())])
    assert cold_run(tmp_path, action) == {'result': 2}
    with sqlite3.connect(tmp_path / 'db/mediaflux.db') as conn:
        row = json.loads(conn.execute("SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'").fetchone()[0])
    assert row['phase'] == ('stable' if cut == 'commit_after' else 'changing')
    assert initialize(tmp_path) == {'result': 2}


@pytest.mark.parametrize('production', [False, True])
def test_existing_env_generated_secret_is_durable_and_unrelated_values_preserved(tmp_path, production):
    from app.env_file import read_env_bytes
    import hashlib
    config = tmp_path / 'config'
    config.mkdir()
    env = config / 'user.env'
    env.write_text("ENV_WEB_PASSPORT=operator\nENV_WEB_PASSWORD=test-bootstrap-password\nKEEP_THIS=value\nAPP_ENV=" + ('production' if production else 'development') + '\n')
    expected = hashlib.sha256(env.read_bytes()).hexdigest()
    assert initialize(tmp_path, {'APP_ENV': 'production' if production else 'development'}, expected) == {'result': 0}
    values = read_env_bytes(env.read_bytes())
    assert values['KEEP_THIS'] == 'value'
    assert len(values['WEB_SECRET_KEY']) >= 32
    assert not (config / '.web-secret-key').exists()


@pytest.mark.parametrize('shape', ['database', '-wal', '-shm', '-journal', 'env_backup'])
def test_existing_artifact_refused_without_publication(tmp_path, shape):
    db = tmp_path / 'db'
    db.mkdir()
    config = tmp_path / 'config'
    config.mkdir()
    target = config / '.user.env.recovery.leftover.bak' if shape == 'env_backup' else db / ('mediaflux.db' + ('' if shape == 'database' else shape))
    target.write_bytes(b'preserve')
    assert initialize(tmp_path) == {'result': 2}
    assert target.read_bytes() == b'preserve'
    assert not (config / 'user.env').exists()


def test_fd_zero_bounded_pipe_input(tmp_path):
    def action():
        import os
        from app.cli import main
        read, write = os.pipe()
        os.write(write, b'{"ENV_WEB_PASSPORT":"operator","ENV_WEB_PASSWORD":"test-bootstrap-password"}')
        os.close(write)
        os.dup2(read, 0)
        os.close(read)
        return main(['initialize', '--data-dir', str(tmp_path), '--confirm-new-installation', '--expect-env-sha256', 'absent', '--settings-fd', '0'])
    assert cold_run(tmp_path, action) == {'result': 0}


def test_release_flat_layout_fd_zero_selected_settings(tmp_path):
    def action():
        import os
        import tempfile
        from app.cli import main
        os.environ['MEDIAFLUX_CONFIG_DIR'] = str(tmp_path)
        os.environ['MEDIAFLUX_CONTAINER'] = '1'
        with tempfile.TemporaryFile() as payload:
            payload.write(json.dumps({'MEDIAFLUX_INITIALIZED': '1', 'ENV_WEB_PASSPORT': 'operator', 'ENV_WEB_PASSWORD': 'test-bootstrap-password', 'WEB_SECRET_KEY': 'synthetic-selected-release-secret'}).encode())
            payload.seek(0)
            os.dup2(payload.fileno(), 0)
            return main(['initialize', '--data-dir', str(tmp_path), '--confirm-new-installation', '--expect-env-sha256', 'absent', '--settings-fd', '0'])
    assert cold_run(tmp_path, action) == {'result': 0}
    assert (tmp_path / 'user.env').is_file()
    with sqlite3.connect(tmp_path / 'mediaflux.db') as conn:
        assert json.loads(conn.execute("SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'").fetchone()[0])['phase'] == 'stable'


@pytest.mark.parametrize('secret', [True, False])
def test_normal_mode_legacy_migration_keeps_real_backup_and_history(tmp_path, secret):
    import zipfile
    config = tmp_path / 'config'
    config.mkdir()
    (config / 'user.env').write_text('MEDIAFLUX_INITIALIZED=1\nENV_WEB_PASSPORT=operator\nENV_WEB_PASSWORD=test-bootstrap-password\n' + ('WEB_SECRET_KEY=synthetic-secret\n' if secret else ''))
    original_env = (config / 'user.env').read_bytes()
    db = tmp_path / 'db'
    db.mkdir()
    with sqlite3.connect(db / 'mediaflux.db') as conn:
        conn.execute('CREATE TABLE organize_log (id INTEGER PRIMARY KEY, source TEXT NOT NULL, original_path TEXT NOT NULL, new_path TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL)')
        conn.execute("INSERT INTO organize_log VALUES(1,'local','old','new','success','2020-01-01')")
    def start():
        from app.main import create_app
        from fastapi.testclient import TestClient
        with TestClient(create_app()) as client:
            return client.get('/readyz').status_code
    assert cold_run(tmp_path, start) == {'result': 200}
    archives = list((tmp_path / 'backups').glob('*.zip'))
    assert archives
    with zipfile.ZipFile(archives[0]) as archive:
        assert archive.read('config/user.env') == original_env
        snapshot = tmp_path / 'backup-observation.db'
        snapshot.write_bytes(archive.read('database/mediaflux.db'))
    with sqlite3.connect(snapshot) as conn:
        assert conn.execute('SELECT original_path,new_path FROM organize_log WHERE id=1').fetchone() == ('old', 'new')
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='settings_kv'").fetchone() is None
    with sqlite3.connect(db / 'mediaflux.db') as conn:
        assert conn.execute('SELECT original_path,new_path FROM organize_log WHERE id=1').fetchone() == ('old', 'new')
        assert json.loads(conn.execute("SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'").fetchone()[0])['phase'] == 'stable'
