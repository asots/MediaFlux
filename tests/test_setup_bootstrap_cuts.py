"""真实安装事务的持久化失败与旧数据保护，不合成 stable fence。"""
import json
import sqlite3
from contextlib import contextmanager

import pytest

from tests.test_setup_bootstrap import cold_run, initialize


@pytest.mark.parametrize('cut', ['env_directory', 'commit_before', 'commit_after'])
def test_failed_durability_never_reports_success_or_replays(tmp_path, cut):
    def action():
        from app import config, database
        from app.modules import backup
        from app.runtime_paths import get_runtime_paths
        from tests.test_setup_bootstrap_support import initialize_paths
        paths = get_runtime_paths()
        sync, get_conn = backup._fsync_directory, database.get_conn
        if cut == 'env_directory':
            def fail(directory):
                if directory == paths.config_dir and paths.env_file.exists():
                    raise OSError('injected directory durability failure')
                sync(directory)
            backup._fsync_directory = fail
        else:
            @contextmanager
            def uncertain():
                canonical = database.resolve_db_path() == paths.database_path
                with get_conn() as conn:
                    yield conn
                    if canonical and cut == 'commit_before':
                        raise OSError('before commit')
                if canonical and cut == 'commit_after':
                    raise OSError('after actual commit')
            database.get_conn = uncertain
        before = config._cache
        try:
            initialize_paths(paths)
        except Exception:  # noqa: BLE001 - 故障注入边界需要检查任意失败后的状态
            assert database.DB_PATH == paths.database_path
            assert config._cache == before
            return 2
        return 0
    assert cold_run(tmp_path, action) == {'result': 2}
    with sqlite3.connect(tmp_path / 'db/mediaflux.db') as conn:
        row = json.loads(conn.execute("SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'").fetchone()[0])
    assert row['phase'] == ('stable' if cut == 'commit_after' else 'changing')
    before = (tmp_path / 'db/mediaflux.db').stat().st_ino, (tmp_path / 'config/user.env').read_bytes()
    assert initialize(tmp_path) == {'result': 2}
    assert before == ((tmp_path / 'db/mediaflux.db').stat().st_ino, (tmp_path / 'config/user.env').read_bytes())


@pytest.mark.parametrize('production', [False, True])
def test_existing_env_generated_secret_is_durable_and_unrelated_values_preserved(tmp_path, production):
    from app.env_file import read_env_bytes
    config = tmp_path / 'config'
    config.mkdir()
    env = config / 'user.env'
    env.write_text('ENV_WEB_PASSPORT=stale-only\nKEEP_THIS=value\n')
    assert initialize(tmp_path, {'APP_ENV': 'production' if production else 'development', 'ENV_WEB_PASSPORT': 'operator'}, env.read_bytes()) == {'result': 0}
    values = read_env_bytes(env.read_bytes())
    assert values['KEEP_THIS'] == 'value'
    assert len(values['WEB_SECRET_KEY']) >= 32
    assert not (config / '.web-secret-key').exists()


@pytest.mark.parametrize('shape', ['database', '-wal', '-shm', '-journal', 'env_backup', 'restore_journal', 'orphan_stage'])
def test_existing_artifact_refused_without_publication(tmp_path, shape):
    db, config = tmp_path / 'db', tmp_path / 'config'
    db.mkdir()
    config.mkdir()
    if shape == 'env_backup':
        target = config / '.user.env.recovery.leftover.bak'
    elif shape == 'restore_journal':
        def select():
            from app.modules.backup import _restore_journal_path
            from app.runtime_paths import get_runtime_paths
            return str(_restore_journal_path(get_runtime_paths()))
        from pathlib import Path
        target = Path(cold_run(tmp_path, select)['result'])
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target = db / ('.installation-orphan.db' if shape == 'orphan_stage' else 'mediaflux.db' + ('' if shape == 'database' else shape))
    target.write_bytes(b'preserve')
    assert initialize(tmp_path) == {'result': 2}
    assert target.read_bytes() == b'preserve'
    assert not (config / 'user.env').exists()
    if shape != 'database':
        assert not (db / 'mediaflux.db').exists()


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
