"""Opt-in installed fixtures retain normal storage and cleanup boundaries."""
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from app import config, database, runtime_paths
from tests.support import initialized_web_runtime


def test_installed_web_fixture_uses_physical_stable_storage_and_restores_scope():
    from app.agent.feature_gate import read_runtime_fence
    previous = (database.DB_PATH, database._configured_test_mode,
                runtime_paths._configured_paths, config.PATHS, config.ENV_FILE)
    environment = dict(os.environ)
    with initialized_web_runtime() as paths:
        assert database.resolve_db_path() == paths.database_path
        assert database._configured_test_mode is False
        assert os.environ['MEDIAFLUX_TEST_MODE'] == '0'
        assert paths.env_file.is_file()
        with closing(sqlite3.connect(paths.database_path)) as connection:
            fence = read_runtime_fence(connection)
            assert fence.phase == 'stable'
            assert fence.config_digest == config.capture_runtime_config_snapshot().config_digest
            assert connection.execute("SELECT name FROM sqlite_master WHERE name='organize_log'").fetchone()
        root = paths.data_dir.parent
    assert not root.exists()
    assert previous == (database.DB_PATH, database._configured_test_mode,
                        runtime_paths._configured_paths, config.PATHS, config.ENV_FILE)
    assert dict(os.environ) == environment



def test_installed_web_fixture_isolates_ambient_marker_and_restores_it():
    from unittest.mock import patch

    from app.agent.feature_gate import read_runtime_fence
    before = _runtime_scope()
    with patch.dict(os.environ, {'MEDIAFLUX_INITIALIZED': '1'}):
        ambient = dict(os.environ)
        with initialized_web_runtime() as paths:
            assert paths.env_file.is_file()
            assert database._configured_test_mode is False
            with database.get_conn() as conn:
                assert read_runtime_fence(conn).phase == 'stable'
        assert dict(os.environ) == ambient
        assert os.environ['MEDIAFLUX_INITIALIZED'] == '1'
    assert _runtime_scope() == before

def test_installed_web_fixture_does_not_initialize_unselected_fresh_paths(tmp_path):
    paths = runtime_paths.RuntimePaths(
        program_dir=Path(__file__).resolve().parents[1],
        data_dir=tmp_path / 'data', config_dir=tmp_path / 'config',
        cache_dir=tmp_path / 'cache', log_dir=tmp_path / 'logs',
        strm_dir=tmp_path / 'strm', trash_dir=tmp_path / 'trash',
    )
    with initialized_web_runtime():
        assert config.PATHS != paths
        assert runtime_paths.get_runtime_paths() != paths
        assert not paths.database_path.exists()
        assert not paths.env_file.exists()
    assert not paths.database_path.exists()
    assert not paths.env_file.exists()


def _runtime_scope():
    return (database.DB_PATH, database._configured_test_mode,
            runtime_paths._configured_paths, config.PATHS, config.ENV_FILE, dict(os.environ))


def _run_fixture_class(case):
    import unittest
    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(case).run(result)
    return result


def test_web_test_class_performs_real_installation_and_restores_after_suite():
    from fastapi.testclient import TestClient

    from app.agent.feature_gate import read_runtime_fence
    from app.main import create_app
    from tests.support import InitializedWebTestCase

    before, roots = _runtime_scope(), []

    class Installed(InitializedWebTestCase):
        def test_installed(self):
            roots.append(config.PATHS.data_dir.parent)
            self.assertFalse(database._configured_test_mode)
            with database.get_conn() as conn:
                self.assertIsNotNone(conn.execute(
                    "SELECT name FROM sqlite_master WHERE name='organize_log'",
                ).fetchone())
                self.assertEqual(read_runtime_fence(conn).phase, 'stable')
            # 原生Web启动也必须认可该安装；不能只检查fixture自称initialized。
            with TestClient(create_app(start_background=False)) as client:
                self.assertEqual(client.get('/login').status_code, 200)

    result = _run_fixture_class(Installed)
    assert result.testsRun == 1 and result.wasSuccessful(), result.errors + result.failures
    assert len(roots) == 1 and not roots[0].exists()
    assert _runtime_scope() == before


def test_web_test_classes_do_not_share_settings_or_database():
    from tests.support import InitializedWebTestCase

    before, paths = _runtime_scope(), []

    class First(InitializedWebTestCase):
        def test_write(self):
            paths.append(database.resolve_db_path())
            with database.get_conn() as conn:
                conn.execute('CREATE TABLE class_fixture_probe(value TEXT)')
                conn.execute("INSERT INTO class_fixture_probe VALUES('first-class')")

    class Second(InitializedWebTestCase):
        def test_read(self):
            paths.append(database.resolve_db_path())
            with database.get_conn() as conn:
                self.assertIsNone(conn.execute(
                    "SELECT name FROM sqlite_master WHERE name='class_fixture_probe'",
                ).fetchone())

    for case in (First, Second):
        result = _run_fixture_class(case)
        assert result.testsRun == 1 and result.wasSuccessful(), result.errors + result.failures
    assert len(paths) == 2 and paths[0] != paths[1]
    assert all(not path.exists() for path in paths)
    assert _runtime_scope() == before


def test_web_test_class_setup_failure_still_cleans_installation():
    from tests.support import InitializedWebTestCase

    before, roots = _runtime_scope(), []

    class Broken(InitializedWebTestCase):
        @classmethod
        def setUpClass(cls):
            super().setUpClass()
            roots.append(config.PATHS.data_dir.parent)
            raise RuntimeError('deliberate class-setup failure after installation')

        def test_unreachable(self):
            raise AssertionError('test body must not run')

    result = _run_fixture_class(Broken)
    assert result.testsRun == 0 and len(result.errors) == 1
    assert 'deliberate class-setup failure' in result.errors[0][1]
    assert len(roots) == 1 and not roots[0].exists()
    assert _runtime_scope() == before


def test_isolated_database_class_restores_nested_installation_scope():
    from tests.support import IsolatedDatabaseTestCase

    before, paths = _runtime_scope(), []

    class Database(IsolatedDatabaseTestCase):
        def test_nested_database(self):
            paths.extend((database.resolve_db_path(), config.PATHS.data_dir.parent))
            self.assertTrue(database._configured_test_mode)
            self.assertEqual(database.resolve_db_path(), self.test_db_path)
            with database.get_conn() as conn:
                self.assertIsNotNone(conn.execute(
                    "SELECT name FROM sqlite_master WHERE name='organize_log'",
                ).fetchone())

    result = _run_fixture_class(Database)
    assert result.testsRun == 1 and result.wasSuccessful(), result.errors + result.failures
    assert len(paths) == 2 and all(not path.exists() for path in paths)
    assert _runtime_scope() == before


def test_installed_web_fixture_restores_stale_startup_override_snapshot(monkeypatch):
    keys = frozenset({
        'ENV_WEB_PASSPORT', 'ENV_WEB_PASSWORD', 'WEB_SECRET_KEY', 'MEDIAFLUX_INITIALIZED',
    })
    snapshot = config._STARTUP_ENV_OVERRIDES | keys
    monkeypatch.setattr(config, '_STARTUP_ENV_OVERRIDES', snapshot)
    environment = dict(os.environ)
    with initialized_web_runtime():
        _payload, values = config.read_env_snapshot_no_recovery(config.ENV_FILE)
        assert all(values.get(key) for key in keys)
        assert keys.isdisjoint(config._STARTUP_ENV_OVERRIDES)
    assert config._STARTUP_ENV_OVERRIDES is snapshot
    assert dict(os.environ) == environment
