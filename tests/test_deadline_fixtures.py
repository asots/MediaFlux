"""Opt-in installed fixtures retain normal storage and cleanup boundaries."""
import os
from pathlib import Path
import sqlite3
from contextlib import closing

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


def test_installed_web_fixture_does_not_initialize_unselected_fresh_paths(tmp_path):
    from app.modules.setup_bootstrap import needs_guidance
    paths = runtime_paths.RuntimePaths(
        program_dir=Path(__file__).resolve().parents[1],
        data_dir=tmp_path / 'data', config_dir=tmp_path / 'config',
        cache_dir=tmp_path / 'cache', log_dir=tmp_path / 'logs',
        strm_dir=tmp_path / 'strm', trash_dir=tmp_path / 'trash',
    )
    assert needs_guidance(paths)
    with initialized_web_runtime():
        assert needs_guidance(paths)
        assert not paths.database_path.exists()
        assert not paths.env_file.exists()
    assert needs_guidance(paths)
