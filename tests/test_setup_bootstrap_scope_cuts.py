"""Scope identity, exact fsync outcomes and real checkpoint contention."""
import hashlib
import json
import sqlite3

import pytest

from tests.test_setup_bootstrap import cold_run, initialize


@pytest.mark.parametrize('cut',['env_inode_aba','path_change','path_change_late','db_fsync','db_fsync_unsupported','unsupported','checkpoint_busy'])
def test_selected_scope_and_durability_controls(tmp_path, cut):
    expected='absent'
    if cut=='env_inode_aba':
        (tmp_path/'config').mkdir()
        (tmp_path/'config/user.env').write_text('KEEP=value\n')
        expected=hashlib.sha256((tmp_path/'config/user.env').read_bytes()).hexdigest()
    def action():
        import os
        import tempfile
        from app.cli import main
        from app.modules import setup_bootstrap as setup
        from app import database, config
        import errno
        original_update=config.update_env_file
        def update(*args, **kwargs):
            result=original_update(*args, **kwargs)
            if cut=='path_change_late':
                os.environ['MEDIAFLUX_CONFIG_DIR']=str(tmp_path/'different')
            return result
        config.update_env_file=update
        original_read=setup.read_settings_fd
        original_sync=setup._sync_directory
        original_init=database.init_db
        retained=[]
        def read(fd):
            result=original_read(fd)
            if cut=='env_inode_aba':
                env=tmp_path/'config/user.env'
                old=tmp_path/'config/retained-inode'
                env.rename(old)
                env.write_bytes(old.read_bytes())
                assert env.stat().st_ino!=old.stat().st_ino
            if cut=='path_change':
                os.environ['MEDIAFLUX_CONFIG_DIR']=str(tmp_path/'different')
            return result
        def sync(path):
            if cut in ('db_fsync','db_fsync_unsupported') and path==tmp_path/'db' and (path/'mediaflux.db').exists():
                raise OSError(errno.EINVAL if cut=='db_fsync_unsupported' else errno.EIO, 'injected strict DB directory fsync failure')
            original_sync(path)
        def unsupported(path):
            raise setup.BootstrapError('unsupported_initialization_storage')
        def initialize_stage():
            original_init()
            if cut=='checkpoint_busy':
                # Open a real reader before the owner's fence write; its old
                # WAL snapshot prevents TRUNCATE from completing.
                conn=sqlite3.connect(database.resolve_db_path())
                conn.execute('BEGIN')
                conn.execute('SELECT count(*) FROM settings_kv').fetchone()
                retained.append(conn)
        setup.read_settings_fd=read
        setup._sync_directory=sync
        database.init_db=initialize_stage
        if cut=='unsupported':
            setup._local_storage=unsupported
        try:
            with tempfile.TemporaryFile() as payload:
                payload.write(json.dumps({'ENV_WEB_PASSPORT':'operator','ENV_WEB_PASSWORD':'test-bootstrap-password'}).encode());payload.seek(0)
                return main(['initialize','--data-dir',str(tmp_path),'--confirm-new-installation','--expect-env-sha256',expected,'--settings-fd',str(payload.fileno())])
        finally:
            for conn in retained:
                conn.close()
    assert cold_run(tmp_path,action)=={'result':2}
    if cut in ('db_fsync','db_fsync_unsupported','path_change_late'):
        with sqlite3.connect(tmp_path/'db/mediaflux.db') as conn:
            assert json.loads(conn.execute("SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'").fetchone()[0])['phase']=='changing'
        assert initialize(tmp_path)=={'result':2}
    else:
        assert not (tmp_path/'db/mediaflux.db').exists()
    if cut=='env_inode_aba':
        assert (tmp_path/'config/user.env').read_bytes()==(tmp_path/'config/retained-inode').read_bytes()
    elif cut=='path_change_late':
        assert (tmp_path/'config/user.env').is_file()
    else:
        assert not (tmp_path/'config/user.env').exists()
