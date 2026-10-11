"""真实 exec 竞争/SIGKILL 验收；无旧 FD0 或授权服务。"""
import json
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

import pytest

from tests.bootstrap_completion_worker import SETTINGS


@contextmanager
def worker(root, mode):
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('bootstrap_completion_worker.py')),str(root),mode], stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:
            yield proc, log
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
            with (root/'exec-lifecycle.jsonl').open('a') as receipt:
                receipt.write(json.dumps({'pid':proc.pid,'argv':proc.args,'returncode':proc.returncode,'reaped':proc.poll() is not None,'mode':mode})+'\n')


def finish(proc, log):
    result = proc.wait(timeout=25)
    log.seek(0)
    output = log.read().decode()
    records = [json.loads(line.partition('=')[2]) for line in output.splitlines() if line.startswith('WORKER_RESULT=')]
    assert records, output
    assert records[-1]['code'] == result
    assert records[-1]['guard'] == 'all-inet-and-dns-denied'
    return result


def fence(root):
    with sqlite3.connect('file:'+str(root/'mediaflux.db')+'?mode=ro', uri=True) as conn:
        assert Path(conn.execute('PRAGMA database_list').fetchone()[2]) == root/'mediaflux.db'
        return json.loads(conn.execute("SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'").fetchone()[0])


def wait_for(fn):
    end = time.monotonic()+15
    while not fn():
        assert time.monotonic() < end, 'bounded handshake deadline'
        time.sleep(.01)


def test_flat_layout_single_installation_exec_then_cold_login(tmp_path):
    with worker(tmp_path,'initialize') as args:
        assert finish(*args) == 0
    from app.env_file import read_env_bytes
    assert read_env_bytes((tmp_path/'user.env').read_bytes()) == SETTINGS
    assert fence(tmp_path)['phase'] == 'stable'
    with worker(tmp_path,'login') as args:
        assert finish(*args) == 0


def test_two_exec_initializers_commit_at_most_once(tmp_path):
    with worker(tmp_path,'race') as first, worker(tmp_path,'race') as second:
        wait_for(lambda: len(list(tmp_path.glob('ready-*'))) == 2)
        (tmp_path/'go').write_text('go')
        assert sorted([finish(*first),finish(*second)]) == [0,2]
    before = (tmp_path/'mediaflux.db').stat().st_ino, (tmp_path/'user.env').read_bytes(), fence(tmp_path)
    with worker(tmp_path,'initialize') as args:
        assert finish(*args) == 2
    assert before == ((tmp_path/'mediaflux.db').stat().st_ino, (tmp_path/'user.env').read_bytes(), fence(tmp_path))


@pytest.mark.parametrize('cut',['before','after'])
def test_real_sigkill_at_canonical_boundaries(tmp_path, cut):
    with worker(tmp_path,cut) as (proc, log):
        wait_for(lambda: (tmp_path/'boundary').exists())
        proc.send_signal(signal.SIGKILL)
        assert proc.wait(timeout=10) == -signal.SIGKILL
    if cut == 'before':
        assert not (tmp_path/'mediaflux.db').exists()
        assert list(tmp_path.glob('.installation-*.db'))
        # A fresh exec at the SAME flat paths cannot promote the orphan.
        with worker(tmp_path,'initialize') as args:
            assert finish(*args) == 2
        assert not (tmp_path/'mediaflux.db').exists()
    else:
        assert fence(tmp_path)['phase'] == 'changing'
        with worker(tmp_path,'initialize') as args:
            assert finish(*args) == 2
        with worker(tmp_path,'refuse') as args:
            assert finish(*args) == 0
        assert fence(tmp_path)['phase'] == 'changing'
    assert not (tmp_path/'user.env').exists()
