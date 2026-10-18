"""Restore a verified runtime snapshot without reviving archived process authority."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from app import config
from app import database as db
from app.agent import feature_gate as gate
from app.agent.kernel import bootstrap
from app.env_file import read_env_bytes
from app.modules import backup
from tests.support import initialized_web_runtime
from tests.test_backup import make_paths


def _cold_start(paths) -> dict[str, object]:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("MEDIAFLUX_", "ENV_WEB_", "WEB_", "APP_ENV", "AGENT_", "PYTHON")):
            env.pop(key, None)
    env.update(
        MEDIAFLUX_DATA_DIR=str(paths.data_dir),
        MEDIAFLUX_CONFIG_DIR=str(paths.config_dir),
        MEDIAFLUX_CACHE_DIR=str(paths.cache_dir),
        MEDIAFLUX_LOG_DIR=str(paths.log_dir),
        MEDIAFLUX_STRM_DIR=str(paths.strm_dir),
        MEDIAFLUX_DISABLE_FILE_LOGGING="1",
        AGENT_LLM_ENABLED="0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    script = r'''
import json, socket, sys

def deny_network(event, args):
    if event in {"socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"}:
        sock = args[0]
        if getattr(sock, "family", None) != socket.AF_UNIX:
            raise AssertionError("network access denied in cold-start test")
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo"}:
        raise AssertionError("DNS access denied in cold-start test")

sys.addaudithook(deny_network)
from app.main import create_app
create_app(start_background=False)
from app.agent.feature_gate import initialize_runtime_fence, read_runtime_fence
from app import database
with database.get_conn() as conn:
    persisted = read_runtime_fence(conn)
initialized = initialize_runtime_fence()
assert initialized == persisted
assert initialized.phase == "stable"
print(json.dumps({
    "incarnation": initialized.incarnation,
    "phase": initialized.phase,
    "config_digest": initialized.config_digest,
}, sort_keys=True))
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _runtime_paths(root: Path):
    return make_paths(root)


@contextmanager
def _initialized_runtime(paths):
    # The install bootstrap intentionally rejects group-writable parents; make
    # this fixture private regardless of the caller's default umask.
    previous_umask = os.umask(0o077)
    try:
        with initialized_web_runtime(paths):
            yield
    finally:
        os.umask(previous_umask)


def test_create_restore_then_cold_start_reinitializes_stable_fence(tmp_path):
    paths = _runtime_paths(tmp_path)
    with _initialized_runtime(paths):
        config.set_and_save({
            "AGENT_LLM_MODEL": "archived-model",
            "AGENT_LLM_API_URL": "https://invalid.example/v1",
        })
        archive = tmp_path / "runtime.zip"
        backup.create_backup(paths, output=archive)
        config.set_and_save({"AGENT_LLM_MODEL": "changed-after-backup"})
        old_runtime = bootstrap.get_agent_kernel_runtime()
        assert old_runtime.creation_binding is not None
        old_fence = old_runtime.creation_binding.fence

        backup.restore_backup(paths, archive)

        with db.get_conn() as conn:
            restored_fence = gate.read_runtime_fence(conn)
        restored_values = read_env_bytes(paths.env_file.read_bytes())
        assert restored_fence.phase == "stable"
        assert restored_fence.incarnation != old_fence.incarnation
        assert restored_fence.config_digest == config._runtime_digest(restored_values)
        with db.get_conn() as conn, pytest.raises(gate.RuntimeFenceError):
            gate.require_runtime_fence(conn, old_fence)
        assert bootstrap.get_agent_kernel_runtime() is not old_runtime

    cold = _cold_start(paths)
    assert cold == {
        "incarnation": restored_fence.incarnation,
        "phase": "stable",
        "config_digest": restored_fence.config_digest,
    }


def test_restore_without_archived_env_binds_retained_env(tmp_path):
    paths = _runtime_paths(tmp_path)
    with _initialized_runtime(paths):
        archive = tmp_path / "database-only.zip"
        backup.create_backup(paths, output=archive, include_settings=False)
        config.set_and_save({"AGENT_LLM_MODEL": "retained-current-config"})
        retained_env = paths.env_file.read_bytes()

        backup.restore_backup(paths, archive)

        assert paths.env_file.read_bytes() == retained_env
        values = read_env_bytes(retained_env)
        with db.get_conn() as conn:
            fence = gate.read_runtime_fence(conn)
            assert fence.phase == "stable"
            assert fence.config_digest == config._runtime_digest(values)
            assert gate.initialize_runtime_fence() == fence


def test_interrupted_restore_rolls_back_database_and_env_together(tmp_path):
    paths = _runtime_paths(tmp_path)
    with _initialized_runtime(paths):
        archive = tmp_path / "before-change.zip"
        backup.create_backup(paths, output=archive)
        config.set_and_save({
            "AGENT_LLM_MODEL": "live-before-interrupted-restore",
            "AGENT_LLM_API_URL": "https://invalid.example/v1",
        })
        live_env = paths.env_file.read_bytes()
        live_runtime = bootstrap.get_agent_kernel_runtime()
        assert live_runtime.creation_binding is not None
        live_fence = live_runtime.creation_binding.fence

        real_replace = backup.os.replace
        interrupted = False

        def replace_then_interrupt(source, destination):
            nonlocal interrupted
            result = real_replace(source, destination)
            if (
                not interrupted
                and Path(destination) == paths.database_path
                and Path(source).name.endswith(".tmp")
            ):
                interrupted = True
                raise KeyboardInterrupt("simulated interruption after DB publication")
            return result

        with patch.object(
            backup.os, "replace", side_effect=replace_then_interrupt
        ), pytest.raises(KeyboardInterrupt):
            backup.restore_backup(paths, archive)

        journal = paths.data_dir / ".mediaflux-restore.journal.json"
        assert journal.is_file()
        assert backup.recover_pending_restore(paths)
        assert not journal.exists()
        assert paths.env_file.read_bytes() == live_env
        with db.get_conn() as conn:
            recovered_fence = gate.read_runtime_fence(conn)
            assert recovered_fence == live_fence
            gate.require_runtime_fence(conn, live_fence)
        assert bootstrap.get_agent_kernel_runtime() is live_runtime


def test_committed_restore_recovers_new_fence_and_env_together(tmp_path):
    paths = _runtime_paths(tmp_path)
    with _initialized_runtime(paths):
        config.set_and_save({
            "AGENT_LLM_MODEL": "archived-before-commit",
            "AGENT_LLM_API_URL": "https://invalid.example/v1",
        })
        archive = tmp_path / "committed-recovery.zip"
        backup.create_backup(paths, output=archive)
        archived_env = paths.env_file.read_bytes()
        config.set_and_save({"AGENT_LLM_MODEL": "live-before-commit"})
        old_runtime = bootstrap.get_agent_kernel_runtime()
        assert old_runtime.creation_binding is not None
        old_fence = old_runtime.creation_binding.fence

        with patch.object(
            backup,
            "_recover_committed",
            side_effect=KeyboardInterrupt("simulated interruption after commit marker"),
        ), pytest.raises(KeyboardInterrupt):
            backup.restore_backup(paths, archive)

        journal_path = paths.data_dir / ".mediaflux-restore.journal.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        assert journal["phase"] == "committed"
        assert paths.env_file.read_bytes() == archived_env
        assert backup.recover_pending_restore(paths)
        assert not journal_path.exists()

        values = read_env_bytes(archived_env)
        with db.get_conn() as conn:
            restored_fence = gate.read_runtime_fence(conn)
            assert restored_fence.phase == "stable"
            assert restored_fence.incarnation != old_fence.incarnation
            assert restored_fence.config_digest == config._runtime_digest(values)
            with pytest.raises(gate.RuntimeFenceError):
                gate.require_runtime_fence(conn, old_fence)
        assert bootstrap.get_agent_kernel_runtime() is not old_runtime
