"""首次安装的唯一事务：共享服务生命周期，按配置快照发布数据库与配置。"""
from __future__ import annotations

import os
import secrets
import sqlite3
import tempfile
from pathlib import Path

from app.runtime_paths import RuntimePaths


class BootstrapError(RuntimeError):
    """不含凭据的失败类别；outcome_unknown 必须人工核验，不能重试冒报成功。"""


def initialize_installation(
    paths: RuntimePaths,
    settings: dict[str, str],
    *,
    expected_env: bytes | None,
) -> None:
    """基于冻结的 env 字节快照创建安装；已有安装或恢复证据绝不覆盖。

    first_run 验证输入并传入当前会话 secret。未提供 secret 时生成随机值；
    只有 stable 事务提交成功才应用进程配置。发布后的失败保留数据供恢复。
    """
    from app import config
    from app.modules.backup import (
        _fsync_directory,
        _restore_journal_path,
        config_snapshot_guard,
        runtime_lifecycle_guard,
    )

    try:
        # 正常 Web 服务已经持有 L；同进程共享租约，固定顺序为 L -> C。
        with runtime_lifecycle_guard(paths), config_snapshot_guard(paths):
            stage = None

            def capture():
                database_path = paths.database_path
                targets = (
                    database_path,
                    *(Path(str(database_path) + suffix) for suffix in ('-wal', '-shm', '-journal')),
                    _restore_journal_path(paths),
                )
                if any(os.path.lexists(target) for target in targets) or any(
                    artifact != stage for artifact in database_path.parent.glob('.installation-*.db*')
                ):
                    raise BootstrapError('existing_storage_or_restore_artifact')
                if config._pending_recovery_backups(paths.env_file):
                    raise BootstrapError('pending_env_recovery')
                payload, values = config.read_env_snapshot_no_recovery(paths.env_file)
                if payload != expected_env:
                    raise BootstrapError('env_compare_and_swap_mismatch')
                effective = {key: os.getenv(key) or values.get(key, '') for key in (
                    'ENV_WEB_PASSPORT', 'ENV_WEB_PASSWORD', 'MEDIAFLUX_INITIALIZED',
                )}
                if (
                    (values.get('ENV_WEB_PASSPORT') and values.get('ENV_WEB_PASSWORD'))
                    or values.get('MEDIAFLUX_INITIALIZED') == '1'
                    or effective['MEDIAFLUX_INITIALIZED'] == '1'
                    or (effective['ENV_WEB_PASSPORT'] and effective['ENV_WEB_PASSWORD'])
                ):
                    raise BootstrapError('existing_initialized_configuration')
                return values

            values = capture()
            for key, value in settings.items():
                if (
                    type(key) is not str or type(value) is not str
                    or key == 'STRM_ROOT'
                    or (key.startswith('MEDIAFLUX_') and key != 'MEDIAFLUX_INITIALIZED')
                ):
                    raise BootstrapError('invalid_or_path_selection_key')
                config._validate_env_item(key, value)
            if settings.get('MEDIAFLUX_INITIALIZED', '1') != '1':
                raise BootstrapError('invalid_initialization_marker')
            from app.modules.first_run import _validate_credentials
            try:
                username = _validate_credentials(settings.get('ENV_WEB_PASSPORT', ''), settings.get('ENV_WEB_PASSWORD', ''))
            except ValueError:
                raise BootstrapError('invalid_administrator_credentials') from None
            merged = dict(values)
            merged.update(settings)
            merged['ENV_WEB_PASSPORT'] = username
            merged['MEDIAFLUX_INITIALIZED'] = '1'
            if not merged.get('WEB_SECRET_KEY', '').strip():
                merged['WEB_SECRET_KEY'] = secrets.token_urlsafe(48)
            for key in ('ENV_WEB_PASSPORT', 'ENV_WEB_PASSWORD', 'WEB_SECRET_KEY', 'MEDIAFLUX_INITIALIZED'):
                if os.getenv(key) and os.environ[key] != merged[key]:
                    raise BootstrapError('external_conflict:' + key)

            from app import database
            from app.agent.feature_gate import (
                begin_installation_fence,
                finish_runtime_change,
            )
            test_mode = database._configured_test_mode
            try:
                for directory in {paths.database_path.parent, paths.env_file.parent}:
                    directory.mkdir(parents=True, exist_ok=True)
                descriptor, name = tempfile.mkstemp(prefix='.installation-', suffix='.db', dir=paths.database_path.parent)
                os.close(descriptor)
                stage = Path(name)
                database.configure_database(stage, test_mode=test_mode)
                database.init_db()
                with database.get_conn() as conn:
                    conn.execute('PRAGMA synchronous=FULL')
                    conn.execute('BEGIN IMMEDIATE')
                    changing = begin_installation_fence(conn, config._runtime_digest(merged))
                conn = sqlite3.connect(stage)
                try:
                    if conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0] != 0:
                        raise BootstrapError('stage_checkpoint_busy')
                    if conn.execute('PRAGMA journal_mode=DELETE').fetchone()[0].lower() != 'delete':
                        raise BootstrapError('stage_journal_not_closed')
                finally:
                    conn.close()
                if any(os.path.lexists(str(stage) + suffix) for suffix in ('-wal', '-shm', '-journal')):
                    raise BootstrapError('stage_sidecar_remaining')
                # 使用可写且不截断的句柄，兼容 Windows 的文件同步。
                with stage.open('r+b') as source:
                    os.fsync(source.fileno())
                staged_identity = stage.lstat()
                capture()
                config._publish_noreplace(stage, paths.database_path)
                _fsync_directory(paths.database_path.parent)
                result = config.update_env_file(paths.env_file, merged, expected=expected_env)
                if config.read_env_snapshot_no_recovery(paths.env_file)[0] != result.payload:
                    raise BootstrapError('published_env_changed')
                _fsync_directory(paths.env_file.parent)
                if config._pending_recovery_backups(paths.env_file):
                    raise BootstrapError('pending_env_recovery')
                published_identity = paths.database_path.lstat()
                if (published_identity.st_dev, published_identity.st_ino) != (staged_identity.st_dev, staged_identity.st_ino):
                    raise BootstrapError('published_database_changed')
                if any(os.path.lexists(str(paths.database_path) + suffix) for suffix in ('-wal', '-shm', '-journal')):
                    raise BootstrapError('existing_storage_or_restore_artifact')
                database.configure_database(paths.database_path, test_mode=test_mode)
                try:
                    with database.get_conn() as conn:
                        conn.execute('PRAGMA synchronous=FULL')
                        conn.execute('BEGIN IMMEDIATE')
                        finish_runtime_change(conn, changing, config._runtime_digest(result.data))
                except Exception:  # noqa: BLE001 -- 统一失败类别，不泄漏底层异常中的凭据。
                    # COMMIT 可能已持久化；绝不以读取结果推断成功、重放或删除。
                    raise BootstrapError('outcome_unknown') from None
                with config._lock:
                    if paths.env_file == Path(config.ENV_FILE):
                        config._cache = dict(result.data)
                    for key, value in result.data.items():
                        if not config.has_external_override(key):
                            os.environ[key] = value
            finally:
                # 在 C 释放前恢复标准路径和模式；仅清理本次私有暂存文件。
                database.configure_database(paths.database_path, test_mode=test_mode)
                if stage is not None:
                    for temporary in (stage, *(Path(str(stage) + suffix) for suffix in ('-wal', '-shm', '-journal'))):
                        try:
                            temporary.unlink(missing_ok=True)
                        except OSError:
                            pass
    except BootstrapError:
        raise
    except config.ConcurrentConfigUpdateError:
        raise BootstrapError('env_compare_and_swap_mismatch') from None
    except FileExistsError:
        raise BootstrapError('existing_storage_or_restore_artifact') from None
    except Exception:  # noqa: BLE001 -- 统一失败类别，不泄漏底层异常中的凭据。
        raise BootstrapError('installation_failed') from None
