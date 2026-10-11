"""Explicit one-shot installation. Importable before config/auth/main.

Local command invocation is authority; neither missing storage nor HTTP is.
Only the owner thread executes storage while holding L, then C. No replay.
"""
import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import queue
import re
import secrets
import select
import sqlite3
import stat
import sys
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit

LIMIT = 65536
TTL = 900


class BootstrapError(RuntimeError):
    """Non-secret failure category suitable for a command receipt."""


def read_settings_fd(fd: int) -> dict[str, str]:
    mode = os.fstat(fd).st_mode
    if not (stat.S_ISREG(mode) or stat.S_ISFIFO(mode)):
        raise BootstrapError('settings_fd_requires_file_or_pipe')
    data = bytearray()
    deadline = time.monotonic() + 10
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            raise BootstrapError('settings_read_deadline')
        chunk = os.read(fd, min(8192, LIMIT + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > LIMIT:
            raise BootstrapError('settings_too_large')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise BootstrapError('duplicate_settings_key')
            result[key] = value
        return result
    try:
        values = json.loads(data.decode('utf-8'), object_pairs_hook=pairs)
    except (UnicodeError, ValueError):
        raise BootstrapError('invalid_settings_json') from None
    if not isinstance(values, dict) or any(type(k) is not str or type(v) is not str for k, v in values.items()):
        raise BootstrapError('settings_require_string_mapping')
    return values


def _identity(path):
    info = path.lstat()
    return info.st_dev, info.st_ino


def _safe_parent(path: Path):
    for part in reversed((path, *path.parents)):
        info = part.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o022 or info.st_uid not in {0, os.geteuid()}:
            raise BootstrapError('unsafe_target_directory')


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_ancestors(path):
    # Includes directories created earlier by CLI/entrypoint, not just this call.
    for part in reversed((path, *path.parents)):
        _sync_directory(part)


def _local_storage(path):
    selected = ('', '')
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        mount = fields[4].replace('\\040', ' ')
        if path == Path(mount) or Path(mount) in path.parents:
            if len(mount) >= len(selected[0]):
                selected = mount, fields[fields.index('-') + 1]
    if selected[1] not in {'ext4', 'ext3', 'ext2', 'xfs', 'btrfs', 'zfs', 'tmpfs', 'overlay'}:
        raise BootstrapError('unsupported_initialization_storage')


def _capture(paths, expected):
    from app.modules.backup import _restore_journal_path
    db = paths.database_path
    targets = [db, *(Path(str(db) + suffix) for suffix in ('-wal', '-shm', '-journal')), _restore_journal_path(paths)]
    if any(os.path.lexists(target) for target in targets):
        raise BootstrapError('existing_storage_or_restore_artifact')
    if list(paths.env_file.parent.glob('.' + paths.env_file.name + '.recovery.*.bak')):
        raise BootstrapError('pending_env_recovery')
    for parent in {db.parent, paths.env_file.parent}:
        _safe_parent(parent)
        _local_storage(parent)
    if db == paths.env_file:
        raise BootstrapError('aliased_targets')
    # Import only after recovery-artifact refusal; no auto-recovery reader.
    from app import config
    payload, values = config.read_env_snapshot_no_recovery(paths.env_file, max_bytes=LIMIT)
    digest = 'absent' if payload is None else hashlib.sha256(payload).hexdigest()
    if digest != expected:
        raise BootstrapError('env_compare_and_swap_mismatch')
    identity = None if payload is None else _identity(paths.env_file)
    return payload, values, identity, _identity(db.parent), _identity(paths.env_file.parent)


def _settings(values, supplied):
    from app import config
    from app.modules.first_run import _validate_credentials
    forbidden = {'STRM_ROOT', 'MEDIAFLUX_TEST_DB_PATH', 'MEDIAFLUX_TEST_MODE', 'MEDIAFLUX_TESTING'}
    for key, value in supplied.items():
        if type(key) is not str or type(value) is not str or (key.startswith('MEDIAFLUX_') and key != 'MEDIAFLUX_INITIALIZED') or key in forbidden:
            raise BootstrapError('invalid_or_path_selection_key')
        if key == 'MEDIAFLUX_INITIALIZED' and value != '1':
            raise BootstrapError('invalid_initialization_marker')
        try:
            config._validate_env_item(key, value)
        except ValueError:
            raise BootstrapError('invalid_settings_item') from None
    merged = dict(values)
    merged.update(supplied)
    try:
        merged['ENV_WEB_PASSPORT'] = _validate_credentials(merged.get('ENV_WEB_PASSPORT', ''), merged.get('ENV_WEB_PASSWORD', ''))
    except ValueError:
        raise BootstrapError('invalid_administrator_credentials') from None
    for key in ('ENV_WEB_PASSPORT', 'ENV_WEB_PASSWORD', 'WEB_SECRET_KEY'):
        if os.environ.get(key) and os.environ[key] != merged.get(key):
            raise BootstrapError('external_conflict:' + key)
    if os.environ.get('APP_ENV', merged.get('APP_ENV', 'development')).lower() == 'production' and merged['ENV_WEB_PASSWORD'] == '123456':
        raise BootstrapError('production_default_password')
    if not merged.get('WEB_SECRET_KEY', '').strip():
        merged['WEB_SECRET_KEY'] = secrets.token_urlsafe(48)
    merged['MEDIAFLUX_INITIALIZED'] = '1'
    return merged


def needs_guidance(paths):
    """Read-only early branch; existing databases still use normal admission."""
    return not os.path.lexists(paths.database_path)


def guidance_app(paths):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
    from html import escape
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def lifespan(_app):
        from app.modules.backup import runtime_lifecycle_guard
        with runtime_lifecycle_guard(paths):
            yield
    app = FastAPI(lifespan=lifespan)
    @app.get('/setup')
    def setup():
        return HTMLResponse('停止服务，检查或恢复已有存储。仅在明确创建新安装时运行 initialize --confirm-new-installation；成功后重启并登录。<pre>' + escape(str(paths.database_path)) + '\n' + escape(str(paths.env_file)) + '</pre>', headers={'Cache-Control': 'no-store', 'X-Frame-Options': 'DENY', 'X-Content-Type-Options': 'nosniff'})
    @app.post('/setup')
    def refuse():
        return PlainTextResponse('local initialize operation required', status_code=403)
    @app.get('/readyz')
    @app.get('/healthz')
    def unready():
        return PlainTextResponse('initialization required', status_code=503)
    @app.get('/')
    @app.get('/login')
    def redirect():
        return RedirectResponse('/setup', status_code=302)
    return app


def setup_host(requested):
    enabled = {'1', 'true', 'yes', 'on'}
    explicit = os.environ.get('MEDIAFLUX_ALLOW_REMOTE_SETUP', '').strip().lower()
    remote = (explicit in enabled) if explicit else os.environ.get('MEDIAFLUX_CONTAINER', '').strip().lower() in enabled
    host = requested or ('0.0.0.0' if remote else '127.0.0.1')
    try:
        local = host == 'localhost' or ipaddress.ip_address(host.strip('[]')).is_loopback
    except ValueError:
        local = False
    if not local and not remote:
        raise BootstrapError('unsafe_first_run_binding')
    return host


def run_new_installation(paths, expected_env, *, settings_fd=None, web=False, grant_file=None, host=None, port=1258, origin=None):
    """Local entry owns L; requests can submit to its private closure only."""
    if not sys.platform.startswith('linux'):
        raise BootstrapError('new_initialization_requires_linux_local_storage')
    if not re.fullmatch(r'absent|[0-9a-f]{64}', expected_env):
        raise BootstrapError('invalid_expected_env_digest')
    if web == (settings_fd is not None):
        raise BootstrapError('choose_exactly_one_input_mode')
    from app.modules.backup import _exclusive_runtime_lifecycle_guard, config_snapshot_guard
    with _exclusive_runtime_lifecycle_guard(paths):
        with config_snapshot_guard(paths):
            captured = _capture(paths, expected_env)
            for parent in {paths.database_path.parent, paths.env_file.parent}:
                _durable_ancestors(parent)
        path_scope = {key: value for key, value in os.environ.items() if key.startswith("MEDIAFLUX_") and key.endswith("_DIR")}
        owner = threading.get_ident()
        attempted = False
        def check_path_scope():
            from app.runtime_paths import get_runtime_paths
            if get_runtime_paths() != paths or {key: value for key, value in os.environ.items() if key.startswith("MEDIAFLUX_") and key.endswith("_DIR")} != path_scope:
                raise BootstrapError("installation_path_scope_changed")
        def consume(supplied):
            nonlocal attempted
            if threading.get_ident() != owner or attempted:
                raise BootstrapError('operation_already_consumed')
            attempted = True
            with config_snapshot_guard(paths):
                check_path_scope()
                if _capture(paths, expected_env) != captured:
                    raise BootstrapError('installation_scope_changed')
                merged = _settings(captured[1], supplied)
                from app import config, database
                from app.agent.feature_gate import begin_installation_fence, finish_runtime_change
                descriptor, name = tempfile.mkstemp(prefix='.installation-', suffix='.db', dir=paths.database_path.parent)
                os.close(descriptor)
                stage = Path(name)
                database.configure_database(stage, test_mode=False)
                database.init_db()
                with database.get_conn() as conn:
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
                with stage.open('rb') as source:
                    os.fsync(source.fileno())
                check_path_scope()
                if _capture(paths, expected_env) != captured:
                    raise BootstrapError('installation_scope_changed')
                config._publish_noreplace(stage, paths.database_path)
                canonical_identity = _identity(paths.database_path)
                _sync_directory(paths.database_path.parent)
                result = config.update_env_file(paths.env_file, merged, expected=captured[0])
                descriptor = config._open_verified_config_file(paths.env_file)
                try:
                    published_identity = (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
                    if os.read(descriptor, LIMIT + 1) != result.payload or _identity(paths.env_file) != published_identity:
                        raise BootstrapError('published_env_changed')
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                _sync_directory(paths.env_file.parent)
                if _identity(paths.database_path) != canonical_identity or _identity(paths.env_file) != published_identity:
                    raise BootstrapError('published_identity_changed')
                if (_identity(paths.database_path.parent), _identity(paths.env_file.parent)) != captured[3:]:
                    raise BootstrapError("published_parent_changed")
                check_path_scope()
                database.configure_database(paths.database_path, test_mode=False)
                try:
                    with database.get_conn() as conn:
                        conn.execute('PRAGMA synchronous=FULL')
                        conn.execute('BEGIN IMMEDIATE')
                        finish_runtime_change(conn, changing, config._runtime_digest(result.data))
                except Exception:
                    # The commit may already be durable. Never report rollback,
                    # replay, delete, or turn cold observation into success.
                    raise BootstrapError('outcome_unknown') from None
                return 'initialized; restart service and log in'
        if not web:
            return consume(read_settings_fd(settings_fd))
        return _web_operation(paths, captured, consume, grant_file, host, port, origin)


def _web_operation(paths, captured, consume, grant_file, host, port, origin):
    """Volatile grant + request queue. Storage never runs in an HTTP worker."""
    import uvicorn
    from app.runtime_paths import protected_runtime_output_target
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    bind = setup_host(host)
    exact_origin = origin or ('http://' + ('[' + bind + ']' if ':' in bind else bind) + ':' + str(port))
    parsed = urlsplit(exact_origin)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password or parsed.hostname in {'0.0.0.0', '::'}:
        raise BootstrapError('exact_setup_origin_required')
    target = Path(grant_file).absolute() if grant_file else None
    if target is None or protected_runtime_output_target(paths, target) or target == paths.database_path or target == paths.env_file:
        raise BootstrapError('unsafe_grant_target')
    from app.modules.backup import _restore_journal_path
    reserved = {_restore_journal_path(paths), paths.pid_file, paths.data_dir / '.mediaflux-runtime-lifecycle.lock', paths.config_dir / '.web-secret-key'}
    if target in reserved or target.name.endswith('.lock') or target.parent == paths.runtime_dir / 'locks':
        raise BootstrapError('unsafe_grant_target')
    _safe_parent(target.parent)
    if target.name.startswith('.user.env.recovery.') or target.name in {paths.database_path.name + s for s in ('-wal', '-shm', '-journal')}:
        raise BootstrapError('unsafe_grant_target')
    code = secrets.token_urlsafe(32)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(descriptor, code.encode('ascii'))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _sync_directory(target.parent)
    code_digest = hashlib.sha256(code.encode()).digest()
    del code
    deadline = time.monotonic() + TTL
    csrf = secrets.token_urlsafe(32)
    mutex = threading.Lock()
    state = 'armed'
    work = queue.Queue(maxsize=1)
    result_ready = threading.Event()
    result = {'message': '', 'status': 500}
    attempts = []
    templates = Environment(loader=FileSystemLoader(Path(__file__).parents[1] / 'templates'), autoescape=select_autoescape())
    app = FastAPI()
    app.mount('/static', StaticFiles(directory=Path(__file__).parents[1] / 'static'), name='static')
    @app.middleware('http')
    async def boundary(request, call_next):
        if request.headers.get('host') != parsed.netloc:
            return PlainTextResponse('invalid host', status_code=403)
        response = await call_next(request)
        response.headers.update({'Cache-Control': 'no-store', 'X-Frame-Options': 'DENY', 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer', 'Content-Security-Policy': "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; frame-ancestors 'none'; form-action 'self'"})
        return response
    def static_asset(path):
        asset = Path(__file__).parents[1] / 'static' / path
        return '/static/' + path + '?v=' + hashlib.sha256(asset.read_bytes()).hexdigest()[:16]
    @app.get('/setup')
    async def form():
        return HTMLResponse(templates.get_template('login.html').render(setup_mode=True, bootstrap_grant=True, csrf_token=lambda: csrf, static_url=static_asset, url_for=lambda name: '/setup', error='', login_wallpaper=None))
    @app.post('/setup')
    async def submit(request: Request):
        nonlocal state
        if request.headers.get('origin') != exact_origin:
            return PlainTextResponse('invalid origin', status_code=403)
        now = time.monotonic()
        attempts[:] = [when for when in attempts if now - when < 60]
        if len(attempts) >= 10:
            return PlainTextResponse('rate limited', status_code=429)
        attempts.append(now)
        async def body():
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > LIMIT:
                    raise ValueError
            return data.decode('utf-8')
        try:
            raw = await asyncio.wait_for(body(), timeout=10)
            fields = parse_qs(raw, keep_blank_values=True, strict_parsing=True, max_num_fields=8)
            if any(len(value) != 1 for value in fields.values()):
                raise ValueError
            fields = {key: value[0] for key, value in fields.items()}
            if not hmac.compare_digest(fields.get('csrf_token', '').encode(), csrf.encode()) or not hmac.compare_digest(hashlib.sha256(fields.get('setup_code', '').encode()).digest(), code_digest):
                return PlainTextResponse('invalid authorization', status_code=403)
            if fields.get('password') != fields.get('password_confirm'):
                raise ValueError
            supplied = {'ENV_WEB_PASSPORT': fields.get('username', ''), 'ENV_WEB_PASSWORD': fields.get('password', '')}
            _settings(captured[1], supplied)
        except (ValueError, UnicodeError, BootstrapError, asyncio.TimeoutError):
            return PlainTextResponse('invalid setup form', status_code=400)
        with mutex:
            if state != 'armed' or time.monotonic() >= deadline:
                return PlainTextResponse('grant unavailable', status_code=409)
            state = 'consuming'
            work.put_nowait(supplied)
        # Disconnect/expiry cannot cancel the owner's storage operation.
        await asyncio.to_thread(result_ready.wait)
        return PlainTextResponse(result['message'], status_code=result['status'])
    server = uvicorn.Server(uvicorn.Config(app, host=bind, port=port, access_log=False, log_config=None))
    thread = threading.Thread(target=server.run, name='mediaflux-setup-http')
    thread.start()
    try:
        while True:
            try:
                supplied = work.get(timeout=.1)
                break
            except queue.Empty:
                with mutex:
                    if state == 'consuming':
                        continue
                    if not thread.is_alive() or time.monotonic() >= deadline:
                        state = 'expired'
                        raise BootstrapError('grant_expired_or_server_stopped')
        try:
            result.update(message=consume(supplied), status=200)
        except Exception:
            result.update(message='initialization failed; inspect storage before restarting', status=500)
            raise
        finally:
            with mutex:
                state = 'consumed'
            result_ready.set()
        return result['message']
    finally:
        server.should_exit = True
        result_ready.set()
        thread.join(timeout=15)
        if thread.is_alive():
            raise BootstrapError('setup_server_shutdown_timeout')
