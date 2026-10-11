from __future__ import annotations

import errno
import importlib
import os
import re
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.runtime_paths import RuntimePaths, configure_runtime_paths


class FirstRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.paths = RuntimePaths(
            program_dir=root / "program",
            data_dir=root / "data",
            config_dir=root / "config",
            cache_dir=root / "cache",
            log_dir=root / "logs",
            strm_dir=root / "strm-data",
            trash_dir=root / "trash",
        )
        self.environment = {
            key: os.environ.get(key)
            for key in (
                "ENV_WEB_PASSPORT",
                "ENV_WEB_PASSWORD",
                "WEB_HOST",
                "MEDIAFLUX_INITIALIZED",
                "MEDIAFLUX_TEST_MODE",
                "MEDIAFLUX_TEST_DB_PATH",
                "WEB_SECRET_KEY",
                "APP_ENV",
                "MEDIAFLUX_CONTAINER",
                "MEDIAFLUX_ALLOW_REMOTE_SETUP",
            )
        }
        for key in self.environment:
            os.environ.pop(key, None)
        os.environ["MEDIAFLUX_TEST_MODE"] = "0"
        configure_runtime_paths(self.paths)
        from app.security import clear_setup_failures
        clear_setup_failures("testclient")
        self._reload_runtime_modules()

    def tearDown(self) -> None:
        for key, value in self.environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        from app.security import clear_setup_failures
        clear_setup_failures("testclient")
        configure_runtime_paths(None)
        self._reload_runtime_modules(restore_default_database=True)
        self.temp.cleanup()

    def _reload_runtime_modules(
        self,
        paths: RuntimePaths | None = None,
        *,
        restore_default_database: bool = False,
    ) -> None:
        import app.config as config
        import app.database as database
        import app.modules.first_run as first_run

        importlib.reload(config)
        importlib.reload(database)
        if not restore_default_database:
            database.configure_database((paths or self.paths).database_path, test_mode=False)
        importlib.reload(first_run)
        first_run._reset_startup_state_for_tests()
        self.config = config
        self.database = database
        self.first_run = first_run

    def _new_client(self) -> TestClient:
        from app.main import create_app

        return TestClient(create_app(), raise_server_exceptions=False)

    @staticmethod
    def _csrf_token(response) -> str:
        match = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
        if not match:
            raise AssertionError("页面未输出 CSRF Token")
        return match.group(1)

    def _setup_page(self, client: TestClient):
        response = client.get("/setup")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertRegex(response.text, r"/static/js/lucide\.min\.js\?v=[0-9a-f]{16}")
        self.assertRegex(response.text, r"/static/js/app\.js\?v=[0-9a-f]{16}")
        return response


    def _initialize(self, username='admin', password='correct-horse'):
        from tests.test_setup_bootstrap_support import initialize_paths
        return initialize_paths(self.paths, {'ENV_WEB_PASSPORT':username, 'ENV_WEB_PASSWORD':password})

    def _web(self, username='admin', password='correct-horse', inspect=None):
        from tests.test_setup_bootstrap_support import initialize_web_paths
        result = initialize_web_paths(self.paths, username, password, inspect)
        self._assert_stable()
        return result

    @staticmethod
    def _origin():
        return {'Origin':'http://127.0.0.1:1258'}

    def _fields(self, page):
        return {'username':'admin','password':'correct-horse','password_confirm':'correct-horse',
                'csrf_token':self._csrf_token(page),
                'setup_code':(self.paths.runtime_dir / 'test-setup-grant').read_text()}

    def _assert_unpublished(self):
        self.assertFalse(self.paths.env_file.exists())
        self.assertFalse(self.paths.database_path.exists())
        self.assertFalse((self.paths.config_dir / '.web-secret-key').exists())

    def _fence(self):
        import json
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect('file:' + str(self.paths.database_path) + '?mode=ro', uri=True)) as conn:
            self.assertEqual(Path(conn.execute('PRAGMA database_list').fetchone()[2]), self.paths.database_path)
            return json.loads(conn.execute("SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'").fetchone()[0])

    def _assert_stable(self):
        self.assertEqual(self._fence()['phase'], 'stable')

    def _assert_failed_installation(self):
        from app.agent.feature_gate import RuntimeFenceError
        self.assertEqual(self._fence()['phase'], 'changing')
        before = self.paths.env_file.read_bytes()
        with self.assertRaises(RuntimeError):
            self._initialize()
        self._reload_runtime_modules()
        with self.assertRaises(RuntimeFenceError):
            self._new_client()
        self.assertEqual(self.paths.env_file.read_bytes(), before)
        self.assertEqual(self._fence()['phase'], 'changing')

    def _login(self, client, username='admin', password='correct-horse'):
        page = client.get('/login')
        self.assertEqual(page.status_code, 200)
        response = client.post('/login', data={'username':username,'password':password,'csrf_token':self._csrf_token(page)}, follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['location'], '/')

    def test_retired_api_and_ungranted_http_cannot_create_authority(self):
        with self.assertRaises(self.first_run.InitializationError):
            self.first_run.initialize_admin('admin', 'correct-horse')
        self._assert_unpublished()
        with self._new_client() as client:
            response = client.post('/setup', data={
                'username': 'admin', 'password': 'correct-horse',
                'password_confirm': 'correct-horse', 'csrf_token': 'invented',
                'setup_code': 'leftover-file-is-not-live-authority',
            })
            self.assertEqual(response.status_code, 403)
            self.assertNotIn('session', client.cookies)
        self._assert_unpublished()
        self._web()
        self._assert_stable()

    def test_empty_data_dir_guidance_never_creates_schema(self):
        self.assertFalse(self.paths.config_dir.exists())
        self.assertTrue(self.first_run.needs_initialization())
        with self._new_client() as client:
            response = client.get('/login', follow_redirects=False)
            self.assertEqual(client.get('/readyz').status_code, 503)
            page = client.get('/setup')
            self.assertEqual(page.headers['cache-control'], 'no-store')
            self.assertIn('initialize', page.text)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['location'], '/setup')
        self._assert_unpublished()

    def test_fresh_login_submit_cannot_initialize_or_authenticate(self):
        with self._new_client() as client:
            response = client.post('/login', data={'username':'admin','password':'correct-horse'}, follow_redirects=False)
            self.assertEqual(response.status_code, 405)
            self.assertNotIn('session', client.cookies)
        self._assert_unpublished()

    def test_incomplete_artifacts_never_gain_initialization_authority(self):
        cases = {'empty_user_env':'', 'ordinary_config':'TMDB_API_KEY=kept\nWEB_HOST=0.0.0.0\n', 'username_only':None, 'password_only':None, 'initialized_flag_only':None, 'database_only':None}
        for name, content in cases.items():
            with self.subTest(name=name):
                self.paths.config_dir.mkdir(parents=True, exist_ok=True)
                if content is not None:
                    self.paths.env_file.write_text(content)
                key = {'username_only':'ENV_WEB_PASSPORT','password_only':'ENV_WEB_PASSWORD','initialized_flag_only':'MEDIAFLUX_INITIALIZED'}.get(name)
                if key:
                    os.environ[key] = '1' if key == 'MEDIAFLUX_INITIALIZED' else 'partial-value'
                if name == 'database_only':
                    self.paths.database_path.parent.mkdir(parents=True, exist_ok=True)
                    self.paths.database_path.touch()
                before = self.paths.env_file.read_bytes() if self.paths.env_file.exists() else None
                self._reload_runtime_modules()
                self.assertTrue(self.first_run.needs_initialization())
                from app.main import _resolve_bind_host
                self.assertEqual(_resolve_bind_host(None), '127.0.0.1')
                if name == 'database_only':
                    from app.agent.feature_gate import RuntimeFenceError
                    with self.assertRaises(RuntimeFenceError):
                        self._new_client()
                    self.assertEqual(self.paths.database_path.read_bytes(), b'')
                else:
                    with self._new_client() as client:
                        self.assertEqual(client.get('/login', follow_redirects=False).headers['location'], '/setup')
                        self.assertEqual(client.post('/setup', data={}).status_code, 403)
                        self.assertEqual(client.get('/readyz').status_code, 503)
                    self.assertFalse(self.paths.database_path.exists())
                self.assertEqual(self.paths.env_file.read_bytes() if self.paths.env_file.exists() else None, before)
                if key:
                    os.environ.pop(key)
                self.paths.env_file.unlink(missing_ok=True)
                self.paths.database_path.unlink(missing_ok=True)

    def test_remote_first_run_binding_requires_explicit_opt_in(self):
        self._reload_runtime_modules()

        with self.assertRaises(self.first_run.UnsafeFirstRunBindingError):
            self.first_run.resolve_bind_host("0.0.0.0")

        os.environ["MEDIAFLUX_ALLOW_REMOTE_SETUP"] = "1"
        self.assertEqual(self.first_run.resolve_bind_host("0.0.0.0"), "0.0.0.0")

    def test_official_container_defaults_fresh_setup_to_all_interfaces(self):
        os.environ["MEDIAFLUX_CONTAINER"] = "1"
        self._reload_runtime_modules()

        self.assertEqual(self.first_run.resolve_bind_host(None), "0.0.0.0")
        self.assertEqual(self.first_run.resolve_bind_host("0.0.0.0"), "0.0.0.0")

    def test_only_explicitly_initialized_credentials_skip_setup(self):
        self.paths.config_dir.mkdir(parents=True, exist_ok=True)
        self.paths.env_file.write_text(
            "MEDIAFLUX_INITIALIZED=1\n"
            "ENV_WEB_PASSPORT=admin\n"
            "ENV_WEB_PASSWORD=strong-password\n",
            encoding="utf-8",
        )
        self._reload_runtime_modules()
        self.assertFalse(self.first_run.needs_initialization())
        self.paths.env_file.unlink()

        os.environ["MEDIAFLUX_INITIALIZED"] = "1"
        os.environ["ENV_WEB_PASSPORT"] = "deployed-admin"
        os.environ["ENV_WEB_PASSWORD"] = "deployed-password"
        self._reload_runtime_modules()
        self.assertFalse(self.first_run.needs_initialization())

    def test_credentials_without_initialization_marker_enter_setup(self):
        self.paths.config_dir.mkdir(parents=True, exist_ok=True)
        self.paths.env_file.write_text(
            "ENV_WEB_PASSPORT=admin\nENV_WEB_PASSWORD=strong-password\n",
            encoding="utf-8",
        )
        self._reload_runtime_modules()
        self.assertTrue(self.first_run.needs_initialization())

    def test_runtime_path_reset_uses_a_new_startup_snapshot(self):
        self.paths.config_dir.mkdir(parents=True)
        self.paths.env_file.write_text(
            "MEDIAFLUX_INITIALIZED=1\n"
            "ENV_WEB_PASSPORT=admin\nENV_WEB_PASSWORD=strong-password\n",
            encoding="utf-8",
        )
        self._reload_runtime_modules()
        self.assertFalse(self.first_run.needs_initialization())

        root = Path(self.temp.name) / "second"
        second_paths = RuntimePaths(
            program_dir=root / "program",
            data_dir=root / "data",
            config_dir=root / "config",
            cache_dir=root / "cache",
            log_dir=root / "logs",
            strm_dir=root / "strm-data",
            trash_dir=root / "trash",
        )
        configure_runtime_paths(second_paths)
        os.environ.pop("ENV_WEB_PASSPORT", None)
        os.environ.pop("ENV_WEB_PASSWORD", None)
        self._reload_runtime_modules(second_paths)

        self.assertTrue(self.first_run.needs_initialization())

    def test_live_setup_page_avoids_username_autofocus_to_prevent_mobile_keyboard_layout_shift(self):
        def inspect(page, client):
            self._setup_page(client)
            self.assertNotIn('id="setup-username" type="text" name="username" class="form-input" placeholder="admin" value="" autocomplete="username" autofocus', page.text)
        self._web(inspect=inspect)

    def test_live_setup_avoids_wallpaper_and_restarted_login_keeps_wallpaper(self):
        with patch('app.routes.auth.get_login_wallpaper', return_value=None) as wallpaper:
            self._web(inspect=lambda page, client: wallpaper.assert_not_called())
            self._reload_runtime_modules()
            with self._new_client() as client:
                self.assertEqual(client.get('/login').status_code, 200)
            wallpaper.assert_called_once()

    def test_live_setup_post_requires_csrf_without_consuming_grant(self):
        def inspect(page, client):
            fields = self._fields(page)
            fields.pop('csrf_token')
            response = client.post('/setup', data=fields, headers=self._origin())
            self.assertEqual(response.status_code, 403)
            self._assert_unpublished()
            self.assertIn('class="login-form setup-form"', self._setup_page(client).text)
        self._web(inspect=inspect)

    def test_live_setup_post_validates_fields_without_writing_configuration(self):
        invalid_forms = (
            {'username':'   '}, {'username':'admin\nname'},
            {'password':'short','password_confirm':'short'},
            {'password_confirm':'different-horse'},
            {'password':'bad\npassword','password_confirm':'bad\npassword'},
        )
        def inspect(page, client):
            for invalid in invalid_forms:
                with self.subTest(fields=tuple(invalid)):
                    response = client.post('/setup', data={**self._fields(page), **invalid}, headers=self._origin())
                    self.assertEqual(response.status_code, 400)
                    self._assert_unpublished()
                    self.assertIn('class="login-form setup-form"', self._setup_page(client).text)
        self._web(inspect=inspect)

    def test_explicit_initialize_rejects_control_characters_before_username_stripping(self):
        from app.modules.setup_bootstrap import BootstrapError
        with self.assertRaises(BootstrapError):
            self._initialize('admin\r\n')
        self._assert_unpublished()

    def test_explicit_initialize_rejects_all_splitline_separators_without_publishing_file(self):
        from app.modules.setup_bootstrap import BootstrapError
        for separator in ('\v','\f','\x1c','\x1d','\x1e','\x85','\u2028','\u2029'):
            with self.subTest(separator=repr(separator)):
                with self.assertRaises(BootstrapError):
                    self._initialize(f'admin{separator}name', f'correct{separator}horse')
                self._assert_unpublished()

    def test_explicit_initialize_uses_owned_paths_not_early_config_module_path(self):
        other_env = Path(self.temp.name) / 'early-config/config/user.env'
        with patch.object(self.config, 'ENV_FILE', other_env):
            self._initialize()
        self.assertTrue(self.paths.env_file.exists())
        self.assertFalse(other_env.exists())
        self._assert_stable()

    def test_failed_canonical_publish_requires_a_new_explicit_operation(self):
        with patch('app.config._publish_noreplace', side_effect=OSError('publish failed')):
            with self.assertRaises(OSError):
                self._initialize()
        self._assert_unpublished()
        self._initialize()
        self.assertTrue(self.paths.env_file.exists())
        self._assert_stable()

    def test_explicit_initialize_preserves_unrelated_config_and_replaces_partial_credentials(self):
        self.paths.config_dir.mkdir(parents=True)
        self.paths.env_file.write_text('TMDB_API_KEY=keep-me\nENV_WEB_PASSPORT=stale-only\n', encoding='utf-8')
        self._reload_runtime_modules()
        self._initialize('new-admin')
        values = self.config._read_env_file(self.paths.env_file)
        self.assertEqual(values['TMDB_API_KEY'], 'keep-me')
        self.assertEqual(values['ENV_WEB_PASSPORT'], 'new-admin')
        self.assertEqual(values['ENV_WEB_PASSWORD'], 'correct-horse')
        self.assertEqual(values['MEDIAFLUX_INITIALIZED'], '1')
        self._assert_stable()

    def test_explicit_initialize_refuses_changed_env_cas(self):
        import hashlib
        import json
        from app.modules.setup_bootstrap import run_new_installation, BootstrapError
        self.paths.ensure_writable_dirs()
        self.paths.env_file.write_text('TMDB_API_KEY=original\n')
        expected = hashlib.sha256(self.paths.env_file.read_bytes()).hexdigest()
        competitor = b'TMDB_API_KEY=competitor\nENV_WEB_PASSPORT=other\nENV_WEB_PASSWORD=other-password\n'
        self.paths.env_file.write_bytes(competitor)
        with tempfile.TemporaryFile() as payload:
            payload.write(json.dumps({'ENV_WEB_PASSPORT':'admin','ENV_WEB_PASSWORD':'correct-horse'}).encode())
            payload.seek(0)
            with self.assertRaisesRegex(BootstrapError, 'env_compare_and_swap_mismatch'):
                run_new_installation(self.paths, expected, settings_fd=payload.fileno())
        self.assertEqual(self.paths.env_file.read_bytes(), competitor)
        self.assertFalse(self.paths.database_path.exists())

    def test_complete_competitor_is_preserved_but_never_auto_admitted(self):
        competitor = b'ENV_WEB_PASSPORT=other\nENV_WEB_PASSWORD=other-password\nMEDIAFLUX_INITIALIZED=1\n'
        real_publish = self.config._publish_noreplace
        def competing_publish(temporary, target):
            if target == self.paths.env_file:
                target.write_bytes(competitor)
                raise FileExistsError(errno.EEXIST, 'exists', str(target))
            return real_publish(temporary, target)
        with patch('app.config._publish_noreplace', side_effect=competing_publish):
            with self.assertRaises(Exception):
                self._initialize()
        self.assertEqual(self.paths.env_file.read_bytes(), competitor)
        self._assert_failed_installation()

    def test_partial_competitor_is_preserved_and_failed_installation_is_not_admitted(self):
        competitor = b'TMDB_API_KEY=competitor\nENV_WEB_PASSPORT=partial\n'
        real_publish = self.config._publish_noreplace
        def competing_publish(temporary, target):
            if target == self.paths.env_file:
                target.write_bytes(competitor)
                raise FileExistsError(errno.EEXIST, 'exists', str(target))
            return real_publish(temporary, target)
        with patch('app.config._publish_noreplace', side_effect=competing_publish):
            with self.assertRaises(Exception):
                self._initialize()
        self.assertEqual(self.paths.env_file.read_bytes(), competitor)
        self._assert_failed_installation()

    def test_explicit_initialize_writes_private_configuration_without_network_binding(self):
        self._initialize('  admin  ')
        self._reload_runtime_modules()
        self.assertFalse(self.first_run.needs_initialization())
        if os.name != 'nt':
            self.assertEqual(stat.S_IMODE(self.paths.env_file.stat().st_mode), 0o600)
        values = self.config._read_env_file()
        self.assertEqual({key:value for key,value in values.items() if key != 'WEB_SECRET_KEY'}, {'ENV_WEB_PASSPORT':'admin','ENV_WEB_PASSWORD':'correct-horse','MEDIAFLUX_INITIALIZED':'1'})
        self.assertTrue(values['WEB_SECRET_KEY'])
        self._assert_stable()

    def test_explicit_initialize_preserves_password_whitespace_in_user_env(self):
        password = '  correct-$horse"  '
        self._initialize('admin', password)
        self.assertEqual(self.config._read_env_file()['ENV_WEB_PASSWORD'], password)
        self._assert_stable()
        self._reload_runtime_modules()
        with self._new_client() as client:
            self._login(client, password=password)

    def test_explicit_initialize_only_allows_one_concurrent_creator(self):
        self.paths.ensure_writable_dirs()
        barrier = threading.Barrier(3)
        errors = []
        successes = []
        def initialize(username):
            barrier.wait(timeout=5)
            try:
                self._initialize(username)
                successes.append(username)
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=initialize, args=(name,)) for name in ('one','two')]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=15)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(successes), 1)
        self.assertEqual(self.config._read_env_file()['ENV_WEB_PASSPORT'], successes[0])
        self._assert_stable()
        before = self.paths.env_file.read_bytes()
        with self.assertRaises(RuntimeError):
            self._initialize('three')
        self.assertEqual(self.paths.env_file.read_bytes(), before)

    def test_live_setup_requires_restart_login_without_writing_network_binding(self):
        def inspect(page, client):
            self.assertNotIn('name="allow_lan"', page.text)
            self.assertNotIn('session', client.cookies)
            # Submit the old forged allow_lan field through the real handler.
            real_post = client.post
            def submit_with_ignored_lan(url, **kwargs):
                kwargs['data'] = {**kwargs['data'], 'allow_lan': '1'}
                return real_post(url, **kwargs)
            client.post = submit_with_ignored_lan
        self._web(inspect=inspect)
        self._reload_runtime_modules()
        with self._new_client() as client:
            self.assertEqual(client.get('/', follow_redirects=False).headers['location'], str(client.base_url.join('/login')))
            self._login(client)
            self.assertEqual(client.get('/').status_code, 200)
        self.assertNotIn('WEB_HOST', self.config._read_env_file())

    def test_removed_network_applying_route_returns_not_found(self):
        with self._new_client() as client:
            response = client.get("/network-applying", follow_redirects=False)

        self.assertEqual(response.status_code, 404)

    def test_unicode_credentials_survive_live_setup_restart_logout_and_login(self):
        username, password = '管理员', '密码足够长度1234'
        self._web(username, password)
        self._reload_runtime_modules()
        with self._new_client() as client:
            self._login(client, username, password)
            dashboard = client.get('/')
            logout = client.post('/logout', data={'csrf_token':self._csrf_token(dashboard)}, follow_redirects=False)
            self.assertEqual(logout.status_code, 302)
            self.assertEqual(logout.headers['location'], '/login')
            self._login(client, username, password)

    def test_live_grant_rate_limit_is_operation_local_and_does_not_block_login(self):
        from app.modules.setup_bootstrap import BootstrapError
        from app.security import login_rate_limited
        from tests.test_setup_bootstrap_support import initialize_web_paths
        def inspect(page, client):
            fields = {**self._fields(page), 'setup_code':'invalid'}
            for _ in range(10):
                self.assertEqual(client.post('/setup', data=fields, headers=self._origin()).status_code, 403)
            self.assertEqual(client.post('/setup', data=self._fields(page), headers=self._origin()).status_code, 429)
            self.assertFalse(login_rate_limited('testclient'))
            self._assert_unpublished()
            raise BootstrapError('test finishes limited operation')
        with self.assertRaisesRegex(BootstrapError, 'test finishes limited operation'):
            initialize_web_paths(self.paths, 'admin', 'correct-horse', inspect)
        self._assert_unpublished()
        # The stale delivery file is not authority; only a new explicit live operation is.
        (self.paths.runtime_dir / 'test-setup-grant').unlink()
        self._web()
        self._reload_runtime_modules()
        with self._new_client() as client:
            self._login(client)


    def test_corrupt_non_utf8_config_is_preserved_and_all_initialization_refused(self):
        self.paths.config_dir.mkdir(parents=True)
        broken = b'\xff\xfeBROKEN'
        self.paths.env_file.write_bytes(broken)
        self._reload_runtime_modules()
        with self._new_client() as client:
            page = client.get('/setup')
            self.assertEqual(page.status_code, 200)
            self.assertIn('user.env', page.text)
            self.assertIn('恢复', page.text)
            self.assertEqual(client.post('/setup', data={'username':'admin','password':'correct-horse'}).status_code, 403)
        with self.assertRaises(UnicodeDecodeError):
            self._initialize()
        self.assertEqual(self.paths.env_file.read_bytes(), broken)
        self.assertFalse(self.paths.database_path.exists())

    @unittest.skipIf(os.name == "nt", "POSIX 链接安全 Web 回归")
    def test_symlink_config_never_initializes_or_touches_target(self):
        self.paths.config_dir.mkdir(parents=True)
        external = Path(self.temp.name) / 'outside.env'
        external.write_text('TMDB_API_KEY=outside\n', encoding='utf-8')
        self.paths.env_file.symlink_to(external)
        self._reload_runtime_modules()
        with self._new_client() as client:
            page = client.get('/setup')
            self.assertIn('user.env', page.text)
            self.assertEqual(client.post('/setup', data={}).status_code, 403)
        with self.assertRaises(self.config.UnsafeConfigFileError):
            self._initialize()
        self.assertEqual(external.read_text(encoding='utf-8'), 'TMDB_API_KEY=outside\n')
        self.assertTrue(self.paths.env_file.is_symlink())
        self.assertFalse(self.paths.database_path.exists())

    def test_permission_failure_never_promotes_competitor_credentials(self):
        self.paths.config_dir.mkdir(parents=True)
        self.paths.env_file.write_text('TMDB_API_KEY=keep-me\nENV_WEB_PASSPORT=partial\n')
        competitor = b'ENV_WEB_PASSPORT=other\nENV_WEB_PASSWORD=other-password\nWEB_SECRET_KEY=competitor-secret\nMEDIAFLUX_INITIALIZED=1\n'
        calls = []
        def fail_permissions(path):
            calls.append(path)
            self.paths.env_file.write_bytes(competitor)
            raise self.config.AtomicPublishError('private permissions failed')
        with patch('app.config._apply_private_permissions', side_effect=fail_permissions):
            with self.assertRaises(self.config.AtomicPublishError):
                self._initialize()
        self.assertTrue(calls)
        self.assertEqual(self.paths.env_file.read_bytes(), competitor)
        self._assert_failed_installation()

    def test_successful_explicit_publish_verifies_and_syncs_physical_env(self):
        events = []
        real_fsync, real_read = os.fsync, os.read
        def is_env(fd):
            if not self.paths.env_file.exists():
                return False
            current, target = os.fstat(fd), self.paths.env_file.stat()
            return (current.st_dev, current.st_ino) == (target.st_dev, target.st_ino)
        def observe_read(fd, size):
            payload = real_read(fd, size)
            if is_env(fd):
                events.append(('read', payload))
            return payload
        def observe_sync(fd):
            result = real_fsync(fd)
            if is_env(fd):
                events.append(('sync', None))
            return result
        with patch('os.read', side_effect=observe_read), patch('os.fsync', side_effect=observe_sync):
            self._initialize()
        payload = self.paths.env_file.read_bytes()
        self.assertIn(('read', payload), events)
        read_index = events.index(('read', payload))
        self.assertIn(('sync', None), events[read_index + 1:])
        self._assert_stable()
        self._reload_runtime_modules()
        self.assertFalse(self.first_run.needs_initialization())
        self.assertTrue(self.paths.env_file.exists())


if __name__ == "__main__":
    unittest.main()
