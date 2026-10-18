"""普通应用的真实 Web 安装生命周期；不使用授权服务或替代安装实现。"""
import json
import sqlite3

from tests.test_setup_bootstrap import cold_run


def test_concurrent_web_requests_publish_once_and_preserve_the_winner(tmp_path):
    def action():
        import re
        import threading
        from fastapi.testclient import TestClient
        from app import config
        from app.main import create_app
        from app.modules import backup, first_run
        entered, second_entered, release = threading.Event(), threading.Event(), threading.Event()
        original_sync, original_needs, original_publish = backup._fsync_directory, first_run.needs_initialization, config._publish_noreplace
        publications, results = [], []
        def sync(directory):
            if directory == tmp_path / 'db' and (directory / 'mediaflux.db').exists():
                entered.set()
                assert release.wait(8), 'owner release deadline'
            original_sync(directory)
        def needs_initialization():
            if entered.is_set() and not release.is_set():
                second_entered.set()
            return original_needs()
        def publish(temporary, target):
            publications.append(str(target))
            return original_publish(temporary, target)
        backup._fsync_directory, first_run.needs_initialization, config._publish_noreplace = sync, needs_initialization, publish
        with TestClient(create_app()) as client:
            page = client.get('/setup')
            search_result = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
            assert search_result is not None
            token = search_result.group(1)
            fields = {'username':'operator', 'password':'test-bootstrap-password', 'password_confirm':'test-bootstrap-password', 'csrf_token':token}
            def post(username):
                try:
                    response = client.post('/setup', data={**fields, 'username':username}, follow_redirects=False)
                    results.append((response.status_code, response.headers.get('location')))
                except Exception as exc:  # noqa: BLE001 - 收集并发请求线程的任意失败
                    results.append(type(exc).__name__)
            first = threading.Thread(target=post, args=('operator',))
            second = threading.Thread(target=post, args=('competitor',))
            first.start()
            try:
                assert entered.wait(8)
                second.start()
                assert second_entered.wait(8)
            finally:
                release.set()
                first.join(timeout=8)
                if second.ident is not None:
                    second.join(timeout=8)
            assert not first.is_alive() and not second.is_alive()
            assert publications.count(str(tmp_path / 'db/mediaflux.db')) == 1
            assert publications.count(str(tmp_path / 'config/user.env')) == 1
            assert config.web_credentials() == ('operator', 'test-bootstrap-password')
            assert client.get('/readyz').status_code == 200
            assert len(results) == 2
            assert all(status in (302, 303) and location == '/login' for status, location in results)
            assert any(status == 303 for status, _ in results)
            return 'installed-once'
    assert cold_run(tmp_path, action) == {'result': 'installed-once'}
    with sqlite3.connect(tmp_path / 'db/mediaflux.db') as conn:
        assert json.loads(conn.execute("SELECT value FROM settings_kv WHERE key='agent.runtime_fence.v1'").fetchone()[0])['phase'] == 'stable'


def test_web_installation_starts_and_stops_real_background_services_once(tmp_path):
    def action():
        import re
        from fastapi.testclient import TestClient
        from app import main
        from app.agent.feature_gate import read_runtime_fence
        events = []
        start, stop = main.start_background_services, main.stop_background_services
        def start_services():
            events.append('start')
            return start()
        def stop_services():
            events.append('stop')
            return stop()
        main.start_background_services, main.stop_background_services = start_services, stop_services
        app = main.create_app(start_background=True)
        with TestClient(app) as client:
            page = client.get('/setup')
            search_result_2 = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
            assert search_result_2 is not None
            token = search_result_2.group(1)
            assert events == []
            assert client.get('/readyz').status_code == 503
            fields = {'username':'operator', 'password':'test-bootstrap-password', 'password_confirm':'test-bootstrap-password', 'csrf_token':token}
            response = client.post('/setup', data=fields, follow_redirects=False)
            assert response.status_code == 303 and response.headers['location'] == '/login'
            assert events == ['start']
            assert app.state.ready and client.get('/readyz').status_code == 200
            assert client.portal is not None
            client.portal.call(app.state.activate_runtime)
            assert events == ['start']
            page = client.get('/login')
            search_result_3 = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
            assert search_result_3 is not None
            token = search_result_3.group(1)
            response = client.post('/login', data={'username':'operator', 'password':'test-bootstrap-password', 'csrf_token':token}, follow_redirects=False)
            assert response.status_code == 302
            assert client.get('/').status_code == 200
            main.config.set_and_save({'AGENT_LLM_MODEL':'synthetic-after-setup'})
            with main.database.get_conn() as conn:
                row = read_runtime_fence(conn)
                assert row.phase == 'stable'
                assert row.config_digest == main.config.capture_runtime_config_snapshot().config_digest
            assert client.get('/readyz').status_code == 200
        assert not app.state.ready and app.state.activate_runtime is None
        return events
    assert cold_run(tmp_path, action) == {'result': ['start', 'stop']}


def test_cancelled_request_does_not_cancel_the_storage_transaction_or_allow_reinstallation(tmp_path):
    def action():
        import asyncio
        import re
        import threading
        import httpx
        from fastapi.testclient import TestClient
        from app import config, database
        from app.agent.feature_gate import read_runtime_fence
        from app.main import create_app
        from app.modules import backup, first_run
        entered, release, completed = threading.Event(), threading.Event(), threading.Event()
        original_sync, original_initialize = backup._fsync_directory, first_run.initialize_admin
        def sync(directory):
            if directory == tmp_path / 'db' and (directory / 'mediaflux.db').exists():
                entered.set()
                assert release.wait(8), 'owner release deadline'
            original_sync(directory)
        def initialize(*args, **kwargs):
            try:
                return original_initialize(*args, **kwargs)
            finally:
                completed.set()
        backup._fsync_directory, first_run.initialize_admin = sync, initialize
        app = create_app()
        with TestClient(app) as client:
            page = client.get('/setup')
            search_result_4 = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
            assert search_result_4 is not None
            token = search_result_4.group(1)
            fields = {'username':'operator', 'password':'test-bootstrap-password', 'password_confirm':'test-bootstrap-password', 'csrf_token':token}
            async def disconnect():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=str(client.base_url), cookies=dict(client.cookies)) as browser:
                    task = asyncio.create_task(browser.post('/setup', data=fields))
                    try:
                        assert await asyncio.to_thread(entered.wait, 8), 'transaction handshake deadline'
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        else:
                            raise AssertionError('request was not cancelled')
                    finally:
                        release.set()
                    assert await asyncio.to_thread(completed.wait, 8), 'storage completion deadline'
            assert client.portal is not None
            client.portal.call(disconnect)
            with database.get_conn() as conn:
                assert read_runtime_fence(conn).phase == 'stable'
            before = (tmp_path / 'config/user.env').read_bytes(), (tmp_path / 'db/mediaflux.db').stat().st_ino
            try:
                first_run.initialize_admin('replacement', 'replacement-password')
            except first_run.InitializationError:
                pass
            else:
                raise AssertionError('cancelled request allowed administrator replacement')
            assert before == ((tmp_path / 'config/user.env').read_bytes(), (tmp_path / 'db/mediaflux.db').stat().st_ino)
            assert config.web_credentials() == ('operator', 'test-bootstrap-password')
        return 'storage-committed-without-replay'
    assert cold_run(tmp_path, action) == {'result': 'storage-committed-without-replay'}
