"""Fresh exec acceptance worker. No pytest/test-package bootstrap imports."""
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys
import time


def main():
    root = Path(sys.argv[1]).resolve()
    mode = sys.argv[2]
    source = Path(__file__).resolve().parents[1]
    for key in list(os.environ):
        if key.startswith(('MEDIAFLUX_', 'ENV_WEB_', 'WEB_', 'APP_ENV', 'AGENT_', 'PYTHON')):
            os.environ.pop(key)
    os.environ.update(MEDIAFLUX_DATA_DIR=str(root), MEDIAFLUX_CONFIG_DIR=str(root),
                      MEDIAFLUX_CACHE_DIR=str(root/'cache'), MEDIAFLUX_LOG_DIR=str(root/'logs'),
                      MEDIAFLUX_STRM_DIR=str(root/'strm'), MEDIAFLUX_DISABLE_FILE_LOGGING='1',
                      MEDIAFLUX_CONTAINER='1', AGENT_LLM_ENABLED='0', APP_ENV='production')
    assert not any(n == 'app' or n.startswith('app.') for n in sys.modules)
    def audit(event, args):
        if event in ('socket.connect', 'socket.bind', 'socket.sendto', 'socket.sendmsg') and args[0].family != socket.AF_UNIX:
            raise AssertionError('exec inet denied')
        if event in ('socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyaddr', 'socket.getnameinfo'):
            raise AssertionError('exec DNS denied')
    sys.addaudithook(audit)
    for action in (lambda: socket.getaddrinfo('localhost', 80), lambda: socket.socket().bind(('127.0.0.1', 0))):
        try:
            action()
        except AssertionError:
            pass
        else:
            raise AssertionError('missing exec guard')
    sys.path.insert(0, str(source))
    os.chdir(source)
    if mode == 'guidance':
        from fastapi.testclient import TestClient
        from app.main import create_app
        with TestClient(create_app()) as client:
            assert client.get('/readyz').status_code == 503
            assert client.post('/setup',data={}).status_code == 403
        assert not (root/'mediaflux.db').exists()
        assert not (root/'user.env').exists()
        code = 0
    elif mode == 'refuse':
        from app.agent.feature_gate import RuntimeFenceError
        try:
            from app.main import create_app
            create_app()
        except RuntimeFenceError:
            code = 0
        else:
            raise AssertionError('changing storage admitted')
        assert not (root/'user.env').exists()
    elif mode == 'login':
        from fastapi.testclient import TestClient
        from app.main import create_app
        from app import database, config
        from app.agent.feature_gate import read_runtime_fence
        with TestClient(create_app()) as client:
            page = client.get('/login')
            token = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
            response = client.post('/login', data={'username':'operator', 'password':'test-bootstrap-password','csrf_token':token},follow_redirects=False)
            assert response.status_code == 302 and response.headers['location'] == '/'
            assert client.get('/readyz').status_code == 200
            with database.get_conn() as conn:
                assert Path(conn.execute('PRAGMA database_list').fetchone()[2]).resolve() == root/'mediaflux.db'
                assert read_runtime_fence(conn).phase == 'stable'
            assert config.ENV_FILE == root/'user.env'
        code = 0
    else:
        if mode in ('before', 'after'):
            from app import config
            original = config._publish_noreplace
            def publish(stage, target):
                if target == root/'mediaflux.db':
                    if mode == 'after':
                        original(stage, target)
                    (root/'boundary').write_text(mode)
                    # Parent kills at an actual publication boundary, no timing guess.
                    while True:
                        time.sleep(.05)
                return original(stage, target)
            config._publish_noreplace = publish
        if mode == 'race':
            (root/('ready-'+str(os.getpid()))).write_text('ready')
            end = time.monotonic()+15
            while not (root/'go').exists():
                assert time.monotonic() < end
                time.sleep(.01)
        sys.argv = [str(source/'mediaflux.py'),'initialize','--data-dir',str(root),'--confirm-new-installation','--expect-env-sha256','absent','--settings-fd','0']
        try:
            runpy.run_path(str(source/'mediaflux.py'),run_name='__main__')
        except SystemExit as exc:
            code = exc.code
        else:
            code = 0
    origins={n:str(m.__file__) for n,m in sys.modules.items() if (n=='app' or n.startswith('app.')) and getattr(m,'__file__',None)}
    assert origins and all(Path(p).resolve().is_relative_to(source) for p in origins.values())
    print('WORKER_RESULT='+json.dumps({'code':code,'pid':os.getpid(),'origins_sha256':hashlib.sha256(json.dumps(origins,sort_keys=True).encode()).hexdigest(),'guard':'all-inet-and-dns-denied','mode':mode}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
