"""Approved explicit-operation prerequisites; no synthetic fence or test mode."""
import hashlib
import json
import tempfile


def initialize_paths(paths, settings=None):
    from app.modules.setup_bootstrap import run_new_installation
    paths.ensure_writable_dirs()
    expected = hashlib.sha256(paths.env_file.read_bytes()).hexdigest() if paths.env_file.exists() else 'absent'
    with tempfile.TemporaryFile() as payload:
        payload.write(json.dumps(settings or {}).encode())
        payload.seek(0)
        return run_new_installation(paths, expected, settings_fd=payload.fileno())


def initialize_web_paths(paths, username, password, inspect_page=None):
    """Real ASGI grant consumption; only the TCP server is substituted."""
    import re
    import uvicorn
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from app.modules.setup_bootstrap import run_new_installation
    paths.ensure_writable_dirs()
    grant = paths.runtime_dir / 'test-setup-grant'
    failures = []
    def serve(server):
        try:
            with TestClient(server.config.app, base_url='http://127.0.0.1:1258') as client:
                page = client.get('/setup')
                assert page.status_code == 200
                assert not paths.env_file.exists()
                assert not (paths.config_dir / '.web-secret-key').exists()
                if inspect_page:
                    inspect_page(page, client)
                token = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
                again = client.get('/setup')
                assert token in again.text  # live memory-only CSRF material
                response = client.post('/setup', data={
                    'username': username, 'password': password,
                    'password_confirm': password, 'csrf_token': token,
                    'setup_code': grant.read_text(),
                }, headers={'Origin': 'http://127.0.0.1:1258'})
                assert response.status_code == 200
                assert 'restart' in response.text
                assert 'session' not in client.cookies
        except BaseException as exc:
            failures.append(exc)
    with patch.object(uvicorn.Server, 'run', serve):
        try:
            result = run_new_installation(paths, 'absent', web=True, grant_file=grant)
        except Exception:
            if failures:
                raise failures[0]
            raise
    if failures:
        raise failures[0]
    return result
