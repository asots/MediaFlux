"""真实单一安装夹具；不注入 schema、fence 或测试安装分支。"""
import os
import secrets


def initialize_paths(paths, settings=None):
    """可预配业务配置，但所有写入均由唯一安装事务完成。"""
    from app import config
    from app.modules.setup_bootstrap import initialize_installation

    expected, values = config.read_env_snapshot_no_recovery(paths.env_file)
    values.setdefault('ENV_WEB_PASSPORT', 'operator')
    values.setdefault('ENV_WEB_PASSWORD', 'test-bootstrap-password')
    values.setdefault('WEB_SECRET_KEY', os.environ.get('WEB_SECRET_KEY') or secrets.token_urlsafe(48))
    values.update(settings or {})
    values['MEDIAFLUX_INITIALIZED'] = '1'
    return initialize_installation(paths, values, expected_env=expected)


def initialize_web_paths(paths, username, password, inspect_page=None):
    """通过普通应用真实完成表单安装，同一客户端立即登录，不替换服务器。"""
    import re
    from fastapi.testclient import TestClient
    from app.main import create_app

    with TestClient(create_app(), base_url='http://127.0.0.1:1258') as client:
        page = client.get('/setup')
        assert page.status_code == 200
        assert 'name="setup_code"' not in page.text
        assert not paths.env_file.exists()
        assert not paths.database_path.exists()
        assert not (paths.config_dir / '.web-secret-key').exists()
        search_result = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        assert search_result is not None
        token = search_result.group(1)
        assert token in client.get('/setup').text
        if inspect_page:
            inspect_page(page, client)
        page = client.get('/setup')
        search_result_2 = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        assert search_result_2 is not None
        token = search_result_2.group(1)
        response = client.post('/setup', data={
            'username': username, 'password': password,
            'password_confirm': password, 'csrf_token': token,
        }, headers={'Origin': str(client.base_url).rstrip('/')}, follow_redirects=False)
        assert response.status_code in (302, 303), response.text
        assert response.headers['location'] == '/login'
        assert not client.cookies.get('session')
        page = client.get('/login')
        assert page.status_code == 200
        search_result_3 = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
        assert search_result_3 is not None
        token = search_result_3.group(1)
        response = client.post('/login', data={
            'username': username, 'password': password, 'csrf_token': token,
        }, follow_redirects=False)
        assert response.status_code == 302, response.text
        assert response.headers['location'] == '/'
        assert client.get('/readyz').status_code == 200
