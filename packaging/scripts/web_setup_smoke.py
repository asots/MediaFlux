"""在空卷容器中通过真实 Web 表单安装并登录；凭据不进入参数、环境或日志。"""
import http.cookiejar
import re
import secrets
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main() -> None:
    base = "http://127.0.0.1:1258"
    cookies = http.cookiejar.CookieJar()
    client = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies), NoRedirect())

    def request(path, data=None):
        body = None if data is None else urllib.parse.urlencode(data).encode()
        req = urllib.request.Request(base + path, data=body, headers={"Origin": base})
        try:
            return client.open(req, timeout=15)
        except urllib.error.HTTPError as response:
            if response.code not in (302, 303):
                raise
            return response

    def csrf(page):
        match = re.search(r'name="csrf_token" value="([^"]+)"', page)
        assert match is not None, "Setup page did not contain a CSRF token"
        return match.group(1)

    with request("/setup") as response:
        assert response.code == 200
        page = response.read().decode()
    assert 'name="setup_code"' not in page
    for name in ("mediaflux.db", "user.env", ".web-secret-key"):
        assert not (Path("/app/db") / name).exists(), "GET setup wrote installation storage"
    password = secrets.token_urlsafe(32)
    with request("/setup", {"username": "admin", "password": password,
                           "password_confirm": password, "csrf_token": csrf(page)}) as response:
        assert response.code == 303 and response.headers["Location"] == "/login"
    assert not any(cookie.name == "session" for cookie in cookies), "setup logged in automatically"
    with request("/login") as response:
        assert response.code == 200
        token = csrf(response.read().decode())
    with request("/login", {"username": "admin", "password": password, "csrf_token": token}) as response:
        assert response.code == 302 and response.headers["Location"] == "/"
    with request("/readyz") as response:
        assert response.code == 200


if __name__ == "__main__":
    main()
