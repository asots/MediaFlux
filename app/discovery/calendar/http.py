"""公开日历HTTP：精确HTTPS主机、公网DNS固定、有界请求，不读取浏览器凭据。"""
from __future__ import annotations

import asyncio
import json
import re
import socket
import threading
import time

import httpx

from app.indexers.http import FixedHostHttpClient
from app.indexers.providers.base import is_likely_challenge_page
from .models import SourceUnavailable


CALENDAR_USER_AGENT = "Mozilla/5.0 (compatible; MediaFluxCalendar/1.0)"
_PUBLIC_SESSION_ENDPOINT = "https://acs.youku.com/h5/mtop.youku.columbus.home.query/1.0/"
_PUBLIC_SESSION_HEADERS = {
    "accept": "application/json",
    "content-type": "application/x-www-form-urlencoded",
    "origin": "https://www.youku.com",
    "referer": "https://www.youku.com/ku/webcomic",
}
_ANONYMOUS_COOKIE = re.compile(
    r"_m_h5_tk=[A-Za-z0-9_%=+./-]{1,1024}(?:; _m_h5_tk_enc=[A-Za-z0-9_%=+./-]{1,1024})?"
)


def _public_session_headers(url, headers):
    if headers is None or headers == {}:
        return {}
    if url != _PUBLIC_SESSION_ENDPOINT or not isinstance(headers, dict) or len(headers) > 5:
        raise SourceUnavailable("公开会话请求头范围无效")
    clean = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise SourceUnavailable("公开会话请求头无效")
        name = key.lower()
        if name in clean:
            raise SourceUnavailable("公开会话请求头重复")
        if name == "cookie":
            if not _ANONYMOUS_COOKIE.fullmatch(value):
                raise SourceUnavailable("仅允许当前公开接口签发的匿名会话状态")
        elif _PUBLIC_SESSION_HEADERS.get(name) != value:
            raise SourceUnavailable("公开会话请求头无效")
        clean[name] = value
    return clean


# OS DNS 不能被协程取消；跨来源/服务重建最多保留三个尚未返回的解析。
# 槽位在真实 resolver 返回时才释放，超时重试不能不断创建新的挂起解析。
_DNS_SLOTS = threading.BoundedSemaphore(3)


def _bounded_resolver(resolver):
    resolve = resolver or FixedHostHttpClient._default_resolver

    def bounded(host, port, *args, **kwargs):
        slots = _DNS_SLOTS
        if not slots.acquire(blocking=False):
            raise socket.gaierror("calendar DNS capacity unavailable")
        try:
            return resolve(host, port, *args, **kwargs)
        finally:
            slots.release()

    return bounded


def _install_bounded_dns(loop):
    """只约束元资料独占 loop 的原生解析，不改进程 socket 或 ASGI loop。

    HTTPX 的直连/HTTP(S)/SOCKS 代理仍自行处理地址、Host、SNI 和认证；
    原生 getaddrinfo 的所有参数/结果原样传递。协程超时不释放 OS DNS 槽位。
    """
    resolve = _bounded_resolver(socket.getaddrinfo)

    async def getaddrinfo(host, port, **kwargs):
        return await asyncio.to_thread(resolve, host, port, **kwargs)

    loop.getaddrinfo = getaddrinfo


class CalendarHttp:
    def __init__(self, allowed_hosts, *, transport=None, resolver=None,
                 max_requests: int = 8, min_interval: float = 1.0):
        hosts = frozenset(allowed_hosts)
        self._client = FixedHostHttpClient(
            allowed_hosts=hosts, timeout_seconds=10,
            max_response_bytes=2 * 1024 * 1024, max_redirects=0,
            pin_resolved_address=True, resolver=_bounded_resolver(resolver),
            require_identity_encoding=True,
            transport=transport or httpx.AsyncHTTPTransport(
                retries=0,
                # 多逻辑host可能固定到同一IP，不能跨Host/SNI复用连接。
                limits=httpx.Limits(max_keepalive_connections=0) if len(hosts) > 1 else httpx.Limits(),
            ),
            user_agent=CALENDAR_USER_AGENT,
        )
        self._max_requests = max(1, min(int(max_requests), 8))
        self._min_interval = max(0.0, float(min_interval))
        self._requests = 0
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def request(self, method, url, *, params=None, json_body=None, headers=None):
        """统一原始请求与预算；最小间隔按准入计时，包含其后的DNS与响应耗时。"""
        extra_headers = _public_session_headers(url, headers)
        if extra_headers and method != "GET":
            raise SourceUnavailable("公开会话仅支持固定只读 GET")
        async with self._lock:
            if self._requests >= self._max_requests:
                raise SourceUnavailable("本轮公开数据请求已达到上限")
            wait = self._min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._requests += 1
            self._last_request = time.monotonic()
            self._client.clear_cookies()
            try:
                async with asyncio.timeout(12):
                    # 每次 GET/POST/匿名初始化都计入相同配额；不调用隐式重试或自动跳转。
                    response = await self._client._request(
                        method, url, params=params, json_body=json_body, max_redirects=0,
                        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "Accept-Encoding": "identity",
                                 **extra_headers},
                    )
            finally:
                # 显式匿名状态只活在调用方内存；固定IP的自动Jar既不能用于grant判断，也不保留其它Cookie。
                self._client.clear_cookies()
        if response.status_code != 200:
            raise SourceUnavailable(f"公开页面暂不可用（HTTP {response.status_code}）")
        if is_likely_challenge_page(response.body):
            raise SourceUnavailable("公开页面要求验证，未继续请求")
        return response

    async def get_text(self, url, *, params=None):
        response = await self.request("GET", url, params=params)
        content_type = str(response.headers.get("content-type", "")).lower()
        charset = "gb18030" if any(x in content_type for x in ("gbk", "gb2312", "gb18030")) else "utf-8"
        return response.body.decode(charset, errors="replace")

    async def get_json(self, url, *, params=None):
        return self._parse_json(await self.get_text(url, params=params))

    async def post_json(self, url, *, json_body):
        # 平台匿名排期接口使用只读 POST 查询；与 GET 共用全部安全边界和配额。
        if not isinstance(json_body, dict) or len(json.dumps(json_body)) > 16 * 1024:
            raise SourceUnavailable("公开排期查询参数无效")
        response = await self.request("POST", url, json_body=json_body)
        return self._parse_json(response.body.decode("utf-8", errors="replace"))

    @staticmethod
    def _parse_json(text):
        try:
            payload = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise SourceUnavailable("公开接口未返回有效结构化数据") from exc
        if not isinstance(payload, (dict, list)):
            raise SourceUnavailable("公开接口数据结构无效")
        return payload

    async def aclose(self):
        self._client.clear_cookies()
        await self._client.aclose()
