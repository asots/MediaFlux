"""匿名豆瓣 suggest：固定公开主机，复用日历安全 HTTP，不读取站点凭据。"""
from __future__ import annotations

import asyncio
import math
import time

import httpx

from .http import CalendarHttp
from .models import SourceUnavailable

_MAX_BYTES = 2 * 1024 * 1024


class _AnonymousTransport(httpx.AsyncBaseTransport):
    """连接池可复用，但不转发 Cookie/Auth，也不接受压缩后的无界膨胀。"""

    def __init__(self, transport):
        self.transport = transport

    async def handle_async_request(self, request):
        for header in ("cookie", "authorization", "proxy-authorization"):
            request.headers.pop(header, None)
        request.headers["Accept"] = "application/json"
        request.headers["Accept-Encoding"] = "identity"
        response = await self.transport.handle_async_request(request)
        response.headers.pop("set-cookie", None)  # 匿名会话也不保留上游 Cookie。
        encoding = response.headers.get("content-encoding", "identity").lower()
        length = response.headers.get("content-length")
        if (encoding not in {"", "identity"}
                or (length is not None and (not length.isascii() or not length.isdigit()
                                             or len(length) > 7 or int(length) > _MAX_BYTES))):
            await response.aclose()
            raise SourceUnavailable("豆瓣元资料编码或大小无效")
        return response

    async def aclose(self):
        await self.transport.aclose()


class CalendarDoubanClient:
    def __init__(self, *, transport=None, resolver=None, max_requests=8, min_interval=2):
        self._http = CalendarHttp(
            {"movie.douban.com"}, resolver=resolver, max_requests=max_requests,
            min_interval=min_interval,
            transport=_AnonymousTransport(transport if transport is not None
                                          else httpx.AsyncHTTPTransport(retries=0)),
        )

    async def suggest(self, title, *, deadline_at):
        if (not isinstance(title, str) or not title.strip() or len(title) > 200
                or any(ord(char) < 32 for char in title)
                or not isinstance(deadline_at, (float, int)) or not math.isfinite(deadline_at)):
            raise SourceUnavailable("豆瓣元资料查询无效")
        if deadline_at <= time.monotonic():
            raise TimeoutError("豆瓣元资料截止时间已到")
        try:
            # 包括限频等待、DNS、连接、慢滴响应；不是 requests 的软 read timeout。
            async with asyncio.timeout(max(0, deadline_at - time.monotonic())):
                payload = await self._http.get_json(
                    "https://movie.douban.com/j/subject_suggest", params={"q": title},
                )
            # 不使用 [:20]，被截断或损坏的完整候选集不能证明唯一匹配。
            if not isinstance(payload, list) or len(payload) > 20 or any(not isinstance(row, dict) for row in payload):
                raise SourceUnavailable("豆瓣元资料结构无效")
            return payload
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001 -- 安全 HTTP 可能抛多类错误，统一无凭据边界。
            # 上游异常不得把请求、环境、Cookie 或认证参数带进卡片/日志。
            raise SourceUnavailable("豆瓣元资料暂不可用") from None

    async def aclose(self):
        await self._http.aclose()
