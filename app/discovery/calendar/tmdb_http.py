"""复用 TMDB 配置的可取消元数据读取；不让慢响应卡住免费事实刷新。"""
from __future__ import annotations

import asyncio
import json
import math
import time

import httpx

from app.clients.tmdb import TMDBClient, close_tmdb_client

from .models import SourceUnavailable


class CalendarTMDBClient:
    def __init__(self, *, settings_factory=TMDBClient, transport=None):
        # 统一复用既有 API URL/密钥/语言/代理解析，不引入第二套用户配置。
        settings = settings_factory()
        try:
            self.api_key = settings.api_key
            self.base_url = settings.base_url
            self.language = settings.language
            self.config_error = settings.config_error
            proxy = settings.session.proxies.get("https") or settings.session.proxies.get("http")
        finally:
            close_tmdb_client(settings)
        self._client = httpx.AsyncClient(
            proxy=proxy, transport=transport, timeout=httpx.Timeout(3, connect=2),
            follow_redirects=False, trust_env=False, headers={"Accept-Encoding": "identity"},
        )

    async def get(self, path, params, *, deadline_at, retries=0):
        if path != "/search/tv" or retries != 0 or self.config_error or not self.api_key:
            raise SourceUnavailable("TMDB 元资料配置或请求无效")
        if (not isinstance(deadline_at, (float, int)) or not math.isfinite(deadline_at)
                or not isinstance(params, dict)
                or set(params) - {"query", "page", "include_adult", "first_air_date_year"}):
            raise SourceUnavailable("TMDB 元资料请求无效")
        if deadline_at <= time.monotonic():
            raise TimeoutError("TMDB 元资料截止时间已到")
        query = {**params, "api_key": self.api_key, "language": self.language}
        limit = 2 * 1024 * 1024
        try:
            async with asyncio.timeout(max(0, deadline_at - time.monotonic())):
                async with self._client.stream("GET", f"{self.base_url}{path}", params=query) as response:
                    if response.status_code != 200:
                        raise SourceUnavailable("TMDB 元资料暂不可用")
                    if response.headers.get("content-encoding", "identity").lower() not in {"", "identity"}:
                        raise SourceUnavailable("TMDB 元资料编码未接受")
                    length = response.headers.get("content-length")
                    if length is not None and (not length.isascii() or not length.isdigit()
                                               or len(length) > 7 or int(length) > limit):
                        raise SourceUnavailable("TMDB 元资料大小超过上限")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(body) + len(chunk) > limit:
                            raise SourceUnavailable("TMDB 元资料大小超过上限")
                        body.extend(chunk)
                    payload = json.loads(body)
        except (httpx.HTTPError, ValueError):
            raise SourceUnavailable("TMDB 元资料暂不可用") from None
        if not isinstance(payload, dict):
            raise SourceUnavailable("TMDB 元资料结构无效")
        return payload

    async def aclose(self):
        await self._client.aclose()
