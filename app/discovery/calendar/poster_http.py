"""平台海报单次有界读取：签名之外仍执行固定主机、DNS/IP 与图片内容检查。"""
from __future__ import annotations

import asyncio

import httpx

from app.indexers.http import FixedHostHttpClient
from app.modules.image_payload import ImagePayloadError, SAFE_IMAGE_MIME_TYPES, matches_image_magic
from .http import _bounded_resolver
from .posters import canonical_platform_poster_key

MAX_PLATFORM_IMAGE_BYTES = 2 * 1024 * 1024


async def fetch_platform_poster(source: str, key: str, *, transport=None, resolver=None) -> tuple[bytes, str]:
    key = canonical_platform_poster_key(source, key)
    if not key:
        raise ImagePayloadError("invalid platform poster key")
    client = FixedHostHttpClient(
        allowed_hosts=frozenset({key.split("/", 1)[0]}), timeout_seconds=10,
        max_response_bytes=MAX_PLATFORM_IMAGE_BYTES, max_redirects=0,
        pin_resolved_address=True, resolver=_bounded_resolver(resolver), require_identity_encoding=True,
        transport=transport or httpx.AsyncHTTPTransport(retries=0),
        user_agent="Mozilla/5.0 (compatible; MediaFluxCalendar/1.0)",
    )
    try:
        async with asyncio.timeout(12):
            # 全新无凭据客户端；不使用带整请求重试的 get，不跟随任何跳转。
            response = await client._request("GET", "https://" + key, max_redirects=0, headers={
                "Accept": "image/jpeg,image/png,image/webp,image/avif,image/gif",
                "Accept-Encoding": "identity",
            })
        if response.status_code != 200:
            raise ImagePayloadError("upstream platform image failed")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if (content_type not in SAFE_IMAGE_MIME_TYPES
                or not matches_image_magic(content_type, response.body)):
            raise ImagePayloadError("invalid upstream platform image")
        return response.body, content_type
    finally:
        await client.aclose()
