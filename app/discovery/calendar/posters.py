"""平台原图只用于日历展示；固定资产路径，不作为元资料或收藏身份。"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

# 仅收录公开排期响应中核验过的精确 CDN 主机，不接受域名后缀/通配符。
PLATFORM_IMAGE_HOSTS: dict[str, frozenset[str]] = {
    "tencent": frozenset({"vcover-hz-pic.puui.qpic.cn"}),
    "iqiyi": frozenset({
        "pic0.iqiyipic.com", "pic1.iqiyipic.com", "pic2.iqiyipic.com", "pic3.iqiyipic.com",
        "pic4.iqiyipic.com", "pic5.iqiyipic.com", "pic6.iqiyipic.com", "pic7.iqiyipic.com",
        "pic8.iqiyipic.com", "pic9.iqiyipic.com",
    }),
    "youku": frozenset({"liangcang-material.alicdn.com", "m.ykimg.com"}),
}
_SAFE_PATH = re.compile(r"/[A-Za-z0-9._~!(),=/-]+")
_ASSET_PATHS = {
    "vcover-hz-pic.puui.qpic.cn": re.compile(r"/vcover_hz_pic/0/[A-Za-z0-9_-]{1,128}/[0-9]{1,4}"),
    **dict.fromkeys(PLATFORM_IMAGE_HOSTS["iqiyi"], re.compile(r"/image/[A-Za-z0-9_./-]+\.(?:jpg|jpeg|png|webp)")),
    "liangcang-material.alicdn.com": re.compile(r"/prod/upload/[0-9a-f]{32}(?:\.webp)?\.(?:jpg|jpeg|png|webp)"),
    "m.ykimg.com": re.compile(r"/[0-9A-F]{32}"),
}


def canonical_platform_poster_key(source: str, value: object) -> str:
    """验证已存储的 host/path；失败仅丢图片，不丢平台排期。"""
    if not isinstance(source, str) or not isinstance(value, str) or not value or len(value) > 1024:
        return ""
    host, separator, path = value.partition("/")
    if (not separator or host not in PLATFORM_IMAGE_HOSTS.get(source, ())
            or not _SAFE_PATH.fullmatch("/" + path)
            or host not in _ASSET_PATHS or not _ASSET_PATHS[host].fullmatch("/" + path)
            or any(part in {"", ".", ".."} for part in path.split("/"))):
        return ""
    return value


def platform_poster_key(source: str, value: object) -> str:
    """从来源明确给出的资产 URL 提取 key；不转发参数、凭据或 HTTP。"""
    if not isinstance(source, str) or not isinstance(value, str) or len(value) > 2048 or re.search(r"[\s\\\x00-\x1f\x7f]", value):
        return ""
    try:
        url = urlsplit("https:" + value if value.startswith("//") else value)
        if (url.scheme not in {"http", "https"} or url.netloc not in PLATFORM_IMAGE_HOSTS.get(source, ())
                or url.username or url.password or url.port or url.fragment):
            return ""
        # 只读取路径标识的公开 HTTPS 原图；不保存/转发查询中的变换或凭据，失败不另行鉴权。
        return canonical_platform_poster_key(source, url.netloc + url.path)
    except ValueError:
        return ""
