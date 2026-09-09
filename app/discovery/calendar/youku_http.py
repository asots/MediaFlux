"""官方 WEBCOMIC 公开 H5 协议：仅本次调用的匿名 grant，至多两个有界 GET。"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from email.utils import parsedate_to_datetime
from http.cookies import SimpleCookie

from .http import CALENDAR_USER_AGENT
from .models import SourceUnavailable

_URL = "https://acs.youku.com/h5/mtop.youku.columbus.home.query/1.0/"
_HOST = "acs.youku.com"
_PATH = "/h5/mtop.youku.columbus.home.query/1.0/"
_API = "mtop.youku.columbus.home.query"
_APP_KEY = "24679788"  # 官方 Page 的公开客户端标识，不是账号凭据。
_COOKIE_NAMES = ("_m_h5_tk", "_m_h5_tk_enc")
_COOKIE_DOMAINS = {_HOST, "." + _HOST, "youku.com", ".youku.com"}
_COOKIE_VALUE = re.compile(r"[A-Za-z0-9_%=+./-]{1,1024}")
_COOKIE_SPLIT = re.compile(r",(?= *[!#$%&'*+.^_`|~0-9A-Za-z-]+=)")
_COOKIE_ATTRS = {"domain", "path", "expires", "max-age", "secure", "httponly", "samesite"}
_TOKEN_CODES = {"FAIL_SYS_TOKEN_EMPTY", "FAIL_SYS_TOKEN_EXOIRED"}  # 官方 SDK 的拼写。
_ERROR = "优酷公开排期暂不可用，未继续请求"


def _compact(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


# 与官方普通 pcweb 页面及已成功的匿名探测一致；不读取配置、浏览器或账号状态。
_DATA = _compact({
    "ms_codes": "2019061000",
    "params": _compact({
        "debug": 0, "utdid": "empty_cna", "appPackageKey": "com.youku.pcweb",
        "appPackageId": "com.youku.pcweb", "ip": "127.0.0.1", "reqSubNode": 0,
        "gray": "0", "pageNo": 1, "bizKey": "kuflix_pc_home", "showNodeList": 0,
        "nodeKey": "WEBCOMIC", "appKey": _APP_KEY, "bizContext": "{}",
    }),
    "system_info": _compact({
        "appPackageKey": "com.youku.pcweb", "appPackageId": "com.youku.pcweb",
        "device": "pcweb", "os": "pcweb", "ver": "1.0.0.0",
        "userAgent": CALENDAR_USER_AGENT, "guid": "1590141704165YXe", "young": 0,
        "brand": "", "network": "", "ouid": "", "idfa": "", "scale": "",
        "operator": "", "resolution": "", "pid": "", "childGender": 0,
        "zx": 0, "zx_list": "", "appkey": _APP_KEY, "disableUserRec": "0",
    }),
})


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError
    return number


def _invalid_constant(_value):
    raise ValueError


def _envelope(response):
    if (response.status_code != 200 or not isinstance(response.body, bytes)
            or len(response.body) > 2 * 1024 * 1024
            or response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json"):
        raise ValueError
    payload = json.loads(
        response.body.decode("utf-8"), object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant, parse_float=_finite_float,
    )
    if (not isinstance(payload, dict) or not isinstance(payload.get("data"), dict)
            or payload.get("api", _API) != _API or payload.get("v", "1.0") != "1.0"):
        raise ValueError
    ret = payload.get("ret")
    # 拒绝混合SUCCESS+error、重复/未知ret；绝不把错误描述中的SUCCESS当成功。
    if not isinstance(ret, list) or len(ret) != 1 or not isinstance(ret[0], str) or len(ret[0]) > 512:
        raise ValueError
    code = ret[0].split("::", 1)[0]
    if code != "SUCCESS" and (code not in _TOKEN_CODES or payload["data"]):
        raise ValueError
    return payload, code


def _cookie_expiry(morsel, now):
    expiry = None
    if morsel["expires"]:
        date = parsedate_to_datetime(morsel["expires"])
        if date.tzinfo is None:
            raise ValueError
        expiry = date.timestamp()
    if morsel["max-age"]:
        age = morsel["max-age"]
        if not re.fullmatch(r"[0-9]{1,10}", age) or not 0 < int(age) <= 2**31 - 1:
            raise ValueError
        expiry = now + int(age)  # 标准cookie语义：Max-Age优先于Expires。
    if expiry is not None and (not math.isfinite(expiry) or expiry <= now):
        raise ValueError
    return expiry


def _anonymous_grant(raw, now):
    """按固定逻辑origin解释coalesced Set-Cookie，不信任固定IP的自动Jar。"""
    cookies, pieces, jar = {}, [], SimpleCookie()
    piece = value = ""
    morsel = None
    deadline = None
    try:
        if (not isinstance(raw, str) or not raw or len(raw) > 16384
                or any(ord(char) < 32 or ord(char) >= 127 for char in raw)):
            raise ValueError
        pieces = _COOKIE_SPLIT.split(raw)
        if len(pieces) > 32:
            raise ValueError
        for piece in pieces:
            piece = piece.strip()
            name = piece.partition("=")[0]
            if name.strip() in _COOKIE_NAMES and name != name.strip():
                raise ValueError
            if name not in _COOKIE_NAMES:
                continue  # 不接收账号、CDR、验证或任意其它Cookie。
            value = piece.split(";", 1)[0].partition("=")[2]
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            # SimpleCookie会解反斜杠/八进制转义；正常SDK读的是原样cookie，禁止这种改写。
            if not _COOKIE_VALUE.fullmatch(value):
                raise ValueError
            attrs = set()
            for part in piece.split(";")[1:]:
                if not part.strip():
                    continue
                attr, separator, attribute_value = part.strip().partition("=")
                attr = attr.lower()
                if (attr not in _COOKIE_ATTRS or attr in attrs
                        or (attr in {"secure", "httponly"}) == bool(separator)
                        or (separator and not attribute_value.strip())):
                    raise ValueError
                attrs.add(attr)
            jar.clear()
            jar.load(piece)
            if set(jar) != {name} or name in cookies:
                raise ValueError  # 同名cookie即使值相同，也不合并模糊的scope/expiry。
            morsel = jar[name]
            domain = morsel["domain"].lower() or _HOST
            path = morsel["path"] or _PATH.rsplit("/", 1)[0]
            if domain not in _COOKIE_DOMAINS or not path.startswith("/"):
                raise ValueError
            if not (_PATH == path or (_PATH.startswith(path)
                    and (path.endswith("/") or _PATH[len(path):].startswith("/")))):
                raise ValueError
            if morsel["samesite"] and morsel["samesite"].lower() not in {"lax", "strict", "none"}:
                raise ValueError
            if morsel.value != value:
                raise ValueError
            if name == "_m_h5_tk" and not value.split("_", 1)[0]:
                raise ValueError
            expiry = _cookie_expiry(morsel, now)
            if expiry is not None:
                deadline = expiry if deadline is None else min(deadline, expiry)
            cookies[name] = value
        if "_m_h5_tk" not in cookies:
            raise ValueError
        return cookies.copy(), deadline
    finally:
        # 尽早释放引用；Python不可变字符串不宣称具有内存物理擦除保证。
        cookies.clear()
        pieces.clear()
        jar.clear()
        raw = piece = value = ""
        morsel = None


async def fetch_youku_calendar(http) -> dict:
    """返回已验证SUCCESS的原MTOP envelope；账号/验证/失败均不续接。"""
    cookies, query = {}, {}
    headers = {
        "Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://www.youku.com", "Referer": "https://www.youku.com/ku/webcomic",
    }
    response = payload = token_prefix = None
    deadline = None
    try:
        for attempt in range(2):
            now = time.time()
            if not math.isfinite(now) or (deadline is not None and now >= deadline):
                raise ValueError
            timestamp = str(int(now * 1000))
            token_prefix = cookies.get("_m_h5_tk", "undefined").split("_", 1)[0]
            query.clear()
            query.update({
                "jsv": "2.7.4", "appKey": _APP_KEY, "t": timestamp,
                "sign": hashlib.md5(
                    (token_prefix + "&" + timestamp + "&" + _APP_KEY + "&" + _DATA).encode("ascii"),
                    usedforsecurity=False,
                ).hexdigest(),
                "api": _API, "v": "1.0", "dataType": "json", "type": "originaljson",
                "jsonpIncPrefix": timestamp, "data": _DATA,
            })
            if cookies:
                headers["Cookie"] = "; ".join(name + "=" + cookies[name] for name in _COOKIE_NAMES if name in cookies)
            # 唯一I/O入口：复用主线factory、全局配额、限频、DNS/TLS/raw大小和deadline。
            response = await http.get_response(_URL, params=query, headers=headers)
            payload, code = _envelope(response)
            if code == "SUCCESS":
                return payload
            if attempt:
                raise ValueError
            cookies, deadline = _anonymous_grant(response.headers.get("set-cookie", ""), time.time())
    except Exception:  # noqa: BLE001, S110 -- 不把上游异常正文、URL、Cookie或sign传给调用方。
        pass
    finally:
        cookies.clear()
        query.clear()
        headers.clear()
        response = payload = token_prefix = None
        deadline = None
    # 在except作用域之外抛出，连__context__都不保留原始网络/解析异常。
    raise SourceUnavailable(_ERROR)
