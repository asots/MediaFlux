"""只读页面拒访证据：不执行JS，不把SDK函数/示例字符串当成服务端错误响应。"""
from __future__ import annotations

from html.parser import HTMLParser
import json
import re
from urllib.parse import unquote

_CODES = ("fail_sys_user_validate", "fail_sys_illegal_access", "fail_sys_token_empty",
          "fail_sys_token_expired", "fail_sys_token_exoired")
_VISIBLE = ("请完成安全验证", "滑动验证", "验证码验证", "安全验证", "访问验证", "访问过于频繁", "captcha")
_STRUCTURAL = ("rgv587_flag", "punish?x5sec")
_TARGETS = ("captcha", "verify", "validate", "login", "punish", "x5sec")
_MAX_TEXT = 2 * 1024 * 1024  # 与公开 HTTP 响应上限一致，不扩大来源预算。
_MAX_DEPTH = 256


class _MalformedScript(ValueError):
    """超过边界或字符串未闭合；只读扫描停止，不猜执行结果。"""


def script_tokens(script):
    """单趟产出字符串/正则/注释/结构符范围；不在失败引号内部重新搜索。"""
    if len(script) > _MAX_TEXT:
        raise _MalformedScript
    i, size = 0, len(script)
    expression_start, word, parentheses, member_access = True, "", [], False
    while i < size:
        char, start = script[i], i
        if char.isspace():
            i += 1
            continue
        if char in "\"'`":
            i += 1
            while i < size:
                current = script[i]
                i += 1
                if current == "\\":
                    i += 1  # 转义字符整体跳过；包括转义引号，不回退。
                elif current == char:
                    yield "string", start, i
                    break
            else:
                raise _MalformedScript
            expression_start, word, member_access = False, "", False
        elif script.startswith("//", i):
            i += 2
            while i < size and script[i] not in "\r\n":
                i += 1
            yield "comment", start, i
        elif script.startswith("/*", i):
            end = script.find("*/", i + 2)
            if end < 0:
                raise _MalformedScript
            i = end + 2
            yield "comment", start, i
        elif char == "/" and expression_start:
            # 表达式起点的 / 是正则；字符类内的斜线和任意转义均不结束字面量。
            # 正则里的引号/花括号/拒访示例都只是数据，不是 JS 语句。
            i, character_class = i + 1, False
            while i < size:
                current = script[i]
                i += 1
                if current in "\r\n":
                    raise _MalformedScript
                if current == "\\":
                    if i >= size or script[i] in "\r\n":
                        raise _MalformedScript
                    i += 1
                elif current == "[":
                    character_class = True
                elif current == "]":
                    character_class = False
                elif current == "/" and not character_class:
                    while i < size and (script[i].isalnum() or script[i] in "_$"):
                        i += 1  # flags 属于字面量；不校验或执行正则本身。
                    yield "regex", start, i
                    break
            else:
                raise _MalformedScript
            expression_start, word, member_access = False, "", False
        elif char.isalnum() or char in "_$":
            i += 1
            while i < size and (script[i].isalnum() or script[i] in "_$"):
                i += 1
            word = "" if member_access else script[start:i]
            member_access = False
            expression_start = word in {"return", "throw", "case", "delete", "void", "typeof",
                                        "yield", "await", "in", "instanceof", "of", "else", "do", "new"}
        else:
            i += 1
            if char == "(":
                if len(parentheses) >= _MAX_DEPTH:
                    raise _MalformedScript
                parentheses.append(word in {"if", "while", "for", "with", "switch", "catch"})
                expression_start = True
            elif char == ")":
                # 控制语句之后可直接是 /regex/.test(...)；调用/分组之后则是除法。
                expression_start = parentheses.pop() if parentheses else False
            elif char in "].":
                expression_start = False
            elif char in "+-" and i < size and script[i] == char:
                i += 1  # 后缀 ++/-- 仍是值；前缀后仍等待表达式。
            else:
                expression_start = True  # 分隔符、操作符或除法之后等待下一表达式。
            word, member_access = "", char == "."
            if char in "{}[]();":
                yield "punctuation", start, i


_ASSIGNMENT = re.compile(
    r"(?:(?:var|let|const)\s+)?([\w$]+(?:\s*(?:\.\s*[\w$]+|\[\s*[\"'][\w$]+[\"']\s*\]))*)"
    r"\s*=\s*([\[{].*)\Z", re.DOTALL,
)
_LOCATION = r"(?:(?:window|document|self|top)\s*(?:\.\s*location|\[\s*[\"']location[\"']\s*\])|location)\s*"
_NAVIGATION = re.compile(
    _LOCATION + r"(?:(?:\.\s*href|\[\s*[\"']href[\"']\s*\])\s*)?=\s*[\"']([^\"'\r\n]{1,2048})[\"']\s*\Z"
    r"|" + _LOCATION + r"(?:\.\s*(?:assign|replace)|\[\s*[\"'](?:assign|replace)[\"']\s*\])"
    r"\s*\(\s*[\"']([^\"'\r\n]{1,2048})[\"']\s*\)\s*\Z", re.IGNORECASE,
)


def _verification_target(value):
    return any(marker in unquote(value).casefold() for marker in _TARGETS)


class _PageEvidence(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text = []
        self.scripts = []
        self.script = None
        self.style = False
        self.redirect = False

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.script = []
        elif tag == "style":
            self.style = True
        elif tag == "meta":
            attributes = dict(attrs)
            if str(attributes.get("http-equiv") or "").casefold() == "refresh":
                self.redirect |= _verification_target(str(attributes.get("content") or ""))

    def handle_endtag(self, tag):
        if tag == "script" and self.script is not None:
            self.scripts.append("".join(self.script))
            self.script = None
        elif tag == "style":
            self.style = False

    def handle_data(self, data):
        if self.script is not None:
            self.script.append(data)
        elif not self.style:
            self.text.append(data)


def _page_evidence(text):
    if not isinstance(text, str) or len(text) > _MAX_TEXT:
        raise _MalformedScript
    page = _PageEvidence()
    page.feed(text)
    page.close()
    if page.script is not None:
        raise _MalformedScript
    return page


def script_contents(text):
    """只取真实 HTML script 内容，不能把注释/属性中的伪 script 当 SSR。"""
    return _page_evidence(text).scripts


def _statements(script):
    # 注释不干扰语句首；字符串内的//、分号与括号不能改变层级。
    parts, end = [], 0
    for kind, start, stop in script_tokens(script):
        if kind == "comment":
            parts.extend((script[end:start], " "))
            end = stop
    parts.append(script[end:])
    clean = "".join(parts)
    stack, start = [], 0
    for kind, offset, end in script_tokens(clean):
        if kind != "punctuation":
            continue
        value = clean[offset]
        if value in "{[(":
            if len(stack) >= _MAX_DEPTH:
                raise _MalformedScript
            stack.append(value)
        elif value in "}])":
            if not stack or stack.pop() != {"}": "{", "]": "[", ")": "("}[value]:
                return  # 不对非本子集的脚本猜执行结果；结构标记仍独立检查。
            if value == "}" and not stack:
                yield clean[start:end].strip()
                start = end
        elif value == ";" and not stack:
            yield clean[start:offset].strip()
            start = end
    if not stack:
        yield clean[start:].strip()


def _error_data(statement):
    assignment = _ASSIGNMENT.fullmatch(statement)
    data, target = statement, ""
    if assignment:
        target, data = assignment.groups()
    if not data.startswith(("{", "[")):
        return False
    try:
        payload = json.loads(data)
    except (ValueError, RecursionError):
        return False
    if isinstance(payload, dict):
        ret = payload.get("ret")
    elif re.search(r"(?:\.\s*ret|\[\s*[\"']ret[\"']\s*\])\s*\Z", target):
        ret = payload
    else:
        return False
    # 混合SUCCESS+拒访同样停止，不能只检查第一项。
    return isinstance(ret, list) and any(isinstance(code, str) and code.split("::", 1)[0].casefold() in _CODES for code in ret)


def has_access_challenge(text):
    if not isinstance(text, str) or len(text) > _MAX_TEXT:
        return True
    if any(marker in text.casefold() for marker in _STRUCTURAL):
        return True
    try:
        page = _page_evidence(text)
        visible = re.sub(r"\s+", "", "".join(page.text)).casefold()
        if page.redirect or any(marker in visible for marker in (*_VISIBLE, *_CODES)):
            return True
        for script in page.scripts:
            try:
                for statement in _statements(script):
                    if _error_data(statement):
                        return True
                    navigation = _NAVIGATION.fullmatch(statement)
                    if navigation and _verification_target(navigation[1] or navigation[2]):
                        return True
            except _MalformedScript:
                # 不是完整 JS 解析器：超出词法子集或未闭合的 SDK 不能据此推定拒访。
                # 只停止该脚本的词法扫描；其它脚本、可见拒访和 HTTP/API 边界仍核验。
                continue
    except Exception:  # noqa: BLE001 -- 畸形/超限输入保守停止，不执行脚本。
        return True
    return False
