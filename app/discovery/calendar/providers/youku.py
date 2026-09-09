"""优酷官方动漫周历：有效SSR优先，旧周以正常匿名动态接口补取；不涉及播放或账号。"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from app.discovery.calendar.models import CalendarEntry, CalendarEvent, SourceResult, SourceUnavailable
from app.discovery.calendar.posters import platform_poster_key
from app.discovery.calendar.youku_challenge import (
    has_access_challenge as _has_access_challenge,
    script_contents,
    script_tokens,
)

_URL = "https://www.youku.com/ku/webcomic"
_TIMEZONE = ZoneInfo("Asia/Shanghai")
_INITIAL_DATA = re.compile(r"\s*window\.__INITIAL_DATA__\s*=\s*", re.IGNORECASE)
_UNDEFINED = re.compile(r"\bundefined\b")
_SHOW_ID = re.compile(r"[0-9a-f]{20}")
_UPDATE = re.compile(
    r"(?P<time>(?:[01][0-9]|2[0-3]):[0-5][0-9])\s*"
    r"(?P<audience>SVIP|VIP|非会员|会员|免费)?\s*更新[1-9][0-9]*[话集]"
)
_MEMBER_LABELS = {"VIP", "SVIP", "会员", "会员专享", "VIP专享", "SVIP专享"}
_DAY_NAMES = ("一", "二", "三", "四", "五", "六", "日")


def _text(value, limit=400):
    if not isinstance(value, str) or len(value) > limit or re.search(r"[<>\x00-\x1f\x7f]", value):
        return ""
    return value.strip()


def _list(value):
    return value if isinstance(value, list) else []


def _initial_data(text):
    try:
        for script in script_contents(text):
            match = _INITIAL_DATA.match(script)
            if not match:
                continue
            raw = script[match.end():]
            # 仅替换字符串之外的 undefined；线性扫描，不 eval JS 或搜索转义引号。
            parts, end = [], 0
            for kind, start, stop in script_tokens(raw):
                if kind == "string":
                    parts.extend((_UNDEFINED.sub("null", raw[end:start]), raw[start:stop]))
                    end = stop
            parts.append(_UNDEFINED.sub("null", raw[end:]))
            payload = "".join(parts)
            data, end = json.JSONDecoder().raw_decode(payload)
            if payload[end:].strip() not in {"", ";"}:
                return None
            return data if isinstance(data, dict) and isinstance(data.get("moduleList"), list) else None
    except (ValueError, RecursionError):
        return None
    return None


def _calendars(data):
    for module in data["moduleList"][:100]:
        if not isinstance(module, dict):
            continue
        for component in _list(module.get("components"))[:100]:
            if (isinstance(component, dict) and component.get("typeName") == "KU_FLIX_MULTI_TAB_A"
                    and component.get("type") == 35 and component.get("title") == "每日更新"):
                yield component


def _week_rows(component, today):
    tabs, rows = component.get("tabList"), component.get("itemList")
    if not isinstance(tabs, list) or not isinstance(rows, list) or len(tabs) != 7 or len(rows) != 7:
        return None
    monday = today - timedelta(days=today.weekday())
    dates = [monday + timedelta(days=i) for i in range(7)]
    for i, (tab, row, day) in enumerate(zip(tabs, rows, dates)):
        # 按完整的当前上海周匹配 MM.DD，跨年周的十二月和一月各归正确年份。
        # “今”只作展示标签，不能凭 selectedIndex 把其他日期挪成今天。
        if (not isinstance(tab, dict) or not isinstance(row, list)
                or tab.get("date") != day.strftime("%m.%d")
                or tab.get("title") not in {_DAY_NAMES[i], "今"}):
            return None
    return zip(dates, rows)


def _programme(card):
    if not isinstance(card, dict):
        return None
    action = card.get("action")
    if not isinstance(action, dict) or action.get("type") != "JUMP_TO_SHOW":
        return None
    source_id = action.get("value")
    extra = action.get("extra")
    category = extra.get("category") if isinstance(extra, dict) else None
    title = _text(card.get("title"), 200)
    if (not isinstance(source_id, str) or not _SHOW_ID.fullmatch(source_id) or not title
            or category != "动漫"):
        return None
    return source_id, title


def _event(card, day):
    reason = card.get("reason")
    reason_text = reason.get("text") if isinstance(reason, dict) else None
    schedule = _text(reason_text.get("title")) if isinstance(reason_text, dict) else ""
    # 周列已经证明日期。没有排播时刻时保留未知，不借用普通集数/其他日期的时刻。
    schedule = schedule or "每日更新"
    match = _UPDATE.fullmatch(schedule)
    audience = "unknown"
    declared = match["audience"] if match else None
    if declared in {"非会员", "免费"}:
        audience = "free"
    elif declared in {"SVIP", "VIP", "会员"}:
        audience = "member"
    else:
        mark = card.get("mark")
        labels = [_text(mark.get("text"))] if isinstance(mark, dict) else []
        for tag in _list(card.get("tags")):
            if isinstance(tag, dict) and isinstance(tag.get("text"), dict):
                labels.append(_text(tag["text"].get("title")))
        if any(label.upper() in _MEMBER_LABELS for label in labels):
            audience = "member"
    return CalendarEvent(date=day.isoformat(), update_time=match["time"] if match else "",
                         schedule=schedule, audience=audience)


def _dynamic_calendar_data(payload):
    """Columbus节点树投影为既有周历输入；只读公开节目字段，丢弃session/跟踪上下文。"""
    if not isinstance(payload, dict) or payload.get("api") != "mtop.youku.columbus.home.query" or payload.get("v") != "1.0":
        return None
    ret = payload.get("ret")
    if (not isinstance(ret, list) or not ret or len(ret) > 5
            or any(not isinstance(value, str) or value.split("::", 1)[0] != "SUCCESS" for value in ret)):
        return None
    data = payload.get("data")
    response = data.get("2019061000") if isinstance(data, dict) else None
    tree = response.get("data") if isinstance(response, dict) else None
    roots = tree.get("nodes") if isinstance(tree, dict) else None
    if not isinstance(roots, list) or not roots or not isinstance(roots[0], dict):
        return None
    root = roots[0]
    root_data = root.get("data")
    if (type(root.get("level")) is not int or root["level"] != 0
            or not isinstance(root_data, dict) or root_data.get("nodeKey") != "WEBCOMIC"):
        return None
    components = []
    for group in _list(root.get("nodes"))[:100]:
        if not isinstance(group, dict) or type(group.get("level")) is not int or group["level"] != 1:
            continue
        for raw in _list(group.get("nodes"))[:100]:
            if not isinstance(raw, dict) or type(raw.get("id")) is not int or raw["id"] != 35:
                continue
            settings = raw.get("data")
            days = raw.get("nodes")
            if (raw.get("typeName") != "KU_FLIX_MULTI_TAB_A" or type(raw.get("level")) is not int or raw["level"] != 2
                    or not isinstance(settings, dict) or settings.get("title") != "每日更新"
                    or not isinstance(days, list) or len(days) != 7):
                continue
            tabs, rows = [], []
            for day in days:
                if not isinstance(day, dict) or not isinstance(day.get("data"), dict) or not isinstance(day.get("nodes"), list):
                    break
                tabs.append({key: day["data"].get(key) for key in ("date", "title")})
                cards = []
                for item in day["nodes"][:100]:
                    card = item.get("data") if isinstance(item, dict) else None
                    if not isinstance(card, dict):
                        continue
                    action = card.get("action")
                    extra = action.get("extra") if isinstance(action, dict) else None
                    if not isinstance(action, dict) or not isinstance(extra, dict):
                        continue
                    mark = card.get("mark")
                    mark_data = mark.get("data") if isinstance(mark, dict) and mark.get("type") == "SIMPLE" else None
                    reason = card.get("reason")
                    reason_text = reason.get("text") if isinstance(reason, dict) else None
                    tags = [{"text": {"title": _text(tag["text"].get("title"))}} for tag in _list(card.get("tags"))[:20]
                            if isinstance(tag, dict) and isinstance(tag.get("text"), dict)]
                    cards.append({"title": card.get("title"), "img": card.get("img"),
                                  "action": {"type": action.get("type"), "value": action.get("value"),
                                             "extra": {"category": extra.get("category")}},
                                  "reason": {"text": {"title": _text(reason_text.get("title")) if isinstance(reason_text, dict) else ""}},
                                  "mark": {"text": _text(mark_data.get("text")) if isinstance(mark_data, dict) else ""},
                                  "tags": tags})
                rows.append(cards)
            if len(tabs) == len(rows) == 7:
                components.append({"typeName": raw["typeName"], "type": raw["id"], "title": settings["title"],
                                   "tabList": tabs, "itemList": rows})
    return {"moduleList": [{"components": components}]} if components else None


class YoukuCalendarProvider:
    source = "youku"
    allowed_hosts = frozenset({"www.youku.com", "acs.youku.com"})

    def __init__(self, *, clock=None):
        self.clock = clock or (lambda: datetime.now(_TIMEZONE))

    def _today(self):
        now = self.clock()
        if isinstance(now, datetime):
            return now.astimezone(_TIMEZONE).date() if now.tzinfo else now.date()
        if isinstance(now, date):
            return now
        raise TypeError("clock 必须返回 datetime 或 date")

    async def fetch(self, http) -> SourceResult:
        # 有效当周SSR仅1GET；已识别官方周历但日期过旧时，最多再用2次正常匿名动态请求。
        try:
            text = await http.get_text(_URL)
        except Exception:  # noqa: BLE001 -- HTTP 源错误统一脱敏，不捕获取消信号。
            return SourceResult(status="unavailable", message="优酷公开周历请求失败或要求验证，已停止请求。")
        if not isinstance(text, str):
            return SourceResult(status="unavailable", message="优酷公开周历响应格式未识别，未取得可核验的动漫日期。")
        if _has_access_challenge(text):
            return SourceResult(status="unavailable", message="优酷公开周历要求访问验证，已停止请求。")
        data = _initial_data(text)
        if data is None:
            return SourceResult(status="unavailable", message="优酷公开周历结构未识别，未取得可核验的动漫日期。")
        today = self._today()
        result = _parse_calendar_data(data, today)
        if result.status != "unavailable" or not any(_calendars(data)):
            return result
        # 不是把旧日期挪成本周；动态接口必须再次通过同一严格七日校验。
        try:
            from app.discovery.calendar.youku_http import fetch_youku_calendar
            payload = await fetch_youku_calendar(http)
            dynamic_data = _dynamic_calendar_data(payload)
            if dynamic_data is None:
                raise SourceUnavailable("优酷动态周历结构未识别")
            dynamic = _parse_calendar_data(dynamic_data, today)
            if dynamic.status == "unavailable" or not dynamic.entries:
                raise SourceUnavailable("优酷动态周历未提供有效的本周排期")
        except Exception:  # noqa: BLE001 -- 不泄露动态请求签名/匿名状态；取消信号仍向上传递。
            return SourceResult(status="unavailable", message=(result.message + "动态来源本次也未取得有效排期。")[:400])
        return SourceResult(entries=dynamic.entries, status=dynamic.status,
                            message=dynamic.message.replace("优酷官方追漫日历", "优酷官方动态追漫日历", 1),
                            sampled=dynamic.sampled, ignored=dynamic.ignored)


def _parse_calendar_data(data, today):
    programmes = {}
    sampled = set()
    conflicts = set()
    valid_calendars = 0
    bounded = False
    for component in _calendars(data):
        week_rows = _week_rows(component, today)
        if week_rows is None:
            continue
        valid_calendars += 1
        for day, row in week_rows:
            for card in row:
                identity = _programme(card)
                if identity is None:
                    continue
                source_id, title = identity
                sampled.add(source_id)
                if source_id in conflicts:
                    continue
                if source_id not in programmes:
                    if len(programmes) >= 100:
                        bounded = True
                        continue
                    programmes[source_id] = {"title": title, "events": set(), "poster_keys": set()}
                programme = programmes[source_id]
                if programme["title"] != title:
                    programmes.pop(source_id)
                    conflicts.add(source_id)
                    continue
                # 只用官方img，经共享精确主机/资产命名空间校验；不以hImg补缺。
                # 最多保留两个不同安全key，冲突只清图且本轮不可被后续卡片恢复。
                key = platform_poster_key("youku", card.get("img"))
                if key and len(programme["poster_keys"]) < 2:
                    programme["poster_keys"].add(key)
                event = _event(card, day)
                if len(programme["events"]) < 50 or event in programme["events"]:
                    programme["events"].add(event)
                else:
                    bounded = True
    entries = tuple(CalendarEntry(
        source="youku", source_id=source_id, title=programme["title"], category="animation",
        url=f"https://v.youku.com/video?s={source_id}",
        platform_poster_key=(next(iter(programme["poster_keys"])) if len(programme["poster_keys"]) == 1 else ""),
        events=tuple(sorted(programme["events"], key=lambda e: (e.date, e.update_time, e.audience, e.schedule))),
        evidence="优酷官方动漫频道“每日更新”周历：tabList.date对应itemList节目，reason.text.title为排播文案；会员性质与免费进度独立。",
    ) for source_id, programme in programmes.items())
    count = sum(len(entry.events) for entry in entries)
    unknown = sum(not event.update_time for entry in entries for event in entry.events)
    message = f"优酷官方追漫日历：本周{len(entries)}部动漫、{count}条排期，{unknown}条时刻未注明；平台排期与免费进度独立。"
    if not valid_calendars:
        monday = today - timedelta(days=today.weekday())
        current = f"{monday:%m.%d}–{monday + timedelta(days=6):%m.%d}"
        dates = [tab.get("date") for component in _calendars(data)
                 for tab in _list(component.get("tabList")) if isinstance(tab, dict)]
        safe_dates = [value for value in dates if isinstance(value, str)
                      and re.fullmatch(r"(?:0[1-9]|1[0-2])\.(?:0[1-9]|[12][0-9]|3[01])", value)]
        observed = f"页面日期为{safe_dates[0]}–{safe_dates[-1]}；" if safe_dates else ""
        return SourceResult(status="unavailable", message=(
            f"优酷未提供有效的当前上海周七日排期（本周{current}）；{observed}"
            "不使用旧周或普通推荐数据，不能据此判断本周没有动漫更新。"))
    if conflicts:
        message += f"{len(conflicts)}个节目身份信息冲突，已忽略。"
    if bounded:
        message += "已达到节目或单节目事件上限，保留有限结果。"
    return SourceResult(entries=entries, status="partial", message=message,
                        sampled=len(sampled), ignored=len(sampled) - len(entries))
