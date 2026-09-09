"""爱奇艺官方动漫追番表；免费进度独立，不访问播放接口。"""
from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, datetime, timedelta
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx

from app.discovery.calendar.models import (
    CalendarEntry,
    CalendarEvent,
    SourceResult,
    SourceUnavailable,
)
from app.discovery.calendar.posters import platform_poster_key
from app.indexers.errors import IndexerError

_TIMEZONE = ZoneInfo("Asia/Shanghai")
_CALENDAR_URL = "https://mesh.if.iqiyi.com/portal/lw/v7/channel/page/tracking"
_PARAMS = {"channelId": "4", "mode": "page", "page": "1", "v": "17.091.26283"}
_WEEKDAYS = {"jmd_Mon": 1, "jmd_Tues": 2, "jmd_Wed": 3, "jmd_Thur": 4,
             "jmd_Fri": 5, "jmd_Sat": 6, "jmd_Sun": 7}
_DAY_NAMES = {1: "周一", 2: "周二", 3: "周三", 4: "周四", 5: "周五", 6: "周六", 7: "周日"}
_NEXT_UPDATE = re.compile(r"(今天|今日|明天|明日)((?:[01][0-9]|2[0-3]):[0-5][0-9])更新")


def _text(value, limit=200):
    if not isinstance(value, str) or len(value) > limit or re.search(r"[<>\x00-\x1f]", value):
        return ""
    return value.strip()


def _programme_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return ""
    value = str(value)
    return value if re.fullmatch(r"[1-9][0-9]{0,31}", value) else ""


def _page_url(value):
    if not isinstance(value, str) or len(value) > 2048 or re.search(r"[\s\\\x00-\x1f]", value):
        return ""
    try:
        url = urlsplit(value)
        if (url.scheme != "https" or url.hostname not in {"www.iqiyi.com", "m.iqiyi.com"}
                or url.username or url.password or url.port not in {None, 443}
                or not re.fullmatch(r"/(?:v|a)_[A-Za-z0-9]+\.html", url.path)):
            return ""
    except ValueError:
        return ""
    return f"https://{url.hostname}{url.path}"


def _reference_day(payload, dates):
    """相对标签的参考日必须由源今天组证明，不能借抓取日重映射旧快照。"""
    reference = set()
    for block in payload["items"]:
        if (not isinstance(block, dict) or block.get("channel") != "4"
                or not isinstance(block.get("temp"), dict) or block["temp"].get("id") != 159
                or not isinstance(block.get("video"), list)):
            continue
        for group in block["video"]:
            if not isinstance(group, dict) or group.get("title") != "今天":
                continue
            block_id, subtitle = group.get("block_id"), group.get("sub_title")
            weekday = _WEEKDAYS.get(block_id) if isinstance(block_id, str) else None
            day = dates.get(subtitle) if isinstance(subtitle, str) else None
            if not day or day.isoweekday() != weekday:
                return None
            reference.add(day)
    return next(iter(reference)) if len(reference) == 1 else None


def _time_labels(row, event_date, reference_day):
    """卡片标签是下一次更新：只给日期吻合的分组赋时刻，不复制到别的星期。"""
    times = {}
    if reference_day is None:
        return times
    tags = row.get("tag3lines")
    if isinstance(tags, list):
        for tag in tags:
            label = _text(tag.get("text"), 120) if isinstance(tag, dict) else ""
            match = _NEXT_UPDATE.fullmatch(label)
            if match:
                target = reference_day + timedelta(days=int(match[1] in {"明天", "明日"}))
                if target == event_date:
                    times[match[2]] = label
    return times


class IqiyiCalendarProvider:
    source = "iqiyi"
    allowed_hosts = frozenset({"mesh.if.iqiyi.com"})

    def __init__(self, *, clock: Callable[[], datetime | date] | None = None):
        # naive datetime 按上海墙上时间理解；aware datetime 转上海后取日期。
        self._clock = clock or (lambda: datetime.now(_TIMEZONE))

    def _today(self):
        now = self._clock()
        if isinstance(now, datetime):
            return now.astimezone(_TIMEZONE).date() if now.tzinfo else now.date()
        if isinstance(now, date):
            return now
        raise TypeError("clock 必须返回 datetime 或 date")

    async def fetch(self, http) -> SourceResult:
        today = self._today()
        monday = today - timedelta(days=today.weekday())
        dates = {day.strftime("%m-%d"): day for day in (monday + timedelta(days=i) for i in range(7))}
        try:
            payload = await http.get_json(_CALENDAR_URL, params=dict(_PARAMS))
        except (SourceUnavailable, TimeoutError, OSError, httpx.HTTPError, IndexerError):
            return SourceResult(status="unavailable", message="爱奇艺追番表请求失败或要求验证，已停止请求。")
        if (isinstance(payload, dict) and isinstance(payload.get("code"), (str, int))
                and payload["code"] not in (0, "0")):
            return SourceResult(status="unavailable", message="爱奇艺追番表返回非成功状态，已停止请求。")
        if (not isinstance(payload, dict) or type(payload.get("code")) is not int
                or payload["code"] != 0 or payload.get("cname") != "追番二级页-v7-PCW"
                or not isinstance(payload.get("items"), list)):
            return SourceResult(status="partial", message="爱奇艺公开追番表结构未核验，未将其他目录当作排期。")

        reference_day = _reference_day(payload, dates)
        if reference_day is not None and reference_day > today:
            reference_day = None  # 源今天不能来自尚未到来的日期，矛盾时只保留绝对日期。
        programmes = {}
        sampled_ids = set()
        invalid_ids = set()
        for block in payload["items"]:
            if (not isinstance(block, dict) or block.get("channel") != "4"
                    or not isinstance(block.get("temp"), dict) or block["temp"].get("id") != 159
                    or not isinstance(block.get("video"), list)):
                continue
            for group in block["video"]:
                if not isinstance(group, dict):
                    continue
                block_id = group.get("block_id")
                weekday = _WEEKDAYS.get(block_id) if isinstance(block_id, str) else None
                subtitle = group.get("sub_title")
                event_date = dates.get(subtitle) if isinstance(subtitle, str) else None
                if not weekday or not event_date or event_date.isoweekday() != weekday:
                    continue  # 不把源窗口的旧周日迁移成本周日，也不处理 coming。
                day_label = group.get("title")
                if not (day_label == _DAY_NAMES[weekday] or (day_label == "今天" and event_date == reference_day)):
                    continue
                rows = group.get("data")
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    source_id = _programme_id(row.get("album_id"))
                    if not source_id:
                        continue
                    sampled_ids.add(source_id)
                    if (type(row.get("channel_id")) is not int or row["channel_id"] != 4
                            or row.get("content_type") != "FEATURE_FILM" or row.get("is_episode") is not False
                            or row.get("isAd") is True):
                        continue
                    title = _text(row.get("title"))
                    url = _page_url(row.get("page_url"))
                    if not title or not url:
                        invalid_ids.add(source_id)
                        continue
                    release = row.get("date")
                    year = str(release.get("year", "")) if isinstance(release, dict) else ""
                    if not re.fullmatch(r"(?:19|20)[0-9]{2}", year):
                        year = ""
                    programme = programmes.setdefault(source_id, {"titles": set(), "urls": set(),
                        "years": set(), "events": {}, "vip_badge": False, "poster_keys": set()})
                    programme["titles"].add(title)
                    programme["urls"].add(url)
                    if year:
                        programme["years"].add(year)
                    # 只取节目正式原封面，不从剧情背景图/hover 缩略图猜另一张图。
                    key = platform_poster_key(self.source, row.get("image_cover"))
                    if key and len(programme["poster_keys"]) < 2:
                        programme["poster_keys"].add(key)
                    # VIP 标识只是内容属性，不推定这次排期的受众或免费集数。
                    mark = row.get("pay_mark")
                    programme["vip_badge"] |= isinstance(mark, str) and "VIP" in mark
                    event = programme["events"].setdefault(event_date, {"group": f"{day_label} {subtitle}", "times": {}})
                    event["times"].update(_time_labels(row, event_date, reference_day))

        entries = []
        for source_id, programme in sorted(programmes.items()):
            if source_id in invalid_ids or len(programme["titles"]) != 1:
                continue
            events = []
            for event_date, event in sorted(programme["events"].items()):
                time = label = ""
                if len(event["times"]) == 1:
                    time, label = next(iter(event["times"].items()))
                events.append(CalendarEvent(date=event_date.isoformat(), update_time=time,
                    schedule="；".join(filter(None, (event["group"], label))), audience="unknown"))
            evidence = "官方追番表：items[].video[] 的星期/日期分组与 data[].album_id；时刻仅取日期吻合的 tag3lines[].text。"
            if programme["vip_badge"]:
                evidence += "节目带VIP标识；排期受众未声明，不构成免费承诺。"
            else:
                evidence += "平台排期受众未声明，不构成免费承诺。"
            entries.append(CalendarEntry(source="iqiyi", source_id=source_id,
                title=next(iter(programme["titles"])), category="animation", url=min(programme["urls"]),
                year=next(iter(programme["years"])) if len(programme["years"]) == 1 else "",
                # 同一节目多张有效原图时不按响应顺序任选，缺图/坏图不丢排期。
                platform_poster_key=(next(iter(programme["poster_keys"]))
                    if len(programme["poster_keys"]) == 1 else ""),
                evidence=evidence, events=tuple(events[:50])))
            if len(entries) >= 100:
                break
        count = sum(len(entry.events) for entry in entries)
        reference_note = (f"源今天为{reference_day.isoformat()}，相对时刻按源日期解释。"
                          if reference_day is not None and reference_day != today else "")
        return SourceResult(entries=tuple(entries), status="partial", sampled=len(sampled_ids),
            ignored=len(sampled_ids) - len(entries), message=(
                f"爱奇艺官方追番表：本周{len(entries)}个动漫节目、{count}条日期排期；"
                f"仅覆盖源已列明的动漫日期。{reference_note}未知时刻留空；平台排期不等于免费进度。"))
