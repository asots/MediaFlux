"""Agent只读追漫日历：复用页面缓存与后台采集，不另开来源或强制刷新。"""
from __future__ import annotations

from datetime import date, datetime, timedelta
import json
import logging
import re
from typing import Any
from zoneinfo import ZoneInfo

from app import config
from app.agent.errors import AgentToolError
from app.agent.models import Evidence, ToolResult
from app.agent.public_safety import sanitize_public_text
from app.discovery.calendar.models import CalendarEvent, SOURCE_NAMES
from app.discovery.calendar.service import get_calendar_service

logger = logging.getLogger(__name__)
_TIMEZONE = "Asia/Shanghai"
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_DAYS = {"today", "tomorrow", "week", *_WEEKDAYS}
_ARGUMENTS = {"day", "source", "query", "page", "limit"}
_SOURCE_STATUSES = {"ok", "partial", "stale", "loading", "unavailable"}
_SCOPE = "平台公开动漫排期，非全站完整覆盖；排期不等于免费进度，访问资格以平台为准。来源不可用不代表当天没有更新。"
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _date(value: Any) -> date:
    if not isinstance(value, str) or not _DATE.fullmatch(value):
        raise ValueError("日期格式无效")
    return date.fromisoformat(value)


def _safe(value: Any, limit: int) -> str:
    return sanitize_public_text(value, limit=limit) if isinstance(value, str) else ""


def anime_calendar_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict) or set(arguments) - _ARGUMENTS:
        raise AgentToolError("追漫日历仅支持 day、source、query、page、limit 参数")
    day = arguments.get("day", "today")
    source = arguments.get("source", "all")
    query = arguments.get("query", "")
    if not isinstance(day, str) or len(day) > 10:
        raise AgentToolError("day 必须是 today、tomorrow、week、英文星期或 YYYY-MM-DD")
    day = day.strip().casefold()
    if day not in _DAYS:
        try:
            _date(day)
        except ValueError:
            raise AgentToolError("day 必须是 today、tomorrow、week、英文星期或有效的 YYYY-MM-DD") from None
    if not isinstance(source, str) or source not in {"all", *SOURCE_NAMES}:
        raise AgentToolError("source 仅支持 all、tencent、iqiyi、youku")
    if not isinstance(query, str) or len(query) > 80 or any(ord(char) < 32 or ord(char) == 127 for char in query):
        raise AgentToolError("query 必须是最多80字符的片名关键词")
    query = query.strip()
    if query:
        query = _safe(query, 80)
        if not query:
            raise AgentToolError("query 只能提供片名关键词，不能包含路径、地址或凭据")
    normalized = {"day": day, "source": source, "query": query}
    for name, default, maximum in (("page", 1, 100), ("limit", 20, 20)):
        value = arguments.get(name, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise AgentToolError(f"{name} 必须是1到{maximum}的整数")
        normalized[name] = value
    return normalized


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 40 or "T" not in value:
        return ""
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return ""
    return value


def _sources(snapshot: dict, selected: str) -> list[dict]:
    raw = snapshot.get("sources")
    if not isinstance(raw, list):
        raw = []
    sources = []
    for source, name in SOURCE_NAMES.items():
        if selected not in {"all", source}:
            continue
        matches = [value for value in raw[:3] if isinstance(value, dict) and value.get("id") == source]
        row = matches[0] if len(matches) == 1 else {}
        status = row.get("status")
        status = status if isinstance(status, str) and status in _SOURCE_STATUSES else "unavailable"
        sources.append({"id": source, "name": name, "status": status,
                        "fetched_at": _timestamp(row.get("fetched_at")),
                        "message": _safe(row.get("message"), 300) or "未取得可核验的来源状态"})
    return sources


def _requested_dates(day: str, today: date, start: date) -> list[date]:
    if day == "week":
        return [start + timedelta(days=index) for index in range(7)]
    if day in _WEEKDAYS:
        return [start + timedelta(days=_WEEKDAYS.index(day))]
    if day == "today":
        return [today]
    if day == "tomorrow":
        return [today + timedelta(days=1)]
    return [_date(day)]


def _events(card: dict, day: date) -> list[CalendarEvent]:
    raw = card.get("events", ())
    if not isinstance(raw, (list, tuple)):
        return []
    result = []
    for value in raw[:50]:
        if not isinstance(value, dict) or not isinstance(value.get("update_time", ""), str):
            continue
        try:
            event = CalendarEvent(value.get("date"), value.get("update_time", ""),
                                  value.get("schedule"), value.get("audience", "unknown"))
        except (TypeError, ValueError):
            continue
        if event.date == day.isoformat():
            result.append(event)
    # 兼容页面已经核验的非会员周规则；只有免费进度而无星期/规则时，绝不分配日期。
    weekdays = card.get("free_weekdays")
    schedule = card.get("free_schedule")
    if (not raw and isinstance(weekdays, (list, tuple)) and weekdays
            and all(type(value) is int and 1 <= value <= 7 for value in weekdays)
            and day.isoweekday() in weekdays and isinstance(schedule, str) and schedule.strip()
            and isinstance(card.get("free_update_time", ""), str)):
        try:
            result.append(CalendarEvent(day.isoformat(), card.get("free_update_time", ""), schedule, "free"))
        except (TypeError, ValueError):
            pass
    return result


def _items(snapshot: dict, requested: list[date], sources: list[dict], query: str) -> list[dict]:
    selected = {source["id"]: source for source in sources}
    dates = {day.isoformat() for day in requested}
    records = {}
    for day in snapshot["days"]:
        if day["date"] not in dates:
            continue
        day_date = _date(day["date"])
        for card in day["items"][:300]:
            if not isinstance(card, dict) or card.get("category") != "animation":
                continue
            source, source_id = card.get("source"), card.get("source_id")
            if not isinstance(source, str) or source not in selected or selected[source]["status"] not in {"ok", "partial", "stale"}:
                continue
            if not isinstance(source_id, str) or not _ID.fullmatch(source_id):
                continue
            title = _safe(card.get("title"), 200)
            if not title or (query and query.casefold() not in title.casefold()):
                continue
            for event in _events(card, day_date):
                # 一行即一条已核验排期；按事件分页，不裁掉同日会员/非会员分支，也不无限展开嵌套数组。
                key = (source, source_id, event.date, event.update_time, event.schedule, event.audience)
                records[key] = {"stable_id": f"{source}:{source_id}", "source": source,
                                "source_name": SOURCE_NAMES[source], "source_id": source_id, "title": title,
                                "date": event.date, "weekday": event.weekday, "update_time": event.update_time,
                                "schedule": _safe(event.schedule, 240), "audience": event.audience,
                                "free_progress": _safe(card.get("free_progress"), 120),
                                "stale": selected[source]["status"] == "stale" or card.get("stale") is True}
    return sorted(records.values(), key=lambda item: (
        item["date"], item["update_time"] or "99:99", item["source"], item["title"],
        item["source_id"], item["audience"], item["schedule"],
    ))


def _result(ok: bool, status: str, summary: str, data: dict, *, error: str = "", suggestions=()) -> ToolResult:
    result = ToolResult(ok=ok, status=status, summary=summary, data=data, error=error,
                        suggestions=list(suggestions), evidence=[Evidence(
                            "anime_calendar", "读取追漫日历受控缓存/后台刷新快照；来源事实时间独立列出，不查询播放资格。",
                            datetime.now(ZoneInfo(_TIMEZONE)).isoformat(timespec="seconds"),
                        )])
    # 按真正的JSON转义体积预留projector余量，不让合法长文案把整页事件降成纯摘要。
    def model_size():
        return len(json.dumps(result.to_model_dict(), ensure_ascii=False, separators=(",", ":")))

    if model_size() > 22_000:
        compact = {**data,
                   "items": [{key: value for key, value in item.items() if key != "source_id"} for item in data["items"]],
                   "sources": [{key: value for key, value in source.items() if key != "message"} for source in data["sources"]],
                   "model_omitted_fields": ["items.source_id", "sources.message"],
                   "model_view_note": "仅省略重复身份和长说明，没有删减本页事件；完整文案可打开追漫日历查看。"}
        result.model_data = compact
        if model_size() > 22_000:
            for item in compact["items"]:
                item.pop("schedule", None)
            compact["model_omitted_fields"].append("items.schedule")
            compact["model_view_note"] = "长排期文案仅在模型视图省略，不得据此推断集数；日期、时刻、受众、免费进度和本页事件均保留。"
    return result


def anime_calendar(arguments: dict[str, Any]) -> ToolResult:
    normalized = anime_calendar_arguments(arguments)
    data = {**normalized, "timezone": _TIMEZONE, "calendar_url": "/discovery/calendar", "scope_note": _SCOPE,
            "today": "", "week_start": "", "week_end": "", "requested_dates": [], "total": 0,
            "total_programmes": 0, "returned": 0, "has_more": False, "items": [], "sources": [],
            "refreshing": False, "retry_after": 0, "pagination_limited": False}
    if not config.get_bool("DISCOVERY_ENABLED"):
        return _result(False, "disabled", "影视探索功能当前已关闭", data,
                       error="请先由管理员启用影视探索。")
    try:
        snapshot = get_calendar_service().get_week()  # 与页面完全相同的缓存和单飞策略，不传force。
        if not isinstance(snapshot, dict) or snapshot.get("timezone") != _TIMEZONE:
            raise ValueError("日历时区无效")
        today, start = _date(snapshot.get("today")), _date(snapshot.get("week_start"))
        if start.weekday() != 0 or today - start != timedelta(days=today.weekday()):
            raise ValueError("日历周范围无效")
        days = snapshot.get("days")
        if not isinstance(days, list) or len(days) != 7 or any(
            not isinstance(day, dict) or day.get("date") != (start + timedelta(days=index)).isoformat()
            or not isinstance(day.get("items"), list) for index, day in enumerate(days)
        ):
            raise ValueError("日历七日结构无效")
        sources = _sources(snapshot, normalized["source"])
        requested = _requested_dates(normalized["day"], today, start)
        end = start + timedelta(days=6)
        data.update(today=today.isoformat(), week_start=start.isoformat(), week_end=end.isoformat(), sources=sources,
                    requested_dates=[day.isoformat() for day in requested], refreshing=snapshot.get("refreshing") is True)
        if any(not start <= day <= end for day in requested):
            return _result(False, "unsupported_range", f"追漫日历仅提供当前上海周 {start} 至 {end} 的排期", data,
                           error="查询日期不在当前周，未用本周或旧周数据代替。")
        items = _items(snapshot, requested, sources, normalized["query"])
    except Exception as exc:  # noqa: BLE001 -- 上游URL/Cookie/配置/异常正文不得进入Agent输出。
        logger.warning("Agent追漫日历读取失败 type=%s", type(exc).__name__)
        return _result(False, "unavailable", "追漫日历暂时不可用，不能据此判断没有动漫更新", data,
                       error="未取得可核验的追漫日历快照。", suggestions=["请稍后重试，或打开追漫日历查看来源状态。"])
    offset = (normalized["page"] - 1) * normalized["limit"]
    page = items[offset:offset + normalized["limit"]]
    data.update(items=page, total=len(items), total_programmes=len({item["stable_id"] for item in items}),
                returned=len(page), has_more=offset + len(page) < len(items) and normalized["page"] < 100,
                pagination_limited=offset + len(page) < len(items) and normalized["page"] == 100)
    states = {source["status"] for source in sources}
    scope = "本周" if len(requested) == 7 else requested[0].isoformat()
    if page:
        partial = states != {"ok"} or any(item["stale"] for item in page)
        suggestions = ["请结合来源状态和上次成功时间阅读排期；未注明时刻保持未知，会员排期不代表可免费观看。"]
        if data["pagination_limited"]:
            suggestions.append("已到第100页上限，仍有匹配记录；请收窄日期、平台或片名，不要继续查询第101页。")
        return _result(True, "partial" if partial else "success",
                       f"{scope}追漫日历返回{len(page)}条排期（已收录匹配{len(items)}条），非全站完整覆盖", data,
                       suggestions=suggestions)
    if items:
        return _result(True, "empty", f"所选范围共有{len(items)}条已收录排期，本页没有结果", data)
    if "loading" in states:
        data["retry_after"] = 5
        return _result(False, "loading", "追漫日历正在后台获取，暂不能确认所选日期的更新", data,
                       suggestions=["排期正在后台获取，可稍后查看或打开追漫日历了解同步进度。"])
    if states & {"unavailable", "stale"}:
        return _result(False, "unavailable", "所选来源不可用或仅有旧缓存，暂无可核验的匹配排期；不代表当天没有更新", data,
                       suggestions=["可查看其他平台或稍后重试，不能把未取得数据解释为没有更新。"])
    return _result(True, "empty", f"{scope}已收录的公开排期没有匹配内容，不代表全站没有更新", data)
