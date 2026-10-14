"""能力查询参数与配置事实；目录由 Kernel 注入，领域层不反向创建运行时。"""

from __future__ import annotations

from typing import Any

from app import config
from app.agent.errors import AgentToolError
from app.agent.models import ToolContext, ToolResult

CAPABILITY_PARAMETERS = {
    "type": "object",
    "description": "每次最多一个顶层字段，不能混用模式；空对象查询概况。已知工具名称用tool_names，无需同时描述query。",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
            "description": "不知道工具名时用能力描述检索；仅单独传query。已知名称改用tool_names。",
        },
        "queries": {
            "type": "array", "minItems": 1, "maxItems": 4,
            "items": {"type": "string", "minLength": 1, "maxLength": 200},
            "description": "仅单独传queries，联合检索最多4组能力；共享有限工具窗口，下一轮生效。",
        },
        "catalog": {
            "type": "object",
            "description": "词法未命中时按领域分页探索目录；只返回摘要，再用tool_names加载。空对象浏览全部领域。",
            "properties": {
                "domains": {"type": "array", "minItems": 1, "maxItems": 4,
                            "items": {"type": "string", "minLength": 1, "maxLength": 80}},
                "offset": {"type": "integer", "minimum": 0, "maximum": 10000},
                "limit": {"type": "integer", "minimum": 1, "maximum": 12},
            },
            "additionalProperties": False,
        },
        "window": {
            "type": "object",
            "description": "本轮有界工作集：retain仅保留已加载或同批已成功加载的canonical工具；release取消保留，不立即卸载。不授予执行权限。",
            "properties": {
                key: {"type": "array", "maxItems": 12,
                      "items": {"type": "string", "minLength": 1, "maxLength": 120, "pattern": r"^\S+$"}}
                for key in ("retain", "release")
            },
            "additionalProperties": False,
        },
        "tool_names": {
            "type": "array",
            "maxItems": 6,
            "minItems": 1,
            "items": {"type": "string", "maxLength": 120},
            "description": "已知工具名称时仅传tool_names，最多6个；不再附加query、queries、catalog或window。",
        },
    },
    "additionalProperties": False,
    "maxProperties": 1,
}


def capability_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict) or set(arguments) - {"query", "queries", "tool_names", "catalog", "window"}:
        raise AgentToolError("只允许 query、queries、tool_names、catalog 或 window；留空查看能力概况。")
    if not arguments:
        return {}
    if len(arguments) != 1:
        raise AgentToolError("query、queries、tool_names、catalog 和 window 只能提供其中一个。")
    if "window" in arguments:
        window = arguments["window"]
        if not isinstance(window, dict) or set(window) - {"retain", "release"}:
            raise AgentToolError("window 只允许 retain、release。")
        normalized = {}
        for key in ("retain", "release"):
            names = window.get(key, [])
            if (not isinstance(names, list) or len(names) > 12
                    or any(not isinstance(n, str) or not 1 <= len(n) <= 120
                           or any(c.isspace() for c in n) for n in names)):
                raise AgentToolError("retain/release 必须是至多12个精确canonical工具名称。")
            normalized[key] = list(dict.fromkeys(names))
        if set(normalized["retain"]) & set(normalized["release"]):
            raise AgentToolError("retain 与 release 不可重叠。")
        return {"window": normalized}
    if "queries" in arguments:
        queries = arguments["queries"]
        if not isinstance(queries, list) or not 1 <= len(queries) <= 4:
            raise AgentToolError("queries 必须是1至4组能力描述。")
        return {"queries": list(dict.fromkeys(capability_arguments({"query": q})["query"] for q in queries))}
    if "catalog" in arguments:
        page = arguments["catalog"]
        if not isinstance(page, dict) or set(page) - {"domains", "offset", "limit"}:
            raise AgentToolError("catalog 只允许 domains、offset、limit。")
        domains = page.get("domains", [])
        if (not isinstance(domains, list) or len(domains) > 4
                or ("domains" in page and not domains)
                or any(not isinstance(d, str) or not 1 <= len(d.strip()) <= 80 for d in domains)):
            raise AgentToolError("domains 必须是1至4个领域名称。")
        offset, limit = page.get("offset", 0), page.get("limit", 12)
        if type(offset) is not int or not 0 <= offset <= 10000 or type(limit) is not int or not 1 <= limit <= 12:
            raise AgentToolError("目录 offset 必须为0至10000整数，limit为1至12整数。")
        return {"catalog": {"domains": list(dict.fromkeys(d.strip() for d in domains)), "offset": offset, "limit": limit}}
    if "query" in arguments:
        query = arguments["query"]
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 200:
            raise AgentToolError("query 必须是1至200字符的能力描述。")
        return {"query": query.strip()}
    names = arguments["tool_names"]
    if (
        not isinstance(names, list)
        or not 1 <= len(names) <= 6
        or any(not isinstance(n, str) or not n.strip() or len(n) > 120 for n in names)
    ):
        raise AgentToolError("tool_names 必须是1至6个工具名称。")
    return {"tool_names": list(dict.fromkeys(n.strip() for n in names))}


def discover_capabilities(
    arguments: dict[str, Any], context: ToolContext
) -> ToolResult:
    if context.capability_search is None:
        return ToolResult(
            False, "unavailable", "当前调用缺少会话能力目录，未加载任何工具。"
        )
    data = context.capability_search(arguments)
    return ToolResult(
        True, "completed", "已查询项目能力，尚未执行业务操作。", data=data
    )


def web_capability_status() -> dict[str, str]:
    if not config.get_bool("WEB_SEARCH_ENABLED"):
        return {
            "status": "disabled",
            "reason": "公开网页搜索/读取已接入，但Web Search开关当前关闭。",
        }
    if not str(config.get("TAVILY_API_KEY", "") or "").strip():
        return {
            "status": "configuration_missing",
            "reason": "公开网页搜索/读取已接入，但尚未配置Tavily API Key。",
        }
    return {
        "status": "configured",
        "reason": "开关与必要配置已就绪；网络可达性、密钥有效性与剩余额度须实际调用核验。",
    }
