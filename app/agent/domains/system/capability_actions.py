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
            "description": "按具体任务检索有限的相关工具候选，仅单独传query；完整功能清单或按读写类型汇总用catalog，不能用候选数代表全部功能。",
        },
        "queries": {
            "type": "array", "minItems": 1, "maxItems": 4,
            "items": {"type": "string", "minLength": 1, "maxLength": 200},
            "description": "仅单独传queries，联合检索最多4组能力；共享有限工具窗口，下一轮生效。",
        },
        "catalog": {
            "type": "object",
            "description": "一次返回所选领域的完整目录摘要及读写数量统计，不加载工具。domains使用概况返回的domain标识，不填中文需求描述；描述需求用query。空对象浏览全部领域；超大结果使用通用结果句柄读取，不接受目录分页参数。",
            "properties": {
                "domains": {"type": "array", "minItems": 1, "maxItems": 4,
                            "items": {"type": "string", "minLength": 1, "maxLength": 80}},
            },
            "additionalProperties": False,
        },
        "window": {
            "type": "object",
            "description": "本轮有界工作集：retain仅保留已加载或同批已成功加载的canonical工具；release取消保留，不立即卸载。不授予执行权限。",
            "properties": {
                key: {"type": "array", "maxItems": 32,
                      "items": {"type": "string", "minLength": 1, "maxLength": 120, "pattern": r"^\S+$"}}
                for key in ("retain", "release")
            },
            "additionalProperties": False,
        },
        "tool_names": {
            "type": "array",
            "maxItems": 20,
            "minItems": 1,
            "items": {"type": "string", "maxLength": 120},
            "description": "已知工具名称时仅传tool_names，最多20个；不再附加query、queries、catalog或window。",
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
            if (not isinstance(names, list) or len(names) > 32
                    or any(not isinstance(n, str) or not 1 <= len(n) <= 120
                           or any(c.isspace() for c in n) for n in names)):
                raise AgentToolError("retain/release 必须是至多32个精确canonical工具名称。")
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
        scope = arguments["catalog"]
        if not isinstance(scope, dict) or set(scope) - {"domains"}:
            raise AgentToolError("catalog 只允许 domains，返回所选范围的完整目录摘要。")
        domains = scope.get("domains", [])
        if (not isinstance(domains, list) or len(domains) > 4
                or ("domains" in scope and not domains)
                or any(not isinstance(d, str) or not 1 <= len(d.strip()) <= 80 for d in domains)):
            raise AgentToolError("domains 必须是1至4个领域名称。")
        return {"catalog": {"domains": list(dict.fromkeys(d.strip().casefold() for d in domains))} if domains else {}}
    if "query" in arguments:
        query = arguments["query"]
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 200:
            raise AgentToolError("query 必须是1至200字符的能力描述。")
        return {"query": query.strip()}
    names = arguments["tool_names"]
    if (
        not isinstance(names, list)
        or not 1 <= len(names) <= 20
        or any(not isinstance(n, str) or not n.strip() or len(n) > 120 for n in names)
    ):
        raise AgentToolError("tool_names 必须是1至20个工具名称。")
    return {"tool_names": list(dict.fromkeys(n.strip() for n in names))}


def discover_capabilities(
    arguments: dict[str, Any], context: ToolContext
) -> ToolResult:
    if context.capability_search is None:
        return ToolResult(
            False, "unavailable", "当前调用缺少会话能力目录，未加载任何工具。"
        )
    data = context.capability_search(arguments)
    ranked = bool(arguments.keys() & {"query", "queries"})
    return ToolResult(
        True, "completed",
        "已查询相关能力候选（不是完整目录），尚未执行业务操作。" if ranked else "已查询项目能力，尚未执行业务操作。",
        data=data,
        suggestions=["要列全某领域的能力，请用catalog按领域读取完整目录；不能把本次候选数当作该领域工具总数。"] if ranked else [],
    )


def web_search_capability_status() -> dict[str, str]:
    if not config.get_bool("WEB_SEARCH_ENABLED"):
        return {
            "status": "disabled",
            "reason": "公开网页搜索已接入，但Web Search开关当前关闭。",
        }
    if not (config.get("TAVILY_API_KEY", "") or "").strip():
        return {
            "status": "configuration_missing",
            "reason": "公开网页搜索已接入，但尚未配置Tavily API Key。",
        }
    return {
        "status": "configured",
        "reason": "开关与必要配置已就绪；网络可达性、密钥有效性与剩余额度须实际调用核验。",
    }


def web_read_capability_status() -> dict[str, str]:
    if not config.get_bool("WEB_READ_ENABLED", True):
        return {"status": "disabled", "reason": "本地网页读取开关当前关闭。"}
    return {
        "status": "configured",
        "reason": "本机读取公开 HTTPS 正文，无需 Tavily Key 或额度；不执行 JavaScript，页面可读性须实际调用核验。",
    }
