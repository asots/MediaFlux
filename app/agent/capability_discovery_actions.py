"""能力查询参数与配置事实；目录由 Kernel 注入，领域层不反向创建运行时。"""

from __future__ import annotations

from typing import Any

from app import config
from app.agent.errors import AgentToolError
from app.agent.models import ToolContext, ToolResult

CAPABILITY_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
            "description": "描述需要查找的项目能力，不必知道工具名称。",
        },
        "tool_names": {
            "type": "array",
            "maxItems": 6,
            "minItems": 1,
            "items": {"type": "string", "maxLength": 120},
            "description": "查询已知能力并加载其Schema，最多6个。与query二选一。",
        },
    },
    "additionalProperties": False,
}


def capability_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict) or set(arguments) - {"query", "tool_names"}:
        raise AgentToolError("只允许 query 或 tool_names；留空查看能力概况。")
    if not arguments:
        return {}
    if len(arguments) != 1:
        raise AgentToolError("query 和 tool_names 只能提供其中一个。")
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
