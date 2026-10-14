"""回合内能力发现；只更新下一次模型调用的工具窗口，不执行领域动作。"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .capabilities import CapabilityRetriever, KernelToolSpec, ToolCatalog
from .pipeline import ToolPipelineError

logger = logging.getLogger(__name__)

DISCOVERY_TOOL = "agent.capabilities"
# Only these bounded kernel utilities are exempt from the business window.
WINDOW_UTILITIES = frozenset({DISCOVERY_TOOL, "agent.read_result", "agent.task"})


class CapabilityDiscovery:
    """每回合独立。已批准范围、权限、工具执行仍由原 ToolPipeline 管理。"""

    def __init__(
        self, catalog: ToolCatalog, *, context: Mapping[str, Any], maximum: int = 10
    ):
        self.catalog = catalog
        self.context = dict(context)
        self.maximum = max(2, min(12, maximum))
        self.requests = 0
        # Cache selection facts only; permissions/status/window are always rechecked.
        self._cache: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        self._pending: tuple[KernelToolSpec, ...] = ()
        self._current: tuple[KernelToolSpec, ...] = ()
        self._retained: tuple[str, ...] = ()

    def _visible(self) -> tuple[KernelToolSpec, ...]:
        result = []
        for tool in self.catalog.visible(self.context):
            if tool.name == "agent.read_result" and not {"tool_result", "telegram_reference"}.intersection(self.context.get("reference_kinds", ())):
                continue
            try:
                if tool.authorize is not None and not tool.authorize(self.context):
                    continue
            except Exception as exc:  # noqa: BLE001 - optional capability policy fails closed
                logger.warning(
                    "Agent能力授权检查失败 tool=%s type=%s",
                    tool.name,
                    type(exc).__name__,
                )
                continue
            result.append(tool)
        names = {tool.name for tool in result}
        self._retained = tuple(name for name in self._retained if name in names)
        return tuple(result)

    def window(
        self,
        previous: Sequence[KernelToolSpec],
        additions: Sequence[KernelToolSpec] = (),
    ) -> tuple[KernelToolSpec, ...]:
        self._current = self._window(previous, additions, self._visible())
        return self._current

    def _window(
        self,
        previous: Sequence[KernelToolSpec],
        additions: Sequence[KernelToolSpec],
        available: Sequence[KernelToolSpec],
    ) -> tuple[KernelToolSpec, ...]:
        visible = {tool.name: tool for tool in available}
        ordered: list[KernelToolSpec] = []
        if DISCOVERY_TOOL in visible:
            ordered.append(visible[DISCOVERY_TOOL])
        if "agent.read_result" in visible:
            ordered.append(visible["agent.read_result"])
        if self.context.get("has_task") and "agent.task" in visible:
            ordered.append(visible["agent.task"])
        business_count = 0
        retained = tuple(visible[name] for name in self._retained if name in visible)
        for tool in (*retained, *additions, *previous):
            if tool.name not in visible or any(item.name == tool.name for item in ordered):
                continue
            if tool.name not in WINDOW_UTILITIES:
                if business_count >= self.maximum:
                    continue
                business_count += 1
            ordered.append(visible[tool.name])
        return tuple(ordered)

    @staticmethod
    def _status(tool: KernelToolSpec) -> dict[str, str]:
        unchecked = {
            "status": "unchecked",
            "reason": "能力已注册；网络、额度与业务运行条件尚未检查。",
        }
        if tool.runtime_status is None:
            return unchecked
        try:
            status = dict(tool.runtime_status())
        except Exception as exc:  # noqa: BLE001 - optional status probe is not a health guarantee
            logger.warning(
                "Agent能力状态检查失败 tool=%s type=%s", tool.name, type(exc).__name__
            )
            return {
                "status": "unknown",
                "reason": "无法读取当前运行条件；不能据此宣称未接入该能力。",
            }
        if status.get("status") not in {
            "configured",
            "disabled",
            "configuration_missing",
            "unchecked",
        }:
            return unchecked
        return {
            "status": str(status["status"]),
            "reason": str(status.get("reason", ""))[:240],
        }

    def search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        visible = self._visible()
        by_name = {tool.name: tool for tool in visible}
        domains = [
            {"domain": domain, "count": count}
            for domain, count in sorted(Counter(t.domain for t in visible).items())
        ]
        if not arguments:
            return {
                "domains": domains,
                "visible_count": len(visible),
                "retained_tools": list(self._retained),
                "retention_scope": "current_turn_only; release removes retention, not execution permission; no immediate unloading",
                "hint": "用queries一次联合查找多组能力；词法未命中可用catalog按domains/offset分页探索，再用tool_names加载。目录摘要不等于已加载或健康。",
            }
        key = json.dumps([arguments, sorted(by_name)], sort_keys=True, ensure_ascii=False)
        cache_hit = key in self._cache
        if not cache_hit and self.requests >= 8:
            raise ToolPipelineError(
                "本轮已达8次不同能力发现上限；相同查询可复用缓存，业务执行与模型轮次仍受会话预算限制。",
                code="capability_discovery_budget",
            )
        requested = arguments.get("tool_names", [])
        page = arguments.get("catalog")
        window = arguments.get("window")
        missing: list[str] = []
        found: list[KernelToolSpec] = []
        next_offset = None
        if window is not None:
            retained = tuple(dict.fromkeys([
                *(name for name in self._retained if name not in window.get("release", [])),
                *window.get("retain", []),
            ]))
            loaded = {t.name for t in (*self._current, *self._pending)}
            if any(name not in by_name or name not in loaded for name in window.get("retain", [])):
                raise ToolPipelineError("只能保留当前已加载且获授权的canonical工具。", code="capability_window_unavailable")
            promised = set(retained) | {t.name for t in self._pending if t.name in by_name}
            if len(retained) > 12 or len(promised - WINDOW_UTILITIES) > self.maximum:
                raise ToolPipelineError("保留工作集与同批已承诺工具超出窗口容量；请先释放或下一批重试。", code="capability_window_capacity")
            self._retained = retained
        elif page is not None:
            listing = sorted((t for t in visible if not page.get("domains") or t.domain in page["domains"]), key=lambda t: t.name)
            offset, limit = page.get("offset", 0), page.get("limit", 12)
            found = listing[offset:offset + limit]
            if offset + limit < len(listing):
                next_offset = offset + limit
        elif cache_hit:
            names, absent = self._cache[key]
            found = [by_name[name] for name in names if name in by_name]
            missing = list(absent)
        elif requested:
            for name in requested:
                try:
                    tool = self.catalog.get(name)
                except KeyError:
                    tool = None
                if tool is None or tool.name not in by_name:
                    missing.append(name)
                elif tool not in found:
                    found.append(tool)
        else:
            groups = [self._query(query, visible) for query in arguments.get("queries", [arguments.get("query", "")])]
            # Interleave ranks so the first query cannot monopolize the union.
            for rank in range(6):
                for group in groups:
                    if rank < len(group) and group[rank] not in found:
                        found.append(group[rank])
            found = found[:12]
        if not cache_hit:
            self.requests += 1
            self._cache[key] = (tuple(t.name for t in found), tuple(missing))
        candidates = tuple(tool for tool in found if tool.name != DISCOVERY_TOOL) if page is None else ()
        # Earlier successful calls in this batch retain their promised slots.
        additions = tuple(dict.fromkeys(t.name for t in (*self._pending, *candidates)))
        combined = tuple(by_name[name] for name in additions if name in by_name)
        planned = self._window(self._current, combined, visible)
        planned_names = {tool.name for tool in planned}
        self._pending = tuple(tool for tool in combined if tool.name in planned_names)
        business_count = sum(tool.name not in WINDOW_UTILITIES for tool in planned)
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description[:500],
                    "effect": tool.effect.value,
                    "availability": self._status(tool),
                    "related_tools": [
                        name
                        for name in tool.metadata.get("related_tools", ())
                        if name in by_name
                    ][:6],
                }
                for tool in found
            ],
            "not_available_in_context": missing,
            "domains": domains if not found else [],
            "loaded_tools": [tool.name for tool in self._pending],
            "retained_tools": list(self._retained),
            "retention_scope": "current_turn_only; release removes retention, not execution permission; no immediate unloading",
            "window_update": "_".join(key for key in ("retain", "release") if window.get(key)) or "inspect" if window is not None else None,
            "not_loaded_tools": [tool.name for tool in candidates if tool.name not in planned_names],
            "unloaded_tools": [
                tool.name for tool in self._current
                if tool.name in by_name and tool.name not in planned_names
            ],
            "window_policy": "retained_then_pending_then_requested_then_previous; overflow unloads unretained previous tail; new overflow is not_loaded",
            "window_budget": {
                "business_limit": self.maximum,
                "business_count": business_count,
                "utility_count": len(planned) - business_count,
                "total_limit": self.maximum + len(WINDOW_UTILITIES),
            },
            "remaining_discoveries": 8 - self.requests,
            "discovery_cost": {"distinct_requests": self.requests, "limit": 8},
            "cache_hit": cache_hit,
            "next_offset": next_offset,
            "catalog_only": page is not None,
            "execution": "未执行任何业务动作；发现的工具仅在下一次模型调用加载。写操作仍需确认。",
        }

    def _query(self, query: str, visible: Sequence[KernelToolSpec]) -> list[KernelToolSpec]:
        # 先保留实际命中，只有剩余槽位才补关联能力，避免第5/6名被无关邻居挤掉。
        selection = CapabilityRetriever(minimum=6, maximum=6).retrieve(
            query, ToolCatalog(visible), context=self.context,
        )
        positive = [tool for tool in selection.tools if selection.scores.get(tool.name, 0) > 0]
        related = set()
        for tool in positive:
            names = tool.metadata.get("related_tools", ())
            if isinstance(names, str):
                names = (names,)
            if isinstance(names, (list, tuple, set, frozenset)):
                related.update(str(name) for name in names)
        neighbors = [tool for tool in visible if tool.name in related and tool not in positive]
        return (positive + neighbors)[:6]

    def checkpoint(self) -> tuple[tuple[KernelToolSpec, ...], tuple[str, ...]]:
        return self._pending, self._retained

    def restore(self, checkpoint: tuple[tuple[KernelToolSpec, ...], tuple[str, ...]]) -> None:
        self._pending, self._retained = checkpoint

    def consume(self) -> tuple[KernelToolSpec, ...]:
        result, self._pending = self._pending, ()
        return result
