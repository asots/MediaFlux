"""原子工具目录与本地词法能力召回。"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ToolEffect(StrEnum):
    READ = "read"
    WRITE = "write"
    DANGER = "danger"


ToolCallable = Callable[..., Any]
Validator = Callable[[dict[str, Any]], dict[str, Any]]
Availability = Callable[[Mapping[str, Any]], bool]
Authorizer = Callable[[Mapping[str, Any]], bool]


def identity_validator(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise TypeError("tool arguments must be an object")
    return dict(arguments)


@dataclass(frozen=True, slots=True, kw_only=True)
class KernelToolSpec:
    name: str
    domain: str
    description: str
    input_schema: Mapping[str, Any]
    effect: ToolEffect
    effect_resolver: Callable[[Mapping[str, Any]], ToolEffect] | None = None
    examples: tuple[str, ...] = ()
    cost: float = 1.0
    context_requirements: frozenset[str] = frozenset()
    validator: Validator = identity_validator
    read: ToolCallable | None = None
    observe: ToolCallable | None = None
    prepare: ToolCallable | None = None
    execute_confirmed: ToolCallable | None = None
    verify: ToolCallable | None = None
    availability: Availability | None = None
    runtime_status: Callable[[], Mapping[str, str]] | None = None
    authorize: Authorizer | None = None
    approval_scope: Callable[[str, Mapping[str, Any], Mapping[str, Any]], dict[str, str] | None] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    model_name: str = ""

    def __post_init__(self) -> None:
        name = (self.name or "").strip()
        domain = (self.domain or "").strip().casefold()
        description = (self.description or "").strip()
        if not name or not re.fullmatch(r"[a-z][a-z0-9_.-]{2,119}", name):
            raise ValueError(f"invalid tool name: {name!r}")
        if not domain or not re.fullmatch(r"[a-z][a-z0-9_.-]{1,79}", domain):
            raise ValueError(f"invalid tool domain: {domain!r}")
        if not description:
            raise ValueError("tool description cannot be empty")
        if not isinstance(self.input_schema, Mapping):
            raise TypeError("input_schema must be an object")
        if self.effect is ToolEffect.READ and self.effect_resolver is not None:
            raise ValueError("READ tools cannot resolve write effects")
        if self.effect is ToolEffect.READ:
            if (
                self.read is None
                or self.prepare is not None
                or self.execute_confirmed is not None
            ):
                raise ValueError("READ tool must expose only a read handler")
        elif (
            self.prepare is None
            or self.execute_confirmed is None
            or self.read is not None
        ):
            raise ValueError(
                "WRITE/DANGER tool must expose prepare and execute_confirmed handlers"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "description", description)
        object.__setattr__(
            self,
            "examples",
            tuple(item.strip() for item in self.examples if item.strip()),
        )
        object.__setattr__(
            self,
            "context_requirements",
            frozenset(
                item.strip().casefold()
                for item in self.context_requirements
                if item.strip()
            ),
        )
        object.__setattr__(self, "cost", max(0.0, float(self.cost)))
        model_name = (self.model_name or name.replace(".", "__").replace("-", "_")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", model_name):
            raise ValueError(f"invalid provider tool name: {model_name!r}")
        object.__setattr__(self, "model_name", model_name)

    def for_arguments(self, arguments: Mapping[str, Any]) -> KernelToolSpec:
        """参数可提升组合风险，不能把目录声明的风险降级。"""
        if self.effect_resolver is None:
            return self
        effect = self.effect_resolver(arguments)
        if not isinstance(effect, ToolEffect) or effect not in {self.effect, ToolEffect.DANGER}:
            raise ValueError("resolved tool effect cannot downgrade risk")
        return replace(self, effect=effect, effect_resolver=None)

    def model_definition(self) -> dict[str, Any]:
        description = self.description
        if self.effect is not ToolEffect.READ:
            description += " 调用本工具只准备确认卡，不执行写入，无需先征求生成卡片的许可；实际写入必须等用户确认。"
        return {
            "name": self.model_name,
            "description": description[:800],
            "parameters": dict(self.input_schema),
        }


class ToolCatalog:
    """只保存工具事实，不理解用户自然语言。"""

    def __init__(self, tools: Iterable[KernelToolSpec] = (), *, projector: Any = None, selection: Any = None) -> None:
        self.projector = projector
        self.selection = selection
        self._tools: dict[str, KernelToolSpec] = {}
        self._aliases: dict[str, str] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: KernelToolSpec) -> None:
        if tool.name in self._tools or tool.model_name in self._aliases:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool
        self._aliases[tool.model_name] = tool.name

    def get(self, name: str) -> KernelToolSpec:
        try:
            key = (name or "").strip()
            return self._tools[self._aliases.get(key, key)]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def visible(
        self, context: Mapping[str, Any] | None = None
    ) -> tuple[KernelToolSpec, ...]:
        runtime = context or {}
        result: list[KernelToolSpec] = []
        available_kinds = {
            str(item).strip().casefold()
            for item in runtime.get("reference_kinds", ())
            if str(item).strip()
        }
        for tool in self._tools.values():
            if tool.context_requirements and not tool.context_requirements.issubset(
                available_kinds
            ):
                continue
            if tool.availability is not None:
                try:
                    if not tool.availability(runtime):
                        continue
                except Exception as exc:  # noqa: BLE001 - optional plug-in boundary
                    logger.warning(
                        "Agent 工具可用性检查失败 tool=%s type=%s",
                        tool.name,
                        type(exc).__name__,
                    )
                    continue
            result.append(tool)
        return tuple(result)

    def has(self, name: str) -> bool:
        key = (name or "").strip()
        return key in self._tools or key in self._aliases

    def __len__(self) -> int:
        return len(self._tools)


_WORD_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+", re.IGNORECASE)
_NEGATED_PHRASE_RE = re.compile(
    r"(?:不要|无需|不需要|不使用|禁止|别用|排除|"
    r"没(?:有)?(?:让|叫)(?:你)?|不是(?:让|叫)(?:你)?|并非(?:让|叫)(?:你)?|"
    r"(?:^|(?<=[，。；;,.!?！？\n、]))\s*(?:不|别)(?!是|同|仅|只))"
    r"\s*([^，。；;,.!?！？\n、]{1,32})",
    re.IGNORECASE,
)

_QUOTED_TEXT_RE = re.compile(r'《[^》]*》|「[^」]*」|“[^”]*”|"[^"\n]*"|`[^`\n]*`')


def _normalize(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold().strip()


def _tokens(value: object) -> list[str]:
    text = _normalize(value)
    tokens: list[str] = []
    for chunk in _WORD_RE.findall(text):
        if not chunk:
            continue
        if "\u3400" <= chunk[0] <= "\u9fff":
            if len(chunk) == 1:
                tokens.append(chunk)
            for width in (2, 3, 4):
                if len(chunk) >= width:
                    tokens.extend(
                        chunk[index : index + width]
                        for index in range(len(chunk) - width + 1)
                    )
            if len(chunk) <= 12:
                tokens.append(chunk)
        else:
            tokens.append(chunk)
            tokens.extend(
                part for part in re.split(r"[._/-]+", chunk) if part and part != chunk
            )
    return tokens


def _query_terms(value: object) -> tuple[str, Counter[str], Counter[str]]:
    text = _normalize(value)
    # 片名/引用里的“不要”不是操作指令；保留等长空白以按原始位置切分。
    directives = _QUOTED_TEXT_RE.sub(lambda match: " " * len(match[0]), text)
    positive, excluded, offset = [], Counter(), 0
    for match in _NEGATED_PHRASE_RE.finditer(directives):
        positive.append(text[offset:match.start()])
        excluded.update(_tokens(match[1]))
        offset = match.end()
    positive.append(text[offset:])
    query = " ".join(positive)
    terms = Counter(_tokens(query))
    # 全句是否定式请求时也可能是在取消既有任务，不能抹掉整个意图。
    if not terms:
        return text, Counter(_tokens(text)), Counter()
    # 同一词也出现在明确正向目标里时，不因另一排除条件而反向惩罚它。
    return query, terms, Counter({token: count for token, count in excluded.items() if token not in terms})


def _bm25_overlap(
    terms: Counter[str],
    document_counts: Counter[str],
    *,
    document_length: int,
    document_frequency: Counter[str],
    total_documents: int,
    average_length: float,
    weight: float,
) -> float:
    score = 0.0
    for token, query_frequency in terms.items():
        frequency = document_counts.get(token, 0)
        if not frequency:
            continue
        df = document_frequency[token]
        idf = math.log(
            1.0 + (total_documents - df + 0.5) / (df + 0.5)
        )
        denominator = frequency + 1.5 * (
            1.0 - 0.75 + 0.75 * document_length / max(1.0, average_length)
        )
        score += (
            idf
            * ((frequency * 2.5) / denominator)
            * min(2, query_frequency)
        )
    return score * weight


@dataclass(frozen=True, slots=True)
class CapabilitySelection:
    tools: tuple[KernelToolSpec, ...]
    scores: Mapping[str, float]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(tool.model_name for tool in self.tools)


class CapabilityRetriever:
    """BM25 风格候选召回；只缩小能力集合，不裁决最终意图。"""

    def __init__(self, *, minimum: int = 6, maximum: int = 10) -> None:
        if minimum < 1 or maximum < minimum or maximum > 24:
            raise ValueError("invalid capability retrieval bounds")
        self.minimum = minimum
        self.maximum = maximum

    def retrieve(
        self,
        message: str,
        catalog: ToolCatalog,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> CapabilitySelection:
        runtime = context or {}
        visible = [
            tool for tool in catalog.visible(runtime)
            if not tool.metadata.get("kernel_utility") or tool.metadata.get("query_discoverable") is True
        ]
        if not visible:
            return CapabilitySelection((), {})
        normalized_message, query_counts, negated_counts = _query_terms(message)
        raw_hints = runtime.get("recent_user_messages", ())
        if isinstance(raw_hints, str):
            raw_hints = (raw_hints,)
        # 历史只补足省略表达，越旧越弱；当前显式引用不能被旧话题压过。
        history_scale = 0.2 if runtime.get("has_current_reference") else 1.0
        hints = []
        current_text = _normalize(message[:2_000])
        for item in raw_hints:
            text = str(item)[:2_000]
            # 调用方可能包含当前输入：不重复加权，也不占用历史衰减位置。
            if not text.strip() or _normalize(text) == current_text:
                continue
            hint, terms, _ = _query_terms(text)
            hints.append((terms, hint, 0.35 ** len(hints)))
            if len(hints) == 3:
                break
        raw_recent_tools = runtime.get("recent_tool_names", ())
        if isinstance(raw_recent_tools, str):
            raw_recent_tools = (raw_recent_tools,)
        recent_tool_names = list(dict.fromkeys(
            str(item).strip() for item in raw_recent_tools if str(item).strip()
        ))[:6]
        explicit_weights = runtime.get("recent_tool_weights") or {}
        # 工具可在同一回合连续调用，不能把列表中的第几个工具当成回合远近。
        recent_tool_weights = {
            name: max(0.0, min(1.0, float(explicit_weights.get(name, 1.0))))
            for name in recent_tool_names
        }
        documents: dict[str, list[str]] = {}
        negative_documents: dict[str, Counter[str]] = {}
        document_frequency: Counter[str] = Counter()
        for tool in visible:
            metadata_terms: list[str] = []
            for metadata_value in tool.metadata.values():
                if isinstance(metadata_value, str):
                    metadata_terms.append(metadata_value)
                elif isinstance(metadata_value, (list, tuple, set, frozenset)):
                    metadata_terms.extend(str(item) for item in metadata_value)
            document = " ".join(
                (
                    tool.name.replace(".", " "),
                    tool.model_name.replace("_", " "),
                    tool.domain,
                    tool.description,
                    *tool.examples,
                    *metadata_terms,
                )
            )
            tokens = _tokens(document)
            documents[tool.name] = tokens
            negative_documents[tool.name] = Counter(
                _tokens(
                    " ".join(
                        (
                            tool.name.replace(".", " "),
                            tool.model_name.replace("_", " "),
                            *tool.examples,
                        )
                    )
                )
            )
            document_frequency.update(set(tokens))
        average_length = sum(len(tokens) for tokens in documents.values()) / max(
            1, len(documents)
        )
        total_docs = len(documents)
        scores: dict[str, float] = {}
        negated_names: set[str] = set()
        for tool in visible:
            tokens = documents[tool.name]
            counts = Counter(tokens)
            overlap_options = {
                "document_length": len(tokens),
                "document_frequency": document_frequency,
                "total_documents": total_docs,
                "average_length": average_length,
            }
            score = _bm25_overlap(
                query_counts,
                counts,
                weight=1.0,
                **overlap_options,
            )
            for hint_counts, _normalized_hint, recency in hints:
                score += _bm25_overlap(
                    hint_counts, counts, weight=0.55 * recency * history_scale,
                    **overlap_options,
                )
            normalized_name = _normalize(tool.name)
            normalized_domain = _normalize(tool.domain)
            if normalized_name in normalized_message:
                score += 8.0
            if normalized_domain and normalized_domain in normalized_message:
                score += 3.0
            for example in tool.examples:
                normalized_example = _normalize(example)
                if normalized_example and (
                    normalized_example in normalized_message
                    or normalized_message in normalized_example
                ):
                    score += 5.0
                elif normalized_example:
                    score += max((
                        1.5 * recency * history_scale
                        for _counts, hint, recency in hints
                        if hint and (normalized_example in hint or hint in normalized_example)
                    ), default=0.0)
            score += 8.0 * history_scale * max(
                recent_tool_weights.get(tool.name, 0.0),
                recent_tool_weights.get(tool.model_name, 0.0),
            )
            negative_overlap = 0.0
            negative_document = negative_documents[tool.name]
            for token, negative_frequency in negated_counts.items():
                frequency = negative_document.get(token, 0)
                if not frequency:
                    continue
                df = document_frequency[token]
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                negative_overlap += idf * min(frequency, negative_frequency)
            if negative_overlap:
                negated_names.add(tool.name)
            score -= negative_overlap * 6.0
            score -= min(2.0, tool.cost * 0.05)
            scores[tool.name] = score
        # 查询型内核工具与业务工具使用同一可见性规则，但不挤占业务槽位；
        # 只匹配其名称/使用示例，不能靠“check”是“checklist”的子串混入普通问答。
        utilities = tuple(tool for tool in visible if tool.metadata.get("kernel_utility")
                          and scores[tool.name] > 0
                          and query_counts.keys() & set(_tokens(" ".join((tool.name, tool.model_name, *tool.examples)))))
        ranked = sorted(
            (tool for tool in visible if not tool.metadata.get("kernel_utility")),
            key=lambda item: (-scores[item.name], item.cost, item.name),
        )
        positive = sum(1 for tool in ranked if scores[tool.name] > 0)
        # 为工具声明的直接上下游能力预留少量位置。召回器仍不决定意图，
        # 但避免候选恰好占满时把写入预检/确认工具挤掉。
        reserve = min(2, max(0, self.maximum - self.minimum))
        base_limit = max(self.minimum, self.maximum - reserve)
        count = min(base_limit, max(self.minimum, positive))
        selected = list(ranked[:count])
        selected_names = {tool.name for tool in selected}
        related_candidates: list[tuple[int, KernelToolSpec]] = []
        workflow_neighbors: dict[str, int] = {}
        # 邻接依赖仍按当前相关性排序；最近真实调用的来源仅得到有限加权。
        # 无条件让历史来源优先会挤掉新任务的必需读取入口；无条件让写工具
        # 优先则会让无关写能力挤掉搜索→提交。这里不按业务意图或风险分流。
        def adjacency_score(tool: KernelToolSpec) -> float:
            score = scores[tool.name]
            if tool.name in recent_tool_names or tool.model_name in recent_tool_names:
                score += min(5.0, max(0.0, score) * 0.5) * history_scale * max(
                    recent_tool_weights.get(tool.name, 0.0),
                    recent_tool_weights.get(tool.model_name, 0.0),
                )
            return score

        sources = sorted(selected, key=lambda tool: (-adjacency_score(tool), tool.name))
        for source in sources:
            related = source.metadata.get("related_tools", ())
            if isinstance(related, str):
                related = (related,)
            if not isinstance(related, (list, tuple, set, frozenset)):
                continue
            for related_name in related:
                if not catalog.has(str(related_name)):
                    continue
                target = catalog.get(str(related_name))
                if (target.name in selected_names or target.name in negated_names or target not in visible
                        or target.metadata.get("kernel_utility")):
                    continue
                workflow = str(source.metadata.get("workflow") or target.name)
                position = workflow_neighbors.get(workflow, 0)
                related_candidates.append((position, target))
                workflow_neighbors[workflow] = position + 1
                selected_names.add(target.name)
        # 同一流程不能独占邻接预留位；先覆盖各流程的下一步，再补后续阶段。
        # 稳定排序保留原相关性顺序，未声明 workflow 的工具仍按原优先级处理。
        neighbors = sorted(related_candidates, key=lambda item: item[0])
        selected.extend(tool for _, tool in neighbors[: self.maximum - len(selected)])
        selected.sort(
            key=lambda item: (-scores.get(item.name, 0.0), item.cost, item.name)
        )
        combined = sorted((*selected[: self.maximum], *utilities),
                          key=lambda tool: (-scores[tool.name], tool.cost, tool.name))
        return CapabilitySelection(tuple(combined), scores)
