"""工具调用上下文与结果契约；不依赖执行器实现。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.agent.kernel.state_types import CancellationToken, PublicationLease

    from . import error_repair
    from .capabilities import KernelToolSpec
    from .effects import EffectPlan, PreparedEffect
    from .projection import ToolOutcome
    from .task_state import TaskPatch


ProgressSink = Callable[[Mapping[str, Any]], Awaitable[None]]

@dataclass(frozen=True, slots=True)
class ToolCallContext:
    owner: str
    session_id: str
    request_id: str
    turn_id: str
    lease: PublicationLease
    cancellation: CancellationToken
    report_progress: ProgressSink
    wait_for_completion: bool = False
    capability_search: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    selection_arguments: Mapping[str, Any] | None = None
    resource_candidate_ref: str = ""
    channel: str = "api"
    completion_scope: Any = None
    continuation_binding: Any = None
    continuation_policy: Mapping[str, Any] | None = None
    diagnostic_only: bool = False
    delivery_route: Mapping[str, Any] = field(default_factory=dict)
    confirmed_effect: PipelineResult | None = None
    current_task: dict | None = None
    active_effect_plan: EffectPlan | None = None
    batch_progress: dict | None = None
    user_source: str = ""
    user_history: tuple[str, ...] = ()
    user_reply_context: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = ""
    schema_replacements: dict[str, error_repair.SchemaReplacement] = field(default_factory=dict)
    reobservation_source: Mapping[str, Any] | None = field(default=None, repr=False)
    scoped_trust: bool = False

    def policy_context(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "turn_id": self.turn_id,
            "generation": self.lease.generation,
        }

@dataclass(frozen=True, slots=True)
class PipelineResult:
    tool: KernelToolSpec
    arguments: Mapping[str, Any]
    outcome: ToolOutcome
    effect_plan: EffectPlan | None = None
    elapsed_ms: int = 0
    task_patch: TaskPatch | None = None
    prepared_effect: PreparedEffect | None = None
    recovered_failures: tuple[str, ...] = ()


def independent_read(tool, task=None) -> bool:
    from .capabilities import ToolEffect
    from .task_state import tracks_read

    return (tool.effect is ToolEffect.READ and tool.observe is None and not tracks_read(task, tool.name)
            and tool.metadata.get("parallel_read") is True
            and not tool.metadata.get("kernel_utility")
            and not tool.metadata.get("publishes_selection"))
