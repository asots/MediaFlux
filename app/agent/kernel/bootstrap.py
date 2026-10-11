"""Agent Kernel 的唯一生产依赖组合入口。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from app import database as db
from app.agent.feature_gate import RuntimeFence, RuntimeFenceError, read_runtime_fence, require_runtime_fence

from app import config
from app.agent.confirmation import SQLiteConfirmationStore
from app.agent.discovery_mapping_actions import configure_discovery_mapping_context
from app.agent.domain_catalog import build_tool_specs
from app.agent.guangya_cleanup_actions import configure_guangya_cleanup_context
from app.agent.guangya_directory_scrape_actions import (
    configure_directory_scrape_context,
)
from app.agent.guangya_fs_change_actions import configure_guangya_fs_change_context
from app.agent.guangya_rename_actions import configure_guangya_rename_context
from app.agent.guangya_workspace_actions import configure_guangya_workspace_context
from app.agent.ingest_actions import AgentIngestSessionStore
from app.agent.local_media_task_actions import configure_local_media_agent_context
from app.agent.missing_media_workflow_runtime import MissingMediaWorkflowRuntime
from app.agent.missing_media_workflows import SQLiteMissingMediaWorkflowRepository
from app.agent.provider_actions import clear_provider_session_state
from app.agent.recent_download_submissions import RecentDownloadSubmissionStore
from app.agent.recent_resource_candidates import RecentResourceCandidateStore
from app.agent.session_context import SQLiteAgentSessionContextRepository

from .capabilities import CapabilityRetriever
from .effects import ConfirmationEffectPlanStore
from .metrics import KernelMetrics
from .lifecycle import AgentSessionLifecycle
from .model import ModelAdapter
from .persistence import SQLiteKernelStore
from .pipeline import ToolPipeline
from .ports import catalog_from_tool_specs
from .ports.mediaflux_effects import MediaFluxEffectLifecycle
from .ports.mediaflux_policy import (
    MediaFluxAuthorizationPolicy,
    MediaFluxToolRateLimiter,
    MediaFluxTurnAdmission,
)
from .provider_model import OpenAICompatibleModelAdapter, ProviderSettings
from .session import AgentSession, SessionLimits
from .transports import TelegramKernelTransport, WebKernelTransport


@dataclass(frozen=True, slots=True)
class RuntimeCreationBinding:
    fence: RuntimeFence
    effective_digest: str
    snapshot: config.RuntimeConfigSnapshot = field(repr=False, compare=False)


def _capture_creation_binding() -> RuntimeCreationBinding:
    snapshot = config.capture_runtime_config_snapshot()
    if not db.resolve_db_path().is_file():
        raise RuntimeFenceError("runtime_database_unavailable")
    with db.get_conn() as conn:
        fence = read_runtime_fence(conn)
        require_runtime_fence(conn, fence)
    if fence.config_digest != snapshot.config_digest:
        raise RuntimeFenceError("runtime_snapshot_mismatch")
    return RuntimeCreationBinding(fence, snapshot.effective_digest, snapshot)


def _recheck_creation_binding(binding: RuntimeCreationBinding) -> None:
    try:
        current = _capture_creation_binding()
    except RuntimeFenceError as exc:
        raise RuntimeFenceError("runtime_changed_during_creation") from exc
    if current != binding:
        raise RuntimeFenceError("runtime_changed_during_creation")


@dataclass(frozen=True, slots=True)
class AgentKernelRuntime:
    session: AgentSession
    store: SQLiteKernelStore
    lifecycle: AgentSessionLifecycle
    metrics: KernelMetrics
    web: WebKernelTransport
    telegram: TelegramKernelTransport
    creation_binding: RuntimeCreationBinding | None = None


_lock = threading.Lock()
_runtime: AgentKernelRuntime | None = None


def _configure_domain_contexts() -> tuple[
    SQLiteAgentSessionContextRepository,
    RecentResourceCandidateStore,
    AgentIngestSessionStore,
    MissingMediaWorkflowRuntime,
]:
    context_repository = SQLiteAgentSessionContextRepository()
    configure_local_media_agent_context(context_repository)
    configure_discovery_mapping_context(context_repository)
    configure_directory_scrape_context(context_repository)
    configure_guangya_rename_context(context_repository)
    configure_guangya_cleanup_context(context_repository)
    configure_guangya_workspace_context(context_repository)
    configure_guangya_fs_change_context(context_repository)
    resource_store = RecentResourceCandidateStore(repository=context_repository)
    ingest_store = AgentIngestSessionStore()
    missing_media_runtime = MissingMediaWorkflowRuntime(
        repository=SQLiteMissingMediaWorkflowRepository(),
        recent_download_store=RecentDownloadSubmissionStore(
            repository=context_repository
        ),
    )
    return context_repository, resource_store, ingest_store, missing_media_runtime


def _session_limits_from_config() -> SessionLimits:
    context_window = config.get_int("AGENT_LLM_CONTEXT_WINDOW_TOKENS", 128_000)
    return SessionLimits(
        context_window_tokens=max(16_384, min(2_000_000, context_window))
    )


def build_agent_kernel_runtime(
    *,
    model: ModelAdapter | None = None,
    store: SQLiteKernelStore | None = None,
    confirmation_store: SQLiteConfirmationStore | None = None,
) -> AgentKernelRuntime:
    if model is not None:
        # Explicit injection never becomes a default paid adapter, even if falsey.
        return _build_agent_kernel_runtime(model=model, store=store, confirmation_store=confirmation_store)
    with config.runtime_config_read_guard():
        binding = _capture_creation_binding()
    with config.use_runtime_config_snapshot(binding.snapshot):
        runtime = _build_agent_kernel_runtime(store=store, confirmation_store=confirmation_store, creation_binding=binding)
    with config.runtime_config_read_guard():
        _recheck_creation_binding(binding)
        return runtime


def _build_agent_kernel_runtime(
    *,
    model: ModelAdapter | None = None,
    store: SQLiteKernelStore | None = None,
    confirmation_store: SQLiteConfirmationStore | None = None,
    creation_binding: RuntimeCreationBinding | None = None,
) -> AgentKernelRuntime:
    """组合唯一 Kernel、领域能力、持久化与两个薄入口。"""
    (
        _context_repository,
        resource_store,
        ingest_store,
        missing_media_runtime,
    ) = _configure_domain_contexts()
    declarations = build_tool_specs(
        resource_store,
        ingest_store,
        missing_media_runtime=missing_media_runtime,
    )
    catalog = catalog_from_tool_specs(declarations)
    kernel_store = store or SQLiteKernelStore()
    effect_store = ConfirmationEffectPlanStore(
        confirmation_store or SQLiteConfirmationStore(),
        record_actions=True,
    )
    pipeline = ToolPipeline(
        catalog=catalog,
        state_store=kernel_store,
        reference_store=kernel_store,
        effect_store=effect_store,
        authorization=MediaFluxAuthorizationPolicy(),
        rate_limiter=MediaFluxToolRateLimiter(),
        effect_lifecycle=MediaFluxEffectLifecycle(
            missing_media_runtime=missing_media_runtime
        ),
    )
    if model is not None:
        bool(model)  # Preserve one observable evaluation, never use it to route.
        active_model = model
        limits = _session_limits_from_config()
    else:
        settings = ProviderSettings.from_config()
        active_model = OpenAICompatibleModelAdapter(settings)
        limits = SessionLimits(context_window_tokens=settings.context_window_tokens)
    session = AgentSession(
        model=active_model,
        catalog=catalog,
        retriever=CapabilityRetriever(),
        pipeline=pipeline,
        state_store=kernel_store,
        journal=kernel_store,
        limits=limits,
        turn_admission=MediaFluxTurnAdmission(),
        stream_display=lambda: config.get_bool("AGENT_STREAM_DISPLAY_ENABLED"),
    )
    metrics = KernelMetrics()
    session.read_wait_binding = creation_binding
    lifecycle = AgentSessionLifecycle(
        session=session,
        store=kernel_store,
        effect_store=effect_store,
        clear_provider_state=clear_provider_session_state,
    )
    return AgentKernelRuntime(
        session=session,
        store=kernel_store,
        lifecycle=lifecycle,
        metrics=metrics,
        web=WebKernelTransport(session, metrics=metrics),
        telegram=TelegramKernelTransport(session, metrics=metrics),
        creation_binding=creation_binding,
    )


def build_agent_kernel(
    *,
    model: ModelAdapter | None = None,
    store: SQLiteKernelStore | None = None,
    confirmation_store: SQLiteConfirmationStore | None = None,
) -> AgentSession:
    """测试与嵌入式调用的便捷入口。"""
    return build_agent_kernel_runtime(
        model=model,
        store=store,
        confirmation_store=confirmation_store,
    ).session


def get_agent_kernel_runtime() -> AgentKernelRuntime:
    global _runtime
    # Never acquire the config lock while holding the cache lock.
    with config.runtime_config_read_guard():
        binding = _capture_creation_binding()
        with _lock:
            if _runtime is not None and _runtime.creation_binding == binding:
                return _runtime
    with config.use_runtime_config_snapshot(binding.snapshot):
        runtime = _build_agent_kernel_runtime(creation_binding=binding)
    with config.runtime_config_read_guard():
        _recheck_creation_binding(binding)
        with _lock:
            if _runtime is not None and _runtime.creation_binding == binding:
                return _runtime
            _runtime = runtime
        return runtime


def get_agent_kernel() -> AgentSession:
    return get_agent_kernel_runtime().session


def invalidate_agent_kernel_runtime() -> None:
    """使下一次入口调用按最新模型与上下文配置重建 Kernel。"""
    global _runtime
    with _lock:
        _runtime = None


def reset_agent_kernel_for_tests() -> None:
    invalidate_agent_kernel_runtime()
