"""按领域组合 MediaFlux Agent 的原子工具声明。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agent.domains.activity.tools import register_specs as register_activity_specs
from app.agent.domains.automation.tools import (
    register_specs as register_automation_rules_specs,
)
from app.agent.domains.configuration.tools import (
    register_specs as register_configuration_management_specs,
)
from app.agent.domains.discovery.mapping_actions import DiscoveryMappingActions
from app.agent.domains.discovery.tools import register_specs as register_discovery_specs
from app.agent.domains.downloads.tools import register_specs as register_download_specs
from app.agent.domains.guangya.cleanup_actions import GuangyaCleanupActions
from app.agent.domains.guangya.directory_scrape_actions import DirectoryScrapeActions
from app.agent.domains.guangya.fs_change_actions import GuangyaFsChangeActions
from app.agent.domains.guangya.rename_actions import GuangyaRenameActions
from app.agent.domains.guangya.sdk_tools import (
    register_specs as register_cloud_sdk_specs,
)
from app.agent.domains.guangya.tools import register_specs as register_cloud_specs
from app.agent.domains.guangya.workspace_actions import GuangyaWorkspaceActions
from app.agent.domains.library.missing_workflow_runtime import (
    MissingMediaWorkflowRuntime,
)
from app.agent.domains.library.quality_tools import (
    register_specs as register_library_quality_specs,
)
from app.agent.domains.library.tools import register_specs as register_library_specs
from app.agent.domains.local_media.task_actions import LocalMediaTaskActions
from app.agent.domains.local_media.tools import (
    register_specs as register_local_media_specs,
)
from app.agent.domains.playback.tools import register_specs as register_playback_specs
from app.agent.domains.resources.candidate_context import RecentResourceCandidateStore
from app.agent.domains.resources.ingest_actions import (
    AgentIngestSessionStore,
    IngestActions,
)
from app.agent.domains.resources.tools import register_specs as register_resource_specs
from app.agent.domains.strm.metadata_tools import (
    register_specs as register_strm_metadata_specs,
)
from app.agent.domains.strm.tools import register_specs as register_strm_specs
from app.agent.domains.subscriptions.tools import (
    register_specs as register_subscription_specs,
)
from app.agent.domains.system.tools import register_specs as register_system_specs
from app.agent.domains.workspace.tools import register_specs as register_workspace_specs
from app.agent.models import ToolSpec
from app.agent.session_context import AgentSessionContextRepository


@dataclass(slots=True)
class ToolSpecCollector:
    items: list[ToolSpec] = field(default_factory=list)

    def register(self, spec: ToolSpec) -> None:
        if any(item.name == spec.name for item in self.items):
            raise ValueError(f"duplicate tool: {spec.name}")
        self.items.append(spec)


class DomainActions:
    """一个 runtime 的领域依赖；绑定后不再修改模块级仓储。"""

    def __init__(self, repository: AgentSessionContextRepository | None = None) -> None:
        self.local_media = LocalMediaTaskActions(repository)
        self.discovery_mapping = DiscoveryMappingActions(repository)
        self.directory_scrape = DirectoryScrapeActions(repository)
        self.rename = GuangyaRenameActions(repository)
        self.cleanup = GuangyaCleanupActions(repository)
        self.workspace = GuangyaWorkspaceActions(repository)
        self.fs_change = GuangyaFsChangeActions(repository, workspace=self.workspace)

    def clear_session(self, *, owner: str, session_id: str) -> None:
        self.local_media.clear_local_media_agent_context(owner=owner, session_id=session_id)
        self.discovery_mapping.clear_session(owner=owner, session_id=session_id)
        self.rename.clear_session(owner=owner, session_id=session_id)
        self.directory_scrape.clear_session(owner=owner, session_id=session_id)
        self.cleanup.clear_session(owner=owner, session_id=session_id)


def build_tool_specs(
    recent_resource_store: RecentResourceCandidateStore | None = None,
    ingest_store: AgentIngestSessionStore | None = None,
    missing_media_runtime: MissingMediaWorkflowRuntime | None = None,
    domain_actions: DomainActions | None = None,
) -> tuple[ToolSpec, ...]:
    collector = ToolSpecCollector()
    resource_store = recent_resource_store or RecentResourceCandidateStore()
    active_ingest_store = ingest_store or AgentIngestSessionStore()
    ingest_actions = IngestActions(
        store=active_ingest_store,
        recent_resource_store=resource_store,
    )
    actions = domain_actions if domain_actions is not None else DomainActions()
    register_activity_specs(collector)
    register_library_quality_specs(collector)
    register_automation_rules_specs(collector)
    register_configuration_management_specs(collector)
    register_system_specs(collector)
    register_local_media_specs(collector, task_actions=actions.local_media)
    register_download_specs(collector)
    register_subscription_specs(collector)
    register_playback_specs(collector)
    register_strm_specs(collector)
    register_strm_metadata_specs(collector)
    register_cloud_specs(
        collector,
        cleanup_actions=actions.cleanup,
        directory_scrape_actions=actions.directory_scrape,
        fs_change_actions=actions.fs_change,
        rename_actions=actions.rename,
        workspace_actions=actions.workspace,
    )
    register_cloud_sdk_specs(collector, workspace_actions=actions.workspace)
    register_discovery_specs(collector, mapping_actions=actions.discovery_mapping)
    register_resource_specs(collector, ingest_actions=ingest_actions)
    register_workspace_specs(collector)
    register_library_specs(collector, missing_media_runtime=missing_media_runtime)
    return tuple(collector.items)
