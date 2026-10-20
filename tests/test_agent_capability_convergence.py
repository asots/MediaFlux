"""Agent 能力收敛契约：防止旧工具、隐藏写入口和领域覆盖回归。"""

from __future__ import annotations

from pathlib import Path

from app.agent.domain_catalog import build_tool_specs
from app.agent.provider_operations import build_provider_catalog
from app.agent.public_safety import public_tool_label

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_TOOL_NAMES = {
    "downloads.pause_task",
    "downloads.resume_task",
    "downloads.delete_task",
    "library.refresh_library",
    "indexer.submit_resource",
    "indexer.submit_resource_batch",
    "rss.set_subscription_enabled",
    "rss.set_refresh_interval",
    "guangya.change_plan.execute",
    "guangya.fs.change.preview",
    "guangya.media_hygiene.execute",
    "guangya.organize.clean_empty",
    "discovery.get_watchlist_summary",
}
_REQUIRED_PROJECT_TOOLS = {
    "agent.capabilities",
    "agent.runtime_status",
    "automation.diagnose_pipeline",
    "workspace.briefing",
    "workspace.health",
    "workspace.next_actions",
    "workspace.search",
    "workspace.todo",
    "provider.capabilities",
    "provider.query",
    "provider.change.preview",
    "provider.change.execute",
    "provider.job.status",
    "config.diagnose",
    "config.diagnose_media_servers",
    "config.explain_component",
    "config.feature_summary",
    "config.safe_policy_summary",
    "config.set_feature_state",
    "config.set_safe_policy",
    "library.search",
    "library.batch_presence",
    "library.audit_episodes",
    "library.audit_library_episodes",
    "library.check_updates",
    "library.count_series_episodes",
    "library.search_missing_episode_resources",
    "library.search_missing_season_resources",
    "library.patrol_status",
    "library.set_patrol_policy",
    "library.trigger_patrol_now",
    "downloads.diagnose_queue",
    "downloads.request_summaries",
    "downloads.retry_submission",
    "config.indexer_sites_summary",
    "config.set_indexer_sites",
    "indexer.diagnose_readiness",
    "indexer.search_resources",
    "ingest.inspect",
    "ingest.submit",
    "ingest.status",
    "web.read",
    "web.search",
    "bangumi.calendar",
    "discovery.search",
    "discovery.person_filmography",
    "discovery.detail",
    "discovery.recommend",
    "discovery.lookup_rating",
    "discovery.mapping_candidates",
    "discovery.confirm_mapping",
    "discovery.watchlist_summaries",
    "discovery.add_watchlist",
    "discovery.remove_watchlist",
    "media.subscription_summaries",
    "media.subscription_updates",
    "media.get_subscription_summary",
    "media.get_subscription_policy",
    "media.create_subscription",
    "media.set_subscription_policy",
    "media.set_subscription_enabled",
    "media.delete_subscription",
    "media.subscription_notification_rule",
    "media.set_subscription_notification_rule",
    "media.reset_subscription_notification_rule",
    "media.continue_watching",
    "media.recently_added",
    "media.recently_played",
    "media.recommend_from_library",
    "guangya.account.status",
    "guangya.connection_status",
    "guangya.fs.query",
    "guangya.fs.change.execute",
    "guangya.recycle.list",
    "guangya.recycle.restore",
    "guangya.recycle.clear",
    "guangya.operation.status",
    "guangya.share.list",
    "guangya.share.create",
    "guangya.share.revoke",
    "guangya.organize.preview",
    "guangya.organize.run_once",
    "guangya.organize.status",
    "guangya.organize.cleanup.classify",
    "guangya.organize.cleanup.preview",
    "guangya.organize.cleanup.execute",
    "guangya.rename.preview",
    "guangya.rename.execute",
    "guangya.directory_scrape.inspect",
    "guangya.directory_scrape.search",
    "guangya.directory_scrape.preview",
    "guangya.directory_scrape.run",
    "strm.diagnose",
    "strm.status",
    "strm.run_history",
    "strm.run_once",
    "strm.retry_failures",
    "strm.schedule_policy",
    "strm.set_schedule_policy",
    "local_media.diagnose",
    "local_media.source_summaries",
    "local_media.get_source_summary",
    "local_media.task_summaries",
    "local_media.inspect_task",
    "local_media.preview_task",
    "local_media.scan_sources",
    "local_media.retry_task",
    "local_media.refresh_task_library",
    "local_media.verify_task_library_visibility",
    "local_media.set_source_trigger_enabled",
    "rss.diagnose",
    "rss.subscription_summaries",
    "rss.get_subscription_summary",
    "rss.create_subscription",
    "rss.update_subscription",
    "rss.delete_subscription",
    "rss.recent_activity",
    "rss.entry_summaries",
    "rss.mark_entries",
    "rss.refresh_subscription",
    "rss.refresh_subscriptions",
    "rss.submit_entries_to_qb",
    "rss.submit_pending_to_qb",
    "rss.retry_failed",
}
_REQUIRED_PROVIDER_OPERATIONS = {
    "media.system.info",
    "media.libraries.list",
    "media.library.counts",
    "media.items.search",
    "media.items.recent_added",
    "media.items.recent_played",
    "media.items.continue_watching",
    "media.series.search",
    "media.series.episodes",
    "media.library.refresh",
    "media.item.refresh",
    "qb.app.version",
    "qb.transfer.info",
    "qb.torrents.info",
    "qb.torrents.files",
    "qb.torrents.pause",
    "qb.torrents.resume",
    "qb.torrents.delete_task",
}


def test_provider_catalog_is_the_only_external_media_and_qb_write_surface() -> None:
    operations = {
        spec.operation_id: spec for spec in build_provider_catalog().operations()
    }
    assert _REQUIRED_PROVIDER_OPERATIONS <= operations.keys()
    for operation in (
        "media.library.refresh",
        "media.item.refresh",
        "qb.torrents.pause",
        "qb.torrents.resume",
        "qb.torrents.delete_task",
    ):
        assert operations[operation].risk.value != "read"


def test_all_registered_tools_have_one_complete_kernel_lifecycle() -> None:
    """防止新增能力只有声明、只有预览或引用到不存在的下一环。"""
    specs = build_tool_specs()
    by_name = {spec.name: spec for spec in specs}
    assert len(by_name) == len(specs)
    assert not _FORBIDDEN_TOOL_NAMES & by_name.keys()
    assert _REQUIRED_PROJECT_TOOLS <= by_name.keys()

    for spec in specs:
        assert public_tool_label(spec.name) != "项目操作", spec.name
        for related in spec.related_tools:
            assert related in by_name, f"{spec.name} -> {related}"
        if spec.risk.value == "read":
            assert spec.handler is not None or spec.context_handler is not None, spec.name
            assert not spec.requires_confirmation, spec.name
            continue
        assert spec.requires_confirmation, spec.name
        assert spec.context_confirmation_preparer is not None, spec.name
        assert spec.context_confirmed_handler is not None, spec.name


def test_watchlist_reads_have_one_read_only_entry_and_no_retired_handler_or_alias() -> None:
    from app.agent import domain_tool_metadata, public_safety, rate_limit
    from app.agent.domains.discovery import (
        watchlist_actions as discovery_watchlist_actions,
    )
    from app.agent.kernel.capabilities import ToolEffect
    from app.agent.kernel.ports import catalog_from_tool_specs

    specs = build_tool_specs()
    watchlist_specs = {spec.name: spec for spec in specs if spec.name.startswith('discovery.') and 'watchlist' in spec.name}
    assert set(watchlist_specs) == {
        'discovery.watchlist_summaries', 'discovery.add_watchlist', 'discovery.remove_watchlist',
    }
    spec = watchlist_specs['discovery.watchlist_summaries']
    assert spec.risk.value == 'read'
    assert spec.handler is discovery_watchlist_actions.list_watchlist_summaries
    assert spec.validator is discovery_watchlist_actions.watchlist_summaries_arguments
    assert not spec.requires_confirmation and spec.parallel_read

    catalog = catalog_from_tool_specs(specs)
    retired = 'discovery.get_watchlist_summary'
    assert not catalog.has(retired)
    assert not catalog.has(retired.replace('.', '__'))
    assert not hasattr(discovery_watchlist_actions, 'get_watchlist_summary')
    for mapping in (
        domain_tool_metadata.TOOL_RETRIEVAL_TERMS, public_safety._PUBLIC_TOOL_LABELS,
        public_safety._PUBLIC_TOOL_ALIASES, rate_limit._TOOL_RATE_LIMIT_SCOPES,
        rate_limit._TOOL_RATE_LIMITS,
    ):
        assert retired not in mapping
        assert retired.replace('.', '__') not in mapping
    assert retired not in domain_tool_metadata.loaded_tool_guidance([spec.name])

    tool = catalog.get(spec.name)
    assert tool.effect is ToolEffect.READ and tool.read is not None
    assert tool.prepare is None and tool.execute_confirmed is None
    assert tool.metadata['parallel_read'] is True
    assert set(tool.input_schema['properties']) == {'mode', 'before_number', 'watchlist_number'}
    assert tool.input_schema['additionalProperties'] is False


def test_guangya_fs_change_is_one_full_plan_confirmation_tool() -> None:
    from app.agent.kernel.capabilities import ToolEffect
    from app.agent.kernel.ports import catalog_from_tool_specs

    specs = build_tool_specs()
    spec = next(item for item in specs if item.name == "guangya.fs.change.execute")
    assert spec.risk.value == "danger"
    assert spec.requires_confirmation
    assert spec.handler is None and spec.context_handler is None
    assert spec.parameters["required"] == ["operations"]
    assert set(spec.parameters["properties"]) == {"operations", "observation_ref", "trigger_strm"}
    assert spec.parameters["additionalProperties"] is False
    operations = spec.parameters["properties"]["operations"]
    assert operations["minItems"] == 1
    assert operations["maxItems"] == 200
    operation_names = {
        name
        for branch in operations["items"]["oneOf"]
        for name in branch["properties"]["op"]["enum"]
    }
    assert operation_names == {
        "rename", "move", "copy", "relocate", "batch_relocate", "batch_files", "create_directory", "trash",
    }

    catalog = catalog_from_tool_specs(specs)
    assert not catalog.has("guangya.fs.change.preview")
    assert not catalog.has("guangya__fs__change__preview")
    tool = catalog.get(spec.name)
    assert tool.effect is ToolEffect.DANGER
    assert tool.read is None
    assert tool.prepare is not None and tool.execute_confirmed is not None
    assert tool.metadata["workflow"] == "guangya_fs_change"
    assert tool.metadata["workflow_stage"] == 10
    assert tool.metadata["parallel_read"] is False


def test_guangya_fs_change_retired_preview_has_no_runtime_metadata_or_alias() -> None:
    from app.agent import (
        action_history,
        confirmation_contract,
        domain_tool_metadata,
        public_safety,
        rate_limit,
    )

    execute = "guangya.fs.change.execute"
    retired = "guangya.fs.change.preview"
    mappings = (
        action_history._TOOL_LABELS,
        action_history._SAFE_FIELDS,
        confirmation_contract._CONFIRMATION_COPY,
        domain_tool_metadata.TOOL_RETRIEVAL_TERMS,
        public_safety._PUBLIC_TOOL_LABELS,
        public_safety._PUBLIC_TOOL_ALIASES,
        rate_limit._TOOL_RATE_LIMIT_SCOPES,
        rate_limit._TOOL_RATE_LIMITS,
    )
    for mapping in mappings:
        assert retired not in mapping
        assert retired.replace(".", "__") not in mapping
    assert action_history._TOOL_LABELS[execute] == public_tool_label(execute) == "光鸭文件变更"
    assert public_tool_label(execute.replace(".", "__")) == "光鸭文件变更"
    assert "复制" in domain_tool_metadata.TOOL_RETRIEVAL_TERMS[execute]
    guidance = domain_tool_metadata.loaded_tool_guidance([execute])
    assert retired not in guidance
    assert "必填的完整 operations" in guidance
    assert "只有用户确认后才执行" in guidance
    assert "不能用空参数" in guidance
    reference = (_REPO_ROOT / "docs/tutorials/09_Agent工具与能力参考.md").read_text()
    assert retired not in reference
    assert "不接受 `{}`" in reference


def test_guangya_fs_change_intents_retrieve_the_single_execute_entry() -> None:
    from app.agent.kernel.capabilities import CapabilityRetriever
    from app.agent.kernel.ports import catalog_from_tool_specs

    catalog = catalog_from_tool_specs(build_tool_specs())
    for message in ("把刚才这个文件复制到备份目录", "在光鸭新建目录", "移动光鸭文件并生成文件变更确认卡"):
        names = CapabilityRetriever().retrieve(message, catalog).names
        assert "guangya.fs.change.execute" in names, (message, names)
        assert "guangya.fs.change.preview" not in names


def test_guangya_fs_change_uses_one_policy_with_separate_confirmation_budget(monkeypatch) -> None:
    from app.agent import rate_limit

    execute = "guangya.fs.change.execute"
    assert rate_limit.tool_rate_limit_policy(execute) == (f"tool:{execute}", 3, 1)
    limiter = rate_limit.AgentRateLimiter(clock=lambda: 100.0)
    monkeypatch.setattr(rate_limit, "agent_rate_limiter", limiter)
    for remaining in (2, 1, 0):
        decision = rate_limit.consume_agent_tool("fs-owner", execute)
        assert decision.allowed and decision.remaining == remaining
    assert not rate_limit.consume_agent_tool("fs-owner", execute).allowed
    # 准备额度耗尽不能阻止用户确认已有卡片；仍使用唯一 execute 的限流合同。
    confirmed = rate_limit.consume_agent_tool("fs-owner", execute, scope_suffix="confirmed")
    assert confirmed.allowed and confirmed.limit == 3 and confirmed.remaining == 2


def test_provider_operations_are_bounded_and_writes_are_reference_scoped() -> None:
    for spec in build_provider_catalog().operations():
        assert spec.max_items > 0, spec.operation_id
        assert spec.timeout_seconds > 0, spec.operation_id
        if spec.risk.value != "read":
            assert spec.reference_arguments, spec.operation_id


def test_batch_updates_use_one_read_and_recheck_the_persisted_list(monkeypatch) -> None:
    """十部片单与省略标题的续问穿过真实目录/Kernel/审计；不连接真实媒体库。"""
    import asyncio
    import json
    from collections import Counter

    from app.agent.domains.library import episode_audit
    from app.agent.kernel.capabilities import CapabilityRetriever
    from app.agent.kernel.events import AgentEventType
    from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
    from app.agent.kernel.persistence import SQLiteKernelStore
    from app.agent.kernel.pipeline import ToolPipeline
    from app.agent.kernel.ports import catalog_from_tool_specs
    from app.agent.kernel.session import AgentSession, SessionLimits
    from app.agent.kernel.state_types import AgentInput
    from tests.support import isolated_test_database
    from tests.test_agent_kernel_core import ScriptedModel

    titles = ["师兄啊师兄", "凡人修仙传", "光阴之外", "择日飞升", "大主宰",
              "牧神记", "沧元图", "一斩苍穹", "东大高武学院", "将夜"]
    identities = {title: str(10000 + index) for index, title in enumerate(titles)}
    inventory = [(1, 1)]
    reads = []

    def inspect(query, **_kwargs):
        reads.append(query)
        return [{
            "server_type": "jellyfin", "server_name": "测试媒体库", "status": "ready",
            "selected": {"name": query, "year": "2026", "tmdb_id": identities[query]},
            "episodes": list(inventory), "truncated": False,
        }]

    class TMDB:
        def detail(self, tmdb_id, _media_type):
            return {"name": next(title for title in titles if identities[title] == tmdb_id),
                    "first_air_date": "2026-01-01", "seasons": [{"season_number": 1}]}

        def tv_season_detail(self, _tmdb_id, _season):
            return {"episodes": [{"episode_number": number, "air_date": "2026-01-01"} for number in (1, 2)]}

    monkeypatch.setattr(episode_audit, "inspect_series_episode_sources", inspect)
    monkeypatch.setattr(episode_audit, "TMDBClient", TMDB)
    monkeypatch.setattr(episode_audit, "close_tmdb_client", lambda _client: None)
    episode_audit.reset_episode_audit_cache_for_tests()

    async def run():
        catalog = catalog_from_tool_specs(build_tool_specs())
        state = SQLiteKernelStore()
        for turn in (1, 2):
            if turn == 2:
                inventory.append((1, 2))
            model = ScriptedModel([
                [ModelEvent(ModelEventType.TOOL_CALL_COMPLETED, tool_call=ModelToolCall(
                    f"batch-{turn}", "library.check_updates",
                    {"queries": titles, "media_type": "tv", "as_of": "2026-08-01"},
                )), ModelEvent(ModelEventType.FINISH, finish_reason="tool_calls")],
                [ModelEvent(ModelEventType.TEXT_DELTA, text="本轮已逐部核对，未检索全网资源。"),
                 ModelEvent(ModelEventType.FINISH, finish_reason="stop")],
            ])
            # 重建会话对象，验证续问依赖持久对话，而不是旧实例中的临时片单。
            session = AgentSession(
                model=model, catalog=catalog, retriever=CapabilityRetriever(),
                pipeline=ToolPipeline(catalog=catalog, state_store=state), state_store=state,
                limits=SessionLimits(max_tool_calls=2, max_model_rounds=4),
            )
            events = [event async for event in session.run(AgentInput(
                owner="batch-owner", session_id="batch-session", request_id=f"request-{turn}",
                message="媒体库中的 " + "/".join(titles) + " 都有更新吗？" if turn == 1 else "再检查一下更新",
            ))]
            selected = next(event for event in events if event.type == AgentEventType.CAPABILITIES_SELECTED)
            assert "library.check_updates" in selected.payload["tools"]
            assert events[-1].type == AgentEventType.TURN_COMPLETED
            assert events[-1].payload["tool_calls"] == 1
            assert not any(event.type == AgentEventType.EFFECT_APPROVAL_REQUIRED for event in events)
            tool_message = next(message for message in reversed(model.requests[-1].messages) if message.role == "tool")
            data = json.loads(tool_message.content)["data"]
            assert [item["query"] for item in data["items"]] == titles
            assert {item["status"] for item in data["items"]} == {"updates_available" if turn == 1 else "up_to_date"}
            assert data["counts"]["requested"] == 10
            if turn == 2:
                history = "\n".join(message.content for message in model.requests[0].messages)
                assert all(title in history for title in titles)
        assert Counter(reads) == Counter({title: 2 for title in titles})

    try:
        with isolated_test_database():
            asyncio.run(run())
    finally:
        episode_audit.reset_episode_audit_cache_for_tests()
