"""构建第二个 runtime 不能重定向第一个 runtime 的领域上下文。"""
from __future__ import annotations
import hashlib
import json
from app.agent.session_context import AgentSessionContextRepository
from app.agent.kernel.model import ModelAdapter
from app.agent.kernel.persistence import SQLiteKernelStore
from typing import cast

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.agent.domain_catalog import DomainActions, build_tool_specs
from app.agent.errors import AgentToolError
from app.agent.kernel.bootstrap import build_agent_kernel_runtime
from app.agent.kernel.state import InMemorySessionStateStore
from app.agent.models import ToolContext


class EmptyRepository:
    def __init__(self) -> None:
        self.reads: list[tuple[str, str]] = []

    def get_latest(self, *, owner, context_type, now):
        self.reads.append((owner, context_type))
        return None


@pytest.mark.parametrize(
    "tool_name, entrypoint, arguments, context_type",
    [
        ("local_media.inspect_task", "context_handler", {"task_number": 1}, "local_media_tasks"),
        ("discovery.confirm_mapping", "context_confirmation_preparer", {"candidate_number": 1}, "discovery_mapping"),
        ("guangya.directory_scrape.run", "context_confirmation_preparer", {}, "directory_scrape"),
        ("guangya.rename.execute", "context_confirmation_preparer", {}, "guangya_rename"),
        ("guangya.organize.cleanup.execute", "context_confirmation_preparer", {}, "guangya_cleanup"),
        ("guangya.fs.change.execute", "context_confirmed_handler", {}, "guangya_fs_change"),
    ],
)
def test_existing_runtime_keeps_its_repository_after_another_build(
    tool_name, entrypoint, arguments, context_type,
):
    first, second = EmptyRepository(), EmptyRepository()
    with patch(
        "app.agent.kernel.bootstrap.SQLiteAgentSessionContextRepository",
        side_effect=[first, second],
    ):
        runtime = build_agent_kernel_runtime(model=cast(ModelAdapter | None, object()), store=cast(SQLiteKernelStore | None, InMemorySessionStateStore()))
        tools = {tool.name: tool for tool in build_tool_specs(domain_actions=runtime.domain_actions)}
        callback = getattr(tools[tool_name], entrypoint)
        build_agent_kernel_runtime(model=cast(ModelAdapter | None, object()), store=cast(SQLiteKernelStore | None, InMemorySessionStateStore()))

    context = ToolContext(owner="same-owner", session_id="same-session")
    with pytest.raises(AgentToolError):
        if entrypoint == "context_confirmed_handler":
            callback(arguments, "old-fingerprint", context)
        else:
            callback(arguments, context)
    expected_owner = (
        runtime.domain_actions.local_media._context_store._scope_key("same-owner", "same-session")
        if context_type == "local_media_tasks" else
        hashlib.sha256(json.dumps([context.owner, context.session_id]).encode()).hexdigest()
        if context_type == "discovery_mapping" else "same-owner"
    )
    assert first.reads == [(expected_owner, context_type)]
    assert second.reads == []


def test_workspace_cursor_uses_the_original_runtime_repository():
    first, second = EmptyRepository(), EmptyRepository()
    with patch(
        "app.agent.kernel.bootstrap.SQLiteAgentSessionContextRepository",
        side_effect=[first, second],
    ):
        runtime = build_agent_kernel_runtime(model=cast(ModelAdapter | None, object()), store=cast(SQLiteKernelStore | None, InMemorySessionStateStore()))
        build_agent_kernel_runtime(model=cast(ModelAdapter | None, object()), store=cast(SQLiteKernelStore | None, InMemorySessionStateStore()))
    assert runtime.domain_actions.workspace.latest_guangya_observation_ref("same-owner") == ""
    assert first.reads == [("same-owner", "guangya_workspace")]
    assert second.reads == []


def test_independent_runtime_memory_does_not_share_local_task_numbers():
    first, second = DomainActions(), DomainActions()
    task = SimpleNamespace(id=41, version=1, status="failed", source_id=7)
    first.local_media._context_store.capture_tasks(owner="same-owner", session_id="same-session", tasks=[task])
    with patch("app.agent.domains.local_media.task_actions.db.get_local_media_task", return_value=task):
        assert first.local_media._current_task("same-owner", "same-session", 1)[1] is task
        with pytest.raises(AgentToolError, match="任务序号不存在"):
            second.local_media._current_task("same-owner", "same-session", 1)


def test_file_change_uses_its_runtime_workspace_for_latest_selection():
    first, second = DomainActions(cast(AgentSessionContextRepository | None, EmptyRepository())), DomainActions(cast(AgentSessionContextRepository | None, EmptyRepository()))
    with (
        patch.object(first.workspace, "latest_guangya_observation_ref", return_value="" ) as original,
        patch.object(second.workspace, "latest_guangya_observation_ref", return_value="other-runtime") as other,
    ):
        with pytest.raises(AgentToolError):
            first.fs_change.prepare_guangya_fs_change_confirmation(
                {"operation": "rename", "operations": []}, ToolContext(owner="same-owner"),
            )
    original.assert_called_once_with("same-owner")
    other.assert_not_called()
