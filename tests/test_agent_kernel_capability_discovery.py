"""实际Kernel循环的动态能力窗口，不调用外部模型或真实Provider。"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import pytest

from app.agent.capability_discovery_actions import (
    capability_arguments,
    web_capability_status,
)
from app.agent.domain_catalog import build_tool_specs
from app.agent.errors import AgentToolError
from app.agent.kernel.capabilities import (
    CapabilityRetriever,
    CapabilitySelection,
    KernelToolSpec,
    ToolCatalog,
    ToolEffect,
)
from app.agent.kernel.discovery import CapabilityDiscovery
from app.agent.kernel.effects import PreparedEffect
from app.agent.kernel.events import AgentEventType
from app.agent.kernel.model import ModelEvent, ModelEventType, ModelToolCall
from app.agent.kernel.pipeline import ToolPipeline, ToolPipelineError
from app.agent.kernel.ports.existing_actions import adapt_tool_spec
from app.agent.kernel.projection import DefaultProjector
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import AgentInput, InMemorySessionStateStore
from tests.test_agent_kernel_core import ScriptedModel, collect, read_tool


def capability_tool():
    return adapt_tool_spec(
        next(spec for spec in build_tool_specs() if spec.name == "agent.capabilities")
    )


def call(name, args=None, call_id="call-1"):
    return ModelEvent(
        ModelEventType.TOOL_CALL_COMPLETED,
        tool_call=ModelToolCall(call_id, name, args or {}),
    )


def final():
    return [ModelEvent(ModelEventType.TEXT_DELTA, text="仅依据工具返回的结果回答。")]


class WrongInitialRetriever(CapabilityRetriever):
    """模拟词法初选漏掉真正的目标；发现阶段必须能独立检索。"""

    def retrieve(self, message, catalog, *, context=None):
        return CapabilitySelection((catalog.get("noise.status"),), {"noise.status": 1})


def session_for(targets, rounds, *, projector=None):
    catalog = ToolCatalog(
        [
            capability_tool(),
            read_tool(
                "noise.status", domain="noise", description="irrelevant old activity"
            ),
            *targets,
        ]
    )
    store = InMemorySessionStateStore()
    model = ScriptedModel(rounds)
    pipeline = ToolPipeline(catalog=catalog, state_store=store, projector=projector)
    session = AgentSession(
        model=model,
        catalog=catalog,
        retriever=WrongInitialRetriever(),
        pipeline=pipeline,
        state_store=store,
    )
    return session, model


def request():
    return AgentInput(
        message="查另一部作品的演员，必要时联网核实",
        owner="owner-1",
        session_id="session-1",
    )


class DynamicDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_missed_read_capability_is_discovered_and_called_in_same_loop(self):
        invoked = []
        target = read_tool(
            "metadata.cast",
            description="查询影视演员角色演职员详情",
            handler=lambda a, c: (
                invoked.append(c.owner) or {"summary": "已取得真实演员表"}
            ),
        )
        session, model = session_for(
            [target],
            [
                [call("agent__capabilities", {"query": "查询演员角色演职员详情"})],
                [call("metadata__cast")],
                final(),
            ],
        )
        events = await collect(session.run(request()))
        self.assertEqual(events[-1].type, AgentEventType.TURN_COMPLETED)
        self.assertEqual(invoked, ["owner-1"])
        self.assertNotIn(
            "metadata__cast", [tool["name"] for tool in model.requests[0].tools]
        )
        self.assertIn(
            "agent__capabilities", [tool["name"] for tool in model.requests[0].tools]
        )
        self.assertIn(
            "metadata__cast", [tool["name"] for tool in model.requests[1].tools]
        )
        selected = [
            event
            for event in events
            if event.type is AgentEventType.CAPABILITIES_SELECTED
        ]
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[1].payload["reason"], "discovery")
        self.assertTrue(all(len(req.tools) <= 10 for req in model.requests))

    async def test_new_schema_cannot_be_guessed_in_same_model_batch(self):
        invoked = []
        target = read_tool(
            "metadata.cast",
            handler=lambda a, c: invoked.append(True) or {"summary": "演员表"},
        )
        session, _model = session_for(
            [target],
            [
                [
                    call(
                        "agent.capabilities", {"tool_names": ["metadata.cast"]}, "find"
                    ),
                    call("metadata.cast", call_id="guess"),
                ],
                [call("metadata.cast", call_id="loaded")],
                final(),
            ],
        )
        events = await collect(session.run(request()))
        failures = [
            event for event in events if event.type is AgentEventType.TOOL_FAILED
        ]
        self.assertEqual(failures[0].payload["code"], "tool_not_available")
        self.assertEqual(failures[0].payload["call_id"], "guess")
        self.assertEqual(invoked, [True])

    async def test_second_discovery_in_same_batch_cannot_erase_successful_first(self):
        invoked = []
        foo = read_tool(
            "metadata.foo",
            handler=lambda a, c: invoked.append("foo") or {"summary": "foo"},
        )
        bar = read_tool(
            "metadata.bar",
            handler=lambda a, c: invoked.append("bar") or {"summary": "bar"},
        )
        session, model = session_for(
            [foo, bar],
            [
                [
                    call(
                        "agent.capabilities", {"tool_names": ["metadata.foo"]}, "first"
                    ),
                    call(
                        "agent.capabilities", {"tool_names": ["metadata.bar"]}, "second"
                    ),
                ],
                [
                    call("metadata.foo", call_id="foo"),
                    call(
                        "agent.capabilities", {"tool_names": ["metadata.bar"]}, "retry"
                    ),
                ],
                [call("metadata.bar", call_id="bar")],
                final(),
            ],
        )
        events = await collect(session.run(request()))
        self.assertEqual(invoked, ["foo", "bar"])
        self.assertIn("metadata__foo", [t["name"] for t in model.requests[1].tools])
        self.assertIn("metadata__bar", [t["name"] for t in model.requests[2].tools])
        failures = [
            event for event in events if event.type is AgentEventType.TOOL_FAILED
        ]
        self.assertEqual(
            [event.payload["code"] for event in failures],
            ["capability_discovery_pending"],
        )

    async def test_discovered_write_only_prepares_and_confirmation_does_not_use_model(
        self,
    ):
        written = []
        target = KernelToolSpec(
            name="metadata.cancel",
            domain="metadata",
            description="取消冻结积压任务",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            effect=ToolEffect.WRITE,
            prepare=lambda a, c: PreparedEffect(
                preview={"summary": "取消待办，不删文件"},
                snapshot_fingerprint="queue-v1",
            ),
            execute_confirmed=lambda a, fingerprint, c: (
                written.append(fingerprint) or {"ok": True, "summary": "已取消待办"}
            ),
        )
        session, model = session_for(
            [target],
            [
                [call("agent.capabilities", {"tool_names": ["metadata.cancel"]})],
                [call("metadata.cancel")],
            ],
        )
        events = await collect(session.run(request()))
        self.assertFalse(written)
        approval = next(
            event
            for event in events
            if event.type is AgentEventType.EFFECT_APPROVAL_REQUIRED
        )
        count = len(model.requests)
        confirmed = await collect(
            session.confirm(
                owner="owner-1",
                session_id="session-1",
                plan_id=approval.payload["plan"]["plan_id"],
            )
        )
        self.assertEqual(written, ["queue-v1"])
        self.assertEqual(len(model.requests), count)
        self.assertIn(
            AgentEventType.EFFECT_COMPLETED, [event.type for event in confirmed]
        )

    async def test_failed_projection_does_not_grant_discovered_tools(self):
        class FailedProjection(DefaultProjector):
            def project(self, value):
                if hasattr(value, "data") and "loaded_tools" in value.data:
                    raise ToolPipelineError(
                        "Projection failed", code="invalid_tool_result"
                    )
                return super().project(value)

        target = read_tool("metadata.cast")
        session, model = session_for(
            [target],
            [
                [call("agent.capabilities", {"tool_names": ["metadata.cast"]})],
                final(),
            ],
            projector=FailedProjection(),
        )
        events = await collect(session.run(request()))
        self.assertNotIn(
            "metadata__cast", [tool["name"] for tool in model.requests[1].tools]
        )
        self.assertEqual(
            len([e for e in events if e.type is AgentEventType.CAPABILITIES_SELECTED]),
            1,
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"availability": lambda c: False},
        {"authorize": lambda c: c.get("owner") == "other-owner"},
        {"context_requirements": frozenset({"media_item"})},
    ],
)
def test_hidden_or_unauthorized_capabilities_are_not_exposed(extra):
    tool = replace(read_tool("secret.inspect", description="演员信息"), **extra)
    finder = CapabilityDiscovery(ToolCatalog([tool]), context={"owner": "owner-1"})
    result = finder.search({"tool_names": ["secret.inspect"]})
    assert result["tools"] == []
    assert result["loaded_tools"] == []
    assert "演员信息" not in json.dumps(result, ensure_ascii=False)
    assert not finder.consume()


def test_discovery_has_its_own_budget_and_no_cross_turn_mutation():
    catalog = ToolCatalog([read_tool("metadata.cast")])
    finder = CapabilityDiscovery(catalog, context={})
    for _ in range(4):
        finder.search({"tool_names": ["metadata.cast"]})
        finder.consume()
    with pytest.raises(ToolPipelineError, match="4次"):
        finder.search({"query": "演员"})
    assert CapabilityDiscovery(catalog, context={}).requests == 0
    assert len(catalog) == 1


def test_zero_match_exposes_only_domain_overview_not_fake_available_tools():
    finder = CapabilityDiscovery(
        ToolCatalog([read_tool("library.inspect", description="媒体库存状态")]),
        context={},
    )
    result = finder.search({"query": "zzzzzzzzzzzzzzzzzzzz"})
    assert not result["tools"]
    assert not result["loaded_tools"]
    assert result["domains"] == [{"domain": "library", "count": 1}]


@pytest.mark.parametrize(
    ("enabled", "key", "status"),
    [
        (False, "", "disabled"),
        (True, "", "configuration_missing"),
        (True, "private-api-key-do-not-leak", "configured"),
    ],
)
def test_web_configuration_is_not_confused_with_provider_reachability(
    enabled, key, status
):
    with (
        patch(
            "app.agent.capability_discovery_actions.config.get_bool",
            return_value=enabled,
        ),
        patch("app.agent.capability_discovery_actions.config.get", return_value=key),
    ):
        result = web_capability_status()
    assert result["status"] == status
    assert "private-api-key" not in json.dumps(result)
    if status == "configured":
        assert "实际调用核验" in result["reason"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": ""},
        {"query": 2},
        {"tool_names": "web.search"},
        {"tool_names": []},
        {"tool_names": ["x"] * 7},
        {"query": "actors", "tool_names": ["discovery.credits"]},
        {"execute": True},
    ],
)
def test_discovery_arguments_are_bounded(arguments):
    with pytest.raises(AgentToolError):
        capability_arguments(arguments)
