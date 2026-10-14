"""实际Kernel循环的动态能力窗口，不调用外部模型或真实Provider。"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from itertools import combinations
from unittest.mock import patch

import pytest

from app.agent.capability_discovery_actions import (
    CAPABILITY_PARAMETERS,
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
from app.agent.kernel.pipeline import (
    ToolPipeline,
    ToolPipelineError,
    _validate_json_schema,
)
from app.agent.kernel.ports.existing_actions import adapt_tool_spec
from app.agent.kernel.projection import DefaultProjector
from app.agent.kernel.session import AgentSession
from app.agent.kernel.state import AgentInput, InMemorySessionStateStore
from app.agent.models import ToolContext
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
    # This helper describes complete scripted rounds, not interrupted protocol streams.
    model = ScriptedModel([
        [*events, ModelEvent(ModelEventType.FINISH, finish_reason=(
            "tool_calls" if any(e.type == ModelEventType.TOOL_CALL_COMPLETED for e in events) else "stop"
        ))] for events in rounds
    ])
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
            [],  # Same-batch discoveries now form one bounded union.
        )

    async def test_discovered_write_prepares_then_confirm_continues_after_execution(
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
                final(),
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
        self.assertEqual(len(model.requests), count + 1)
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
@pytest.mark.parametrize("arguments", [
    {"tool_names": ["secret.inspect"]}, {"catalog": {}},
    {"catalog": {"domains": ["secret"]}},
])
def test_hidden_or_unauthorized_capabilities_are_not_exposed(extra, arguments):
    tool = replace(read_tool("secret.inspect", description="演员信息"), **extra)
    finder = CapabilityDiscovery(ToolCatalog([tool]), context={"owner": "owner-1"})
    result = finder.search(capability_arguments(arguments))
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
    assert finder.requests == 1  # Identical selection reuses the per-turn cache.
    for index in range(7):
        finder.search({"query": f"unmatched-{index}"})
    with pytest.raises(ToolPipelineError, match="8次"):
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


_DISCOVERY_MODES = [
    {"query": "媒体库"}, {"queries": ["媒体库", "资源搜索"]},
    {"tool_names": ["library.search"]}, {"catalog": {"domains": ["rss"]}},
    {"window": {"retain": ["library.search"]}},
]


@pytest.mark.parametrize("arguments", [{}, {"catalog": {}}, *_DISCOVERY_MODES])
def test_discovery_each_mode_and_overview_keep_public_contract(arguments):
    _validate_json_schema(arguments, CAPABILITY_PARAMETERS)
    assert capability_arguments(arguments).keys() == arguments.keys()


@pytest.mark.parametrize("left,right", list(combinations(_DISCOVERY_MODES, 2)))
def test_discovery_all_mixed_modes_rejected_before_business_execution(left, right):
    arguments = {**left, **right}
    with pytest.raises(ToolPipelineError):
        _validate_json_schema(arguments, CAPABILITY_PARAMETERS)
    with pytest.raises(AgentToolError):
        capability_arguments(arguments)


@pytest.mark.parametrize("scope", [{}, {"domains": ["rss"]}])
@pytest.mark.parametrize("legacy", [{"offset": 0}, {"limit": 12}, {"offset": 0, "limit": 12}])
def test_catalog_legacy_pagination_rejected_by_schema_and_domain_validator(scope, legacy):
    arguments = {"catalog": {**scope, **legacy}}
    with pytest.raises(ToolPipelineError, match="未知参数") as error:
        _validate_json_schema(arguments, CAPABILITY_PARAMETERS)
    assert error.value.code == "invalid_arguments"
    with pytest.raises(AgentToolError, match="catalog.*domains"):
        capability_arguments(arguments)


@pytest.mark.parametrize("arguments", [{}, *[mode for mode in _DISCOVERY_MODES if "catalog" not in mode]])
def test_non_catalog_modes_do_not_report_catalog_totals(arguments):
    tool = read_tool("library.search", description="媒体库")
    finder = CapabilityDiscovery(ToolCatalog([tool]), context={})
    finder.window([tool])
    result = finder.search(capability_arguments(arguments))
    # 空参数沿用既有概况形状；不要求补一个无业务含义的null字段。
    assert result.get("catalog_summary") is None
    assert "next_offset" not in result


@pytest.mark.parametrize("restriction", ["availability", "authorize", "context_requirements"])
def test_overview_uses_same_live_visible_catalog_as_discovery(restriction):
    spec = next(s for s in build_tool_specs() if s.name == "agent.capabilities")
    allowed = [False]
    options = ({restriction: lambda _: allowed[0]} if restriction != "context_requirements"
               else {restriction: frozenset({"resource_candidates"})})
    hidden = replace(read_tool("library.private"), **options)
    finder = CapabilityDiscovery(ToolCatalog([adapt_tool_spec(spec), hidden]), context={})
    context = ToolContext(capability_search=finder.search)
    result = spec.context_handler({}, context)
    assert result.ok and result.status == "success"
    assert result.data["total_tools"] == result.data["discovery"]["visible_count"] == 1
    assert sum(g["tool_count"] for g in result.data["groups"]) == 1
    assert finder.search({"tool_names": [hidden.name]})["loaded_tools"] == []
    assert "private" not in json.dumps(result.to_dict())
    allowed[0] = True
    finder.context["reference_kinds"] = ["resource_candidates"]
    refreshed = spec.context_handler({}, context)
    assert refreshed.data["total_tools"] == refreshed.data["discovery"]["visible_count"] == 2
    assert sum(g["tool_count"] for g in refreshed.data["groups"]) == 2
    assert finder.search({"tool_names": [hidden.name]})["loaded_tools"] == [hidden.name]


def test_overview_without_session_catalog_does_not_fall_back_to_global_registry():
    spec = next(s for s in build_tool_specs() if s.name == "agent.capabilities")
    result = spec.context_handler({}, ToolContext())
    assert not result.ok and result.status == "unavailable"
    assert "total_tools" not in result.data


@pytest.mark.parametrize("name,group_title", [
    ("guangya.fs.query", "光鸭云盘"), ("library.batch_presence", "媒体与媒体库"),
])
def test_overview_groups_by_runtime_domain_instead_of_tool_name_prefix(name, group_title):
    specs = {spec.name: spec for spec in build_tool_specs()}
    finder = CapabilityDiscovery(ToolCatalog([adapt_tool_spec(specs["agent.capabilities"]), adapt_tool_spec(specs[name])]), context={})
    result = specs["agent.capabilities"].context_handler({}, ToolContext(capability_search=finder.search))
    groups = {group["title"]: group["tool_count"] for group in result.data["groups"]}
    assert groups[group_title] == 1
    assert "其他项目能力" not in groups


def test_discovery_matches_are_not_displaced_by_unrelated_workflow_neighbors():
    matches = [read_tool(f"library.match{i}", description="stellar inventory") for i in range(6)]
    neighbors = [read_tool(f"unrelated.neighbor{i}", description="other business") for i in range(2)]
    matches[0] = replace(matches[0], metadata={"related_tools": tuple(t.name for t in neighbors)})
    finder = CapabilityDiscovery(ToolCatalog([*matches, *neighbors]), context={})
    result = finder.search({"query": "stellar inventory"})
    assert {t["name"] for t in result["tools"]} == {t.name for t in matches}
    assert len(result["loaded_tools"]) == 6
    assert finder.search({"query": "unfindablephrase"})["tools"] == []


def test_actual_filesystem_capability_survives_discovery_ranking():
    finder = CapabilityDiscovery(ToolCatalog(adapt_tool_spec(s) for s in build_tool_specs()), context={})
    result = finder.search({"query": "光鸭云盘文件移动、复制、改名"})
    assert "guangya.fs.change.execute" in {tool["name"] for tool in result["tools"]}
    assert len(result["tools"]) <= 6
    assert result["remaining_discoveries"] == 7


@pytest.mark.parametrize("domain", [" RSS ", " Rſſ "])
def test_catalog_domain_casefold_keeps_filter_scope(domain):
    tools = [read_tool("rss.item", domain="rss"), read_tool("library.item", domain="library")]
    finder = CapabilityDiscovery(ToolCatalog(tools), context={})
    raw = {"catalog": {"domains": [domain, "rss"]}}
    _validate_json_schema(raw, CAPABILITY_PARAMETERS)
    arguments = capability_arguments(raw)
    assert arguments == {"catalog": {"domains": ["rss"]}}
    result = finder.search(arguments)
    assert [t["name"] for t in result["tools"]] == ["rss.item"]
    assert result["catalog_only"] is True
    assert result["loaded_tools"] == []
    assert result["catalog_summary"]["total"] == 1
    assert result["catalog_summary"]["by_effect"]["read"] == 1
    assert "next_offset" not in result
    assert not finder.consume()


def test_ranked_discovery_is_explicitly_a_subset_not_domain_total():
    spec = next(s for s in build_tool_specs() if s.name == "agent.capabilities")
    finder = CapabilityDiscovery(ToolCatalog([adapt_tool_spec(spec), *[read_tool(f"rss.item{i}", description="feed inspection", domain="rss") for i in range(14)]]), context={})
    context = ToolContext(capability_search=finder.search)
    result = spec.context_handler({"query": "feed inspection"}, context)
    assert result.ok and len(result.data["tools"]) == 6
    assert "不是完整目录" in result.summary
    assert result.data["catalog_summary"] is None
    assert any("catalog" in hint and "完整目录" in hint for hint in result.suggestions)
    assert all("next_offset" not in hint for hint in result.suggestions)
    full = spec.context_handler(capability_arguments({"catalog": {"domains": ["RSS"]}}), context)
    assert len(full.data["tools"]) == full.data["catalog_summary"]["total"] == 14
    assert full.data["catalog_summary"]["by_effect"]["read"] == 14
    assert full.data["loaded_tools"] == result.data["loaded_tools"]
    assert "next_offset" not in full.data
    assert {t.name for t in finder.consume()} == set(result.data["loaded_tools"])


def test_cloud_inventory_has_one_authoritative_session_catalog():
    specs = build_tool_specs()
    assert 'guangya.capabilities' not in {s.name for s in specs}
    catalog = ToolCatalog(adapt_tool_spec(s) for s in specs)
    visible = [t for t in catalog.visible() if t.name.startswith('guangya.')]
    domains = sorted({t.domain for t in visible})
    finder = CapabilityDiscovery(catalog, context={})
    result = finder.search({'catalog': {'domains': domains}})
    returned = {t['name']: t for t in result['tools']}
    assert {t.name for t in visible} <= returned.keys()
    assert all(returned[t.name]['effect'] == t.effect.value for t in visible)
    assert result['catalog_only'] and not result['loaded_tools']
    assert result['catalog_summary']['total'] == len(result['tools'])
    assert not any('upload' in name for name in returned)

    # 当前授权/可见性裁掉的工具不能从另一份静态清单重新出现。
    restricted = ToolCatalog([t for t in catalog.visible() if t.name != 'guangya.recycle.clear'])
    restricted_result = CapabilityDiscovery(restricted, context={}).search({'catalog': {'domains': domains}})
    assert 'guangya.recycle.clear' not in {t['name'] for t in restricted_result['tools']}
