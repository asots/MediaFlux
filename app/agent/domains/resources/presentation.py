"""媒体资源结果和候选卡展示；由领域目录注入唯一工具执行链。"""
from __future__ import annotations

import json
from dataclasses import replace

from app.agent.domains.resources.selection import (
    CANDIDATE_VIEW_KEY,
    RESOURCE_KIND,
    annotate_resource_candidates,
    candidate_result,
    issue_candidate_view,
    resource_candidate_items,
    resource_model_content,
)
from app.agent.kernel.projection import DefaultProjector
from app.agent.kernel.state_types import StateUpdate
from app.agent.model_context_budget import decode_tool_content


class MediaFluxProjector(DefaultProjector):
    async def before_read(self, tool, context, state_store) -> None:
        if tool.metadata.get("source_kind") in {"resource_index", "resource_presentation"}:
            # 新查询在执行前废弃旧卡，空结果或失败也不能恢复旧选择。
            await state_store.commit(context.lease, updates=(StateUpdate(f"metadata.{CANDIDATE_VIEW_KEY}", None),))
            await context.report_progress({"tool": tool.name, "candidate_view": None})

    def project_read(self, tool, value):
        outcome = self.project(value)
        source = tool.metadata.get("source_kind")
        if source == "resource_presentation":
            outcome = replace(outcome, telemetry={**outcome.telemetry,
                "selected_positions": outcome.public_content["data"]["positions"]})
        if source in {"resource_index", "resource_presentation"} and not any(ref.kind == RESOURCE_KIND for ref in outcome.refs):
            outcome = replace(outcome,
                public_content=candidate_result(outcome.public_content, None),
                model_content=resource_model_content(outcome.model_content, maximum=self.max_model_chars),
                full_model_content=resource_model_content(outcome.full_model_content,
                    maximum=max(len(outcome.full_model_content) * 2, self.max_model_chars)) if outcome.full_model_content else "")
        return outcome

    async def present_references(self, outcome, references, *, context, reference_store):
        candidate_view = None
        candidate_items = []
        has_candidates = False
        positions = outcome.telemetry.get("selected_positions")
        for item, reference in references:
            if reference.kind != RESOURCE_KIND:
                continue
            has_candidates = True
            candidate_items = resource_candidate_items(item.value, outcome.public_content)
            candidate_view = await issue_candidate_view(
                store=reference_store, owner=context.owner, session_id=context.session_id,
                generation=context.lease.generation, ref=reference.ref, items=candidate_items,
                ttl_seconds=item.ttl_seconds, turn_id=context.turn_id, selected_positions=positions,
            )
        if not has_candidates:
            return outcome
        content = outcome.full_model_content or outcome.model_content
        payload = decode_tool_content(resource_model_content(content,
            maximum=max(len(content) * 2, self.max_model_chars)))
        numbered = candidate_view["items"] if candidate_view else candidate_items
        if positions is not None:
            numbered = [item for item in numbered if item["position"] in positions]
        if numbered:
            annotate_resource_candidates(payload, numbered, selected_positions=positions)
            if candidate_view and candidate_view["recommended_positions"]:
                payload["recommended_ingest_arguments"] = {
                    "source_type": RESOURCE_KIND, "resource_candidates_ref": candidate_view["ref"],
                    "positions": candidate_view["recommended_positions"], "target": "preferred",
                }
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return replace(outcome, public_content=candidate_result(outcome.public_content, candidate_view),
            model_content=content, full_model_content=content if outcome.full_model_content else "",
            state_updates=(*outcome.state_updates, StateUpdate(f"metadata.{CANDIDATE_VIEW_KEY}", candidate_view)))


class ResourceSelection:
    """资源候选选择、冻结计划和执行回执的领域绑定。"""

    async def admit(self, agent_input, *, state_store, reference_store, catalog):
        from app.agent.domains.resources.selection import validate_selection
        from app.agent.errors import ToolPipelineError
        from app.agent.kernel.capabilities import ToolEffect
        from app.agent.kernel.state_types import SelectionInvalidError

        selection = candidate = tool = None
        if "selection" in agent_input.metadata:
            state = await state_store.load(owner=agent_input.owner, session_id=agent_input.session_id)
            selection = await validate_selection(agent_input.metadata["selection"], state=state, store=reference_store)
            try:
                tool = catalog.get("ingest.submit")
                if tool.effect is ToolEffect.READ:
                    raise KeyError("selection must require confirmation")
            except KeyError as exc:
                raise ToolPipelineError("候选预览能力暂不可用", code="selection_unavailable") from exc
        if agent_input.metadata.get("candidate_context") and selection is None:
            state = await state_store.load(owner=agent_input.owner, session_id=agent_input.session_id)
            items = (state.metadata.get(CANDIDATE_VIEW_KEY) or {}).get("items") or []
            if not items:
                raise SelectionInvalidError()
            candidate = await validate_selection({"ref": agent_input.metadata["candidate_context"],
                "positions": [items[0]["position"]], "target": "guangya"},
                state=state, store=reference_store, for_preview=False)
        return selection, candidate, tool

    def validate_call(self, tool, arguments, context):
        from app.agent.errors import ToolPipelineError
        from app.agent.kernel.capabilities import ToolEffect

        if context.selection_arguments is None:
            return
        actual = {**arguments, "target": arguments.get("target", "preferred")}
        positions = arguments.get("positions")
        if (tool.name != "ingest.submit" or tool.effect is ToolEffect.READ
                or actual != dict(context.selection_arguments) or not isinstance(positions, list)
                or any(type(position) is not int for position in positions)):
            raise ToolPipelineError("只能预览本次已验证的候选选择", code="selection_mismatch")

    async def authorize(self, tool, arguments, resolved, context, reference_store):
        from app.agent.errors import ToolPipelineError

        if (context.resource_candidate_ref and tool.name == "ingest.submit"
                and arguments.get("source_type") == RESOURCE_KIND
                and arguments.get("resource_candidates_ref") != context.resource_candidate_ref):
            bound = await reference_store.resolve(context.resource_candidate_ref,
                owner=context.owner, session_id=context.session_id, expected_kind=RESOURCE_KIND)
            candidate = resolved.get(RESOURCE_KIND)
            if not isinstance(bound, dict) or not isinstance(candidate, dict) or (
                    {key: value for key, value in bound.items() if key != "_query_arguments"}
                    != {key: value for key, value in candidate.items() if key != "_query_arguments"}):
                raise ToolPipelineError("请从回复消息所属的候选批次选择，不可替换为其他搜索编号", code="candidate_scope_mismatch")

    def plan_updates(self, tool, arguments, plan):
        if tool.name == "ingest.submit" and arguments.get("source_type") == RESOURCE_KIND:
            return (StateUpdate("metadata.ux_candidate_plan", {
                "plan_id": plan.plan_id, "ref": arguments.get("resource_candidates_ref"),
            }),)
        return ()

    def confirmed_result(self, result, public_result):
        from app.agent.public_view import format_public_result

        if not result or result.tool.name != "ingest.submit" or result.arguments.get("source_type") != RESOURCE_KIND:
            return None
        data = public_result.get("data")
        items = data.get("items", []) if isinstance(data, dict) else []
        items = items if isinstance(items, list) else []
        return {"ref": result.arguments.get("resource_candidates_ref"),
            "text": format_public_result(public_result), "target": result.arguments.get("target"),
            "handled_positions": [item.get("position") for item in items if isinstance(item, dict)
                and type(item.get("position")) is int and item.get("status") != "failed"]}
