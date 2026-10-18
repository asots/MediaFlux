"""Offline model-input adapter: explicitly choose a server repair entry.

Only nominated call IDs are migrated. No production method is replaced.
"""
import json
from dataclasses import replace


class ExplicitRepairModel:
    def __init__(self, model, call_ids):
        self.model = model
        self.call_ids = call_ids

    def __getattr__(self, name):
        return getattr(self.model, name)

    async def stream(self, request, **kwargs):
        entries = []
        for message in request.messages:
            if message.role == "tool":
                try:
                    payload = json.loads(message.content)
                except (ValueError, TypeError):
                    continue
                if isinstance(payload, dict) and "repair" in payload:
                    entries.append(payload["repair"])
        async for event in self.model.stream(request, **kwargs):
            call = event.tool_call
            if call is not None and call.call_id in self.call_ids and entries:
                event = replace(event, tool_call=replace(call, name="agent.repair", arguments={
                    "ticket": entries[-1]["ticket"], "arguments": dict(call.arguments)}))
            yield event


def repair_entries(value: object) -> list[dict[str, object]]:
    """Extract real tool feedback across the three native request encodings."""
    if isinstance(value, str):
        try:
            return repair_entries(json.loads(value))
        except (ValueError, TypeError):
            return []
    if isinstance(value, dict):
        if isinstance(value.get("repair"), dict):
            return [value["repair"]]
        return [entry for child in value.values() for entry in repair_entries(child)]
    if isinstance(value, list):
        return [entry for child in value for entry in repair_entries(child)]
    return []


def explicit_repair_wire(protocol, replies, *, index, arguments, reply_factory=None):
    from tests.test_agent_tool_argument_ingress import Wire
    from tests.test_agent_recovered_progress import reply
    class RepairWire(Wire):
        async def handler(self, request):
            if len(self.requests) == index:
                entries = repair_entries(json.loads(request.content))
                assert entries, "real tool feedback must offer an explicit repair"
                parameters = {"ticket": entries[-1]["ticket"], "arguments": arguments}
                self.replies[0] = (reply_factory(parameters) if reply_factory else
                    reply(protocol, name="agent__repair", arguments=parameters, index=index))
            return await super().handler(request)
    return RepairWire(protocol, replies)
