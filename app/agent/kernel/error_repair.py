"""Turn-local evidence for correcting rejected calls, never execution authority."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import secrets
from typing import Any

from .capabilities import ToolEffect

REPAIR_TOOL = "agent.repair"


@dataclass(frozen=True)
class SchemaReplacement:
    scope: tuple
    tool: str
    locked: dict[tuple, Any]
    predecessors: tuple[str, ...] = ()
    equivalent_arguments: str = ""
    required_keys: frozenset[str] = frozenset()
    collision_keys: frozenset[str] = frozenset()
    guard_key: tuple[str, str] | None = None
    offered: bool = True


def scope(context):
    return (context.owner, context.session_id, context.lease.generation, context.turn_id)


def _guard_key(tool, arguments):
    # A nonempty alternative READ mode is independent; omitting a filter or
    # falling back to an empty/default request is not a mode switch.
    properties = tool.input_schema.get("properties", {})
    keys = arguments.keys() & properties.keys()
    mode = (next(iter(keys)) if tool.effect is ToolEffect.READ and tool.input_schema.get("maxProperties") == 1
            and len(properties) > 1 and len(keys) == 1 else "")
    return json.dumps([arguments.get(key) for key in tool.input_schema.get("required", ())], sort_keys=True, default=str), mode


def pending(tool, arguments, context):
    if not context.schema_replacements:
        return {}
    key = _guard_key(tool, arguments)
    return {ticket: entry for ticket, entry in context.schema_replacements.items()
            if entry.scope == scope(context) and entry.tool == tool.name and entry.guard_key is not None
            and entry.guard_key[0] == key[0]
            and not (entry.guard_key[1] and key[1] and entry.guard_key[1] != key[1])}


def require_presence(entries, arguments, error_type):
    omitted = set()
    for entry in entries:
        omitted.update(entry.required_keys - arguments.keys())
        if entry.collision_keys and not any(arguments.get(key) is not None for key in entry.collision_keys):
            omitted.update(entry.collision_keys)
    if omitted:
        raise error_type("参数修复不能省略此前已提供的字段：" + "、".join(sorted(omitted))
                         + "；请修正值，不要删除字段退回默认值。", code="invalid_arguments")


def validated(entries, context):
    # Admission is not completion: keep offered proof/tickets until a real
    # successful result or an explicit replacement settles the old failure.
    for ticket, entry in entries.items():
        if entry.offered:
            context.schema_replacements[ticket] = replace(entry, guard_key=None)
        else:
            context.schema_replacements.pop(ticket)


def effective_schema(schema, argument_schemas=None):
    """Share the registered domain contract between repair proofs and model feedback."""
    if not argument_schemas:
        return schema
    properties = schema.get("properties", {})
    if not argument_schemas.keys() <= properties.keys():
        return None
    return {**schema, "properties": {**properties, **argument_schemas}}


def issue(tool, arguments, context, validate=None, error_type=Exception, *, collisions=frozenset(), invalid_fields=(), argument_schemas=None, preserve_fields=False):
    locked = {}
    schema = effective_schema(tool.input_schema, argument_schemas)
    if schema is None:
        return ""
    equivalent = deepcopy(arguments)
    repairable = arguments is None  # Undecodable call: no inferred old object identity.
    supported = True
    editable = set(invalid_fields) | (set(collisions) - set(tool.input_schema.get("required", ())))
    if not editable <= tool.input_schema.get("properties", {}).keys():
        return ""

    def visit(value, schema, path):
        nonlocal repairable, supported
        if len(path) == 1 and path[0] in editable:
            repairable = True
            return
        try:
            validate(value, schema)
        except error_type:
            if isinstance(value, dict):
                properties = dict(schema.get("properties", {}))
                branches = [*schema.get("oneOf", ()), *schema.get("anyOf", ())]
                for branch in branches:
                    for key, definition in branch.get("properties", {}).items():
                        properties[key] = ({"anyOf": [properties[key], definition]}
                                           if key in properties else definition)
                # Keep values valid under any declared alternative fixed; a
                # replacement must still satisfy the full original composition.
                repairable |= any(set(part.get("required", ())) - value.keys()
                                  for part in [schema, *branches])
                for key, item in value.items():
                    if key not in properties and schema.get("additionalProperties") is False:
                        repairable = True
                        parent = equivalent
                        for part in path:
                            parent = parent[part]
                        del parent[key]
                    else:
                        visit(item, properties.get(key, {}), (*path, key))
            elif isinstance(value, (list, tuple)):
                # Removing valid elements is a different selection, not a value repair.
                if (len(value) > schema.get("maxItems", len(value)) or schema.get("uniqueItems")
                        and len({json.dumps(v, sort_keys=True) for v in value}) != len(value)):
                    supported = False
                repairable |= len(value) < schema.get("minItems", 0)
                for index, item in enumerate(value):
                    visit(item, schema.get("items", {}), (*path, index))
            else:
                repairable = True
        else:
            # Domain validation can identify an otherwise schema-valid root field.
            if not path and editable:
                repairable = True
                for key, item in value.items():
                    visit(item, schema.get("properties", {}).get(key, {}), (key,))
            else:
                locked[path] = deepcopy(value)

    if arguments is not None:
        visit(arguments, schema, ())
    offered = repairable and supported
    if not offered and not preserve_fields:
        return ""  # Unknown domain failures cannot invent repair authority.
    required_keys = frozenset(key for key, value in (arguments or {}).items()
                              if value is not None and key in schema.get("properties", {})
                              and (key not in collisions or key in schema.get("required", ())))
    exact = ""
    if offered and equivalent != arguments:
        try:
            validate(equivalent, schema)
        except error_type:
            pass
        else:
            # Only deleting undeclared fields gives a unique equivalent query.
            # Missing values, changed targets and domain guesses still need explicit repair.
            exact = json.dumps(equivalent, sort_keys=True, ensure_ascii=False)
    ticket = secrets.token_urlsafe(24)
    context.schema_replacements[ticket] = SchemaReplacement(
        scope(context), tool.name, locked, equivalent_arguments=exact, required_keys=required_keys,
        collision_keys=frozenset(collisions), guard_key=_guard_key(tool, arguments) if preserve_fields else None, offered=offered,
    )
    return ticket if offered else ""


def preserves(entry, arguments):
    for path, expected in entry.locked.items():
        value = arguments
        try:
            for key in path:
                value = value[key]
        except (KeyError, IndexError, TypeError):
            return False
        if json.dumps(value, sort_keys=True, ensure_ascii=False) != json.dumps(expected, sort_keys=True, ensure_ascii=False):
            return False  # In particular, True is not the number 1.
    return True


def completed_result(tool, content):
    """查询的found/empty是已返回事实，不是后台任务的受理状态。"""
    statuses = {"success", "completed", "found", "empty"} if tool.effect is ToolEffect.READ else {"success", "completed"}
    return content.get("ok") is not False and content.get("status", "success") in statuses


def completed(tool, arguments, context, content):
    """Associate a successful exact correction, never another object's success."""
    if not context.schema_replacements or not completed_result(tool, content):
        return ()
    encoded = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    recovered = []
    for ticket, entry in list(context.schema_replacements.items()):
        if entry.scope == scope(context) and entry.tool == tool.name and entry.equivalent_arguments == encoded:
            recovered.extend((*entry.predecessors, ticket))
            context.schema_replacements.pop(ticket)
    return tuple(dict.fromkeys(recovered))
