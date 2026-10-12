"""Turn-local evidence for correcting rejected calls, never execution authority."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import secrets
from typing import Any

REPAIR_TOOL = "agent.repair"


@dataclass(frozen=True)
class SchemaReplacement:
    scope: tuple
    tool: str
    locked: dict[tuple, Any]
    predecessors: tuple[str, ...] = ()
    equivalent_arguments: str = ""


def scope(context):
    return (context.owner, context.session_id, context.lease.generation, context.turn_id)


def effective_schema(schema, argument_schemas=None):
    """Share the registered domain contract between repair proofs and model feedback."""
    if not argument_schemas:
        return schema
    properties = schema.get("properties", {})
    if not argument_schemas.keys() <= properties.keys():
        return None
    return {**schema, "properties": {**properties, **argument_schemas}}


def issue(tool, arguments, context, validate=None, error_type=Exception, *, collisions=frozenset(), invalid_fields=(), argument_schemas=None):
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
    if not repairable or not supported:
        return ""  # No speculative ticket when the invalid field is unknown.
    exact = ""
    if equivalent != arguments:
        try:
            validate(equivalent, schema)
        except error_type:
            pass
        else:
            # Only deleting undeclared fields gives a unique equivalent query.
            # Missing values, changed targets and domain guesses still need explicit repair.
            exact = json.dumps(equivalent, sort_keys=True, ensure_ascii=False)
    ticket = secrets.token_urlsafe(24)
    context.schema_replacements[ticket] = SchemaReplacement(scope(context), tool.name, locked, equivalent_arguments=exact)
    return ticket


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


def completed(tool, arguments, context, content):
    """Associate a successful exact correction, never another object's success."""
    if not context.schema_replacements or content.get("ok") is False or content.get("status", "success") not in {"success", "completed"}:
        return ()
    encoded = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    recovered = []
    for ticket, entry in list(context.schema_replacements.items()):
        if entry.scope == scope(context) and entry.tool == tool.name and entry.equivalent_arguments == encoded:
            recovered.extend((*entry.predecessors, ticket))
            context.schema_replacements.pop(ticket)
    return tuple(dict.fromkeys(recovered))
