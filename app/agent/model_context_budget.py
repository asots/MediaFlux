"""Provider 请求的模型消息预算与安全压缩。"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, NoReturn

RESULT_PATH_MAX = 1_000


def estimated_tokens(value: object) -> int:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            text = str(value)
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    return max(1, (ascii_chars + 3) // 4 + len(text) - ascii_chars)


def conservative_tokens(value: object) -> int:
    """UTF-8 byte proxy, not an exact tokenizer or final wire-size guarantee."""
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), default=str,
    )
    return max(1, len(text.encode("utf-8")))


def decode_tool_content(content: str) -> dict[str, Any]:
    """读取规范 JSON；只在恢复旧历史时迁移旧的多行引用附录。"""
    text = str(content or "")
    body, separator, suffix = text.partition("\nopaque_refs=")
    if text.startswith("opaque_refs="):
        body, separator, suffix = "{}", "legacy", text.removeprefix("opaque_refs=")
    try:
        value = json.loads(body)
    except (TypeError, ValueError):
        value = {"summary": body}
    payload = value if isinstance(value, dict) else {"data": value}
    if separator:
        for line in ("opaque_refs=" + suffix).splitlines():
            key, _, raw = line.partition("=")
            if key in {"opaque_refs", "reference_arguments", "candidate_numbers", "recommended_ingest_arguments"}:
                try:
                    payload[key] = json.loads(raw)
                except ValueError:
                    continue
    return payload


def _invalid_result() -> NoReturn:
    from app.agent.errors import ToolPipelineError

    raise ToolPipelineError("结果续取元数据无效，不能将不完整证据当作完整结果；请重新观察。", code="invalid_tool_result")


def _validate_result_metadata(payload: dict[str, Any], *, result_page: bool) -> None:
    """Validate only trusted envelopes, never infer protocol identity from JSON shape."""
    required = {"path", "kind", "total", "offset", "returned", "has_more", "next_offset"}
    scalars = ("int", "float", "bool", "NoneType")
    kinds = {"dict": "object", "list": "array", "str": "string"}

    def pointer(value: Any) -> bool:
        return (isinstance(value, str) and len(value) <= RESULT_PATH_MAX
                and (not value or value.startswith("/")) and not re.search(r"~(?![01])", value))

    def counts(row: dict[str, Any]) -> bool:
        return (all(type(row.get(key)) is int and row[key] >= 0 for key in ("total", "offset", "returned"))
                and row["returned"] <= max(0, row["total"] - row["offset"]))

    rows = payload.get("truncation", [])
    if (not isinstance(rows, list) or ("truncated" in payload and type(payload["truncated"]) is not bool)
            or bool(rows) != bool(payload.get("truncated"))):
        _invalid_result()
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or not required <= row.keys() or row.keys() - required - {"scope"}
                or not pointer(row.get("path")) or not counts(row)):
            _invalid_result()
        if row["path"] in seen or row["kind"] not in (*kinds, *kinds.values(), "scalar", *scalars):
            _invalid_result()
        seen.add(row["path"])
        subtree = row.get("scope") == "subtree"
        if ("scope" in row and not subtree or row["has_more"] is not True
                or type(row["next_offset"]) is not int or row["offset"] > row["total"]
                or row["next_offset"] != row["offset"] + (0 if subtree else row["returned"])
                or not subtree and row["next_offset"] >= row["total"]
                or row["kind"] in ("scalar", *scalars) and row["total"] != 1):
            _invalid_result()
    if not result_page or payload.get("ok") is False:
        return
    page = payload.get("data")
    handle = payload.get("result_handle")
    if (not isinstance(handle, str) or not 1 <= len(handle) <= 200 or payload.get("read_tool") != "agent.read_result"
            or not isinstance(page, dict) or page.keys() != required | {"items"}
            or not pointer(page.get("path")) or not counts(page)):
        _invalid_result()
    kind, items = page["kind"], page["items"]
    if kind not in (*kinds, *scalars) or not isinstance(items, str if kind == "str" else list):
        _invalid_result()
    if kind == "dict":
        keys = []
        for item in items:
            if not isinstance(item, dict) or item.keys() != {"key", "value"} or not isinstance(item["key"], str):
                _invalid_result()
            keys.append(item["key"])
        if len(set(keys)) != len(keys):
            _invalid_result()
    end = page["offset"] + len(items)
    if (len(items) != page["returned"] or page["has_more"] is not (end < page["total"])
            or page["next_offset"] != (end if end < page["total"] else None)
            or end < page["total"] and (not items or type(page["next_offset"]) is not int)
            or kind in scalars and (page["total"] != 1 or any(type(item).__name__ != kind for item in items))):
        _invalid_result()
    for row in rows:
        if page["path"] and row["path"] != page["path"] and not row["path"].startswith(page["path"] + "/"):
            _invalid_result()
        if row["path"] == page["path"] and (
            any(row[key] != page[key] for key in ("total", "offset", "returned"))
            or row["kind"] not in (kind, kinds.get(kind, "scalar"))
        ):
            _invalid_result()


def compact_tool_content(
    content: str, *, maximum: int, result_page: bool = False, trusted: bool = False, tool_name: str = "",
) -> str:
    """Budget a source or a canonical projection, retaining original snapshot coordinates.

    Raw tool DTOs/arguments use the default. Only executor-produced projections and
    persisted tool messages use trusted=True; result_page comes from the executor's
    tool_name, not any field inside content. No model-visible magic marker grants trust.
    """
    payload = decode_tool_content(content)
    # The built-in reader has one catalog-published alias. Match exact identities,
    # not a reversible spelling heuristic (nor payload.read_tool/page-like fields).
    result_page = result_page or tool_name in ("agent.read_result", "agent__read_result")
    protocol_keys = {"result_handle", "read_tool", "truncated", "truncation"}
    if trusted:
        _validate_result_metadata(payload, result_page=result_page)
    elif result_page:
        _invalid_result()
    collisions = {} if trusted else {key: payload[key] for key in protocol_keys & payload.keys()}
    if len(content) <= maximum and not collisions:
        return content
    protected = {"ok", "status", "code", "result_handle", "read_tool"}
    prior = {row["path"]: dict(row) for row in payload.get("truncation", [])} if trusted else {}
    omitted = dict(prior)
    # Reader identity comes from the executor/message tool_name, never business fields.
    page = payload.get("data") if result_page and payload.get("ok") is not False else None
    root = page["path"] if page is not None else ""

    def dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def source_path(path: str) -> str:
        if page is None:
            return path
        tail = path.removeprefix("/data/items")
        if not tail or page["kind"] == "str":
            return root
        index, separator, rest = tail[1:].partition("/")
        if page["kind"] == "dict":
            key = page["items"][int(index)]["key"]
            return root + "/" + key.replace("~", "~0").replace("/", "~1") + rest.removeprefix("value")
        prefix = "/" + str(page["offset"] + int(index)) if page["kind"] == "list" else ""
        return root + prefix + ("/" + rest if separator else "")

    def branches(node: Any, path: str = ""):
        if not isinstance(node, (dict, list)):
            return
        for key, value in (node.items() if isinstance(node, dict) else enumerate(node)):
            if not path and key in protected | {"truncated", "truncation"}:
                continue
            wrapper = page is not None and page["kind"] == "dict" and path.startswith("/data/items/") and path.count("/") == 3
            if wrapper and key == "key":
                continue
            child = path + "/" + str(key).replace("~", "~0").replace("/", "~1")
            # key/value is a display wrapper, not a source object.
            if not (page is not None and page["kind"] == "dict" and path == "/data/items"):
                yield node, key, child, value
            yield from branches(value, child)

    def fields():
        if page is not None:
            yield page, "items", "/data/items", page["items"]
            yield from branches(page["items"], "/data/items")
        else:
            yield from branches(payload)

    def measure(value: Any) -> tuple[str, int]:
        return ("object", len(value)) if isinstance(value, dict) else ("array", len(value)) if isinstance(value, list) else ("string", len(value)) if isinstance(value, str) else ("scalar", 1)

    originals = {source_path(path): measure(value) for _, _, path, value in fields()}
    root_members = payload.keys() - (protocol_keys if trusted else set())
    if trusted and page is None:
        # Before root coverage exists, displaced members remain in child obligations.
        # Union direct keys, not child totals: descendants/visible parents count once.
        # An existing root record below remains authoritative after subtree merging.
        root_members.update(path.split("/", 2)[1].replace("~1", "/").replace("~0", "~")
                            for path in prior if path)
    originals[root] = (page["kind"], page["total"]) if page is not None else ("object", len(root_members))
    originals.update(("/" + key, measure(value)) for key, value in collisions.items())
    for path, row in omitted.items():
        originals[path] = (row["kind"], row["total"])

    def covers(ancestor: str, child: str) -> bool:
        return ancestor == child or not ancestor or child.startswith(ancestor + "/")

    def record(path: str, returned: int, *, subtree: bool = False) -> dict[str, Any]:
        kind, total = originals[path]
        offset = prior.get(path, {}).get("offset", page["offset"] if page is not None and path == root else 0)
        row = {"path": path, "kind": kind, "total": total, "offset": offset,
               "returned": returned, "has_more": True, "next_offset": offset if subtree else offset + returned}
        if subtree:
            row["scope"] = "subtree"
        return row

    # Business protocol-name collisions stay at their original snapshot pointers.
    # Omit them explicitly from the view; do not overwrite the frozen source or wrap data.
    for key in collisions:
        payload.pop(key, None)
        omitted["/" + key] = record("/" + key, 0, subtree=True)

    def fail_budget() -> NoReturn:
        from app.agent.errors import ToolPipelineError

        raise ToolPipelineError("结果的原子键或续取元数据超过上下文预算，无法安全分页；请缩小原查询。", code="context_budget_exceeded")

    def encode() -> str:
        current = {source_path(path): measure(value)[1] for _, _, path, value in fields()}
        current[root] = len(page["items"]) if page is not None else len(payload.keys() - {"result_handle", "read_tool", "truncated", "truncation"})
        for path in list(omitted):
            ancestor = path
            while len(ancestor) > RESULT_PATH_MAX:
                ancestor = ancestor.rpartition("/")[0]
            if ancestor != path:
                del omitted[path]
                omitted[ancestor] = record(ancestor, current.get(ancestor, 0), subtree=True)
        # An ancestor plus a descendant is one subtree obligation, never summed counts.
        for path in sorted(list(omitted), key=len):
            if path not in omitted:
                continue
            descendants = [child for child in omitted if child != path and covers(path, child)]
            if descendants or omitted[path].get("scope") == "subtree":
                omitted[path] = record(path, current.get(path, 0), subtree=True)
                for child in descendants:
                    del omitted[child]
        if len(omitted) > 4 or len(dumps(list(omitted.values()))) > max(180, maximum // 3):
            paths = list(omitted)
            ancestor = paths[0]
            while not all(covers(ancestor, path) for path in paths):
                ancestor = ancestor.rpartition("/")[0]
            while ancestor not in originals:
                ancestor = ancestor.rpartition("/")[0]
            omitted.clear()
            omitted[ancestor] = record(ancestor, current.get(ancestor, 0), subtree=True)
        if omitted:
            # Compact JSON alone can fit: truncation denotes loss, not re-encoding.
            payload["truncated"] = True
            payload["truncation"] = list(omitted.values())
        if page is not None:
            count = len(page["items"])
            end = page["offset"] + count
            page.update(returned=count, has_more=end < page["total"], next_offset=end if end < page["total"] else None)
        return dumps(payload)

    encoded = encode()
    while len(encoded) > maximum:
        choices = []
        for parent, key, path, value in fields():
            if isinstance(value, (dict, list, str)) and len(value) > (48 if isinstance(value, str) else 1):
                priority = 1 if isinstance(value, str) else -1 if isinstance(value, dict) else 0
                choices.append((priority, len(dumps(value)), parent, key, path, value))
        minimum_string = 48
        if not choices and page is not None:
            minimum_string = 1
            choices = [(1, len(dumps(value)), parent, key, path, value)
                       for parent, key, path, value in fields() if isinstance(value, str) and len(value) > 1]
        if not choices:
            # Atomic page keys/coordinates cannot be shortened without changing identity.
            if page is not None:
                break
            omitted.clear()
            payload = {key: value for key, value in payload.items() if key in protected}
            omitted[root] = record(root, 0, subtree=True)
            encoded = encode()
            break
        _, _, parent, key, path, value = max(choices, key=lambda item: item[:2])
        source = source_path(path)
        if page is not None and len(source) > RESULT_PATH_MAX:
            fail_budget()
        retained = max(minimum_string if isinstance(value, str) else 1, len(value) // 2)
        parent[key] = dict(list(value.items())[:retained]) if isinstance(value, dict) else value[:retained]
        previous = omitted.get(source, {})
        omitted[source] = record(source, retained, subtree=previous.get("scope") == "subtree")
        encoded = encode()
    if len(encoded) > maximum:
        fail_budget()
    _validate_result_metadata(payload, result_page=result_page)
    return encoded


def _compact_chain(messages: Sequence[Any], *, max_tool_chars: int) -> list[Any]:
    result = []
    for message in messages:
        if message.role == "tool":
            content = compact_tool_content(
                message.content, maximum=max_tool_chars, trusted=True,
                tool_name=message.tool_name,
            )
            if content != message.content:
                message = replace(message, content=content)
        result.append(message)
    return result


def _compact_legacy_history(messages: Sequence[Any]) -> list[Any]:
    return [
        replace(message, content=compact_tool_content(message.content, maximum=800, trusted=True))
        if message.role == "tool" and message.tool_name == "agent.capabilities" and len(message.content) > 4_000
        else message for message in messages
    ]


def _native_history(messages: Sequence[Any], *, reserved_ids: set[str] | None = None) -> list[Any]:
    """保留原生工具问答角色；历史调用只是证据，不是重新执行的授权。"""
    result = []
    pending: dict[str, str] = {}
    reserved = set(reserved_ids or ()) | {
        call.call_id for message in messages if message.provider_state is not None
        for call in message.tool_calls
    }
    for index, message in enumerate(messages):
        if message.role == "assistant" and message.tool_calls:
            calls = []
            for position, call in enumerate(message.tool_calls):
                # 旧历史去重ID；有原生私有状态的调用保留供应商关联ID，不能改写签名链。
                identifier = call.call_id
                if message.provider_state is None:
                    identifier = f"history_{index}_{position}"
                    while identifier in reserved:
                        identifier += "_"
                    reserved.add(identifier)
                pending[call.call_id] = identifier
                calls.append(replace(call, call_id=identifier))
            result.append(replace(message, tool_calls=tuple(calls)))
        elif message.role == "tool":
            identifier = pending.pop(message.tool_call_id, "")
            if identifier:
                result.append(replace(message, tool_call_id=identifier))
            else:
                # 仅迁移旧80条截断留下的孤立观察，不能伪造已执行的调用参数。
                result.append(replace(message, role="user", tool_call_id="", tool_name="",
                                      content="历史观察（原调用已缺失，不是当前查询或执行授权）：\n" + message.content))
        else:
            result.append(message)
    return result


def protocol_history_groups(messages: Sequence[Any], *, history_end: int):
    """Select whole user groups by native identity only, before any budget loss."""
    split_at = max(0, min(int(history_end), len(messages)))
    history = list(messages[:split_at])
    current = list(messages[split_at:])
    current_counts = Counter(call.call_id for message in current for call in message.tool_calls)
    current_native_ids = {call.call_id for message in current if message.provider_state is not None
                          for call in message.tool_calls}
    if any(current_counts[identifier] > 1 for identifier in current_native_ids):
        from app.agent.errors import ToolPipelineError

        raise ToolPipelineError(
            "模型复用了原生状态关联标识，无法安全续传；已保留真实工具结果，请重新发起查询",
            code="provider_state_conflict",
        )
    groups: list[list[Any]] = []
    for message in history:
        if message.role == "user" or not groups:
            groups.append([])
        groups[-1].append(message)
    # Native state can bind provider call IDs. Do not rewrite those IDs to
    # deduplicate history; omit an entire older user turn on a collision.
    reserved_ids = {call.call_id for message in current for call in message.tool_calls}
    unique_groups: list[list[Any]] = []
    for group in reversed(groups):
        native_ids = {call.call_id for message in group if message.provider_state is not None
                      for call in message.tool_calls}
        counts = Counter(call.call_id for message in group for call in message.tool_calls)
        if native_ids & reserved_ids or any(counts[identifier] > 1 for identifier in native_ids):
            continue
        reserved_ids.update(native_ids)
        unique_groups.append(group)
    return list(reversed(unique_groups)), current, reserved_ids


def protocol_history_baseline(messages: Sequence[Any], *, history_end: int) -> list[Any]:
    """Uncompacted completeness obligations; budget eviction is never exempted."""
    groups, current, _ = protocol_history_groups(messages, history_end=history_end)
    return [message for group in groups for message in group] + current


def bounded_model_messages(
    messages: Sequence[Any],
    *,
    history_end: int,
    tool_definitions: Sequence[Mapping[str, Any]],
    system_prompt: str,
    context_window_tokens: int,
    output_tokens: int,
    conservative: bool = False,
) -> tuple[Any, ...]:
    """保留最新完整回合，裁剪旧历史，并硬性阻止超窗口 Provider 请求。"""
    groups, current, reserved_ids = protocol_history_groups(messages, history_end=history_end)
    groups = [_compact_legacy_history(group) for group in groups]
    history = [message for group in groups for message in group]
    estimate = conservative_tokens if conservative else estimated_tokens
    message_budget = max(
        0,
        int(context_window_tokens)
        - estimate(system_prompt)
        - estimate(tool_definitions)
        - int(output_tokens)
        - 512,
    )

    def cost(items: Sequence[Any]) -> int:
        return sum(8 + estimate(item.to_dict()) for item in items)

    current_cost = cost(current)
    for maximum in (1_200, 400):
        if current_cost <= message_budget:
            break
        current = _compact_chain(current, max_tool_chars=maximum)
        current_cost = cost(current)
    if current_cost > message_budget:
        from app.agent.errors import ToolPipelineError

        raise ToolPipelineError(
            "当前工具链超过模型上下文上限，请缩小查询范围后重试",
            code="context_budget_exceeded",
        )
    remaining = max(0, message_budget - current_cost)
    history_view = _native_history(history, reserved_ids=reserved_ids)
    if cost(history_view) <= remaining:
        return tuple(history_view + current)

    kept: list[list[Any]] = []
    for group in reversed(groups):
        candidate, candidate_cost = group, cost(_native_history(group, reserved_ids=reserved_ids))
        for maximum in (1_200, 400):
            if candidate_cost <= remaining or kept:
                break
            from app.agent.errors import ToolPipelineError

            try:
                candidate = _compact_chain(group, max_tool_chars=maximum)
            except ToolPipelineError as exc:
                if exc.code != "context_budget_exceeded":
                    raise
                break  # 不可压缩的旧整组留在存储中，不阻塞当前请求。
            candidate_cost = cost(_native_history(candidate, reserved_ids=reserved_ids))
        if candidate_cost > remaining:
            # 历史整组超出窗口时停止装载，不让旧调用锁死新问题。
            # Session 的历史索引保留遗漏标记；历史本身不授予写操作权限。
            break
        kept.append(candidate)
        remaining -= candidate_cost
    return tuple(
        _native_history([message for group in reversed(kept) for message in group], reserved_ids=reserved_ids) + current
    )
