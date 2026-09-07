"""定时策略确认的唯一快照与指纹算法，领域模块只提供字段和公开投影。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from app import config


def capture_policy_state(
    arguments: dict[str, Any],
    *,
    targets: Mapping[str, tuple[str, Any]],
    current_policy: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    snapshot, values = config.read_env_snapshot(config.ENV_FILE)
    current = current_policy()
    requested = {**current, **arguments}
    requested_keys = tuple(name for name in targets if name in arguments)
    return {
        "snapshot": snapshot,
        "snapshot_present": snapshot is not None,
        "snapshot_sha256": hashlib.sha256(snapshot or b"").hexdigest(),
        "persisted": {
            key: values.get(key, "<unset>") for key, _default in targets.values()
        },
        "current": current,
        "requested": requested,
        "requested_keys": requested_keys,
        "changed_keys": tuple(
            name for name in requested_keys if current[name] != requested[name]
        ),
        "external_overrides": tuple(
            name
            for name in requested_keys
            if config.has_external_override(targets[name][0])
        ),
    }


def policy_state_fingerprint(
    state: dict[str, Any], *, public_policy: Callable[[dict[str, Any]], dict[str, Any]]
) -> str:
    # 不改序列化、缺省值或摘要域：重构前签发的确认必须保持同一指纹。
    payload = {
        "snapshot_present": state["snapshot_present"],
        "snapshot_sha256": state["snapshot_sha256"],
        "persisted": state["persisted"],
        "current": public_policy(state["current"]),
        "requested": public_policy(state["requested"]),
        "requested_keys": list(state["requested_keys"]),
        "changed_keys": list(state["changed_keys"]),
        "external_overrides": list(state["external_overrides"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
