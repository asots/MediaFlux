"""本地媒体领域共用的任务边界与确定性摘要。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

_WORKSPACE_OWNER = "admin"
_MAX_TASK_NUMBER = 100
_MAX_INSPECTION_NUMBER = 2_147_483_647
_MAX_SOURCE_NUMBER = 10_000
_RETRYABLE = frozenset({"failed", "requires_manual"})


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _fingerprint(
    payload: dict[str, Any], *, default: Callable[[Any], Any] | None = None
) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=default
        ).encode("utf-8")
    ).hexdigest()
