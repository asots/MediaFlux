"""Provider 查询 Artifact 与对象引用的 owner-scoped 短期存储。"""
from __future__ import annotations

import secrets
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from itertools import islice
from typing import Any

from app.agent.provider_models import ProviderGatewayError
from app.agent.provider_projection import project_provider_value


@dataclass(slots=True)
class _ProviderObject:
    object_ref: str
    provider: str
    profile_ref: str
    kind: str
    raw_id: str
    snapshot: dict[str, Any]


@dataclass(slots=True)
class _ProviderArtifact:
    artifact_ref: str
    owner: str
    session_id: str
    provider: str
    profile_ref: str
    operation: str
    created_at: float
    expires_at: float
    data: dict[str, Any]
    objects: dict[str, _ProviderObject]


class ProviderArtifactStore:
    def __init__(self, *, ttl_seconds: int = 900, max_entries: int = 512) -> None:
        self.ttl_seconds = max(60, min(int(ttl_seconds), 3600))
        self.max_entries = max(32, min(int(max_entries), 4096))
        self._items: dict[str, _ProviderArtifact] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _ref(prefix: str) -> str:
        return f"{prefix}-{secrets.token_hex(12).upper()}"

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def clear_owner(self, *, owner: str) -> int:
        """清除 owner 的全部短期对象引用。"""
        owner_key = str(owner or "").strip()
        if not owner_key:
            return 0
        with self._lock:
            refs = [
                ref for ref, item in self._items.items()
                if item.owner == owner_key
            ]
            for ref in refs:
                self._items.pop(ref, None)
        return len(refs)

    def clear_session(self, *, owner: str, session_id: str) -> int:
        """仅清除指定 owner/session 的短期对象引用。"""
        owner_key = str(owner or "").strip()
        session_key = str(session_id or "").strip()
        if not owner_key or not session_key:
            return 0
        with self._lock:
            refs = [
                ref
                for ref, item in self._items.items()
                if item.owner == owner_key and item.session_id == session_key
            ]
            for ref in refs:
                self._items.pop(ref, None)
        return len(refs)

    def _prune_locked(self, now: float) -> None:
        expired = [key for key, item in self._items.items() if item.expires_at <= now]
        for key in expired:
            self._items.pop(key, None)
        if len(self._items) <= self.max_entries:
            return
        oldest = sorted(self._items.values(), key=lambda item: item.created_at)
        for item in oldest[:len(self._items) - self.max_entries]:
            self._items.pop(item.artifact_ref, None)

    def put(
        self,
        *,
        owner: str,
        session_id: str,
        provider: str,
        profile_ref: str,
        operation: str,
        data: dict[str, Any],
        max_items: int = 32,
    ) -> tuple[str, dict[str, Any]]:
        if not owner:
            raise ProviderGatewayError("Provider 查询需要已登录会话", code="precondition_failed")
        now = time.monotonic()
        artifact_ref = self._ref("PA")
        objects: dict[str, _ProviderObject] = {}
        allow_open_url = str(provider or "").strip().casefold() == "media"

        item_limit = max(1, min(int(max_items), 500))
        projection_truncated = False

        def visit(value: Any, depth: int = 0) -> Any:
            nonlocal projection_truncated
            if depth > 4:
                if value is not None:
                    projection_truncated = True
                return None
            if isinstance(value, (list, tuple)):
                if len(value) > item_limit:
                    projection_truncated = True
                return [visit(item, depth + 1) for item in value[:item_limit]]
            if not isinstance(value, dict):
                return value
            raw_id = str(value.get("__object_id") or "").strip()
            kind = str(value.get("__object_kind") or "item").strip().casefold()
            if len(value) > 32:
                projection_truncated = True
            projected = {
                str(key): visit(item, depth + 1)
                for key, item in islice(value.items(), 32)
                if key not in {"__object_id", "__object_kind"}
            }
            # count 若明确对应本层被裁剪的集合，保留来源计数并改为实际
            # 展示数量；total 仍是上游库存总量，不能据展示上限推断缺集。
            for key, item in islice(value.items(), 32):
                if isinstance(item, (list, tuple)) and len(item) > item_limit:
                    projected["truncated"] = True
                    if type(value.get("count")) is int and value["count"] == len(item):
                        projected["source_count"] = value["count"]
                        projected["count"] = item_limit
            if raw_id:
                object_ref = self._ref("PO")
                projected["object_ref"] = object_ref
                safe_snapshot = project_provider_value(
                    projected,
                    max_items=max(32, item_limit),
                    allow_open_url=allow_open_url,
                )
                objects[object_ref] = _ProviderObject(
                    object_ref=object_ref,
                    provider=provider,
                    profile_ref=profile_ref,
                    kind=kind,
                    raw_id=raw_id,
                    snapshot=(safe_snapshot if isinstance(safe_snapshot, dict) else {}),
                )
            return projected

        # 先按预算遍历再复制公开投影，不能为最终不可见的条目签发句柄。
        visited = visit(data)
        public_data = project_provider_value(
            visited,
            max_items=max(32, item_limit),
            allow_open_url=allow_open_url,
        )
        if not isinstance(public_data, dict):
            raise ProviderGatewayError("Provider 响应结构无效", code="invalid_response")
        if projection_truncated:
            public_data["projection_truncated"] = True
            public_data["truncated"] = True
        item = _ProviderArtifact(
            artifact_ref=artifact_ref,
            owner=owner,
            session_id=str(session_id or ""),
            provider=provider,
            profile_ref=profile_ref,
            operation=operation,
            created_at=now,
            expires_at=now + self.ttl_seconds,
            data=deepcopy(public_data),
            objects=objects,
        )
        with self._lock:
            self._prune_locked(now)
            self._items[artifact_ref] = item
        return artifact_ref, public_data

    def resolve_object(
        self,
        *,
        owner: str,
        session_id: str,
        object_ref: str,
        provider: str,
        profile_ref: str,
        expected_kind: str,
    ) -> tuple[str, dict[str, Any]]:
        normalized = str(object_ref or "").strip().upper()
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            for artifact in self._items.values():
                candidate = artifact.objects.get(normalized)
                if candidate is None:
                    continue
                if artifact.owner != owner or artifact.session_id != str(session_id or ""):
                    break
                if candidate.provider != provider or candidate.profile_ref != profile_ref:
                    break
                if expected_kind and candidate.kind != expected_kind:
                    raise ProviderGatewayError(
                        "对象引用类型不匹配", code="invalid_arguments"
                    )
                return candidate.raw_id, deepcopy(candidate.snapshot)
        raise ProviderGatewayError(
            "对象引用已失效，请重新查询", code="artifact_expired"
        )
