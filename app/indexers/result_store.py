from __future__ import annotations

import copy
import heapq
import re
import secrets
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .errors import IndexerResultExpired, IndexerResultNotFound
from .models import IndexerItem

_RESULT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


@dataclass(slots=True)
class _StoredEntry:
    item: IndexerItem
    expires_at: datetime


class IndexerResultStore:
    """Bounded in-memory store mapping opaque short-lived IDs to provider results."""

    def __init__(
        self,
        *,
        ttl_seconds: int = 600,
        max_entries: int = 10_000,
        clock: Callable[[], datetime] | None = None,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.ttl_seconds = int(ttl_seconds)
        self.max_entries = int(max_entries)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._entries: OrderedDict[str, _StoredEntry] = OrderedDict()
        self._expired_ids: OrderedDict[str, None] = OrderedDict()
        self._expiry_heap: list[tuple[datetime, str]] = []
        self._lock = threading.RLock()

    def put(self, item: IndexerItem) -> str:
        now = self._clock()
        result_id = secrets.token_urlsafe(24)
        with self._lock:
            self._prune_expired(now)
            while len(self._entries) >= self.max_entries:
                evicted_id, _ = self._entries.popitem(last=False)
                self._expired_ids.pop(evicted_id, None)
            self._store_entry(result_id, item, now)
        return result_id

    def restore(self, result_id: str, item: IndexerItem) -> str:
        """用原 opaque ID 恢复已认证的短期结果，供跨进程确认继续执行。"""
        token = str(result_id or "").strip()
        if not _RESULT_ID_PATTERN.fullmatch(token):
            raise ValueError("invalid result id")
        if not isinstance(item, IndexerItem):
            raise TypeError("item must be IndexerItem")
        now = self._clock()
        with self._lock:
            self._prune_expired(now)
            while token not in self._entries and len(self._entries) >= self.max_entries:
                evicted_id, _ = self._entries.popitem(last=False)
                self._expired_ids.pop(evicted_id, None)
            self._store_entry(token, item, now)
            self._entries.move_to_end(token)
            self._expired_ids.pop(token, None)
        return token

    def get(self, result_id: str) -> IndexerItem:
        token = str(result_id or "").strip()
        if not token:
            raise IndexerResultNotFound()
        now = self._clock()
        with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                if token in self._expired_ids:
                    raise IndexerResultExpired()
                raise IndexerResultNotFound()
            if entry.expires_at <= now:
                self._entries.pop(token, None)
                self._remember_expired(token)
                raise IndexerResultExpired()
            return copy.deepcopy(entry.item)

    def _store_entry(self, token: str, item: IndexerItem, now: datetime) -> None:
        entry = _StoredEntry(copy.deepcopy(item), now + timedelta(seconds=self.ttl_seconds))
        self._entries[token] = entry
        heapq.heappush(self._expiry_heap, (entry.expires_at, token))
        # restore/容量淘汰留下的旧节点惰性清理；最多 2N，重建成本摊到此前
        # 至少 N 次写入。TTL/时钟倒退不要求 OrderedDict 按到期时间有序。
        if len(self._expiry_heap) > 2 * self.max_entries:
            self._expiry_heap = [
                (stored.expires_at, result_id) for result_id, stored in self._entries.items()
            ]
            heapq.heapify(self._expiry_heap)

    def _remember_expired(self, token: str) -> None:
        self._expired_ids[token] = None
        self._expired_ids.move_to_end(token)
        while len(self._expired_ids) > self.max_entries:
            self._expired_ids.popitem(last=False)

    def _prune_expired(self, now: datetime) -> None:
        while self._expiry_heap and self._expiry_heap[0][0] <= now:
            expires_at, token = heapq.heappop(self._expiry_heap)
            entry = self._entries.get(token)
            # 原 ID 恢复后有了新 TTL，旧节点不可误删新值。
            if entry is not None and entry.expires_at == expires_at:
                self._entries.pop(token)
                self._remember_expired(token)
