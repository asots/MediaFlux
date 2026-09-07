"""光鸭改名、文件变更与残留清理共用的追加式执行日志。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.private_files import protect_private_file


def append_guangya_journal(path: Path, event: dict[str, Any]) -> None:
    """目录与事件时间由业务入口确定；统一保持旧 JSONL 协议及 fsync 语义。"""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        protect_private_file(path)
