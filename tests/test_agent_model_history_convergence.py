"""模型窗口反复缩小时，旧观察索引稳定且用户条件只累计一次。"""
from __future__ import annotations
from app.agent.kernel.conversation_state import ConversationState
from typing import cast

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.agent.kernel.conversation_state import preserve_history


@pytest.mark.parametrize("count", [12, 13, 25])
def test_omitted_fact_index_converges_after_its_retention_limit(count):
    state = SimpleNamespace(summary="", metadata={})
    original = [
        {
            "role": "tool", "tool_name": "library.search", "tool_call_id": str(index),
            "content": json.dumps({"ok": True, "summary": f"旧观察 {index}"}),
        }
        for index in range(count)
    ]
    memory = preserve_history(cast(ConversationState, state), original, [])
    expected = deepcopy(memory)
    for _ in range(3):
        memory = preserve_history(cast(ConversationState, state), original, [], memory=memory)
        assert memory == expected
    assert [row["call_id"] for row in memory["observations"]] == [
        str(index) for index in range(max(0, count - 12), count)
    ]
    assert state.summary == "" and state.metadata == {}


def test_window_changes_keep_previously_omitted_user_conditions():
    state = SimpleNamespace(summary="", metadata={})
    original = [
        {"role": "user", "content": "只要 1080p", "history_id": "resolution"},
        {"role": "user", "content": "排除已看", "history_id": "unwatched"},
    ]
    memory = preserve_history(cast(ConversationState, state), original, original[1:])
    previous = deepcopy(memory)
    memory = preserve_history(cast(ConversationState, state), original, original[:1], memory=memory)
    assert [row["text"] for row in memory["requests"]] == ["只要 1080p", "排除已看"]
    assert previous["requests"] == [memory["requests"][0]]
    assert preserve_history(cast(ConversationState, state), original, [], memory=memory) == memory
