"""三项定时策略已签发确认的指纹兼容与唯一算法回归。"""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "module_name,snapshot,expected",
    [
        (
            "app.agent.guangya_schedule_config_actions",
            None,
            "96d5602c696c0dfa256b491cf325b023bc4f02491f3c9c574781f4e0a0ee4071",
        ),
        (
            "app.agent.guangya_schedule_config_actions",
            b"",
            "188fb76173ee83487428e7ebb518d994918bd557d4f76c2f0fda3fb731bbda72",
        ),
        (
            "app.agent.guangya_schedule_config_actions",
            b"non-sensitive fixture snapshot\n",
            "895bbdb5508882535f9a3292c2442480d3f17065c1f0763faeb170d3128873f8",
        ),
        (
            "app.agent.library_patrol_config_actions",
            None,
            "c38f4084abf637d2c0f88ceb31acd4aba37fd16e612df2306b267d0188025956",
        ),
        (
            "app.agent.library_patrol_config_actions",
            b"",
            "4d0e25177f7e279c0c17a779e27c706a2d00440dc6beaae81c827e1b031b15c4",
        ),
        (
            "app.agent.library_patrol_config_actions",
            b"non-sensitive fixture snapshot\n",
            "a5788ae0d645520402edfbc1d1174c6eebfa860eba845b085087e7c43f529af8",
        ),
        (
            "app.agent.strm_schedule_config_actions",
            None,
            "1db51050c56fd625e18786e0f10d18155b91d5a7a7ba2f1ff0f527cf26746679",
        ),
        (
            "app.agent.strm_schedule_config_actions",
            b"",
            "67a8d490d8a250250c60650364c987e98daf48e1ee52393dc5834dde3fc67f87",
        ),
        (
            "app.agent.strm_schedule_config_actions",
            b"non-sensitive fixture snapshot\n",
            "0dba62b0d9ed525c7a6d4949ca66b4f63e3c14efe06c5773281d555ca9061261",
        ),
    ],
)
def test_history_policy_fingerprint_is_byte_compatible_with_7068c54(
    module_name, snapshot, expected
):
    module = import_module(module_name)
    current = {name: value[1] for name, value in module._TARGETS.items()}
    with (
        patch.object(
            module.config, "read_env_snapshot", return_value=(snapshot, {})
        ) as read,
        patch.object(module, "_current_policy", return_value=current),
        patch.object(module.config, "has_external_override", return_value=False),
    ):
        state = module._capture({"enabled": not current["enabled"]})
        assert module._fingerprint(state) == expected
        assert state["requested_keys"] == state["changed_keys"] == ("enabled",)
        read.assert_called_once_with(module.config.ENV_FILE)
        assert current["enabled"] is False  # 请求快照不得修改当前策略。


@pytest.mark.parametrize(
    "module_name",
    [
        "guangya_schedule_config_actions",
        "library_patrol_config_actions",
        "strm_schedule_config_actions",
    ],
)
def test_normal_policy_shared_snapshot_still_invalidates_each_confirmation_barrier(
    module_name,
):
    module = import_module("app.agent." + module_name)
    current = {name: value[1] for name, value in module._TARGETS.items()}
    with (
        patch.object(module.config, "read_env_snapshot", return_value=(b"fixture", {})),
        patch.object(module, "_current_policy", return_value=current),
        patch.object(module.config, "has_external_override", return_value=False),
    ):
        state = module._capture({"enabled": True})
        fingerprint = module._fingerprint(state)
    variants = []
    for key, value in (
        ("snapshot_present", False),
        ("snapshot_sha256", "0" * 64),
        ("requested_keys", ()),
        ("changed_keys", ()),
        ("external_overrides", ("enabled",)),
    ):
        variants.append({**state, key: value})
    for key in ("current", "requested", "persisted"):
        changed = deepcopy(state)
        name = "enabled" if key != "persisted" else module._TARGETS["enabled"][0]
        changed[key][name] = not changed[key][name] if key != "persisted" else "1"
        variants.append(changed)
    assert all(module._fingerprint(variant) != fingerprint for variant in variants)
