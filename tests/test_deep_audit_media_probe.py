"""本轮本地探测预算与缓存恢复；不启动 ffprobe/真实外部连接。"""

from __future__ import annotations

import tests  # noqa: F401
import json
import subprocess
from unittest.mock import patch

import pytest

from app.modules import media_probe as probe
from tests.support import isolated_test_database


@pytest.fixture(autouse=True)
def no_network():
    with (
        patch(
            "socket.socket.connect",
            side_effect=AssertionError("real network forbidden"),
        ),
        patch(
            "socket.create_connection",
            side_effect=AssertionError("real network forbidden"),
        ),
    ):
        yield


@pytest.mark.parametrize("remaining_at_timeout", [0.0, 0.001])
def test_local_budget_timeout_does_not_suppress_next_task(
    tmp_path, remaining_at_timeout
):
    with isolated_test_database():
        video = tmp_path / "Film.mkv"
        video.write_bytes(b"local-media-fixture")
        stat = video.stat()
        snapshot = dict(
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            device=stat.st_dev,
            inode=stat.st_ino,
        )
        clock = [100.0]
        with patch.object(probe.time, "monotonic", side_effect=lambda: clock[0]):
            budget = probe.ProbeBudget(attempts=24, max_seconds=20)

            def budget_timeout(executable, path, timeout):
                assert timeout == 20
                clock[0] = 120.0 - remaining_at_timeout
                raise subprocess.TimeoutExpired("fake-ffprobe", timeout)

            with patch.object(probe, "_run_ffprobe", side_effect=budget_timeout):
                assert (
                    probe.probe_local_media_profile(video, **snapshot, budget=budget)
                    is None
                )
            assert budget.timeouts == 1
            payload = json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "h264",
                            "width": 1920,
                            "height": 1080,
                        }
                    ]
                }
            )
            with patch.object(
                probe,
                "_run_ffprobe",
                return_value=subprocess.CompletedProcess([], 0, payload, ""),
            ) as ffprobe:
                profile = probe.probe_local_media_profile(
                    video, **snapshot, budget=probe.ProbeBudget(max_seconds=20)
                )
                assert profile is not None, "总任务预算耗尽不能让健康文件负缓存 300 秒"
                assert profile.resolution == "1080p"
                assert ffprobe.call_count == 1
                assert probe.probe_local_media_profile(video, **snapshot) == profile
                assert ffprobe.call_count == 1, "恢复后再次调用应复用真实数据库缓存"


def test_actual_file_timeout_keeps_failure_backoff(tmp_path):
    with isolated_test_database():
        video = tmp_path / "Slow.mkv"
        video.write_bytes(b"slow-file")
        stat = video.stat()
        kwargs = dict(
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            device=stat.st_dev,
            inode=stat.st_ino,
        )
        with patch.object(
            probe, "_run_ffprobe", side_effect=subprocess.TimeoutExpired("fake", 30)
        ) as ffprobe:
            assert (
                probe.probe_local_media_profile(
                    video, **kwargs, budget=probe.ProbeBudget(max_seconds=100)
                )
                is None
            )
            assert probe.probe_local_media_profile(video, **kwargs) is None
            assert ffprobe.call_count == 1
