from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.agent.domain_catalog.cloud_runtime import (
    guangya_organize_status,
    wait_for_guangya_operation,
)
from app.agent.kernel.ports.existing_actions import adapt_tool_spec
from app.agent.kernel.state import CancellationToken, InMemorySessionStateStore
from app.agent.models import RiskLevel, ToolContext, ToolResult, ToolSpec

_OPERATION_REF = "GY-0000-0000-0000-0000-0000-0000-0000-0001"


def _snapshot(status: str, *, stats: dict[str, int] | None = None) -> ToolResult:
    return ToolResult(
        ok=status in {"queued", "running", "completed"},
        status=status,
        summary=f"snapshot:{status}",
        data={
            "task": {
                "status": status,
                "running": status == "running",
                "stats": stats or {},
                "started_at": "2026-09-19T00:00:00+08:00",
                "finished_at": "" if status in {"queued", "running"} else "2026-09-19T00:00:02+08:00",
            },
            "queue": {"pending_count": 1 if status == "queued" else 0},
        },
    )


class GuangYaOperationTrackingTests(unittest.IsolatedAsyncioTestCase):
    def test_status_binds_public_ref_to_owner_and_keeps_safe_stats(self) -> None:
        manager = Mock()
        manager.status.return_value = {
            "operation_queue": {"total": 0},
            "schedule": {},
        }
        manager.task_result.return_value = {
            "status": "completed",
            "stats": {
                "relocated": 1,
                "created": 2,
                "trashed": 3,
                "strm_scope_unknown": 1,
                "strm_trigger_skipped": 1,
                "secret": 99,
            },
            "stoppable": False,
        }
        with patch("app.modules.organize_tasks.get_organize_manager", return_value=manager):
            result = guangya_organize_status(
                {"operation_ref": _OPERATION_REF},
                ToolContext(owner="webk:v1:owner-safe"),
            )

        manager.task_result.assert_called_once_with(
            _OPERATION_REF, owner="webk:v1:owner-safe"
        )
        self.assertEqual(
            result.data["task"]["stats"],
            {"relocated": 1, "created": 2, "trashed": 3, "strm_scope_unknown": 1, "strm_trigger_skipped": 1},
        )
        self.assertNotIn("secret", result.data["task"]["stats"])
        self.assertIn("未触发 STRM 联动", " ".join(result.suggestions))

    async def test_waits_on_persistent_snapshots_and_reports_safe_progress(self) -> None:
        accepted = ToolResult(
            True,
            "accepted",
            "已提交",
            data={
                "operation_ref": _OPERATION_REF,
                "total": 4,
                "relocate_count": 2,
                "cloud_write": False,
            },
            model_data={"total": 4, "cloud_write": False},
        )
        progress: list[dict] = []
        snapshots = iter(
            (
                _snapshot("queued"),
                _snapshot("running"),
                _snapshot("completed", stats={"relocated": 2, "created": 1}),
            )
        )

        async def report(payload):
            progress.append(dict(payload))

        with (
            patch(
                "app.agent.domain_catalog.cloud_runtime.guangya_organize_status",
                side_effect=lambda *_args, **_kwargs: next(snapshots),
            ) as status,
            patch(
                "app.agent.domain_catalog.cloud_runtime.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
        ):
            result = await wait_for_guangya_operation(
                accepted,
                tool="guangya.fs.change.execute",
                context=ToolContext(owner="tg:v1:owner-safe"),
                report_progress=report,
            )

        self.assertEqual(result.status, "completed")
        self.assertTrue(result.ok)
        self.assertEqual(result.data["total"], 4)
        self.assertEqual(result.data["relocate_count"], 2)
        self.assertEqual(result.data["operation_ref"], _OPERATION_REF)
        self.assertEqual(result.data["background_job"]["stats"]["relocated"], 2)
        self.assertNotIn("cloud_write", result.data)
        self.assertNotIn("cloud_write", result.to_model_dict()["data"])
        self.assertEqual(result.to_model_dict()["data"]["background_job"]["status"], "completed")
        self.assertEqual(status.call_count, 3)
        self.assertEqual(sleep.await_count, 2)
        self.assertEqual(
            [item["phase"] for item in progress],
            ["background_job", "background_job", "background_job"],
        )
        self.assertEqual(
            [item["status"] for item in progress], ["queued", "running", "completed"]
        )
        self.assertTrue(all(item["operation_ref"] == _OPERATION_REF for item in progress))
        self.assertTrue(all(item["tool"] == "guangya.fs.change.execute" for item in progress))
        self.assertNotIn("owner", progress[0])

    async def test_completed_cloud_write_keeps_optional_sync_warning(self):
        snapshot = _snapshot("completed", stats={"renamed": 1, "strm_scope_unknown": 1})
        snapshot.suggestions.append("同步范围未能确认，本次未触发 STRM 联动。")
        with patch("app.agent.domain_catalog.cloud_runtime.guangya_organize_status", return_value=snapshot):
            result = await wait_for_guangya_operation(
                ToolResult(True, "accepted", "已提交", data={"operation_ref": _OPERATION_REF}),
                tool="guangya.fs.change.execute", context=ToolContext(owner="owner"),
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "completed")
        self.assertIn("未触发 STRM 联动", " ".join(result.suggestions))
        self.assertEqual(result.data["stats"]["renamed"], 1)

    async def test_terminal_states_are_returned_without_collapsing_their_status(self) -> None:
        for terminal in ("partial", "failed", "cancelled", "manual_review", "stopped"):
            with self.subTest(terminal=terminal):
                accepted = ToolResult(
                    True,
                    "accepted",
                    "已提交",
                    data={
                        "operation_ref": _OPERATION_REF,
                        "total": 1,
                        "cloud_write": False,
                    },
                )
                with (
                    patch(
                        "app.agent.domain_catalog.cloud_runtime.guangya_organize_status",
                        return_value=_snapshot(terminal),
                    ) as status,
                    patch(
                        "app.agent.domain_catalog.cloud_runtime.asyncio.sleep",
                        new=AsyncMock(),
                    ) as sleep,
                ):
                    result = await wait_for_guangya_operation(
                        accepted,
                        tool="guangya.rename.execute",
                        context=ToolContext(owner="webk:v1:owner-safe"),
                        report_progress=AsyncMock(),
                    )
                self.assertEqual(result.status, terminal)
                self.assertEqual(result.ok, terminal == "stopped")
                self.assertEqual(result.data["operation_ref"], _OPERATION_REF)
                self.assertEqual(result.data["total"], 1)
                self.assertNotIn("cloud_write", result.data)
                status.assert_called_once()
                sleep.assert_not_awaited()

    async def test_missing_job_is_unknown_and_non_gy_submission_is_unchanged(self) -> None:
        accepted = ToolResult(
            True,
            "accepted",
            "已提交",
            data={"operation_ref": _OPERATION_REF, "total": 1},
        )
        with patch(
            "app.agent.domain_catalog.cloud_runtime.guangya_organize_status",
            return_value=ToolResult(
                False,
                "empty",
                "没有找到这个光鸭操作编号",
                data={"operation_ref": _OPERATION_REF, "found": False},
            ),
        ):
            unknown = await wait_for_guangya_operation(
                accepted,
                tool="guangya.fs.change.execute",
                context=ToolContext(owner="webk:v1:owner-safe"),
            )

        self.assertEqual(unknown.status, "outcome_unknown")
        self.assertEqual(unknown.data["background_job"]["status"], "unknown")
        self.assertEqual(unknown.data["operation_ref"], _OPERATION_REF)

        normal_write = ToolResult(
            True, "accepted", "已提交", data={"accepted": True, "total": 1}
        )
        with patch(
            "app.agent.domain_catalog.cloud_runtime.guangya_organize_status"
        ) as status:
            unchanged = await wait_for_guangya_operation(
                normal_write,
                tool="ingest.submit",
                context=ToolContext(owner="webk:v1:owner-safe"),
            )
        self.assertIs(unchanged, normal_write)
        status.assert_not_called()

    async def test_timeout_keeps_last_running_fact_as_unknown(self) -> None:
        accepted = ToolResult(
            True,
            "accepted",
            "已提交",
            data={
                "operation_ref": _OPERATION_REF,
                "total": 1,
                "cloud_write": False,
            },
        )
        with patch(
            "app.agent.domain_catalog.cloud_runtime.guangya_organize_status",
            return_value=_snapshot("running"),
        ):
            result = await wait_for_guangya_operation(
                accepted,
                tool="guangya.fs.change.execute",
                context=ToolContext(owner="webk:v1:owner-safe"),
                report_progress=AsyncMock(),
                timeout_seconds=0,
            )

        self.assertEqual(result.status, "outcome_unknown")
        self.assertFalse(result.ok)
        self.assertEqual(result.data["operation_ref"], _OPERATION_REF)
        self.assertEqual(result.data["background_job"]["last_status"], "running")
        self.assertTrue(result.data["background_job"]["timed_out"])
        self.assertNotIn("cloud_write", result.data)
        self.assertNotIn("failed", result.summary)
        self.assertNotIn("completed", result.summary)

    async def test_sync_post_write_verifier_runs_off_event_loop(self) -> None:
        verifier_threads: list[int] = []
        main_thread = threading.get_ident()
        accepted = ToolResult(
            True,
            "accepted",
            "已提交",
            data={"operation_ref": _OPERATION_REF},
        )

        def verifier(_arguments, value):
            verifier_threads.append(threading.get_ident())
            return value

        spec = ToolSpec(
            name="guangya.fs.change.execute",
            description="测试同步写后验证",
            risk=RiskLevel.DANGER,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            validator=lambda value: dict(value),
            requires_confirmation=True,
            context_confirmation_preparer=lambda _args, _ctx: (
                ToolResult(True, "ready", "预检"),
                "fingerprint",
            ),
            context_confirmed_handler=lambda _args, _snapshot, _ctx: accepted,
            post_write_verifier=verifier,
        )
        tool = adapt_tool_spec(spec)
        state = InMemorySessionStateStore()
        lease, _ = await state.begin_turn(
            owner="webk:v1:owner-safe", session_id="session", request_id="request"
        )
        from app.agent.kernel.pipeline import ToolCallContext

        context = ToolCallContext(
            owner="webk:v1:owner-safe",
            session_id="session",
            request_id="request",
            turn_id=lease.turn_id,
            lease=lease,
            cancellation=CancellationToken(),
            report_progress=AsyncMock(),
        )
        with patch(
            "app.agent.kernel.ports.existing_actions.wait_for_guangya_operation",
            new=AsyncMock(side_effect=lambda value, **_kwargs: value),
        ) as wait:
            result = await tool.verify({}, accepted, context)

        self.assertIs(result, accepted)
        self.assertEqual(len(verifier_threads), 1)
        self.assertNotEqual(verifier_threads[0], main_thread)
        wait.assert_awaited_once()

    async def test_cancel_drains_sync_verifier_before_verify_task_finishes(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        accepted = ToolResult(True, "success", "已核验", data={"total": 1})

        def verifier(_arguments, value):
            entered.set()
            release.wait(timeout=5)
            finished.set()
            return value

        spec = ToolSpec(
            name="config.cancel_drain",
            description="测试取消时收稳同步核验",
            risk=RiskLevel.WRITE,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            validator=lambda value: dict(value),
            requires_confirmation=True,
            context_confirmation_preparer=lambda _args, _ctx: (
                ToolResult(True, "ready", "预检"),
                "fingerprint",
            ),
            context_confirmed_handler=lambda _args, _snapshot, _ctx: accepted,
            post_write_verifier=verifier,
        )
        tool = adapt_tool_spec(spec)
        state = InMemorySessionStateStore()
        lease, _ = await state.begin_turn(
            owner="webk:v1:owner-safe", session_id="session", request_id="request"
        )
        from app.agent.kernel.pipeline import ToolCallContext

        context = ToolCallContext(
            owner="webk:v1:owner-safe",
            session_id="session",
            request_id="request",
            turn_id=lease.turn_id,
            lease=lease,
            cancellation=CancellationToken(),
            report_progress=AsyncMock(),
        )
        verify_done: list[bool] = []

        async def run_verify():
            await tool.verify({}, accepted, context)
            verify_done.append(True)

        task = asyncio.create_task(run_verify())
        self.assertTrue(await asyncio.to_thread(entered.wait, 2))
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertFalse(finished.is_set())
        self.assertFalse(verify_done)

        release.set()
        await task
        self.assertTrue(finished.is_set())
        self.assertEqual(verify_done, [True])

    async def test_kernel_verify_uses_one_generic_gy_wait_without_custom_verifier(self) -> None:
        accepted = ToolResult(
            True,
            "accepted",
            "已提交",
            data={"operation_ref": _OPERATION_REF},
        )
        spec = ToolSpec(
            name="guangya.fs.change.execute",
            description="测试 GY 后台写入",
            risk=RiskLevel.DANGER,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            validator=lambda value: dict(value),
            requires_confirmation=True,
            context_confirmation_preparer=lambda _args, _ctx: (
                ToolResult(True, "ready", "预检"),
                "fingerprint",
            ),
            context_confirmed_handler=lambda _args, _snapshot, _ctx: accepted,
        )
        tool = adapt_tool_spec(spec)
        state = InMemorySessionStateStore()
        lease, _ = await state.begin_turn(
            owner="webk:v1:owner-safe", session_id="session", request_id="request"
        )
        from app.agent.kernel.pipeline import ToolCallContext

        context = ToolCallContext(
            owner="webk:v1:owner-safe",
            session_id="session",
            request_id="request",
            turn_id=lease.turn_id,
            lease=lease,
            cancellation=CancellationToken(),
            report_progress=AsyncMock(),
        )
        terminal = ToolResult(True, "completed", "已完成", data={"operation_ref": _OPERATION_REF})
        with patch(
            "app.agent.kernel.ports.existing_actions.wait_for_guangya_operation",
            new=AsyncMock(return_value=terminal),
        ) as wait:
            result = await tool.verify({}, accepted, context)

        self.assertIs(result, terminal)
        wait.assert_awaited_once()
        self.assertEqual(wait.await_args.kwargs["tool"], "guangya.fs.change.execute")


if __name__ == "__main__":
    unittest.main()
