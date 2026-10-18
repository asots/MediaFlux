from __future__ import annotations
from app.agent.kernel.model import ModelAdapter
from typing import cast

import json
import unittest
from pathlib import Path

from app.agent.kernel.bootstrap import build_agent_kernel

FIXTURE = Path(__file__).parent / "fixtures" / "agent_kernel_capability_cases.jsonl"


class NoModel:
    async def stream(self, request, *, cancellation):
        if False:  # pragma: no cover
            yield request


class AgentKernelCapabilityCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.session = build_agent_kernel(model=cast(ModelAdapter | None, NoModel()))

    def test_global_and_cloud_questions_use_one_capability_catalog(self) -> None:
        for message in ("你是谁？你能做什么？", "列出光鸭能力和哪些操作需要确认"):
            with self.subTest(message=message):
                selection = self.session.retriever.retrieve(
                    message, self.session.catalog,
                    context={"owner": "corpus-owner", "session_id": "corpus-session", "channel": "test"},
                )
                self.assertIn("agent.capabilities", selection.names)
                self.assertNotIn("guangya.capabilities", selection.names)

    def test_real_failure_corpus_retrieves_required_atomic_capabilities(self) -> None:
        failures: list[str] = []
        for raw_line in FIXTURE.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            case = json.loads(raw_line)
            selection = self.session.retriever.retrieve(
                case["message"],
                self.session.catalog,
                context={
                    "owner": "corpus-owner",
                    "session_id": "corpus-session",
                    "channel": "test",
                    "reference_kinds": (),
                },
            )
            selected = set(selection.names)
            missing = [name for name in case["required"] if name not in selected]
            if missing:
                failures.append(
                    f"{case['id']}: missing={missing}, selected={list(selection.names)}"
                )
            self.assertGreaterEqual(len(selection.tools), 6)
            business = [tool for tool in selection.tools if not tool.metadata.get("kernel_utility")]
            self.assertLessEqual(len(business), self.session.retriever.maximum)
            utilities = {tool.name for tool in selection.tools if tool.metadata.get("kernel_utility")}
            self.assertLessEqual(utilities, {"agent.batch", "agent.task", "agent.pending_plan"})
        self.assertEqual(failures, [], "\n".join(failures))
