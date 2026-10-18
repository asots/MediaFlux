"""运行期导入必须保持单向；函数内导入不能掩盖领域与内核的回边。"""
from __future__ import annotations

import ast
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1] / "app"
AGENT_ROOT = APP_ROOT / "agent"


def _runtime_import_graph(root: Path = AGENT_ROOT, *, package: str = "app.agent") -> dict[str, set[str]]:
    modules = {}
    for path in root.rglob("*.py"):
        suffix = list(path.relative_to(root).with_suffix("").parts)
        if suffix[-1] == "__init__":
            suffix.pop()
        modules[".".join((package, *suffix))] = path
    graph: dict[str, set[str]] = {module: set() for module in modules}

    class Imports(ast.NodeVisitor):
        def __init__(self, module: str, path: Path) -> None:
            self.module = module
            self.package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]

        def visit_If(self, node: ast.If) -> None:
            test = node.test
            negated = isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
            if negated:
                assert isinstance(test, ast.UnaryOp)
                test = test.operand
            type_only = (
                isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
                or isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if type_only:
                for statement in node.body if negated else node.orelse:
                    self.visit(statement)
            else:
                self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            graph[self.module].update(alias.name for alias in node.names if alias.name in modules)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.level:
                parts = self.package.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1])
                if node.module:
                    base += "." + node.module
            else:
                base = node.module or ""
            for alias in node.names:
                child = base + "." + alias.name
                target = child if child in modules else base
                if target in modules:
                    graph[self.module].add(target)

    for module, path in modules.items():
        Imports(module, path).visit(ast.parse(path.read_text(encoding="utf-8")))
    return graph


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    indexes: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    result: list[list[str]] = []

    def visit(module: str) -> None:
        indexes[module] = low[module] = len(indexes)
        stack.append(module)
        active.add(module)
        for target in sorted(graph[module]):
            if target not in indexes:
                visit(target)
                low[module] = min(low[module], low[target])
            elif target in active:
                low[module] = min(low[module], indexes[target])
        if low[module] != indexes[module]:
            return
        group = []
        while True:
            target = stack.pop()
            active.remove(target)
            group.append(target)
            if target == module:
                break
        if len(group) > 1 or module in graph[module]:
            result.append(sorted(group))

    for module in sorted(graph):
        if module not in indexes:
            visit(module)
    return sorted(result)


def test_agent_runtime_import_graph_has_no_cycles() -> None:
    assert _cycles(_runtime_import_graph()) == []


def test_receipt_contract_has_no_execution_or_business_observer_dependency() -> None:
    graph = _runtime_import_graph()
    pending = ["app.agent.kernel.completion_receipts"]
    reached: set[str] = set()
    while pending:
        module = pending.pop()
        if module not in reached:
            reached.add(module)
            pending.extend(graph[module] - reached)
    assert reached.isdisjoint({
        "app.agent.effect_completion", "app.agent.completion_observers",
        "app.agent.kernel.pipeline", "app.agent.kernel.session",
        "app.agent.kernel.continuation", "app.agent.kernel.batch_effects",
        "app.agent.kernel.state", "app.agent.kernel.persistence",
    })


def test_scoped_trust_policy_does_not_depend_on_executor_or_session() -> None:
    graph = _runtime_import_graph()
    pending = ["app.agent.kernel.scoped_trust"]
    reached: set[str] = set()
    while pending:
        module = pending.pop()
        if module not in reached:
            reached.add(module)
            pending.extend(graph[module] - reached)
    assert reached.isdisjoint({
        "app.agent.kernel.pipeline", "app.agent.kernel.session",
        "app.agent.kernel.turn", "app.agent.kernel.tool_batches",
    })


def test_dependency_check_detects_function_local_relative_import_cycles(tmp_path: Path) -> None:
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "first.py").write_text("def call():\n    from . import second\n")
    (tmp_path / "second.py").write_text("def call():\n    from .first import call\n")
    assert _cycles(_runtime_import_graph(tmp_path)) == [["app.agent.first", "app.agent.second"]]


def test_type_checking_import_does_not_create_runtime_cycle(tmp_path: Path) -> None:
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "first.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from .second import Value\n"
    )
    (tmp_path / "second.py").write_text("from . import first\nclass Value: pass\n")
    assert _cycles(_runtime_import_graph(tmp_path)) == []


def test_execution_kernel_has_no_domain_or_transport_imports() -> None:
    graph = _runtime_import_graph(APP_ROOT, package="app")
    violations = {}
    for module, targets in graph.items():
        if not module.startswith("app.agent.kernel."):
            continue
        if module == "app.agent.kernel.bootstrap" or module.startswith("app.agent.kernel.ports."):
            continue
        forbidden = {
            target for target in targets
            if target.startswith(("app.agent.domains", "app.bot", "app.routes", "app.modules.telegram_"))
            or target in {
                "app.agent.transports", "app.agent.effect_completion",
                "app.agent.completion_observers",
            }
        }
        if forbidden:
            violations[module] = forbidden
    assert not violations


def test_dependency_check_detects_cycles_through_external_transport(tmp_path: Path) -> None:
    (tmp_path / "agent").mkdir()
    (tmp_path / "bot").mkdir()
    (tmp_path / "agent" / "core.py").write_text(
        "def deliver():\n    from app.bot.delivery import send\n"
    )
    (tmp_path / "bot" / "delivery.py").write_text(
        "def send():\n    from app.agent.core import deliver\n"
    )
    assert _cycles(_runtime_import_graph(tmp_path, package="app")) == [
        ["app.agent.core", "app.bot.delivery"],
    ]
