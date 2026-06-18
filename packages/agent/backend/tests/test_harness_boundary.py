"""Boundary check: harness layer must not import from app layer.

The deerflow-harness package (packages/harness/deerflow/) is a standalone,
publishable agent framework. It must never depend on the app layer (app/).

This test scans all Python files in the harness package and fails if any
``from app.`` or ``import app.`` statement is found.
"""

import ast
from pathlib import Path

HARNESS_ROOT = Path(__file__).parent.parent / "packages" / "harness" / "deerflow"

BANNED_PREFIXES = ("app.",)

# Top-level imports (module-level, not nested in functions/classes) of these
# packages are banned: a top-level import runs at harness import time and would
# (a) make the harness package un-importable without this workspace dep, and
# (b) re-introduce the import-cycle / sync-wipe failure modes the SSOT
# PROTECTED list guards against. Lazy imports inside functions are allowed.
BANNED_TOP_LEVEL_PREFIXES = ("ethoinsight",)


def _collect_imports(filepath: Path) -> list[tuple[int, str]]:
    """Return (line_number, module_path) for every import in *filepath*."""
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    results: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                results.append((node.lineno, node.module))
    return results


def test_harness_does_not_import_app():
    violations: list[str] = []

    for py_file in sorted(HARNESS_ROOT.rglob("*.py")):
        for lineno, module in _collect_imports(py_file):
            if any(module == prefix.rstrip(".") or module.startswith(prefix) for prefix in BANNED_PREFIXES):
                rel = py_file.relative_to(HARNESS_ROOT.parent.parent.parent)
                violations.append(f"  {rel}:{lineno}  imports {module}")

    assert not violations, "Harness layer must not import from app layer:\n" + "\n".join(violations)


def _collect_top_level_imports(filepath: Path) -> list[tuple[int, str]]:
    """Return (line_number, module_path) for imports at MODULE TOP LEVEL only.

    Walks ``tree.body`` (the module's top-level statement list) rather than
    ``ast.walk`` so imports nested inside functions/classes (lazy imports) are
    excluded — those do not run at harness import time and are allowed.
    """
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    results: list[tuple[int, str]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                results.append((node.lineno, node.module))
    return results


def test_harness_does_not_top_level_import_ethoinsight():
    """The harness must not import ethoinsight at module top level.

    ethoinsight is an app-layer workspace dep. A top-level import would make
    the harness un-importable without ethoinsight installed and re-introduce
    the "tests green via mock, real import crashes" failure mode (see memory
    feedback_harness_must_import_without_ethoinsight). Lazy imports inside
    function bodies are allowed.
    """
    violations: list[str] = []

    for py_file in sorted(HARNESS_ROOT.rglob("*.py")):
        for lineno, module in _collect_top_level_imports(py_file):
            if any(
                module == prefix.rstrip(".") or module.startswith(prefix)
                for prefix in BANNED_TOP_LEVEL_PREFIXES
            ):
                rel = py_file.relative_to(HARNESS_ROOT.parent.parent.parent)
                violations.append(f"  {rel}:{lineno}  top-level imports {module}")

    assert not violations, (
        "Harness layer must not import ethoinsight at module top level "
        "(breaks publishability / re-introduces import cycle):\n"
        + "\n".join(violations)
    )
