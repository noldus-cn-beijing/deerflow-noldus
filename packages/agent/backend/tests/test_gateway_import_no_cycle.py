"""Regression guard: production entrypoints must import without circular-import errors.

Why a subprocess test (not a plain `import` in-process):
`tests/conftest.py` injects a ``sys.modules`` mock for
``deerflow.subagents.executor`` to break a *pre-existing* import cycle so the
rest of the suite can run. That mock means an in-process ``import`` here would
NOT exercise the real module graph — exactly the blind spot that let a Spec C
regression (executor.py top-level ``from ...seal_handoff_tools import
_seal_handoff_to_workspace``) ship "green" yet crash ``make dev`` with::

    ImportError: cannot import name '_seal_handoff_to_workspace' from partially
    initialized module 'deerflow.tools.builtins.seal_handoff_tools'
    (most likely due to a circular import)

These tests spawn a clean Python subprocess (no conftest, no mock) and import
the two real startup entrypoints the way uvicorn / langgraph do. If any harness
module reintroduces a top-level edge that closes an import cycle, the subprocess
exits non-zero and the test fails — before it reaches a user's `make dev`.

See: memory feedback_harness_must_import_without_ethoinsight (same "tests green
via mock, real import crashes" failure mode).
"""

import subprocess
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _import_in_clean_subprocess(import_stmt: str) -> subprocess.CompletedProcess:
    """Run ``import_stmt`` in a fresh interpreter with PYTHONPATH=backend, no conftest."""
    return subprocess.run(
        [sys.executable, "-c", import_stmt],
        cwd=str(_BACKEND_ROOT),
        env={"PYTHONPATH": str(_BACKEND_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_gateway_app_imports_without_circular_import():
    """The Gateway entrypoint (uvicorn imports ``app.gateway``) must load cleanly.

    This is the exact chain that crashed: app.gateway -> routers -> skills ->
    deerflow.agents -> ... -> subagents.executor -> seal_handoff_tools.
    """
    result = _import_in_clean_subprocess("import app.gateway")
    assert result.returncode == 0, (
        "Importing app.gateway failed (likely a circular import reintroduced at "
        f"module top-level). stderr:\n{result.stderr}"
    )
    assert "partially initialized module" not in result.stderr
    assert "circular import" not in result.stderr


def test_lead_agent_imports_without_circular_import():
    """The LangGraph entrypoint (``make_lead_agent``) must load cleanly."""
    result = _import_in_clean_subprocess("from deerflow.agents import make_lead_agent")
    assert result.returncode == 0, (
        "Importing make_lead_agent failed (likely a circular import). "
        f"stderr:\n{result.stderr}"
    )
    assert "partially initialized module" not in result.stderr
    assert "circular import" not in result.stderr
