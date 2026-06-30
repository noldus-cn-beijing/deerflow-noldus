"""ls tool file-path disambiguation regression.

Forensic root cause of the 2026-06-30 EPM-28 dogfood「plan_metrics.json 缺失」false
failure (thread 8827351d-e2b1-4292-b69e-a3bde14b5fb0):

The code-executor subagent's system_prompt instructs it to confirm
`plan_metrics.json` exists by calling ``ls /mnt/user-data/workspace/plan_metrics.json``
(a **file** path). But ``ls`` is a directory-listing tool: ``list_dir`` returns ``[]``
for any non-directory path (``if not root_path.is_dir(): return result``), and
``ls_tool`` then emits ``"(empty)"``. The subagent cannot distinguish three states,
all of which return ``"(empty)"``:

    ls <existing file>      -> "(empty)"   <- misread as "file missing" -> seal failed
    ls <missing path>       -> "(empty)"
    ls <empty directory>    -> "(empty)"

Meanwhile the lead's own ``ls /mnt/user-data/workspace`` (a **directory**) correctly
listed the file as a child — proving the file was on disk all along. So the guard
was right to allow; the false failure is the ``ls``-on-file ambiguity, not a TOCTOU
window (H-A) or a guard gap (H-B).

This test pins the fix: ``ls`` must disambiguate「file」and「not found」from「empty dir」.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from deerflow.sandbox.tools import ls_tool


def _wire_runtime(monkeypatch, tmp_path: Path) -> SimpleNamespace:
    """Wire ls_tool to resolve /mnt/user-data/workspace/* onto tmp_path.

    Mirrors the lightweight pattern in tests/test_sandbox_tools_security.py
    (drive ls_tool.func directly with a SimpleNamespace runtime + monkeypatched
    helpers), so we exercise the real ls_tool control flow without spinning up a
    full LangGraph runtime.
    """
    runtime = SimpleNamespace(state={}, context={"thread_id": "thread-ls-file"}, config={})

    # A real LocalSandbox whose list_dir walks the resolved host path.
    from deerflow.sandbox.local.local_sandbox import LocalSandbox

    sandbox = LocalSandbox("local-ls-file")

    monkeypatch.setattr("deerflow.sandbox.tools.ensure_sandbox_initialized", lambda runtime: sandbox)
    monkeypatch.setattr("deerflow.sandbox.tools.ensure_thread_directories_exist", lambda runtime: None)
    monkeypatch.setattr("deerflow.sandbox.tools.is_local_sandbox", lambda runtime: True)
    monkeypatch.setattr(
        "deerflow.sandbox.tools.get_thread_data",
        lambda runtime: {"workspace_path": str(tmp_path)},
    )
    monkeypatch.setattr("deerflow.sandbox.tools.validate_local_tool_path", lambda path, thread_data, **kw: None)

    def _resolve(path, thread_data):  # noqa: ANN001
        # Map the single virtual workspace dir onto tmp_path so real files exist.
        return path.replace("/mnt/user-data/workspace", str(tmp_path))

    monkeypatch.setattr("deerflow.sandbox.tools._resolve_and_validate_user_data_path", _resolve)
    return runtime


def _run_ls(runtime: SimpleNamespace, virtual_path: str) -> str:
    return ls_tool.func(runtime=runtime, description="probe", path=virtual_path)


# ============================================================
# Reproduction: ls on an EXISTING FILE must not say "(empty)"
# ============================================================

class TestLsOnFilePath:
    """The core regression: ls <existing file> must signal the file exists."""

    def test_existing_file_is_not_empty(self, monkeypatch, tmp_path):
        plan = tmp_path / "plan_metrics.json"
        plan.write_text('{"metrics": [1]}', encoding="utf-8")
        runtime = _wire_runtime(monkeypatch, tmp_path)

        result = _run_ls(runtime, "/mnt/user-data/workspace/plan_metrics.json")

        # The bug: this returned "(empty)", which the subagent misread as missing.
        assert result != "(empty)", "ls on an existing file must not be indistinguishable from an empty directory"

    def test_existing_file_message_signals_file_exists(self, monkeypatch, tmp_path):
        plan = tmp_path / "plan_metrics.json"
        plan.write_text('{"metrics": [1]}', encoding="utf-8")
        runtime = _wire_runtime(monkeypatch, tmp_path)

        result = _run_ls(runtime, "/mnt/user-data/workspace/plan_metrics.json")

        lowered = result.lower()
        assert "file" in lowered or "plan_metrics.json" in result, (
            "ls on a file must clearly indicate it is a file (so the subagent infers "
            f"the file exists). Got: {result!r}"
        )


# ============================================================
# Disambiguation: ls on a MISSING path must not equal "(empty)"
# ============================================================

class TestLsOnMissingPath:
    def test_missing_path_is_not_empty(self, monkeypatch, tmp_path):
        runtime = _wire_runtime(monkeypatch, tmp_path)

        result = _run_ls(runtime, "/mnt/user-data/workspace/nope.json")

        assert result != "(empty)", (
            "ls on a missing path must not be indistinguishable from an empty directory"
        )

    def test_missing_path_message_signals_not_found(self, monkeypatch, tmp_path):
        runtime = _wire_runtime(monkeypatch, tmp_path)

        result = _run_ls(runtime, "/mnt/user-data/workspace/nope.json")

        lowered = result.lower()
        assert "not found" in lowered or "no such" in lowered or "missing" in lowered or "does not exist" in lowered, (
            f"ls on a missing path must clearly say not-found. Got: {result!r}"
        )


# ============================================================
# Positive non-regression: directories behave exactly as before
# ============================================================

class TestLsOnDirectoryUnchanged:
    def test_directory_with_files_lists_children(self, monkeypatch, tmp_path):
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "b.txt").write_text("b", encoding="utf-8")
        runtime = _wire_runtime(monkeypatch, tmp_path)

        result = _run_ls(runtime, "/mnt/user-data/workspace")

        assert "a.txt" in result
        assert "b.txt" in result

    def test_empty_directory_still_says_empty(self, monkeypatch, tmp_path):
        runtime = _wire_runtime(monkeypatch, tmp_path)

        result = _run_ls(runtime, "/mnt/user-data/workspace")

        assert result == "(empty)", "a genuinely empty directory must still report (empty)"
