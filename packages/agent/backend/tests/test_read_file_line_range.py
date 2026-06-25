"""TDD coverage for read_file line-range slicing (spec 2026-06-25).

Root cause (取证坐实，EPM dogfood thread `0e72d605`): read_file had two silent
out-of-range bugs that sent code-executor into a 20+ turn read_file death loop:

- R1 (M1): passing only ``start_line`` (no ``end_line``) silently skipped the
  whole slicing branch (``if start_line is not None AND end_line is not None``)
  → returned the full file, head-truncated → the LLM always saw the *beginning*
  and believed the file "wrapped around".
- R2 (M2): ``start_line`` beyond EOF sliced to an empty list → returned "" →
  the LLM could not tell "past EOF" from "this range is genuinely empty" → it
  retried with another start_line → death loop.

This file exercises the fixed slicing logic both at the pure-helper level
(deterministic, no sandbox) and through the real ``read_file_tool.func`` entry
point (proving the tool surfaces the fail-loud error / correct slice).
"""

from types import SimpleNamespace
from unittest.mock import patch

from deerflow.sandbox.tools import _slice_content_by_lines, read_file_tool

# ---------------------------------------------------------------------------
# Pure-helper tests (the actual bug site) — T1/T2/T3/T4
# ---------------------------------------------------------------------------

def _make_lines(n: int) -> str:
    """Build an n-line file whose line i is the 1-indexed str(i)."""
    return "\n".join(str(i) for i in range(1, n + 1))


def test_t1_start_line_only_reads_to_end():
    """R1 红→绿: only start_line → from start_line to EOF (not the full head)."""
    content = _make_lines(100)
    # start_line=90 → lines 90..100 inclusive
    sliced = _slice_content_by_lines(content, start_line=90, end_line=None)
    assert sliced.splitlines() == [str(i) for i in range(90, 101)]


def test_t2_end_line_only_reads_from_start():
    """R1: only end_line → from line 1 to end_line inclusive."""
    content = _make_lines(100)
    sliced = _slice_content_by_lines(content, start_line=None, end_line=10)
    assert sliced.splitlines() == [str(i) for i in range(1, 11)]


def test_t3_start_line_beyond_eof_errors():
    """R2 红→绿: start_line > total_lines → fail-loud error, not silent empty."""
    content = _make_lines(100)
    result = _slice_content_by_lines(content, start_line=2895, end_line=None)
    # Fail-loud: a clear error string naming the line and the real line count.
    assert isinstance(result, str)
    assert "start_line=2895" in result
    assert "100 行" in result
    # Crucially NOT an empty string (the death-loop root cause).
    assert result.strip() != ""


def test_t3b_start_line_exactly_at_eof_is_legal():
    """R2 boundary: start_line == total_lines is the last line, NOT out of range."""
    content = _make_lines(100)
    sliced = _slice_content_by_lines(content, start_line=100, end_line=None)
    assert sliced.splitlines() == ["100"]


def test_t4_both_lines_still_work():
    """Regression: both start_line + end_line slice as before (behavior unchanged)."""
    content = _make_lines(100)
    sliced = _slice_content_by_lines(content, start_line=10, end_line=20)
    assert sliced.splitlines() == [str(i) for i in range(10, 21)]


def test_slice_no_range_returns_content_unchanged():
    """Neither bound passed → no slicing (helper returns content untouched)."""
    content = _make_lines(5)
    assert _slice_content_by_lines(content, start_line=None, end_line=None) == content


# ---------------------------------------------------------------------------
# Integration through the real read_file_tool.func entry point
# ---------------------------------------------------------------------------
# These prove the fail-loud error / correct slice actually surface from the
# tool, not just the helper. We bypass sandbox acquisition + path validation by
# patching the helpers read_file_tool calls, so the only logic under test is the
# (now helper-backed) slicing branch — exactly the bug site.


def _fake_runtime():
    """Minimal Runtime stand-in: state/context/config as attributes."""
    return SimpleNamespace(
        state={"thread_data": None, "sandbox": {"sandbox_id": "local"}},
        context={"thread_id": "t-test"},
        config={"configurable": {"thread_id": "t-test"}},
    )


class _StubSandbox:
    """Returns a fixed file body, bypassing the real filesystem."""

    def __init__(self, body: str):
        self._body = body

    def read_file(self, path):  # noqa: D401 - stub signature mirrors Sandbox.read_file
        return self._body


def _patch_sandbox(body: str):
    """Patch the three helpers read_file_tool.func calls before slicing."""
    return (
        patch("deerflow.sandbox.tools.ensure_sandbox_initialized", return_value=_StubSandbox(body)),
        patch("deerflow.sandbox.tools.ensure_thread_directories_exist", return_value=None),
        patch("deerflow.sandbox.tools.is_local_sandbox", return_value=False),
    )


def test_integration_start_line_only_reads_to_end():
    """read_file_tool.func(start_line=90) on a 100-line file → lines 90..100."""
    body = _make_lines(100)
    p1, p2, p3 = _patch_sandbox(body)
    with p1, p2, p3:
        out = read_file_tool.func(_fake_runtime(), "why", "/mnt/x/plan.json", start_line=90, end_line=None)
    assert out.splitlines() == [str(i) for i in range(90, 101)]


def test_integration_start_line_beyond_eof_returns_fail_loud_error():
    """read_file_tool.func(start_line=2895) on a 100-line file → fail-loud error (not empty)."""
    body = _make_lines(100)
    p1, p2, p3 = _patch_sandbox(body)
    with p1, p2, p3:
        out = read_file_tool.func(_fake_runtime(), "why", "/mnt/x/plan.json", start_line=2895, end_line=None)
    assert "start_line=2895" in out
    assert "100 行" in out
    assert out.strip() != ""  # not the silent-empty death-loop behavior


# ---------------------------------------------------------------------------
# T5: code-executor system_prompt warns against reading the big plan
# ---------------------------------------------------------------------------

def test_t5_code_executor_prompt_warns_against_reading_plan():
    """R3 (prompt): code-executor system_prompt must steer toward run_metric_plan
    and away from read_file-ing the (large) plan content."""
    from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG

    prompt = CODE_EXECUTOR_CONFIG.system_prompt
    # Positive steering: call run_metric_plan directly.
    assert "run_metric_plan" in prompt
    # Explicit guard against read_file-ing the plan body to "verify" it
    # (the dogfood death-loop trigger).
    assert "不要 read_file" in prompt or "不 read_file" in prompt or "不要 read" in prompt
