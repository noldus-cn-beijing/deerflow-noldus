"""Unit tests for Sprint 6: experiment_summary memory fact injection into seal_report_writer_handoff."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from deerflow.tools.builtins.seal_handoff_tools import (
    _extract_key_findings_count,
    _extract_n_per_group,
    _write_experiment_summary_memory,
    seal_report_writer_handoff,
)


def _make_runtime(workspace_dir: str, *, user_id: str | None = None, thread_id: str | None = None) -> MagicMock:
    """Create a mock Runtime with workspace_path in thread_data."""
    runtime = MagicMock()
    thread_data = {"workspace_path": workspace_dir}
    if user_id is not None:
        thread_data["user_id"] = user_id
    state = {"thread_data": thread_data}
    if thread_id is not None:
        state["thread_id"] = thread_id
    runtime.state = state
    return runtime


def _make_workspace(tmp_path: Path, *, paradigm: str = "epm", config_id: str = "test-config-id") -> Path:
    """Create a workspace directory with an experiment-context.json."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "experiment-context.json").write_text(
        json.dumps({"paradigm": paradigm, "analysis_config_id": config_id}),
        encoding="utf-8",
    )
    return ws


def _write_handoff_files(
    workspace: Path,
    *,
    paradigm: str = "epm",
    n_per_group: int | None = 8,
    key_findings_count: int | None = 3,
    config_id: str = "test-config-id",
) -> None:
    """Write handoff_code_executor.json and handoff_data_analyst.json for testing."""
    ce_data = {
        "status": "completed",
        "paradigm": paradigm,
        "analysis_config_id": config_id,
        "metrics_summary": {},
    }
    if n_per_group is not None:
        ce_data["metrics_summary"]["control"] = {"time_in_open_arms": {"n": n_per_group}}
    (workspace / "handoff_code_executor.json").write_text(
        json.dumps(ce_data, ensure_ascii=False), encoding="utf-8"
    )

    da_data = {
        "status": "completed",
        "key_findings": ["finding"] * (key_findings_count or 0),
        "analysis_config_id": config_id,
    }
    (workspace / "handoff_data_analyst.json").write_text(
        json.dumps(da_data, ensure_ascii=False), encoding="utf-8"
    )


# ============================================================================
# Test _extract_n_per_group
# ============================================================================


class TestExtractNPerGroup:
    def test_from_metrics_summary(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_handoff_files(ws, n_per_group=8)
        assert _extract_n_per_group(ws) == "8"

    def test_from_metadata_priority(self, tmp_path):
        ws = _make_workspace(tmp_path)
        ce_data = {"status": "completed", "metadata": {"n_per_group": 12}, "metrics_summary": {"g": {"m": {"n": 6}}}}
        (ws / "handoff_code_executor.json").write_text(json.dumps(ce_data), encoding="utf-8")
        assert _extract_n_per_group(ws) == "12"

    def test_missing_file_returns_unknown(self, tmp_path):
        ws = _make_workspace(tmp_path)
        assert _extract_n_per_group(ws) == "unknown"

    def test_invalid_json_returns_unknown(self, tmp_path):
        ws = _make_workspace(tmp_path)
        (ws / "handoff_code_executor.json").write_text("NOT JSON", encoding="utf-8")
        assert _extract_n_per_group(ws) == "unknown"


class TestExtractKeyFindingsCount:
    def test_counts_findings(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_handoff_files(ws, key_findings_count=3)
        assert _extract_key_findings_count(ws) == 3

    def test_missing_file_returns_zero(self, tmp_path):
        ws = _make_workspace(tmp_path)
        assert _extract_key_findings_count(ws) == 0

    def test_empty_findings(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_handoff_files(ws, key_findings_count=0)
        assert _extract_key_findings_count(ws) == 0


# ============================================================================
# Test _write_experiment_summary_memory
# ============================================================================


class TestWriteExperimentSummaryMemory:
    """Tests for the memory fact injection helper."""

    def test_writes_fact_with_correct_content(self, tmp_path):
        ws = _make_workspace(tmp_path, paradigm="epm", config_id="abc123456789")
        _write_handoff_files(ws, paradigm="epm", n_per_group=8, key_findings_count=3, config_id="abc123456789")

        with patch("deerflow.agents.memory.updater.create_memory_fact") as mock_fact:
            _write_experiment_summary_memory(
                workspace=ws,
                paradigm="epm",
                config_id="abc123456789",
                thread_id="thread-001",
                user_id="user-001",
            )
            mock_fact.assert_called_once()
            kwargs = mock_fact.call_args.kwargs
            content = kwargs["content"]
            assert "epm analysis" in content
            assert "n_per_group=8" in content
            assert "key_findings_count=3" in content
            assert "analysis_config_id=abc123456789" in content

    def test_category_is_experiment_summary(self, tmp_path):
        ws = _make_workspace(tmp_path, paradigm="fst")
        _write_handoff_files(ws, paradigm="fst")

        with patch("deerflow.agents.memory.updater.create_memory_fact") as mock_fact:
            _write_experiment_summary_memory(workspace=ws, paradigm="fst", config_id="x", thread_id="t1", user_id=None)
            assert mock_fact.call_args.kwargs["category"] == "experiment_summary"

    def test_confidence_is_1(self, tmp_path):
        ws = _make_workspace(tmp_path, paradigm="epm")
        _write_handoff_files(ws)

        with patch("deerflow.agents.memory.updater.create_memory_fact") as mock_fact:
            _write_experiment_summary_memory(workspace=ws, paradigm="epm", config_id="c", thread_id="t1", user_id="u1")
            assert mock_fact.call_args.kwargs["confidence"] == 1.0

    def test_lineage_in_content_includes_thread_and_config(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_handoff_files(ws)

        with patch("deerflow.agents.memory.updater.create_memory_fact") as mock_fact:
            _write_experiment_summary_memory(workspace=ws, paradigm="epm", config_id="cfg-99", thread_id="th-42", user_id=None)
            # Lineage is folded into content (create_memory_fact has no `source` kwarg).
            content = mock_fact.call_args.kwargs["content"]
            assert "thread=th-42" in content
            assert "analysis_config_id=cfg-99" in content
            # Regression guard: source must NOT be passed (real fn would TypeError).
            assert "source" not in mock_fact.call_args.kwargs

    def test_n_per_group_unknown_when_missing(self, tmp_path):
        ws = _make_workspace(tmp_path, paradigm="epm")
        _write_handoff_files(ws, n_per_group=None, key_findings_count=2)

        with patch("deerflow.agents.memory.updater.create_memory_fact") as mock_fact:
            _write_experiment_summary_memory(workspace=ws, paradigm="epm", config_id="c1", thread_id="t1", user_id=None)
            content = mock_fact.call_args.kwargs["content"]
            assert "n_per_group=unknown" in content

    def test_key_findings_count_zero_when_empty(self, tmp_path):
        ws = _make_workspace(tmp_path, paradigm="epm")
        _write_handoff_files(ws, key_findings_count=0)

        with patch("deerflow.agents.memory.updater.create_memory_fact") as mock_fact:
            _write_experiment_summary_memory(workspace=ws, paradigm="epm", config_id="c1", thread_id="t1", user_id=None)
            content = mock_fact.call_args.kwargs["content"]
            assert "key_findings_count=0" in content

    def test_failure_does_not_raise(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_handoff_files(ws)

        with patch("deerflow.agents.memory.updater.create_memory_fact", side_effect=OSError("disk full")):
            # Should not raise — memory failure is non-fatal
            _write_experiment_summary_memory(workspace=ws, paradigm="epm", config_id="c1", thread_id="t1", user_id=None)

    def test_user_id_passed_through(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_handoff_files(ws)

        with patch("deerflow.agents.memory.updater.create_memory_fact") as mock_fact:
            _write_experiment_summary_memory(workspace=ws, paradigm="epm", config_id="c1", thread_id="t1", user_id="alice")
            assert mock_fact.call_args.kwargs["user_id"] == "alice"

    def test_no_handoff_files_still_writes_with_unknowns(self, tmp_path):
        """Even when handoff files don't exist, fact is still written (with unknown values)."""
        ws = _make_workspace(tmp_path, paradigm="oft")
        # No handoff files written

        with patch("deerflow.agents.memory.updater.create_memory_fact") as mock_fact:
            _write_experiment_summary_memory(workspace=ws, paradigm="oft", config_id="c2", thread_id="t2", user_id=None)
            content = mock_fact.call_args.kwargs["content"]
            assert "oft analysis" in content
            assert "n_per_group=unknown" in content

    def test_reads_n_per_group_from_code_executor_metadata(self, tmp_path):
        """n_per_group should prefer code-executor handoff metadata over metrics_summary."""
        ws = _make_workspace(tmp_path, paradigm="epm")
        ce_data = {
            "status": "completed",
            "paradigm": "epm",
            "metadata": {"n_per_group": 12},
        }
        (ws / "handoff_code_executor.json").write_text(json.dumps(ce_data), encoding="utf-8")
        da_data = {"status": "completed", "key_findings": ["f1"], "analysis_config_id": "c3"}
        (ws / "handoff_data_analyst.json").write_text(json.dumps(da_data), encoding="utf-8")

        with patch("deerflow.agents.memory.updater.create_memory_fact") as mock_fact:
            _write_experiment_summary_memory(workspace=ws, paradigm="epm", config_id="c3", thread_id="t3", user_id=None)
            content = mock_fact.call_args.kwargs["content"]
            assert "n_per_group=12" in content


class TestSealReportWriterMemoryIntegration:
    """Integration: seal_report_writer_handoff triggers memory write."""

    def test_seal_report_writer_writes_memory_fact(self, tmp_path):
        ws = _make_workspace(tmp_path, paradigm="epm", config_id="cfg-mem-test")
        _write_handoff_files(ws, paradigm="epm", n_per_group=6, key_findings_count=2, config_id="cfg-mem-test")
        runtime = _make_runtime(str(ws), user_id="bob", thread_id="thread-mem")

        with patch("deerflow.tools.builtins.seal_handoff_tools._write_experiment_summary_memory") as mock_mem:
            # Call the underlying function directly (bypass StructuredTool wrapper)
            result = seal_report_writer_handoff.func(
                status="completed",
                report_path="/mnt/user-data/outputs/report.md",
                sections_written=["Results", "Discussion"],
                runtime=runtime,
            )
            assert result.startswith("OK: sealed handoff_report_writer.json")
            mock_mem.assert_called_once()
            call_kwargs = mock_mem.call_args.kwargs
            assert call_kwargs["paradigm"] == "epm"
            assert call_kwargs["config_id"] == "cfg-mem-test"
            assert call_kwargs["thread_id"] == "thread-mem"
            assert call_kwargs["user_id"] == "bob"

    def test_seal_report_writer_failed_does_not_write_memory(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(str(ws))

        with patch("deerflow.tools.builtins.seal_handoff_tools._write_experiment_summary_memory") as mock_mem:
            result = seal_report_writer_handoff.func(
                status="failed",
                report_path="",
                errors=["something broke"],
                runtime=runtime,
            )
            assert result.startswith("OK: sealed handoff_report_writer.json")
            mock_mem.assert_not_called()

    def test_seal_report_writer_no_user_id_passes_none(self, tmp_path):
        ws = _make_workspace(tmp_path, paradigm="fst")
        _write_handoff_files(ws, paradigm="fst")
        runtime = _make_runtime(str(ws))  # no user_id

        with patch("deerflow.tools.builtins.seal_handoff_tools._write_experiment_summary_memory") as mock_mem:
            seal_report_writer_handoff.func(
                status="completed",
                report_path="/mnt/user-data/outputs/report.md",
                sections_written=["Results"],
                runtime=runtime,
            )
            call_kwargs = mock_mem.call_args.kwargs
            assert call_kwargs["user_id"] is None

    def test_seal_report_writer_no_thread_id_passes_unknown(self, tmp_path):
        ws = _make_workspace(tmp_path, paradigm="fst")
        _write_handoff_files(ws, paradigm="fst")
        runtime = _make_runtime(str(ws))  # no thread_id

        with patch("deerflow.tools.builtins.seal_handoff_tools._write_experiment_summary_memory") as mock_mem:
            seal_report_writer_handoff.func(
                status="completed",
                report_path="/mnt/user-data/outputs/report.md",
                sections_written=["Results"],
                runtime=runtime,
            )
            call_kwargs = mock_mem.call_args.kwargs
            assert call_kwargs["thread_id"] == "unknown"


# ============================================================================
# Regression: exercise the REAL create_memory_fact (no mock on the fn itself).
# Only file I/O is isolated, so a signature mismatch surfaces as a test failure
# instead of being swallowed by the helper's try/except. This is the guard the
# original suite lacked — mocking create_memory_fact hid that it rejects `source=`.
# ============================================================================


class TestRealCreateMemoryFactIntegration:
    """The fact must actually be created by the real updater, not a mock."""

    def test_fact_is_written_through_real_create_memory_fact(self, tmp_path):
        ws = _make_workspace(tmp_path, paradigm="epm", config_id="real-cfg")
        _write_handoff_files(ws, paradigm="epm", n_per_group=8, key_findings_count=3, config_id="real-cfg")

        captured: dict = {}

        def _fake_save(memory_data, agent_name=None, *, user_id=None):
            captured["memory_data"] = memory_data
            return True

        # Patch only the I/O boundary; create_memory_fact runs for real.
        with (
            patch("deerflow.agents.memory.updater.get_memory_data", return_value={"facts": []}),
            patch("deerflow.agents.memory.updater._save_memory_to_file", side_effect=_fake_save),
        ):
            # Must NOT raise and must NOT swallow a TypeError.
            _write_experiment_summary_memory(
                workspace=ws,
                paradigm="epm",
                config_id="real-cfg",
                thread_id="th-real",
                user_id=None,
            )

        # The real updater appended a fact with our content/category/confidence.
        facts = captured.get("memory_data", {}).get("facts", [])
        assert len(facts) == 1, "real create_memory_fact did not append a fact (was the call swallowed?)"
        fact = facts[0]
        assert fact["category"] == "experiment_summary"
        assert fact["confidence"] == 1.0
        assert "epm analysis" in fact["content"]
        assert "n_per_group=8" in fact["content"]
        assert "analysis_config_id=real-cfg" in fact["content"]
        assert "thread=th-real" in fact["content"]
