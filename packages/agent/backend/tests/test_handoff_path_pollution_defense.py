"""Layer B path-pollution defense:

Subagents (code-executor in particular) must record raw_files using virtual
/mnt/user-data/uploads/... paths. Host-side absolute paths leak through
Path.resolve() and cause sandbox guardrail rejections downstream (chart-maker
calls catalog.resolve with raw_files; host paths trigger local.host_path_blocked).

The schema-level field_validator enforces the contract; read_handoff applies
reverse-masking + soft schema validation so legacy/leaked handoffs surface as
warnings without breaking gate-2 critical-warning extraction.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from deerflow.subagents.handoff_schemas import CodeExecutorHandoff, CodeExecutorInputs

_VALID_PATHS = [
    "/mnt/user-data/uploads/trial-1.txt",
    "/mnt/user-data/uploads/trial-2.txt",
]


def _make_handoff_dict(raw_files: list[str], extras: dict | None = None) -> dict:
    base = {
        "status": "completed",
        "summary": "OK",
        "paradigm": "fst",
        "analysis_config_id": "test-config-id",
        "inputs": {"raw_files": raw_files, "groups": {"Arena 1": "Treatment"}},
    }
    if extras:
        base.update(extras)
    return base


class TestCodeExecutorInputsVirtualPathValidator:
    def test_accepts_virtual_paths(self):
        inputs = CodeExecutorInputs(raw_files=_VALID_PATHS)
        assert inputs.raw_files == _VALID_PATHS

    def test_accepts_empty_list(self):
        """A handoff with no raw_files is unusual but legal — e.g. paradigm-only response."""
        inputs = CodeExecutorInputs(raw_files=[])
        assert inputs.raw_files == []

    def test_rejects_host_absolute_path(self):
        with pytest.raises(ValidationError) as exc_info:
            CodeExecutorInputs(
                raw_files=[
                    "/home/wangqiuyang/noldus-insight/packages/agent/backend/.deer-flow/users/x/threads/y/user-data/uploads/trial.txt",
                ]
            )
        msg = str(exc_info.value)
        assert "virtual paths" in msg
        assert "/mnt/user-data/" in msg

    def test_rejects_mixed_paths(self):
        """All offenders surface in a single error message so subagent can fix in one pass."""
        with pytest.raises(ValidationError) as exc_info:
            CodeExecutorInputs(
                raw_files=[
                    "/mnt/user-data/uploads/ok.txt",
                    "/home/wangqiuyang/x.txt",
                    "/tmp/y.txt",
                ]
            )
        msg = str(exc_info.value)
        assert "/home/wangqiuyang/x.txt" in msg
        assert "/tmp/y.txt" in msg

    def test_rejects_relative_path(self):
        with pytest.raises(ValidationError):
            CodeExecutorInputs(raw_files=["uploads/trial.txt"])

    def test_rejects_skills_path(self):
        """Skills paths are not user-data — must not appear in inputs."""
        with pytest.raises(ValidationError):
            CodeExecutorInputs(raw_files=["/mnt/skills/custom/foo.md"])


class TestCodeExecutorHandoffFullDoc:
    def test_valid_handoff_with_virtual_inputs(self):
        handoff = CodeExecutorHandoff.model_validate(_make_handoff_dict(_VALID_PATHS))
        assert handoff.inputs is not None
        assert handoff.inputs.raw_files == _VALID_PATHS

    def test_handoff_without_inputs_is_legal(self):
        """Backward compatibility — older handoffs omit inputs altogether."""
        handoff = CodeExecutorHandoff.model_validate({"status": "completed", "summary": "OK", "paradigm": "fst", "analysis_config_id": "test-config-id"})
        assert handoff.inputs is None

    def test_handoff_with_host_paths_rejected(self):
        with pytest.raises(ValidationError):
            CodeExecutorHandoff.model_validate(
                _make_handoff_dict(
                    [
                        "/home/wangqiuyang/noldus-insight/.deer-flow/users/x/threads/y/user-data/uploads/trial.txt",
                    ]
                )
            )


class TestReadHandoffReverseMask:
    """read_handoff() should reverse-mask host paths back to virtual when thread_data given."""

    def _build_thread_data(self, tmp_path: Path) -> dict:
        workspace = tmp_path / "workspace"
        uploads = tmp_path / "uploads"
        outputs = tmp_path / "outputs"
        for d in (workspace, uploads, outputs):
            d.mkdir(parents=True, exist_ok=True)
        return {
            "workspace_path": str(workspace),
            "uploads_path": str(uploads),
            "outputs_path": str(outputs),
        }

    def test_returns_none_when_missing(self, tmp_path: Path):
        from deerflow.agents.middlewares.experiment_context import read_handoff
        result = read_handoff(str(tmp_path))
        assert result is None

    def test_reads_clean_handoff(self, tmp_path: Path):
        from deerflow.agents.middlewares.experiment_context import read_handoff
        from unittest.mock import patch
        from deerflow.agents.middlewares.experiment_context import HandoffStrictMode
        td = self._build_thread_data(tmp_path)
        handoff_path = Path(td["workspace_path"]) / "handoff_code_executor.json"
        clean_handoff = _make_handoff_dict(_VALID_PATHS)
        handoff_path.write_text(json.dumps(clean_handoff), encoding="utf-8")

        with patch("deerflow.agents.middlewares.experiment_context._get_strict_mode", return_value=HandoffStrictMode.WARN):
            result = read_handoff(td["workspace_path"], thread_data=td)
        assert result is not None
        assert result["inputs"]["raw_files"] == _VALID_PATHS
        assert "_schema_violations" not in result

    def test_reverse_masks_host_paths_in_text(self, tmp_path: Path):
        """Polluted handoff with host paths is rewritten back to virtual before parsing."""
        from deerflow.agents.middlewares.experiment_context import read_handoff
        from unittest.mock import patch
        from deerflow.agents.middlewares.experiment_context import HandoffStrictMode
        td = self._build_thread_data(tmp_path)
        handoff_path = Path(td["workspace_path"]) / "handoff_code_executor.json"
        # Simulate a polluted handoff (code-executor ran Path.resolve()).
        polluted_uploads = td["uploads_path"]
        polluted_handoff = _make_handoff_dict(
            [f"{polluted_uploads}/trial.txt"]
        )
        handoff_path.write_text(json.dumps(polluted_handoff, ensure_ascii=False), encoding="utf-8")

        with patch("deerflow.agents.middlewares.experiment_context._get_strict_mode", return_value=HandoffStrictMode.WARN):
            result = read_handoff(td["workspace_path"], thread_data=td)
        assert result is not None
        assert result["inputs"]["raw_files"] == ["/mnt/user-data/uploads/trial.txt"]
        # After masking, the schema validates cleanly — no violation recorded.
        assert "_schema_violations" not in result

    def test_records_violation_when_path_unmappable(self, tmp_path: Path):
        """If host paths cannot be reverse-mapped (different host root), record soft violation."""
        from deerflow.agents.middlewares.experiment_context import read_handoff
        from unittest.mock import patch
        from deerflow.agents.middlewares.experiment_context import HandoffStrictMode
        td = self._build_thread_data(tmp_path)
        handoff_path = Path(td["workspace_path"]) / "handoff_code_executor.json"
        # Path outside any allowed root — mask cannot translate it.
        polluted = _make_handoff_dict(["/some/unrelated/host/path/trial.txt"])
        handoff_path.write_text(json.dumps(polluted), encoding="utf-8")

        with patch("deerflow.agents.middlewares.experiment_context._get_strict_mode", return_value=HandoffStrictMode.WARN):
            result = read_handoff(td["workspace_path"], thread_data=td)
        assert result is not None
        # Raw value preserved (we don't drop the field), but schema violation surfaces.
        assert result["_schema_violations"]
        assert any("virtual paths" in v for v in result["_schema_violations"])

    def test_get_critical_warnings_works_on_polluted_handoff(self, tmp_path: Path):
        """gate2 critical-warning extraction must succeed even when inputs are polluted."""
        from deerflow.agents.middlewares.experiment_context import get_critical_warnings
        from unittest.mock import patch
        from deerflow.agents.middlewares.experiment_context import HandoffStrictMode
        td = self._build_thread_data(tmp_path)
        handoff_path = Path(td["workspace_path"]) / "handoff_code_executor.json"
        polluted = _make_handoff_dict(
            [f"{td['uploads_path']}/trial.txt"],
            extras={
                "data_quality_warnings": [
                    {"severity": "critical", "metric": "all", "message": "n=1", "code": "SAMPLE.TOO_SMALL", "blocks_downstream": True}
                ]
            },
        )
        handoff_path.write_text(json.dumps(polluted), encoding="utf-8")

        with patch("deerflow.agents.middlewares.experiment_context._get_strict_mode", return_value=HandoffStrictMode.WARN):
            warnings = get_critical_warnings(td["workspace_path"], thread_data=td)
        assert len(warnings) == 1
        assert warnings[0]["message"] == "n=1"
