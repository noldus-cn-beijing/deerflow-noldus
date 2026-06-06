"""Unit tests for handoff strict mode (Sprint 0)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deerflow.agents.middlewares.experiment_context import (
    HandoffSchemaError,
    HandoffStrictMode,
    _get_strict_mode,
    read_handoff,
)


def _write_handoff(workspace: Path, data: dict) -> None:
    (workspace / "handoff_code_executor.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def _valid_handoff() -> dict:
    return {
        "status": "completed",
        "summary": "test",
        "paradigm": "fst",
        "analysis_config_id": "x",
    }


def _invalid_handoff() -> dict:
    return {
        "status": "completed",
        "summary": "test",
    }


class TestStrictModeOff:
    def test_returns_dict_on_violation(self, tmp_path):
        _write_handoff(tmp_path, _invalid_handoff())
        with patch("deerflow.agents.middlewares.experiment_context._get_strict_mode", return_value=HandoffStrictMode.OFF):
            result = read_handoff(str(tmp_path))
            assert result is not None
            assert "_schema_violations" not in result


class TestStrictModeWarn:
    def test_logs_violation(self, tmp_path, caplog):
        import logging

        _write_handoff(tmp_path, _invalid_handoff())
        with patch("deerflow.agents.middlewares.experiment_context._get_strict_mode", return_value=HandoffStrictMode.WARN), \
             caplog.at_level(logging.WARNING, logger="deerflow.agents.middlewares.experiment_context"):
            result = read_handoff(str(tmp_path))
            assert result is not None
            assert "_schema_violations" in result
            assert len(result["_schema_violations"]) > 0

    def test_valid_handoff_no_violation(self, tmp_path):
        _write_handoff(tmp_path, _valid_handoff())
        with patch("deerflow.agents.middlewares.experiment_context._get_strict_mode", return_value=HandoffStrictMode.WARN):
            result = read_handoff(str(tmp_path))
            assert result is not None
            assert "_schema_violations" not in result


class TestStrictModeFailClosed:
    def test_raises_on_violation(self, tmp_path):
        _write_handoff(tmp_path, _invalid_handoff())
        with patch("deerflow.agents.middlewares.experiment_context._get_strict_mode", return_value=HandoffStrictMode.FAIL_CLOSED):
            with pytest.raises(HandoffSchemaError, match="FAIL_CLOSED"):
                read_handoff(str(tmp_path))

    def test_valid_handoff_passes(self, tmp_path):
        _write_handoff(tmp_path, _valid_handoff())
        with patch("deerflow.agents.middlewares.experiment_context._get_strict_mode", return_value=HandoffStrictMode.FAIL_CLOSED):
            result = read_handoff(str(tmp_path))
            assert result is not None


class TestGetStrictMode:
    def test_emergency_file_forces_warn(self, tmp_path):
        emergency = tmp_path / "disable_strict_handoff"
        emergency.touch()
        with patch("deerflow.agents.middlewares.experiment_context._EMERGENCY_DOWNGRADE_FILE", str(emergency)):
            mode = _get_strict_mode()
            assert mode == HandoffStrictMode.WARN

    def test_reads_from_config(self):
        mock_config = MagicMock()
        mock_config.handoff_strict_mode = "fail_closed"
        with patch("deerflow.config.get_app_config", return_value=mock_config), \
             patch.object(Path, "exists", return_value=False):
            mode = _get_strict_mode()
            assert mode == HandoffStrictMode.FAIL_CLOSED

    def test_invalid_config_falls_back_to_warn(self):
        mock_config = MagicMock()
        mock_config.handoff_strict_mode = "invalid_value"
        with patch("deerflow.config.get_app_config", return_value=mock_config), \
             patch.object(Path, "exists", return_value=False):
            mode = _get_strict_mode()
            assert mode == HandoffStrictMode.WARN
