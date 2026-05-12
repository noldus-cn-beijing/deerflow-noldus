"""Tests for ethoinsight.scripts._cli helper module."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest


class TestEmitResult:
    def test_emit_result_prints_marker_with_payload(self, capsys):
        from ethoinsight.scripts._cli import emit_result

        emit_result({"metric": "open_arm_time_ratio", "value": 0.35})

        captured = capsys.readouterr()
        assert "[result]" in captured.out
        # The JSON payload must follow the marker on the same line
        line = next(l for l in captured.out.splitlines() if l.startswith("[result]"))
        payload = json.loads(line[len("[result] "):])
        assert payload == {"metric": "open_arm_time_ratio", "value": 0.35}


class TestSaveOutputJson:
    def test_save_output_writes_json_atomically(self, tmp_path: Path):
        from ethoinsight.scripts._cli import save_output_json

        out_path = tmp_path / "out.json"
        save_output_json(out_path, {"value": 0.5})

        assert out_path.exists()
        assert json.loads(out_path.read_text()) == {"value": 0.5}

    def test_save_output_creates_parent_dirs(self, tmp_path: Path):
        from ethoinsight.scripts._cli import save_output_json

        out_path = tmp_path / "deep/nested/out.json"
        save_output_json(out_path, {"value": 1})

        assert out_path.exists()


class TestReadInputsJson:
    def test_read_inputs_json_returns_list_of_paths(self, tmp_path: Path):
        from ethoinsight.scripts._cli import read_inputs_json

        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps(["/tmp/a.txt", "/tmp/b.txt"]))

        result = read_inputs_json(inputs_file)
        assert result == ["/tmp/a.txt", "/tmp/b.txt"]

    def test_read_inputs_json_rejects_non_array(self, tmp_path: Path):
        from ethoinsight.scripts._cli import read_inputs_json

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"not": "array"}))

        with pytest.raises(ValueError, match="must be a JSON array"):
            read_inputs_json(bad)


class TestReadGroupsJson:
    def test_read_groups_json_returns_dict(self, tmp_path: Path):
        from ethoinsight.scripts._cli import read_groups_json

        groups_file = tmp_path / "groups.json"
        groups_file.write_text(json.dumps({"control": ["s1"], "treatment": ["s2"]}))

        result = read_groups_json(groups_file)
        assert result == {"control": ["s1"], "treatment": ["s2"]}
