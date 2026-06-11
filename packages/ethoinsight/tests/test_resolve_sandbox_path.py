"""Tests for resolve_sandbox_path — virtual /mnt/... → real path resolution.

The sandbox injects DEERFLOW_PATH_* env vars so Python scripts can resolve
virtual paths at runtime.  ``resolve_sandbox_path`` consumes these env vars
to turn ``/mnt/user-data/workspace/m_x.json`` into a real host path that
``Path().read_text()`` can open.

Coverage:
  - Virtual path resolved via DEERFLOW_PATH_* env
  - Real (non-/mnt) path passthrough
  - Virtual path with no matching env → fail-safe passthrough
  - Longest-prefix match (workspace before user-data)
  - Exact prefix match (no trailing /)
  - End-to-end: validate_plan_results uses resolved path
  - read_inputs_json / read_groups_json resolve paths
"""

import json
import os
from pathlib import Path

import pytest

from ethoinsight.scripts._cli import (
    resolve_sandbox_path,
    read_inputs_json,
    read_groups_json,
)
from ethoinsight.validate_catalog import validate_plan_results


# ---------------------------------------------------------------------------
# env key format verified against sandbox code (local_sandbox.py:338)
#   env_key = "DEERFLOW_PATH_" + container_path.strip("/").replace("/", "_").replace("-", "_").upper()
#
# /mnt/user-data/workspace → DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE
# /mnt/user-data/uploads   → DEERFLOW_PATH_MNT_USER_DATA_UPLOADS
# /mnt/user-data/outputs   → DEERFLOW_PATH_MNT_USER_DATA_OUTPUTS
# /mnt/user-data           → DEERFLOW_PATH_MNT_USER_DATA
# /mnt/shared              → DEERFLOW_PATH_MNT_SHARED
# /mnt/skills              → DEERFLOW_PATH_MNT_SKILLS
# /mnt/acp-workspace       → DEERFLOW_PATH_MNT_ACP_WORKSPACE
# ---------------------------------------------------------------------------

ENV_KEY_WORKSPACE = "DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE"
ENV_KEY_UPLOADS = "DEERFLOW_PATH_MNT_USER_DATA_UPLOADS"
ENV_KEY_OUTPUTS = "DEERFLOW_PATH_MNT_USER_DATA_OUTPUTS"
ENV_KEY_USER_DATA = "DEERFLOW_PATH_MNT_USER_DATA"
ENV_KEY_SHARED = "DEERFLOW_PATH_MNT_SHARED"
ENV_KEY_SKILLS = "DEERFLOW_PATH_MNT_SKILLS"


class TestResolveSandboxPath:
    """Unit tests for resolve_sandbox_path."""

    def test_virtual_workspace_path_resolved(self, tmp_path, monkeypatch):
        """Virtual /mnt/user-data/workspace/... → real path via env."""
        real_ws = tmp_path / "real_workspace"
        real_ws.mkdir()
        monkeypatch.setenv(ENV_KEY_WORKSPACE, str(real_ws))

        resolved = resolve_sandbox_path("/mnt/user-data/workspace/m_x.json")
        assert resolved == real_ws / "m_x.json"

    def test_virtual_workspace_path_exact_match(self, tmp_path, monkeypatch):
        """Exact prefix match (no trailing sub-path)."""
        real_ws = tmp_path / "ws"
        real_ws.mkdir()
        monkeypatch.setenv(ENV_KEY_WORKSPACE, str(real_ws))

        resolved = resolve_sandbox_path("/mnt/user-data/workspace")
        assert resolved == real_ws

    def test_virtual_uploads_path_resolved(self, tmp_path, monkeypatch):
        """Virtual /mnt/user-data/uploads/... → real path via env."""
        real_up = tmp_path / "real_uploads"
        real_up.mkdir()
        monkeypatch.setenv(ENV_KEY_UPLOADS, str(real_up))

        resolved = resolve_sandbox_path("/mnt/user-data/uploads/subject1.txt")
        assert resolved == real_up / "subject1.txt"

    def test_virtual_outputs_path_resolved(self, tmp_path, monkeypatch):
        """Virtual /mnt/user-data/outputs/... → real path via env."""
        real_out = tmp_path / "real_outputs"
        real_out.mkdir()
        monkeypatch.setenv(ENV_KEY_OUTPUTS, str(real_out))

        resolved = resolve_sandbox_path("/mnt/user-data/outputs/chart.png")
        assert resolved == real_out / "chart.png"

    def test_longest_prefix_match(self, tmp_path, monkeypatch):
        """Longer prefix (/mnt/user-data/workspace) wins over shorter (/mnt/user-data)."""
        real_ws = tmp_path / "real_ws"
        real_ud = tmp_path / "real_ud"
        real_ws.mkdir()
        real_ud.mkdir()
        monkeypatch.setenv(ENV_KEY_WORKSPACE, str(real_ws))
        monkeypatch.setenv(ENV_KEY_USER_DATA, str(real_ud))

        resolved = resolve_sandbox_path("/mnt/user-data/workspace/m_x.json")
        # Must resolve via workspace (longer), NOT user-data (shorter)
        assert resolved == real_ws / "m_x.json"
        assert resolved != real_ud / "workspace/m_x.json"

    def test_fallback_to_shorter_prefix(self, tmp_path, monkeypatch):
        """When longer prefix has no env, fall back to shorter prefix."""
        real_ud = tmp_path / "real_ud"
        real_ud.mkdir()
        monkeypatch.setenv(ENV_KEY_USER_DATA, str(real_ud))
        # workspace env NOT set

        resolved = resolve_sandbox_path("/mnt/user-data/workspace/m_x.json")
        assert resolved == real_ud / "workspace/m_x.json"

    def test_real_path_passthrough(self):
        """Already-real host path → returned as-is."""
        p = "/home/user/data/file.json"
        assert resolve_sandbox_path(p) == Path(p)

    def test_real_path_passthrough_relative(self):
        """Relative path → returned as-is."""
        p = "relative/path/to/file.txt"
        assert resolve_sandbox_path(p) == Path(p)

    def test_virtual_path_no_env_failsafe(self, monkeypatch):
        """/mnt but no matching DEERFLOW_PATH_* env → returned as-is (no crash)."""
        for k in list(os.environ):
            if k.startswith("DEERFLOW_PATH_"):
                monkeypatch.delenv(k)
        result = resolve_sandbox_path("/mnt/user-data/workspace/x.json")
        assert result == Path("/mnt/user-data/workspace/x.json")

    def test_path_object_accepted(self, tmp_path, monkeypatch):
        """Path input works same as str input."""
        real_ws = tmp_path / "ws"
        real_ws.mkdir()
        monkeypatch.setenv(ENV_KEY_WORKSPACE, str(real_ws))

        resolved = resolve_sandbox_path(Path("/mnt/user-data/workspace/data.json"))
        assert resolved == real_ws / "data.json"

    def test_shared_prefix_resolved(self, tmp_path, monkeypatch):
        """/mnt/shared → real path."""
        real_shared = tmp_path / "shared"
        real_shared.mkdir()
        monkeypatch.setenv(ENV_KEY_SHARED, str(real_shared))

        resolved = resolve_sandbox_path("/mnt/shared/some_file.txt")
        assert resolved == real_shared / "some_file.txt"

    def test_skills_prefix_resolved(self, tmp_path, monkeypatch):
        """/mnt/skills → real path."""
        real_skills = tmp_path / "skills"
        real_skills.mkdir()
        monkeypatch.setenv(ENV_KEY_SKILLS, str(real_skills))

        resolved = resolve_sandbox_path("/mnt/skills/ethoinsight/SKILL.md")
        assert resolved == real_skills / "ethoinsight/SKILL.md"


class TestValidateCatalogReadsResolvedPath:
    """End-to-end: validate_plan_results reads metric files via resolve_sandbox_path."""

    def test_validator_reads_via_resolved_path(self, tmp_path, monkeypatch):
        """Red anchor: before fix, validator read /mnt string → result_file_unreadable.

        After fix: resolve_sandbox_path resolves the virtual path so the real
        metric file is found and validated.
        """
        real_ws = tmp_path / "ws"
        real_ws.mkdir()

        # Create a real metric output file
        metric_data = {"metric": "center_time_ratio", "value": 0.5}
        (real_ws / "m_center_time.json").write_text(
            json.dumps(metric_data), encoding="utf-8"
        )

        monkeypatch.setenv(ENV_KEY_WORKSPACE, str(real_ws))

        # Plan with virtual output path (the bug scenario)
        plan = {
            "paradigm": "epm",
            "metrics": [
                {
                    "id": "center_time_ratio",
                    "output": "/mnt/user-data/workspace/m_center_time.json",
                    "output_unit": "ratio",
                    "subject_index": 0,
                }
            ],
        }

        violations = validate_plan_results(plan)

        # Before fix: result_file_unreadable. After fix: no file errors.
        file_errors = [v for v in violations if v["issue"] == "result_file_unreadable"]
        assert not file_errors, (
            f"result_file_unreadable should be empty after fix, got: {file_errors}"
        )

    def test_validator_still_reports_genuine_unreadable(self, tmp_path, monkeypatch):
        """When a file truly doesn't exist, result_file_unreadable is still raised."""
        real_ws = tmp_path / "ws"
        real_ws.mkdir()
        monkeypatch.setenv(ENV_KEY_WORKSPACE, str(real_ws))

        plan = {
            "paradigm": "epm",
            "metrics": [
                {
                    "id": "missing_metric",
                    "output": "/mnt/user-data/workspace/nonexistent_file.json",
                    "output_unit": "ratio",
                    "subject_index": 0,
                }
            ],
        }

        violations = validate_plan_results(plan)
        file_errors = [v for v in violations if v["issue"] == "result_file_unreadable"]
        assert len(file_errors) == 1

    def test_validator_no_env_still_works(self):
        """No DEERFLOW_PATH_* env → fail-safe passthrough (behavior unchanged from pre-fix)."""
        plan = {
            "paradigm": "epm",
            "metrics": [
                {
                    "id": "some_metric",
                    "output": "/mnt/user-data/workspace/nonexistent.json",
                    "output_unit": "ratio",
                    "subject_index": 0,
                }
            ],
        }
        violations = validate_plan_results(plan)
        # In fail-safe mode, the virtual path won't resolve → file not found → error
        file_errors = [v for v in violations if v["issue"] == "result_file_unreadable"]
        assert len(file_errors) == 1


class TestReadInputsJsonResolvesPaths:
    """D3: read_inputs_json resolves virtual paths in returned list."""

    def test_virtual_paths_resolved(self, tmp_path, monkeypatch):
        """Virtual /mnt/... paths in inputs JSON → real paths."""
        real_uploads = tmp_path / "real_uploads"
        real_uploads.mkdir()
        monkeypatch.setenv(ENV_KEY_UPLOADS, str(real_uploads))

        inputs_json = tmp_path / "raw_files.json"
        inputs_json.write_text(json.dumps([
            "/mnt/user-data/uploads/subject1.txt",
            "/mnt/user-data/uploads/subject2.txt",
        ]))

        paths = read_inputs_json(inputs_json)
        assert paths == [
            str(real_uploads / "subject1.txt"),
            str(real_uploads / "subject2.txt"),
        ]

    def test_real_paths_passthrough(self, tmp_path):
        """Already-real paths → returned as-is."""
        inputs_json = tmp_path / "raw_files.json"
        inputs_json.write_text(json.dumps([
            "/home/user/data/subject1.txt",
            "/home/user/data/subject2.txt",
        ]))

        paths = read_inputs_json(inputs_json)
        assert paths == [
            "/home/user/data/subject1.txt",
            "/home/user/data/subject2.txt",
        ]

    def test_mixed_paths_resolved(self, tmp_path, monkeypatch):
        """Mix of virtual and real paths → only virtual resolved."""
        real_uploads = tmp_path / "real_up"
        real_uploads.mkdir()
        monkeypatch.setenv(ENV_KEY_UPLOADS, str(real_uploads))

        inputs_json = tmp_path / "raw_files.json"
        inputs_json.write_text(json.dumps([
            "/mnt/user-data/uploads/subject1.txt",
            "/home/user/data/subject2.txt",
        ]))

        paths = read_inputs_json(inputs_json)
        assert paths == [
            str(real_uploads / "subject1.txt"),
            "/home/user/data/subject2.txt",
        ]


class TestReadGroupsJsonResolvesPaths:
    """D3: read_groups_json resolves subject paths in group lists."""

    def test_subject_paths_resolved(self, tmp_path, monkeypatch):
        """Virtual /mnt/... subject paths in groups → real paths."""
        real_uploads = tmp_path / "real_uploads"
        real_uploads.mkdir()
        monkeypatch.setenv(ENV_KEY_UPLOADS, str(real_uploads))

        groups_json = tmp_path / "groups.json"
        groups_json.write_text(json.dumps({
            "Control": [
                "/mnt/user-data/uploads/ctrl_1.txt",
                "/mnt/user-data/uploads/ctrl_2.txt",
            ],
            "Treatment": [
                "/mnt/user-data/uploads/trt_1.txt",
                "/mnt/user-data/uploads/trt_2.txt",
            ],
        }))

        groups = read_groups_json(groups_json)
        assert groups == {
            "Control": [
                str(real_uploads / "ctrl_1.txt"),
                str(real_uploads / "ctrl_2.txt"),
            ],
            "Treatment": [
                str(real_uploads / "trt_1.txt"),
                str(real_uploads / "trt_2.txt"),
            ],
        }

    def test_real_paths_passthrough(self, tmp_path):
        """Already-real subject paths → returned as-is."""
        groups_json = tmp_path / "groups.json"
        groups_json.write_text(json.dumps({
            "Control": ["/home/user/data/ctrl_1.txt"],
            "Treatment": ["/home/user/data/trt_1.txt"],
        }))

        groups = read_groups_json(groups_json)
        assert groups == {
            "Control": ["/home/user/data/ctrl_1.txt"],
            "Treatment": ["/home/user/data/trt_1.txt"],
        }
