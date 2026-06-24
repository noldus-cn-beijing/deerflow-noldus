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


# ---------------------------------------------------------------------------
# Spec 2026-06-24-run-chart-plan-permissionerror — F2 fail-safe 可观测
# (T6/T7：收到 /mnt 虚拟路径却无 env 无 workspace_base 兜底 → 原样返回时记 WARNING，
#  给「虚拟路径漏解析」留响亮 grep 锚点；真实路径不 warn)
# ---------------------------------------------------------------------------


class TestFailSafeWarning:
    """F2：fail-safe 原样返回 /mnt 路径时记 WARNING（原 debug 在生产日志级别不可见 →
    静默退化无痕，本次 chart 渲染崩塌排查被「疑似伪造」误导正因此处无响亮信号）。
    """

    def _clear_env(self, monkeypatch):
        for k in list(os.environ):
            if k.startswith("DEERFLOW_PATH_"):
                monkeypatch.delenv(k)

    def test_f2_t6_unresolved_virtual_path_warns(self, monkeypatch, caplog):
        """T6：/mnt 路径无 env 无 workspace_base → 原样返回 + 记 WARNING（含原始路径）。"""
        import logging

        self._clear_env(monkeypatch)
        virtual = "/mnt/user-data/workspace/x.json"

        with caplog.at_level(logging.WARNING, logger="ethoinsight.scripts._cli"):
            result = resolve_sandbox_path(virtual)

        assert result == Path(virtual)  # fail-safe: still returned as-is (no crash)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "F2 red: unresolved /mnt virtual path should emit a WARNING"
        # The original path string must appear in the warning (grep anchor).
        assert any(virtual in r.getMessage() for r in warnings), [
            r.getMessage() for r in warnings
        ]

    def test_f2_t7_real_path_does_not_warn(self, caplog):
        """T7：真实路径（非 /mnt）→ 原样返回，不 warn（合法路径，无信号噪音）。"""
        import logging

        with caplog.at_level(logging.WARNING, logger="ethoinsight.scripts._cli"):
            result = resolve_sandbox_path("/real/path/x.json")

        assert result == Path("/real/path/x.json")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warnings, f"F2 over-warns on real path: {[r.getMessage() for r in warnings]}"
