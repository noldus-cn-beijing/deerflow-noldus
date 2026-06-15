"""Sprint 3 单元测试 — seal_data_analyst_handoff 透传 parameter_audit_findings。

验证 seal 能正确写入含 parameter_audit_findings 的 handoff JSON，
拒绝非法 mismatch_kind，以及不传 findings 时默认为空列表。

直接调用 _seal_handoff (和现有 test_seal_handoff_tools.py 同风格)，
避免 LangChain tool .invoke() 的 runtime 注入问题。

测试清单（对应 spec §2.8）：
1. test_seal_writes_findings_to_handoff — seal 透传 findings 到 JSON 文件
2. test_seal_validates_finding_schema — 非法 mismatch_kind → Pydantic 拒绝
3. test_seal_handles_empty_findings — 不传 findings → 默认 []
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deerflow.subagents.handoff_schemas import DataAnalystHandoff
from deerflow.tools.builtins.seal_handoff_tools import _seal_handoff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime(workspace: Path) -> MagicMock:
    """Build a mock Runtime with workspace_path in state."""
    runtime = MagicMock()
    runtime.state = {
        "thread_data": {
            "workspace_path": str(workspace),
        },
    }
    return runtime


def _make_workspace(tmp_path: Path) -> Path:
    """Create a workspace directory with experiment-context.json."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "experiment-context.json").write_text(
        json.dumps({"analysis_config_id": "deadbeef12345678"}), encoding="utf-8"
    )
    return ws


def _make_finding_dict(**overrides) -> dict:
    """Minimal valid ParameterAuditFinding dict."""
    base = {
        "parameter": "velocity_threshold",
        "metric": "immobility_time",
        "severity": "warning",
        "used_value": 30.0,
        "observed_distribution": {"median": 5.0, "p90": 12.0, "max": 25.0, "n_subjects": 12},
        "mismatch_kind": "threshold_too_high",
        "suggestion": "当前阈值偏高，建议参考 paradigm md 调参指南段",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. seal writes findings to handoff
# ---------------------------------------------------------------------------
class TestSealWritesFindings:
    def test_seal_writes_findings_to_handoff(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(ws)
        findings = [
            _make_finding_dict(),
            _make_finding_dict(parameter="total_entry_threshold", metric="total_entry_count", used_value=8, mismatch_kind="threshold_too_low"),
        ]

        result = _seal_handoff(
            DataAnalystHandoff,
            "handoff_data_analyst.json",
            {
                "status": "completed",
                "key_findings": ["发现参数不匹配"],
                "parameter_audit_findings": findings,
                "gate_signals": {
                    "parameter_audit_findings_count": 2,
                    "parameter_audit_critical_count": 0,
                },
            },
            runtime,
        )

        assert result.startswith("OK: sealed handoff_data_analyst.json")

        handoff_path = ws / "handoff_data_analyst.json"
        assert handoff_path.exists()
        data = json.loads(handoff_path.read_text(encoding="utf-8"))

        # Validate via Pydantic
        handoff = DataAnalystHandoff(**data)
        assert handoff.status == "completed"
        assert len(handoff.parameter_audit_findings) == 2
        assert handoff.parameter_audit_findings[0].parameter == "velocity_threshold"
        assert handoff.parameter_audit_findings[0].mismatch_kind == "threshold_too_high"
        assert handoff.parameter_audit_findings[1].parameter == "total_entry_threshold"
        assert handoff.parameter_audit_findings[1].mismatch_kind == "threshold_too_low"

        # Gate signals also preserved
        assert handoff.gate_signals is not None
        assert handoff.gate_signals.parameter_audit_findings_count == 2

    def test_seal_writes_manifest_with_findings(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(ws)

        _seal_handoff(
            DataAnalystHandoff,
            "handoff_data_analyst.json",
            {
                "status": "completed",
                "key_findings": ["finding"],
                "parameter_audit_findings": [_make_finding_dict()],
            },
            runtime,
        )

        manifest_path = ws / ".lineage" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "handoff_data_analyst.json" in manifest
        assert "sha256" in manifest["handoff_data_analyst.json"]


# ---------------------------------------------------------------------------
# 2. seal validates finding schema
# ---------------------------------------------------------------------------
class TestSealValidatesSchema:
    def test_seal_validates_finding_schema(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(ws)

        with pytest.raises(ValueError, match="schema validation failed"):
            _seal_handoff(
                DataAnalystHandoff,
                "handoff_data_analyst.json",
                {
                    "status": "completed",
                    "parameter_audit_findings": [
                        _make_finding_dict(mismatch_kind="INVALID_KIND"),
                    ],
                },
                runtime,
            )

    def test_seal_rejects_empty_parameter(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(ws)

        with pytest.raises(ValueError, match="schema validation failed"):
            _seal_handoff(
                DataAnalystHandoff,
                "handoff_data_analyst.json",
                {
                    "status": "completed",
                    "parameter_audit_findings": [
                        _make_finding_dict(parameter=""),
                    ],
                },
                runtime,
            )


# ---------------------------------------------------------------------------
# 3. seal handles empty findings
# ---------------------------------------------------------------------------
class TestSealHandlesEmptyFindings:
    def test_seal_handles_empty_findings(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(ws)

        result = _seal_handoff(
            DataAnalystHandoff,
            "handoff_data_analyst.json",
            {"status": "completed", "key_findings": ["finding"]},
            runtime,
        )

        assert result.startswith("OK: sealed handoff_data_analyst.json")
        handoff_path = ws / "handoff_data_analyst.json"
        data = json.loads(handoff_path.read_text(encoding="utf-8"))
        handoff = DataAnalystHandoff(**data)
        assert handoff.parameter_audit_findings == []

    def test_seal_handles_explicit_empty_findings(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(ws)

        result = _seal_handoff(
            DataAnalystHandoff,
            "handoff_data_analyst.json",
            {"status": "completed", "key_findings": ["finding"], "parameter_audit_findings": []},
            runtime,
        )

        assert result.startswith("OK: sealed handoff_data_analyst.json")
        handoff_path = ws / "handoff_data_analyst.json"
        data = json.loads(handoff_path.read_text(encoding="utf-8"))
        handoff = DataAnalystHandoff(**data)
        assert handoff.parameter_audit_findings == []
