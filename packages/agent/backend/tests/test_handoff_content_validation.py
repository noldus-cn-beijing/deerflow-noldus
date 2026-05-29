"""Sprint 5.5 — Handoff 内容非空校验单元测试。

测试 _validate_handoff_emitted 在文件存在后追加的「核心字段非空」检查。
空内容 handoff（调了 seal 但字段残缺如 key_findings=[]）应返回诊断 str，
而非 None（之前版本静默通过，导致下游产垃圾）。

复用 test_handoff_emission_validator.py 的 importlib 加载模式，
绕过 conftest.py 对 deerflow.subagents.executor 的 mock。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Load the REAL executor module source (bypassing conftest's sys.modules mock)
# ---------------------------------------------------------------------------
_EXECUTOR_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "subagents" / "executor.py"
)

_REAL_EXECUTOR: ModuleType | None = None


def _get_real_executor() -> ModuleType:
    global _REAL_EXECUTOR
    if _REAL_EXECUTOR is not None:
        return _REAL_EXECUTOR

    spec = importlib.util.spec_from_file_location(
        "deerflow.subagents.executor_real_55",
        _EXECUTOR_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None, f"Could not find executor.py at {_EXECUTOR_FILE}"
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        pytest.skip("Could not load real executor module")

    _REAL_EXECUTOR = module
    return _REAL_EXECUTOR


@pytest.fixture(autouse=True)
def _ensure_module_loaded():
    _get_real_executor()


def _validate(subagent_name: str, workspace_path: str | None) -> str | None:
    return _get_real_executor()._validate_handoff_emitted(subagent_name, workspace_path)


def _make_workspace(
    tmp_path: Path,
    *,
    filename: str | None = None,
    content: dict | list | str | None = None,
) -> str:
    """Create a temporary workspace dir, optionally writing a handoff file."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    if filename is not None:
        if content is None:
            content = {}
        if isinstance(content, (dict, list)):
            text = json.dumps(content)
        else:
            text = content
        (ws / filename).write_text(text, encoding="utf-8")
    return str(ws)


# ==================== DATA-ANALYST ====================


class TestDataAnalystContentValidation:
    """data-analyst: key_findings 非空 list"""

    def test_empty_key_findings_fails(self, tmp_path: Path):
        ws = _make_workspace(
            tmp_path,
            filename="handoff_data_analyst.json",
            content={"status": "completed", "key_findings": []},
        )
        result = _validate("data-analyst", ws)
        assert result is not None
        assert "incomplete" in result
        assert "key_findings" in result

    def test_nonempty_key_findings_passes(self, tmp_path: Path):
        ws = _make_workspace(
            tmp_path,
            filename="handoff_data_analyst.json",
            content={"status": "completed", "key_findings": ["Finding 1"]},
        )
        assert _validate("data-analyst", ws) is None


# ==================== CODE-EXECUTOR ====================


class TestCodeExecutorContentValidation:
    """code-executor: metrics_summary 非空 dict"""

    def test_empty_metrics_summary_fails(self, tmp_path: Path):
        ws = _make_workspace(
            tmp_path,
            filename="handoff_code_executor.json",
            content={"status": "completed", "metrics_summary": {}},
        )
        result = _validate("code-executor", ws)
        assert result is not None
        assert "metrics_summary" in result

    def test_nonempty_metrics_summary_passes(self, tmp_path: Path):
        ws = _make_workspace(
            tmp_path,
            filename="handoff_code_executor.json",
            content={"status": "completed", "metrics_summary": {"group1": {"m1": {}}}},
        )
        assert _validate("code-executor", ws) is None


# ==================== CHART-MAKER ====================


class TestChartMakerContentValidation:
    """chart-maker: chart_files 非空 list 或 failed_charts 有说明"""

    def test_both_empty_fails(self, tmp_path: Path):
        ws = _make_workspace(
            tmp_path,
            filename="handoff_chart_maker.json",
            content={"status": "completed", "chart_files": [], "failed_charts": []},
        )
        result = _validate("chart-maker", ws)
        assert result is not None
        assert "chart_files" in result or "failed_charts" in result

    def test_failed_charts_only_passes(self, tmp_path: Path):
        """chart_files=[] 但 failed_charts 有说明 → 合法（图表确实没生成但记录了原因）"""
        ws = _make_workspace(
            tmp_path,
            filename="handoff_chart_maker.json",
            content={
                "status": "partial",
                "chart_files": [],
                "failed_charts": [{"chart_id": "heatmap", "reason": "data too sparse"}],
            },
        )
        assert _validate("chart-maker", ws) is None

    def test_chart_files_nonempty_passes(self, tmp_path: Path):
        ws = _make_workspace(
            tmp_path,
            filename="handoff_chart_maker.json",
            content={
                "status": "completed",
                "chart_files": ["/mnt/user-data/outputs/chart.png"],
                "failed_charts": [],
            },
        )
        assert _validate("chart-maker", ws) is None


# ==================== REPORT-WRITER ====================


class TestReportWriterContentValidation:
    """report-writer: report_path 非空 str"""

    def test_empty_report_path_fails(self, tmp_path: Path):
        ws = _make_workspace(
            tmp_path,
            filename="handoff_report_writer.json",
            content={"status": "completed", "report_path": ""},
        )
        result = _validate("report-writer", ws)
        assert result is not None
        assert "report_path" in result

    def test_nonempty_report_path_passes(self, tmp_path: Path):
        ws = _make_workspace(
            tmp_path,
            filename="handoff_report_writer.json",
            content={"status": "completed", "report_path": "/mnt/user-data/outputs/report.md"},
        )
        assert _validate("report-writer", ws) is None


# ==================== FAIL-OPEN ====================


class TestFailOpenOnBadJson:
    """读/解析失败 → fail-open（不阻断正常 task）"""

    def test_unparseable_json_fails_open(self, tmp_path: Path):
        ws = _make_workspace(
            tmp_path,
            filename="handoff_data_analyst.json",
            content="{this is not valid json!!!",
        )
        assert _validate("data-analyst", ws) is None

    def test_non_dict_json_fails(self, tmp_path: Path):
        """handoff 是 [1,2,3]（合法 JSON 但非 dict）→ str "not a JSON object" """
        ws = _make_workspace(
            tmp_path,
            filename="handoff_data_analyst.json",
            content=[1, 2, 3],
        )
        result = _validate("data-analyst", ws)
        assert result is not None
        assert "not a JSON object" in result


# ==================== REGRESSION: 5.7 行为不受影响 ====================


class TestExistingBehaviorPreserved:
    """5.5 内容检查不影响 5.7 的存在性检查"""

    def test_missing_file_still_terminated_without_emitting(self, tmp_path: Path):
        """文件不存在 → 仍返回 5.7 原诊断 "terminated without emitting" """
        ws = _make_workspace(tmp_path)  # no handoff file
        result = _validate("data-analyst", ws)
        assert result is not None
        assert "terminated without emitting" in result

    def test_general_purpose_no_content_check(self, tmp_path: Path):
        """general-purpose（白名单外）→ None（不进内容检查）"""
        ws = _make_workspace(tmp_path)
        assert _validate("general-purpose", ws) is None
