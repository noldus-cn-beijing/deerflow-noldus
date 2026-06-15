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


# ==================== CODE-EXECUTOR: 字段名分裂回归（2026-06-04 核实发现） ====================


class TestCodeExecutorFieldNameDivergence:
    """契约：code-executor handoff 校验器认「字段唯一真相」的等价集。

    metrics_summary = 规范字段（Sprint 0 起），metrics / metrics_results = 历史等价字段
    （Sprint 0 前的旁路写入产物）。校验器承认所有三个字段的数据有效性以免误判残缺，
    但数据只存在于非规范字段时记 warning 暴露格式漂移。

    背景（2026-06-04 核实）：对 .deer-flow 下 56 个真实 handoff_code_executor.json
    做统计——27 个顶层用 `metrics`、1 个用 `metrics_results`，仅 24 个用
    `metrics_summary`。_check_code_executor_content 曾只查 metrics_summary，
    导致 28/56 个 status=completed、数据完整的成功分析被误判为「核心内容残缺」→
    触发 seal-resume 补轮 → 补轮也不会把数据搬进 metrics_summary →
    第二次校验仍失败 → executor 把成功的 task 标 FAILED（executor.py:1032）。

    这两个 fixture 是从真实 thread 9f77adcc（FST, completed）/ 7db437e7（FST,
    completed）落盘内容裁剪而来，结构原样保留。
    """

    # thread 9f77adcc-2a18 的真实结构裁剪：顶层字段叫 `metrics`（list），完整数据。
    _REAL_METRICS_FIELD_HANDOFF = {
        "status": "completed",
        "constitution_acknowledged": True,
        "paradigm": "fst",
        "metrics": [
            {
                "id": "immobility_time",
                "display_name_zh": "不动时间",
                "unit_zh": "秒",
                "values": {"treatment": 5.52, "control": 14.56},
            },
            {
                "id": "immobility_latency",
                "display_name_zh": "首次不动潜伏期",
                "unit_zh": "秒",
                "values": {"treatment": 60.0, "control": 30.0},
            },
        ],
        "statistics": {},
        "data_quality_warnings": [],
        "errors": [],
    }

    # thread 7db437e7 的真实结构裁剪：顶层字段叫 `metrics_results`（list）。
    _REAL_METRICS_RESULTS_HANDOFF = {
        "status": "completed",
        "paradigm": "fst",
        "metrics_results": [
            {
                "metric_id": "immobility_time",
                "display_name_zh": "不动时间",
                "values": [
                    {"group": "Treatment", "subject_index": 0, "value": 0.56},
                    {"group": "Control", "subject_index": 1, "value": 1.92},
                ],
            }
        ],
        "errors": [],
    }

    def test_real_metrics_field_handoff_should_pass(self, tmp_path: Path):
        """顶层 `metrics`（完整数据，status=completed）不应被判残缺。"""
        ws = _make_workspace(
            tmp_path,
            filename="handoff_code_executor.json",
            content=self._REAL_METRICS_FIELD_HANDOFF,
        )
        assert _validate("code-executor", ws) is None

    def test_real_metrics_results_handoff_should_pass(self, tmp_path: Path):
        """顶层 `metrics_results`（完整数据，status=completed）不应被判残缺。"""
        ws = _make_workspace(
            tmp_path,
            filename="handoff_code_executor.json",
            content=self._REAL_METRICS_RESULTS_HANDOFF,
        )
        assert _validate("code-executor", ws) is None

    def test_all_three_fields_empty_fails(self, tmp_path: Path):
        """三字段全空 → 判残缺（保留 Sprint 5.5 的核心保护）。"""
        ws = _make_workspace(
            tmp_path,
            filename="handoff_code_executor.json",
            content={"status": "completed"},
        )
        result = _validate("code-executor", ws)
        assert result is not None
        assert "metrics data is empty" in result

    def test_non_canonical_field_logs_warning(self, tmp_path: Path, caplog):
        """metrics（非规范字段）有数据 → 放行但记 warning 暴露格式漂移。"""
        import logging

        caplog.set_level(logging.WARNING)
        ws = _make_workspace(
            tmp_path,
            filename="handoff_code_executor.json",
            content=self._REAL_METRICS_FIELD_HANDOFF,
        )
        assert _validate("code-executor", ws) is None
        assert any(
            "non-canonical metrics field" in rec.message and "'metrics'" in rec.message
            for rec in caplog.records
        )


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


# ==================== status=failed 短路（2026-06-15 spec #3） ====================


class TestFailedStatusShortCircuit:
    """status=failed 的 handoff 合法地没有核心数据（失败的定义就是没产出）。

    背景（2026-06-15 EPM dogfood）：_validate_handoff_emitted 调 content check 只看
    核心字段非空，无视 status。code-executor 因环境错误正确 seal status=failed
    （failed 本就没指标数据可填），却被判 "incomplete → re-dispatch"，触发 lead
    规则 #7 自动重派死循环（gateway.log L288-361）。

    修复：status=failed 时短路放行（一处覆盖四个 subagent）。partial 不放宽。
    """

    def test_failed_code_executor_passes_empty_metrics(self, tmp_path: Path):
        """red 锚点：status=failed + metrics_summary 空 → 放行（修复前判 incomplete）。"""
        ws = _make_workspace(
            tmp_path,
            filename="handoff_code_executor.json",
            content={
                "status": "failed",
                "summary": "环境层错误，140 指标全失败",
                "metrics_summary": {},  # 失败本就该空
                "errors": ["FileNotFoundError ..."],
            },
        )
        assert _validate("code-executor", ws) is None

    def test_failed_code_executor_all_metrics_fields_empty_passes(self, tmp_path: Path):
        """status=failed + 三字段全空 → 仍放行（守护：failed 短路先于 content check）。"""
        ws = _make_workspace(
            tmp_path,
            filename="handoff_code_executor.json",
            content={"status": "failed"},  # metrics/metrics_results/metrics_summary 全缺
        )
        assert _validate("code-executor", ws) is None

    def test_failed_data_analyst_passes_empty_key_findings(self, tmp_path: Path):
        """四个 subagent 一致：failed data-analyst（key_findings 空）放行。"""
        ws = _make_workspace(
            tmp_path,
            filename="handoff_data_analyst.json",
            content={
                "status": "failed",
                "summary": "前置 code-executor 失败，无法判读",
                "key_findings": [],
            },
        )
        assert _validate("data-analyst", ws) is None

    def test_failed_report_writer_passes_empty_report_path(self, tmp_path: Path):
        """守护一致性：failed report-writer（report_path 空）放行。"""
        ws = _make_workspace(
            tmp_path,
            filename="handoff_report_writer.json",
            content={
                "status": "failed",
                "summary": "上游数据缺失，无法生成报告",
                "report_path": "",
            },
        )
        assert _validate("report-writer", ws) is None

    def test_failed_chart_maker_passes_empty_charts(self, tmp_path: Path):
        """守护一致性：failed chart-maker（chart_files + failed_charts 均空）放行。"""
        ws = _make_workspace(
            tmp_path,
            filename="handoff_chart_maker.json",
            content={"status": "failed", "chart_files": [], "failed_charts": []},
        )
        assert _validate("chart-maker", ws) is None

    def test_partial_status_still_requires_metrics(self, tmp_path: Path):
        """守护：partial 不在放宽范围——partial + 核心字段空仍判 incomplete。"""
        ws = _make_workspace(
            tmp_path,
            filename="handoff_code_executor.json",
            content={"status": "partial", "metrics_summary": {}},
        )
        result = _validate("code-executor", ws)
        assert result is not None
        assert "incomplete" in result

    def test_completed_status_still_requires_metrics(self, tmp_path: Path):
        """守护：放宽 failed 不误伤 completed——completed + 核心字段空仍判 incomplete。"""
        ws = _make_workspace(
            tmp_path,
            filename="handoff_code_executor.json",
            content={"status": "completed", "metrics_summary": {}},
        )
        result = _validate("code-executor", ws)
        assert result is not None
        assert "incomplete" in result

