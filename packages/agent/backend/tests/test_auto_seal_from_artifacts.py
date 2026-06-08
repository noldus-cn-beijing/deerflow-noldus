"""Spec C — Integration tests for _attempt_auto_seal_from_artifacts.

Tests the deterministic auto-seal fallback: when seal-resume fails but the
subagent has produced output files (report.md / plot_*.png), the harness
constructs a valid handoff JSON from those artifacts without relying on LLM.

Because conftest.py pre-mocks deerflow.subagents.executor, we load the real
module via importlib (same pattern as test_seal_resume / test_executor_handoff_emission).
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Load the real executor module (same pattern as test_executor_handoff_emission)
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
        "deerflow.subagents.executor_real_auto_seal",
        _EXECUTOR_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    _REAL_EXECUTOR = module
    return module


@pytest.fixture(autouse=True)
def _load_module():
    _get_real_executor()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_thread(tmp_path: Path):
    """Create thread directory structure: user-data/{workspace,outputs}.

    Returns (workspace_path_str, outputs_path).
    """
    user_data = tmp_path / "user-data"
    ws = user_data / "workspace"
    out = user_data / "outputs"
    ws.mkdir(parents=True)
    out.mkdir(parents=True)
    return str(ws), out


def _auto_seal(subagent_name: str, workspace_path: str | None) -> bool:
    """Shortcut to call _attempt_auto_seal_from_artifacts on the real module."""
    mod = _get_real_executor()
    return mod._attempt_auto_seal_from_artifacts(subagent_name, workspace_path)


# ===================================================================
# Test: report-writer auto-seal (core path)
# ===================================================================

class TestAutoSealReportWriter:
    """report-writer 有产出 (outputs/report.md) 无 handoff → auto-seal 成功。"""

    def test_report_exists_no_handoff_auto_seals(self, tmp_path: Path):
        """report.md 存在、无 handoff → auto-seal 成功，产物过内容校验。"""
        mod = _get_real_executor()
        ws, out = _mk_thread(tmp_path)
        # 写 report.md（模拟 subagent 已完成产出）
        (out / "report.md").write_text(
            "# 实验概况\n\n内容...\n\n## 结果\n\n统计分析结果...\n\n## 讨论\n\n讨论内容...\n",
            encoding="utf-8",
        )
        # experiment-context.json（_seal_handoff_to_workspace 需要 analysis_config_id）
        (Path(ws) / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test-config-123"}),
            encoding="utf-8",
        )

        ok = _auto_seal("report-writer", ws)
        assert ok is True

        # handoff 文件已写入
        h = Path(ws) / "handoff_report_writer.json"
        assert h.exists()
        data = json.loads(h.read_text(encoding="utf-8"))
        assert data["status"] == "completed"
        assert data["report_path"] == "/mnt/user-data/outputs/report.md"
        assert len(data["sections_written"]) >= 2  # 从 markdown 标题解析
        assert "实验概况" in data["sections_written"]
        assert "结果" in data["sections_written"]
        assert any("auto-seal" in e for e in data["errors"])  # auto-seal 标记
        # analysis_config_id 已注入（由 _seal_handoff_to_workspace）
        assert data["analysis_config_id"] == "test-config-123"
        # 文件权限 644（Spec1 教训）
        assert stat.S_IMODE(os.stat(h).st_mode) == 0o644

        # 集成断言：auto-seal 产物能过 _validate_handoff_emitted（含内容非空检查）
        validation_error = mod._validate_handoff_emitted("report-writer", ws)
        assert validation_error is None, f"auto-sealed handoff should validate, got: {validation_error}"

    def test_no_report_does_not_auto_seal(self, tmp_path: Path):
        """outputs/ 空（无 report.md）→ 不兜底，返回 False。"""
        ws, _out = _mk_thread(tmp_path)  # outputs 空
        ok = _auto_seal("report-writer", ws)
        assert ok is False  # 没产出 → 不兜底 → 外层走 FAILED

    def test_existing_handoff_not_overwritten(self, tmp_path: Path):
        """workspace 已有非空 handoff → 不覆盖，返回 False。"""
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# X\n", encoding="utf-8")
        # 已有 handoff（模拟 seal-resume 成功后又被 auto-seal 检查到——不该到这，保险）
        (Path(ws) / "handoff_report_writer.json").write_text(
            '{"status":"completed","report_path":"/x","sections_written":["orig"]}',
            encoding="utf-8",
        )
        ok = _auto_seal("report-writer", ws)
        assert ok is False  # 已有 handoff → 不覆盖

    def test_empty_report_does_not_auto_seal(self, tmp_path: Path):
        """report.md 为空文件 → 不兜底（无实质产出）。"""
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("", encoding="utf-8")  # 空文件
        ok = _auto_seal("report-writer", ws)
        assert ok is False

    def test_sections_parsed_from_headings(self, tmp_path: Path):
        """验证从 markdown 标题解析 sections_written 的准确性。"""
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text(
            "# 方法\n方法内容\n## 被试\n被试信息\n### 分组\n分组信息\n# 结果\n结果内容\n",
            encoding="utf-8",
        )
        (Path(ws) / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test"}),
            encoding="utf-8",
        )

        ok = _auto_seal("report-writer", ws)
        assert ok is True
        data = json.loads((Path(ws) / "handoff_report_writer.json").read_text(encoding="utf-8"))
        # 各级标题都被提取（去除 # 前缀和空格）
        assert "方法" in data["sections_written"]
        assert "被试" in data["sections_written"]
        assert "分组" in data["sections_written"]
        assert "结果" in data["sections_written"]
        assert len(data["sections_written"]) == 4


# ===================================================================
# Test: chart-maker auto-seal
# ===================================================================

class TestAutoSealChartMaker:
    """chart-maker 有产出 (outputs/plot_*.png) 无 handoff → auto-seal 成功。"""

    def test_charts_exist_no_handoff_auto_seals(self, tmp_path: Path):
        """plot_*.png 存在 → auto-seal 成功，chart_files 带正确虚拟路径前缀。"""
        mod = _get_real_executor()
        ws, out = _mk_thread(tmp_path)
        # 模拟 chart-maker 已产出图表
        (out / "plot_trajectory.png").write_text("fake png", encoding="utf-8")
        (out / "plot_heatmap.png").write_text("fake png", encoding="utf-8")
        # code-executor handoff（提供 paradigm）
        (Path(ws) / "handoff_code_executor.json").write_text(
            json.dumps({"paradigm": "epm", "metrics_summary": {"epm": {}}}),
            encoding="utf-8",
        )
        (Path(ws) / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test-config"}),
            encoding="utf-8",
        )

        ok = _auto_seal("chart-maker", ws)
        assert ok is True

        h = Path(ws) / "handoff_chart_maker.json"
        assert h.exists()
        data = json.loads(h.read_text(encoding="utf-8"))
        assert data["status"] == "completed"
        assert data["paradigm"] == "epm"
        assert "auto-seal" in data["summary"]
        # chart_files 有正确的 /mnt/user-data/outputs/ 前缀（schema 校验要求）
        assert len(data["chart_files"]) == 2
        for cf in data["chart_files"]:
            assert cf.startswith("/mnt/user-data/outputs/")
        chart_names = [Path(cf).name for cf in data["chart_files"]]
        assert "plot_trajectory.png" in chart_names
        assert "plot_heatmap.png" in chart_names
        # 文件权限 644
        assert stat.S_IMODE(os.stat(h).st_mode) == 0o644

        # 集成断言：能过 _validate_handoff_emitted
        validation_error = mod._validate_handoff_emitted("chart-maker", ws)
        assert validation_error is None, f"auto-sealed chart-maker handoff should validate, got: {validation_error}"

    def test_no_charts_does_not_auto_seal(self, tmp_path: Path):
        """outputs/ 无 plot_*.png → 不兜底。"""
        ws, _out = _mk_thread(tmp_path)  # outputs 空
        ok = _auto_seal("chart-maker", ws)
        assert ok is False

    def test_paradigm_fallback_empty_string(self, tmp_path: Path):
        """无 code_executor handoff → paradigm 回退空字符串。"""
        ws, out = _mk_thread(tmp_path)
        (out / "plot_test.png").write_text("fake", encoding="utf-8")
        (Path(ws) / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test"}),
            encoding="utf-8",
        )
        # 无 handoff_code_executor.json

        ok = _auto_seal("chart-maker", ws)
        assert ok is True
        data = json.loads((Path(ws) / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        assert data["paradigm"] == ""  # 回退空字符串（schema 允许）


# ===================================================================
# Test: scope — data-analyst / code-executor NEVER auto-sealed
# ===================================================================

class TestAutoSealScope:
    """认知产物 subagent（data-analyst / code-executor）永不 auto-seal。"""

    def test_data_analyst_never_auto_sealed(self, tmp_path: Path):
        """data-analyst 即便有报告文件也绝不 auto-seal（判读是认知产物）。"""
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# X\n", encoding="utf-8")
        ok = _auto_seal("data-analyst", ws)
        assert ok is False

    def test_code_executor_never_auto_sealed(self, tmp_path: Path):
        """code-executor 绝不 auto-seal（指标数值是认知产物）。"""
        ws, _out = _mk_thread(tmp_path)
        ok = _auto_seal("code-executor", ws)
        assert ok is False

    def test_unknown_subagent_never_auto_sealed(self, tmp_path: Path):
        """未在白名单的 subagent → 不兜底。"""
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# X\n", encoding="utf-8")
        ok = _auto_seal("general-purpose", ws)
        assert ok is False

    def test_no_workspace_fails_safe(self, tmp_path: Path):
        """workspace_path=None → 不抛异常，返回 False。"""
        assert _auto_seal("report-writer", None) is False


# ===================================================================
# Test: robustness — never raise
# ===================================================================

class TestAutoSealRobustness:
    """_attempt_auto_seal_from_artifacts 绝不抛异常（调用点在 executor try/except 内）。"""

    def test_never_raises_on_garbage(self):
        """垃圾路径 → 返回 False，不抛。"""
        try:
            result = _auto_seal("report-writer", "/nonexistent/\x00bad")
            assert result is False
        except Exception as e:
            pytest.fail(f"must not raise, got {e}")

    def test_never_raises_on_empty_string_workspace(self):
        """空字符串 workspace → 返回 False，不抛。"""
        try:
            result = _auto_seal("report-writer", "")
            assert result is False
        except Exception as e:
            pytest.fail(f"must not raise, got {e}")

    def test_never_raises_on_non_existent_outputs(self, tmp_path: Path):
        """workspace 存在但 outputs 目录不存在 → 返回 False，不抛。"""
        ws_path = tmp_path / "workspace_only"
        ws_path.mkdir(parents=True)
        # 没有 outputs 目录
        try:
            result = _auto_seal("report-writer", str(ws_path))
            assert result is False
        except Exception as e:
            pytest.fail(f"must not raise, got {e}")

    def test_never_raises_when_report_unreadable(self, tmp_path: Path):
        """report.md 存在但不可读 → 返回 False，不抛。"""
        ws, out = _mk_thread(tmp_path)
        report = out / "report.md"
        report.write_text("# X\n", encoding="utf-8")
        # 模拟不可读：删除文件后指向不存在的路径
        # 实际上我们测试 outputs 目录存在但 glob 返回空
        # 更实际的场景：workspace 路径下有 outputs 但 outputs 不是目录
        bad_tmp = tmp_path / "bad_thread"
        bad_tmp.mkdir()
        bad_ws = bad_tmp / "workspace"
        bad_ws.mkdir(parents=True)
        # outputs 是文件而非目录
        (bad_tmp / "outputs").write_text("not a dir")
        try:
            result = _auto_seal("report-writer", str(bad_ws))
            assert result is False
        except Exception as e:
            pytest.fail(f"must not raise, got {e}")


# ===================================================================
# Test: auto-seal 不破坏 _AUTO_SEALABLE 白名单完整性
# ===================================================================

class TestAutoSealableMap:
    """验证 _AUTO_SEALABLE 只包含 report-writer 和 chart-maker，不含认知产物 subagent。"""

    def test_auto_sealable_only_contains_mechanical_subagents(self):
        mod = _get_real_executor()
        sealable = mod._AUTO_SEALABLE
        assert "report-writer" in sealable
        assert "chart-maker" in sealable
        assert "code-executor" not in sealable, "code-executor handoff is cognitive product"
        assert "data-analyst" not in sealable, "data-analyst handoff is cognitive product"
        assert "general-purpose" not in sealable
        assert len(sealable) == 2  # 只有两个，防止意外扩大
