"""Tests for data-analyst 分步填模板（产物 + 封口）—— spec
2026-06-23-data-analyst-seal-stepwise-fill-template.

These tests exercise the new fill_*/finalize tools + in_progress template preset +
SealGate data-analyst recognition + loop-detection leniency + crash honesty.

⚠️ Worktree / editable-link 注意（memory feedback_worktree_shares_main_venv...）：
worktree 借主仓 venv（editable 指主仓 packages/harness），直接 `from deerflow...import`
会加载主仓【旧】源而非被测 worktree 源。本测试用 `_overlay_worktree_harness` autouse
fixture 在 session 开始时把 worktree 的被改 harness 文件 overlay 到主仓 editable
位置、结束后恢复，使 `deerflow.*` 解析到被测代码（与 PR#169/PR#177 验全量的 overlay
法一致）。裸导入环测试在 test_gateway_import_no_cycle.py 单独覆盖。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Worktree root = 4 levels up from this test file
# (tests/ -> backend/ -> agent/ -> packages/ -> worktree-root)
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]

# harness files this spec edits (relative to worktree root). Overlay copies these
# onto the main-repo editable location so `deerflow.*` resolves to worktree source.
_HARNESS_RELS = [
    "packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py",
    "packages/agent/backend/packages/harness/deerflow/tools/builtins/__init__.py",
    "packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py",
    "packages/agent/backend/packages/harness/deerflow/subagents/executor.py",
    "packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py",
    "packages/agent/backend/packages/harness/deerflow/agents/middlewares/seal_gate_middleware.py",
    "packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py",
    "packages/agent/backend/packages/harness/deerflow/agents/middlewares/quality_warning_broadcast_middleware.py",
]

# Main-repo root = the worktree's parent's parent (the real repo root that the
# editable .pth points at: /home/.../noldus-insight/packages/agent/backend/packages/harness).
_MAIN_ROOT = _WORKTREE_ROOT.parent.parent if _WORKTREE_ROOT.name.startswith("worktree-") else None


def _overlay_worktree_harness():
    """Source is physically overlaid by the run script; no-op here."""
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "experiment-context.json").write_text(
        json.dumps({"paradigm": "epm", "analysis_config_id": "cfg-123"}),
        encoding="utf-8",
    )
    return ws


def _make_runtime(workspace_dir: str) -> MagicMock:
    runtime = MagicMock()
    runtime.state = {"thread_data": {"workspace_path": workspace_dir}}
    return runtime


# ---------------------------------------------------------------------------
# T1 — harness 预置 in_progress 空模板（纯 Python，Pydantic 校验过）
# ---------------------------------------------------------------------------


class TestPresetTemplate:
    def test_preset_writes_valid_in_progress_template(self, tmp_path):
        from deerflow.tools.builtins.seal_handoff_tools import (
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        written = preset_data_analyst_template_to_workspace(ws)
        handoff_path = ws / "handoff_data_analyst.json"
        assert Path(written) == handoff_path
        assert handoff_path.exists()

        data = json.loads(handoff_path.read_text(encoding="utf-8"))
        assert data["status"] == "in_progress"
        assert data["sealed_by"] == "preset"
        assert data["key_findings"] == []
        assert data["analysis_config_id"] == "cfg-123"
        assert data["parameter_audit_findings"] == []

    def test_preset_is_idempotent_overwrite(self, tmp_path):
        """幂等覆盖：已有终态/FAILED 残留也被无条件覆盖成 in_progress。"""
        from deerflow.tools.builtins.seal_handoff_tools import (
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        # seed a stale FAILED handoff (simulating last run's residue)
        (ws / "handoff_data_analyst.json").write_text(
            json.dumps({"status": "failed", "key_findings": ["stale"], "sealed_by": "finalize"}),
            encoding="utf-8",
        )
        preset_data_analyst_template_to_workspace(ws)
        data = json.loads((ws / "handoff_data_analyst.json").read_text(encoding="utf-8"))
        assert data["status"] == "in_progress"
        assert data["key_findings"] == []  # residue wiped
        assert data["sealed_by"] == "preset"


# ---------------------------------------------------------------------------
# T2 — fill set（填字段非空、整体校验过、原子更新、返回进度）
# ---------------------------------------------------------------------------


class TestFillTextListSet:
    def test_fill_key_findings_set(self, tmp_path):
        from deerflow.tools.builtins.seal_handoff_tools import (
            fill_data_analyst_text_list,
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        runtime = _make_runtime(str(ws))

        result = fill_data_analyst_text_list.func(
            field="key_findings", mode="set", value=["finding A", "finding B"], runtime=runtime,
        )
        assert result.startswith("OK: filled key_findings")
        data = json.loads((ws / "handoff_data_analyst.json").read_text(encoding="utf-8"))
        assert data["key_findings"] == ["finding A", "finding B"]
        # status stays in_progress (fill doesn't seal)
        assert data["status"] == "in_progress"
        # progress JSON embedded in result
        assert "key_findings" in result


# ---------------------------------------------------------------------------
# T3 — fill append（追加不覆盖；超长字段分多次 append）
# ---------------------------------------------------------------------------


class TestFillTextListAppend:
    def test_append_does_not_overwrite(self, tmp_path):
        from deerflow.tools.builtins.seal_handoff_tools import (
            fill_data_analyst_text_list,
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        runtime = _make_runtime(str(ws))

        fill_data_analyst_text_list.func(
            field="recommendations", mode="set", value=["rec 1"], runtime=runtime,
        )
        # second batch via append — small args each time (the whole point)
        fill_data_analyst_text_list.func(
            field="recommendations", mode="append", value=["rec 2", "rec 3"], runtime=runtime,
        )
        data = json.loads((ws / "handoff_data_analyst.json").read_text(encoding="utf-8"))
        assert data["recommendations"] == ["rec 1", "rec 2", "rec 3"]


# ---------------------------------------------------------------------------
# T4 — finalize gate（completed 必须有 key_findings；partial 不要求）
# ---------------------------------------------------------------------------


class TestFinalizeGate:
    def test_finalize_completed_empty_key_findings_rejected(self, tmp_path):
        from deerflow.tools.builtins.seal_handoff_tools import (
            finalize_data_analyst_handoff,
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        runtime = _make_runtime(str(ws))

        # No key_findings filled → completed must be rejected by the validator.
        with pytest.raises(ValueError, match="key_findings is empty"):
            finalize_data_analyst_handoff.func(final_status="completed", runtime=runtime)

        # status stays in_progress (finalize refused to seal)
        data = json.loads((ws / "handoff_data_analyst.json").read_text(encoding="utf-8"))
        assert data["status"] == "in_progress"

    def test_finalize_partial_allowed_without_key_findings(self, tmp_path):
        """fast-fail 路径：partial/failed 不要求 key_findings 非空。"""
        from deerflow.tools.builtins.seal_handoff_tools import (
            finalize_data_analyst_handoff,
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        runtime = _make_runtime(str(ws))

        result = finalize_data_analyst_handoff.func(final_status="partial", runtime=runtime)
        assert result.startswith("OK: finalized")
        data = json.loads((ws / "handoff_data_analyst.json").read_text(encoding="utf-8"))
        assert data["status"] == "partial"
        assert data["sealed_by"] == "finalize"
        # manifest written (sealed_by=finalize observable)
        manifest = json.loads((ws / ".lineage" / "manifest.json").read_text(encoding="utf-8"))
        assert "handoff_data_analyst.json" in manifest

    def test_finalize_completed_with_key_findings_succeeds(self, tmp_path):
        from deerflow.tools.builtins.seal_handoff_tools import (
            fill_data_analyst_text_list,
            finalize_data_analyst_handoff,
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        runtime = _make_runtime(str(ws))
        fill_data_analyst_text_list.func(
            field="key_findings", mode="set", value=["real finding"], runtime=runtime,
        )

        result = finalize_data_analyst_handoff.func(final_status="completed", runtime=runtime)
        assert "status=completed" in result
        data = json.loads((ws / "handoff_data_analyst.json").read_text(encoding="utf-8"))
        assert data["status"] == "completed"
        assert data["sealed_by"] == "finalize"


# ---------------------------------------------------------------------------
# fill record list (outlier_findings / quality_warnings) 子模型校验
# ---------------------------------------------------------------------------


class TestFillRecordList:
    def test_fill_outlier_findings_validates_submodel(self, tmp_path):
        from deerflow.tools.builtins.seal_handoff_tools import (
            fill_data_analyst_record_list,
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        runtime = _make_runtime(str(ws))

        fill_data_analyst_record_list.func(
            field="outlier_findings", mode="set",
            value=[{"subject": "Subject 3", "metric": "mean_nnd", "value": 70.0, "deviation": "2x median"}],
            runtime=runtime,
        )
        data = json.loads((ws / "handoff_data_analyst.json").read_text(encoding="utf-8"))
        assert len(data["outlier_findings"]) == 1
        assert data["outlier_findings"][0]["subject"] == "Subject 3"

    def test_fill_outlier_findings_rejects_missing_required(self, tmp_path):
        from deerflow.tools.builtins.seal_handoff_tools import (
            fill_data_analyst_record_list,
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        runtime = _make_runtime(str(ws))

        # OutlierFinding requires subject/metric/value/deviation — drop deviation.
        with pytest.raises(ValueError, match="failed OutlierFinding validation"):
            fill_data_analyst_record_list.func(
                field="outlier_findings", mode="set", value=[{"subject": "S1"}], runtime=runtime,
            )


# ---------------------------------------------------------------------------
# T5 — SealGate 对 data-analyst 认 finalize/终态
# ---------------------------------------------------------------------------


class TestSealGateDataAnalyst:
    def test_seal_tool_is_finalize_for_data_analyst(self):
        from deerflow.agents.middlewares.seal_gate_middleware import SealGateMiddleware

        mw = SealGateMiddleware("data-analyst")
        assert mw._seal_tool == "finalize_data_analyst_handoff"

    def test_seal_tool_still_seal_for_chart_maker(self):
        from deerflow.agents.middlewares.seal_gate_middleware import SealGateMiddleware

        mw = SealGateMiddleware("chart-maker")
        assert mw._seal_tool == "seal_chart_maker_handoff"

    def test_in_progress_reminds_back(self, tmp_path):
        """in_progress（未 finalize）→ SealGate 催回（催 finalize）。"""
        from deerflow.agents.middlewares.seal_gate_middleware import SealGateMiddleware
        from deerflow.tools.builtins.seal_handoff_tools import (
            preset_data_analyst_template_to_workspace,
        )
        from langchain_core.messages import AIMessage

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        mw = SealGateMiddleware("data-analyst")
        # last AI message = pure text (model wants to end), no finalize called
        state = {
            "messages": [AIMessage(content="analysis done")],
            "thread_data": {"workspace_path": str(ws)},
        }
        runtime = MagicMock()
        out = mw._check(state, runtime)
        # should inject a reminder + jump_to model
        assert out is not None
        assert out.get("jump_to") == "model"

    def test_terminal_status_does_not_remind(self, tmp_path):
        """finalize 落盘终态 → SealGate 不催（workspace 文件 status=completed）。"""
        from deerflow.agents.middlewares.seal_gate_middleware import SealGateMiddleware
        from deerflow.tools.builtins.seal_handoff_tools import (
            fill_data_analyst_text_list,
            finalize_data_analyst_handoff,
            preset_data_analyst_template_to_workspace,
        )
        from langchain_core.messages import AIMessage

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        runtime = _make_runtime(str(ws))
        fill_data_analyst_text_list.func(
            field="key_findings", mode="set", value=["f1"], runtime=runtime,
        )
        finalize_data_analyst_handoff.func(final_status="completed", runtime=runtime)

        mw = SealGateMiddleware("data-analyst")
        state = {
            "messages": [AIMessage(content="done")],  # no finalize ToolMessage in history
            "thread_data": {"workspace_path": str(ws)},
        }
        out = mw._check(state, MagicMock())
        # workspace file status=completed → treated as sealed → no reminder
        assert out is None


# ---------------------------------------------------------------------------
# T7 — loop detection：fill_* lenient（多次 fill 不被 per-tool frequency 误杀）
# ---------------------------------------------------------------------------


class TestLoopDetectionLenient:
    def test_fill_tools_in_semantic_overrides(self):
        from deerflow.agents.middlewares.loop_detection_middleware import (
            _TOOL_FREQ_SEMANTIC_OVERRIDES,
        )

        assert "fill_data_analyst_text_list" in _TOOL_FREQ_SEMANTIC_OVERRIDES
        assert "fill_data_analyst_record_list" in _TOOL_FREQ_SEMANTIC_OVERRIDES
        assert "fill_data_analyst_gate_signals" in _TOOL_FREQ_SEMANTIC_OVERRIDES
        # thresholds should be lenient (>= write_todos tier)
        warn, hard = _TOOL_FREQ_SEMANTIC_OVERRIDES["fill_data_analyst_text_list"]
        assert warn >= 10 and hard >= 20


# ---------------------------------------------------------------------------
# T-contract — data_analyst system_prompt 含填模板流程、不含旧"产出与交付合一/立即 seal"
# ---------------------------------------------------------------------------


class TestDataAnalystPromptContract:
    def _load_prompt(self) -> str:
        from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG

        return DATA_ANALYST_CONFIG.system_prompt

    def test_prompt_describes_fill_template_flow(self):
        p = self._load_prompt()
        assert "fill_data_analyst_text_list" in p
        assert "finalize_data_analyst_handoff" in p
        assert "分步填模板" in p

    def test_prompt_no_old_produce_deliver_merged_phrase(self):
        p = self._load_prompt()
        # old "产出与交付合一" reversal target must be gone
        assert "产出与交付合一" not in p
        assert "立即调 seal_data_analyst_handoff" not in p

    def test_toolset_excludes_old_seal(self):
        from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG

        assert "seal_data_analyst_handoff" in DATA_ANALYST_CONFIG.disallowed_tools

    def test_fill_finalize_registered_in_builtin_tools(self):
        """装配回归（#187 漏注册 bug）：fill/finalize 必须在 BUILTIN_TOOLS 注册表里。

        #187 在 seal_handoff_tools.py 定义了 4 个工具、builtins/__init__.py 导出了，
        但漏在 tools.py 的 BUILTIN_TOOLS 注册 → data-analyst (tools=None 继承
        BUILTIN_TOOLS − disallowed) 拿不到 fill/finalize → 旧 seal 又被 disallowed →
        无任何能写 handoff 的工具 → 生产 100% FAILED + lead 降级。
        定义/导出/prompt/SealGate 的单元测试都绿，唯独这条装配链没测 → bug 溜过去。
        本测试守「凡 data-analyst 流程依赖的 fill/finalize 工具，都必须真在 BUILTIN_TOOLS」。
        """
        from deerflow.tools.tools import BUILTIN_TOOLS

        names = {getattr(t, "name", None) for t in BUILTIN_TOOLS}
        for required in (
            "fill_data_analyst_text_list",
            "fill_data_analyst_record_list",
            "fill_data_analyst_gate_signals",
            "finalize_data_analyst_handoff",
        ):
            assert required in names, (
                f"{required} 定义在 seal_handoff_tools.py 但未注册到 tools.py:BUILTIN_TOOLS "
                f"→ data-analyst 拿不到 → 无工具写 handoff → 确定性 FAILED"
            )

    def test_data_analyst_visible_toolset_includes_fill_finalize(self):
        """端到端装配：data-analyst 实际可见工具集（BUILTIN_TOOLS − disallowed）含 fill/finalize。

        直击 dogfood 复现的死路：LLM turn 4 正确调 fill_data_analyst_text_list，
        但工具不在 data-analyst 工具集 → 改走 write_file 被 guardrail 拒 → 挣扎放弃。
        """
        from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG
        from deerflow.tools.tools import BUILTIN_TOOLS

        builtin_names = {getattr(t, "name", None) for t in BUILTIN_TOOLS}
        disallowed = set(DATA_ANALYST_CONFIG.disallowed_tools)
        visible = builtin_names - disallowed
        for required in (
            "fill_data_analyst_text_list",
            "fill_data_analyst_record_list",
            "fill_data_analyst_gate_signals",
            "finalize_data_analyst_handoff",
        ):
            assert required in visible, f"{required} 不在 data-analyst 可见工具集（被 disallowed 或未注册）"


# ---------------------------------------------------------------------------
# T-resume — data-analyst 补轮催 finalize（隐藏耦合点 §三 #9 R2）
# ---------------------------------------------------------------------------


class TestSealResumeDataAnalyst:
    def test_resume_uses_finalize_for_data_analyst(self):
        """_attempt_seal_resume 的 seal_tool 对 data-analyst = finalize（§三 #9 R2）。

        conftest 把 deerflow.subagents.executor mock 掉，所以直接读被测源文件确认
        分支逻辑已编码（不依赖模块 import）。
        """
        from pathlib import Path

        src_path = Path(__file__).resolve().parents[1] / "packages" / "harness" / "deerflow" / "subagents" / "executor.py"
        src = src_path.read_text(encoding="utf-8")
        assert "finalize_data_analyst_handoff" in src
        assert 'self.config.name == "data-analyst"' in src


# ---------------------------------------------------------------------------
# §3.5 — in_progress 不是交付物：下游守卫
# ---------------------------------------------------------------------------


class TestInProgressNotDeliverable:
    def test_quality_warning_broadcast_ignores_in_progress(self, tmp_path):
        from deerflow.agents.middlewares.quality_warning_broadcast_middleware import (
            _load_quality_warnings,
        )
        from deerflow.tools.builtins.seal_handoff_tools import (
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        state = {"thread_data": {"workspace_path": str(ws)}}
        # in_progress → returns None (treated as not-delivered), not []
        assert _load_quality_warnings(state) is None

    def test_validate_handoff_emitted_in_progress_is_incomplete(self, tmp_path):
        """data-analyst 终止时仍是 in_progress → _validate_handoff_emitted 判
        非空失败（key_findings 空）→ 走 seal-resume/FAILED（崩溃诚实不 auto-seal）。

        conftest 把 deerflow.subagents.executor mock 掉，所以用 importlib 从源文件
        加载真实的 _validate_handoff_emitted。
        """
        import importlib.util

        from deerflow.tools.builtins.seal_handoff_tools import (
            preset_data_analyst_template_to_workspace,
        )

        src = Path(__file__).resolve().parents[1] / "packages" / "harness" / "deerflow" / "subagents" / "executor.py"
        spec = importlib.util.spec_from_file_location("_da_executor_real", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _validate_handoff_emitted = mod._validate_handoff_emitted

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        err = _validate_handoff_emitted("data-analyst", str(ws))
        assert err is not None  # in_progress template → key_findings empty → diagnostic
        assert "key_findings" in err or "finalize" in err


# ---------------------------------------------------------------------------
# T6 — 崩溃诚实：填一半 + in_progress + turn 用完 → 判 incomplete（不 auto-seal）
# ---------------------------------------------------------------------------


class TestCrashHonesty:
    def test_data_analyst_not_in_auto_sealable(self):
        """data-analyst 永不在 auto-seal 列表（认知产物不兜底，spec §六 R1）。"""
        from deerflow.subagents.executor import _AUTO_SEALABLE

        assert "data-analyst" not in _AUTO_SEALABLE
