"""Spec 2026-06-25-chart-maker-seal-once-and-results-key-uniqueness — M1 (R1) tests.

治 R1（double-seal 覆盖）：chart-maker 手调 ``seal_chart_maker_handoff`` 覆盖
``run_chart_plan`` 的确定性封存（``sealed_by=run_plan``）。M1 在
``_seal_handoff_to_workspace``（chart-maker 分支）加「封存只允许一次」不变式：
磁盘已有 ``sealed_by=run_plan`` 封存时，任何**非 force** seal raise ValueError。

- T1：模拟 chart-maker 手调 seal（``seal_chart_maker_handoff`` 不暴露 force）→ 撞门拒绝。
- T2：``run_chart_plan`` 重跑全量（force=True，合法覆盖场景）→ 不被门误伤。
- T3：harness auto-seal（force=False）撞已有 run_plan 封存 → 被拒，且不抛出逃逸（auto-seal
  内部 try/except 吞掉，return False，subagent 不会因此升级 FAILED）。

模块加载方式照抄 test_run_chart_plan.py：importlib.util 加载 worktree 源（绕 conftest 的
``deerflow.subagents.executor`` sys.modules mock；worktree 借主仓 venv，editable .pth 指向
主仓，故必须 spec_from_file_location 才能加载 worktree 代码）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from langchain.tools import ToolRuntime

# Load the tool module fresh via importlib (bypasses conftest sys.modules mock of
# deerflow.subagents.executor; worktree's editable pth points at the MAIN repo).
_TOOL_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "tools" / "builtins" / "run_chart_plan_tool.py"
)
_SEAL_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "tools" / "builtins" / "seal_handoff_tools.py"
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[])
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# 关键：worktree 借主仓 venv（editable .pth 指向**主仓** harness 包），故裸
# ``from deerflow.tools.builtins.seal_handoff_tools import ...`` 会拿到主仓（未改）版本。
# M1 门在 seal_handoff_tools.py，run_chart_plan_tool 的惰性 import 也走 deerflow.* → 主仓。
# 要测 worktree 改动，必须把 worktree 的 seal 模块以**真实 deerflow 名**塞进 sys.modules，
# 让 run_chart_plan_tool 函数体内 ``from deerflow...import _seal_handoff_to_workspace`` 解析到
# worktree（带门）版本。seal 模块自身只 import handoff_schemas / Runtime（不在环上，主仓版本即可）。
_SEAL = _load_module("deerflow.tools.builtins.seal_handoff_tools", _SEAL_FILE)
# 把 worktree 的 seal 模块以**真实 deerflow 名**塞进 sys.modules，让 run_chart_plan_tool
# 函数体内 ``from deerflow...import _seal_handoff_to_workspace`` 解析到 worktree（带门）版本。
# seal 模块自身只 import handoff_schemas / Runtime（不在环上，主仓版本即可）。
sys.modules["deerflow.tools.builtins.seal_handoff_tools"] = _SEAL

# run_chart_plan_tool 现在惰性 import 会拿到上面塞入的 worktree seal 模块。
_TOOL = _load_module("deerflow.tools.builtins.run_chart_plan_tool_real_seal_once", _TOOL_FILE)


# ---------------------------------------------------------------------------
# Fixtures + helpers（镜像 test_run_chart_plan.py）
# ---------------------------------------------------------------------------


def _runtime(workspace: Path, outputs: Path) -> ToolRuntime:
    """ToolRuntime with thread_data mapping /mnt/user-data onto real tmp dirs."""
    return ToolRuntime(
        state={
            "thread_data": {
                "workspace_path": str(workspace),
                "outputs_path": str(outputs),
                "uploads_path": str(workspace.parent / "uploads"),
            }
        },
        context=None,
        config={},
        stream_writer=None,
        tool_call_id="test-id",
        store=None,
    )


def _plan(charts: list[dict], *, paradigm: str = "epm") -> dict:
    for c in charts:
        c.setdefault("output_mode", "per_subject")
        c.setdefault("subject_index", 0)
        if "output" not in c:
            c["output"] = f"/mnt/user-data/outputs/{c['id']}.png"
        if "args" not in c:
            c["args"] = ["--input", "/mnt/user-data/uploads/fake.txt", "--output", c["output"]]
        c.setdefault("script", "ethoinsight.scripts.epm.plot_fake")
    return {
        "schema_version": "1.1",
        "paradigm": paradigm,
        "ev19_template": paradigm,
        "generated_at": "2026-06-25T00:00:00",
        "inputs": {"raw_files": ["/mnt/user-data/uploads/fake.txt"], "groups_file": None, "columns_file": None},
        "charts": charts,
        "charts_fallback_available": [],
        "charts_budget_remaining": [],
        "skipped": [],
        "user_intent": None,
        "notes": [],
    }


def _write_plan(workspace: Path, plan: dict) -> None:
    (workspace / "plan_charts.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    (workspace / "experiment-context.json").write_text(
        json.dumps({"analysis_config_id": "test-config-seal-once"}), encoding="utf-8"
    )


def _runner_success(*, outputs: Path):
    def _r(script: str, args: list[str], task_id: str):
        out = None
        for i, a in enumerate(args):
            if a == "--output" and i + 1 < len(args):
                out = args[i + 1]
        if out:
            name = out.rsplit("/", 1)[-1]
            (outputs / name).parent.mkdir(parents=True, exist_ok=True)
            (outputs / name).write_bytes(b"\x89PNG\r\n\x1a\n")
        return (task_id, 0, "")

    return _r


@pytest.fixture
def ws_and_outputs(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return workspace, outputs


def _call_run_chart_plan(runtime, **kwargs):
    """run_chart_plan_tool 现返 Command(update={"messages":[ToolMessage(json(result))...]})（spec
    2026-06-25-auto-register-artifacts）。解包首条 ToolMessage 的 json content 回 result dict，
    让既有读 ``res["status"]`` 的断言零改动继续工作（同 test_run_chart_plan.py:_call 模式）。
    """
    cmd = _TOOL.run_chart_plan_tool.func(runtime, tool_call_id="test-tcid", **kwargs)
    msgs = cmd.update.get("messages", [])
    assert msgs, f"Command 缺 ToolMessage: {cmd.update}"
    return json.loads(msgs[0].content)


def _read_handoff(workspace: Path) -> dict:
    return json.loads((workspace / "handoff_chart_maker.json").read_text(encoding="utf-8"))


# ===================================================================
# T1（R1 红→绿）：手调 seal_chart_maker_handoff 不能覆盖 run_plan 封存
# ===================================================================


class TestSealOnceRejectsModelOverwrite:
    """chart-maker 手调 seal_chart_maker_handoff（force 默认 False）撞 run_plan 封存 → 拒绝。

    红态：当前 ``_seal_handoff_to_workspace`` 无条件覆盖 → 第 2 步成功，sealed_by=model。
    绿态：M1 后第 2 步 raise ValueError「已存在确定性封存，拒绝覆盖」。
    """

    def test_t1_model_seal_rejected_after_run_plan(self, ws_and_outputs, monkeypatch):
        workspace, outputs = ws_and_outputs
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_success(outputs=outputs))
        charts = [{"id": "box_open_arm", "output_mode": "aggregate"}]
        _write_plan(workspace, _plan(charts))

        # 1. run_chart_plan 先封存（force=True 合法）→ sealed_by=run_plan
        res = _call_run_chart_plan(_runtime(workspace, outputs))
        assert res["status"] == "completed", res
        h1 = _read_handoff(workspace)
        assert h1["sealed_by"] == "run_plan"

        # 2. 模拟 chart-maker 手调 seal_chart_maker_handoff（LLM 工具，不暴露 force）。
        #    _seal_handoff → _seal_handoff_to_workspace 不传 force → 撞门拒绝。
        runtime = _runtime(workspace, outputs)
        with pytest.raises(ValueError, match="已存在确定性封存"):
            _SEAL.seal_chart_maker_handoff.func(
                paradigm="epm",
                summary="chart-maker 手调覆盖叙事（dogfood thread a6e3775c 复现）",
                chart_files=["/mnt/user-data/outputs/box_open_arm.png"],
                status="completed",
                runtime=runtime,
            )

        # 磁盘上仍是 run_plan 封存，未被覆盖（R1 的核心断言）。
        h2 = _read_handoff(workspace)
        assert h2["sealed_by"] == "run_plan", (
            f"R1 red: 手调 seal 覆盖了 run_plan 封存 → sealed_by={h2['sealed_by']!r}"
        )


# ===================================================================
# T2（R1）：run_chart_plan 重跑全量 force=True 仍合法覆盖
# ===================================================================


class TestRunChartPlanForceOverrides:
    """用户追加图重跑场景：run_chart_plan 带 force=True 合法覆盖已有 run_plan 封存。"""

    def test_t2_rerun_all_force_overrides_existing(self, ws_and_outputs, monkeypatch):
        workspace, outputs = ws_and_outputs
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_success(outputs=outputs))
        charts = [{"id": "box_open_arm", "output_mode": "aggregate"}]
        _write_plan(workspace, _plan(charts))

        # 首次 run_chart_plan 封存。
        _call_run_chart_plan(_runtime(workspace, outputs))
        assert _read_handoff(workspace)["sealed_by"] == "run_plan"

        # 重跑（用户追加图重画）——run_chart_plan 内部调 _seal_handoff_to_workspace(force=True)，
        # 不应被 M1 门拒绝。summary 机械重写，sha256 变（覆盖语义正常）。
        res2 = _call_run_chart_plan(_runtime(workspace, outputs))
        assert res2["status"] == "completed", res2
        h2 = _read_handoff(workspace)
        assert h2["sealed_by"] == "run_plan"
        assert h2["summary"] == "1/1 charts rendered (run_chart_plan)"


# ===================================================================
# T3（R1）：auto-seal 撞已有 run_plan 封存被拒（不抛逃逸）
# ===================================================================


class TestAutoSealRejectedWhenRunPlanSealed:
    """harness auto-seal（force=False）撞 run_chart_plan 已封存 → 拒绝。

    守 spec §五 风险 3：auto-seal 的 ValueError 必须被其内部 try/except 吞掉，
    return False，**不升级为 subagent FAILED**（auto-seal 失败应静默跳过）。
    实测：_attempt_auto_seal_from_artifacts 本身对已存在的非空 handoff 早退（line 365），
    但若到了 seal 调用，M1 门拒也必须被吞掉。这里直接测 _seal_handoff_to_workspace 行为。
    """

    def test_t3_seal_helper_force_false_rejected_after_run_plan(self, ws_and_outputs, monkeypatch):
        from deerflow.subagents.handoff_schemas import ChartMakerHandoff

        workspace, outputs = ws_and_outputs
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_success(outputs=outputs))
        charts = [{"id": "box_open_arm", "output_mode": "aggregate"}]
        _write_plan(workspace, _plan(charts))

        # run_chart_plan 先确定性封存。
        _call_run_chart_plan(_runtime(workspace, outputs))
        assert _read_handoff(workspace)["sealed_by"] == "run_plan"

        # auto-seal 等价路径：直接调 _seal_handoff_to_workspace(force 默认 False) → 撞门。
        # executor._attempt_auto_seal_from_artifacts 的 chart-maker 分支就是这么调的（不带 force）。
        payload = {
            "status": "completed",
            "paradigm": "epm",
            "summary": "harness auto-seal 试图覆盖 run_plan",
            "chart_files": ["/mnt/user-data/outputs/box_open_arm.png"],
            "failed_charts": [],
            "gate_signals": None,
        }
        with pytest.raises(ValueError, match="已存在确定性封存"):
            _SEAL._seal_handoff_to_workspace(
                ChartMakerHandoff, "handoff_chart_maker.json", payload, workspace
            )

        # 磁盘未被 auto-seal 覆盖。
        assert _read_handoff(workspace)["sealed_by"] == "run_plan"


# ===================================================================
# T3b：auto-seal 不传 force = 调用方契约（守 §五 风险 2 grep 断言）
# ===================================================================


class TestForceOnlyInRunChartPlan:
    """force=True 只在 run_chart_plan 工具内部传，LLM 工具签名不暴露 force。

    守 spec §五 风险 2：否则 LLM 学会传 force 绕门（reward hacking）。
    """

    def test_t3b_seal_chart_maker_handoff_signature_has_no_force(self):
        import inspect

        sig = inspect.signature(_SEAL.seal_chart_maker_handoff.func)
        assert "force" not in sig.parameters, (
            f"R1 reward-hacking risk: seal_chart_maker_handoff 暴露了 force 参数 → LLM 可绕门: {sig}"
        )

    def test_t3b_run_chart_plan_internal_seal_passes_force_true(self):
        """run_chart_plan_tool 源码里主封存调用（Step 9）必须带 force=True。

        排除 _seal_failed（failed 路径，无 prior run_plan 封存要覆盖，走默认即可）。
        主封存调用是单行 ``_seal_handoff_to_workspace(ChartMakerHandoff, ..., force=True)``。
        """
        src = _TOOL_FILE.read_text(encoding="utf-8")
        # 主封存调用锚点：Step 9 的 force=True 调用（排除 _seal_failed 里不带 force 的同款前缀）。
        anchor = "_seal_handoff_to_workspace(ChartMakerHandoff, \"handoff_chart_maker.json\", payload, workspace, force=True)"
        assert anchor in src, (
            "未找到 run_chart_plan 主封存调用（带 force=True 的单行形态）"
        )
