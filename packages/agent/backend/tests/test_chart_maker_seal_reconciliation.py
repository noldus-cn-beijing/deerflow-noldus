"""Spec 2026-06-22: chart-maker 伪造失败原因 + 漏执行 plan 内 aggregate 图 —— 透传真实 skip reason + 执行完整性门。

本测试套件钉死两类故障（见 spec 第 1 节根因）：
  - 根因 A：failed_charts[].reason 是 LLM free-text，可伪造（本次对 box_open_arm 编
    "missing columns"，但 plan_charts.json.skipped=[] 证伪）。封存时用 plan 的
    skipped[] 机读真相覆盖/订正。
  - 根因 B：status=completed 但 plan 内 aggregate 图未全部落盘 → 现有 validator
    只看 chart_files 非空（trajectory 在），放行了。修法：seal 对账 plan 的
    aggregate 图集合 ⊆ outputs/ 实际 png，缺口非空且 status=completed → 响亮拒绝。

单一注入点：_seal_handoff_to_workspace（seal_chart_maker_handoff 工具与 executor
auto-seal 兜底都过它）。对账仅在 model_cls is ChartMakerHandoff 时触发，不影响其余
3 个 handoff。

fixture 用 prod 真实形态（memory feedback_pr115_stage1_equivalence_baseline_is_hollow_error_string：
别造空壳）——charts[] 条目含 id/output/output_mode/confidence（fb3ed752 控制组实测字段）。
"""

import json

import pytest

from deerflow.subagents.handoff_schemas import ChartMakerHandoff
from deerflow.tools.builtins.seal_handoff_tools import _seal_handoff_to_workspace

# ---------------------------------------------------------------------------
# fixture builders —— 复刻 prod plan_charts.json 真实形态
# ---------------------------------------------------------------------------


def _write_plan_charts(
    workspace,
    *,
    charts,
    skipped=None,
    budget_remaining=None,
):
    """写一份 prod 形态的 plan_charts.json 到 workspace。

    charts: list of dict，每条至少 {id, output, output_mode}（缺字段补 prod 默认）。
    skipped: list of {id, reason, detail}（默认 []）。
    budget_remaining: list of chart entries（默认 []）。
    """
    plan = {
        "schema_version": "1",
        "paradigm": "epm",
        "ev19_template": "elevated_plus_maze",
        "charts": charts,
        "charts_fallback_available": [],
        "charts_budget_remaining": budget_remaining or [],
        "skipped": skipped or [],
        "user_intent": "test",
        "notes": "Generated N catalog charts",
    }
    (workspace / "plan_charts.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _epm_prod_charts(*, with_box=True):
    """复刻 fb3ed752 控制组的 charts[]（aggregate box + per_subject bar/trajectory）。"""
    charts = []
    if with_box:
        charts.append(
            {
                "id": "box_open_arm",
                "script": "ethoinsight.scripts.epm.plot_box_open_arm",
                "input": "/mnt/user-data/workspace/inputs_box_open_arm.json",
                "output": "/mnt/user-data/outputs/plot_box_open_arm.png",
                "subject_index": 0,
                "display_name_zh": "开放臂箱线图",
                "confidence": "optional",
                "output_mode": "aggregate",
                "args": [
                    "--inputs",
                    "/mnt/user-data/workspace/inputs_box_open_arm.json",
                    "--groups",
                    "/mnt/user-data/workspace/groups_box_open_arm.json",
                    "--output",
                    "/mnt/user-data/outputs/plot_box_open_arm.png",
                    "--parameters-json",
                    '{"closed_arm_zones": ["closed"], "open_arm_zones": ["open"]}',
                ],
            }
        )
    # per_subject：4 张 trajectory（被 chart_budget 截断是合法的）
    for i in range(4):
        charts.append(
            {
                "id": "trajectory",
                "script": "ethoinsight.scripts._common.plot_trajectory",
                "input": f"/mnt/user-data/workspace/inputs_trajectory_s{i}.json",
                "output": f"/mnt/user-data/outputs/plot_trajectory_s{i}.png",
                "subject_index": i,
                "display_name_zh": "轨迹图",
                "confidence": "optional",
                "output_mode": "per_subject",
                "args": [
                    "--inputs",
                    f"/mnt/user-data/workspace/inputs_trajectory_s{i}.json",
                    "--output",
                    f"/mnt/user-data/outputs/plot_trajectory_s{i}.png",
                ],
            }
        )
    return charts


def _seed_outputs(outputs_dir, *basenames):
    """在 outputs/ 下创建占位 png（0 字节即可，对账只看存在性）。"""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for name in basenames:
        (outputs_dir / name).write_bytes(b"")


def _seal_chart_maker(workspace, **payload_overrides):
    """调 _seal_handoff_to_workspace 封存 chart-maker handoff（prod 契约）。"""
    payload = {
        "status": "completed",
        "paradigm": "epm",
        "summary": "generated charts",
        "chart_files": [],
        "failed_charts": [],
        "remaining_charts": [],
        "gate_signals": None,
    }
    payload.update(payload_overrides)
    _seal_handoff_to_workspace(
        ChartMakerHandoff, "handoff_chart_maker.json", payload, workspace
    )
    return json.loads(
        (workspace / "handoff_chart_maker.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# Test 1 (2.1)：伪造 reason 被机读 plan 真相订正
# ---------------------------------------------------------------------------


class TestSealOverwritesFabricatedReasonWithPlanTruth:
    """堵根因 A：chart-maker 对一个 plan 里没 skip 的 chart_id 编 "missing columns"
    之类的假 reason，封存时必须用 plan_charts.json 的真实状态订正。

    复现并锁死 2026-06-22 prod bug：box_open_arm 在 plan 里 args 完备、skipped=[]，
    但 chart-maker 自报 reason="catalog.resolve skipped: missing columns ..."。
    """

    def test_fabricated_missing_columns_reason_is_corrected(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        # plan：box_open_arm 在 charts[]、skipped=[]（即 resolver 没 skip 它）
        _write_plan_charts(ws, charts=_epm_prod_charts(with_box=True), skipped=[])
        # outputs/ 只有 trajectory，没有 box png（执行漏画）
        _seed_outputs(
            outputs,
            "plot_trajectory_s0.png",
            "plot_trajectory_s1.png",
            "plot_trajectory_s2.png",
            "plot_trajectory_s3.png",
        )

        # chart-maker 自报的伪造 reason（抄了 6/16 旧列门 reason）。
        # 用 status=partial 隔离 2.1（reason 订正）与 2.2（completed 完整性门）：
        # 本用例只钉 reason 订正；completed + 漏 aggregate 的拒绝另由
        # TestSealRejectsCompletedWithMissingAggregate 覆盖。
        written = _seal_chart_maker(
            ws,
            status="partial",
            chart_files=[
                f"/mnt/user-data/outputs/plot_trajectory_s{i}.png" for i in range(4)
            ],
            failed_charts=[
                {
                    "chart_id": "box_open_arm",
                    "reason": "catalog.resolve skipped: missing columns in_zone_open_arms_* (raw files are xlsx without zone column)",
                }
            ],
        )

        box_entry = next(
            c for c in written["failed_charts"] if c["chart_id"] == "box_open_arm"
        )
        # 权威 reason 以机读形态开头（"resolved in plan but not rendered"），
        # 不再让 LLM 编的 "missing columns" 作为权威失败因。LLM 原文仅作引用
        # 保留在括号注里（chart-maker note: ...），可被下游识别为非权威。
        assert box_entry["reason"].startswith("resolved in plan but not rendered"), (
            "权威 reason 必须是机读形态（resolved in plan but not rendered），"
            f"got: {box_entry['reason']!r}"
        )

    def test_real_skipped_reason_is_passed_through(self, tmp_path):
        """若 chart_id 确实在 plan.skipped[] 里 → 用 plan 的真实 detail 覆盖
        LLM 自述（plan 是 resolver 机读真相，权威）。"""
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        _write_plan_charts(
            ws,
            charts=_epm_prod_charts(with_box=False),  # box 不在 charts[]，在 skipped[]
            skipped=[
                {
                    "id": "box_open_arm",
                    "reason": "columns.missing",
                    "detail": "Chart box_open_arm skipped: missing columns ['in_zone_open_arms_*'].",
                }
            ],
        )
        _seed_outputs(outputs, "plot_trajectory_s0.png")

        written = _seal_chart_maker(
            ws,
            status="partial",
            chart_files=["/mnt/user-data/outputs/plot_trajectory_s0.png"],
            failed_charts=[
                {
                    "chart_id": "box_open_arm",
                    "reason": "I think it failed somehow",  # LLM 模糊自述
                }
            ],
        )

        box_entry = next(
            c for c in written["failed_charts"] if c["chart_id"] == "box_open_arm"
        )
        # 用 plan 的真实 detail 覆盖
        assert "missing columns" in box_entry["reason"]
        assert "in_zone_open_arms" in box_entry["reason"]


# ---------------------------------------------------------------------------
# Test 2 (2.2)：completed + plan 内 aggregate 图未落盘 → 响亮拒绝
# ---------------------------------------------------------------------------


class TestSealRejectsCompletedWithMissingAggregate:
    """堵根因 B：plan 要画的 aggregate 图（box/bar 这类组间对比 must_have）漏画，
    却标 status=completed → seal 必须响亮拒绝（ValueError），而非放行"画了一半"
    的哑完成。

    chart_files 非空（4 张 trajectory）以绕过现有空 check，证明是新门在拦。
    """

    def test_completed_with_missing_aggregate_raises(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        _write_plan_charts(ws, charts=_epm_prod_charts(with_box=True), skipped=[])
        # outputs/ 只有 trajectory —— aggregate box_open_arm 漏画
        _seed_outputs(outputs, "plot_trajectory_s0.png")

        with pytest.raises(ValueError) as exc_info:
            _seal_chart_maker(
                ws,
                status="completed",
                chart_files=["/mnt/user-data/outputs/plot_trajectory_s0.png"],
                failed_charts=[],
            )

        msg = str(exc_info.value)
        assert "plot_box_open_arm.png" in msg, (
            "拒绝消息必须列出缺失的 aggregate 图 basename，让 chart-maker 知道补画哪张"
        )
        assert "aggregate" in msg.lower() or "render" in msg.lower()

    def test_partial_with_missing_aggregate_is_allowed(self, tmp_path):
        """status=partial 时允许 aggregate 缺失（partial 本就表示部分失败）。
        门只拦 completed 的哑完成，不与 partial 路径打架。"""
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        _write_plan_charts(ws, charts=_epm_prod_charts(with_box=True), skipped=[])
        _seed_outputs(outputs, "plot_trajectory_s0.png")

        # partial 不抛——partial 是诚实的部分成功
        written = _seal_chart_maker(
            ws,
            status="partial",
            chart_files=["/mnt/user-data/outputs/plot_trajectory_s0.png"],
            failed_charts=[
                {"chart_id": "box_open_arm", "reason": "ran out of turns"}
            ],
        )
        assert written["status"] == "partial"


# ---------------------------------------------------------------------------
# Test 3 (绿)：所有 aggregate 都落盘 → completed 正常封存（防误伤）
# ---------------------------------------------------------------------------


class TestSealCompletedPassesWhenAllAggregateRendered:
    """对应本地 fb3ed752 控制组跑通的形态：plan 有 box_open_arm，outputs/ 也有
    plot_box_open_arm.png → completed 应正常封存，不抛错，reason 不被乱改。"""

    def test_completed_passes(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        _write_plan_charts(ws, charts=_epm_prod_charts(with_box=True), skipped=[])
        # aggregate + per_subject 都在 outputs/
        _seed_outputs(
            outputs,
            "plot_box_open_arm.png",
            "plot_trajectory_s0.png",
            "plot_trajectory_s1.png",
        )

        written = _seal_chart_maker(
            ws,
            status="completed",
            chart_files=[
                "/mnt/user-data/outputs/plot_box_open_arm.png",
                "/mnt/user-data/outputs/plot_trajectory_s0.png",
                "/mnt/user-data/outputs/plot_trajectory_s1.png",
            ],
            failed_charts=[],
        )
        assert written["status"] == "completed"
        assert len(written["chart_files"]) == 3


# ---------------------------------------------------------------------------
# Test 4 (绿)：per_subject 预算截断不触发门（防与 P5 打架）
# ---------------------------------------------------------------------------


class TestSealPerSubjectBudgetTruncationNotBlocked:
    """per_subject 图被 chart_budget 截断本就允许不画（remaining_charts 指纹机制）。
    门只管 aggregate；缺 per_subject png 不触发拒绝。否则会和 P5 预算规则打架。"""

    def test_missing_per_subject_does_not_block(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        charts = _epm_prod_charts(with_box=True)
        # plan charts[] 有 4 张 trajectory per_subject + 1 box aggregate
        _write_plan_charts(
            ws,
            charts=charts,
            skipped=[],
            # 预算截断了 s1/s2/s3 trajectory（进 remaining 指纹）
            budget_remaining=[c for c in charts if c["id"] == "trajectory"][1:],
        )
        # outputs/ 只有 box aggregate + 1 张 trajectory；其余 per_subject 缺
        _seed_outputs(outputs, "plot_box_open_arm.png", "plot_trajectory_s0.png")

        written = _seal_chart_maker(
            ws,
            status="completed",
            chart_files=[
                "/mnt/user-data/outputs/plot_box_open_arm.png",
                "/mnt/user-data/outputs/plot_trajectory_s0.png",
            ],
            failed_charts=[],
        )
        # 不触发拒绝：aggregate（box）已落盘，per_subject 截断豁免
        assert written["status"] == "completed"


# ---------------------------------------------------------------------------
# Test 5 (绿)：plan_charts.json 不存在 → 不 crash，原样封存 + 鲁棒
# ---------------------------------------------------------------------------


class TestSealToleratesMissingPlanCharts:
    """极端早退场景：plan_charts.json 可能不存在（prep 没跑到 / 别的早退路径）。
    读不到 plan 不能 crash 掉 seal —— 跳过对账、按原样封存（spec 2.1 实现要点）。"""

    def test_no_plan_charts_seals_anyway(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        # 不写 plan_charts.json
        _seed_outputs(outputs, "plot_trajectory_s0.png")

        written = _seal_chart_maker(
            ws,
            status="completed",
            chart_files=["/mnt/user-data/outputs/plot_trajectory_s0.png"],
            failed_charts=[],
        )
        assert written["status"] == "completed"
        assert written["chart_files"] == ["/mnt/user-data/outputs/plot_trajectory_s0.png"]

    def test_no_plan_charts_keeps_failed_reason_as_is(self, tmp_path):
        """无 plan 时无法对账 reason → 保留 LLM 原文（不做订正也不 crash）。"""
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        _seed_outputs(outputs, "plot_trajectory_s0.png")

        written = _seal_chart_maker(
            ws,
            status="partial",
            chart_files=["/mnt/user-data/outputs/plot_trajectory_s0.png"],
            failed_charts=[{"chart_id": "x", "reason": "some stderr"}],
        )
        assert written["failed_charts"][0]["reason"] == "some stderr"


# ---------------------------------------------------------------------------
# Test 6：auto-seal 兜底路径也走同一对账门（spec 第 4 节风险边界）
# ---------------------------------------------------------------------------


class TestAutoSealGateInteraction:
    """executor._attempt_auto_seal_from_artifacts 复用 _seal_handoff_to_workspace，
    所以 chart-maker 的对账门同样作用于 auto-seal 兜底路径。

    bug 场景（spec 第 0 节）：chart-maker 漏调 seal、撞 max_turns 终止，outputs/ 只有
    trajectory。auto-seal 看到 plot_*.png 想机械兜底成 completed，但 plan 里还有
    aggregate（box_open_arm）没画 → 门抛 ValueError，auto-seal 的 ROBUSTNESS try/except
    捕获 → 返回 False（诚实 FAILED），而不是兜底成"画了一半的 completed"。
    """

    def test_auto_seal_returns_false_when_aggregate_missing(self, tmp_path):
        mod = _load_executor_module()

        ws = tmp_path / "user-data" / "workspace"
        out = tmp_path / "user-data" / "outputs"
        ws.mkdir(parents=True)
        out.mkdir(parents=True)
        # plan 有 aggregate box_open_arm；outputs/ 只有 trajectory（aggregate 漏画）
        _write_plan_charts(ws, charts=_epm_prod_charts(with_box=True), skipped=[])
        _seed_outputs(out, "plot_trajectory_s0.png")
        # code-executor handoff（auto-seal 读 paradigm）
        (ws / "handoff_code_executor.json").write_text(
            json.dumps({"paradigm": "epm", "metrics_summary": {"epm": {}}}),
            encoding="utf-8",
        )
        (ws / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "cfg"}), encoding="utf-8"
        )

        ok = mod._attempt_auto_seal_from_artifacts("chart-maker", str(ws))
        assert ok is False, (
            "auto-seal 不应把漏画 aggregate 的产出兜底成 completed；应返回 False "
            "（诚实 FAILED），让 lead/人看到真实失败而非伪造成功"
        )
        # 不留伪造的 completed handoff
        assert not (ws / "handoff_chart_maker.json").exists()

    def test_auto_seal_succeeds_when_no_plan(self, tmp_path):
        """auto-seal 老场景（无 plan_charts.json）不受影响：对账早退，正常兜底。"""
        mod = _load_executor_module()

        ws = tmp_path / "user-data" / "workspace"
        out = tmp_path / "user-data" / "outputs"
        ws.mkdir(parents=True)
        out.mkdir(parents=True)
        _seed_outputs(out, "plot_trajectory.png", "plot_heatmap.png")
        (ws / "handoff_code_executor.json").write_text(
            json.dumps({"paradigm": "epm", "metrics_summary": {"epm": {}}}),
            encoding="utf-8",
        )
        (ws / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "cfg"}), encoding="utf-8"
        )

        ok = mod._attempt_auto_seal_from_artifacts("chart-maker", str(ws))
        assert ok is True
        data = json.loads((ws / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        assert data["status"] == "completed"


def _load_executor_module():
    """惰性加载真实 executor 模块。

    conftest.py 在 sys.modules 里 mock 了 deerflow.subagents.executor（破导入环），
    直接 importlib.import_module 会拿到 mock。用 spec_from_file_location 从源文件
    加载真实模块（与 test_auto_seal_from_artifacts._get_real_executor 同模式）。
    """
    import importlib.util
    from pathlib import Path

    executor_file = (
        Path(__file__).resolve().parents[1]
        / "packages"
        / "harness"
        / "deerflow"
        / "subagents"
        / "executor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "deerflow.subagents.executor_real_seal_gate_test",
        executor_file,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
