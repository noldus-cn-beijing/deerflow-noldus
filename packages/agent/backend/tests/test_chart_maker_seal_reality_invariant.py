"""Spec 2026-06-24 ETHO-10: chart-maker 伪完成根治 —— 立「产物真实性不变式」。

封存 chart-maker handoff 时，`chart_files` 里每条路径必须确定性核对磁盘真存在；
不存在的不准留在 chart_files（剔除挪进 remaining_charts 留痕）。单一注入点
`_reconcile_chart_maker_payload`（seal 工具 + executor auto-seal 两路径）覆盖。

与 PR#165（test_chart_maker_seal_reconciliation）是同一道封存对账门的补全 + 升维：
PR#165 只对账 aggregate 图（plan output_mode=aggregate 必须全落盘），per_subject
完全没校验「磁盘真存在」→ 本次 dogfood 漏洞（chart-maker 把 57 张没画的 png 塞进
chart_files 还标 completed）。

本套件钉死「实质正确」维度（磁盘真相），补 PR#165 只测「形式正确」（非空/前缀）的漏。

TDD 顺序：本文件先红（实现前），实现后转绿。
"""

import json

import pytest

from deerflow.subagents.handoff_schemas import ChartMakerHandoff
from deerflow.tools.builtins.seal_handoff_tools import _seal_handoff_to_workspace

# ---------------------------------------------------------------------------
# fixture builders —— 复刻 prod plan_charts.json 真实形态（与
# test_chart_maker_seal_reconciliation.py 同源，独立复制以保持本文件自洽）
# ---------------------------------------------------------------------------


def _write_plan_charts(
    workspace,
    *,
    charts,
    skipped=None,
    budget_remaining=None,
):
    """写一份 prod 形态的 plan_charts.json 到 workspace。"""
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


def _per_subject_only_charts(n=4):
    """plan 里全是 per_subject 图（每 subject 一张 trajectory），无 aggregate。

    复现 2026-06-24 EPM 28-subject dogfood 场景：57 张图若全 per_subject，
    planned_aggregate 为空 → 2.2 门 `if not planned_aggregate: return` 整体放行。
    """
    charts = []
    for i in range(n):
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


def _aggregate_charts(*names):
    """plan 里的 aggregate 图（box/bar 组间对比 must_have）。"""
    return [
        {
            "id": name,
            "script": f"ethoinsight.scripts.epm.plot_{name}",
            "input": f"/mnt/user-data/workspace/inputs_{name}.json",
            "output": f"/mnt/user-data/outputs/plot_{name}.png",
            "subject_index": 0,
            "display_name_zh": f"{name}图",
            "confidence": "optional",
            "output_mode": "aggregate",
            "args": [
                "--inputs",
                f"/mnt/user-data/workspace/inputs_{name}.json",
                "--output",
                f"/mnt/user-data/outputs/plot_{name}.png",
            ],
        }
        for name in names
    ]


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
# T1：幻影剔除 —— chart_files 含 3 条路径，磁盘只有 1 个 png → 只留 1 条真实，
#     另 2 条进 remaining_charts 留痕
# ---------------------------------------------------------------------------


class TestPhantomChartFilesArePurged:
    """堵根因：chart_files 只查「非空 + 前缀」不查「真存在」。封存时每条路径必须
    确定性核磁盘，不存在的剔除挪进 remaining_charts（被截断/未画的留痕语义）。"""

    def test_phantom_paths_purged_real_kept(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        # plan：3 张 per_subject trajectory，无 aggregate
        _write_plan_charts(ws, charts=_per_subject_only_charts(3), skipped=[])
        # 磁盘只有 s0 这一张真 png
        _seed_outputs(outputs, "plot_trajectory_s0.png")

        written = _seal_chart_maker(
            ws,
            status="completed",
            chart_files=[
                "/mnt/user-data/outputs/plot_trajectory_s0.png",
                "/mnt/user-data/outputs/plot_trajectory_s1.png",  # 幻影
                "/mnt/user-data/outputs/plot_trajectory_s2.png",  # 幻影
            ],
            failed_charts=[],
        )

        # chart_files 只剩磁盘上真存在的 1 条
        assert written["chart_files"] == [
            "/mnt/user-data/outputs/plot_trajectory_s0.png"
        ], f"幻影路径必须剔除，只留真落盘的；got: {written['chart_files']!r}"
        # 2 条幻影进 remaining_charts 留痕（chart_id + reason）
        remaining_ids = sorted(r["chart_id"] for r in written["remaining_charts"])
        assert remaining_ids == ["plot_trajectory_s1.png", "plot_trajectory_s2.png"], (
            "被剔的幻影路径必须进 remaining_charts 留痕（供用户再要）；"
            f"got remaining_charts: {written['remaining_charts']!r}"
        )
        for r in written["remaining_charts"]:
            assert "rendered" in r["reason"].lower(), (
                "remaining_charts 的 reason 必须机读说明「声称画了但磁盘没落盘」"
            )


# ---------------------------------------------------------------------------
# T2：全幻影 —— chart_files 3 条全不存在 + status=completed → 剔空后抛 ValueError
#     （核心图一张没真画）
# ---------------------------------------------------------------------------


class TestAllPhantomCompletedRaises:
    """chart_files 全是幻影（磁盘 0 png）却标 completed = 哑完成的极致形态。
    剔除后 chart_files 空 → 抛 ValueError（逼补画或改 partial）。"""

    def test_all_phantom_completed_raises(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        _write_plan_charts(ws, charts=_per_subject_only_charts(3), skipped=[])
        # outputs/ 0 png（dogfood 真实：loop-detection 熔断致全没画）
        outputs.mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError) as exc_info:
            _seal_chart_maker(
                ws,
                status="completed",
                chart_files=[
                    "/mnt/user-data/outputs/plot_trajectory_s0.png",
                    "/mnt/user-data/outputs/plot_trajectory_s1.png",
                    "/mnt/user-data/outputs/plot_trajectory_s2.png",
                ],
                failed_charts=[],
            )
        msg = str(exc_info.value)
        assert "completed" in msg.lower(), "拒绝消息必须点明 completed 名不副实"


# ---------------------------------------------------------------------------
# T3：plan 无 aggregate 不再放行 —— 复现 dogfood 场景
#     plan 0 aggregate（全 per_subject）+ outputs 0 png + completed → ValueError
#     （堵 2.2 门 `if not planned_aggregate: return` 放行漏洞）
# ---------------------------------------------------------------------------


class TestNoAggregateNoLongerBypassesGate:
    """2026-06-24 dogfood 坐实的根因路径：plan 里全是 per_subject（无 aggregate）→
    planned_aggregate 为空 → 旧 2.2 门 `if not planned_aggregate: return` 整体放行
    → outputs 0 png 也标 completed。

    修法不依赖 plan 有没有 aggregate，直接核「真画出来的图」是不是空：剔除幻影后
    chart_files 空 + completed → ValueError。"""

    def test_per_subject_only_no_png_completed_raises(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        # plan 全 per_subject，无 aggregate（堵放行漏洞的关键前提）
        _write_plan_charts(ws, charts=_per_subject_only_charts(4), skipped=[])
        # outputs/ 0 png
        outputs.mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError) as exc_info:
            _seal_chart_maker(
                ws,
                status="completed",
                # chart-maker 误把没画的 per_subject 全塞进 chart_files（dogfood 形态）
                chart_files=[
                    f"/mnt/user-data/outputs/plot_trajectory_s{i}.png"
                    for i in range(4)
                ],
                failed_charts=[],
            )
        assert "completed" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# T4：aggregate 对账不回归 —— 既有 2.2 门（aggregate 缺失抛错）仍绿
# ---------------------------------------------------------------------------


class TestAggregateReconciliationNotRegressed:
    """守 PR#165 的 2.2 aggregate 对账门：plan 有 aggregate 但 outputs 没对应 png
    + completed → 仍响亮拒绝。本改动（产物真实性核对）跑在 2.2 门之前，不能吞掉它。"""

    def test_completed_missing_aggregate_still_raises(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        _write_plan_charts(
            ws, charts=_aggregate_charts("box_open_arm") + _per_subject_only_charts(1)
        )
        # outputs/ 只有 trajectory，缺 aggregate box png
        _seed_outputs(outputs, "plot_trajectory_s0.png")

        with pytest.raises(ValueError) as exc_info:
            _seal_chart_maker(
                ws,
                status="completed",
                chart_files=["/mnt/user-data/outputs/plot_trajectory_s0.png"],
                failed_charts=[],
            )
        # aggregate 对账门的消息特征（列出缺失 aggregate basename + 提 aggregate/render）
        msg = str(exc_info.value)
        assert "plot_box_open_arm.png" in msg


# ---------------------------------------------------------------------------
# T5：真实图全留 —— chart_files 全部真存在 + completed → 原样封存，不剔不抛
# ---------------------------------------------------------------------------


class TestRealChartFilesAllKept:
    """防误伤：chart_files 每条路径磁盘上真存在 → 全保留，不剔不抛。"""

    def test_all_real_completed_passes(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        _write_plan_charts(ws, charts=_per_subject_only_charts(3), skipped=[])
        _seed_outputs(
            outputs,
            "plot_trajectory_s0.png",
            "plot_trajectory_s1.png",
            "plot_trajectory_s2.png",
        )

        written = _seal_chart_maker(
            ws,
            status="completed",
            chart_files=[
                "/mnt/user-data/outputs/plot_trajectory_s0.png",
                "/mnt/user-data/outputs/plot_trajectory_s1.png",
                "/mnt/user-data/outputs/plot_trajectory_s2.png",
            ],
            failed_charts=[],
        )
        assert written["status"] == "completed"
        assert len(written["chart_files"]) == 3
        # 全真实 → remaining_charts 不应被污染
        assert written["remaining_charts"] == []


# ---------------------------------------------------------------------------
# T6：partial 豁免 —— status=partial + chart_files 含幻影 → 不抛（partial 允许
#     不全），但幻影仍剔进 remaining（产物真实性对所有 status 生效）
# ---------------------------------------------------------------------------


class TestPartialExemptButPhantomsStillPurged:
    """partial/failed 本就允许产物不全，不因 chart_files 剔空而拒（与现有
    _completed_requires_core_output 一致）。但「产物真实性」对所有 status 生效：
    幻影路径仍须剔进 remaining_charts——partial 的 chart_files 也不能含磁盘上
    不存在的路径（下游 present 时仍会 404）。"""

    def test_partial_phantom_purged_not_raised(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        _write_plan_charts(ws, charts=_per_subject_only_charts(3), skipped=[])
        _seed_outputs(outputs, "plot_trajectory_s0.png")

        written = _seal_chart_maker(
            ws,
            status="partial",
            chart_files=[
                "/mnt/user-data/outputs/plot_trajectory_s0.png",  # 真
                "/mnt/user-data/outputs/plot_trajectory_s1.png",  # 幻影
                "/mnt/user-data/outputs/plot_trajectory_s2.png",  # 幻影
            ],
            failed_charts=[],
        )
        # partial 不抛
        assert written["status"] == "partial"
        # chart_files 只留真实的（幻影剔掉，下游不 404）
        assert written["chart_files"] == [
            "/mnt/user-data/outputs/plot_trajectory_s0.png"
        ]
        # 幻影进 remaining_charts 留痕（真实性对所有 status 生效）
        remaining_ids = sorted(r["chart_id"] for r in written["remaining_charts"])
        assert remaining_ids == ["plot_trajectory_s1.png", "plot_trajectory_s2.png"]


# ---------------------------------------------------------------------------
# T7：auto-seal 路径同样核 —— executor auto-seal 构造的 chart-maker payload 过同一门
# ---------------------------------------------------------------------------


class TestAutoSealPathAlsoRealityChecked:
    """executor._attempt_auto_seal_from_artifacts 的 chart-maker 分支用 glob plot_*.png
    构造 chart_files（当前已是真实的），但它仍过 _seal_handoff_to_workspace →
    _reconcile_chart_maker_payload，受同一产物真实性核对约束。

    断言：auto-seal payload 若塞了磁盘上不存在的路径，同样被剔/拒（单一注入点保证）。"""

    def test_auto_seal_chart_files_are_disk_real(self, tmp_path):
        """auto-seal 正常路径：glob 出来的 plot_*.png 都是真的 → completed 正常封存。"""
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
        # glob 出来的都是真路径，全部保留
        assert data["status"] == "completed"
        assert len(data["chart_files"]) == 2
        assert data["remaining_charts"] == []


# ---------------------------------------------------------------------------
# T8：鲁棒性 —— outputs 读盘异常 → warning 跳过核对、不 crash
# ---------------------------------------------------------------------------


class TestRealityCheckRobustness:
    """读盘异常（OSError，如权限问题、坏链接遍历）不应 crash seal。
    spec 六.边界：核心图全空的 ValueError 是有意响亮拒绝，不在「绝不 crash」豁免内；
    但读盘机制本身的异常应 warning + 跳过核对，原样封存（同 2.2 门「plan 读不到不 crash」）。

    注：Path.exists() 本身吞掉 OSError 返回 False，所以「outputs 是文件」这类静默场景
    会走正常 purge 路径（合法：那些路径确实不存在）。本测试钉的是真正的 OSError 路径
    （权限/坏链接），通过 patch helper 制造确定性异常。
    """

    def test_disk_read_error_does_not_crash(self, tmp_path, monkeypatch):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = ws.parent / "outputs"
        _write_plan_charts(ws, charts=_per_subject_only_charts(2), skipped=[])
        _seed_outputs(outputs, "plot_trajectory_s0.png", "plot_trajectory_s1.png")

        import deerflow.tools.builtins.seal_handoff_tools as mod

        # 制造真正的 OSError（权限/坏链接遍历类），而非静默 exists()=False。
        def _boom(virtual_path, outputs_dir):
            raise OSError("permission denied reading outputs")

        monkeypatch.setattr(mod, "_chart_file_exists_on_disk", _boom)

        # 读盘异常 → 不 crash：保守保留所有 chart_files（绝不误剔真图）。
        written = _seal_chart_maker(
            ws,
            status="completed",
            chart_files=[
                "/mnt/user-data/outputs/plot_trajectory_s0.png",
                "/mnt/user-data/outputs/plot_trajectory_s1.png",
            ],
            failed_charts=[],
        )
        # 读盘异常 → 跳过核对，原样保留 chart_files（不 crash、不误剔）
        assert written["status"] == "completed"
        assert len(written["chart_files"]) == 2


def _load_executor_module():
    """惰性加载真实 executor 模块（与 test_chart_maker_seal_reconciliation 同模式）。"""
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
        "deerflow.subagents.executor_real_seal_gate_test_etho10",
        executor_file,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
