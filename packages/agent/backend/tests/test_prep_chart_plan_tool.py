"""Tests for prep_chart_plan_tool — Spec 3 (P3) charts 路径列对齐自读 context。

红线二正模式 1：column_aliases / groups 由工具内部从 session 状态自取，
取代 chart-maker 在 bash 里手拼 --column-aliases-file / --groups-json。

覆盖 spec §3 三个判据：
1. test_charts_resolve_picks_up_aliases_from_context —— experiment-context.json 有
   column_aliases，不手动传，box/bar 真能解析（修复前必红：旧路径 LLM 漏拼 alias）
2. test_charts_skip_when_truly_no_alignment —— 无 alias 且列名非标准 → box 合理 skip
3. test_charts_path_picks_up_groups_from_context —— groups.json 存在时被透传（parity
   与 metrics 路径：两路径都从 session 状态自取 groups）
"""

import json

from langchain.tools import ToolRuntime

from deerflow.tools.builtins.prep_chart_plan_tool import prep_chart_plan_tool


def _runtime_with_paths(workspace, uploads) -> ToolRuntime:
    return ToolRuntime(
        state={
            "thread_data": {
                "workspace_path": str(workspace),
                "uploads_path": str(uploads),
            }
        },
        context=None,
        config={},
        stream_writer=None,
        tool_call_id="test-id",
        store=None,
    )


def _write_ethovision_file(path: str, columns: list[str]):
    """Write a UTF-16 LE EthoVision trajectory file (mirrors prep_metric_plan test helper)."""
    header_lines = 36
    lines: list[str] = []
    lines.append(f'"Number of header lines:";"{header_lines}"')
    metadata = [
        ("Experiment", "Mock EPM"),
        ("Trial name", "Trial 1"),
        ("Subject", "Subject 1"),
        ("Start time", "2026-01-01 00:00:00"),
        ("Trial duration", "300"),
        ("Arena name", "Arena 1"),
        ("Number of Subjects", "1"),
    ]
    for k, v in metadata:
        lines.append(f'"{k}";"{v}"')
    while len(lines) < header_lines - 2:
        lines.append('""')
    lines.append('"' + '";"'.join(columns) + '"')
    lines.append('"' + '";"'.join(["s"] * len(columns)) + '"')
    lines.append(";".join(["-1.0"] * len(columns)))
    content = "\r\n".join(lines) + "\r\n"
    with open(path, "wb") as f:
        f.write(b"\xff\xfe")
        f.write(content.encode("utf-16-le"))


def _write_experiment_context(workspace, column_aliases=None):
    """Write experiment-context.json with an optional column_aliases projection."""
    ctx = {"paradigm": "epm"}
    if column_aliases is not None:
        ctx["column_aliases"] = column_aliases
    (workspace / "experiment-context.json").write_text(
        json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_groups_json(workspace, groups: dict):
    """Write groups.json exactly as prep_metric_plan_tool does."""
    (workspace / "groups.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# FewZones 列名：dogfood 实证真实结构（用户把开放臂/封闭臂简写成 open/closed）。
# parse_header normalize 后这些列名仍是 "open"/"closed"，需要 column_aliases 映射到
# catalog 概念关键词 "open_arms" / "closed_arms"。
FEW_ZONES_COLUMNS = [
    "Trial time",
    "Recording time",
    "X center",
    "Y center",
    "open",
    "closed",
]

# 标准 in_zone 列名（无须 alias 即可被 catalog 识别）。
STANDARD_EPM_COLUMNS = [
    "Trial time",
    "Recording time",
    "X center",
    "Y center",
    "in zone Open arms 1 / Center-point",
    "in zone Open arms 2 / Center-point",
    "in zone Closed arms 1 / Center-point",
    "in zone Closed arms 2 / Center-point",
]


class TestPrepChartPlanPicksUpAliasesFromContext:
    """spec §3 判据 1：column_aliases 来自 context，不手动传，box/bar 不被 skip。"""

    def test_charts_resolve_picks_up_aliases_from_context(self, tmp_path):
        """FewZones open/closed 列 + context 有 alias → plan_charts 含 open_arm_time_ratio_bar。

        修复前必红：旧路径 chart-maker bash 拼 catalog.resolve 漏传 --column-aliases-file，
        resolve 收到 None → in_zone_open_arms_* 缺列 → box/bar 被 skip，只出 trajectory/heatmap。
        新路径 prep_chart_plan 内部自读 context 拿 alias，bar 图必须出现。
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "few_zones_epm.txt"
        _write_ethovision_file(str(data_file), FEW_ZONES_COLUMNS)

        # context 投影出 alias（Gate 1 列语义对齐的产物）
        _write_experiment_context(
            workspace,
            column_aliases={"open": "open_arms", "closed": "closed_arms"},
        )

        runtime = _runtime_with_paths(workspace, uploads)
        # 关键：不传 column_aliases —— 工具必须自己从 context 读
        result = prep_chart_plan_tool.invoke({
            "uploaded_files": ["/mnt/user-data/uploads/few_zones_epm.txt"],
            "paradigm": "epm",
            "runtime": runtime,
            "total_subjects": 1,
        })

        assert result["status"] == "ok", f"expected ok, got: {result}"
        assert result["plan_summary"]["column_aliases_applied"] is True

        plan_path = workspace / "plan_charts.json"
        assert plan_path.exists()
        plan_data = json.loads(plan_path.read_text())
        chart_ids = [c["id"] for c in plan_data["charts"]]
        # open_arm_time_ratio_bar 只 requires in_zone_open_arms_*，alias 后应命中
        assert "open_arm_time_ratio_bar" in chart_ids, (
            f"FewZones alias 后 open_arm_time_ratio_bar 必须出现，实际 charts={chart_ids}"
        )
        # alias 已生效的证据：skipped 不应因 in_zone_open_arms_* 缺列而含 bar/box
        skipped_ids = [s["id"] for s in plan_data.get("skipped", [])]
        assert "open_arm_time_ratio_bar" not in skipped_ids

    def test_no_aliases_no_context_means_no_zone_charts(self, tmp_path):
        """spec §3 判据 2：无 context（无 alias）+ FewZones 列 → box/bar 合理 skip。

        守住「真缺列才跳」：open/closed 既无 alias 又非标准 in_zone_* 列名，
        zone 类图必须 skip，但 trajectory/heatmap（只要 x/y 坐标）仍应可用（fallback 路径）。
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "few_zones_no_ctx.txt"
        _write_ethovision_file(str(data_file), FEW_ZONES_COLUMNS)

        # 故意不写 experiment-context.json —— 无 alias
        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_chart_plan_tool.invoke({
            "uploaded_files": ["/mnt/user-data/uploads/few_zones_no_ctx.txt"],
            "paradigm": "epm",
            "runtime": runtime,
            "total_subjects": 1,
        })

        assert result["status"] == "ok"
        assert result["plan_summary"]["column_aliases_applied"] is False

        plan_data = json.loads((workspace / "plan_charts.json").read_text())
        chart_ids = [c["id"] for c in plan_data["charts"]]
        # 无 alias：zone 类图（依赖 in_zone_open_arms_*）不该出现
        assert "open_arm_time_ratio_bar" not in chart_ids
        # 但坐标类 fallback 图可用（_common catalog 只要 x_center/y_center）
        fallback_ids = [c["id"] for c in plan_data.get("charts_fallback_available", [])]
        all_ids = set(chart_ids) | set(fallback_ids)
        assert any("trajectory" == i or "heatmap" == i for i in all_ids), (
            f"坐标类 fallback 图必须可用，实际 all_ids={all_ids}"
        )

    def test_standard_columns_need_no_alias(self, tmp_path):
        """标准 in_zone 列名无需 alias：bar 图直接命中（对照基线，证明 alias 只是补救）。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "standard_epm.txt"
        _write_ethovision_file(str(data_file), STANDARD_EPM_COLUMNS)
        _write_experiment_context(workspace)  # 无 alias

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_chart_plan_tool.invoke({
            "uploaded_files": ["/mnt/user-data/uploads/standard_epm.txt"],
            "paradigm": "epm",
            "runtime": runtime,
            "total_subjects": 1,
        })

        assert result["status"] == "ok"
        plan_data = json.loads((workspace / "plan_charts.json").read_text())
        chart_ids = [c["id"] for c in plan_data["charts"]]
        assert "open_arm_time_ratio_bar" in chart_ids


class TestPrepChartPlanGroupsFromContext:
    """spec §3 + §4 风险边界：groups 也从 session 状态自取（groups.json），
    与 metrics 路径对称。needs_groups:true 的 box 图依赖它。"""

    def test_groups_json_is_threaded_into_resolve(self, tmp_path):
        """groups.json 存在 → groups_applied=True，且 plan.inputs.groups_file 指向它。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "grp_epm.txt"
        _write_ethovision_file(str(data_file), STANDARD_EPM_COLUMNS)
        _write_groups_json(
            workspace,
            {"/mnt/user-data/uploads/grp_epm.txt": "treatment"},
        )

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_chart_plan_tool.invoke({
            "uploaded_files": ["/mnt/user-data/uploads/grp_epm.txt"],
            "paradigm": "epm",
            "runtime": runtime,
            "total_subjects": 1,
        })

        assert result["status"] == "ok"
        assert result["plan_summary"]["groups_applied"] is True
        plan_data = json.loads((workspace / "plan_charts.json").read_text())
        assert plan_data["inputs"]["groups_file"] == "/mnt/user-data/workspace/groups.json"


class TestPrepChartPlanErrors:
    def test_workspace_missing(self):
        runtime = ToolRuntime(
            state={"thread_data": None},
            context=None,
            config={},
            stream_writer=None,
            tool_call_id="test-id",
            store=None,
        )
        result = prep_chart_plan_tool.invoke({
            "uploaded_files": ["/mnt/user-data/uploads/x.txt"],
            "paradigm": "epm",
            "runtime": runtime,
        })
        assert result["status"] == "error"
        assert result["error_code"] == "workspace_missing"

    def test_no_files_provided(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()
        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_chart_plan_tool.invoke({
            "uploaded_files": [],
            "paradigm": "epm",
            "runtime": runtime,
        })
        assert result["status"] == "error"
        assert result["error_code"] == "no_files_provided"

    def test_file_not_found(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()
        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_chart_plan_tool.invoke({
            "uploaded_files": ["/mnt/user-data/uploads/nonexistent.txt"],
            "paradigm": "epm",
            "runtime": runtime,
        })
        assert result["status"] == "error"
        assert result["error_code"] == "file_not_found"

    def test_unknown_paradigm(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()
        data_file = uploads / "epm.txt"
        _write_ethovision_file(str(data_file), STANDARD_EPM_COLUMNS)
        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_chart_plan_tool.invoke({
            "uploaded_files": ["/mnt/user-data/uploads/epm.txt"],
            "paradigm": "not_a_paradigm",
            "runtime": runtime,
        })
        assert result["status"] == "error"
        assert result["error_code"] == "unknown_paradigm"
