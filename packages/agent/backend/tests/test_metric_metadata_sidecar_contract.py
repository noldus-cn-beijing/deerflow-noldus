"""spec 2026-06-22-metric-metadata-sidecar：backend 侧契约。

覆盖 spec §四 测试项：
  - 项 2：prep_metric_plan 落盘时同源同次写 _metric_metadata.json，且是 plan_metrics.json 的
    去重元数据投影（按 metric id 一条，而非按 subject 重复 N×M 条）。
  - 项 4：report-writer / data-analyst system_prompt 静态契约 —— 含旁路措辞 + thinking 铁律，
    不含旧诱导串（「按 metric id 在 metrics[] 数组中匹配」）、report-writer 不再 read plan_metrics.json。

ethoinsight 侧 metric_metadata_to_dict 的去重单测见
packages/ethoinsight/tests/test_metric_metadata_sidecar.py。
"""
import json

# 复用现有 prep 工具测试的 EthoVision 文件构造器（同一批 EV 导出列结构）。
# 同目录裸 import（pytest prepend import mode 把 tests/ 加到 sys.path）。
from test_prep_metric_plan_tool import (
    EPM_COLUMNS,
    _runtime_with_paths,
    _write_ethovision_file,
)

from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG
from deerflow.subagents.builtins.report_writer import REPORT_WRITER_CONFIG
from deerflow.tools.builtins.prep_metric_plan_tool import prep_metric_plan_tool


def _run_prep_epm(tmp_path, n_subjects: int = 1) -> dict:
    """跑一次成功的 EPM prep，返回 tool 结果；workspace/uploads 已就绪并写了 EPM 数据。

    n_subjects>1 时传同一物理文件的多个虚拟路径——resolve_metrics 按 raw_files 列表长度
    展开 N 个 PlanMetric（每 subject 一条），模拟 spec 场景的「按 subject 重复」。
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    data_file = uploads / "test_epm.txt"
    _write_ethovision_file(str(data_file), EPM_COLUMNS)
    runtime = _runtime_with_paths(workspace, uploads)
    uploaded_files = ["/mnt/user-data/uploads/test_epm.txt"] * n_subjects
    return prep_metric_plan_tool.invoke(
        {
            "uploaded_files": uploaded_files,
            "paradigm": "epm",
            "runtime": runtime,
        }
    )


def test_metric_metadata_sidecar_written_alongside_plan(tmp_path) -> None:
    """项 2：prep 成功后 workspace 同时存在 plan_metrics.json 与 _metric_metadata.json。

    多 subject 场景（3 subject × 5 metric = 15 条 plan）：旁路是 plan 的去重元数据投影，
    metric id 集合一致，但旁路按 id 一条（dict，5 条），plan 按 subject 重复（list，15 条）。
    """
    workspace = tmp_path / "workspace"
    result = _run_prep_epm(tmp_path, n_subjects=3)
    assert result["status"] == "ok", f"prep 应成功，实际 {result}"

    plan_path = workspace / "plan_metrics.json"
    metadata_path = workspace / "_metric_metadata.json"
    assert plan_path.exists(), "plan_metrics.json 未落盘"
    assert metadata_path.exists(), "_metric_metadata.json 未落盘（spec §3.1 要求同源同次写旁路）"

    plan_data = json.loads(plan_path.read_text())
    meta_data = json.loads(metadata_path.read_text())

    # 顶层结构契约
    assert meta_data["paradigm"] == "epm"
    assert isinstance(meta_data["metrics"], dict), "旁路 metrics 必须是 dict（按 id 直查）"

    plan_metric_ids = {m["id"] for m in plan_data["metrics"]}
    sidecar_ids = set(meta_data["metrics"].keys())
    # 旁路反映 plan 实际集合（id 集合一致）
    assert sidecar_ids == plan_metric_ids, (
        f"旁路 metric id 集合 != plan：旁路 {sidecar_ids} vs plan {plan_metric_ids}"
    )

    # 去重契约（多 subject 场景）：plan 的 metrics[] 按 subject 重复，长度 = subject × metric；
    # 旁路按 id 去重，长度 = metric 数。spec 场景 28×5=140 → 旁路 5。
    n_metrics = len(sidecar_ids)
    assert len(plan_data["metrics"]) == 3 * n_metrics, (
        f"3 subject 场景 plan metrics[] 应 = 3×{n_metrics}={3 * n_metrics}，"
        f"实际 {len(plan_data['metrics'])}（按 subject 重复契约）"
    )
    assert len(sidecar_ids) == n_metrics

    # 元数据一致：旁路每条的字段值 == plan 同 id 首条
    for mid, meta in meta_data["metrics"].items():
        first_plan_metric = next(m for m in plan_data["metrics"] if m["id"] == mid)
        for field in (
            "display_name_zh",
            "unit_zh",
            "one_liner",
            "output_unit",
            "direction_for_anxiety",
            "statistical_default",
        ):
            assert field in meta, f"{mid}: 旁路缺字段 {field}"
            assert meta[field] == first_plan_metric[field], (
                f"{mid}.{field}: 旁路值 {meta[field]!r} != plan 首条 {first_plan_metric[field]!r}"
            )


def test_metric_metadata_sidecar_is_small_enough(tmp_path) -> None:
    """旁路体积应远小于施工文件（防回归：去重失效会让旁路膨胀到接近 plan）。

    单 subject 场景两者接近，故这里用「旁路 metrics 是 dict（去重结构）」+ 「每条只 6 字段」
    做结构断言，而非纯字节比较。
    """
    workspace = tmp_path / "workspace"
    _run_prep_epm(tmp_path)
    meta_data = json.loads((workspace / "_metric_metadata.json").read_text())
    for mid, meta in meta_data["metrics"].items():
        # 旁路每条只含 6 个元数据字段，不含施工字段（script/input/output/args/parameters_in_use/subject_index）
        assert set(meta.keys()) == {
            "display_name_zh",
            "unit_zh",
            "one_liner",
            "output_unit",
            "direction_for_anxiety",
            "statistical_default",
        }, f"{mid}: 旁路条目应只含 6 元数据字段，不应含施工字段，实际 {set(meta.keys())}"


# ---------------------------------------------------------------------------
# spec §四 项 4：report-writer / data-analyst prompt 静态契约（防回归）
# ---------------------------------------------------------------------------


def test_report_writer_prompt_reads_sidecar() -> None:
    """report-writer system_prompt 含 _metric_metadata.json 措辞 + thinking 铁律。"""
    p = REPORT_WRITER_CONFIG.system_prompt
    assert "_metric_metadata.json" in p, "report-writer prompt 未指引读 _metric_metadata.json"
    assert "<thinking_discipline>" in p, "report-writer prompt 缺 <thinking_discipline> 铁律"
    # 旁路 fallback 段（spec §2.5）
    assert "<metadata_fallback>" in p, "report-writer prompt 缺 <metadata_fallback>"


def test_report_writer_prompt_drops_old_induction() -> None:
    """report-writer system_prompt 不含旧诱导串「按 metric id 在 metrics[] 数组中匹配」。"""
    p = REPORT_WRITER_CONFIG.system_prompt
    assert "按 metric id 在" not in p, "report-writer prompt 仍含旧诱导串「按 metric id 在...匹配」"
    assert "metrics[] 数组中匹配" not in p, "report-writer prompt 仍含「metrics[] 数组中匹配」"
    # report-writer 不再 read plan_metrics.json（数值在 handoff，元数据在旁路）
    # 检查 workflow 段不再把 plan_metrics.json 列为 read_file 目标。
    assert (
        "read_file /mnt/user-data/workspace/plan_metrics.json" not in p
    ), "report-writer workflow 仍指示 read plan_metrics.json（应改读 _metric_metadata.json）"


def test_data_analyst_prompt_reads_sidecar() -> None:
    """data-analyst system_prompt 含 _metric_metadata.json 措辞 + thinking 铁律。"""
    p = DATA_ANALYST_CONFIG.system_prompt
    assert "_metric_metadata.json" in p, "data-analyst prompt 未指引读 _metric_metadata.json"
    assert "<thinking_discipline>" in p, "data-analyst prompt 缺 <thinking_discipline> 铁律"
    assert "<metadata_fallback>" in p, "data-analyst prompt 缺 <metadata_fallback>"


def test_data_analyst_prompt_drops_old_induction() -> None:
    """data-analyst system_prompt 不含旧诱导串「按 metric id 在 metrics[] 数组中匹配」。"""
    p = DATA_ANALYST_CONFIG.system_prompt
    assert "按 metric id 在" not in p, "data-analyst prompt 仍含旧诱导串「按 metric id 在...匹配」"
    assert "metrics[] 数组中匹配" not in p, "data-analyst prompt 仍含「metrics[] 数组中匹配」"
    # data-analyst 读 plan 唯一目的是判读元数据，应改读旁路
    assert (
        "read_file /mnt/user-data/workspace/plan_metrics.json" not in p
    ), "data-analyst workflow 仍指示 read plan_metrics.json（应改读 _metric_metadata.json）"
