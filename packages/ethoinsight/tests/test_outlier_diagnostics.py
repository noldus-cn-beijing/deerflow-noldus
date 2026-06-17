"""Tests for outlier + leave-one-out diagnostics pushed down to the statistics layer.

背景（spec 2026-06-17-data-analyst-loo-counterfactual-pushdown）：data-analyst 曾在
自然语言推理里手算 leave-one-out 反事实（反复验算耗尽预算）。把 outlier 识别 + LOO
下沉为 `compute_outlier_diagnostics` 纯确定性函数，由 `compare_groups` 产出并落盘进
statistics.json，data-analyst 只读不算。

判据与 data-analyst prompt step 2.7 b 字面一致：偏离组均值 ≥ 1.5 SD，或偏离组中位数
≥ 2×（取并集）。
"""

from __future__ import annotations

import numpy as np

from ethoinsight import statistics
from ethoinsight.statistics import compute_outlier_diagnostics


# ---------------------------------------------------------------------------
# 纯函数 compute_outlier_diagnostics
# ---------------------------------------------------------------------------


class TestComputeOutlierDiagnostics:
    def test_flags_high_sd_outlier(self):
        """一个 ≥1.5 SD 的离群 subject 必须被标，LOO 数值手工可核验。"""
        # 9 个 ~10 + 1 个 100：100 远超 1.5 SD
        group_values = {"control": [10, 10, 10, 10, 9, 11, 10, 10, 10, 100]}
        diags = compute_outlier_diagnostics(group_values)
        assert len(diags) == 1
        d = diags[0]
        assert d["group"] == "control"
        assert d["value"] == 100.0
        assert d["deviation_sd"] >= 1.5

        # LOO：排除 100 后只剩 9 个 ~10 的值
        loo_expected = np.array([10, 10, 10, 10, 9, 11, 10, 10, 10], dtype=float)
        assert abs(d["loo_mean"] - float(loo_expected.mean())) < 1e-9
        assert abs(d["loo_std"] - float(loo_expected.std(ddof=1))) < 1e-9
        # counterfactual 串必须含组名 + 变化方向信息，data-analyst 原样引用
        assert isinstance(d["counterfactual"], str)
        assert "control" in d["counterfactual"]

    def test_median_ratio_rule(self):
        """≥2× median 但 <1.5 SD 的离群（并集判据的另一支）必须被标。"""
        # 中位数 10，值 100 是 10× median；但若 SD 也被这个值拉大，deviation_sd 可能
        # 不超 1.5——构造一组让 median ratio 命中、SD 判据不命中的数据。
        # [1,1,1,1,1,1,1,1,1,100]：median=1，100/1=100≥2×；SD 判据看 deviation_sd。
        group_values = {"treatment": [1, 1, 1, 1, 1, 1, 1, 1, 1, 100]}
        diags = compute_outlier_diagnostics(group_values)
        values = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 100], dtype=float)
        median = float(np.median(values))
        d_by_value = {d["value"]: d for d in diags}
        assert 100.0 in d_by_value, "≥2× median 的离群必须被标"
        d = d_by_value[100.0]
        # 命中 median 支
        assert d["deviation_median_ratio"] >= 2.0
        assert abs(d["deviation_median_ratio"] - (100.0 / median)) < 1e-9

    def test_no_outlier_uniform(self):
        """均匀分布无离群 → 空列表。"""
        group_values = {"control": [10.0, 10.0, 10.0, 10.0, 10.0]}
        diags = compute_outlier_diagnostics(group_values)
        assert diags == []

    def test_loo_matches_manual_dogfood(self):
        """真实 dogfood control 值 → 库算 LOO == data-analyst trace 里手算的 0.1285。

        等价性探针：证明下沉后库算结果与之前手算结果一致，杜绝行为漂移。
        """
        # spec §4.1 line 131 的真实 dogfood control open_arm_time_ratio
        control = [0.099, 0.084, 1.0, 0.258, 0.133, 0.138, 0.059]
        group_values = {"control": control}
        diags = compute_outlier_diagnostics(group_values)
        # 只有 1.0 这个值是离群（2.23 SD）
        by_value = {d["value"]: d for d in diags}
        assert 1.0 in by_value, "1.0 应被标为离群（≥1.5 SD 且 ≥2× median）"
        d = by_value[1.0]
        # spec 明确：排除 1.0 后 mean ≈ 0.1285（±1e-3）
        assert abs(d["loo_mean"] - 0.1285) < 1e-3
        # 完整 control 组均值 0.253
        assert abs(d["group_mean"] - 0.253) < 1e-3

    def test_empty_and_singleton_groups_skipped(self):
        """<2 值的组无法算 SD/LOO，必须跳过（不抛、不产幽灵条目）。"""
        group_values = {
            "control": [5.0],  # 单值组
            "treatment": [],  # 空组
            "other": [10.0, 10.0, 10.0, 10.0],  # 正常组：全相同，无离群
        }
        diags = compute_outlier_diagnostics(group_values)
        # control/treatment 被跳过，other 全相同无离群
        assert diags == []

    def test_subject_identifier_index_fallback(self):
        """compare_groups 拿不到 group→subject 名映射（dispatcher 落盘时丢了），
        组内 index 兜底：subject 写 `subject #i`，i 是该 subject 在组 values 里的位置。

        subject 真名映射是 Issue #98 列对齐家族的另一轴，本 spec 不引入（见 spec §6.1）。
        """
        # 100 在组内 index 3
        group_values = {"control": [10, 10, 10, 100]}
        diags = compute_outlier_diagnostics(group_values)
        assert len(diags) == 1
        d = diags[0]
        assert "subject" in d
        assert d["subject"] == "subject #3"

    def test_subject_identifier_real_names_when_provided(self):
        """纯函数支持可选 subject_names：调用方若有真名则透传，data-analyst 引用更可读。"""
        group_values = {"control": [10, 10, 10, 100]}
        diags = compute_outlier_diagnostics(
            group_values, subject_names={"control": ["c0", "c1", "c2", "c3"]}
        )
        assert diags[0]["subject"] == "c3"


# ---------------------------------------------------------------------------
# compare_groups 接线：附加 outlier_diagnostics，不破坏 comparisons schema
# ---------------------------------------------------------------------------


def _metrics_result_with(values_by_group: dict[str, list[float]], metric: str = "m") -> dict:
    """构造与 dispatcher.compute_paradigm_metrics 输出兼容的最小 metrics_result。"""
    group_summary: dict[str, dict] = {}
    for grp, vals in values_by_group.items():
        arr = np.array(vals, dtype=float)
        group_summary[grp] = {
            metric: {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n": len(arr),
                "values": [float(v) for v in arr],
            }
        }
    return {"group_summary": group_summary}


class TestCompareGroupsAttachesOutlierDiagnostics:
    def test_outlier_diagnostics_key_present(self):
        metrics_result = _metrics_result_with(
            {
                "control": [10, 10, 10, 10, 9, 11, 10, 100],
                "treatment": [20, 20, 20, 21, 19, 20, 20, 20],
            }
        )
        result = statistics.compare_groups(metrics_result)
        assert "outlier_diagnostics" in result
        diags = result["outlier_diagnostics"]
        # 控制组 100 是离群
        flagged_values = {d["value"] for d in diags}
        assert 100.0 in flagged_values

    def test_comparisons_schema_unchanged(self):
        """新增 outlier_diagnostics 不得污染 comparisons 既有 schema（charts.py 消费）。"""
        metrics_result = _metrics_result_with(
            {
                "control": [10, 11, 9, 10, 10],
                "treatment": [20, 21, 19, 20, 20],
            }
        )
        result = statistics.compare_groups(metrics_result)
        # comparisons schema 必须原样：metric -> list[dict] with group1/group2/p_value
        assert "comparisons" in result
        comp = result["comparisons"]["m"][0]
        for key in ("group1", "group2", "p_value"):
            assert key in comp, f"comparisons schema 残缺：缺 {key}"
        # outlier_diagnostics 与 comparisons 并列，不塞进每条 comparison
        assert "outlier_diagnostics" not in comp
        assert isinstance(result["outlier_diagnostics"], list)

    def test_outlier_diagnostics_index_fallback_in_compare_groups(self):
        """compare_groups 默认拿不到 group→subject 名映射 → 组内 index 兜底。

        control 的 1.0 在 index 2（[0.099,0.084,1.0,...]）；counterfactual 用 `subject #2`。
        """
        control_vals = [0.099, 0.084, 1.0, 0.258, 0.133, 0.138, 0.059]
        metrics_result = _metrics_result_with(
            {"control": control_vals, "treatment": [0.05, 0.06, 0.05, 0.07]}
        )
        result = statistics.compare_groups(metrics_result)
        diags = result["outlier_diagnostics"]
        subjects = {d["subject"] for d in diags}
        assert "subject #2" in subjects  # control 的 1.0
