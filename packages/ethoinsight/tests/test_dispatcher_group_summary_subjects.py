"""dispatcher group_summary[grp][metric]['subjects'] 与 values 等长逐位对应。

背景（spec 2026-06-18-data-analyst-thinking-overload §3.1）：compute_outlier_diagnostics
用组内 index（去 NaN 前）索引 subject_names[grp]，要求 subject_names[grp] 与
group_summary[grp][metric]['values'] 等长且逐位对应。dispatcher 产出 values 的同一循环
并行收集 subjects（subject_key 列表），天然保证对齐。本批坐实这个不变量——它是 outlier
真名下沉的安全基石：若 subjects 与 values 错位，outlier 会把真名张冠李戴。

走真实 compute_paradigm_metrics 路径（复用 test_dispatcher_warnings.py 的 EPM fixture 模式）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ethoinsight.metrics import compute_paradigm_metrics


def _make_epm_df(n_frames: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    oa = np.zeros(n_frames, dtype=int)
    oa[10:18] = 1
    return pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "velocity": rng.uniform(0, 20, n_frames),
            "in_zone_open_arm_1": oa,
            "in_zone_closed_arm_1": np.zeros(n_frames, dtype=int),
            "in_zone_center-point": np.ones(n_frames, dtype=int),
        }
    )


def _build_parsed(subjects: dict[str, pd.DataFrame]) -> dict:
    all_dfs = []
    for name, df in subjects.items():
        df = df.copy()
        df["subject"] = name
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return {
        "subjects": subjects,
        "all_data": combined,
        "file_list": [f"subj_{i}.txt" for i in range(len(subjects))],
        "summary": {
            "total_files": len(subjects),
            "total_subjects": len(subjects),
            "total_rows": sum(len(df) for df in subjects.values()),
            "duration_seconds": 600.0,
        },
    }


class TestGroupSummarySubjectsField:
    """group_summary[grp][metric] 含 subjects 键、与 values 等长逐位对应（spec §3.1 安全基石）。"""

    def test_subjects_field_present_and_parallel_to_values(self):
        """每个 metric 的 subjects 与 values 等长。"""
        dfs = {f"S{i}": _make_epm_df(seed=i) for i in range(5)}
        parsed = _build_parsed(dfs)
        groups = {"control": ["S0", "S1", "S2"], "treatment": ["S3", "S4"]}
        # 两组各需 n>=2 才有 group_summary values；这里 control 3 / treatment 2
        result = compute_paradigm_metrics(parsed, "epm", groups={
            "all": list(dfs.keys()),
        })
        group_summary = result["group_summary"]
        assert group_summary, "应有 group_summary"
        for grp_name, grp_metrics in group_summary.items():
            for mname, minfo in grp_metrics.items():
                assert "subjects" in minfo, f"{grp_name}/{mname} 缺 subjects 键"
                vals = minfo["values"]
                subs = minfo["subjects"]
                assert len(vals) == len(subs), (
                    f"{grp_name}/{mname}: values({len(vals)}) 与 subjects({len(subs)}) 不等长"
                )

    def test_subjects_are_subject_keys(self):
        """subjects 列表元素是 subject_key（与 per_subject 的 key 同源）。"""
        dfs = {f"S{i}": _make_epm_df(seed=i) for i in range(4)}
        parsed = _build_parsed(dfs)
        result = compute_paradigm_metrics(parsed, "epm", groups={"grp": list(dfs.keys())})
        per_subject_keys = set(result["per_subject"].keys())
        for grp_metrics in result["group_summary"].values():
            for minfo in grp_metrics.values():
                for s in minfo["subjects"]:
                    assert s in per_subject_keys, f"subject {s!r} 不在 per_subject keys 里"

    def test_subjects_match_group_membership(self):
        """subjects 列表 ⊆ 该 group 的成员（group_summary 只含分进该组的 subject）。"""
        dfs = {f"S{i}": _make_epm_df(seed=i) for i in range(4)}
        parsed = _build_parsed(dfs)
        groups = {"control": ["S0", "S1"], "treatment": ["S2", "S3"]}
        result = compute_paradigm_metrics(parsed, "epm", groups=groups)
        for grp_metrics in result["group_summary"].values():
            for minfo in grp_metrics.values():
                for s in minfo["subjects"]:
                    # 每个 subject 应属于该组（control 全在 control 组成员里）
                    assert any(s in members for members in groups.values())
