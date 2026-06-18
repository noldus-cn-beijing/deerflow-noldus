"""dispatcher group_summary 按文件路径桥接到 subject_key（box plot "No data" 回归）。

根因（2026-06-18 第四轮 EPM dogfood）：box plot / 直调 compute_paradigm_metrics 的
groups 成员是**文件路径**（groups.json 里的 /mnt/.../Trial 1.xlsx），而 per_subject 的
key 是 EV19 对象名（常为空串 "" / 合成 "_1"/"_2"…）。旧 dispatcher 在 group_summary
聚合时直接 `s in per_subject` 匹配 → 文件路径与 subject_key 零交集 → 每组 matched 恒空
→ group_summary={} → 箱线图对每个 metric 渲染 "No data"（charts.py:_extract_group_data
拿不到任何 group）。

修复：parse_batch 返回的 file_subjects {path: subject_key} 是 file→subject 权威桥
（与 statistics 路径同源，spec 2026-06-16）。dispatcher 聚合前先按文件路径桥接，桥不到
再按 subject_key 直配——两种 group 成员形态都正确。

statistics 路径当初已修（compare_groups 内桥接），但 box plot 走 compute_paradigm_metrics
直调这条路从未接上同一个桥——这是该 bug 家族的第 11 个结构缺口。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ethoinsight.metrics import compute_paradigm_metrics


def _make_epm_df(n_open_frames: int = 8, n_frames: int = 100, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    oa = np.zeros(n_frames, dtype=int)
    oa[10 : 10 + n_open_frames] = 1
    ca = np.zeros(n_frames, dtype=int)
    ca[30:50] = 1
    return pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "velocity": rng.uniform(0, 20, n_frames),
            "in_zone_open_arm_1": oa,
            "in_zone_closed_arm_1": ca,
            "in_zone_center-point": np.ones(n_frames, dtype=int),
        }
    )


def _build_parsed_with_file_subjects(
    subjects: dict[str, pd.DataFrame],
    file_subjects: dict[str, str],
) -> dict:
    """parsed_data with the EV19-style empty/synthetic subject keys + file_subjects bridge."""
    return {
        "subjects": subjects,
        "file_subjects": file_subjects,
        "summary": {
            "total_files": len(file_subjects),
            "total_subjects": len(subjects),
            "duration_seconds": 4.0,
        },
    }


class TestGroupSummaryFilePathBridge:
    """groups 成员为文件路径时，group_summary 必须经 file_subjects 桥接、非空。"""

    def _fixture(self):
        # per_subject keyed by EV19 object names — first is empty string, rest synthetic.
        # This is exactly the shape parse_batch produces when EV19 names are blank.
        subjects = {
            "": _make_epm_df(n_open_frames=8, seed=1),
            "_1": _make_epm_df(n_open_frames=4, seed=2),
            "_2": _make_epm_df(n_open_frames=20, seed=3),
            "_3": _make_epm_df(n_open_frames=2, seed=4),
        }
        # groups.json-style: members are FILE PATHS, not subject keys.
        paths = [
            "/mnt/user-data/uploads/Trial 1.xlsx",
            "/mnt/user-data/uploads/Trial 2.xlsx",
            "/mnt/user-data/uploads/Trial 3.xlsx",
            "/mnt/user-data/uploads/Trial 4.xlsx",
        ]
        file_subjects = dict(zip(paths, ["", "_1", "_2", "_3"]))
        groups = {"control": paths[:2], "treatment": paths[2:]}
        parsed = _build_parsed_with_file_subjects(subjects, file_subjects)
        return parsed, groups

    def test_filepath_groups_produce_nonempty_group_summary(self):
        """REGRESSION: 文件路径 groups → group_summary 非空（旧代码这里恒空 → 箱线图 No data）。"""
        parsed, groups = self._fixture()
        result = compute_paradigm_metrics(parsed, "epm", groups=groups, zone_overrides={"open_arm_zones": ["in_zone_open_arm_1"]})
        gs = result["group_summary"]
        assert gs, "group_summary 不应为空（文件路径未桥接 = box plot No data 的根因）"
        assert set(gs.keys()) == {"control", "treatment"}

    def test_filepath_groups_metric_values_present_and_correct_n(self):
        """两组的 open_arm_time_ratio 都有值，且 n 与文件数一致（control=2, treatment=2）。"""
        parsed, groups = self._fixture()
        result = compute_paradigm_metrics(parsed, "epm", groups=groups, zone_overrides={"open_arm_zones": ["in_zone_open_arm_1"]})
        gs = result["group_summary"]
        for grp in ("control", "treatment"):
            assert "open_arm_time_ratio" in gs[grp], f"{grp} 缺 open_arm_time_ratio（box plot 会显示 No data）"
            assert gs[grp]["open_arm_time_ratio"]["n"] == 2
            assert len(gs[grp]["open_arm_time_ratio"]["values"]) == 2

    def test_subject_key_groups_still_work_backward_compat(self):
        """向后兼容：groups 成员已是 subject_key（非文件路径）时仍正确匹配（桥不到 → 直配）。"""
        parsed, _ = self._fixture()
        # Pass subject keys directly (the statistics path / pre-bridge callers).
        groups = {"g1": ["", "_1"], "g2": ["_2", "_3"]}
        result = compute_paradigm_metrics(parsed, "epm", groups=groups, zone_overrides={"open_arm_zones": ["in_zone_open_arm_1"]})
        gs = result["group_summary"]
        assert set(gs.keys()) == {"g1", "g2"}
        assert gs["g1"]["open_arm_time_ratio"]["n"] == 2
        assert gs["g2"]["open_arm_time_ratio"]["n"] == 2

    def test_no_file_subjects_falls_back_to_subject_key_match(self):
        """parsed_data 无 file_subjects（老数据）时不报错，按 subject_key 直配。"""
        subjects = {"S0": _make_epm_df(seed=1), "S1": _make_epm_df(seed=2)}
        parsed = {
            "subjects": subjects,
            # no file_subjects key at all
            "summary": {"total_subjects": 2, "duration_seconds": 4.0},
        }
        result = compute_paradigm_metrics(parsed, "epm", groups={"g": ["S0", "S1"]}, zone_overrides={"open_arm_zones": ["in_zone_open_arm_1"]})
        assert result["group_summary"]["g"]["open_arm_time_ratio"]["n"] == 2
