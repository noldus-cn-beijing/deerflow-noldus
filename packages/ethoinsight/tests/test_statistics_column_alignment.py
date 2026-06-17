"""红→绿测试：statistics 路径列对齐缺口（spec 2026-06-16-statistics-path-column-alignment）。

真实 FewZones EPM 原始列名是 `open`/`closed`（用户自定义归属列），EPM 指标函数默认查
`in_zone.*open_arm` 正则不命中 → 返 None。需要 zone overrides（`open_arm_zones=["open"]`）
才能算对。compute 路径（code-executor 逐指标跑）已接，**statistics 路径（dispatcher 批量算）
从未接入这套机制** → EPM 专属指标在 statistics 链全 None。

三组测试独立立证（与第三层 file→subject 桥接分开）：
  1. dispatcher 单元：open/closed 列，不传 zone_overrides → None；传 → 非 None。
  2. resolve 单元：column_aliases → plan.statistics.parameters 含 open_arm_zones=["open"]。
  3. 端到端：FewZones 列名跑 run_groupwise_stats 子进程 → comparisons 含 EPM 专属指标。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# helpers（与 test_metrics_epm.py 同族，但用 FewZones 真实列名 open/closed）
# ============================================================================


def _make_fewzones_epm_df(
    n_frames: int = 100,
    *,
    open_col: str = "open",
    closed_col: str = "closed",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic EPM df with FewZones-style column names (`open`/`closed`).

    默认 `in_zone*` 列不存在 → EPM 指标函数自动检测必返 None，模拟真实 FewZones 数据。
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "velocity": rng.uniform(0, 20, n_frames),
        }
    )
    # 5 open-arm bouts + 5 closed-arm bouts（互斥），用 FewZones 列名
    open_vals = np.zeros(n_frames, dtype=int)
    closed_vals = np.zeros(n_frames, dtype=int)
    for start in [10, 25, 45, 65, 85]:
        open_vals[start : start + 8] = 1
    for start in [18, 35, 55, 75, 93]:
        closed_vals[start : start + 6] = 1
    df[open_col] = open_vals
    df[closed_col] = closed_vals
    return df


def _build_parsed_data(subjects: dict[str, pd.DataFrame]) -> dict:
    combined = pd.concat(
        [df.assign(subject=name) for name, df in subjects.items()], ignore_index=True
    )
    return {
        "subjects": subjects,
        "all_data": combined,
        "file_list": [f"subj_{i}.txt" for i in range(len(subjects))],
        "file_subjects": {f"subj_{i}.txt": name for i, name in enumerate(subjects)},
        "summary": {
            "total_files": len(subjects),
            "total_subjects": len(subjects),
            "total_rows": sum(len(df) for df in subjects.values()),
            "duration_seconds": 600.0,
        },
    }


# ============================================================================
# 1. dispatcher 单元：zone_overrides 透传通道
# ============================================================================


class TestDispatcherZoneOverrides:
    """compute_paradigm_metrics 的 zone_overrides 参数（spec §3.1 断点 2）。"""

    def test_without_zone_overrides_open_metrics_are_none(self):
        """FewZones 列名 + 不传 zone_overrides → EPM 专属指标 None（红锚点）。"""
        from ethoinsight.metrics.dispatcher import compute_paradigm_metrics

        df = _make_fewzones_epm_df()
        parsed = _build_parsed_data({"s1": df, "s2": df.copy()})
        result = compute_paradigm_metrics(parsed, paradigm="epm")
        per = result["per_subject"]
        for subj in per:
            assert per[subj]["open_arm_time_ratio"] is None, (
                "FewZones 列名下不传 zone_overrides，open_arm_time_ratio 必为 None"
            )

    def test_with_zone_overrides_open_metrics_non_none(self):
        """FewZones 列名 + 传 zone_overrides → EPM 专属指标非 None（绿锚点）。"""
        from ethoinsight.metrics.dispatcher import compute_paradigm_metrics

        df = _make_fewzones_epm_df()
        parsed = _build_parsed_data({"s1": df, "s2": df.copy()})
        zone_overrides = {"open_arm_zones": ["open"], "closed_arm_zones": ["closed"]}
        result = compute_paradigm_metrics(
            parsed, paradigm="epm", zone_overrides=zone_overrides
        )
        per = result["per_subject"]
        for subj in per:
            ratio = per[subj]["open_arm_time_ratio"]
            assert ratio is not None, "传 zone_overrides 后 open_arm_time_ratio 必非 None"
            assert 0.0 <= ratio <= 1.0

    def test_zone_overrides_default_none_backward_compatible(self):
        """zone_overrides 默认 None → 行为与改前完全一致（向后兼容）。"""
        from ethoinsight.metrics.dispatcher import compute_paradigm_metrics

        df = _make_fewzones_epm_df()
        parsed = _build_parsed_data({"s1": df})
        # 标准列名 df（in_zone_open_arm_1）→ 自动检测应工作，不传 zone_overrides 不破坏
        from tests.test_metrics_epm import _make_epm_df

        std_df = _make_epm_df(n_frames=50)
        std_parsed = _build_parsed_data({"s1": std_df})
        result = compute_paradigm_metrics(std_parsed, paradigm="epm")
        assert result["per_subject"]["s1"]["open_arm_time_ratio"] is not None

    def test_oft_center_zone_str_override(self):
        """OFT center_zone 是 str（非 list），透传形态天然匹配（spec §2 约束）。"""
        from ethoinsight.metrics.dispatcher import compute_paradigm_metrics

        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n, dtype=float) * 0.04,
                "x_center": rng.uniform(0, 500, n),
                "y_center": rng.uniform(0, 500, n),
                "distance_moved": rng.uniform(0, 5, n),
                "velocity": rng.uniform(0, 20, n),
                "中心区": np.where(np.arange(n) % 3 == 0, 1, 0),  # FewZones 自定义列名
            }
        )
        parsed = _build_parsed_data({"s1": df})
        # 不传 → None（无 in_zone_center 列）
        r_none = compute_paradigm_metrics(parsed, paradigm="open_field")
        assert r_none["per_subject"]["s1"]["center_time_ratio"] is None
        # 传 center_zone=str → 非 None
        r_zone = compute_paradigm_metrics(
            parsed, paradigm="open_field", zone_overrides={"center_zone": "中心区"}
        )
        assert r_zone["per_subject"]["s1"]["center_time_ratio"] is not None


# ============================================================================
# 2. resolve 单元：statistics 段 parameters（spec §3.2 断点 1）
# ============================================================================


class TestResolveStatisticsParameters:
    """resolve_metrics 把 zone_aliases_overrides 投影进 plan.statistics.parameters。"""

    def test_statistics_segment_carries_zone_parameters(self, tmp_path):
        """column_aliases → plan.statistics.parameters 含 open_arm_zones=["open"]（红：当前无此字段）。"""
        from ethoinsight.catalog.resolve import resolve_metrics

        columns = [
            "trial_time", "recording_time",
            "x_center", "y_center",
            "distance_moved", "velocity",
            "open", "closed", "result_1",
        ]
        plan = resolve_metrics(
            paradigm="epm",
            columns=columns,
            raw_files=[
                "/mnt/user-data/uploads/c1.txt",
                "/mnt/user-data/uploads/t1.txt",
                "/mnt/user-data/uploads/t2.txt",
                "/mnt/user-data/uploads/t3.txt",
            ],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
            column_aliases={"open": "open_arms", "closed": "closed_arms"},
            groups_file="/mnt/user-data/workspace/groups.json",
            n_per_group=3,
            n_groups=2,
        )
        assert plan.statistics is not None, "n_per_group=3, n_groups=2 应触发 statistics"
        params = plan.statistics.parameters
        assert "open_arm_zones" in params, (
            f"statistics.parameters 应含 open_arm_zones，实际 {params!r}"
        )
        assert params["open_arm_zones"] == ["open"], params["open_arm_zones"]
        assert params.get("closed_arm_zones") == ["closed"], params.get("closed_arm_zones")

    def test_statistics_parameters_empty_without_aliases(self, tmp_path):
        """无 column_aliases → statistics.parameters 空（向后兼容）。"""
        from ethoinsight.catalog.resolve import resolve_metrics

        columns = [
            "trial_time", "x_center", "y_center",
            "distance_moved", "velocity",
            "in_zone_open_arms_center", "in_zone_closed_arms_center", "result_1",
        ]
        plan = resolve_metrics(
            paradigm="epm",
            columns=columns,
            raw_files=[
                "/mnt/user-data/uploads/c1.txt",
                "/mnt/user-data/uploads/t1.txt",
                "/mnt/user-data/uploads/t2.txt",
                "/mnt/user-data/uploads/t3.txt",
            ],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
            groups_file="/mnt/user-data/workspace/groups.json",
            n_per_group=3,
            n_groups=2,
        )
        assert plan.statistics is not None
        assert plan.statistics.parameters == {}

    def test_statistics_parameters_same_source_as_metrics(self, tmp_path):
        """SSOT：statistics.parameters 是 metrics 同一份 zone_aliases_overrides 的投影。"""
        from ethoinsight.catalog.resolve import resolve_metrics

        columns = [
            "trial_time", "x_center", "y_center",
            "distance_moved", "velocity",
            "open", "closed", "result_1",
        ]
        plan = resolve_metrics(
            paradigm="epm",
            columns=columns,
            raw_files=[
                "/mnt/user-data/uploads/c1.txt",
                "/mnt/user-data/uploads/t1.txt",
                "/mnt/user-data/uploads/t2.txt",
                "/mnt/user-data/uploads/t3.txt",
            ],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
            column_aliases={"open": "open_arms", "closed": "closed_arms"},
            groups_file="/mnt/user-data/workspace/groups.json",
            n_per_group=3,
            n_groups=2,
        )
        # metrics 段的 parameters_in_use 与 statistics 段 parameters 同源
        open_metric = next(m for m in plan.metrics if m.id == "open_arm_time_ratio")
        assert open_metric.parameters_in_use.get("open_arm_zones") == ["open"]
        assert plan.statistics.parameters["open_arm_zones"] == ["open"]


# ============================================================================
# 3. 端到端：run_groupwise_stats 子进程 + FewZones 列名（spec §4.3）
# ============================================================================


def _write_fewzones_trajectory(path: Path) -> None:
    """写一份 FewZones 列名的 EPM 轨迹文件（CSV，列 open/closed）。"""
    df = _make_fewzones_epm_df(n_frames=100)
    df.to_csv(path, index=False, sep="\t")


class TestRunGroupwiseStatsColumnAlignment:
    """端到端：FewZones 列名 + parameters → comparisons 含 EPM 专属指标。

    走完整 CLI 链（argparse → read_inputs/groups → bridge → compute → compare →
    save_output_json），仅用 monkeypatch 注入 parse_batch 的返回（FewZones 列名合成
    数据，绕开 EthoVision 文件格式解析——本测试聚焦 zone 参数透传链而非文件解析）。
    """

    @pytest.fixture
    def fewzones_files(self, tmp_path, monkeypatch):
        """造 4 份 FewZones 轨迹文件 + patch parse_batch 返回 FewZones 列名 parsed_data。"""
        files = [str(tmp_path / f"subj_{i}.txt") for i in range(4)]
        for p in files:
            _write_fewzones_trajectory(Path(p))

        # patch 脚本内 parse_batch：读文件（已 resolve 成真实路径），但把列名换成
        # FewZones 形态（open/closed），file_subjects 用文件路径→stem 映射。
        from ethoinsight.scripts.epm import run_groupwise_stats as stats_mod

        def _fake_parse_batch(paths):
            subjects = {}
            file_subjects = {}
            for p in paths:
                import pandas as _pd

                df = _pd.read_csv(p, sep="\t")
                df.attrs["subject"] = Path(p).stem
                key = Path(p).stem
                subjects[key] = df
                file_subjects[p] = key
            combined = _pd.concat(
                [df.assign(subject=k) for k, df in subjects.items()], ignore_index=True
            )
            return {
                "subjects": subjects,
                "all_data": combined,
                "file_list": list(paths),
                "file_subjects": file_subjects,
                "summary": {
                    "total_files": len(paths),
                    "total_subjects": len(subjects),
                    "total_rows": sum(len(d) for d in subjects.values()),
                    "duration_seconds": 600.0,
                },
            }

        monkeypatch.setattr(stats_mod, "parse_batch", _fake_parse_batch)
        return files

    def test_comparisons_include_epm_metrics_with_parameters(self, tmp_path, fewzones_files):
        from ethoinsight.scripts.epm.run_groupwise_stats import main as stats_main

        files = fewzones_files
        inputs_json = tmp_path / "inputs.json"
        inputs_json.write_text(json.dumps(files))
        # SSOT 形态 {file: group}，read_groups_json 会反转
        groups_json = tmp_path / "groups.json"
        groups_json.write_text(
            json.dumps({files[0]: "control", files[1]: "control",
                        files[2]: "treatment", files[3]: "treatment"})
        )
        out_json = tmp_path / "stats.json"

        parameters = {"open_arm_zones": ["open"], "closed_arm_zones": ["closed"]}
        argv = [
            "--inputs", str(inputs_json),
            "--groups", str(groups_json),
            "--output", str(out_json),
            "--parameters-json", json.dumps(parameters),
        ]
        rc = stats_main(argv)
        assert rc == 0
        payload = json.loads(out_json.read_text())
        comparisons = payload.get("comparisons", {})
        # EPM 专属指标应在 comparisons 里（不传 parameters 时这些会被 None 过滤掉）
        assert "open_arm_time_ratio" in comparisons, (
            f"comparisons 应含 open_arm_time_ratio，实际 keys={list(comparisons)}"
        )

    def test_comparisons_miss_epm_metrics_without_parameters(self, tmp_path, fewzones_files):
        """不传 parameters → EPM 专属指标 None 被过滤，comparisons 缺该 key（红锚点反向坐实）。"""
        from ethoinsight.scripts.epm.run_groupwise_stats import main as stats_main

        files = fewzones_files
        inputs_json = tmp_path / "inputs.json"
        inputs_json.write_text(json.dumps(files))
        groups_json = tmp_path / "groups.json"
        groups_json.write_text(
            json.dumps({files[0]: "control", files[1]: "control",
                        files[2]: "treatment", files[3]: "treatment"})
        )
        out_json = tmp_path / "stats.json"

        argv = ["--inputs", str(inputs_json), "--groups", str(groups_json), "--output", str(out_json)]
        rc = stats_main(argv)
        assert rc == 0
        payload = json.loads(out_json.read_text())
        comparisons = payload.get("comparisons", {})
        assert "open_arm_time_ratio" not in comparisons, (
            "FewZones 列名 + 不传 parameters，open_arm_time_ratio 应为 None 被过滤掉"
        )
