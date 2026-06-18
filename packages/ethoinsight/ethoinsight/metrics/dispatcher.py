"""compute_paradigm_metrics 派发器：按 paradigm 路由到 metrics/<范式>.py 的函数。"""

from __future__ import annotations

import functools
import warnings

import numpy as np

from ethoinsight.metrics._common import compute_distance_moved, compute_velocity_stats
from ethoinsight.metrics.oft import (
    compute_center_time_ratio,
    compute_thigmotaxis_index,
    compute_center_distance_ratio,
    compute_center_entry_count,
)
from ethoinsight.metrics.epm import (
    compute_open_arm_time_ratio,
    compute_open_arm_entry_count,
    compute_open_arm_entry_ratio,
    compute_open_arm_time,
    compute_total_entry_count,
)
from ethoinsight.metrics.zero_maze import (
    compute_open_zone_time_ratio,
    compute_open_zone_time,
    compute_open_zone_distance,
    compute_hesitation_count,
)
from ethoinsight.metrics.ldb import (
    compute_light_time_ratio,
    compute_transition_count,
    compute_light_latency,
)
from ethoinsight.metrics.fst import (
    compute_immobility_time_fst,
    compute_immobility_latency_fst,
    compute_immobility_bout_count_fst,
)
from ethoinsight.metrics.tst import (
    compute_immobility_time_tst,
    compute_immobility_latency_tst,
    compute_immobility_bout_count_tst,
)


# ============================================================================
# Sprint 2a: catalog-backed parameter lookup
# ============================================================================


@functools.lru_cache(maxsize=16)
def _get_shared_param(param_name: str, default: float | int | str) -> float | int | str:
    """从 _common.yaml.shared_parameters 读参数,失败时回退到 default 并 warn。"""
    try:
        from ethoinsight.catalog.loader import load_common_catalog
        common = load_common_catalog()
        spec = common.shared_parameters.parameters.get(param_name)
        if spec is not None:
            return spec.default
    except Exception as e:
        warnings.warn(f"catalog load failed for shared param '{param_name}': {e}", stacklevel=2)
    return default


@functools.lru_cache(maxsize=32)
def _get_paradigm_param(paradigm: str, param_name: str, default: float | int | str) -> float | int | str:
    """从 <paradigm>.yaml.paradigm_parameters 读参数,失败时回退到 default 并 warn。"""
    try:
        from ethoinsight.catalog.loader import load_catalog
        cat = load_catalog(paradigm)
        spec = cat.paradigm_parameters.parameters.get(param_name)
        if spec is not None:
            return spec.default
    except Exception as e:
        warnings.warn(f"catalog load failed for paradigm '{paradigm}' param '{param_name}': {e}", stacklevel=2)
    return default


# ============================================================================
# Paradigm dispatcher
# ============================================================================


def compute_paradigm_metrics(
    parsed_data: dict,
    paradigm: str,
    groups: dict[str, list[str]] | None = None,
    metrics: list[str] | None = None,
    zone_overrides: dict[str, list[str] | str] | None = None,
) -> dict:
    """Compute metrics for a specific paradigm.

    Args:
        parsed_data: Output of ``parse.parse_batch()``.
        paradigm: Paradigm name (e.g. "epm", "open_field", "zero_maze",
            "light_dark_box", "forced_swim"). v0.1 仅支持这 5 个。
        groups: Optional grouping ``{group_name: [subject_name, ...]}``.
            If None, all subjects are in a single "all" group.
        metrics: Optional list of metric names to compute.
            If None, computes all metrics for the paradigm.
        zone_overrides: Optional zone column overrides keyed by the底层 metric
            function's kwarg name（e.g. ``{"open_arm_zones": ["open"],
            "center_zone": "中心区"}``）。形态（list vs str）由各范式 catalog 的
            ``wrap_list`` 决定，与底层函数签名天然匹配。None 或缺 key 时底层函数
            走原有自动检测，完全向后兼容。这是 statistics 路径接入列对齐机制的透传
            通道（compute 路径经 ``parameters_in_use`` 已接，spec 2026-06-16）。

    Returns:
        {
            "paradigm": str,
            "per_subject": {subject: {metric: value, ...}, ...},
            "group_summary": {group: {metric: {"mean", "std", "n", "values", "subjects"}, ...}, ...},
            "timeseries": {metric_name: DataFrame, ...},
            "metadata": {"n_subjects", "n_files", "duration_s", "computed_metrics"},
        }
    """
    subjects = parsed_data.get("subjects", {})
    summary = parsed_data.get("summary", {})

    # Default: all subjects in one group
    if groups is None:
        groups = {"all": list(subjects.keys())}

    # Compute per-subject scalar metrics
    # zone_overrides 透传：key 用底层函数的 kwarg 名（open_arm_zones / center_zone /
    # open_zones / light_zone 等），与 resolve 的 zone_aliases_overrides 输出 key 同源。
    # 形态（list/str）由各范式 catalog 的 wrap_list 决定，与底层函数签名天然匹配。
    #
    # 只在 zo 有非空值时透传对应 kwarg；缺 key 时**不传**该 kwarg，让底层函数用各自默认值
    # 走原有自动检测（EPM/zero_maze 默认 None=自动检测；OFT/ldb 默认是具体列名字符串，
    # 显式传 None 会破坏 `in df.columns` 成员测试 → 必须用"不传"而非"传 None"）。
    # 这是 statistics 路径接入列对齐机制的透传通道（compute 路径经 parameters_in_use 已接，
    # spec 2026-06-16）。
    zo = zone_overrides or {}

    def _zone_kwargs(keys: list[str]) -> dict:
        """从 zo 取存在的 zone 参数，缺/空 key 不进 kwargs（让底层用默认）。"""
        return {k: zo[k] for k in keys if zo.get(k)}

    per_subject: dict[str, dict] = {}
    for name, df in subjects.items():
        m: dict[str, float | dict | None] = {}
        m["distance_moved"] = compute_distance_moved(df)
        m["velocity_stats"] = compute_velocity_stats(df)
        if paradigm == "open_field":
            zk = _zone_kwargs(["center_zone"])
            m["center_time_ratio"] = compute_center_time_ratio(df, **zk)
            m["thigmotaxis_index"] = compute_thigmotaxis_index(df)  # 几何指标，无 zone 注入点
            m["center_distance_ratio"] = compute_center_distance_ratio(df, **zk)
            m["center_entry_count"] = compute_center_entry_count(df, **zk)
        elif paradigm == "epm":
            zk = _zone_kwargs(["open_arm_zones", "closed_arm_zones"])
            m["open_arm_time_ratio"] = compute_open_arm_time_ratio(df, **_zone_kwargs(["open_arm_zones"]))
            m["open_arm_entry_ratio"] = compute_open_arm_entry_ratio(df, **zk)
            m["open_arm_entry_count"] = compute_open_arm_entry_count(df, **_zone_kwargs(["open_arm_zones"]))
            m["open_arm_time"] = compute_open_arm_time(df, **_zone_kwargs(["open_arm_zones"]))
            m["total_entry_count"] = compute_total_entry_count(df, **zk)
        elif paradigm == "zero_maze":
            zk = _zone_kwargs(["open_zones", "closed_zones"])
            ozk = _zone_kwargs(["open_zones"])
            m["open_zone_time_ratio"] = compute_open_zone_time_ratio(df, **ozk)
            m["open_zone_time"] = compute_open_zone_time(df, **ozk)
            m["open_zone_distance"] = compute_open_zone_distance(df, **ozk)
            m["hesitation_count"] = compute_hesitation_count(df, **zk)
        elif paradigm == "light_dark_box":
            ldk = _zone_kwargs(["light_zone", "dark_zone"])
            lk = _zone_kwargs(["light_zone"])
            m["light_time_ratio"] = compute_light_time_ratio(df, **lk)
            m["transition_count"] = compute_transition_count(df, **ldk)
            m["light_latency"] = compute_light_latency(df, **lk)
        elif paradigm == "forced_swim":
            m["immobility_time"] = compute_immobility_time_fst(df)
            m["immobility_latency"] = compute_immobility_latency_fst(df)
            m["immobility_bout_count"] = compute_immobility_bout_count_fst(df)
        elif paradigm == "tail_suspension":
            m["immobility_time"] = compute_immobility_time_tst(df)
            m["immobility_latency"] = compute_immobility_latency_tst(df)
            m["immobility_bout_count"] = compute_immobility_bout_count_tst(df)
        per_subject[name] = m

    # Group-level timeseries / metrics — reserved for paradigms that produce group-aggregate
    # output (e.g. shoaling IID/NND/polarity once that paradigm is re-introduced).
    # v0.1 paradigms (epm/oft/zero_maze/ldb/fst) are all per-subject scalar, so these are empty.
    timeseries: dict = {}
    group_level_metrics: dict = {}

    # Filter metrics if requested
    if metrics:
        for name in per_subject:
            per_subject[name] = {
                k: v for k, v in per_subject[name].items() if k in metrics
            }

    # Build group summary
    #
    # group 成员可能是两种形态之一：
    #   (a) subject_key（per_subject 的 key，如 EV19 对象名 ""/"_1"…）——statistics
    #       路径在 compare_groups 内已自行桥接后传入；
    #   (b) **文件路径**（如 groups.json / chart 脚本传的 /mnt/.../Trial 1.xlsx）——
    #       这是 box plot / 直调 compute_paradigm_metrics 的常见形态。
    # per_subject 的 key 永远是 subject_key（b 形态与之零交集 → 旧代码 matched 恒空 →
    # group_summary={} → 箱线图 "No data"）。parse_batch 返回的 file_subjects
    # {path: subject_key} 是 file→subject 的权威桥（与 statistics 路径同源，spec
    # 2026-06-16）。下面对每个 group 成员先按文件路径桥接，桥不到再按 subject_key 直配，
    # 两形态都正确。
    file_subjects = parsed_data.get("file_subjects", {}) or {}

    def _to_subject_keys(members: list[str]) -> list[str]:
        """把 group 成员（文件路径或 subject_key）统一解析为 per_subject 的 key。"""
        resolved: list[str] = []
        for m in members:
            if m in file_subjects:            # 文件路径 → 桥接到 subject_key
                resolved.append(file_subjects[m])
            else:                              # 已是 subject_key（或老调用方）
                resolved.append(m)
        return resolved

    group_summary: dict[str, dict] = {}
    for grp_name, grp_subjects in groups.items():
        grp_metrics: dict[str, dict] = {}
        resolved_subjects = _to_subject_keys(grp_subjects)
        matched = [s for s in resolved_subjects if s in per_subject]
        if not matched:
            continue
        # Collect all scalar metric names
        all_metric_names = set()
        for s in matched:
            for k, v in per_subject[s].items():
                if isinstance(v, (int, float)) and v is not None:
                    all_metric_names.add(k)

        for mname in sorted(all_metric_names):
            # values 与 subjects 并行收集：同一过滤条件，保证两列表逐位对应、等长。
            # subjects 是 subject_key（EV19 对象名称，常为空串）——真名美化（文件名 stem）
            # 由 statistics runner 持有的 subject_label_map 在 compare_groups 层翻译；
            # dispatcher 内部拿不到文件路径，只保证 subject_key 列表与 values 对齐
            # （spec 2026-06-18 outlier 真名下沉：消除 data-analyst thinking 里的 subject 映射）。
            values: list[float] = []
            pair_subjects: list[str] = []
            for s in matched:
                v = per_subject[s].get(mname)
                if v is None or not isinstance(v, (int, float)):
                    continue
                values.append(v)
                pair_subjects.append(s)
            if values:
                arr = np.array(values, dtype=float)
                grp_metrics[mname] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "n": len(arr),
                    "values": [float(v) for v in arr],
                    # 与 values 逐位对应的 subject_key 列表；compare_groups 据此把
                    # compute_outlier_diagnostics 的组内 index 解析回具体 subject。
                    "subjects": pair_subjects,
                }
        group_summary[grp_name] = grp_metrics

    computed = sorted(
        {k for subj in per_subject.values() for k, v in subj.items() if v is not None}
    )

    # Data quality warnings — surface critical sample-size concerns without blocking.
    data_quality_warnings: list[dict] = []
    for grp_name, grp_metrics in group_summary.items():
        if not grp_metrics:
            continue
        # All metrics within a group share the same n (they come from the same subjects)
        sample_n = next(iter(grp_metrics.values())).get("n", 0)
        if sample_n < 3:
            data_quality_warnings.append(
                {
                    "severity": "critical",
                    "code": "SAMPLE.TOO_SMALL",
                    "metric": "all",
                    "message": (
                        f"Group '{grp_name}' has n={sample_n} (<3). "
                        "Statistical inference will be unreliable; descriptive statistics only."
                    ),
                    "evidence": {
                        "n": sample_n,
                        "threshold": 3,
                        "group": grp_name,
                    },
                    "blocks_downstream": True,
                }
            )
    if paradigm == "epm":
        _epm_sample_threshold = _get_shared_param("sample_size_underpowered_threshold", 5)
        _epm_motor_threshold = _get_paradigm_param("epm", "motor_low_entries_threshold", 8)
        for grp_name, grp_metrics in group_summary.items():
            if not grp_metrics:
                continue
            sample_n = next(iter(grp_metrics.values())).get("n", 0)
            if 0 < sample_n < _epm_sample_threshold:
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "code": "SAMPLE.UNDERPOWERED",
                        "metric": "all",
                        "message": (
                            f"Group '{grp_name}' has n={sample_n} (<{_epm_sample_threshold}). "
                            "统计功效不足，结论需谨慎。"
                        ),
                        "evidence": {
                            "n": sample_n,
                            "threshold": _epm_sample_threshold,
                            "paradigm": "epm",
                            "group": grp_name,
                        },
                        "blocks_downstream": False,
                    }
                )
        for name, m in per_subject.items():
            te = m.get("total_entry_count")
            if te is not None and isinstance(te, (int, float)) and te < _epm_motor_threshold:
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "code": "MOTOR.LOW_ENTRIES",
                        "metric": "total_entry_count",
                        "message": (
                            f"Subject '{name}' 总进臂次数={int(te)} (<{_epm_motor_threshold})。"
                            "开臂指标的下降可能为运动抑制而非焦虑增加，需标注警告。"
                        ),
                        "evidence": {
                            "subject": name,
                            "total_entry_count": int(te),
                            "threshold": _epm_motor_threshold,
                            "paradigm": "epm",
                        },
                        "blocks_downstream": False,
                    }
                )
    if paradigm == "zero_maze":
        _zm_sample_threshold = _get_shared_param("sample_size_underpowered_threshold", 5)
        _zm_distance_threshold = _get_paradigm_param("zero_maze", "zm_low_distance_threshold", 10.0)
        for grp_name, grp_metrics in group_summary.items():
            if not grp_metrics:
                continue
            sample_n = next(iter(grp_metrics.values())).get("n", 0)
            if 0 < sample_n < _zm_sample_threshold:
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "code": "SAMPLE.UNDERPOWERED",
                        "metric": "all",
                        "message": (
                            f"Group '{grp_name}' has n={sample_n} (<{_zm_sample_threshold}). "
                            "统计功效不足，结论需谨慎。"
                        ),
                        "evidence": {
                            "n": sample_n,
                            "threshold": _zm_sample_threshold,
                            "paradigm": "zero_maze",
                            "group": grp_name,
                        },
                        "blocks_downstream": False,
                    }
                )
        for name, m in per_subject.items():
            td = m.get("distance_moved")
            if (
                td is not None
                and isinstance(td, (int, float))
                and td < _zm_distance_threshold
            ):
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "code": "MOTOR.LOW_DISTANCE",
                        "metric": "distance_moved",
                        "message": (
                            f"Subject '{name}' 总移动距离={td:.2f} (<{_zm_distance_threshold})。"
                            "开放区指标的下降可能为运动抑制而非焦虑增加，需标注警告。"
                        ),
                        "evidence": {
                            "subject": name,
                            "distance_moved": td,
                            "threshold": _zm_distance_threshold,
                            "paradigm": "zero_maze",
                        },
                        "blocks_downstream": False,
                    }
                )
    if paradigm == "light_dark_box":
        _ldb_sample_threshold = _get_shared_param("sample_size_underpowered_threshold", 5)
        _ldb_transition_threshold = _get_paradigm_param("light_dark_box", "signal_low_transition_threshold", 4)
        for grp_name, grp_metrics in group_summary.items():
            if not grp_metrics:
                continue
            sample_n = next(iter(grp_metrics.values())).get("n", 0)
            if 0 < sample_n < _ldb_sample_threshold:
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "code": "SAMPLE.UNDERPOWERED",
                        "metric": "all",
                        "message": (
                            f"Group '{grp_name}' has n={sample_n} (<{_ldb_sample_threshold}). "
                            "统计功效不足，结论需谨慎。"
                        ),
                        "evidence": {
                            "n": sample_n,
                            "threshold": _ldb_sample_threshold,
                            "paradigm": "light_dark_box",
                            "group": grp_name,
                        },
                        "blocks_downstream": False,
                    }
                )
        for name, m in per_subject.items():
            tc = m.get("transition_count")
            if tc is not None and isinstance(tc, (int, float)) and tc < _ldb_transition_threshold:
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "code": "SIGNAL.LOW_TRANSITION_COUNT",
                        "metric": "transition_count",
                        "message": (
                            f"Subject '{name}' 穿梭次数={int(tc)} (<{_ldb_transition_threshold})。"
                            "明箱时间百分比的下降可能为探索动机不足而非焦虑增加，需标注警告。"
                        ),
                        "evidence": {
                            "subject": name,
                            "transition_count": int(tc),
                            "threshold": _ldb_transition_threshold,
                            "paradigm": "light_dark_box",
                        },
                        "blocks_downstream": False,
                    }
                )
    if paradigm in ("forced_swim", "tail_suspension"):
        _fst_sample_threshold = _get_shared_param("sample_size_underpowered_threshold", 5)
        for grp_name, grp_metrics in group_summary.items():
            if not grp_metrics:
                continue
            sample_n = next(iter(grp_metrics.values())).get("n", 0)
            if 0 < sample_n < _fst_sample_threshold:
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "code": "SAMPLE.UNDERPOWERED",
                        "metric": "all",
                        "message": (
                            f"Group '{grp_name}' has n={sample_n} (<{_fst_sample_threshold}). "
                            "统计功效不足，结论需谨慎。"
                        ),
                        "evidence": {
                            "n": sample_n,
                            "threshold": _fst_sample_threshold,
                            "paradigm": paradigm,
                            "group": grp_name,
                        },
                        "blocks_downstream": False,
                    }
                )

    return {
        "paradigm": paradigm,
        "per_subject": per_subject,
        "group_summary": group_summary,
        "group_level_metrics": group_level_metrics,
        "timeseries": timeseries,
        "data_quality_warnings": data_quality_warnings,
        "metadata": {
            "n_subjects": len(subjects),
            "n_files": summary.get("total_files", 0),
            "duration_s": summary.get("duration_seconds", 0),
            "computed_metrics": computed,
        },
    }
