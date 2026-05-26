"""compute_paradigm_metrics 派发器：按 paradigm 路由到 metrics/<范式>.py 的函数。"""

from __future__ import annotations


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
# Paradigm dispatcher
# ============================================================================


def compute_paradigm_metrics(
    parsed_data: dict,
    paradigm: str,
    groups: dict[str, list[str]] | None = None,
    metrics: list[str] | None = None,
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

    Returns:
        {
            "paradigm": str,
            "per_subject": {subject: {metric: value, ...}, ...},
            "group_summary": {group: {metric: {"mean", "std", "n", "values"}, ...}, ...},
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
    per_subject: dict[str, dict] = {}
    for name, df in subjects.items():
        m: dict[str, float | dict | None] = {}
        m["distance_moved"] = compute_distance_moved(df)
        m["velocity_stats"] = compute_velocity_stats(df)
        if paradigm == "open_field":
            m["center_time_ratio"] = compute_center_time_ratio(df)
            m["thigmotaxis_index"] = compute_thigmotaxis_index(df)
            m["center_distance_ratio"] = compute_center_distance_ratio(df)
            m["center_entry_count"] = compute_center_entry_count(df)
        elif paradigm == "epm":
            m["open_arm_time_ratio"] = compute_open_arm_time_ratio(df)
            m["open_arm_entry_ratio"] = compute_open_arm_entry_ratio(df)
            m["open_arm_entry_count"] = compute_open_arm_entry_count(df)
            m["open_arm_time"] = compute_open_arm_time(df)
            m["total_entry_count"] = compute_total_entry_count(df)
        elif paradigm == "zero_maze":
            m["open_zone_time_ratio"] = compute_open_zone_time_ratio(df)
            m["open_zone_time"] = compute_open_zone_time(df)
            m["open_zone_distance"] = compute_open_zone_distance(df)
            m["hesitation_count"] = compute_hesitation_count(df)
        elif paradigm == "light_dark_box":
            m["light_time_ratio"] = compute_light_time_ratio(df)
            m["transition_count"] = compute_transition_count(df)
            m["light_latency"] = compute_light_latency(df)
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
    group_summary: dict[str, dict] = {}
    for grp_name, grp_subjects in groups.items():
        grp_metrics: dict[str, dict] = {}
        matched = [s for s in grp_subjects if s in per_subject]
        if not matched:
            continue
        # Collect all scalar metric names
        all_metric_names = set()
        for s in matched:
            for k, v in per_subject[s].items():
                if isinstance(v, (int, float)) and v is not None:
                    all_metric_names.add(k)

        for mname in sorted(all_metric_names):
            values = [
                per_subject[s][mname]
                for s in matched
                if mname in per_subject[s]
                and per_subject[s][mname] is not None
                and isinstance(per_subject[s][mname], (int, float))
            ]
            if values:
                arr = np.array(values, dtype=float)
                grp_metrics[mname] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "n": len(arr),
                    "values": [float(v) for v in arr],
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
                    "metric": "all",
                    "message": (
                        f"Group '{grp_name}' has n={sample_n} (<3). "
                        "Statistical inference will be unreliable; descriptive statistics only."
                    ),
                }
            )
    if paradigm == "epm":
        # Per epm.md: n < 5 per group → low statistical power
        for grp_name, grp_metrics in group_summary.items():
            if not grp_metrics:
                continue
            sample_n = next(iter(grp_metrics.values())).get("n", 0)
            if 0 < sample_n < 5:
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "metric": "all",
                        "message": (
                            f"Group '{grp_name}' has n={sample_n} (<5). "
                            "统计功效不足，结论需谨慎。"
                        ),
                    }
                )
        # Per epm.md: total entries < 8 → motor suppression confound
        for name, m in per_subject.items():
            te = m.get("total_entry_count")
            if te is not None and isinstance(te, (int, float)) and te < 8:
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "metric": "total_entry_count",
                        "message": (
                            f"Subject '{name}' 总进臂次数={int(te)} (<8)。"
                            "开臂指标的下降可能为运动抑制而非焦虑增加，需标注警告。"
                        ),
                    }
                )
    if paradigm == "zero_maze":
        # Per zero_maze.md: n < 5 per group → low statistical power
        for grp_name, grp_metrics in group_summary.items():
            if not grp_metrics:
                continue
            sample_n = next(iter(grp_metrics.values())).get("n", 0)
            if 0 < sample_n < 5:
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "metric": "all",
                        "message": (
                            f"Group '{grp_name}' has n={sample_n} (<5). "
                            "统计功效不足，结论需谨慎。"
                        ),
                    }
                )
        # Per zero_maze.md: total distance too low → motor suppression confound
        _ZM_LOW_DISTANCE_THRESHOLD = 10.0  # cm; very low → movement suppressed
        for name, m in per_subject.items():
            td = m.get("distance_moved")
            if (
                td is not None
                and isinstance(td, (int, float))
                and td < _ZM_LOW_DISTANCE_THRESHOLD
            ):
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "metric": "distance_moved",
                        "message": (
                            f"Subject '{name}' 总移动距离={td:.2f} (<{_ZM_LOW_DISTANCE_THRESHOLD})。"
                            "开放区指标的下降可能为运动抑制而非焦虑增加，需标注警告。"
                        ),
                    }
                )
    if paradigm == "light_dark_box":
        # Per ldb.md: n < 5 per group → low statistical power
        for grp_name, grp_metrics in group_summary.items():
            if not grp_metrics:
                continue
            sample_n = next(iter(grp_metrics.values())).get("n", 0)
            if 0 < sample_n < 5:
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "metric": "all",
                        "message": (
                            f"Group '{grp_name}' has n={sample_n} (<5). "
                            "统计功效不足，结论需谨慎。"
                        ),
                    }
                )
        # Per ldb.md: transitions < 4 → insufficient exploration motivation
        for name, m in per_subject.items():
            tc = m.get("transition_count")
            if tc is not None and isinstance(tc, (int, float)) and tc < 4:
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "metric": "transition_count",
                        "message": (
                            f"Subject '{name}' 穿梭次数={int(tc)} (<4)。"
                            "明箱时间百分比的下降可能为探索动机不足而非焦虑增加，需标注警告。"
                        ),
                    }
                )
    if paradigm in ("forced_swim", "tail_suspension"):
        # n < 5 per group → low statistical power for immobility paradigms
        for grp_name, grp_metrics in group_summary.items():
            if not grp_metrics:
                continue
            sample_n = next(iter(grp_metrics.values())).get("n", 0)
            if 0 < sample_n < 5:
                data_quality_warnings.append(
                    {
                        "severity": "warning",
                        "metric": "all",
                        "message": (
                            f"Group '{grp_name}' has n={sample_n} (<5). "
                            "统计功效不足，结论需谨慎。"
                        ),
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
