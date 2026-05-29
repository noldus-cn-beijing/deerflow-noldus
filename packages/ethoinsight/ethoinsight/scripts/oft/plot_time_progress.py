"""OFT: 时间进程图（5 分钟 bin，运动距离 + 中心区滞留双折线）。

CLI:
  单文件:  python -m ethoinsight.scripts.oft.plot_time_progress \\
             --input <轨迹文件> --output <png>
  多文件:  python -m ethoinsight.scripts.oft.plot_time_progress \\
             --inputs <inputs.json> --output <png>
  分组聚合: python -m ethoinsight.scripts.oft.plot_time_progress \\
             --inputs <inputs.json> --groups <groups.json> --output <png>

per-subject plot: reads ``paths[0]`` when given an inputs.json.
Group-aggregate: computes group mean ± SEM per 5-min bin across subjects.
catalog when: total_duration_seconds > 300
bin 长度固定 300 秒，末尾不足并入最后一个 bin。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ethoinsight.charts import time_progress_plot
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_plot_parser, read_groups_json, read_inputs_json, resolve_per_subject_input


def _find_center_col(df: pd.DataFrame) -> str | None:
    if "in_zone_center" in df.columns:
        return "in_zone_center"
    for col in df.columns:
        cl = col.lower()
        if cl.startswith("in_zone") and ("center" in cl or "centre" in cl):
            if not any(bad in cl for bad in ("wall", "edge", "peripher", "border", "outer")):
                return col
    return None


def _compute_bins(df: pd.DataFrame) -> list[dict]:
    """Slice df into 5-min bins and compute distance + center_time per bin."""
    t = pd.to_numeric(df["trial_time"], errors="coerce")
    t_min = float(t.min())
    t_max = float(t.max())
    total_dur = t_max - t_min
    n_bins = max(1, math.floor(total_dur / 300))

    center_col = _find_center_col(df)
    has_distance = "distance_moved" in df.columns

    bins = []
    for i in range(n_bins):
        bin_start = t_min + i * 300
        # last bin absorbs remainder
        bin_end = t_max if i == n_bins - 1 else t_min + (i + 1) * 300
        mask = (t >= bin_start) & (t < bin_end)
        if i == n_bins - 1:
            mask = t >= bin_start
        sub = df[mask]

        # distance: sum distance_moved in bin, or estimate from velocity*dt
        if has_distance:
            dist = float(pd.to_numeric(sub["distance_moved"], errors="coerce").fillna(0).sum())
        elif "velocity" in sub.columns and len(sub) >= 2:
            v = pd.to_numeric(sub["velocity"], errors="coerce").fillna(0)
            tt = pd.to_numeric(sub["trial_time"], errors="coerce")
            dt = float(tt.diff().median())
            dist = float((v * dt).sum())
        else:
            dist = 0.0

        # center_time: sum of frame durations where center_col == 1
        if center_col and center_col in sub.columns:
            in_center = pd.to_numeric(sub[center_col], errors="coerce").fillna(0)
            if len(sub) >= 2:
                tt = pd.to_numeric(sub["trial_time"], errors="coerce")
                dt = float(tt.diff().median())
            else:
                dt = 0.04  # fallback 25Hz
            center_time = float((in_center * dt).sum())
        else:
            center_time = 0.0

        bins.append({
            "bin_start_sec": bin_start - t_min,
            "bin_end_sec": bin_end - t_min,
            "distance": dist,
            "center_time": center_time,
        })

    return bins


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=True).parse_args(argv)

    # Group-aggregate mode: multiple subjects with group labels
    if args.inputs and args.groups:
        paths = read_inputs_json(args.inputs)
        groups = read_groups_json(args.groups)

        # Build subject_name → group_name lookup
        subject_group: dict[str, str] = {}
        for gname, subjects in groups.items():
            for s in subjects:
                subject_group[s] = gname

        # Parse all subjects, compute per-subject bins
        group_bins: dict[str, list[list[dict]]] = {}
        for p in paths:
            df = parse_trajectory(p)
            if "trial_time" not in df.columns:
                continue
            subject_name = Path(p).stem
            gname = subject_group.get(subject_name, "ungrouped")
            bins = _compute_bins(df)
            group_bins.setdefault(gname, []).append(bins)

        if not group_bins:
            print("error: no valid data found", file=sys.stderr)
            return 1

        # Aggregate: per-bin mean ± SEM across subjects within each group.
        # Bins are aligned by index (bin 0 = first 300s, bin 1 = next 300s, ...)
        # since all subjects share the same trial duration assumption.
        for gname, subjects_bins in group_bins.items():
            n_subjects = len(subjects_bins)
            # Use the most common bin count as reference
            bin_counts = [len(b) for b in subjects_bins]
            n_bins = max(set(bin_counts), key=bin_counts.count)
            aggregated = []
            for bin_i in range(n_bins):
                bin_distances = []
                bin_center_times = []
                bin_start = None
                bin_end = None
                for sb in subjects_bins:
                    if bin_i < len(sb):
                        bin_distances.append(sb[bin_i]["distance"])
                        bin_center_times.append(sb[bin_i]["center_time"])
                        if bin_start is None:
                            bin_start = sb[bin_i]["bin_start_sec"]
                            bin_end = sb[bin_i]["bin_end_sec"]

                n_contrib = len(bin_distances)
                aggregated.append({
                    "bin_start_sec": bin_start or 0,
                    "bin_end_sec": bin_end or 0,
                    "distance": float(np.mean(bin_distances)) if n_contrib > 0 else 0.0,
                    "distance_sem": float(np.std(bin_distances, ddof=1) / np.sqrt(n_contrib)) if n_contrib > 1 else None,
                    "center_time": float(np.mean(bin_center_times)) if n_contrib > 0 else 0.0,
                    "center_time_sem": float(np.std(bin_center_times, ddof=1) / np.sqrt(n_contrib)) if n_contrib > 1 else None,
                })

            output_path = time_progress_plot(aggregated, output_path=args.output, group_label=gname)
            emit_result({"plot": "time_progress", "group": gname, "n_subjects": n_subjects, "n_bins": n_bins, "path": output_path})

        return 0

    # Per-subject mode
    try:
        path = resolve_per_subject_input(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    df = parse_trajectory(path)

    if "trial_time" not in df.columns:
        print("error: trial_time column missing", file=sys.stderr)
        return 1

    t = pd.to_numeric(df["trial_time"], errors="coerce")
    total_dur = float(t.max() - t.min())
    if total_dur <= 0:
        print("error: trial_time has zero duration", file=sys.stderr)
        return 1

    per_bin_data = _compute_bins(df)
    output_path = time_progress_plot(per_bin_data, output_path=args.output)
    emit_result({"plot": "time_progress", "path": output_path, "n_bins": len(per_bin_data), "total_duration_seconds": total_dur})
    return 0


if __name__ == "__main__":
    sys.exit(main())
