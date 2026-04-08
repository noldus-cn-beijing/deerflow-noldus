"""Shoaling (zebrafish group behavior) analysis template.

get_analysis_template tool reads this file, fills the PARAMETERS section,
and returns a ready-to-run script. Lines marked CUSTOMIZABLE can be
modified by code-executor via str_replace to meet user-specific needs.
"""

import json
import os

from ethoinsight import parse, metrics, charts

# Try importing statistics (P3 — may not be implemented yet)
try:
    from ethoinsight import statistics as stats

    _HAS_STATISTICS = hasattr(stats, "compare_groups")
except ImportError:
    _HAS_STATISTICS = False

# ===== PARAMETERS (code-executor: filled by get_analysis_template, do not modify) =====
FILE_PATTERN = "/mnt/user-data/uploads/轨迹*.txt"
PARADIGM = "shoaling"
GROUPS = {
    "control": ["Subject 1", "Subject 2"],
    "treatment": ["Subject 3", "Subject 4", "Subject 5"],
}
# ===== END PARAMETERS =====

METRICS_TO_COMPUTE = [  # CUSTOMIZABLE: add/remove metrics as needed
    "distance_moved",
    "mean_iid",
    "mean_nnd",
    "mean_polarity",
]
CHART_TYPES = ["box_plot"]  # CUSTOMIZABLE: box_plot, violin_plot, bar_chart, raincloud_plot, beeswarm_plot, correlogram
OUTPUT_DIR = "/mnt/user-data/workspace/output/"
HANDOFF_PATH = "/mnt/user-data/workspace/handoff_code_executor.json"
# ===== END PARAMETERS =====


def _resolve_path(virtual_path: str) -> str:
    """Resolve /mnt/... virtual paths using DEERFLOW_PATH_* env vars set by sandbox."""
    if not virtual_path.startswith("/mnt/"):
        return virtual_path
    # Try progressively shorter prefixes: /mnt/user-data/uploads -> /mnt/user-data -> /mnt
    parts = virtual_path.strip("/").split("/")
    for end in range(len(parts), 0, -1):
        key = "DEERFLOW_PATH_" + "_".join(parts[:end]).replace("-", "_").upper()
        mapped = os.environ.get(key)
        if mapped:
            suffix = "/".join(parts[end:])
            return os.path.join(mapped, suffix) if suffix else mapped
    return virtual_path


def main() -> None:
    global FILE_PATTERN, OUTPUT_DIR, HANDOFF_PATH
    # Resolve virtual paths to physical paths via sandbox env vars
    FILE_PATTERN = _resolve_path(FILE_PATTERN)
    OUTPUT_DIR = _resolve_path(OUTPUT_DIR)
    HANDOFF_PATH = _resolve_path(HANDOFF_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    errors: list[str] = []

    # 1. Parse — fixed workflow, do not modify
    print("Step 1: Parsing data...")
    data = parse.parse_batch(FILE_PATTERN)
    print(parse.get_summary(data))

    if data["summary"]["total_files"] == 0:
        _write_handoff("failed", "No trajectory files found", [], errors=["No files"])
        return

    # 2. Compute metrics — fixed workflow, do not modify
    print("\nStep 2: Computing metrics...")
    m = metrics.compute_paradigm_metrics(
        data, PARADIGM, groups=GROUPS, metrics=METRICS_TO_COMPUTE
    )

    # 3. Statistical tests — fixed workflow, do not modify
    stat_results = {}
    if _HAS_STATISTICS:
        print("\nStep 3: Running statistical tests...")
        try:
            stat_results = stats.compare_groups(m, list(GROUPS.keys()))
        except Exception as e:
            errors.append(f"Statistics error: {e}")
            print(f"Warning: statistics failed: {e}")
    else:
        print("\nStep 3: Statistics module not yet implemented, skipping.")

    # 4. Charts — fixed workflow, do not modify
    print("\nStep 4: Generating charts...")
    chart_paths = []
    for metric in METRICS_TO_COMPUTE:
        for chart_type in CHART_TYPES:
            try:
                fn = getattr(charts, chart_type)
                path = fn(
                    m,
                    [metric],
                    significance=stat_results or None,
                    output_path=os.path.join(OUTPUT_DIR, f"{metric}_{chart_type}.png"),
                )
                chart_paths.append(path)
                print(f"  Saved: {path}")
            except Exception as e:
                errors.append(f"Chart error ({metric} {chart_type}): {e}")
                print(f"  Warning: {metric} {chart_type} failed: {e}")

    # Trajectory plot — fixed workflow, do not modify
    try:
        traj_path = charts.trajectory_plot(
            data["all_data"],
            output_path=os.path.join(OUTPUT_DIR, "trajectory.png"),
        )
        chart_paths.append(traj_path)
        print(f"  Saved: {traj_path}")
    except Exception as e:
        errors.append(f"Trajectory plot error: {e}")

    # Timeseries plots — fixed workflow, do not modify
    for ts_name, y_col in [
        ("inter_individual_distance", "mean_iid"),
        ("group_polarity", "polarity"),
    ]:
        ts_df = m.get("timeseries", {}).get(ts_name)
        if ts_df is not None:
            try:
                ts_path = charts.timeseries_plot(
                    ts_df,
                    y_col=y_col,
                    output_path=os.path.join(OUTPUT_DIR, f"{ts_name}_timeseries.png"),
                )
                chart_paths.append(ts_path)
                print(f"  Saved: {ts_path}")
            except Exception as e:
                errors.append(f"Timeseries plot error ({ts_name}): {e}")

    # CUSTOMIZABLE: add extra charts or custom analysis below >>>
    # Example: charts.violin_plot(m, ["mean_iid"], output_path=os.path.join(OUTPUT_DIR, "iid_violin.png"))
    # Example: charts.bar_chart(m, ["distance_moved"], output_path=os.path.join(OUTPUT_DIR, "distance_bar.png"))
    # <<<

    # 5. Save data — fixed workflow, do not modify
    print("\nStep 5: Saving outputs...")
    metrics_csv = os.path.join(OUTPUT_DIR, "metrics.csv")
    metrics.save_to_csv(m, metrics_csv)
    print(f"  Saved: {metrics_csv}")

    if stat_results:
        stats_json = os.path.join(OUTPUT_DIR, "statistics.json")
        with open(stats_json, "w") as f:
            json.dump(stat_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved: {stats_json}")

    # 6. Handoff — fixed workflow, do not modify
    _write_handoff(
        status="completed",
        summary_text=f"Analyzed {data['summary']['total_files']} files, "
                     f"{data['summary']['total_rows']} rows, "
                     f"{len(m.get('per_subject', {}))} subjects",
        chart_paths=chart_paths,
        metrics_csv=metrics_csv,
        stat_results=stat_results,
        errors=errors,
        data_summary=data["summary"],
    )


def _write_handoff(
    status: str,
    summary_text: str,
    chart_paths: list[str],
    metrics_csv: str = "",
    stat_results: dict | None = None,
    errors: list[str] | None = None,
    data_summary: dict | None = None,
) -> None:
    handoff = {
        "status": status,
        "summary": summary_text,
        "output_files": {
            "metrics": metrics_csv,
            "statistics": os.path.join(OUTPUT_DIR, "statistics.json") if stat_results else "",
            "charts": chart_paths,
        },
        "metadata": {
            "paradigm": PARADIGM,
            "n_files": (data_summary or {}).get("total_files", 0),
            "groups": GROUPS,
        },
        "errors": errors or [],
    }
    os.makedirs(os.path.dirname(HANDOFF_PATH) or ".", exist_ok=True)
    with open(HANDOFF_PATH, "w") as f:
        json.dump(handoff, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nHANDOFF: {HANDOFF_PATH}")


if __name__ == "__main__":
    main()
