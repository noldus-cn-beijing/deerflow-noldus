"""Shoaling (zebrafish group behavior) analysis template.

code-executor reads this template, modifies the PARAMETERS section,
and executes it in the sandbox. All parameters are at the top for
easy LLM modification.
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

# ===== PARAMETERS (code-executor modifies this section) =====
FILE_PATTERN = "/mnt/user-data/uploads/轨迹*.txt"
PARADIGM = "shoaling"
GROUPS = {
    "control": ["Subject 1", "Subject 2"],
    "treatment": ["Subject 3", "Subject 4", "Subject 5"],
}
METRICS_TO_COMPUTE = [
    "distance_moved",
    "mean_iid",
    "mean_nnd",
    "mean_polarity",
]
CHART_TYPES = ["box_plot"]  # box_plot, violin_plot, bar_chart
OUTPUT_DIR = "/mnt/user-data/workspace/output/"
HANDOFF_PATH = "/mnt/user-data/workspace/handoff_code_executor.json"
# ===== END PARAMETERS =====


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    errors: list[str] = []

    # 1. Parse
    print("Step 1: Parsing data...")
    data = parse.parse_batch(FILE_PATTERN)
    print(parse.get_summary(data))

    if data["summary"]["total_files"] == 0:
        _write_handoff("failed", "No trajectory files found", [], errors=["No files"])
        return

    # 2. Compute metrics
    print("\nStep 2: Computing metrics...")
    m = metrics.compute_paradigm_metrics(
        data, PARADIGM, groups=GROUPS, metrics=METRICS_TO_COMPUTE
    )

    # 3. Statistical tests (skip if not available)
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

    # 4. Charts
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

    # Trajectory plot
    try:
        traj_path = charts.trajectory_plot(
            data["all_data"],
            output_path=os.path.join(OUTPUT_DIR, "trajectory.png"),
        )
        chart_paths.append(traj_path)
        print(f"  Saved: {traj_path}")
    except Exception as e:
        errors.append(f"Trajectory plot error: {e}")

    # Timeseries plots for IID and polarity
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

    # 5. Save data
    print("\nStep 5: Saving outputs...")
    metrics_csv = os.path.join(OUTPUT_DIR, "metrics.csv")
    metrics.save_to_csv(m, metrics_csv)
    print(f"  Saved: {metrics_csv}")

    if stat_results:
        stats_json = os.path.join(OUTPUT_DIR, "statistics.json")
        with open(stats_json, "w") as f:
            json.dump(stat_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved: {stats_json}")

    # 6. Handoff
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
