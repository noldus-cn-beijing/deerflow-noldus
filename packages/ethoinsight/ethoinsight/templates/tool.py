"""Analysis template tools for code-executor subagent.

Provides:
- get_analysis_template_tool: Returns a ready-to-run Python script (legacy, fallback).
- run_paradigm_analysis_tool: Executes the full analysis pipeline in one tool call.
- run_paradigm_analysis_core: Pure function (no langchain dependency) for testing.

NOTE: This module imports langchain at the top level. It only runs inside the
DeerFlow agent process where langchain is available. Standalone ethoinsight
tests do not import this module (they test statistics/metrics/parse directly).
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from pathlib import Path

from langchain.tools import ToolRuntime, tool

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent
_EXCLUDED_FILES = {"__init__.py", "tool.py"}
_PARAM_START = "# ===== PARAMETERS"
_PARAM_END = "# ===== END PARAMETERS ====="


def get_available_paradigms() -> list[str]:
    """Return list of available paradigm names (template files in templates/)."""
    return sorted(
        p.stem
        for p in _TEMPLATES_DIR.glob("*.py")
        if p.name not in _EXCLUDED_FILES
    )


def render_template(
    paradigm: str,
    file_pattern: str,
    groups: dict[str, list[str]],
    metrics: list[str] | None = None,
    chart_types: list[str] | None = None,
    output_dir: str = "/mnt/user-data/outputs/",
    handoff_path: str = "/mnt/user-data/workspace/handoff_code_executor.json",
) -> str:
    """Read a paradigm template and fill in the PARAMETERS section.

    Args:
        paradigm: Paradigm name (e.g. "shoaling", "open_field").
        file_pattern: Glob pattern for input trajectory files.
        groups: Mapping of group name to list of subject identifiers.
        metrics: Optional list of metrics to compute. If None, uses template defaults.
        chart_types: Optional list of chart types. If None, uses template defaults.
        output_dir: Directory for output files.
        handoff_path: Path for handoff JSON.

    Returns:
        Complete Python script as a string, ready to execute.

    Raises:
        FileNotFoundError: If the paradigm template does not exist.
    """
    template_path = _TEMPLATES_DIR / f"{paradigm}.py"
    if not template_path.exists():
        raise FileNotFoundError(
            f"No template for paradigm '{paradigm}'. "
            f"Available: {', '.join(get_available_paradigms())}"
        )

    source = template_path.read_text(encoding="utf-8")

    # Build the new PARAMETERS block
    param_lines = [
        f'# ===== PARAMETERS (filled by get_analysis_template, do not modify) =====',
        f'FILE_PATTERN = {file_pattern!r}',
        f'PARADIGM = {paradigm!r}',
        f'GROUPS = {_format_dict(groups)}',
    ]

    if metrics is not None:
        param_lines.append(f'METRICS_TO_COMPUTE = {metrics!r}  # CUSTOMIZABLE: add/remove metrics')
    if chart_types is not None:
        param_lines.append(f'CHART_TYPES = {chart_types!r}  # CUSTOMIZABLE: box_plot, violin_plot, bar_chart, raincloud_plot, beeswarm_plot, correlogram')

    param_lines.append(f'OUTPUT_DIR = {output_dir!r}')
    param_lines.append(f'HANDOFF_PATH = {handoff_path!r}')
    param_lines.append(_PARAM_END)

    new_params = "\n".join(param_lines)

    # Replace between PARAMETERS markers
    pattern = re.compile(
        rf"^{re.escape(_PARAM_START)}.*?^{re.escape(_PARAM_END)}",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(source)
    if match:
        result = source[: match.start()] + new_params + source[match.end() :]
    else:
        # No markers found — prepend parameters (fallback)
        logger.warning("No PARAMETERS markers in template %s, prepending params", paradigm)
        result = new_params + "\n\n" + source

    return result


def _format_dict(d: dict) -> str:
    """Format a dict as a readable Python literal."""
    items = []
    for k, v in d.items():
        items.append(f"    {k!r}: {v!r},")
    return "{\n" + "\n".join(items) + "\n}"


@tool("get_analysis_template", parse_docstring=True)
def get_analysis_template_tool(
    paradigm: str,
    file_pattern: str,
    groups: str,
    metrics: str | None = None,
    chart_types: str | None = None,
    output_dir: str = "/mnt/user-data/outputs/",
    handoff_path: str = "/mnt/user-data/workspace/handoff_code_executor.json",
) -> str:
    """Get a ready-to-run analysis script for the specified paradigm.

    Returns a complete Python script with parameters filled in. Write the
    result to a .py file and execute it with bash. Lines marked with
    '# CUSTOMIZABLE' can be modified with str_replace if the user has
    special requirements.

    Args:
        paradigm: Analysis paradigm name. Unsupported paradigms return an error with the list of available ones.
        file_pattern: Glob pattern for input data files. Example: "/mnt/user-data/uploads/*.txt"
        groups: JSON string mapping group names to subject lists. Example: '{"control": ["Subject 1", "Subject 2"], "treatment": ["Subject 3", "Subject 4"]}'
        metrics: Optional comma-separated metric names to compute. If omitted, uses paradigm defaults. Example: "distance_moved,mean_iid,mean_nnd"
        chart_types: Optional comma-separated chart types. If omitted, uses paradigm defaults. Example: "box_plot,violin_plot"
        output_dir: Output directory path. Default: "/mnt/user-data/outputs/"
        handoff_path: Handoff JSON path. Default: "/mnt/user-data/workspace/handoff_code_executor.json"

    Returns:
        Complete Python script as a string, ready to write to a file and execute.
    """
    # Parse groups JSON
    try:
        groups_dict = json.loads(groups)
    except (json.JSONDecodeError, TypeError) as e:
        available = get_available_paradigms()
        return (
            f"Error: invalid groups JSON: {e}\n"
            f"Expected format: '{{\"control\": [\"Subject 1\"], \"treatment\": [\"Subject 3\"]}}'\n"
            f"Available paradigms: {', '.join(available)}"
        )

    # Parse optional comma-separated lists
    metrics_list = [m.strip() for m in metrics.split(",")] if metrics else None
    chart_types_list = [c.strip() for c in chart_types.split(",")] if chart_types else None

    try:
        return render_template(
            paradigm=paradigm,
            file_pattern=file_pattern,
            groups=groups_dict,
            metrics=metrics_list,
            chart_types=chart_types_list,
            output_dir=output_dir,
            handoff_path=handoff_path,
        )
    except FileNotFoundError as e:
        return str(e)


def run_paradigm_analysis_core(
    paradigm: str,
    file_pattern: str,
    groups: dict[str, list[str]],
    metrics: list[str] | None = None,
    chart_types: list[str] | None = None,
    output_dir: str = "",
    handoff_path: str = "",
) -> dict:
    """Execute the full analysis pipeline and return structured results.

    This is a pure Python function with no langchain dependency. It accepts
    already-resolved physical paths. The DeerFlow-side tool wrapper handles
    virtual path resolution and calls this function.

    Args:
        paradigm: Paradigm name (e.g. "shoaling", "open_field", "epm").
        file_pattern: Physical glob pattern for input trajectory files.
        groups: Mapping of group name to list of subject identifiers.
        metrics: Optional list of metrics to compute.
        chart_types: Optional list of chart types. Defaults to ["box_plot"].
        output_dir: Physical directory for output files.
        handoff_path: Physical path for handoff JSON file.

    Returns:
        Dict with keys: status, summary, output_files, metrics_summary,
        statistics, assessment, metadata, errors.
    """
    # Gate: reject paradigms that have no analysis template (incomplete support)
    available = get_available_paradigms()
    if paradigm not in available:
        return {
            "status": "failed",
            "summary": f"范式 '{paradigm}' 尚未支持完整自动分析",
            "errors": [
                f"当前支持完整分析的范式: {', '.join(available)}",
                "建议：请告知用户该范式暂不支持，征求用户意见后再决定下一步",
            ],
            "available_paradigms": available,
        }

    from ethoinsight import parse, metrics as etho_metrics, charts as etho_charts

    try:
        from ethoinsight import statistics as stats
        _has_statistics = hasattr(stats, "compare_groups")
    except ImportError:
        _has_statistics = False

    try:
        from ethoinsight import assess
        _has_assess = hasattr(assess, "assess_results")
    except ImportError:
        _has_assess = False

    if chart_types is None:
        chart_types = ["box_plot"]

    os.makedirs(output_dir, exist_ok=True)
    errors: list[str] = []
    chart_paths: list[str] = []
    stat_results: dict = {}
    m_result: dict = {}

    # Step 1: Parse
    try:
        data = parse.parse_batch(file_pattern)
    except Exception as e:
        return {"status": "failed", "summary": "Data parsing failed", "errors": [f"Parse error: {e}"]}

    if data["summary"]["total_files"] == 0:
        matched = glob.glob(file_pattern)
        return {
            "status": "failed",
            "summary": f"No trajectory files found matching pattern",
            "errors": [f"Pattern '{file_pattern}' matched 0 files. Cwd contents: {matched[:5]}"],
        }

    # Step 2: Compute metrics
    try:
        m_result = etho_metrics.compute_paradigm_metrics(
            data, paradigm, groups=groups, metrics=metrics
        )
    except Exception as e:
        return {"status": "failed", "summary": "Metrics computation failed", "errors": [f"Metrics error: {e}"]}

    # Step 3: Statistical tests
    if _has_statistics:
        try:
            stat_results = stats.compare_groups(m_result, list(groups.keys()))
        except Exception as e:
            errors.append(f"Statistics error: {e}")

    # Step 4: Charts
    computed_metrics = m_result.get("metadata", {}).get("computed_metrics", [])
    metrics_to_plot = metrics if metrics else computed_metrics
    for metric in metrics_to_plot:
        for ct in chart_types:
            try:
                fn = getattr(etho_charts, ct, None)
                if fn is None:
                    errors.append(f"Unknown chart type: {ct}")
                    continue
                path = fn(
                    m_result,
                    [metric],
                    significance=stat_results or None,
                    output_path=os.path.join(output_dir, f"{metric}_{ct}.png"),
                )
                chart_paths.append(path)
            except Exception as e:
                errors.append(f"Chart error ({metric} {ct}): {e}")

    # Trajectory plot
    if data.get("all_data") is not None:
        try:
            traj_path = etho_charts.trajectory_plot(
                data["all_data"],
                output_path=os.path.join(output_dir, "trajectory.png"),
            )
            chart_paths.append(traj_path)
        except Exception as e:
            errors.append(f"Trajectory plot error: {e}")

    # Timeseries plots (shoaling-specific)
    for ts_name, y_col in [
        ("inter_individual_distance", "mean_iid"),
        ("group_polarity", "polarity"),
    ]:
        ts_df = m_result.get("timeseries", {}).get(ts_name)
        if ts_df is not None:
            try:
                ts_path = etho_charts.timeseries_plot(
                    ts_df, y_col=y_col,
                    output_path=os.path.join(output_dir, f"{ts_name}_timeseries.png"),
                )
                chart_paths.append(ts_path)
            except Exception as e:
                errors.append(f"Timeseries plot error ({ts_name}): {e}")

    # Step 5: Save metrics CSV
    metrics_csv = ""
    try:
        metrics_csv = os.path.join(output_dir, "metrics.csv")
        etho_metrics.save_to_csv(m_result, metrics_csv)
    except Exception as e:
        errors.append(f"CSV save error: {e}")

    # Save statistics JSON
    stats_json = ""
    if stat_results:
        try:
            stats_json = os.path.join(output_dir, "statistics.json")
            with open(stats_json, "w", encoding="utf-8") as f:
                json.dump(stat_results, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            errors.append(f"Statistics JSON save error: {e}")

    # Step 6: Domain assessment
    assessment: dict = {}
    if _has_assess and stat_results:
        try:
            assessment = assess.assess_results(stat_results, paradigm, metrics_result=m_result)
        except Exception as e:
            errors.append(f"Assessment error: {e}")

    # Build compact metrics_summary (no raw values)
    compact_group_summary: dict = {}
    group_summary = m_result.get("group_summary", {})
    for grp, metric_dict in group_summary.items():
        compact_group_summary[grp] = {
            metric: {k: v for k, v in vals.items() if k != "values"}
            for metric, vals in metric_dict.items()
        }

    # Step 7: Write handoff JSON
    n_files = data["summary"]["total_files"]
    n_subjects = len(m_result.get("per_subject", {}))
    summary_text = f"Analyzed {n_files} files, {n_subjects} subjects, paradigm: {paradigm}"
    if errors:
        summary_text += f" ({len(errors)} warning(s))"

    handoff = {
        "status": "completed",
        "summary": summary_text,
        "output_files": {
            "metrics": metrics_csv,
            "statistics": stats_json,
            "charts": chart_paths,
        },
        "metrics_summary": compact_group_summary,
        "statistics": stat_results,
        "assessment": assessment,
        "metadata": {
            "paradigm": paradigm,
            "n_files": n_files,
            "groups": groups,
        },
        "errors": errors,
    }

    if handoff_path:
        try:
            os.makedirs(os.path.dirname(handoff_path) or ".", exist_ok=True)
            with open(handoff_path, "w", encoding="utf-8") as f:
                json.dump(handoff, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            errors.append(f"Handoff write error: {e}")

    return handoff


def _resolve_virtual_path(path: str, thread_data: dict | None) -> str:
    """Resolve /mnt/user-data virtual path to physical path using thread_data.

    In local sandbox mode, thread_data maps virtual dirs to physical paths.
    In Docker (aio sandbox), /mnt/user-data is already mounted — thread_data
    is None, so the path is returned as-is.
    """
    if thread_data is None or not path.startswith("/mnt/user-data"):
        return path
    mapping = {
        "/mnt/user-data/workspace": thread_data.get("workspace_path"),
        "/mnt/user-data/uploads": thread_data.get("uploads_path"),
        "/mnt/user-data/outputs": thread_data.get("outputs_path"),
    }
    for prefix, real in mapping.items():
        if real and path.startswith(prefix):
            suffix = path[len(prefix):]
            return real + suffix if suffix else real
    return path


@tool("run_paradigm_analysis", parse_docstring=True)
def run_paradigm_analysis_tool(
    runtime: ToolRuntime,
    paradigm: str,
    file_pattern: str,
    groups: str,
    metrics: str | None = None,
    chart_types: str | None = None,
    output_dir: str = "/mnt/user-data/outputs/",
    handoff_path: str = "/mnt/user-data/workspace/handoff_code_executor.json",
) -> str:
    """Run the complete behavioral data analysis pipeline in one call.

    Executes parse, metrics, statistics, charts, and assessment, then returns
    structured JSON results. Use this instead of get_analysis_template for
    supported paradigms. Returns status="failed" with available_paradigms list
    when the requested paradigm is not yet supported.

    Args:
        paradigm: Paradigm name. Call with any name; unsupported paradigms return a clear error with the list of available ones. Example: "shoaling"
        file_pattern: Glob pattern for input trajectory files. Example: "/mnt/user-data/uploads/*.txt"
        groups: JSON string mapping group names to subject lists. Example: '{"control": ["Subject 1"], "treatment": ["Subject 3"]}'
        metrics: Optional comma-separated metric names. If omitted, uses paradigm defaults. Example: "distance_moved,mean_iid"
        chart_types: Optional comma-separated chart types. Options: box_plot, violin_plot, bar_chart, raincloud_plot, beeswarm_plot. Default: box_plot.
        output_dir: Output directory. Default: "/mnt/user-data/outputs/"
        handoff_path: Handoff JSON path. Default: "/mnt/user-data/workspace/handoff_code_executor.json"

    Returns:
        JSON string with analysis results: status, summary, output_files, metrics_summary, statistics, assessment, metadata, errors.
    """
    # Extract thread_data from runtime for virtual path resolution
    thread_data = None
    try:
        if runtime is not None and runtime.state is not None:
            thread_data = runtime.state.get("thread_data")
    except Exception:
        pass

    # Resolve virtual paths
    real_file_pattern = _resolve_virtual_path(file_pattern, thread_data)
    real_output_dir = _resolve_virtual_path(output_dir, thread_data)
    real_handoff_path = _resolve_virtual_path(handoff_path, thread_data)

    # Parse string parameters
    try:
        groups_dict = json.loads(groups)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"status": "failed", "error": f"Invalid groups JSON: {e}", "errors": [str(e)]}, ensure_ascii=False)

    metrics_list = [m.strip() for m in metrics.split(",")] if metrics else None
    chart_types_list = [c.strip() for c in chart_types.split(",")] if chart_types else None

    # Delegate to pure function
    result = run_paradigm_analysis_core(
        paradigm=paradigm,
        file_pattern=real_file_pattern,
        groups=groups_dict,
        metrics=metrics_list,
        chart_types=chart_types_list,
        output_dir=real_output_dir,
        handoff_path=real_handoff_path,
    )
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)
