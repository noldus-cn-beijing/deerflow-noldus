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
        "group_level_metrics": m_result.get("group_level_metrics", {}),
        "statistics": stat_results,
        "assessment": assessment,
        "metadata": {
            "paradigm": paradigm,
            "n_files": n_files,
            "groups": groups,
        },
        "data_quality_warnings": m_result.get("data_quality_warnings", []),
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


# ============================================================================
# Granular tool wrappers (introduced 2026-04-17)
# ============================================================================

import pickle


def _get_thread_data(runtime: ToolRuntime | None) -> dict | None:
    """Extract thread_data from runtime.state safely."""
    try:
        if runtime is not None and runtime.state is not None:
            return runtime.state.get("thread_data")
    except Exception:
        return None
    return None


def _write_json(path: str, obj: dict) -> None:
    """Write a dict as JSON to path, creating parent dirs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _read_json(path: str) -> dict:
    """Read a JSON file to dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fail(error: str, **extra) -> str:
    """Return a failure JSON string for tool outputs."""
    result = {"status": "failed", "error": error, **extra}
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


def _ok(**fields) -> str:
    """Return a success JSON string for tool outputs."""
    result = {"status": "completed", **fields}
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


@tool("parse_trajectories", parse_docstring=True)
def parse_trajectories_tool(
    runtime: ToolRuntime,
    file_pattern: str,
    workspace_dir: str = "/mnt/user-data/workspace/",
) -> str:
    """Parse EthoVision XT trajectory files (auto-handles UTF-16 encoding).

    Matches files by glob pattern, parses all matched files into DataFrames,
    then saves both a full pickle (for downstream tools) and a JSON summary
    (for LLM inspection) to workspace_dir.

    Args:
        file_pattern: Glob pattern. Example: "/mnt/user-data/uploads/*.txt"
        workspace_dir: Workspace directory for intermediate files. Default: "/mnt/user-data/workspace/"

    Returns:
        JSON with status, n_files, subjects, columns, pkl_path, and quality_warnings on success.
        On failure, returns status="failed" with error details and matched_files hint.
    """
    from ethoinsight import parse as etho_parse

    thread_data = _get_thread_data(runtime)
    real_pattern = _resolve_virtual_path(file_pattern, thread_data)
    real_workspace = _resolve_virtual_path(workspace_dir, thread_data)
    os.makedirs(real_workspace, exist_ok=True)

    try:
        data = etho_parse.parse_batch(real_pattern)
    except Exception as e:
        matched = glob.glob(real_pattern)
        return _fail(f"parse_batch raised: {e}", file_pattern=file_pattern, matched_files=matched[:5])

    if data["summary"]["total_files"] == 0:
        matched = glob.glob(real_pattern)
        return _fail(
            f"No files matched pattern '{file_pattern}'",
            matched_files=matched[:5],
        )

    # Save full data as pickle
    pkl_path = os.path.join(real_workspace, "parsed.pkl")
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        return _fail(f"Failed to write pickle: {e}", pkl_path=pkl_path)

    # Build LLM-facing summary
    summary = data["summary"]
    quality_warnings = []
    if summary["total_files"] < 3:
        quality_warnings.append(f"Only {summary['total_files']} file(s) — sample size likely insufficient for reliable statistics")
    if summary["total_rows"] < 100:
        quality_warnings.append(f"Very few data rows ({summary['total_rows']}) — check data integrity")

    summary_path = os.path.join(real_workspace, "parsed_summary.json")
    summary_payload = {
        "status": "completed",
        "n_files": summary["total_files"],
        "total_rows": summary["total_rows"],
        "subjects": summary.get("subjects", []),
        "paradigm_hint": summary.get("paradigm", "unknown"),
        "columns": summary.get("columns", []),
        "duration_seconds": summary.get("duration_seconds", 0.0),
        "quality_warnings": quality_warnings,
        "pkl_path": pkl_path,
    }
    _write_json(summary_path, summary_payload)

    return _ok(
        n_files=summary_payload["n_files"],
        subjects=summary_payload["subjects"],
        columns=summary_payload["columns"],
        quality_warnings=quality_warnings,
        pkl_path=pkl_path,
        summary_path=summary_path,
    )


@tool("compute_metrics", parse_docstring=True)
def compute_metrics_tool(
    runtime: ToolRuntime,
    paradigm: str,
    groups: str,
    metrics: str | None = None,
    workspace_dir: str = "/mnt/user-data/workspace/",
    output_dir: str = "/mnt/user-data/outputs/",
) -> str:
    """Compute paradigm-specific behavioral metrics from parsed trajectory data.

    Requires parse_trajectories to have run first (reads workspace/parsed.pkl).
    Writes metrics.pkl (for downstream tools) and metrics_summary.json (for LLM).

    Args:
        paradigm: Paradigm name. Example: "shoaling"
        groups: JSON string mapping group name to subject list. Example: '{"control":["Subject 1"],"treatment":["Subject 3"]}'
        metrics: Optional comma-separated metric names. Uses paradigm defaults if omitted.
        workspace_dir: Workspace directory. Default: "/mnt/user-data/workspace/"
        output_dir: Output directory for metrics.csv. Default: "/mnt/user-data/outputs/"

    Returns:
        JSON with status, computed_metrics, group_summary (mean/std/n, no raw values), and file paths.
    """
    from ethoinsight import metrics as etho_metrics

    thread_data = _get_thread_data(runtime)
    real_workspace = _resolve_virtual_path(workspace_dir, thread_data)
    real_output = _resolve_virtual_path(output_dir, thread_data)
    os.makedirs(real_output, exist_ok=True)

    pkl_path = os.path.join(real_workspace, "parsed.pkl")
    if not os.path.exists(pkl_path):
        return _fail(
            f"missing dependency: {pkl_path}. Run parse_trajectories first.",
            missing_file=pkl_path,
        )

    try:
        groups_dict = json.loads(groups)
    except (json.JSONDecodeError, TypeError) as e:
        return _fail(f"Invalid groups JSON: {e}. Expected format: '{{\"control\":[\"Subject 1\"]}}'")

    metrics_list = [m.strip() for m in metrics.split(",")] if metrics else None

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return _fail(f"Failed to load parsed.pkl: {e}")

    try:
        m_result = etho_metrics.compute_paradigm_metrics(
            data, paradigm, groups=groups_dict, metrics=metrics_list,
        )
    except Exception as e:
        return _fail(f"compute_paradigm_metrics raised: {e}", paradigm=paradigm)

    # Save full metrics as pickle
    metrics_pkl = os.path.join(real_workspace, "metrics.pkl")
    try:
        with open(metrics_pkl, "wb") as f:
            pickle.dump(m_result, f)
    except Exception as e:
        return _fail(f"Failed to write metrics.pkl: {e}")

    # Save CSV
    csv_path = os.path.join(real_output, "metrics.csv")
    try:
        etho_metrics.save_to_csv(m_result, csv_path)
    except Exception as e:
        csv_path = ""  # non-fatal

    # Build compact group_summary (drop raw values)
    compact: dict = {}
    for grp, metric_dict in m_result.get("group_summary", {}).items():
        compact[grp] = {
            mname: {k: v for k, v in vals.items() if k != "values"}
            for mname, vals in metric_dict.items()
        }

    # Quality warnings
    quality_warnings = []
    for grp, metric_dict in compact.items():
        for mname, vals in metric_dict.items():
            if vals.get("n", 0) < 3:
                quality_warnings.append(f"group '{grp}' metric '{mname}': n={vals.get('n')} — underpowered")
            if vals.get("std", 1.0) == 0:
                quality_warnings.append(f"group '{grp}' metric '{mname}': zero variance — all subjects identical")
    # Structured warnings from compute_paradigm_metrics (e.g. single-subject shoaling).
    for w in m_result.get("data_quality_warnings", []):
        quality_warnings.append(f"[{w.get('severity', 'info')}] {w.get('metric', '?')}: {w.get('message', '')}")

    summary_path = os.path.join(real_workspace, "metrics_summary.json")
    computed = m_result.get("metadata", {}).get("computed_metrics", [])
    summary_payload = {
        "status": "completed",
        "paradigm": paradigm,
        "computed_metrics": computed,
        "group_summary": compact,
        "quality_warnings": quality_warnings,
        "pkl_path": metrics_pkl,
        "csv_path": csv_path,
    }
    _write_json(summary_path, summary_payload)

    return _ok(
        paradigm=paradigm,
        computed_metrics=computed,
        group_summary=compact,
        quality_warnings=quality_warnings,
        pkl_path=metrics_pkl,
        csv_path=csv_path,
        summary_path=summary_path,
    )


@tool("run_statistics", parse_docstring=True)
def run_statistics_tool(
    runtime: ToolRuntime,
    alpha: float = 0.05,
    correction: str = "bonferroni",
    workspace_dir: str = "/mnt/user-data/workspace/",
    output_dir: str = "/mnt/user-data/outputs/",
) -> str:
    """Run statistical comparisons across groups with automatic test selection.

    Uses Shapiro-Wilk normality + Levene homogeneity internally to choose between
    parametric (t-test/ANOVA) and non-parametric (Mann-Whitney/Kruskal-Wallis).
    Requires compute_metrics to have run first (reads workspace/metrics.pkl).

    Args:
        alpha: Significance threshold. Default: 0.05
        correction: Multiple comparison correction method. Default: "bonferroni"
        workspace_dir: Workspace directory. Default: "/mnt/user-data/workspace/"
        output_dir: Output directory for statistics.json. Default: "/mnt/user-data/outputs/"

    Returns:
        JSON with status, comparisons (by metric), summary, and method_warnings.
    """
    from ethoinsight import statistics as etho_stats

    thread_data = _get_thread_data(runtime)
    real_workspace = _resolve_virtual_path(workspace_dir, thread_data)
    real_output = _resolve_virtual_path(output_dir, thread_data)
    os.makedirs(real_output, exist_ok=True)

    metrics_pkl = os.path.join(real_workspace, "metrics.pkl")
    if not os.path.exists(metrics_pkl):
        return _fail(
            f"missing dependency: {metrics_pkl}. Run compute_metrics first.",
            missing_file=metrics_pkl,
        )

    try:
        with open(metrics_pkl, "rb") as f:
            m_result = pickle.load(f)
    except Exception as e:
        return _fail(f"Failed to load metrics.pkl: {e}")

    groups_list = list(m_result.get("group_summary", {}).keys())
    if len(groups_list) < 2:
        return _fail(f"Need at least 2 groups for comparison, got {len(groups_list)}: {groups_list}")

    try:
        stat_results = etho_stats.compare_groups(
            m_result, groups_list, alpha=alpha, correction=correction,
        )
    except Exception as e:
        return _fail(f"compare_groups raised: {e}")

    # Method warnings: detect small-n + parametric mismatch
    method_warnings = []
    for metric_name, comps in stat_results.get("comparisons", {}).items():
        for comp in comps if isinstance(comps, list) else [comps]:
            test_used = comp.get("test_used", "")
            n1 = comp.get("n1", 0)
            n2 = comp.get("n2", 0)
            if "t-test" in test_used.lower() and min(n1, n2) < 5:
                method_warnings.append(
                    f"{metric_name}: test={test_used} with n=({n1},{n2}) — consider non-parametric"
                )

    stats_path = os.path.join(real_output, "statistics.json")
    _write_json(stats_path, stat_results)

    ws_stats_path = os.path.join(real_workspace, "statistics.json")
    full_payload = {
        "status": "completed",
        "comparisons": stat_results.get("comparisons", {}),
        "summary": stat_results.get("summary", ""),
        "alpha": stat_results.get("alpha", alpha),
        "correction": stat_results.get("correction", correction),
        "method_warnings": method_warnings,
        "output_path": stats_path,
    }
    _write_json(ws_stats_path, full_payload)

    return _ok(
        n_metrics=len(stat_results.get("comparisons", {})),
        summary=stat_results.get("summary", ""),
        method_warnings=method_warnings,
        output_path=stats_path,
        workspace_path=ws_stats_path,
    )


@tool("generate_charts", parse_docstring=True)
def generate_charts_tool(
    runtime: ToolRuntime,
    chart_types: str = "box_plot",
    metrics: str | None = None,
    include_trajectory: bool = True,
    include_timeseries: bool = True,
    workspace_dir: str = "/mnt/user-data/workspace/",
    output_dir: str = "/mnt/user-data/outputs/",
) -> str:
    """Generate publication-quality charts from metrics and statistics results.

    Requires compute_metrics to have run. Reads workspace/metrics.pkl,
    workspace/statistics.json (optional), workspace/parsed.pkl (for trajectory).
    Generates PNG files in output_dir and records paths in workspace/charts.json.

    Args:
        chart_types: Comma-separated chart types from {box_plot, violin_plot, bar_chart, raincloud_plot, beeswarm_plot}. Default: "box_plot"
        metrics: Optional comma-separated metric names to plot. Defaults to all computed metrics.
        include_trajectory: Whether to generate trajectory_plot from parsed data. Default: True
        include_timeseries: Whether to generate shoaling timeseries plots (inter_individual_distance, group_polarity). Default: True
        workspace_dir: Workspace directory. Default: "/mnt/user-data/workspace/"
        output_dir: Output directory for PNG files. Default: "/mnt/user-data/outputs/"

    Returns:
        JSON with status, chart_paths (list of PNG files), and errors per chart attempt.
    """
    from ethoinsight import charts as etho_charts

    thread_data = _get_thread_data(runtime)
    real_workspace = _resolve_virtual_path(workspace_dir, thread_data)
    real_output = _resolve_virtual_path(output_dir, thread_data)
    os.makedirs(real_output, exist_ok=True)

    metrics_pkl = os.path.join(real_workspace, "metrics.pkl")
    if not os.path.exists(metrics_pkl):
        return _fail(f"missing dependency: {metrics_pkl}. Run compute_metrics first.")

    with open(metrics_pkl, "rb") as f:
        m_result = pickle.load(f)

    stat_results: dict = {}
    stats_ws_path = os.path.join(real_workspace, "statistics.json")
    if os.path.exists(stats_ws_path):
        try:
            stat_payload = _read_json(stats_ws_path)
            stat_results = stat_payload.get("comparisons", {})
        except Exception:
            pass

    chart_type_list = [c.strip() for c in chart_types.split(",")]
    computed = m_result.get("metadata", {}).get("computed_metrics", [])
    metric_list = [m.strip() for m in metrics.split(",")] if metrics else computed

    chart_paths: list[str] = []
    errors: list[str] = []

    # Per-metric charts
    for metric in metric_list:
        for ct in chart_type_list:
            fn = getattr(etho_charts, ct, None)
            if fn is None:
                errors.append(f"Unknown chart type: {ct}")
                continue
            try:
                path = fn(
                    m_result, [metric],
                    significance=stat_results or None,
                    output_path=os.path.join(real_output, f"{metric}_{ct}.png"),
                )
                chart_paths.append(path)
            except Exception as e:
                errors.append(f"Chart {metric} {ct}: {e}")

    # Trajectory plot
    if include_trajectory:
        parsed_pkl = os.path.join(real_workspace, "parsed.pkl")
        if os.path.exists(parsed_pkl):
            try:
                with open(parsed_pkl, "rb") as f:
                    data = pickle.load(f)
                if data.get("all_data") is not None:
                    traj_path = etho_charts.trajectory_plot(
                        data["all_data"],
                        output_path=os.path.join(real_output, "trajectory.png"),
                    )
                    chart_paths.append(traj_path)
            except Exception as e:
                errors.append(f"Trajectory plot: {e}")

    # Timeseries plots (shoaling-specific)
    if include_timeseries:
        for ts_name, y_col in [("inter_individual_distance", "mean_iid"), ("group_polarity", "polarity")]:
            ts_df = m_result.get("timeseries", {}).get(ts_name)
            if ts_df is not None:
                try:
                    ts_path = etho_charts.timeseries_plot(
                        ts_df, y_col=y_col,
                        output_path=os.path.join(real_output, f"{ts_name}_timeseries.png"),
                    )
                    chart_paths.append(ts_path)
                except Exception as e:
                    errors.append(f"Timeseries {ts_name}: {e}")

    charts_json_path = os.path.join(real_workspace, "charts.json")
    payload = {"status": "completed", "chart_paths": chart_paths, "errors": errors}
    _write_json(charts_json_path, payload)

    return _ok(
        chart_paths=chart_paths,
        errors=errors,
        n_generated=len(chart_paths),
        output_path=charts_json_path,
    )


@tool("assess_and_handoff", parse_docstring=True)
def assess_and_handoff_tool(
    runtime: ToolRuntime,
    paradigm: str,
    groups: str,
    workspace_dir: str = "/mnt/user-data/workspace/",
    output_dir: str = "/mnt/user-data/outputs/",
    handoff_path: str = "/mnt/user-data/workspace/handoff_code_executor.json",
) -> str:
    """Run domain threshold assessment and synthesize the final handoff JSON.

    Requires compute_metrics and run_statistics to have run. Reads all intermediate
    workspace files, runs assess_results, and writes the handoff JSON expected by
    downstream data-analyst subagent.

    Args:
        paradigm: Paradigm name.
        groups: JSON string mapping group name to subject list (same as used in compute_metrics).
        workspace_dir: Workspace directory. Default: "/mnt/user-data/workspace/"
        output_dir: Output directory. Default: "/mnt/user-data/outputs/"
        handoff_path: Handoff JSON output path. Default: "/mnt/user-data/workspace/handoff_code_executor.json"

    Returns:
        JSON with status, handoff_path, summary text, and errors. The handoff file
        itself contains: metrics_summary, statistics, assessment, output_files, metadata.
    """
    from ethoinsight import assess as etho_assess

    thread_data = _get_thread_data(runtime)
    real_workspace = _resolve_virtual_path(workspace_dir, thread_data)
    real_output = _resolve_virtual_path(output_dir, thread_data)
    real_handoff = _resolve_virtual_path(handoff_path, thread_data)

    metrics_pkl = os.path.join(real_workspace, "metrics.pkl")
    if not os.path.exists(metrics_pkl):
        return _fail(f"missing dependency: {metrics_pkl}. Run compute_metrics first.")

    with open(metrics_pkl, "rb") as f:
        m_result = pickle.load(f)

    try:
        groups_dict = json.loads(groups)
    except Exception as e:
        return _fail(f"Invalid groups JSON: {e}")

    # Load stats
    stats_ws = os.path.join(real_workspace, "statistics.json")
    stat_results: dict = {}
    if os.path.exists(stats_ws):
        payload = _read_json(stats_ws)
        stat_results = {k: v for k, v in payload.items() if k in {"comparisons", "summary", "alpha", "correction"}}

    # Load chart paths
    charts_json = os.path.join(real_workspace, "charts.json")
    chart_paths: list[str] = []
    if os.path.exists(charts_json):
        chart_paths = _read_json(charts_json).get("chart_paths", [])

    # Load parse summary for n_files
    parsed_summary_path = os.path.join(real_workspace, "parsed_summary.json")
    n_files = 0
    if os.path.exists(parsed_summary_path):
        n_files = _read_json(parsed_summary_path).get("n_files", 0)

    # Run assessment
    assessment: dict = {}
    assess_errors: list[str] = []
    if stat_results:
        try:
            assessment = etho_assess.assess_results(
                stat_results, paradigm, metrics_result=m_result,
            )
        except Exception as e:
            assess_errors.append(f"assess_results raised: {e}")

    # Compact group_summary (drop values)
    compact: dict = {}
    for grp, metric_dict in m_result.get("group_summary", {}).items():
        compact[grp] = {
            mname: {k: v for k, v in vals.items() if k != "values"}
            for mname, vals in metric_dict.items()
        }

    # Aggregate all upstream warnings for errors field
    errors: list[str] = list(assess_errors)
    for ws_file in ["parsed_summary.json", "metrics_summary.json", "statistics.json", "charts.json"]:
        p = os.path.join(real_workspace, ws_file)
        if os.path.exists(p):
            try:
                payload = _read_json(p)
                for key in ["quality_warnings", "method_warnings", "errors"]:
                    errors.extend(payload.get(key, []))
            except Exception:
                pass

    n_subjects = len(m_result.get("per_subject", {}))
    summary_text = f"Analyzed {n_files} files, {n_subjects} subjects, paradigm: {paradigm}"
    if errors:
        summary_text += f" ({len(errors)} warning(s))"

    handoff = {
        "status": "completed",
        "summary": summary_text,
        "output_files": {
            "metrics": os.path.join(real_output, "metrics.csv"),
            "statistics": os.path.join(real_output, "statistics.json"),
            "charts": chart_paths,
        },
        "metrics_summary": compact,
        "group_level_metrics": m_result.get("group_level_metrics", {}),
        "statistics": stat_results,
        "assessment": assessment,
        "metadata": {
            "paradigm": paradigm,
            "n_files": n_files,
            "groups": groups_dict,
        },
        "data_quality_warnings": m_result.get("data_quality_warnings", []),
        "errors": errors,
    }
    _write_json(real_handoff, handoff)

    return _ok(
        handoff_path=real_handoff,
        summary=summary_text,
        n_errors=len(errors),
        errors=errors[:10],  # preview only
    )
