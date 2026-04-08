"""Analysis template tool for code-executor subagent.

Provides a LangChain tool that returns a ready-to-run Python analysis script
with user-specified parameters filled in, based on paradigm-specific templates.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.tools import tool

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
        paradigm: Analysis paradigm name. Examples: "shoaling", "open_field", "epm", "novel_object", "y_maze", "forced_swim".
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
