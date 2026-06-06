"""统一的脚本 CLI helper。

所有 ethoinsight.scripts.* 下的脚本通过本模块统一参数解析、I/O 接口。

接口约定见 docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md §3.2
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


# ============================================================================
# Stdout marker - subagent uses this to extract per-script result
# ============================================================================

RESULT_MARKER = "[result]"


def emit_result(payload: dict[str, Any]) -> None:
    """Print a `[result] {json}` line to stdout for subagent extraction.

    Also runs metric validation (NaN / Inf / out-of-range) on the payload
    and prints VALIDATION_ERROR lines when violations are found.  These are
    informational — the result is still emitted so downstream can decide
    how to handle partial data.
    """
    from ethoinsight.validate import validate_metrics

    # Validate the result value if it looks like a single-metric payload.
    if "metric" in payload and "value" in payload:
        violations = validate_metrics({payload["metric"]: payload["value"]})
        for v in violations:
            print(
                f"VALIDATION_ERROR: {v['metric']}: {v['issue']} (value={v['value']})"
            )

    print(f"{RESULT_MARKER} {json.dumps(payload, ensure_ascii=False)}")


# ============================================================================
# File I/O
# ============================================================================


def save_output_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write `data` to `path` atomically (temp file + rename), creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, p)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def read_inputs_json(path: str | Path) -> list[str]:
    """Read a JSON file containing a list of input file paths.

    Format: ``["/path/to/subject1.txt", "/path/to/subject2.txt", ...]``
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"{path} must be a JSON array of file paths, got {type(data).__name__}"
        )
    return [str(item) for item in data]


def read_groups_json(path: str | Path) -> dict[str, list[str]]:
    """Read a JSON file containing the groups mapping.

    Format: ``{"group_name": ["subject_name_1", "subject_name_2"], ...}``
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must be a JSON object mapping group names to subject lists"
        )
    return {k: list(v) for k, v in data.items()}


def resolve_per_subject_input(args: argparse.Namespace) -> str:
    """For per-subject plots: return the single trajectory path from args.

    Accepts either:
      - ``--input <path>`` (legacy single-file form)
      - ``--inputs <json>`` (uniform form; reads the FIRST path from the JSON array)

    Raises ValueError if neither is provided or inputs.json is empty.
    """
    if getattr(args, "input", None):
        return args.input
    if getattr(args, "inputs", None):
        paths = read_inputs_json(args.inputs)
        if not paths:
            raise ValueError(f"{args.inputs} is an empty JSON array")
        return paths[0]
    raise ValueError("plot script requires --input or --inputs")


# ============================================================================
# Standard argparse builders for the three script types
# ============================================================================


def make_compute_parser(description: str) -> argparse.ArgumentParser:
    """Argparse for ``compute_*.py``: --input + --output + --parameters-json."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--input", required=True, help="Path to a single EthoVision trajectory file"
    )
    ap.add_argument("--output", required=True, help="Path to write the metric JSON")
    # === Sprint 2b ===
    ap.add_argument(
        "--parameters-json",
        default="{}",
        help=(
            "JSON dict of parameters (e.g. velocity_threshold, pendulum_*). "
            "Empty dict means use script-side defaults. "
            "Populated by catalog.resolve from PlanMetric.parameters_in_use."
        ),
    )
    return ap


def parse_parameters(args_namespace: argparse.Namespace) -> dict[str, float | int | str]:
    """Parse args.parameters_json into dict. Returns {} on empty/invalid (with stderr warning).

    Sprint 2b: shared by all compute_*.py scripts for uniform parameter parsing.
    """
    raw = getattr(args_namespace, "parameters_json", "") or "{}"
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[warning] --parameters-json parse failed ({e}); falling back to script defaults", file=sys.stderr)
        return {}
    if not isinstance(result, dict):
        print(f"[warning] --parameters-json must be a JSON object, got {type(result).__name__}", file=sys.stderr)
        return {}
    return result


def make_plot_parser(
    description: str, *, supports_groups: bool = False
) -> argparse.ArgumentParser:
    """Argparse for ``plot_*.py``.

    Single-file plots use ``--input``; aggregated plots use ``--inputs`` + optional ``--groups``.
    """
    ap = argparse.ArgumentParser(description=description)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", help="Path to a single trajectory file (single-file plots)"
    )
    group.add_argument(
        "--inputs",
        help="Path to a JSON file containing a list of trajectory file paths",
    )
    if supports_groups:
        ap.add_argument(
            "--groups",
            help="Path to a JSON file mapping group_name -> [subject_name, ...]",
        )
    ap.add_argument("--output", required=True, help="Path to write the PNG plot")
    return ap


def make_stats_parser(description: str) -> argparse.ArgumentParser:
    """Argparse for ``run_*_stats.py``: --inputs --groups --output."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--inputs",
        required=True,
        help="Path to a JSON file containing a list of trajectory file paths",
    )
    ap.add_argument(
        "--groups",
        required=True,
        help="Path to a JSON file mapping group_name -> [subject_name, ...]",
    )
    ap.add_argument("--output", required=True, help="Path to write the stats JSON")
    return ap
