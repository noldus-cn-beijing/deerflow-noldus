"""inspect_uploaded_file — 解析上传的 EthoVision 数据文件，返回列清单 + 列识别评估。

供 lead agent 在 Gate 1 之后、prep_metric_plan 之前调用。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState
from ethoinsight.parse._core import detect_ethovision, parse_header, parse_trajectory
from ethoinsight.utils import assess_column_confidence

logger = logging.getLogger(__name__)

_ERROR_HINTS: dict[str, str] = {
    "no_files_provided": "uploaded_files 为空。把当前 <uploaded_files> 中所有数据文件传进来。",
    "workspace_missing": "thread_data.workspace_path 未设置——基础设施 bug。present_files 把错误信息呈现给用户。",
    "file_not_found": "数据文件不存在。用 ask_clarification 让用户重新上传。",
    "parse_failed": "数据文件解析失败。用 ask_clarification 让用户确认文件格式。",
    "format_unrecognized": "文件不是 EthoVision XT 导出格式。用 ask_clarification 让用户确认导出方式。",
}


def _error_result(code: str, message: str, failed_file: str | None = None) -> dict:
    hint = _ERROR_HINTS.get(code, "未知错误，请联系开发者。")
    result: dict = {"status": "error", "error_code": code, "message": message, "hint": hint}
    if failed_file:
        result["failed_file"] = failed_file
    return result


def _extract_required_patterns(paradigm: str) -> list[str]:
    """Extract deduplicated requires_columns patterns from a paradigm's catalog."""
    from ethoinsight.catalog.loader import load_catalog

    try:
        cat = load_catalog(paradigm)
    except Exception:
        return []

    patterns: set[str] = set()
    for m in cat.default_metrics:
        patterns.update(m.requires_columns)
    for m in cat.optional_metrics:
        patterns.update(m.requires_columns)
    return sorted(patterns)


def _compute_evidence(raw_column: str, values: list, column_index: int) -> dict:
    """Compute per-column evidence for unrecognized columns.

    For 0/1 columns: report proportion of 1s.
    For non-0/1 columns: flag as "疑似连续值/距离列".
    """
    import numpy as np

    evidence: dict = {"column_index": column_index}

    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        evidence["type"] = "empty_or_all_nan"
        return evidence

    unique = set(finite)
    if unique <= {0.0, 1.0} or unique <= {0, 1}:
        # Binary column — typical zone indicator
        ones = int(np.sum(finite == 1.0))
        total = len(finite)
        evidence["type"] = "binary_zone"
        evidence["proportion_ones"] = round(ones / total, 4) if total > 0 else 0.0
        evidence["total_rows"] = total
    else:
        evidence["type"] = "continuous_or_distance"
        evidence["min"] = round(float(np.min(finite)), 4)
        evidence["max"] = round(float(np.max(finite)), 4)
        evidence["mean"] = round(float(np.mean(finite)), 4)

    return evidence


@tool("inspect_uploaded_file", parse_docstring=True)
def inspect_uploaded_file_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    uploaded_files: list[str],
    paradigm: str | None = None,
) -> dict:
    """Parse uploaded EthoVision file(s) and assess column recognition.

    Call BEFORE prep_metric_plan when you need to know which columns the system
    recognises and which ones (e.g. custom zone columns) need user alignment.

    Args:
      uploaded_files: Virtual paths like ["/mnt/user-data/uploads/file1.txt", ...].
      paradigm: Optional paradigm key (e.g. "open_field"). When provided, catalog
          requires_columns patterns are used for rule (b) recognition, giving more
          accurate recognised/unrecognised splits. Without it only rules (a) and (c)
          apply — conservative, may miss zone columns in standard-named data.

    Returns:
      status="ok":
        {"status": "ok",
         "columns": [...],
         "column_assessment": {
             "recognized": [{"raw": str, "normalized": str}, ...],
             "unrecognized": [{"raw": str, "normalized": str, "evidence": {...}}, ...]
         },
         "open_questions": ["中心区", "边缘区", ...],   # unrecognized raw column names
         "header_metadata": {...}}
      status="error":
        {"status": "error", "error_code": str, "message": str, "hint": str}
    """
    if not uploaded_files:
        return _error_result("no_files_provided", "uploaded_files is empty")

    # Step 1: resolve workspace
    thread_data = runtime.state.get("thread_data") if runtime.state else None
    if not thread_data or not thread_data.get("workspace_path"):
        return _error_result("workspace_missing", "thread_data.workspace_path is not set")

    from deerflow.sandbox.tools import replace_virtual_path

    # Step 2: resolve required_patterns from catalog (if paradigm provided)
    required_patterns: list[str] = []
    if paradigm:
        required_patterns = _extract_required_patterns(paradigm)

    # Step 3: parse first file
    first_file = uploaded_files[0]
    clean_virtual = first_file.split("::")[0] if "::" in first_file else first_file
    real_first = replace_virtual_path(clean_virtual, thread_data)

    effective_path: str = real_first
    if "::" in first_file:
        effective_path = real_first + "::" + first_file.split("::", 1)[1]

    if not Path(real_first).exists():
        return _error_result(
            "file_not_found",
            f"File not found: {clean_virtual} (resolved to {real_first})",
            failed_file=clean_virtual,
        )

    if not detect_ethovision(effective_path):
        return _error_result(
            "format_unrecognized",
            f"Not an EthoVision export: {first_file}",
            failed_file=first_file,
        )

    try:
        header = parse_header(effective_path)
    except Exception as e:
        logger.warning("parse_header failed for %s: %s", first_file, e)
        return _error_result(
            "parse_failed",
            f"Failed to parse header: {e}",
            failed_file=first_file,
        )

    try:
        df = parse_trajectory(effective_path)
    except Exception as e:
        logger.warning("parse_trajectory failed for %s: %s", first_file, e)
        return _error_result(
            "parse_failed",
            f"Failed to parse trajectory data: {e}",
            failed_file=first_file,
        )

    columns = list(df.columns)
    raw_columns = [str(c).strip().strip('"').strip() for c in columns]

    # Step 4: assess column confidence
    assessment = assess_column_confidence(raw_columns, required_patterns)

    # Step 5: add evidence to unrecognized columns
    for entry in assessment.get("unrecognized", []):
        raw = entry["raw"]
        if raw in df.columns:
            col_idx = list(df.columns).index(raw)
            values = df.iloc[:, col_idx].values
            entry["evidence"] = _compute_evidence(raw, values, col_idx)
        else:
            entry["evidence"] = {"type": "column_not_in_dataframe"}

    # Step 6: build open_questions
    open_questions = [e["raw"] for e in assessment.get("unrecognized", [])]

    # Step 7: extract header metadata (Treatment/Dose/Animal ID for lead)
    header_metadata: dict = {}
    raw_metadata = header.get("raw_metadata", {})
    for key in ("Treatment", "Dose", "Animal ID", "Subject", "Arena"):
        if key in raw_metadata:
            header_metadata[key] = raw_metadata[key]

    return {
        "status": "ok",
        "columns": raw_columns,
        "column_assessment": assessment,
        "open_questions": open_questions,
        "header_metadata": header_metadata,
    }
