"""ethoinsight.parse — EthoVision XT data file parser.

Handles EthoVision XT exported trajectory files in three formats:
- TXT: UTF-16 LE with BOM, semicolon-delimited, EthoVision header-line-count marker
- CSV: UTF-16 LE with BOM, semicolon-delimited
- XLSX/XLS: Excel export with EthoVision header-line-count marker in row 0, column 0

The marker is locale-dependent: "标题行数：" (Chinese) or "Number of header lines:" (English).
"""

from __future__ import annotations

import glob as glob_module
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ethoinsight.utils import detect_paradigm, normalize_columns


def detect_ethovision(file_path: str) -> bool:
    """Detect whether a file is an EthoVision export.

    Supports TXT/CSV (UTF-16 LE with BOM) and XLSX/XLS (Excel export
    with locale-dependent header-line-count marker in row 0, column 0:
    "标题行数：" or "Number of header lines:").
    """
    real_path, sheet_name = _parse_path_and_sheet(file_path)
    path = real_path
    if not path.exists() or not path.is_file():
        return False

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return _detect_ethovision_xlsx(path, sheet_name)
    if suffix not in (".txt", ".csv"):
        return False

    try:
        with open(path, "rb") as f:
            header = f.read(512)
    except OSError:
        return False

    # Check UTF-16 LE BOM
    if not header.startswith(b"\xff\xfe"):
        return False

    # Decode and check for EthoVision markers
    try:
        text = header.decode("utf-16-le")
    except (UnicodeDecodeError, ValueError):
        return False

    first_line = text.split("\r\n")[0].strip("\ufeff").strip()

    # Trajectory file: starts with locale-dependent header-line-count marker
    if "标题行数" in first_line or "number of header lines" in first_line.lower():
        return True

    # Statistics file: semicolon-separated quoted fields
    if '"' in first_line and ";" in first_line:
        return True

    return False


def _parse_path_and_sheet(file_path: str) -> tuple[Path, str | int]:
    """Split file_path into (real_path, sheet_name).

    Supports the ``path::SheetName`` convention for referencing a specific
    sheet in a multi-sheet XLSX file.  If no ``::`` separator is present,
    sheet_name defaults to 0 (first sheet).
    """
    if "::" in file_path:
        path_str, sheet = file_path.rsplit("::", 1)
        return Path(path_str), sheet
    return Path(file_path), 0


def _detect_ethovision_xlsx(path: Path, sheet_name: str | int = 0) -> bool:
    """Detect whether an XLSX/XLS file (or specific sheet) is an EthoVision export.

    Reads only the first cell (row 0, column 0) and checks for
    the EthoVision header-line-count marker (locale-dependent:
    "标题行数" for Chinese, "Number of header lines" for English).
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=1)
    except Exception:
        return False
    if df.empty or df.iloc[0, 0] is None:
        return False
    first_cell = str(df.iloc[0, 0]).lower()
    return "标题行数" in first_cell or "number of header lines" in first_cell


def parse_header(file_path: str) -> dict:
    """Parse the header section of an EthoVision trajectory file.

    Returns a dict with:
        header_lines: int — number of header lines declared
        experiment: str — experiment name
        trial_name: str — trial name
        subject: str — subject/object name
        start_time: str — recording start time
        duration: str — trial duration
        arena: str — arena name
        paradigm: str | None — auto-detected paradigm
        columns: list[str] — normalized column names
        units: list[str] — unit strings
        raw_metadata: dict — all key-value pairs from header
    """
    real_path, sheet_name = _parse_path_and_sheet(file_path)
    path = real_path
    if path.suffix.lower() in (".xlsx", ".xls"):
        return _parse_header_xlsx(path, sheet_name)

    with open(path, "r", encoding="utf-16-le") as f:
        lines = f.readlines()

    # Strip BOM from first line
    lines[0] = lines[0].lstrip("\ufeff")

    # Parse header line count
    first_line = lines[0].strip()
    match = re.search(r'"(\d+)"', first_line)
    if not match:
        raise ValueError(f"Cannot parse header line count from: {first_line}")
    header_lines = int(match.group(1))

    # Parse metadata key-value pairs from header
    raw_metadata: dict[str, str] = {}
    for i in range(1, min(header_lines - 2, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        parts = _split_semicolons(line)
        if len(parts) >= 2:
            key = parts[0].strip('"').strip()
            value = parts[1].strip('"').strip()
            if key:
                raw_metadata[key] = value

    # Parse column names (line at header_lines - 1, 0-indexed: header_lines - 2)
    col_line_idx = header_lines - 2
    unit_line_idx = header_lines - 1
    raw_col_names = _split_semicolons(lines[col_line_idx].strip())
    raw_col_names = [
        c.strip('"').strip() for c in raw_col_names if c.strip('"').strip()
    ]
    columns = normalize_columns(raw_col_names)

    # Parse units
    units: list[str] = []
    if unit_line_idx < len(lines):
        raw_units = _split_semicolons(lines[unit_line_idx].strip())
        units = [u.strip('"').strip() for u in raw_units]
        # Trim to match column count
        units = units[: len(columns)]

    # Extract common metadata fields
    experiment = raw_metadata.get("实验", raw_metadata.get("Experiment", ""))
    paradigm = detect_paradigm(experiment)

    return {
        "header_lines": header_lines,
        "experiment": experiment,
        "trial_name": raw_metadata.get("试验名称", raw_metadata.get("Trial Name", "")),
        "subject": raw_metadata.get("对象名称", raw_metadata.get("Subject", "")),
        "start_time": raw_metadata.get("开始时间", raw_metadata.get("Start Time", "")),
        "duration": raw_metadata.get(
            "试验持续时间", raw_metadata.get("Trial Duration", "")
        ),
        "arena": raw_metadata.get("观察区名称", raw_metadata.get("Arena", "")),
        "paradigm": paradigm,
        "columns": columns,
        "units": units,
        "raw_metadata": raw_metadata,
    }


def _parse_header_xlsx(path: Path, sheet_name: str | int = 0) -> dict:
    """Parse the header section of an EthoVision XLSX trajectory file.

    XLSX structure (rows):
      R0:         标题行数: <N>
      R1..R(N-3): metadata key-value pairs (col 0 = key, col 1 = value)
      R(N-2):     column names
      R(N-1):     units
      RN:         data starts
    """
    df_header = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=None)
    header_lines = int(df_header.iloc[0, 1])

    raw_metadata: dict[str, str] = {}
    for i in range(1, header_lines - 2):
        if i >= len(df_header):
            break
        key = df_header.iloc[i, 0]
        val = df_header.iloc[i, 1]
        if pd.isna(key) or (isinstance(key, str) and not key.strip()):
            continue
        raw_metadata[str(key).strip()] = str(val).strip() if not pd.isna(val) else ""

    col_line_idx = header_lines - 2
    unit_line_idx = header_lines - 1

    raw_col_names = [
        str(c).strip() for c in df_header.iloc[col_line_idx].tolist()
        if pd.notna(c) and str(c).strip()
    ]
    columns = normalize_columns(raw_col_names)

    units: list[str] = []
    if unit_line_idx < len(df_header):
        raw_units = df_header.iloc[unit_line_idx].tolist()
        units = [
            str(u).strip() if pd.notna(u) else ""
            for u in raw_units[: len(columns)]
        ]

    experiment = raw_metadata.get("实验", raw_metadata.get("Experiment", ""))
    paradigm = detect_paradigm(experiment)

    return {
        "header_lines": header_lines,
        "experiment": experiment,
        "trial_name": raw_metadata.get("试验名称", raw_metadata.get("Trial Name", "")),
        "subject": raw_metadata.get("对象名称", raw_metadata.get("Subject", "")),
        "start_time": raw_metadata.get("开始时间", raw_metadata.get("Start Time", "")),
        "duration": raw_metadata.get("试验持续时间", raw_metadata.get("Trial Duration", "")),
        "arena": raw_metadata.get("观察区名称", raw_metadata.get("Arena", "")),
        "paradigm": paradigm,
        "columns": columns,
        "units": units,
        "raw_metadata": raw_metadata,
    }


def parse_trajectory(file_path: str) -> pd.DataFrame:
    """Parse a single EthoVision trajectory file into a DataFrame.

    - Skips header rows (count read from line 1)
    - Normalizes column names to English snake_case
    - Converts "-" to NaN
    - Converts numeric columns
    - Stores metadata in df.attrs
    """
    real_path, sheet_name = _parse_path_and_sheet(file_path)
    path = real_path
    if path.suffix.lower() in (".xlsx", ".xls"):
        return _parse_trajectory_xlsx(path, sheet_name)

    header = parse_header(file_path)
    header_lines = header["header_lines"]

    # Read data rows, skipping header + units line
    # Data starts at line header_lines + 1 (0-indexed: header_lines)
    # But pandas skiprows is 0-indexed, and we also skip the units line
    path = Path(file_path)
    with open(path, "r", encoding="utf-16-le") as f:
        all_lines = f.readlines()

    # BOM strip
    all_lines[0] = all_lines[0].lstrip("\ufeff")

    # Data lines start after header_lines (column names) + 1 (units)
    data_start = header_lines  # 0-indexed, this is the units line; data is next
    data_lines = all_lines[data_start:]

    if not data_lines:
        df = pd.DataFrame(columns=header["columns"])
        df.attrs.update(_header_to_attrs(header))
        return df

    # Parse data rows
    rows = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        values = _split_semicolons(line)
        # Strip quotes
        values = [v.strip('"').strip() for v in values]
        # Trim to column count
        values = values[: len(header["columns"])]
        rows.append(values)

    df = pd.DataFrame(rows, columns=header["columns"])

    # Convert "-" to NaN
    df = df.replace("-", np.nan)
    df = df.replace("", np.nan)

    # Convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    # Store metadata
    df.attrs.update(_header_to_attrs(header))

    return df


def _parse_trajectory_xlsx(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Parse a single EthoVision XLSX trajectory file (or sheet) into a DataFrame.

    Uses parse_header (XLSX branch) for metadata and column names,
    then reads data rows with pd.read_excel(skiprows=header_lines).
    """
    header = _parse_header_xlsx(path, sheet_name)
    header_lines = header["header_lines"]

    df = pd.read_excel(path, sheet_name=sheet_name, header=None, skiprows=header_lines)
    df.columns = header["columns"]

    # Convert "-" to NaN
    df = df.replace("-", np.nan)
    df = df.replace("", np.nan)

    # Convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    df.attrs.update(_header_to_attrs(header))
    return df


def parse_batch(file_paths: list[str] | str) -> dict:
    """Batch-parse EthoVision trajectory files.

    Args:
        file_paths: List of file paths, or a glob pattern string.

    Returns:
        {
            "metadata": dict — merged header metadata,
            "subjects": dict[str, DataFrame] — per-subject DataFrames,
            "all_data": DataFrame — concatenated with 'subject' and 'file' columns,
            "summary": dict — summary statistics,
        }
    """
    # Resolve file paths
    if isinstance(file_paths, str):
        paths = sorted(glob_module.glob(file_paths))
    else:
        paths = list(file_paths)

    # Filter to trajectory files only (must have header count).
    # Supports ::sheet_name suffix for multi-sheet XLSX — strip it before
    # filesystem checks, pass the full path to detect_ethovision.
    trajectory_paths = []
    for p in paths:
        clean = p.split("::")[0] if "::" in p else p
        if not Path(clean).exists():
            continue
        try:
            if detect_ethovision(p):
                trajectory_paths.append(p)
        except Exception:
            continue

    if not trajectory_paths:
        return {
            "metadata": {},
            "subjects": {},
            "all_data": pd.DataFrame(),
            "summary": {
                "total_files": 0,
                "total_rows": 0,
                "subjects": [],
                "paradigm": None,
                "columns": [],
                "duration_seconds": 0,
            },
        }

    # Parse all files
    subjects: dict[str, pd.DataFrame] = {}
    all_dfs: list[pd.DataFrame] = []
    first_header = None

    for p in trajectory_paths:
        df = parse_trajectory(p)
        subject = df.attrs.get("subject", Path(p).stem)

        # Handle duplicate subject names (e.g., multiple trials)
        key = subject
        if key in subjects:
            # Append trial info to distinguish
            trial = df.attrs.get("trial_name", "")
            key = f"{subject}_{trial}" if trial else f"{subject}_{len(subjects)}"

        subjects[key] = df

        # Add subject/file columns for concatenation
        df_copy = df.copy()
        df_copy["subject"] = subject
        df_copy["file"] = Path(p).name
        all_dfs.append(df_copy)

        if first_header is None:
            first_header = df.attrs.copy()

    # Concatenate
    all_data = pd.concat(all_dfs, ignore_index=True)

    # Build summary
    paradigm = first_header.get("paradigm") if first_header else None
    columns = first_header.get("columns", []) if first_header else []

    # Parse duration from first file
    duration_str = first_header.get("duration", "") if first_header else ""
    duration_seconds = _parse_duration(duration_str)

    unique_subjects = sorted(
        set(df.attrs.get("subject", "") for df in subjects.values())
    )

    return {
        "metadata": first_header or {},
        "subjects": subjects,
        "all_data": all_data,
        "summary": {
            "total_files": len(trajectory_paths),
            "total_rows": len(all_data),
            "subjects": unique_subjects,
            "paradigm": paradigm,
            "columns": columns,
            "duration_seconds": duration_seconds,
        },
    }


def get_summary(parsed_data: dict, max_chars: int = 2000) -> str:
    """Generate an Agent-friendly text summary of parsed data.

    Designed to fit within LLM context limits while providing
    enough information for the Agent to understand the data.
    """
    summary = parsed_data.get("summary", {})
    metadata = parsed_data.get("metadata", {})
    all_data = parsed_data.get("all_data", pd.DataFrame())

    lines = []
    lines.append("=== EthoVision Data Summary ===")
    lines.append(f"Experiment: {metadata.get('experiment', 'Unknown')}")
    lines.append(f"Paradigm: {summary.get('paradigm', 'Unknown')}")
    lines.append(f"Files: {summary.get('total_files', 0)}")
    lines.append(f"Total rows: {summary.get('total_rows', 0)}")
    lines.append(f"Subjects: {', '.join(summary.get('subjects', []))}")
    lines.append(f"Duration: {summary.get('duration_seconds', 0):.1f}s per trial")
    lines.append(f"Columns: {', '.join(summary.get('columns', []))}")

    # Add basic statistics for numeric columns
    if not all_data.empty:
        lines.append("")
        lines.append("--- Numeric Column Stats ---")
        numeric_cols = all_data.select_dtypes(include="number").columns
        # Exclude metadata columns
        skip = {"result_1", "result_2", "subject", "file"}
        numeric_cols = [c for c in numeric_cols if c not in skip]

        for col in numeric_cols[:10]:  # Limit to 10 columns
            series = all_data[col].dropna()
            if len(series) == 0:
                continue
            line = f"  {col}: mean={series.mean():.4f}, std={series.std():.4f}, range=[{series.min():.4f}, {series.max():.4f}]"
            lines.append(line)

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


# ============================================================================
# Internal helpers
# ============================================================================


def _split_semicolons(line: str) -> list[str]:
    """Split a semicolon-delimited line, respecting quoted fields."""
    parts = line.split(";")
    # Remove trailing empty part (lines often end with ";")
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return parts


def _parse_duration(duration_str: str) -> float:
    """Parse EthoVision duration string like "+ 00:05:00.200" to seconds."""
    if not duration_str:
        return 0.0
    # Remove leading "+", spaces
    s = duration_str.strip().lstrip("+").strip()
    match = re.match(r"(\d+):(\d+):(\d+(?:\.\d+)?)", s)
    if match:
        h, m, sec = float(match.group(1)), float(match.group(2)), float(match.group(3))
        return h * 3600 + m * 60 + sec
    return 0.0


def _header_to_attrs(header: dict) -> dict:
    """Convert header dict to DataFrame attrs (flat, serializable)."""
    return {
        "experiment": header.get("experiment", ""),
        "trial_name": header.get("trial_name", ""),
        "subject": header.get("subject", ""),
        "start_time": header.get("start_time", ""),
        "duration": header.get("duration", ""),
        "arena": header.get("arena", ""),
        "paradigm": header.get("paradigm"),
        "columns": header.get("columns", []),
        "units": header.get("units", []),
    }


def infer_groups_from_result_block(subjects: list[dict]) -> dict[str, list[str]] | None:
    """Infer groupings from EthoVision 'result block name'.

    EthoVision 数据选择配置的"结果块命名"出现在 raw 文件末列。
    - 默认占位名（"Result 1", "Result 2", ...）→ 视为未分组，返回 None
    - 规范命名（"Drug" / "Saline" / "Control" 等）→ 按命名聚合

    Source: 2026-05-13 同事反馈 Q3。

    Args:
        subjects: list of dicts with keys 'name' (subject name) and optional
                  'result_block_name' (str)

    Returns:
        {group_name: [subject_name, ...]} 或 None（未分组）
    """
    import re

    if not subjects:
        return None

    block_names: dict[str, list[str]] = {}
    for s in subjects:
        block = s.get("result_block_name")
        if not block:
            return None
        if re.fullmatch(r"Result\s+\d+", block.strip()):
            return None
        block_names.setdefault(block, []).append(s["name"])

    if len(block_names) < 2:
        return None
    return block_names
