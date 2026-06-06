"""inspect_uploaded_file — lead 探查上传文件结构(sheets/columns/EV19 元数据/data preview),无需 bash。

lead 在调 prep_metric_plan 之前,需要知道:
- xlsx 文件有多少 sheet (FST/EPM 多 arena 场景)
- 每个 sheet 含哪些列
- **EV19 metadata header 中的分组信息**(Treatment / Dose / Animal ID / Group)
- **前几行数据**(data preview),作为 hard fact 供 lead 理解数据内容

Sprint 0/B fix 2026-05-28: 增加本 tool 解决 dogfood thread 485a899d 暴露的根因——
lead 没有 bash / general-purpose subagent,无法探查 xlsx 列出分组信息;之前死循环
反问"drug 列具体值是什么"但用户不知道。

2026-06-02 增强: 新增 data_preview 字段(前 N 行真实数据),让 lead 调一次
inspect 就拿到列名 + 前几行值 + EV19 metadata,无需 bash head。

**关键设计**: EV19 文件头部已经写明 Treatment / Dose / Animal ID 等元数据
(`raw_metadata` from parse_header),分组信息在头部不在数据列里。本 tool 直接
提取这些字段,lead 多数情况下根本不需要反问用户即可构造 groups dict。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState

logger = logging.getLogger(__name__)

# EV19 metadata 中表示"分组"的关键字段 (中英文都覆盖)
# 按出现频率排序: Treatment 是 Noldus 默认分组字段
_GROUPING_METADATA_KEYS = (
    "Treatment", "treatment",
    "Group", "group", "组别",
    "Drug", "drug",
    "Condition", "condition",
    "Dose", "dose", "剂量",
    "Compound", "compound",
)

# Number of data rows to include in data_preview
_DATA_PREVIEW_N_ROWS = 5


def _extract_required_patterns(paradigm: str) -> list[str]:
    """Extract deduplicated requires_columns patterns from a paradigm's catalog.

    Used by the Sprint 1 column-assessment to decide which custom columns are
    recognised (match a catalog glob) vs need HITL alignment.
    """
    from ethoinsight.catalog.loader import load_catalog

    patterns: set[str] = set()
    try:
        cat = load_catalog(paradigm)
    except Exception:
        return []
    for m in list(cat.default_metrics) + list(cat.optional_metrics):
        patterns.update(m.requires_columns)
    for ch in cat.charts:
        patterns.update(getattr(ch, "requires_columns", []) or [])
    return sorted(patterns)


def _compute_evidence(raw_column: str, values: list, column_index: int) -> dict[str, Any]:
    """Per-column objective evidence for an unrecognized column (no guessing).

    Binary (0/1) zone-like columns get an occupancy proportion; numeric columns
    get min/max/mean. This is shown to the user so THEY decide the column's
    meaning — the agent never asserts identity from the column name.
    """
    import numpy as np

    evidence: dict[str, Any] = {"column_index": column_index}
    finite = [v for v in values if v is not None]
    if not finite:
        evidence["type"] = "empty_or_all_nan"
        return evidence
    try:
        nums = np.array([float(v) for v in finite], dtype=float)
        nums = nums[np.isfinite(nums)]
    except (TypeError, ValueError):
        evidence["type"] = "non_numeric"
        return evidence
    if nums.size == 0:
        evidence["type"] = "empty_or_all_nan"
        return evidence
    uniq = set(np.unique(nums).tolist())
    if uniq <= {0.0, 1.0}:
        ones = int((nums == 1.0).sum())
        total = int(nums.size)
        evidence["type"] = "binary_zone"
        evidence["proportion_ones"] = round(ones / total, 4) if total > 0 else 0.0
        evidence["total_rows"] = total
    else:
        evidence["type"] = "continuous_or_distance"
        evidence["min"] = round(float(np.min(nums)), 4)
        evidence["max"] = round(float(np.max(nums)), 4)
        evidence["mean"] = round(float(np.mean(nums)), 4)
    return evidence


def _attach_column_assessment(result: dict[str, Any], paradigm: str | None) -> dict[str, Any]:
    """Layer Sprint 1 column_assessment + open_questions onto an inspect result.

    Uses result["columns"] (already present in dev's txt/csv/single-sheet xlsx
    paths) + result["data_preview"] (for per-column evidence). No-op when there
    are no top-level columns (e.g. multi-sheet xlsx — assessment is per-sheet
    territory left to Sprint 2).
    """
    from ethoinsight.utils import assess_column_confidence

    raw_columns = result.get("columns") or []
    if not raw_columns:
        return result

    required_patterns = _extract_required_patterns(paradigm) if paradigm else []
    assessment = assess_column_confidence(raw_columns, required_patterns)

    # Attach per-column evidence to unrecognized entries from data_preview.
    preview = result.get("data_preview") or {}
    preview_cols = preview.get("columns") or []
    preview_rows = preview.get("rows") or []
    for entry in assessment.get("unrecognized", []):
        raw = entry["raw"]
        if raw in preview_cols:
            idx = preview_cols.index(raw)
            col_values = [row[idx] for row in preview_rows if idx < len(row)]
            entry["evidence"] = _compute_evidence(raw, col_values, idx)
        else:
            entry["evidence"] = {"type": "column_not_in_preview"}

    result["column_assessment"] = assessment
    result["open_questions"] = [e["raw"] for e in assessment.get("unrecognized", [])]
    return result


def _compute_anonymous_zone_evidence(df: "pandas.DataFrame") -> dict[str, Any] | None:
    """Compute occupancy ratio evidence for bare in_zone column.

    When data has a bare ``in_zone`` column (unnamed EthoVision analysis zone),
    this computes the proportion of time the animal spends in each zone value,
    which helps the user identify which zone the anonymous column represents.

    Returns None if ``in_zone`` column is absent or the data is empty.
    """
    if "in_zone" not in df.columns:
        return None
    s = df["in_zone"].dropna()
    if s.empty:
        return None
    vc = s.value_counts(normalize=True)
    return {
        "column": "in_zone",
        "occupancy_ratio": {str(int(k)): round(float(v), 4) for k, v in vc.items()},
        "n_frames": int(len(s)),
        "note": "in_zone=1 表示在该匿名区内；动物在焦虑回避区（中心/开放臂/亮室）通常占时较低。",
    }


@tool("inspect_uploaded_file", parse_docstring=True)
def inspect_uploaded_file_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    uploaded_file: str,
    paradigm: str | None = None,
) -> dict[str, Any]:
    """探查单个上传文件的结构与 EV19 元数据,无需 bash。

    Lead 在准备 prep_metric_plan 之前调本 tool, 多数情况下可以直接从
    EV19 头部的 Treatment / Dose / Animal ID 字段推断分组, 无需反问用户。

    当传入 paradigm 时, 额外对每列做识别评估 (Sprint 1 列语义对齐): 返回
    column_assessment (recognized / unrecognized + 每个未识别列的取值证据) 和
    open_questions (未被系统识别的自定义分析区列名)。open_questions 非空时,
    lead 应走 ethoinsight-column-confirmation skill 与用户对齐列语义。

    Args:
      uploaded_file: 虚拟路径,如 "/mnt/user-data/uploads/foo.xlsx" 或 "/mnt/user-data/uploads/foo.txt"。多文件场景需要分别调用本 tool 探查每个文件。不要传 sheet 后缀(双冒号 sheet_name 形式)，本 tool 会自动列出所有 sheet。
      paradigm: 可选范式 key (如 "open_field")。提供时用 catalog requires_columns 做列识别评估更准; 不提供则只按 COLUMN_MAP / L1 固定列判定。

    Returns:
      status="ok" 时:
        {
          "status": "ok",
          "file": "/mnt/user-data/uploads/foo.txt",
          "format": "xlsx" | "txt" | "csv",
          "ev19_metadata": {       # 从文件头部提取的关键字段(EV19 txt + xlsx 都试)
            "experiment": "Porsolt forced swim test XT190",
            "trial_name": "Trial 1",
            "arena": "Arena 1",
            "subject": "Subject 1",
            "grouping_fields": {   # 检测到的分组相关字段
              "Treatment": "Drug",
              "Dose": "5 mg/L",
              "Animal ID": "1",
              ...
            }
          },
          "sheets": [              # 仅 xlsx 多 sheet 时填; 单 sheet/csv/txt 为空
            {
              "name": "...",
              "virtual_path": "/mnt/.../foo.xlsx::sheet_name",
              "n_rows": 1234,
              "columns": [...]
            }
          ],
          "columns": [...]         # 数据部分的列名(仅 csv/txt/单 sheet xlsx)
        }
      status="error" 时:
        {"status": "error", "error_code": "file_not_found"|"unsupported_format"|"parse_failed",
         "message": str}
    """
    # 1. 校验 thread_data + workspace
    thread_data = runtime.state.get("thread_data") if runtime.state else None
    if not thread_data or not thread_data.get("workspace_path"):
        return {
            "status": "error",
            "error_code": "workspace_missing",
            "message": "thread_data.workspace_path is not set",
        }

    # 2. 解析虚拟路径 → 真实路径
    from deerflow.sandbox.tools import replace_virtual_path

    # 拒绝带 sheet 后缀的输入 (lead 应该传整个文件)
    if "::" in uploaded_file:
        return {
            "status": "error",
            "error_code": "invalid_input",
            "message": (
                f"uploaded_file should NOT include sheet suffix (`::`). "
                f"Got: {uploaded_file!r}. Pass just the file path; this tool will list all sheets."
            ),
        }

    real_path = replace_virtual_path(uploaded_file, thread_data)
    if not Path(real_path).exists():
        return {
            "status": "error",
            "error_code": "file_not_found",
            "message": f"File not found: {uploaded_file} (resolved to {real_path})",
        }

    # 3. 根据扩展名分派
    ext = Path(real_path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        result = _inspect_excel(uploaded_file, real_path)
    elif ext == ".csv":
        result = _inspect_csv(uploaded_file, real_path)
    elif ext == ".txt":
        result = _inspect_txt(uploaded_file, real_path)
    else:
        return {
            "status": "error",
            "error_code": "unsupported_format",
            "message": (
                f"Unsupported file extension {ext!r}. "
                f"Supported: .xlsx / .xls / .csv / .txt"
            ),
        }

    # 4. Sprint 1 列语义对齐: layer column_assessment + open_questions onto the result.
    if isinstance(result, dict) and result.get("status") == "ok":
        result = _attach_column_assessment(result, paradigm)
    return result


def _extract_grouping_fields(raw_metadata: dict[str, str]) -> dict[str, str]:
    """从 EV19 raw_metadata 中提取分组相关字段。"""
    result: dict[str, str] = {}
    for key in _GROUPING_METADATA_KEYS:
        if key in raw_metadata and raw_metadata[key]:
            result[key] = raw_metadata[key]
    # 额外: Animal ID 不一定是分组字段但是 subject identifier, 一起返回方便 lead 理解
    for k in ("Animal ID", "Animal", "动物 ID", "动物编号"):
        if k in raw_metadata and raw_metadata[k] and k not in result:
            result[k] = raw_metadata[k]
            break
    return result


def _try_parse_header(file_path: str) -> dict[str, Any] | None:
    """复用 ethoinsight.parse.parse_header,失败返回 None。"""
    try:
        from ethoinsight.parse._core import parse_header
        return parse_header(file_path)
    except Exception as e:
        logger.warning("inspect_uploaded_file: parse_header failed for %s: %s", file_path, e)
        return None


def _build_data_preview_txt(real_path: str, header: dict[str, Any]) -> dict[str, Any] | None:
    """从 EV19 txt 文件提取前 N 行数据预览。

    Uses ethoinsight.parse.parse_header's header_lines to know where data starts,
    then reads the first _DATA_PREVIEW_N_ROWS data lines directly (no full file parse).
    """
    try:
        columns = header.get("columns", [])
        if not columns:
            return None
        header_lines = header.get("header_lines", 0)
        if header_lines <= 0:
            return None
        with open(real_path, "r", encoding="utf-16-le") as f:
            all_lines = f.readlines()
        # BOM strip
        all_lines[0] = all_lines[0].lstrip("﻿")
        # Data starts after header_lines (which includes column names + units line)
        # header_lines is 1-indexed count of header rows; data is at 0-indexed [header_lines:]
        data_start = header_lines
        from ethoinsight.parse._core import _split_semicolons

        rows: list[list[Any]] = []
        total_data_rows = 0
        for line in all_lines[data_start:]:
            line = line.strip()
            if not line:
                continue
            total_data_rows += 1
            if len(rows) < _DATA_PREVIEW_N_ROWS:
                values = _split_semicolons(line)
                values = [v.strip('"').strip() for v in values]
                values = values[: len(columns)]
                # Pad if too short
                values += [None] * (len(columns) - len(values))
                # Convert numeric strings
                converted: list[Any] = []
                for v in values:
                    if v is None or v == "-" or v == "":
                        converted.append(None)
                    else:
                        try:
                            converted.append(float(v))
                        except (ValueError, TypeError):
                            converted.append(v)
                rows.append(converted)
        if not rows:
            return None
        return {
            "columns": columns,
            "rows": rows,
            "n_rows_total": total_data_rows,
        }
    except Exception as e:
        logger.debug("data_preview txt failed for %s: %s", real_path, e)
        return None


def _build_data_preview_df(df: "pandas.DataFrame", n_total: int | None = None) -> dict[str, Any] | None:
    """从 pandas DataFrame 提取前 N 行数据预览。

    Works for xlsx/csv paths where pandas is already used.
    """
    try:
        import pandas as pd

        columns = [str(c) for c in df.columns]
        preview_df = df.head(_DATA_PREVIEW_N_ROWS)
        rows: list[list[Any]] = []
        for _, row in preview_df.iterrows():
            converted: list[Any] = []
            for val in row:
                if pd.isna(val):
                    converted.append(None)
                elif isinstance(val, float) and val == int(val):
                    converted.append(int(val))
                else:
                    converted.append(val)
            rows.append(converted)
        if not rows:
            return None
        n_total = n_total if n_total is not None else len(df)
        return {
            "columns": columns,
            "rows": rows,
            "n_rows_total": n_total,
        }
    except Exception as e:
        logger.debug("data_preview df failed: %s", e)
        return None


def _inspect_txt(virtual_path: str, real_path: str) -> dict[str, Any]:
    """探查 EV19 txt 文件 (UTF-16 编码, header + tabular data)。"""
    header = _try_parse_header(real_path)
    if not header:
        # txt 解析失败大概率是非 EV19 格式
        return {
            "status": "error",
            "error_code": "parse_failed",
            "message": f"Failed to parse {virtual_path} as EthoVision XT txt (UTF-16 header expected).",
        }

    raw_metadata = header.get("raw_metadata", {}) or {}
    result: dict[str, Any] = {
        "status": "ok",
        "file": virtual_path,
        "format": "txt",
        "ev19_metadata": {
            "experiment": header.get("experiment", ""),
            "trial_name": header.get("trial_name", ""),
            "arena": header.get("arena", ""),
            "subject": header.get("subject", ""),
            "start_time": header.get("start_time", ""),
            "duration": header.get("duration", ""),
            "grouping_fields": _extract_grouping_fields(raw_metadata),
        },
        "sheets": [],
        "columns": header.get("columns", []),
    }
    # Add data preview
    data_preview = _build_data_preview_txt(real_path, header)
    if data_preview is not None:
        result["data_preview"] = data_preview
    # Add anonymous zone evidence when bare in_zone exists (requires full parse)
    columns = result["columns"]
    if "in_zone" in columns:
        try:
            from ethoinsight.parse._core import parse_trajectory

            df = parse_trajectory(real_path)
            evidence = _compute_anonymous_zone_evidence(df)
            if evidence is not None:
                result["anonymous_zone_evidence"] = evidence
        except Exception as e:
            logger.debug("anonymous_zone_evidence txt failed for %s: %s", real_path, e)
    return result


def _inspect_excel(virtual_path: str, real_path: str) -> dict[str, Any]:
    """探查 xlsx/xls 文件 sheets + 每个 sheet 的 EV19 metadata + data preview。"""
    try:
        import pandas as pd
    except ImportError:
        return {
            "status": "error",
            "error_code": "parse_failed",
            "message": "pandas not installed",
        }

    try:
        xl = pd.ExcelFile(real_path)
    except Exception as e:
        return {
            "status": "error",
            "error_code": "parse_failed",
            "message": f"Failed to open Excel file: {e}",
        }

    sheet_names = xl.sheet_names
    if len(sheet_names) > 1:
        sheets: list[dict[str, Any]] = []
        for sn in sheet_names:
            # 对每个 sheet 尝试 EV19 header 解析(parse_header 接受 ::sheet 后缀)
            sheet_full = f"{real_path}::{sn}"
            sheet_header = _try_parse_header(sheet_full)
            sheet_info: dict[str, Any] = {
                "name": sn,
                "virtual_path": f"{virtual_path}::{sn}",
            }
            if sheet_header:
                raw_metadata = sheet_header.get("raw_metadata", {}) or {}
                sheet_info["ev19_metadata"] = {
                    "arena": sheet_header.get("arena", ""),
                    "subject": sheet_header.get("subject", ""),
                    "grouping_fields": _extract_grouping_fields(raw_metadata),
                }
                sheet_info["columns"] = sheet_header.get("columns", [])
            else:
                # 退化到原始 pandas 读
                try:
                    df = pd.read_excel(real_path, sheet_name=sn, nrows=50)
                    sheet_info["columns"] = [str(c) for c in df.columns]
                    sheet_info["n_rows_preview"] = "50+"
                except Exception as e:
                    sheet_info["error"] = f"sheet read failed: {e}"
            sheets.append(sheet_info)
        return {
            "status": "ok",
            "file": virtual_path,
            "format": "xlsx",
            "ev19_metadata": None,  # 多 sheet 时元数据在每个 sheet 里
            "sheets": sheets,
            "columns": [],
        }
    else:
        # 单 sheet xlsx — 试 EV19 header
        sheet_full = f"{real_path}::{sheet_names[0]}" if sheet_names else real_path
        header = _try_parse_header(sheet_full)
        if header:
            raw_metadata = header.get("raw_metadata", {}) or {}
            result: dict[str, Any] = {
                "status": "ok",
                "file": virtual_path,
                "format": "xlsx",
                "ev19_metadata": {
                    "experiment": header.get("experiment", ""),
                    "trial_name": header.get("trial_name", ""),
                    "arena": header.get("arena", ""),
                    "subject": header.get("subject", ""),
                    "grouping_fields": _extract_grouping_fields(raw_metadata),
                },
                "sheets": [],
                "columns": header.get("columns", []),
            }
            # Add data preview for EV19 xlsx via header info
            try:
                from ethoinsight.parse._core import parse_trajectory

                df = parse_trajectory(sheet_full)
                data_preview = _build_data_preview_df(df)
                if data_preview is not None:
                    result["data_preview"] = data_preview
                # Add anonymous zone evidence when bare in_zone exists
                columns = result["columns"]
                if "in_zone" in columns:
                    evidence = _compute_anonymous_zone_evidence(df)
                    if evidence is not None:
                        result["anonymous_zone_evidence"] = evidence
            except Exception as e:
                logger.debug("data_preview xlsx ev19 failed for %s: %s", sheet_full, e)
            return result
        else:
            # 不是 EV19 格式, 只返回列名 + data preview
            try:
                df = pd.read_excel(real_path, sheet_name=sheet_names[0], nrows=_DATA_PREVIEW_N_ROWS)
            except Exception as e:
                return {"status": "error", "error_code": "parse_failed", "message": f"Excel read failed: {e}"}
            result = {
                "status": "ok",
                "file": virtual_path,
                "format": "xlsx",
                "ev19_metadata": None,
                "sheets": [],
                "columns": [str(c) for c in df.columns],
            }
            data_preview = _build_data_preview_df(df)
            if data_preview is not None:
                result["data_preview"] = data_preview
            return result


def _inspect_csv(virtual_path: str, real_path: str) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "error", "error_code": "parse_failed", "message": "pandas not installed"}
    # CSV 试 EV19 header (parse_header 支持 csv)
    header = _try_parse_header(real_path)
    if header:
        raw_metadata = header.get("raw_metadata", {}) or {}
        result: dict[str, Any] = {
            "status": "ok",
            "file": virtual_path,
            "format": "csv",
            "ev19_metadata": {
                "experiment": header.get("experiment", ""),
                "trial_name": header.get("trial_name", ""),
                "arena": header.get("arena", ""),
                "subject": header.get("subject", ""),
                "grouping_fields": _extract_grouping_fields(raw_metadata),
            },
            "sheets": [],
            "columns": header.get("columns", []),
        }
        # Try data preview via parse_trajectory for EV19 CSV
        try:
            from ethoinsight.parse._core import parse_trajectory

            df = parse_trajectory(real_path)
            data_preview = _build_data_preview_df(df)
            if data_preview is not None:
                result["data_preview"] = data_preview
            # Add anonymous zone evidence when bare in_zone exists
            if "in_zone" in result["columns"]:
                evidence = _compute_anonymous_zone_evidence(df)
                if evidence is not None:
                    result["anonymous_zone_evidence"] = evidence
        except Exception as e:
            logger.debug("data_preview csv ev19 failed for %s: %s", real_path, e)
        return result
    # 退化到原始 CSV
    try:
        df = pd.read_csv(real_path, nrows=_DATA_PREVIEW_N_ROWS)
    except Exception as e:
        return {"status": "error", "error_code": "parse_failed", "message": f"CSV read failed: {e}"}
    result = {
        "status": "ok",
        "file": virtual_path,
        "format": "csv",
        "ev19_metadata": None,
        "sheets": [],
        "columns": [str(c) for c in df.columns],
    }
    data_preview = _build_data_preview_df(df)
    if data_preview is not None:
        result["data_preview"] = data_preview
    return result
