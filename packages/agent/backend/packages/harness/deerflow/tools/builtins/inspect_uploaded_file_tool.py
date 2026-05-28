"""inspect_uploaded_file — lead 探查上传文件结构(sheets/columns/EV19 元数据),无需 bash。

lead 在调 prep_metric_plan 之前,需要知道:
- xlsx 文件有多少 sheet (FST/EPM 多 arena 场景)
- 每个 sheet 含哪些列
- **EV19 metadata header 中的分组信息**(Treatment / Dose / Animal ID / Group)

Sprint 0/B fix 2026-05-28: 增加本 tool 解决 dogfood thread 485a899d 暴露的根因——
lead 没有 bash / general-purpose subagent,无法探查 xlsx 列出分组信息;之前死循环
反问"drug 列具体值是什么"但用户不知道。

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


@tool("inspect_uploaded_file", parse_docstring=True)
def inspect_uploaded_file_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    uploaded_file: str,
) -> dict[str, Any]:
    """探查单个上传文件的结构与 EV19 元数据,无需 bash。

    Lead 在准备 prep_metric_plan 之前调本 tool, 多数情况下可以直接从
    EV19 头部的 Treatment / Dose / Animal ID 字段推断分组, 无需反问用户。

    Args:
      uploaded_file: 虚拟路径,如 "/mnt/user-data/uploads/foo.xlsx" 或 "/mnt/user-data/uploads/foo.txt"。多文件场景需要分别调用本 tool 探查每个文件。不要传 sheet 后缀(双冒号 sheet_name 形式)，本 tool 会自动列出所有 sheet。

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
        return _inspect_excel(uploaded_file, real_path)
    elif ext == ".csv":
        return _inspect_csv(uploaded_file, real_path)
    elif ext == ".txt":
        return _inspect_txt(uploaded_file, real_path)
    else:
        return {
            "status": "error",
            "error_code": "unsupported_format",
            "message": (
                f"Unsupported file extension {ext!r}. "
                f"Supported: .xlsx / .xls / .csv / .txt"
            ),
        }


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
    return {
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


def _inspect_excel(virtual_path: str, real_path: str) -> dict[str, Any]:
    """探查 xlsx/xls 文件 sheets + 每个 sheet 的 EV19 metadata。"""
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
            return {
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
        else:
            # 不是 EV19 格式, 只返回列名
            try:
                df = pd.read_excel(real_path, sheet_name=sheet_names[0], nrows=50)
            except Exception as e:
                return {"status": "error", "error_code": "parse_failed", "message": f"Excel read failed: {e}"}
            return {
                "status": "ok",
                "file": virtual_path,
                "format": "xlsx",
                "ev19_metadata": None,
                "sheets": [],
                "columns": [str(c) for c in df.columns],
            }


def _inspect_csv(virtual_path: str, real_path: str) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "error", "error_code": "parse_failed", "message": "pandas not installed"}
    # CSV 试 EV19 header (parse_header 支持 csv)
    header = _try_parse_header(real_path)
    if header:
        raw_metadata = header.get("raw_metadata", {}) or {}
        return {
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
    # 退化到原始 CSV
    try:
        df = pd.read_csv(real_path, nrows=50)
    except Exception as e:
        return {"status": "error", "error_code": "parse_failed", "message": f"CSV read failed: {e}"}
    return {
        "status": "ok",
        "file": virtual_path,
        "format": "csv",
        "ev19_metadata": None,
        "sheets": [],
        "columns": [str(c) for c in df.columns],
    }
