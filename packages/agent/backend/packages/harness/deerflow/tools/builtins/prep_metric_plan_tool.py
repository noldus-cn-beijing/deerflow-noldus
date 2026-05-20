"""prep_metric_plan — 一步生成 plan_metrics.json，无需 bash。

lead agent 专用：直接调 ethoinsight Python 函数解析 EthoVision 文件 +
catalog resolve，结果写入 workspace/plan_metrics.json。

这是 P0 fix 的核心：lead 不再有 bash tool，所有 ethoinsight CLI 调用
强制走此 Python 工具。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState
from ethoinsight.catalog.resolve import ResolveError, plan_metrics_to_dict, resolve_metrics
from ethoinsight.parse._core import detect_ethovision, parse_header

logger = logging.getLogger(__name__)

# 错误码→hint 模板（给 lead 看的下一步建议）
_ERROR_HINTS: dict[str, str] = {
    "file_not_found": (
        "数据文件不存在，可能用户上传失败。用 ask_clarification 让用户重新上传。"
    ),
    "format_unrecognized": (
        "文件不是 EthoVision XT 导出格式。用 ask_clarification 让用户确认导出方式。"
    ),
    "parse_failed": (
        "数据文件损坏，无法解析。用 ask_clarification 让用户重新导出。"
    ),
    "unknown_paradigm": (
        "范式不在 catalog 内。用 ask_clarification 让用户确认范式，"
        "或检查 set_experiment_paradigm 调用是否正确。"
    ),
    "columns_missing": (
        "数据缺关键列（可能录制设置漏了 Open/Closed arms 进入次数或相关区域）。"
        "用 ask_clarification 让用户确认实验录制设置。"
    ),
    "schema_violation": (
        "catalog YAML 损坏——这是项目内部 bug。present_files 把错误信息呈现给用户，让他报 bug。"
    ),
    "empty_plan": (
        "按当前参数一项指标都跑不了。用 ask_clarification 确认用户需求。"
    ),
    "unknown_metric": (
        "用户要求的指标不在 catalog 中。用 ask_clarification 让用户从可用指标中选择。"
    ),
    "workspace_missing": (
        "thread_data.workspace_path 未设置——这是基础设施 bug（ThreadDataMiddleware 应该先建好 workspace）。"
        "present_files 把错误信息呈现给用户，让他报 bug。"
    ),
}


@tool("prep_metric_plan", parse_docstring=True)
def prep_metric_plan_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    uploaded_file: str,
    paradigm: str,
) -> dict:
    """一步生成 plan_metrics.json，无需 bash。

    Args:
      uploaded_file: 虚拟路径如 /mnt/user-data/uploads/xxx.txt
      paradigm: 范式如 'epm' / 'oft' / 'fst' / 'ldb' / 'tst' / 'zero_maze'
                / 'shoaling'

    Returns:
      status="ok" 时:
        {"status": "ok",
         "plan_path": "/mnt/user-data/workspace/plan_metrics.json",
         "plan_summary": {"paradigm": "epm", "metric_count": 5,
                          "metric_ids": ["open_arm_time_ratio", ...]}}
      status="error" 时:
        {"status": "error",
         "error_code": "file_not_found"|"format_unrecognized"|"parse_failed"|
                       "unknown_paradigm"|"columns_missing"|"schema_violation"|
                       "empty_plan"|"unknown_metric"|"workspace_missing",
         "message": str,
         "hint": str}
    """
    # Step 1: resolve thread_data — workspace_path is mandatory, fail fast if missing
    thread_data = runtime.state.get("thread_data") if runtime.state else None
    if not thread_data or not thread_data.get("workspace_path"):
        return _error_result(
            "workspace_missing",
            "thread_data.workspace_path is not set",
        )
    real_workspace_path = thread_data["workspace_path"]
    # Lazy import to avoid circular dependency (sandbox.tools → agents.factory → tools.builtins → here)
    from deerflow.sandbox.tools import replace_virtual_path

    real_file_path = replace_virtual_path(uploaded_file, thread_data)

    # Step 2: check file exists
    if not Path(real_file_path).exists():
        return _error_result(
            "file_not_found",
            f"File not found: {uploaded_file} (resolved to {real_file_path})",
        )

    # Step 3: detect EthoVision format
    if not detect_ethovision(real_file_path):
        return _error_result(
            "format_unrecognized",
            f"File {uploaded_file} is not an EthoVision XT export.",
        )

    # Step 4: parse header to get column names
    try:
        header = parse_header(real_file_path)
    except Exception as e:
        logger.warning("parse_header failed for %s: %s", uploaded_file, e)
        return _error_result(
            "parse_failed",
            f"Failed to parse header: {e}",
        )

    columns = header.get("columns", [])
    if not columns:
        return _error_result(
            "parse_failed",
            "Parsed header contains no column names.",
        )

    # Step 5: resolve catalog → PlanMetrics
    # raw_files 走虚拟路径,避免宿主机路径泄漏到 plan_metrics.json 后被 subagent
    # 照抄进 bash --input。IO 部分(detect_ethovision / parse_header)已在 Step 3-4
    # 完成,resolve_metrics 内部只把 raw_files 透传到 PlanMetric.input + PlanInputs.raw_files。
    try:
        plan = resolve_metrics(
            paradigm=paradigm,
            columns=columns,
            raw_files=[uploaded_file],
            workspace_dir=real_workspace_path,
            virtual_workspace_dir="/mnt/user-data/workspace",
        )
    except ResolveError as e:
        return _error_result(
            e.code,
            str(e),
            extra_details=e.details,
        )
    except Exception as e:
        logger.exception("Unexpected error during resolve for paradigm=%s", paradigm)
        return _error_result(
            "parse_failed",
            f"Unexpected error during catalog resolve: {e}",
        )

    # Step 6: serialize plan to workspace/plan_metrics.json
    plan_dict = plan_metrics_to_dict(plan)
    plan_path = Path(real_workspace_path) / "plan_metrics.json"
    try:
        plan_path.write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        return _error_result(
            "parse_failed",
            f"Failed to write plan_metrics.json: {e}",
        )

    # Step 7: build summary (只 paradigm/metric_count/metric_ids，不含完整 plan)
    metric_ids = [m.get("id", "") for m in plan_dict.get("metrics", [])]

    logger.info(
        "prep_metric_plan success: paradigm=%s, metric_count=%d, plan=%s",
        paradigm,
        len(metric_ids),
        plan_path,
    )

    return {
        "status": "ok",
        "plan_path": "/mnt/user-data/workspace/plan_metrics.json",
        "plan_summary": {
            "paradigm": paradigm,
            "metric_count": len(metric_ids),
            "metric_ids": metric_ids,
        },
    }


def _error_result(code: str, message: str, extra_details: dict | None = None) -> dict:
    """Build a standardised error response dict."""
    hint = _ERROR_HINTS.get(code, "未知错误，请联系开发者。")
    result: dict = {
        "status": "error",
        "error_code": code,
        "message": message,
        "hint": hint,
    }
    if extra_details:
        result["details"] = extra_details
    return result
