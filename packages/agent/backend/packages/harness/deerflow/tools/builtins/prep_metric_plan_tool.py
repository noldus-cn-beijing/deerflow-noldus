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
    "no_files_provided": (
        "uploaded_files 为空。把当前 <uploaded_files> 中所有相关数据文件路径传进来再调一次。"
    ),
}


@tool("prep_metric_plan", parse_docstring=True)
def prep_metric_plan_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    uploaded_files: list[str],
    paradigm: str,
) -> dict:
    """一步生成 plan_metrics.json，无需 bash。

    Args:
      uploaded_files: 虚拟路径列表如 ["/mnt/user-data/uploads/arena1.txt",
                      "/mnt/user-data/uploads/arena2.txt"]。多文件场景每个文件
                      代表 1 个 subject;catalog 会为每个指标 × 每个文件生成一个
                      PlanMetric(N 文件 × M 指标 = N×M 个调用)。单文件场景传单元素 list。
                      **请把当前 <uploaded_files> 里所有相关数据文件全传进来**,
                      不要只传第 1 个,否则其余 subject 在分析中会被静默丢失。
      paradigm: 范式 canonical key（学术名）, v0.1 仅支持以下 5 个:
                'epm' / 'open_field' / 'forced_swim' / 'light_dark_box' / 'zero_maze'
                （filename-style 缩写如 'oft'/'fst'/'ldb' 也接受，向后兼容）
                其他 paradigm_key (如 'shoaling'/'tail_suspension'/'morris_water_maze' 等)
                会在 catalog.resolve 阶段报错; lead 应在 identify_ev19_template 看到
                status=unsupported 时就反问用户, 不要走到这一步

    Returns:
      status="ok" 时:
        {"status": "ok",
         "plan_path": "/mnt/user-data/workspace/plan_metrics.json",
         "plan_summary": {"paradigm": "epm", "metric_count": 5, "subject_count": 2,
                          "metric_ids": ["open_arm_time_ratio", ...]}}
      status="error" 时:
        {"status": "error",
         "error_code": "file_not_found"|"format_unrecognized"|"parse_failed"|
                       "unknown_paradigm"|"columns_missing"|"schema_violation"|
                       "empty_plan"|"unknown_metric"|"workspace_missing"|
                       "no_files_provided",
         "message": str,
         "hint": str,
         "failed_file": str | None}
    """
    # Step 0: validate inputs — must have at least one file
    if not uploaded_files:
        return _error_result(
            "no_files_provided",
            "uploaded_files is empty; provide at least one file path.",
        )

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

    # Step 2-4: per-file validation (existence + EthoVision detect + header parse)
    # 任何文件失败立即返回,带 failed_file 字段给 lead 看是哪个文件出问题。
    # header 用第 1 个文件的(同一批次 EV 导出列结构一致;若不一致 catalog resolve 会报 columns_missing)。
    columns: list[str] = []
    for idx, uploaded_file in enumerate(uploaded_files):
        real_file_path = replace_virtual_path(uploaded_file, thread_data)

        if not Path(real_file_path).exists():
            return _error_result(
                "file_not_found",
                f"File not found: {uploaded_file} (resolved to {real_file_path})",
                failed_file=uploaded_file,
            )

        if not detect_ethovision(real_file_path):
            return _error_result(
                "format_unrecognized",
                f"File {uploaded_file} is not an EthoVision XT export.",
                failed_file=uploaded_file,
            )

        if idx == 0:
            try:
                header = parse_header(real_file_path)
            except Exception as e:
                logger.warning("parse_header failed for %s: %s", uploaded_file, e)
                return _error_result(
                    "parse_failed",
                    f"Failed to parse header: {e}",
                    failed_file=uploaded_file,
                )
            columns = header.get("columns", [])
            if not columns:
                return _error_result(
                    "parse_failed",
                    "Parsed header contains no column names.",
                    failed_file=uploaded_file,
                )

    # Step 5: resolve catalog → PlanMetrics
    # raw_files 走虚拟路径,避免宿主机路径泄漏到 plan_metrics.json 后被 subagent
    # 照抄进 bash --input。IO 部分(detect_ethovision / parse_header)已在 Step 2-4
    # 完成,resolve_metrics 内部按 raw_files 展开 N 个 PlanMetric(每个 subject 一个)。
    try:
        plan = resolve_metrics(
            paradigm=paradigm,
            columns=columns,
            raw_files=list(uploaded_files),
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

    # Step 7: build summary (paradigm/metric_count/subject_count/metric_ids,不含完整 plan)
    metric_dicts = plan_dict.get("metrics", [])
    metric_ids = sorted({m.get("id", "") for m in metric_dicts})
    subject_count = len(uploaded_files)

    logger.info(
        "prep_metric_plan success: paradigm=%s, subjects=%d, metric_count=%d, plan_metric_count=%d, plan=%s",
        paradigm,
        subject_count,
        len(metric_ids),
        len(metric_dicts),
        plan_path,
    )

    return {
        "status": "ok",
        "plan_path": "/mnt/user-data/workspace/plan_metrics.json",
        "plan_summary": {
            "paradigm": paradigm,
            "metric_count": len(metric_ids),
            "subject_count": subject_count,
            "metric_ids": metric_ids,
        },
    }


def _error_result(
    code: str,
    message: str,
    extra_details: dict | None = None,
    failed_file: str | None = None,
) -> dict:
    """Build a standardised error response dict."""
    hint = _ERROR_HINTS.get(code, "未知错误，请联系开发者。")
    result: dict = {
        "status": "error",
        "error_code": code,
        "message": message,
        "hint": hint,
    }
    if failed_file is not None:
        result["failed_file"] = failed_file
    if extra_details:
        result["details"] = extra_details
    return result
