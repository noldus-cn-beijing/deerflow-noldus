"""prep_chart_plan — 一步生成 plan_charts.json，无需 bash。

Spec 3 (P3, 2026-06-17): charts 路径列对齐靠 LLM 拼 CLI —— 改为从 experiment-context 自读。

lead / chart-maker 调用本工具，一步产出 plan_charts.json。工具内部确定性自读
session 级横切状态（column_aliases / groups / paradigm），与 metrics 路径的
``prep_metric_plan`` 共用同一个 reader（红线二正模式 1：横切状态有唯一 reader
收口，工具内部自取，绝不让 LLM 拼 CLI 参数传 session 常量）。

修法对应 spec 方案 A（结构对称）：charts 路径获取 column_aliases 像 metrics
路径一样确定性，不依赖 LLM 在 bash 里记得拼 ``--column-aliases-file``。

column_aliases 永远来自 experiment-context.json（Gate 1 column_semantics 投影），
LLM 无从遗漏；groups 永远来自 groups.json（prep_metric_plan 落盘），LLM 无从遗漏。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState

logger = logging.getLogger(__name__)

# 错误码→hint 模板（给调用方看的下一步建议）
_ERROR_HINTS: dict[str, str] = {
    "file_not_found": (
        "数据文件不存在，可能用户上传失败或 raw_files 路径错。"
        "确认从 plan_metrics.json.inputs.raw_files 原样取路径，不要用 realpath。"
    ),
    "format_unrecognized": (
        "文件不是 EthoVision XT 导出格式。让 lead 用 ask_clarification 让用户确认导出方式。"
    ),
    "parse_failed": (
        "数据文件解析失败（损坏 / header 读不出列）。让 lead 确认是否需要重新导出。"
    ),
    "unknown_paradigm": (
        "范式不在 catalog 内。从 handoff_code_executor.json 复制 paradigm 字段，"
        "不要自己猜范式名。"
    ),
    "workspace_missing": (
        "thread_data.workspace_path 未设置——基础设施 bug（ThreadDataMiddleware 应先建好 workspace）。"
    ),
    "no_files_provided": (
        "uploaded_files 为空。把 plan_metrics.json.inputs.raw_files 列表原样传进来再调一次。"
    ),
}


@tool("prep_chart_plan", parse_docstring=True)
def prep_chart_plan_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    uploaded_files: list[str],
    paradigm: str,
    user_intent: str | None = None,
    total_subjects: int | None = None,
    n_per_group: int | None = None,
    n_groups: int | None = None,
) -> dict:
    """一步生成 plan_charts.json，无需 bash 拼 catalog.resolve CLI。

    工具内部自读 experiment-context.json 拿 column_aliases（Gate 1 列语义对齐投影）、
    读 groups.json 拿分组（若 prep_metric_plan 落盘了），调 resolve_charts 产出
    plan_charts.json。**column_aliases / groups 永远来自 context，调用方无从遗漏**——
    这是取代「chart-maker 在 bash 里手拼 --column-aliases-file / --groups-json」的
    确定性入口（红线二正模式 1）。

    Args:
      uploaded_files: 虚拟路径列表（**原样取自 plan_metrics.json.inputs.raw_files**），
                      如 ["/mnt/user-data/uploads/arena1.txt"]。多文件 = 多 subject。
                      不要用 Path.resolve()/realpath，不要从 handoff_code_executor.json
                      抄（其 inputs 历史上有宿主路径污染）。
      paradigm: 范式 canonical key，**原样取自 handoff_code_executor.json.paradigm**。
      user_intent: 用户原图原话（"箱线图" / "轨迹图" / "未明确指定"）。用于 resolve
                   的 user_intent 过滤——不传则不过滤。原样取自 chart-maker 派遣 prompt。
      total_subjects: subject 总数（可选）。原样取自 handoff_code_executor.json。
      n_per_group: 每组 subject 数（可选）。原样取自 handoff_code_executor.json。
      n_groups: 组数（可选）。原样取自 handoff_code_executor.json。

    Returns:
      status="ok" 时:
        {"status": "ok",
         "plan_path": "/mnt/user-data/workspace/plan_charts.json",
         "plan_summary": {"paradigm": "epm", "chart_count": 4,
                          "fallback_count": 0, "skipped_count": 0,
                          "chart_ids": ["box_open_arm", ...],
                          "column_aliases_applied": true|false,
                          "groups_applied": true|false}}
      status="error" 时:
        {"status": "error",
         "error_code": "file_not_found"|"format_unrecognized"|"parse_failed"|
                       "unknown_paradigm"|"workspace_missing"|"no_files_provided",
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

    # Lazy imports to avoid circular dependency + keep deerflow harness importable
    # without ethoinsight installed (same discipline as prep_metric_plan_tool).
    from deerflow.sandbox.tools import replace_virtual_path

    from ethoinsight.catalog.resolve import (
        ResolveError,
        plan_charts_to_dict,
        resolve_charts,
    )
    from ethoinsight.parse._core import detect_ethovision, parse_header

    # Step 2: per-file validation (existence + EthoVision detect) + header parse on first file.
    # header 用第 1 个文件（同一批次 EV 导出列结构一致；若不一致 resolve_charts 会 columns_missing skip）。
    columns: list[str] = []
    for idx, uploaded_file in enumerate(uploaded_files):
        clean_virtual = uploaded_file.split("::")[0] if "::" in uploaded_file else uploaded_file
        real_fs_path = replace_virtual_path(clean_virtual, thread_data)
        effective_path: str = real_fs_path
        if "::" in uploaded_file:
            effective_path = real_fs_path + "::" + uploaded_file.split("::", 1)[1]

        if not Path(real_fs_path).exists():
            return _error_result(
                "file_not_found",
                f"File not found: {clean_virtual} (resolved to {real_fs_path})",
                failed_file=clean_virtual,
            )

        if not detect_ethovision(effective_path):
            return _error_result(
                "format_unrecognized",
                f"File {uploaded_file} is not an EthoVision XT export.",
                failed_file=uploaded_file,
            )

        if idx == 0:
            try:
                header = parse_header(effective_path)
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

    # Step 3: 自读 session 级横切状态（红线二正模式 1 —— 工具内部自取，不靠 LLM 拼 CLI）。
    # column_aliases: experiment-context.json 的 column_semantics 投影（Gate 1 落盘）。
    #                 不读它 → resolve_charts 拿不到别名 → 自定义分析区列仍 columns_missing skip
    #                 （dogfood 实证：FewZones open/closed 列被 skip 掉 box/bar/rose）。
    # groups: groups.json（prep_metric_plan 落盘）。catalog needs_groups:true 的 aggregate
    #         chart（box/bar 等）依赖它做组间对比。
    from deerflow.agents.middlewares.experiment_context import read_context

    ctx = read_context(real_workspace_path)
    column_aliases = ctx.get("column_aliases") if ctx else None

    groups_file_virtual: str | None = None
    groups_file_real: str | None = None
    groups_dict: dict | None = None
    groups_path = Path(real_workspace_path) / "groups.json"
    if groups_path.exists():
        try:
            groups_data = json.loads(groups_path.read_text(encoding="utf-8"))
            if isinstance(groups_data, dict):
                groups_dict = groups_data
                groups_file_real = str(groups_path)
                groups_file_virtual = "/mnt/user-data/workspace/groups.json"
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("prep_chart_plan: failed to read groups.json: %s", e)

    virtual_workspace_dir = "/mnt/user-data/workspace"

    # Step 4: resolve catalog charts → PlanCharts
    try:
        pc = resolve_charts(
            paradigm=paradigm,
            columns=columns,
            raw_files=list(uploaded_files),
            workspace_dir=real_workspace_path,
            user_intent=user_intent,
            total_subjects=total_subjects,
            n_per_group=n_per_group,
            n_groups=n_groups,
            groups_file=groups_file_virtual,
            groups=groups_dict,
            columns_file=None,
            virtual_workspace_dir=virtual_workspace_dir,
            column_aliases=column_aliases,
        )
    except ResolveError as e:
        return _error_result(
            e.code,
            str(e),
            extra_details=e.details,
        )
    except Exception as e:
        logger.exception("Unexpected error during resolve_charts for paradigm=%s", paradigm)
        return _error_result(
            "parse_failed",
            f"Unexpected error during catalog resolve: {e}",
        )

    # Step 5: serialize plan to workspace/plan_charts.json
    plan_dict = plan_charts_to_dict(pc)
    plan_path = Path(real_workspace_path) / "plan_charts.json"
    try:
        plan_path.write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        return _error_result(
            "parse_failed",
            f"Failed to write plan_charts.json: {e}",
        )

    chart_ids = [c.get("id", "") for c in plan_dict.get("charts", [])]
    logger.info(
        "prep_chart_plan success: paradigm=%s, charts=%d, fallback=%d, skipped=%d, "
        "column_aliases_applied=%s, groups_applied=%s, plan=%s",
        paradigm,
        len(chart_ids),
        len(plan_dict.get("charts_fallback_available", [])),
        len(plan_dict.get("skipped", [])),
        bool(column_aliases),
        bool(groups_dict),
        plan_path,
    )

    return {
        "status": "ok",
        "plan_path": "/mnt/user-data/workspace/plan_charts.json",
        "plan_summary": {
            "paradigm": paradigm,
            "chart_count": len(chart_ids),
            "fallback_count": len(plan_dict.get("charts_fallback_available", [])),
            "skipped_count": len(plan_dict.get("skipped", [])),
            "chart_ids": chart_ids,
            "column_aliases_applied": bool(column_aliases),
            "groups_applied": bool(groups_dict),
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
