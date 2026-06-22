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
        "缺失的列是 in_zone_* 分析区列时，通常不是数据缺列，而是用户自定义了分析区列名"
        "（如把中心区命名为「中心区」/「Center」）——先调 inspect_uploaded_file(paradigm=...) 看是否有"
        "未识别的自定义分析区列，按 ethoinsight-column-confirmation skill 与用户对齐列语义，"
        "再调 set_experiment_paradigm(column_semantics={...}) 落盘后重试 prep_metric_plan。"
        "仅当确认数据真的没有该分析区列（录制设置漏了）时，才用 ask_clarification 让用户确认实验录制设置。"
    ),
    "zone_unnamed": (
        "数据里有一个未命名分析区(in_zone)，需要确认它代表哪个目标区域后再分析。"
        "第一步：调 inspect_uploaded_file 查看该文件的 anonymous_zone_evidence，"
        "它给出 in_zone=1 与 in_zone=0 的占时比例。"
        "第二步：用 ask_clarification 把占时证据呈现给用户并请其确认。"
        "行为学常识可辅助判断：动物在焦虑回避区（旷场中心区 / 零迷宫开放臂 / 明暗箱亮室）通常停留时间较短，"
        "占时低的一侧更可能是目标区，最终以用户确认为准。"
        "第三步：用户确认后写 parameter_overrides={\"anonymous_zone_is\": \"in_zone\"} "
        "再重调 prep_metric_plan。"
        "若用户判断该区不是目标区，请说明数据需在 EthoVision 重新命名区域后导出，"
        "以保证分析建立在明确区域定义之上。"
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
    groups: dict[str, str] | None = None,
) -> dict:
    """一步生成 plan_metrics.json，无需 bash。

    Args:
      uploaded_files: 虚拟路径列表如 ["/mnt/user-data/uploads/arena1.txt",
                      "/mnt/user-data/uploads/arena2.txt"]。多文件场景每个文件
                      代表 1 个 subject;catalog 会为每个指标 × 每个文件生成一个
                      PlanMetric(N 文件 × M 指标 = N×M 个调用)。单文件场景传单元素 list。
                      **请把当前 <uploaded_files> 里所有相关数据文件全传进来**,
                      不要只传第 1 个,否则其余 subject 在分析中会被静默丢失。
      paradigm: 范式 canonical key（学术名）。v0.1 支持的范式以
                `ethoinsight.ev19_facts.SUPPORTED_PARADIGMS_V01` 为准（权威清单），
                调用方应通过 `identify_ev19_template` 工具返回的 `supported_paradigms`
                字段获取当前清单，而非在此处手抄。常见 key 形如 'epm' / 'open_field'
                等（filename-style 缩写如 'oft'/'fst'/'ldb' 也接受，向后兼容）。
                不在 SUPPORTED_PARADIGMS_V01 里的 paradigm_key 会在 catalog.resolve
                阶段报错; lead 应在 identify_ev19_template 看到 status=unsupported 时
                就反问用户, 不要走到这一步
      groups: 可选的 subject -> group_name 映射。当用户已经在 ask_clarification 中说清分组(如"第一个是实验组，第二个是对照组")，lead 必须把它翻译成 dict 传进来。dict 的 key 必须是 uploaded_files 列表中的某个文件路径(普通多文件直接用文件路径；FST 多 sheet 用 sheet-suffixed virtual path 如 "/mnt/.../foo.xlsx::轨迹-Arena 1-Subject 1")，value 是 group_name 字符串(如 "treatment" / "control" / "vehicle")。详见 ethoinsight-grouping skill 的 references/lead-translates-answer.md。传入后会写入 /mnt/user-data/workspace/groups.json 并在 plan_metrics.json 的 inputs.groups_file 字段记录路径，下游 code-executor 据此进行分组聚合统计，避免 code-executor 看不到分组幻觉脚本去探测 drug 列。None = 无分组信息(单组分析，或 lead 还没收集到分组)。

    Returns:
      status="ok" 时:
        {"status": "ok",
         "plan_path": "/mnt/user-data/workspace/plan_metrics.json",
         "plan_summary": {"paradigm": "epm", "metric_count": 5, "subject_count": 2,
                          "metric_ids": ["open_arm_time_ratio", ...]}}
      status="error" 时:
        {"status": "error",
         "error_code": "file_not_found"|"format_unrecognized"|"parse_failed"|
                       "unknown_paradigm"|"columns_missing"|"zone_unnamed"|
                       "schema_violation"|"empty_plan"|"unknown_metric"|
                       "workspace_missing"|"no_files_provided",
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

    # Lazy import of the ethoinsight domain library: keep the deerflow harness
    # importable (agent stack / middleware / other tools) without ethoinsight
    # installed. The package is only required when this tool actually runs.
    from ethoinsight.catalog.loader import load_common_catalog
    from ethoinsight.catalog.resolve import (
        ResolveError,
        metric_metadata_to_dict,
        plan_metrics_to_dict,
        resolve_metrics,
    )
    from ethoinsight.parse._core import detect_ethovision, parse_header

    # Step 1.5: expand multi-sheet XLSX files into individual sheet entries.
    # FST-style exports pack 2 subjects in 1 XLSX via separate sheets;
    # each sheet needs its own PlanMetric so the downstream pipeline sees
    # N subjects rather than 1.
    expanded_files: list[str] = []
    for uf in uploaded_files:
        if uf.endswith((".xlsx", ".xls")):
            real_path = replace_virtual_path(uf, thread_data)
            if Path(real_path).exists():
                try:
                    import pandas as pd
                    xl = pd.ExcelFile(real_path)
                    if len(xl.sheet_names) > 1:
                        for sn in xl.sheet_names:
                            expanded_files.append(f"{uf}::{sn}")
                        continue
                except Exception:
                    pass  # fall through to single-entry
        expanded_files.append(uf)
    uploaded_files = expanded_files

    # Step 2-4: per-file validation (existence + EthoVision detect + header parse)
    # 任何文件失败立即返回,带 failed_file 字段给 lead 看是哪个文件出问题。
    # header 用第 1 个文件的(同一批次 EV 导出列结构一致;若不一致 catalog resolve 会报 columns_missing)。
    # Supports ::sheet_name suffix for multi-sheet XLSX files (stripped before
    # path resolution, re-attached for the detect / parse calls that consume it).
    columns: list[str] = []
    for idx, uploaded_file in enumerate(uploaded_files):
        # Strip ::sheet suffix for filesystem path resolution
        clean_virtual = uploaded_file.split("::")[0] if "::" in uploaded_file else uploaded_file
        real_fs_path = replace_virtual_path(clean_virtual, thread_data)
        # Effective path passed to detect_ethovision / parse_header (may include ::sheet)
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

    # Step 4.5/4.6: read parameter_overrides + column_aliases from experiment-context.json.
    # parameter_overrides (Sprint 4.5): 用户确认的参数覆盖 + analysis_config_id。
    # column_aliases (Sprint 1 列语义对齐): set_experiment_paradigm(column_semantics=...)
    #   投影出的别名表；不读到它 → resolve 拿不到别名 → 自定义分析区列仍 columns_missing。
    from deerflow.agents.middlewares.experiment_context import read_context

    ctx = read_context(real_workspace_path)
    parameter_overrides = ctx.get("parameter_overrides", {}) if ctx else {}
    analysis_config_id = ctx.get("analysis_config_id", "PENDING") if ctx else "PENDING"
    column_aliases = ctx.get("column_aliases") if ctx else None

    # Step 5: resolve catalog → PlanMetrics
    # raw_files 走虚拟路径,避免宿主机路径泄漏到 plan_metrics.json 后被 subagent
    # 照抄进 bash --input。IO 部分(detect_ethovision / parse_header)已在 Step 2-4
    # 完成,resolve_metrics 内部按 raw_files 展开 N 个 PlanMetric(每个 subject 一个)。

    # Step 4.5 (Bug #3 fix 2026-05-28): 若 lead 传入 groups 映射,写入 groups.json
    # 并透传给 resolve_metrics;否则 code-executor 看不到分组会幻觉脚本探测 drug 列。
    groups_file_virtual: str | None = None
    groups_file_real: str | None = None
    if groups:
        if not isinstance(groups, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in groups.items()
        ):
            return _error_result(
                "schema_violation",
                f"groups must be dict[str, str] (subject -> group_name), got {type(groups).__name__}",
            )
        # 校验所有 key 都对应 uploaded_files 中的某个文件
        unknown_keys = [k for k in groups if k not in uploaded_files]
        if unknown_keys:
            return _error_result(
                "schema_violation",
                f"groups keys 必须出现在 uploaded_files 中: 未知键 {unknown_keys}; "
                f"可用键 {uploaded_files}",
            )
        groups_path = Path(real_workspace_path) / "groups.json"
        try:
            groups_path.write_text(
                json.dumps(groups, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            return _error_result(
                "parse_failed",
                f"Failed to write groups.json: {e}",
            )
        groups_file_real = str(groups_path)
        groups_file_virtual = "/mnt/user-data/workspace/groups.json"
        logger.info(
            "prep_metric_plan: wrote groups.json with %d entries, groups=%s",
            len(groups),
            sorted(set(groups.values())),
        )

    # #6a 层③ prep 侧显式派生组计数（双保险 + 可读性）：
    # groups 是分组事实的源；在写完 groups.json、调 resolve 前显式派生计数传入。
    # 与 resolve 内自派生互为冗余校验（resolve 自派生兜住所有调用方；这里让 prep 的
    # plan 意图自解释，且即便 resolve 自派生因故失效 prep 仍正确）。
    # n_groups=不同组数=len(set(group_names))；n_per_group=最小组 size（gate 语义"每组都≥2"）。
    n_groups_val: int | None = None
    n_per_group_val: int | None = None
    if groups:
        from collections import Counter

        counts = Counter(groups.values())
        if counts:
            n_groups_val = len(counts)
            n_per_group_val = min(counts.values())

    try:
        plan = resolve_metrics(
            paradigm=paradigm,
            columns=columns,
            raw_files=list(uploaded_files),
            workspace_dir=real_workspace_path,
            virtual_workspace_dir="/mnt/user-data/workspace",
            column_aliases=column_aliases,
            groups_file=groups_file_virtual,
            n_per_group=n_per_group_val,  # #6a: 显式派生传入（resolve 也自派生兜底）
            n_groups=n_groups_val,
            overrides=parameter_overrides,  # Sprint 4.5: 把用户确认的参数覆盖真正传入计算（非仅展示）
            common_catalog=load_common_catalog(),  # Sprint 4.5: shared_parameters 来源；缺它则 velocity_*/pendulum_* 等共享参数进不了 parameters_in_use，override 无可覆盖
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
    # Sprint 4.5: embed analysis_config_id in plan for lineage tracing
    plan_dict["analysis_config_id"] = analysis_config_id
    if parameter_overrides:
        plan_dict["parameter_overrides"] = parameter_overrides
    plan_path = Path(real_workspace_path) / "plan_metrics.json"
    try:
        plan_path.write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        return _error_result(
            "parse_failed",
            f"Failed to write plan_metrics.json: {e}",
        )

    # Step 6.1: 写去重元数据旁路文件 _metric_metadata.json（spec 2026-06-22-metric-metadata-sidecar）。
    # plan_metrics.json 的 metrics[] 按 subject 重复（28×5=140 条，133K），report-writer /
    # data-analyst 只需展示+判读元数据（5 条）。旁路是 plan 的去重元数据投影，与主 plan 同源同次
    # 写避免漂移；SSOT 仍是 catalog，旁路是只读投影。几 KB，subagent 单次 read + 按 id 直查，
    # 不再啃 133K 施工文件撑爆 thinking。
    metadata_path = Path(real_workspace_path) / "_metric_metadata.json"
    try:
        metadata_path.write_text(
            json.dumps(metric_metadata_to_dict(plan), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        # 旁路写入失败不阻断主流程——report-writer/data-analyst 有 metric id 兜底（spec §2.5）。
        logger.warning("Failed to write _metric_metadata.json (non-critical): %s", e)

    # Step 6.5: lineage — write overrides file when overrides exist (Sprint 4.5)
    if parameter_overrides:
        overrides_path = Path(real_workspace_path) / f"overrides_{analysis_config_id}.json"
        try:
            overrides_path.write_text(
                json.dumps({"analysis_config_id": analysis_config_id, "parameter_overrides": parameter_overrides}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Failed to write overrides file %s (non-critical)", overrides_path)

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
        "analysis_config_id": analysis_config_id,
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
