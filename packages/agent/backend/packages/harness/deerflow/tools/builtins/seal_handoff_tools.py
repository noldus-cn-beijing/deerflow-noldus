"""4 个 first-party tool — subagent 调用本 tool 结构化 seal handoff 到 workspace。

设计原则（grill 锁定 Sprint 0）：
1. LLM 只填 tool 参数（LangChain tool_call schema 自动校验类型/必填）
2. tool 内部 Pydantic 校验 + atomic write + .lineage/manifest.json 记录
3. 4 个 tool 共享 _seal_handoff helper，避免重复
4. 调用方:
    - code-executor → seal_code_executor_handoff
    - data-analyst → seal_data_analyst_handoff
    - chart-maker → seal_chart_maker_handoff
    - report-writer → seal_report_writer_handoff
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain.tools import tool
from pydantic import ValidationError

from deerflow.subagents.handoff_schemas import (
    ChartMakerHandoff,
    CodeExecutorHandoff,
    DataAnalystHandoff,
    ReportWriterHandoff,
)
from deerflow.tools.types import Runtime

logger = logging.getLogger(__name__)


# ============================================================================
# Sprint 6: experiment_summary memory fact
# ============================================================================


def _extract_n_per_group(workspace: Path) -> str:
    """Read n_per_group from handoff_code_executor.json (deterministic, no LLM).

    Priority: metadata.n_per_group > first metrics_summary group's n > "unknown".
    """
    ce_path = workspace / "handoff_code_executor.json"
    if not ce_path.exists():
        return "unknown"
    try:
        ce = json.loads(ce_path.read_text(encoding="utf-8"))
        # Priority 1: explicit metadata field
        meta = ce.get("metadata")
        if isinstance(meta, dict) and "n_per_group" in meta:
            return str(meta["n_per_group"])
        # Priority 2: scan metrics_summary for first group's first metric's n
        ms = ce.get("metrics_summary")
        if isinstance(ms, dict):
            for _group, metrics in ms.items():
                if isinstance(metrics, dict):
                    for _metric, stats in metrics.items():
                        if isinstance(stats, dict) and "n" in stats:
                            return str(stats["n"])
        return "unknown"
    except Exception:
        return "unknown"


def _extract_key_findings_count(workspace: Path) -> int:
    """Read key_findings count from handoff_data_analyst.json."""
    da_path = workspace / "handoff_data_analyst.json"
    if not da_path.exists():
        return 0
    try:
        da = json.loads(da_path.read_text(encoding="utf-8"))
        return len(da.get("key_findings", []))
    except Exception:
        return 0


def _write_experiment_summary_memory(
    workspace: Path,
    paradigm: str,
    config_id: str,
    thread_id: str,
    user_id: str | None,
) -> None:
    """Write an experiment_summary fact to memory (deterministic, no LLM).

    Non-fatal: all exceptions are caught and logged as warnings.
    """
    try:
        from deerflow.agents.memory.updater import create_memory_fact

        n_per_group = _extract_n_per_group(workspace)
        key_findings_count = _extract_key_findings_count(workspace)

        # Lineage (thread_id/config_id) is folded into content because
        # create_memory_fact() hardcodes source="manual" and does not accept
        # a source kwarg — keeping it in content preserves traceability.
        content = (
            f"{paradigm} analysis on {datetime.now(UTC).strftime('%Y-%m-%d')}: "
            f"n_per_group={n_per_group}; "
            f"key_findings_count={key_findings_count}; "
            f"analysis_config_id={config_id}; "
            f"thread={thread_id}"
        )
        create_memory_fact(
            content=content,
            category="experiment_summary",
            confidence=1.0,
            user_id=user_id,
        )
        logger.info("experiment_summary fact written for config_id=%s", config_id)
    except Exception as e:
        logger.warning("Failed to write experiment_summary memory fact: %s", e)


# ============================================================================
# 内部 helper
# ============================================================================


def _resolve_workspace(runtime: Runtime) -> Path:
    """从 runtime state 取 host-side workspace 路径。"""
    state = runtime.state
    if not isinstance(state, dict):
        raise RuntimeError("seal_*_handoff: runtime.state is not a dict")
    thread_data = state.get("thread_data")
    if not isinstance(thread_data, dict):
        raise RuntimeError("seal_*_handoff: thread_data missing from state")
    workspace_path = thread_data.get("workspace_path")
    if not workspace_path:
        raise RuntimeError("seal_*_handoff: workspace_path missing")
    return Path(workspace_path)


# Regex matching markdown image paths that do NOT use the canonical virtual
# prefix. Covers three LLM-written variants that must be normalised to canonical:
#   (outputs/file.png)             — relative, no prefix
#   (mnt/user-data/outputs/…)     — correct prefix but missing the leading slash
#   (/mnt/user-data/outputs/…)    — already canonical; matched idempotently
# The optional ``/?mnt/user-data/`` prefix absorbs both slash-less and slashed
# forms; the literal ``outputs/`` then matches; capture group 1 = the file name.
_BAD_IMG_PATH_RE = re.compile(
    r"\((?:/?mnt/user-data/)?outputs/([^)]+\.(?:png|jpg|jpeg|svg|gif|webp))\)"
)

# Regex matching {{img:<basename>}} placeholders in report.md.
# Layer 1 of the chart image placeholder resolution system: LLM writes
#   ![Figure 1]({{img:plot_trajectory_s0.png}})
# and seal_report_writer_handoff resolves it to the canonical virtual path
# from handoff_chart_maker.json.chart_files.
_IMG_PLACEHOLDER_RE = re.compile(r"\{\{img:([^}]+)\}\}")


# ============================================================================
# report 图片路径规范形态（SSOT）—— 2026-06-18
# spec: docs/superpowers/specs/2026-06-18-report-image-path-ssot-spec.md
#
# report.md 内图片路径的**唯一规范形态**是带前导斜杠的虚拟绝对路径：
#   /mnt/user-data/outputs/<name>.<ext>
#
# 前导斜杠让前端 ``src.startsWith("/mnt/")`` 判断稳定命中；前端把它原样
# 交给 artifact API（``/api/threads/{tid}/artifacts/mnt/user-data/outputs/…``），
# 后端 ``resolve_virtual_path`` 内部 ``lstrip("/")`` 后命中 ``mnt/user-data/`` 前缀。
#
# 这是 report 图片路径规范形态的**唯一定义点**——seal 的两个产出点
# (placeholder 解析 + path normalize) 都调本函数，保证字节一致（SSOT 铁律）。
# ============================================================================
_CANONICAL_PREFIX = "/mnt/user-data/outputs/"


def _to_canonical_artifact_path(name: str) -> str:
    """Return the canonical report-image path for *name*.

    规范形态 = ``/mnt/user-data/outputs/<name>``（带前导斜杠）。
    对已是规范形态（或带多余前导斜杠）的输入幂等归一，不会二次加前缀。

    Args:
        name: 图片文件名（basename 或已是 ``/mnt/user-data/outputs/…`` 全路径）。

    Returns:
        带前导斜杠的规范虚拟绝对路径。
    """
    stripped = name.lstrip("/")
    if stripped.startswith("mnt/user-data/outputs/"):
        return f"/{stripped}"
    return f"{_CANONICAL_PREFIX}{stripped}"


def _load_chart_files_map(workspace: Path) -> dict[str, str]:
    """Return {basename: canonical_virtual_path} from handoff_chart_maker.json.

    Returns empty dict when file absent, unparseable, or chart_files empty.
    The value is the **canonical** form (leading slash) via
    ``_to_canonical_artifact_path`` — the SSOT for report image paths.
    """
    chart_handoff = workspace / "handoff_chart_maker.json"
    if not chart_handoff.exists():
        return {}
    try:
        data = json.loads(chart_handoff.read_text(encoding="utf-8"))
        chart_files = data.get("chart_files", [])
        if not isinstance(chart_files, list):
            return {}
        result: dict[str, str] = {}
        for f in chart_files:
            if isinstance(f, str):
                result[Path(f).name] = _to_canonical_artifact_path(Path(f).name)
        return result
    except Exception:
        return {}


def _resolve_report_image_placeholders(
    report_host_path: Path,
    workspace: Path,
) -> None:
    """Resolve {{img:<basename>}} placeholders in report.md.

    Reads handoff_chart_maker.json from workspace, builds a {basename: full_path}
    mapping, then replaces every {{img:<basename>}} placeholder in the report
    with the canonical virtual path.

    Unmatched basenames → replaced with visible error stub listing available files.
    Missing/empty chart_files → no-op (placeholders survive for human diagnosis).
    """
    if not report_host_path.is_file():
        return

    chart_files_map = _load_chart_files_map(workspace)

    try:
        original = report_host_path.read_text(encoding="utf-8")

        def _replace(match: re.Match[str]) -> str:
            basename = match.group(1).strip()
            if not chart_files_map:
                return match.group(0)          # 无映射时保留原样
            if basename in chart_files_map:
                return chart_files_map[basename]
            # 不匹配 → 可见错误文本
            available = ", ".join(
                sorted(chart_files_map.keys())[:5]
            )
            suffix = f"；可用: {available}" if available else ""
            return f"[图表 '{basename}' 未找到{suffix}]"

        resolved = _IMG_PLACEHOLDER_RE.sub(_replace, original)

        if resolved != original:
            report_host_path.write_text(resolved, encoding="utf-8")
            logger.info(
                "seal_report_writer_handoff: resolved image placeholders in %s",
                report_host_path,
            )
    except Exception:
        logger.warning(
            "seal_report_writer_handoff: image placeholder resolution skipped",
            exc_info=True,
        )


def _normalize_report_image_paths(report_host_path: Path) -> None:
    """Rewrite image paths in report.md to the canonical form (SSOT).

    规范形态 = 带前导斜杠的虚拟绝对路径 ``/mnt/user-data/outputs/<name>.<ext>``
    （见 ``_to_canonical_artifact_path``）。LLM 常写相对形态 ``outputs/file.png``
    或缺前导斜杠的 ``mnt/user-data/outputs/file.png``，两者前端
    ``startsWith("/mnt/")`` 都不命中 → 404。本函数把它们统一到规范形态，
    这样无论 LLM 写成什么，结果都正确（前端只认这一种）。

    幂等：已是规范形态的路径不变。report 文件不存在时静默跳过。
    """
    if not report_host_path.is_file():
        return
    try:
        original = report_host_path.read_text(encoding="utf-8")
        # 替换目标用 canonical helper 拼接，与 _load_chart_files_map 字节一致（SSOT）。
        normalised = _BAD_IMG_PATH_RE.sub(
            lambda m: f"({_to_canonical_artifact_path(m.group(1))})",
            original,
        )
        if normalised != original:
            report_host_path.write_text(normalised, encoding="utf-8")
            logger.info(
                "seal_report_writer_handoff: normalised image paths in %s",
                report_host_path,
            )
    except Exception as e:
        # Non-fatal: log and continue; the seal still succeeds.
        logger.warning("seal_report_writer_handoff: image path normalisation skipped: %s", e)


def _read_analysis_config_id(workspace: Path) -> str:
    """从 experiment-context.json 读 analysis_config_id (Sprint 4.5 填)。

    Sprint 0 阶段: experiment-context.json 可能还没有此字段，返回 "PENDING_SPRINT_4.5"
    占位，Sprint 4.5 实施后会自动正常填入。
    """
    ctx_path = workspace / "experiment-context.json"
    if not ctx_path.exists():
        return "PENDING_SPRINT_4.5"
    try:
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
        return ctx.get("analysis_config_id", "PENDING_SPRINT_4.5")
    except Exception as e:
        logger.warning("read experiment-context.json failed: %s", e)
        return "PENDING_SPRINT_4.5"


def _update_manifest(workspace: Path, handoff_filename: str, sha256: str, analysis_config_id: str) -> None:
    """写 .lineage/manifest.json。

    Sprint 5.5 会进一步用本 manifest 做下游 hash 校验；Sprint 0 只负责写。
    """
    lineage_dir = workspace / ".lineage"
    lineage_dir.mkdir(exist_ok=True)
    manifest_path = lineage_dir / "manifest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                manifest = {}
        except Exception:
            manifest = {}
    else:
        manifest = {}

    manifest[handoff_filename] = {
        "sha256": sha256,
        "analysis_config_id": analysis_config_id,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # atomic write for manifest itself
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(manifest_path)


def _build_task_context(payload: dict[str, Any]) -> dict[str, Any]:
    """从 handoff payload 已有字段确定性组装 task_context 的 4 个字段。

    纯函数、无 LLM、无副作用、任何异常返回部分结果（防御性）。
    """
    tc: dict[str, Any] = {
        "file_changes": [],
        "verify_commands": [],
        "failed_paths": [],
        "pending_items": [],
    }
    try:
        # file_changes: 从 output_files 的 value（路径）提取
        output_files = payload.get("output_files") or {}
        paths: list[str] = []
        for v in output_files.values():
            if isinstance(v, str):
                paths.append(v)
            elif isinstance(v, list):
                paths.extend(p for p in v if isinstance(p, str))
        tc["file_changes"] = paths

        # verify_commands: 对每个产物文件生成存在性 + JSON 校验命令（模板）
        cmds: list[str] = []
        for p in paths:
            if p.endswith(".json"):
                cmds.append(f"python -m json.tool {p} > /dev/null")
            else:
                cmds.append(f"ls {p}")
        tc["verify_commands"] = cmds

        # failed_paths: 从 errors 提取（errors 是产出方记录的失败事实）
        errors = payload.get("errors") or []
        tc["failed_paths"] = [e for e in errors if isinstance(e, str)]

        # pending_items: 暂留空。
        # 真实 partial 语义是"指标已算完、但统计检验因样本量(n=1/n=2)被跳过"，
        # 而非"指标未算完"——partial 的原因已由 gate_signals.statistical_validity="skipped"
        # + data_quality_warnings 充分表达，lead 据此决策无需本字段。
        # 当前无可靠的"未完成明细"数据源（errors 在此类 partial 时恒空）。
        # TODO: 若未来出现"指标脚本失败导致 partial"的真实场景，再从
        #   plan_metrics.json(计划) vs metrics_summary(实际) 的差集派生。
        # 不从 errors 派生（恒空且误导）。
    except Exception:
        pass  # 防御性：组装失败不影响 seal 主流程
    return tc


def _seal_handoff_to_workspace(
    model_cls: type,
    filename: str,
    payload: dict[str, Any],
    workspace: Path,
) -> str:
    """Pure-function variant of _seal_handoff: Pydantic validate → atomic write → manifest.

    Takes an explicit workspace ``Path`` instead of resolving it from ``Runtime``.
    Used by harness-level auto-seal where no Runtime is available (Spec C).

    Same contract as _seal_handoff: all failure paths raise ValueError.
    """
    # 1. 注入 analysis_config_id (subagent 不用手动传)
    payload.setdefault("analysis_config_id", _read_analysis_config_id(workspace))

    # 1.5. 自动组装 task_context —— 仅当目标 schema 仍声明该字段时注入。
    # ethoinsight 4 个 handoff 已移除该字段（拆为旁路 lineage，spec 2026-06-18）；
    # 通用 handoff schema 若仍有该字段则保持组装，向前兼容。task_context 是死重量
    # （下游不消费），无条件注入会把主 handoff 顶过 sandbox read_file 50K 截断线。
    if "task_context" in getattr(model_cls, "model_fields", {}):
        payload.setdefault("task_context", _build_task_context(payload))

    # 2. Pydantic 校验
    try:
        handoff = model_cls(**payload)
    except ValidationError as e:
        raise ValueError(
            f"seal_{filename}: schema validation failed: {e}. "
            f"Check field names/types against {model_cls.__name__} schema."
        ) from e

    # 3. Atomic write (tmp + rename)
    final_path = workspace / filename
    tmp_path = workspace / f"{filename}.tmp"
    json_bytes = handoff.model_dump_json(indent=2, exclude_none=False).encode("utf-8")
    tmp_path.write_bytes(json_bytes)
    os.rename(tmp_path, final_path)  # POSIX atomic

    # 3.5. chmod 0o644（Spec1 教训：文件权限 — downstream 工具需可读）
    os.chmod(final_path, 0o644)

    # 4. 写 manifest
    sha256 = hashlib.sha256(json_bytes).hexdigest()
    _update_manifest(workspace, filename, sha256, payload["analysis_config_id"])

    return f"OK: sealed {filename} (sha256={sha256[:12]}...)"


def _seal_handoff(
    model_cls: type,
    filename: str,
    payload: dict[str, Any],
    runtime: Runtime,
) -> str:
    """共享 helper: Pydantic 校验 → atomic write → 写 manifest → 返回 OK。

    所有失败路径都返回 ValueError（LangChain 会自动转 error ToolMessage 给 LLM）。
    """
    workspace = _resolve_workspace(runtime)
    return _seal_handoff_to_workspace(model_cls, filename, payload, workspace)


# ============================================================================
# 4 个 first-party tool
# ============================================================================


@tool("seal_code_executor_handoff", parse_docstring=True)
def seal_code_executor_handoff(
    status: str,
    summary: str,
    paradigm: str,
    metrics_summary: dict[str, dict[str, dict[str, Any]]] | None = None,
    per_subject: dict[str, dict[str, Any]] | None = None,
    statistics: dict[str, Any] | None = None,
    output_files: dict[str, Any] | None = None,
    data_quality_warnings: list[dict[str, Any]] | None = None,
    errors: list[str] | None = None,
    confidence: float | None = None,
    ev19_template: str | None = None,
    inputs: dict[str, Any] | None = None,
    gate_signals: dict[str, Any] | None = None,
    runtime: Runtime = None,
) -> str:
    """Code-executor 完成指标计算后，封存 handoff_code_executor.json。

    严禁直接用 write_file 写 handoff_code_executor.json，必须走本 tool。

    Args:
        status: 执行状态: "completed" / "partial" / "failed"
        summary: 一句话总结
        paradigm: 范式名，如 "fst" / "epm"
        metrics_summary: 嵌套 dict: group -> metric -> {mean, std, n, parameters_used, ...}
        per_subject: 每个 subject 的原始数据: {subject_name: {metric: value}}
        statistics: 组间统计检验结果
        output_files: 产物文件路径表
        data_quality_warnings: 警告列表，每条含 severity/code/metric/message/evidence/blocks_downstream
        errors: 错误信息列表
        confidence: 整体置信度 [0,1]
        ev19_template: EV19 模板 ID，如 'fst-modified'
        inputs: 输入信息: {raw_files: [...], groups: {...}}
        gate_signals: 决策信号
    """
    payload = {
        "status": status,
        "summary": summary,
        "paradigm": paradigm,
        "metrics_summary": metrics_summary or {},
        "per_subject": per_subject or {},
        "statistics": statistics or {},
        "output_files": output_files or {},
        "data_quality_warnings": data_quality_warnings or [],
        "errors": errors or [],
        "confidence": confidence,
        "ev19_template": ev19_template,
        "inputs": inputs,
        "gate_signals": gate_signals,
    }
    return _seal_handoff(CodeExecutorHandoff, "handoff_code_executor.json", payload, runtime)


@tool("seal_data_analyst_handoff", parse_docstring=True)
def seal_data_analyst_handoff(
    status: str,
    key_findings: list[str] | None = None,
    outlier_findings: list[dict[str, Any]] | None = None,
    excluded_metrics: list[str] | None = None,
    method_warnings: list[str] | None = None,
    recommendations: list[str] | None = None,
    errors: list[str] | None = None,
    gate_signals: dict[str, Any] | None = None,
    quality_warnings: list[dict[str, Any]] | None = None,
    parameter_audit_findings: list[dict[str, Any]] | None = None,
    runtime: Runtime = None,
) -> str:
    """Data-analyst 完成分析后，封存 handoff_data_analyst.json。

    严禁直接用 write_file 写 handoff_data_analyst.json，必须走本 tool。

    Args:
        status: "completed" / "failed"
        key_findings: 1-5 条核心发现
        outlier_findings: 异常 subject 列表，每条含 subject/metric/value/deviation/counterfactual
        excluded_metrics: 因质量问题被排除的指标
        method_warnings: 统计方法警告
        recommendations: 建议后续操作
        errors: 错误信息
        gate_signals: 决策信号
        quality_warnings: 从 handoff_code_executor.json 透传的 data_quality_warnings
        parameter_audit_findings: 恒传空数组 []。2026-06-18 起 data-analyst 不再产出
            参数审计（判据行为学上造不出来，移出判读路径）。字段保留为向前兼容 + 将来以
            确定性代码接入时复用；调用时 parameter_audit_findings=[] 即可，gate_signals
            的 parameter_audit_findings_count / parameter_audit_critical_count 恒为 0。
    """
    payload = {
        "status": status,
        "key_findings": key_findings or [],
        "outlier_findings": outlier_findings or [],
        "excluded_metrics": excluded_metrics or [],
        "method_warnings": method_warnings or [],
        "recommendations": recommendations or [],
        "errors": errors or [],
        "gate_signals": gate_signals,
        "quality_warnings": quality_warnings or [],
        "parameter_audit_findings": parameter_audit_findings or [],
    }
    return _seal_handoff(DataAnalystHandoff, "handoff_data_analyst.json", payload, runtime)


@tool("seal_chart_maker_handoff", parse_docstring=True)
def seal_chart_maker_handoff(
    paradigm: str,
    summary: str,
    chart_files: list[str] | None = None,
    failed_charts: list[dict[str, str]] | None = None,
    remaining_charts: list[dict[str, str]] | None = None,
    status: str = "completed",
    gate_signals: dict[str, Any] | None = None,
    runtime: Runtime = None,
) -> str:
    """Chart-maker 完成绘图后，封存 handoff_chart_maker.json。

    严禁直接用 write_file 写 handoff_chart_maker.json，必须走本 tool。

    Args:
        paradigm: 范式名
        summary: 一句话描述生成的图表
        chart_files: 成功的图表 png 路径（必须在 /mnt/user-data/outputs/ 下）
        failed_charts: 失败列表，每条 {chart_id, reason}
        remaining_charts: P5 预算降级指纹——被 chart_budget 截断未画的 per_subject 图，
            每条 {chart_id, reason}（reason 通常 "chart_budget_truncated"）。
            来自 prep_chart_plan 返回的 plan_summary.budget_remaining_ids。无截断时省略/[]。
        status: "completed" / "partial" / "failed"（全部失败时为 failed）
        gate_signals: 决策信号
    """
    payload = {
        "status": status,
        "paradigm": paradigm,
        "summary": summary,
        "chart_files": chart_files or [],
        "failed_charts": failed_charts or [],
        "remaining_charts": remaining_charts or [],
        "gate_signals": gate_signals,
    }
    return _seal_handoff(ChartMakerHandoff, "handoff_chart_maker.json", payload, runtime)


@tool("seal_report_writer_handoff", parse_docstring=True)
def seal_report_writer_handoff(
    status: str,
    report_path: str,
    sections_written: list[str] | None = None,
    errors: list[str] | None = None,
    gate_signals: dict[str, Any] | None = None,
    runtime: Runtime = None,
) -> str:
    """Report-writer 完成写报告后，封存 handoff_report_writer.json。

    严禁直接用 write_file 写 handoff_report_writer.json，必须走本 tool。

    Args:
        status: "completed" / "failed"
        report_path: 报告 md 文件路径
        sections_written: 已写的段落，如 ["Results", "Discussion"]
        errors: 错误信息
        gate_signals: 决策信号
    """
    payload = {
        "status": status,
        "report_path": report_path,
        "sections_written": sections_written or [],
        "errors": errors or [],
        "gate_signals": gate_signals,
    }

    # 0. 解析 {{img:...}} 占位符（chart image placeholder system）
    # Layer 1: resolves LLM-written placeholders to canonical virtual paths
    # from handoff_chart_maker.json.chart_files before the Layer 2
    # _normalize_report_image_paths prefix fix.
    try:
        _ws = _resolve_workspace(runtime)
        _report_host = _ws.parent / "outputs" / Path(report_path).name
        _resolve_report_image_placeholders(_report_host, _ws)
    except Exception as _e:
        logger.warning("seal_report_writer_handoff: image placeholder resolution failed: %s", _e)

    # 1. 规范化图片路径前缀（现有逻辑，保留）
    # Normalise image paths in the report file before sealing: the artifacts API
    # requires ``mnt/user-data/outputs/file.png`` (no leading slash), but LLMs
    # often write ``outputs/file.png`` or ``/mnt/user-data/outputs/file.png``,
    # both of which return 400 Bad Request in the frontend. Fix it server-side
    # so the result is correct regardless of what the model wrote.
    try:
        _ws = _resolve_workspace(runtime)
        # report_path is a virtual path like /mnt/user-data/outputs/report.md;
        # derive the host path by replacing the virtual prefix with the outputs dir.
        _report_host = _ws.parent / "outputs" / Path(report_path).name
        _normalize_report_image_paths(_report_host)
    except Exception as _e:
        logger.warning("seal_report_writer_handoff: image normalisation pre-step failed: %s", _e)

    result = _seal_handoff(ReportWriterHandoff, "handoff_report_writer.json", payload, runtime)

    # Sprint 6: write experiment_summary memory fact on successful completion
    if status == "completed":
        try:
            workspace = _resolve_workspace(runtime)
            thread_data = runtime.state.get("thread_data", {})
            config_id = payload.get("analysis_config_id", _read_analysis_config_id(workspace))

            # Read paradigm from experiment-context.json
            ctx_path = workspace / "experiment-context.json"
            paradigm = "unknown"
            if ctx_path.exists():
                try:
                    ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
                    paradigm = ctx.get("paradigm", "unknown")
                except Exception:
                    pass

            thread_id = runtime.state.get("thread_id", "unknown")
            user_id = thread_data.get("user_id")

            _write_experiment_summary_memory(
                workspace=workspace,
                paradigm=paradigm,
                config_id=config_id,
                thread_id=thread_id,
                user_id=user_id,
            )
        except Exception as e:
            logger.warning("S6 memory injection skipped: %s", e)

    return result
