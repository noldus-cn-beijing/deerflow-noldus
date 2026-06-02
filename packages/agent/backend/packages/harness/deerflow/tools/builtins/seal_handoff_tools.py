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

    # 1. 注入 analysis_config_id (subagent 不用手动传)
    payload.setdefault("analysis_config_id", _read_analysis_config_id(workspace))

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

    # 4. 写 manifest
    sha256 = hashlib.sha256(json_bytes).hexdigest()
    _update_manifest(workspace, filename, sha256, payload["analysis_config_id"])

    return f"OK: sealed {filename} (sha256={sha256[:12]}...)"


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
        parameter_audit_findings: Sprint 3 新增。参数适配性审计发现列表，
            每条含 parameter/metric/severity/used_value/observed_distribution/
            mismatch_kind/suggestion/blocks_downstream
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
        status: "completed" / "partial" / "failed"（全部失败时为 failed）
        gate_signals: 决策信号
    """
    payload = {
        "status": status,
        "paradigm": paradigm,
        "summary": summary,
        "chart_files": chart_files or [],
        "failed_charts": failed_charts or [],
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
