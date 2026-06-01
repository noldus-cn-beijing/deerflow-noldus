"""present_assumptions — 聚合分析假设并渲染为可折叠 Markdown 卡片。

Sprint 7: 轻量不强制，lead 按需主动调用。不做 GateProvider。
聚合来源:
  - experiment-context.json → analysis_config_id, parameter_overrides, gate_completed
  - plan_metrics.json → parameters_in_use (S2b)
  - handoff_data_analyst.json → quality_warnings, parameter_audit_findings
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from deerflow.agents.middlewares.experiment_context import read_context
from deerflow.agents.thread_state import ThreadState

logger = logging.getLogger(__name__)

CONTAINER_WORKSPACE = "/mnt/user-data/workspace/"


def _read_json_safe(path: Path) -> dict | None:
    """Read a JSON file, return None on any failure."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def _build_assumptions_markdown(
    ctx: dict | None,
    plan: dict | None,
    da_handoff: dict | None,
) -> str:
    """Build the markdown content for the assumptions panel.

    All inputs may be None (file missing / unreadable) — the function
    gracefully omits sections that have no data.
    """
    config_id = "?"
    overrides_section = ""
    data_quality_section = ""
    parameter_audit_section = ""
    parameters_in_use_section = ""

    # --- Parameter configuration ---
    if ctx:
        config_id = ctx.get("analysis_config_id", "?")
        overrides = ctx.get("parameter_overrides")
        if overrides and isinstance(overrides, dict) and len(overrides) > 0:
            override_lines = "\n".join(f"  - `{k}`: `{v}`" for k, v in sorted(overrides.items()))
            overrides_section = f"\n### 参数覆盖\n{override_lines}\n"
        else:
            overrides_section = "\n### 参数覆盖\n（使用 catalog 默认值）\n"

        gates = ctx.get("gate_completed", [])
        if isinstance(gates, list):
            gate_str = ", ".join(gates) if gates else "（无）"
        else:
            gate_str = str(gates)

    # --- Parameters in use (from plan_metrics.json) ---
    if plan:
        params = plan.get("parameters_in_use")
        if isinstance(params, dict) and params:
            param_lines = "\n".join(f"  - `{k}`: `{v}`" for k, v in sorted(params.items()))
            parameters_in_use_section = f"\n### 运行时参数\n{param_lines}\n"

    # --- Data quality ---
    if da_handoff:
        quality_warnings = da_handoff.get("quality_warnings", [])
        if isinstance(quality_warnings, list):
            critical = [w for w in quality_warnings if isinstance(w, dict) and w.get("severity") == "critical"]
            blocks = [w for w in critical if w.get("blocks_downstream")]
            critical_count = len(critical)
            blocks_count = len(blocks)
        else:
            critical_count = 0
            blocks_count = 0

        if critical_count > 0:
            data_quality_section = (
                f"\n### 数据质量\n"
                f"- critical warnings: **{critical_count}** 条（blocks_downstream: **{blocks_count}** 条）\n"
            )
            for w in critical:
                code = w.get("code", "unknown")
                msg = w.get("message", "")
                data_quality_section += f"  - `{code}`: {msg}\n"

        # --- Parameter audit ---
        audit = da_handoff.get("parameter_audit_findings", [])
        if isinstance(audit, list) and len(audit) > 0:
            crit_audit = [a for a in audit if isinstance(a, dict) and a.get("severity") == "critical"]
            data_quality_section += (
                f"\n### 参数审计\n"
                f"- findings: **{len(audit)}** 条（critical: **{len(crit_audit)}** 条）\n"
            )
            for a in audit[:5]:  # Cap at 5 to avoid bloating
                param = a.get("parameter", "?")
                sev = a.get("severity", "?")
                data_quality_section += f"  - `{param}` [{sev}]\n"
            if len(audit) > 5:
                data_quality_section += f"  - ... and {len(audit) - 5} more\n"

    # --- Assemble ---
    has_content = overrides_section or data_quality_section or parameters_in_use_section
    if not has_content:
        return ""

    lines = [
        "<details>",
        f"<summary>分析假设摘要 (config_id={config_id})</summary>",
        "",
        overrides_section,
    ]
    if parameters_in_use_section:
        lines.append(parameters_in_use_section)
    if data_quality_section:
        lines.append(data_quality_section)
    lines.append("</details>")

    return "\n".join(lines)


@tool("present_assumptions", parse_docstring=True)
def present_assumptions_tool(
    workspace_dir: str = CONTAINER_WORKSPACE,
    runtime: ToolRuntime[ContextT, ThreadState] = None,
) -> str:
    """聚合分析假设并渲染为可折叠 Markdown 卡片。

    读取 experiment-context.json / plan_metrics.json / handoff_data_analyst.json，
    提取参数配置、数据质量警告、参数审计发现，渲染为 &lt;details&gt; Markdown。

    当所有数据均为默认值（无覆盖、无警告、无审计发现）时返回空字符串，
    表明无需向用户展示假设面板。

    Args:
        workspace_dir: 工作目录。默认 "/mnt/user-data/workspace/"。

    Returns:
        Markdown 文本（前端渲染为折叠卡片），或空字符串表示无内容。
    """
    # Resolve host workspace path
    actual_workspace = workspace_dir
    if runtime is not None and runtime.state is not None:
        thread_data = runtime.state.get("thread_data")
        if isinstance(thread_data, dict):
            host_ws = thread_data.get("workspace_path")
            if host_ws:
                actual_workspace = host_ws

    ws = Path(actual_workspace)

    ctx = read_context(actual_workspace)
    plan = _read_json_safe(ws / "plan_metrics.json")
    da_handoff = _read_json_safe(ws / "handoff_data_analyst.json")

    return _build_assumptions_markdown(ctx, plan, da_handoff)
