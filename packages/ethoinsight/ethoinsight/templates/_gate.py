"""Shared soft gate for paradigm template entrypoints.

Each template's analysis steps must check ev19_template is set before doing work,
to avoid silently writing wrong-template results when the lead agent skipped
set_experiment_paradigm.
"""

from __future__ import annotations

import json
from pathlib import Path


def require_ev19_template(workspace_dir: str) -> dict | None:
    """Return None if ev19_template is set; return structured error dict if missing.

    Caller (template entrypoint) returns the dict directly to its caller,
    short-circuiting the analysis. The error dict contains a `remediation` field
    so the lead agent (reading it via handoff) knows what to do next.
    """
    ctx_path = Path(workspace_dir) / "experiment-context.json"
    if not ctx_path.exists():
        return {
            "status": "error",
            "reason": "experiment-context.json 不存在 — ev19_template 字段未设置",
            "remediation": (
                "lead agent 应先调用 set_experiment_paradigm(paradigm, ..., ev19_template) "
                "确定模板。如不能确定，先 ask_clarification 反问用户。"
            ),
        }
    try:
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {
            "status": "error",
            "reason": f"无法解析 experiment-context.json: {e}",
            "remediation": "lead agent 应重新调用 set_experiment_paradigm 写入正确的 context。",
        }
    if not ctx.get("ev19_template"):
        return {
            "status": "error",
            "reason": "experiment-context.json 缺少 ev19_template 字段",
            "remediation": (
                "lead agent 应调用 set_experiment_paradigm(..., ev19_template=...) 补齐字段。"
                "参考 ethovision-paradigm-knowledge skill 的 _facts.md 选择白名单内变体。"
            ),
        }
    return None
