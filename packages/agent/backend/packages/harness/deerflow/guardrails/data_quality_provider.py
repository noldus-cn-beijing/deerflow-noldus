"""Guardrail provider that blocks downstream subagent dispatch on critical data quality warnings.

In manual workflow_mode, this provider blocks task() calls to data-analyst,
chart-maker, and report-writer when the code-executor handoff contains
severity='critical' AND blocks_downstream=true warnings that haven't been
acknowledged by the user.

Auto mode: NOT mounted — warnings are displayed via QualityWarningBanner only.
Manual mode: Mounted alongside GateEnforcementMiddleware (Gate 2 already blocks
data-analyst on any critical warning; this provider adds blocks_downstream=true
granularity and extends coverage to chart-maker / report-writer).

Deny messages carry the full structured warning payload (code, message, evidence,
blocks_downstream) so the frontend can render them with the same QualityWarningBanner.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import override

from deerflow.guardrails.ev19_template_provider import set_ev19_workspace
from deerflow.guardrails.provider import GuardrailDecision, GuardrailReason, GuardrailRequest

logger = logging.getLogger(__name__)

# Subagent types subject to this guardrail
_BLOCKED_SUBAGENTS = {"data-analyst", "chart-maker", "report-writer"}

# Reuse the ev19 workspace contextvar — set by Ev19WorkspaceBridgeMiddleware
# which runs before GuardrailMiddleware in the middleware chain.


class DataQualityGuardrailProvider:
    """Block downstream subagent dispatch on critical+blocks_downstream warnings.

    Only active in manual mode (mounted in agent.py conditionally).
    Agent sees the structured deny and must call ask_clarification to surface
    warnings to the user, then set_experiment_paradigm(acknowledge_quality=True)
    before retrying.
    """

    name = "data-quality-guardrail"

    def __init__(self, workspace_resolver=None):
        if workspace_resolver is not None:
            self._resolve_workspace = workspace_resolver
        else:
            # Reuse the ev19 workspace contextvar — set by Ev19WorkspaceBridgeMiddleware
            from deerflow.guardrails.ev19_template_provider import _ev19_workspace

            self._resolve_workspace = lambda: _ev19_workspace.get()

    # --- core check (sync) ---

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Only inspect task() calls
        if request.tool_name != "task":
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        # Only inspect task() to blocked subagent types
        subagent = (request.tool_input or {}).get("subagent_type", "")
        if subagent not in _BLOCKED_SUBAGENTS:
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        workspace = self._resolve_workspace()
        if workspace is None:
            logger.debug("DataQualityGuardrailProvider: workspace unresolvable, allowing task call")
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        # Check if quality is already acknowledged
        if self._is_quality_acknowledged(workspace):
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        # Read blocking warnings from handoff
        blocking_warnings = self._get_blocking_warnings(workspace)
        if not blocking_warnings:
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        # Build structured deny message
        warning_payloads = [
            {
                "code": w.get("code", "UNKNOWN"),
                "message": w.get("message", ""),
                "metric": w.get("metric", ""),
                "severity": w.get("severity", "critical"),
                "evidence": w.get("evidence", {}),
                "blocks_downstream": True,
            }
            for w in blocking_warnings
        ]

        warning_lines = "\n".join(
            f"  - [{w.get('code', 'UNKNOWN')}] {w.get('message', '')}"
            for w in blocking_warnings
        )

        deny_message = (
            f"数据质量检查发现 {len(blocking_warnings)} 条 critical 且 blocks_downstream=true "
            f"的警告，需用户确认后方可派遣 {subagent}：\n{warning_lines}\n\n"
            f"请调用 ask_clarification 告知用户上述质量问题，等用户确认后调 "
            f"set_experiment_paradigm(acknowledge_quality=True) 再继续。"
        )

        return GuardrailDecision(
            allow=False,
            reasons=[
                GuardrailReason(
                    code="ethoinsight.quality_blocks_downstream",
                    message=deny_message,
                )
            ],
            policy_id="data-quality-guardrail",
            metadata={"warnings": warning_payloads},
        )

    # --- async wrapper ---

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)

    # --- helpers ---

    def _is_quality_acknowledged(self, workspace: str) -> bool:
        """Check if gate2_quality_acknowledged is in experiment-context.json."""
        ctx = self._read_context(workspace)
        if not ctx:
            return False
        gate_completed = ctx.get("gate_completed", [])
        if not isinstance(gate_completed, list):
            return False
        return "gate2_quality_acknowledged" in gate_completed

    def _get_blocking_warnings(self, workspace: str) -> list[dict]:
        """Extract severity='critical' AND blocks_downstream=true warnings from handoff."""
        handoff = self._read_handoff(workspace)
        if not handoff:
            return []
        warnings = handoff.get("data_quality_warnings", [])
        if not isinstance(warnings, list):
            return []
        return [
            w
            for w in warnings
            if isinstance(w, dict)
            and w.get("severity") == "critical"
            and w.get("blocks_downstream") is True
        ]

    def _read_context(self, workspace: str) -> dict | None:
        path = Path(workspace) / "experiment-context.json"
        try:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read experiment-context.json: %s", e)
            return None

    def _read_handoff(self, workspace: str) -> dict | None:
        path = Path(workspace) / "handoff_code_executor.json"
        try:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read handoff_code_executor.json: %s", e)
            return None
