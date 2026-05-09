"""Guardrail provider that blocks task(code-executor) when ev19_template is unset.

Works alongside the existing GateEnforcementMiddleware (which checks the `paradigm`
field). The two have orthogonal responsibilities:
  - GateEnforcementMiddleware: paradigm field present and valid
  - Ev19TemplateGuardrailProvider: ev19_template field present in 62-variant whitelist

The provider only blocks task(code-executor). Other tool calls pass through.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from deerflow.guardrails.provider import GuardrailDecision, GuardrailReason, GuardrailRequest

logger = logging.getLogger(__name__)


def _default_workspace_resolver() -> str | None:
    """Default workspace resolver — caller should pass a callable that returns the host workspace path."""
    return None


class Ev19TemplateGuardrailProvider:
    """Block task(code-executor) when experiment-context.json lacks ev19_template.

    Agent sees the error reason and is expected to call set_experiment_paradigm
    (with ev19_template) or ask_clarification before retrying.
    """

    name = "ev19-template-guardrail"

    def __init__(self, workspace_resolver: Callable[[], str | None] | None = None):
        self._resolve_workspace = workspace_resolver or _default_workspace_resolver

    # --- core check (sync) ---

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Only inspect task() calls
        if request.tool_name != "task":
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        # Only inspect task(code-executor) — other subagents are unaffected
        subagent = request.tool_input.get("subagent_type", "") if request.tool_input else ""
        if "code-executor" not in subagent:
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        workspace = self._resolve_workspace()
        if workspace is None:
            # No workspace context available — fail-open (don't block)
            logger.debug("Ev19TemplateGuardrailProvider: workspace unresolvable, allowing task call")
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        ctx = self._read_context(workspace)
        if ctx is None or not ctx.get("ev19_template"):
            return GuardrailDecision(
                allow=False,
                reasons=[
                    GuardrailReason(
                        code="ethoinsight.no_ev19_template",
                        message=(
                            "EV19 模板尚未设置。请先调用 set_experiment_paradigm(..., ev19_template=...) "
                            "确定模板变体（参考 ethovision-paradigm-knowledge skill 中 references/_facts.md "
                            "的 62 变体白名单）。如果信息不足，先 ask_clarification 反问用户。"
                        ),
                    )
                ],
                policy_id="ev19-template-guardrail",
            )

        return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

    # --- async wrapper ---

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)

    # --- helpers ---

    def _read_context(self, workspace: str) -> dict | None:
        path = Path(workspace) / "experiment-context.json"
        try:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read experiment-context.json: %s", e)
            return None
