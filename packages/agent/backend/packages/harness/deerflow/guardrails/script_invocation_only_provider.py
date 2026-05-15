"""ScriptInvocationOnlyProvider — gate code-executor's bash to script invocations + file ops only.

This Guardrail enforces the 'script-per-metric' architecture: code-executor's
bash tool must only be used to either invoke an `ethoinsight.scripts.*` script
via ``python -m`` or perform safe file operations (mkdir / cp / mv / ls / cat /
grep / head / tail). Any other bash command (including ``python -c``,
``pip install``, arbitrary scripts) is denied with a reason that tells the
agent the correct path forward.

White-list rather than black-list: the allowed shape is small and stable; new
scripts are auto-allowed by the same pattern without touching this provider.

Only applies to subagents whose agent_id starts with 'subagent:code-executor'.
Lead agent and other subagents pass through unchanged.

See: docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md §4
"""

from __future__ import annotations

import re

from deerflow.guardrails.provider import (
    GuardrailDecision,
    GuardrailReason,
    GuardrailRequest,
)

# Match `python -m ethoinsight.scripts.<paradigm>.<script>` at the start of the command.
# Supports leading whitespace and any args after the module name.
_ALLOWED_PYTHON_PATTERN = re.compile(
    r"^\s*python\s+-m\s+ethoinsight\.scripts\.\w+\.\w+(\s|$)"
)

# Match safe file-operation commands at start of command.
_ALLOWED_FILE_OPS = re.compile(
    r"^\s*(mkdir|cp|mv|ls|cat|grep|head|tail)(\s|$)"
)


_DENY_MESSAGE = (
    "该 bash 命令不是脚本调用。code-executor 仅可：\n"
    "  1. 调脚本：python -m ethoinsight.scripts.<paradigm>.<name> --input ... --output ...\n"
    "  2. 文件操作：mkdir / cp / mv / ls / cat / grep / head / tail\n"
    "请改用脚本调用形式。可用脚本清单见 by-paradigm/<范式>.md。"
)


class ScriptInvocationOnlyProvider:
    """Whitelist bash commands for code-executor to script invocations + file ops."""

    name = "script_invocation_only"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Only gate bash tool calls.
        if request.tool_name != "bash":
            return GuardrailDecision(allow=True)

        # Only gate code-executor subagent.
        if "code-executor" not in (request.agent_id or ""):
            return GuardrailDecision(allow=True)

        cmd = request.tool_input.get("command", "")

        if _ALLOWED_PYTHON_PATTERN.match(cmd):
            return GuardrailDecision(allow=True)
        if _ALLOWED_FILE_OPS.match(cmd):
            return GuardrailDecision(allow=True)

        return GuardrailDecision(
            allow=False,
            reasons=[
                GuardrailReason(
                    code="script_invocation_only.not_a_script_call",
                    message=_DENY_MESSAGE,
                )
            ],
            policy_id="script_invocation_only",
        )

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Pure sync logic; expose async for protocol compliance.
        return self.evaluate(request)
