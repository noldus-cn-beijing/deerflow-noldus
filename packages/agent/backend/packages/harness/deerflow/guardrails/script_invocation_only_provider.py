"""ScriptInvocationOnlyProvider — gate code-executor's bash to script invocations + file ops only.

This Guardrail enforces the 'script-per-metric' architecture: code-executor's
bash tool must only be used to either invoke an `ethoinsight.scripts.*` script
via ``python -m`` or perform safe file operations (mkdir / cp / mv / ls / cat /
grep / head / tail). Any other bash command (including ``python -c``,
``pip install``, arbitrary scripts) is denied with a reason that tells the
agent the correct path forward.

Additionally, all 4 handoff-writing subagents (code-executor, data-analyst,
chart-maker, report-writer) are blocked from using ``write_file`` to write
handoff_*.json — they must use the corresponding seal_*_handoff first-party tool.

White-list rather than black-list: the allowed shape is small and stable; new
scripts are auto-allowed by the same pattern without touching this provider.

Only applies to subagents whose agent_id starts with 'subagent:code-executor'
(for bash gating) or 'subagent:' + handoff subagent name (for write_file gating).
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

_HANDOFF_SUBAGENT_NAMES = {"code-executor", "data-analyst", "chart-maker", "report-writer"}

_HANDOFF_WRITE_FILE_DENY = (
    "严禁用 write_file 写 handoff JSON 文件。"
    "请改用对应的 first-party tool: "
    "seal_code_executor_handoff / seal_data_analyst_handoff / "
    "seal_chart_maker_handoff / seal_report_writer_handoff，"
    "按结构化参数调用。"
)


class ScriptInvocationOnlyProvider:
    """Whitelist bash commands for code-executor + block write_file for handoff JSONs."""

    name = "script_invocation_only"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        agent_id = request.agent_id or ""

        # --- Gate 1: write_file handoff path blocking ---
        if request.tool_name == "write_file":
            if any(f"subagent:{name}" in agent_id for name in _HANDOFF_SUBAGENT_NAMES):
                path = request.tool_input.get("path", "")
                if "handoff_" in path and path.endswith(".json"):
                    return GuardrailDecision(
                        allow=False,
                        reasons=[
                            GuardrailReason(
                                code="handoff.write_file_forbidden",
                                message=f"{_HANDOFF_WRITE_FILE_DENY} 目标文件: {path}",
                            )
                        ],
                        policy_id="script_invocation_only",
                    )
            return GuardrailDecision(allow=True)

        # --- Gate 2: bash command whitelisting (code-executor only) ---
        if request.tool_name != "bash":
            return GuardrailDecision(allow=True)

        # Only gate code-executor subagent.
        if "code-executor" not in agent_id:
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
