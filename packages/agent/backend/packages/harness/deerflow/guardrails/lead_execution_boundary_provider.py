"""LeadAgentExecutionBoundaryProvider — gate the lead agent's write_file/bash
to enforce the role boundary "lead is a scheduler, not an executor".

This Guardrail enforces spec §5.5.1 (output-constitution Article 6):
- write_file: deny if path ends with executable script extensions
  (.py / .sh / .ipynb / .bash / .zsh)
- bash: whitelist to:
  1. python -m ethoinsight.parse.*   (parse EthoVision data)
  2. python -m ethoinsight.catalog.* (generate metric_plan.json)
  3. safe file ops: mkdir / cp / mv / ls / cat / grep / head / tail

The provider self-gates by agent_id: subagent calls (agent_id starts with
"subagent:") pass through unchanged. Other subagents (code-executor /
data-analyst / report-writer) have their own providers attached in
subagents/executor.py.

White-list rather than black-list: the allowed shape is small and stable;
adding new ethoinsight.parse/catalog modules auto-allowed by the same
pattern without touching this provider.

See: docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md §5.5.1
"""

from __future__ import annotations

import re

from deerflow.guardrails.provider import (
    GuardrailDecision,
    GuardrailReason,
    GuardrailRequest,
)

# Allow `python -m ethoinsight.parse.*` or `python -m ethoinsight.catalog.*`
# (with or without `python3`/leading whitespace).
_LEAD_BASH_ALLOWED = re.compile(
    r"^\s*python3?\s+-m\s+ethoinsight\.(parse|catalog)\.\w+(\s|$)"
)

# Safe file operations at start of command.
_LEAD_BASH_SAFE_OPS = re.compile(
    r"^\s*(mkdir|cp|mv|ls|cat|grep|head|tail)(\s|$)"
)

# Executable extensions lead must not write. Matched case-insensitively.
_FORBIDDEN_SCRIPT_EXTENSIONS = (".py", ".sh", ".ipynb", ".bash", ".zsh")

_WRITE_FILE_DENY_MESSAGE = (
    "lead 是调度员，不写脚本。补充分析/图表请：\n"
    "  a) 更新 metric_plan.json → 重派 code-executor\n"
    "  b) ask_clarification 问用户是否要做\n"
    "执行分析脚本是 code-executor 的工作。"
)

_BASH_DENY_MESSAGE = (
    "lead 的 bash 仅可：\n"
    "  1. python -m ethoinsight.parse.* （解析数据）\n"
    "  2. python -m ethoinsight.catalog.* （生成 plan.json）\n"
    "  3. 文件操作：mkdir / cp / mv / ls / cat / grep / head / tail\n"
    "执行分析脚本请走 task(code-executor)。"
)


class LeadAgentExecutionBoundaryProvider:
    """Gate lead agent's write_file/bash to enforce role boundary."""

    name = "lead_execution_boundary"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Subagents have their own providers; pass through.
        if request.agent_id and request.agent_id.startswith("subagent:"):
            return GuardrailDecision(allow=True)

        if request.tool_name == "write_file":
            path = (request.tool_input or {}).get("path", "") or ""
            if path.lower().endswith(_FORBIDDEN_SCRIPT_EXTENSIONS):
                return GuardrailDecision(
                    allow=False,
                    reasons=[
                        GuardrailReason(
                            code="lead_execution_boundary.script_write_forbidden",
                            message=_WRITE_FILE_DENY_MESSAGE,
                        )
                    ],
                    policy_id="lead_execution_boundary",
                )
            return GuardrailDecision(allow=True)

        if request.tool_name == "bash":
            cmd = (request.tool_input or {}).get("command", "") or ""
            if not cmd:
                # Empty bash command is never legitimate; deny so the agent
                # gets a clear error rather than silently no-op.
                return GuardrailDecision(
                    allow=False,
                    reasons=[
                        GuardrailReason(
                            code="lead_execution_boundary.bash_not_allowed",
                            message=_BASH_DENY_MESSAGE,
                        )
                    ],
                    policy_id="lead_execution_boundary",
                )
            if _LEAD_BASH_ALLOWED.match(cmd) or _LEAD_BASH_SAFE_OPS.match(cmd):
                return GuardrailDecision(allow=True)
            return GuardrailDecision(
                allow=False,
                reasons=[
                    GuardrailReason(
                        code="lead_execution_boundary.bash_not_allowed",
                        message=_BASH_DENY_MESSAGE,
                    )
                ],
                policy_id="lead_execution_boundary",
            )

        # All other tools (read_file, ls, glob, grep, task, ask_clarification,
        # present_files, str_replace, ...) pass through unconditionally.
        return GuardrailDecision(allow=True)

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Pure sync logic; expose async for protocol compliance.
        return self.evaluate(request)
