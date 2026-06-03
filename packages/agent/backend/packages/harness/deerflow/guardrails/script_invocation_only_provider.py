"""ScriptInvocationOnlyProvider — gate code-executor + chart-maker bash to script invocations + file ops only.

This Guardrail enforces the 'script-per-metric' architecture: code-executor's and
chart-maker's bash tool must only be used to either invoke an `ethoinsight.scripts.*`
script via ``python -m`` or perform safe file operations. Write file-ops (mkdir/cp/mv)
are further validated to ensure the target path is within the sandbox workspace
(/mnt/user-data/) — not into .venv, site-packages, /mnt/skills, or other protected paths.

Any other bash command (including ``python -c``, ``pip install``, arbitrary scripts)
is denied with a reason that tells the agent the correct path forward.

Additionally, all 4 handoff-writing subagents (code-executor, data-analyst,
chart-maker, report-writer) are blocked from using ``write_file`` to write
handoff_*.json — they must use the corresponding seal_*_handoff first-party tool.

White-list rather than black-list: the allowed shape is small and stable; new
scripts are auto-allowed by the same pattern without touching this provider.

Applies to subagents whose agent_id contains 'code-executor' or 'chart-maker'
(for bash gating) or 'subagent:' + handoff subagent name (for write_file gating).
Lead agent and other subagents pass through unchanged.

See: docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md §4
"""

from __future__ import annotations

import re
import shlex

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

# Match read-only file-operation commands at start of command (no path validation needed).
_READ_ONLY_FILE_OPS = re.compile(
    r"^\s*(ls|cat|grep|head|tail)(\s|$)"
)

# Match write file-operation commands at start of command (requires path validation).
_WRITE_FILE_OPS = re.compile(
    r"^\s*(mkdir|cp|mv)(\s|$)"
)

# Combined pattern for quick allow/deny check before path validation.
_ALLOWED_FILE_OPS = re.compile(
    r"^\s*(mkdir|cp|mv|ls|cat|grep|head|tail)(\s|$)"
)

_DENY_MESSAGE = (
    "该 bash 命令不是脚本调用。code-executor / chart-maker 仅可：\n"
    "  1. 调脚本：python -m ethoinsight.scripts.<paradigm>.<name> --input ... --output ...\n"
    "  2. 文件操作：mkdir / cp / mv / ls / cat / grep / head / tail\n"
    "请改用脚本调用形式。可用脚本清单见 by-paradigm/<范式>.md。"
)

_FILE_OP_PATH_DENY_MESSAGE = (
    "文件写入操作的目标路径必须在沙箱工作区内（/mnt/user-data/）。"
    "脚本由 ethoinsight 库维护，不可写入已安装包路径（.venv/site-packages）"
    "或沙箱外路径（/mnt/skills 为只读）。"
    "缺脚本请在 handoff 标 status=failed 让 lead 处理。"
)

_HANDOFF_SUBAGENT_NAMES = {"code-executor", "data-analyst", "chart-maker", "report-writer"}

_HANDOFF_WRITE_FILE_DENY = (
    "严禁用 write_file 写 handoff JSON 文件。"
    "请改用对应的 first-party tool: "
    "seal_code_executor_handoff / seal_data_analyst_handoff / "
    "seal_chart_maker_handoff / seal_report_writer_handoff，"
    "按结构化参数调用。"
)

# Subagent types whose bash is gated (code-executor + chart-maker).
_BASH_GATED_AGENTS = {"code-executor", "chart-maker"}


def _extract_target_path(cmd: str) -> str | None:
    """Extract the target/destination path from mkdir/cp/mv commands.

    Returns None if the command cannot be reliably parsed.
    """
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return None

    if len(parts) < 2:
        return None

    op = parts[0]
    if op == "mkdir":
        # mkdir [-p] [-m mode] target — skip flags, take last non-flag arg
        args = [p for p in parts[1:] if not p.startswith("-")]
        return args[-1] if args else None
    else:
        # cp/mv source [...] target — target is always the last argument
        return parts[-1]


def _is_path_safe(target: str) -> bool:
    """Check if target path is within the sandbox workspace and not in protected dirs.

    Returns True only if the target is clearly within /mnt/user-data/ and does
    not point to .venv, site-packages, or /mnt/skills (read-only).
    """
    # Deny paths that try to write into Python package directories.
    if ".venv" in target or "site-packages" in target:
        return False

    # Absolute paths: must start with /mnt/user-data/ (the sandbox workspace).
    if target.startswith("/"):
        if target.startswith("/mnt/skills"):
            return False
        if not target.startswith("/mnt/user-data/"):
            return False
        return True

    # Relative path: allow (resolves within sandbox cwd),
    # .venv/site-packages already checked above.
    return True


class ScriptInvocationOnlyProvider:
    """Whitelist bash commands for code-executor + chart-maker + block write_file for handoff JSONs."""

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

        # --- Gate 2: bash command whitelisting (code-executor + chart-maker) ---
        if request.tool_name != "bash":
            return GuardrailDecision(allow=True)

        # Only gate code-executor and chart-maker subagents.
        if not any(name in agent_id for name in _BASH_GATED_AGENTS):
            return GuardrailDecision(allow=True)

        cmd = request.tool_input.get("command", "")

        # Allow python -m ethoinsight.scripts.* invocations.
        if _ALLOWED_PYTHON_PATTERN.match(cmd):
            return GuardrailDecision(allow=True)

        # Allow file operations; write ops get additional path validation.
        if _ALLOWED_FILE_OPS.match(cmd):
            # Read-only file ops: always allowed.
            if _READ_ONLY_FILE_OPS.match(cmd):
                return GuardrailDecision(allow=True)

            # Write file ops (mkdir/cp/mv): validate target path.
            target = _extract_target_path(cmd)
            if target is None:
                # Cannot parse command — deny (strict).
                return GuardrailDecision(
                    allow=False,
                    reasons=[
                        GuardrailReason(
                            code="script_invocation_only.unsafe_file_op_path",
                            message=f"无法解析该命令的目标路径，为确保安全已拒绝: {cmd[:120]}\n{_FILE_OP_PATH_DENY_MESSAGE}",
                        )
                    ],
                    policy_id="script_invocation_only",
                )

            if not _is_path_safe(target):
                return GuardrailDecision(
                    allow=False,
                    reasons=[
                        GuardrailReason(
                            code="script_invocation_only.unsafe_file_op_path",
                            message=f"文件写入目标路径不在沙箱工作区内: {target}\n{_FILE_OP_PATH_DENY_MESSAGE}",
                        )
                    ],
                    policy_id="script_invocation_only",
                )

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
