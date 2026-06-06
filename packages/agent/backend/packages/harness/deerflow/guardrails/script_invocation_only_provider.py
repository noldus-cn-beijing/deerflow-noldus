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

import os
import re
import shlex

from deerflow.guardrails.provider import (
    GuardrailDecision,
    GuardrailReason,
    GuardrailRequest,
)

# Allowed `python -m ethoinsight.*` invocations at the start of the command.
# Supports leading whitespace and any args after the module name.
#
# Three shapes are whitelisted (white-list stays small + stable):
#   1. ethoinsight.scripts.<paradigm>.<script>  — per-metric compute / plot scripts
#   2. ethoinsight.catalog.resolve              — chart-maker self-runs `--mode charts`
#      to produce plan_charts.json (execution-conventions.md §CLI 例外; chart-maker
#      SKILL.md step 2). prep_metric_plan runs `--mode metrics` internally as a tool,
#      so only the charts path reaches this guardrail via chart-maker bash.
#   3. ethoinsight.parse.dump_headers           — produces columns.json, which
#      catalog.resolve consumes via its REQUIRED --columns-file arg.
# Whitelisting resolve/dump_headers is safe: both only read inputs + write JSON into
# the workspace; they execute no arbitrary code. Path safety for their args is the
# same as for scripts (the plotting scripts themselves still re-validate paths).
_ALLOWED_PYTHON_PATTERN = re.compile(
    r"^\s*python\s+-m\s+ethoinsight\.(scripts\.\w+\.\w+|catalog\.resolve|parse\.dump_headers)(\s|$)"
)

# Allow parallel script execution via bash -c wrapper.
# Matches: bash -c "python -m ethoinsight.scripts.epm.compute_... --input ... & ... wait"
# Also allows optional cd prefix: cd /mnt/user-data/workspace && bash -c "..."
# The inner content between quotes is validated separately (see _validate_parallel_bash).
_PARALLEL_BASH_PATTERN = re.compile(
    r"^\s*(?:cd\s+\S+\s*&&\s*)?bash\s+-c\s+([\"'])(.+?)\1\s*$",
    re.DOTALL,
)

# Inside a bash -c block: allowed tokens between script invocations.
# Each script call is python -m ethoinsight.scripts.* ... optionally followed by & or ;
# The last line must be wait or echo.
# Split on &, ;, and newlines (wait and echo are typically on their own lines).
_PARALLEL_INNER_SPLITTER = re.compile(r"\s*[&;\n]\s*")

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
    "  2. chart-maker 专用：python -m ethoinsight.parse.dump_headers --input ... --output columns.json\n"
    "     与 python -m ethoinsight.catalog.resolve --mode charts ... --output plan_charts.json\n"
    "  3. 文件操作：mkdir / cp / mv / ls / cat / grep / head / tail\n"
    "请改用以上形式之一。可用脚本清单见 by-paradigm/<范式>.md。"
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

# Sandbox workspace base used to resolve relative paths before boundary check.
_SANDBOX_WORKSPACE_BASE = "/mnt/user-data/workspace"


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

    Uses os.path.normpath to resolve ``..`` before checking boundaries,
    eliminating relative-path escape (e.g. ``../../../../usr/lib/...``).

    A sandbox base of ``/mnt/user-data/workspace`` is used to resolve
    relative paths to absolute before the boundary check.
    """
    # Resolve .. via normpath to close relative-path escape.
    normalized = os.path.normpath(target)

    # Resolve relative paths against the sandbox workspace base.
    if not normalized.startswith("/"):
        normalized = os.path.normpath(
            os.path.join(_SANDBOX_WORKSPACE_BASE, normalized)
        )

    # Deny paths that try to write into Python package directories.
    if ".venv" in normalized or "site-packages" in normalized:
        return False

    # Must be within /mnt/user-data/ (the sandbox boundary).
    if not normalized.startswith("/mnt/user-data/"):
        return False

    # Must not write into /mnt/skills (read-only).
    if normalized.startswith("/mnt/skills"):
        return False

    return True


def _validate_parallel_bash_content(inner: str) -> bool:
    """Validate the content inside a bash -c "..." block for parallel execution.

    Each "line" (split by ``&`` or ``;``) must be either:
    - A ``python -m ethoinsight.scripts.*`` invocation (with args)
    - ``wait``
    - ``echo ...`` (status message)
    - Empty/whitespace-only

    Shell metacharacters (``|``, ``>``, ``<``, ``$()``, backticks) are
    rejected because parallel script execution never needs them, and they
    are common injection vectors.
    """
    # Reject shell metacharacters that are never needed for parallel script exec
    _DANGEROUS_META = re.compile(r"[|><`$]")
    if _DANGEROUS_META.search(inner):
        return False

    segments = _PARALLEL_INNER_SPLITTER.split(inner.strip())
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # Allow wait and echo as control/status commands
        if seg == "wait":
            continue
        if seg.startswith("echo "):
            continue
        # Each non-empty segment must be a valid script invocation
        if not _ALLOWED_PYTHON_PATTERN.match(seg):
            return False
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

        # Allow parallel script execution via bash -c wrapper.
        # e.g. bash -c "python -m ethoinsight.scripts.epm.compute_... & ... wait"
        m = _PARALLEL_BASH_PATTERN.match(cmd)
        if m:
            inner = m.group(2)
            if _validate_parallel_bash_content(inner):
                return GuardrailDecision(allow=True)
            return GuardrailDecision(
                allow=False,
                reasons=[
                    GuardrailReason(
                        code="script_invocation_only.parallel_bash_invalid_content",
                        message=(
                            "bash -c 内仅允许 python -m ethoinsight.scripts.* 调用"
                            "（用 & 或 ; 分隔，最后以 wait 或 echo 结尾）。"
                            "不允许其他命令。"
                        ),
                    )
                ],
                policy_id="script_invocation_only",
            )

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
