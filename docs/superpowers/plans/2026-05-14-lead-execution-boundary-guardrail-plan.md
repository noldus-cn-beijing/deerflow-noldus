# Lead Agent Execution Boundary Guardrail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 spec §5.5.1 中定义的 `LeadAgentExecutionBoundaryProvider`，从机制层（Guardrail）禁止 lead agent 调用 `write_file` 写可执行脚本 (`.py / .sh / .ipynb / .bash / .zsh`) 或调用 `bash` 跑非白名单命令。直接修复 thread `b0d3a611-071e-41a5-a952-36c3772c167f` E2E 测试暴露的根因 A。

**Architecture:** 仿照 `deerflow/guardrails/script_invocation_only_provider.py` 的模板，新写一个 `LeadAgentExecutionBoundaryProvider`（实现 deerflow 的 `GuardrailProvider` Protocol）。Provider 注册位置：`packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py`。Wire 到 `lead_agent/agent.py:314` 已有的 `Ev19TemplateGuardrailProvider` 后面，构成 lead 中间件链上的第二个 `GuardrailMiddleware`。通过 `passport=None`（lead 调用时不传 passport）与下游 subagent (`passport="subagent:..."`) 区分；provider 内部用 `agent_id` 自门控，只对 lead 生效，subagent 透传。

**Tech Stack:** Python 3.12+, pytest, pytest-asyncio, ruff (line length 240), langchain.agents middleware framework, deerflow `GuardrailProvider` Protocol。

**Context links（实现时必读）：**
- spec：`docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md` §5.5.1（line 619-712 附近）
- 诊断材料：`docs/problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md`
- 模板 Provider：`packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py`
- 模板单测：`packages/agent/backend/tests/test_script_invocation_only_provider.py`
- Guardrail 接口契约：`packages/agent/backend/packages/harness/deerflow/guardrails/provider.py`
- Wire 位置（已有 Ev19 provider 注册示例）：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py:295-315`
- 项目约束（必读）：`CLAUDE.md`（特别是第 12 条"复用 deerflow 现成功能优先于自造轮子"、TDD 强制、Git 规范用中文 commit message）

**Scope（明确不做）：**
- 不实现 `HandoffPendingActionsProvider`（spec §5.5.2）—— 它依赖阶段 1 L1 schema 的 `pending_actions[]` 字段落地，单独写后续 plan
- 不改 `lead_agent/prompt.py`（commit `24715250` 已写入的 3 条 prompt 约束保留作为双重保险）
- 不改 sandbox/tools.py（根因 B + 触发点在本 plan 落地后不可触发，spec §10.3 论证）
- 不改 spec 文件

**前置假设（执行前用 `git log -1` 验证）：**
- 当前在 `dev` 分支，工作目录 `/home/wangqiuyang/noldus-insight/`
- spec draft v2 已合入主 spec 文件（含 §5.5.1）—— 应能 `grep "LeadAgentExecutionBoundaryProvider" docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md` 找到
- dogfood-fix 的 11 个 commit 已在本地（`git log --oneline origin/dev..dev | wc -l` 应 ≥ 11）
- backend tests 在本基线上全绿（执行前先跑一次 `cd packages/agent/backend && make test` 确认）

---

## File Structure（决策已锁定，照此实施）

**新建 1 个文件**（生产代码）：
- `packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py`（~85 行）— Provider 实现，单一职责：lead agent 的 write_file/bash 白名单门控

**新建 1 个文件**（测试代码）：
- `packages/agent/backend/tests/test_lead_execution_boundary_provider.py`（~210 行）— 覆盖：write_file 扩展名 deny、bash 白名单 allow/deny、subagent 透传、其他工具透传、deny reason 内容、async/sync 一致、agent_id=None 处理

**修改 1 个文件**（wire 到中间件链）：
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` 在 line 314 之后追加 ~10 行：注册新 provider 到 lead 的中间件链。注意：`GuardrailMiddleware` 的 `name` property 会自动以 provider.name 去重（`GuardrailMiddleware[lead_execution_boundary]` vs `GuardrailMiddleware[ev19-template-guardrail]`），无需手动指定 name。

**修改 0 个文件**（subagent executor）：
- subagent executor (`subagents/executor.py`) **不动**。Provider 自门控逻辑（`if agent_id starts with "subagent:" → pass through`）确保即使未来 subagent 链上加 lead provider 也安全；但本 plan 不在 subagent 链上加，所以 subagent 路径 0 改动。

---

### Task 1: Provider 单元测试（先写测试，TDD）

**Files:**
- Create: `packages/agent/backend/tests/test_lead_execution_boundary_provider.py`

**为什么先写测试**：TDD 是项目硬规约（CLAUDE.md "TDD 强制"）。这步会失败（provider 未实现），下一 task 才让它通过。

- [ ] **Step 1: 写完整测试文件**

完整文件内容：

```python
"""Tests for LeadAgentExecutionBoundaryProvider.

Modeled after test_script_invocation_only_provider.py.

Coverage:
- write_file with executable extension (.py/.sh/.ipynb/.bash/.zsh) → deny
- write_file with data extension (.md/.json/.csv/.txt) → allow
- bash whitelist: ethoinsight.parse / ethoinsight.catalog / safe file ops
- bash deny: python -c, python /path/to/file.py, pip install, arbitrary commands
- subagent passport (agent_id starts with "subagent:") → always allow (passthrough)
- non-bash/non-write_file tools (read_file, ls, task, ask_clarification) → always allow
- deny reason code stability + message helpfulness
- sync evaluate() and async aevaluate() agree
"""

from __future__ import annotations

import pytest

from deerflow.guardrails.provider import GuardrailRequest


@pytest.fixture
def provider():
    from deerflow.guardrails.lead_execution_boundary_provider import (
        LeadAgentExecutionBoundaryProvider,
    )
    return LeadAgentExecutionBoundaryProvider()


def _req(tool_name: str, *, path: str = "", command: str = "", agent_id: str | None = None) -> GuardrailRequest:
    """Build a GuardrailRequest. agent_id=None simulates lead; "subagent:foo" simulates subagent."""
    tool_input: dict = {}
    if path:
        tool_input["path"] = path
    if command:
        tool_input["command"] = command
    return GuardrailRequest(tool_name=tool_name, tool_input=tool_input, agent_id=agent_id)


class TestSubagentPassportPassthrough:
    """Subagents (agent_id starting with 'subagent:') are never gated by this provider."""

    def test_subagent_write_py_allowed(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/x.py", agent_id="subagent:code-executor"))
        assert decision.allow

    def test_subagent_bash_python_c_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="python -c 'import x'", agent_id="subagent:data-analyst"))
        assert decision.allow

    def test_subagent_bash_arbitrary_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="curl https://example.com", agent_id="subagent:report-writer"))
        assert decision.allow


class TestNonGatedToolsAlwaysAllowed:
    """For lead (agent_id=None), tools other than write_file/bash always pass."""

    def test_read_file_allowed(self, provider):
        decision = provider.evaluate(_req("read_file"))
        assert decision.allow

    def test_ls_allowed(self, provider):
        decision = provider.evaluate(_req("ls"))
        assert decision.allow

    def test_glob_allowed(self, provider):
        decision = provider.evaluate(_req("glob"))
        assert decision.allow

    def test_grep_allowed(self, provider):
        decision = provider.evaluate(_req("grep"))
        assert decision.allow

    def test_task_allowed(self, provider):
        decision = provider.evaluate(_req("task"))
        assert decision.allow

    def test_ask_clarification_allowed(self, provider):
        decision = provider.evaluate(_req("ask_clarification"))
        assert decision.allow

    def test_present_files_allowed(self, provider):
        decision = provider.evaluate(_req("present_files"))
        assert decision.allow

    def test_str_replace_allowed(self, provider):
        # str_replace is for editing existing files; not gated here
        decision = provider.evaluate(_req("str_replace"))
        assert decision.allow


class TestWriteFileForbiddenExtensions:
    """Lead writing executable script files is denied."""

    def test_write_py_denied(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/gen_charts.py"))
        assert not decision.allow
        assert decision.reasons
        assert decision.reasons[0].code == "lead_execution_boundary.script_write_forbidden"

    def test_write_sh_denied(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/run.sh"))
        assert not decision.allow

    def test_write_ipynb_denied(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/analysis.ipynb"))
        assert not decision.allow

    def test_write_bash_denied(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/script.bash"))
        assert not decision.allow

    def test_write_zsh_denied(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/script.zsh"))
        assert not decision.allow

    def test_write_py_case_insensitive(self, provider):
        # Tolerate uppercase extension (.PY)
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/X.PY"))
        assert not decision.allow


class TestWriteFileAllowedExtensions:
    """Lead writing data/doc files is allowed."""

    def test_write_md_allowed(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/notes.md"))
        assert decision.allow

    def test_write_json_allowed(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/metric_plan.json"))
        assert decision.allow

    def test_write_csv_allowed(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/data.csv"))
        assert decision.allow

    def test_write_txt_allowed(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/log.txt"))
        assert decision.allow

    def test_write_no_extension_allowed(self, provider):
        # Files without extension are typically data/config — allow
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/README"))
        assert decision.allow

    def test_write_empty_path_allowed(self, provider):
        # Defensive: if path missing in tool_input, don't crash
        decision = provider.evaluate(_req("write_file"))
        assert decision.allow


class TestBashAllowList:
    """Lead bash whitelist: ethoinsight.parse / ethoinsight.catalog / safe file ops."""

    def test_dump_headers_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.parse.dump_headers --input /mnt/user-data/uploads/x.txt --output /mnt/user-data/workspace/columns.json",
        ))
        assert decision.allow

    def test_catalog_resolve_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.catalog.resolve --paradigm epm --columns-file /mnt/user-data/workspace/columns.json --output /mnt/user-data/workspace/metric_plan.json",
        ))
        assert decision.allow

    def test_mkdir_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="mkdir -p /mnt/user-data/workspace/outputs"))
        assert decision.allow

    def test_cp_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="cp /mnt/user-data/uploads/a.txt /mnt/user-data/workspace/a.txt"))
        assert decision.allow

    def test_mv_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="mv /tmp/a /tmp/b"))
        assert decision.allow

    def test_ls_bash_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="ls /mnt/user-data/uploads/"))
        assert decision.allow

    def test_cat_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="cat /mnt/user-data/workspace/metric_plan.json"))
        assert decision.allow

    def test_grep_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="grep -r 'open_arm' /mnt/user-data/workspace/"))
        assert decision.allow

    def test_head_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="head -10 /mnt/user-data/uploads/data.txt"))
        assert decision.allow

    def test_tail_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="tail -10 /mnt/user-data/workspace/log.txt"))
        assert decision.allow

    def test_leading_whitespace_allowed(self, provider):
        # Tolerate leading whitespace (lead sometimes emits indented commands)
        decision = provider.evaluate(_req("bash", command="  python -m ethoinsight.parse.dump_headers"))
        assert decision.allow


class TestBashDenyList:
    """Lead bash deny: arbitrary scripts, python -c, pip, etc."""

    def test_python_c_denied(self, provider):
        decision = provider.evaluate(_req("bash", command='python -c "print(1)"'))
        assert not decision.allow
        assert decision.reasons
        assert decision.reasons[0].code == "lead_execution_boundary.bash_not_allowed"

    def test_python_run_script_denied(self, provider):
        # Lead's exact failure mode in thread b0d3a611
        decision = provider.evaluate(_req("bash", command="python3 /mnt/user-data/workspace/gen_charts.py"))
        assert not decision.allow

    def test_python_heredoc_denied(self, provider):
        # Heredoc form (also lead's failure mode)
        decision = provider.evaluate(_req("bash", command="python3 << 'PYEOF'\nimport pandas\nPYEOF"))
        assert not decision.allow

    def test_python_dash_m_other_module_denied(self, provider):
        # Only ethoinsight.parse and ethoinsight.catalog are allowed
        decision = provider.evaluate(_req("bash", command="python -m ethoinsight.scripts.epm.compute_open_arm_time_ratio"))
        assert not decision.allow

    def test_python_dash_m_pip_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="python -m pip install pandas"))
        assert not decision.allow

    def test_pip_install_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="pip install pandas"))
        assert not decision.allow

    def test_rm_denied(self, provider):
        # rm is destructive, not in safe-op whitelist
        decision = provider.evaluate(_req("bash", command="rm -rf /mnt/user-data/workspace/"))
        assert not decision.allow

    def test_curl_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="curl https://example.com"))
        assert not decision.allow

    def test_git_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="git status"))
        assert not decision.allow

    def test_empty_command_denied(self, provider):
        # Defensive: empty bash command should not be allowed silently
        decision = provider.evaluate(_req("bash", command=""))
        assert not decision.allow


class TestDenyReasonContent:
    """Deny reasons should guide the agent to the correct path."""

    def test_write_file_reason_mentions_redispatch_path(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/x.py"))
        assert not decision.allow
        msg = decision.reasons[0].message
        assert "metric_plan.json" in msg or "code-executor" in msg
        assert "ask_clarification" in msg

    def test_bash_reason_mentions_allowed_modules(self, provider):
        decision = provider.evaluate(_req("bash", command="python /tmp/x.py"))
        assert not decision.allow
        msg = decision.reasons[0].message
        assert "ethoinsight.parse" in msg
        assert "ethoinsight.catalog" in msg


class TestPolicyId:
    """policy_id is stable for log aggregation."""

    def test_write_file_policy_id(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/x.py"))
        assert decision.policy_id == "lead_execution_boundary"

    def test_bash_policy_id(self, provider):
        decision = provider.evaluate(_req("bash", command="curl x"))
        assert decision.policy_id == "lead_execution_boundary"


@pytest.mark.asyncio
async def test_aevaluate_matches_evaluate_on_deny(provider):
    req = _req("write_file", path="/mnt/user-data/workspace/x.py")
    sync_decision = provider.evaluate(req)
    async_decision = await provider.aevaluate(req)
    assert sync_decision.allow == async_decision.allow
    assert sync_decision.reasons[0].code == async_decision.reasons[0].code
    assert sync_decision.policy_id == async_decision.policy_id


@pytest.mark.asyncio
async def test_aevaluate_matches_evaluate_on_allow(provider):
    req = _req("write_file", path="/mnt/user-data/workspace/x.md")
    sync_decision = provider.evaluate(req)
    async_decision = await provider.aevaluate(req)
    assert sync_decision.allow == async_decision.allow
```

- [ ] **Step 2: 运行测试，确认全部失败（ImportError）**

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_lead_execution_boundary_provider.py -v
```

Expected: 所有测试 `collection error` 或 `ImportError`（`No module named 'deerflow.guardrails.lead_execution_boundary_provider'`）。这是预期失败——证明测试存在但实现还没写。

- [ ] **Step 3: Commit 测试文件**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/tests/test_lead_execution_boundary_provider.py
git commit -m "test(guardrails): 加 LeadAgentExecutionBoundaryProvider 单测（先红后绿，TDD）

照 test_script_invocation_only_provider.py 模板写。覆盖：
- subagent passport 透传
- write_file 可执行扩展名 deny / 数据扩展名 allow
- bash 白名单（ethoinsight.parse/catalog + safe ops）vs deny（python -c / 任意脚本 / pip / curl）
- deny reason 内容引导
- async/sync 一致
"
```

---

### Task 2: 实现 LeadAgentExecutionBoundaryProvider

**Files:**
- Create: `packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py`
- Test: `packages/agent/backend/tests/test_lead_execution_boundary_provider.py`（Task 1 已写）

- [ ] **Step 1: 写 Provider 实现**

完整文件内容：

```python
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
```

- [ ] **Step 2: 运行 provider 单测，确认全绿**

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_lead_execution_boundary_provider.py -v
```

Expected: 全部测试通过（约 35 条）。如有失败：**不要修测试**，先回头看 provider 实现是不是有 bug——测试是契约的固化。

- [ ] **Step 3: 跑 ruff lint，确认格式合规**

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make lint
```

Expected: 无错误。如果新文件有 ruff 警告，按 ruff 提示修；line length 限制 240 字符（CLAUDE.md "Python > ruff 格式化"）。

- [ ] **Step 4: Commit provider 实现**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py
git commit -m "feat(guardrails): 加 LeadAgentExecutionBoundaryProvider（spec §5.5.1）

机制层禁止 lead agent 越权执行：
- write_file 写 .py/.sh/.ipynb/.bash/.zsh → deny
- bash 限白名单：python -m ethoinsight.parse|catalog 模块 + 安全文件操作
- subagent passport 透传，不影响 code-executor 现有职责

修复 thread b0d3a611 E2E 失败的根因 A（lead 自己 write_file 写脚本）。
照 deerflow.guardrails.script_invocation_only_provider 模板实现。
"
```

---

### Task 3: Wire 到 lead 中间件链

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py:314-315`

**为什么独立成 Task**：注册 wire 改动仅 ~10 行，但它**触发整个 lead agent 中间件链**——任何细节错（顺序、import path）会让所有 thread 跑挂。独立 task + 独立 commit，方便回退。

- [ ] **Step 1: 读 wire 上下文（确认插入点）**

Run:
```bash
cd /home/wangqiuyang/noldus-insight
sed -n '295,330p' packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py
```

Expected 看到：
- 第 295-299 行：ThinkTagMiddleware 注册
- 第 301-314 行：`if guardrails_cfg.enabled` 块内注册 `Ev19WorkspaceBridgeMiddleware` + `GuardrailMiddleware(provider=Ev19TemplateGuardrailProvider())`
- 第 316-318 行：`if custom_middlewares` 块
- 第 320-325 行：GateEnforcementMiddleware（manual mode）
- 第 327-328 行：ClarificationMiddleware（must be last）

**关键约束**：新 provider 加在 Ev19 provider **之后**、`custom_middlewares` extend **之前**——保持 ClarificationMiddleware 始终最后。

- [ ] **Step 2: 修改 lead_agent/agent.py，在 Ev19 provider 注册之后追加 LeadAgentExecutionBoundary 注册**

用 Edit 工具修改 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py`：

old_string（**包含 Ev19 注册结尾 + 紧接着的空行 + custom_middlewares 注释**，提供足够上下文确保唯一匹配）：
```python
        provider = Ev19TemplateGuardrailProvider()
        middlewares.append(Ev19WorkspaceBridgeMiddleware())
        middlewares.append(GuardrailMiddleware(provider=provider, fail_closed=guardrails_cfg.fail_closed))

    # Inject custom middlewares before ClarificationMiddleware
```

new_string：
```python
        provider = Ev19TemplateGuardrailProvider()
        middlewares.append(Ev19WorkspaceBridgeMiddleware())
        middlewares.append(GuardrailMiddleware(provider=provider, fail_closed=guardrails_cfg.fail_closed))

        # LeadAgentExecutionBoundary — block lead from writing scripts or running
        # non-whitelisted bash. Self-gates by agent_id; subagents pass through.
        # See: spec §5.5.1, fix thread b0d3a611 E2E failure root cause A.
        from deerflow.guardrails.lead_execution_boundary_provider import (
            LeadAgentExecutionBoundaryProvider,
        )

        middlewares.append(GuardrailMiddleware(
            provider=LeadAgentExecutionBoundaryProvider(),
            fail_closed=guardrails_cfg.fail_closed,
        ))

    # Inject custom middlewares before ClarificationMiddleware
```

**注意事项**：
- import 写在 `if guardrails_cfg.enabled:` 块**内**——保持和 Ev19 一致的 lazy import 风格
- `fail_closed=guardrails_cfg.fail_closed` 复用现有配置，不引入新 config 字段
- 不传 `name=` 参数——`GuardrailMiddleware` 会自动用 `f"GuardrailMiddleware[{provider.name}]"` 即 `"GuardrailMiddleware[lead_execution_boundary]"` 去重，已经够独特

- [ ] **Step 3: 跑 backend 全部单测，确认无回归**

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make test
```

Expected: 全绿。如果有任何回归（特别是 `test_create_deerflow_agent.py` / `test_lead_agent_*` / `test_*_middleware.py`），先**仔细读失败信息**——可能是：
- 中间件链顺序假设的测试，需要更新（这种是合理改动）
- 中间件总数计数的测试，需要 +1（这种是合理改动）
- Guardrail 配置 disabled 时的覆盖，**不要**改 provider 让它无条件注册——保持"只在 guardrails_cfg.enabled 时注册"

- [ ] **Step 4: 跑 ruff lint**

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make lint
```

Expected: 无错误。

- [ ] **Step 5: Commit wire 改动**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py
git commit -m "feat(lead_agent): wire LeadAgentExecutionBoundaryProvider 到中间件链

加在 Ev19TemplateGuardrailProvider 之后、custom_middlewares 之前，
保持 ClarificationMiddleware 始终最后。复用 guardrails_cfg.fail_closed
配置。GuardrailMiddleware.name 自动去重（按 provider.name）。
"
```

---

### Task 4: 集成测试 + dogfood 复现验证

**Files:**
- Modify: `packages/agent/backend/tests/test_create_deerflow_agent.py`（可能需要更新中间件计数 / 顺序断言，按 Task 3 Step 3 跑出的结果决定）

**目的**：验证新 provider 在真实 lead agent 创建路径上生效，且能阻止 thread b0d3a611 同款故障。

- [ ] **Step 1: 检查 test_create_deerflow_agent.py 是否需要更新**

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_create_deerflow_agent.py -v 2>&1 | tail -30
```

如果有失败：
- 失败信息含 `GuardrailMiddleware not in mw_types` 或类似——读测试代码，按 provider.name 判断要不要更新（如果它在断言中间件链组成的具体名字，要加上 `GuardrailMiddleware[lead_execution_boundary]`）
- 失败信息含中间件计数 mismatch——按 +1 更新
- 如果没失败，跳过 Step 2

如果全绿，**跳到 Step 3**。

- [ ] **Step 2: 如需更新测试断言，最小化更新**

用 Edit 工具，按 Step 1 的失败信息精确更新断言。**只更新与 LeadAgentExecutionBoundaryProvider 相关的断言，不动其他**。
更新后再跑一次 Step 1 的命令，确认全绿。

- [ ] **Step 3: 跑端到端验证（半自动）**

启动开发服务：

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop  # 清理旧进程（如果有）
make dev
```

等 ~30 秒服务起来；用 `tail -f packages/agent/logs/langgraph.log | grep -E "lead_execution_boundary|guardrails|Error"` 看启动日志没有 import 报错（应能看到中间件链初始化、不应有 `Failed to load` / `ImportError`）。

- [ ] **Step 4: 浏览器 dogfood 复现 thread b0d3a611 路径**

1. 打开 `http://localhost:2026`
2. 新建 thread
3. 上传 `/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点/` 任意一份 EPM 单只数据
4. 跟流程走（反问 → 单只描述 → 是否要洞察 → 选"需要"）
5. 完成单只指标分析后，提示用户输入「**需要！补充轨迹图和汇总表格图表**」（与 thread b0d3a611 同款触发点）
6. 观察 UI + langgraph.log：
   - **预期**：lead 不再 `write_file gen_charts.py`；要么调 `ask_clarification` 问用户、要么尝试 `task(code-executor)` 重派
   - **如果**仍然看到 lead `write_file` 写 `.py`：**plan 失败**，回头看 wire 是否真的接上
   - **如果**看到 `Guardrail denied: tool 'write_file' was blocked (lead_execution_boundary.script_write_forbidden)` ToolMessage：**plan 成功，guardrail 已机制层阻断**

- [ ] **Step 5: 截图 + 记录验证结果**

在 `docs/handoffs/2026-05/` 下新建（或追加到现有验证 handoff）：

```bash
cd /home/wangqiuyang/noldus-insight
# 如果 dogfood-followup-handoff 还在，追加；否则新写
ls docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md 2>/dev/null
```

在该文件追加段落（手动编辑）：

```markdown
## 阶段 1.5 LeadAgentExecutionBoundaryProvider 验证（YYYY-MM-DD）

复现 thread b0d3a611 路径：
- 步骤 1-5：[上传 / 反问 / 描述 / 洞察]
- 步骤 6（补充图表请求）：lead 行为 = [ask_clarification / 重派 code-executor / 被 guardrail block 后转向]

grep 验证：
```
grep "lead_execution_boundary" packages/agent/logs/langgraph.log | tail -5
```

结果：[贴关键日志行]

判定：[plan 成功 / 失败 + 理由]
```

- [ ] **Step 6: 停服务、commit 验证文档**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop
```

```bash
cd /home/wangqiuyang/noldus-insight
git add docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md
git add packages/agent/backend/tests/test_create_deerflow_agent.py  # 仅当 Step 2 改了它
git commit -m "test(guardrails): 验证 LeadAgentExecutionBoundaryProvider 端到端阻断 thread b0d3a611 同款故障

dogfood 复现：在 EPM 单只分析后用户问'补充轨迹图'，lead 不再 write_file 写 .py，
而是被 guardrail 在机制层 deny + 回到 ask_clarification / 重派 code-executor 路径。
"
```

如 Step 2 没改 test_create_deerflow_agent.py 就**不要** add 它。

---

### Task 5: 收尾验证

- [ ] **Step 1: 整体单测全绿**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make test
```

Expected: 全绿。包括：
- `test_lead_execution_boundary_provider.py`（新加，~35 条）
- `test_script_invocation_only_provider.py`（既有，无变化）
- `test_handoff_isolation_provider.py`（既有，无变化）
- `test_ev19_template_guardrail_provider.py`（既有，无变化）
- `test_guardrail_middleware.py`（既有，无变化）
- `test_create_deerflow_agent.py`（既有 ± Task 4 Step 2 改动）

- [ ] **Step 2: ethoinsight 库单测全绿（防回归）**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/
```

Expected: 全绿。不应受本 plan 影响——它只动 agent backend。

- [ ] **Step 3: 全 commit 列表自查**

Run:
```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline origin/dev..dev | head -20
```

Expected 看到本 plan 增加的 4 个 commit（按时间序）：
1. `test(guardrails): 加 LeadAgentExecutionBoundaryProvider 单测...`
2. `feat(guardrails): 加 LeadAgentExecutionBoundaryProvider...`
3. `feat(lead_agent): wire LeadAgentExecutionBoundaryProvider 到中间件链`
4. `test(guardrails): 验证 LeadAgentExecutionBoundaryProvider 端到端阻断...`

加上 dogfood-fix 之前已有的 11 个 commit + 可能的 spec 修订 commit。

- [ ] **Step 4: 完成清单**

把本 plan 文件顶部加一行 frontmatter（或在末尾追加）：

```markdown
## 实施完成

- 完成日期：YYYY-MM-DD
- 执行 agent：[名字 / commit author]
- 完成的 commit hash：
  - test commit: <hash>
  - provider commit: <hash>
  - wire commit: <hash>
  - dogfood validation commit: <hash>
- dogfood 验证结果：[plan 成功 / 失败 + 一句话总结]
- 已知遗留：[如有，例如某测试用例因 fixture 限制 skip]
```

不需要 commit 此 plan 文件本身的更新（用户决定何时 commit plan 文档变更）。

- [ ] **Step 5: 回到用户**

向用户回报：
- 4 个 commit hash
- dogfood 验证结论
- 全测试状态
- 任何遗留问题（如有）

---

## 不要做的事（防止越权）

- ❌ **不要 push 到 origin** —— 用户没明确授权，11+ 个本地 commit 仍未 push 是已知状态
- ❌ **不要改 spec 文件** —— 本 plan 落地 §5.5.1，spec 已 draft v2 含此段
- ❌ **不要改 prompt 文件** —— `lead_agent/prompt.py` 现有约束保留作为双重保险
- ❌ **不要改 sandbox/tools.py** —— spec §10.3 论证不修
- ❌ **不要实现 HandoffPendingActionsProvider** —— 它依赖阶段 1 L1 schema，本 plan 范围外
- ❌ **不要碰这 3 个无关文件**：`docs/specs/llm-finetuning-strategy.md` / `docs/plans/2026-05-13-base-model-decision-memo.md` / `packages/agent/frontend/src/app/page.tsx`
- ❌ **不要用 --no-verify 跳过 pre-commit hook**
- ❌ **不要 git push --force** 任何分支
- ❌ **不要做 git rebase -i** 整理 commit（每个 commit 直接生效即可）

---

## 实施完成后的状态

- 新增 2 个文件（1 个 provider 实现 + 1 个测试），改动 1 个文件（lead agent middleware wire）
- 可选改动 1 个测试文件（test_create_deerflow_agent.py，仅当中间件链测试断言要更新时）
- 4 个 commit（local only，未 push）
- thread b0d3a611 同款故障在机制层不可触发
- backend 单测、ethoinsight 单测全绿
- dogfood 端到端验证记录在 handoff 文档中

如出现意料外的回归或失败，**停下来**联系用户而不是自己改 scope。
