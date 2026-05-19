# Spec 修订草稿 v1 — 基于 thread b0d3a611 E2E 失败 + deepseek co-analysis + GuardrailProvider 机制发现

> **状态**：draft，等用户 review 后才合入主 spec（`2026-05-14-handoff-protocol-and-runtime-isolation-spec.md`）
> **日期**：2026-05-14
> **触发**：thread `b0d3a611-071e-41a5-a952-36c3772c167f` E2E 测试在「补充轨迹图和汇总表格」阶段卡死（lead 自己 write_file 写脚本 → sandbox 路径不替换 → FileNotFoundError → 反复猜测 → 7 次归档卡死）
> **诊断材料**：[docs/problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md](../../problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md)
> **核心发现**：deerflow 已有完整的 `GuardrailProvider` 机制（`packages/agent/backend/packages/harness/deerflow/guardrails/`），且已有 `ScriptInvocationOnlyProvider` + `HandoffIsolationProvider` 两个生产级实现可直接借鉴
> **修订哲学**：复用 deerflow 现成机制（CLAUDE.md 第 12 条），用 **机制层防护** 替代 prompt 自觉约束

---

## 修订总览

| # | 章节 | 改动类型 | 关联根因 |
|---|---|---|---|
| 1 | §2.4 L1 schema | 补充 `pending_actions[]` 字段 | 遗漏根因 1：lead 忽视 handoff errors |
| 2 | §5 输出宪法 | 补充第 6-7 条「执行边界」+ 「pending_actions acknowledge」 | 根因 A：lead 越权写脚本 |
| 3 | §5 输出宪法 → 新增 §5.5 | 「机制层执行边界」—— 引入 2 个新 GuardrailProvider 描述 | 根因 A + 遗漏 1 |
| 4 | §7 与现有 plan/spec 关系 | 更新表格，把 24715250 commit 的 prompt 约束标记为「过渡方案，本 spec 用机制层根治」 | — |
| 5 | §8 实施路径 | 新增「阶段 1.5：执行边界 guardrail」夹在阶段 1 和阶段 2 之间 | — |
| 6 | §10 风险与未决项 | 新增「sandbox 路径不对称 (根因 B) 留长期 backlog 不修」的论证 | 根因 B、触发点 |

---

## 修订 1：§2.4 L1 schema 补 `pending_actions[]` 字段

**改动位置**：spec line 124-191 之间，在 `gate_signals` 字段定义之后

**新增内容**：

```markdown
#### 2.4.3 `pending_actions[]` 字段（根因 1 修复）

`pending_actions[]` 列 **subagent 自己处理不了、需要 lead 后续处理** 的未决事项。**与 errors 不同**——errors 是 "本次任务过程中遇到的错误"，pending_actions 是 "lead 必须显式回应的指令"。

```json
"pending_actions": [
  {
    "action_id": "supplement_trajectory_chart",
    "trigger": "user_request_out_of_plan",          // 触发原因
    "requested_by": "user_message_at_run_2",         // 谁要求的
    "description": "用户请求轨迹图，但 plan.charts 未列",
    "resolution_options": [
      "update_plan_and_redispatch_code_executor",    // 推荐路径
      "ask_clarification_with_user"                  // 备选路径
    ],
    "must_acknowledge": true                         // ← 关键：lead 必须 ack
  }
]
```

**为什么从 `errors` 字段分离**：

- thread b0d3a611 的 `handoff_code_executor.json` 把"轨迹图未在 plan"信息埋在 `errors[]`，lead 看完后**忘了**这是强制指令
- `errors[]` 在 prompt 语义上是"已完成的任务遇到的问题、可参考"，没有 "lead 必须处理" 的语气
- `pending_actions[]` 明确"待办"语义 + `must_acknowledge=true` 字段——配合 §5.5 的 `HandoffPendingActionsProvider` 实现 **机制层强制 ack**

**lead 处理 `pending_actions[]` 的规则**（写进 lead prompt）：

收到含 `pending_actions[]` 的 handoff 后，lead **下一次 task() 前必须** 二选一：

1. **更新 plan 重派**：更新 `metric_plan.json` 把未决事项加入 charts/metrics → 重派 code-executor（推荐）
2. **澄清用户**：调用 `ask_clarification` 问用户是否要做这件事 / 用什么参数（备选）

**机制保证**：`HandoffPendingActionsProvider` （§5.5）会 block `task()` 调用直到 lead 用上面两种方式之一 acknowledge。
```

**理由**：
- 把 "lead 必须处理" 的语义从隐含（埋在 errors）变成显式（独立字段 + `must_acknowledge`）
- 与机制层 `HandoffPendingActionsProvider`（§5.5）配合，从 prompt 自觉升级到机制强制
- L1 体量影响：单个 pending_action ~100 token，单次 handoff 一般 0-2 个，可忽略

---

## 修订 2：§5 输出宪法 草案补充第 6-7 条

**改动位置**：spec line 537 「## 6-10：未来扩展」之前

**新增内容**：

```markdown
## 6. 执行边界（角色不可越权）

**lead 是调度员，不是执行员**。即使用户提出"补充图表/补充分析/换个参数算"等指标外请求，lead **不能**：

1. **不能 `write_file` 写可执行脚本** — 包括 `.py` / `.sh` / `.ipynb` / `.bash` / `.zsh` 扩展名的文件
2. **不能用 `bash` 跑分析脚本** — bash 仅限：
   - `python -m ethoinsight.parse.*`（解析 EthoVision 文件）
   - `python -m ethoinsight.catalog.*`（生成 metric_plan.json）
   - 安全文件操作：`mkdir / cp / mv / ls / cat / grep / head / tail`
3. **不能 `python -c "..."` heredoc 写分析逻辑** — 同上禁令包含这种内联写法

**当用户提出指标外请求时，lead 必须二选一**：

a) **更新 plan 重派 code-executor**：把请求映射到 metric_plan.json 的 `charts` 或新 metric → 重派
b) **澄清用户**：用 `ask_clarification` 问"该请求需要：i) 走标准 plan 重新计算 / ii) 暂不做 / iii) 走 ad-hoc 路径（未来工作）"

**为什么是机制层禁止、不只是 prompt 约束**：

thread b0d3a611 暴露了 prompt 自觉的脆弱性——commit `24715250` 已经在 lead prompt 加了 3 条角色边界禁令，但 lead 仍然在 msg 52 的 thinking 里说 "I can generate them with a simple Python script. Let me do this directly rather than re-running the full code-executor pipeline"。**LLM 在压力下会自作主张绕开 prompt 禁令**。

§5.5 节定义机制层 GuardrailProvider，把本条从"应当"升级为"做不到"。

## 7. Pending Actions 强制 Acknowledge

任何 subagent 返回的 L1 中含 `pending_actions[]`（详见 §2.4.3）时，lead **必须**在下一次 `task()` 调用前完成 acknowledge：

| Acknowledge 方式 | 实现 |
|---|---|
| 更新 plan 重派 | 更新 `metric_plan.json` 包含未决事项 → 调 task(code-executor) |
| 澄清用户 | 调 `ask_clarification` 问用户决策 |

**机制保证**：§5.5 节的 `HandoffPendingActionsProvider` 会 deny `task()` 调用直到 lead 用上面两种方式之一 ack。

---
```

---

## 修订 3：§5 新增 §5.5「机制层执行边界」

**改动位置**：spec line 547 「### 5.4 与本 spec 其它部分的协同」之后

**新增内容**：

```markdown
### 5.5 机制层执行边界（GuardrailProvider）

宪法第 6 条（执行边界）和第 7 条（pending_actions ack）通过 deerflow 现成的 [GuardrailProvider 协议](../../../packages/agent/backend/packages/harness/deerflow/guardrails/provider.py) 实现机制层防护。

**新增 2 个 Provider**（仿照 [`ScriptInvocationOnlyProvider`](../../../packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py) 的实现模板）：

#### 5.5.1 `LeadAgentExecutionBoundaryProvider`

**职责**：强制宪法第 6 条——lead 不能写脚本、不能跑非白名单 bash

**新文件**：`packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py`

**逻辑**：
```python
class LeadAgentExecutionBoundaryProvider:
    name = "lead_execution_boundary"

    _LEAD_BASH_ALLOWED = re.compile(
        r"^\s*python\s+-m\s+ethoinsight\.(parse|catalog)\."
    )
    _LEAD_BASH_SAFE_OPS = re.compile(
        r"^\s*(mkdir|cp|mv|ls|cat|grep|head|tail)(\s|$)"
    )
    _FORBIDDEN_SCRIPT_EXTENSIONS = (".py", ".sh", ".ipynb", ".bash", ".zsh")

    def evaluate(self, request):
        # Only gate lead — subagents have their own providers
        if request.agent_id and request.agent_id.startswith("subagent:"):
            return GuardrailDecision(allow=True)

        if request.tool_name == "write_file":
            path = request.tool_input.get("path", "")
            if path.endswith(self._FORBIDDEN_SCRIPT_EXTENSIONS):
                return GuardrailDecision(allow=False, reasons=[
                    GuardrailReason(
                        code="lead_execution_boundary.script_write_forbidden",
                        message=(
                            "lead 是调度员，不写脚本。补充分析/图表请：\n"
                            "  a) 更新 metric_plan.json → 重派 code-executor\n"
                            "  b) ask_clarification 问用户是否要做\n"
                            "执行分析脚本是 code-executor 的工作。"
                        )
                    )
                ])

        if request.tool_name == "bash":
            cmd = request.tool_input.get("command", "")
            if self._LEAD_BASH_ALLOWED.match(cmd) or self._LEAD_BASH_SAFE_OPS.match(cmd):
                return GuardrailDecision(allow=True)
            return GuardrailDecision(allow=False, reasons=[
                GuardrailReason(
                    code="lead_execution_boundary.bash_not_allowed",
                    message=(
                        "lead 的 bash 仅可：\n"
                        "  1. python -m ethoinsight.parse.* （解析数据）\n"
                        "  2. python -m ethoinsight.catalog.* （生成 plan.json）\n"
                        "  3. 文件操作：mkdir / cp / mv / ls / cat / grep / head / tail\n"
                        "执行分析脚本请走 task(code-executor)。"
                    )
                )
            ])

        return GuardrailDecision(allow=True)

    async def aevaluate(self, request):
        return self.evaluate(request)
```

**Wire 位置**：[`lead_agent/agent.py` line 314](../../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py) 之后追加：

```python
from deerflow.guardrails.lead_execution_boundary_provider import LeadAgentExecutionBoundaryProvider
middlewares.append(GuardrailMiddleware(
    provider=LeadAgentExecutionBoundaryProvider(),
    fail_closed=True,
))
```

#### 5.5.2 `HandoffPendingActionsProvider`

**职责**：强制宪法第 7 条——`pending_actions[]` 未 ack 时 block `task()`

**新文件**：`packages/agent/backend/packages/harness/deerflow/guardrails/handoff_pending_actions_provider.py`

**逻辑**：
```python
class HandoffPendingActionsProvider:
    """Block task() when prior handoff_*.json has unacknowledged pending_actions."""
    name = "handoff_pending_actions"

    def __init__(self, workspace_resolver):
        self._resolve_workspace = workspace_resolver  # 复用 Ev19WorkspaceBridgeMiddleware 的 contextvar 模式

    def evaluate(self, request):
        # 只在 lead 调用 task() 时检查
        if request.agent_id and request.agent_id.startswith("subagent:"):
            return GuardrailDecision(allow=True)
        if request.tool_name not in ("task", "ask_clarification"):
            return GuardrailDecision(allow=True)

        # ask_clarification 是合法 ack 方式，pass
        if request.tool_name == "ask_clarification":
            return GuardrailDecision(allow=True)

        workspace = self._resolve_workspace()
        if workspace is None:
            return GuardrailDecision(allow=True)  # fail-open

        # 扫描所有 handoff_*.json 找未 ack 的 pending_actions
        unacked = self._find_unacked_pending_actions(workspace)
        if not unacked:
            return GuardrailDecision(allow=True)

        # 检查本次 task 是否在 ack 范围内（lead 派的 subagent_type 或 prompt 内容覆盖了 unacked.action_id）
        if self._task_acks_pending(request.tool_input, unacked):
            return GuardrailDecision(allow=True)

        action_descs = "\n".join(f"  - {a['action_id']}: {a['description']}" for a in unacked)
        return GuardrailDecision(allow=False, reasons=[
            GuardrailReason(
                code="handoff_pending_actions.unacknowledged",
                message=(
                    f"前序 handoff 中有未处理的 pending_actions：\n{action_descs}\n\n"
                    "处理方式：\n"
                    "  a) 更新 metric_plan.json 包含这些事项 → 重派 code-executor\n"
                    "  b) ask_clarification 问用户决策\n"
                    "完成后再调用 task()。"
                )
            )
        ], policy_id="handoff_pending_actions")
```

**Wire 位置**：同 `LeadAgentExecutionBoundaryProvider`，加到 lead 中间件链。需要配套加 `WorkspaceBridgeMiddleware`（仿 [`Ev19WorkspaceBridgeMiddleware`](../../../packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py)）注入 workspace contextvar。

**实现细节留给 plan 阶段**：
- `_find_unacked_pending_actions` 的具体匹配规则（按时间戳？按 run_id？）
- `_task_acks_pending` 的匹配启发式（task prompt 包含 action_id？subagent_type=code-executor + plan 包含未决事项？）

---
```

---

## 修订 4：§7 更新 dogfood plan 关系表

**改动位置**：spec line 622

**改动前**：
```markdown
| Task 2 lead 角色边界 prompt | 本 spec 第 5 章（output-constitution）的预热——后续会迁到宪法 |
```

**改动后**：
```markdown
| Task 2 lead 角色边界 prompt | **过渡方案**——commit 24715250 修了 3 条输出禁令（不写判读 / 不引未告知元数据 / 不用绝对参考术语），但 thread b0d3a611 验证 prompt 自觉不够。本 spec §5.5 用机制层 `LeadAgentExecutionBoundaryProvider` 根治 "lead 越权写脚本" 这一执行层漏洞。Task 2 的 3 条输出禁令保留并迁入 §5 宪法第 1-3 条。 |
```

**额外加一行**：

```markdown
### 7.4 与 thread b0d3a611 E2E 失败的关系

本 spec 修订（§2.4.3 pending_actions / §5 宪法第 6-7 条 / §5.5 机制层 guardrail）直接响应 thread b0d3a611 E2E 失败的根因 A 和遗漏根因 1。详细诊断材料见 [docs/problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md](../../problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md)。

**根因 B（sandbox 路径不对称）+ 触发点（`//` 误报）的处置**：详见本 spec §10.3 长期 backlog。在 §5.5 的执行边界 guardrail 落地后，这两个 bug 在 lead 路径上**不再可触发**（lead 跑不了 `write_file *.py` 和 `python heredoc` 了），降级为非紧急维护项。
```

---

## 修订 5：§8 实施路径加阶段 1.5

**改动位置**：spec line 668「### 阶段 2：run-scoped 路径绝缘」之前

**新增内容**：

```markdown
### 阶段 1.5：执行边界 GuardrailProvider（~2-3 天，独立可先做）

**优先级**：高——直接修 thread b0d3a611 暴露的根因 A，可独立于阶段 1 落地

1. 新建 `guardrails/lead_execution_boundary_provider.py`（~80 行，照 `script_invocation_only_provider.py` 模板）
2. 新建 `guardrails/handoff_pending_actions_provider.py`（~120 行，照 `ev19_template_provider.py` 模板含 WorkspaceBridge）
3. Wire 到 `lead_agent/agent.py` 中间件链（line 314 之后）
4. 单测覆盖：
   - lead write_file `.py` → deny + 错误信息检查
   - lead write_file `.md` → allow
   - lead bash `python -m ethoinsight.catalog.resolve ...` → allow
   - lead bash `python -c "..."` → deny
   - lead bash `python /path/to/file.py` → deny
   - 含 pending_actions[] 的 handoff 存在 → lead task() deny
   - lead ask_clarification → 始终 allow（合法 ack 方式）
   - subagent (passport="subagent:...") 的 bash/write_file → pass through（不受 lead 边界限制）
5. 集成测试：用 thread b0d3a611 的复现流跑一遍——确认 lead 在 msg 52 想 write_file `gen_charts.py` 时被 deny + 看到错误信息后转向 update plan 或 ask_clarification

**阶段 1.5 可以与阶段 1 并行**——不依赖 L1/L2 双层 handoff 协议落地。
```

**阶段 3 的修订**：line 676 改为：

```markdown
### 阶段 3：输出宪法 + Constitution acknowledged 机制（~3-5 天）

1. 落 `output-constitution.md` 初版（含本 spec §5 第 1-7 条）
2. 4 个 subagent prompt 加"开工前 read_file"
3. L1 schema 加 `constitution_acknowledged` + `pending_actions` 字段
4. 同事 PR 进一步完善宪法内容
```

---

## 修订 6：§10 新增 §10.3 长期 backlog

**改动位置**：spec line 731「## 11. 修订记录」之前

**新增内容**：

```markdown
### 10.3 长期 backlog（不在本 spec 实施范围）

#### 10.3.1 sandbox 路径不对称（thread b0d3a611 根因 B）

**问题**：[`sandbox/tools.py:1568-1607`](../../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py) `write_file_tool` 写文件时不替换 content 里的 `/mnt/user-data/*` 字面量；而 `bash_tool` 的命令字符串会被 `replace_virtual_paths_in_command` 替换。**语义不对称**：

- `bash python -c "open('/mnt/user-data/uploads/x.txt')"` ✅ 工作（命令字符串被替换）
- `write_file '/mnt/.../x.py'` 写 `open('/mnt/user-data/uploads/x.txt')` 然后 `bash python /mnt/.../x.py` ❌ FileNotFoundError（脚本内容不被替换）

**为何不修**：在 §5.5 `LeadAgentExecutionBoundaryProvider` 落地后，lead 不能 `write_file *.py`，code-executor 又只能 `python -m ethoinsight.scripts.*`（预置模块自己读 `DEERFLOW_PATH_*` env var）——**两条都堵了之后这个 bug 在生产路径上无法触发**。

**未来何时该修**：当我们允许新的 subagent 自由写 `.py` 脚本（如未来 ad-hoc analysis subagent）时，需要回头根治：在 sandbox write_file_tool 中按文件扩展名白名单跑路径替换，或注入引导脚本头自动读 `DEERFLOW_PATH_*`。

#### 10.3.2 `_ABSOLUTE_PATH_PATTERN` 正则误报（thread b0d3a611 触发点）

**问题**：[`sandbox/tools.py:24`](../../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py) 的 `_ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![:\w])(?<!:/)/(?:[^\s\"'`;&|<>()]+)")` 是纯正则扫描，**不识别 shell 上下文**（heredoc body / `python -c "..."` 引号内容 / Python 整除 `//`）。thread b0d3a611 的 `n // 5000` 被误判为绝对路径。

**为何不修**：

- lead 路径上不再可触发（执行边界 guardrail 堵死）
- code-executor 路径上用 `python -m ...`，不进 heredoc，也不会写 `n // 5000` 这类表达式作为命令字符串

**未来何时该修**：当某个合法 bash 调用确实被误报（如 grep 含 `//` 的 URL pattern）时，根治方向：删 `_ABSOLUTE_PATH_PATTERN` 全局扫描，所有路径校验走 `_split_shell_tokens` 的 token-level 判断 + 加 heredoc-aware tokenizer（识别 `<<DELIM` 边界，body 整段视为一个 token 不扫描）。这是 `shlex` 标准库的局限，需要手写 bash parser 或引入 `bashlex`。

#### 10.3.3 ArchivingSummarizationMiddleware 压缩 pending_actions

**问题**：lead 看完 handoff 后过几轮触发 archive，pending_actions 内容可能被压缩出 lead 视野。

**目前缓解**：

- L2 文件不删，lead 任何时候可 read_file 找回
- `HandoffPendingActionsProvider`（§5.5.2）会**主动扫描 workspace 的 handoff_*.json**，不依赖 message history——所以即使被 archive 压掉，guardrail 仍然能 block 未 ack 的 task()

**因此本问题在 §5.5 落地后自然消解**，不需独立修复。
```

---

## 修订 7：§11 修订记录

**改动位置**：spec line 732 表格新增一行

**新增内容**：

```markdown
| 2026-05-14 | draft v2 | 根据 thread b0d3a611 E2E 失败和 deepseek co-analysis 更新：§2.4 加 pending_actions[]、§5 加宪法第 6-7 条 + §5.5 机制层 guardrail、§7 标 Task 2 为过渡方案、§8 加阶段 1.5、§10.3 加长期 backlog 论证。复用 deerflow GuardrailProvider 机制实现根治。 |
```

---

## 验收清单（spec 修订是否完整）

- [x] 修复根因 A（lead 越权写脚本）—— §5 宪法第 6 条 + §5.5.1 `LeadAgentExecutionBoundaryProvider`
- [x] 修复遗漏根因 1（lead 忽视 handoff errors）—— §2.4.3 `pending_actions[]` + §5 宪法第 7 条 + §5.5.2 `HandoffPendingActionsProvider`
- [x] 处置根因 B（sandbox 路径不对称）—— §10.3.1 论证为何不修
- [x] 处置触发点（`//` 误报）—— §10.3.2 论证为何不修
- [x] 处置遗漏根因 2（archive 压缩 errors）—— §10.3.3 论证为何自然消解
- [x] 复用 deerflow 现成机制（CLAUDE.md 第 12 条）—— §5.5 全部基于 GuardrailProvider
- [x] 实施阶段独立可落地 —— §8 阶段 1.5 标"~2-3 天，独立可先做"
- [x] 关联诊断材料 —— §7.4 引用 problems 文档

---

## 给用户的决策点（review 前请确认）

1. **是否同意"根因 B 留长期 backlog 不修"** —— §10.3.1 的论证站不站得住？lead 路径堵死后 code-executor 真的不会再写带硬编码 `/mnt/user-data/*` 的脚本吗？（答：code-executor 走 `python -m`，调用预置模块，预置模块自己读 env var；这是 spec 5-13 "脚本是工具不是 LLM 产物" 的延续）
2. **是否同意"`//` 误报降级"** —— §10.3.2 的论证站不站得住？现有合法 bash 路径里没人会跑含 `//` 的 grep？（答：风险存在但低；建议留 backlog，跑 dogfood 时再判断要不要补做）
3. **`HandoffPendingActionsProvider` 的匹配启发式** —— §5.5.2 的 `_task_acks_pending` 留给 plan 阶段决定，OK 吗？（这部分有一定设计自由度，要不要现在就拍板）
4. **阶段 1.5 是否真能与阶段 1 并行** —— L1 schema 没落地前，pending_actions 字段从哪里来？是不是要等阶段 1 才能跑端到端？（答：可以拆分——阶段 1.5 先上 `LeadAgentExecutionBoundaryProvider`（不依赖 L1）；`HandoffPendingActionsProvider` 等阶段 1 L1 schema 落地后再上）
5. **新 provider 的 fail_closed 策略** —— 默认 `fail_closed=True`（provider 出错也拒）—— 这对 lead 调度是否过于激进？是不是该 `fail_closed=False`（出错时放行 + warning）？

---

## 不在本修订范围

- 不写代码（任何 .py 实现）
- 不改任何 prompt 文件
- 不动 lead_agent/prompt.py 已有内容（24715250 的 prompt 约束保留，作为机制层之外的双重保险）
- 不改 sandbox/tools.py（§10.3 论证为何不动）
- 不开新 plan（plan 在本 spec 定稿后另开）
