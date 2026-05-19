# Subagent 协作信息传递：files are facts

> **状态**: 已通过第 2 轮评估 + 修订。
> **评估 agent**: 本文档不要求实施，只需评估设计正确性、潜在遗漏、边界条件。
> **第 2 轮修订要点**: 补全顶层设计隐喻（指挥不抄谱），新增 2 项改动（gate_signals 契约 + Guardrail 隔离拦截），明确范围排除（inbox 物理隔离作为 follow-up）。

## 真实动因

本 spec 的动因是**dogfooding 观察 + 架构演进方向冲突**，不是生产事故。

- **dogfooding 观察**：开发者使用追问场景时注意到 knowledge-assistant 的回答缺少对 `outlier_findings` / `method_warnings` 的引用。code_summary.json 字段清单（lead prompt.py:1068）只含 `paradigm` / `groups` / `metrics_summary` / `statistics` / `chart_paths` / `data_quality_warnings`——data-analyst 产出的离群分析和方法学警告完全不在内。
- **架构演进方向**：EV19 模板重构（2026-04-29 / 2026-05-08 spec）的方向是"subagent 直接读权威 handoff + 渐进披露知识"，code_summary 这种"lead 手工捏摘要"的中转层和新方向不协调，积压会增加未来重构成本。

未观察到 code_summary 与 handoff 内容真实冲突的事故，"以谁为准"是设计风险而非现实问题。

## 顶层设计：乐团指挥隐喻

多 subagent 协作的角色定位：

| 角色 | 职责 | 不做 |
|------|------|------|
| **lead = 乐团指挥** | 调度（决定谁演奏什么）、倾听（subagent 最终消息）、替用户把关（gate_signals 决策）、用户呈现（转述 + 必要时从 handoff 提数字） | 不演奏（不做数据加工）、不抄谱（不生成派生摘要文件如 code_summary） |
| **subagent = 乐手** | 演奏（数据处理 / 分析 / 写报告）、写自己的产物（handoff_*.json）、用结构化契约报告指挥 | 不偷看其他乐手的乐谱（不读其他 subagent 的 handoff，除非指挥明确授权） |

这个隐喻导出的几条强约束：

1. **指挥不抄谱**——lead 不为下游 subagent 生成派生摘要文件
2. **乐手不偷看**——subagent 之间不直接通信，下游需要上游产出时由 lead 在 task() prompt 中明确点名授权
3. **指挥听汇报，不读乐谱细节**——lead 优先看 subagent 最终消息（含 gate_signals + 摘要）做决策；仅在异常时（特权角色）read_file outbox 追细节
4. **文件 handoff 是事实源**——subagent 产出的 handoff JSON 是该 subagent 工作结果的权威表达，不存在二手摘要

## 当前架构

```
code-executor → handoff_code_executor.json
                  ↓
            lead 读 → 精简 → 写 code_summary.json    ← 违反"指挥不抄谱"
                  ↓
            knowledge-assistant 追问时读 code_summary.json
            （不读 handoff_data_analyst.json，丢失 outlier/method_warnings）
            ← 违反"指挥不抄谱"导致的信息缺失
```

```
data-analyst.py / report-writer.py system prompt 明文写
"可读 /mnt/shared/code_summary.json 作为兜底"          ← 诱导横向通信 + dead instruction
```

## 问题清单

### 1. code_summary.json 违反"指挥不抄谱"

两份文件描述同一件事（分析结果），但来源不同：
- `handoff_*.json` = subagent 直接产出（事实源）
- `code_summary.json` = lead 手工精简（二手摘要）

精简过程**确认**丢失关键字段：code_summary.json 不含 `outlier_findings` / `method_warnings` / `counterfactual`——这些恰恰是用户追问"为什么 p 不显著"时最有价值的内容。当两份文件内容不一致时，无判定依据。

### 2. knowledge-assistant 追问场景信息不足

dogfooding 观察到：用户看完报告后追问（如"为什么 p 不显著"），lead 路由到 knowledge-assistant。当前 knowledge-assistant 只读 `code_summary.json`，data-analyst 已经做过的离群分析、反事实推理、method_warnings 完全丢失，追问回答质量取决于一份摘要而非完整上下文。

### 3. lead 做了数据中转（不是决策）

写 code_summary 是非决策性工作——调度员不应该做数据精简。**这是符号性问题**，单看 lead 多一次 write_file 负担不大；但和未来 EV19 重构方向冲突，积压会增加重构成本。

### 4. data-analyst / report-writer 的兜底引用掩盖真实失败

data_analyst.py:27-28 和 report_writer.py:34 将 `code_summary.json` 作为兜底输入。这是 dead instruction（lead 派遣时已经只传 handoff 路径，从不传 code_summary 路径），但它存在于 contract 段会诱导模型"对照"两份事实，影响推理稳定性；删掉后失败更可见，不再被 code_summary 静默掩盖。

### 5. 缺少 subagent 间隔离的机制约束

当前 subagent system_prompt 中明文允许"读 code_summary.json"，等于允许跨 subagent 文件横向通信的雏形。改完 1-4 后只是删了一个具体文件名，但底层问题仍在：**subagent 仍可 read_file 任意 workspace 文件**，未来 prompt drift 可能再次出现"data-analyst 读 handoff_code_executor.json + handoff_planning.json 兜底"这类横向通信。需要机制层防腐烂，而不是只靠 prompt 纪律。

### 6. lead 做数据质量决策（Step 1.5）时需 read_file 完整 handoff，膨胀上下文

lead 在 Step 1.5 拦截决策需要看 handoff 的 `data_quality_warnings` 字段是否含 critical 级条目。当前实现是 `read_file handoff_code_executor.json` 把整个 JSON（含 metrics_summary / per_subject / statistics 等几 KB 到几十 KB）灌入 lead 对话历史。lead 真正需要的只是几条门禁信号，但拿到了演奏数据全文，上下文浪费。

## 设计原则

```
subagent 产出的 handoff JSON = 事实源
任何 subagent 的输入 = lead 在 task() prompt 中给的信息 + lead 授权读的 handoff JSON
任何 subagent 不直接读另一个 subagent 的输出文件（除 lead 在 task prompt 明确授权）
不创建对同一事实的"摘要"副本（如 code_summary.json）
lead 用 subagent 最终消息中的结构化 gate_signals 做决策，不读 handoff 演奏数据
lead 是特权角色：可 read_file 任意 handoff（用于异常排查），但正态决策路径不依赖 read_file
重新分析后旧 handoff 被覆盖，之前的追问回答不应被视为对新分析有效
```

## 范围决策

讨论中提出过两个更彻底的方向，最终**排除在本 spec 之外**：

### 排除 1: inbox/outbox 目录改造

每个 subagent 独立 `workspace/<subagent>/inbox/` + `outbox/`，lead 调度时显式投递文件到下游 inbox。
**排除理由**：纯文件路径重命名，不影响 agent 表现或正确性；改动会动到 SubagentExecutor / LocalSandboxProvider / ThreadDataMiddleware / task_tool 等多个受保护文件（CLAUDE.md 同步规则列为高风险），未来 deerflow 上游同步成本翻倍；v0.1 路径上不需要。

### 排除 2: subagent sandbox 物理隔离（per-subagent 虚拟路径视图）

subagent 的 sandbox 只暴露自己的目录，物理上看不到其他 subagent 的 outbox。
**排除理由**：subagent 是项目自有代码，互信；模型不会主动 ls workspace 翻文件，"乱读"非常罕见；用 GuardrailMiddleware 拦截 + prompt 严令已足以防腐烂（改动 6），物理隔离是过度防御；需改 LocalSandboxProvider per-subagent 视图，CLAUDE.md 明示"血泪教训"——这块和上游 Tier 4 体系冲突严重。

两项作为未来路径（v0.1 之后）保留可能，但**不在本 spec 实施范围**。本 spec 仅做**真正影响 agent 表现和正确性的改动**。

## 改动方案

**改动范围：4 个 prompt 文件 + 1 个 handoff_schemas 字段升级 + 1 个新 GuardrailProvider + task_tool/SubagentExecutor 小幅扩展（占位符 + 授权透传）。**

| 改动 | 类型 | 涉及文件 |
|------|------|---------|
| 1 | prompt 清理 | `subagents/builtins/knowledge_assistant.py` |
| 2 | prompt 清理 | `subagents/builtins/data_analyst.py` |
| 3 | prompt 清理 | `subagents/builtins/report_writer.py` |
| 4 | prompt 清理（最复杂） | `agents/lead_agent/prompt.py` |
| 5 | schema + prompt 升级 | `subagents/handoff_schemas.py` + 3 个 subagent prompt |
| 6 | 占位符机制 + GuardrailProvider | `tools/builtins/task_tool.py`（占位符扩展）+ `subagents/executor.py`（透传授权列表）+ `guardrails/handoff_isolation_provider.py`（新建）+ `lead_agent/prompt.py`（lead 用占位符派遣） |

### 改动 1: knowledge-assistant system prompt

文件：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py`

场景 A（追问）的 `<contract>` 段改为：

```diff
- 输入:
-   - 场景 A（追问）: lead agent 提供问题 + {{shared://code_summary.json}} 引用（如有分析结果）
-   - 场景 B（纯知识）: lead agent 只提供问题
+ 输入:
+   - 场景 A（追问）: lead agent 提供问题 + 占位符授权的 handoff 文件
+     （lead 派遣时通过 {{handoff://code_executor}} 等占位符传递；
+     subagent 看到的是已解析的真实路径 /mnt/user-data/workspace/handoff_*.json）
+   - 场景 B（纯知识）: lead agent 只提供问题
```

场景 A 的工作流改为：

```diff
- - read_file prompt 中引用的 /mnt/shared/code_summary.json
- - 结合领域知识解释结果
+ - read_file lead 在 prompt 中授权的 handoff JSON 文件（路径已由占位符解析），
+   结合 handoff 中的具体数据 + 领域知识回答
+ - 不要尝试 read_file 其他 handoff 文件——未经占位符授权的读取会被 Guardrail 拦截
```

**max_turns 边界**: 当前 `max_turns=6`。noldus-kb 禁用时，场景 A 典型路径为"读 handoff (1 turn) → 回答 (1 turn)"，2-3 turn 足够。若未来 noldus-kb 启用 + 同时读 2 个 handoff，路径变为"读 handoff A (1) + 读 handoff B (1) + 调 MCP (2) + 回答 (1)" = 5 turn，仍在 6 以内。暂不调 max_turns，noldus-kb 启用后按实测决定是否提到 8。

场景 B（纯知识问答）不读 handoff，不受影响，不改。

### 改动 2: data-analyst system prompt

文件：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`

`<contract>` 输入段删掉 code_summary.json 兜底引用（行 27-28）：

```diff
  输入:
    - /mnt/user-data/workspace/handoff_code_executor.json — code-executor 的结构化交接文件
-   - /mnt/shared/code_summary.json — code-executor 的精简数据快照（和 handoff 重叠度高，可作为兜底）
```

**副作用**: 删掉后，如果 `handoff_code_executor.json` 读取失败（磁盘错、JSON 损坏），data-analyst 直接进 `<failure>` 分支返回 `status=failed`，不再有摘要兜底路径。这不是退化——失败更可见，不再被 code_summary 静默掩盖。

**说明**: lead 派遣 data-analyst 时传的 prompt（prompt.py:1075）已经只提 `handoff_code_executor.json`，不提 code_summary。本改动实质是清理 dead instruction，不影响实际运行行为。

### 改动 3: report-writer system prompt

文件：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`

`<contract>` 输入段删掉 code_summary.json 兜底引用（行 34）：

```diff
  输入（两个 handoff 文件 + 可选数据快照）:
    - /mnt/user-data/workspace/handoff_code_executor.json
    - /mnt/user-data/workspace/handoff_data_analyst.json
-   - /mnt/shared/code_summary.json — 可选兜底，和 handoff_code_executor 重叠度高
    - /mnt/user-data/workspace/handoff_planning.json — 若存在，可读 group_semantics 字段
```

### 改动 4: lead agent system prompt（最复杂）

文件：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

经 grep 验证，`code_summary` 在 prompt.py 中出现 **7 处**，必须全部处理：

| 行号 | 上下文 | 处理方式 |
|------|--------|----------|
| 308 | 调度职责表 knowledge-assistant 输入栏 `{{shared://code_summary.json}}` | 改为 `workspace/handoff_code_executor.json（如有 handoff_data_analyst.json 也附上）` |
| 377 | "你负责将 code-executor 的 handoff 精简后写入 /mnt/shared/code_summary.json" | 删除整句 |
| 412 | 呈现模板 "### 关键指标（从 code_summary.json 提取）" | 改为 "### 关键指标（从 handoff_code_executor.json 提取）"；lead 自己 `read_file` handoff_code_executor.json 取 M±SD 填表格（不是委托 data-analyst） |
| 529 | code-executor 调度示例 `{{shared://code_summary.json}}` | 改为 `workspace/handoff_code_executor.json` |
| 573 | knowledge-assistant 调度示例 `{{shared://code_summary.json}}` | 改为 `workspace/handoff_code_executor.json 和 workspace/handoff_data_analyst.json` |
| 1061-1070 | 整段 Step 1.5「数据质量校验 + 写共享摘要」 | **整段重写**：保留 `read_file` 动作 + 数据质量校验逻辑，删"精简写入 code_summary.json"部分，改名为「数据质量校验 + 准备呈现」。lead 自己从 handoff 中提取数字填后续呈现模板。配合改动 5，Step 1.5 拦截决策**改为优先看 code-executor 最终消息中的 gate_signals**，仅在 gate_signals 不充分时 read_file handoff 追细节 |
| 1083 | "用自然语言整合 code_summary.json + handoff_data_analyst.json 的内容呈现给用户" | 改为 "用自然语言整合 handoff_code_executor.json + handoff_data_analyst.json 的内容呈现给用户" |

**lead 呈现策略**: 采用"lead 自己读 handoff 提取数字"（而非委托 data-analyst）。理由：data-analyst 已经返回了 `key_findings` + `outlier_findings` 等判断，lead 呈现 M±SD 表格是"转述事实"不是"自己做分析"。lead 在 Step 1.5 已经 `read_file` 了 handoff_code_executor.json，直接从读到的内容里提取数字填表格即可。

**Step 4 分支（"先帮我解释 XX"）**: prompt.py:1104 已经写「派遣 knowledge-assistant，prompt 附 handoff_code_executor.json 和 handoff_data_analyst.json 路径」——这已经是目标状态，不改。仅改 529/573 行的示例代码使之与 1104 一致。

### 改动 5: subagent 最终消息契约升级（gate_signals）

**目的**: lead 默认不读 handoff 演奏数据，靠 subagent 最终消息中的结构化 gate_signals 做决策（Step 1.5 拦截 / 路由判断）。read_file handoff 退化为异常排查路径。

#### 5.1 升级 `handoff_schemas.py`

文件：`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`

为 `CodeExecutorHandoff` 和 `DataAnalystHandoff`（如已有）添加 `gate_signals` 字段，作为给 lead 的结构化决策依据：

```python
class GateSignals(BaseModel):
    """Structured decision signals from subagent to lead.

    Lead reads these from subagent's final AIMessage (not from handoff file)
    to make Step 1.5 quality-gate decisions without inflating context.
    """

    model_config = ConfigDict(extra="allow")

    data_quality: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Summary of data_quality_warnings: "
            "{'critical_count': int, 'warning_count': int, "
            "'critical_items': [str, ...]  # 关键 critical 条目摘要，每条 <80 字}"
        ),
    )
    statistical_validity: Literal["ok", "warning", "failed"] = "ok"
    errors_count: int = 0


class CodeExecutorHandoff(BaseModel):
    # ... 现有字段 ...
    gate_signals: GateSignals | None = Field(
        default=None,
        description=(
            "Structured signals for lead's decision-making. "
            "Optional in JSON file (lead reads from final AIMessage instead), "
            "but recommended to include for stability."
        ),
    )
```

`gate_signals` 在 JSON 文件中是 optional（lead 主要从 final AIMessage 解析），写入文件用于审计 / 回放。

#### 5.2 升级 subagent 最终消息契约

修改 3 个 subagent 的 system_prompt 中"最终 AIMessage 应输出什么"的约定：

**code-executor**（`subagents/builtins/code_executor.py`）：
最终消息除现有"OK: handoff written"外，加一段 `[gate_signals]` 块，列出关键 critical/warning 条目（不超过 5 条）。

**data-analyst**（`subagents/builtins/data_analyst.py`）：
最终 AIMessage 现有的"2-3 段关键发现摘要"基础上，追加 `[gate_signals]` 块——含 method_warnings_count、outlier_count 等。

**report-writer**（`subagents/builtins/report_writer.py`）：
最终消息加 `[gate_signals]` 块——含 sections_written 完整性、是否有失败章节。

格式约定（自然语言 + 半结构化，便于 lead 解析）：

```
[gate_signals]
data_quality:
  critical_count: 1
  warning_count: 2
  critical_items:
    - IID 为常数（单鱼模式 vs 群体模式不匹配）
statistical_validity: warning
errors_count: 0
```

**为什么不强制结构化工具调用（`report_completion` 工具）**: deerflow 现有"subagent write_file handoff JSON + 返回最终 AIMessage" 已经是事实结构化契约，handoff_schemas.py 提供 Pydantic 校验。再加一个工具会重复造轮子，且要改 SubagentExecutor 处理工具调用结果——动受保护文件。当前方案是 prompt 工程 + schema 增强，不动框架。

#### 5.3 升级 lead Step 1.5

如改动 4 表格 1061-1070 行所述，Step 1.5 决策路径变为：

```
1. 收到 code-executor 最终 AIMessage
2. 解析消息中的 [gate_signals] 块
3. 如果 gate_signals.data_quality.critical_count > 0:
   → 调 ask_clarification 询问用户是否继续（不变）
4. 如果 gate_signals 不完整或异常（subagent 没按契约输出）:
   → 兜底 read_file handoff_code_executor.json，按原逻辑校验 data_quality_warnings
5. 否则正常进入 Step 2 派遣 data-analyst
```

**预期收益**: 正态情况 lead 在 Step 1.5 不再 read_file，对话上下文减少 5-30 KB（取决于 handoff 大小）。异常路径仍可读，无回退风险。

### 改动 6: 占位符授权机制 + GuardrailProvider 拦截 subagent 偷读 handoff

**目的**: 用 deerflow 现有的占位符机制（`{{shared://}}`）作为蓝本，扩展出 `{{handoff://}}` 占位符同时解决"文件路径正确性"+"权限授权"两个问题，并配合 GuardrailProvider 在机制层拦截违规读取，防止未来 prompt drift（subagent prompt 不小心又写"可读 handoff_X 作为兜底"）。

**核心洞察**：lead 在 prompt 中用 `{{handoff://code_executor}}` 这个动作有双重语义——既"指了路径"，也"授了权"。没有占位符 = 没有授权。框架层一次完成路径解析 + 权限注入，避免靠正则猜 prompt 内容。

#### 6.1 扩展 task_tool 占位符机制

文件：`packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py`

现有 `{{shared://filename}}` → `/mnt/shared/filename` 是字符串替换。**保留不动**（作为 follow-up 清理项，本 spec 不删）。

新增 `{{handoff://<subagent_name>}}` 占位符，按 subagent 名（不是文件名）查表得到真实路径：

```python
# 注册表：subagent name → handoff filename
HANDOFF_FILE_REGISTRY: dict[str, str] = {
    "code_executor": "handoff_code_executor.json",
    "data_analyst": "handoff_data_analyst.json",
    "report_writer": "handoff_report_writer.json",
    "planning": "handoff_planning.json",  # 若 planning skill 启用
}

_HANDOFF_PLACEHOLDER_RE = re.compile(r"\{\{handoff://([^}]+)\}\}")


def _resolve_handoff_placeholders(prompt: str) -> tuple[str, set[str]]:
    """Replace ``{{handoff://<subagent_name>}}`` with full workspace path,
    and return the set of authorized paths for this task delegation.

    Returns: (replaced_prompt, authorized_absolute_paths)

    Unknown subagent names raise ValueError immediately (fail-fast on typo).
    """
    authorized: set[str] = set()

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        if name not in HANDOFF_FILE_REGISTRY:
            raise ValueError(
                f"Unknown handoff subagent '{name}' in placeholder. "
                f"Known: {sorted(HANDOFF_FILE_REGISTRY)}"
            )
        filename = HANDOFF_FILE_REGISTRY[name]
        full_path = f"/mnt/user-data/workspace/{filename}"
        authorized.add(full_path)
        return full_path

    replaced = _HANDOFF_PLACEHOLDER_RE.sub(_replace, prompt)
    return replaced, authorized
```

`task_tool` 的占位符解析流程（在 `executor.execute_async()` 之前）：

```python
# Existing: resolve {{shared://...}}
prompt = _resolve_placeholders(prompt)

# New: resolve {{handoff://...}} and capture authorized paths
prompt, authorized_handoffs = _resolve_handoff_placeholders(prompt)

# Pass authorized_handoffs to the executor so GuardrailProvider inside
# subagent's middleware chain can read it
executor = SubagentExecutor(
    ...,
    authorized_handoff_paths=authorized_handoffs,
)
```

**lead 使用方式**（替代当前 prompt 里写完整路径的方式）：

```python
# Before
task(subagent_type="data-analyst",
     prompt="请分析 /mnt/user-data/workspace/handoff_code_executor.json 中的数据...")

# After
task(subagent_type="data-analyst",
     prompt="请分析 {{handoff://code_executor}} 中的数据...")
```

#### 6.2 SubagentExecutor 透传授权列表

文件：`packages/agent/backend/packages/harness/deerflow/subagents/executor.py`

`SubagentExecutor.__init__` 新增 `authorized_handoff_paths: set[str] | None = None` 参数，在 `_build_middlewares()`（或对应位置）实例化 `HandoffIsolationProvider` 时传入：

```python
self.guardrail_provider = HandoffIsolationProvider(
    authorized_paths=authorized_handoff_paths or set(),
    self_outbox_subagent_name=config.name,  # 允许 subagent 读自己的产出
)
middlewares.append(GuardrailMiddleware(provider=self.guardrail_provider))
```

这是对 SubagentExecutor 的最小侵入修改（加一个可选参数 + 一行中间件挂载）。不动 SubagentExecutor 的核心调度循环。

#### 6.3 新建 `handoff_isolation_provider.py`

文件：`packages/agent/backend/packages/harness/deerflow/guardrails/handoff_isolation_provider.py`（新建，约 80 行）

```python
class HandoffIsolationProvider:
    """Block subagents from reading peer subagents' handoff files unless
    lead has authorized the path via {{handoff://...}} placeholder in task prompt.

    Authorization is supplied by task_tool at dispatch time (parsed from
    {{handoff://<name>}} placeholders), not by parsing the prompt text inside
    the provider. This avoids regex guessing and keeps semantics explicit:
    no placeholder = no authorization.

    Rationale: enforce 'files are facts' principle in mechanism, not just prompt.
    Lead is the authorizing party — only files lead names via placeholder are
    accessible to the subagent.
    """

    name = "handoff_isolation"

    def __init__(self, authorized_paths: set[str], self_outbox_subagent_name: str | None = None):
        self.authorized_paths = authorized_paths
        self.self_outbox_subagent_name = self_outbox_subagent_name

    def _is_own_handoff(self, file_path: str) -> bool:
        """Allow subagent to read its own handoff file (it just wrote it).

        e.g. data-analyst writes handoff_data_analyst.json then may re-read
        for self-validation. This is not 'peeking at peer'.
        """
        if not self.self_outbox_subagent_name:
            return False
        # Convert 'data-analyst' → 'handoff_data_analyst.json'
        normalized = self.self_outbox_subagent_name.replace("-", "_")
        return f"handoff_{normalized}.json" in file_path

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if not request.is_subagent:
            return GuardrailDecision(allow=True)
        if request.tool_name != "read_file":
            return GuardrailDecision(allow=True)
        file_path = request.tool_input.get("file_path", "")
        # Only gate handoff_*.json reads; other files unrestricted
        if "handoff_" not in file_path or not file_path.endswith(".json"):
            return GuardrailDecision(allow=True)
        # Allow self-outbox reads
        if self._is_own_handoff(file_path):
            return GuardrailDecision(allow=True)
        # Check authorization list
        if file_path in self.authorized_paths:
            return GuardrailDecision(allow=True)
        return GuardrailDecision(
            reasons=[GuardrailReason(
                code="handoff_isolation.unauthorized",
                message=f"Subagent attempted to read {file_path} without lead authorization. "
                        f"Authorized paths: {sorted(self.authorized_paths)}",
            )],
            policy_id="handoff_isolation",
        )
```

#### 6.4 lead system prompt 改用占位符

文件：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

改动 4 中所有"改为 `workspace/handoff_X.json`"的位置，**统一改为占位符形式**：

| 行号 | 原值 | v2 占位符形式 |
|------|------|---------------|
| 308 | `{{shared://code_summary.json}}` | `{{handoff://code_executor}}（如有 {{handoff://data_analyst}} 也附上）` |
| 529 | `{{{{shared://code_summary.json}}}}` | `{{{{handoff://code_executor}}}}` |
| 573 | `{{{{shared://code_summary.json}}}}` | `{{{{handoff://code_executor}}}} 和 {{{{handoff://data_analyst}}}}` |
| 1075 | `/mnt/user-data/workspace/handoff_code_executor.json`（已是真实路径） | 改为 `{{{{handoff://code_executor}}}}` |
| 1104 / 1109 | 类似 | 类似改占位符 |
| 1116 / 1117 | 重入场景的 prompt 示例 | 同上 |

**注意**: prompt.py 中现有"已经写完整路径"的位置（如 1075）也建议改占位符——保持机制一致，让 lead 习惯只用占位符派遣 subagent。lead 自己在 Step 1.5 read_file 时仍用真实路径（lead 是特权角色，不通过占位符）。

#### 6.5 与现有 GuardrailProvider 的关系

deerflow 已有：
- `GateEnforcementMiddleware`（管 paradigm 字段，CLAUDE.md 第 10 条）
- EV19 重构计划中的 `ev19_template_provider`（管 ev19_template 字段）

`HandoffIsolationProvider` 是第三个独立 provider，职责正交（管 subagent 跨文件读取），并列存在。

## 边界条件

### 已有分析数据的重入场景（prompt.py:1115-1117）

- 用户说"只帮我重新写个报告" + workspace 已有 handoff → 直接派 report-writer
- 用户说"帮我重新解读一下" + 已有 handoff → 直接派 data-analyst
- 用户说"用不同的分组重新分析" → 从 Step 1 重新派 code-executor

这些完全不依赖 code_summary，本身符合 files are facts。✅ **无影响。**

报告/解读重入派 subagent 时，lead 必须在 task prompt 中明确提到所需的 handoff 路径（否则改动 6 的 Guardrail 会拦截）——现有 prompt.py:1109 / 1075 已经这么写。

### 多轮追问中 handoff 被覆盖

用户追问 → knowledge-assistant 读 handoff 答 → 用户说"用不同分组重新跑" → code-executor 覆盖 handoff_code_executor.json。旧追问回答不再对新的 handoff 有效。这不是本 spec 引入的问题——handoff 文件覆盖是现有语义。本 spec 设计原则里已声明"重新分析后旧 handoff 被覆盖，之前的追问回答不应被视为对新分析有效"。

### gate_signals 缺失或不规范

某些场景 subagent 可能未按契约输出 `[gate_signals]` 块（早期 fine-tune 模型、deepseek 输出不稳定）。改动 5.3 的 Step 1.5 兜底：**gate_signals 不完整时回退到 read_file handoff 校验**——回退路径保留，不引入正确性风险。

随着微调数据累积，subagent 输出 gate_signals 的合规率会提升，正态路径占比越来越高。

### Guardrail 误拦截

占位符方案下，授权列表是 task_tool 从 `{{handoff://}}` 占位符精确收集的 set，不靠正则猜 prompt 文本——误判面比之前小得多。仍存在两种边界情况：

- **lead 直接写完整路径而不用占位符**：subagent 看到完整路径但 authorized_paths 为空 → 拒绝。**这是设计意图**——强制 lead 走占位符通道。lead system prompt 必须明确"派遣 subagent 时引用 handoff 一律用 `{{handoff://<name>}}` 占位符，不要写完整路径"。
- **subagent 读自己刚写的 handoff**（如 data-analyst 写完 handoff_data_analyst.json 后想再读自己核对）：`HandoffIsolationProvider._is_own_handoff()` 按 subagent config.name 匹配文件名前缀，允许。

未在注册表 `HANDOFF_FILE_REGISTRY` 中的占位符 → `_resolve_handoff_placeholders()` 抛 ValueError，fail-fast 暴露 typo。lead 收到错误信息后会修改 prompt 重试。

### /mnt/shared/ 目录和 {{shared://}} 占位符

改完之后 production 中没有任何 prompt 再用 `{{shared://}}` 占位符，`/mnt/shared/` 目录无 production 读写流量。机制本身保留（无需删），占空间忽略不计。**列为 follow-up 清理项**，不阻塞本 spec：后续可清 `config/paths.py:8` 的 `shared_dir()` 和相关路径定义。

## 单测影响

- `tests/test_sandbox_tools_security.py`: 多处用 `/mnt/shared/code_summary.json` 作为 fixture 验证路径合法性——这些测试测的是路径校验机制而非业务逻辑，**不算 production 引用，不改**。验收 grep 时排除 test 文件。
- `tests/test_subagent_prompt_clarity.py:69`: 注释 `# Primary input is now the code-executor handoff; code_summary.json` 说明该测试已经在校验"不再依赖 code_summary"——**这是助力，不是阻塞**。
- 新增测试需求：
  - `test_handoff_isolation_provider.py`: 单测 `HandoffIsolationProvider` 的 allow/deny 决策（含授权列表解析、subagent 读自己 outbox 的允许、未授权读取的拒绝）
  - `test_gate_signals_schema.py`: 校验 `GateSignals` schema 字段
  - `test_lead_step_1_5_gate_signals.py`: 模拟 subagent 最终消息含/不含 gate_signals 时 lead 的决策路径

## 不变的部分

- **反问 / 阶段确认机制完全不变**：`ask_clarification` 工具 + `ClarificationMiddleware` + 各 Gate 反问机制（Gate 1 范式确认、Gate 2 数据质量 critical 反问、Gate 3 subagent 失败反问、Step 3 用户三选一）—— **本 spec 的所有改动都不动这层**。`ClarificationMiddleware` 是中间件链最后一位，触发 `Command(goto=END)` 中断对话等用户回复，与改动 1-6 无任何耦合。
- code-executor 工作流不变（T5 之后 tools + skills 已切换 SOTA）
- data-analyst / report-writer 的核心工作流不变（仅 contract 段微调）
- handoff JSON schema 主体不变（仅新增 optional `gate_signals` 字段）
- deerflow 框架代码 0 改动（受保护文件全部不动：SubagentExecutor / LocalSandboxProvider / ThreadDataMiddleware / task_tool 等）
- subagent 之间的信息传递仍然通过文件（不是 agent 间直接聊天），遵守 Delegation 模式
- GateEnforcementMiddleware / EV19 ev19_template_provider 等已有 Guardrail 不动，HandoffIsolationProvider 并列新增

## 验收条件

1. grep `code_summary` 在 `packages/agent/backend/packages/harness/deerflow/` 下无 production 引用（排除 `tests/` 目录和注释/文档字符串）
2. knowledge-assistant 追问场景能读到 handoff_data_analyst.json 的 outlier_findings 和 method_warnings（dogfooding 实测："为什么 p 不显著"的回答能引用 method_warnings 中 n<5 的警告）
3. data-analyst / report-writer / lead agent 的 prompt 中不再提到 code_summary.json
4. lead prompt.py 中 7 处 code_summary 引用全部处理（按改动 4 的表逐项验收）
5. `handoff_schemas.py` 含 `GateSignals` 模型；3 个 subagent prompt 含 `[gate_signals]` 输出约定
6. `task_tool.py` 含 `HANDOFF_FILE_REGISTRY` + `{{handoff://}}` 占位符解析；未知 subagent 名抛 ValueError
7. `SubagentExecutor` 接受 `authorized_handoff_paths` 参数并透传给 `HandoffIsolationProvider`
8. `HandoffIsolationProvider` 实现 + 注册到 subagent 中间件链；单测覆盖核心 allow/deny 路径（含 self_outbox 允许）
9. 实测：subagent 在未使用占位符授权时 read_file `handoff_*.json` 被拦截，返回 error ToolMessage
10. 实测：lead Step 1.5 正态路径不再 read_file handoff_code_executor.json（看消息中 gate_signals）；异常路径仍可回退 read_file
11. lead system prompt 中所有派遣 subagent 的示例都用 `{{handoff://<name>}}` 占位符，不写完整 handoff 路径
12. 现有单测（agent backend）全过

---

## 修订记录

- **v1.0** (2026-05-11): brainstorming 会话初始产出
- **v1.1** (2026-05-11): 第 1 轮评估修订——改动 4 重写为 7 处完整列表；明确 lead 呈现策略；补 max_turns/兜底副作用/单测影响/边界条件等
- **v2.0** (2026-05-11): 第 2 轮评估 + 顶层设计深化——
  - 新增"真实动因"段：dogfooding 观察 + 架构演进方向冲突
  - 新增"顶层设计：乐团指挥隐喻"段：lead = 指挥，subagent = 乐手，导出 4 条强约束
  - 新增"范围决策"段：明确排除 inbox/outbox 目录改造 + 物理隔离（作为 v0.1 后 follow-up）
  - 新增改动 5：subagent 最终消息含 gate_signals 契约，lead Step 1.5 优先看消息不读 handoff
  - 新增改动 6：`HandoffIsolationProvider` 机制层拦截 subagent 偷读
  - 修订"问题清单"：新增问题 5（缺机制约束）和问题 6（上下文膨胀），原 4 个问题语义微调
  - 修订"不变的部分"：明确反问/阶段确认机制 0 影响
  - 修订"验收条件"：从 5 条扩为 9 条
- **v2.1** (2026-05-11): 改动 6 实现方式升级——
  - 引入 `{{handoff://<subagent_name>}}` 占位符（复用 deerflow 现成 `{{shared://}}` 机制蓝本）
  - 占位符同时解决"路径正确性"+"权限授权"两个问题——指占位符 = 授权
  - `task_tool.py` 扩展 `HANDOFF_FILE_REGISTRY` + `_resolve_handoff_placeholders()`，返回授权 set
  - `SubagentExecutor` 新增 `authorized_handoff_paths` 参数，透传给 `HandoffIsolationProvider`
  - `HandoffIsolationProvider` 不再正则猜 prompt，授权列表从 task_tool 精确传入
  - 加入 `_is_own_handoff()` 允许 subagent 读自己刚写的 handoff（避免 data-analyst 自核对被拦）
  - 新增改动 6.4：lead system prompt 中所有派遣示例统一用占位符（不写完整路径）
  - 验收条件从 9 条扩为 12 条
- **后续路径**（不在本 spec）：inbox/outbox 目录改造 + LocalSandboxProvider 物理隔离 + `{{shared://}}` 占位符清理——v0.1 之后视实际需要再议
