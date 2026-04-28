# Gate Enforcement 生产级改造设计

**日期**: 2026-04-28
**状态**: 待审批
**讨论来源**: [2026-04-28-p0-permission-fix-handoff.md](../handoffs/2026-04-28-p0-permission-fix-handoff.md)

---

## 1. 问题定义

### 1.1 当前架构的致命缺陷

EthoInsight 的端到端分析流水线有 5 个 Gate（Gate 0/1/1.5/2/3/4/5），但 **所有 Gate 逻辑 100% 依赖 prompt 文本**。无论 prompt 措辞多强硬（"必须"、"硬规则"、"不可跳过"），LLM 在推理时都可能选择忽略——这在 E2E 中已经发生：

- Gate 1：用户说"鱼群行为"，agent 跳过两级确认直接调 `set_experiment_paradigm`
- Gate 2：`data_quality_warnings` 有 critical 条目（n=2<3），agent 选择不阻塞继续流水线

**这不是措辞问题，是架构问题**：prompt engineering 对控制流没有强制力。

### 1.2 设计目标

| 维度 | 目标 | 衡量方式 |
|------|------|----------|
| 可靠性 | Gate 触发不依赖 LLM 自觉，代码层强制执行 | 单元测试覆盖所有 gate 路径 |
| 可观测性 | 每次 gate 检查有结构化日志，可追溯决策 | `grep "gate_check"` 即可复盘 |
| DeerFlow 原生 | 复用框架已有机制，不发明新抽象 | 使用 `AgentMiddleware.wrap_tool_call` + `ClarificationMiddleware` + `experiment-context.json` |

---

## 2. 当前已有什么

### 2.1 GateEnforcementMiddleware（已存在但不完整）

文件：`deerflow/agents/middlewares/gate_enforcement_middleware.py`（今日创建，28 行核心逻辑）

**已实现**：
- 在 `workflow_mode="manual"` 时激活
- 拦截所有 `task()` 调用
- 检查 `experiment-context.json` 是否存在
- 不存在 → 返回 `ToolMessage` error，引导 agent 调用 `ask_clarification`
- 存在 → 放行

**未实现（本次设计要补的）**：
- **Gate 1 条件分支**：不做任何判断，只要 json 不存在就拦截。error message 是固定文本，没有区分"用户已提供大类+细分范式"（应引导 agent 直接调 set_experiment_paradigm 后重试）vs"只提供了大类"（应引导只问细分）
- **Data quality hard stop**：完全缺失。code-executor 返回后，没有任何机制检查 `data_quality_warnings`
- **拦截方式**：用 `ToolMessage`（软拦），agent 理论上可以不理会。没有用 `Command(goto=END)`（硬中断）

### 2.2 experiment_context.py（已存在）

提供 4 个函数：
- `read_context(workspace_dir)` → dict | None
- `context_exists(workspace_dir)` → bool
- `resolve_workspace_from_state(state)` → str | None
- `set_experiment_paradigm_tool()` — LangChain `@tool`，写入 experiment-context.json

### 2.3 experiment-context.json 当前 schema

```json
{
  "paradigm": "shoaling",
  "paradigm_cn": "斑马鱼鱼群行为",
  "category": "zebrafish",
  "subject": "fish",
  "paradigm_confirmed_at": "2026-04-28T10:00:00Z",
  "gate_completed": ["gate1_paradigm"]
}
```

### 2.4 中间件链中 GateEnforcementMiddleware 的位置

```
agent.py:_build_middlewares():
  ...
  LoopDetectionMiddleware
  ThinkTagMiddleware
  custom_middlewares (if any)
  GateEnforcementMiddleware(enabled=True)   ← 在 ClarificationMiddleware 之前
  ClarificationMiddleware()                  ← 最后一个
```

位置正确：在 `ClarificationMiddleware` 之前，所以 `ToolMessage` error → agent 调用 `ask_clarification` → `ClarificationMiddleware` 拦截 → `Command(goto=END)` 的中断链条是通的。

### 2.5 关键设计约束

- `workflow_mode` 通过 `config.configurable.workflow_mode` 注入，默认 `"auto"`
- 中间件只能从 `request.state` 获取 thread 状态
- `ClarificationMiddleware` 拦截 `ask_clarification` 工具调用，格式化为用户可见的问题，然后 `Command(goto=END)` 中断图执行
- Agent 恢复后，用户的回答作为新的 HumanMessage 出现在消息历史中

---

## 3. 设计方案

### 3.1 整体思路

**双层 enforcement**：中间件层（强制拦截）+ Tool 层（二次校验，defense in depth）。

```
Agent 调用 task(code-executor)
  │
  ├─ 第一层: GateEnforcementMiddleware.wrap_tool_call()
  │   ├─ Gate 1 检查: experiment-context.json 是否存在且 paradigm 字段有效？
  │   │   ├─ 存在且有效 → 放行
  │   │   └─ 不存在 → 返回 ToolMessage error，内容包含条件分支引导
  │   │         (引导 agent 根据用户已提供信息量决定跳过/简化/完整走两级确认)
  │   └─ （data quality 检查不在这里，在 code-executor 之后）
  │
  ├─ 第二层: task_tool 函数体内
  │   ├─ （同上检查，defense in depth）
  │   └─ 正常 dispatch subagent
  │
  └─ code-executor 执行…

Agent 调用 task(data-analyst)
  │
  ├─ 第一层: GateEnforcementMiddleware.wrap_tool_call()
  │   ├─ Gate 2 检查: handoff_code_executor.json 是否有 critical warning？
  │   │   ├─ 有 critical 且未 acknowledge → 返回 ToolMessage error
  │   │   └─ 无 critical 或已 acknowledge → 放行
  │
  └─ 第二层: task_tool 函数体内（同上）
```

### 3.2 Gate 1：范式确认（增强已有中间件）

**当前行为**：`experiment-context.json` 不存在 → 拦截所有 `task()` 调用。

**改为**：增加条件分支，区分三种情况：

| 用户输入 | experiment-context.json | 行为 |
|----------|------------------------|------|
| 大类 + 细分范式都明确（如"斑马鱼鱼群行为"） | 不存在 | **拦截**，但 error message 引导 agent 直接调 `set_experiment_paradigm` 后重试（最快路径：1 轮往返） |
| 只明确大类（如"焦虑相关实验"） | 不存在 | 拦截，error message 引导只问细分 |
| 什么都没提供（如"帮我分析"） | 不存在 | 拦截，error message 引导完整两级流程 |

**实现方式**：中间件本身不做"推断"（那是 LLM 的事），而是改变拦截后的 error message：

```python
def _build_block_message(self, request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(
        content=(
            "实验范式尚未确认。请执行以下步骤：\n"
            "1. 如果用户已经明确提到了大类名和具体范式名（如"斑马鱼鱼群行为""斑马鱼"）：\n"
            "   直接调用 set_experiment_paradigm(...)，然后重新调用 task()。\n"
            "2. 如果用户只提到了大类但未指定细分（如只说"焦虑迷宫"没说具体是 EPM 还是零迷宫）：\n"
            "   调用 ask_clarification 只问细分范式那一级。\n"
            "3. 如果用户什么都没提供：\n"
            "   调用 ask_clarification 分两步：先问大类，再问细分。\n\n"
            "范式分类表见 system prompt 中的"识别实验范式与实验设计类型"章节。"
        ),
        tool_call_id=request.tool_call.get("id", ""),
        name="gate_enforcement",
    )
```

**关键设计决策**：不在中间件中做范式推断（中间件不知道 7 大类→18 范式的映射），推断仍是 LLM 的职责。中间件的职责是：**确保 experiment-context.json 存在且有效**，而不检查它被写入之前是怎么确认的。

### 3.3 Data Quality Gate（新增）

**触发点**：`task(subagent_type="data-analyst")` 被调用时。

**检查逻辑**：
1. 读取 `handoff_code_executor.json`（在 workspace 目录下）
2. 检查 `data_quality_warnings` 数组
3. 如果存在 `severity="critical"` 的条目：
   - 检查 `experiment-context.json` 的 `gate_completed` 是否包含 `"gate2_quality_acknowledged"`
   - 如果未 acknowledge → 返回 ToolMessage error，列出所有 critical warning
   - 如果已 acknowledge → 放行

**experiment-context.json schema 扩展**：

```json
{
  "paradigm": "shoaling",
  "paradigm_cn": "斑马鱼鱼群行为",
  "category": "zebrafish",
  "subject": "fish",
  "paradigm_confirmed_at": "2026-04-28T10:00:00Z",
  "gate_completed": ["gate1_paradigm", "gate2_quality_acknowledged"],
  "quality_acknowledged_at": "2026-04-28T10:05:00Z",
  "quality_warnings_acknowledged": [
    "轨迹中断（missing data > 10%）: Subject 3"
  ]
}
```

**拦截消息模板**：

```python
def _build_quality_block_message(self, warnings: list[dict]) -> ToolMessage:
    warning_lines = "\n".join(
        f"- [{w.get('severity', 'warning')}] {w.get('message', str(w))}"
        for w in warnings
    )
    return ToolMessage(
        content=(
            f"数据质量检查发现以下 critical 问题，必须先获得用户确认才能继续：\n\n"
            f"{warning_lines}\n\n"
            f"请调用 ask_clarification 告知用户这些问题，提供以下选项：\n"
            f"(a) 排除异常个体并重算 (b) 保留并继续 (c) 查看详情\n\n"
            f"用户确认后，调用 write_file 更新 experiment-context.json，"
            f"在 gate_completed 中添加 'gate2_quality_acknowledged'，然后重新调用 task(data-analyst)。"
        ),
        tool_call_id=request.tool_call.get("id", ""),
        name="gate_enforcement",
    )
```

### 3.4 experiment_context.py 新增函数

```python
def read_handoff(workspace_dir: str) -> dict | None:
    """Read handoff_code_executor.json. Returns None if absent."""
    ...

def get_critical_warnings(workspace_dir: str) -> list[dict]:
    """Extract critical-severity warnings from handoff. Returns empty list if none."""
    ...

def is_quality_acknowledged(workspace_dir: str) -> bool:
    """Check if data quality gate has been acknowledged in experiment-context.json."""
    ...
```

### 3.5 可观测性

每次 gate check 输出结构化日志：

```
logger.info("gate_check | gate=%s | thread=%s | result=%s | detail=%s",
            gate_name, thread_id, result, detail)
```

示例：
```
gate_check | gate=gate1_paradigm | thread=abc123 | result=blocked | detail=experiment-context.json not found
gate_check | gate=gate1_paradigm | thread=abc123 | result=allowed | detail=paradigm=shoaling, confirmed_at=2026-04-28T10:00:00Z
gate_check | gate=gate2_quality | thread=abc123 | result=blocked | detail=1 critical warning(s), not acknowledged
gate_check | gate=gate2_quality | thread=abc123 | result=allowed | detail=acknowledged at 2026-04-28T10:05:00Z
```

日志以 `gate_check` 为固定前缀，方便 grep/聚合/监控。

### 3.6 Prompt 变化

中间件接管 enforcement 后，prompt 从"规则 + 执行指令"退化为"执行指导"：

| 内容 | 之前（prompt 承担） | 之后（中间件承担） |
|------|---------------------|-------------------|
| Gate 1 何时触发 | prompt 说"必须先执行" | 中间件检查 json 是否存在 |
| Gate 1 条件分支 | 无（无条件两级） | 中间件 error message 引导条件分支 |
| Data quality hard stop | prompt 说"如果有 warnings → ask_clarification" | 中间件检查 handoff + context |
| 范式分类表 | prompt 内联 | 保持不变（LLM 推断需要） |
| ask_clarification 格式 | prompt 内联 | 保持不变（LLM 执行需要） |

**具体修改**：

- `prompt.py:279-311`（Gate 1 段）：保留"必须"正面指令和分类表 + ask_clarification 模板示例，删除被中间件取代的手写 JSON 格式等执行细节。措辞从"你必须先执行这两步检查"改为"你必须先确认实验范式（系统会在 task 调度时强制执行此检查）"。
- `prompt.py:1125-1136`（Step 1.5）：拆为两半
  - **Data quality 检查那半**：简化为引用句——"检查 handoff 中的 data_quality_warnings，如有 critical 条目系统会在你调度 data-analyst 时拦截，届时你需按拦截提示调用 ask_clarification"。
  - **write_file 共享摘要那半**：保留完整指令（中间件管不着 write_file，这个动作必须 agent 自己执行）。
- `quality-gates.md:5-16`（Gate 0 段）：添加条件分支说明
- `quality-gates.md:44-58`（Gate 2 段）：标注"由 GateEnforcementMiddleware 在 task(data-analyst) 调度时强制执行"

---

## 4. 实现步骤

### Step 0：Schema rename（前置依赖）

文件：[experiment_context.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py)

`set_experiment_paradigm_tool` 中 `gate_completed: ["gate1"]` → `gate_completed: ["gate1_paradigm"]`。
同步更新所有引用 `["gate1"]` 的现有单测。纯 rename，独立 commit。

### Step 1：扩展 experiment_context.py

文件：[experiment_context.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py)

新增 3 个函数：
- `read_handoff(workspace_dir)` — 读 handoff_code_executor.json
- `get_critical_warnings(workspace_dir)` — 提取 severity="critical" 的 warning
- `is_quality_acknowledged(workspace_dir)` — 读 experiment-context.json 检查 gate_completed 是否包含 "gate2_quality_acknowledged"

### Step 2：增强 GateEnforcementMiddleware

文件：[gate_enforcement_middleware.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/gate_enforcement_middleware.py)

改动：
1. `_should_block()` → 重命名为 `_check_gate1()`，逻辑不变
2. 新增 `_check_gate2(state)` — 检查 handoff 中的 critical warnings
3. 新增 `_build_quality_block_message(warnings)` — data quality 专用 error message
4. `wrap_tool_call()` — 增加 `subagent_type` 区分：
   - `task(code-executor)` → 只做 Gate 1 检查
   - `task(data-analyst)` → 只做 Gate 2 检查
   - 其他 `task(X)` → 只做 Gate 1 检查（分析流水线之外的任务也需要范式上下文）
5. 所有 gate check 加上结构化日志

### Step 3：简化 prompt.py

文件：[prompt.py](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py)

改动：
1. Line 279-311（Gate 1 段）：保留"必须"正面指令和分类表，删除被中间件取代的手写 JSON 格式等执行细节。加入条件分支指导文本。
2. Line 1125-1136（Step 1.5）：拆为两半——data quality 检查那半改为引用中间件拦截，write_file 共享摘要那半保留完整指令（中间件管不着文件写入）。

### Step 4：更新 quality-gates.md

文件：[quality-gates.md](packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md)

改动：
1. Gate 0 段：补充条件分支逻辑（与 prompt 对齐）
2. Gate 2 段：标注"由 GateEnforcementMiddleware 在 task(data-analyst) 调度时强制执行"

### Step 5：写测试

新增文件：`tests/test_gate_enforcement.py`

测试用例：
1. `test_gate1_blocks_when_no_context` — 无 json → 拦截
2. `test_gate1_allows_when_context_exists` — 有 json → 放行
3. `test_gate1_handles_missing_workspace_path` — workspace_path 缺失 → 放行（降级安全）
4. `test_gate2_blocks_when_critical_unacknowledged` — 有 critical 未确认 → 拦截
5. `test_gate2_allows_when_no_critical` — 无 critical → 放行
6. `test_gate2_allows_when_acknowledged` — 有 critical 已确认 → 放行
7. `test_gate2_handles_missing_handoff` — handoff 文件不存在 → 放行
8. `test_non_task_tools_pass_through` — 非 task 工具 → 直接放行
9. `test_disabled_middleware_passes_through` — enabled=False → 全部放行
10. `test_gate_check_logging` — 验证结构化日志输出

### Step 6：运行全量测试确认基线

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. python -m pytest tests/ --tb=short 2>&1 | tail -5
```

---

## 5. 设计决策

### 5.1 为什么用 ToolMessage 而不是 Command(goto=END)？

中间件的拦截点不是 `ask_clarification`，而是 `task()`。如果在这里直接 `Command(goto=END)`：
- 用户会看到一个技术性的"被拦截"消息，而不是精心格式化的选择题
- Agent 没有机会解释为什么停下来了
- 无法复用 ClarificationMiddleware 对 ask_clarification 的格式化

所以正确的链条是：
```
GateEnforcementMiddleware 拦截 task() 
  → 返回 ToolMessage error（"请先 ask_clarification"）
  → Agent 读取 error，调用 ask_clarification
  → ClarificationMiddleware 拦截 ask_clarification
  → Command(goto=END)，用户看到格式化的选择题
```

ToolMessage 方案在实际中的可靠性极高，因为 agent 的 goal 是完成分析，而到 code-executor 的唯一路径就是通过 task()——error 返回后没有替代路径。

### 5.2 为什么不在中间件中做范式推断？

中间件从 state 读取的是 `thread_data.workspace_path`，无法访问用户的原始消息内容。范式推断（"斑马鱼鱼群行为" → shoaling）需要理解用户语义，这是 LLM 的职责，不应在中间件中复制。

中间件的职责边界：**确保实验上下文已持久化**。至于 agent 怎么确认的（跳过了还是完整走了两级 flow），中间件不关心，只检查结果。

### 5.3 为什么不在 task_tool.py 中做 enforcement？

已经在做了——作为 defense in depth 的第二层。中间件是第一层（早拦截，省掉不必要的 LLM 推理），task_tool 函数体是第二层（即使中间件被意外绕过，工具层也能拦住）。

### 5.4 workflow_mode 切换

- `workflow_mode="auto"`：GateEnforcementMiddleware 不激活，零开销。用于生产环境自动模式。
- `workflow_mode="manual"`：激活所有 gate enforcement。用于飞轮期交互式质量保证。

---

## 6. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| Agent 忽略 ToolMessage error，不调 ask_clarification | 低 | Error 消息给出精确的下一步指令；如果 agent 反复重试 task()，中间件每次都拦截，形成 feedback loop 迫使 agent 改变行为 |
| 老 thread 缺少 experiment-context.json | 中 | 所有 gate check 在 `resolve_workspace_from_state` 返回 None 或 `thread_data` 缺失时 **fail open**（放行），不阻断已有会话。只有完整 thread_data 的新 thread 才被 enforcement 覆盖。此外，`gate_completed` 数组预留 schema 版本号字段，未来扩展时不破坏老 json |
| handoff_code_executor.json 格式变化 | 低 | `get_critical_warnings()` 用 `.get()` 防御式访问，格式不匹配返回空列表（fail open） |
| 中间件逻辑过于复杂 | 低 | 当前 GateEnforcementMiddleware 28 行，加 data quality 后约 80 行。如果超过 200 行（含 docstring 和注释），考虑拆分为 Gate1Middleware + Gate2Middleware |

---

## 7. 文件变更范围

| 文件 | 变更类型 | 行数估计 |
|------|----------|----------|
| `gate_enforcement_middleware.py` | 增强 | +50 |
| `experiment_context.py` | 新增函数 | +40 |
| `prompt.py` | 简化 | -30 |
| `quality-gates.md` | 对齐 | ±10 |
| `test_gate_enforcement.py` | 新增 | +120 |

---

## 8. 审批记录

**审批结论**: 有条件批准
**审批日期**: 2026-04-28

### 修复项（已合入本文档）

1. Gate 1 表格"放行" → "拦截引导"（与 3.1 流程图和 5.2 决策一致）
2. Step 1.5 拆为两半——data quality 检查引用中间件，write_file 共享摘要保留完整指令
3. 老 thread 风险评估补显式 fail-open 条件（`resolve_workspace_from_state` 返回 None 时放行）
4. gate_completed 命名统一为 `["gate1_paradigm", "gate2_quality_acknowledged"]`
5. 中间件复杂度门槛从 150 行提到 200 行

### Follow-up（不阻塞实施，写入 PR 描述）

- **defense in depth 第二层**: task_tool 函数体暂不加二次校验，作为隐式兜底留到后续迭代。当前中间件层已足够覆盖两个 choke point
- **schema 版本号**: `experiment-context.json` 加 `schema_version` 字段，未来迁移老 json 时按版本号处理
- **`set_experiment_paradigm_tool`**: 已在 Step 0 中完成 rename，无需额外操作
