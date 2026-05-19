# files-are-facts: subagent 协作信息传递重构实施 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> 🔔 **2026-05-11 PATCH（如果你正在实施本 plan，请读）**:
> 实施过程中发现 deerflow skill 注入机制和 code-executor 工作模式有特殊性，原 plan 不完整。已追加两个 task：
> - **Task 5a**：清理 ethoinsight-planning skill 中残留的 code_summary 引用 + handoff 路径改占位符语义。**插在 Task 5 和 Task 6 之间**。
> - **Task 7a**：在 ethoinsight-code skill 的 6 个范式胶水脚本模板中追加 [gate_signals] 输出代码（code-executor 最终消息来自胶水脚本 stdout，不是模型自由生成，因此 Task 7 对 code-executor system_prompt 的改动**不会单独生效**——必须配合 Task 7a 改胶水脚本才能让 lead 真正收到 [gate_signals] 块）。**插在 Task 7 和 Task 8 之间**。
>
> Task 7 文本内已加 ⚠️ 警告，澄清 code-executor 最终消息的机制 + Task 7a 的必要性。Task 13 验收清单也已更新（Step 1a/1b + 用例 C）。

**Goal:** 落实 spec `docs/superpowers/specs/2026-05-11-subagent-file-is-facts-design.md`——删除 code_summary.json 中转层，引入 `{{handoff://}}` 占位符 + GuardrailProvider 隔离机制，强化 subagent 最终消息契约（gate_signals），让 lead 不读 handoff 演奏数据也能做决策。

**Architecture:** 4 个 subagent prompt 清理 + lead system prompt 7 处清理（含改占位符）+ `handoff_schemas.py` 加 `GateSignals` 字段 + `task_tool.py` 扩展 `{{handoff://}}` 占位符并捕获授权列表 + `SubagentExecutor` 加 `authorized_handoff_paths` 参数 + 新建 `HandoffIsolationProvider` GuardrailProvider 接入 subagent 中间件链。零改动到 LocalSandboxProvider / ThreadDataMiddleware（保留受保护文件原状）。

**Tech Stack:** Python 3.12+ / Pydantic / LangChain / LangGraph / pytest / deerflow harness 现有 GuardrailMiddleware 协议

**任务执行顺序原则:**
- Task 1-4: 低风险、独立、不影响运行——纯 prompt 文本清理 + dead instruction 删除（subagent contract 段）
- Task 5: 仍是 prompt 清理但量大、复杂——lead system prompt 7 处命中
- Task 6-7: schema + prompt 升级——引入 gate_signals 契约
- Task 8-11: 框架扩展（task_tool 占位符、SubagentExecutor 参数、GuardrailProvider 新建、中间件接入）
- Task 12: lead system prompt 改用占位符（依赖 8-11 已落地）
- Task 13: e2e 验收 + grep 清理验证

每个 task 独立 commit，失败可回滚。

**仓库结构提示:**
- 所有源码在 `packages/agent/backend/packages/harness/deerflow/`，import 前缀 `deerflow.*`
- 所有测试在 `packages/agent/backend/tests/`
- 命令必须在 `packages/agent/backend/` 目录运行
- 测试运行：`source .venv/bin/activate && make test` 或 `PYTHONPATH=. uv run pytest tests/<file>.py -v`
- Lint：`make lint`
- 项目根 CLAUDE.md / backend CLAUDE.md 是受保护文件清单的权威——任何改动遇到这些规则冲突时立即停止并问

---

## Task 1: 清理 knowledge-assistant system prompt 中的 code_summary 引用

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py:15-16, 32`
- Test: `packages/agent/backend/tests/test_subagent_prompt_clarity.py`（已存在，校验内容）

- [ ] **Step 1: 读取当前文件，确认要改的行号**

Run: `grep -n "code_summary\|/mnt/shared" packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py`

Expected output:
```
15:  - 场景 A（追问）: lead agent 提供问题 + {{shared://code_summary.json}} 引用（如有分析结果）
32:- read_file prompt 中引用的 /mnt/shared/code_summary.json
```

- [ ] **Step 2: 改 system_prompt 的 `<contract>` 段（行 15）**

把：
```
  - 场景 A（追问）: lead agent 提供问题 + {{shared://code_summary.json}} 引用（如有分析结果）
```

改为：
```
  - 场景 A（追问）: lead agent 提供问题 + 占位符授权的 handoff 文件
    （lead 派遣时通过 {{handoff://code_executor}} 等占位符传递；
    subagent 看到的是已解析的真实路径 /mnt/user-data/workspace/handoff_*.json）
```

注意：`{{shared://...}}` 和 `{{handoff://...}}` 在 Python 字符串里是普通字符。`SubagentConfig.system_prompt` 是用三引号字符串定义的，不需要双花括号转义。**直接写就行**。

- [ ] **Step 3: 改场景 A 工作流（行 32）**

把：
```
- read_file prompt 中引用的 /mnt/shared/code_summary.json
- 结合领域知识解释结果
```

改为：
```
- read_file lead 在 prompt 中授权的 handoff JSON 文件（路径已由占位符解析），
  结合 handoff 中的具体数据 + 领域知识回答
- 不要尝试 read_file 其他 handoff 文件——未经占位符授权的读取会被 Guardrail 拦截
```

- [ ] **Step 4: 验证 grep 无 code_summary 残留**

Run: `grep -n "code_summary\|/mnt/shared" packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py`

Expected: 无输出（exit code 1）

- [ ] **Step 5: 运行现有 prompt 清晰度测试**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_subagent_prompt_clarity.py -v`

Expected: 全部 PASS（不应因本改动失败；若失败大概率是 fixture 写死了 code_summary 字样，记录并 stop）

- [ ] **Step 6: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py
git commit -m "prompt(knowledge-assistant): 移除 code_summary.json 引用，改为 handoff 占位符语义"
```

---

## Task 2: 清理 data-analyst system prompt 中的 code_summary 兜底

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py:27-28`

- [ ] **Step 1: 确认要删的行**

Run: `grep -n "code_summary\|/mnt/shared" packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`

Expected:
```
27:  - /mnt/shared/code_summary.json — code-executor 的精简数据快照
28:    （和 handoff 重叠度高，可作为兜底）
```

- [ ] **Step 2: 删除这两行**

把：
```
  - /mnt/user-data/workspace/handoff_code_executor.json — code-executor 的
    结构化交接文件，包含 metrics_summary / per_subject / group_level_metrics /
    statistics / assessment / data_quality_warnings 等全部分析结果
  - /mnt/shared/code_summary.json — code-executor 的精简数据快照
    （和 handoff 重叠度高，可作为兜底）
```

改为（删后两行，保留第一项）：
```
  - /mnt/user-data/workspace/handoff_code_executor.json — code-executor 的
    结构化交接文件，包含 metrics_summary / per_subject / group_level_metrics /
    statistics / assessment / data_quality_warnings 等全部分析结果
```

- [ ] **Step 3: 验证 grep 无 code_summary 残留**

Run: `grep -n "code_summary\|/mnt/shared" packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`

Expected: 无输出

- [ ] **Step 4: 运行所有 subagent 相关测试**

Run: `cd packages/agent/backend && make test 2>&1 | tail -30`

Expected: 全部 PASS。如果某个 fixture 写死了 code_summary 字样，记录失败 stop。

- [ ] **Step 5: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py
git commit -m "prompt(data-analyst): 删除 code_summary.json 兜底（dead instruction，掩盖失败）"
```

---

## Task 3: 清理 report-writer system prompt 中的 code_summary 兜底

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py:34`

- [ ] **Step 1: 确认要删的行**

Run: `grep -n "code_summary\|/mnt/shared" packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`

Expected:
```
34:  - /mnt/shared/code_summary.json —— 可选兜底，和 handoff_code_executor 重叠度高
```

- [ ] **Step 2: 删除行 34**

把：
```
  - /mnt/user-data/workspace/handoff_data_analyst.json —— 专业解读
    （key_findings / outlier_findings / method_warnings / excluded_metrics /
    recommendations）
  - /mnt/shared/code_summary.json —— 可选兜底，和 handoff_code_executor 重叠度高
  - /mnt/user-data/workspace/handoff_planning.json —— 若存在，可读 group_semantics
```

改为（删中间行）：
```
  - /mnt/user-data/workspace/handoff_data_analyst.json —— 专业解读
    （key_findings / outlier_findings / method_warnings / excluded_metrics /
    recommendations）
  - /mnt/user-data/workspace/handoff_planning.json —— 若存在，可读 group_semantics
```

- [ ] **Step 3: 验证 grep**

Run: `grep -n "code_summary\|/mnt/shared" packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`

Expected: 无输出

- [ ] **Step 4: 运行测试**

Run: `cd packages/agent/backend && make test 2>&1 | tail -30`

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py
git commit -m "prompt(report-writer): 删除 code_summary.json 兜底（dead instruction）"
```

---

## Task 4: 清理 lead system prompt 表格描述行（行 308 / 377）

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:308, 377`

> 注：lead prompt.py 中 `code_summary` 共 7 处命中，本 task 处理结构最简单的 2 处（描述性表格行 + 简单删除）。其余 5 处复杂改动放 Task 5、Task 12。

- [ ] **Step 1: 读上下文（行 305-310）**

Run: `sed -n '305,310p' packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

应能看到调度职责表的一行：
```
| knowledge-assistant | 问题 + 可选 {{shared://code_summary.json}} | 文本回答 | 查询 noldus-kb + ethoinsight skill 知识 | — |
```

- [ ] **Step 2: 改行 308**

把 knowledge-assistant 这一行第二列中的 `{{shared://code_summary.json}}` 改为 `handoff 文件路径（由 lead 用 {{handoff://code_executor}} 等占位符授权）`。

完整改后：
```
| knowledge-assistant | 问题 + 可选 handoff 文件路径（由 lead 用 {{handoff://code_executor}} 等占位符授权） | 文本回答 | 查询 noldus-kb + ethoinsight skill 知识 | — |
```

- [ ] **Step 3: 读上下文（行 374-380）**

Run: `sed -n '374,380p' packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

应能看到：
```
### 共享 workspace 机制
- /mnt/shared/ 是 lead agent 和 subagent 之间的数据中继目录
- 你负责将 code-executor 的 handoff 精简后写入 /mnt/shared/code_summary.json
- data-analyst 的交付物是 /mnt/user-data/workspace/handoff_data_analyst.json（由 data-analyst 自己写入，你不需要干预）
```

- [ ] **Step 4: 删行 377（"你负责将..."整行）**

删除：
```
- 你负责将 code-executor 的 handoff 精简后写入 /mnt/shared/code_summary.json
```

注意：行 376 提到 `/mnt/shared/` 的存在性是事实（机制保留作为 follow-up 清理项），**不删行 376**。

- [ ] **Step 5: 验证只剩 5 处命中**

Run: `grep -cn "code_summary" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

Expected: 5（从 7 → 5）

- [ ] **Step 6: 运行测试**

Run: `cd packages/agent/backend && make test 2>&1 | tail -10`

Expected: 全部 PASS

- [ ] **Step 7: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
git commit -m "prompt(lead): 清理调度表 + 共享 workspace 段的 code_summary 引用（2/7 处）"
```

---

## Task 5: 重写 lead system prompt 中 Step 1.5 / 呈现模板 / 示例片段（行 412 / 529 / 573 / 1061-1070 / 1083）

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` 多处

> 注：本 task 的占位符**用 v0 形式（写真实路径 `workspace/handoff_*.json`）**。Task 12 会统一升级为 `{{handoff://}}` 占位符——拆分原因：先把 lead prompt 中 code_summary 死引用全清零（不依赖框架扩展），再统一改占位符（依赖 Task 8-11）。

- [ ] **Step 1: 改行 412（呈现模板）**

Run: `sed -n '410,415p' packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

应能看到：
```
### 关键指标（从 code_summary.json 提取）
```

改为：
```
### 关键指标（从 handoff_code_executor.json 提取）
```

- [ ] **Step 2: 改行 529（code-executor 调度示例 prompt）**

Run: `sed -n '527,531p' packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

应能看到：
```python
task(subagent_type="data-analyst", description="解读分析结果",
     prompt="请分析 {{{{shared://code_summary.json}}}} 中的旷场实验数据。")
```

改为：
```python
task(subagent_type="data-analyst", description="解读分析结果",
     prompt="请分析 /mnt/user-data/workspace/handoff_code_executor.json 中的旷场实验数据。")
```

- [ ] **Step 3: 改行 573（knowledge-assistant 调度示例 prompt）**

Run: `sed -n '571,575p' packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

应能看到：
```python
task(subagent_type="knowledge-assistant", description="解答 NND 指标含义",
     prompt="用户问题: NND 偏高说明什么？\n已有分析结果: {{{{shared://code_summary.json}}}}")
```

改为：
```python
task(subagent_type="knowledge-assistant", description="解答 NND 指标含义",
     prompt="用户问题: NND 偏高说明什么？\n相关数据在 /mnt/user-data/workspace/handoff_code_executor.json 和 /mnt/user-data/workspace/handoff_data_analyst.json，请 read_file 这两份文件后回答。")
```

- [ ] **Step 4: 重写 Step 1.5（行 1061-1070）**

Run: `sed -n '1060,1072p' packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

应能看到现有的"### Step 1.5: 数据质量校验 + 写共享摘要"段。

把整段（行 1061-1070）：

```
### Step 1.5: 数据质量校验 + 写共享摘要
1. read_file /mnt/user-data/workspace/handoff_code_executor.json
2. 检查 "data_quality_warnings" 字段：如有 critical 条目，系统会在你调度 task(data-analyst) 时拦截，届时你需按拦截提示调用 ask_clarification 告知用户并询问是否继续。
3. **写共享摘要**（使用 write_file 工具）：
   ```
   write_file("/mnt/shared/code_summary.json", <JSON 字符串，包含以下字段>)
   ```
   code_summary.json 包含：paradigm, groups, metrics_summary, statistics, chart_paths, data_quality_warnings
   （从 handoff 中直接提取这些字段，只包含分析结果，跳过 output_files.metrics 等原始文件路径）
   **写完后直接进入下一步，回复消息保持简洁即可**
```

整段替换为：

```
### Step 1.5: 数据质量校验 + 准备呈现

1. **首先看 code-executor 最终消息中的 `[gate_signals]` 块**（subagent 按契约输出，含 critical_count / warning_count / critical_items / statistical_validity / errors_count）：
   - 如果 `data_quality.critical_count > 0` → 调 `ask_clarification` 询问用户是否继续（系统会在你随后调度 task(data-analyst) 时拦截，按拦截提示走）
   - 如果 `statistical_validity == "failed"` → 同上，反问用户
   - 否则进入第 2 步

2. **如果 gate_signals 不完整或异常**（subagent 没按契约输出 `[gate_signals]` 块，或解析失败）：
   - 兜底 `read_file /mnt/user-data/workspace/handoff_code_executor.json`，按 `data_quality_warnings` 字段中是否有 severity=critical 条目做拦截判断（原逻辑）
   - 这是异常路径，正态情况不走

3. **准备呈现**：在 Step 3 自然语言整合时，你将从 handoff_code_executor.json 中提取 M±SD / n / p / 效应量等数字填表格。
   - 如果你在第 2 步**没有**为了校验而 read_file（即 gate_signals 正常使用），那么 Step 3 整合时你**首次** read_file handoff_code_executor.json 取数字
   - 如果你在第 2 步**已经** read_file 了（异常路径走过），那么 Step 3 直接复用读到的内容，不重复 read_file
   - **绝对不要再 write_file 到 /mnt/shared/，code_summary.json 中转层已废弃**
```

- [ ] **Step 5: 改行 1083（自然语言整合）**

Run: `sed -n '1081,1085p' packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

应能看到：
```
2. 按"分析结果呈现模板"（见前面章节）用自然语言整合 code_summary.json + handoff_data_analyst.json 的内容呈现给用户
```

改为：
```
2. 按"分析结果呈现模板"（见前面章节）用自然语言整合 handoff_code_executor.json + handoff_data_analyst.json 的内容呈现给用户
```

- [ ] **Step 6: 验证 code_summary 在 prompt.py 中清零**

Run: `grep -cn "code_summary" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

Expected: 0

Run: `grep -n "shared://code_summary\|/mnt/shared/code_summary" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

Expected: 无输出

- [ ] **Step 7: 全仓 grep code_summary（应只剩 tests/ 和 follow-up 项目）**

Run: `grep -rn "code_summary" packages/agent/backend/packages/harness/deerflow/ --include="*.py"`

Expected: 0 命中（所有 production 文件 0 处；CLAUDE.md 同步规则也不应有 production import 残留）。如果 tests/ 目录下有命中，Task 13 会确认那些是 fixture 不动。

- [ ] **Step 8: 运行全部测试**

Run: `cd packages/agent/backend && make test 2>&1 | tail -30`

Expected: 全部 PASS。注意 `test_subagent_prompt_clarity.py` 如校验了 code_summary 字样，需查看测试期望值——大概率是验证"不再依赖"而非"必须含"，应该 PASS。

- [ ] **Step 9: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
git commit -m "prompt(lead): 重写 Step 1.5 + 清理剩余 5 处 code_summary 引用

- Step 1.5 改名为「数据质量校验 + 准备呈现」，删 write code_summary 逻辑
- 优先看 code-executor 最终消息 [gate_signals] 决策，异常时回退 read handoff
- 呈现模板/示例代码改为引用 handoff_code_executor.json
- 自此 production 代码无 code_summary 引用"
```

---

## Task 5a: 清理 ethoinsight-planning skill 中的 code_summary 残留 + handoff 路径改占位符

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight-planning/SKILL.md:120`
- Modify: `packages/agent/skills/custom/ethoinsight-planning/references/intent-classification.md:22`

> 背景：ethoinsight-planning skill 是给 lead 用的（lead prompt.py:1013 主动 read_file）。其中两处和本 spec 冲突：
> - SKILL.md:120 的失败降级表里写"data-analyst 超时 → 把 code_summary 展示给用户"——code_summary 已废弃
> - intent-classification.md:22 写"派遣 knowledge-assistant 附 handoff 完整路径"——按 Task 12 的统一占位符规约应改为 `{{handoff://}}`

- [ ] **Step 1: 确认 SKILL.md 命中行**

Run: `grep -n "code_summary" packages/agent/skills/custom/ethoinsight-planning/SKILL.md`

Expected:
```
120:| data-analyst 超时/空返回 | 跳过，直接把 code_summary 展示给用户 |
```

- [ ] **Step 2: 改 SKILL.md 行 120**

把：
```
| data-analyst 超时/空返回 | 跳过，直接把 code_summary 展示给用户 |
```

改为：
```
| data-analyst 超时/空返回 | 跳过，lead 自己 read_file handoff_code_executor.json，把统计摘要（metrics_summary + statistics）转述给用户 |
```

理由：code_summary 已删；lead 作为特权角色可以直接 read_file handoff，从中提取摘要呈现。这与 spec "lead 是特权角色" 原则一致。

- [ ] **Step 3: 确认 intent-classification.md 命中行**

Run: `grep -n "handoff_code_executor\|handoff_data_analyst" packages/agent/skills/custom/ethoinsight-planning/references/intent-classification.md`

Expected:
```
22:    │   │       → 动作: 派遣 knowledge-assistant，附 workspace/handoff_code_executor.json 和 workspace/handoff_data_analyst.json 路径
40:- 前置条件: workspace 中有 handoff_code_executor.json
```

- [ ] **Step 4: 改 intent-classification.md 行 22**

把：
```
    │   │       → 动作: 派遣 knowledge-assistant，附 workspace/handoff_code_executor.json 和 workspace/handoff_data_analyst.json 路径
```

改为：
```
    │   │       → 动作: 派遣 knowledge-assistant，prompt 用 {{handoff://code_executor}} 和 {{handoff://data_analyst}} 占位符授权读取（参见 Task 12 引入的派遣规约）
```

**行 40 不改**——"前置条件: workspace 中有 handoff_code_executor.json" 描述的是 workspace 状态判断（lead 看 workspace 里有没有这个文件来决定走哪条分支），不是派遣 subagent 时的 prompt 内容，符合 lead 特权角色直接判断文件存在的语义。

- [ ] **Step 5: 验证 skill 目录无 code_summary 残留**

Run: `grep -rn "code_summary" packages/agent/skills/`

Expected: **完全无输出**（之前唯一一处已删）

- [ ] **Step 6: 验证 quality-gates.md 不需要改**

Run: `grep -n "code_summary\|handoff_code_executor\|handoff_data_analyst" packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md`

Expected output（仅展示，**不改**）：
```
71:**强制执行**: 由 GateEnforcementMiddleware 在 task(data-analyst) 调度时检查 handoff_code_executor.json，存在 critical 且未在 experiment-context.json 中 acknowledge 则拦截。
90:| 返回正常解读 | 继续流水线，读 handoff_data_analyst.json 向用户呈现洞察 |
91:| 超时 | 跳过 report-writer，直接把 handoff_code_executor.json 的统计摘要展示给用户 |
101:| 超时 | 用 data-analyst 的 handoff_data_analyst.json 里的 key_findings 作为最终输出 |
```

这些都是 **lead 行为描述**（lead 作为特权角色直接读 handoff），符合新设计，**保持不动**。`GateEnforcementMiddleware` 是 deerflow 现有中间件（管 paradigm 字段），和我们新加的 `HandoffIsolationProvider` 职责正交，描述准确无需改。

- [ ] **Step 7: 跑全套测试**

Run: `cd packages/agent/backend && make test 2>&1 | tail -10`

Expected: 全部 PASS（skill 文件改动不影响 Python 测试）

- [ ] **Step 8: Commit**

```bash
git add packages/agent/skills/custom/ethoinsight-planning/SKILL.md \
        packages/agent/skills/custom/ethoinsight-planning/references/intent-classification.md
git commit -m "skill(ethoinsight-planning): 清理 code_summary 残留 + handoff 路径改占位符语义

- SKILL.md:120 失败降级：lead 改为直接 read_file handoff 转述摘要
- intent-classification.md:22 派遣 knowledge-assistant 改占位符语义
- quality-gates.md 中的 lead 读 handoff 行为描述符合特权角色原则，保留"
```

---

## Task 6: 在 `handoff_schemas.py` 加 `GateSignals` 模型

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`
- Test: `packages/agent/backend/tests/test_gate_signals_schema.py`（新建）

- [ ] **Step 1: 写失败测试**

新建文件 `packages/agent/backend/tests/test_gate_signals_schema.py`：

```python
"""Tests for GateSignals Pydantic model in subagent handoff_schemas.

GateSignals is the structured decision payload lead reads from subagent's
final AIMessage (Step 1.5 quality-gate decisions without inflating context).
"""

import pytest
from pydantic import ValidationError


def test_gate_signals_default_construction_is_safe():
    """Empty GateSignals must construct cleanly with safe defaults."""
    from deerflow.subagents.handoff_schemas import GateSignals

    g = GateSignals()
    assert g.data_quality == {}
    assert g.statistical_validity == "ok"
    assert g.errors_count == 0


def test_gate_signals_full_construction():
    """All fields populated; ensure structured types come through."""
    from deerflow.subagents.handoff_schemas import GateSignals

    g = GateSignals(
        data_quality={
            "critical_count": 1,
            "warning_count": 2,
            "critical_items": ["IID 为常数（单鱼模式）"],
        },
        statistical_validity="warning",
        errors_count=0,
    )
    assert g.data_quality["critical_count"] == 1
    assert g.data_quality["critical_items"][0].startswith("IID")
    assert g.statistical_validity == "warning"


def test_gate_signals_rejects_invalid_validity():
    """statistical_validity is a constrained literal."""
    from deerflow.subagents.handoff_schemas import GateSignals

    with pytest.raises(ValidationError):
        GateSignals(statistical_validity="unknown")  # noqa: PIE837


def test_gate_signals_allows_extra_fields():
    """Future-proof: extra keys allowed (extra='allow' in model_config)."""
    from deerflow.subagents.handoff_schemas import GateSignals

    g = GateSignals.model_validate({
        "data_quality": {},
        "statistical_validity": "ok",
        "errors_count": 0,
        "future_field": "ignored gracefully",
    })
    # extra='allow' makes future_field accessible
    assert g.statistical_validity == "ok"


def test_code_executor_handoff_has_optional_gate_signals():
    """CodeExecutorHandoff must accept gate_signals=None (optional in file)."""
    from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

    h = CodeExecutorHandoff(status="completed", summary="ok")
    assert h.gate_signals is None


def test_code_executor_handoff_accepts_gate_signals():
    from deerflow.subagents.handoff_schemas import CodeExecutorHandoff, GateSignals

    h = CodeExecutorHandoff(
        status="completed",
        summary="ok",
        gate_signals=GateSignals(statistical_validity="warning"),
    )
    assert h.gate_signals.statistical_validity == "warning"
```

- [ ] **Step 2: 运行测试验证失败（GateSignals 还没定义）**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_gate_signals_schema.py -v`

Expected: 全部 FAIL with `ImportError: cannot import name 'GateSignals'`

- [ ] **Step 3: 实现 `GateSignals` + 在 `CodeExecutorHandoff` 加 optional 字段**

在 `packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py` 中：

(a) 在文件开头 import 段，确认已有 `from typing import Any, Literal` 和 `from pydantic import BaseModel, ConfigDict, Field`（行 14-17 已经有）。

(b) 在 `DataQualityWarning` 类后面（约行 50 之后、`CodeExecutorHandoff` 之前）插入：

```python
class GateSignals(BaseModel):
    """Structured decision signals from subagent to lead.

    Lead reads these from subagent's final AIMessage (not from the handoff
    file itself) to make Step 1.5 quality-gate decisions without inflating
    context with the full handoff JSON. Also persisted in handoff JSON for
    audit/replay (optional field).
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
```

(c) 在 `CodeExecutorHandoff` 类中加 optional 字段（在 `confidence` 字段后面，约行 88）：

```python
    gate_signals: GateSignals | None = Field(
        default=None,
        description=(
            "Structured signals for lead's decision-making. "
            "Optional in JSON file (lead reads from final AIMessage instead), "
            "but recommended to include for audit/replay."
        ),
    )
```

(d) 在 `DataAnalystHandoff` 类的尾部（`errors` 字段后）也加 optional 字段：

```python
    gate_signals: GateSignals | None = Field(default=None)
```

(e) 在 `ReportWriterHandoff` 同样加：

```python
    gate_signals: GateSignals | None = Field(default=None)
```

(f) 把 `GateSignals` 加到 `__all__` 列表：

```python
__all__ = [
    "CodeExecutorHandoff",
    "DataAnalystHandoff",
    "DataQualityWarning",
    "GateSignals",     # 新增
    "MetricStat",
    "OutlierFinding",
    "ReportWriterHandoff",
]
```

- [ ] **Step 4: 运行测试验证 PASS**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_gate_signals_schema.py -v`

Expected: 6 个测试全 PASS

- [ ] **Step 5: 跑全套测试确保不破坏现有 schema 测试**

Run: `cd packages/agent/backend && make test 2>&1 | tail -10`

Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py packages/agent/backend/tests/test_gate_signals_schema.py
git commit -m "schema: 新增 GateSignals 模型 + 在 3 个 handoff schema 加 optional gate_signals 字段"
```

---

## Task 7: 升级 3 个 subagent system_prompt——要求输出 `[gate_signals]` 块

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`

> 此 task 不写测试——prompt 输出契约是模型行为，不是函数。验收靠 Task 13 e2e 实测。

- [ ] **Step 1: 读 code_executor.py 当前最终消息约定**

Run: `sed -n '20,40p' packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`

应能看到：
```
工作完成后输出 1 行确认（如 "OK: handoff written"），handoff JSON 已写盘 ${workspace_path}/handoff_code_executor.json。
```

- [ ] **Step 2: 在 code-executor 中追加 `[gate_signals]` 输出契约**

找到行 32 附近的 "工作完成后输出 1 行确认" 句，把所在段（约 1-2 段）改为：

```
工作完成后输出最终消息，包含两部分：

1. 一行确认（如 `OK: handoff written`），表示 handoff JSON 已写盘 `${workspace_path}/handoff_code_executor.json`。

2. `[gate_signals]` 块——结构化决策信号给 lead，让 lead 不读 handoff 也能做数据质量决策。格式：

```
[gate_signals]
data_quality:
  critical_count: <int>
  warning_count: <int>
  critical_items:
    - <每条 <80 字的 critical 警告摘要>
    - ...（最多 5 条；超出条数省略号即可）
statistical_validity: ok | warning | failed
errors_count: <int>
```

字段语义：
- `critical_count`: handoff.data_quality_warnings 中 severity=="critical" 的条目数
- `warning_count`: severity=="warning" 的条目数
- `critical_items`: critical 条目的 message 字段摘要（每条 <80 字，截断时用 "…" 结尾）
- `statistical_validity`: "ok" = 统计结果可用；"warning" = 警告（如 n<5）；"failed" = 统计完全失败
- `errors_count`: handoff.errors 数组长度

即便所有 count 为 0，仍必须输出完整 `[gate_signals]` 块。**lead 用这个块的存在性判断是否走 gate_signals 路径**。
```

- [ ] **Step 3: 在 data-analyst 中追加 `[gate_signals]` 输出契约**

在 `data_analyst.py` 找到 `<workflow>` 段第 4 步（行 97-99 附近）"最终 AIMessage..." 后追加新一段（保留原"2-3 段关键发现摘要"约定）：

在 `<workflow>` 第 4 步之后、`<principles>` 之前插入：

```
<gate_signals_contract>
**最终 AIMessage 必须以 `[gate_signals]` 块结尾**，给 lead 提供结构化决策信号。
紧贴在 2-3 段自然语言摘要之后输出。格式：

```
[gate_signals]
method_warnings_count: <int>          # method_warnings 数组长度
outlier_count: <int>                  # outlier_findings 数组长度
excluded_metrics_count: <int>         # excluded_metrics 数组长度
statistical_validity: ok | warning | failed
errors_count: <int>
```

- `statistical_validity`: "ok" = 解读可用；"warning" = 有 method_warnings 但仍可参考；"failed" = handoff_code_executor.json 读取失败，无法解读
- 即便所有 count 为 0，仍必须输出完整 `[gate_signals]` 块
</gate_signals_contract>
```

- [ ] **Step 4: 在 report-writer 中追加 `[gate_signals]` 输出契约**

在 `report_writer.py` 找到 `<workflow>` 段第 5 步（"最终 AIMessage..."）后追加：

在 `<workflow>` 段末尾、`<write_file_chunking>` 之前插入：

```
<gate_signals_contract>
**最终 AIMessage 必须以 `[gate_signals]` 块结尾**：

```
[gate_signals]
sections_written_count: <int>         # sections_written 数组长度（期望 6）
sections_missing: [<str>, ...]        # 6 段骨架中未写成功的章节名（中文），为空则 []
statistical_validity: ok | failed     # report-writer 不评估统计有效性，按 handoff_code_executor 透传
errors_count: <int>                   # handoff_report_writer.json 中 errors 数组长度
```

- `sections_missing` 为空数组时表示 6 段骨架全部成功写入；非空表示有章节失败
- 即便所有 count 为 0、sections_missing 为空，仍必须输出完整 `[gate_signals]` 块
</gate_signals_contract>
```

- [ ] **Step 5: 跑测试，确保 prompt 改动不破坏现有 subagent config 单测**

Run: `cd packages/agent/backend && make test 2>&1 | tail -20`

Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py \
        packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py \
        packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py
git commit -m "prompt(subagents): 3 个 subagent 最终消息必须含 [gate_signals] 块，让 lead 不读 handoff 也能决策"
```

> ⚠️ **重要说明（执行 Task 7 时务必读）**: code-executor 的最终消息**不是模型自由生成**——它是胶水脚本（`analysis.py`）通过 `print()` 输出 的 stdout 经 bash 工具回到模型上下文，模型再转给 lead。所以即便 Task 7 改了 code-executor 的 system_prompt，模型也不会"自由发挥"在最终消息里加 [gate_signals] 块——因为它的最终消息本质上就是胶水脚本 stdout 的转述。
>
> **真正让 code-executor 输出 [gate_signals] 的落地点是 ethoinsight-code skill 的胶水脚本模板**——见 Task 7a。
>
> Task 7 对 code-executor system_prompt 的改动仍要做，作用是**告诉模型"看到 stdout 里有 [gate_signals] 块要原样保留转给 lead"**（防止它"贴心地"把这段不像普通日志的内容删掉）。data-analyst / report-writer 不依赖胶水脚本，Task 7 对它们的 system_prompt 改动**直接生效**。

---

## Task 7a: 在 ethoinsight-code skill 胶水脚本模板中追加 [gate_signals] 输出

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight-code/SKILL.md`（加 1 段约定）
- Modify: `packages/agent/skills/custom/ethoinsight-code/templates/output-contract.md`（加 [gate_signals] 输出规范段）
- Modify: 6 个范式 reference：
  - `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md`
  - `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/oft.md`
  - `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/ldb.md`
  - `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/tst.md`
  - `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/fst.md`
  - `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/zero-maze.md`

> 背景：code-executor 通过"写 analysis.py → bash 执行 → 把 stdout 转给 lead"的模式工作。要让 code-executor 的最终消息含 [gate_signals] 块，**必须改胶水脚本模板**让 Python 代码直接 print 这个块。这是确定性生成（不依赖模型推理），可靠性 100%。
>
> 6 个范式 reference 文件的结尾胶水脚本模板**结构一致**（都是 write_text + print "OK"），所以本 task 的改动模式可复用。

- [ ] **Step 1: 看 epm.md 当前胶水脚本结尾结构**

Run: `sed -n '85,95p' packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md`

应能看到类似：
```python
(WORKSPACE / "handoff_code_executor.json").write_text(
    json.dumps(handoff, ensure_ascii=False, indent=2)
)
print(f"OK: handoff written to {WORKSPACE / 'handoff_code_executor.json'}")
```

其他 5 个范式 reference 末尾结构基本一致，行号略有差异（spec 已经 grep 确认）。

- [ ] **Step 2: 改 epm.md 的胶水脚本模板**

找到 `(WORKSPACE / "handoff_code_executor.json").write_text(...)` 之后、`print(f"OK: handoff written...")` 之前/之后的位置。

把原来的：
```python
(WORKSPACE / "handoff_code_executor.json").write_text(
    json.dumps(handoff, ensure_ascii=False, indent=2)
)
print(f"OK: handoff written to {WORKSPACE / 'handoff_code_executor.json'}")
```

替换为：
```python
(WORKSPACE / "handoff_code_executor.json").write_text(
    json.dumps(handoff, ensure_ascii=False, indent=2)
)

# 给 lead 的结构化决策信号——让 lead 不读 handoff 也能做 Step 1.5 拦截。
# 由 Python 代码确定性生成，不依赖模型推理。lead 解析这个块的存在 +
# critical_count > 0 → 反问用户。详见 spec docs/superpowers/specs/2026-05-11-subagent-file-is-facts-design.md
warnings = handoff.get("data_quality_warnings", [])
critical = [w for w in warnings if w.get("severity") == "critical"]
warn_only = [w for w in warnings if w.get("severity") == "warning"]

print(f"OK: handoff written to {WORKSPACE / 'handoff_code_executor.json'}")
print()
print("[gate_signals]")
print("data_quality:")
print(f"  critical_count: {len(critical)}")
print(f"  warning_count: {len(warn_only)}")
print("  critical_items:")
if critical:
    for w in critical[:5]:
        msg = (w.get("message") or "")[:80]
        print(f"    - {msg}")
else:
    print("    (none)")
status = handoff.get("status", "completed")
if status == "failed":
    validity = "failed"
elif critical:
    validity = "warning"
else:
    validity = "ok"
print(f"statistical_validity: {validity}")
print(f"errors_count: {len(handoff.get('errors', []))}")
```

- [ ] **Step 3: 验证 epm.md 改动 grep**

Run: `grep -n "gate_signals\|critical_count" packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md`

Expected: 至少看到 `[gate_signals]` / `critical_count` / `warning_count` / `statistical_validity` 等字段的输出代码命中。

- [ ] **Step 4: 对其他 5 个范式做相同改动**

按 Step 2 的替换模式（**完全相同的代码块**），处理：

1. `oft.md`
2. `ldb.md`
3. `tst.md`
4. `fst.md`
5. `zero-maze.md`

每个文件的胶水脚本末尾结构都是 `write_text(...)` + `print("OK: ...")`，逐一找到对应位置替换。

实施提示：可以打开 epm.md 改后的版本作为模板，对照修改其他 5 个文件。**代码块在 6 个范式间应当一字不差**——这是有意为之，因为输出契约对 lead 而言是统一的。

- [ ] **Step 5: 验证全部 6 个范式都加上了 [gate_signals] 输出**

Run:
```bash
for f in epm oft ldb tst fst zero-maze; do
    echo "=== $f.md ==="
    grep -c "gate_signals" packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/$f.md
done
```

Expected: 每个文件输出 ≥ 2（一处英文注释 + 一处 print "[gate_signals]"）。如果某个文件输出 0，说明漏改。

- [ ] **Step 6: 在 ethoinsight-code/SKILL.md 加输出约定段**

打开 `packages/agent/skills/custom/ethoinsight-code/SKILL.md`，在"## 通用资源"段下、"## 反模式（永远禁止）"段上插入一段新约定：

```markdown
## 最终消息约定（必读）

胶水脚本 stdout 最后必须包含 `[gate_signals]` 块（已在每个范式 reference 的胶水脚本模板末尾给出代码）。这是 lead 做数据质量决策的依据，不输出会导致 lead 退化到回读 handoff 的兜底路径（仍能跑通但浪费 token）。

输出格式见 `templates/output-contract.md` 的 `[gate_signals]` 段。

由 Python 代码生成，不要靠模型自己加这个块——它一定会忘或写错。
```

- [ ] **Step 7: 在 output-contract.md 加 [gate_signals] 段**

打开 `packages/agent/skills/custom/ethoinsight-code/templates/output-contract.md`，在文件末尾追加：

```markdown

## 胶水脚本 stdout 最终输出契约

`handoff_code_executor.json` 写盘后，**胶水脚本 stdout 必须按以下格式输出 `[gate_signals]` 块**：

```
OK: handoff written to <path>

[gate_signals]
data_quality:
  critical_count: <int>     # 等于 data_quality_warnings 中 severity=="critical" 的条目数
  warning_count: <int>      # severity=="warning" 的条目数
  critical_items:
    - <每条 <80 字的 critical 警告 message，最多 5 条>
    - ...
    （如无 critical 条目，输出"    (none)"占位行）
statistical_validity: ok | warning | failed   # failed = handoff.status=="failed"; warning = 有 critical; 否则 ok
errors_count: <int>         # handoff.errors 数组长度
```

**为什么必须输出**：lead 在 Step 1.5 读这个块的字段做拦截决策，无须 read_file handoff（节省上下文 5-30 KB）。块缺失时 lead 会回退到 read handoff 的兜底路径——能跑通但效率退化。

**确定性约束**：由 Python 代码直接生成，不依赖模型推理。每个范式 reference 的胶水脚本模板已经包含完整的输出代码块，复制使用即可。
```

- [ ] **Step 8: 验证 grep**

Run:
```bash
grep -l "gate_signals" packages/agent/skills/custom/ethoinsight-code/SKILL.md \
                       packages/agent/skills/custom/ethoinsight-code/templates/output-contract.md
```

Expected: 两个文件都命中

- [ ] **Step 9: 干跑一个 epm.md 胶水脚本验证 [gate_signals] 输出格式正确**

切到 backend 目录写个临时验证脚本：

```bash
cd packages/agent/backend
source .venv/bin/activate
python3 -c "
import json

# 构造一个测试 handoff——模拟胶水脚本 dispatcher 输出
handoff = {
    'status': 'completed',
    'data_quality_warnings': [
        {'severity': 'critical', 'metric': 'IID', 'message': 'IID 为常数（单鱼模式 vs 群体模式不匹配）'},
        {'severity': 'warning', 'metric': 'all', 'message': 'n=2 偏小'},
    ],
    'errors': [],
}

# 复制 epm.md 里的胶水脚本 gate_signals 输出段（手工 paste 进来跑一遍）
warnings = handoff.get('data_quality_warnings', [])
critical = [w for w in warnings if w.get('severity') == 'critical']
warn_only = [w for w in warnings if w.get('severity') == 'warning']

print('OK: handoff written to /tmp/fake.json')
print()
print('[gate_signals]')
print('data_quality:')
print(f'  critical_count: {len(critical)}')
print(f'  warning_count: {len(warn_only)}')
print('  critical_items:')
if critical:
    for w in critical[:5]:
        msg = (w.get('message') or '')[:80]
        print(f'    - {msg}')
else:
    print('    (none)')
status = handoff.get('status', 'completed')
if status == 'failed':
    validity = 'failed'
elif critical:
    validity = 'warning'
else:
    validity = 'ok'
print(f'statistical_validity: {validity}')
print(f'errors_count: {len(handoff.get(\"errors\", []))}')
"
```

Expected 输出：
```
OK: handoff written to /tmp/fake.json

[gate_signals]
data_quality:
  critical_count: 1
  warning_count: 1
  critical_items:
    - IID 为常数（单鱼模式 vs 群体模式不匹配）
statistical_validity: warning
errors_count: 0
```

如果输出格式不对（缩进 / 字段名 / critical_items 处理），返回 Step 2 修复模板。

- [ ] **Step 10: 跑全套测试**

Run: `cd packages/agent/backend && make test 2>&1 | tail -10`

Expected: 全部 PASS（skill 文件改动不影响 Python 测试）

- [ ] **Step 11: Commit**

```bash
git add packages/agent/skills/custom/ethoinsight-code/SKILL.md \
        packages/agent/skills/custom/ethoinsight-code/templates/output-contract.md \
        packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md \
        packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/oft.md \
        packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/ldb.md \
        packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/tst.md \
        packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/fst.md \
        packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/zero-maze.md
git commit -m "skill(ethoinsight-code): 胶水脚本模板追加 [gate_signals] 输出

让 code-executor 的最终消息 stdout 含结构化决策信号，lead 不读
handoff 也能做 Step 1.5 拦截。由 Python 代码确定性生成，6 个范式
模板代码统一一致。"
```

---

## Task 8: 在 `task_tool.py` 加 `HANDOFF_FILE_REGISTRY` + `{{handoff://}}` 占位符解析

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py`
- Test: `packages/agent/backend/tests/test_task_tool_handoff_placeholders.py`（新建）

- [ ] **Step 1: 写失败测试**

新建文件 `packages/agent/backend/tests/test_task_tool_handoff_placeholders.py`：

```python
"""Tests for {{handoff://<name>}} placeholder resolution in task_tool.

Placeholder serves two roles:
1. Replace {{handoff://code_executor}} → full workspace path
2. Collect authorized paths for HandoffIsolationProvider
"""

import pytest


def test_resolve_handoff_placeholder_basic():
    """Single placeholder resolves to full path and returns authorized set."""
    from deerflow.tools.builtins.task_tool import _resolve_handoff_placeholders

    prompt = "请分析 {{handoff://code_executor}} 中的数据"
    resolved, authorized = _resolve_handoff_placeholders(prompt)

    assert "/mnt/user-data/workspace/handoff_code_executor.json" in resolved
    assert "{{handoff://" not in resolved
    assert authorized == {"/mnt/user-data/workspace/handoff_code_executor.json"}


def test_resolve_handoff_placeholder_multiple():
    """Multiple placeholders all resolved + all collected in authorized set."""
    from deerflow.tools.builtins.task_tool import _resolve_handoff_placeholders

    prompt = (
        "结合 {{handoff://code_executor}} 的数据和 "
        "{{handoff://data_analyst}} 的解读回答"
    )
    resolved, authorized = _resolve_handoff_placeholders(prompt)

    assert "handoff_code_executor.json" in resolved
    assert "handoff_data_analyst.json" in resolved
    assert authorized == {
        "/mnt/user-data/workspace/handoff_code_executor.json",
        "/mnt/user-data/workspace/handoff_data_analyst.json",
    }


def test_resolve_handoff_placeholder_no_placeholder():
    """Prompt without placeholders: unchanged prompt + empty authorized set."""
    from deerflow.tools.builtins.task_tool import _resolve_handoff_placeholders

    prompt = "这是一个不含占位符的普通 prompt"
    resolved, authorized = _resolve_handoff_placeholders(prompt)

    assert resolved == prompt
    assert authorized == set()


def test_resolve_handoff_placeholder_unknown_name_raises():
    """Unknown subagent name → ValueError (fail-fast on typo)."""
    from deerflow.tools.builtins.task_tool import _resolve_handoff_placeholders

    with pytest.raises(ValueError, match="Unknown handoff subagent 'foo'"):
        _resolve_handoff_placeholders("分析 {{handoff://foo}}")


def test_resolve_handoff_placeholder_duplicate_same_name():
    """Same placeholder appearing twice: both resolved, dedup'd in set."""
    from deerflow.tools.builtins.task_tool import _resolve_handoff_placeholders

    prompt = "{{handoff://code_executor}} 和再次 {{handoff://code_executor}}"
    resolved, authorized = _resolve_handoff_placeholders(prompt)

    # Both occurrences in prompt are replaced
    assert resolved.count("handoff_code_executor.json") == 2
    # Set dedup
    assert authorized == {"/mnt/user-data/workspace/handoff_code_executor.json"}


def test_registry_known_subagents():
    """HANDOFF_FILE_REGISTRY exposes the canonical mapping."""
    from deerflow.tools.builtins.task_tool import HANDOFF_FILE_REGISTRY

    assert HANDOFF_FILE_REGISTRY["code_executor"] == "handoff_code_executor.json"
    assert HANDOFF_FILE_REGISTRY["data_analyst"] == "handoff_data_analyst.json"
    assert HANDOFF_FILE_REGISTRY["report_writer"] == "handoff_report_writer.json"
    assert HANDOFF_FILE_REGISTRY["planning"] == "handoff_planning.json"
```

- [ ] **Step 2: 跑测试看失败**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_task_tool_handoff_placeholders.py -v`

Expected: 全部 FAIL with `ImportError: cannot import name '_resolve_handoff_placeholders'`

- [ ] **Step 3: 在 `task_tool.py` 加注册表和解析函数**

打开 `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py`，在现有 `_SHARED_PLACEHOLDER_RE = re.compile(r"\{\{shared://([^}]+)\}\}")`（行 22）之后插入：

```python

# Handoff-file placeholder registry: subagent name → handoff filename.
# Lead uses {{handoff://<name>}} in task prompt; task_tool resolves it to the
# full workspace path AND adds the path to the per-task authorized_handoff_paths
# set that flows into HandoffIsolationProvider.
#
# Pointing to a path via this placeholder IS the authorization. No placeholder =
# no authorization (subagent cannot read peer handoff files).
HANDOFF_FILE_REGISTRY: dict[str, str] = {
    "code_executor": "handoff_code_executor.json",
    "data_analyst": "handoff_data_analyst.json",
    "report_writer": "handoff_report_writer.json",
    "planning": "handoff_planning.json",
}

_HANDOFF_PLACEHOLDER_RE = re.compile(r"\{\{handoff://([^}]+)\}\}")


def _resolve_handoff_placeholders(prompt: str) -> tuple[str, set[str]]:
    """Replace ``{{handoff://<subagent_name>}}`` with the full workspace path.

    Returns:
        (replaced_prompt, authorized_absolute_paths) — the set is what
        HandoffIsolationProvider uses as its allowlist for the subagent.

    Raises:
        ValueError: if any placeholder references an unknown subagent name
            (fail-fast on typo so lead immediately learns about the error
            rather than silently dispatching a subagent with a broken prompt).
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

- [ ] **Step 4: 在 `task_tool` 函数体里调用新解析函数**

在 `task_tool` 函数体中，找到现有 `prompt = _resolve_placeholders(prompt)` 这一行（行 137 附近，紧邻 `_resolve_placeholders` 调用）。

在它之后**加一行**：

```python
    # Existing: resolve {{shared://...}}
    prompt = _resolve_placeholders(prompt)

    # New: resolve {{handoff://...}} and capture authorized paths
    prompt, authorized_handoff_paths = _resolve_handoff_placeholders(prompt)
```

**注意**: 暂时**不在本 task** 把 `authorized_handoff_paths` 传给 `SubagentExecutor`（SubagentExecutor 此时还不接受该参数）。Task 9 修改 SubagentExecutor 签名，Task 10 再修改这里把参数传过去。本 task 保留变量但暂时不消费。

为避免 lint 报"变量未使用"，可在变量后加一行：

```python
    _ = authorized_handoff_paths  # Threaded into SubagentExecutor in Task 10
```

- [ ] **Step 5: 跑测试验证 PASS**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_task_tool_handoff_placeholders.py -v`

Expected: 6 个测试全 PASS

- [ ] **Step 6: 跑全套测试**

Run: `cd packages/agent/backend && make test 2>&1 | tail -10`

Expected: 全部 PASS。`task_tool` 现有行为不变（authorized_handoff_paths 暂未消费）。

- [ ] **Step 7: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py packages/agent/backend/tests/test_task_tool_handoff_placeholders.py
git commit -m "feat(task_tool): 新增 {{handoff://<name>}} 占位符解析 + HANDOFF_FILE_REGISTRY

返回 (resolved_prompt, authorized_paths) 元组。authorized_paths 将在
Task 10 传给 SubagentExecutor 供 HandoffIsolationProvider 使用。"
```

---

## Task 9: 给 `SubagentExecutor` 加 `authorized_handoff_paths` 参数

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/executor.py:264-302`

> 此 task 仅扩展 `SubagentExecutor.__init__` 签名 + 存为实例属性。**不实例化 Guardrail**——那是 Task 11。
> 这是为了把对受保护文件 `executor.py` 的改动拆为最小单元：现在只加一个 optional 参数，行为完全不变。

- [ ] **Step 1: 写单测确认现有签名 + 期望新签名**

在 `packages/agent/backend/tests/test_subagent_executor_authorized_paths.py`（新建）：

```python
"""Test that SubagentExecutor accepts authorized_handoff_paths and stores it."""

from deerflow.subagents.config import SubagentConfig
from deerflow.subagents.executor import SubagentExecutor


def _minimal_config():
    return SubagentConfig(
        name="test-agent",
        description="test",
        system_prompt="you are a test",
    )


def test_executor_accepts_authorized_handoff_paths():
    """Executor accepts the new parameter as a keyword argument."""
    config = _minimal_config()
    executor = SubagentExecutor(
        config=config,
        tools=[],
        authorized_handoff_paths={"/mnt/user-data/workspace/handoff_code_executor.json"},
    )
    assert executor.authorized_handoff_paths == {
        "/mnt/user-data/workspace/handoff_code_executor.json"
    }


def test_executor_authorized_paths_defaults_to_empty_set():
    """When the parameter is omitted, attribute defaults to empty set."""
    config = _minimal_config()
    executor = SubagentExecutor(config=config, tools=[])
    assert executor.authorized_handoff_paths == set()


def test_executor_authorized_paths_none_normalized_to_empty():
    """Passing None explicitly normalizes to empty set (downstream code can
    treat the attribute uniformly without None-checks)."""
    config = _minimal_config()
    executor = SubagentExecutor(
        config=config, tools=[], authorized_handoff_paths=None
    )
    assert executor.authorized_handoff_paths == set()
```

- [ ] **Step 2: 跑测试看失败**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_subagent_executor_authorized_paths.py -v`

Expected: 失败 with `TypeError: __init__() got an unexpected keyword argument 'authorized_handoff_paths'`

- [ ] **Step 3: 改 `SubagentExecutor.__init__`**

打开 `packages/agent/backend/packages/harness/deerflow/subagents/executor.py`。

找到 `__init__` 签名（行 267-276），改为：

```python
    def __init__(
        self,
        config: SubagentConfig,
        tools: list[BaseTool],
        parent_model: str | None = None,
        sandbox_state: SandboxState | None = None,
        thread_data: ThreadDataState | None = None,
        thread_id: str | None = None,
        trace_id: str | None = None,
        authorized_handoff_paths: set[str] | None = None,
    ):
```

在 docstring 的 Args 段（行 279-286）末尾加一行：

```
            authorized_handoff_paths: Set of absolute workspace paths the
                subagent is authorized to read via read_file. Populated by
                task_tool from {{handoff://<name>}} placeholders in the lead's
                task prompt. None or empty set = no authorization (subagent
                cannot read peer handoff files). Consumed by
                HandoffIsolationProvider attached to the subagent's middleware
                chain (see Task 11).
```

在 `__init__` 函数体内（约行 292），与其他 `self.xxx = xxx` 行并列，加一行：

```python
        self.authorized_handoff_paths = authorized_handoff_paths or set()
```

- [ ] **Step 4: 跑测试验证 PASS**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_subagent_executor_authorized_paths.py -v`

Expected: 3 个测试全 PASS

- [ ] **Step 5: 跑全套测试确保不破坏现有 executor 测试**

Run: `cd packages/agent/backend && make test 2>&1 | tail -20`

Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/executor.py packages/agent/backend/tests/test_subagent_executor_authorized_paths.py
git commit -m "feat(executor): SubagentExecutor 新增 authorized_handoff_paths 参数

参数默认 None（归一化为空 set）。本 task 仅存储属性，
HandoffIsolationProvider 实例化在 Task 11 完成。"
```

---

## Task 10: 把 `authorized_handoff_paths` 从 `task_tool` 透传给 `SubagentExecutor`

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py:125-140`

> 此 task 只是把 Task 8 在 task_tool 里采集的 set 传给 Task 9 加好的 executor 参数。改动只有 2-3 行。

- [ ] **Step 1: 改 `task_tool.py` 中 `SubagentExecutor(...)` 实例化处**

打开 `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py`。

找到 `SubagentExecutor(...)` 实例化处（约行 125-134），现在长这样：

```python
    executor = SubagentExecutor(
        config=config,
        tools=tools,
        parent_model=parent_model,
        sandbox_state=sandbox_state,
        thread_data=thread_data,
        thread_id=thread_id,
        trace_id=trace_id,
    )
```

改为：

```python
    executor = SubagentExecutor(
        config=config,
        tools=tools,
        parent_model=parent_model,
        sandbox_state=sandbox_state,
        thread_data=thread_data,
        thread_id=thread_id,
        trace_id=trace_id,
        authorized_handoff_paths=authorized_handoff_paths,
    )
```

- [ ] **Step 2: 删 Task 8 留下的占位行**

找到 Task 8 留下的：

```python
    _ = authorized_handoff_paths  # Threaded into SubagentExecutor in Task 10
```

删除这一行。

- [ ] **Step 3: 跑测试**

Run: `cd packages/agent/backend && make test 2>&1 | tail -10`

Expected: 全部 PASS。Task 8 / Task 9 单测仍然通过。

- [ ] **Step 4: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py
git commit -m "feat(task_tool): 把占位符采集的 authorized_handoff_paths 透传给 SubagentExecutor"
```

---

## Task 11: 新建 `HandoffIsolationProvider` 并接入 subagent 中间件链

**Files:**
- Create: `packages/agent/backend/packages/harness/deerflow/guardrails/handoff_isolation_provider.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/executor.py:304-323` (即 `_create_agent` 方法)
- Test: `packages/agent/backend/tests/test_handoff_isolation_provider.py`（新建）

- [ ] **Step 1: 写 provider 单测（先确认行为期望）**

新建 `packages/agent/backend/tests/test_handoff_isolation_provider.py`：

```python
"""Tests for HandoffIsolationProvider — gates subagent read_file on handoff_*.json."""

from deerflow.guardrails.handoff_isolation_provider import HandoffIsolationProvider
from deerflow.guardrails.provider import GuardrailRequest


def _request(
    *,
    tool_name: str = "read_file",
    file_path: str = "",
    is_subagent: bool = True,
) -> GuardrailRequest:
    return GuardrailRequest(
        tool_name=tool_name,
        tool_input={"file_path": file_path},
        is_subagent=is_subagent,
    )


def test_lead_calls_always_allowed():
    """Provider only restricts subagent calls; lead reads pass through."""
    p = HandoffIsolationProvider(authorized_paths=set())
    decision = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json",
        is_subagent=False,
    ))
    assert decision.allow is True


def test_non_read_file_tools_always_allowed():
    """Other tools (write_file, ls, bash...) are not gated."""
    p = HandoffIsolationProvider(authorized_paths=set())
    decision = p.evaluate(_request(
        tool_name="write_file",
        file_path="/mnt/user-data/workspace/handoff_code_executor.json",
    ))
    assert decision.allow is True


def test_non_handoff_files_always_allowed():
    """read_file on non-handoff files is not gated."""
    p = HandoffIsolationProvider(authorized_paths=set())
    decision = p.evaluate(_request(file_path="/mnt/user-data/workspace/metrics.csv"))
    assert decision.allow is True


def test_authorized_handoff_path_allowed():
    p = HandoffIsolationProvider(authorized_paths={
        "/mnt/user-data/workspace/handoff_code_executor.json"
    })
    decision = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json"
    ))
    assert decision.allow is True


def test_unauthorized_handoff_path_denied():
    p = HandoffIsolationProvider(authorized_paths={
        "/mnt/user-data/workspace/handoff_data_analyst.json"
    })
    decision = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json"
    ))
    assert decision.allow is False
    assert decision.reasons[0].code == "handoff_isolation.unauthorized"
    assert "handoff_code_executor.json" in decision.reasons[0].message


def test_self_outbox_always_allowed():
    """data-analyst can read its own handoff_data_analyst.json even without
    being in authorized_paths (it just wrote it)."""
    p = HandoffIsolationProvider(
        authorized_paths=set(),
        self_outbox_subagent_name="data-analyst",
    )
    decision = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_data_analyst.json"
    ))
    assert decision.allow is True


def test_self_outbox_with_hyphenated_name():
    """Subagent names use '-' (e.g. 'code-executor'), filename uses '_' (handoff_code_executor.json).
    Provider normalizes."""
    p = HandoffIsolationProvider(
        authorized_paths=set(),
        self_outbox_subagent_name="code-executor",
    )
    decision = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json"
    ))
    assert decision.allow is True


def test_self_outbox_does_not_allow_peer_handoff():
    """data-analyst CANNOT read handoff_code_executor.json via _is_own_handoff."""
    p = HandoffIsolationProvider(
        authorized_paths=set(),
        self_outbox_subagent_name="data-analyst",
    )
    decision = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json"
    ))
    assert decision.allow is False


async def test_aevaluate_delegates_to_evaluate():
    """Async path returns same decision as sync."""
    p = HandoffIsolationProvider(authorized_paths={
        "/mnt/user-data/workspace/handoff_code_executor.json"
    })
    sync = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json"
    ))
    async_d = await p.aevaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json"
    ))
    assert sync.allow == async_d.allow
```

- [ ] **Step 2: 跑测试看失败**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_handoff_isolation_provider.py -v`

Expected: 全部 FAIL with `ImportError` (provider 还没创建)

- [ ] **Step 3: 创建 `handoff_isolation_provider.py`**

新建 `packages/agent/backend/packages/harness/deerflow/guardrails/handoff_isolation_provider.py`：

```python
"""HandoffIsolationProvider — gates subagent reads of peer handoff_*.json files.

Authorization is supplied by task_tool at dispatch time (parsed from
{{handoff://<name>}} placeholders in the lead's task prompt). The provider
DOES NOT parse the prompt text itself — keeping semantics explicit: no
placeholder = no authorization.

This enforces the 'files are facts' principle in mechanism, not just prompt.
Lead is the sole authorizing party; only paths lead names via placeholder
are accessible to the subagent. A subagent may always read its own outbox
(self-validation case).
"""

from __future__ import annotations

from deerflow.guardrails.provider import (
    GuardrailDecision,
    GuardrailReason,
    GuardrailRequest,
)


class HandoffIsolationProvider:
    """Block subagents from reading peer subagents' handoff files unless lead
    has authorized the path via {{handoff://...}} placeholder in task prompt.
    """

    name = "handoff_isolation"

    def __init__(
        self,
        authorized_paths: set[str],
        self_outbox_subagent_name: str | None = None,
    ):
        self.authorized_paths = authorized_paths
        self.self_outbox_subagent_name = self_outbox_subagent_name

    def _is_own_handoff(self, file_path: str) -> bool:
        """Allow subagent to read its own handoff file (it just wrote it).

        e.g. data-analyst writes handoff_data_analyst.json, then may re-read
        for self-validation. This is not "peeking at peer".
        """
        if not self.self_outbox_subagent_name:
            return False
        # Subagent names use hyphens ('code-executor'); filenames use
        # underscores (handoff_code_executor.json). Normalize for comparison.
        normalized = self.self_outbox_subagent_name.replace("-", "_")
        return f"handoff_{normalized}.json" in file_path

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Lead calls are never restricted by this provider.
        if not request.is_subagent:
            return GuardrailDecision(allow=True)
        # Only gate read_file; other tools pass through.
        if request.tool_name != "read_file":
            return GuardrailDecision(allow=True)
        file_path = request.tool_input.get("file_path", "") or ""
        # Only gate handoff_*.json reads; other files unrestricted.
        if "handoff_" not in file_path or not file_path.endswith(".json"):
            return GuardrailDecision(allow=True)
        # Subagent may always read its own outbox.
        if self._is_own_handoff(file_path):
            return GuardrailDecision(allow=True)
        # Check explicit authorization.
        if file_path in self.authorized_paths:
            return GuardrailDecision(allow=True)
        return GuardrailDecision(
            allow=False,
            reasons=[
                GuardrailReason(
                    code="handoff_isolation.unauthorized",
                    message=(
                        f"Subagent attempted to read {file_path} without "
                        f"lead authorization. Authorized paths: "
                        f"{sorted(self.authorized_paths)}. To authorize, "
                        f"lead must include {{{{handoff://<subagent_name>}}}} "
                        f"placeholder in the task prompt."
                    ),
                )
            ],
            policy_id="handoff_isolation",
        )

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Same logic, async signature for GuardrailProvider protocol compliance.
        return self.evaluate(request)
```

- [ ] **Step 4: 跑 provider 单测验证 PASS**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_handoff_isolation_provider.py -v`

Expected: 9 个测试全 PASS

- [ ] **Step 5: 把 provider 接入 `SubagentExecutor._create_agent`**

打开 `packages/agent/backend/packages/harness/deerflow/subagents/executor.py`。

找到 `_create_agent` 方法（行 304-323），里面有：

```python
        from deerflow.agents.middlewares.tool_error_handling_middleware import build_subagent_runtime_middlewares

        # Reuse shared middleware composition with lead agent.
        middlewares = build_subagent_runtime_middlewares(lazy_init=True)
```

在 `middlewares = build_subagent_runtime_middlewares(...)` 这一行**之后**加一段：

```python
        # Attach HandoffIsolationProvider so subagent's read_file on
        # handoff_*.json files is gated by lead's {{handoff://}} authorization.
        from deerflow.guardrails.handoff_isolation_provider import HandoffIsolationProvider
        from deerflow.guardrails.middleware import GuardrailMiddleware

        handoff_isolation = HandoffIsolationProvider(
            authorized_paths=self.authorized_handoff_paths,
            self_outbox_subagent_name=self.config.name,
        )
        middlewares.append(GuardrailMiddleware(provider=handoff_isolation))
```

**重要**: `GuardrailMiddleware` 接受 `provider`, 单测可参考 `packages/agent/backend/packages/harness/deerflow/guardrails/middleware.py` 行 29 的 `__init__(self, provider, *, fail_closed=True, passport=None)`。这里只传 `provider=` 一个关键字参数，其它 default。`is_subagent=True` 标志由 GuardrailMiddleware 内部填充——查看其 `_build_request()` 即可知晓；如果 `is_subagent` 默认为 False，那么需要给 GuardrailMiddleware 加 `passport` 标识。**先按上面 import + append 写法跑测试**，如果集成测试拦截行为不对，再去查 GuardrailMiddleware 怎么标识 subagent 身份。

- [ ] **Step 6: 写集成测试，验证 GuardrailMiddleware 真的拦截**

新建 `packages/agent/backend/tests/test_subagent_handoff_isolation_integration.py`：

```python
"""Integration: HandoffIsolationProvider attached to SubagentExecutor blocks
unauthorized handoff reads at the middleware level.

We don't run a full subagent here; we just verify the provider is in the
middleware chain and would deny an unauthorized read."""

from deerflow.subagents.config import SubagentConfig
from deerflow.subagents.executor import SubagentExecutor
from deerflow.guardrails.handoff_isolation_provider import HandoffIsolationProvider
from deerflow.guardrails.middleware import GuardrailMiddleware


def _minimal_config():
    return SubagentConfig(
        name="data-analyst",
        description="test",
        system_prompt="you are a test",
    )


def test_create_agent_attaches_handoff_isolation_provider():
    """SubagentExecutor._create_agent() appends a GuardrailMiddleware whose
    provider is HandoffIsolationProvider bound to the executor's
    authorized_handoff_paths and config.name."""
    config = _minimal_config()
    executor = SubagentExecutor(
        config=config,
        tools=[],
        authorized_handoff_paths={"/mnt/user-data/workspace/handoff_code_executor.json"},
    )

    # Trigger middleware chain construction. We don't care about the agent
    # itself, only that a HandoffIsolationProvider ended up wired in.
    # _create_agent reads model config, so we test the middleware-build path
    # by inspecting what _create_agent would assemble.
    # Easier: call the internal builder directly.
    from deerflow.agents.middlewares.tool_error_handling_middleware import build_subagent_runtime_middlewares
    middlewares = build_subagent_runtime_middlewares(lazy_init=True)

    handoff_isolation = HandoffIsolationProvider(
        authorized_paths=executor.authorized_handoff_paths,
        self_outbox_subagent_name=executor.config.name,
    )
    middlewares.append(GuardrailMiddleware(provider=handoff_isolation))

    # Find our provider in the chain
    matched = [
        mw for mw in middlewares
        if isinstance(mw, GuardrailMiddleware)
        and isinstance(mw.provider, HandoffIsolationProvider)
    ]
    assert len(matched) >= 1
    provider = matched[-1].provider
    assert provider.authorized_paths == {
        "/mnt/user-data/workspace/handoff_code_executor.json"
    }
    assert provider.self_outbox_subagent_name == "data-analyst"
```

- [ ] **Step 7: 跑集成测试**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_subagent_handoff_isolation_integration.py -v`

Expected: PASS

- [ ] **Step 8: 跑全套测试，验证不破坏现有子代理**

Run: `cd packages/agent/backend && make test 2>&1 | tail -20`

Expected: 全部 PASS

**如果失败**: 可能原因是 `GuardrailMiddleware._build_request` 默认 `is_subagent=False`——这种情况下 HandoffIsolationProvider 的 lead/subagent 区分逻辑失效，所有调用都被当成 lead。**修复方法**：检查 `GuardrailMiddleware.__init__` 是否有 `passport` 参数标识身份，并在 Step 5 实例化时传入对应值（如 `passport=f"subagent:{config.name}"`），同时 HandoffIsolationProvider 改用 passport 而非 `is_subagent` 判断身份。

- [ ] **Step 9: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/guardrails/handoff_isolation_provider.py \
        packages/agent/backend/packages/harness/deerflow/subagents/executor.py \
        packages/agent/backend/tests/test_handoff_isolation_provider.py \
        packages/agent/backend/tests/test_subagent_handoff_isolation_integration.py
git commit -m "feat(guardrails): 新增 HandoffIsolationProvider + 接入 SubagentExecutor 中间件链

机制层防腐烂：subagent read_file handoff_*.json 必须 lead 通过
{{handoff://<name>}} 占位符明确授权，否则返回 error ToolMessage。
subagent 读自己 outbox（self-validation）允许；lead 调用不受限。"
```

---

## Task 12: lead system prompt 改用 `{{handoff://}}` 占位符

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`（多处）

> 此 task 把 Task 5 留下的"v0 占位符"（直接写真实路径）统一升级为 `{{handoff://}}` 形式。这样 lead 派遣 subagent 时既"指了路径"又"授了权"。同步在 prompt 里写明使用规约——让 lead 自己习惯用占位符。

- [ ] **Step 1: 查找当前 prompt 中所有 `handoff_*.json` 真实路径出现**

Run: `grep -n "handoff_code_executor.json\|handoff_data_analyst.json\|handoff_report_writer.json\|handoff_planning.json" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

记录所有命中行。预期 10+ 处。

- [ ] **Step 2: 在 prompt.py 中加入"派遣规约"小节**

在共享 workspace 机制段附近（接近行 376-380，即 `/mnt/shared/` 描述附近），找一个合适的位置（建议紧贴 `prompt 中使用 {{shared://filename}} 占位符...` 那一行之后），新增一段：

```
### 派遣 subagent 时引用 handoff 文件的规约

派遣 subagent 时如果需要让它读上游的 handoff 文件，**必须使用 `{{handoff://<subagent_name>}}` 占位符**，不要在 prompt 中写完整路径：

| subagent | 占位符 | 解析后的路径 |
|---|---|---|
| code-executor | `{{handoff://code_executor}}` | `/mnt/user-data/workspace/handoff_code_executor.json` |
| data-analyst | `{{handoff://data_analyst}}` | `/mnt/user-data/workspace/handoff_data_analyst.json` |
| report-writer | `{{handoff://report_writer}}` | `/mnt/user-data/workspace/handoff_report_writer.json` |
| planning | `{{handoff://planning}}` | `/mnt/user-data/workspace/handoff_planning.json` |

**关键语义**：占位符既"指路径"也"授权"。系统会自动把占位符引用的文件加入 subagent 本次任务的授权读取列表；未通过占位符引用的 handoff 文件，subagent read_file 时会被 Guardrail 拦截。

**正确示例**：
```python
task(subagent_type="data-analyst", description="解读数据",
     prompt="请分析 {{handoff://code_executor}} 中的数据，注意效应量与混杂因素。")
```

**错误示例**（subagent 看到了完整路径但**没有授权**，read_file 会被拦截）：
```python
task(subagent_type="data-analyst", description="解读数据",
     prompt="请分析 /mnt/user-data/workspace/handoff_code_executor.json 中的数据")
```

注：你自己（lead）read_file 时不需要走占位符——你是特权角色，直接 read_file 真实路径即可（如 Step 1.5 异常路径的兜底校验）。
```

注意 prompt.py 是用 Python f-string 渲染的 — 如果上面这段是写在 f-string 模板中（如 `lead_prompt` 函数内），需要把所有 `{` `}` 转义为 `{{` `}}`（Python f-string 规则）。**先查行 376 附近的字符串是否是 f-string**——若是普通字符串，直接复制粘贴；若是 f-string，把上述代码里的所有 `{` 改成 `{{`，`}` 改成 `}}`。

- [ ] **Step 3: 把 Task 5 留下的真实路径示例改为占位符**

| 行号附近 | 当前内容 | 改后 |
|---|---|---|
| 528-529 | `prompt="请分析 /mnt/user-data/workspace/handoff_code_executor.json 中的旷场实验数据。"` | `prompt="请分析 {{handoff://code_executor}} 中的旷场实验数据。"` |
| 572-573 | `prompt="用户问题: NND 偏高说明什么？\n相关数据在 /mnt/user-data/workspace/handoff_code_executor.json 和 /mnt/user-data/workspace/handoff_data_analyst.json，请 read_file 这两份文件后回答。"` | `prompt="用户问题: NND 偏高说明什么？\n相关数据在 {{handoff://code_executor}} 和 {{handoff://data_analyst}}，请 read_file 这两份文件后回答。"` |

> **f-string 注意**: 如果第 528-529、572-573 行所在的字符串是 f-string（不是普通字符串字面量），那么模板中的 `{` 要写成 `{{`，`}` 要写成 `}}`。grep 出 `{{{{shared://code_summary.json}}}}` 这种四花括号正是 f-string 嵌套占位符的写法。**先检查上下文** —— 若示例代码本身被包在 f-string 里，对应模板就要写 `{{{{handoff://code_executor}}}}`（解析后输出 `{{handoff://code_executor}}`，再被 task_tool 解析为路径）。

- [ ] **Step 4: 检查 1075 / 1104 / 1109 / 1115-1117 等位置**

按 spec 改动 6.4 表格逐一处理：

| 行号附近 | 当前内容（v0 形式） | 改后（v2 占位符形式） |
|---|---|---|
| 1075 | `prompt="请分析 /mnt/user-data/workspace/handoff_code_executor.json 中的数据。\\n范式: <范式名>\\n请写出专业的行为学解读..."` | `prompt="请分析 {{handoff://code_executor}} 中的数据。\\n范式: <范式名>\\n请写出专业的行为学解读..."` |
| 1104 | "派遣 knowledge-assistant，prompt 附 handoff_code_executor.json 和 handoff_data_analyst.json 路径" | "派遣 knowledge-assistant，prompt 用 `{{handoff://code_executor}}` 和 `{{handoff://data_analyst}}` 占位符授权" |
| 1109 | `prompt="请基于 /mnt/user-data/workspace/handoff_code_executor.json 的数据和 /mnt/user-data/workspace/handoff_data_analyst.json 的分析解读，撰写..."` | `prompt="请基于 {{handoff://code_executor}} 的数据和 {{handoff://data_analyst}} 的分析解读，撰写..."` |
| 1115 | "用户说"只帮我重新写个报告" + workspace 已有 handoff_code_executor.json + handoff_data_analyst.json → 直接派遣 report-writer" | 同左（这里描述的是 workspace 状态判断，不是 prompt 内容，不改） |
| 1116 | "用户说"帮我重新解读一下" + 已有 handoff_code_executor.json → 直接派 data-analyst" | 同左（同上） |

- [ ] **Step 5: 验证 prompt.py 中没有 lead 派遣 subagent 时直接写完整 handoff 路径的示例**

Run:
```bash
grep -n "/mnt/user-data/workspace/handoff_" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
```

**Expected 输出说明**：
- 应该看到 `Step 1.5` / lead 自己 read_file 的描述里仍有真实路径（lead 是特权角色，**不**改）
- **不应该**看到 lead 派遣 subagent 时 prompt 字符串里直接写真实路径（应都改为 `{{handoff://...}}`）

人工 review 每行，确保上述区分正确。

- [ ] **Step 6: 跑全套测试**

Run: `cd packages/agent/backend && make test 2>&1 | tail -20`

Expected: 全部 PASS

- [ ] **Step 7: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
git commit -m "prompt(lead): 派遣 subagent 时统一用 {{handoff://<name>}} 占位符

派遣规约新增独立章节，所有示例代码 prompt 字段改占位符形式。
lead 自己 read_file 仍用真实路径（特权角色，不通过占位符）。"
```

---

## Task 13: 端到端验收 + grep 清理验证

**Files:** 无新增/修改；纯验证

- [ ] **Step 1: 验证 production 代码 0 处 code_summary 引用**

Run:
```bash
grep -rn "code_summary" packages/agent/backend/packages/harness/deerflow/ --include="*.py"
```

Expected: **完全无输出**（production 代码全部清零）

Run:
```bash
grep -rn "code_summary" packages/agent/backend/tests/ --include="*.py"
```

Expected: 仍有命中（`test_sandbox_tools_security.py` fixture / `test_subagent_prompt_clarity.py` 注释）—— **这些是允许的**，按 spec "单测影响" 段说明，路径校验机制的 fixture 不算 production 引用。

- [ ] **Step 1a: 验证 skill 目录 0 处 code_summary 引用**

Run:
```bash
grep -rn "code_summary" packages/agent/skills/
```

Expected: **完全无输出**（Task 5a 已清理 ethoinsight-planning 中的最后一处残留）

- [ ] **Step 1b: 验证 6 个范式胶水脚本模板都含 [gate_signals] 输出代码**

Run:
```bash
for f in epm oft ldb tst fst zero-maze; do
    count=$(grep -c "gate_signals" packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/$f.md)
    echo "$f.md: $count"
done
```

Expected: 每个文件输出 ≥ 2（注释 + print 语句）。任何一个为 0 都表示 Task 7a 漏改某个范式。

- [ ] **Step 2: 验证 spec 验收条件 1-5**

逐项检查 spec 验收条件：

1. ✅ `grep code_summary` 在 packages/harness/deerflow/ 下无 production 引用 — Step 1 已验证
2. 🔲 knowledge-assistant 追问能读到 handoff_data_analyst.json 的 outlier_findings / method_warnings — **需 dogfooding 实测**，在 Step 8 完成
3. ✅ data-analyst / report-writer / lead agent prompt 不再提 code_summary.json — Step 1 已验证（覆盖全 production）
4. ✅ lead prompt.py 7 处 code_summary 引用全部处理 — Task 4 + Task 5 完成
5. ✅ handoff_schemas.py 含 GateSignals 模型；3 个 subagent prompt 含 [gate_signals] 输出约定 — Task 6 + Task 7 完成

Run:
```bash
grep -l "GateSignals\|gate_signals" \
  packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py \
  packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py \
  packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py \
  packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py
```

Expected: 4 个文件全部命中

- [ ] **Step 3: 验证 spec 验收条件 6-8（task_tool 占位符 + Guardrail）**

```bash
grep -l "HANDOFF_FILE_REGISTRY\|_resolve_handoff_placeholders" \
  packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py
```

Expected: 命中

```bash
grep -l "authorized_handoff_paths" \
  packages/agent/backend/packages/harness/deerflow/subagents/executor.py
```

Expected: 命中

```bash
ls packages/agent/backend/packages/harness/deerflow/guardrails/handoff_isolation_provider.py
```

Expected: 文件存在

- [ ] **Step 4: 跑全套测试，再次确认**

Run: `cd packages/agent/backend && make test 2>&1 | tail -30`

Expected: 全部 PASS（其中包含本 plan 新增的 Task 6 / 8 / 9 / 11 测试）

- [ ] **Step 5: lint**

Run: `cd packages/agent/backend && make lint`

Expected: 0 errors

如有 warning，按 ruff 提示修复后再 commit fix。

- [ ] **Step 6: 验证 lead prompt 改占位符后没有 dangling 真实路径**

人工 review `agents/lead_agent/prompt.py` 中每一处出现 `handoff_code_executor.json` / `handoff_data_analyst.json` / `handoff_report_writer.json` / `handoff_planning.json` 的位置，确认：
- lead 自己 read_file 的位置 → 真实路径（特权角色）
- lead 派遣 subagent 的 prompt 字符串里 → 占位符 `{{handoff://...}}`

Run:
```bash
grep -n "handoff_code_executor.json\|handoff_data_analyst.json\|handoff_report_writer.json\|handoff_planning.json" \
  packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
```

人工 review 每一行 context，给每行打勾。

- [ ] **Step 7: 启动 dev 环境做手工 e2e 验证**

按 backend CLAUDE.md 的指引，启动 dev 环境：

```bash
cd packages/agent
make dev
```

打开 `http://localhost:2026`，做两个测试用例：

**用例 A**：上传 demo-data 里的 shoaling 数据集，让 agent 跑完整分析流程。验证：
- code-executor 完成后**最终消息中含 `[gate_signals]` 块**（由胶水脚本 print 生成）—— 翻 `make dev` 日志或前端最终消息找 `[gate_signals]` 字符串
- code-executor 完成后 lead **没有** read_file handoff_code_executor.json 来做 Step 1.5（在日志里 grep `read_file.*handoff_code_executor` 不应出现 critical_count=0 情况下的命中）
- lead 用 ask_clarification 给出三选一时正常
- 在 ask_clarification 后选 "先帮我解释 XX" 并输入 "为什么 p 不显著"——验证 knowledge-assistant 的回答**含 method_warnings / outlier_findings 的引用**（dogfooding 用户反馈本来就是想要这个）

**用例 B**：构造 critical 数据质量警告场景（用单鱼模式数据触发 IID 常数警告），验证：
- 胶水脚本输出 `[gate_signals]` 块中 `critical_count >= 1`，`critical_items` 列出关键警告条目
- lead 收到 `[gate_signals]` 后**不读 handoff**直接调用 ask_clarification 拦截用户
- lead 在 ask_clarification 中给出三选项（按 prompt.py 现有的 Gate 2 反问语义）

**用例 C**：subagent 偷读测试。构造一个调用：lead 在 task() prompt 里**不**用 `{{handoff://...}}` 占位符，而是直接写完整路径如 `prompt="读 /mnt/user-data/workspace/handoff_data_analyst.json"`。验证 subagent read_file 时被 Guardrail 拦截，返回 error ToolMessage 提示授权要求。

把三次实测的日志/截图归档到 `docs/handoffs/2026-05/`（创建文件 `2026-05-11-files-are-facts-handoff.md` 记录验收结果）。

- [ ] **Step 8: 最终 commit + 写交接文档**

```bash
cat > docs/handoffs/2026-05/2026-05-11-files-are-facts-handoff.md <<'EOF'
# files-are-facts 重构交接（2026-05-11）

## 实施完成情况
- spec: docs/superpowers/specs/2026-05-11-subagent-file-is-facts-design.md (v2.1)
- plan: docs/superpowers/plans/2026-05-11-files-are-facts.md
- 全部 15 task 完成（含 Task 5a / 7a 中途追加）

## 验收结果

### 静态验证
- production 代码 0 处 code_summary 引用 ✅
- skill 目录 0 处 code_summary 引用（Task 5a 清理 ethoinsight-planning） ✅
- 6 个范式胶水脚本模板全部含 [gate_signals] 输出代码（Task 7a） ✅
- handoff_schemas 含 GateSignals ✅
- task_tool 含 HANDOFF_FILE_REGISTRY + {{handoff://}} 占位符 ✅
- HandoffIsolationProvider 实现 + 接入 ✅
- lead prompt 派遣 subagent 全部用占位符 ✅
- make test 全过 ✅
- make lint 0 errors ✅

### dogfooding 实测
- 用例 A（正态流程 + knowledge-assistant 追问 + [gate_signals] 输出验证）：[填写实测结果]
- 用例 B（critical 数据质量拦截 + lead 不读 handoff）：[填写实测结果]
- 用例 C（subagent 未占位符授权时 read_file handoff 被 Guardrail 拦截）：[填写实测结果]

## 后续路径
- {{shared://}} 占位符清理（follow-up）
- inbox/outbox 目录改造（v0.1 之后）
- LocalSandboxProvider 物理隔离（v0.1 之后）
EOF

git add docs/handoffs/2026-05/2026-05-11-files-are-facts-handoff.md
git commit -m "docs(handoff): files-are-facts 重构实施完成 + dogfooding 验收"
```

---

## 自查记录（implementer 跳过；该段供 plan 作者审核）

- ✅ 占位符扫描：无 TBD / TODO / "待定" / "implement later"
- ✅ 类型一致性：`authorized_handoff_paths: set[str]`、`HANDOFF_FILE_REGISTRY: dict[str, str]`、`HandoffIsolationProvider(authorized_paths, self_outbox_subagent_name)` 在 task_tool / executor / provider / 测试中保持一致命名
- ✅ spec 覆盖：
  - 改动 1（knowledge-assistant prompt）→ Task 1
  - 改动 2（data-analyst prompt）→ Task 2
  - 改动 3（report-writer prompt）→ Task 3
  - 改动 4（lead prompt 7 处）→ Task 4（行 308 / 377）+ Task 5（行 412 / 529 / 573 / 1061-1070 / 1083）
  - 改动 5.1（handoff_schemas + GateSignals）→ Task 6
  - 改动 5.2（3 个 subagent prompt 加 [gate_signals]）→ Task 7
  - 改动 5.3（lead Step 1.5 改造）→ Task 5 Step 4（重写 Step 1.5）
  - 改动 6.1（task_tool 占位符）→ Task 8
  - 改动 6.2（SubagentExecutor 透传授权）→ Task 9 + Task 10
  - 改动 6.3（HandoffIsolationProvider 新建）→ Task 11
  - 改动 6.4（lead prompt 改占位符）→ Task 12
  - 验收条件 1-12 → Task 13
- ✅ TDD 顺序：Task 6/8/9/11 都是先写测试看 FAIL → 再实现 → 再看 PASS → commit。Prompt 类 task（1-5/7/12）通过 grep + e2e 验收，无适合写单测的可执行行为
- ✅ 频繁 commit：每个 task 至少 1 个 commit，13 个 task 至少 13 个 commit
- ✅ 受保护文件影响最小化：Task 9 / Task 10 / Task 11 触及 SubagentExecutor.py 和 task_tool.py，但都是**追加性**改动（新参数 / 新 import / 新 middleware append），不改核心调度循环。CLAUDE.md 同步规则的 surgical merge 仍然友好。
- ⚠️ Task 11 Step 5/Step 8 提到了"如果 GuardrailMiddleware 不区分身份"的兜底方案——这是已知不确定性，标注得很明确。Implementer 遇到该情况时按 Step 8 的修复建议处理。
