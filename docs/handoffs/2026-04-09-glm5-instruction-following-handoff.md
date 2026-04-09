# GLM-5.1 Subagent 指令遵循改进 — 交接文档

> 日期: 2026-04-09
> 上一份交接: `docs/handoffs/2026-04-09-glm5-timeout-resilience-handoff.md`

---

## 1. 当前任务目标

**问题**: 上传旷场实验数据后，code-executor subagent 完全无视 system prompt 中的工作流程——没有调用 `get_analysis_template`，而是自己探索文件系统、反复 makedirs、从头写代码，跑了 22 轮触发 `GraphRecursionError`。

**根因**: GLM-5.1 对复杂 system prompt 的指令遵循能力不足，加上 prompt 中大量"不要做X"的否定指令反而激活了模型去试探这些行为。

**预期产出**: 通过 prompt 精简 + 正面指令改写 + 代码层硬限制，确保 code-executor 第一个 tool call 就是 `get_analysis_template`，整体 AI message ≤ 10 轮。

**完成标准**: 清理 checkpoint 后重新测试，code-executor 按 checklist 执行，无 `GraphRecursionError`。

---

## 2. 当前进展

### Prompt 层面 ✅ 全部完成

| 编号 | 任务 | 文件 | 状态 |
|------|------|------|------|
| 1.1 | Lead Agent 路由意图确认锁 | `prompt.py:219-227` | ✅ |
| 1.2 | 契约表加失败降级列 | `prompt.py:246-252` | ✅ (用户手动编辑) |
| 1.3 | Code-executor prompt 精简 (~3K→~1.5K tokens) + max_turns 15→10 | `code_executor.py` | ✅ |
| 1.4 | 编排示例替换为行为学场景 (腾讯股价→旷场/EPM/知识问答) | `prompt.py:298-315` | ✅ |
| 1.5 | 澄清场景示例替换为行为学场景 | `prompt.py:425-478` | ✅ |
| 1.6 | 全局否定指令→正面表述 ("禁止"→"工作范围"/"执行原则") | 所有 subagent + prompt.py | ✅ |
| 1.7 | data-analyst + report-writer 加 noldus-kb MCP 工具 | `data_analyst.py`, `report_writer.py` | ✅ |

### 代码层面 ✅ 全部完成（上一轮会话已实现）

| 编号 | 任务 | 文件 | 状态 |
|------|------|------|------|
| 2.1 | AI message 硬限制 (超过 max_turns 时 break) | `executor.py:332-337` | ✅ |
| 2.2 | recursion_limit 公式 `max_turns * 3` → `max_turns * 2 + 1` | `executor.py:271` | ✅ |

---

## 3. 关键上下文

### 核心发现：GLM-5.1 的 "否定指令反转" 问题
- GLM-5.1 对"不要做X"、"禁止Y"类指令会反向激活该行为
- **解决方案**: 所有 subagent prompt 和 lead agent prompt 中的否定指令全部改为正面表述
- 例: "不得跳过 Step 2" → "Step 2 必须首先执行"; "禁止读原始数据" → "唯一数据来源是 code_summary.json"
- 已记录到 memory: `~/.claude/projects/-home-qiuyangwang/memory/feedback_positive_prompting.md`

### 架构背景
- **DeerFlow 框架**: LangGraph-based multi-agent system
- **Lead Agent**: 调度员角色，通过 `task()` tool 派遣 subagent
- **4 个 Noldus subagent**: code-executor, data-analyst, report-writer, knowledge-assistant
- **LangGraph recursion_limit**: 计算 node steps (model=1 + tools=1 per turn)，不是 AI message 数
- **`{{shared://filename}}`**: `task_tool.py:22-31` 正则替换机制，subagent 收到的是 `/mnt/shared/filename`

### 已做的关键决定
1. data-analyst 和 report-writer 的 `tools` 从显式列表改为 `None`（继承所有工具）+ `disallowed_tools` 黑名单，以获取 noldus-kb MCP 工具
2. 编排指南保留通用框架（DECOMPOSE/DELEGATE/SYNTHESIZE + 并发限制），但示例全部替换为行为学场景
3. 中间件层的首次 tool call 校验（Plan 2.4）暂不实现，先看 prompt + 硬限制的效果

---

## 4. 关键发现

### 修改的文件清单

| 文件 | 改动要点 |
|------|----------|
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | 路由回退规则、契约表改"工作范围"列、编排示例换行为学场景、澄清示例换行为学场景、否定指令正面化 |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` | prompt 精简为 checklist 格式、"绝对禁令"→"执行原则"、max_turns 15→10 |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` | "禁止"→"工作范围"、加 noldus-kb MCP 工具、tools=None + disallowed_tools |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py` | "禁止"→"工作范围"、加 noldus-kb MCP 工具、tools=None + disallowed_tools |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py` | "禁止"→"工作范围"、否定工具调用建议→正面表述 |
| `packages/agent/backend/packages/harness/deerflow/subagents/executor.py` | AI message 硬限制 + recursion_limit 公式修正（上一轮已完成） |

### prompt.py 结构（供参考）

- `_build_subagent_section()` (167-392): 生成 `<subagent_system>` 块
  - 167-206: 动态 subagent 列表构建
  - 208-266: Noldus 调度规则（路由判断、契约表、共享 workspace、输出规则）
  - 267-392: 通用编排指南（已替换为行为学示例）
- `SYSTEM_PROMPT_TEMPLATE` (395-579): prompt 骨架模板
  - `<clarification_system>` (412-479): 已替换为行为学澄清示例
  - `{orchestration_guide}` (566): Noldus 端到端流水线 Step 0-4
- `apply_prompt_template()` (753+): 最终组装函数

---

## 5. 未完成事项

### 高优先级
1. **端到端验证测试** — 清理 checkpoint → 重启 → 上传旷场数据 → 检查 log
   - code-executor 第一个 tool call 是否为 `get_analysis_template`
   - AI message 总数是否 ≤ 10
   - 无 `GraphRecursionError`
   - data-analyst 和 report-writer 是否能成功使用 noldus-kb

### 中优先级
2. **首次 tool call 校验中间件** (Plan 2.4) — 如果测试后 code-executor 仍不调 `get_analysis_template`
   - 利用 `AgentMiddleware.after_model()` 钩子
   - 在第一次 LLM 响应后检查 tool_calls，如果不是 `get_analysis_template` → 替换为错误消息
   - 文件位置: 新建 `packages/harness/deerflow/subagents/middlewares/first_tool_validator.py`

### 低优先级
3. **citations 区块** (`prompt.py:503-564`) 仍有一些 "NEVER"/"DO NOT" — 可在后续统一正面化
4. **report-writer max_turns=15** 偏高，正常流程 4-5 步足够，可考虑降到 8

---

## 6. 建议接手路径

### 第一步：验证测试
```bash
cd /home/qiuyangwang/noldus-insight
make stop
rm -f packages/agent/backend/.deer-flow/checkpoints.db*
rm -rf packages/agent/backend/.deer-flow/threads/*
make dev
```

然后在 UI 中：
1. 新建 thread
2. 上传旷场实验数据
3. 发送"帮我分析这些数据，Subject 1-3 是对照组，4-6 是实验组"
4. 检查 `packages/agent/logs/langgraph.log`

### 需要重点关注的 log 关键词
- `get_analysis_template` — code-executor 的第一个 tool call
- `captured AI message #` — 计数是否 ≤ 10
- `GraphRecursionError` — 应该不再出现
- `noldus-kb` / `search_knowledge` — data-analyst 和 report-writer 是否调用了 MCP 工具

### 如果测试失败
- 检查 code-executor 的第一个 tool call 是什么
- 如果是 `ls` / `bash` / `read_file` → 实现 Plan 2.4 的首次 tool call 校验中间件
- 如果是其他问题 → 读 log 诊断

---

## 7. 风险与注意事项

1. **`tools=None` 的副作用**: data-analyst 和 report-writer 现在继承了所有工具（包括 bash、str_replace 等），虽然通过 `disallowed_tools` 过滤了，但如果 `_filter_tools()` 逻辑有 bug，可能泄露不该有的工具。验证时检查 log 中 subagent 实际可用的工具列表。

2. **noldus-kb MCP 工具可能增加 subagent 轮次**: data-analyst 和 report-writer 原来 3-4 轮就够，加了 MCP 查询后可能需要更多轮。如果发现超时，考虑增加 max_turns 或在 prompt 中限制查询次数（"最多查询 2 次知识库"）。

3. **prompt.py 中 `{direct_execution_example}` 和 `{direct_tool_examples}` 变量**: 这两个仍在 CRITICAL WORKFLOW 部分引用，但对应的 counter-example 代码块已被删除。如果通用编排的 CRITICAL WORKFLOW 部分仍然引用它们但找不到匹配的模板变量，可能会报 KeyError。需要确认 f-string 中这些变量是否还在使用。

4. **已验证的方向**: `{{shared://filename}}` 占位符机制工作正常（`task_tool.py:22-31`），无需修改。DeepSeek 对此的担忧不成立。

---

## 下一位 Agent 的第一步建议

1. 读取本文档了解全貌
2. 运行验证测试（第 6 节的 bash 命令）
3. 检查 log 确认改动生效
4. 如果 code-executor 仍然不调 `get_analysis_template`，实现首次 tool call 校验中间件
5. 如果一切正常，考虑处理低优先级事项（citations 正面化、report-writer max_turns 调整）
