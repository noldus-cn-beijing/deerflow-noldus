# Handoff — 2026-06-05 E2E Dogfood: Thinking 显示问题 + 速度实测

> 给下一个 AI Agent 的上下文总结

## 背景

用户重新开始 dogfood，发现前端最大问题：**DeepSeek 模型的 thinking 被隐藏了**，导致用户在某些长时间段只看到 loading dots，不知道 agent 在做什么。

用户要求用 `/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点` 的 EPM 数据跑完整 E2E，用 Playwright 观察前端行为并记录总耗时。

## E2E 测试条件

- **数据**：高架十字迷宫_小鼠_三点（1 个 xlsx + 1 个 txt，仅 Subject 1，Drug 处理组，n=1）
- **提示词**：分析这个高架十字迷宫的数据
- **用户回复**：
  - 模板选择："A，就这一个subject，是Drug处理组，没有对照组"（PlusMaze-AllZones）
  - 可视化："A. 是，画出图表"
  - 报告："需要，生成报告"
- **运行环境**：本地 `make dev`（localhost:2026）
- **模型配置**：Lead Agent = deepseek-v4-pro（PatchedReasoningChatOpenAI），Subagents = deepseek-v4-pro-summary（除 data-analyst 用 deepseek-v4-pro）

## E2E 完整时间线

| 时刻 | 耗时 | 事件 |
|------|------|------|
| 09:58:11 | 0:00 | 上传文件 + 提交提示词 |
| ~0:15 | 0:15 | Lead Agent Gate 1：identify_ev19_template → ambiguous |
| ~0:35 | 0:20 | ask_clarification（模板 A/B + 分组确认） |
| ~0:45 | 0:10 | 用户回复，set_experiment_paradigm + prep_metric_plan |
| ~1:05 | 0:20 | 派遣 code-executor |
| ~2:35 | 1:30 | code-executor 完成（5 个指标 + gate_signals） |
| ~2:55 | 0:20 | 展示结果表格 + ask_clarification(viz) |
| ~3:05 | 0:10 | 用户选 A（画图），Lead 试图直接 dispatch chart-maker → **被 Guardrail 拦截** |
| ~3:10 | 0:05 | Guardrail："chart-maker 之前必须先 data-analyst"，Lead 纠正 |
| ~4:40 | 1:30 | data-analyst 完成（关键发现摘要） |
| ~5:25 | 0:45 | chart-maker 完成（2 张柱状图） |
| ~5:50 | 0:25 | Lead 总结 + 反问是否需要报告 |
| ~6:00 | 0:10 | 用户回复"需要，生成报告" |
| ~7:00 | 1:00 | report-writer 完成（6 段完整报告） |
| ~7:20 | 0:20 | Lead 浏览报告 + 最终摘要 |

**总计：约 13 分 30 秒**（含用户交互等待 ~3.5 分钟）
**纯 Agent 工作时间：约 10 分钟**

## 发现的问题

### 问题 1：DeepSeek thinking 在 streaming 期间不是实时显示的（核心问题）

**现象**：用户在 Lead Agent 思考阶段（比如 Gate 1 判断、计划生成）看到的是 loading dots 动画，看不到 agent 的实时思考过程。只有等模型输出完完整内容后，"思考 Lead Agent" 面板才一次性展开。

**前因**：
- DeepSeek API 返回 `reasoning_content` 作为独立的 streaming delta 字段
- `PatchedReasoningChatOpenAI` 正确将其捕获到 `additional_kwargs.reasoning_content`
- `ThinkTagMiddleware` 正确提取 `<think>` 标签内容到同一字段
- 前端 `extractReasoningContentFromMessage` 正确读取该字段
- 前端 `message-list.tsx:338` 在 `hasReasoning(message)` 为 true 时渲染"思考"面板

**根因推测**：
1. LangGraph streaming 的 checkpoint 频率可能不跟随每个 token——`messages-tuple` SSE event 发送的是完整 AIMessage，不是逐 token 的 chunk。如果 checkpoint 在模型输出完整回复后才触发，前端会一次性收到 reasoning + content，而不是分步收到。
2. 前端 `message-list.tsx` 的 `assistant:processing` group 渲染逻辑（line 317-369）同时检查 `hasReasoning` 和 `hasContent`，但在 streaming 初始阶段可能两者都不满足（空 reasoning + 空 content = 不渲染任何东西），直到 reasoning_content 到达后才触发渲染。此时如果 content 仍为空，只渲染"思考"面板——这实际上是正确的，但前提是 LangGraph 在 reasoning token 到达时就推送 event。
3. 没有找到任何"故意隐藏 thinking"的代码——架构是正确的，问题更可能在 streaming 推送频率上。

**后果**：
- 用户在 agent 长时间思考时（DeepSeek thinking 可以持续 30-60 秒）只看到 loading dots
- 用户体验差，不知道 agent 在做什么、是否卡住
- 与 LLM chat UI 的期望不一致（ChatGPT/DeepSeek 官网都会实时流式展示思考过程）

### 问题 2：Subagent 的 thinking 内容在前端不可见

**现象**：data-analyst 使用 deepseek-v4-pro（thinking enabled），但前端只显示"🔬 指标已完成，正在请专家解读"的状态卡片，没有可展开的思考面板。

**前因**：
- data-analyst 是 subagent，通过 `task` tool 被派遣
- subagent 的 thinking 内容写入 `assistant:subagent` group
- `subtask-card.tsx:232` 检查 `step.type === "reasoning"` 来渲染思考内容
- 但 subagent 的 reasoning 是否被正确序列化到 subtask events 中存疑

**后果**：
- data-analyst 的执行时间最长（~90s），但用户完全看不到它在"想什么"
- 和 Lead Agent 形成对比——Lead 有"思考"面板，subagent 没有

### 问题 3：E2E 总耗时 ~13.5 min，未达 10-12 min 目标

**前因**：
- 两个最长瓶颈：code-executor ~90s + data-analyst ~90s
- LLM 推理时间占大头（每个 turn ~13-15s × ~7 个带思考的 turn）
- n=1 场景下 data-analyst 还是完整跑了（虽然解读空间有限）
- perf/e2e-pipeline-speed-optimization 分支未合入（n=1 快速路径、反问合并、并行 bash、batch 读文件）

**后果**：
- 当前 dev 分支仍然 ~21 min → 修复后 ~13.5 min，有改善但未达标
- 合入 perf 分支的 4 项优化后预计可达 8-10 min

### 问题 4（正面发现）：Path Guardrail 正确工作

chart-maker 在 data-analyst 之前被正确拦截：
```
Guardrail denied: tool 'task' was blocked (ethoinsight.path_sequence_violation).
Reason: 按 E2E_FULL_ASKVIZ 路径，chart-maker 之前必须先完成 data-analyst。
```
Lead Agent 收到 deny 后正确纠正，按正确顺序派遣 data-analyst → chart-maker。

## 已验证正常的功能

| 功能 | 状态 |
|------|------|
| Lead Agent "思考"面板（`<think>` 标签提取） | ✅ 正常显示 |
| QualityWarningBanner | ✅ 正常显示（"1 条数据质量警告"） |
| ToolCall 摘要渲染（inspect_uploaded_file/set_experiment_paradigm/prep_metric_plan） | ✅ 正确显示 |
| 图表图片在消息中渲染 | ✅ 正确显示 |
| Artifact 面板展示图表和报告 | ✅ 正确显示 |
| n=1 降级处理 | ✅ 各处如实反映限制 |
| 双源数据（xlsx + txt）交叉验证 | ✅ 正确检测并标注一致 |

## 建议解决方案讨论

以下问题需要和 opus 讨论：

### 问题 1 的解决方向

1. **增加 LangGraph streaming checkpoint 频率**：让 reasoning token 到达时立即推送 `messages-tuple` event，而不是等完整 turn 结束。需要检查 LangGraph server 的 `stream_mode` 配置。

2. **前端在 streaming 阶段主动显示"思考中"指示器**：当 `thread.isLoading` 为 true 且没有新消息时，显示一个 animated "Agent 正在思考..." 而非空白 loading dots。这是前端改动，相对简单。

3. **前端在收到仅有 reasoning 的 partial message 时立即渲染**：目前 `message-list.tsx` 在 `assistant:processing` group 中，如果 `hasReasoning` 为 true 但 `hasContent` 为 false，只渲染"思考"面板——这应该已经是正确的。问题可能在 SSE event 的推送频率上。

4. **使用 LangGraph 的 `custom` stream mode 单独推送 reasoning events**：后端在 `ThinkTagMiddleware` 或 `PatchedReasoningChatOpenAI` 中 emit custom event，前端监听并实时更新。

### 问题 2 的解决方向

1. **Subagent executor 在 task events 中包含 reasoning 内容**：`SubagentExecutor` 已有 `task_running` events，可以在其中附带 reasoning 片段。
2. **`subtask-card.tsx` 实时展示 subagent reasoning**：当前只展示已完成 step 的 reasoning，需要支持 streaming 中的 partial reasoning。

### 问题 3 的解决方向

1. **合入 `perf/e2e-pipeline-speed-optimization` 分支**：n=1 快速路径 + 反问合并 + 并行 bash + batch 读文件。
2. **n=1 场景跳过 data-analyst**：当 `statistical_validity: skipped` 时，data-analyst 的额外解读价值有限，可直接跳转到 chart-maker。

## 建议接手路径

1. 读本文档了解问题全貌
2. 验证问题 1 根因：检查 LangGraph SSE streaming 的 checkpoint 频率，确认 reasoning_content 是否延迟推送
3. 验证问题 2：检查 subagent executor 的 event 序列化是否包含 reasoning
4. 根据 opus 的方案决策实施修复

## 相关文件

| 文件 | 用途 |
|------|------|
| `packages/agent/backend/packages/harness/deerflow/models/patched_reasoning.py` | DeepSeek reasoning_content 捕获 |
| `packages/agent/backend/packages/harness/deerflow/agents/middlewares/think_tag_middleware.py` | `<think>` 标签提取 |
| `packages/agent/frontend/src/core/messages/utils.ts` | reasoning 提取/判断逻辑 |
| `packages/agent/frontend/src/components/workspace/messages/message-list.tsx` | 前端消息渲染（行 317-369 处理 assistant:processing group） |
| `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx` | Subagent 卡片渲染（行 232 处理 reasoning step） |
| `packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx` | LeadAgentThinkingBlock（行 411+ 渲染"思考 Lead Agent"面板） |
| `packages/agent/backend/packages/harness/deerflow/subagents/executor.py` | Subagent 执行引擎（event 序列化） |
| `packages/agent/config.yaml` | 模型配置（deepseek-v4-pro 使用 PatchedReasoningChatOpenAI） |
| `docs/plans/2026-06-04-e2e-latency-bottleneck-analysis.md` | E2E 耗时瓶颈分析 |
| `docs/superpowers/specs/2026-06-04-e2e-pipeline-speed-optimization-spec.md` | E2E 加速 spec |
