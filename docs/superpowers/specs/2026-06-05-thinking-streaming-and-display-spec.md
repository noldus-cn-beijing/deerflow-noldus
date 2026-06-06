# Spec：Thinking 流式与显示修复 — Lead 实时思考 + Subagent 思考可见

> **状态**：待实施 | **作者**：调研/诊断 agent | **日期**：2026-06-05
> **实施者**：新 agent | **预计规模**：中（前端为主 2-4 文件；后端可选 1 文件 surgical）
> **风险**：低（前端纯渲染逻辑）~中（若做后端 subagent 流式，碰受保护文件 executor.py）
> **承接**：docs/handoffs/2026-06/2026-06-05-e2e-dogfood-thinking-display-issues.md

---

## 0. 这份 spec 解决什么、不解决什么

**解决**：
1. **Lead Agent 思考阶段（30-60s）实时流式展示**，而不是只看 loading dots、思考面板在 turn 结束才一次性展开。
2. **Subagent（仅 data-analyst，~90s）思考过程对用户可见**（至少在完成时可展开，理想是流式）。
   - 注：code-executor / chart-maker / report-writer 使用 `deepseek-v4-pro-summary`（`supports_thinking: false`，deepseek-v4-flash），不产 thinking，不存在"隐藏"问题。

**明确不解决（不要混淆）**：
- ❌ 不是修后端 reasoning 捕获 —— `PatchedReasoningChatOpenAI` + `ThinkTagMiddleware` + `data-analyst thinking_enabled` 已全部正确（见 §2 证据）。
- ❌ 不是 E2E 速度优化（问题 3 走 `perf/e2e-pipeline-speed-optimization` 分支，与本 spec 正交）。
- ❌ 不改 DashScope/模型配置 —— 已实测模型按 token 增量流式 reasoning（§2.1）。

---

## 1. 问题描述（现状 vs 期望）

### 问题 1：Lead thinking 流式期间不实时显示

| | 现状 | 期望 |
|---|---|---|
| 思考阶段（Gate 1 判断/计划生成，30-60s） | 只见 loading dots + 一个**默认折叠**的"思考 Lead Agent"灰条；思考正文不可见 | 思考正文随 token 增量出现（边想边显示），用户看得到 agent 在想什么 |
| turn 结束 | 思考面板"一次性展开"（因为消息从 processing group 切到 assistant group，换了个默认展开的组件） | 平滑延续，不出现"突然展开"的跳变 |

### 问题 2：Subagent thinking 不可见

| | 现状 | 期望 |
|---|---|---|
| data-analyst 执行（~90s） | 只见"🔬 正在请专家解读"状态卡片（默认折叠），无思考内容 | 卡片内可展开看到 data-analyst 的推理；理想是流式；至少完成时有完整推理可看 |

---

## 2. 根因分析（代码引用 + 证据）

### 2.1 关键证据：后端 reasoning 是增量流式的（实测）

对 DashScope `deepseek-v4-pro` + `enable_thinking:true` 做 streaming 探针：
```
reasoning: 26 chunks, 211 chars, first@2.9s
content:    6 chunks,  39 chars, first@5.2s
```
**结论：reasoning_content 按 26 个增量 delta 流式返回，且在 content 之前。** 后端"checkpoint 才推送"的假设被排除。

后端链路逐环已核实可增量传递 reasoning：
| 环节 | 位置 | 行为 |
|---|---|---|
| DashScope delta.reasoning_content | — | 26 个增量 chunk |
| 注入 additional_kwargs | `models/patched_reasoning.py:63-90` `_convert_chunk_to_generation_chunk` | 每个 chunk 把 reasoning 累加进 `additional_kwargs.reasoning_content` |
| on_llm_new_token 逐 chunk 触发 | `langchain_core/language_models/chat_models.py:689` | **即使 content 为空**也对每个 chunk 调用 |
| LangGraph emit messages-tuple | `langgraph/pregel/_messages.py:145-158` `StreamMessagesHandler.on_llm_new_token` | 无条件 emit 每个 `chunk.message` |
| additional_kwargs 跨 chunk 合并 | `langchain_core/utils/_merge.py:45-64` `merge_dicts` | string 字段（含 reasoning_content）拼接累加 |
| 前端读取 reasoning | `frontend/src/core/messages/utils.ts:343-363` `extractReasoningContentFromMessage` | 优先读 `additional_kwargs.reasoning_content` |
| output_version | config.yaml 未设 → 默认 v0 | reasoning 留在 additional_kwargs，不进 content_blocks |

→ **后端到前端 SDK 的链路完全正确，reasoning 增量到达 `thread.messages`。**

### 2.2 问题 1 真根因：前端 MessageGroup 思考面板「流式期默认折叠」，与最终答案组件「默认展开」不对称

流式/中间步骤阶段，带 reasoning 且无 content/无 tool_call 的 AIMessageChunk 进入 `assistant:processing` group（`message-list.tsx:111-124` `hasReasoning(message) || hasToolCalls(message)` 为真）→ 渲染 `MessageGroup`（`message-list.tsx:338-345`）。

在 `MessageGroup` 里，纯思考（无尾随 tool call）的 reasoning 成为 `lastReasoningStep`（`message-group.tsx:71-79`），渲染为一个**默认折叠**的"思考 Lead Agent"按钮：
```tsx
// message-group.tsx:56-57
const [showLastThinking, setShowLastThinking] = useState(
  env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true",   // 运行时 = false → 折叠
);
// message-group.tsx:~200-218：仅当 showLastThinking 才渲染 reasoning 正文
```
→ 流式期间 reasoning 正文被折叠藏起，用户只见折叠灰条 + 全局 loading dots（`message-list.tsx:396` `StreamingIndicator`）。

turn 结束、最终答案（有 content、无 tool_call）到达时，消息改归 `assistant` group（`message-list.tsx:105-110`）→ 渲染 `MessageListItem` → `MessageContent_` → `ReasoningPanel`，其 `showThinking` **默认展开**：
```tsx
// message-list-item.tsx:426
const [showThinking, setShowThinking] = useState(true);   // 展开
// message-list-item.tsx:431 open={true}
```
→ **两个组件默认折叠态不一致**，造成"思考结束才一次性展开"的错觉。本质：流式期的实时思考一直在更新，只是被默认折叠藏起来了。

### 2.3 问题 2 真根因：subagent reasoning 仅在 turn 结束时单次推送 + 卡片默认折叠

reasoning 捕获正确：`data_analyst.py:310 thinking_enabled=True`，`executor.py:667` 按 config 开 think，reasoning 进 `additional_kwargs.reasoning_content`。

事件携带正确：`task_tool.py:540-556`，每个**新**AIMessage 发一个 `task_running` 事件，`message` 字段是 `last_message.model_dump()`（含 additional_kwargs）。前端 `hooks.ts:499-513 onCustomEvent` 收 `task_running` → `updateSubtask({latestMessage})`；`context.tsx useUpdateSubtask` 把 latestMessage 追加进 `task.messages`（按 id 去重/就地替换）；`subtask-card.tsx:166 SubtaskCoTTimeline` → `convertToSteps` → `CoTStepRenderer`（`subtask-card.tsx:232`）渲染 reasoning step。

**但两处导致不可见**：
1. **单次、滞后**：`executor.py:898 agent.astream(..., stream_mode="values")` —— values 模式每次 yield 是完整 state 快照，`result.ai_messages` 只在一个 turn 的 AIMessage **完整**后才追加（`executor.py:913-931`）。data-analyst 被 prompt 要求"一次性完成核心分析推理（单轮 LLM 思考）"（`data_analyst.py:102`）→ 整段 reasoning 在 ~90s 后随**唯一一个** AIMessage 一次性出现，`task_running` 只发一次。流式期间 `task.messages` 为空，timeline 不渲染。
2. **卡片默认折叠**：`subtask-card.tsx:52 const [collapsed, setCollapsed] = useState(true)` → 即使 reasoning 到了也要用户手动展开；折叠态只显示 `getStageBroadcastForSubagent`（"正在请专家解读"）。

---

## 3. 解决方案设计

### 问题 1 —— 方案对比

#### 方案 1A（推荐）：流式期 MessageGroup 思考面板默认展开 + 修复折叠态不对称

**改动点**：`frontend/src/components/workspace/messages/message-group.tsx`
- 让 `lastReasoningStep` 的展开默认值在 `isLoading`（流式中）时为 `true`，与最终答案的 `ReasoningPanel` 对齐。
- 同理 `showAbove`（历史思考）可保持折叠，只动"当前/最后一段思考"。

关键逻辑（伪代码）：
```tsx
// message-group.tsx
const [showLastThinking, setShowLastThinking] = useState(
  env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true" || isLoading,  // 流式中默认展开
);
// 注意：useState 初值只在 mount 时取一次。若组件先以非 loading mount，
// 需用 useEffect 在 isLoading 变 true 时同步 setShowLastThinking(true)（仅一次，
// 不要覆盖用户手动折叠）：用一个 ref 记录"用户是否手动操作过"，未操作过才跟随 isLoading。
```

**优劣**：
- ✅ 最小改动、纯前端、无后端风险；直接消除"一次性展开"错觉。
- ✅ 与 `ReasoningPanel`（最终答案，默认展开）行为一致，体验连贯。
- ⚠️ 需处理 useState 初值只取一次的 React 陷阱（用 useEffect + 用户操作 ref 兜底）。

#### 方案 1B：流式期统一用一个 ReasoningPanel 风格组件渲染思考（合并两套）

把 `assistant:processing` group 的思考渲染也改用 `message-list-item.tsx` 的 `ReasoningPanel`（默认展开、带 Lead Agent 徽章、isStreaming 透传），让"流式中"和"最终答案"用**同一个**组件，彻底消除两套默认态。

**优劣**：
- ✅ 根治不对称（单一组件单一默认态）。
- ⚠️ 改动更大：`MessageGroup` 同时承载 tool call 时间线，不能整体替换；需要在 processing group 渲染分支里，把"reasoning 步骤"单独抽出来用 ReasoningPanel，tool call 仍走 MessageGroup —— 触及 `message-list.tsx:317-389` 渲染结构。回归面更宽。

**结论**：先做 **1A**（小、稳、直接见效）；1B 作为后续统一重构（与 2026-05-21 "统一双 thinking layout" 同向，可并入那条线）。

### 问题 2 —— 方案对比

#### 方案 2A（推荐第一步）：SubtaskCard 流式期默认展开 + 完成后保留可展开

**改动点**：`frontend/src/components/workspace/messages/subtask-card.tsx`
- `collapsed` 初值：`in_progress` 时默认展开（`useState(task.status !== "in_progress")` 起步，配合 useEffect 同步），让用户在 subagent 跑动时就能看到 timeline（含 reasoning step 一到就显示）。
- 完成后可折叠，但保留"展开看推理"的入口（已有）。

**优劣**：
- ✅ 纯前端、零后端风险。
- ✅ 与问题 1A 的体验一致（思考默认可见）。
- ⚠️ **单靠 2A 仍滞后**：因为 reasoning 仍只在 ~90s 后随单个 task_running 事件到达（值模式 + 单轮）。2A 让"到了就显示"，但"何时到"不变。需要配合 2B 或可接受"完成时一次性可见"。

#### 方案 2B（推荐第二步，若要真流式）：executor 增量推送 data-analyst reasoning

> **适用范围**：仅 data-analyst（唯一使用 deepseek-v4-pro + thinking 的 subagent）。code-executor / chart-maker / report-writer 使用 deepseek-v4-pro-summary（`supports_thinking: false`），无 thinking 可推送，此方案对它们无影响。

让 subagent 执行也按 token/部分消息流式，使 `task_running` 在 reasoning 累积过程中多次推送。

**改动点（受保护文件，surgical）**：`backend/.../subagents/executor.py`
- `_aexecute` 的 `agent.astream(...)` 改为 `stream_mode=["values","messages"]`（或单独再开一路 messages），从 `messages` 流拿到 `AIMessageChunk`（含增量 reasoning_content）。
- 在循环里：对 `messages` 模式产出的部分 chunk，按 message id 累积/更新，并节流（如每 N 个 chunk 或每 500ms）调用 `task_tool` 提供的回调/writer 推一个"部分 message"。
- `task_tool.py` 当前是后台轮询 `result.ai_messages` 后再 `writer({"type":"task_running", "message":...})`（`task_tool.py:540-556`）。要支持 partial，需要把"部分消息"也经由这条路推出去 —— 可在 `SubagentResult` 上加一个"最新 partial message"字段（带版本号/序号），轮询时若 partial 更新则推 `task_running`（前端 `context.tsx` 已支持同 id 就地替换，见 `useUpdateSubtask` 的 `existingIndex >= 0` 分支）。

前端侧已基本就绪：`context.tsx useUpdateSubtask` 对相同 message.id 的 latestMessage 做就地替换（streaming chunk 友好），`subtask-card.tsx` 的 timeline 会重渲染。所以 2B 主要是**后端把 partial 推出来**。

**优劣**：
- ✅ 真正实现 subagent 边想边显示。
- ⚠️ 碰受保护文件 `executor.py`（sync 守护成本）；轮询→推送有 5s 间隔（`task_tool.py:592 asyncio.sleep(5)`），要么缩短轮询间隔、要么改事件驱动，否则"流式"粒度仍是 5s。
- ⚠️ `result.ai_messages` 去重逻辑（`executor.py:920-931`）和 max_turns 早停（`executor.py:933-936`）基于"完整 AIMessage 计数"，引入 partial 不能破坏这套计数 —— partial 只用于 UI 推送，**不计入** ai_messages 的 turn 计数（保持两条独立轨道）。

**结论**：先做 **2A**（让"到了即显示"+ 默认展开，立即改善"完全看不到"）；若用户要求真流式，再做 **2B**（后端 surgical + 节流推送）。2A 单独即可把"完全黑盒 90s"变成"完成即展开可见"，性价比高、零风险。

### 推荐组合

- **必做**：1A + 2A（纯前端，零后端风险，直接解决"看不到/一次性展开"）。
- **可选增强**：2B（subagent 真流式，碰受保护文件，按用户对"实时性"的要求决定）。
- **后续重构**：1B（统一思考组件，并入 2026-05-21 双 layout 统一线）。

---

## 4. 影响范围

| 文件 | 改动 | 影响现有功能？ |
|---|---|---|
| `frontend/.../message-group.tsx` | 1A：`showLastThinking` 流式默认展开 + useEffect 同步 + 用户操作 ref | 仅改思考面板默认折叠态；tool call 时间线、历史折叠（showAbove）不变 |
| `frontend/.../subtask-card.tsx` | 2A：`collapsed` 在 in_progress 时默认展开 + useEffect 同步 | 仅改卡片默认折叠态；完成态/失败态/result 渲染不变 |
| `backend/.../subagents/executor.py`（仅 2B） | 加 `messages` 流 + partial 推送；**不动** turn 计数/seal-resume/recursion_limit | 受保护文件，需 surgical；partial 不计入 ai_messages 计数 |
| `backend/.../tools/builtins/task_tool.py`（仅 2B） | 轮询时若 partial 更新则推 `task_running` | 终态事件/usage/cleanup 路径不变 |

**不影响**：
- 后端 reasoning 捕获链（patched_reasoning / think_tag_middleware / 模型配置）—— 全部已正确，不动。
- `extractReasoningContentFromMessage` / `getMessageGroups` 分组逻辑（已正确路由 reasoning）。
- `ReasoningPanel`（最终答案路径，已默认展开）。
- IM channels（feishu/slack）—— 它们走 `runs.stream` 自行累积文本，不依赖这两个组件。

---

## 5. 测试策略

### 5.1 前端单测（vitest，`frontend/src/.../*.test.ts(x)`）
- **1A**：给 `MessageGroup` 传一条「仅 reasoning、无 content、无 tool_call」的 AIMessage + `isLoading=true`，断言 reasoning 正文**可见**（不被折叠）；`isLoading=false` 且无用户操作时按既有默认。模拟用户点击折叠后，再来新 chunk **不**强制展开（尊重用户操作）。
- **2A**：`SubtaskCard` 在 `task.status==="in_progress"` 时断言 timeline 区域展开；`completed` 时默认折叠但可展开。
- 复用既有 `utils.test.ts` 锚点：`extractReasoningContentFromMessage` 对 `additional_kwargs.reasoning_content` 的读取不回归。

### 5.2 后端单测（仅 2B，`backend/tests/`）
- executor `stream_mode=["values","messages"]` 下，partial chunk 累积后 `result.ai_messages` 的**完整消息计数不变**（max_turns 早停语义不破）。
- partial 推送节流：mock writer，断言 reasoning 流式过程中 `task_running` 被多次调用且 message.id 稳定（前端就地替换前提）。
- **全量回归铁律**（改 executor 是共享逻辑）：`cd packages/agent/backend && PYTHONPATH=packages/harness:. .venv/bin/python -m pytest tests/ -q`，确认 failed 数不超过既有 pre-existing。

### 5.3 Dogfood（必做，环境能跑真 deepseek）
1. `cd packages/agent && make stop && make dev`（改前端需前端重建；改 executor 必须重启）。
2. 跑 EPM n=1 dogfood（同 handoff 数据 `高架十字迷宫_小鼠_三点`）。用 Playwright/手动观察：
   - **问题 1**：Gate 1 / 计划生成阶段，思考正文**随 token 增量出现**（不再只有 loading dots + 折叠条）；turn 结束**无**"突然一次性展开"跳变。
   - **问题 2（2A）**：data-analyst 卡片在 in_progress 时默认展开；完成时推理可见。（2B 若做：观察 ~90s 内推理是否分批流式刷新。）
3. 回归确认：最终答案的 `ReasoningPanel` 仍正常；图表/报告/quality banner 不受影响。

---

## 6. 实施顺序建议

1. **第一步（纯前端，立即见效，零后端风险）**：1A（message-group.tsx）+ 2A（subtask-card.tsx）+ 5.1 前端单测。先跑 dogfood 验收"实时思考可见 + subagent 推理完成即可见"。
2. **第二步（按需，真流式）**：2B（executor.py + task_tool.py，surgical）+ 5.2 后端单测 + 全量回归。仅当第一步 dogfood 后用户仍要求 subagent "边想边显示"才做。
3. **第三步（后续重构，可并入其它线）**：1B 统一思考组件，消除两套默认态的根。

### 注意 / 禁区
- ⚠️ React `useState` 初值只在 mount 取一次 —— 1A/2A 必须用 useEffect 跟随 `isLoading`/`status` 同步默认展开，且用 ref 记录"用户已手动操作"避免覆盖用户折叠意图。
- ❌ 不动后端 reasoning 捕获链（已正确）。
- ❌ 2B 不改 executor 的 turn 计数（`executor.py:913-936`）、seal-resume（`executor.py:726-820`）、recursion_limit（`executor.py:822-883`）。partial 只服务 UI，独立于 ai_messages 计数。
- ✅ 全程正面措辞（CLAUDE.md §6 deepseek 正面提示原则）。
- ✅ executor.py / task_tool.py 改动在 PR 标注"受保护文件 surgical"，便于 sync 守护。

---

## 7. 回滚

- 1A/2A：把默认展开改回原折叠默认（删 useEffect/ref）即恢复，纯前端零状态。
- 2B：executor 改回 `stream_mode="values"`、移除 partial 推送字段即恢复原行为（reasoning 仍在完成时单次到达）。

---

## 关键代码文件速查

| 文件 | 行号 | 用途 |
|------|------|------|
| `frontend/.../message-group.tsx` | 56-57 | **问题 1 根因** — `showLastThinking` 默认折叠 |
| `frontend/.../message-group.tsx` | 71-79 | `lastReasoningStep` 判定逻辑 |
| `frontend/.../subtask-card.tsx` | 52 | **问题 2 前端** — `collapsed` 默认 true |
| `frontend/.../subtask-card.tsx` | 232 | CoT reasoning step 渲染 |
| `frontend/.../message-list-item.tsx` | 426 | **基准参考** — `ReasoningPanel.showThinking=true`（展开） |
| `frontend/.../message-list.tsx` | 111-124 | `assistant:processing` group 分组逻辑 |
| `frontend/.../message-list.tsx` | 317-389 | processing group 渲染分支 |
| `backend/.../subagents/executor.py` | 898 | **问题 2 后端** — `stream_mode="values"`（2B 改动点） |
| `backend/.../tools/builtins/task_tool.py` | 540-556 | `task_running` event 发射 |
| `backend/.../models/patched_reasoning.py` | 63-90 | reasoning_content 捕获（已验证正确，不动） |
| `frontend/.../core/messages/utils.ts` | 343-363 | `extractReasoningContentFromMessage` |
