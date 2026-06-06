# Frontend 信息架构改进设计文档

> 基于 2026-06-04 OFT 旷场 E2E dogfood 发现的前端问题分析与改进方案
>
> **审查状态**: 已由 Opus agent 对照代码库逐条核实（2026-06-04），关键修正：
> - Level 4 全链路已实现，剩一个边界 bug 待修
> - Level 1 warning badge 数据通路不存在
> - Level 2 代码草图字段与实际工具返回不符
> - 优先级已重排

## 问题总览

| # | 问题 | 严重度 | 状态 |
|---|------|--------|------|
| 1 | 消息重复输出（reasoning+content 渲染两遍） | P0 | ✅ 已修复 `utils.ts:getMessageGroups` |
| 2 | 图表 markdown 图片 404（`/mnt/` 路径未解析） | P0 | ✅ 已修复 `markdown-content.tsx` |
| 3 | Subagent 卡片标题状态滞后（completed 后仍显示 "正在…"） | P1 | ✅ 已修复 `subtask-card.tsx` |
| 4 | data-analyst quality warning 在 subagent group 中静默丢失 | P1 | 待修（边界 bug，真实根因见 §2.3.1） |
| 5 | Lead agent 工具调用渲染路径与实施目标错位 | P2 | 待进一步设计 |
| 6 | Subagent 实时进度不可见 | P3 | 待设计（需后端配合） |

---

## 1. 已修复问题详述

### 1.1 消息重复输出

**根因**：`core/messages/utils.ts:getMessageGroups()` 中，`hasReasoning && !hasToolCalls && hasContent` 的 AI 消息同时进入 `assistant:processing` 和 `assistant` 两个 group，两个 group 各自渲染完整的 reasoning + content。

**修复**：将 `hasContent && !hasToolCalls` 提升为独立 `else-if` 分支，排在 `hasReasoning || hasToolCalls` 之前，使此类消息只进入 `assistant` group。

### 1.2 图表图片 404

**根因**：`MarkdownContent` 组件在渲染 markdown 中的 `![image](/mnt/user-data/outputs/X.png)` 时，没有将 `/mnt/` 虚拟路径解析为 artifact API URL。浏览器将裸路径解析为相对 URL（`/workspace/chats/mnt/user-data/...`），导致 404。

**修复**：
1. `MarkdownContent` 新增 `threadId` prop
2. 在默认 `components.img` 中检测 `/mnt/` 前缀或其他 artifact 路径，自动调用 `resolveArtifactURL` / `normalizeArtifactImageSrc` 转换为正确的 API URL
3. `message-list.tsx` 中 4 处 `MarkdownContent` 调用均传入 `threadId`

**影响范围**：所有通过 `MarkdownContent` 渲染的 markdown 图片。

### 1.3 Subagent 卡片状态滞后

**根因**：`SubtaskCard` 的卡片标签始终使用 `getStageBroadcastForSubagent(subagent_type)` 返回的固定文案，不随 `task.status` 变化。

**修复**：
- `completed` → 灰色 muted "— 子任务已完成"
- `failed` → 红色 "— 子任务失败"
- `in_progress` → Shimmer 动画（不变）
- 折叠时显示 `task.result` 第一行作为摘要

---

## 2. 现有基础设施（已实现，勿重复造轮）

以下组件/通路已存在，后续方案应复用它而非新建：

| 基础设施 | 位置 | 说明 |
|---------|------|------|
| `QualityWarningBroadcastMiddleware` | `backend/.../middlewares/quality_warning_broadcast_middleware.py` | 后端读 data-analyst handoff 的 quality_warnings，注入到 lead AIMessage 的 `additional_kwargs.quality_warnings` |
| `extractQualityWarnings` | `frontend/src/core/messages/utils.ts:661` | 前端从 message 提取 quality_warnings |
| `QualityWarningBanner` | `frontend/.../messages/quality-warning-banner.tsx` | 前端渲染 warning banner |
| Banner 渲染点 | `message-list-item.tsx:228` | `MessageContent_` 中，仅 `assistant` group 触发 |
| `convertToSteps` 已带 `step.result` | `message-group.tsx:558-567` | 每个 tool call step 已通过 `findToolCallResult` 挂载了解析后的 result |
| `Subtask` 类型 | `core/tasks/types.ts` | **不含** `quality_warnings` 字段，`result` 是字符串 |

---

## 3. 核心议题：关键信息突破折叠

### 3.1 真实待修 bug — data-analyst warning 在 subagent group 中丢失

**这是 OFT dogfood 里 "n=1 critical warning 在默认视图中看不到" 的真实根因。**

#### 已有通路

`QualityWarningBroadcastMiddleware`（已注册、已启用）的工作流程：
1. `data-analyst` 完成 → handoff JSON 含 `quality_warnings` 数组
2. 中间件在 lead 产出**下一条无 tool_call 的 AIMessage** 时，读 handoff 注入 `additional_kwargs.quality_warnings`
3. 该 AIMessage 走 `assistant` group → `MessageContent_` → `QualityWarningBanner` 渲染 ✅

#### 边界 bug

当 lead 把 "播报质量警告的文字" 和 "下一个 task 派遣" **打包进同一条 AIMessage**（常见模式：`content = "n=1 无法统计..." + tool_calls = [task("chart-maker")]`）时：

- `hasToolCalls(message)` = true → 消息进入 `assistant:subagent` group
- `assistant:subagent` handler（`message-list.tsx:234-287`）**不调用 `extractQualityWarnings`，不渲染 `QualityWarningBanner`**
- Banner 静默丢失

这与历史记忆中的 "同一 AIMessage 并行 set_viz_choice+task 致竞态" 是同型打包问题。

#### 修复（二选一）

**方案 A（前端）**：`assistant:subagent` group handler 中对每条 `hasContent` 的 AIMessage 也调 `extractQualityWarnings` + 渲染 banner。

```tsx
// message-list.tsx assistant:subagent handler
if (hasContent(message)) {
  const narrative = extractContentFromMessage(message);
  if (narrative) {
    results.push(<MarkdownContent ... />);
  }
  // 新增：渲染 quality warnings
  const qw = extractQualityWarnings(message as unknown as Record<string, unknown>);
  if (qw.length > 0) {
    results.push(<QualityWarningBanner warnings={qw} />);
  }
}
```

**方案 B（后端）**：去掉中间件 `_maybe_inject` 的 `if last.tool_calls: return None` 早退条件，即使 lead 同消息里打包了 task，也注入 warnings。

**推荐方案 A**（纯前端，改动最小，不影响后端契约）。

### 3.2 Level 1 — SubtaskCard 折叠增强（已部分完成，暂时不需扩展）

**已完成**：折叠状态下显示 `task.result` 第一行作为摘要。

**原计划但暂不可做**：在折叠行显示 quality warning badge。原因是：
- `Subtask` 类型的 `result` 是纯字符串（从 ToolMessage split 出来），不是结构化对象
- `Subtask` 没有 `quality_warnings` 字段
- 要拿到 warning 计数需要改动后端数据通路（在后端把 warnings 塞进 task ToolMessage，前端在 `updateSubtask` 中解析）
- 而且 Level 1 的 badge 与 §3.1 banner 展 示的是同一份数据，按 SSOT 原则不应两处独立取

**结论**：Level 1 当前状态已足够。warning 展示统一用 lead banner（§3.1），不在 SubtaskCard 重复。

### 3.3 Level 2 — Lead tool calls 摘要化（需先解决渲染路径问题，再谈摘要）

#### 前置问题：lead 的 tool call 在哪个 group 渲染？

当前 `assistant:processing` handler（`message-list.tsx:296-339`）对 lead 的 AIMessage 只做了两件事：
1. `hasReasoning` → 渲染 `MessageGroup`（含 `convertToSteps` → `ToolCall`）
2. `hasContent` → 渲染 narrative `MarkdownContent`

**对既无 reasoning 又无 content 的 tool-call-only AIMessage，tool call 直接丢弃不渲染**。"查看其他 N 个步骤" 只在 `MessageGroup` 内部出现。

**必须先决定**：这些 tool call 应该在哪个 group 被消费。选项：
- 让 `assistant:processing` handler 对无 reasoning 的消息也调 `convertToSteps` + 渲染
- 或新增一个专用渲染路径

#### 摘要化方案（路径确定后实施）

`convertToSteps`（`message-group.tsx:558-567`）**已经**通过 `findToolCallResult` 把 tool result 解析成 `step.result` 挂在每个 step 上，只是 `ToolCall` 组件除 `web_search` / `image_search` 外不消费它。所以不需要新建 `tools/summary.ts`——直接在 `ToolCall` 里对已知高频工具加 `step.result` 摘要分支即可。

**注意**（对照真实工具返回）：
- `inspect_uploaded_file` 返回 `{sheets, columns, raw_metadata, ...}`，**没有** `row_count` 字段
- `prep_metric_plan` 返回 `{status: "ok", plan_summary: {paradigm, metric_count, subject_count, ...}}`，`metric_count` 嵌在 `plan_summary` 内，**不是**顶层字段

### 3.4 Level 3 — Subagent 实时进度事件（收窄范围后可行）

原方案的 `quality_warning` / `handoff_ready` 等终态信号与 §3.1 的 handoff 通路重叠（same data, dual source → 违反 SSOT）。Level 3 应该**只覆盖 handoff 通路做不到的事**：subagent 进行中的实时进度。

**收窄后的事件类型**：
```typescript
type SubagentProgressEvent = {
  type: "subagent_progress";
  task_id: string;
  progress:
    | { kind: "scripts_started"; total: number }
    | { kind: "script_completed"; script_name: string; status: "ok" | "failed" }
    | { kind: "chart_started"; chart_type: string };
};
```

**数据流**：
```
subagent → custom event (SSE) → lead stream → frontend useSubtask
                                                   ↳ 折叠行进度更新
```

**工作量**：后端需改动 subagent executor 的 SSE event 协议 + `Subtask` 类型；估时应上调到 4-6h（前后端合计）。

### 3.5 实施路径（重排后）

| Phase | 内容 | 预计工作量 | 依赖 |
|-------|------|-----------|------|
| **Phase 1** | §3.1: 修复 subagent group 中 banner 丢失（方案 A） | 0.5h | 无 |
| **Phase 2** | §3.3: 决定 lead tool call 的渲染路径 + 在 ToolCall 中加摘要 | 2-3h | Phase 1 |
| **Phase 3** | §3.4: Subagent 实时进度事件（收窄范围） | 后端 3h + 前端 3h | Phase 1 |
| — | §3.2: SubtaskCard warning badge | **暂不做**（数据通路缺失 + 与 banner 重复） | — |

---

## 4. 新对话 Sidebar 行为

待明确具体需求。相关组件：`recent-chat-list.tsx`、`workspace-sidebar.tsx`。当前观察到的行为已在之前测试中记录，需要用户给出具体的期望行为后再立项。

---

## 5. 改动清单

### 已修改文件

| 文件 | 改动 |
|------|------|
| `core/messages/utils.ts` | `getMessageGroups`: `hasContent && !hasToolCalls` 提前为 `else-if`（修复 #1） |
| `messages/markdown-content.tsx` | 新增 `threadId` prop + 默认 `img` 组件自动解析 artifact 路径（修复 #2） |
| `messages/message-list.tsx` | 4 处 `MarkdownContent` 传入 `threadId`（修复 #2） |
| `messages/subtask-card.tsx` | 卡片标签随 status 变化 + 折叠时显示 result 摘要（修复 #3） |

### 待实施

| 优先级 | 改动 | 涉及文件 | 估时 |
|--------|------|---------|------|
| P1 | assistant:subagent handler 渲染 QualityWarningBanner（§3.1 方案 A） | `message-list.tsx` | 0.5h |
| P2 | 决定 lead tool call 渲染路径 + ToolCall 摘要化（§3.3） | `message-group.tsx`（新增分支在 `ToolCall` 中消费已有 `step.result`） | 2-3h |
| P3 | Subagent 实时进度 custom events（§3.4，收窄范围） | backend executor + frontend `Subtask` 类型 + SSE | 6h |

---

## 审查记录

- **2026-06-04 Opus agent 审查**：逐条对照前端源码和后端 harness 核实。修正：Level 4 已实现→改为边界 bug；Level 1 warning badge 数据通路缺失；Level 2 字段名与实际工具返回不符；Level 3 与 Level 4 终态信号重叠→收窄范围。关键文件均已标注绝对路径。
