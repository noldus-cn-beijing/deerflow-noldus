# Frontend 信息架构改进设计文档

> 基于 2026-06-04 OFT 旷场 E2E dogfood 发现的前端问题分析与改进方案

## 问题总览

| # | 问题 | 严重度 | 状态 |
|---|------|--------|------|
| 1 | 消息重复输出（reasoning+content 渲染两遍） | 🔴 P0 | ✅ 已修复 `utils.ts:getMessageGroups` |
| 2 | 图表 markdown 图片 404（`/mnt/` 路径未解析） | 🔴 P0 | ✅ 已修复 `markdown-content.tsx` |
| 3 | Subagent 卡片标题状态滞后（completed 后仍显示 "正在…"） | 🟡 P1 | ✅ 已修复 `subtask-card.tsx` |
| 4 | Subagent 关键信息全在折叠中，用户不展开就看不到 | 🟡 P1 | 📝 本文档主议题 |
| 5 | Lead agent 工具调用隐藏在 "查看其他 N 个步骤" 中 | 🟡 P2 | 📝 本文档覆盖 |
| 6 | 新对话 sidebar 行为需优化 | 🟢 P3 | 📝 待进一步明确需求 |

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

**影响范围**：所有通过 `MarkdownContent` 渲染的 markdown 图片（present_files 内容、subagent handoff 摘要、报告中的图片等）。

### 1.3 Subagent 卡片状态滞后

**根因**：`SubtaskCard` 的卡片标签始终使用 `getStageBroadcastForSubagent(subagent_type)` 返回的固定文案（如 "🧮 正在计算指标，预计 30-60 秒..."），不随 `task.status` 变化。

**修复**：
- `completed` → 显示 "🧮 正在计算指标 — 子任务已完成"（灰色 muted）
- `failed` → 显示 "🧮 正在计算指标 — 子任务失败"（红色）
- `in_progress` → 保持 Shimmer 动画（同前）
- 折叠时显示 `task.result` 第一行作为摘要，而非仅 "子任务已完成"

---

## 2. 核心议题：关键信息突破折叠

### 2.1 当前状态

E2E 流程中用户默认看到的信息层级：

```
👤 分析数据（含4个文件）
├── 🤖 [intent] E2E_FULL_ASKVIZ …收到数据，先识别模板
│   └── 📦 查看其他 1 个步骤  ← identify_ev19_template 被折叠
├── 🤖 数据已探查完毕…先确认 EV19 模板类型
│   └── 📦 查看其他 2 个步骤  ← inspect_uploaded_file ×2 被折叠
├── ❓ ask_clarification：选 A/B 模板
├── 🤖 好的，确认模板…还需确认分组
│   ├── 📦 查看其他 1 个步骤  ← set_experiment_paradigm 被折叠
│   └── ❓ ask_clarification：control vs treatment
├── 🤖 确认分组…还需确认 zone
│   ├── 📦 查看其他 1 个步骤  ← prep_metric_plan 被折叠
│   └── ❓ ask_clarification：in_zone=1 是中心区？
├── 🤖 好的，启动分析
│   ├── 💭 思考 Lead Agent  ← lead 思考被折叠
│   ├── [SubtaskCard: code-executor]  ← 默认折叠
│   │   └── 内部：10 个 bash 命令 + reasoning 全部折叠
│   ├── [SubtaskCard: data-analyst]   ← 默认折叠
│   │   └── 内部：read_file ×N + reasoning + handoff 全部折叠
│   ├── [SubtaskCard: chart-maker]    ← 默认折叠
│   │   └── 内部：bash 命令 + reasoning 全部折叠
│   └── [SubtaskCard: report-writer]  ← 默认折叠
│       └── 内部：read_file ×N + reasoning 全部折叠
└── 🤖 最终回复：摘要/报告/图表
```

**问题**：用户不展开任何折叠区时，只能看到 lead 的顶层文字和最终回复。以下关键信号全部被隐藏：

- 数据质量 critical warning（n=1 无法统计检验）
- 指标计算结果（5 个中心区指标的具体数值）
- 图表生成失败原因（柱状图因列缺失而失败）
- 统计跳过原因（每组仅 n=1）
- Subagent 的实际工作过程（10 个 bash 脚本执行、判读文档查阅等）

### 2.2 设计原则

| 原则 | 说明 |
|------|------|
| **渐进披露** | 默认层 = 摘要/关键信号；展开层 = 完整细节 |
| **关键信号前置** | data quality warnings、errors、blockers 必须在折叠状态下可见 |
| **状态可感知** | 用户不需要展开就能知道 "进行到哪了、有没有问题" |
| **不推翻现有架构** | 利用现有的 custom events、SubtaskCard、QualityWarningBanner 组件 |

### 2.3 改进方案

#### Level 1：SubtaskCard 折叠状态增强（低风险，本 PR 可做）

```
┌─────────────────────────────────────────────────┐
│ 🧮 正在计算指标 — 子任务已完成        [展开 ▲]   │
│ ✓ 10/10 脚本成功，统计跳过（n=1）                 │  ← 新增：result 第一行摘要
│ ⚠️ 1 critical: SAMPLE_TOO_SMALL                   │  ← 新增：quality warning badge
└─────────────────────────────────────────────────┘
```

**实现**：
1. ✅ 已完成：result 第一行作为折叠摘要
2. 🆕 待做：从 `task.latestMessage.additional_kwargs` 或 `task.result` 中提取 `quality_warnings`，在折叠行中渲染 warning badge

```tsx
// SubtaskCard 折叠行新增 quality warning 指示
{task.status === "completed" && qualityWarnings.length > 0 && (
  <span className="text-amber-500 text-xs">
    ⚠️ {qualityWarnings.length} warning{qualityWarnings.length > 1 ? "s" : ""}
  </span>
)}
```

#### Level 2：Lead Agent tool calls — "查看其他 N 个步骤" 摘要化（中风险）

**现状**：Lead agent 的 identify/inspect/set_paradigm/prep_metric_plan 等 tool call 默认折叠，只显示 "使用 XX 工具" + "查看其他 N 个步骤"。

**改进**：在 "使用 XX 工具" 右侧显示该工具的关键输出摘要。

```
使用 "inspect_uploaded_file" 工具
  ↳ Trial 1: dark_mice 列, Trial 2: white_mice 列  ← 摘要

使用 "prep_metric_plan" 工具
  ↳ 5 个指标, Control vs Treatment 两组  ← 摘要

📦 查看其他 3 个步骤  ← 次要工具折叠
```

**实现思路**：
1. 为 `identify_ev19_template` / `inspect_uploaded_file` / `prep_metric_plan` 等高频工具编写 `extractToolCallSummary(toolName, result)` 函数
2. 在 `message-group.tsx` 的 tool call 渲染中，展开前 1-2 个"高信息量" tool call，其余折叠

```ts
// 新增：tools/summary.ts
export function getToolCallSummary(name: string, result: unknown): string | null {
  if (name === "inspect_uploaded_file") {
    const r = result as { columns?: string[]; row_count?: number };
    if (r.columns) return `${r.row_count ?? "?"} 行, 列: ${r.columns.join(", ")}`;
  }
  if (name === "prep_metric_plan") {
    const r = result as { metric_count?: number; paradigms?: string[] };
    if (r.metric_count) return `${r.metric_count} 个指标`;
  }
  return null;
}
```

#### Level 3：关键事件广播 — Custom Events 通道（中等风险，需后端配合）

**思路**：Subagent 在执行过程中，将关键 milestone 以 custom event 发送给 lead → lead 转发到前端 → 前端在 SubtaskCard 折叠行中展示。

**事件类型**：
```typescript
type SubagentMilestoneEvent = {
  type: "subagent_milestone";
  task_id: string;
  milestone: 
    | { kind: "scripts_completed"; total: number; succeeded: number; failed: number }
    | { kind: "quality_warning"; code: string; message: string }
    | { kind: "chart_generated"; chart_type: string; status: "ok" | "failed" }
    | { kind: "handoff_ready"; summary: string };
};
```

**数据流**：
```
subagent → ToolMessage (custom event) → lead → stream → frontend
                                                      ↳ SubtaskCard 折叠行更新
```

**优势**：不需要改变 subagent 的工作流程，只需要在 key points 发送一个轻量 event。

#### Level 4：QualityWarning 全局广播（低风险，前端优先）

**思路**：在 data-analyst handoff 中的 `quality_warnings` 数组，直接渲染为独立的 warning banner，不隐藏在卡片中。

**实现**：
1. 在 `message-list.tsx` 的 `assistant:subagent` 组渲染中，检测 `data-analyst` handoff 的 `quality_warnings`
2. 如果有 critical warning，在 SubtaskCard 上方渲染一个 `QualityWarningBanner`

```tsx
// 在 assistant:subagent handler 中
{dataAnalystQualityWarnings.length > 0 && (
  <QualityWarningBanner warnings={dataAnalystQualityWarnings} />
)}
```

这个改动已经在 `message-list-item.tsx` 中有基础设施（`extractQualityWarnings` + `QualityWarningBanner`），只需要在 message-list 的 group handler 中复用。

### 2.4 推荐实施路径

| Phase | 内容 | 预计工作量 |
|-------|------|-----------|
| **Phase 1** (本 PR) | Level 1: SubtaskCard 折叠摘要 + quality warning badge | 1-2h |
| **Phase 2** | Level 4: data-analyst warning 独立 banner | 1h |
| **Phase 3** | Level 2: Lead tool calls 摘要化 | 2-3h |
| **Phase 4** (需后端) | Level 3: Custom events 通道 | 后端 2h + 前端 2h |

---

## 3. 新对话 Sidebar 行为（待明确）

观察到的行为：
- "新对话" 链接在 `/workspace/chats/new` 时高亮
- 发送消息后 URL 变为 `/workspace/chats/<uuid>`，"新对话" 失去高亮
- Sidebar 的 RecentChatList 在对话开始后自动出现当前对话

需要与用户确认：期望的 sidebar 行为是什么？可能的方向：
1. 正在进行的对话在 sidebar 中应有视觉区分（如加粗/动画点）
2. "新对话" 链接在活跃对话中应保持可点击但不干扰
3. Sidebar 折叠时，"新对话" 应保持便捷访问

---

## 4. 改动清单

### 已修改文件

| 文件 | 改动 |
|------|------|
| `core/messages/utils.ts` | `getMessageGroups`: 修复重复输出（#1） |
| `messages/markdown-content.tsx` | 新增 `threadId` prop + 默认 `img` 组件自动解析 artifact 路径（#2） |
| `messages/message-list.tsx` | 4 处 `MarkdownContent` 传入 `threadId`（#2） |
| `messages/subtask-card.tsx` | 卡片标签随 status 变化 + 折叠时显示 result 摘要（#3, #4） |

### 待实施改动（建议下个 agent 执行）

| 优先级 | 改动 | 涉及文件 |
|--------|------|---------|
| P1 | SubtaskCard 折叠行 quality warning badge | `subtask-card.tsx` |
| P1 | assistant:subagent 组中独立渲染 QualityWarningBanner | `message-list.tsx` |
| P2 | Lead tool calls 前 1-2 个展开摘要 | `message-group.tsx`, 新增 `tools/summary.ts` |
| P3 | Custom events 通道（需后端配合） | backend + frontend |
