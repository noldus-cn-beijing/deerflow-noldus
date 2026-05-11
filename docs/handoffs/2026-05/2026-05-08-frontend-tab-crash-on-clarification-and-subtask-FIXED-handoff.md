# 2026-05-08 前端 tab 崩溃 (ask_clarification + subtask 终态) — 已修复

## TL;DR

端到端跑 shoaling 范式，subagent 完成任务、或 lead 发出 `ask_clarification` 时，**前端 tab 崩溃** ("Aw, Snap" / DevTools 红屏)。重新打开历史 thread 也立刻崩。

根因 1 处，改动 1 个文件：[`packages/agent/frontend/src/core/tasks/context.tsx`](../../packages/agent/frontend/src/core/tasks/context.tsx) 的 `useUpdateSubtask`，移除了 render 期会触发 `setTasks` 的 terminal-status 分支。已端到端验证通过。

## 症状

| 触发场景 | 表现 |
|---------|------|
| streaming 中 subagent 完成 task | tab 卡死，浏览器报"网页崩溃" |
| 关闭 tab 后重开历史 thread | 进入页面立刻崩，DevTools 一屏红 |
| 全部场景 | DevTools console 持续刷 `Cannot update a component (SubtasksProvider) while rendering a different component (MessageList).` |

二级症状（虚晃，与本 bug 无关）：
- DevTools console 偶发 `Unable to add filesystem: <illegal path>` —— Chromium DevTools 协议层对 Next 16 / Turbopack source-map workspace 注册的报错，非应用层 bug。
- 第二次修复尝试中由 useEffect 引入的 `Cannot read properties of undefined (reading 'status')` —— 已回滚。

## 根因

`tasks/context.tsx` 的 `useUpdateSubtask` 在本仓 commit `81f67bd7` ("useUpdateSubtask 恢复双路径")时引入了三条岔路，其中一条在 **render 期** 会触发 `setTasks`：

```ts
// 旧逻辑（崩）
if (
  (update.status === "completed" || update.status === "failed") &&
  existing?.status !== update.status
) {
  tasks[update.id] = { ...existing, ...update, messages: existing?.messages ?? [] };
  setTasks({ ...tasks });   // ← 在 MessageList render 期被调用，触发 React 报错
  return;
}
```

调用链：

1. subagent 完成 → ToolMessage `"Task Succeeded. Result: …"` 进 `thread.messages`
2. langgraph SDK `useStream` 触发 `MessageList` re-render
3. `MessageList` render 期 walk `group.messages`，对 ToolMessage 调 `updateSubtask({status: "completed"})`
4. 进 terminal-status 分支 → `setTasks` → React `Cannot update a component while rendering` 警告 → 在 dev 模式触发 recoverable error，多次累积导致 tab 崩

`ask_clarification` 路径同因：lead 发 `ask_clarification` 时 `groupMessages` 把 ToolMessage 既塞进前一个 subagent group 又开新的 clarification group；subagent 渲染分支命中 `else if (message.type === "tool")` 用 ask_clarification 的 `tool_call_id` 走 `updateSubtask({status: "in_progress"})`。这条路径**不会**进 terminal 分支不会崩，但在其他 task 已完成的状态下会接力触发同一报错。

## 修复

对齐上游 deerflow 的 `useUpdateSubtask` 极简版本：**只在 `latestMessage` 存在时 `setTasks`**，其他全部 in-place mutation。

```ts
// 新逻辑
if (incoming) {
  // SSE task_running: dedup messages, commit via setTasks (在 SSE handler 内，非 render 期)
  ...
  setTasks({ ...tasks });
  return;
}

// 其他全部 in-place（无 setTasks）：
//   - render 期 message-list 派发的 task_init / task_completed / task_failed 元数据
//   - SSE task_completed / task_failed（不带 latestMessage）
// 终态卡片翻状态由后续 React rerender 自然完成：
//   ToolMessage 进 thread.messages → useStream 触发 rerender → 这次 render 期同代码再跑
//   一遍，把 status 写进 tasks → 该帧渲染 SubtaskCard 时读到 "completed"。
tasks[update.id] = { ...existing, ...update, messages: existing?.messages ?? [] };
```

[diff (HEAD~1..HEAD)](../../packages/agent/frontend/src/core/tasks/context.tsx)：单文件单段，~30 行净增减。

## 与上游的关系

排查时 fetch 了 `deerflow/main` 对比，结论：

| 上游提交 | 与本 bug 的关系 |
|---------|----------------|
| `0431a67b fix(frontend): filter task tool calls when rendering SubtaskCard (#1242)` | 治的是同一类 `TypeError on task.status`，filter `name === "task"` 防止非 task 的 toolCall id 渲染成 SubtaskCard。**我们本地早已有此 filter**（在 `message-list.tsx` SubtaskCard 渲染段），所以这条不需要再合。 |
| 上游当前 `useUpdateSubtask` (`tasks/context.tsx`) | 极简版：`if (latestMessage) setTasks else 纯 in-place`。本修复**就是对齐这个语义**（保留我们本地的 messages dedup/append 增强，因为本地用 `messages: AIMessage[]` 累积而上游只用 `latestMessage`）。 |
| 上游 `getMessageGroups` `lastOpenGroup()?.messages.push(message)` | 上游也保留这行；之前误以为它是 phantom task 来源，本会话第一轮误删过，已回滚。它本身无害，因为 phantom 的 task 元数据虽进 store 但没有 SubtaskCard 引用它。 |

**结论：上游没有专门治这个 bug 的 commit；但上游 `useUpdateSubtask` 的极简形态本身就不会触发该报错**——本地 commit `81f67bd7` 那次"恢复双路径"加了多余的 `setTasks` 分支，是 regression 来源。

## 验证

1. `pnpm typecheck` clean。
2. 用户端到端实测：upload → 触发 ask_clarification → subagent 完成（shoaling pipeline 全跑）→ tab 不再崩，历史 thread 正常加载。

## 相关代码位置

- 修复点：[`packages/agent/frontend/src/core/tasks/context.tsx:42-104`](../../packages/agent/frontend/src/core/tasks/context.tsx)
- subtask 渲染 filter（已有，无需改）：`packages/agent/frontend/src/components/workspace/messages/message-list.tsx` 渲染 `SubtaskCard` 处的 `?.filter((toolCall) => toolCall.name === "task")`
- 本会话期间一度被改后回滚的文件：
  - `packages/agent/frontend/src/core/messages/utils.ts` — `groupMessages` 中 `lastOpenGroup()?.messages.push(message)` 那行不要再删
  - `packages/agent/frontend/src/components/workspace/messages/message-list.tsx` — 不要把 `updateSubtask` 派发挪进 `useEffect`（会让 SubtaskCard 第一帧 `useSubtask(taskId)!` 拿到 undefined 立刻崩）

## 经验沉淀

1. **render 期调 setState 不只是 warning** —— React 19 / Next 16 dev 模式下，`Cannot update a component while rendering a different component` 会被 `react-dom-client.development.js` 当作 recoverable error 反复记录到 `on-recoverable-error.ts`，在频繁 rerender（streaming）场景下最终把 tab 撑崩。
2. **Subtask store 是双源数据** —— 既由 SSE 事件（`task_running` / `task_completed` / `task_failed`）写，又由 `MessageList` render 期 walk `thread.messages` 写。后者必须 in-place mutation；前者可以安全 setState。`useUpdateSubtask` 的判别只能依赖 payload 的差异（有无 `latestMessage`），不能依赖 status 字段——status 字段在两条路径上都会被设。
3. **遇到本地 fork 的 React 报错先去比对上游 ——** 本仓的"useUpdateSubtask 双路径"是本地独家改动，上游极简版从未引入这个崩溃。同步策略文档（[docs/sop/deerflow-sync-sop.md](../sop/deerflow-sync-sop.md)）的"取长补短不直接覆盖"原则反过来也成立：本地新增的逻辑分支应该和上游对照评估，避免引入 regression。

## 待办 / 不在本次范围

- DevTools 那条 `Unable to add filesystem: <illegal path>` 没有动；是浏览器内部 source-map workspace 报错，不影响功能。如果它在 Next 17 / Turbopack 升级后还在，再单独排。
- `useSubtask(taskId)!` 的 non-null 断言全仓只此一处，第一帧 race 风险被 render-期 in-place mutation 兜住了——只要 `MessageList` 在渲染 `SubtaskCard` 同一帧已经先调过 `updateSubtask` 注册元数据，就安全。如果未来重构 `MessageList` 把派发延迟到 effect，记得同步加上 `if (!task) return null` 兜底。
