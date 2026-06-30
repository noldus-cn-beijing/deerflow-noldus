# 修复 spec：subtask 卡片在后端 run success 后仍卡在「正在运行」（in_progress 终态翻转依赖脆弱重渲染假设）（2026-06-30）

> dogfood 现场发现（2026-06-30 prod build，thread `772ec083-6aa7-469d-bfd4-f9f7ca9082a3`，EPM 28 文件）。**本 spec 只写不实施，交别的 agent。** 已坐实根因（前后端双侧取证）。

---

## 一、现象

report-writer 子任务卡片显示「📋 解读已完成，正在生成中文研究报告...」+ `Loader2Icon animate-spin`（转圈），**一直停在运行态**。但后端该 run 早已完成、报告已生成并成功 present。

---

## 二、根因（前后端双侧取证，已坐实）

### 后端：早已 success（`logs/gateway.log`，thread 772ec083）

```
10:48:06  report-writer completed async execution
10:48:07  Task call_40b31334... status: completed  (completed after 32 polls)
10:48:19  Run 657df193 -> success
10:48:21  report.html 成功解析返回 200，present_files 成功
```
其后**无新 run、无 ask_clarification**——后端彻底终态。前端却仍 in_progress。**纯前端状态 bug。**

### 前端：subtask 终态翻转依赖一个会失效的假设

`task.status`（驱动转圈 + 文案，`subtask-card.tsx:67-79`、`195`）来自 `useSubtask`（`core/tasks/context.tsx:37`）。翻到 `completed` 的**唯一路径**是 `message-list.tsx:288-309`：**render-time walk `thread.messages`，遇到以 `"Task Succeeded"` 开头的 ToolMessage 时**才 `updateSubtask({status:"completed"})`。

而 `useUpdateSubtask`（`core/tasks/context.tsx:79-97`）对**不带 `latestMessage` 的 terminal 事件刻意 NOT setTasks**，只原地 mutate `tasks[id]`，注释明写其假设：

> 「Because the matching ToolMessage ("Task Succeeded…") has either already arrived in thread.messages or is about to. When MessageList re-renders for that message change, this same code path runs with status:"completed" and the in-place mutation makes the card render its terminal state.」

**这个假设脆弱**：subtask 卡片翻成终态，**完全依赖「之后还会有一次由 `Task Succeeded` ToolMessage 触发的 MessageList 重渲染」**。report-writer 是流水线**最后一步**，其 `Task Succeeded` ToolMessage 到达后，若没有进一步的 message 变化触发重渲染（run 已 success、流已停、无后续 token / 无 ask_clarification），那次「兜底重渲染」就不发生 → 原地 mutate 的 `completed` 状态**没有任何东西把它 flush 到 UI** → 卡片永远 `in_progress`。

> 即：terminal 状态被写进了 store 对象，但**没触发 React 重渲染**，UI 读的是上一次渲染的 in_progress。这是「靠后续事件兜底 flush」的设计在「我就是最后一个事件」时必然失效。

---

## 三、Step 1：prod 复现坐实（不可跳，确认翻转为何没触发）

> 前后端日志已强证，但前端「那次重渲染为何没发生」需断点坐实，避免改错层。

1. prod build（`make start`）跑完整 EPM dogfood 到 report-writer 完成。
2. 在 `message-list.tsx:288` 和 `context.tsx:79`（terminal 分支）打 console：打印 `taskId`、`status`、以及该分支是否被命中、命中后 UI 是否重渲染。
3. 判别：
   - 若 terminal 分支**根本没被命中**（`Task Succeeded` ToolMessage 没进 render-walk / 顺序问题）→ 根因 = **事件未到达翻转点**。
   - 若**命中了但 UI 不更新**（原地 mutate 无 setTasks，无后续重渲染 flush）→ 根因 = **terminal 翻转不触发重渲染**（最可能）。

**产出**：复现小报告（落 `docs/superpowers/reports/`），明确「卡在哪：未命中 vs 命中不 flush」。

---

## 四、Step 2：修法（按 Step1 选，倾向修法 A）

### 修法 A（若根因=命中但不 flush，最可能）
让 subtask **terminal 状态翻转可靠触发一次重渲染**，不再赌「后续会有别的重渲染」。
- 在 `context.tsx` 的 terminal 分支：当 status 从非终态变为 `completed`/`failed`（真实状态跃迁）时，**setTasks 提交一次**——但要避开它原本规避的「render 中调 setTasks 警告」。
  - 关键约束（注释 line 88-92）：MessageList 在 **render 期间**同步调 `updateSubtask` 镜像 tool_call 元数据，此时 setTasks 会 warn「Cannot update while rendering」。
  - 因此 terminal flush 不能裸调 setTasks，需**脱离 render 阶段**：用 microtask / `queueMicrotask` 或在 effect 中提交终态（仅当检测到 in_progress→completed 跃迁时），避免 render 内同步 setState。
- 守约束：只在**真实状态跃迁**时 flush（不是每次 render walk 都 setTasks，否则回到性能问题）。

### 修法 B（若根因=事件未到达翻转点）
确保 report-writer（最后一步）的 `Task Succeeded` ToolMessage 必然进入一次 render-walk，或在 SSE `task_completed` 事件直接驱动终态（`hooks.ts:539` 目前只在带 `latestMessage` 时 updateSubtask；可让 `task_completed` 不带 message 时也走一次可靠的终态提交）。

### 兜底（与 A/B 正交，强烈建议加）
**run 进入 success/error 终态时，把该 run 下所有仍 `in_progress` 的 subtask 强制翻 `completed`/`failed`**。这是确定性兜底：无论上面哪条路失效，run 都不可能终态了还留着转圈的卡片。落点：thread run-state 变 success 时遍历 tasks。

---

## 五、验收

1. **Step 1 复现报告**先行。
2. TDD（CLAUDE.md 强制）：
   - 红→绿：构造「subtask 收到 `Task Succeeded` terminal 事件、其后无更多 message 变化」→ 断言卡片渲染 `completed`（无 `animate-spin`、文案非「正在...」）。**改前红、改后绿。**
   - 防 vacuous：去掉修法后该断言应变红。
   - 不回归：流式中（in_progress 真未完成）卡片**仍转圈**。
3. 真机复核（prod）：跑完整 dogfood 到 report-writer 完成，**卡片显示「子任务已完成」不再转圈**（守 `feedback_code_has_fix_not_equal_bug_eliminated_seal_react_floor`——现象消除才算过）。
4. `npx vitest run` 绿 + `pnpm check` 0。
5. 守红线：不动 `useStream`/`mergeMessages`/dedupe；**不破坏 `context.tsx` 规避 render-中-setTasks 的既有约束**（别引入 React 警告 / 死循环重渲染）。

---

## 六、不做什么

- ❌ 不改后端（后端 run/seal 正确终态，bug 纯前端）。
- ❌ 不在 Step1 坐实前盲改 `context.tsx` 的 commit 时机（那段注释记录了它规避的 render-setTasks 陷阱，改错会引入新 bug）。
- ❌ 不每次 render walk 都 setTasks（会回退性能）。
- ❌ 不动流式核心。

---

## 七、关联

- 同源家族：`#247`（ask_clarification dots，已修 commit `e74757b1`）+ 本 spec，都属「run 终态后某个 UI 状态不复位」。本 spec 是 subtask 卡片层，#247 是 thread 底部指示器层，落点不同、可分别验收。
- 与 generative-UX 输出理念重构（2026-06-30 调研中）相关：subtask 运行态如何呈现本就是重构对象，本 spec 先确定性止血，重构时这块呈现会再设计。
