# Step 1 复现分析报告：subtask 卡片在 run success 后仍卡「正在运行」（2026-06-30）

> spec：`docs/superpowers/specs/2026-06-30-subtask-card-stuck-in-progress-after-run-success-fix-spec.md`
> thread：`772ec083-6aa7-469d-bfd4-f9f7ca9082a3`（EPM 28 文件，prod build dogfood 现场）

## ⚠️ 取证方式声明

spec 三 Step 1 要求「不可跳」地在 prod build 跑完整 EPM dogfood、打 console 坐实「未命中 vs 命中不 flush」。

**本次实施环境跑不了 prod build + 完整 dogfood。** 经用户拍板（`/implement-spec` 问答），采用 **静态代码取证** 替代实跑：逐文件读 spec 引用的源码，判别翻转为何没触发。判别结论与 spec 二自洽（命中但不 flush）。**真机复核（prod dogfood 跑到 report-writer 完成观察卡片不再转圈）作为最终验收留给用户**，见 spec 五.3。

## 判别结论：命中但不 flush（修法 A，spec 也判最可能）

### 证据 1：terminal 翻转唯一路径 = render-time walk + in-place mutate

`task.status`（驱动 `Loader2Icon animate-spin` + 文案，`subtask-card.tsx:67-79`）来自 `useSubtask`。翻到 `completed` 的唯一路径在 `message-list.tsx:288-309`：**render 期间 walk `thread.messages`**，遇到以 `"Task Succeeded"` 开头的 ToolMessage 时调 `updateSubtask({ id, status: "completed", result })`。

该调用 `update` **不带 `latestMessage`** → 落入 `useUpdateSubtask` 的 in-place-mutate 分支（`context.tsx` 原 line 79-97）：

```ts
// 不带 latestMessage 的所有路径 —— 都原地 mutate，刻意 NOT setTasks
tasks[update.id] = { ...(existing ?? …), ...update, messages: existing?.messages ?? [] };
```

注释（原 line 83-92）明写其假设：「`Task Succeeded` ToolMessage 已经/即将进入 thread.messages，MessageList 为那次 message 变化重渲染时，同一代码路径会以 status:"completed" 跑，原地 mutate 让卡片渲染终态」。

**这个假设要求「之后还有一次由别的 message 变化触发的 MessageList 重渲染」来 flush。**

### 证据 2：report-writer 是最后一步，那次兜底重渲染不发生

report-writer 是流水线**最后一步**。其 `Task Succeeded` ToolMessage 抵达后：
- 后端 run 已 success（`logs/gateway.log`：`Run 657df193 -> success`，spec 二已取证）
- 流已停
- 无后续 token、无 ask_clarification

→ **没有任何进一步的 message 变化触发 MessageList 重渲染** → 那次「兜底 flush」不发生 → 原地 mutate 的 `completed` 状态没有任何东西把它 flush 到 UI → 卡片永远 `in_progress`。

即：terminal 状态被写进了 store 对象，但**没触发 React 重渲染**，UI 读的是上一次渲染的 in_progress。

### 证据 3：SSE 路径不存在 task_completed 终态事件

`hooks.ts` 的 `onCustomEvent` 只处理 `task_running`（带 `latestMessage` → 走 setTasks flush）。**没有 `task_completed` / `task_failed` SSE 事件**驱动终态。终态翻转 100% 依赖证据 1 的 render-walk。spec 二 line 63 提「SSE `task_completed`（hooks.ts:539）」实为 `task_running` 的笔误——但不影响判别结论（终态确实无独立 SSE 通道）。

### 判别：命中但不 flush（不是「事件未到达翻转点」）

- 「未命中」需 `Task Succeeded` ToolMessage 没进 render-walk。但后端标准产物就是该 ToolMessage（message-list.tsx:288 已有完整 walk + 多格式兼容处理它），且后端日志显示 report.html 已成功 present——ToolMessage 必然已抵达 thread.messages。
- 因此根因 = **命中了翻转点，但 in-place mutate 不触发重渲染，且无后续事件兜底 flush**。

→ 选 **修法 A**（让 terminal 翻转可靠触发一次重渲染）+ **兜底**（run 终态时遍历 tasks 强制翻终态）。

## 静态取证交叉核（守 `feedback-handoff-bug-claims-expire-check-head-before-execution`）

spec 写于 2026-06-30。同源家族 #247（ask_clarification dots）同日合入 `e74757b1`——核其 diff：只改 `message-list.tsx` 底部 streaming indicator + 新增 streaming-dots 测试，**未碰 subtask 卡片终态翻转逻辑**。`context.tsx` 的 in-place-mutate 分支自 `81f67bd7`（恢复双路径）后未再动。**spec 根因在 HEAD 仍成立。**

## 实施落点（已实现，见 PR）

1. **修法 A**（`context.tsx`）：terminal 分支检测「非终态 → completed/failed」的真实状态跃迁时，`queueMicrotask(() => setTasks({...snapshot}))` 延后一次 flush——脱离 render 阶段，不触发「Cannot update while rendering」警告；幂等（同终态重复 walk 不排）。
2. **兜底**（`context.tsx` 新增 `finalizeRunning` + `hooks.ts` onFinish/onError 调用）：run 正常结束→所有仍 in_progress 的 subtask 翻 completed；run 出错→翻 failed。确定性兜底，与修法 A 正交。

## 待办：真机复核

- [ ] prod build（`make start`）跑完整 EPM dogfood 到 report-writer 完成，**观察 subtask 卡片显示「子任务已完成」、Loader2Icon 不再 animate-spin**（守 `feedback_code_has_fix_not_equal_bug_eliminated_seal_react_floor`——现象消除才算过）。
