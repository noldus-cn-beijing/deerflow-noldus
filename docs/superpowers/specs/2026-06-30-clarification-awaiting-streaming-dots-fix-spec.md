# 修复 spec：ask_clarification 等待用户回复时，底部流式三点指示器仍在跳动（2026-06-30）

> dogfood 现场新发现（2026-06-30，thread `993e8b83-3d25-405a-a923-3086eac58fe3`，EPM 多文件）。确定性单点 bug，读代码即定论，复用现成纯函数即可修。**本 spec 只写不实施，交别的 agent。**

---

## 一、现象

`ask_clarification` 决策卡（DecisionCard，「分析已暂停·等待你的确认」）渲染、agent 停下等用户回复时，**对话流底部的三点跳动指示器（`StreamingIndicator`，`● ● ●` animate-bouncing）仍在动**，给用户「agent 还在输出/运行」的错觉——实际 run 已中断、在等用户决策。

截图佐证：决策卡下方紧跟着跳动的三个点。

---

## 二、根因（已坐实）

`src/components/workspace/messages/message-list.tsx:548`：

```tsx
const trailing = (
  <>
    {thread.isLoading && <StreamingIndicator className="my-4" />}
    ...
  </>
);
```

那三个跳动点**只看 `thread.isLoading` 一个条件**（`message-list.tsx:102` 的 `const isLoading = thread.isLoading`）。

而 `ask_clarification` 走 `ClarificationMiddleware` 的 `Command(goto=END)` 中断后，run 处于「中断等待用户」态，但前端 `thread.isLoading` 仍为 `true`（SSE 连接未断、前端把中断等待当成「仍在流式」）→ `StreamingIndicator` 继续渲染。

**真 bug 在前端渲染条件漏判「正在等待澄清」这个状态**，不是后端问题（与 memory `feedback_todo_middleware_must_not_force_reengage_while_awaiting_clarification` 同源：「等待澄清」是该被识别并短路掉「进行中」表现的特殊态）。

---

## 三、修法（最小、确定性，复用现成纯函数）

**现成判定已存在**：`src/core/messages/clarification-state.ts` 的 `lastClarificationIsAwaiting(messages: Message[]): boolean`——「整条消息流是否止于一个未答的 ask_clarification」，纯函数、已有单测、专为「agent 在等任何决策」设计（非局限列对齐）。

在 `message-list.tsx:548` 的渲染条件叠加排除：

```tsx
{thread.isLoading &&
  !lastClarificationIsAwaiting(mergedMessages) &&
  <StreamingIndicator className="my-4" />}
```

> 注意：传给 `lastClarificationIsAwaiting` 的必须是**该组件实际渲染所用的同一份 messages**（`message-list.tsx` 里驱动渲染的那个数组，确认变量名——可能是 `mergedMessages` / `messages` / `thread.messages` 之一，**实施时 grep 确认，别想当然**）。`import { lastClarificationIsAwaiting } from "@/core/messages/clarification-state"`（同文件已 import 了同模块的 `answeredOptionIndex`，加个具名导入即可）。

---

## 四、TDD（CLAUDE.md 强制：先写断言再改）

`message-list` 已有测试基建（`message-list.perf.test.tsx`）。新增断言：

1. **红 → 绿**：构造一份「最后一条 AI 消息含 ask_clarification tool_call、其后无 human message」且 `thread.isLoading=true` 的 messages → 断言 `StreamingIndicator` **不渲染**（query 那三个 `animate-bouncing` dot 不存在）。改前红、改后绿。
2. **不回归**：`thread.isLoading=true` 且**普通流式中**（最后一条不是 ask_clarification）→ 断言 `StreamingIndicator` **照常渲染**。
3. **已答澄清恢复**：ask_clarification 之后已有 human message（用户已回复、run 重新跑）+ `isLoading=true` → 断言 `StreamingIndicator` **照常渲染**（已答 → 不再 awaiting → 不该被短路）。

> 防 vacuous：断言 2/3 保证「不是把 indicator 一刀切关掉」。去掉修法那行后断言 1 应变红——实施时验证这点（revert 行为仍绿 = 假 guard）。

---

## 五、验收

1. `npx vitest run`（前端）：新断言全绿，旧测试不回归。
2. `pnpm check` 0。
3. 真机复核：dogfood 跑到 ask_clarification 决策卡出现时，**底部三点不再跳动**；用户回复后 run 继续，三点恢复正常。
4. 守红线：不动 `useStream`/`mergeMessages`/dedupe（这是纯渲染条件改动，本就不该碰它们）。

---

## 六、不做什么

- ❌ 不改后端（`ask_clarification` 中断语义、`thread.isLoading` 的后端来源都不动）。
- ❌ 不新造「是否等待澄清」判定——`lastClarificationIsAwaiting` 已存在，复用它（守 single-source-of-truth）。
- ❌ 不顺手改决策卡 / clarification-options 的其它渲染。
