# Spec：前端 message grouping 流式瞬态孤儿 tool message 修复

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-25
> 代码基线：dev HEAD `c410b7b6`
> 性质：🟡 中 · 前端 grouping 健壮性（console 报错 + 流式中途消息瞬时丢失）。纯前端改动，零后端。
> **方向（用户拍板 2026-06-25）**：EPM dogfood（thread `0e72d605`）浏览器 console 报 `Unexpected tool message outside a processing group {}`。取证（解 langgraph checkpoint 拿真实 37 条消息序列 + 核实各 group 渲染安全性）坐实根因=**流式分片时序**：AI message 的 tool_calls 分片尚未到达、content 暂空时，grouping 不开任何组，其 tool result 流到即成孤儿 → console.error + 丢弃。修法=方向 1（AI message 永远占一个 open processing 组，治本）+ 方向 2（孤儿降级不丢弃，兜底）。
> 受保护文件（sync surgical）：`frontend/src/core/messages/utils.ts`（含 Noldus grouping 定制）。

---

## ⚠️ 〇、根因（取证坐实，解真实 checkpoint，不凭推断）

> 本 spec 来自 EPM dogfood（thread `0e72d605`，2026-06-25）的 console 报错。**最初表象误导性极强**（"空 {} tool message"像是 DanglingToolCall 占位或 content 为空），取证逐一推翻了三个错误假设，本章如实记录。

### 取证手段
1. 解 langgraph `checkpoints.db`（`SqliteSaver.get`）拿该 thread **真实 37 条消息序列**。
2. 逐条核实 `utils.ts:getMessageGroups` 的分组判定。
3. 核实 `message-list.tsx` 各 group 类型对 tool message 的渲染安全性。

### 被取证推翻的错误假设
| 假设 | 推翻依据 |
|---|---|
| `{}` 是 content 为空的 tool message | 真实序列里所有 tool message 都有 content（如 `Successfully presented files` / JSON）。`{}` 是 `console.error("...", message)` 第二参数**流式瞬态 message 对象的浏览器浅显示** |
| `{}` 是 DanglingToolCallMiddleware 注入的占位 | 占位有合成 content + `status="error"`（[dangling_tool_call_middleware.py:169-174](../../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/dangling_tool_call_middleware.py#L169)），不是空；且本 thread 无中断悬挂 |
| 修法是「回退挂最后一个 group」 | 核实否决——挂 `human`/`assistant` 组会渲染原始工具输出气泡；挂 `assistant:subagent` 组会触发 `updateSubtask` 副作用（见 §二安全性表）|

### 真根因（坐实）
**流式分片时序**。前端 grouping 在 streaming 时每来一个分片重算一次。LangGraph 流式中，一个 AI message 的 `content` / `reasoning` / `tool_calls` 分片**分批到达、不同时**。在 **tool_calls 尚未到达、content 仍为空** 的那一帧：

- AI message 此刻 `hasContent`=false、`hasToolCalls`=false、`hasReasoning`=false → [utils.ts:92-125](../../../packages/agent/frontend/src/core/messages/utils.ts#L92) 的 AI 分支**五个条件全不命中 → 不开任何组**。
- 此时该 tool 的 result 已流到 → [utils.ts:79](../../../packages/agent/frontend/src/core/messages/utils.ts#L79) `lastOpenGroup()` 看最后是终结组（`human`/`clarification`）→ 返回 null → [L83](../../../packages/agent/frontend/src/core/messages/utils.ts#L83) `console.error` + **丢弃**。
- 流式结束、tool_calls 到齐后重算 grouping 即正常 → **只在流式中途报错，最终态正常**（解释了「dogfood 端到端成功但 console 有 error」）。

**最易触发点（真实序列坐实）**：消息 `[19] ai content='' tool_calls=['set_viz_choice']`（用户回答「A.是画图」后 lead 调 set_viz_choice）。`[19]` 的 content 本就空（lead 调 set_viz_choice 时无思考文本），「content 空 + tool_calls 未到」的瞬态窗口比别的 AI message 更长更稳定地命中 → 流式中途 `[20] set_viz_choice` 的 result 成孤儿。

### 危害
- console.error 污染（每次流式经过这种瞬态都报）。
- 流式中途那一帧 tool result 不渲染（最终态恢复，但用户看到内容闪烁/瞬时缺失）。
- **非破坏性**（最终态正确），但属 v0.1 演示观感硬伤 + console 噪声。

---

## 一、给实施 agent 的一句话

改 `utils.ts:getMessageGroups` 两处：M1（治本）——AI message 只要 `type==="ai"` 就至少并入/开一个 `assistant:processing` 组（即使 content/tool_calls/reasoning 暂时全空），让其后续 tool result 永远有 open processing 组可挂；M2（兜底）——`lastOpenGroup()` 找不到时不再 `console.error` 丢弃，而是**新开一个 `assistant:processing` 组**容纳该 tool message（processing 组对 tool message 渲染安全），仅在非流式态保留一条 `console.warn`（那才是真异常）。TDD 用真实瞬态序列复现。

---

## 二、各 group 对 tool message 的渲染安全性（核实表，决定修法）

> 核实 `message-list.tsx` 每个 group 类型挂入「无关 tool message」的影响——这决定为什么不能简单「回退挂最后一组」，以及为什么选 `assistant:processing` 作容器。

| group 类型 | 渲染逻辑（message-list.tsx）| 挂入孤儿 tool message 的影响 | 安全 |
|---|---|---|---|
| `human` / `assistant` (L74-85) | `messages.map(MessageListItem)` → tool 被当 assistant 气泡渲染原始 content | ❌ 渲染出 `Successfully presented files` 等原始输出气泡 | 否 |
| `assistant:subagent` (L156-316) | tool message 进 L174 → `updateSubtask(tool_call_id)` | ❌ 无关 tool_call_id 触发 updateSubtask 副作用 | 否 |
| `assistant:clarification` (L86-134) | 只渲染 `group.messages[0]` | ✅ 后挂的 tool message 被忽略 | 是 |
| `assistant:present-files` (L135-155) | 只渲染有 present_files 的 + `messages[0]` content | ✅ tool message 无 present_files，忽略 | 是 |
| **`assistant:processing`** (L317-389) | **`if (message.type !== "ai") continue`（L334）显式跳过 tool** | ✅ **tool message 安全跳过，不渲染** | **是** |

**结论**：`assistant:processing` 是唯一「既能作 open group 容纳 tool message、又对 tool message 渲染安全」的组。M1/M2 都用它作容器。**绝不回退挂 human/subagent 组。**

---

## 三、修法（M1 治本 + M2 兜底）

### M1：AI message 永远占一个 open processing 组（治本）

**改 [utils.ts:92-125](../../../packages/agent/frontend/src/core/messages/utils.ts#L92) 的 AI message 分支**。当前末尾 `else if (hasReasoning || hasToolCalls)` 才开 processing 组；改为：**present-files / subagent / 终结 assistant 三种特例之外的所有 AI message（含暂时全空的流式瞬态）都进 processing 组**：

```js
if (message.type === "ai") {
  if (hasPresentFiles(message)) {
    groups.push({ id: message.id, type: "assistant:present-files", messages: [message] });
  } else if (hasSubagent(message)) {
    groups.push({ id: message.id, type: "assistant:subagent", messages: [message] });
  } else if (hasContent(message) && !hasToolCalls(message)) {
    groups.push({ id: message.id, type: "assistant", messages: [message] });
  } else {
    // 改动：原 `else if (hasReasoning || hasToolCalls)` → `else`
    // 流式瞬态：content 空 + tool_calls 未到 + 无 reasoning 的 AI message 也开/并入
    // processing 组，让其后续 tool result 有 open group 可挂（消除孤儿源头）。
    // 空 processing 组由 message-list.tsx:387 的 `if (results.length===0) return null` 兜底不渲染。
    const lastGroup = groups[groups.length - 1];
    if (lastGroup?.type !== "assistant:processing") {
      groups.push({ id: message.id, type: "assistant:processing", messages: [message] });
    } else {
      lastGroup.messages.push(message);
    }
  }
}
```

> 关键：把最后一个 `else if (hasReasoning(message) || hasToolCalls(message))` 改成无条件 `else`。这样**任何**非特例 AI message（包括流式中途 content/tool_calls 都还没到的空壳）都开/并入 processing 组。其 tool result 流到时 `lastOpenGroup()` 命中该 processing 组 → 不再孤儿。

### M2：孤儿降级不丢弃（兜底未来未知孤儿）

**改 [utils.ts:78-87](../../../packages/agent/frontend/src/core/messages/utils.ts#L78) 的 tool message else 分支**：

```js
} else {
  const open = lastOpenGroup();
  if (open) {
    open.messages.push(message);
  } else {
    // M2 兜底：找不到 open group（M1 后理论上不再发生，但防未来未知时序）。
    // 不再 console.error 丢弃——新开一个 processing 组容纳（对 tool message 渲染安全，见 §二）。
    // 仅非流式态仍 warn：流式态孤儿是已知瞬态（M1 已治），非流式态孤儿才是真异常。
    groups.push({
      id: message.id,
      type: "assistant:processing",
      messages: [message],
    });
  }
}
```

> M2 是 M1 的安全网。M1 后流式孤儿不再发生，但 M2 兜住任何未来未预料的时序——孤儿 tool message 进一个独立 processing 组（不渲染其内容，但不丢、不报 error）。**不再 console.error**。

> **关于 console.warn**：若要保留可观测，可在 M2 加 `if (!isStreaming)` 守卫的 warn——但 `getMessageGroups` 当前不接收 isStreaming 参数，加参数会改签名+所有调用方。**建议 M2 直接静默兜底**（不 warn），保持函数纯净；可观测交给 React DevTools/手动排查。若坚持要 warn，单列为 M2b 改签名。

---

## 四、TDD（红→绿）

> frontend 当前**无测试框架**（CLAUDE.md: "No test framework is configured"）。本 spec 需先决策：(a) 引入 vitest 测 `utils.ts` 纯函数，或 (b) 用类型级 + 手动 dogfood 验收。**建议 (a)**——`getMessageGroups` 是纯函数、易测、是 grouping 回归高发区（历史多次踩，见 message-list.tsx 注释里的 2026-05-25 / 2026-06-04 回归）。

### 若引入 vitest（建议）：`utils.test.ts`

```ts
// T1（治本）：流式瞬态——空 AI message + 其 tool result，不产生孤儿
test("streaming transient: empty AI message gets processing group, tool result attaches", () => {
  const msgs = [
    { type: "human", id: "h1", content: "A" },
    { type: "ai", id: "a1", content: "", tool_calls: [] },  // 流式中途：tool_calls 未到
    { type: "tool", id: "t1", name: "set_viz_choice", tool_call_id: "tc1", content: "{...}" },
  ];
  const groups = getMessageGroups(msgs);
  // 红：当前 a1 不开组 → t1 孤儿 → console.error
  // 绿：M1 后 a1 开 processing 组 → t1 挂上，无孤儿
  expect(groups.some(g => g.type === "assistant:processing")).toBe(true);
  // 断言无 console.error（spy）
});

// T2（兜底）：纯孤儿 tool message（前面是 human）不丢弃、不 error
test("orphan tool after human goes to a processing group, not dropped", () => {
  const msgs = [
    { type: "human", id: "h1", content: "A" },
    { type: "tool", id: "t1", name: "x", tool_call_id: "tc1", content: "r" },
  ];
  const groups = getMessageGroups(msgs);
  // 绿：M2 后 t1 进一个 processing 组（不再 console.error）
  expect(groups.length).toBeGreaterThan(1);
});

// T3（回归）：正常 AI+tool 序列分组不变
test("normal ai-with-toolcall + tool result groups unchanged", () => { ... });

// T4（回归）：present_files / subagent / clarification 终结组行为不变
test("present-files / subagent / clarification grouping unchanged", () => { ... });

// T5（回归）：真实 37 条 dogfood 序列分组与现状一致（除孤儿消除）
test("real dogfood 0e72d605 sequence: no orphan, all 37 grouped", () => { ... });
```

### 若不引入测试框架
- `pnpm check`（lint + typecheck）+ 手动 dogfood：重跑 EPM，浏览器 console 应**无** `Unexpected tool message` 报错。

---

## 五、风险与注意事项

1. **空 processing 组渲染**：M1 可能产生暂时 messages 全空/全 tool 的 processing 组。[message-list.tsx:387](../../../packages/agent/frontend/src/components/workspace/messages/message-list.tsx#L387) 已有 `if (results.length === 0) return null` 兜底——空组不渲染。**核实该兜底对「组里只有 tool message（被 L334 跳过）」也生效**（results 为空 → return null）✅。
2. **绝不回退挂 human/subagent 组**（§二核实）。M1/M2 都用 `assistant:processing` 作容器。
3. **grouping 是回归高发区**：message-list.tsx 注释记录了 2026-05-25（report+clarification 同 message）、2026-06-04（tool-only AI message 丢失）等多次回归。改 grouping 必跑全部 group 类型的回归（T3/T4/T5）。守 `feedback_pr_merge_must_run_full_suite_on_shared_logic`。
4. **sync surgical**：`utils.ts` 含 Noldus grouping 定制（clarification 双 push、subagent 组等），改时只动 AI 分支末尾 + tool else 分支，保留其余。
5. **isStreaming 不入 getMessageGroups**：M2 不加 isStreaming 守卫（避免改签名+所有调用方）。若产品要「非流式态孤儿仍可观测」，单列 M2b。
6. **`getMessageGroups` 多处调用**：grep 所有调用方（message-list / getAssistantTurnUsageMessages / 等），确认 M1 多开的 processing 组不破坏它们（如 turn usage 聚合按 group 遍历，多一个空 processing 组应无害——它 messages 非空，含那个 AI message）。

---

## 六、实施步骤

1. （建议）引入 vitest：`pnpm add -D vitest`，配 `vitest.config.ts`，加 `pnpm test` script。
2. TDD 红：写 T1，确认当前红（空 AI message → 孤儿 → error）。
3. M1：改 AI 分支末尾 `else if(...)` → `else`，跑 T1/T3/T4/T5 绿。
4. M2：改 tool else 分支孤儿兜底，跑 T2 绿。
5. `pnpm check`（lint+typecheck）+ 手动 dogfood：重跑 EPM，console 无 `Unexpected tool message`。
6. 全 group 类型回归（T3/T4/T5）确认无渲染变化。

---

## 七、与其他 spec 的关系

- 与 spec A/B/C/D **全部正交**——纯前端 grouping，不碰后端。
- 与 spec C（read_file + run_metric_plan 中断）**曾疑同源**（最初猜孤儿是中断悬挂占位），取证**排除**——本 spec 是流式时序，spec C 是后端中断，不同源。

---

## 八、milestone 建议

归入「前端信息架构 / 流式渲染健壮性」track。checkpoint：「EPM dogfood（thread 0e72d605）console 报 Unexpected tool message outside a processing group {}。取证（解 langgraph checkpoint 真实 37 条序列 + 核实各 group 渲染安全性）**推翻三个错误假设**（非空 content、非 DanglingToolCall 占位、不能回退挂 human/subagent 组），坐实根因=流式分片时序（AI message 的 tool_calls 分片未到、content 暂空时 grouping 不开组，其 tool result 流到成孤儿）。修法=M1 AI message 永远占 processing 组（治本，processing 组对 tool message 渲染安全）+ M2 孤儿降级进 processing 组不丢弃不 error（兜底）。最易触发点=set_viz_choice（content 本就空）。建议引入 vitest 测 grouping 纯函数（回归高发区）。」
