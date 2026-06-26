# Spec：输入框遮挡底部对话内容修复（前端布局 · 悬浮底栏 scroll-padding）

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-26
> 代码基线：dev HEAD `5616a73f`
> 性质：🟢 低 · 纯前端布局（CSS padding，**不碰流式核心/不碰 generated 源**），独立可做
> 取证：用户截图——决策卡选项（A/B）+ "画图"按钮被悬浮输入框盖住，看不到也点不到

---

## 〇、问题（截图实测）

对话流最底部的内容**被固定在底部的输入框盖住**：用户截图里 `ask_clarification` 决策卡的选项（"A. 是，把刚才的结论画成图" / "B. 不用，直接给我报告"）和右下角"画图"按钮，被悬浮输入框压在背后——**既看不全也点不到**（这些是可交互元素，比单纯看不到更严重）。输入框越高（多行/带附件堆叠）盖住越多。

---

## 一、根因（读码坐实）

布局结构（`app/workspace/chats/[thread_id]/page.tsx`）：

```jsx
// :164 输入框容器 —— absolute 浮在消息流之上（z-30 盖住消息）
<div className="absolute right-0 bottom-0 left-0 z-30 flex justify-center px-4 pb-4">
  <InputBox ... />
</div>

// :152-153 消息流 —— 只有 pt-4，无底部 padding
<div className="flex size-full justify-center">
  <MessageList className={cn("size-full", !isNewThread && "pt-4")} ... />
```

消息内容区 `ConversationContent`（`message-list.tsx:454`）当前 `className="... pt-12"` —— **只有顶部 padding，没有底部 padding**。

**遮挡机理**：输入框 `absolute bottom-0` 悬浮在消息流上，但消息流滚动内容区**底部没有预留输入框高度的留白**。最后的内容滚到底时延伸进输入框背后区域 → 被盖住。**且随输入框高度变化恶化**（附件堆叠/多行时输入框更高，盖住更多）。

---

## 二、修复：消息流底部预留输入框高度的 scroll-padding（动态）

经典"悬浮底栏 + 内容区 scroll-padding"模式。输入框已有 `promptRootRef`（`input-box.tsx:400`），**高度可测**，做**动态** padding（不写死，因输入框高度随附件/行数变）：

### 2.1 测输入框高度 → CSS 变量

`input-box.tsx`（或 page.tsx 父层）用 `ResizeObserver` 观测 `promptRootRef.current` 高度，写进一个 CSS 变量（挂在对话区容器或 `:root` 局部）：
```ts
// input-box.tsx 内，或父层
useEffect(() => {
  const el = promptRootRef.current;
  if (!el) return;
  const ro = new ResizeObserver(() => {
    const h = el.offsetHeight;
    el.closest("[data-chat-root]")?.style.setProperty("--input-box-height", `${h}px`);
  });
  ro.observe(el);
  return () => ro.disconnect();
}, []);
```

### 2.2 消息流底部用该变量留白

`message-list.tsx:454` 的 `ConversationContent` className 加底部 padding（**业务侧 className，不碰 generated `conversation.tsx` 源**）：
```jsx
<ConversationContent className="mx-auto w-full max-w-(--container-width-md) gap-8 pt-12 pb-[calc(var(--input-box-height,7rem)+1.5rem)]">
```
- `var(--input-box-height, 7rem)`：动态高度 + fallback 7rem（输入框未测量时的合理默认）。
- `+1.5rem`：额外呼吸间距，让最后内容**完全滚过**输入框上沿，不贴边。

### 2.3 贴底滚动同步（StickToBottom）

`use-stick-to-bottom` 贴底时要滚到"内容真实底部"（含新 padding）。确认加 `pb` 后 `scrollToBottom` 仍把最后一条消息滚到**输入框上方可见区**，而非被盖。若 StickToBottom 按 scrollHeight 计算，新 padding 会自然纳入；**dogfood 验证贴底后最后一条完全可见**。

---

## 三、备选（更简但不够动态）

若动态测量嫌复杂，**退化方案**：消息流底部给一个**固定 `pb`**（如 `pb-32` ≈ 输入框常态高度 + 余量）。缺点：输入框变高（附件堆叠多行）时仍可能盖住一点。**推荐 §二动态**，附件堆叠场景下输入框高度变化大，固定值不够稳。

---

## 四、验证

1. `pnpm check`。
2. **dogfood 关键场景**（`make dev`）：
   - 对话流最后是 `ask_clarification` 决策卡 → 选项 + 按钮**完全可见可点**，不被输入框盖。
   - 输入框带附件堆叠（多文件）变高时 → 底部内容仍不被盖（动态 padding 生效）。
   - 滚到最底 → 最后一条消息完全在输入框上方可见。
   - 输入框多行展开 → 底部留白同步增大。
3. **回归**：空 thread / 短对话（不足一屏）布局正常，无多余大片空白（fallback 值合理）。
4. **不碰**：`ai-elements/conversation.tsx`（generated 源）、流式核心、StickToBottom 内部逻辑——只加业务侧 className padding + 一个 ResizeObserver。

---

## 五、关键文件

- `packages/agent/frontend/src/app/workspace/chats/[thread_id]/page.tsx:152-164`（消息流 + 悬浮输入框布局）
- `packages/agent/frontend/src/components/workspace/messages/message-list.tsx:454`（ConversationContent 加 `pb`）
- `packages/agent/frontend/src/components/workspace/input-box.tsx:400`（promptRootRef，挂 ResizeObserver 测高度）
- **不改**：`ai-elements/conversation.tsx`（generated，padding 走业务侧 className）

---

*依据：用户截图（决策卡+画图按钮被输入框遮挡）+ 读码坐实 page.tsx:164 输入框 absolute bottom-0 z-30 悬浮 + message-list ConversationContent 无底部 padding。纯布局修复，不碰流式核心/generated。*
