# Spec：前端三问题修复（输入框遮挡 + 移除模型选择器 + 输入框 HCI 打磨 + 运行轨迹空 panel）

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-26
> 代码基线：dev HEAD `1c1033a7`
> 性质：🟡 中 · 纯前端（布局 + 组件移除 + HCI 打磨 + 一处数据取证后修）
> 取证：owner 账号本地 dev（:2026）+ 后端 checkpoint 解码 + 读码坐实。dogfood thread `339512dd-96dc-4a26-a712-71dee5c74c27`

---

## 〇、四件事（用户拍板）

1. **输入框遮挡对话流最后输出**——浮动输入框盖住最后一条消息。
2. **输入框 HCI 打磨**——间距/圆角/阴影 + 按钮布局/对齐 + placeholder/行为，三方面都按 HCI 最佳实践 + 日式克制重做。
3. **移除输入框里的模型选择器**——右下角 `deepseek` 模型名连同整个 ModelSelector 一起去掉（用户：**整个移除**，不再让用户切模型）。
4. **运行轨迹 panel 空**——右上角"N 步进行中"点开后"运行轨迹"抽屉只有标题没内容。

---

## 一、问题 1：输入框遮挡（#219 漏改 agents 路由）

### 根因（已坐实）

#219（commit `613c445f`）的动态 paddingBottom 修复**只加在标准路由** `src/app/workspace/chats/[thread_id]/page.tsx`（line 42 state + 118-120 callback + 129-133 计算 + 给 InputBox 传 `onHeightChange`）。

**agents 路由 `src/app/workspace/agents/[agent_name]/chats/[thread_id]/page.tsx:111-114` 仍是静态版**：

```tsx
const messageListPaddingBottom = showFollowups
  ? MESSAGE_LIST_DEFAULT_PADDING_BOTTOM + MESSAGE_LIST_FOLLOWUPS_EXTRA_PADDING_BOTTOM
  : undefined;
```

没有 `inputBoxHeight` state、**没给 InputBox 传 `onHeightChange`**——输入框一旦长高（多行/附件/followups）就盖住最后消息。这是 catastrophic-forgetting 式漏改（同一逻辑两路由只改一条）。

### 测高机制已正确（无需改）

`src/components/workspace/input-box.tsx:166-181` 的 ResizeObserver 已正确：measure `promptRootRef.current.offsetHeight` → 设 `--input-box-height` CSS 变量 + 调 `onHeightChange?.(h)`，对所有态（followups/decision-card/初始）可靠触发。spacer `src/components/workspace/messages/message-list.tsx:474`（`<div style={{height: paddingBottom}}/>`）已正确在滚动容器内、最后消息后。**问题纯粹是 agents 路由没把 prop 接上。**

### 修复

把标准路由的动态逻辑搬到 agents 路由：
- 加 `const [inputBoxHeight, setInputBoxHeight] = useState<number>(0);`
- 加 `handleInputBoxHeightChange` callback（同标准路由 118-120）。
- `messageListPaddingBottom` 改动态版（`inputBoxHeight > 0 ? inputBoxHeight + INPUT_BOX_PADDING_BREATHING_ROOM_PX : MESSAGE_LIST_DEFAULT_PADDING_BOTTOM` + followups extra，同标准路由 129-133）。
- 给该路由 `<InputBox>` 传 `onHeightChange={handleInputBoxHeightChange}`。

### ⚠️ 常量陷阱（必做，否则又埋 drift）

`INPUT_BOX_PADDING_BREATHING_ROOM_PX = 24` 是标准路由 `page.tsx:37` 的**模块内局部 const**，agents 路由 import 不到。**把它抽到共享模块导出**（建议放 `message-list.tsx` 与 `MESSAGE_LIST_DEFAULT_PADDING_BOTTOM`/`MESSAGE_LIST_FOLLOWUPS_EXTRA_PADDING_BOTTOM` 同处），两路由都 import。**别复制常量**——复制就是下一次 drift。改完 grep 确认两路由都 import 同一个 const、且都 wire 了 `onHeightChange`。

---

## 二、问题 2：移除整个模型选择器（用户：整个移除）

### 位置（已坐实）

`src/components/workspace/input-box.tsx:617-655`，右侧 `PromptInputTools` 里的整段 `<ModelSelector>...</ModelSelector>`，其中 line ~626 `<ModelSelectorName>{selectedModel?.display_name}</ModelSelectorName>` 就是右下角那个 `deepseek`。

### 修复

- **整段删除** `<ModelSelector>` 到 `</ModelSelector>`（line 618-655）。删后右侧 `PromptInputTools` 只剩 `<PromptInputSubmit>`（line 656-661，保留）。
- 清理随之 unused 的 import：`ModelSelector`/`ModelSelectorTrigger`/`ModelSelectorName`/`ModelSelectorContent`/`ModelSelectorInput`/`ModelSelectorList`/`ModelSelectorItem`、`modelDialogOpen` state（`setModelDialogOpen`）、`handleModelSelect`（确认仅此处用再删）。
- **保留模型内部选定逻辑不破坏**：`context.model_name` 默认解析 + fallback（line 187-209）继续存在——发送仍带正确 model_name，只是 UI 不暴露切换。
- `selectedModel`（line 204-209）若删 selector 后仍被 `supports_thinking`（line ~214）等消费 → 保留该 useMemo；若再无消费者则一并删，避免 `pnpm check` unused 报错。**逐个核 `selectedModel` 引用点再决定删/留。**

---

## 三、问题 1b：输入框 HCI 打磨（三维度都修）

### ⚠️ registry 红线（先读，决定改法）

`PromptInput*` / `ModelSelector*` / `PromptInputSubmit` / `PromptInputTextarea` / `PromptInputFooter` / `PromptInputTools` 全来自 `@/components/ai-elements/prompt-input`——**registry-pulled（`components.json` registries `@ai-elements`），禁改源**（re-pull 会覆盖）。**所有 HCI 打磨只能在消费侧 `input-box.tsx` 通过 className override / 外层 wrapper 做，不动 ai-elements 源。** 守 memory `feedback_shadcn_ui_copyin_editable_vs_registry_pulled_generated`。

设计纪律：日式克制（memory `feedback_frontend_design_japanese_minimal_motion_craft`）+ spec#1/#6 token（`--shadow-rest`/`--shadow-modal`、`--ease-brand-out`/`--dur-*`）。**性能即高效实现非视觉降级**（memory `feedback_perf_is_efficient_impl_not_visual_downgrade`）。

### A. 间距 / 圆角 / 阴影

- 浮动框与视口底部/两侧留白对称统一（核当前外层容器 margin/padding 是否左右不均）。
- 圆角与全局体系一致（别让输入框单独一个怪半径）；阴影统一走 `--shadow-rest`（静置）/ `--shadow-modal`（聚焦/悬浮），不自定义杂阴影值。
- textarea 当前 `min-h-14 py-4 text-[15px] leading-6`（`input-box.tsx:506`）——确认上下内边距对称、首行不贴边、与 footer 间距协调。

### B. 按钮布局 / 对齐

- footer 当前裸 `flex`（`input-box.tsx:517` `<PromptInputFooter className="flex">`）：左 `PromptInputTools`（附件 `AddAttachmentsButton` + 数据飞轮 mode menu）+ 右 `PromptInputTools`（移除 ModelSelector 后只剩 Submit）。**改成 `justify-between` + `items-center`**，左右组干净分列、垂直居中。
- 移除 ModelSelector 后右侧只剩 Submit（`rounded-full bg-brand`，主操作，保持），重新平衡左右视觉权重。
- 左侧 `px-2!`/`gap-1!`（line 528/533）用了 `!important` hack——统一附件钮 / mode toggle 的 size / gap / icon 权重，能去 `!` 就去（用规范 className 而非 important 覆盖）。

### C. placeholder / 行为

- 审 placeholder 文案：`t.inputBox.placeholder` 与 `awaitingClarification` 态 `t.clarification.awaitingPlaceholder`（line 508-512）。若改文案走 i18n 三齐（key + en-US + zh-CN）。
- 确认 textarea 随字增高顺滑、focus 态/空态表现自然（聚焦边框/环用 token，不突兀）。

> 改文案涉 i18n：`src/core/i18n/locales/{en-US,zh-CN}.ts` 必须 key + en + zh 三齐（memory PR review 方法论 ④）。

---

## 四、问题 3：运行轨迹 panel 空（根因未坐实，实现前必先取证）

### 已排除（别往这改）

- widget 徽章与 panel 内容**用同一个 `useRunTrace({messages,t})`**（`run-trace-widget.tsx:47` + `run-trace-panel.tsx:35`），同一 `SubtasksProvider`、同一 `messages` prop、drawer 无独立 Portal（`run-trace-drawer.tsx` 直接 `DialogPrimitive.Content` 在 widget 子树内）。**所以"徽章2步 + panel空"在纯逻辑上自相矛盾**——Explore 提的"SubtaskContext race / 数据源分叉"假设与代码矛盾，**已证伪，不要按它改**。
- 后端 checkpoint 实测该 thread **47 条消息、20 条 AI 带 tool_calls**（含 `task`×4 / `set_experiment_paradigm` / `ask_clarification`×2 等），`buildRunTrace`（`src/core/trace/build-run-trace.ts:139`）理应从这些 tool_calls 产出多个事件。
- headless 复现时该 thread **历史根本没加载（0 消息渲染）**——这是自动化 harness 局限（用户截图明确显示消息渲染了），**不是用户的 bug**，别据此误判。

### 实现第一步：浏览器内坐实根因（在能正常加载该 thread 的真机/会话里）

打开 thread `339512dd...`，消息渲染出来后，在 console 取 `thread.messages`，对照后端 checkpoint 的 20 条 tool_calls，判断落哪个分支：

1. **若 reload 后 `thread.messages` 的 AI 消息丢了 `tool_calls`** → 根因在前端历史加载路径：`src/core/threads/hooks.ts:978-986` 的 `/run/{id}/messages` 拉取后 `.filter(...).map((m) => m.content)`，或后端 `/messages` 端点返回的消息形状缺 tool_calls。修法 = 让 reload 的历史消息保留 tool_calls（核 `m.content` 是否含 `tool_calls` 字段；缺则后端端点补，或前端映射别丢）。
2. **若 `thread.messages` 有 tool_calls 但 panel 仍空** → 根因在 `buildRunTrace`（如 `message.type !== "ai"` continue（line 144）对 reload 消息类型判定失效——reload 的消息 `type` 可能不是 `"ai"`），或 panel/drawer 内容被 CSS clip（截图看着像 panel body 被裁，需核 `run-trace-drawer.tsx` 的 `min-h-0 flex-1` 容器 + `run-trace-panel.tsx` 的 ScrollArea 高度）。
3. **若是 `ask_clarification` 等被 summarize 算进 running、但对应 events 渲染被 buildRunTrace 某分支吞** → 查该 tool 名在 `buildRunTrace` 是否有对应 push 分支（line 160-247 逐 toolCall 分发；没匹配分支的 toolCall 可能既不计 step 也不渲染，或反之）。

> **红线**：`buildRunTrace` / `useRunTrace` 是 #221 之后的敏感区（spec §六流式只读消费侧）。**必须先取证再改**，不可凭推断动它。改完跑 trace 相关单测（见验证）。

---

## 五、关键文件

| 问题 | 文件 |
|---|---|
| 1 遮挡 | `src/app/workspace/agents/[agent_name]/chats/[thread_id]/page.tsx`（补 #219 动态 padding）；`src/app/workspace/chats/[thread_id]/page.tsx`（参照源 + 抽 `INPUT_BOX_PADDING_BREATHING_ROOM_PX` 常量）；`src/components/workspace/messages/message-list.tsx`（共享常量落点） |
| 2 移除选择器 + 1b HCI | `src/components/workspace/input-box.tsx`（删 ModelSelector + className 打磨；**不动 ai-elements 源**） |
| 1b/1c i18n | `src/core/i18n/locales/{en-US,zh-CN}.ts`（若改 placeholder 文案，三齐） |
| 3 trace 空 | 取证后定：`src/core/threads/hooks.ts:978-986` 或 `src/core/trace/build-run-trace.ts` 或 `run-trace-panel.tsx`/`run-trace-drawer.tsx`（CSS clip） |

---

## 六、验证

1. `pnpm check`（lint + typecheck，必过）。
2. **问题1**：**标准路由 + agents 路由都测**——打开有内容的 thread，输入框单行/多行/带附件/带 followups 四态，最后一条消息都不被遮挡；ResizeObserver 触发后 spacer 高度跟随输入框高度变化。
3. **问题2**：输入框右下角不再有模型名/选择器；模型仍正确选中（发送请求带正确 `model_name`，`supports_thinking` 等依赖不报错）。
4. **问题1b**：浮动框间距对称、圆角/阴影统一、footer 左右组 `justify-between` 对齐、Submit 权重正确；placeholder 正常；focus/空态自然。出效果图给用户拍板（memory `feedback_frontend_design_japanese_minimal_motion_craft`）。
5. **问题3**：reload thread `339512dd...` → 点开运行轨迹 → panel 显示事件时间线（非空 spinner）；**徽章步数与 panel 节点数一致**。
6. **回归（流式红线，必跑）**：worktree 用 `node_modules/.bin/vitest run`（别用 npx，memory PR#221 教训）——`mergeMessages.test.ts` / `utils.test.ts` / `build-run-trace` 相关测试 / `message-list.perf.test.tsx` / `virtualized-groups.test.tsx` 全绿。
7. **catastrophic forgetting 自检**（CLAUDE.md 三病理）：问题1 改的是两路由共享的布局常量/逻辑——grep `INPUT_BOX_PADDING_BREATHING_ROOM_PX` / `onHeightChange` 所有消费者，确认两条 chat 路由都接上、无第三条路由遗漏。
8. 真机 dogfood owner 账号（:2026，`qiuyang.wang@noldus.com.cn` / `19961031`，注意 `.com.cn`）复看四项。

---

## 七、不做 / 边界

- **不改** ai-elements / MagicUI / React Bits registry 源（红线，HCI 打磨走消费侧 className）。
- **不凭推断改** `buildRunTrace`/`useRunTrace`（#221 敏感区）——问题3 必须先浏览器取证。
- 问题1b 是设计打磨，**第一版出效果图与用户对齐**再定稿，不闷头大改。

---

*依据：后端 checkpoint 解码（thread `339512dd...` 47 消息/20 带 tool_calls）+ 读码坐实（#219 仅标准路由、agents 路由 `page.tsx:111-114` 静态 padding 未 wire `onHeightChange`；模型名 `input-box.tsx:626`；ai-elements registry 红线 `components.json`）+ 真机自动化（headless 该 thread 历史未加载＝harness 局限非 bug，故 panel 空根因留作实现时浏览器取证）。诊断脚本 `/tmp/inspect-thread.cjs`、截图 `/tmp/thread-state.png`。*
