# 前端 Phase0#7 运行时性能 — 实施 handoff

> 日期：2026-06-25
> Spec：[2026-06-24-frontend-phase0-7-runtime-performance-spec.md](../../specs/2026-06-24-frontend-phase0-7-runtime-performance-spec.md)
> PR：见 `gh pr view`（分支 `worktree-frontend-phase0-7-runtime-performance` ← `dev`）

## 一句话

在前端**渲染层**治三个真实卡顿热点——流式 token 全列重渲染、热组件无 memo 边界、重模块全量上首屏——外加一处真违规的 box-shadow 动画。**绝不碰 `core/threads/hooks.ts` 的 SSE/合并核心**（diff 自证）。

## 实施期对 spec 前提的修正（重要）

调研发现 spec 部分前提已过时（spec 撰写于 #202-#207 前），实施按真实代码走：

1. **没有 per-message `backdrop-blur`**：spec §5.5 整段前提（"现状 message.tsx per-message blur 玻璃保留但改省写法"）作废——messages/ 目录无任何 per-message 玻璃层。Step 5.5 跳过此子项（不存在的东西无需优化）。
2. **`groupMessages` 已是 `groupMessages(messages, mapper): T[]`**（`utils.ts:152`），不是 spec §3.2 草图的"render 内每次重算未 memo"形态。但 perf 问题真实：`thread.messages` 在 `useThreadStream` 每次 render 都是新引用（`hooks.ts:813-830` `mergeMessages` 重算 + `...thread` 新对象），致 `getMessageGroups`（O(n)）每 render 都跑。修法仍是 `useDeferredValue` + `useMemo`。
3. **`transition-all` 基本都在不可改 primitive 里**：2 处 `ai-elements/`（registry 禁改）+ 4 处 Shadcn copy-in（API-preserving 约束）。**唯一真违规 = `globals.css:549` `pulse-soft` keyframe 动 `box-shadow`**，已修。

用户拍板边界：① 只修真违规可编辑项 ② 虚拟化尝试，不稳标已知限制 ③ 引入 `@testing-library/react` + jsdom 做组件测试。

## 落地清单（按 Step）

| Step | 内容 | 文件 | 状态 |
|---|---|---|---|
| 1 | `useDeferredValue(thread.messages)` + `useMemo(groupMessages)` | `message-list.tsx` | ✅ |
| 2 | `MessageListItem`/`MessageGroup`/`SubtaskCard` 包 `React.memo` | 三个组件 | ✅ |
| 3 | `RunTraceDrawer` 经 `next/dynamic` 懒加载（`everOpened` 门控，首次打开才请求 chunk） | `run-trace-widget.tsx` | ✅ |
| 4 | 消息列表虚拟化（`@tanstack/react-virtual`，≥30 组才开，复用 `StickToBottom` 的 `contextRef` 取滚动元素） | `virtualized-groups.tsx` + `message-list.tsx` | ✅（可降级） |
| 5 | `pulse-soft` 改 transform/opacity-only（伪元素 opacity/scale 脉动，box-shadow 静态） | `globals.css` | ✅ |
| 6 | reduced-motion 校验（全局 `*` 块 + pulse-soft 显式块） | — | ✅ |

## 铁律遵守（diff 自证）

- **`core/threads/hooks.ts` 零改动**——`useStream`/`mergeMessages`/`dedupeMessagesByIdentity`/optimistic/summarization 一行未碰。节流全加在消费侧接缝（`message-list.tsx` 读 `thread.messages` 之后）。
- **`groupMessages`/`getMessageGroups` 分组算法零改动**——只套 `useMemo` 壳。
- **不引新状态/渲染框架**——React 19 `useDeferredValue`/`useMemo`/`memo` + 已装 `react-virtual`。
- **零后端改动**。
- registry-pulled primitive（`ai-elements/`、`magic-bento.css`、`word-rotate.tsx`）未碰。

## 测试

- **新增测试基建**：`@testing-library/react` + `@testing-library/jest-dom` + `jsdom`（devDeps）。`vitest.config.ts` 加 `setupFiles`（jest-dom + ResizeObserver stub）。组件测试用 per-file `// @vitest-environment jsdom` docblock，**纯逻辑测试保持 Node 环境不变**（`utils.test.ts`/`stream-error.test.ts` 不受影响）。
- **新增测试**：
  - `message-list.perf.test.tsx`：spy `groupMessages`，断言无关重渲染（同 messages 引用）不重算 + messages 变化时重算。
  - `message-list-item.memo.test.tsx`：Profiler 等价——mock 叶子组件计 render 数，断言相同 props 下父级重渲染不穿透子树。
  - `virtualized-groups.test.tsx`：threshold 常量 + 大列表不崩 + trailing 必渲染 + null scroll context 不崩。
  - `test-infra.smoke.test.tsx`：jsdom + jest-dom setup 烟测。
- **结果**：`pnpm test` = **98 pass / 2 fail**（2 fail 是 pre-existing baseline：`utils.test.ts` 的 `isStreaming` pinning，非本次回归，见 memory `project_frontend_vitest_already_set_up_2_red_streaming_tests`）。
- **lint**：净减（baseline 7 problems → 现 4，全在 pre-existing 未碰文件；`lint:fix` 顺手清了所碰文件的 import/order，无关文件 quality-warning-banner/workspace-header 的 lint:fix 改动已 revert 保持 PR 聚焦）。
- **typecheck**：0 错。**build**：`pnpm build` 成功（Next 16 Turbopack，`next/dynamic` + 虚拟化 + CSS 全编译通过）。

## 已知限制（按 spec §3.1/Step4 明示，不静默）

- **Step 4 虚拟化的真窗口化（DOM 节点恒定）未在单测断言**：`@tanstack/react-virtual` 依赖真实 DOM 几何（`clientHeight`/`ResizeObserver`/`getBoundingClientRect`），jsdom 不计算布局，无法在单测里断言"挂载节点 < 总数"。该验收项需**手动**验：DevTools Elements 面板，50+ 组滚动时数 DOM 节点（spec §五）。
- **虚拟化与 `StickToBottom`（registry Conversation）的组合是较高风险项**：通过 `contextRef` 只读滚动元素（未改 registry 组件）。若 dogfood 出现展开态跳变/流式追加抖动/贴底断裂，降级方法：把 `VIRTUALIZATION_THRESHOLD` 调到 `Infinity`（或从 `message-list.tsx` 移除 `VirtualizedGroups`）——Step 1-3 的 defer+memo 已消除主卡顿源，虚拟化是叠加增益。

## 手动验收待办（PR body 已列，需 dogfood 实跑）

| 项 | 工具 | 预期 |
|---|---|---|
| 流式 token 暴雨主线程帧 | Performance 面板 | 无连续长帧（<16ms） |
| 交互延迟 | input-latency | <100ms |
| `groupMessages` 不每 token 重算 | React DevTools Profiler | ✓ |
| 历史消息组重渲染 = 0 | Profiler | ✓（memo 生效） |
| 50+ 组 DOM 恒定 | Elements 面板 | 或标已知限制 |
| 首屏 bundle 下降 | `next build` 输出对比 | RunTraceDrawer 移出主 chunk |
| CLS / TTI / TBT | Lighthouse | CLS<0.1 |
| 长会话合成层稳定 + GPU 显存不涨 | Layers / Performance Monitor | pulse-soft 改省写法后改善 |

## before/after 性能基线（回归锚）

> 待 dogfood 实跑后填入数值。当前仅记录"改了什么"，数值验收见上表。

- **改前**：`thread.messages` 每次 render 新引用 → `groupMessages` 每 token 全量重算 + 全列 diff；`MessageListItem`/`MessageGroup`/`SubtaskCard` 无 memo，流式时历史消息全量重渲染；`RunTraceDrawer` 全量进首屏 bundle；`pulse-soft` 每 frame 重绘 box-shadow。
- **改后**：defer + memo 把重算/重渲染降到"仅消息变化时"；drawer 按需加载；pulse-soft 只动 transform/opacity（合成器）。

## milestone 建议

本 PR 完成母方案 [前端/人机体验大升级](../../../plans/2026-06-24-frontend-generative-ux-upgrade.md) Phase 0 的 **#7 运行时性能**。至此 Phase 0 八项中已完成 #1（token）/ #2（trace）/ #3（gallery）/ #6（设计语言）/ #7（本 PR）；仍待实施 **#4 rail / #5 决策卡 / #8 多文件上传**。建议 milestone 索引更新 #7 状态为「已实施，待 dogfood 数值验收」。
