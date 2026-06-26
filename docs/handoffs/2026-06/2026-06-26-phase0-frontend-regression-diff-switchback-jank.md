# Phase 0 前端回归对比：切回卡顿「我们引入了什么」

> spec `2026-06-26-conversation-gallery-empty-progress-rail-semantics-and-switchback-jank-spec.md` §四（问题4）交付物。
> 日期：2026-06-26
> 基线：Phase 0 前端改造前 `fe78b551`（#201 之前）→ HEAD `1eb9d68a`。

## 结论（先读）

切回/打开卡顿的**完整修复已在 #223 合入 dev**（`6d921a36`「治理对话页打开/切回渲染卡顿」），含
jank spec 的三条独立可回退修复：

| jank spec 修复 | #223 实现 | 测试 |
|---|---|---|
| **Fix 1** 降虚拟化阈值 30→15 | `virtualized-groups.tsx` `VIRTUALIZATION_THRESHOLD` | `virtualized-groups.test.tsx` |
| **Fix 2** 分帧挂载历史消息 | `progressive-mount.ts` `useProgressiveMount`（requestIdleCallback 分批） | `progressive-mount.test.ts` |
| **Fix 3** 历史思考块默认折叠 | `message-list-item.tsx` ReasoningPanel `useState(isStreaming)` | `message-list-item.reasoning-collapse.test.tsx` |

本 PR（gallery/rail/viewport）**不重复 #223 的工作**——它已落地且全绿（33 message 测试通过）。

## Phase 0 在消息流上叠加了什么（DeerFlow 原生没有）

git 对比 `fe78b551..HEAD`，Phase 0（spec#1~#8，#201-#217 + 后续）在消息流容器层引入的
**派生组件**（都订阅 `thread.messages` 做 `useMemo` 派生）：

| 组件 | 引入 PR | 派生来源 | 切回时行为 |
|---|---|---|---|
| **AnalysisRail**（spec#4） | #214 | `useRunTrace` + `deriveWorkflowStages` over `messages` | sticky 常驻，messages 变即重推导 |
| **RunTraceWidget**（spec#2/#7） | #203/#212 | `useRunTrace` over `messages` | 入口徽章 + 抽屉，messages 变即重算 |
| **InlineArtifactSummary**（spec#3） | #205/#216 | `normalizeArtifacts` over `thread.values.artifacts` | present-files 块挂载 |
| **DecisionCard**（spec#5） | #217 | clarification 选项 over `messages` | 反问块挂载 |
| **虚拟化阈值 30**（spec#7） | #212 | `shouldVirtualize = groups.length >= 30` | <30 组全量挂载（#223 已降到 15） |

## useMemo-over-messages 是不是切回卡顿根因？——**不是**（profile 证伪）

本会话早先 CPU profile（thread `bd7ca7f7`，17149 采样）的**前 25 热点**里没有
`buildRunTrace` / `deriveWorkflowStages` / `mergeMessages` / `dedupeMessagesByIdentity`。
热点全是 React 渲染/commit 一次性挂载成本：

- `jsxDEV`/`createElement` ~6%（一次性创建大量 element）
- `commitMutationEffectsOnFiber`/`commitLayoutEffectOnFiber` ~8%（一次 commit 挂载全部子树）
- **`CollapsibleContentImpl.useLayoutEffect` 2.5%**（Radix Collapsible 每块同步测高）← #223 Fix 3 治
- `get/set scrollTop` ~1.6%（StickToBottom 抖动）

即：**卡顿 = 一次性挂载 N 个重子树（含 Collapsible layout effect），不是 useMemo 重算**。
`useRunTrace` 的 `useMemo([messages, tasks, t])` 在切回时只重算一次（O(n)，25 条消息可忽略），
非热点。故本 PR **不动这些派生组件的 useMemo 依赖**（守 memory `feedback_perf_is_efficient_impl_not_visual_downgrade`：
是省开销，不是改派生语义；profile 未证其为根因前不盲改，避免 over-engineering）。

## 治法归属

- **挂载成本**（jsxDEV/commit/Collapsible）→ #223 已治（降阈值 + 分帧 + 折叠）。
- **派生重算**（buildRunTrace 等）→ 非热点，不动。
- 本 PR 额外修的 **视口高度链**（§四点五，底部裁切 + 滚动条不见）与卡顿同属消息流容器层，
  但根因独立（`size-full` 溢出 + `overflow-y-hidden`），本 PR 已修。

## 验收（守 jank spec §三）

- `pnpm check`（lint + typecheck）✅（本 PR）。
- 流式回归测试全绿：`utils.test.ts` / `mergeMessages.test.ts` / `message-list.perf.test.tsx` /
  `virtualized-groups.test.tsx` / `progressive-mount.test.ts` /
  `message-list-item.reasoning-collapse.test.tsx` ✅（33 message 测试通过；仅 `utils.test.ts`
  2 个 pre-existing 红 baseline 非 jank 相关，见 memory `project_frontend_vitest_already_set_up_2_red_streaming_tests`）。
- `dedupeMessagesByIdentity`/`mergeMessages` 不在 profile 热点 ✅（#221 红线逻辑保持）。
- prod build Long Task 下降：#223 已落地，prod 复测以 #223 的 dogfood 为准（本 PR 不重测）。
