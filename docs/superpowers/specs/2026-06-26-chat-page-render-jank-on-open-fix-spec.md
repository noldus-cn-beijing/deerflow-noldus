# Spec：对话页打开/切回渲染卡顿——消息流挂载开销治理

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-26
> 代码基线：dev HEAD `1c1033a7`
> 性质：🟡 中 · 前端渲染性能（非流式核心、非消息合并逻辑）
> 取证：真机自动化 CPU profile + Long Task 实测（thread `bd7ca7f7`，owner 账号，本地 dev :2026）

---

## 〇、问题与已坐实根因（先读，避免重走错路）

用户观察：**打开 / 切回对话页时要卡一下才渲染出来**（不是流式过程卡，是整页挂载卡）。

handoff 原猜测「与历史乱序同源、在消息合并/流式层」。**本次自动化 profile 证伪了这个方向**，必须记下，别再往那走：

### 已证伪（不要改这些）

- `dedupeMessagesByIdentity` / `mergeMessages` / `interleaveMessages` / `messageIdentity`：在 17149 个 CPU 采样里**连前 25 都进不去**。它们是 O(n) Map/Set，25 条消息成本可忽略。**#221 重写的归并保序逻辑没有性能代价，不要碰。**
- 流式层 SSE 累积：合成 `visibilitychange` 后 `after_return` 的 Long Task = **0**。

### 已坐实的真根因（CPU profile self-time，dev build）

打开 thread `bd7ca7f7` → **6 个 Long Task，累计 ~716ms，max 173ms，全部在加载阶段**。非 idle 时间几乎 100% 是 React 渲染/commit：

| 热点 | self% | 含义 |
|---|---|---|
| `jsxDEV`/`jsx`/`createElement`/`ReactElement` | ~6% | 一次性创建大量 React element |
| `commitMutationEffectsOnFiber`/`commitLayoutEffectOnFiber`/`recursivelyTraverse*` | ~8% | React 一次 commit 挂载全部消息子树 |
| **`CollapsibleContentImpl.useLayoutEffect`** | **2.5%** | Radix Collapsible 每个 reasoning/task/plan 块同步测量布局 |
| `get/set scrollTop` | ~1.6% | StickToBottom 滚动测量抖动 |

**结构性根因 = 三条叠加**：

1. **消息列表虚拟化阈值 30，本 thread 分组 <30 → 虚拟化未生效**（`virtualized-groups.tsx:33` `VIRTUALIZATION_THRESHOLD = 30`，`message-list.tsx:470` `shouldVirtualize = renderedGroups.length >= VIRTUALIZATION_THRESHOLD`）。25 条消息分组数 <30，走 plain map 路径，**全部重子树（markdown/图表/Collapsible）在一次 commit 同步挂载**。
2. **Radix Collapsible 的 `useLayoutEffect` 同步测高**（ai-elements 的 reasoning/task/plan/chain-of-thought 全用 Collapsible）——每个块挂载时同步读布局，N 个块 = N 次 layout 读，阻塞 commit。
3. **StickToBottom scrollTop 读写**在大挂载期间抖动。

> ⚠️ **dev build 失真说明**：profile 是 `pnpm dev`（Turbopack dev）跑的——`jsxDEV` / `runWithFiberInDEV` / `logComponentRender` / StrictMode 双调 effect 都在采样里，**dev React 比 prod 慢数倍**。所以绝对值 716ms 被放大，prod 会低很多。**但结构（非虚拟化挂载 + Collapsible layout effect）与 build 无关，prod 同样存在、只是程度轻。** 验收必须以 `pnpm build && pnpm start` 的 prod 数为准，别拿 dev 数当目标，避免过度工程。

---

## 一、修复方案（按性价比排序，建议全做；每条独立可回退）

### 修复 1（主治）：降低虚拟化阈值 + 验证小列表也走窗口化

`VIRTUALIZATION_THRESHOLD = 30` 偏高——常见研究会话 10~30 条消息正好落在阈值下、吃满非虚拟化挂载成本。

- 把阈值降到能覆盖典型重会话的值（建议 **12~15**，需 dogfood 校准：低于此值的会话挂载成本可接受、不值得虚拟化测量开销）。
- **风险**：虚拟化叠在第三方 StickToBottom 上是 spec#7（#212）标注的最高风险项，注释明说「若不稳可把阈值设 `Infinity` 退回」。降阈值会让更多会话进虚拟化路径，**必须 dogfood 验证**：流式追加时不跳动、stick-to-bottom 仍工作、动态测高不抖、`data-message-id` 仍在（#214 依赖它）。
- 若降阈值引入流式不稳，改走**修复 2**（不动虚拟化，治挂载本身）。

### 修复 2（与 1 互补，治"首屏一次性挂载"）：分帧/延迟挂载历史消息

不论虚拟化是否生效，**打开 thread 时把全部历史一次 commit** 是卡顿主因。让首屏只同步挂载视口附近的消息，其余在 `requestIdleCallback` / `startTransition` 里分批挂载：

- 历史加载完成后的首次渲染，用 `startTransition` 包住非视口消息的挂载，让浏览器先画出顶部/底部可见部分，再低优先级补齐中间。
- 或：初始只渲染最后 N 条（用户最关心的），向上滚动时再挂载更早的（与虚拟化天然契合）。

### 修复 3（治 Collapsible layout 抖动）：reasoning/task 等折叠块默认折叠 + 懒挂载内容

`CollapsibleContentImpl.useLayoutEffect` 占 2.5%——每个 reasoning/task/plan 块挂载即同步测高。

- 历史消息里的 reasoning/思考链/task 块**默认折叠**（多数研究员看结论不看思考过程），折叠态不挂载/不测量 `CollapsibleContent` 内容，点击展开才挂载。
- 坐实当前这些块默认是否展开；若默认展开，改默认折叠即可砍掉绝大部分 layout effect 成本。
- **守纪律**：这是历史消息的折叠，**不动流式进行中**的展示（进行中思考链该实时可见）。

### 修复 4（可选，低优先）：StickToBottom scrollTop 抖动

scrollTop get/set ~1.6%，挂载期间的滚动测量。优先级最低——修复 1/2 让挂载量下降后这个自然缓解。先不单独动，dogfood 后若仍显著再看。

---

## 二、关键文件

- `packages/agent/frontend/src/components/workspace/messages/virtualized-groups.tsx:33`（`VIRTUALIZATION_THRESHOLD`，修复 1）
- `packages/agent/frontend/src/components/workspace/messages/message-list.tsx:470`（`shouldVirtualize` 判定 + plain map 路径，修复 1/2）
- `packages/agent/frontend/src/components/workspace/messages/message-group.tsx` / `message-list-item.tsx`（消息子树挂载，修复 2/3）
- ai-elements 的 `reasoning.tsx`/`task.tsx`/`plan.tsx`/`chain-of-thought.tsx`（Collapsible 默认态——**注意这些是 registry-pulled，禁直接改**；改默认折叠须在 `workspace/` 的消费侧通过 prop 控制，或包一层 wrapper，不动 registry 源。见 frontend/CLAUDE.md 与 memory `feedback_shadcn_ui_copyin_editable_vs_registry_pulled_generated`）
- token：spec#1 `--ease-brand-out`/`--dur-fast`（若展开/折叠加动效）

> ⚠️ **registry 红线**：ai-elements/* 与 ui/ 里的 MagicUI/React Bits 是 registry 拉取的，re-pull 会覆盖手改。修复 3 要改 Collapsible 默认态，**只能在消费侧（workspace/messages）传 prop 或包 wrapper 控制**，不改 ai-elements 源。改前先核 `components.json` registries 段。

---

## 三、验证（验收以 prod build 为准）

1. `pnpm check`（lint + typecheck）。
2. **prod profile 复测**（关键）：`pnpm build && pnpm start`，复用本次诊断脚本 `/tmp/perf-profile.cjs`（或搬入仓库 scripts）对同一 thread `bd7ca7f7` 录 CPU profile：
   - **目标**：加载阶段 Long Task 累计显著下降（dev 基线 716ms；prod 基线先测出来再定目标，别拿 dev 数当门）。
   - `dedupeMessagesByIdentity`/`mergeMessages` **仍不在热点**（确认没把合并逻辑改回退化版）。
3. **回归（流式红线，必跑）**：
   - `node_modules/.bin/vitest run`（或 `pnpm vitest run`）—— `utils.test.ts` / `mergeMessages.test.ts` / `message-list.perf.test.tsx` / `virtualized-groups.test.tsx` 全绿（降阈值/分帧不得破坏既有断言）。
   - 真机 dogfood：① 打开重 thread 不再明显卡顿 ② **流式追加时不跳动、stick-to-bottom 工作** ③ 切到别 tab 再切回顺滑 ④ `data-message-id` 仍在每条消息（#214 分析轨依赖）⑤ 折叠的 reasoning 块点击能展开。
4. **catastrophic forgetting 自检**（CLAUDE.md 三病理）：降阈值改的是共享渲染路径——grep `VIRTUALIZATION_THRESHOLD` / `shouldVirtualize` 所有消费者，确认没有别处假设阈值=30。

---

## 四、不做 / 边界

- **不碰** `mergeMessages`/`dedupeMessagesByIdentity`/`interleaveMessages`/`messageIdentity`（profile 证伪，#221 红线逻辑保持）。
- **不改** ai-elements / MagicUI / React Bits registry 源（红线）。
- **不追求** dev build 的绝对数达标（dev 失真，以 prod 为准）。
- **真实后台节流 + SSE 突发 flush 那条路本 spec 不覆盖**——本次自动化（headless / headed Playwright 合成 visibilitychange）实证测不到 OS 级后台节流，`after_return` Long Task 恒为 0。若用户后续报告「**仅**切后台跑完再切回才卡、前台开着不卡」，那是另一类问题，需真人最小化窗口跑 live run 录 Performance 单独诊断，不在此 spec。

---

*依据：真机自动化 CPU profile（thread `bd7ca7f7`，owner 账号，本地 dev :2026，17149 采样）—— 加载阶段 6 个 Long Task 累计 716ms 全为 React 渲染/commit（jsxDEV + commit*Effects + Radix CollapsibleContentImpl.useLayoutEffect 2.5% + scrollTop 1.6%）；`dedupeMessagesByIdentity`/`mergeMessages` 不在前 25 热点（证伪 handoff「同源流式层」猜测）；message-list 虚拟化阈值 30、本 thread 分组 <30 故未生效。诊断脚本 `/tmp/perf-profile.cjs` + `/tmp/perf-trace-chat.cjs`。*
