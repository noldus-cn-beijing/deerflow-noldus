# Spec：运行时性能与丝滑度（Phase 0 · 第 7 项）

> 类型：**一次性实施 spec**（前端渲染层性能优化，**零后端、绝不改流式核心**）
> 日期：2026-06-25
> 母方案：[docs/plans/2026-06-24-frontend-generative-ux-upgrade.md](../../plans/2026-06-24-frontend-generative-ux-upgrade.md)（§1.1 地基"够用且现代" + §9 工程纪律"不动流式核心"）
> 依赖：[2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md](2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md)（动效 transform-only/reduced-motion）；与 [#3 artifact-gallery](2026-06-24-frontend-phase0-3-artifact-gallery-spec.md) 的虚拟化**同一引擎复用**（`@tanstack/react-virtual`）
> 适用层：`packages/agent/frontend/src/components/workspace/messages/`（消息列表渲染）+ `core/threads/hooks.ts` 的**消费侧**（不改合并侧）+ `app/`（代码分割）。**全部在 `useStream`/`mergeMessages`/`dedupe` 的下游渲染层，不进上游合并逻辑。**
> 设计准则来源：`ui-ux-pro-max`（`virtualize-lists` / `main-thread-budget` 16ms/60fps / `transform-performance` / `debounce-throttle` / `reduce-reflows` / `content-jumping` CLS / `lazy-loading` / `bundle-splitting` / `progressive-loading` / `input-latency` <100ms）
> 一句话：用户怕"开始用就卡又慢"。调研坌实**三个真实卡顿主因**——① 消息列表 `message-list.tsx:75` 纯 `map()` **未虚拟化** ② `groupMessages` **render 内每次重算未 memo** + 流式 `setMessages` **零节流**致每个 SSE token 全列重渲染 ③ `next/dynamic` **零使用**（画廊/轨迹/图表全量上首屏）。本 spec **只在渲染层**加：消息列表虚拟化 + 流式更新 `useDeferredValue` 节流 + `groupMessages` memo + 重组件 memo 边界 + 路由级代码分割 + 60fps transform-only 动效 + 防 CLS。**铁律：绝不碰 `useStream`/`mergeMessages`/`dedupeMessagesByIdentity`/optimistic/summarization（踩坑沉淀，重写必复发）——节流加在 `thread.messages` 被列表消费的接缝处，不进 SSE handler。**
>
> **2026-06-25 补强：§3.7 视觉特效的高效实现（硬闸门，所有视觉 spec #1/#3/#5/#6/#8 必过）**。用户顾虑（已澄清）："新增很多特效要简单有效——**不是视觉降级,而是写代码别繁琐造成额外不必要的开销**。" §3.1-§3.6 治渲染管线,§3.7 治**特效的实现方式**:同一个玻璃/阴影/动效,笨重写法卡、高效写法丝滑——**保视觉,省开销**。硬规则:动画只 transform/opacity(同视觉不重排)、特效组件 memo+稳定 props(不被无关 render 带着重建)、同质重复层在肉眼无差时合并(消除冗余非删视觉)、will-change 不常驻、资源用完释放。**现状 `message.tsx` per-message blur 玻璃保留但改省写法**(不随每条消息重渲染重建)。进 §五红线,违反不准合。**默认保视觉,只优化实现。**

---

## 〇、为什么需要这份 spec（先量后改）

母方案 §1.1 说前端地基"够用且现代"——**架构对，但运行时性能没专门治过**。用户 2026-06-25 原话：

> "我们也需要对于前端的响应速度、刷新速率，是不是足够丝滑来进行优化。就怕用户开始用，但是又卡又慢。"

EthoInsight 的负载特征**天生考验渲染性能**：一次分析几十条消息 + 上百 per_subject 工具调用 + 高频 SSE 流式 token + 上百张图。这正是"未虚拟化 + 零节流 + 零分割"会暴露的场景。**先量（§一带证据），再只在渲染层改（§三），不碰踩坑沉淀的合并核心。**

> **守 CLAUDE.md 反 under-exploration + reward-hacking**：性能优化最容易"改了 prompt/加了感觉"而无实测。本 spec 每条改动配**可测验收**（Performance 面板帧数 / DOM 节点数 / React DevTools Profiler 重渲染次数 / Lighthouse），不靠"感觉变快了"。

---

## 一、现状（调研所得，带证据 —— 三个真实热点）

### 1.1 热点 ①：消息列表未虚拟化（DOM 随消息数线性增长）
- `message-list.tsx:71-75`：`groupMessages(messages, ...)` 后 `group.messages.map((msg) => {...})` **全量渲染**——无 windowing。
- 一次复杂分析几十组消息，每组内含 subtask 卡 / 工具调用 / reasoning，**全部常驻 DOM**。滚动时 layout/paint 全量，且与硬伤 A 的图墙叠加（画廊已在 #3 单独治，但消息流本身的卡片也未虚拟化）。
- `ui-ux-pro-max` `virtualize-lists`：**50+ 项必虚拟化**。消息流轻松超 50 个可视块。

### 1.2 热点 ②：流式更新零节流 + groupMessages render 内重算（每 token 全列重渲染）
- `hooks.ts:867/934/1004`：`setMessages` 在 SSE 事件里**直接调用，无 `throttle`/`startTransition`/`useDeferredValue`**。流式时每个 token 一次 `setMessages` → 一次 re-render。
- `message-list.tsx:62/71`：`const messages = thread.messages` 后 **`groupMessages(messages)` 在 render 体内直接调用，未 `useMemo`**——每次 render（含每个流式 token 触发的）都**重新分组全部消息**。
- 叠加效果：流式高频 → 每 token 触发 `groupMessages` 全量重算 + 全列 diff。这是**最主要的流式卡顿源**。
- `ui-ux-pro-max` `debounce-throttle`（高频事件节流）+ `main-thread-budget`（每帧 <16ms 保 60fps）。

### 1.3 热点 ③：next/dynamic 零使用（重模块全量上首屏）
- grep `next/dynamic` / `dynamic(` 在 `components`+`app`+`core` = **0 处**。
- 后果：画廊（含虚拟化库）、运行轨迹抽屉、图表/markdown 渲染、artifact 详情 iframe 等**重模块全部进首屏 bundle**，拖慢首次加载（TTI）。
- `ui-ux-pro-max` `bundle-splitting` / `lazy-loading`：按路由/feature 分割，非首屏组件 `dynamic import`。

### 1.4 已有的好底子（保留、复用）
| 已有 | 位置 | 用途 |
|---|---|---|
| 部分 memo | `message-group.tsx`(5) / `message-list-item.tsx`(6) / `subtask-card.tsx`(3) | 已有 memo 意识,但被 §1.1/§1.2 的列表级缺口抵消 |
| reduced-motion 全局降级 | `globals.css:357/422` | perf + a11y 共用闸,新动效接入 |
| `@tanstack/react-query` | `package.json:48` | 同源 `@tanstack/react-virtual`（#3 已选,本 spec 复用同引擎） |
| `motion@12` 用量克制（5 处） | shimmer/flip/word-rotate/number-ticker/terminal | spring 未滥用,保持 |
| tabular-nums | `globals.css:332` | 流式数字防跳（spec#1 推广） |

### 1.5 安全接缝（关键：节流加在哪不碰核心）
`message-list.tsx:62` `const messages = thread.messages` → `groupMessages(messages)` 是**消费侧**，与 `hooks.ts` 的**合并侧**（`setMessages`/`mergeMessages`/`dedupe`）隔开。**节流/defer 全部加在这条消费接缝**（list 拿到 messages 之后、分组之前/之后），`hooks.ts` 的 SSE handler **一行不动**——这样既丝滑又不碰踩坑沉淀。

---

## 二、目标与非目标

### 目标
1. **消息列表虚拟化**：长会话滚动时 DOM 节点恒定（不随消息数线性增长），复用 #3 的 `@tanstack/react-virtual`。
2. **流式渲染节流**：高频 SSE 更新经 `useDeferredValue`/render 层节流，token 流式时不每帧全列重渲染；`groupMessages` `useMemo` 缓存。
3. **重渲染边界**：热组件（消息项/分组/subtask 卡）`React.memo` + 稳定 props，切断无关重渲染。
4. **代码分割**：画廊/轨迹抽屉/图表/markdown/artifact 详情等非首屏重模块 `next/dynamic` 懒加载。
5. **60fps 动效**：所有动效 transform/opacity-only（不动 layout），骨架屏替空转，防 CLS。
6. **可测验收**：每条配 Performance/Profiler/Lighthouse 可测指标（不靠"感觉"）。

### 非目标
- ❌ **绝不改流式核心**：`useStream` / `mergeMessages` / `dedupeMessagesByIdentity` / optimistic / summarization 一行不动（CLAUDE.md/memory 铁律,踩坑沉淀,重写必复发）。节流只在**消费侧接缝**（§1.5）。
- ❌ 不改 `groupMessages` 的**分组算法**（只给它套 `useMemo` 缓存,不改逻辑）。
- ❌ 不引新状态库/新渲染框架（用 React 19 现成 `useDeferredValue`/`useMemo`/`memo` + 已装 react-virtual）。
- ❌ 不做 SSR/边缘渲染改造（超 Phase 0）。
- ❌ 画廊图片虚拟化/缩略图**在 #3 做**（本 spec 不重复,只做"消息流卡片"虚拟化 + 全局性能基线）。
- ❌ 不动后端（零后端依赖）。

---

## 三、设计

### 3.1 消息列表虚拟化（热点 ①）

- 用 `@tanstack/react-virtual`（headless,与 #3 同引擎,`package.json` 已有 react-query 同源）对 `groupMessages` 后的**分组列表**做 windowing：任意时刻只挂载可视区 + 缓冲的若干组,滚出真正卸载。
- **难点 = 动态高度**：消息组高度不定（reasoning 展开/折叠、图、表）。用 react-virtual 的 `measureElement`（动态测量）+ `estimateSize` 兜底。
- **流式追加场景**：新消息在底部追加 + 自动滚到底（stick-to-bottom）。虚拟化要兼容"追加时保持贴底"——用 `scrollToIndex(last)` 或 react-virtual 的 reverse/sticky 方案。**当前无 scrollIntoView 实现**（grep 证实），所以贴底逻辑随虚拟化一并设计,不破坏现有滚动。
- **降级**：消息数 < 阈值（如 30 组）时可不虚拟化（直接 map,避免小列表虚拟化的测量开销）；超阈值才开。

> ⚠️ 虚拟化消息流比虚拟化图墙难（高度动态 + 流式追加 + 贴底 + 展开态）。**实施期若动态高度虚拟化不稳,退而求其次**：先做 §3.2 节流 + §3.3 memo（这两个收益大且低风险）,消息流虚拟化作本 spec 内**可独立推进的较高风险项**,不阻塞其余。验收按"50+ 组时 DOM 节点恒定"判,做不到就标记为已知限制 + 后续迭代（不静默假装做了——`ui-ux-pro-max` 无声截断同理）。

### 3.2 流式更新节流：`useDeferredValue` + `groupMessages` memo（热点 ②，最高收益）

**在消费接缝（`message-list.tsx`）改,不碰 `hooks.ts`**：

```tsx
// message-list.tsx —— 现状: const messages = thread.messages; groupMessages(messages) 在 render 体
// 改为:
const messages = thread.messages;
// 1) defer: 流式高频更新时,React 可中断/合并渲染,优先保持交互响应（React 19 并发）
const deferredMessages = useDeferredValue(messages);
// 2) memo: 分组只在 deferredMessages 变化时重算,不是每 render
const groups = useMemo(() => groupMessages(deferredMessages, /*...*/), [deferredMessages /*, 其他稳定依赖*/]);
```

- `useDeferredValue`：React 19 并发特性,流式 token 暴雨时,渲染落后于最新值但**主线程不被每 token 阻塞**,交互（点击/滚动）保持响应。**纯消费侧,零碰合并核心。**
- `useMemo(groupMessages)`：把"每 render 全量重分组"降为"仅消息变化时分组"。`groupMessages` **逻辑不改**,只加缓存壳。
- **正在流式的那一条**仍要实时（用户要看 token 蹦出来）：`useDeferredValue` 只 defer **重渲染优先级**,不是延迟显示;视觉上仍连续,只是高负载时合并掉中间帧（正是 60fps 要的）。
- 可选增强：若 defer 仍不够,对"流式中的活跃消息"与"已完成的历史消息"**分治**——历史部分 memo 到极致（几乎不重渲染）,只活跃尾部高频更新。但**先上 defer+memo 量一版**,够了就不加复杂度（避免过度工程）。

> 这一条是**最高性价比 + 最低风险**：纯渲染层 React 原语,不动核心,直接干掉"每 token 全列重算"主卡顿源。**应最先做。**

### 3.3 重渲染边界：memo + 稳定 props（热点 ②配套）

- 热组件 `React.memo`：`SubtaskCard` / `MessageGroup` / `MessageListItem`（已部分有,补全 + 校验 props 稳定性）。
- **props 必须稳定否则 memo 失效**：传给 memo 组件的回调用 `useCallback`、对象/数组用 `useMemo`。重点查 `message-list.tsx` 传给子组件的 inline `{...}`/`() => {}`（每 render 新引用 → memo 白做）。
- React DevTools Profiler 验收：流式一条消息时,**已完成的历史消息组重渲染次数 = 0**（只活跃组重渲染）。

### 3.4 代码分割（热点 ③）

`next/dynamic` 懒加载非首屏重模块（首屏只留聊天主干）：

| 模块 | 现状 | 改 |
|---|---|---|
| 产物画廊（#3,含 react-virtual + 网格） | 全量上首屏 | `dynamic(() => import('.../gallery'), { ssr:false, loading: skeleton })` |
| 运行轨迹抽屉（#2） | 全量 | `dynamic`,开抽屉才加载 |
| artifact 详情（iframe/markdown/代码高亮） | 全量 | `dynamic`,选中产物才加载 |
| 图表/markdown 渲染重库 | 全量 | `dynamic`,按需 |

- `loading:` 用骨架屏（`ui-ux-pro-max` `progressive-loading`,现有 `skeleton.tsx` 复用）,不空转。
- **首屏预算**：分割后首屏 bundle 显著降（验收用 `next build` 输出 + Lighthouse TTI/TBT）。

### 3.5 60fps 动效 + 防 CLS（全局基线）

- **transform/opacity-only**（`transform-performance`）：所有动效（spec#1 曲线 + spec#6 hover lift/脉冲）只动 transform/opacity,**禁动 width/height/top/left**。grep `transition-all` 收窄成 `transition-[transform,opacity]`/`transition-colors`（spec#1 Step4 已起头,本 spec 校验闭环）。
- **防 CLS**（`content-jumping`/`image-dimension`）：流式内容、图、骨架屏预留尺寸（`aspect-ratio`/`min-height`）,不让内容到达时跳动。Lighthouse CLS < 0.1。
- **input latency <100ms**（`input-latency`/`tap-feedback-speed`）：点击/滚动/输入 100ms 内有反馈(§3.2 的 defer 正是为保此);press 反馈 spec#6 scale。
- **骨架屏 >300ms**（`loading-states`）：任何 >300ms 加载（画廊、报告、图）用 skeleton/shimmer（现有 `shimmer.tsx`/`skeleton.tsx`）,不空转 spinner。
- 全部 reduced-motion 降级（复用 `globals.css:357`；motion 库 spring 用 `useReducedMotion()`）。

### 3.6 滚动与流式贴底（丝滑度细节）
- 当前无 `scrollIntoView`（grep 证实）。设计 stick-to-bottom 时**避免每 token `scrollTo`**（高频 scroll = jank）——用"接近底部才自动贴底 + rAF 合并滚动",或 react-virtual 的贴底能力。
- 用户上滚看历史时**不强拽回底部**（`back-behavior`/`state-preservation`）：只有已在底部时新消息才自动跟。

### 3.7 视觉特效的高效实现（硬闸门，**所有视觉 spec #1/#3/#5/#6/#8 必过**）

> **这是用户 2026-06-25 的核心顾虑（已澄清）**："新增很多视觉效果和特效,应尽量简单有效——**不是视觉降级,而是写代码时别写那种特别繁琐、造成额外不必要开销的代码**。" 即:**视觉效果该有就有,但实现它的代码要省**——不写冗余 re-render、不每帧重算、该缓存就缓存、资源用完释放、用便宜手段达成同一效果。**这一节不限制"做什么视觉",只规范"怎么高效实现"。**
>
> ⚠️ **原则边界(必须守):性能问题 99% 出在"实现方式"而非"视觉本身"。** 同一个玻璃/阴影/动效,笨重写法卡、高效写法丝滑。**默认保视觉,优化实现**;只有当"视觉上本来就看不出区别"时(如十几个完全相同的同质层可合并为一层背景)才合并——**那也不是降级,是消除肉眼无差的冗余**。

#### 3.7.1 开销从哪来：同一效果的"贵写法 vs 省写法"
特效贵不在"它存在",而在**实现它的代码强制浏览器每帧/每次重复做昂贵的事**。同样的视觉,换个写法就不贵：

| 视觉效果 | ❌ 贵写法（繁琐/冗余开销） | ✅ 省写法（同视觉，低开销） |
|---|---|---|
| 卡片淡入/hover | 动画 `width/height/top/left`/`box-shadow 模糊半径` → 每帧 layout/paint 重排 | 只动 `transform`/`opacity`（合成层搞定,不碰 layout） |
| 玻璃/模糊层 | 给**每个列表项/每条消息**各挂一个 `backdrop-blur`（几十个同质合成层各自离屏渲染） | 同样观感,但**让模糊层是稳定的、不随列表重渲染重建**;同质背景可由**一层容器**承载而非每项一个（视觉一致,合成层从几十→一） |
| 常驻动画(pulse/shimmer) | 同屏铺很多 + 每个独立 `setInterval`/JS 驱动 | CSS 动画(合成器驱动,不占主线程);只给**真正在进行**的那个元素加,完成即停 |
| 重渲染驱动的特效 | 特效组件因父级每次 render 跟着重渲染（props 不稳定）→ 反复重建 DOM/重算样式 | `React.memo` + 稳定 props,**特效只在自身状态变时更新**,不被无关 render 带着重算 |
| `will-change` | 常驻挂在很多元素 → 每个长期占合成层+显存不释放 | **只在交互瞬间加,结束移除**（或靠 transform 自动提层,根本不写） |
| 重复计算/重建 | 每帧/每 render 重算样式值、重建对象/回调、重新 measure | `useMemo`/`useCallback` 缓存;measure 结果复用;rAF 合并高频更新 |

> 一句话:**同一个视觉,"省写法"和"贵写法"差的是浏览器每帧的重复劳动量,不是观感。** 这一节要的就是全程用省写法。

#### 3.7.2 高效实现的硬规则（违反不准合，进 §五红线）

**A. 动画只用合成器属性（同视觉，不碰 layout/paint）**
- 动画**只动 `transform`/`opacity`**;**禁** `width/height/top/left/box-shadow/filter` 进 `transition`/`animation`（每帧 layout/paint 重排是纯浪费——同样的位移/缩放/淡入用 transform 看起来一模一样却不重排）。
- hover lift 用 spec#6 阴影**阶梯切换**（切 class,不在 transition 里 animate 模糊半径）——视觉是阴影变化,实现不付逐帧模糊光栅化的代价。

**B. 特效组件不被无关重渲染带着重算**
- 特效/动画组件 `React.memo` + 稳定 props（`useCallback`/`useMemo`），**只在自身状态变时更新**——别因父级每次 render 就重建。这是"繁琐开销"最常见来源（承 §3.3 重渲染边界）。
- 样式值/对象/回调用 `useMemo`/`useCallback` 缓存,不每 render 重建。

**C. 同质重复的特效层用便宜结构承载（消除冗余，非删视觉）**
- 大量**视觉相同**的效果层（如长列表每项一个同样的玻璃/模糊背景）：**视觉保持不变**,但优先**用一层稳定容器/单一背景**承载,而非每项各挂一个昂贵层——肉眼无差,合成层数从 O(n)→O(1)。
- 具体到现状 `message.tsx` 的 per-message `backdrop-blur`：**玻璃质感保留**（那是设计），但**改成"不随每条消息重渲染而重建、且模糊范围受限"的省写法**（见 Step 5.5）——视觉不变,开销大降。**不是删玻璃。**

**D. 资源用完释放，高频合并**
- `will-change` 只交互瞬间加、结束移除(或不用)；监听器/计时器/`URL.createObjectURL` 等用完释放，不泄漏。
- 高频事件(scroll/resize/stream)用 rAF/节流合并,不每次都跑昂贵回调(承 §3.2/§3.6)。

**E. 便宜手段优先（达到同效果时）**
- 同一观感,优先开销低的实现:能用 CSS 动画(合成器)就不用 JS 逐帧;能复用缓存就不重算;能一层承载就不 N 层。**这不是"少做视觉",是"同样视觉用更聪明的代码"。**
- `motion-meaning`:动画表达因果/层级/状态时才加——这是**避免写无意义的冗余动画代码**,不是砍掉有意义的视觉。

#### 3.7.3 验收方法（可测，看"有没有多余开销"，不是"有没有视觉"）
- **重渲染**：React DevTools Profiler——特效组件不因无关 render 重渲染（验 B）。
- **合成层/重绘**：DevTools → Rendering → Layers / "Paint flashing"——动画时只动 transform 的元素**不触发 paint**（验 A）；同质层没有 O(n) 个冗余合成层（验 C）。
- **GPU 显存**：Performance Monitor——长会话/多图不持续涨（验 C/D：层稳定复用、资源释放）。
- **帧率**：Performance 面板——滚动/动画无连续长帧（<16ms 60fps,`main-thread-budget`）。
- **基准**：**视觉效果保持不变的前提下**，上述指标改善（before/after 对比，§Step6 基线）——这是"实现变省"的证明，不是"视觉变少"。

---

## 四、实施步骤

> 全在 `frontend/` 渲染层,**零碰 `hooks.ts` 的 SSE handler / 合并逻辑**。按收益/风险排序。

### Step 1（最高收益、最低风险）：流式节流 + groupMessages memo（§3.2）
- `message-list.tsx`：`useDeferredValue(thread.messages)` + `useMemo(groupMessages)`。**不动 hooks.ts。**
- 验收：React DevTools Profiler 录一段流式,确认 `groupMessages` 不再每 token 重算 + 主线程帧 <16ms。
- **先上这步量一版**——可能单这步就大幅缓解卡顿。

### Step 2：重渲染边界 memo + 稳定 props（§3.3）
- 补全 `SubtaskCard`/`MessageGroup`/`MessageListItem` 的 `React.memo`;查 `message-list.tsx` 传子组件的 inline 回调/对象,`useCallback`/`useMemo` 稳定化。
- 验收：Profiler 确认流式时历史消息组重渲染 = 0。

### Step 3：代码分割（§3.4）
- 画廊/轨迹/artifact 详情/重渲染库 `next/dynamic` + 骨架 loading。
- 验收：`next build` 输出首屏 bundle 体积下降 + Lighthouse TTI/TBT 改善。

### Step 4（较高风险，可独立/可降级）：消息列表虚拟化（§3.1）
- `@tanstack/react-virtual` 对分组列表 windowing,动态高度 `measureElement`,兼容流式追加 + 贴底。
- 验收：50+ 组时滚动 DOM 节点恒定（devtools Elements 数节点不随滚动线性增长）。
- **若动态高度虚拟化不稳**：保留 Step 1-3 收益,本步标记已知限制 + 后续迭代（不静默假装）。

### Step 5：60fps + CLS + 贴底（§3.5 + §3.6）
- grep 清 `transition-all`;预留尺寸防 CLS;骨架屏 >300ms;stick-to-bottom 用 rAF 合并不每 token scroll。
- 验收：Lighthouse CLS<0.1 + 滚动/流式 Performance 面板无长帧。

### Step 5.5（特效高效实现，"同视觉省开销"）：per-message blur 改省写法 + 特效组件 memo（§3.7）
- **`message.tsx` 的 per-message `backdrop-blur` 玻璃质感保留**（设计要的视觉,不删），但改省写法:① 让该消息组件 `React.memo` + 稳定 props,**已完成的历史消息不再因新消息流入而重渲染**（消除"每条消息反复重建模糊层"的冗余开销，承 §3.3）;② 模糊层不随每次 render 重建（稳定挂载）;③ 若同屏几十条**视觉相同**的玻璃可由一层稳定背景承载,在不改观感前提下合并（消除 O(n) 同质冗余层，**肉眼无差**）。**视觉不变,合成层/重绘从随消息数增长 → 稳定。**
- 审计其余 `backdrop-blur`/特效引用：凡因父级 render 跟着重渲染的（props 不稳定），加 memo/稳定 props——**保视觉，去无谓重算**。
- `will-change` 若有常驻的，改为交互瞬间加/结束移除。
- 验收：**视觉效果不变**前提下，DevTools Profiler 特效组件不被无关 render 带着重渲染 + Layers 面板长会话合成层稳定 + GPU 显存不随消息持续涨。
- **这步可独立于 Step1-4 先做**（纯高效化，零依赖，不动视觉）。

### Step 6：reduced-motion + 量化回归
- reduced-motion 全降级校验。
- 留一份**性能基线记录**（before/after：流式时帧率、长会话 DOM 节点数、首屏 bundle、Lighthouse 分、**长会话合成层数 + GPU 显存**）——**视觉不变前提下指标改善 = 实现变省的证明**，作回归锚（CLAUDE.md「兜底/优化要可观测」）。

---

## 五、验收标准（每条可测，不靠"感觉"）

### 流式丝滑（核心诉求）
- [ ] 流式 token 暴雨时主线程帧 <16ms（Performance 面板无连续长帧）；交互（点击/滚动）不被流式阻塞（`input-latency`<100ms）。
- [ ] `groupMessages` 经 `useMemo`,不再每 render/每 token 重算（Profiler 验）。
- [ ] `useDeferredValue` 接在消费侧;**`hooks.ts` SSE handler / 合并逻辑零改动**（diff 证实）。
- [ ] 流式时已完成历史消息组重渲染次数 = 0（Profiler 验）。

### 长会话 / 大列表
- [ ] 50+ 消息组滚动时 DOM 节点恒定（不随消息数线性增长）——或明确标注已知限制（Step 4 降级时）。
- [ ] 长会话滚动 60fps,无明显掉帧。

### 首屏 / 加载
- [ ] 画廊/轨迹/artifact 详情等 `next/dynamic` 懒加载;首屏 bundle 体积较改前下降（`next build` 输出对比）。
- [ ] 非首屏模块加载用骨架屏,不空转 spinner（>300ms 加载）。
- [ ] Lighthouse：TTI/TBT 改善,CLS < 0.1。

### 动效 / 渲染
- [ ] 所有动效 transform/opacity-only,无 width/height/top/left 动画（grep `transition-all` 已收窄）。
- [ ] 流式内容/图/骨架预留尺寸,无 CLS 跳动。
- [ ] stick-to-bottom 不每 token scroll（rAF 合并）;用户上滚看历史不被强拽回底。
- [ ] reduced-motion 全降级（含 motion spring）。

### 视觉特效高效实现（§3.7 硬闸门，**所有视觉 spec #1/#3/#5/#6/#8 共用红线；保视觉、省开销**）
- [ ] 动画**只动 transform/opacity**——无 width/height/top/left/box-shadow/filter 进 transition（同位移/缩放/淡入用 transform，视觉一样不重排）。
- [ ] 特效/动画组件 `React.memo` + 稳定 props，**不因无关父级 render 重渲染重建**（Profiler 验）。
- [ ] **`message.tsx` per-message 玻璃视觉保留但改省写法**：历史消息不被新消息流入带着重渲染；模糊层不每 render 重建；同质层在**肉眼无差**前提下可合并（Layers 验长会话合成层稳定、不随消息数 O(n) 增长）。
- [ ] box-shadow 用 spec#6 五档；hover lift 切 class 不在 transition 里 animate 模糊半径。
- [ ] `will-change` 不常驻（仅交互瞬间）；监听器/计时器/objectURL 用完释放。
- [ ] **视觉效果保持不变前提下**，GPU 显存长会话/多图不持续涨、paint flashing 动画时不大面积重绘（证明"实现变省"非"视觉变少"）。

### 工程纪律（红线）
- [ ] **不改 `useStream`/`mergeMessages`/`dedupeMessagesByIdentity`/optimistic/summarization**——节流/虚拟化全在消费侧渲染层（diff 自证）。
- [ ] 不改 `groupMessages` 分组算法（只套 memo）。
- [ ] 不引新状态/渲染框架（React 19 原语 + 已装 react-virtual）。
- [ ] `pnpm check` 通过;零后端改动。
- [ ] 留 before/after 性能基线记录作回归锚。

---

## 六、风险与回退

| 风险 | 缓解 |
|---|---|
| 误改流式核心引回归（dedupe/optimistic 踩坑复发） | **铁律:节流只加消费侧接缝（§1.5）,hooks.ts 一行不动**;diff review 自证;Step 1 先做且最易隔离 |
| `useDeferredValue` 让流式"看起来延迟" | defer 只降重渲染优先级,不延迟显示;高负载才合并中间帧（正是 60fps 要的）;实测调 |
| 消息流动态高度虚拟化不稳（展开/流式追加/贴底） | §3.1/Step4 列为较高风险可降级项:Step1-3 收益不依赖它;不稳则标已知限制不静默 |
| memo 因 props 不稳定失效（白加） | Step2 强制 useCallback/useMemo 稳定 props;Profiler 验重渲染真降 |
| 代码分割致首次打开某功能闪骨架 | 骨架屏体验可接受;关键路径（聊天主干）不分割,只分割画廊/轨迹等次级 |
| 过度工程（分治/复杂节流上太多） | §3.2 先 defer+memo 量一版,够了不加;按实测决定是否要分治活跃/历史 |

**回退**：纯渲染层 React 原语 + 组件包装,`git revert` 即可;不动数据/不动后端/不动流式核心,回退零风险。每步可独立灰度（Step1 单独就有大收益）。

---

## 七、给实施 agent 的交接

- **改动文件**：`message-list.tsx`（defer+memo+虚拟化,主战场）、热组件补 memo（subtask-card/message-group/message-list-item）、`app/`+组件入口 `next/dynamic`、`globals.css`/className 清 `transition-all`。**`core/threads/hooks.ts` 只读消费,不改其 SSE/合并。**
- **绝不碰**（memory/CLAUDE.md 铁律）：`useStream` / `mergeMessages` / `dedupeMessagesByIdentity` / optimistic / summarization / `groupMessages` 分组算法。
- **复用**：#3 的 `@tanstack/react-virtual`（同引擎,消息流与图墙各自虚拟化但同库）；现有 `skeleton.tsx`/`shimmer.tsx`/`number-ticker`;spec#1 transform-only + reduced-motion。
- **顺序（按收益/风险）**：Step 1（defer+memo,最高收益最低风险,先做量一版）→ Step 2（memo 边界）→ Step 3（代码分割）→ Step 5（60fps/CLS/贴底）→ Step 4（虚拟化,较高风险可降级）→ Step 6（reduced-motion+基线）。
- **量化优先**：每步**先量后改、改后再量**（Performance 面板 + React Profiler + Lighthouse + DOM 节点数）。**不靠"感觉变快了"**——留 before/after 基线。
- **本会话已定**：① 新建本 spec#7（用户拍板）② **绝不改流式核心**（用户拍板,节流只在消费侧渲染层）。
- **与其他 spec**：#3 画廊图片虚拟化/缩略图归 #3,本 spec 做消息流卡片虚拟化 + 全局性能基线;spec#1/#6 的动效本 spec 只校验其 transform-only/60fps 闭环,不重定义。

---

*依据：母方案 §1.1/§9 + 用户 2026-06-25"响应速度/刷新丝滑/防卡又慢"诉求 + 现状三热点实证（message-list.tsx:75 未虚拟化 / hooks.ts setMessages 零节流 + groupMessages render 内重算 / next/dynamic 零使用）+ `ui-ux-pro-max`（virtualize-lists/main-thread-budget/transform-performance/debounce-throttle/bundle-splitting/content-jumping/input-latency）。未写代码。*
