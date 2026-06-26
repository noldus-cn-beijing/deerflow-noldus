# 前端 / 人机体验大升级方案（EthoInsight）

> 2026-06-24 · 设计调研稿（产品 + 工程双视角）
> 调研基准：当前 `packages/agent/frontend`（Next.js 16 / React 19 / Tailwind 4 / shadcn / `@langchain/langgraph-sdk` `useStream`）
> 设计语言基准：日式简洁美学 + 真正的「设计感」（克制、留白、动效曲线讲究），**不是**千篇一律的 BI 仪表盘味
> 工具基准：`ui-ux-pro-max` skill（Data-Dense Dashboard 风格、large-dataset 图表准则、reduced-motion / 虚拟化 / faceting）

---

## 决策已锁定（2026-06-24 用户拍板）

> 这一段是后续所有 spec 的前提,改动须重新对齐用户。

| # | 决策 | 结论 | 影响 |
|---|---|---|---|
| 硬伤A | 几百张图怎么展示 | **聊天流里只展示少量"代表图"(aggregate / 精选),全部图收进一个"单独打开的产物画廊"**——独立路由页 或 全屏弹出层(模态/抽屉),走图墙最佳实践(虚拟化+分面+小倍数对比) | §4.2 重写为"代表图 inline + 画廊另开" |
| 决策1 | 画廊后端契约 | **路 A**:后端 `present_file` 附带 chart 元数据(`chart_id/output_mode/paradigm/metric/subject`),`artifacts` 从 `string[]` 升级成 `ArtifactMeta[]` | 需后端协同 + grep 所有消费方 + 向后兼容 |
| 决策2 | 进度轨位置 | **同意做**(横/纵在 §3.1 给出推荐,实施期定) | §3.1 |
| 决策3 | 配色基调 | **不变**——保留 Forest Green `#1A4840` 品牌主色 + 暖白纸底,不往"全中性"走。日式克制体现在**留白/降噪/去装饰/动效**,不动色相 | §2.1 收窄:不改配色,只改"信息密度+装饰+动效" |
| 决策4 | dark mode | **先不做**,留到之后 | §2.3 降级为 P2/后续;Phase 0 token 阶段**不含** dark |
| 决策5 | 运行轨迹/重放面板 | **现在就做**(用户已确认要)。**经后端核查拆成两半**:`live trace`(实时全保真,今天就能做)+ `replay`(回看历史 run,lead 层今天能做、subagent 内部步骤需 ~30 行后端持久化) | §5.2 重写 |

**决策 5 的"重放面板"是什么(回答用户反问)**:指一个**"运行轨迹"侧抽屉**,把本次分析里 agent 的完整行为(每次派遣哪个子代理、每次工具调用、每个质检 gate 判定绿/黄/红、每次反问决策)按**时间线**铺开,每步可展开看 input/output。"重放"= 选一次历史分析(thread 里的某次 run),让这条时间线**按当时的真实时序重新播一遍**(后端每条消息都存了 `created_at` 墙钟时间,所以能做"带时序的回放",不只是静态列表)。价值:①研究员看清 agent 每步在干嘛(可审计)②训练飞轮反馈打得更准③出错时定位"卡在哪一步"。

### Phase 0 实施 spec（2026-06-24 已出，按顺序）

| # | spec | 一句话 | 后端依赖 |
|---|---|---|---|
| 1 | [design-tokens-motion](../superpowers/specs/2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md) | `--ease-*`/`--dur-*`/语义色 token + 换掉全站 ease-in-out→非对称减速曲线 + 退役装饰动画 | 无 |
| 2 | [run-trace-live](../superpowers/specs/2026-06-24-frontend-phase0-2-run-trace-live-spec.md) | 运行轨迹侧抽屉（实时全保真,`useRunTrace` 纯派生） | 无 |
| 3 | [artifact-gallery](../superpowers/specs/2026-06-24-frontend-phase0-3-artifact-gallery-spec.md) | 代表图 inline + 独立画廊（分面/虚拟化/对比/失败显式） | `ArtifactMeta` 契约（路 A） |
| 4 | [analysis-rail](../superpowers/specs/2026-06-24-frontend-phase0-4-analysis-rail-spec.md) | 7 阶段进度轨（前端推导,复用 spec#2 派生） | 无 |
| 5 | [decision-card](../superpowers/specs/2026-06-24-frontend-phase0-5-decision-card-spec.md) | 反问→显眼决策卡（accent bar + 依据 + 键盘 + 联动） | 无 |
| 6 | [design-language-craft-semiotics](../superpowers/specs/2026-06-24-frontend-phase0-6-design-language-craft-semiotics-spec.md) | 细节工艺（重叠 card elevation 阶梯 + radius 纪律 + 不抖布局）+ 尺寸/比例/层级（一屏一主角）+ 符号系统（Peirce+Morris 8 硬规则 + label-off 门）+ Mœbius 插画**只借语法**（线型/平涂/负空间,硬拒青橙 palette,clay 暖中性 accent） | 无 |
| 7 | [runtime-performance](../superpowers/specs/2026-06-24-frontend-phase0-7-runtime-performance-spec.md) | 运行时丝滑度（防"卡又慢"）：消息列虚拟化 + 流式 `useDeferredValue` 节流 + `groupMessages` memo + 重渲染边界 + `next/dynamic` 代码分割 + 60fps transform-only + 防 CLS。**+§3.7 视觉特效的高效实现（保视觉省开销:动画只 transform/opacity、特效组件 memo、同质层合并、不常驻 will-change；所有视觉 spec 必过）**。绝不改流式核心,只在消费侧渲染层 | 无 |
| 8 | [stacked-upload-attachments](../superpowers/specs/2026-06-24-frontend-phase0-8-stacked-upload-attachments-spec.md) | 多文件上传堆叠（Seedance 式）：超阈值(≈5)后续文件堆叠成一叠 + "+N" 计数,hover(桌面)/点击(触屏)扇开可删。**自建组件替 generated 用法,复用 `usePromptInputAttachments` store 不改 generated** | 无 |

**实施顺序**：#1（token,零风险打头,是后面所有组件的动效/语义色 SSOT）→ #3 Step 3（inline 代表图,立即缓解图墙痛）→ #2（live trace）→ #4（进度轨,复用 #2 派生）→ #5（决策卡,联动依赖 #4 但本体可独立）。#3 的后端 `ArtifactMeta` 契约与前端并行。**#6 设计语言是横切纪律**：其 token（shadow 阶梯 + clay accent）紧跟 #1 落（同属零风险样式层）；其工艺/尺寸/符号/插画**验收项融进 #2-#5 各组件交付**（不单独大重构），空状态插画落 `ui/empty.tsx` 可独立做。

> **⏱️ 实施进度（2026-06-25，以 `git show HEAD:` 在 dev commit `4d847228` 坐实，勿凭本表"按顺序"重做已合项）**：
> - ✅ **#1 已合并 dev**（PR#201，commit `4d847228`）——`--ease-brand-*`/`--dur-*`/`@utility duration-*`/`--status-*` 两层/`--stage-*` 两层全在 `globals.css`，写法已核对正确。**不要再把 #1 当待办起头。** 按 `code≠bug` 原则，#1 仍需 `make dev` 真机实测（duration 是否死类、status 是否自指、ease 精确值、装饰退役、tabular-nums）——验证清单见 [2026-06-25-frontend-phase0-dev-code-fix-spec.md](../superpowers/specs/2026-06-25-frontend-phase0-dev-code-fix-spec.md) §二 A 节。
> - 🚧 **#2 / #3 / #6 实施中**（其它 agent 并行；**dev HEAD 上均未合**）。注意 #3 后端 `_build_artifact_meta`/`thumb_path` 只在主仓工作树未提交草稿里（`present_file_tool.py` ` M`），HEAD 未合；前端 `ArtifactMeta` 契约**完全没落**（`types.ts:8` 仍 `artifacts: string[]`），且勿把 `normalizeArtifactImageSrc`（report.md 图片 src 规范化，无关契约）误当"已部分实现"。待合入后的核对清单见 fix-spec §三/四/五。
> - ⏳ **#4 / #5 / #7 / #8 未起**。
> - **核实铁律**：判"某代码在不在 dev"用 `git show HEAD:<file>`，不要 `grep <工作树文件>`——工作树含别的 agent 未提交草稿会误判（本次 #3 后端误判即由此而来）。

---

## 0. 一句话结论

**不重写、不换栈。** 现有前端已经站在一条**正确的生成式 UI 地基**上（LangGraph `useStream` 事件流 + 类型化消息分组 + artifacts 侧栏 + 可中断 HITL）。真正的问题不在技术栈，而在 **「信息架构 + 状态可见性 + 动效/视觉的设计感」三层都还停在『把后端事件被动渲染成卡片』的阶段**。

本方案分三个层次推进：

1. **P0 止血**（修可用性硬伤 + 用户"现在就做"的）：几百张图的「图墙」(→代表图+独立画廊)、workflow 无导向(→进度轨)、审批/中断不显眼(→决策卡)、运行轨迹 live trace。(dark mode 已按决策推迟。)
2. **P1 体验跃迁**（生成式 UI 的"按需生成/销毁" + 可观测性）：把 lead 的"思考-规划-工具-审批"做成一条**可读、可重放、可干预**的工作流主轴；把 artifacts 从"文件列表"升级成**领域感知的产物画廊**。
3. **P2 形态扩展**（desktop / Electron 的共享状态前置铺垫）：把跨端共享状态、命令系统、能力边界沙盒在 web 期就抽象好，desktop 只是壳。

贯穿三层的是一条**质量基线**：日式克制美学 + 动效曲线讲究（§7 专门讲）。这条基线不是单独阶段，而是每个组件交付的验收项。

---

## 1. 现状盘点（调研所得，带证据）

### 1.1 技术栈已经"够用且现代"

| 能力 | 现状 | 证据 |
|---|---|---|
| 流式事件引擎 | `useStream` + `onCustomEvent` / `onUpdateEvent` / `onLangChainEvent` 全通 | `core/threads/hooks.ts:367-539` |
| 类型化消息分组 | `groupMessages` 已把消息路由成 `assistant:subagent` / `:present-files` / `:clarification` / `:processing` 五类 | `components/workspace/messages/message-list.tsx:71-394` |
| 子代理时间线 | SubtaskCard 已渲染 subagent reasoning/tool-call 时间线 + 状态徽章 | `subtask-card.tsx` |
| 工具调用语义化 | `stage-broadcast.ts` 已把 `bash`/`task` 翻成中文状态条；ToolCall 已对 `inspect_uploaded_file`/`prep_metric_plan`/`set_experiment_paradigm` 做领域渲染 | `message-group.tsx:232-526`、`stage-broadcast.ts` |
| HITL 中断 | `ask_clarification` 已渲染成带 options 按钮的可点选项 | `message-list.tsx:86-134`、`clarification-options.tsx` |
| artifacts 侧栏 | 可 resize 的 60/40 分栏 + 文件详情（代码/markdown/html/图片预览 + iframe 沙盒） | `chats/chat-box.tsx:23-178`、`artifact-file-detail.tsx` |
| 反馈飞轮 | 三按钮 verdict + run_id 绑定（history + stream 双源） | `message-list-item.tsx:64-75`、`hooks.ts:836-842` |
| 设计 token 地基 | oklch 色彩空间、Forest Green 品牌（#1A4840）、暖白纸 bg、glass-card、soft shadow、radius 阶梯 | `styles/globals.css:143-262` |

**结论：地基扎实。** 不要碰 `useStream` / `groupMessages` / `merge*` 这套（`hooks.ts` 里大量精心处理过的 dedupe / optimistic / summarization 边界，是踩坑踩出来的，重写必复发）。升级动作**叠加在分组之上**，不动流式合并核心。

### 1.2 三个真正的硬伤

#### 硬伤 A：几百张图 = 不可用的「图墙」

后端 `run_chart_plan` 能并行产出 **per_subject** 图（28 subject × N metric 轻松上百张），每张图通过 `present_file` 进 `thread.values.artifacts`——但前端拿到的是**一个扁平 `string[]`**，`ArtifactFileList` 直接 `grid-cols-2` 全量渲染（`artifact-file-list.tsx:94-135`）。

- 后端**明明有结构化元数据**却没传到前端：`plan_charts.json` 里每张图有 `chart_id` / `output_mode`（`per_subject`|`aggregate`）/ `script` / paradigm（后端调研 §3-4 坐实）。
- 前端把这一切**拍平成路径字符串**，丢光了"哪张是 aggregate、哪张属于 subject 7 的 open_arm"这种研究员唯一关心的维度。
- 后果：研究员要在 100+ 张缩略图里**肉眼大海捞针**找"control 组的 box plot"，且首屏全量 `<img>` 直接违反 `ui-ux-pro-max` 的 `virtualize-lists`（50+ 必虚拟化）和 `large-dataset`（1000+ 点要聚合/抽样/drill-down）。

> 这是**第一优先级**。研究员是这个产品的全部受众，"出图后找不到图"直接抵消整条分析流水线的价值。

#### 硬伤 B：workflow 没有导向性，用户不知道"现在到哪了 / 接下来做什么"

EthoInsight 的分析有清晰的**领域工作流**：`上传 → 范式识别(Gate1) → 列语义对齐(HITL) → 指标计算 → 数据质检(Gate2) → 统计解读 → 报告`。但前端**没有任何地方呈现这条主轴**：

- 用户看到的是一条**线性消息流**，subagent 卡片一个个冒出来，但"整体进度"不可见。
- `write_todos` 工具被渲染成一行 CoT step（`message-group.tsx:450-457`），todo 状态没有做成持久的进度骨架。
- 范式识别、列对齐这些**关键决策点**淹没在消息流里，研究员（非程序员）不知道"我刚才那个反问确认了什么、它锁定了哪条范式"。

`ui-ux-pro-max` 的 `multi-step-progress`（多步流程要有 step indicator）、`nav-state-active`（当前位置必须高亮）在这里完全缺位。

#### 硬伤 C：审批/中断不够"显眼" + 无 dark mode

- `ask_clarification` 目前只是消息流里一段 markdown + 几个按钮（`clarification-options.tsx`），**没有视觉上的"流程在此暂停、等你决策"的强信号**。研究员很容易划过去没注意到 agent 在等他。用户视角原文："可中断提示""审批链"应该是一等公民。
- `:root` 只覆盖了 light 主题（`globals.css:193+`），**dark mode 没做**。`ui-ux-pro-max` 把 dark/light parity 列为 HIGH（`dark-mode-pairing`、`color-dark-mode`）；科研用户长时间盯屏，暗色是刚需不是加分。

### 1.3 用户提出的产品观（六维迁移）与现状的差距

用户给的六维（交互/状态/输出/任务/可观测/安全）本质是 **"传统 UI → 生成式 Agent UI"** 的范式迁移。逐条对现状打分：

| 维度 | 用户的愿景 | 现状 | 差距 |
|---|---|---|---|
| 1 交互模式 | 事件流驱动、界面按需生成/销毁 | ✅ 事件流已有；❌ 但"界面"只有固定几种卡片，没有"按需生成的领域组件" | 需要**生成式组件注册表**（见 §4.1） |
| 2 状态管理 | 前端 ↔ Agent 共享状态、双向同步、冲突调和 | ⚠️ 单向：后端 `thread.values` → 前端渲染；前端只能发消息 | 需要**前端可写共享状态**（见 §6.2） |
| 3 输出方式 | 动态组件树、思考链可视化、可中断 | ✅ 思考链有；⚠️ 可中断有但弱；❌ 动态组件树没有 | §4 + §5 |
| 4 任务模式 | AI 自主规划子任务、人机中断/回退 | ✅ 子任务有；❌ 规划不可见、不可回退 | §3 workflow 主轴 + §5 回退 |
| 5 可观测性 | 记录思考/规划/工具/审批链，可审计/重放 | ⚠️ 单次会话可见；❌ 无重放、无审批链汇总视图 | §5 可观测面板 |
| 6 安全模式 | 基于 Agent 能力边界的沙盒渲染 + 指令校验 | ✅ html artifact 已 iframe sandbox；⚠️ 生成式组件的能力边界没定义 | §6.3 |

**关键判断**：维度 1/3/4 是 **P0-P1 能在 web 端做掉的**；维度 2/6（共享状态、能力边界）是 **desktop 前置铺垫**，web 期只需把抽象立对，不必全实现。

---

## 2. 设计语言：日式简洁 + 真正的设计感

> 用户明确要求：**有设计感、日式简洁美学、不要老土千篇一律的风格、注意细节（举例动效曲线要渐变减速不要匀速）**。这一节定义"好"的标准，后面每个组件都按这条验收。

### 2.1 视觉原则（间 / 余白 / 克制）

`ui-ux-pro-max` 给 EthoInsight 的推荐是 **Data-Dense Dashboard**（BI 味，minimal padding、maximum data visibility）。**我们要有意识地往回拉**——研究员要的是"少而清晰"，不是"满屏 KPI"。日式美学的落地准则：

1. **留白即信息（間）**：用空间分组，不用边框/分隔线堆叠。当前很多卡片 `rounded-lg border` 套 `border` 套 `border`（如 `message-group.tsx` 的 ChainOfThought 嵌套），视觉噪音重。原则：**一个层级只用一种区隔手段**（要么留白、要么底色、要么一条细线，不三者叠加）。
2. **配色基调不变（决策3）**：**保留品牌 Forest Green `#1A4840` 主色 + 暖白纸底,色相不动**。日式克制**不是改配色**,而是改"用色的分布"——大面积负空间/中性暖灰,主色只点在"当前焦点/主 CTA"。不照搬 `ui-ux-pro-max` 默认那套蓝+琥珀高饱和 BI 配色,但也不把品牌绿降级成中性。状态语义色(§2.2)与品牌绿同色系协调,不引入冲突色相。
3. **一屏一主角**（`primary-action`）：每个状态只有一个视觉主角（当前 subagent / 待决策的反问 / 选中的图）。其余降为背景层。
4. **字体克制**：保留 OPPO Sans 作正文（中文友好）；**数字/指标/代码用等宽**（`number-tabular`：表格、效应量、p 值、计时器用 tabular figures 防跳动）。标题不堆字重，靠**字号 + 留白**拉层级，不靠粗体轰炸（`weight-hierarchy` 但克制）。
5. **去装饰**：移除 aurora / shine / 过度 pulse 这类"科技感"装饰动效（`globals.css:121-137` 那些）——它们正是"老土千篇一律 AI 产品"的味道来源。动效只服务于**因果关系的表达**（`motion-meaning`）。

### 2.2 一套语义化设计 token（扩展现有，不另起炉灶）

现有 `@theme inline` 已有 `--color-brand` / `--color-elevated` / radius 阶梯。**补齐三组缺失的语义 token**，全部走 oklch（已是现状）：

```css
/* 状态语义色（質感版，非荧光）—— 给 quality-warning / gate / 审批用 */
--color-status-info:    oklch(0.62 0.10 230);   /* 沉静蓝 */
--color-status-success: oklch(0.60 0.11 155);   /* 苔绿，与品牌同family */
--color-status-warning: oklch(0.72 0.12  75);   /* 琥珀，降饱和 */
--color-status-danger:  oklch(0.58 0.16  25);   /* 朱，非纯红 */

/* 工作流阶段色（7 阶段各一个低饱和 hue，用于进度主轴）*/
--color-stage-upload / -paradigm / -align / -compute / -qc / -interpret / -report;

/* 动效曲线 token（§7 核心）*/
--ease-out-quint:  cubic-bezier(0.22, 1, 0.36, 1);     /* 入场：快起→缓停 */
--ease-in-out-quint: cubic-bezier(0.83, 0, 0.17, 1);   /* 切换 */
--ease-spring: linear(...);  /* 或 motion 库的 spring 配置 */
--dur-fast: 160ms; --dur-base: 240ms; --dur-slow: 360ms;
```

> **关键**：现在 `globals.css` 的 keyframes 全用 `ease-in-out` / `ease-out`（`:83/94/100/106`），这是浏览器默认的对称三次曲线——正是用户批评的"接近匀速"。换成上面的 `--ease-out-quint`，所有入场/退场立刻有"高级感的减速尾巴"。这是**性价比最高的一个改动**（改 token，全站受益）。

### 2.3 dark mode（决策4=先不做，留到后续）

按 `color-dark-mode` 的正确做法是**降饱和 + 重新定义明度阶梯**(不是反相),dark/light 一起设计一起测。**但用户已拍板先不做**——Phase 0 的 token 升级**不含** dark,`--ease-*` 曲线和状态语义色照做(它们与主题无关、light 单主题下即生效)。dark mode 列入 P2/后续,届时单独补 `:root.dark` 一组 token + 双主题验证。

> 注意:即便现在不做 dark,新写组件**仍用语义 token**(`bg-card`/`text-foreground`/`--color-status-*`)而非硬编码 hex——这样将来补 dark 只改 token 不改组件(零返工)。这是"不做 dark"前提下唯一要守的纪律。

---

## 3. P0：workflow 主轴 —— 给整个体验一根脊柱

> 解决硬伤 B。这是**导向性**的核心。

### 3.1 「分析进度轨」(Analysis Rail)

在 chat 区**顶部或左侧**放一条**常驻的、领域感知的工作流进度轨**，把 EthoInsight 的 7 阶段画成一条 stepper：

```
①上传 ─ ②范式 ─ ③列对齐 ─ ④指标计算 ─ ⑤数据质检 ─ ⑥统计解读 ─ ⑦报告
 ✓      ✓        ◖(等你确认)   ○            ○           ○          ○
```

- **状态来源**：不是新后端字段，而是**从已有信号推导**——
  - 阶段完成度：监听 subagent 完成事件（`task_completed`）+ `set_experiment_paradigm` tool_call（→②done）+ `ask_clarification` 类型为列对齐（→③等待）+ artifacts 出现 .png（→④有产出）。
  - 这些信号**前端全都已经能拿到**（§1.1）。先用前端推导跑起来；若推导不稳，**再考虑**让后端在 `thread.values` 加一个 `workflow_stage` 字段（轻量、单一来源，符合 SSOT）。
- **交互**：点某一阶段 → 消息流**锚点滚动**到该阶段对应的消息（deep-link 思想，`ui-ux-pro-max` `breadcrumb-web` / `state-preservation`）。
- **当前阶段高亮**（`nav-state-active`）：用品牌绿 + 一个**呼吸感的细描边**（不是大色块），日式克制。
- **等待决策态**：当某阶段在等 HITL，该节点变成**琥珀脉冲 + "等你确认"微标**，和 §5 的审批信号联动。

> 价值：研究员（非程序员）**第一眼就知道**「分析走到第几步、卡在哪、为什么停」。这是把"线性消息流"升级成"有地图的旅程"。

### 3.2 todo / 子任务规划可视化

`write_todos` 现在只是一行 CoT。升级为**与进度轨联动的可折叠规划面板**：
- todo 列表实时反映 agent 的子任务规划（`thread.values.todos` 已存在，后端调研 §1 坐实）。
- 完成的打勾、进行中的转圈、被 HITL 阻断的标琥珀。
- 这正面回应用户维度 4「AI 自主规划子任务 + 多步执行」**可见化**。

---

## 4. P1：生成式 UI —— 从"卡片"到"领域组件树"

> 解决硬伤 A + 落地用户维度 1/3「界面按需生成、动态组件树」。

### 4.1 核心机制：生成式组件注册表（Generative Component Registry）

**思想**：后端的工具调用 / 产物 / 状态，不再无脑映射成"文字 + 文件卡"，而是**按语义路由到专门的领域组件**。这就是 AG-UI 图里「工具调用 → 生成式 UI 组件」那一格的真正落地。

现状其实**已经是这个模式的雏形**——`ToolCall` 里对 `inspect_uploaded_file` / `prep_metric_plan` 做了专门渲染（`message-group.tsx:458-514`）。把它**抽象成注册表**：

```
toolName / artifactKind / stateSlice  →  registry  →  专用 React 组件
```

- `inspect_uploaded_file` → **数据集卡**（sheet 数、列名 chips、Treatment/Dose 徽章）—— 现在是一行文字摘要，升级成结构化卡。
- `prep_metric_plan` → **指标计划表**（paradigm / metric_count / subject_count + 可展开看每条指标）。
- `present_file`(charts) → **产物画廊**（§4.2，重点）。
- `set_experiment_paradigm` → **范式锁定卡**（显示锁定了哪条 EV19 模板 + 学术范式，可一眼复核）。
- 统计结果（report.md 里的表）→ **可排序统计表**（`sortable-table`：control vs treatment、effect size、p 值，tabular figures）。

**好处**：
1. 新工具/产物来了，只在注册表加一条，不动消息流主干（符合"复用 deerflow 现成"+ overlap 接受重复的工程纪律）。
2. "界面按需生成/销毁"自然成立——组件随对应消息进/出而挂载/卸载。
3. 未知工具 fallback 到通用 CoT step（现状已有此 fallback，保留）。

### 4.2 产物画廊（Artifact Gallery）—— 直击"几百张图"

**最终形态（按用户拍板）**：聊天流里**不铺图墙**——只 inline 展示**少量代表图**（aggregate 汇总图 + 至多几张精选），其余全部图收进一个**单独打开的产物画廊界面**（独立路由页 `/workspace/chats/[id]/gallery` 或全屏弹出层）。聊天流保持干净（日式克制），"看全部图"是一个明确的、用户主动触发的动作。

```
聊天流里（inline，克制）:
┌─────────────────────────────────────────┐
│  📊 已生成 6 张汇总图 + 112 张单样本图     │
│  [aggregate 汇总图 1] [汇总图 2] [汇总图3] │  ← 只展示 aggregate 代表图（数量少、最重要）
│            ⤢ 打开产物画廊查看全部 118 张 →  │  ← 主动入口
└─────────────────────────────────────────┘

点开 → 产物画廊（独立界面 / 全屏层）:
┌─────────────────────────────────────────────────────┐
│ 范式▾ 图类型▾ aggregate/per-subject▾ 组▾ subject▾  搜索  │ ← 分面筛选
│ ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐  …(虚拟滚动,懒加载)          │
│ └──┘└──┘└──┘└──┘└──┘└──┘                              │
│ [✓多选] → 小倍数并排对比模式   [下载选中] [导出CSV]      │
└─────────────────────────────────────────────────────┘
```

**为什么是"另开界面"而不是 inline 折叠**：①聊天流的心智是"对话/叙事",塞进上百张图(哪怕折叠)都破坏节奏;②画廊是"探索/检索"心智,需要全屏的筛选+对比空间;③两种心智分屏,各自体验最优(这正是图墙最佳实践——Figma/Google Photos/Lightroom 都是"流里给入口、专门界面里铺图")。

**数据层（决策1=路 A，关键前置）**：后端把 `plan_charts.json` 的 per-chart 元数据**带到前端**——`present_file` 时附带 `{path, chart_id, output_mode, paradigm, metric, subject?, group?}`,`thread.values.artifacts` 从扁平 `string[]` 升级成 `ArtifactMeta[]`。`plan_charts.json` 已是这份数据的唯一来源(后端调研坐实),只是"接出来"。

> ⚠️ 工程纪律：路 A 改后端产物契约,**必须** grep 所有 `artifacts` 消费方(`chat-box.tsx`、`artifact-file-list.tsx`、`hooks.ts` 的 `merge_artifacts` reducer)一起改 + 加测试。`ArtifactMeta` 向后兼容:老数据(纯字符串)退化成 `{path}` 仍可渲染(画廊里归到"未分类")。

**画廊视图层**（图墙最佳实践，全部命中 `ui-ux-pro-max` chart 准则）：
1. **分面筛选**（`large-dataset` drill-down / faceting）：`范式` / `图类型(box/bar/trajectory)` / `aggregate vs per-subject` / `组(control/treatment)` / `subject` + 文件名搜索。
2. **aggregate 优先**：画廊默认聚焦 aggregate(数量少、最重要);per-subject 折叠成分组,展开才虚拟化加载（`progressive-disclosure` + `lazy-load-below-fold`）。
3. **虚拟化网格**（`virtualize-lists`，50+ 必虚拟化）：per-subject 上百张用虚拟滚动,缩略图 `loading="lazy"` + `aspect-ratio` 占位防 CLS（`image-dimension` / `content-jumping`）。
4. **小倍数对比模式**（small multiples）：勾选多张 → 并排对齐网格,同范式同指标跨组/跨 subject 对比（科研最高频需求）。
5. **图的 a11y**：box plot 是统计图主力(`ui-ux-pro-max` 里 box plot=AA),缩略图旁带统计摘要副标(n / median / 组别) + 提供 CSV/数据表下载（`data-table` / `export-option`）;组别用颜色 + 形状/标签双编码(色盲友好)。
6. **失败/截断显式呈现**：`run_chart_plan` 的 `failed_charts` / `remaining_charts`(budget 截断)在画廊里**明确呈现**——"还有 N 张单样本图未生成(超出预算),[补全]",不静默少图（呼应 CLAUDE.md「无声截断 = 假装覆盖了」铁律）。
7. **单图详情**：点缩略图 → 复用现有 `ArtifactFileDetail`(已支持图片/下载/新窗口打开),或画廊内 lightbox(带键盘左右切换、ESC 关)。

**inline 代表图的选取规则**（确定性，不靠 LLM）：`output_mode == "aggregate"` 的图全部 inline(通常 ≤6 张);若无 aggregate(极少),取每个 metric 的第一张。其余进画廊。这条规则前端纯靠 `ArtifactMeta` 算,不需要 agent 决策。

### 4.3 思考链 / 工具流的视觉降噪

现在 thinking 面板 + CoT + subtask 三套 `ChainOfThought` 嵌套，边框套边框（§2.1 批评点）。统一成**一种**时间线视觉：
- 主轴一条**极细的竖线**（日式），节点是小圆点，不是一堆方框。
- reasoning 默认折叠成一行"思考中…"（克制），点开才展开（现状默认展开，信息过载）。
- 工具调用用 §4.1 的领域组件，不是泛 `WrenchIcon` 一刀切。

---

## 5. P1：可观测性 + HITL —— 审批链做成一等公民

> 解决硬伤 C + 落地用户维度 4/5「人机中断/回退、审批链、可审计/重放」。

### 5.1 决策/审批信号强化（可中断提示）

当 agent 在等 `ask_clarification`：
- **进度轨**对应节点变琥珀脉冲（§3.1）。
- 消息流里的反问块升级成**显眼的决策卡**：左侧琥珀色 accent bar + 图标 + "分析已暂停，等待你的确认"标题 + 选项按钮（`confirmation-dialogs` / `error-recovery` 的"清晰恢复路径"）。
- **输入框态联动**：等待决策时输入框 placeholder 变成"回答上面的问题，或直接输入…"，降低非程序员的茫然感。
- options 按钮：现状已有（`clarification-options.tsx`），加**键盘可达**（数字键 1/2/3 选，`tooltip-keyboard` / `keyboard-nav`）。

> 列语义对齐（`中心区`/`边缘区` 反问）、范式确认这类**领域关键决策**，正是 HITL 铁律所在（CLAUDE.md 第9条「组间比较不猜阈值」、memory「范式必反问」）。决策卡要把**"agent 为什么问、依据是什么列"**讲清楚（呼应 memory `feedback_identify_zone_info_not_persisted` 的教训：lead 要带列依据反问）。

### 5.2 运行轨迹面板（live trace + replay）—— 用户拍板"现在就做"

> 经后端数据核查(2026-06-24)拆成**两半**:能力边界由"哪些数据已持久化、可被前端拉取"决定。

新增一个**可切出的"运行轨迹"侧抽屉**(不默认占空间,按需开),分两个能力档:

#### 档 A — Live Trace（实时全保真，**今天就能做，零后端依赖**）

本次正在跑的分析,实时把 agent 行为铺成时间线:每个 subagent 派遣、每次工具调用、每个 gate 判定、每次 HITL 决策,按时序排,可展开看 input/output。
- **数据来源**:全部来自已有实时事件——`task_started/running/completed/failed`(SSE custom event)、tool_calls、reasoning、`gate_signals`(实时 AIMessage)、clarification。这些前端**实时全收得到**(`hooks.ts:onCustomEvent` 等)。
- subagent 内部步骤(`task_running`)实时**有**,所以 live trace 是**全保真**的。
- **gate 关卡卡片**:绿✓/黄⚠/红✗ + 展开看 `DataQualityWarning` 明细(severity/code/blocks_downstream),由现有 `QualityWarningBanner` 升级。

#### 档 B — Replay（回看历史 run，lead 层今天能做、subagent 内部需小后端补丁）

选 thread 里某次历史 run,把那次的时间线**按真实时序重播**。
- **lead 层全保真,今天就能做**:后端 event store 每条消息都持久化了 `type / tool_calls / tool 结果 / reasoning / caller(区分 lead vs subagent) / seq / created_at(墙钟)`,经现有端点 `GET /api/threads/{tid}/runs/{rid}/messages` 可拉取(`thread_runs.py:384`、`journal.py:353`)。**带时序回放**(每步间隔按 `created_at` 差)因此能做,不只是静态列表。
- **⚠️ 已知缺口(后端核查坐实)**:subagent **内部**步骤(`task_running` 中间态)是**临时事件,跑完即丢、未持久化**(task_tool.py 只 SSE 不写 journal;subagent executor `checkpointer=False`)。所以历史回放里 subagent 只能显示"派遣 → 结果",中间过程缺失。
- **补缺口的最小后端работа(各约 30-50 行,列为 P1 后端协同项,非阻塞 live trace)**:
  1. 把 `task_running` 以 `category="trace"` 落 event store(`task_tool.py`,~30 行)→ 解锁 subagent 内部步骤回放。
  2. 加 handoff JSON 只读端点 `GET /api/threads/{tid}/runs/{rid}/handoffs`(~50 行)→ 解锁历史 `gate_signals` / 质检明细回放(现在 handoff JSON 在 workspace 落盘但**未开 API**)。
  3. (可选,低优先)state-history 端点 → 回放每步 thread state 快照。

> **落地顺序**:先做**档 A(live trace,纯前端,零后端依赖)**——用户"现在就做"的诉求当期即满足;**档 B 的 lead 层回放**紧随(也几乎纯前端,复用现有 history 端点);subagent 内部回放 + 历史 gate 明细等那两个小后端补丁(与后端协同,不阻塞前面)。
>
> 这一层也直接服务**训练飞轮**:研究员看清 agent 每步,反馈(三按钮 verdict + revised_text)才打得准。可观测性 ≈ 飞轮数据质量。

### 5.3 回退（人机回退）

用户维度 4 的"回退"：
- 轻量版（web 先做）：每个 HITL 决策点支持"**改主意**"——重新触发该反问（LangGraph 的 `Command(goto=...)` + checkpointer 支持从某点续跑）。
- 重量版（后续）：从轨迹面板选任一历史节点 fork 一条新分支。这依赖后端 checkpointer fork 能力，列为 P2。

---

## 6. P2：desktop / 共享状态 / 能力边界（前置铺垫）

> 用户提到将来要 Electron desktop。**web 期不实现 desktop，但把抽象立对**，避免将来推倒重来。

### 6.1 Electron 策略

- **壳模式**：Electron 加载同一套 Next.js 前端（web 与 desktop 共用 99% 代码）。desktop 独有的是：本地文件系统直读（研究员的 EthoVision 导出文件不用上传，直接选本地目录）、离线、原生菜单/通知、多窗口（一个窗口看图、一个窗口看报告）。
- **技术选型**：Next.js 静态导出 / 或 Electron + Next 自定义 server；用 `electron-pro` 类方案做签名分发。**不在本方案展开**，单独立 spec。
- **关键**：所有"和环境耦合"的能力（文件访问、通知、存储）从一开始走**适配层接口**（web 实现 = 上传 API + localStorage；desktop 实现 = IPC + fs），UI 组件只依赖接口。

### 6.2 前端可写共享状态（用户维度 2）

现状是**单向**（后端 `thread.values` → 前端）。用户要的是**双向同步 + 冲突调和**：
- 落地形态：前端的 UI 状态（筛选条件、选中的图、展开的 subject、当前 facet）做成**可被 agent 读取的共享状态切片**——agent 能感知"用户正在看 control 组的 box plot"，从而上下文相关地回应。
- 技术路径：LangGraph 支持 `Command` 更新 state；前端可通过一个"UI 状态上报"通道（轻量 custom event 或 state patch）把前端意图回传。**冲突调和**用 last-write-wins + 版本号起步（不过度设计）。
- ⚠️ 这是**架构级改动**，且 deerflow 上游可能演进 AG-UI 双向协议——**先观察上游，不自造**（呼应 CLAUDE.md「复用 deerflow 现成优先」「sync 全量跟随上游」）。web 期只把"前端状态可序列化、可上报"的接口留出来。

### 6.3 能力边界沙盒（用户维度 6）

- 现状：html artifact 已 `iframe sandbox="allow-scripts allow-forms"`（`artifact-file-detail.tsx:453`）——已是正确方向。
- 扩展：生成式组件（§4.1）若将来允许 agent 产出**可交互的自定义 UI**（不只是图），需定义**能力清单**——哪些组件能调哪些 action、渲染在何种沙盒。web 期先约定"生成式组件只读、不可发起副作用"，把校验点（指令校验）的位置预留。

---

## 7. 贯穿始终：动效与细节的"设计感"

> 用户原话举例：**渐入渐出动画移出频率应是逐渐变慢的渐变曲线，不是 y=2x 匀速**。这一节是**质量基线**，不是单独阶段——每个组件交付都按此验收。

### 7.1 动效曲线（直接回应用户的例子）

- **入场（enter）**：`ease-out`（快起 → 缓停），用 `--ease-out-quint: cubic-bezier(0.22,1,0.36,1)`。视觉上元素"冲进来再优雅刹车"，就是用户要的"逐渐变慢的尾巴"。
- **退场（exit）**：用 `ease-in` 且**比入场快**（`exit-faster-than-enter`，约 60-70% 时长）——退场拖沓最显廉价。
- **绝不用 `linear`** 做 UI 过渡（`easing` 准则；`linear` 就是用户批评的 y=2x）。当前 `globals.css` 的 keyframes 全是 `ease-in-out`（对称、偏匀速感）→ 全部换成上面的非对称曲线。
- **spring 物理曲线**（`spring-physics`）：卡片展开/折叠、侧栏滑入用 `motion` 库（已装 `motion@12`）的 spring，比 cubic-bezier 更自然。
- **位移表达层级**（`hierarchy-motion`）：子任务卡从下方淡入（= 更深一层）、退场向上（= 返回）；侧栏从触发源滑出（`modal-motion`：从 artifacts 触发处展开，不是凭空出现）。

### 7.2 微交互细节（日式精致度）

- **stagger 入场**（`stagger-sequence`）：产物画廊的图、todo 列表项，逐项延迟 30-50ms 入场，不是齐刷刷一起出现（齐出 = 廉价感）。
- **press 反馈**（`scale-feedback`）：可点卡片/按钮按下 scale 0.97，松开回弹——但**不位移布局**（`Stable Interaction States`：press 不能让周围元素抖）。
- **可中断**（`interruptible` / `no-blocking-animation`）：动画进行中用户点击立即响应/打断，绝不锁 UI。
- **数字防跳**（`number-tabular`）：流式更新的 token 计数、p 值、计时用等宽数字。
- **骨架屏**（`progressive-loading`）：>300ms 的加载（图缩略图、报告渲染）用 skeleton/shimmer，不用空转圈（现状 `skeleton.tsx` 已有，扩展到画廊）。

### 7.3 全局必守（`ui-ux-pro-max` CRITICAL/HIGH）

- `prefers-reduced-motion`：所有动效在 reduced-motion 下降级/关闭（现状 `globals.css:423` 已对 `pulse-soft` 处理，**推广到所有新动效**）。这是 a11y 红线。
- 对比度 4.5:1（正文）/ 3:1（大字、图形）—— light 主题先验证;dark 推迟做时再独立验证(决策4)。
- 触摸目标 ≥44px、focus ring 不删、icon-only 按钮带 aria-label。
- 图表不靠颜色单独表意（`color-not-only`）—— 组别用颜色 + **形状/纹理/直接标签**双编码（色盲友好；box plot 已是 AA）。

---

## 8. 分期路线图（按 2026-06-24 决策修订）

> 原则：每期都**可独立交付、可灰度**，不阻塞 deerflow sync（叠加在分组之上，不动 `useStream` 核心）。**dark mode 已移出(决策4)**;**运行轨迹 live trace 提前到 Phase 0(决策5"现在就做")**。

### Phase 0（P0 止血 + 现在就要的，~2-2.5 周）

1. **设计 token 升级**：补 `--ease-*` 曲线 + 状态语义色（**不含 dark**）。换掉 `globals.css` 全部 `ease-in-out`→非对称减速曲线。**全站立刻有质感提升**，零风险（纯 CSS token）。← 第一步、性价比最高
2. **运行轨迹面板 · 档 A（live trace）**：实时时间线 + gate 卡。**零后端依赖**,正面交付用户"现在就做"。
3. **产物画廊 v1**：聊天流 inline 代表图 + "打开画廊"入口 + 独立画廊界面(facet 筛选 + aggregate 优先 + 虚拟化 + 失败/截断显式)。**依赖**后端 `ArtifactMeta` 契约(路 A,与后端协同)。
4. **分析进度轨 v1**：7 阶段 stepper,前端推导状态。
5. **审批决策卡**：`ask_clarification` 升级成显眼决策卡 + 进度轨联动 + 键盘可达。

### Phase 1（P1 体验跃迁，~2-3 周）

6. **运行轨迹面板 · 档 B（replay）**：lead 层历史回放(纯前端,复用 history 端点);**后端协同两小项**(task_running 落 journal ~30 行 + handoff 只读端点 ~50 行)解锁 subagent 内部步骤 + 历史 gate 明细回放。
7. **生成式组件注册表**：把 inspect/prep/paradigm/统计表抽象成注册表 + 领域组件。
8. **思考链视觉降噪**：统一时间线、默认折叠 reasoning、stagger/spring 微交互全面铺开。
9. **todo 规划面板** + 回退轻量版。

### Phase 2（P2 形态扩展，按需启动）

10. **dark mode**：补 `:root.dark` 一组 token + 双主题验证(决策4 推迟到此)。
11. **共享状态接口预留**（观察 deerflow 上游 AG-UI 双向协议进展再动）。
12. **Electron 适配层 + 壳**（单独 spec）。
13. **能力边界沙盒**（生成式可交互组件出现时再做）。

### 后端协同清单（非阻塞，与前端并行）

| 项 | 规模 | 解锁 | 阻塞谁 |
|---|---|---|---|
| `present_file` 附带 chart 元数据 → `ArtifactMeta[]` | 中(改产物契约+测试) | 产物画廊分面/对比 | Phase 0 画廊(路 A) |
| `task_running` 落 event store(`category=trace`) | ~30 行 | replay 里 subagent 内部步骤 | Phase 1 档 B(非全部) |
| handoff JSON 只读端点 | ~50 行 | replay 里历史 gate 明细 | Phase 1 档 B(非全部) |
| (可选) state-history 端点 | ~100 行 | 每步 state 快照回放 | 低优先,可不做 |

---

## 9. 工程纪律（守 CLAUDE.md / memory 的硬约束）

1. **不动流式核心**：`useStream` / `mergeMessages` / `dedupeMessagesByIdentity` / optimistic / summarization 边界是踩坑沉淀，升级一律叠加在 `groupMessages` 之上。
2. **改 `artifacts` 契约 = 改共享组件**：grep 所有消费方（`chat-box` / `artifact-file-list` / `merge_artifacts` reducer）一起改 + 测试 + `ArtifactMeta` 向后兼容。
3. **SSOT**：图表元数据唯一来源是后端 `plan_charts.json`，前端不重新发明分类；工作流阶段定义若需后端字段，单一来源。
4. **跟随 deerflow 上游**：双向共享状态 / AG-UI 协议**先看上游**，能复用绝不自造（上游是 infra 底座，默认全合）。前端定制集中在"领域语义层"（领域组件、进度轨、画廊），底座流式/SDK 层跟随上游。
5. **失败显式、不无声截断**：画廊必须呈现 `failed_charts` / `remaining_charts`（reward-hacking / under-exploration 自检）。
6. **reduced-motion 必降级、a11y 红线**：每个组件交付的验收项(dark/light 双测在做 dark 时才适用,见决策4)。
7. **受保护文件**：本方案基本只动 `frontend/`（非受保护）；若触及后端 `present_file` 产物 schema,按受保护文件 surgical 处理 + 跑 `make test` + 裸导入两生产入口。

---

## 10. 决策点（2026-06-24 已全部拍板）

原 5 个待定决策已由用户确认,见文首「决策已锁定」表。要点复述:

1. ✅ 产物画廊后端契约 = **路 A**(`ArtifactMeta`)。
2. ✅ 进度轨 = **做**(横/纵实施期定)。
3. ✅ 配色基调 = **不变**(保留 Forest Green + 暖白纸,日式只动密度/装饰/动效)。
4. ✅ dark mode = **先不做**,推到 Phase 2。
5. ✅ 运行轨迹/重放 = **现在就做**——拆 live trace(Phase 0,零后端依赖) + replay(Phase 1,lead 层纯前端 / subagent 内部 + 历史 gate 需两个小后端补丁)。

**硬伤 A 收敛**:聊天流只放代表图 + "打开画廊"入口,全部图进独立画廊界面(图墙最佳实践)。

**下一步**(2026-06-25 更新):Phase 0 第一步(设计 token + 动效曲线,#1)**已合并 dev**(PR#201);产物画廊(#3)/运行轨迹 live trace(#2)/设计语言(#6)由其它 agent **并行实施中**。后续 #4 进度轨/#5 决策卡/#7 性能/#8 堆叠上传未起。各项落地状态以 §25 spec 表后的「实施进度」块为准(`git show HEAD:` 坐实),已合项的实测验证见 [dev-code-fix-spec](../superpowers/specs/2026-06-25-frontend-phase0-dev-code-fix-spec.md)。每项落地前出独立 spec + plan。

---

*附：本方案基于 2026-06-24 对 `packages/agent/frontend` 全量组件 + 后端事件契约的调研，未写任何代码。落地前每个 Phase 应出独立 spec + plan（参照 `docs/superpowers/specs/` 格式），并补前端契约测试。*
