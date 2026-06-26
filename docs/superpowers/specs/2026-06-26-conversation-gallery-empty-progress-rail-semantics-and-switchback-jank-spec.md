# Spec：对话流图廊看不到 + 进度轨语义/失准 + 切回卡顿（dogfood 实测，多问题归因）

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-26
> 代码基线：dev HEAD `1c1033a7`
> 性质：🔴 高（对话流图廊空＝研究员看不到产物，最痛）+ 🟡 中（进度轨）+ 🔴 红线（切回卡顿，触流式层）
> 取证：dogfood thread `bd7ca7f7`（owner 账号，113 图全成功落盘）+ 读码坐实渲染链。**用户实测：对话流里期待看到图廊，什么都没有。**

> ⚠️ **给实施 agent 的诚实前言**：上一份 handoff 曾断言"#216 修对了画廊、能看到图"——**那是代码推断，被本次 dogfood 推翻**（违反 memory `feedback_code_has_fix_not_equal_bug_eliminated`：代码有修复≠现象消除）。#216 **只修了 `/gallery` 独立页**，**没修对话流内嵌图廊**（用户实际看的地方）。下面每条都以磁盘/dogfood 为准，别再凭"代码里有"判"现象没了"。

---

## 〇、四个问题（用户本轮提出）

1. **🔴 对话流里看不到图廊**——113 图全生成成功、落盘 `threads/<tid>/user-data/outputs/`，但对话流里产物展示**空**。
2. **进度轨与实际流程对不上**——已经在画图了，进度轨还停在"指标计算"。
3. **进度轨在非线性/追问场景的语义**——端到端跑完后继续追问，这个"上传→…→报告"的 7 阶段轨还有什么用？非端到端场景它会变吗？
4. **🔴 切回浏览器 thread 页明显卡几秒**——"原生 DeerFlow 没这问题，我们升级时改了什么？"

---

## 一、🔴 问题1：对话流图廊空（state 冒泡 vs 磁盘端点的双路分裂）

### 根因（已坐实，磁盘+读码双证）

前端取图有**两条独立的路，只修了一条**：

| 入口 | 数据源 | 状态 |
|---|---|---|
| **`/gallery` 独立页**（`app/.../gallery/page.tsx:34`） | `chartsArtifactsURL` → 后端 `/artifacts/charts` **磁盘端点**（`backend/app/gateway/routers/artifacts.py:241` `list_chart_artifacts`，rglob outputs/*.png） | ✅ #216 修对了 |
| **对话流内嵌图廊**（用户实际看的） | `message-list.tsx:200` `<InlineArtifactSummary artifacts={artifacts}>` ← `useArtifacts()` context ← `chat-box.tsx:54` `setArtifacts(thread.values.artifacts)` ＝ **LangGraph state 冒泡** | ❌ **没修，恒空** |

**为什么 state 冒泡恒空**：113 张图是 **chart-maker subagent** 画的，`run_chart_plan` 在 **subagent 独立 graph 的 state** 里登记 artifacts。**DeerFlow executor 只捞 subagent 文本结果、丢弃子 state，不上行到 lead**（memory `feedback_subagent_command_artifacts_not_bubble_to_lead_executor_drops_state`，铁律级）。所以 `thread.values.artifacts`（lead state）里没有这 113 张 → 对话流 `InlineArtifactSummary` 拿到空数组 → 什么都不显示。

**双重落空**：`message-list.tsx:185` 该块只在 `group.type === "assistant:present-files"` 时渲染，且即便渲染，`artifacts` 也是空 state。

### 修复：对话流内嵌图廊也改走磁盘端点（与 `/gallery` 同源）

把"图数据来自 `thread.values.artifacts`（state 冒泡）"换成"来自 `/artifacts/charts` 磁盘端点"——磁盘是唯一真相（同 #216 对 `/gallery` 的处方，只是这次用在对话流内嵌摘要上）。

- 加一个 hook（如 `useChartArtifacts(threadId)`）：fetch `chartsArtifactsURL(threadId)` → `normalizeArtifacts`，**run 结束/有 present-files 时触发或轮询一次**。`/gallery` 页的 fetch 逻辑（`gallery/page.tsx:32-52`，磁盘主 + state 回退）可直接抽成共享 hook，两处复用（守 SSOT，别复制）。
- `InlineArtifactSummary` 的 `artifacts` 改吃这个 hook 的结果，而不是 `useArtifacts()` 的空 context。
- **保留回退**：磁盘端点空/失败时回退 `thread.values.artifacts`（lead present 的代表图仍在 state，至少有 1 张），同 `/gallery`。
- **触发时机**：present-files 消息出现时拉一次；若 run 仍进行中，图陆续落盘，需在 run 完成（`onFinish`）后再拉一次确保全量。**别每帧拉**（性能）。

> **不要**再往"让 subagent Command 上行到 lead state"方向修（memory 铁律：DeerFlow executor 把 subagent 当文本黑盒，上行不了）。磁盘是唯一真相。

### 验证
- dogfood thread `bd7ca7f7`：对话流里 present-files 那条消息下，图廊摘要显示 113 张（aggregate 代表图 + per-subject 折叠）→ 点开能进 `/gallery` 看全部。
- 直接打端点确认返数：`GET /api/threads/bd7ca7f7-.../artifacts/charts` 应返 113 项（实施时先 curl 坐实端点本身 OK，再接前端）。

---

## 二、问题2：进度轨与实际流程对不上（画图了还显示"指标计算"）

### 现状（读码坐实）

`components/workspace/analysis-rail/analysis-rail.tsx`（189 行）——7 阶段（上传/范式识别/列对齐/指标计算/数据质检/统计解读/报告）是**前端从 `thread.messages` 推导**的（spec#4「零后端改动，前端推导」）。失准＝推导规则没覆盖 chart-maker 阶段：画图(chart-maker dispatch)发生时，推导逻辑没有把轨推进到对应阶段，仍停在"指标计算"。

### 修复
- **由问题 3 的动态阶段方案（用户已选 B）一并治**：阶段从"硬编码 7 阶段"改为"本 run 实际 subagent dispatch 动态推导"后，画图时阶段自然推进到"图表生成"，不会停在"指标计算"。**不要**单独再补一套静态 chart-maker→阶段映射（会和动态方案打架）——直接做动态方案，问题 2 是它的子结果。
- **守 spec#4 红线**：纯前端推导、零后端改动；不碰 `mergeMessages`/流式核心。

---

## 三、问题3：进度轨在非线性/追问场景的语义（设计问题，需用户拍板）

### 问题本质（用户的洞察是对的）

当前 7 阶段轨假设**一条从上传到报告的线性端到端流程**。但真实场景：
- **非端到端**：用户只问知识、只画图、只重算某指标——线性轨不适用。
- **追问**：端到端跑完后继续追问（如本 dogfood：跑完分析→追问画图→可能再追问报告）。此时"上传→报告"已全绿，固定轨**对追问无信息量**，甚至误导（追问画图时它已显示"报告 未开始"或全完成）。

### 候选方案（需用户选，见末尾 AskUserQuestion）

- **A. 轨绑定"当前 run"而非"整个 thread"**：每个 run（一次用户请求→agent 完成）推导自己的阶段轨；追问开新 run→新轨。线性假设在单 run 内成立，追问不再错位。
- **B. 轨变成"能力进度"而非"固定流水线"**：阶段动态化——本 run 实际触发了哪些 subagent 就显示哪些（画图 run 只显示"画图"阶段，知识问答 run 不显示轨）。
- **C. 端到端完成后轨折叠/隐藏**：首轮端到端显示完整轨，完成后追问时轨收起成一行摘要（"已完成完整分析 · 点击展开"），不占视觉。

> ✅ **用户已拍板：方案 B「能力进度（动态阶段）」。** 进度轨不再写死 7 阶段，改为**从本 run 实际触发的 subagent dispatch 动态推导**要显示哪些阶段：
> - 画图 run → 只显示"图表生成"阶段；知识问答 run → 不显示轨（无 pipeline）；完整端到端 → 显示实际走过的全部阶段。
> - **这天然解决问题 2**（画图了还显示"指标计算"）——阶段从实际 dispatch 推导，画图就推进到画图，不会停在指标计算。问题 2 与问题 3 合并由动态阶段方案一并治。
> - 实现：`analysis-rail.tsx` 的阶段集合从"硬编码 7 阶段常量"改为"扫本 run（或本 thread 最新 run）的 messages 里出现过的 subagent dispatch / 关键 tool_call → 映射成动态阶段列表"。**仍守 spec#4 红线：纯前端推导、零后端改动、不碰流式核心。**
> - 守 memory `feedback_frontend_design_japanese_minimal_motion_craft`（日式克制，知识问答这种无 pipeline 的 run 干脆不显示轨，别硬塞空轨）。

---

## 四、🔴 问题4：切回卡顿 + "DeerFlow 原生没这问题，升级改了什么"

### 已有的诊断结论（本会话早先 profile 实测，见 `2026-06-26-chat-page-render-jank-on-open-fix-spec.md`）

切回/打开卡顿 = **React 一次性挂载重消息节点**（jsxDEV + commit*Effects + Radix CollapsibleContentImpl.useLayoutEffect + scrollTop 抖动），**不是流式合并逻辑**（profile 证伪 `dedupeMessagesByIdentity`/`mergeMessages` 不在热点）。该 spec 已给治法（降虚拟化阈值 + 分帧挂载 + 折叠块懒挂载）。

### 本问题新增的关键追问："DeerFlow 原生为什么没有？我们改了什么？"

**这是 catastrophic-forgetting 自检的正确直觉——必须回答，因为它能定位"是我们引入的回归"。** 实施 agent 第一步做 **git 回归对比**：

1. `git log --oneline` 找 Phase 0 前端改造前的基线 commit（spec#1~#8 之前，约 #201 之前）。
2. 对比 `message-list.tsx` / `message-group.tsx` / 消息渲染链：**Phase 0 加了什么 DeerFlow 原生没有的东西**？已知嫌疑（本会话发现）：
   - **AnalysisRail（spec#4）**：sticky 常驻条，每次 `thread.messages` 变都重推导 7 阶段（`useMemo` over messages）——切回时大量消息一次性到位，触发重算 + 重渲染。
   - **RunTraceWidget（spec#2）** + **InlineArtifactSummary（spec#3）** + **DecisionCard（spec#5）**：都订阅 `thread.messages` 做 `useMemo` 派生，切回时一起重算。
   - **虚拟化阈值 30（spec#7）**：<30 组不虚拟化，全量挂载。
   - DeerFlow 原生消息流**没有这些叠加的 per-message 派生组件**，所以切回轻。
3. **结论方向**：卡顿大概率是 **Phase 0 在消息流上叠加的多个 `useMemo`-over-all-messages 派生组件**（AnalysisRail/RunTrace/ArtifactSummary）在切回（tab 重新可见、React 恢复渲染）时**同时重算+重挂载**。治法＝这些派生组件的 `useMemo` 依赖收敛（别依赖整个 messages 数组引用）、或切到后台时暂停、或分帧。

> 守 memory `feedback_perf_is_efficient_impl_not_visual_downgrade`：是省开销，不是砍 Phase 0 的视觉/功能。**这条与渲染卡顿 spec 是同一战场**，建议**同一 agent 串行处理**，先做 git 回归对比定位"我们引入了什么"，再治。

---

## 四点五、🔴 问题5：对话页视口高度链断裂（底部被裁 + 滚动条不见）

### 根因（真浏览器实测，thread `71c306bc`，viewport 900px，全部坐实——非推断）

量渲染后的实际高度链，**两个确凿结构错位**：

**① 底部被裁 → `data-chat-root` 高度溢出父容器**
- 父 `<main className="min-h-0 grow">`（`chat-box.tsx:164`）实测 **792px 高**（rectTop=76 → rectBottom=868）——顶部 header（`page.tsx:144` `h-14` + 间距）占了 76px。
- 子 `[data-chat-root]`（`page.tsx:139` `relative flex size-full min-h-0 justify-between`）用 `size-full` ＝ **900px**，**比父高 108px** → 底部（900）落在父底（868）之外，**被推出视口** → "看到的不是真正的底部"。
- 即：`size-full` 错误地相对视口取满高，没扣掉 header 占的 76px。应让 chat-root 受 flex 父约束（`min-h-0 flex-1`）而非 `size-full` 硬取满。

**② 滚动条不见 → 滚动容器 `overflow-y-hidden`**
- `ai-elements/conversation.tsx:14` `relative flex-1 overflow-y-hidden`——滚动容器**垂直方向 hidden** → 右侧垂直滚动条不显示；叠加①的高度溢出 → 滚动行为也算错（内容超出但容器 hidden）。

**③ 这是 Phase 0 引入的回归**（呼应问题4"DeerFlow 原生没这问题"）：absolute header（脱离文档流，不占高）+ `size-full` chat-root 的组合，假设了能拿满视口高，没扣 header → DeerFlow 原生若用正常 flex 列布局则无此问题。git 回归对比时一并核这块。

### 修复（守 ai-elements registry 红线）
- **①**：`data-chat-root` 的 `size-full` 改成受父 flex 约束的高度（如 `h-full min-h-0`，且确保父链 `min-h-0 grow` 真传导）——让它撑满父的 792px 而非硬取视口 900px。或：header 改为占文档流（不用 absolute）让父容器高度自然扣减。**两条路选一，实施时在真浏览器验 chat-root rectBottom ≤ 父 rectBottom**。
- **②**：`conversation.tsx` 的 `overflow-y-hidden` 应是 `overflow-y-auto`——但它是 **`@ai-elements` registry-pulled，禁改源**（`components.json`）。**只能在消费侧覆盖**：给 Conversation 传 `className` override（`cn` 后 `overflow-y-auto` 覆盖 `overflow-y-hidden`），或在外层包一个 `overflow-y-auto` 滚动容器。核 Conversation 是否已接受 `className` prop（:14 用了 `cn(..., className)`，**接受**）→ 消费侧传 `className="overflow-y-auto"` 即可，零改 registry。
- 滚动条**显示/使用问题**：修②后滚动条恢复；若还需样式（细滚动条/不占位），用 token 化的 scrollbar 样式（消费侧），守日式克制。
- **视口随窗口变化**：确认根链用 `h-dvh`（动态视口，随移动端地址栏/窗口 resize）而非 `h-screen`（固定）——实测 html/body/main 顶层 =900=vh 是对的，断点在 chat-root 的 `size-full`，修①即可；但顺带核根 layout 若用 `h-screen` 改 `h-dvh`。

### 验证
- thread `71c306bc` 真浏览器：resize 窗口高度 → 底部内容（输入框、最后消息）始终在视口内可见；右侧垂直滚动条可见可用；chat-root rectBottom ≤ 父 main rectBottom（不溢出）。
- 与问题1（输入框遮挡，已 #219/输入框 spec）、问题4（切回卡顿）**同属消息流容器层**，**同一前端 agent 串行处理**避免互相踩高度链。

---

## 五、关键文件

| 问题 | 文件 |
|---|---|
| 1 对话流图廊空 | `src/components/workspace/messages/message-list.tsx:185-206`（InlineArtifactSummary 挂载）；`src/components/workspace/artifacts/inline-artifact-summary.tsx`（吃磁盘端点 hook）；`src/app/workspace/chats/[thread_id]/gallery/page.tsx:32-52`（抽共享 fetch hook）；`src/core/artifacts/utils.ts:47`（`chartsArtifactsURL`）；`src/components/workspace/chats/chat-box.tsx:54`（state 冒泡填充处） |
| 2 进度轨失准 | `src/components/workspace/analysis-rail/analysis-rail.tsx`（阶段推导补 chart-maker） |
| 3 进度轨语义 | 同上（方案 A/B/C 待定后改） |
| 4 切回卡顿 | `2026-06-26-chat-page-render-jank-on-open-fix-spec.md` + Phase 0 派生组件（analysis-rail / run-trace-widget / inline-artifact-summary / decision-card 的 useMemo 依赖） |
| 5 视口高度链 | `src/components/workspace/chats/chat-box.tsx:164`（父 main 792px）；`src/app/workspace/chats/[thread_id]/page.tsx:139`（chat-root `size-full` 溢出）+ `:144`（absolute header h-14）；`src/components/ai-elements/conversation.tsx:14`（`overflow-y-hidden`，**禁改源，消费侧传 className override**）；agents 路由同款 `page.tsx:119` 同问题 |

---

## 六、验证

1. `pnpm check`。
2. **问题1（最痛，必验）**：dogfood `bd7ca7f7` → 对话流里直接看到图廊（≥113 张可达），不用进独立页。先 curl `/artifacts/charts` 坐实端点返 113，再验前端接上。
3. **问题2**：跑到 chart-maker 阶段时进度轨推进到对应阶段，不停在"指标计算"。
4. **问题4**：git 回归对比产出"Phase 0 引入了什么"的明确清单；切回卡顿在 prod build 下 Long Task 显著下降；流式回归测试全绿（`node_modules/.bin/vitest run`，`mergeMessages.test.ts`/`utils.test.ts`）。
5. **问题5（视口高度链）**：thread `71c306bc` 真浏览器 resize → 底部（输入框/最后消息）始终在视口内；右侧垂直滚动条可见可用；实测 `[data-chat-root]` rectBottom ≤ 父 `main` rectBottom（不溢出）。
6. **后端铁律**：若动 `/artifacts/charts` 端点，改 `app/gateway` 后裸导入 `import app.gateway`。

---

## 七、不做 / 边界

- **不修"让 subagent artifacts 上行到 lead state"**（DeerFlow 黑盒，改不了；走磁盘）。
- **不碰** `mergeMessages`/`dedupeMessagesByIdentity`（#221 红线，profile 已证伪是性能根因）。
- **不砍 Phase 0 视觉/功能**保性能（是收敛 useMemo/分帧，不是删 AnalysisRail）。
- 进度轨语义（问题3）**先对齐用户再实施**。

---

*依据：dogfood thread `bd7ca7f7` 磁盘有 113 png（`threads/<tid>/user-data/outputs/`）+ 读码坐实对话流图廊走 `thread.values.artifacts`（state 冒泡，subagent artifacts 不上行→恒空）vs `/gallery` 独立页走 `/artifacts/charts` 磁盘端点（#216 只修了后者）；进度轨 `analysis-rail.tsx` 前端推导未覆盖 chart-maker 阶段；切回卡顿经本会话 CPU profile 证为 React 重挂载（Phase 0 叠加的 useMemo-over-messages 派生组件嫌疑最大，需 git 回归对比坐实）。memory：`feedback_subagent_command_artifacts_not_bubble_to_lead_executor_drops_state` / `feedback_code_has_fix_not_equal_bug_eliminated`。*
