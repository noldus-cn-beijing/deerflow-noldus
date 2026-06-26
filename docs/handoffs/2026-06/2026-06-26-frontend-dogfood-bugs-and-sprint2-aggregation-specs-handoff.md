# Handoff：前端 dogfood 多 bug + Sprint2 结构聚合 — 4 份 spec 待派（2026-06-26 晚）

> 本会话产出 **4 份实施 spec**（全部 untracked，本 handoff 一并 commit）。前一份 handoff（`2026-06-26-frontend-phase0-finish-and-multi-bug-fix-specs-handoff.md`）的待派项**已全部实施合并**（#218~#222），本会话是它之后的新一轮 dogfood 发现 + 一条主线 spec。
> dev HEAD：`8c2edf40`（#222 画廊折叠提示已合）。核实用 `git show HEAD:`，别 grep 工作树。

---

## 〇、一句话现状

前一轮 handoff 列的待派 bug **已全清**（#218 memory UUID / #219 输入框遮挡 / #220 后端质量 / #221 历史乱序 / #222 画廊折叠提示，全合 dev）。本会话用 owner 账号在本地 dev（:2026）做了**真浏览器自动化诊断**（CPU profile + 高度链实测 + 磁盘核验），发现一批新问题，写成 4 份 spec。**全部 untracked、未派实施。** 另有 Sprint2 结构聚合 spec（阻塞已解除，可开工）。

---

## 一、4 份 spec（全 untracked，待派）

| spec 文件（`docs/superpowers/specs/`） | 范围/风险 | 状态 |
|---|---|---|
| **`2026-06-26-conversation-gallery-empty-progress-rail-semantics-and-switchback-jank-spec.md`** | 前端 5 问题 / 🔴🔴 | ❌ 未派（**最痛，含对话流图廊空**） |
| **`2026-06-26-input-box-overlap-model-removal-hci-and-empty-trace-panel-fix-spec.md`** | 前端 4 问题 / 🟡 | ❌ 未派 |
| **`2026-06-26-chat-page-render-jank-on-open-fix-spec.md`** | 前端渲染性能 / 🟡 | ❌ 未派 |
| **`2026-06-26-column-semantics-sprint2-structural-aggregation-spec.md`** | 后端 ethoinsight / 🟡 | ❌ 未派（阻塞已解除） |

---

## 二、每份 spec 要点 + 关键取证

### A. 对话流图廊空 + 进度轨 + 切回卡顿 + 视口高度（5 问题，最痛）
`...conversation-gallery-empty-progress-rail-semantics-and-switchback-jank-spec.md`

1. **🔴 对话流图廊空**（用户最痛，实测复现）：113 图全成功落盘 `threads/<tid>/user-data/outputs/`，但**对话流里什么都看不到**。根因＝**两条取图路只修了一条**：`/gallery` 独立页走 `/artifacts/charts` 磁盘端点（#216 修对了），但**对话流内嵌图廊走 `chat-box.tsx:54` `thread.values.artifacts`（state 冒泡）**——subagent artifacts 不上行到 lead state（memory 铁律 `feedback_subagent_command_artifacts_not_bubble_to_lead_executor_drops_state`）→ 恒空。修法：对话流也接磁盘端点（抽 `/gallery` 的 fetch 成共享 hook）。
2. **进度轨失准 + 3. 语义** → 合并由**用户已拍板的「能力进度（动态阶段）」方案**治：阶段从本 run 实际 subagent dispatch 动态推导，不写死 7 阶段。画图就显示画图（自然解决"画图了还显示指标计算"），知识问答 run 不显示轨。改 `analysis-rail.tsx`，守 spec#4 红线（纯前端推导）。
4. **🔴 切回卡顿**：本会话 CPU profile 实测＝React 一次性重挂载（**证伪了 mergeMessages/流式层是根因**，它们不在热点）。用户追问"DeerFlow 原生没这问题、升级改了什么"→ spec 要求 **git 回归对比** 第一步：嫌疑是 Phase 0 叠加的 4 个订阅整个 messages 的 useMemo 派生组件（AnalysisRail/RunTrace/InlineArtifactSummary/DecisionCard）。详见配套 `...chat-page-render-jank-on-open-fix-spec.md`。
5. **🔴 视口高度链断裂**（真浏览器 thread `71c306bc` 实测）：① `[data-chat-root]`（`page.tsx:139` `size-full`=900px）溢出父 `main`（`chat-box.tsx:164` 仅 792px，被 `h-14` absolute header 挤）108px→底部被推出视口；② 滚动容器 `ai-elements/conversation.tsx:14` 是 `overflow-y-hidden`→右侧滚动条不见。修法：chat-root 改受父 flex 约束（非 size-full）；conversation 是 **registry-pulled 禁改源**，消费侧传 `className="overflow-y-auto"` override（它已 `cn(...,className)` 接受）。

### B. 输入框遮挡 + 移除模型选择器 + HCI + 运行轨迹空 panel（4 问题）
`...input-box-overlap-model-removal-hci-and-empty-trace-panel-fix-spec.md`
1. **输入框遮挡**：#219 动态 padding **只修了标准路由，agents 路由 `page.tsx:111-114` 漏 wire `onHeightChange`**。修法搬逻辑过去 + **抽 `INPUT_BOX_PADDING_BREATHING_ROOM_PX` 成共享常量别复制**。
2. **移除模型选择器**（用户拍板：整个移除）：删 `input-box.tsx:618-655` 整段 `<ModelSelector>`，保留 `context.model_name` 内部选定。
3. **HCI 打磨**（用户选：间距/圆角/阴影 + 按钮布局 + placeholder 全要）：消费侧 className override，**ai-elements 禁改源**。
4. **运行轨迹 panel 空**：**根因未坐实**（headless 复现不出该 thread 历史）。widget/panel 同源 useRunTrace，Explore 的"SubtaskContext race"假设**已证伪**。spec 要求实现时**先浏览器取证** `thread.messages` 是否带 tool_calls 再改，**别凭推断动 buildRunTrace**（#221 敏感区）。

### C. 打开/切回渲染卡顿
`...chat-page-render-jank-on-open-fix-spec.md` — CPU profile 实测根因（非虚拟化挂载 + Radix Collapsible useLayoutEffect + scrollTop）。**dev build 失真，验收以 prod build 为准**。与 A-4 切回卡顿同战场，建议同一 agent 串行。

### D. Sprint2 结构聚合（后端，阻塞已解除）
`...column-semantics-sprint2-structural-aggregation-spec.md`
- **关键发现**：N 列→1 概念聚合机制**其实已全链路实现**（`resolve.py:_build_zone_aliases_overrides` 多列收集 + `metrics/epm.py:101` `df[cols].max(axis=1)` OR 聚合 + catalog glob）。milestone 标的 blocked **滞后于代码**。
- **真实任务**＝坐实+固化+补测，**不是从零实现**：① 补多列聚合直接测试（含"累积区 A/B/all 三列同 alias 双重计数"陷阱测试）② 固化特殊规则（LDB 忽略隐藏区 / FST·TST 不分区 / MWM 不合并，来自同事方法论 `docs/review-packages/2026-06-09-feedbacks/自定义数据列识别对齐聚合.md`）③ 任务3 修双重计数（仅当陷阱测试暴露才做）④ 更新 milestone 状态。
- **#98 已 CLOSED、同事 PR #115 已交付方法论**——阻塞解除，可开工。

---

## 三、关键陷阱（必读）

1. **「代码有修复 ≠ 现象消除」（本会话亲历踩坑）**：前一 handoff 我断言"#216 修对了画廊"，被本次 dogfood 推翻——#216 只修 `/gallery` 独立页，对话流内嵌图廊仍空。**判前端产物问题必 dogfood 实测，别信代码推断**（memory `feedback_code_has_fix_not_equal_bug_eliminated`）。
2. **subagent artifacts 不上行**（铁律）：走磁盘端点，别再修"让 Command 上行到 lead state"。
3. **ai-elements / MagicUI / React Bits 是 registry-pulled 禁改源**：HCI/overflow 等只能消费侧 className override（`components.json` registries 段）。
4. **`mergeMessages`/`dedupeMessagesByIdentity` 是 #221 红线**：profile 已证伪是性能根因，别碰。
5. **运行轨迹 panel 空根因未坐实**：实现前先浏览器取证，别凭推断改 buildRunTrace。
6. **dogfood 账号**：owner `qiuyang.wang@noldus.com.cn` / `19961031`（`.com.cn`），本地 dev 端口 **2026**。诊断脚本留在 `/tmp/`（`inspect-layout.cjs` / `perf-profile.cjs` / `inspect-thread.cjs` / `perf-trace-chat.cjs`），实施 agent 可复用。

---

## 四、并行/串行建议

- **第 1 波（独立可并行）**：B（输入框 4 问题，但 panel 空需取证）/ D（后端 Sprint2，与前端正交）。
- **第 2 波（前端消息流容器层，串行同一 agent）**：A（5 问题）+ C（渲染卡顿）——它们都碰消息流容器/高度链/useMemo 派生组件，**同一 agent 串行避免互相踩**。A-5 视口高度、A-1 图廊、A-4 切回卡顿、C 都在这一层。
- **红线提醒**：任何碰 `message-list.tsx`/`hooks.ts`/流式核心的改动必跑 `node_modules/.bin/vitest run`（`mergeMessages.test.ts`/`utils.test.ts`）+ worktree 用 `node_modules/.bin/vitest` 别用 npx。

---

## 五、下一位 agent 第一步

1. 4 份 spec + 本 handoff 已 commit 进 dev（本会话同批）。
2. 派 D（后端 Sprint2）+ B（输入框）两路并行——独立、阻塞已清。
3. A+C 留给会 dogfood、能起真浏览器的 agent 串行——A-1 图廊空是研究员最痛，优先。先 curl `/artifacts/charts` 坐实端点返 113，再接前端对话流。

---

## 六、相关 memory / 文档

- memory `feedback_subagent_command_artifacts_not_bubble_to_lead_executor_drops_state`（图廊空根因）
- memory `feedback_code_has_fix_not_equal_bug_eliminated_seal_react_floor`（必 dogfood）
- memory `feedback_shadcn_ui_copyin_editable_vs_registry_pulled_generated`（ai-elements 禁改源）
- 同事方法论 `docs/review-packages/2026-06-09-feedbacks/自定义数据列识别对齐聚合.md`（Sprint2 SSOT）
- milestone `docs/milestone/column-semantics-alignment.md`（Sprint2 状态待更新）
