# 前端生成式 UX 升级 Phase 0（Frontend Generative UX — Phase 0）

**状态**：done（8 份 spec 全部实施合 dev + 后续 dogfood bug 修复批全合）
**时间跨度**：2026-06-24（方案/spec 出齐）~ 2026-06-26（最后一个 PR #227 合 dev）
**dev HEAD**：`07aa0684`（Phase 0 收官点）

## 做了什么

把现有前端（Next 16/React 19/Tailwind 4/shadcn/`@langchain/langgraph-sdk` useStream）从"能用的 chatbox"升级成**有导向的生成式分析界面**，**不重写不换栈**——所有改动叠加在 `groupMessages` 之上，不动 useStream/mergeMessages/dedupe/optimistic/summarization 核心（踩坑沉淀，重写必复发）。

针对三个硬伤设计了 8 份 spec：① 几百张 per_subject 图 = 扁平图墙（→ inline 代表图 + 独立画廊虚拟化）② workflow 无导向（→ 7 阶段进度轨 + 运行轨迹抽屉）③ 审批/中断不显眼（→ 决策卡）。配套横切纪律两份（设计语言工艺 / 运行时性能）。8 份全部实施合 dev，随后一轮真实 EPM Playwright dogfood 跑出一串前后端 bug，逐一取证定根因、修复合入。

判读哲学：**性能=高效实现非视觉降级**（守 `feedback_perf_is_efficient_impl_not_visual_downgrade`）；设计=日式简洁 + 动效曲线讲究（守 `feedback_frontend_design_japanese_minimal_motion_craft`）。

## 8 份 Phase 0 spec → PR 映射（全 done）

| spec | 主题 | PR | spec 文件 |
|---|---|---|---|
| **#1** | 动效曲线/时长/语义色 token 化 | #201 | `2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md` |
| **#6** | 设计语言工艺地基（elevation 阶梯 + clay accent + icon token + Mœbius 空状态插画） | #202 | `2026-06-24-frontend-phase0-6-design-language-craft-semiotics-spec.md` |
| **#2** | 运行轨迹侧抽屉（档 A live trace） | #203 | `2026-06-24-frontend-phase0-2-run-trace-live-spec.md` |
| **#3** | 产物画廊：inline 代表图 + ZIP/契约 + 虚拟化网格 + 分面筛选 + lightbox | #205 + #207 | `2026-06-24-frontend-phase0-3-artifact-gallery-spec.md` |
| **#8** | 多文件上传堆叠 + hover/点击扇开 | #210 | `2026-06-24-frontend-phase0-8-stacked-upload-attachments-spec.md` |
| **#7** | 运行时性能与丝滑度（渲染层，零碰流式核心） | #212 | `2026-06-24-frontend-phase0-7-runtime-performance-spec.md` |
| **#4** | 分析进度轨 AnalysisRail（7 阶段常驻 sticky，前端推导） | #214 | `2026-06-24-frontend-phase0-4-analysis-rail-spec.md` |
| **#5** | 审批/反问决策卡 DecisionCard（显眼强信号 + 键盘可达 + 输入框联动） | #217 | `2026-06-24-frontend-phase0-5-decision-card-spec.md` |

## 配套基础设施 / dogfood bug 修复批（全 done）

| PR | 修复 | 性质 |
|---|---|---|
| #204 | 流式瞬态孤儿 tool message 修复（M1 治本 + M2 兜底） | grouping |
| #206 | read_file 越界静默 bug 工程修 + code-executor「不 read 大 plan」prompt | 后端 |
| #208 | backend test baseline 清零（make test 15 个 pre-existing 失败 → 4615 全绿） | 测试地基 |
| #209 | thinking channel 语言约束：lead + 4 subagent 思考语言跟随用户语言 | 后端 prompt |
| #211 | 清零 5 个 pre-existing eslint error | 前端地基 |
| #213 + #216 | 产物画廊走磁盘端点 `/artifacts/charts`（113/113 全量返回，修「113 张图只显示 1 张」） | 后端端点 |
| #215 | 消息流附件复用 #8 堆叠，修发送后附件平铺淹没 | 前端 |
| #218 | 修 memory context UUID 加载崩溃（lead 不再静默丢记忆） | 后端 |
| #219 | 修悬浮输入框遮挡底部对话内容（动态测量输入框高度驱动底部留白） | 前端布局 |
| #220 | 后端质量三小项：charts legend warning / memory JSON 结构约束 / thread_id 贯通 | 后端 |
| #221 | 修重进 thread 历史合并乱序：mergeMessages 改为按 identity 归并保序 | 前端流式红线 |
| #222 | 画廊单样本图折叠入口加明显可点提示 | 前端 |
| #223 | 治理对话页打开/切回渲染卡顿：降虚拟化阈值 30→15 + 折叠历史思考块 + 分帧挂载 | 前端性能 |
| #224 | 输入框遮挡(agents 路由漏改) + 移除模型选择器 + footer 对齐打磨 | 前端 |
| #227 | 对话流图廊走磁盘端点 + 进度轨动态能力进度 + 视口高度链修复 | 前端收官 |

> #225 是「列语义 Sprint 2」，属另一条 track（[column-semantics-alignment.md](column-semantics-alignment.md)），不计入本 Phase 0。

## 关键节点

| 日期 | 事件 | handoff |
|------|------|---------|
| 6/24 | 前端 UX 大升级调研方案 + Phase 0 五份实施 spec 出齐（用 ui-ux-pro-max 查实准则） | [6/24 升级方案](../plans/2026-06-24-frontend-generative-ux-upgrade.md) |
| 6/25 | spec#6 设计语言 + spec#7 运行时性能 + spec#8 堆叠上传 补齐；画廊渲染压力三成本模型 + PNG 实测拍板缩略图 | [6/25 phase0 specs + gallery rendering](../handoffs/2026-06/2026-06-25-frontend-ux-upgrade-phase0-specs-gallery-rendering-handoff.md) |
| 6/25 | spec#1-#3/#7/#8 等实施合 dev（#201-#213 批） | [6/25 5specs implemented](../handoffs/2026-06/2026-06-25-epm-dogfood-5specs-implemented-merged-handoff.md) |
| 6/26 | 真实 EPM Playwright dogfood 跑出一串 bug，逐一取证定根因写 spec；#4/#5 收尾，三现象修复落 #216 磁盘端点方案 | [6/26 phase0 finish + multi-bug specs](../handoffs/2026-06/2026-06-26-frontend-phase0-finish-and-multi-bug-fix-specs-handoff.md) |
| 6/26 | dogfood bug 修复批（#218-#225）全合 | [6/26 dogfood bugs + sprint2](../handoffs/2026-06/2026-06-26-frontend-dogfood-bugs-and-sprint2-aggregation-specs-handoff.md) |
| 6/26 | **收官 #227**：对话流内嵌图廊从恒空的 `thread.values.artifacts`（state 冒泡）改接磁盘端点 + 进度轨动态能力进度 + 视口高度链修复 | [6/26 切回卡顿回归对比](../handoffs/2026-06/2026-06-26-phase0-frontend-regression-diff-switchback-jank.md) |

## 当前状态

- **完成项**：8 份 Phase 0 spec 全实施 + 配套基础设施 + dogfood bug 修复批全合 dev。前端从扁平图墙升级为 inline 代表图 + 虚拟化画廊（治几百张图卡顿）；7 阶段进度轨 + 运行轨迹抽屉给 workflow 导向；决策卡让审批/反问显眼可达；设计 token + 动效曲线 + elevation 阶梯落地；运行时性能（虚拟化阈值/分帧挂载/历史思考折叠）治打开/切回卡顿。
- **遗留项 / 暂缓**（非阻塞，按需再起）：
  - **dark mode** → 推迟到 Phase 2（`:root` 当前只覆盖 light；组件已用语义 token 不硬编码 hex，将来补 dark 零返工）。
  - **replay 回放**（spec#2 档 B）→ Phase 1（subagent 内部历史回放需 ~30 行 task_running 落 journal + ~50 行 handoff JSON 只读端点）。
  - **Chrome 后台节流切回卡顿** → 仍待真机火焰图（headless 测不到后台节流），未出 spec。
  - **缩略图 thumb_path / run_id 分段** → ArtifactMeta 可选增强，不阻塞（无 thumb 退化原图 + decoding=async）。
- **下一 milestone**：Phase 1（生成式组件注册表 / 思考链降噪 / todo+回退 / replay 档 B），尚未立项。

## 相关 handoff

- [6/24 升级方案](../plans/2026-06-24-frontend-generative-ux-upgrade.md) — 调研稿 + 三硬伤 + 分期决策
- [6/25 phase0 specs + gallery rendering](../handoffs/2026-06/2026-06-25-frontend-ux-upgrade-phase0-specs-gallery-rendering-handoff.md) — spec#6/#7 补齐 + 画廊渲染三成本模型
- [6/26 phase0 finish + multi-bug specs](../handoffs/2026-06/2026-06-26-frontend-phase0-finish-and-multi-bug-fix-specs-handoff.md) — Phase 0 收尾 + dogfood bug spec 清单（含一个方向错的修复纠正记录）
- [6/26 切回卡顿回归对比](../handoffs/2026-06/2026-06-26-phase0-frontend-regression-diff-switchback-jank.md) — #227 切回卡顿「我们引入了什么」profile 证伪分析
