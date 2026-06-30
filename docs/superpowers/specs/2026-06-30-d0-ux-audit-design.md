# 设计 spec：D0 UX audit（资深视角双路证据合流诊断，只看不改）（2026-06-30）

> 本文档是前端设计语言轨道 **D0**（UX audit）的可实施设计。D0 在路线图 spec
> `2026-06-30-frontend-design-language-roadmap-and-d1-design-md.md` 里被排为第 0 步、
> 当时为赶立 D1 被**故意跳过**、留了「D0 可后补，作为 D1 修正输入」。本 spec 补上它。
>
> 缘起：用户要继续 brainstorm 设计语言轨道剩余子项目，按依赖排序 D0→D2→C2→D3→C3，
> D0 无依赖、是 D1-D3/C2 的事实地基，最先做。brainstorm 收敛出「双路证据合流诊断」形态。

---

## 目标

对 noldus-insight 前端现状做一次**资深 UX 视角的诊断走查**，产出一张
**问题清单（findings）**——`问题 | 在哪 | 严重度 | 违背理念 | 修复方向归轨`——
喂给 D1（DESIGN.md/token 修正输入）、A2（进度叙事）、D2（组件库）、C2（画廊布局）排优先级。

**D0 是诊断，只看不改。** 不写代码、不改 token、不动后端。改的事归 D1/D2/A2/C2。

---

## 为什么这么做（核心立场）

1. **旅程不固定**：EthoInsight 的用户旅程不是固定的端到端「上传→反问→分析→产物→追问」。
   会分支/降级：n=1 跳 data-analyst、知识问答无产物、范式各异、有的会话只追问、崩溃重连。
   → 走查若按「单条固定旅程」走会漏掉分支/降级/异常态的 UX 问题。所以采集用
   **硬主路 + 软分支**，诊断用**透镜**（按设计理念组织）而非按旅程步骤组织。
2. **证据要真**：问题清单不能凭空推。靠**双路证据合流**——实跑（真实像素）+ 读码（结构推导）。
3. **采集与判断分离**（守 memory `feedback_e2e_testing_deterministic_playwright_not_llm_browser_use`）：
   采集走**确定性 Playwright**（复用 `noldus-insight-e2e` skill），**不让 LLM 驱动浏览器边跑边主观判断**；
   判断（5 透镜）走读码 + 看采集到的截图，由人/读码 agent 合流下结论。

---

## 产出物（3 件）

### 产出 1：走查指导书（本 spec 的 §「走查指导书」章节）

交给**实跑 agent** 的说明。**硬主路 + 软分支**：

#### 硬主路（确定性，复用 e2e skill）

- **prod build** 起服务（`pnpm build && pnpm start`，或 prod compose `:2026`），`E2E_PERF_BUILD=prod`。
  - prod build 的理由是**视觉真实**（无 dev 开发期假象：未压缩 / React Strict 双渲染 / source-map 拉取 / Turbopack HMR），**不是为跑性能分**（D0 不测 perf）。
- 跑 `/noldus-insight-e2e <EPM 真实数据目录>`，配 `e2e-answers.yaml` 走完整 HITL 到 terminal。
  driver 已自动采集 SSE / 截图 / clarification（read-only forensics，不改源）。
- 在 e2e skill 既有截图点之外，**补 6 个关键态截图**：
  1. 首屏空态（未上传）
  2. 上传后（数据已挂、未开跑）
  3. 反问卡片态（列语义对齐 clarification）
  4. 分析中（subtask 卡片 + 流式指示器）
  5. 产物画廊态（报告 + 图表）
  6. 报告展开态

#### 软分支（agent 自主补走，给场景不给脚本）

至少各采 1 张截图，缺的**标「未触发，原因 X」，不静默漏**：

- **n=1 单样本**：每组 n<2 的降级态（lead fast-path 跳 data-analyst），界面长什么样。
- **知识问答**：无数据纯问答，产物面板应为空/隐藏，验证空态设计。
- **崩溃重连**：刷新/断连后重连（#249 修过的 stale-run 态），验证视觉是否错位。
- **多轮追问**：对话变长后的滚动行为 / 信息密度 / 历史叠加。

### 产出 2：双路证据

- **实跑路**：硬主路 + 软分支跑出的截图 + DOM + SSE 时间线，落
  `docs/reports/d0-audit/<date>/evidence/`。
- **读码路**：读 `packages/agent/frontend/src/components/workspace/` 关键组件推导结构性 UX 问题。
  已勘察的关键面（无 `*stage*/*progress*/*trace*` 组件——顶层 7 阶段进度轨已于 #231 移除，
  阶段叙事现为内联，属 A1/A2 territory）：
  - `messages/message-list.tsx`、`messages/message-group.tsx`、`messages/message-list-item.tsx`（对话流）
  - `messages/subtask-card.tsx`（子任务卡片）
  - `messages/decision-card.tsx`（决策卡）
  - `messages/clarification-options.tsx`（反问选项）
  - `messages/quality-warning-banner.tsx`（质量告警）
  - `artifacts/thread-assets-panel.tsx`（产物画廊——硬编码两段，见 C1/C2 spec）
  - `input-box.tsx`（输入区）
  - `workspace-sidebar.tsx`（导航侧栏）

### 产出 3：问题清单（D0 最终交付）

落 `docs/reports/d0-audit/<date>/findings.md`，markdown 表，每条带：

| 字段 | 说明 |
|---|---|
| 问题 | 一句话描述 |
| 在哪 | 组件路径 + 截图引用（evidence/ 下文件名） |
| 严重度 | 高/中/低（标准见下） |
| 违背理念 | 命中哪条诊断透镜（L1-L5） |
| 修复方向 | 归到哪条后续轨（D1/A2/D2/C2）+ 一句怎么改 |

清单末尾附 **D1 修正输入摘要**：哪些 finding 该改 DESIGN.md / globals.css token。

---

## 诊断透镜（对着截图 + 代码看什么——理念落点）

不是泛泛「资深 UX 走查」，而是 **5 条具体透镜**，每条对应已拍板的设计理念。
每条透镜走查产出 0-N 条问题，带截图引用 + 严重度。

| 透镜 | 问什么 | 来源理念 |
|---|---|---|
| **L1 当前态优先** | 进度/状态呈现是否只露「当前在干什么」，而非摆固定全流程清单？有没有强迫用户消化不相关阶段？ | 用户本会话原话：「没必要每次都告诉用户 0-1-2，只告诉当前在干什么」 |
| **L2 对话为主** | 对话流是不是主角？产物面板有没有抢焦点？空间分配是否「对话为主、产物为辅」？ | D1 路线图重心定调 |
| **L3 日式克制** | 留白是否足够？有没有过载/堆叠/过多网格线/过多高亮叠加？入场动效是否 ease-out 非 linear？ | memory `feedback_frontend_design_japanese_minimal_motion_craft` |
| **L4 不暴露内脏** | 界面有没有泄漏 `gate_signals` / handoff json / 虚拟路径 / 技术黑话给研究员？ | 输出宪法（`skills/custom/ethoinsight/references/output-constitution.md`） |
| **L5 分支态一致** | 降级态（n=1）/ 空态（知识问答）/ 异常态（重连）是否被妥善设计，还是裸奔/错位？ | 用户点的「旅程不固定」 |

---

## 严重度评分标准（免得 agent 随意打分）

- **高**：阻断理解 / 误导研究员 / 暴露内脏。
- **中**：别扭但不阻断，影响专业感。
- **低**：打磨项。

---

## 执行编排（谁跑、怎么合流）

实施时分**两个角色**，最后合流：

1. **实跑 agent**（采集，不下结论）：
   读本走查指导书 → prod build 起服务 → `/noldus-insight-e2e` 跑硬主路 + 软分支补走 →
   落截图/DOM/SSE 到 `docs/reports/d0-audit/<date>/evidence/`。
   **只采集，不带 5 透镜判断**（避免 LLM 边跑边主观漂移；判断留给合流）。

2. **读码 + 合流**（判断）：
   读 `workspace/` 关键组件 + 看实跑 agent 采集的截图 → 用 5 透镜逐条诊断 →
   产出 `findings.md`。

> 这样分守 memory `feedback_e2e_testing_deterministic_playwright_not_llm_browser_use`：
> **采集走确定性 Playwright（e2e skill），判断走人/读码**，不让 LLM 驱动浏览器边跑边判断。

---

## 验收（D0 这张清单怎么算合格）

1. 硬主路 6 个关键态截图齐全（首屏 / 上传后 / 反问 / 分析中 / 画廊 / 报告）。
2. 4 个软分支态至少各 1 张截图（缺的标「未触发，原因 X」，不静默漏）。
3. `findings.md` 每条带：截图引用 + 组件路径 + 严重度 + 违背透镜（L1-L5）+ 修复方向归轨。
4. 清单末尾给 **D1 修正输入摘要**（哪些 finding 该改 DESIGN.md / token）。

---

## 不做什么（守边界）

- ❌ 不改任何代码 / token（D0 是诊断，改归 D1/D2/A2/C2）。
- ❌ 不让 e2e skill 改后端 / 前端源（它本就 read-only forensics）。
- ❌ 不测性能（D0 是视觉/UX；perf 归 e2e skill 自己的 perf panel；prod build 只为视觉真实，不为跑分）。
- ❌ 不另造 Playwright 脚本（复用 `noldus-insight-e2e` skill driver）。
- ❌ 不预设固定旅程（硬主路 + 软分支，守「旅程不固定」）。

---

## 关联

- **复用**：`noldus-insight-e2e` skill（确定性 Playwright driver + HITL 预填 fail-loud + 截图 + read-only forensics）。
- **守 memory**：`feedback_e2e_testing_deterministic_playwright_not_llm_browser_use`（采集确定性、判断分离）、
  `feedback_frontend_design_japanese_minimal_motion_craft`（L3 透镜）、
  `feedback_perf_is_efficient_impl_not_visual_downgrade`（prod build 视觉真实非降级）。
- **上游路线图**：`2026-06-30-frontend-design-language-roadmap-and-d1-design-md.md`（D0 是其第 0 步）。
- **下游**：`findings.md` 喂 D1（DESIGN.md/token 修正）、A2（进度叙事 `cc7b0bdd`）、D2（组件库）、C2（画廊 `8fc7316a` 之上）。
