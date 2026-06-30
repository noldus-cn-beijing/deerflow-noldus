# Handoff：生成式 UX + 前端设计语言两大轨道 brainstorm 立项（spec 全 push dev，部分已实施）（2026-06-30）

> 本会话承接 `2026-06-30-frontend-darkmode-and-dogfood-three-bug-specs-handoff.md`。开局是「读 handoff + 测切回卡顿」，演变成**两条设计轨道的连环 brainstorm**：生成式 UX 演进（输出理念重构）+ 前端设计语言重构。产出 7 份 spec 全 push dev，其中 4 个 bug 已被并发 agent 实施合入。**本会话以 brainstorm + 写 spec 为主，实施交别的 agent。**

---

## 〇、一句话现状

- **git：本地 = origin/dev（0/0 同步）**。今天 2026-06-30。
- 工作区只剩**别人的改动 + 历史 untracked**，**本会话产出全已 commit+push**：
  - `M CLAUDE.md`、`M docs/plans/2026-06-04-skillopt-...md` —— **别人改的，勿碰勿提**
  - `?? docs/reports/`、`?? reports/report for june/`、`?? scripts/repro/run_chart_plan_repro*.py` —— 历史 untracked，**保持原样**
- **下一步主线 = 继续 brainstorm 剩余子项目（C2/C3/D0/D2/D3 任选）**，或派实施已就绪的 spec。被 `/handoff` 打断在「问下一个 brainstorm 哪个」处。

---

## 一、本会话已完成（✅）

### 1.1 切回卡顿（原主线）→ 结论：prod 不复现，续 spec 已删

- prod build 实测（thread `772ec083`，EPM 28 文件 dogfood 跑通：113 图全渲染、data-analyst 流式、多次切走切回）**全程无卡顿**。
- 结论：#238 后的「切回卡顿」是 **dev build 开发期开销假象**（未压缩/Strict 双渲染/source-map 拉取），非生产问题。
- 续 spec `2026-06-29-tab-switchback-jank-238-still-janks-followup-spec.md` **已删除**（commit `03e809b8`，前提作废）。原始 spec + Step0 报告保留存档。
- **alembic 坑**：prod 启动曾炸 `Can't locate revision 20260626_1700`（DB 停在 Stage C #241 已删的 revision），`make dev` 的 bootstrap 自愈到 head `0003`，prod 现在能起。

### 1.2 dogfood 发现 4 个显示 bug，全已实施合入 dev

| bug | spec | 实施状态 |
|---|---|---|
| ask_clarification 等待时底部 dots 仍跳 | `2026-06-30-clarification-awaiting-streaming-dots-fix-spec.md` | ✅ **#247 已合**（`e74757b1`） |
| subtask 卡片 run success 后仍「正在运行」 | `2026-06-30-subtask-card-stuck-in-progress-after-run-success-fix-spec.md` | ✅ **#248 已合**（`7866ec38`） |
| 崩溃重连 SSE 卡死 success run 空转 + cancel 409 | `2026-06-30-crash-reconnect-stale-run-spin-and-cancel-409-fix-spec.md` | ✅ **#249 已合**（`b2fd0e62`） |
| 图表命名 source_filename（上一会话遗留） | — | ✅ #245 已合 |

### 1.3 生成式 UX 演进轨道（brainstorm 立项）

总路线图 spec：`2026-06-30-generative-ux-roadmap-and-a1-event-track-foundation-design.md`（commit `6a86ae36`）。
**北极星**=Agent 驱动生成式 UI；**架构铁律**=infra 全用 DeerFlow 现成（astream 多轨/ThinkTagMiddleware/custom 轨/get_stream_writer）、翻译层=后端中间件由真实事件驱动（不靠 LLM 自报，防漂移+防失真，Air Canada 教训）、不引入 ag-ui 仅借分轨语汇。

| 步 | 子项目 | 状态 |
|---|---|---|
| **A1** | 后端事件分轨 StageNarrationMiddleware（custom 轨发 stage_plan/stage_update） | ✅ **#250 已合** |
| **A1 缺口补齐** | 识别范式阶段派发点 + knowledge-assistant 登记 | ✅ **#251 已合**（`9493a74e`） |
| **A2** | 前端分轨渲染（阶段叙事内联对话流，删 stage-broadcast 改后端 narration） | spec `cc7b0bdd`，**gate 已解除可实施**（见 §四） |
| C1 | 指标结果表导出（CSV+JSON）+ 画廊概览优先呈现 | spec `8fc7316a`，待实施 |
| C2 | 画廊 layout 重设计 | ⬜ 未 brainstorm，依赖 D1 |
| C3 | 产物框选→输入框追问（Gemini 式） | ⬜ 未 brainstorm，依赖 C1 |

### 1.4 前端设计语言重构轨道（brainstorm 立项）

总路线图 + D1 spec：`2026-06-30-frontend-design-language-roadmap-and-d1-design-md.md`（commit `51b35d50`）。
**重心**=对话为主、产物为辅。**sync 兼容命脉**=视觉语言只通过 token 注入（globals.css，sync 零冲突）+ 结构重设计只在 `workspace/`（Noldus 独占 75 文件）做，不改 ai-elements/registry 组件。

| 步 | 子项目 | 状态 |
|---|---|---|
| D0 | UX audit（资深走查现状） | ⬜ 未 brainstorm |
| **D1** | DESIGN.md（Stitch 9 段，日式+色盲安全）+ globals.css token | spec 已就绪，待实施 |
| D2 | 设计系统（分散 UI→统一组件库） | ⬜ 未 brainstorm |
| D3 | A11y + 多设备响应式 | ⬜ 未 brainstorm |

### 1.5 doc-sync 回流

commit `c62a13f9`：milestone README 活跃表「前端生成式 UX Phase 0」→ 替换为「Phase 1（输出理念重构）」，删过期切回卡顿引用。CLAUDE.md 328→328（无堆叠）。

---

## 二、关键发现/决策（取证坐实，下个 agent 直接用）

1. **分轨原理**：DeerFlow 上游已有完整事件分轨地基——`astream(stream_mode=['messages','custom','updates','values'])` 多轨传输、`ThinkTagMiddleware` 已把 `<think>` 剥到 reasoning 轨、`custom` 轨可由 `get_stream_writer()` 写。LLM **不参与分轨**（它只在 messages 轨吐 token），分轨是框架层+中间件代码的事 → 对「用户看什么」有确定性控制。
2. **翻译层最佳落点（调研坐实）**：后端中间件（真实状态机驱动），不是 prompt（漂移+打地鼠）、不是前端查表（漂移）。与 PR#213/SealGateMiddleware 同构。
3. **A2 的核实发现（重要）**：删前端 `stage-broadcast.ts` 前，核实出 A1 有两覆盖缺口——「识别范式」阶段永不点亮（A1 stage_update 只在派 subagent 时发，识别阶段 lead 自调工具不派 subagent）+ knowledge-assistant 漏登记。已写缺口补齐 spec，**#251 已实施合入**，缺口已闭合。
4. **指标结果数据现状**：`handoff_code_executor.json`（22KB 内部文件）的 `per_subject`（每 subject×每指标数值）+ `metrics_summary`（组级 mean/std/n）含全部指标结果，但**没以用户可读格式落盘**（outputs/ 只有 report.html，无 CSV）；`data-table` 端点是 placeholder。C1 要确定性导出干净产物（剥 gate_signals 等内脏）。
5. **画廊布局现状**：`thread-assets-panel.tsx` 是**硬编码两段**（reports section + charts section 上下叠），加新产物类型必须重构成可扩展布局（C1 做最小挂载，C2 做完整重设计）。
6. **前端分层（sync 关键）**：`workspace/`（75，Noldus 独占，可自由设计）/ `ui/`（41 shadcn copy-in，只改 token 保 API）/ `ai-elements/`（27 registry 拉取，禁改结构）。
7. **awesome-design-md** (https://github.com/voltagent/awesome-design-md)：73 个品牌 DESIGN.md（Stitch 9 段格式），D1 借其结构 + Linear/Notion/Claude 范本，不照抄品牌。
8. **数据表呈现最佳实践（调研坐实）**：概览先行（Shneiderman）+ 组摘要懒展开（不平铺/不卡顿，盈亏平衡 50 行）+ 分布缩略（均值会骗人）+ IQR 离群轻量标记（日式克制、色盲安全绝不红绿）。

---

## 三、未完成事项（按优先级）

| # | 事项 | 状态/依赖 |
|---|---|---|
| 1 | **派实施 A2**（gate 已解除：A1 缺口补齐 #251 已合） | spec `cc7b0bdd` 就绪 |
| 2 | **派实施 C1**（指标表导出+概览呈现） | spec `8fc7316a` 就绪，独立 |
| 3 | **派实施 D1**（DESIGN.md + token） | spec `51b35d50` 就绪，是 D2/C2 地基 |
| 4 | **继续 brainstorm**：C3 框选追问（依赖 C1）/ C2 画廊（依赖 D1）/ D0 audit / D2 组件库 / D3 a11y | 被 /handoff 打断在「问选哪个」处 |

---

## 四、关键陷阱 / 注意事项

1. **A2 实施顺序 gate 已解除**：A2 spec 原写「删 stage-broadcast 须 gate 在 A1 缺口补齐之后」——**#251 已合，gate 解除，A2 可全量实施**。
2. **`git add` 永远精确路径**：工作区常驻 `M CLAUDE.md`/`M skillopt`（别人的）+ 4 历史 untracked，**绝不 -A/.**。
3. **改 harness 核心后裸导入两入口**：A1 缺口补齐/C1 改 code-executor/seal 链，除测试外必跑 `import app.gateway` + `make_lead_agent`。
4. **prod vs dev**：本会话切过 prod build 测卡顿，**已 `make stop` 关闭**（三端口释放）。下个 agent 要跑 dogfood 重新 `make dev`。perf 验收只认 prod build。
5. **sync 兼容是设计语言轨道的命脉**：D1/D2/C2 改前端时，视觉只通过 token 注入、结构只动 workspace/，**不改 ai-elements/registry**（否则 sync 冲突 + re-pull 覆盖）。
6. **每次 git 操作前 fetch + 核分叉**：dev 多并发实例高频推进（本会话期间 #247/#248/#249/#250/#251 被并发实施合入）。

---

## 五、下一位 Agent 的第一步

1. `git fetch origin dev` + 核分叉。读本 handoff + 两份路线图 spec（生成式 UX `6a86ae36` + 设计语言 `51b35d50`）。
2. 相关 memory：`feedback_single_source_of_truth`、`feedback_sync_full_follow_upstream_infra`、`feedback_harnessx_ideas_on_deerflow_not_harnessx_mechanisms`、`feedback_frontend_design_japanese_minimal_motion_craft`、`feedback_perf_is_efficient_impl_not_visual_downgrade`、`feedback_shadcn_ui_copyin_editable_vs_registry_pulled_generated`、`feedback_tailwind_v4_dur_namespace_not_real_duration_utility_dead_class`。
3. 问用户：**派实施（A2/C1/D1 三选）还是继续 brainstorm（C2/C3/D0/D2/D3）**。brainstorm 走 `superpowers:brainstorming`（一次一问 + 设计定稿前用户审）。
4. 派实施时提醒目标 agent：A2 gate 已解除；改 harness 核心跑两裸导入；前端改动守 sync token 约束。

---

## 六、milestone 建议

本会话让两条轨道立项 + 一批 bug 修复。milestone README 已回流（`c62a13f9`，活跃表 Phase 1）。下个 agent 若 #250/#251/A2/C1/D1 进一步实施，可再回流：
- 「前端生成式 UX Phase 1」track 状态更新（A1+缺口补齐已合，A2/C1 进度）。
- 可考虑新建「前端设计语言重构」milestone（D0-D3+C2 轨道）——但受 doc-sync 新建配额限制，按需。
