# Handoff：本会话写了 4 份前端 spec（dark mode + dogfood 三 bug）全 push dev，未实施（2026-06-30）

> 本会话承接 `2026-06-29-suspect-columns-fix-and-sync-stage-c-spec-handoff.md`。**本会话只写 spec、不实施**（用户明确：实施交给别的 agent）。产出 4 份前端 spec 全部已 commit + push origin/dev。Stage C（sync alembic）用户已交给别的 agent，本会话没碰、它已被并发合入。

---

## 〇、一句话现状

- **git：本地 = origin/dev（0/0 同步）**。今天是 **2026-06-30**（注意：会话中一度沿用 06-29，但磁盘 date 是 06-30；spec 文件名用了 `2026-06-29` 前缀，是写时按当时认知命名，**不影响内容**，下个 agent 知道即可）。
- 本会话两个 spec commit 因 dev 被并发 rebase，现 hash = `aa442d89`（三 bug spec）、`d2ad7eb2`（dark mode spec）。HEAD 之上还有别人的 `6bff70bf`(doc-sync skill)、`8eb1e54b`(#242 导出)。
- 工作区只剩 4 个**历史 untracked**（`docs/reports/`、`reports/report for june/`、`scripts/repro/run_chart_plan_repro*.py`）—— **非本会话产出，保持原样别动别提交**。
- **下一步主线 = 实施这 4 份 spec 中的任意一份**（都未动手）。优先级见 §二。

---

## 一、本会话已完成（✅）

### 1.1 写了 4 份前端 spec（全 push dev，均未实施）

| spec 文件（`docs/superpowers/specs/`） | 性质 | 实施难度 |
|---|---|---|
| `2026-06-29-dark-mode-toggle-button-and-token-completion-spec.md` | 确定性，直接可做 | 小 |
| `2026-06-29-chart-display-name-source-filename-spec.md` | 确定性，最干净 | 小-中 |
| `2026-06-29-todo-list-expand-clipping-fix-spec.md` | 确定性单点回归 | 极小 |
| `2026-06-29-tab-switchback-jank-238-still-janks-followup-spec.md` | **先量再改**（需真机 profile 才能选修法） | 中（且阻塞在 profile） |

### 1.2 关键诊断结论（取证坐实，下个 agent 直接用）

**dogfood thread = `dcde1446-659e-4504-9e95-7b89f9c23e51`**（EPM 多文件，28 trial，113 张图）。磁盘落点：`packages/agent/backend/.deer-flow/users/cd95effa-d595-441a-bc44-29db0f3e259d/threads/dcde1446-.../`。

- **① dark mode 基本已建好**（里程碑「推迟 Phase 2」**已过期**）：next-themes 已装、`layout.tsx:22` 已配 `attribute="class" enableSystem`、`.dark` token 块已有 ~40 变量、设置面板已有系统/浅/深三选一。**真实缺口只有「无一键切换按钮」+「`.dark` 缺 8 个 token（`--shadow-float`+7 个 `--stage-*`）」**。用户决策：亮↔暗两态循环、按钮放 sidebar 顶部 logo 行、保留落地页 `forcedTheme="dark"` 不动。

- **② 图表命名**：113 图全叫 `plot_heatmap_sN`，不带来源。但 `inputs_heatmap_s0.json`=`Trial 1.xlsx`、`s14.json`=`Trial 15.xlsx`——**subject→trial 映射磁盘上本来就有**，只是没带进展示。根因链：`resolve.py` `_chart_to_plan()`(约1151-1176) 只用 idx 不记文件名 → schema.py:209 `PlanChart` 无字段 → `resolve.py:1440` 序列化漏 → `artifacts.py` ArtifactMeta 空 → `gallery-grid.tsx:41` 展示名不含来源。修法=加 `source_filename` 贯穿，**不改物理文件名**。

- **③ To-dos 压扁**：根因=PR#201(`4d847228`) 把 `todo-list.tsx:68-73` 的 `<main>` 改成 `transition-[height]`+固定 `h-28`(112px)，内部 `QueueList` 是 `max-h-40`(160px)+外层 `overflow-hidden` → 超出被裁。修法=改 grid-rows-[0fr↔1fr] 自适应。确定性，读代码即定论。

- **④ 切回卡顿**：#238(`18b412c3` 候选B `useDocumentVisibility`) 已合 dev，**但用户真机切 VS Code 再切回仍卡十几秒**。这恰是上次 Step0 报告（`docs/superpowers/reports/2026-06-29-tab-switchback-jank-step0-trace-report.md`）说的「headless 测不出、需真机坐实」路径——**现在坐实卡顿真实存在**。最可能主因=`visibilitychange` 真机延迟/不可靠致 hook 感知失败照样积压；叠加大消息 markdown 重解析 + 虚拟化重测量。**deerflow 上游无现成修复可拉**（上游根本不用 `useDeferredValue`，那是我们 #232 独有，故上游无此坑）。用户拍板：**先量再改**（续 spec 的 Step1 真机 profile 定真凶再选修法，不盲改）。

---

## 二、未完成事项（按优先级）

| # | 事项 | 状态/依赖 |
|---|---|---|
| 1 | **实施 To-dos 压扁修复** | 最小、确定性，建议先做。spec 给了 grid-rows 改法 |
| 2 | **实施图表命名 source_filename** | 确定性，跨前后端 4-5 文件、各几行。spec 详尽 |
| 3 | **实施 dark mode 切换按钮** | 确定性，spec 含完整组件源码 + token 值 |
| 4 | 切回卡顿：**先 Step1 真机 profile** | 阻塞在「需有 GUI 的机器 + headed Chrome + OS 级真切 tab」，headless 做不到。出 trace 报告后才选修法 |
| 5 | （可选）`/doc-sync` 回流文档 | 前端 Phase0 里程碑「dark mode 推迟 Phase 2」已过期、应更新为「基本已建、只缺按钮+token」。本会话结尾提过，用户未拍 |

---

## 三、关键陷阱 / 注意事项

1. **dev 多并发实例高频推进**：本会话期间 dev 被并发 rebase（我的 commit hash 变过）、并发合入 #241(Stage C)/#242(导出)/doc-sync。**每次 git 操作前 `git fetch origin dev` + `git rev-list --left-right --count HEAD...origin/dev`**。push 前必核分叉。
2. **`git add` 永远精确路径，绝不 `-A`/`.`**：工作区常驻 4 个历史 untracked（见 §〇），别误提。
3. **spec 文件名是 `2026-06-29` 前缀但今天是 06-30**：写时按当时认知命名，内容不受影响，别因日期困惑。
4. **改 harness 核心（图表命名要改 ethoinsight + backend artifacts.py）后**：除测试外必跑两裸导入 `import app.gateway` + `make_lead_agent`（守闭环铁律）。
5. **图表命名改 `PlanChart` schema 加字段务必带默认值 `""`**：否则按位置构造的调用点错位，改后 grep `PlanChart(` 核所有构造点。
6. **切回卡顿 spec 守红线**：不动 `useStream`/`mergeMessages`/dedupe；dev build perf 不作数（prod build `make start`）；未 trace 坐实真凶前不盲上修法。
7. **本地 dev/prod 状态**：上次 Step0 复现把本地切过 prod build。用户本会话能跑 dogfood 应已是 dev，但若发现是 prod（`make start`）想回 dev，`make stop && make dev`。

---

## 四、下一位 Agent 的第一步

1. `git fetch origin dev` + 核分叉（dev 并发活跃）。若 behind 先 rebase 调和（守线性）。
2. 读本 handoff + 要实施的那份 spec（4 选 1）+ 相关 memory（`feedback_frontend_design_japanese_minimal_motion_craft`、`feedback_perf_is_efficient_impl_not_visual_downgrade`、`feedback_code_has_fix_not_equal_bug_eliminated_seal_react_floor`、`feedback_e2e_testing_deterministic_playwright_not_llm_browser_use`）。
3. 实施建议从 **#1 To-dos**（最小）或 **#2 图表命名**（收益最大、用户最在意）起手。**用 TDD**（确定性 bug 先写断言再改）。切回卡顿（#4）若要碰，**先真机 profile 出 trace 报告**，别直接写修复代码。
4. 完成后：精确路径 commit（拆语义）、`pnpm check`（前端）/ `pytest`+裸导入（后端）验证、问用户是否 push。

---

## 五、milestone 建议

本会话让「前端 Phase0 遗留项」状态发生变化，建议下个 agent（或用 `/doc-sync`）更新 `docs/milestone/frontend-generative-ux-phase0.md`：
- **dark mode**：从「推迟 Phase 2、`:root` 只覆盖 light」改为「**基础设施基本已建（next-themes/`.dark`块/设置面板三选一齐全），仅缺一键切换按钮+8 token，spec 已立**」。
- 新增一批 dogfood bug spec 记录：图表命名 / To-dos 压扁 / 切回卡顿续（4 份 spec 文件名见 §一）。
