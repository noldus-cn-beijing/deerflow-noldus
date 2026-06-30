# Handoff：生成式 UX + 前端设计语言两轨道 brainstorm 续（D0/D2/D3 spec + 两 bug 诊断修复 spec，全 push dev）（2026-06-30 续）

> 承接 `2026-06-30-generative-ux-and-design-language-brainstorm-handoff.md`。本会话开局「继续 brainstorm 剩余子项目」，按依赖排序 D0→D2→C2→D3→C3 逐个深打磨写 spec，中途因端到端 dogfood 发现两个 bug 切去诊断。**本会话以 brainstorm + 写 spec + 诊断为主，实施全交别的 agent。**

---

## 〇、一句话现状

- **git：本地 = origin/dev（已全 push，无分叉）**。dev HEAD = `43111e90`。今天 2026-06-30。
- 工作区只剩历史 untracked（`docs/reports/`、`reports/report for june/`、`scripts/repro/run_chart_plan_repro*.py`）+ 别人的 `M`（CLAUDE.md / skillopt plan 等）——**勿碰**。本会话产出全已 commit+push。
- **本会话新增 5 份 spec 全 push dev**（见 §一）。
- **C1 用户已派 agent 实施中**（不是本会话派的，用户自派）。
- 被 context 耗尽打断在「D3 第③段已确认、D3 spec 刚 push、准备问是否 review」处。

---

## 一、本会话已完成（✅）—— 全是 docs，全 push dev

| 子项目 | spec 文件 | commit | 状态 |
|---|---|---|---|
| **D0 UX audit** | `2026-06-30-d0-ux-audit-design.md` | `2f572259` | ✅ 定稿 push（用户通过） |
| **D2 设计系统** | `2026-06-30-d2-design-system-kit-design.md` | `6a3ad451` | ✅ 定稿 push（用户通过） |
| **C1 修正** | `2026-06-30-c1-metrics-table-export-...-design.md` | `bd9f27cc`→`2b78259d` | ✅ 修正后 push（见 §二.1 教训） |
| **Bug② HITL 越权修复** | `2026-06-30-hitl-over-confirmation-memory-not-current-turn-fix-spec.md` | `84f0dcab` | ✅ push |
| **Bug① plan 缺失假失败修复** | `2026-06-30-code-executor-transient-plan-missing-false-failure-fix-spec.md` | `84f0dcab` | ✅ push |
| **D3 a11y+响应式** | `2026-06-30-d3-a11y-responsive-design-spec.md` | `43111e90` | ✅ 定稿 push（用户通过③段，**未走 ExitPlanMode/最终 review gate**） |
| memory | `feedback_memory_prefill_must_not_be_counted_as_current_turn_confirmation.md` | （在 ~/.claude memory，未随仓库） | ✅ 已存+索引 |

### 各 spec 一句话核心
- **D0**：双路证据合流诊断（硬主路复用 `noldus-insight-e2e` skill 跑 prod build + 软分支补走 → 截图证据；读码路推导 → 5 透镜合流出 findings.md）。**只看不改**，喂 D1/A2/D2/C2。
- **D2**：`workspace/kit/` 业务组件层（StatusCard/StatusBadge+AccentBar/EmptyState/LoadingState 4 原语），顶层自由建 + 底两层 token 对齐守 sync，全量迁移分 3 批，kit 带测+原卡片不回归。**依赖 D1**。
- **C1**：纯后端导出 metrics_table.csv/.json + data-table 端点 + 列举（P3）。**修正核心**：画廊**已实现**（artifact-gallery 280 行很完整），「很多产物显示不出」根因=**后端列表端点只认 .png / .md/.html 两类扩展名**（artifacts.py:296/350），非两类产物无渲染路径直接消失。
- **D3**：a11y（WCAG 2.1 AA 五条）+ 三端全优化（桌面/平板/手机，手机产物面板变抽屉+触摸目标≥44px），横切治理立标准**写进 D1 DESIGN.md 不另立**，接 `vitest-axe`+`axe-core`（新 devDep）组件级护栏 + 对比度脚本。**依赖 D1/D2**。

---

## 二、关键发现/决策（下个 agent 直接用）

### 1. ⚠️ C1 误判教训（已纠正，但要知道）
本会话一度误以为「画廊未实现」，把 C1 收窄成纯后端、前端移交 C2（commit `bd9f27cc`）。**用户纠正：画廊已实现、有各种问题、很多产物显示不出来才有 C1/C2**。读码坐实后恢复（`2b78259d`）。**教训：改/删 spec 前先读真实代码，别凭路线图措辞猜实现状态**（守 read-handoff「git/代码是真相」原则）。

### 2. 两 bug 诊断结论（磁盘证据坐实，失败 thread `8827351d-e2b1-4292-b69e-a3bde14b5fb0`，在 `.deer-flow/users/e6cdba0f-647f-48a9-9480-7737b1b54b6e/threads/`）
- **Bug① plan_metrics.json 缺失 = 瞬时假失败，已自愈**。证据：plan_metrics.json/handoff_code_executor.json(status=completed,28 subject)/data-analyst/chart-maker 全在盘。code-executor 首派落在 plan 写盘前 → seal failed → lead 重跑成功。**非 #251/#252 引入**（#252 纯前端；#251 只加 stage narration 观测，wrap_tool_call 干净）。**开放问题**：已有 `path_sequence_provider.py:95-110` guard 检查 plan 存在，为何没拦住首派？（TOCTOU vs guard 没拦，实施 agent 先坐实）。
- **Bug② HITL 越权 = 真 bug**。用户只回模板 A，agent 拿 memory 历史偏好把分组+列语义标 `confirmed=true` 落盘（experiment-context.json 的 confirmed_at 与 paradigm 同瞬间 08:42:27）。根因：`prompt.py:587` 规则只护模板字段一个方向。修法=**结构门**（确认来源 provenance：memory 预填只能 confirmed=false，本轮用户输入才 confirmed=true），落 `experiment_context.py`。新 memory 已记。

### 3. brainstorm 一路拍板的关键决策
- **5 子项依赖序**：D0(无依赖)→D2(依赖D1)→C2(依赖D1/D2/C1)→D3(依赖D1/D2)→C3(依赖C1)。
- **C2 三段已定稿**（待写 spec，等 C1 产出）：在**已实现画廊**上「连内部一起重做」拿 SOTA，但 **dogfood 行为不变式逐条继承+回归断言**（chart_type 显式映射/防图墙折叠/每类型独立折叠态/失败告警/aggregate 代表图/lightbox+对比+ZIP 共 6 条）；架构=概览优先 + 分类可扩展(注册驱动)；概览层 4 件(报告入口卡+指标表组综述+代表图缩略复用 aggregate+计数概览)。
- **C2b（画廊后端可扩展产物列表）已立为新子项**：注册表内核(ext→kind+build_meta SSOT)+ 多端点门面(保 charts/reports 韧性+新 data 端点)；chart/report build_meta **字节级继承**现有 list_* 逻辑；data 类只 .csv/.json。**①②段定稿，③段+边界暂停等 C1 产出**（避 P3 列举撞车）。
- **sync 友好性已逐处核实**：前端全部(D1/D2/D3/C2/D0) + 后端端点层(C2b/C1 的 data-table) 都在 deerflow 子树外，**100% sync 友好**；唯一 sync 敏感=C1 导出(挂 run_metric_plan_tool.py 等子树内，但是 Noldus 原创文件=前向 feature，遇上游共享受保护文件按规则 surgical merge)。sync 只门控 `backend/packages/harness/deerflow/` 子树（`sync-deerflow.sh:206`）。

---

## 三、未完成事项（按优先级）

| # | 事项 | 状态/依赖 |
|---|---|---|
| 1 | **C2b 第③段 + 边界** | 暂停，**等 C1 产出**校准（避 P3 列举撞车）。①②段已定稿（§二.3） |
| 2 | **C2 写 spec** | 三段已定稿（§二.3），**等 C1 产出**（消费 metrics_table.json + data 端点） |
| 3 | **C3 框选追问 brainstorm** | 未开始，**依赖 C1 数据可框选** |
| 4 | **D3 最终 review gate** | D3 spec 已 push，用户通过③段内容但未走「请 review spec 文件」的最终确认——下个 agent 可补问或直接视作通过 |
| 5 | 两 bug 修复实施 | spec 已 push，**交别的 agent 实施**（含 TDD+防 vacuous+裸导入两入口） |

---

## 四、关键陷阱 / 注意事项

1. **C1 正被实施中**（用户自派）。C2b/C2 都碰 P3 列举/消费 C1 数据 → **必须等 C1 产出再写它们的 spec/边界**，否则撞车（改同片 artifacts.py + 重复造 data 端点）。
2. **画廊已实现**（artifact-gallery.tsx 280 行：分面筛选/aggregate-per_subject 分区/per-type 折叠子区/对比模式/lightbox/ZIP/失败告警）。任何画廊改动都是**在它之上重做**，不是从零。
3. **`git add` 永远精确路径**：工作区常驻别人的 `M` + 历史 untracked，**绝不 -A/.**。
4. **改 harness 核心后裸导入两入口**：两 bug 修复改 experiment_context/guard/seal 链，除测试外必跑 `import app.gateway` + `make_lead_agent`。
5. **sync 命脉**：前端改动只通过 token(globals.css) + workspace/ 结构，**不改 ai-elements/registry 结构、ui/ 只保 API**。
6. **每次 git 操作前 fetch 核分叉**：dev 多并发推进（本会话期间 A2 #252 被并发合入）。
7. **prod 服务**：本会话没起服务（纯 brainstorm）。下个 agent 跑 dogfood 需 `make dev`/prod build。失败 thread 在 per-user 目录（`.deer-flow/users/<uid>/threads/`），不是 flat `.deer-flow/threads/`。

---

## 五、下一位 Agent 的第一步

1. `git fetch origin dev` + 核分叉。读本 handoff + D1 路线图 spec（`51b35d50`）+ C1/C2b/C2 三 spec。
2. 相关 memory：`feedback_memory_prefill_must_not_be_counted_as_current_turn_confirmation`(新)、`feedback_single_source_of_truth`、`feedback_shadcn_ui_copyin_editable_vs_registry_pulled_generated`、`feedback_perf_is_efficient_impl_not_visual_downgrade`、`feedback_frontend_design_japanese_minimal_motion_craft`、`feedback_code_has_fix_not_equal_bug_eliminated`、`feedback_oft_single_zone_must_ask_not_guess`、`project_frontend_vitest_already_set_up_2_red_streaming_tests`。
3. 问用户走哪条：**等 C1 落地后续 C2b/C2 边界** / **brainstorm C3** / **派两 bug 修复实施** / **D3 补最终 review**。
4. brainstorm 走 `superpowers:brainstorming`（一次一问+定稿用户审+写 spec push）。

---

## 六、milestone 建议

本会话让设计语言轨道 D0/D2/D3 spec 就绪（D1 之前已立）、C2/C2b 设计定稿、两 dogfood bug 诊断+修复 spec。可回流 milestone README：
- 「前端设计语言重构」轨道（D0✅spec / D1✅spec / D2✅spec / D3✅spec / C2✅定稿待spec / C2b✅定稿待校准）—— 可考虑新建该 milestone（受 doc-sync 新建配额限制，按需）。
- 生成式 UX 轨道 C 系列：C1 实施中、C2/C2b/C3 待续。
- 两 bug 修复 spec 进「seal/HITL」相关 milestone 或新建 dogfood-fixes 批次。
