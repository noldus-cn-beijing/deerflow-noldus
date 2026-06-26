# Handoff：架构方向立项批次 —— 文件路径可靠性 + experiment 跨范式 + Electron 桌面化 + e2e 增强（2026-06-26 晚二）

> 本会话产出 **4 份 spec/设计文档 + 1 个 GitHub issue + 多份 memory**，**全部已 commit 并 push 到 dev**（HEAD `3f025bed`，本地=远端 `0 0`）。本 handoff 是它们之后的状态快照。
> 核实用 `git show HEAD:` 或在 GitHub **切到 dev 分支**看（这些文档只在 dev、不在 main——符合"commit 先进 dev"规范，用户本会话一度"看不到"就是因为在 main 视角）。

---

## 〇、一句话现状

承接前一份 handoff（`2026-06-26-frontend-dogfood-bugs-and-sprint2-aggregation-specs-handoff.md`）。那批待派的前端 bug + Sprint2 **已全部实施合并**（#223 渲染卡顿 / #224 输入框+移除模型选择器 / #225 Sprint2 聚合）。本会话**没有实施任何产品代码**，而是**回答用户的架构方向性问题并立项**：① 文件系统可靠性 ② 跨范式对比 feature ③ Electron 桌面化 ④ e2e 测试增强（含对 browser-use 的否决）。产出全是设计/spec/探索文档 + 一个给行为学同事的 issue。**4 份文档全 untracked→已 commit→已 push 到 dev。**

---

## 一、本会话产出（全已 commit & push 到 dev）

| commit | 产出 | 性质 |
|---|---|---|
| `9ddcfb7d`+`776450b2`(rebase 后 hash) | 更新文档：结构聚合阻塞已解除、硬阻塞收敛为 Golden Cases | 文档校准 |
| `1539678b` | **文件路径可靠性承重墙 spec** + **experiment 跨范式 init 设计** 两 feature 立项 | 🔴 spec/设计 |
| `0f7f9787` | 跨范式方法论开 **Issue #226** 给同事 + 文档登记 | issue+登记 |
| `df92db3d` | **Electron 桌面化踩坑探索文档** | 🟡 探索 |
| `3f025bed` | **e2e skill 增强 spec**（HITL 预填 + 性能/视觉回归） | 🟡 spec |

四份文档路径（`docs/superpowers/specs/`）：
- `2026-06-26-file-path-reliability-loadbearing-convergence-spec.md`
- `2026-06-26-experiment-cross-paradigm-comparison-init-design.md`
- `2026-06-26-electron-desktop-packaging-pitfalls-exploration.md`
- `2026-06-26-e2e-skill-hitl-prefill-and-perf-visual-regression-spec.md`

---

## 二、四件事的结论 + 关键决策

### A. 文件路径可靠性（用户最在意——"无数 spec/handoff 栽在路径上"）
spec：`...file-path-reliability-loadbearing-convergence-spec.md`
- **立意=承重墙级收敛**，让"路径类 bug"家族（113图/FileNotFound/幻影文件名）结构性消失，不是打补丁。
- **关键坐实**：DeerFlow **已有** `Paths.resolve_virtual_path`（`config/paths.py:351`，段边界匹配+穿越守卫+user-scoped+legacy 迁移）。薄弱点是"机制未强制用全 + 无 DB 外键兜底 + artifact 无 run 隔离"，**非机制缺失**。原则=先查 DeerFlow 复用、缺的自己补。
- 5 任务：①DB 外键级联+删除事务化(P0) ②artifact 去重键加 run 维度治"113图"家族(P1，`thread_state.py:62 merge_artifacts` 受保护 surgical) ③路径解析单点强制+禁裸拼 `/mnt/` 守护测试(P1) ④shared handoff 归属校验(P2) ⑤legacy 观测(P2，**澄清 `user_dir` 已有迁移审计、大半误报**)。

### B. experiment 跨范式对比（"让项目更像 agent 不是 chatbox"）
设计：`...experiment-cross-paradigm-comparison-init-design.md`
- **用户拍板约束**：import=archive；**工程化快照提取 thread 具体文件**（不跨 thread 运行时读，因 thread 间文件本就隔离）；**≥2 thread 才可比**；**不替用户识别是否真同实验**。
- 架构：新增 3 表（experiments/experiment_threads/experiment_synthesis，仿 `thread_meta` persistence）+ Paths experiment 目录方法 + experiments router（仿 threads.py `@require_permission`）+ **快照而非引用**（原 thread 删了结论仍在）+ `experiment-synthesizer` subagent（关进受输出宪法约束的 subagent，别让 lead inline 对比产违禁词）。
- 提取清单 P0=`handoff_data_analyst.json`+`experiment_context.json`，P1=report.md+code_executor handoff，P2=图。
- **判读层阻塞同事**：跨范式"相同/不同"怎么算才科学 → 已开 **Issue #226**，最终落成新 skill `ethoinsight-cross-paradigm`。性质同 Golden Cases #90。
- **路径 spec 是它的地基**（提取依赖路径解析可靠），建议路径 spec 先行。

### C. Electron 桌面化（给不懂技术研究员的一键安装包）
探索：`...electron-desktop-packaging-pitfalls-exploration.md`
- **选 Electron 不选 Tauri**（前端 SSR 重度依赖、非静态 SPA，Tauri 需 3-4 周改造让路）。
- **倾向先走路线 A**（捆绑容器运行时 Podman + 复用 DeerFlow 现成 compose 镜像 + Electron 壳）：本项目后端有 ethoinsight 重科学栈（scipy/matplotlib/numpy），sidecar 冻结这些原生扩展业界公认易碎（Datasette 敢冻结因无科学栈）；A 用现成已验证镜像消除该风险。用户判定"管理员权限/重启可接受"抵消了 A 最大减分项。
- **A 代价**：体积~2GB、首启拉镜像慢、Win WSL2 前提、容器↔宿主文件映射。
- **立项前必过 3 个 go/no-go**：① 干净 Win+WSL2 机器 compose 全链路 dogfood（DooD/sandbox 子进程能跑 ethoinsight 是命门）② 桌面单用户绕登录+localhost CSRF ③ 拿 mac/Win 代码签名证书（不签会被 Gatekeeper/SmartScreen 拦死不懂技术用户）。
- **状态：探索完成，未立项实施**——三个 go/no-go 过了再写实施 spec。

### D. e2e skill 增强（用户痛点：前端"太卡"测不出 + HITL 答案随数据而异）
spec：`...e2e-skill-hitl-prefill-and-perf-visual-regression-spec.md`
- **先否决 browser-use**：LLM 驱动浏览器非确定性、无断言、不测性能、归因崩坏（被测 agent 已是 LLM 非确定性，再叠一层测试器无法归因）。我们已有更对口的 `noldus-insight-e2e`（Playwright/CDP 确定性驱动）→ 该走增强它。
- **缺陷 A（HITL 写死通用"确认"、范式无关）**：data 目录旁 `e2e-answers.yaml`，**按关键词匹配**问题（不按出现顺序——脆），**未匹配 fail-loud 停下报错**（不瞎确认，守"不知道就要问不能猜"）。无答案文件退化为旧行为。
- **缺陷 B（前端太卡测不出）**：加 CDP 长任务+交互耗时(P0，正是当初手动 profile 抓切回卡顿的维度)、FPS/Web Vitals/视觉回归(可选)。**头号约束：性能只对 prod build**（dev Turbopack/HMR/sourcemap 失真，dev 数据进面板标 skipped）。验证要亲手 revert 掉 #223 确认能抓出红。
- 关键文件：`.claude/skills/noldus-insight-e2e/scripts/{run-e2e.cjs(:118 HITL,:43 CDP,:154 截图),lib.js,analyze.py}` + `SKILL.md`。

---

## 三、未完成事项（按优先级，全部待派实施）

| # | spec | 性质 | 依赖/阻塞 |
|---|---|---|---|
| 1 | **对话流图廊空 + 进度轨** | 前端，研究员最痛 | 独立。A 那份 spec（`...conversation-gallery-empty-progress-rail...`）剩余部分；#223 只吃了其"切回卡顿"。**核实 `chat-box.tsx:54` 仍走恒空的 `thread.values.artifacts`**（state 冒泡），要改接磁盘端点 `/artifacts/charts`（#216 给 `/gallery` 独立页修对了、对话流这条没修） |
| 2 | **e2e skill 增强** | 测试工具 | 独立。有它后面回归才守得住 |
| 3 | **文件路径可靠性承重墙** | 后端地基 | 独立。是 #4 的地基，建议先行 |
| 4 | **experiment 跨范式对比 Phase 1** | 新 feature | 工程骨架可建；synthesizer 判读层挂 **#226** 等同事 |
| 5 | **Electron 桌面化** | 探索完成 | 待过 3 个 go/no-go 才立项 |

**用户尚未拍板下一步派哪个**（会话结束时正问"看不到 spec"，已澄清在 dev 分支）。本 agent 建议优先级：1 图廊空（最痛、独立、改动面清晰）→ 2 e2e 增强 → 3 路径承重墙 → 4 experiment Phase1。

---

## 四、关键陷阱 / 注意事项

1. **文档在 dev 不在 main**：用户本会话"看不到 spec"就是因为在 main 视角。所有 commit 先进 dev（CLAUDE.md 规范）。看 spec 切 dev 分支。
2. **本会话 push 多次遇分叉**：dev 被同事并行推进（#223/#224/#225 都是本会话期间别人推的）。每次 push 前 `git fetch` + 看 `comm -12` 零文件重叠 → rebase（非 merge，保线性）→ push。本会话所有 rebase 零冲突（文档 vs 前端源码不重叠）。
3. **3 个 untracked 是历史遗留非本会话产出**：`docs/reports/`（进度报告 html）、`reports/report for june/`（agent ontology/RL roadmap html+png）、`scripts/repro/`（run_chart_plan 复现脚本）。**已与用户确认保持原样**，别误提交。
4. **路径 spec 受保护文件清单**：`config/paths.py`/`thread_state.py`/`executor.py`/`task_tool.py` 都在 PROTECTED（`scripts/sync-deerflow.sh:51`），改这些必 surgical（只加约束不删 Noldus 定制）+ 改完裸导入两生产入口防闭环。
5. **Electron 不要被"捆绑 Docker 太重"轻易否掉**：用户已明确"管理员权限/重启在 Windows 装软件里正常、可接受"——这是路线 A 可行的前提，别忘。

---

## 五、相关 memory（本会话新增/更新）

- `feedback_desktop_packaging_electron_over_tauri_ssr_dependency` — Electron>Tauri + A/B 路线 + 3 go/no-go
- `reference_agentic_context_engineering_session_tree_v1_direction` — blog session-tree 思想=v1.0 方向、启示 experiment
- `project_2026-06-26_experiment_cross_paradigm_comparison_init` — 两 feature 立项全景
- `feedback_e2e_testing_deterministic_playwright_not_llm_browser_use` — 否决 browser-use + e2e 两缺陷增强
- milestone 已校准：结构聚合 #98 已解除、硬阻塞=Golden Cases #90；README 登记 4 新 track（路径/experiment/+#226/+e2e 散在各 spec）

---

## 六、下一位 Agent 的第一步

1. `git log --oneline -6` 确认 HEAD=`3f025bed`（或更新，dev 在动）。读本 handoff + 上面 5 份 memory。
2. 等用户拍板派哪个 spec。若用户说"按你建议"→ 先派 **#1 图廊空**（最痛、独立）：先 `curl localhost:2026/artifacts/charts?...` 坐实磁盘端点返图，再改对话流内嵌图廊从 `chat-box.tsx:54` 的 state 冒泡改接磁盘端点（抽 `/gallery` 的 fetch 成共享 hook）。需 dogfood 实测（"代码有修复≠现象消除"）。
3. 派后端类 spec（路径承重墙）前：读 CLAUDE.md「同步核心规则」+「harness 模块顶层 import 闭环风险」+「三大病理自检」；改受保护文件 surgical + 裸导入两入口验证。
