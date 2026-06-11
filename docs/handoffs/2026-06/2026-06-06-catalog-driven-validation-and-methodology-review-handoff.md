# Handoff — 数据处理方法论调研/grill + Catalog-Driven 验证实施 + 文档收尾

> 日期：2026-06-06 ｜ 分支：`dev`（与 origin/dev 同步，已含 PR #100/#101/#102）
> 这是给下一位 AI Agent 的交接，不是给用户的总结。

---

## 1. 本会话任务目标

从「review 一份数据处理方法论设计文档」（`docs/design/2026-06-06-data-processing-methodology-design.md`）出发，完成了一条完整链路：**调研核实 → grill 用户 → 修订设计 → 发现并修补合并事故 → 实施 catalog-driven 验证 → review → 修复 → 文档收尾**。核心成果是 **AutoResearch 启发的两层指标验证（L-A/L-B）已完整落地 dev**。

---

## 2. 本会话已完成（按时间顺序）

1. ✅ **方法论文档调研**（核实 §9 九问）：直接读代码裁定每条断言。最致命发现——文档反复强调的 `bash head -1 | tr ',' '\n'` 提列名方案**三处全错**（EV19 是 UTF-16/分号/第0行是`标题行数`，列名在第 N-2 行；依据 `parse/_core.py`）。
2. ✅ **深挖 + grill 用户**：解决设计树分支——删 head -1（列名走 inspect/identify 的 parser 或 dump_headers）、Layer 3 聚合 sum→开放问题（因 zone 是否空间重叠未知）、catalog-driven 4 缺口、AutoResearch 类比修正（train.py↔catalog 方向反了）。
3. ✅ **修订 Issue #98 + 设计文档**：把 head -1 和写死的 sum 从两处回收（避免误导行为学专家），zone 空间关系升级为 #98 决定性问题。
4. ✅ **发现并修补合并事故（PR #101）**：PR #100 合 S1-S4 时**漏掉了修正 commit `dee60114`**（时序陷阱：PR head 锁在旧 SHA）。导致 dev 上 validate.py 是错的 prefix 版（匹配不到真实指标）+ 缺 S2→S3 通路。cherry-pick 补合 → PR #101 已 merge。
5. ✅ **catalog-driven 验证实施 + review + 修复（PR #102）**：见 §3。
6. ✅ **文档收尾**（本次会话尾）：方法论设计 §6/§8 标"已实施"、CLAUDE.md 加两层验证说明、本 handoff。

---

## 3. PR #102 核心：两层指标验证（必读架构）

**两层分工**（为什么分两层：compute 脚本进程内只有 `--input/--output`，**拿不到 paradigm**，查不了 catalog；paradigm 只在 code-executor 层的 `plan_metrics.json` 可得）：

| 层 | 文件 | 验证什么 | 需 catalog? |
|----|------|---------|-----------|
| **L-A** | `ethoinsight/validate.py`（进程内 `emit_result`） | 只 NaN/Inf（name-agnostic 安全网） | 否 |
| **L-B** | `ethoinsight/validate_catalog.py`（code-executor 层 CLI） | output_unit 范围 + 复合_stats + 孤儿/未知单位 | 是 |

**L-B 两个入口**（关键区别，别混淆）：
- `validate_plan_results(plan)` — **CLI 用**。吃 `plan_metrics.json`，用每条 metric **自带的 `output_unit`**（resolve 从 catalog 透传），**不 load_catalog**。**按 subject 逐条验证**（plan 按 subject 展开，同 metric_id 多条），违规标签带 `#<subject_index>`。
- `validate_metrics_against_catalog(results, paradigm)` — **直接调函数**。load_catalog 查 output_unit，能检孤儿（`catalog_unknown`）。**CLI 路径不遇孤儿**（plan 只含 catalog 内指标）。

**4 缺口闭合**：mm_s2 枚举（6 单位穷举，未登记报 `unknown_output_unit`）/ 孤儿 `catalog_unknown` / 复合 `_stats` 拆字段（mean/median 套范围、std/sem 只≥0、n/count 验整数）/ 物理上限 `plausible_max` 预留（rule 表 `upper=None`，等 catalog 加字段）。

**接入点**：code-executor SKILL step 5.5 调 `python -m ethoinsight.validate_catalog --plan ...`；guardrail `ScriptInvocationOnlyProvider` 白名单已精准扩 `validate_catalog`（正则 + red 锚点测 lookalike deny）。VALIDATION_ERROR 行汇入 data_quality_warnings（`code=METRIC_VALIDATION`）。

**方法论 skill 下沉**：`ethoinsight-column-confirmation/references/column-processing-methodology.md`（三层框架 + 工具分工 + 禁 head -1 + 列名来源链；不内嵌结构化知识、Layer 3 聚合留白指向 #98）。

---

## 4. Review 发现并修复的缺陷（P0/P1，已在 PR #102）

| ID | 问题 | 修复 |
|----|------|------|
| **P0-1** | feature 分支基于旧 dev（PR #101 前），merge 冲突 | rebase 到最新 dev，手工解 validate.py/test_validate.py（取收窄版+保留 bool 防御） |
| **P0-2** | CLI 按 metric_id 当 key，plan 按 subject 展开 → **N 个 subject 只验最后 1 个** | 新增 `validate_plan_results`，按 subject 逐条 + `#index` 标签 |
| **P1-1** | CLI 多余 load_catalog | 直接用 plan 自带 output_unit |
| **P1-2** | 孤儿 catalog_unknown 在 CLI 路径死代码 | 文档化：孤儿检测仅在直接调函数入口 |
| **P1-3** | emit_result docstring 仍说 out-of-range + bool 防御 | 改 docstring；bool 显式跳过（从 #101 带入） |

测试：ethoinsight 全量 **706 passed**（+56 新测试）、后端 guardrail 29 passed、ruff 干净。补了多 subject 测试（每 subject 独立验证 + 全违规全报告 + plan-driven output_unit 证明）。

---

## 5. 关键陷阱（下一个 agent 必看）

1. **后端测试 editable-install 陷阱**：`.venv` 的 deerflow 是 editable install **指向主仓**（不是 worktree）。在 worktree 跑后端测试要 `PYTHONPATH=<worktree>/packages/agent/backend/packages/harness` 才测到 worktree 代码，否则测主仓 dev 版——本会话差点误判 guardrail fail。
2. **PR 时序陷阱**：PR 创建后又往分支 push，GitHub PR head 不自动跟进，合并时合的是锁定的旧 SHA。合 PR 前核对 GitHub PR 页的 commit 数 / head SHA。本会话 PR #100 就因此漏合 dee60114。
3. **head -1 永远不能提 EV19 列名**：UTF-16/分号/标题行数三处全错。列名只来自 inspect_uploaded_file（完整）或 `python -m ethoinsight.parse.dump_headers`（CLI）。identify 的 `evidence.columns` 是 `[:20]` 截断，不可当完整列名。

---

## 6. 下一步（未阻塞 / 阻塞分类）

**未阻塞，可立刻做**：
- **S5 Experiment Log**：方法论 §8 标"设计已就绪经 Opus review 待实施"。AutoResearch results.tsv 等价物，记录每次实验决策 + user_feedback 信号，为微调飞轮供数据。纯工程，我可写 spec。
- **真实用户反馈 bug triage**：#94（上传原始数据却被要求传统计数据，像核心路由 bug）、#97/#39（上传失败，19txt/103mb）。需调 thread 日志复现，可能影响 v0.1 可用性。

**阻塞，等行为学专家**（Issue #98）：
- Layer 1 固定列清单 / Layer 2 概念完整性 / **Layer 3 zone 聚合**（决定性问题=EV19 zone 是否空间重叠/嵌套，决定 OR vs sum）。**专家未回复**（#98 两条 comment 都是用户自己发的）。在它回来前 Layer 3 一行都不能写（写=猜）。

**阻塞，等专家产出**：#90 Golden Cases / #72 TST / #63 调参指南。

---

## 7. 仓库状态

- **dev HEAD**：`b79285da`（含 PR #100/#101/#102），与 origin/dev 同步
- **本会话产生的远程变更**：PR #101（补合 dee60114）已 merge、PR #102（catalog-driven + 修复）已 merge
- **待 commit/push**：本 handoff + 方法论设计 §6/§8 更新 + CLAUDE.md 更新（本会话尾，见下方 commit 建议）
- **worktree**：已清理（`/tmp/cdv-fix` removed）

---

## milestone 建议

本会话让 **AutoResearch 启发的指标验证体系** track 到达 checkpoint（两层验证 + 方法论 skill 下沉全部落地 dev）。建议**更新或新建** milestone：
- 可并入现有 `docs/milestone/column-semantics-alignment.md`（同属"真实数据处理"主题），或新建 `docs/milestone/catalog-driven-validation.md`。
- 关键摘要：S1-S4（PR #100）+ S2→S3 通路修复（PR #101）+ catalog-driven L-B（PR #102）已 done；Layer 1/2/3 列对齐深化 + S5 Experiment Log 待续；Layer 3 阻塞在 Issue #98（行为学专家 zone 空间关系）。
- README 索引同步更新。

下一 agent 读到本段后执行 milestone 创建/更新；用户也可说"为 catalog-driven validation 写 milestone"手动触发。
