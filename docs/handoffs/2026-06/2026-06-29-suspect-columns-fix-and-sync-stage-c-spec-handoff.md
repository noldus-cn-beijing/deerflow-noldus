# Handoff：suspect_columns 列对齐修复（已合）+ DeerFlow sync Stage A 被并发实例合入 + Stage C spec 立项待实施（2026-06-29 续）

> 本会话承接 `2026-06-29-html-report-defects-and-new-specs-handoff.md`。做了三件事：① 诊断+修复 dogfood 暴露的「列语义已确认却重复找列」bug（已合 dev）；② 评估并为本周 DeerFlow 上游更新写 sync spec（Stage A 在写 spec 期间被**另一个并发实例**实施合入，Stage C 由我立独立 spec 待实施）；③ 全程多次调和 dev 并发分叉。**会话结束时本地与 origin/dev 完全同步（0/0），本会话两个 commit 均已在远端。**

---

## 〇、一句话现状

- **git：本地 = origin/dev（分叉 0/0），无未提交本会话工作**。HEAD=`3075387a`（Stage C spec）。工作区只剩 3 个历史 untracked（`docs/reports/`、`reports/report for june/`、`scripts/repro/*.py`）—— **非本会话产出，保持原样别动**。
- 本会话两 commit 均已落远端：`5010c2d0`（suspect_columns 修复 + 测试）、`3075387a`（Stage C spec）。
- **下一步主线 = 实施 DeerFlow sync Stage C**（persistence/alembic 升级 + 部署链改造），spec 已入库、决策已拍板，**尚未动手**。

---

## 一、本会话已完成（✅）

### 1.1 修复「列语义已确认却重复去找列」bug（已合 `5010c2d0`）
- **现象**（用户报 + thread `3dcac0a0` 磁盘取证）：EPM 数据用自定义区列 `open`/`closed`，用户确认列语义后，agent 仍重复 inspect 找列。
- **根因（坐实，非「agent 无视用户」）**：`identify_ev19_template` 第 0 步就输出 `zone_info.suspect_columns=["open","closed"]`，但 lead prompt 674-675 行把 `inspect_uploaded_file`/列问题**只框定为「分组」问题**，没指引「suspect_columns 已在手 → 折进首个合并反问 + 同一次 `set_experiment_paradigm` 带 `column_semantics`」。结果首次 `set_experiment_paradigm` 漏带 column_semantics → `prep_metric_plan` 报 `columns_missing` → 才回头重 inspect（一次失败 prep + 一次多余 inspect 的浪费来回）。时间戳铁证：`experiment-context.json` 的 `paradigm_confirmed_at`(07:31:41) 早于 `column_semantics.confirmed_at`(07:32:43)。
- **判定 ETHO-5（结构已对、只缺指引）→ 改 prompt**（守 HarnessX §6.6：收紧现有指令、不加第 6 条 reminder；deepseek 正面表述）。
- **改动**：[prompt.py:674-680](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L674-L680) 把单一 inspect 触发拆成两情形（有 suspect_columns→主动对齐；evidence 不足→inspect）。
- **TDD**：先写 [test_lead_prompt_suspect_columns_preflight.py](packages/agent/backend/tests/test_lead_prompt_suspect_columns_preflight.py)（断言 prompt 含 `suspect_columns`+`column_semantics` 且同处 `<paradigm_identification_system>` 段，纯 source-substring 不依赖 LLM），watched RED（`column_semantics` 全文 0 次）→ 改后 GREEN。
- **验证**：全量 **4745 passed, 27 skipped**；两裸导入入口（`import app.gateway` / `make_lead_agent`）exit=0 无环。
- **memory 已记**：`feedback_suspect_columns_must_fold_into_first_clarification_not_after_prep_fail`（含「时间戳辨『漏信号』vs『有信号没用』」判别铁律）+ MEMORY.md 索引。

### 1.2 DeerFlow sync Stage A —— **被并发实例 `affc0494` 实施合入**（非本会话动手）
- 本会话读 `scripts/sync-deerflow.sh`、fetch 上游、分类了 `11415875→b3c312b7` 的 28 commit / 34 harness 文件（**21 NEW + 11 SAFE + 2 PROTECTED**），并写了 Stage A+B 父 spec。
- **写 spec 期间，另一个并发实例 `affc0494` 把 Stage A 实施合入了 dev**，范围与我 spec 几乎逐条一致（A1 9 SAFE 全量、A2 15 个 tui 全量+不加 textual、A4 mcp/tools.py surgical 保 4096 截断合上游隔离），并 **defer 了 Stage C**。`.deerflow-sync-state` 已推进到 `last_sync_commit: b3c312b7`，commit message 极详尽（含 Stage C SKIP 理由 + 指向我的 Stage C spec 文件名）。
- 因此**父 spec 已作废、本会话已删除**（`2026-06-29-sync-deerflow-11415875-to-b3c312b7.md` 不再存在）。

### 1.3 多次调和 dev 并发分叉（已完成）
- 本会话期间 dev 被并发实例高频推进（#237 seal fail-loud / #238 tab-jank 候选B / #239 导出 spike / #240 导出实施 spec / `affc0494` sync / golden-case 对齐）。
- 用 **rebase**（线性）调和过一次分叉（你的 golden-case commit rebase 到远端之上）；后又 fast-forward pull 一次。过程中处理了 3 个 untracked 文件挡 rebase/pull（均证实=远端版的本地副本，删本地零信息损失：`report-export-formats-spike.md`、`report-export-formats-impl.md`、`report-export-formats.md` 的过期改动）。**经验**：dev 有并发实例，每次 git 操作前 `git fetch` + `git rev-list --left-right --count` 看分叉；untracked 挡路先 `diff` 证同内容再删。

### 1.4 顺带查清的两件事（无需再动）
- **前端 typography build error**：用户报「#236 改 lockfile 没 pnpm install」。实测**已自愈**——typography `0.5.20` 已装、`pnpm install --frozen-lockfile`=「Already up to date」、`pnpm build` ✓ 干净。之前的报错是 install 前的陈旧态。
- **tab-switchback-jank spec**：上一会话判 Step 0 inconclusive（headless `Page.frozen` 复现不出真实 hidden-tab SSE 积压）。本会话**未处理**；并发实例 `18b412c3`(#238) 已用候选 B（hidden 时绕过 useDeferredValue）实施合入。

---

## 二、未完成事项（按优先级）

| # | 事项 | 依赖/状态 |
|---|---|---|
| 1 | **实施 DeerFlow sync Stage C**（persistence/alembic 升级 + 部署链改造） | spec 已入库、决策已拍板，**未动手**。详见 §三 |
| 2 | （可选）本会话两 commit 已在远端，**无 push 待办** | — |
| 3 | dark mode → P2、replay → P1（前端遗留，沿用上批 handoff） | 未启动 |

---

## 三、下一步主线：实施 DeerFlow sync Stage C

**Spec（已入库）**：[docs/superpowers/specs/2026-06-29-sync-deerflow-stage-c-alembic-bootstrap-deploy.md](docs/superpowers/specs/2026-06-29-sync-deerflow-stage-c-alembic-bootstrap-deploy.md)（commit `3075387a`）

### 用户已拍板的决策（**直接执行，无需再问**）
- **DB 可随便删**（项目未进生产）→ 折叠成上游单链，不做历史数据迁移。
- **部署链取 C7-b**：`deploy-via-tar.sh` 保留「服务起来前迁移失败即 abort」层，但 `run-db-migrations.sh` 简化为 `alembic upgrade head`、删硬编码 `HEAD_REV`。
- **新 revision 命名跟上游 `000N_` 风格**（`0003_noldus_feedback_ext`），保持单一体系。

### 要做什么（spec 的 C1-C8）
- **C1** 引入上游 alembic 机器（NEW 全量合）：`persistence/bootstrap.py` + `migrations/{_env_filters,_helpers,script.py.mako, versions/0001_baseline, versions/0002_runs_token_usage}.py`。
- **C2** SAFE 全量覆盖：`persistence/run/model.py`、`migrations/env.py`。
- **C3** 删我们旧 4 revision：`versions/{20260512_feedback_verdict_revised,20260601_feedback_paradigm,20260622_run_token_usage_by_model,20260626_thread_cascade_fk}.py`。
- **C4** 新增 `versions/0003_noldus_feedback_ext.py`（down_revision=`0002_runs_token_usage`）：`feedback.rating` 改 nullable + 补回 Noldus 三列 `verdict/revised_text/paradigm`（用 C1 的 `_helpers.safe_add_column` 幂等）；若 C5 核出上游 0001 缺 thread cascade FK 一并补。
- **C5** 写 revision 前核：上游 `0001_baseline` 对 `runs.token_usage_by_model` / `runs|run_events→threads_meta ON DELETE CASCADE FK` / `channel_*` 列的覆盖。
- **C6** `engine.py` surgical（PROTECTED）：**保 idempotency guard**（`if _engine is not None: return`，防 langgraph blockbuster 500）、**采** 上游 `bootstrap_schema` 调用、跟随上游加 `PRAGMA busy_timeout=30000`。head-to-head diff 核。
- **C7-b** 部署链：改 `packages/agent/scripts/run-db-migrations.sh`（简化为 `alembic upgrade head`、删 `HEAD_REV="20260626_1700"`）+ 核 `deploy-via-tar.sh:197-219` 迁移步。
- **C8** wipe 本地 DB 重 bootstrap 验证（`.deer-flow/data/deerflow.db` 等可删；799 个 training-data JSONL **不在 DB 里**、不受影响）。

### 必保的 Noldus 定制（不补回就丢）
1. `persistence/feedback/model.py`（PROTECTED）：`rating` nullable + `verdict/revised_text/paradigm` 三列。**上游 0001_baseline 的 feedback 表 `rating NOT NULL` 且无这三列** → C4 增量补回。
2. `persistence/engine.py`（PROTECTED）：idempotency guard。

### 🔴 核心风险（spec 已详述）
**双 alembic root 冲突**：我们和上游各有一条 root（down_revision 都=None）→ 直接共存 = multiple-heads，`upgrade head` 歧义。解法=删我们旧链、折叠成「上游 baseline 单链 + Noldus 0003 增量」。**绝不**用脚本「全量合 persistence/*」一把梭（会洗掉 engine.py guard + 制造双 root + 丢 feedback 三列）。

---

## 四、关键陷阱 / 注意事项

1. **dev 有多个并发实例高频推进**：每次 git 操作前 `git fetch origin dev` + `git rev-list --left-right --count HEAD...origin/dev`。push 前必 fetch（本会话 push 被挡过两次）。untracked 挡 rebase/pull → 先 `diff <(cat 本地) <(git show origin/dev:文件)` 证同内容再删，**绝不盲删可能本地独有的文件**。
2. **`git add` 永远精确路径，绝不 `-A`/`.`**：工作区常驻 3 个历史 untracked（`docs/reports/`、`reports/report for june/`、`scripts/repro/*.py`）+ 可能有别人遗留的散改动/旧 stash（本会话见过一个 2026-06-25 的旧 stash，未碰）。
3. **改 harness 核心（subagents/tools/builtins/agents/persistence）后**：除 `make test` 外必跑两裸导入 `import app.gateway` + `make_lead_agent`（守闭环铁律）。
4. **Stage C 改 PROTECTED `engine.py`**：surgical 保 guard，head-to-head diff，别整文件覆盖。
5. **验收守「代码有修复≠现象消除」**：Stage C 完成后 wipe DB→bootstrap→前端提交带 verdict 的 feedback→查 `deerflow.db` feedback 行真有那三列值；`alembic heads` 必须单头。
6. **worktree？** 本会话全程在主仓 dev 做（用户未要求 worktree）。Stage C 是较重操作（删 revision + 改 PROTECTED + wipe DB + 改部署脚本），下个 agent 可考虑建议用户开 worktree 隔离，但**仅在用户明确要 worktree 时才开**。

---

## 五、下一位 Agent 的第一步

1. `git fetch origin dev` + `git rev-list --left-right --count HEAD...origin/dev` 看分叉（dev 并发活跃，本 handoff 写时是 0/0，但可能已变）。若 behind 先调和（rebase 优先，守线性）。
2. 读本 handoff + Stage C spec（`docs/superpowers/specs/2026-06-29-sync-deerflow-stage-c-alembic-bootstrap-deploy.md`）+ 相关 memory（`feedback_deploy_alembic_migration_for_added_columns`、`feedback_local_dev_db_also_needs_manual_alembic_migration_after_sync`、`feedback_sync_full_follow_upstream_infra`）。
3. 实施 Stage C：建议先 **C5 核上游 baseline 覆盖**（决定 C4 要不要补 FK）→ TDD 写 `tests/test_persistence_bootstrap.py`（空库 + 旧库两态，断言终态 feedback 含三列且 rating nullable，watched RED）→ C1-C4 引入+折叠链 → C6 engine.py surgical → C7-b 部署脚本 → C8 wipe 本地 DB 验证 → 全量 + 两裸导入 + `alembic heads` 单头。
4. 完成后：精确路径 commit（拆语义）、更新 `.deerflow-sync-state` 注释（Stage C 从 defer 改为已实施）、问用户是否 push。
