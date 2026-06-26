# 2026-06-26 run_chart_plan 核盘竞态「硬化」=证伪竞态 + 落 fail-loud 自洽性守卫 + 修正 problem doc + commit/push dev

> **给下一个 Agent**：本会话已**完整收尾**——任务做完、TDD 全绿、3 个 commit 已 push 到 `origin/dev`。本文档主要是**留档 + 防误解**，不是「半截活待续」。若你被叫来「继续」，大概率是要做**下一个范式 dogfood（OFT/LDB/Zero Maze）**，而不是回头改 run_chart_plan（它没问题，见 §关键发现）。

---

## 1. 当前任务目标

用户初始诉求：「硬化 `run_chart_plan` 核盘时机（TDD 修那个良性竞态）」——来源是 handoff `docs/handoffs/2026-06/2026-06-24-epm-dogfood-chart-success-askviz-gate-handoff.md` §5 + problem doc `docs/problems/2026-06-24-chart-maker-run-chart-plan-disk-race-false-partial.md`，二者称 chart-maker 渲染 112 张图时「进程池未 flush 就核盘」误判 partial（良性）。

**实际交付**（任务性质被取证改变，见 §关键发现）：
- 证伪「工具核盘竞态」假设；
- 落地一个 **fail-loud 结构守卫**钉死核盘自洽性不变式（防未来回归，非治竞态）；
- 修正 problem doc 的错误根因；
- 把 dev 工作区改动（含本次修复 + 别人遗留物，**排除临时产物**）拆 3 个语义 commit 推 `origin/dev`。

---

## 2. 当前进展（全部 ✅ 已完成并验证）

- ✅ **取证**：`run_chart_plan` **本身无核盘竞态**（详见 §关键发现 #1，这是本会话最重要的结论）。
- ✅ **TDD 红→绿**：
  - 红：`TestReconcileGuardFires` 两条单元测试引 `_TOOL._assert_reconcile_consistent` → `AttributeError` 失败。
  - 绿：实现纯函数 + 接入调用点。
  - 工具级守卫测试 + `TestReconcileInvariantHolds` 四路径回归。
- ✅ **测试全绿**：`make test` = **4690 passed / 27 skipped / 0 failed**（136s）。
- ✅ **裸导入铁律**（改了 `tools/builtins/` 必跑）：`uv run python -c "import app.gateway"` + `from deerflow.agents import make_lead_agent` 双双 0 退出（无导入环）。
- ✅ **lint**：`ruff check` 两改动文件 clean。
- ✅ **problem doc 修正**：§0/§1/§3/§5/§6 全部更正，原错误诊断保留删除线对照。
- ✅ **commit + push**：3 个 commit 已上 `origin/dev`，`git rev-list origin/dev...HEAD = 0 0`（完全同步）。

---

## 3. 关键上下文（项目 + 本次改动）

- **工作目录**：`/home/wangqiuyang/noldus-insight`，**直接在 `dev` 分支主仓**（非 worktree）。
- **改动文件**（commit `8429756e`，我的修复）：
  1. `packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py`（+27）—— 新增纯函数 `_assert_reconcile_consistent(n_total, chart_files, failed_charts)`（放在 `_derive_status` 旁，模块级无 deerflow/ethoinsight import），在 Step 7 核盘算完 `n_rendered/n_failed` 后、`_derive_status` 前调用（标 `Step 7.1`）。违反 `len(chart_files)+len(failed_charts)==n_total` 即 fail-loud：`logger.error` + `_seal_failed(...)` 落 failed handoff + `return _error_command(tool_call_id, "reconcile_inconsistent", str(e))`。
  2. `packages/agent/backend/tests/test_run_chart_plan.py`（+131）—— `TestReconcileGuardFires`（守卫单元 mismatch→raise / match→pass + 工具级 monkeypatch 守卫抛错断 `error_code=reconcile_inconsistent`、无 artifacts、磁盘 failed handoff）；`TestReconcileInvariantHolds`（all-success-112 / partial / rc0-no-png / abort 四路径，双向断 `n_rendered+n_failed==n_total`、`n_failed==len(failures)==len(handoff.failed_charts)`、`gate_signals.failed_charts==errors_count==n_failed`）。
  3. `docs/problems/2026-06-24-chart-maker-run-chart-plan-disk-race-false-partial.md`（未跟踪→本次首次纳入 git，commit `93c52ed9`）。
- **测试驱动方式**：测试经 `importlib.spec_from_file_location` 直接加载 worktree 源（绕 conftest mock）；注入同步 `_TASK_RUNNER_OVERRIDE` 绕 ProcessPoolExecutor。所以**单测无法复现真实多进程竞态**——回归靶是「不变式」不是「假竞态」（这是把 problem doc 原提议 #3 调整的原因）。
- **测试命令**：`cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_run_chart_plan.py -v`（注意：`python` 不在 PATH，必须 `uv run python`）。

---

## 4. 关键发现（最重要——防下一个 Agent 重蹈覆辙）

### #1 `run_chart_plan` 工具本身没有核盘竞态（problem doc 原根因错误）

- `_execute_tasks`（`run_chart_plan_tool.py:~423-455`）对**每个** future 调 `fut.result(timeout=)`（阻塞到该 worker 的 `main()` 返回，含 `savefig` 完成关闭文件），再在 `finally` 里 `pool.shutdown(wait=True, cancel_futures=True)`。worker `_run_chart_task` 进程内同步调 `mod.main(args)`。**核盘 `Path(output_real).exists()`（:~256-279）发生在这之后，png 必已落盘。**
- `git log -S "pool.shutdown(wait=True"` 证明该 wait 在**首版 commit `302a7046`（2026-06-24）就有**——正是 problem doc 引用的 commit。所以 doc 提议 #1「加 shutdown wait」**从第一天起就是 no-op**。
- handoff/problem doc 里的 `n_rendered=4 / 108 failed` 是 **chart-maker 子代理自己**在思考里 mid-flight `ls outputs/*.png`（工具返回前）看到的瞬时值，**从未进过任何 handoff 字段**。工具确定性计数从来正确（最终 handoff `chart_files=112 / completed` 即铁证）。

### #2 自洽性不变式在当前代码已结构成立，守卫纯防未来回归

核盘循环每 idx 只 append `chart_files` 或 `failed_charts` 之一（无 continue/break/双 append），`n_failed = n_total - n_rendered` 由减法导出。故 `n_rendered+n_failed==n_total` 与 `n_failed==len(failed_charts)` 恒成立。doc 担心的 `n_failed≠len(failures)` 裂缝（[silent-drop-completed] 同款）在本工具已被 M2/R2「按 task index 核盘」修复堵死。**守卫只防未来重构（如重新按 chart_id keying）静默重引入该裂缝**——这符合 CLAUDE.md「用确定性结构约束行为」。

### #3 不改 prompt（守三大病理自检）

真实症状层（子代理 mid-flight ls）**没改 prompt**：CLAUDE.md 三大病理自检 §3/§6.6——反复误判不靠加提醒规则修，且子代理本次已正确克制走 seal、未污染产物。结构正解=让确定性工具自洽性 bulletproof + 修文档。

---

## 5. 本会话的 git 操作（commit/push 留档）

推上 `origin/dev` 的 3 个语义 commit（`1c1033a7..93c52ed9`）：

| commit | 内容 | 是否本任务 |
|---|---|---|
| `8429756e` | run_chart_plan 核盘自洽性 fail-loud 守卫 + 修正 disk-race problem doc | ✅ 本任务核心 |
| `a5fc2b8d` | deploy 链补 alembic 迁移（`run-db-migrations.sh` 新增）+ 前端 CLAUDE.md 收窄 ui/ 可编辑性 | ❌ 别人遗留，代提 |
| `93c52ed9` | 归档会话文档（handoffs/specs/plans/problems/milestone） | ❌ 别人遗留，代提 |

- 全程 `git add <精确路径>`，**未用 `git add -A`**。
- **排除未提交的临时产物**（用户明确选择留工作区，勿误提）：`recon.js`、`scripts/repro/`、`reports/`、`docs/reports/`。
- **直接 push 到 dev**（用户要求），**绕过了** CLAUDE.md「feature 完成从 dev 提 PR 到 main」流程——只进 dev，没碰 main、未触发生产镜像构建。

---

## 6. 未完成事项（本任务 100% 完成；以下是项目层面的下一步候选）

按 handoff §8 + CLAUDE.md milestone：

1. **【高】OFT/LDB/Zero Maze dogfood**：EPM 已绿（2026-06-24），按 `/noldus-insight-e2e <data-dir>` 跑剩 3 个范式，验证 chart catalog 覆盖。这是「继续」最可能的指向。
2. **【低】临时产物清理**：若不想让 `recon.js` / `scripts/repro/` / `reports/` / `docs/reports/` 留在工作区，可加 `.gitignore` 或删除（用户尚未决定）。
3. **【低】a5fc2b8d 的 deploy 迁移真实性验证**：`run-db-migrations.sh` + deploy-via-tar.sh 改动是别人遗留代提的，**本会话未实跑验证**。若下次 deploy，注意核实该迁移步骤行为（与 memory `feedback_deploy_alembic_migration_for_added_columns` 一致）。

---

## 7. 风险与注意事项

- ⚠️ **别再回头「修 run_chart_plan 的竞态」**——它没竞态（§4 #1）。守卫已落地，problem doc 已更正。
- ⚠️ **别误提临时产物**：4 项 `??` 是用户主动选择留工作区的。
- ⚠️ **`python` 不在 PATH**：跑后端任何脚本/导入用 `uv run python`（在 `packages/agent/backend` 下 + `PYTHONPATH=.`）。
- ⚠️ **dev 工作区可能仍有别人新落的改动**：本会话只处理了当时快照。接手前先 `git status --short` + `git fetch origin dev` 对齐。
- ⚠️ **改 `tools/builtins/` / `subagents/` / `agents/` 核心后必跑两裸导入**（CLAUDE.md 导入环铁律），conftest mock 会让 pytest 假绿。

---

## 8. 下一位 Agent 的第一步建议

1. `cd /home/wangqiuyang/noldus-insight && git status --short && git log --oneline -3` —— 确认起点（应见 `93c52ed9` 在顶，4 项临时产物 `??`）。
2. 若用户要「继续」且未指明 → 默认是 **OFT/LDB/Zero Maze dogfood**，读 `docs/milestone/fst-e2e-7fixes-askviz-intent.md` 的遗留项 + 用 `/noldus-insight-e2e` skill。
3. 若用户问「run_chart_plan 还要不要改」→ 答**不用**，引用本文 §4 #1 + commit `8429756e`。
4. 若要复看本次修复：`packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py` 搜 `_assert_reconcile_consistent` / `reconcile_inconsistent`。
