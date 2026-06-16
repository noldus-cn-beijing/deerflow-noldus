# Handoff: dogfood 四份 spec（S1–S4）全部 review + review-fix + 合入 dev

> 日期：2026-06-15
> 上一会话产出：按序 review S2/S3/S4 三份 spec 的实施 → 各自发现并修复问题（review-fix）→ 四份 PR 全部合入 dev
> 交接对象：下一个接手的 AI Agent
> 关联起点：`docs/handoffs/2026-06/2026-06-12-dogfood-diagnosis-four-specs-handoff.md`（四份 spec 的诊断+拆分起点）

---

## 1. 当前状态速览（全部闭环）

| spec | 主题 | PR | 状态 |
|---|---|---|---|
| **S1** | loop_detection 配置接线 + inspect override | #126 | ✅ 已合 dev（上一会话，含 review-fix） |
| **S2** | identify_ev19_template 返回 per_file_grouping | #128 | ✅ **review + review-fix + 合 dev** |
| **S3** | handoff schema fail-open 防护 + calamine 启动断言 | #129 | ✅ **review + review-fix + 合 dev** |
| **S4** | code-executor run_metric_plan 工具（最大） | #130 | ✅ **review + review-fix + 合 dev** |

**`origin/dev` 当前 HEAD：`d0a7c920`**（线性历史：`a728881d`(S1) → `a60f8957`(S2 #128) → `eb0020c9`(S3 #129) → `d0a7c920`(S4 #130)）。

四份 feature 分支均基于 `a728881d` 独立拉出、独立 review-fix、独立 PR 合入。S2 远程分支 merge 后已删（正常），S3/S4 远程分支仍在。**本批 spec 已完全闭环，无遗留待实施项。**

---

## 2. 每份 spec 的 review 发现与修复（review-fix commit 都已在 dev）

### S2（#128）— review-fix `37bd91b4`
- **发现 1（已修）**：抽共享 `_ev19_grouping.extract_grouping_fields` 时，把 Animal ID 变体折进单循环，**丢了原 inspect `_extract_grouping_fields` 的 `break` 语义**（多别名时旧版只取首个 Animal ID，新版会全收）——重构引入行为漂移。修复=拆 `_ANIMAL_ID_KEYS` 独立第二循环 + break，与原函数 8 边界 case 字节等价，保留 None 防御增强。
- **发现 2（已修）**：新测试文件 `import pytest` 未使用（ruff F401）。删。
- 补 `test_shared_extract_animal_id_takes_first_variant_only` 锁定 break 语义（中英文多变体 + 混合分组字段）。
- 核心实现（identify Step 3.5 遍历提分组、prompt 优先用 per_file_grouping、只读 parse_header 不读 trajectory、SSOT 抽共享）本身忠实，红线全守；红→绿基线为真（parent 无该字段，实测可证）。

### S3（#129）— review-fix `6ceafe5a`
- **阻塞发现（已修）**：Part A 的「completed → 核心产物非空」validator 加到四个共享 handoff schema，**打挂了 9 个测试文件里 23 个构造"空核心 completed handoff"的既有用例**，实施只适配了 4 个（典型「改共享 schema 必须跑全量」教训，见 memory `feedback_pr_merge_must_run_full_suite_on_shared_logic`）。修复=给那些只测别的东西（config_id/path 防污染/strict mode/manifest/memory fact/参数审计）的 completed fixture 补核心字段；ChartMaker「empty chart_files ok」与 seal 工具契约对齐（全失败时为 failed/partial，非 completed），改成对 failed/partial 断言空合法。**validator 实现本身不动**——chart_files/key_findings/metrics_summary 非空契约与各 seal 工具 docstring 一致。
- **次要发现（已确认接受）**：spec A1 表格写 ReportWriter 是 `sections_written 非空（或 report_path 有值）`，实施丢了 `or report_path`。核实后**保留实施者的"严 sections_written"选择**（spec 的 `or report_path` 因 report_path 必填会让 validator 形同虚设；auto-seal 路径已兜底 sections 非空，不受影响）。已在 commit message 记录此有意偏离。
- **可移植性小修**：calamine 子进程测试硬编码绝对 cwd → 改从 `__file__` 推导包根（换机器/CI 可用）。
- Part B（`parse/__init__.py` 顶部 `import python_calamine` 断言）忠实——import 名是 `python_calamine` 不是 `calamine`（实测坐实），放 parse 不放 ethoinsight 根。子进程隔离测试设计正确。

### S4（#130，最大）— review-fix `16348235`
核心实现（run_metric_plan 工具 / ProcessPoolExecutor 进程内调脚本 / `metric_aggregation.aggregate_metrics_to_handoff` 纯函数 SSOT / auto-seal 改调它 / code_executor 收 bash + prompt 重写 / 工具三处注册 / guardrail 不误拦 / 裸导入无环）**全部忠实，7 条红线全守**。review 发现三处工程瑕疵，已修：
- **发现 1（最重要，已修）**：原 Step 4 直接 `os.environ.update(DEERFLOW_PATH_*)` **永久 mutate 父进程 env**——多线程 Gateway 下一个 thread 的 workspace 路径会泄漏成全局污染并发跑的其他 thread。修复=compute 池 worker 经 `ProcessPoolExecutor(initializer=_worker_init, initargs=(path_env,))` 在子进程内设 env；statistics（父进程内跑）用新增 `_scoped_path_env` 上下文管理器临时设 + 退出即还原。父进程 os.environ 调用前后不变（`TestEnvNotGloballeyPolluted` 锁定）。
- **发现 2（已修）**：`on_error="abort"` 原实现一次性 submit 全部再 cancel-in-loop（已运行的 cancel 无效）。改为遇首个失败即 cancel 后续未启动 future + break + `shutdown(cancel_futures=True)`。**仍按提交序逐个 `result(timeout=)`**（每 task 独立超时预算）——特意不用 `as_completed`（挂死 task 永不被 yield 会卡死整轮 wait，中途试过又退回）。
- **发现 3（已修）**：测试 #6（SSOT parity）名不副实——原只断言 `direct1==direct2`（纯函数确定性），不证语义。补 `test_aggregator_output_matches_expected_payload` 逐字段钉死 metrics_summary/per_subject/status/完整性。顺手清 F841。
- **行为等价的真凭据**：`aggregate_metrics_to_handoff` 是对 parent `executor.py:401-559` 原 auto-seal 聚合的忠实抽取（neutral 措辞由 auto-seal 调用方 re-wrap，payload 字节一致）；28 个既有 auto-seal 测试在重构后全绿——这是真正的「抽出无行为变化」证明（不是测试 #6 那条弱断言）。

---

## 3. 跨三次 review 的统一方法学（值得下一个 reviewer 复用）

1. **红→绿基线必须实证**：每份都核实了 parent commit 上「改前真的红/真的缺字段」，不信空壳 baseline（守 PR#115 教训 `feedback_pr115_stage1_equivalence_baseline_is_hollow_error_string`）。
2. **全量回归 + 逐失败归因**：每次跑全量后端，把失败逐一坐实为「基线环境债」还是「本次引入」。**决定性手法**=把本次源改动 stash 回退后重跑，或在 parent 主 checkout 同 venv 重跑——同样红=基线。
3. **被改的既有测试逐个判「适配 vs 削弱」**：S3 的 23 个 + S4 的 5 个改动测试，全部确认是「补核心字段/契约反转对齐」而非削弱断言（如 S4 `test_code_executor_still_has_bash` 正确反转成 `uses_run_metric_plan_not_bash`，因为契约就是反的）。
4. **改 subagents/tools/agents 核心后必跑裸导入两入口**（S2/S3/S4 都跑了 `import app.gateway` + `make_lead_agent`）——conftest mock 藏导入环，pytest 假绿。S4 新工具的所有 cycle-risk import 都放函数体内惰性 import，裸导入证实无环。

---

## 4. 已知基线债（看到这些红别归因后续改动）

全量后端回归稳定出现 **~24 failed**，**全是与本批 spec 无关的环境债**，已多次在 parent 复现坐实：
- `test_subagent_executor` ×15 — `ModuleNotFoundError: 'deerflow.agents.middlewares' is not a package`（本机 venv editable-install 的 namespace 解析债，**import-order 相关、跨 run 不稳定**——某些 run 全红某些 run 不出现，别当 flaky 归因自己）。
- `test_local_sandbox_provider_mounts` ×3、`test_lead_agent_training_middleware` ×2、`test_chart_maker_config` ×1、`test_lead_agent_model_resolution` ×1。
- `test_inspect_gate_guardrail` / `test_paradigm_identification_gate` 的 `test_async_delegates_to_sync` — **单跑就过、full-suite 才红**的测试隔离污染（memory `feedback_known_full_suite_test_pollution_4_tests`）。

判据：用 parent 主 checkout 同 venv 跑同一批，同样红 = 基线。这批是测试卫生 debt，待单独修，不阻塞合入。

---

## 5. 环境/工具注意事项（本批踩过的坑）

1. **worktree 共享 dev 分支的坑（S2 实证，已纠正）**：之前的实施 agent 用 `git worktree add <path> dev` 建 S2 worktree → worktree 和主仓库**共享同一个 dev 分支**，在里面 commit 直接动 dev。本会话发现时 S2 commit 已成游离 commit（dev 被 reset 回 parent）。补救=新建独立 `feat/s2-...` 分支收纳游离 commit。**下一个 agent 建 worktree 务必 `git worktree add -b feat/sX-xxx <path> <base>` 一步建独立分支，别共享 dev。**（S3/S4 都已是独立 feature 分支，无此问题。）
2. **worktree 无 venv**：四个 worktree 都没有 `.venv`。review 跑测试用**主 checkout 的 venv 解释器 + `PYTHONPATH=<worktree>/packages/harness:<worktree>/packages/ethoinsight:<worktree>/backend`**——`PYTHONPATH` 前置能覆盖 editable-install 的 `.pth` finder，把 `deerflow`/`ethoinsight` 解析到 worktree（实测可行；注意 cwd 优先级斗不过 editable finder，必须显式 PYTHONPATH）。
3. **run_metric_plan 的 env 隔离设计（S4 新引入，记牢）**：worker 经 `ProcessPoolExecutor(initializer=_worker_init)` 拿 `DEERFLOW_PATH_*`，父进程内的 statistics 经 `_scoped_path_env` 临时拿——**这是「dev/prod 行为一致 + 不在多线程 Gateway 污染全局 env」的样板**。后续若给其他 first-party 工具加进程池执行，照此模式，别 `os.environ.update` 全局。
4. **两个已存在的 S4 实施期 handoff 已被超越**：`docs/handoffs/2026-06/2026-06-15-s4-run-metric-plan-handoff.md`（进行中）和 `-completed-handoff.md`（停在 commit `63793162`、"PR 待创建"、"17 测试"）是 S4 实施 agent 写的、**未含本次 review + review-fix `16348235` + 四份全合 dev**。两者均 untracked（未入库）。**以本 handoff 为准**；那两个保留作实施记录即可。

---

## 6. 下一个 Agent 的第一步建议

**若用户开新 dogfood/分析任务**：本批四份 spec 已闭环（全合 dev `d0a7c920`），无需回头。建议复跑一次 EPM/OFT dogfood 验证 S2（一次 identify 拿全分组不逐个 inspect）+ S4（code-executor 调一次 run_metric_plan，13 分钟→数分钟）的端到端效果——这是四份 spec 的最终验收，本会话只做了单测/回归层验证，**未跑真实 dogfood**。

**若用户问某份 spec 细节**：四份 spec 自包含在 `docs/superpowers/specs/2026-06-12-s{1,2,3,4}-*.md`；架构 why（尤其 S4 五轮否决记录）在 S4 spec §0.1。

**关于基线测试债**：`test_subagent_executor` namespace 债 + 4 个隔离污染测试值得单开一个清扫任务（与任何 feature 正交），让全量回归回到真·全绿，否则每次 review 都要重新归因这 ~24 个。

---

## 关键文件/commit 速查
- 四份 spec：`docs/superpowers/specs/2026-06-12-s{1,2,3,4}-*.md`
- review-fix commit：S2 `37bd91b4` / S3 `6ceafe5a` / S4 `16348235`（都在 dev `d0a7c920`）
- S4 新增核心：`tools/builtins/run_metric_plan_tool.py`（工具+执行器+env 隔离）、`subagents/metric_aggregation.py`（SSOT 聚合纯函数）
- 起点 handoff：`2026-06-12-dogfood-diagnosis-four-specs-handoff.md`
