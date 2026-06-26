# Spec：文件路径可靠性承重墙收敛 — 让"路径类 bug"结构性消失

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-26
> 代码基线：dev HEAD（写时 `776450b2`，实施前 `git show HEAD:` 核对）
> 性质：🔴 高价值 / 中风险 · 后端 deerflow harness + persistence + ethoinsight 工具
> 立意：**承重墙级收敛**，不是逐个打补丁。目标是让"产物落在哪、谁能读、DB 与磁盘是否一致"这一类反复出现在历史 spec/handoff 里的 bug **结构性消失**。

---

## 〇、为什么写这份 spec（problem 全景）

用户原话：**"我们之前无数个 spec 和 handoff 很多都是因为实际文件存储的路径引起的，这也是为什么我想着重修复这点、增强可靠性。"**

回看历史 memory/handoff，路径类 bug 反复出现且家族庞大：
- `run_metric_plan` argv 虚拟路径谁解析（铁律 11，#192）
- 进程内跑脚本全 `FileNotFoundError`＝脚本不调 `resolve_sandbox_path`
- `validate_catalog` `result_file_unreadable` 误报（虚拟路径解析）
- `present_files` 幻影文件名（`epm_bar_{metric}.png`）污染 artifacts 致前端 404
- "113 张图只显示 1 张"（artifacts 不上行 + 取图路径双轨）
- chart 没出图判别要看 `plan_charts.json skipped[]` + `outputs/` png

这些**不是同一个 bug**，但**同一个病根**：**路径的产生、解析、隔离、登记没有被收敛成"承重墙单点 + 确定性约束"，而是散落在各工具里靠惯例对应**。符合 CLAUDE.md 核心原则的反面——"用 prompt/惯例约束行为"而非"确定性结构定生死"。

**本 spec 的立意**：把路径可靠性做成承重墙——
1. **解析单点**：所有虚拟↔真实路径翻译只走 DeerFlow 已有的 `Paths.resolve_virtual_path`，禁止散落的字符串拼接。
2. **隔离边界显式化**：thread / run 产物的隔离从"命名惯例"升级为"结构约束 + 校验"。
3. **DB↔磁盘一致性兜底**：删除/创建的一致性从"应用层三步靠顺序"升级为"事务 + 外键级联（DeerFlow 有就用，没有补迁移）"。

---

## 一、先查 DeerFlow：哪些机制已存在（必读，避免重造轮子）

> 原则（用户拍板）：**先查 DeerFlow infra 有无对照，有就复用/跟随，没有才自己加；自己加的尽量做成可回贡献上游的形态。**

实读坐实，DeerFlow **已经有相当强的路径承重墙**，薄弱点多是"机制在但没强制用全"，不是"机制缺失"：

| 能力 | DeerFlow 现状 | 证据 | 结论 |
|---|---|---|---|
| **路径解析单点 + path-traversal 守卫** | ✅ 已有且健壮 | `config/paths.py:351` `resolve_virtual_path`：段边界精确匹配防前缀混淆 + `relative_to(base)` 防穿越 + user-scoped | **复用，禁止绕过** |
| **per-thread / per-user 隔离** | ✅ 已有 | `paths.py:222 thread_dir`、`:176 user_dir`、`:285 shared_dir` 全带 `user_id` | **复用** |
| **legacy unsafe-id 迁移** | ✅ 已有 | `paths.py:181-203` `user_dir` 自动把 legacy 摘要桶 rename 到 SHA-256 新格式 + `logger.info` 审计 | **已解决"legacy 无审计"——薄弱点③大半是误报** |
| **目录创建/删除** | ✅ 已有 | `paths.py:319 ensure_thread_dirs`、`:342 delete_thread_dir` | **复用** |
| **alembic 迁移框架** | ✅ 已有 | `persistence/migrations/versions/`（3 个迁移在册） | **加表/约束走这里**（注意部署链不跑 alembic，见 §四） |
| **DB 外键约束** | ❌ 无 | `run/model.py:17` thread_id 仅 `index=True` 无 `ForeignKey`；`thread_meta/model.py` 同 | **DeerFlow 也没有 → 自己补，做成可回贡献形态** |
| **artifact 按 run 隔离** | ❌ 无 | `thread_state.py:62 merge_artifacts` 去重键＝path（`_artifact_path`），无 run_id | **自己补** |

**收敛后的判断**（哪些真做、哪些是误报）：

| 薄弱点 | 判定 | 处置 |
|---|---|---|
| ① DB↔文件无约束绑定 | **真，P0** | 加外键级联 + 删除事务化（DeerFlow 无，自己补） |
| ② run_id 与产物文件名混淆 | **真，P1**（"113 图"家族病根之一） | artifact 去重键加 run 维度（自己补） |
| ③ legacy 无 user 路径无审计 | **大半误报**：`user_dir` 已有迁移+审计日志 | 仅补一条"启动时统计 legacy thread 数"观测，不动路径逻辑 |
| ④ shared 路径跨线程缺校验 | **半真**：`resolve_virtual_path` 已防穿越，但 handoff 读取处未显式校验父目录归属 | 加显式校验（轻量，自己补） |

---

## 二、任务（按风险/价值，全部 TDD 强制）

### 任务 1（P0）：DB↔磁盘一致性 — 外键级联 + 删除事务化

**问题坐实**：删 thread 现在是三步独立操作（`threads.py` 的 `_delete_thread_data` 删文件 / `checkpointer.adelete_thread` 删检查点 / `thread_store.delete` 删 meta），任一步失败留下孤儿：meta 行在但磁盘没了→读 404；或 meta 删了但 `runs` 行还指向不存在的 thread。`runs.thread_id` 无外键（`run/model.py:17`）。

**修法**：
1. **新增 alembic 迁移**（仿 `migrations/versions/20260622_1700_*.py` 模板）：给 `runs.thread_id` 加 `FOREIGN KEY REFERENCES threads_meta(thread_id) ON DELETE CASCADE`。SQLite 需用 batch 模式（`op.batch_alter_table`）。`run_events` 若有 run_id 外链同理级联到 `runs.run_id`。
2. **删除路径事务化**：在 thread 删除入口，把"删 runs 行 → 删 meta → 删检查点 → 删磁盘目录"包进单一逻辑顺序：**先 DB（一个事务内）后磁盘**，DB 事务失败则磁盘不动；磁盘删失败记 error 但不回滚 DB（磁盘孤儿可被后续 GC 清，比 DB 孤儿安全）。
3. **观测**：删除完成 emit 一条结构化日志（thread_id / user_id / runs_deleted_count / dir_existed）。

**受保护文件影响**：`persistence/run/model.py`、`thread_meta/model.py` **非受保护**（可改）；删除入口在 `app/gateway/routers/threads.py`（gateway app 层，非 sync 面，可改）。迁移是新文件。

**TDD**：
- `test_delete_thread_cascades_runs`：建 thread+2 runs，删 thread，断言 runs 表对应行归零。
- `test_delete_thread_db_fail_keeps_disk`：mock DB 事务抛错，断言磁盘目录未被删（不产生"DB 在磁盘没"的反向孤儿）。
- 既有 thread 删除测试全绿（grep 调用方，seesaw 防回归）。

---

### 任务 2（P1）：run_id 维度的 artifact 隔离 —"113 图"家族病根收敛

**问题坐实**：`thread_state.py:62 merge_artifacts` 按 path 去重（`_artifact_path`），同一 thread 内多 run 若产同名 `plot_<chart_id>.png`（chart_id 来自 catalog 固定、与 run 无关），**后一 run 覆盖前一 run 的产物**，前端 artifacts 列表不变但磁盘内容被换 → 用户回看旧 run 拿到新数据；或分批 present 互相覆盖致"只显示 1 张"。

**修法（择一，实施 agent 按改动面定，优先 A）**：
- **A（轻量、不改 schema 承重墙）**：artifact 元数据带 `run_id` 字段，`merge_artifacts` 去重键改为 `(run_id, path)`。同 run 同 path 覆盖（正确），跨 run 同 chart_id 不互覆盖。`thread_state.py` 是受保护文件 → **surgical**：只在 `merge_artifacts` 加 run 维度，不动其它 reducer 语义。
- **B（更彻底，评估后决定是否本 sprint）**：产物落盘路径按 run 分目录 `outputs/<run_id>/plot_*.png`。**风险**：碰 `present_files` / `run_chart_plan` / 前端取图端点（`/artifacts/charts`），改动面大，且与"对话流图廊空"那批前端 spec 有交集 → **本 sprint 默认只做 A，B 记入后续**。

**受保护文件影响**：`thread_state.py` 受保护 → surgical（仅扩 reducer）。`present_file_tool.py` / `run_chart_plan_tool.py` 非受保护。

**TDD**：
- `test_artifacts_not_overwritten_across_runs`：同 chart_id 两个 run 各 present，断言 artifacts 含两条（按 run 区分）、磁盘两份都在。
- `test_same_run_same_path_dedup`：同 run 同 path 仍去重为一条（不回归）。

---

### 任务 3（P1）：路径解析单点强制 — 禁止散落字符串拼接

**问题坐实**：历史路径 bug 大量来自工具/脚本各自拼 `/mnt/...` 字符串或假设脚本自己 resolve（铁律 11、`FileNotFoundError` 家族）。DeerFlow 已有 `resolve_virtual_path` 单点，但没有"禁止绕过"的护栏。

**修法（结构约束，不加 prompt）**：
1. **审计 grep**：扫 `tools/builtins/`、`ethoinsight/scripts/`、`sandbox/` 下所有 hardcoded `/mnt/` 字符串拼接，列出未走 `resolve_virtual_path` / `resolve_sandbox_path` 的点。
2. 把这些点改为统一走解析单点（`Paths.resolve_virtual_path` 或 ethoinsight `_cli.resolve_sandbox_path`，二者本就配套）。
3. **加一条常驻测试守护**：`test_no_raw_mnt_path_concatenation`——grep 关键源目录，断言无新增的裸 `/mnt/` 拼接（白名单已知合法引用），把"别手拼路径"从惯例变成 CI 可抓的约束。

**受保护文件影响**：多为非受保护工具文件；若触及 `sandbox/tools.py`（受保护）则 surgical。

**TDD**：上述守护测试 + 被改工具的解析正确性回归。

---

### 任务 4（P2）：shared 路径 handoff 读取显式归属校验

**问题坐实**：`resolve_virtual_path` 已防路径穿越，但 subagent handoff 读取处（executor / seal）未显式断言 handoff 文件父目录 == 当前 thread 的 `shared_path`/`workspace_path`。当前靠 middleware 注入路径隐式隔离，**未来改动易遗漏**。

**修法**：在 handoff 读取单点加显式校验：读取前 `assert resolved.is_relative_to(thread_data.shared_path 或 workspace_path)`，失败 fail-loud（记 error + 拒读），不静默。`subagents/executor.py`、`task_tool.py` 受保护 → surgical（仅加校验断言，不动控制流）。

**TDD**：`test_handoff_rejects_foreign_thread_path`：构造指向他 thread 的路径，断言被拒。

---

### 任务 5（P2）：legacy thread 观测（澄清误报，最小动作）

**澄清**：调研一度判 legacy 路径"无审计"，**实读发现 `user_dir` 已有迁移 + `logger.info` 审计（paths.py:181-203）**。所以**不动路径逻辑**，仅补一条启动观测：统计 `threads_meta.user_id IS NULL` 行数并日志告警（便于运维判断是否还有老部署遗留）。`config/paths.py` 受保护 → 若要加放在 gateway 启动钩子，不碰 paths.py。

---

## 三、不做 / 边界

- **不重写 `Paths` / `resolve_virtual_path`**——DeerFlow 已健壮，本 spec 是"强制用全 + 补 DB 兜底 + 补 run 隔离"，不动承重墙本体。
- **不做 artifact 落盘按 run 分目录（任务 2 的 B 方案）**——改动面碰前端取图链，与对话流图廊那批 spec 交集，留后续单独评估。
- **不动 agentic 上下文管理 / session tree**（blog 思想）——那是 v1.0 context-engineering 方向，见 memory `reference_agentic_context_engineering_session_tree_v1_direction`，与磁盘路径正交。
- **不引入 experiment 层**——那是配套的新 feature，见 `2026-06-26-experiment-cross-paradigm-comparison-init-design.md`；但**本 spec 的路径承重墙是它的地基**（experiment 提取 thread 文件依赖路径解析单点可靠）。

---

## 四、验证

1. `cd packages/agent/backend && make test`（全绿）+ **裸导入两生产入口**（改了 persistence/executor 等核心，防闭环假绿）：
   ```bash
   PYTHONPATH=. python -c "import app.gateway"
   PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
   ```
2. **迁移真跑**（部署链不跑 alembic！）：本地库 `alembic upgrade head` 验证外键建成；生产/dev 库需手动迁移（守 memory `feedback_deploy_alembic_migration_for_added_columns` + `feedback_local_dev_db_also_needs_manual_alembic_migration_after_sync`）。SQLite 外键需确认 `PRAGMA foreign_keys=ON`（连接级，检查 engine 配置）。
3. **改共享逻辑跑全量 + grep 调用方**（守 `feedback_pr_merge_must_run_full_suite_on_shared_logic`）：任务 1/2 动 persistence + reducer 是共享组件。
4. **dogfood 坐实**（守"代码有修复≠现象消除"）：多 run 同 chart 场景实跑，确认前端 artifacts 不再互覆盖；可用 `/noldus-insight-e2e`。
5. **受保护文件 surgical 自检**：`thread_state.py` / `executor.py` / `task_tool.py` / `paths.py` 的改动逐行确认只加约束不删 Noldus 定制（守 sync 铁律）。

---

## 五、HarnessX 三大病理自检（改 harness 前必过）

- **Reward hacking**：本 spec 加的是确定性 DB 约束 + 路径校验 + 测试守护，验收看真产物（迁移真跑、dogfood 真覆盖），不靠 LLM 自述。✅
- **Catastrophic forgetting**：改 `merge_artifacts`（共享 reducer）、persistence 模型——**先 grep 所有消费者全量回归**（seesaw）。任务 3 的"禁裸拼路径"守护测试本身就是防遗忘的护栏。✅
- **Under-exploration**：本 spec **正是从"反复改 prompt/惯例"转向结构约束**（外键、去重键、解析单点护栏）——是结构改动不是打地鼠。但注意任务 5 是反向自检结果（结构已对、只缺观测，故不上结构门只加日志）。✅

---

*依据：实读坐实 DeerFlow 已有 `Paths.resolve_virtual_path` 路径承重墙（段边界匹配+穿越守卫+user-scoped+legacy 迁移），薄弱点是"机制未强制用全 + 无 DB 外键兜底 + artifact 无 run 隔离"，非机制缺失。先查 DeerFlow 复用、缺的自己补并做成可回贡献形态（用户原则）。立意＝把路径类 bug 从"惯例对应"收敛为"承重墙单点+确定性约束"，让该家族 bug 结构性消失。*
