# Handoff: Spec S4 — run_metric_plan 工具实施完成

**日期**: 2026-06-15
**Worktree**: `/home/wangqiuyang/noldus-insight/.claude/worktrees/s4-run-metric-plan`
**Branch**: `feature/s4-run-metric-plan`
**Commit**: `63793162`（已 push）
**PR**: 待用户在 https://github.com/noldus-cn-beijing/noldus-insight/pull/new/feature/s4-run-metric-plan 创建
**Spec**: `docs/superpowers/specs/2026-06-12-s4-code-executor-run-metric-plan-spec.md`

---

## 完成情况

Spec S4 全部 7 个红线 + §4 测试矩阵实施完成。code-executor subagent 的 LLM bash 编排反模式（逐 token 拼 140 行命令 + cat 聚合 + 手构 handoff）替换为确定性 first-party 工具 `run_metric_plan`。

### 产出（13 文件，+1453/-263）

**新建**：
- `tools/builtins/run_metric_plan_tool.py` — 工具薄壳 + `_execute_tasks` 执行器（ProcessPoolExecutor，per-task timeout=120s，on_error policy）+ 调 `aggregate_metrics_to_handoff` + seal 落盘
- `subagents/metric_aggregation.py` — `aggregate_metrics_to_handoff(plan, workspace, *, run_validation=False)` 纯函数 SSOT（auto-seal 与 run_metric_plan 共用）
- `tests/test_run_metric_plan.py` — 17 个测试覆盖 spec §4 矩阵

**修改**：
- `subagents/handoff_schemas.py` — `sealed_by` 加 `"run_plan"` 枚举值
- `subagents/executor.py` — auto-seal code-executor 分支改调 `aggregate_metrics_to_handoff(run_validation=False)`
- `tools/builtins/__init__.py` + `tools/tools.py` — 注册 `run_metric_plan` 进 `BUILTIN_TOOLS`
- `subagents/builtins/code_executor.py` — 收 bash/write_file/str_replace，加 run_metric_plan；prompt 重写（删 bash_constraints/workflow 编排段，加 `<triage>` 失败分诊三类）；max_turns 40→20
- 5 个旧 prompt 契约测试同步更新（`test_code_executor_config/workflow`、`test_ethoinsight_code_skill`、`test_lead_tool_filtering`、`test_subagent_prompt_clarity`）

---

## 关键设计决策（与 spec 的偏差，已核实合理）

### 1. 聚合器签名：`aggregate_metrics_to_handoff(plan, workspace)` 不接 task_results

spec §1.2 草案写 `_aggregate_to_handoff(plan, task_results, workspace)`（吃执行结果）。实际实现为 `aggregate_metrics_to_handoff(plan, workspace)`（**只从磁盘 m_*.json 读，不接 task_results**）。

**理由**：这符合 spec §1.4 更强的约束（"聚合从磁盘真实产物读，不从 plan/task_results 假设推"），且与 auto-seal 字节一致性更易保证（两路都吃 plan+workspace，不依赖执行结果）。代价：聚合靠执行后磁盘产物，不能把执行结果传给聚合器——但这正是 spec 想要的（聚合反映磁盘真相，不反映执行声称）。

### 2. status 派生：纯由磁盘对账 + 失败计数

`_derive_status(agg_status, n_total, n_failed)`：agg_status（completed/partial/failed 来自磁盘 glob 对账）+ `n_failed>0 → failed`。注释里厘清：task 失败通常意味着产物没写出，聚合器 glob 时自然计进 missing → status 降级，所以 `n_failed` 其实已被磁盘对账捕捉；保留它是为兜底"脚本 rc≠0 但产物写成功"的罕见半成功（守 §3 迁移风险 1：响亮失败不静默吞）。

### 3. 测试可注入 runner（绕开 fork/pickle）

`_execute_tasks(tasks, *, runner=None)` 支持注入同步 runner。模块级 `_TASK_RUNNER_OVERRIDE` 钩子供测试 monkeypatch，生产恒 None 走 ProcessPoolExecutor。理由：ProcessPoolExecutor 的 worker pickle 语义（fork 继承 vs spawn 不继承）让 monkeypatch 当前进程的 worker 函数行为不可靠；注入同步 runner 让 on_error/timeout/selector 等**编排逻辑 100% 可测**且不依赖平台 fork 语义。

### 4. SSOT parity 测试的正确对象

spec §4 #6 字面是"run_metric_plan 与 auto-seal 产出字节一致"。实施时发现：正确对比对象是**同一组 m_*.json 调 `aggregate_metrics_to_handoff` 两次结果一致**（纯函数确定性），而非"聚合原始 dict vs 经 CodeExecutorHandoff Pydantic 序列化的 handoff"——中间有 schema 默认值填充层（MetricStat 加 `applicable`/`std` 默认字段），本就不该字节相同。测试断言改为验证聚合纯函数可重复 + per_subject 透传一致。

---

## 验证结果

| 检查项 | 结果 |
|--------|------|
| 裸导入 `import app.gateway` | ✅ 通过（无闭环） |
| 裸导入 `from deerflow.agents import make_lead_agent` | ✅ 通过 |
| `run_metric_plan` 进 BUILTIN_TOOLS | ✅ |
| code-executor tools = `[run_metric_plan, read_file, ls, seal_code_executor_handoff]` | ✅ |
| test_run_metric_plan.py（17 测试） | ✅ 17 passed |
| 回归套件（auto_seal + gateway_no_cycle + code_executor 系列） | ✅ 55 passed |
| 全量回归 | 4007 passed / 6 failed |

### 6 个全量失败全部坐实为基线既有债（零新增回归）

用 `git stash -u` 回到 HEAD 干净态 + 软链主仓库 config.yaml 跑全量，对比确认：
- `test_chart_maker_config::test_chart_maker_config_basic_fields` — 断言 chart-maker `model=="inherit"`，实际 config 是 `deepseek-v4-pro-summary`。**基线单跑也红**，dev 上 config 与测试断言既有不一致，与 S4 无关。
- `test_local_sandbox_provider_mounts` ×3（SymlinkEscapes / DownloadFileMappings / MultipleMounts）— **基线也红**，LocalSandbox 既有 bug，与 S4 无关。
- `test_inspect_gate_guardrail` / `test_paradigm_identification_gate` 的 `test_async_delegates_to_sync` — 单跑全绿，全量跑时被前置测试 state 污染（test isolation 债，同 memory `feedback_known_full_suite_test_pollution_4_tests`）。

**重要教训**：worktree 无 `config.yaml`（在主仓库 `packages/agent/config.yaml`），导致全量跑时 `test_subagent_executor` 一大片（20+）因 `FileNotFoundError: config.yaml` 失败。软链主仓库 config 后这些全转绿——它们是 config 缺失环境债，非代码债。下次在 worktree 跑全量前先软链 config。

---

## 待办（下一步）

1. **用户建 PR**：https://github.com/noldus-cn-beijing/noldus-insight/pull/new/feature/s4-run-metric-plan
2. **PR 合 dev 后复跑 dogfood EPM**（spec §5 最终验证）：code-executor 调一次 run_metric_plan 即完成（不写 bash），预期 13 分钟 → 数分钟内，handoff 链走通（data-analyst 能消费）。
3. **端到端真实脚本验证**（spec §4 #1/#12）：本次单测用 mock runner（绕开 fork/pickle），未跑真实 ethoinsight compute 脚本经 ProcessPoolExecutor。dogfood 是真实验收。
4. **环境债清理**（独立 sprint，不阻塞本 PR）：
   - worktree 缺 config.yaml 致全量假红 → 考虑 conftest fixture 或文档说明
   - chart_maker_config `model=="inherit"` 断言与 config 不符
   - local_sandbox 3 个既有失败
   - gate 测试 isolation 污染

---

## 风险记录

1. **ProcessPoolExecutor 的 DEERFLOW_PATH_* env 继承**：Linux 默认 fork 继承 parent env，已显式 `os.environ.update(_build_path_env(thread_data))` 保险。spawn 模式（macOS/Windows 默认）未测——dogfood 在 Linux ECS 跑，fork 模式。
2. **statistics argv 手构**：PlanStatistics 无 args 字段，手构 `--inputs/--groups/--output`，避开遗留 `input` 字段（指向 handoff_code_executor.json 错误文件）。
3. **deepseek 正面提示**：prompt 全用正面指令（"用 run_metric_plan 执行"非"不要用 bash"），bash 通过收走工具 + disallow 双保险（不靠 prompt 约束）。
