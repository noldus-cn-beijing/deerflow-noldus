# Handoff: Spec S4 — run_metric_plan Tool 实施（进行中）

**日期**: 2026-06-15  
**Worktree**: `/home/wangqiuyang/noldus-insight/.claude/worktrees/s4-run-metric-plan`  
**Branch**: `feature/s4-run-metric-plan`  
**Spec**: `docs/superpowers/specs/2026-06-12-s4-code-executor-run-metric-plan-spec.md`

---

## 当前任务目标

用确定性的 `run_metric_plan` 第一方工具替换 code-executor subagent 里的 bash LLM 编排，走 `ProcessPoolExecutor` 在进程内执行脚本（无 subprocess 冷启动开销），并通过 SSOT 聚合函数统一 run_metric_plan 和 auto-seal 两个调用路径。

**预期产出**：
- 新工具 `run_metric_plan`（`tools/builtins/run_metric_plan_tool.py`）
- 注册到 `tools/builtins/__init__.py`
- `code_executor.py` 重写（删 bash/write_file/str_replace，加 run_metric_plan）
- 测试套 `tests/test_run_metric_plan.py`
- 推送到 `feature/s4-run-metric-plan`

---

## 当前进展

### ✅ 已完成

1. **`handoff_schemas.py`**：`sealed_by` Literal 扩展加入 `"run_plan"`
   - 文件：`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`
   - 状态：已修改，未 commit（worktree `git status` 显示 `M`）

2. **`metric_aggregation.py`（新建）**：纯函数 SSOT 聚合模块
   - 文件：`packages/agent/backend/packages/harness/deerflow/subagents/metric_aggregation.py`
   - 状态：新建，未 commit（worktree `git status` 显示 `??`）
   - 包含两个函数：
     - `_collect_validation_warnings(plan, workspace)` — L-A+L-B 校验，返回 DataQualityWarning-shaped dicts
     - `aggregate_metrics_to_handoff(plan, workspace, *, run_validation=False)` — 读 m_*.json + groups.json，生成 handoff payload
   - `run_validation=False` 供 auto-seal（保持字节一致）；`run_validation=True` 供 run_metric_plan（含校验）

3. **`executor.py`** 重构 auto-seal 分支
   - `_attempt_auto_seal_from_artifacts` 的 code-executor 分支改为调用 `aggregate_metrics_to_handoff`（惰性 import）
   - 所有 27 个 `test_auto_seal_from_artifacts.py` 测试通过 ✅

### ❌ 未完成

4. `tools/builtins/run_metric_plan_tool.py` — **尚未创建**（这是下一步）
5. `tools/builtins/__init__.py` — 未注册 `run_metric_plan`
6. `subagents/builtins/code_executor.py` — 未改写（仍有 bash/write_file/str_replace）
7. `tests/test_run_metric_plan.py` — 未写
8. Commit + push

---

## 关键上下文

### Worktree 路径

所有文件在 `/home/wangqiuyang/noldus-insight/.claude/worktrees/s4-run-metric-plan/` 下，加前缀 `packages/agent/backend/packages/harness/deerflow/`。

### Python 环境（worktree 无独立 venv）

```bash
# 用主仓库 venv 跑测试：
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/s4-run-metric-plan/packages/agent/backend
PYTHONPATH=packages/harness:/home/wangqiuyang/noldus-insight/packages/ethoinsight \
  .venv/bin/python -m pytest tests/test_auto_seal_from_artifacts.py -q
```

### 循环导入防护（必读）

- harness 内存在已知导入环：`task_tool → subagents/executor`（conftest.py mock 了 `deerflow.subagents.executor`）
- **所有新文件在 `subagents/`、`tools/builtins/` 的跨模块 import 必须放函数体内（惰性），不能放顶层**
- 改完核心文件后必须跑：
  ```bash
  cd /home/wangqiuyang/noldus-insight/.claude/worktrees/s4-run-metric-plan/packages/agent/backend
  PYTHONPATH=packages/harness python -c "import app.gateway"
  PYTHONPATH=packages/harness python -c "from deerflow.agents import make_lead_agent"
  ```

### 关键参考文件

| 文件 | 用途 |
|------|------|
| `tools/builtins/prep_metric_plan_tool.py` | Runtime 用法模式、workspace_path 获取、replace_virtual_path 调用 |
| `sandbox/tools.py:557` | `_build_path_env(thread_data)` — 构建 `DEERFLOW_PATH_*` env |
| `tools/builtins/seal_handoff_tools.py:354` | `_seal_handoff_to_workspace(model_cls, filename, payload, workspace)` |
| `subagents/handoff_schemas.py` | `CodeExecutorHandoff` schema |
| `subagents/metric_aggregation.py` | 已完成的 SSOT 聚合函数 |
| `catalog/schema.py` | `PlanMetric.args: list[str]`，`PlanStatistics.script/input/output` |

---

## 下一步：实现 `run_metric_plan_tool.py`

### 设计要点（已确定）

**工具签名**：
```python
@tool("run_metric_plan", parse_docstring=True)
def run_metric_plan_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    plan_path: str = "/mnt/user-data/workspace/plan_metrics.json",
    only_metric_ids: list[str] | None = None,
    on_error: str = "continue",
) -> dict[str, Any]:
    """一步跑完 plan_metrics.json 的所有 compute + statistics，确定性聚合并落盘 handoff。"""
```

**Worker 函数（必须模块级，供 ProcessPoolExecutor pickle）**：
```python
def _run_metric_task(script: str, args: list[str]) -> tuple[str, int, str]:
    """Returns (script, returncode, error_msg). 0=success."""
    import importlib
    mod = importlib.import_module(script)
    try:
        rc = mod.main(args)
        return (script, rc or 0, "")
    except SystemExit as e:
        return (script, e.code or 0, "")
    except Exception as e:
        return (script, 1, str(e))
```

**核心流程**：
1. 从 `runtime.state["thread_data"]["workspace_path"]` 取 workspace
2. `replace_virtual_path(plan_path, thread_data)` 解析虚拟路径 → 真实路径
3. 读 `plan_metrics.json`，按 `only_metric_ids` 过滤
4. `_build_path_env(thread_data)` + `os.environ.update(path_env)`（worker 继承 env）
5. `ProcessPoolExecutor` 提交所有 compute metrics（`PlanMetric`）
6. 每个 future：`future.result(timeout=PER_TASK_TIMEOUT_SECONDS)`，失败时：`on_error="abort"` 则中止，`"continue"` 则继续
7. Statistics（`PlanStatistics`）：若 `skip_reason is None`，写 `inputs.json`（内容来自 `plan.inputs.raw_files`），调 stats script
8. 调 `aggregate_metrics_to_handoff(plan, workspace, run_validation=True)`
9. 组装 payload（含 `sealed_by="run_plan"`），调 `_seal_handoff_to_workspace(CodeExecutorHandoff, "handoff_code_executor.json", payload, workspace)`
10. 返回 compact dict：`{"status", "handoff_path", "n_total", "n_completed", "n_failed", "failures", "gate_signals"}`

**Statistics argv 构建（注意：`PlanStatistics.input` 是遗留字段，指向错误文件）**：
```python
# 必须从 plan.inputs.raw_files 自己写 inputs.json
inputs_json_path = workspace / "inputs.json"
inputs_json_path.write_text(json.dumps(plan["inputs"]["raw_files"]))
argv = [
    "--inputs", f"/mnt/user-data/workspace/inputs.json",  # 虚拟路径，worker 会 resolve
    "--groups", f"/mnt/user-data/workspace/groups.json",
    "--output", stats.output,  # 来自 PlanStatistics.output（已是虚拟路径）
]
```

**超时常量**：
```python
PER_TASK_TIMEOUT_SECONDS = 120  # per spec
MAX_WORKERS = 4                  # per spec
```

### Statistics script 调用方式

`PlanStatistics` 没有 `args` 字段，需手动构建。`PlanMetric` 有 `args: list[str]`，直接传给 `_run_metric_task`。

---

## 步骤 5：注册 `run_metric_plan`

文件：`tools/builtins/__init__.py`

```python
# 在文件末尾加（惰性 import 放在函数体或模块顶层均可，但要确认不闭环）
from deerflow.tools.builtins.run_metric_plan_tool import run_metric_plan_tool
# 加入 __all__ 列表
```

---

## 步骤 6：重写 `code_executor.py`

文件：`subagents/builtins/code_executor.py`

**工具列表改为**：
```python
["run_metric_plan", "read_file", "ls", "seal_code_executor_handoff"]
```

删除：`bash`, `write_file`, `str_replace`

**System prompt 大幅简化**：删除原来的 `<workflow>` 和 `<bash_constraints>` 章节，改为：
- 调 `run_metric_plan` 执行整个 plan（一步到位）
- 查看结果后调 `seal_code_executor_handoff`
- 只有在工具返回 `status: partial/failed` 时才需要诊断

---

## 步骤 7：测试套

文件：`tests/test_run_metric_plan.py`

Spec §4 要求的 14 个测试（见 spec 文档 §4）：

关键测试：
- `test_run_metric_plan_all_success` — 全部成功，status=completed，handoff 落盘
- `test_run_metric_plan_partial_failure_continue` — 1 个失败，on_error=continue，status=partial
- `test_run_metric_plan_abort_on_first_failure` — on_error=abort，停在第一个失败
- `test_run_metric_plan_only_metric_ids_filter` — 过滤子集
- `test_run_metric_plan_timeout` — 超时处理
- `test_run_metric_plan_ssot_parity` — 与 aggregate_metrics_to_handoff 直接调用结果字节一致（spec §1.4）
- `test_run_metric_plan_sealed_by_run_plan` — handoff 里 sealed_by="run_plan"
- `test_run_metric_plan_validation_warnings` — run_validation=True 时校验警告出现

---

## 步骤 8：最终验证

```bash
# 裸导入（循环导入检测）
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/s4-run-metric-plan/packages/agent/backend
PYTHONPATH=packages/harness python -c "import app.gateway"
PYTHONPATH=packages/harness python -c "from deerflow.agents import make_lead_agent"

# 针对性测试
PYTHONPATH=packages/harness:/path/to/ethoinsight .venv/bin/python -m pytest \
  tests/test_run_metric_plan.py \
  tests/test_auto_seal_from_artifacts.py \
  tests/test_gateway_import_no_cycle.py -v

# 全量
.venv/bin/python -m pytest tests/ -x -q
```

---

## 步骤 9：Commit + Push

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/s4-run-metric-plan

git add packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py
git add packages/agent/backend/packages/harness/deerflow/subagents/metric_aggregation.py
git add packages/agent/backend/packages/harness/deerflow/subagents/executor.py
git add packages/agent/backend/packages/harness/deerflow/tools/builtins/run_metric_plan_tool.py
git add packages/agent/backend/packages/harness/deerflow/tools/builtins/__init__.py
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py
git add packages/agent/backend/tests/test_run_metric_plan.py

git commit -m "feat(s4): 用 run_metric_plan 工具替换 code-executor bash 编排

- handoff_schemas: sealed_by 加入 run_plan 枚举值
- metric_aggregation: 抽出 aggregate_metrics_to_handoff 纯函数 SSOT
- executor: auto-seal 分支改调 aggregate_metrics_to_handoff
- run_metric_plan_tool: ProcessPoolExecutor 执行全部 compute+statistics
- code_executor: 删 bash/write_file/str_replace，加 run_metric_plan
- tests: test_run_metric_plan.py 14 个测试覆盖 spec §4"

git push origin feature/s4-run-metric-plan
```

---

## 风险与注意事项

1. **循环导入**：`run_metric_plan_tool.py` 里所有 `from deerflow...` 必须在函数体内（惰性）。特别是 `metric_aggregation`、`seal_handoff_tools`、`sandbox.tools`。
2. **Statistics argv**：不能用 `PlanStatistics.input`（指向 `handoff_code_executor.json`，是遗留字段），必须从 `plan.inputs.raw_files` 写 `inputs.json`。
3. **DEERFLOW_PATH_\* env**：在 `ProcessPoolExecutor.submit()` 前调用 `os.environ.update(_build_path_env(thread_data))`，否则 worker 里 `resolve_sandbox_path` 不认 `/mnt/user-data/...` 路径。
4. **__init__.py 已有内容**：注册前先读 `tools/builtins/__init__.py`，搜索 `__all__` 和现有 import 列表，外科式加入，绝不整文件覆盖。
5. **conftest mock 假绿陷阱**：`conftest.py` mock 了 `deerflow.subagents.executor`，致导入环 pytest 不可见。改完必须做步骤 8 的裸导入验证。
6. **PER_TASK_TIMEOUT_SECONDS=120**：spec 要求，不要随意调大。
