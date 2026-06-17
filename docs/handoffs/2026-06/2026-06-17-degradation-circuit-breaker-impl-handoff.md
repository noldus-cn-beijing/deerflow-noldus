# Handoff: P2 降级信号源 + P7 降级熔断器合并实施

> 日期：2026-06-17
> 分支：`worktree-degradation-circuit-breaker`（基于 `origin/dev` 4b8d6e2c）
> 关联 spec：
> - [2026-06-17-statistics-loud-failure-spec.md](../../specs/2026-06-17-statistics-loud-failure-spec.md)（P2，信号源 L2）
> - [2026-06-17-data-degradation-circuit-breaker-spec.md](../../specs/2026-06-17-data-degradation-circuit-breaker-spec.md)（P7，熔断器 L1）
> 状态：实施完成 + 全套测试绿，待 PR review。

## 背景与目标

statistics 子步骤崩溃（rc=1，如 ZeroDivision）原本被静默吞成 `statistics={}`：`run_metric_plan` 仍报 `status=completed n_failed=0`，下游 data-analyst 收到空 statistics → 走描述性手算螺旋 → 残废报告。三宗罪：没通知、无有限自救（反而无限螺旋）、没 HITL。

本 PR 把两个 spec 合一落地（信号 + 熔断同 PR，spec §0 明确建议）：

- **P2（L2 信号源）**：`run_metric_plan` 在 `gate_signals` 亮 `statistics_status` 三态（`ok`/`crashed`/`absent_by_design`）可机读信号，崩溃时 status 降级 completed→partial。不再静默。
- **P7（L1 熔断器）**：新增 `DegradationCircuitBreakerMiddleware`（挂 lead 链），读 code-executor handoff 的 `statistics_status`，检测 `crashed` → 自救一次（jump_to=model，重派 code-executor）→ 超限转 HITL（让模型调 `ask_clarification` 问用户）。

产品决策：自救上限 = **1 次**（一次重试足以区分偶发 vs 必然崩溃；spec §5 "figure out 时间不应长"）。

## 改动清单

### P2 信号源

| 文件 | 改动 |
|---|---|
| `packages/harness/deerflow/tools/builtins/run_metric_plan_tool.py` | Step7 追踪 `statistics_status`/`statistics_error`（rc==0+非空→ok；rc==0+空/读失败→crashed；rc!=0→crashed；skip_reason 非空→absent_by_design）；Step8 crashed 时 completed→partial；Step9 透传给 `_build_gate_signals`；`_build_gate_signals` 加两 keyword-only 参数（默认值向后兼容）+ 写入返回 dict。加 `from typing import Literal`。 |
| `packages/harness/deerflow/subagents/handoff_schemas.py` | `GateSignals` 显式声明 `statistics_status: Literal[...]` + `statistics_error: str|None`（SSOT：三态字面量唯一定义处）。原已 `extra="allow"`，显式声明让 SSOT 清晰 + IDE 可用。 |

**SSOT**：`statistics_status` 三态字面量唯一在 `GateSignals.statistics_status: Literal[...]`。tool 赋值与熔断器判断都引用同字面量，不另造。

### P7 熔断器

| 文件 | 改动 |
|---|---|
| `packages/harness/deerflow/agents/middlewares/degradation_circuit_breaker_middleware.py`（**新建**） | `DegradationCircuitBreakerMiddleware`，复刻 `SealGateMiddleware`（`@hook_config(can_jump_to=["model"])` + per-run 计数 + fail-open）+ `QualityWarningBroadcast`（last AIMessage 无 tool_calls 才触发；`resolve_workspace_from_state` 取 workspace 后直接 `json.loads` 读 handoff，**不走 `read_handoff`** 以避开 `get_app_config` 依赖 + schema 校验副作用）。`_MAX_SELF_HELP=1`。per-run mtime 去重防同条 crashed 反复触发。reminder 用正面指令（deepseek）。 |
| `packages/harness/deerflow/agents/lead_agent/agent.py` | `build_middlewares` 在 `QualityWarningBroadcastMiddleware()` 之后、`SafetyFinishReason`/`ClarificationMiddleware` 之前惰性 import + append。 |

**HITL 机制**：熔断器不自造 tool_call。注入 reminder + jump_to=model 让模型自己调 `ask_clarification`，由 `ClarificationMiddleware`（lead 链最后）拦截 → `Command(goto=END)` 中断。

### 测试

| 文件 | 用例 |
|---|---|
| `tests/test_run_metric_plan.py` 新增 `TestStatisticsStatus`（5 用例） | ok / crashed（含 status 降级 + failures 记录）/ absent_by_design / crashed 与 absent 字节可区分 / 空 payload 视作 crashed |
| `tests/test_degradation_circuit_breaker_middleware.py`（**新建**，10 用例） | 首次 crash 自救 / 同 mtime 不重触发 / 超限转 HITL / ok 不动 / absent 不动 / last AI 有 tool_call 不抢断 / 无 handoff 不动 / fail-open / per-run 计数隔离 / `_MAX_SELF_HELP==1` |

## 验证结果

- ✅ 信号源 5 + 熔断器 10 + run_metric_plan 现有（27 含新增 5）全绿。
- ✅ 裸导入两生产入口（`import app.gateway` / `from deerflow.agents import make_lead_agent`，cwd=backend，加载 worktree 源）0 退出——导入环无回归。
- ✅ 全量测试：我的改动**不引入新失败**。干净 origin/dev 基线 10 failed / 4097 passed；加我的改动后同样 10 failed / 4112 passed（多 15 = 新增测试）。这 10 个失败是**已知全量测试隔离污染**（`test_*_async_delegates_to_sync` 系列 + `local_sandbox_provider_mounts` + `chart_maker_config` + `data_analyst_step28_contract`，单跑全绿），与本次改动无关。坐实法=stash 改动跑干净基线同样 10 failed。
- ✅ ruff lint：我引入的代码 All checks passed（修了 3 个 I001 import 排序）。agent.py 的 `F841 workflow_mode` 是 pre-existing（stash 验证 origin/dev 已有）。

## 风险与边界（spec §5 落实）

- **防同条反复触发**：per-run `_processed_mtime` 按 handoff 文件 mtime 去重。同一文件 mtime 只触发一次；handoff 被 code-executor 重写（mtime 变）才允许新一条触发，此时计数已 +1 → 直接进 HITL。
- **只熔断 crashed**：`absent_by_design`（单组合理 skip）/ `ok` 不熔断（已落 handoff，lead 自读即通知）。判据严格 `gate_signals.get("statistics_status") != "crashed"`。
- **status 不加新枚举**：crashed 时 completed→partial（复用现有三态 `completed/partial/failed`），避免全链 status 枚举核对风险（memory `feedback_dataanalyst_reportwriter_handoff_status_missing_partial`）。区分靠 `statistics_status` 信号字段。
- **挂载顺序**：QualityWarningBroadcast 后（都读同 handoff，QWB 只打 additional_kwargs 不抢断）、Clarification 前（熔断器 jump_to=model 后模型发 ask_clarification → Clarification 拦截中断）。
- **导入环**：新 middleware 顶层只 import langchain/langgraph 标准件；`experiment_context` 惰性 import 放 `_check` 函数体；挂载处 import 放 `build_middlewares` 函数体内。永久 guard `test_gateway_import_no_cycle.py` 在 CI 自动验证。
- **正交于 P1**：本 PR 只改"崩溃如何被上报"，不改 statistics 计算逻辑（ZeroDivision 修复是 P1/outlier-diagnostics PR#148，已合）。

## 与其他 spec 的关系

- **P1（ZeroDivision）已合 PR#148**：让"用正确数据不该崩"成立。本 PR 让"万一还崩，响亮报错触发自愈/HITL，而非降级糊弄"。两者互补：P1=本不该崩，P2+P7=崩了也不许藏。
- **statistics 列对齐已合 PR#141**：本 PR 的 Step7 改动在 PR#141 的 `--parameters-json` 透传之上叠加（同函数、正交字段），无冲突。
- **seal-gate（feat/seal-gate-middleware）待 PR**：同为 after_model + jump_to 范式，本熔断器复刻其结构。

## worktree 注意事项（给后续 agent）

- worktree `degradation-circuit-breaker` 借主仓 venv（editable pth 指主仓 `packages/harness`）。跑测试须 cwd=backend + `PYTHONPATH=packages/harness:.`，否则 deerflow 解析到主仓 → 假绿（memory `feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib`）。
- worktree 无 `config.yaml`（gitignored）。跑全量需 `export DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml`。
- worktree 从 `origin/dev` fresh 建出（4b8d6e2c），含 PR#141/#147/#148 等。**注意**：本地 `origin/dev` remote ref 曾经过时（停在 PR#140），实施前已 `git fetch origin dev` + `git reset --hard origin/dev` 对齐。

## milestone 建议

无需新增/更新 milestone。本 PR 是红线一（降级不静默）的基础设施落地，属 harness 健壮性 track，不改变 v0.1 feature 进度。可在既有 milestone（若存在"subagent 鲁棒性 / 降级治理"track）追加一条摘要。
