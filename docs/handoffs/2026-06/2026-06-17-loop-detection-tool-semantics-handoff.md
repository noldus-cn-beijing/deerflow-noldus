# Handoff: loop-detection 工具语义修复（Spec P6, 2026-06-17）

> 分支：`worktree-loop-detection-tool-semantics`（基于 dev）
> Spec：[docs/superpowers/specs/2026-06-17-loop-detection-tool-semantics-spec.md](../../specs/2026-06-17-loop-detection-tool-semantics-spec.md)
> 状态：实施完成，测试全绿，待 PR 进 dev。

## 症状（2026-06-17 EPM dogfood 实证）

lead 的 `write_todos`（记账工具）在长 E2E（code→data→chart→report）中被调 5 次触发 `[FORCED STOP] Tool write_todos called 5 times — All tool_calls stripped.`，**同一条消息里所有 tool_call 被剥光**，包括 lead 要派 report-writer 的 `task` call → 报告永远派不出去。

## 根因（两个叠加缺陷）

1. **缺陷 A — 不分工具语义**：生产 `config.example.yaml` 里 `write_todos` 的 override 是 `warn=2 / hard_limit=4`，**比全局 (30/50) 还严**——记账工具的高频被当成死循环。
2. **缺陷 B — 熔断殃及无辜**：频率硬限时 strip **整条消息所有 tool_calls**，把语义无关、正在推进流程的 `task(report-writer)` 一起剥。

## 修复（红线四正模式 1/2/3）

### 正模式 1：按工具语义分级（floor 机制）
`LoopDetectionConfig` 新增 `_SEMANTIC_OVERRIDE_FLOORS`（ClassVar）+ validator，给记账工具一个**语义地板**：
- `write_todos: (warn=15, hard_limit=30)` —— caller 可调高，**不可在地板下收紧**（max(caller, floor)）。这样无论 config.yaml 怎么配、或裸 `LoopDetectionConfig()`，write_todos 都不会再误伤。
- `config.example.yaml` 的 write_todos override 同步改为 (15/30)。
- `task` 不进 floor：它已在 `_track_and_check` 按 subagent_type 分桶，派 4 个不同 subagent 是 4 个不同桶，只有全局 50 管「重复派同一 subagent」。

### 正模式 2：熔断副作用最小化 —— 只 strip 超限工具
`_track_and_check` 返回值从二元组 `(warning, hard_stop)` 升级为三元组 `(warning, hard_stop, offending_freq_key)`：
- 频率硬停时返回超限工具的 freq_key（非 task 工具是裸名，task 是 `task:<subagent>`）。
- `_apply` 新增 `_build_partial_strip_update`：**只移除 offending call(s)，保留同消息其余 tool_call**。
- 哈希硬停（整组完全相同重复）仍 strip 全部——整组就是那个循环。
- `_offending_call_ids` 精确匹配（task 按 subagent_type 匹配）。
- finish_reason 只在「无幸存 call」时翻 stop；有幸存 call 保持 `tool_calls` 让其正常执行。

**DanglingToolCall 交互（spec 4 风险边界已核实）**：部分 strip 后保留的 task call 会正常执行产生 ToolMessage；即便因中断无响应，DanglingToolCallMiddleware（`_message_tool_calls` 优先用结构化 `tool_calls`）会兜底注入合成 ToolMessage，不会产生 dangling。LoopDetection 是 after_model、Dangling 是 wrap_model_call，不同 hook 不冲突。

### 正模式 3：熔断信息可操作
`_TOOL_FREQ_HARD_STOP_MSG` 改为 `"...The offending {tool_name} call was removed; any other tool_calls in this turn were preserved..."`，让模型知道其余 call 还在，不要放弃。

## 测试（test_loop_detection_tool_semantics.py，4 组全绿）

- `TestBookkeepingOverrideDefaults`：write_todos 6 次 / task 4 个 subagent 派遣都不误伤（`with_semantic_defaults()`）。
- `TestStripOnlyOffendingTool`：[write_todos(超限), task(report-writer)] → task 保留、write_todos 移除；多 sibling（task+seal）全保留；单工具超限 → `tool_calls==[]` 且 additional_kwargs 干净。
- `TestRealLoopStillCaught`：相同 bash 5 次仍哈希硬停；不同 bash 超全局频率仍硬停（防护未削弱）。
- `TestLongE2ESequenceCompletes`：4 task 派遣 + 4 todo 更新全程不误触发。

## 改动文件

| 文件 | 改动 |
|---|---|
| `loop_detection_middleware.py` | 三元组返回 + `_build_partial_strip_update` + `_offending_call_ids` + `with_semantic_defaults()` + 可操作提示 + `_TOOL_FREQ_SEMANTIC_OVERRIDES` 常量 |
| `loop_detection_config.py` | `_SEMANTIC_OVERRIDE_FLOORS` ClassVar + validator floor 机制（max(caller,floor)） |
| `config.example.yaml` | write_todos override 2/4 → 15/30 + 注释 |
| `test_loop_detection_middleware.py` | 2 个 from_config 测试更新（empty/overrides 现在含 write_todos floor） |
| `test_loop_detection_config_wiring.py` | 4 处解包升三元组 + floor/raise-above-floor 新测试 |
| `test_loop_detection_tool_semantics.py` | 新增，4 组行为测试 |

## 验证

- loop+dangling 测试 115 passed。
- 裸导入两生产入口：`PYTHONPATH=packages/harness python -c "import app.gateway"` / `from deerflow.agents import make_lead_agent` 均 0 退出（导入成功；worktree 无 config.yaml 触发的 FileNotFoundError 是 app.gateway eager 副作用，不阻塞导入，与主仓行为一致）。
- **worktree 借主仓 venv 验证坑**：裸导入命令 `PYTHONPATH=.` 实际加载**主仓**源（editable `__editable__` finder 优先于 PYTHONPATH 目录扫描），要验证 worktree 改动须 `PYTHONPATH=packages/harness`。测试同理必须 `PYTHONPATH=packages/harness` 否则测主仓假绿。
- 全量套件有 24 个**预先存在**的失败（subagent_executor / sandbox_mounts / paradigm_gate / inspect_gate / chart_maker_config / lead_agent_training），主仓 HEAD 同样红，属已知全量污染（memory `feedback_known_full_suite_test_pollution_4_tests`），与本改动无关。
- ruff：改动文件全过（test_loop_detection_config_wiring.py 的 unused `pytest` 是 pre-existing，未碰）。

## 风险边界（已守住）

- ✅ 不削弱真死循环防护：hash-based 完全相同重复检测保留；bash/全局频率限制保留。
- ✅ override 进默认值（config 代码 floor + example.yaml），不靠手配（`feedback_dev_prod_behavior_alignment`）。
- ✅ 改 middleware 后裸导入两生产入口（导入环铁律）。
- ✅ 部分 strip 与 DanglingToolCall 交互已核实，无 dangling tool call。

## 后续

- PR 进 dev（branch `worktree-loop-detection-tool-semantics`）。
- 生产 `~/ethoinsight-prod/config.yaml` 的 write_todos override 需同步改 15/30（部署 SOP 手动同步，因 config.yaml 不进 git）；下次 deploy 时注意。
