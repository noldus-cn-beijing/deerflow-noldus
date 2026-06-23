# ETHO-1 SealGate after_agent 兜底 — 实施 handoff

> 日期：2026-06-23
> 分支：`worktree-etho1-seal-gate-after-agent`
> Spec：[docs/superpowers/specs/2026-06-23-etho1-seal-gate-after-agent-reconstructable-vs-cognitive-spec.md](../../superpowers/specs/2026-06-23-etho1-seal-gate-after-agent-reconstructable-vs-cognitive-spec.md)
> 性质：🔴 harness 控制安全（D7）加固。subagent 完成推理后未调 `seal_<name>_handoff` → 报 `terminated without emitting` → Lead 自动重试一整轮（~2-4min 代价 + 用户可见 `Task failed` 中间态）。

## 做了什么

升级现有 `SealGateMiddleware`（**不新建 middleware**，spec §0 用户拍板方向），分两类堵漏：

### 修法 A：可重建产物的 after_agent 兜底（report-writer / chart-maker）

给 `SealGateMiddleware` 加 `after_agent` / `aafter_agent` hook（**不标 `can_jump_to`**，纯副作用）。逻辑：
1. 仅 `_RECONSTRUCTABLE = {report-writer, chart-maker}` 通过；data-analyst / code-executor / 非 seal subagent 直接 return None。
2. history 已有 seal ToolMessage → 已 seal，return None。
3. 从 `state["thread_data"]["workspace_path"]`（复用 `experiment_context.resolve_workspace_from_state`）解析 workspace；解析不出 → fail-open。
4. 惰性 import（守 import 环铁律）调 `executor._attempt_auto_seal_from_artifacts(name, workspace, sealed_by="after_agent_artifacts")`。
5. sealed 与否都 return None（after_agent 不能 jump；executor L3/L4 仍是最终裁决）。

**效果**：把 executor L3 auto-seal（原本埋在 L2 seal-resume 失败之后、且 FAILED 错误已冒给 lead 之后才跑）**提前到终止点** → 消除 `Task failed` 中间态 + 省一轮 lead 重试。堵的是 L1 `_MAX_REMINDERS=2` 的放行口（spec §1.3 缺口 A）。

### 修法 B：data-analyst 诚实不兜底

`after_agent` 对 data-analyst **不做任何事**：判读结论是认知产物，after_agent 既不能 jump 补调、又不能伪造判读。漏调仍走 L1（after_model nudge）+ L2（seal-resume）+ L4（lead 重试）。spec §2.2 明确写：**降触发率不归零**，符合 memory `feedback_code_has_fix_not_equal_bug_eliminated`。

### 修法 C：sealed_by 可观测性（HarnessX trace richness）

- `ReportWriterHandoff` / `ChartMakerHandoff` schema 新增 `sealed_by: Literal["model", "after_agent_artifacts", "executor_artifacts"] = "model"`。
- `_attempt_auto_seal_from_artifacts` 加可选 `sealed_by` kwarg（report-writer/chart-maker 分支透传；code-executor 分支固定 `framework_rebuild` 不受影响）。
- executor L3 调用点传 `sealed_by="executor_artifacts"`；SealGate after_agent 调用点传 `sealed_by="after_agent_artifacts"`。
- 兜底触发率可观测（spec §2.3 + memory `feedback_fallback_trigger_rate_must_be_observable`）：`after_agent_artifacts` 触发率上升 = 上游 L1 在退化的信号。

## 不改的东西（spec §2.4 scope guard）

- `_MAX_REMINDERS=2` **不动**（提高 cap 依赖 thinking 超时线先解决，另一条 spec 处理）。
- **不加** 任何 seal 提醒 prompt 规则（HarnessX §6.6 Telecom 累加 reminder 禁令；我们已改 4 次 prompt 打地鼠）。
- **不改** `_attempt_auto_seal_from_artifacts` 的重建逻辑（复用，只新增调用点 + 透传 sealed_by）。
- **不改** L2 seal-resume / L4 FAILED→lead 重试。
- 测试 `test_max_reminders_unchanged` + `test_after_agent_not_can_jump_to` 锁住 scope。

## 改动清单

| 文件 | 改动 |
|---|---|
| `seal_gate_middleware.py` | +`_RECONSTRUCTABLE` frozenset；+`after_agent`/`aafter_agent`（不标 can_jump_to，惰性 import executor + experiment_context，fail-open）；模块 docstring 补 ETHO-1 升级段 |
| `subagents/executor.py` | `_attempt_auto_seal_from_artifacts` 加 `sealed_by` kwarg（report-writer/chart-maker 透传）；L3 调用点传 `executor_artifacts` |
| `subagents/handoff_schemas.py` | `ReportWriterHandoff` / `ChartMakerHandoff` 加 `sealed_by` 字段 |
| `tests/test_seal_gate_after_agent.py`（新） | 13 个测试覆盖 spec §四 全部 7 项（红→绿 / 边界 / noop / fail-open / smoke / 可观测 / scope guard） |

## 验收

- ✅ **TDD 红→绿**：13 测试改前 5 红（核心 auto-seal ×2 + `_RECONSTRUCTABLE` + sealed_by ×2）改后全绿；8 边界测试（data-analyst skip / already-sealed / fail-open / smoke / scope guard）始终绿。
- ✅ **import 环**：裸导入 `import app.gateway` + `from deerflow.agents import make_lead_agent` 0 退出（惰性 import 守环；after_agent 的 import 全在函数体内）。
- ✅ **seesaw 回归**：seal 邻域全绿（`test_seal_gate_middleware` / `test_auto_seal_from_artifacts` / `test_seal_resume` / `test_seal_handoff_tools` / `test_chart_maker_seal_reconciliation` / `test_executor_handoff_emission` / `test_data_analyst_empty_statistics_seal_rule` / `test_seal_data_analyst_parameter_audit` = 136 邻域 + 循环 guard）。
- ✅ **backend 全量 A/B**（overlay 法跑主仓 editable）：clean baseline 9 failed / 4284 passed；with my overlay **同 9 failed** / 4297 passed（= +13 新测试，零回归）。9 个 failed 全是 pre-existing 污染（3 `local_sandbox_provider_mounts` 确定性 + 5 `resolved_facts_readback` + 1 `s8_feedback` 顺序污染，单独跑全绿），与本次改动无关。
- ✅ **可观测性**：sealed_by 三态（model / after_agent_artifacts / executor_artifacts）进 handoff JSON；after_agent 触发有 `[seal_gate] after_agent auto-sealed ...` WARNING 日志。
- ✅ **scope**：`_MAX_REMINDERS == 2` 未改；无新增 seal 提醒 prompt。
- ✅ ruff（本次改动文件）全绿。

## 测试环境踩坑（worktree 共享主仓 venv）

worktree 无独立 venv，借主仓 `.venv`（editable link 指主仓 `packages/harness`）。普通 import 解析到主仓源 = 测主仓假绿。解法（守 memory `feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib`）：
- 新测试 `test_seal_gate_after_agent.py` 用 `importlib.spec_from_file_location` 加载 worktree 中间件 + autouse fixture 把 worktree executor 注入 `sys.modules["deerflow.subagents.executor"]`（让中间件 after_agent 内的惰性 import 解析到 worktree executor，含新 `sealed_by` kwarg）。
- 全量回归用 **overlay 法**：把 3 个改动文件 + 新测试 cp 到主仓 editable，跑主仓全量（config.yaml 在主仓可用），跑完还原 + 删 .bak。A/B 对照判文件集不判数（守 memory `feedback_pr169_metric_metadata_sidecar_passed`）。

## 三大病理自检（CLAUDE.md 改 harness 前必过）

1. **Reward hacking**：after_agent 兜底会不会让 LLM 学会「反正有兜底」？→ 不构成糊弄验收：report.md / plot_*.png 是真产出非伪造；data-analyst 明确不兜底（不伪造判读）；sealed_by 触发率可观测兜住退化信号。
2. **Catastrophic forgetting**：加 after_agent 会不会回归 after_model L1？→ 两 hook 正交（after_model 逼补调 / after_agent 终止点兜底）；seal 邻域 136 测试 + L1 全绿守 seesaw。
3. **Under-exploration**：本 spec 敢动结构（加 after_agent hook + 改终止点兜底），不是又加一条 prompt 规则——正面避开 HarnessX §6.6 Telecom 累加 reminder 陷阱。

## milestone 建议

「subagent 派遣/生命周期 infra 加固」track：本 PR（ETHO-1 SealGate after_agent 兜底）与 ETHO-2（registry/prompt）同属 subagent 生命周期 infra 系列。checkpoint = 「seal 漏调从『偶发 Task failed 中间态 + 多烧一轮』降到『可重建产物零中间态、认知产物可观测降级』」。
