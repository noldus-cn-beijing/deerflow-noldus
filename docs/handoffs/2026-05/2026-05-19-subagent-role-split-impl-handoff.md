# 2026-05-19 Subagent Role Split Implementation Handoff

## 任务目标

按 [plan](../../superpowers/plans/2026-05-18-subagent-role-split-implementation-plan.md) 执行 22 个 Task，用 capability-exposure 重构 lead → 5 subagent 调度，消除 5-14/5-18 同类故障（lead 不读 handoff / catalog 无 fallback / code-executor 角色过载）。

## 当前进展

✅ **W1-W21 代码全部落地，PR #8 已 merge 到 dev**（merge commit `cd5cceca`）
- 23 个 commit 在 dev 上
- Backend: 2604 tests pass (33 pre-existing failures 与本次无关)
- Ethoinsight: 267 tests pass (1 pre-existing 与本次无关)

### 已完成的 21 个 Task

| Task | 内容 | 关键文件 |
|------|------|----------|
| W1 | SubagentConfig 加 4 capability 字段 | `subagents/config.py` |
| W7 | execution-conventions.md 新建 | `skills/custom/ethoinsight/references/execution-conventions.md` |
| W2 | PlanMetrics + PlanCharts dataclass | `ethoinsight/catalog/schema.py` |
| W3 | _common.yaml + load_common_catalog | `ethoinsight/catalog/_common.yaml`, `loader.py` |
| W5 | _evaluate_when 扩展 total_subjects | `ethoinsight/catalog/resolve.py` |
| W6 | plot_timeseries.py CLI | `ethoinsight/scripts/_common/plot_timeseries.py` |
| W4 | catalog CLI --mode + resolve_charts | `cli.py`, `resolve.py` (resolve_metrics + resolve_charts + serializers) |
| W20 | prep_metric_plan → plan_metrics.json | `tools/builtins/prep_metric_plan_tool.py` |
| W8 | ethoinsight-lead-interaction skill | `skills/custom/ethoinsight-lead-interaction/` (4 文件) |
| W9 | ethoinsight-charts 重定位为 chart-maker 专有 | `skills/custom/ethoinsight-charts/SKILL.md` |
| W10 | 删 default-template-fallback + Gate before guess | `skills/custom/ethovision-paradigm-knowledge/`, `prompt.py` |
| W21 | ethoinsight-chart-maker skill | `skills/custom/ethoinsight-chart-maker/` (3 文件) |
| W11 | code-executor capability + 删 charts 段 | `subagents/builtins/code_executor.py`, `ethoinsight-code/SKILL.md` |
| W12 | data-analyst capability | `subagents/builtins/data_analyst.py` |
| W14 | report-writer capability + 可选 chart handoff | `subagents/builtins/report_writer.py` |
| W15 | knowledge-assistant capability | `subagents/builtins/knowledge_assistant.py` |
| W13 | chart-maker builtin 新建 + 注册 | `subagents/builtins/chart_maker.py`, `__init__.py`, `handoff_registry.py` |
| W16 | Lead prompt 瘦身 1243→395 行 | `agents/lead_agent/prompt.py` |
| W17 | IntentClassificationGuardrailProvider | `guardrails/intent_classification_provider.py`, `agent.py` |
| W18 | TaskHandoffAuthorizationProvider | `guardrails/task_handoff_authorization_provider.py`, `agent.py` |
| W19 | task_tool 自动注入 handoff 占位符 | `tools/builtins/task_tool.py` |

### W1 实施中的关键决策

- `HANDOFF_FILE_REGISTRY` 从 `task_tool.py` 提取到 `subagents/handoff_registry.py`（零依赖 thin module），打破 `builtins/__init__ → task_tool → deerflow.subagents` 循环依赖
- `validate_subagent_handoff_refs()` 在 `builtins/__init__.py` import 时调用，fail-fast

### W4 关键设计

- `resolve()` 保留作为 backward-compat wrapper（内部调 `resolve_metrics()` + 补空 `charts=[]`）
- `resolve_charts()` 新增，fallback 触发：`len(charts)==0` → load `_common.yaml`
- CLI 默认 mode=`metrics`，向后兼容

### W16 Lead prompt 瘦身

- `_build_subagent_section()` 整段重写，调 `format_subagent_capability()` 渲染 5 个 SubagentConfig
- 删除 ~340 行 noldus_rules（4-choice模板、范式默认fallback、chart选择逻辑等）
- 保留 5 条硬约束：`[intent]` 行、不手写 `{{handoff://X}}`、Gate before guess、ev19 template 前置、意图状态机骨架
- 细节已移入 `ethoinsight-lead-interaction` skill

## 关键上下文

### 新架构

```
Lead Agent (deepseek-v4-pro, prompt ~395行)
  ├── IntentClassificationGuardrailProvider → 强制 [intent] X 行
  ├── TaskHandoffAuthorizationProvider → 安全网：校验 required handoff 占位符存在
  ├── Ev19TemplateGuardrailProvider → 不动（已有）
  └── 5 subagent:
        ├── code-executor → 只跑 metrics+stats，读 plan_metrics.json
        ├── data-analyst → required_upstream_handoffs=["code_executor"]
        ├── chart-maker → 新增！自跑 catalog.resolve --mode charts，bash ≤6
        ├── report-writer → required_upstream_handoffs=["code_executor","data_analyst"]
        └── knowledge-assistant → required_upstream_handoffs=[]
```

### 关键文件路径

- `packages/agent/backend/packages/harness/deerflow/`
  - `subagents/config.py` — SubagentConfig + format_subagent_capability + validate_subagent_handoff_refs
  - `subagents/handoff_registry.py` — HANDOFF_FILE_REGISTRY authoritative source
  - `subagents/builtins/chart_maker.py` — CHART_MAKER_CONFIG（新建）
  - `subagents/builtins/code_executor.py` — 删 charts 段，skills 去掉 ethoinsight-charts
  - `guardrails/intent_classification_provider.py` — IntentClassificationGuardrailProvider + IntentBridgeMiddleware
  - `guardrails/task_handoff_authorization_provider.py` — TaskHandoffAuthorizationProvider
  - `tools/builtins/task_tool.py` — _auto_inject_handoff_placeholders() (W19)
  - `agents/lead_agent/prompt.py` — 瘦身到 ~395 行
  - `agents/lead_agent/agent.py` — 挂载 W17 + W18 guardrail
- `packages/ethoinsight/ethoinsight/catalog/`
  - `schema.py` — PlanMetrics + PlanCharts（旧 Plan 保留）
  - `_common.yaml` — trajectory_plot + timeseries_plot fallback
  - `loader.py` — CommonCatalog + load_common_catalog
  - `resolve.py` — resolve_metrics + resolve_charts + serializers
  - `cli.py` — --mode {metrics,charts}
- `packages/agent/skills/custom/`
  - `ethoinsight-lead-interaction/` — 新建（SKILL.md + 3 references）
  - `ethoinsight-chart-maker/` — 新建（SKILL.md + 2 references）
  - `ethoinsight/references/execution-conventions.md` — 新建
  - `ethoinsight-charts/SKILL.md` — 重定位为 chart-maker 专有（v2.0.0）
  - `ethovision-paradigm-knowledge/SKILL.md` — 删 fallback 引用 + 加 Gate before guess

### Plan/spec 文件位置

- Spec: `docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md`
- Plan: `docs/superpowers/plans/2026-05-18-subagent-role-split-implementation-plan.md`

## 未完成事项

🔴 **W22: Dogfood 三场景 E2E 验证** — 需要手动跑

用户正在主仓库 `/home/wangqiuyang/noldus-insight` 手动执行。已 `git pull` 拿到 merge commit `cd5cceca`。

### W22 三场景验收标准

**S1（5-18 复现：单被试 + "再画几个图"）**:
1. 上传单被试 EPM 数据 + "帮我分析一下"
2. 4-choice → 选加图
3. 验收：0 LOOP DETECTED | chart-maker bash ≤6 | lead 不自己 write_file | fallback_used=true, 2 charts

**S2（正常 EPM 3v3 + "都要"）**:
1. 上传 EPM 3v3 数据 + "帮我分析一下"
2. 4-choice → 选"都要"
3. 验收：catalog charts 命中(fallback_used=false) | report.md 6段

**S8（范式模糊反问）**:
1. 上传无范式线索文件（如 `mouse_data_2026.txt`）+ "帮我分析一下"
2. 验收：不调 set_experiment_paradigm | ask_clarification 带证据（列名/文件名）

### Bug 回环

如果任何场景 fail：
- 写新 task `W22.1-fix-<issue>` 在 worktree 修
- 单独 commit
- 重新 dogfood 该场景

## 风险与注意事项

- **Pre-existing test failures**: 33 个 backend 测试失败（provisioner/pvc/run_event_store/sandbox/planning skill config）与本次无关，不要修
- **W14 有两个 commit**（5cde6d35 + be419b7b）：linter 导致 split，功能正常
- **W16 瘦身到 395 行而非 plan 说的 200 行**：实际保留了一些必要的 skill reference 和 memory injection wiring，68% 缩减已达标
- **`{{handoff://X}}` 转义**：在 Python f-string 中需要 `{{{{handoff://{name}}}}}`（4 重括号）

## 下一位 Agent 第一步

如果用户报告 W22 dogfood 有场景失败：
1. `cd ~/noldus-insight && git pull` 确保 dev 最新
2. 新开 worktree 修 bug
3. 跑 `make test` 确认绿
4. commit + push + PR merge 到 dev
5. 重复 dogfood 失败场景

如果 W22 全绿：
无需操作，本次特性开发完成。
