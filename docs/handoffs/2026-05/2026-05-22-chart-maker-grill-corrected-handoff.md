# 2026-05-22 chart-maker grill 修正版交接（取代前版）

> **状态**：grill 完成。3 PR 待实施。本文取代 [2026-05-22-tool-vs-script-grill-handoff.md](2026-05-22-tool-vs-script-grill-handoff.md)（前版误判记录见末段）。

## 起点问题

> "看到 Claude Code 用 git bash 的方式，能不能仿照 git，将指标的代码、图表生成的代码，也做成 tooluse，而不是脚本（更重）？"

## 前置结论（与前版一致，仍有效）

- ❌ 不把指标做成 LangChain/MCP tool —— 5/9-5/12 已彻底否决
- ❌ 不放开 code-executor bash 白名单 —— 5/15 lead-bash-removal 锁定
- ❌ 不新建 free-analyst subagent —— β3 行为学同事否决
- ✅ 重的不是 process 冷启动，而是 LLM 试错 + Gate 死锁 + reasoning_effort=high
- ✅ 真正要做的"美观（git 风格）"已在 stage-broadcast.ts 有半成品（Q10 子树）

## 🚨 前版 handoff 关键判断修正

前版 handoff 把 Q 子树（脚本契约一致）的"用户证据"建模为：

> "code-executor 跑 `plot_trajectory --paradigm fst` 失败，因为 `plot_trajectory` 不接受 `--paradigm` 但 `plot_timeseries` 接受。LLM 试错-反思-重试全过程暴露给用户。"

**该叙事仓库证据完全不支持**。本轮 grill 现场核实结论：

| 验证项 | 结果 |
|---|---|
| training-data jsonl 含 `plot_trajectory --paradigm` 试错 | ❌ 0 处 |
| `.deer-flow/threads/*/archived_messages/` 提到该试错 | ❌ 0 处 |
| `checkpoints.db` writes 表 ormsgpack 解码后含字面 `--paradigm` 紧邻 `plot_trajectory` | ❌ 0 处 |
| 全仓库唯一提到 `plot_trajectory --paradigm` 的文件 | ⚠️ **前版 grill handoff 自己**（孤本） |
| 用户指定的 thread 843fe2b8（FST 实际跑过的）chart-maker 4 张图状态 | ✅ 全成，failed_charts=[] |
| 唯一含 `plot_trajectory` 字面的 thread 78ccb52b（checkpoints 13 处出现） | ⚠️ 全部是文件名 `plot_trajectory_plot.png`，**0 处 --paradigm** |

→ **前版叙事是上一个 agent 演绎而非真实 trace**。继续以此叙事为根因建模会修错地方。

## 真根因 4 处契约错乱（按修复优先级）

### 1. SKILL.md 与脚本签名不一致

`packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md:28` 明文：

> "通用图(`_common.plot_timeseries` / `_common.plot_trajectory`)必须追加 `--paradigm <p>`，让脚本按范式选默认 y_col。"

但实际 `packages/ethoinsight/ethoinsight/scripts/_common/plot_trajectory.py` 的 argparse 不接受 `--paradigm`（轨迹图不需要 y_col）。**SKILL 强制 LLM 对错的脚本传错的参数**。这次 843fe2b8 没炸只是因为 chart-maker LLM 在 fallback 路径下违反 SKILL 直接照 plan_charts.json entry 拼了正确命令。

**根治**：脚本签名下沉到 catalog yaml（见 B4 决策）；SKILL.md:28 整行删除。

### 2. chart_maker.py prompt 模板与 fallback 路径冲突

`chart_maker.py:43` + `:67`：

```
bash python -m ethoinsight.scripts.<paradigm>.<chart_script>
```

但 `_common.yaml:7` 注册的是：

```yaml
script: ethoinsight.scripts._common.plot_trajectory
```

`_common` ≠ `<paradigm>`。模板对 fallback 必错、对 catalog 直接命中碰巧对。

**根治**：prompt 改为 `bash python -m <entry.script> <entry.args 拼接>`（files-are-facts，与 code_executor.py prompt 对齐）。

### 3. plan_metrics.json schema 不含 display_name_zh

前版 Q10 子树假设 plan_metrics.json 已含 `display_name_zh`，实际 schema 只有 `id / script / input / output / required / reason / subject_index`，id 是英文 snake_case。`plan_labels.py` 直接遍历会拿到英文 id。

**根治**：`schema.py:PlanMetric / PlanChart` 加 `display_name_zh: str`；`resolve.py` 写入；`plan_metrics_to_dict` / `plan_charts_to_dict` 序列化。catalog yaml 是 single source of truth，prep_metric_plan 把这字段透传到 plan 文件。

### 4. lead 越过 ScriptInvocationOnlyProvider 直接 bash cp

thread 78ccb52b 真实故障序列（checkpoints 解码）：
- chart-maker 把 PNG 写到 workspace/（违反 SKILL/prompt"必须 outputs/"——chart_maker.py 这段强化是 5-19 18:03 才补，78ccb52b 跑在 15:40，那时 prompt 还没补强）
- lead 看到 chart 在 workspace/ → 自己用 `bash cp /mnt/user-data/workspace/*.png /mnt/user-data/outputs/` 收尾
- `ScriptInvocationOnlyProvider` 只挂在 code-executor（executor.py:407 已确认），**不约束 lead**
- LoopDetectionMiddleware 拦截 lead 反复读 handoff
- lead 改派 `general-purpose` subagent 复制文件 → 该 subagent 不存在 → 链断

**根治分两步**：
- catalog/resolve.py PlanChart 的 `output` 路径直接写 `/mnt/user-data/outputs/`（让 plan_charts.json 是真实施工单，chart-maker 照拼即可零推断）
- IntentPipelineConsistencyProvider（独立子树，另起 handoff）守护 lead 派遣一致性

## 关键决策（grill 已确认）

| 议题 | 决策 | 理由 |
|---|---|---|
| 脚本签名落点 | **B4：catalog yaml 加 `accepts_paradigm` 字段，resolve 据此生成 args 数组** | B2（脚本读 plan）破坏 ethoinsight 库独立性；B4 保持库纯净 + single source of truth |
| Chart 输出路径策略 | **PlanChart.output 直接锁 outputs/，不引入 purpose 字段** | 用户明确不要 purpose 枚举；"中间产物 chart"是 v0.1 不需考虑的场景 |
| display_name_zh 链条 | **prep_metric_plan 阶段写入 plan_metrics/plan_charts**（schema 升级） | 链条最短、与 single source of truth 一致；前端 fetch 即得最终中文 |
| `ethoinsight-charts` 孤儿 | **接入 chart-maker `skills=[...]`，作为 L2「图种 know-what」入口** | lead 不持图种 know-how（lead-interaction:48 + lead-prompt:185 W16 注释 已明文）；chart-maker 通过 SKILL 主文 system prompt 注入 + references 渐进披露 |
| `ethoinsight-chart-maker` vs `ethoinsight-charts` 重叠 | **保留两个，charts 留"图种选择决策树"，chart-maker 留"fallback 决策树"** | 职责正交：知道生成什么图 vs 知道怎么生成；走 deerflow 渐进披露最佳实践 |
| `ethoinsight-planning` skill | **删整目录**；quality-gates / failure-recovery 内容并入 lead-interaction/references/ | 与 lead prompt 第 254-258 行 INTENT 状态机重叠 + 无 guardrail 校验；single source of truth 不允许双存 |
| `/plan` 模式（claude-code 风格） | **v0.2+ 再做（present_plan 工具），v0.1 不引入** | E2E_FULL_ASKVIZ 的 `ask(viz?)` 已是软计划展示 |

## PR 拆分（3 个）

### PR-1：catalog schema + yaml 元数据升级（B4 + γ 数据契约）

**独立可测**。schema_version 升 1.1。

| 文件 | 改动 |
|---|---|
| `packages/ethoinsight/ethoinsight/catalog/schema.py` | `ChartEntry` 加 `display_name_zh: str` + `accepts_paradigm: bool = False`；`PlanMetric` 加 `display_name_zh: str`；`PlanChart` 加 `display_name_zh: str` + `args: list[str]` |
| `packages/ethoinsight/ethoinsight/catalog/loader.py` | yaml→ChartEntry 读 `accepts_paradigm`（可选，默认 False）+ `display_name_zh`（必填） |
| `packages/ethoinsight/ethoinsight/catalog/_common.yaml` | `timeseries_plot` 加 `accepts_paradigm: true` + `display_name_zh: "时间序列图"`；`trajectory_plot` 加 `display_name_zh: "轨迹图"`（不加 paradigm） |
| `packages/ethoinsight/ethoinsight/catalog/{fst,oft,epm,zero_maze,shoaling}.yaml` | 每条 chart entry 加 `display_name_zh`；如有需要也加 `accepts_paradigm` |
| `packages/ethoinsight/ethoinsight/catalog/resolve.py` | `_expand_metric_for_subjects` / `_expand_chart_for_subjects` 把 display_name_zh 传入 PlanMetric/PlanChart；chart 展开按 `accepts_paradigm=True` 把 `--paradigm <p>` 加入 args；PlanChart 的 `output` 写 `/mnt/user-data/outputs/...png`；`plan_metrics_to_dict` + `plan_charts_to_dict` 序列化新字段 |
| `packages/ethoinsight/tests/test_catalog_*.py` | 加单测覆盖：display_name_zh 透传 / accepts_paradigm 控制 args / PlanChart.output 落 outputs/ |

### PR-2：chart-maker workflow + skill 渐进披露（α + β）

**依赖 PR-1**。

**完整 spec 已撰写**：[docs/specs/2026-05-22-pr2-chart-maker-workflow-final-spec.md](../specs/2026-05-22-pr2-chart-maker-workflow-final-spec.md)

**核心 4 项**：
1. chart_maker.py `skills=[..., "ethoinsight-charts"]` 挂载孤儿 skill + workflow Step 2.5 read 它形成 L2 渐进披露入口
2. chart_maker.py prompt 模板改 `bash python -m <entry.script> <entry.args 拼接>`（files-are-facts，与 code_executor 对齐）
3. `ethoinsight-chart-maker/SKILL.md` 删除"通用图必须追加 --paradigm"错指示（PR-1 已把签名声明性下沉到 yaml）
4. 两个 SKILL.md 的 description 调整为 "L2 图种 know-what" / "L3 执行 how-to-execute" 分工定位

详见 spec §2 改动清单、§3 实施顺序、§2.4.2 dogfood 验证用例（用 thread 843fe2b8）、§2.5 grep 验证清除项。

### PR-3：INTENT 状态机守护 + lead 边界硬约束 + planning skill 处置

**独立**（不依赖 PR-1/PR-2，但建议放最后以便 e2e dogfood）。

**完整 spec 已撰写**：[docs/specs/2026-05-22-pr3-lead-robustness-final-spec.md](../specs/2026-05-22-pr3-lead-robustness-final-spec.md)

**核心 6 项**：
1. `task_tool.subagent_type` 改 `Literal[...]` 强类型化（schema 硬约束消除"派幽灵 subagent"affordance）
2. lead `disallowed_tools=["bash"]`（5/15 lead-bash-removal 真正硬约束）
3. 新建 `IntentPostStepAskGateProvider`（拦 task(chart-maker) 当 INTENT=E2E_FULL_ASKVIZ + data-analyst 完成 + gate3_viz_acknowledged 未在 experiment-context.json）
4. 新建 `set_viz_choice(choice)` 工具（写 gate3 到 experiment-context.json，沿用 deerflow 既有 gate_completed 范式）
5. `IntentClassificationGuardrailProvider` 假阳 bug 仅诊断打点 + 单测 reproduce（M2 方案，修留 PR-4）
6. 删 `ethoinsight-planning` skill + quality-gates/failure-recovery 搬到 lead-interaction（single source of truth）

详见 spec §2 改动清单、§4 实施顺序、§5 dogfood 验证用例、§附录 A schema 强类型化 spike 笔记。

## 关键文件速查

| 用途 | 路径 |
|---|---|
| 项目 CLAUDE.md | `/home/wangqiuyang/noldus-insight/CLAUDE.md` |
| chart_maker subagent | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py` |
| chart-maker SKILL（要修 :28） | `packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md` |
| ethoinsight-charts SKILL（要接入） | `packages/agent/skills/custom/ethoinsight-charts/SKILL.md` |
| catalog schema（要升级） | `packages/ethoinsight/ethoinsight/catalog/schema.py` |
| catalog resolve（要改） | `packages/ethoinsight/ethoinsight/catalog/resolve.py` |
| _common.yaml（要加字段） | `packages/ethoinsight/ethoinsight/catalog/_common.yaml` |
| plot_trajectory（无需改） | `packages/ethoinsight/ethoinsight/scripts/_common/plot_trajectory.py` |
| ScriptInvocationOnlyProvider | `packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py` |
| lead prompt（INTENT 状态机段） | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:254-280` |
| lead-interaction SKILL | `packages/agent/skills/custom/ethoinsight-lead-interaction/SKILL.md` |
| 5/15 lead-bash-removal handoff | `docs/handoffs/2026-05/2026-05-15-p0-lead-bash-removal-handoff.md` |

## 验证用现场 thread_id

| Thread | 用途 |
|---|---|
| `843fe2b8-ac75-4692-bdb2-f243d4427168` | FST 跑通现场（chart-maker 4 张全成）；验证 PR-1+PR-2 后此 thread 复跑应仍成且 args 数组来自 yaml |
| `78ccb52b-4d16-4a27-9fb3-f1a04b6ee5b2` | lead 越界 bash cp + 派 general-purpose 故障；验证 PR-3 实施后此故障类型应被 IntentPipelineConsistencyProvider 拦截 |

## 下一棵子树 grill（已完成）

**Q5-b**：IntentPipelineConsistencyProvider 设计 grill — 已闭环 5 轮，4 thread fact-check 推翻多个前置假设。最终 PR-3 设计见 [docs/specs/2026-05-22-pr3-lead-robustness-final-spec.md](../specs/2026-05-22-pr3-lead-robustness-final-spec.md)；grill 过程交接见 [2026-05-22-intent-pipeline-consistency-grill-handoff.md](2026-05-22-intent-pipeline-consistency-grill-handoff.md)（仅含起点，5 轮 grill 内容沉淀在 memory `project_2026-05-22_pr3_lead_robustness_grill_conclusions.md`）。

## 前版误判记录（防止再次传播）

前版 handoff（[2026-05-22-tool-vs-script-grill-handoff.md](2026-05-22-tool-vs-script-grill-handoff.md)）的下列陈述**全部失实**，本版取代：

| 前版陈述 | 失实点 |
|---|---|
| "code-executor 跑 plot_trajectory --paradigm fst 失败" | 全仓库无原始 trace；843fe2b8/78ccb52b 均不发生 |
| "LLM 试错-反思-重试全过程暴露给用户" | 演绎，非真实故障模式 |
| "plan_metrics.json 已含 display_name_zh" | schema 实际不含；PR-1 才补 |
| "code-executor 受 L1+L2+L3 三层约束，所有脚本调用 100% 在 plan 内" | L2 ScriptInvocationOnlyProvider 只对 code-executor 生效；lead 不受约束，78ccb52b 即反例 |

教训沉淀：`feedback_grill_handoff_must_be_verified.md`（grill handoff 接手时必须现场核实证据，不能直接基于前版结论继续 grill）。
