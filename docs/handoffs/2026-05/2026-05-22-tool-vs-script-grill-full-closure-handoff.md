# 2026-05-22 tool-vs-script grill 全闭环 + 3 PR 合并完成

> **状态**：grill 已完成，3 PR 已合入 dev。准备 dogfood 验证 + PR-4（IntentClassification 假阳 bug 修复）。
> **上一棒**：前轮 [2026-05-22-tool-vs-script-grill-handoff.md](2026-05-22-tool-vs-script-grill-handoff.md) grill 40% 的交接

## 本次会话完成了什么

### 阶段 1：推翻前轮 handoff 误判（~30% 时间）

前轮 handoff 把 Q 子树（脚本契约一致）的根因建模为"LLM 试错 plot_trajectory --paradigm"。**全仓库证据不支持**：
- 指定 thread 843fe2b8：chart-maker 4 张图全成，0 试错
- 全仓库唯一提到 `plot_trajectory --paradigm` 的文件就是前轮 handoff 自己
- checkpoints.db ormsgpack 解码 0 处 trace

**真根因 4 处**：
1. `ethoinsight-chart-maker/SKILL.md` 明文要求所有通用图必须 --paradigm，但 plot_trajectory argparse 不接
2. chart_maker.py prompt 模板 `<paradigm>.<chart_script>` 与 fallback `_common.plot_*` 路径冲突
3. plan_metrics.json schema 不含 display_name_zh（前轮 Q10 假设含，按实不含）
4. 78ccb52b lead 越过 ScriptInvocationOnlyProvider（只对 code-executor 生效，不约束 lead）直接 bash cp + 派不存在的 general-purpose subagent

### 阶段 2：B4 决策 + skill 渐进披露分工（~20% 时间）

- **B4**：脚本签名声明性下沉到 catalog yaml（`accepts_paradigm` 字段），resolve 据此生成 args 数组 → LLM 不再推导参数
- **L2/L3 分工**：`ethoinsight-charts` = L2 图种 know-what，`ethoinsight-chart-maker` = L3 how-to-execute；lead 完全不持图选择 know-how
- **删除 planning skill**：因与 lead prompt INTENT 7 类状态机重叠（single source of truth）
- **输出路径**：PlanChart.output 锁死 outputs/（放弃 purpose 枚举，用户 Q3-a 明确拒绝）
- **display_name_zh 链条**：catalog yaml → PlanMetric/PlanChart → JSON 序列化（β 方案，prep_metric_plan 阶段写入字段）

### 阶段 3：Q5-b INTENT pipeline consistency grill + fact check（~30% 时间）

4 个现代 thread（843fe2b8/189e7840/f3fbce44/7456611e）全量 ormsgpack 解码：

| 发现 | 影响 |
|---|---|
| 4/4 thread INTENT 分类全对（全是 E2E_FULL_ASKVIZ），**没有"派错人"证据** | 推翻"IntentPipelineConsistency 拦派遣"的设计前提 |
| 4/4 thread 全都**跳过 ask(viz?) 直接派 chart-maker** | 真问题是「该 ask 时没 ask」，不是「派错谁」 |
| 2/4 thread lead 反复越界（bash 135 次 + general-purpose 113 次）| deny 信息完备但 LLM 知错不改 |
| 2/4 IntentClassification 反复假阳 reject（574 次），但不阻塞执行 | bug 存在但不 priority |
| summarization 未触发（60k token 阈值未到）—— **ASKVIZ 规则未被压掉** | 不是归档问题 |

### 阶段 4：3 PR spec 撰写 + 验证（~20% 时间）

3 份完整 spec + 3 份派遣 prompt，用户交给 other agent 实施。实施后本会话验证全正确。

## 3 个 PR 的核心改动（全部已合入 dev）

### PR-1 (#28): catalog schema 1.0→1.1

- `ChartEntry` 加 `display_name_zh` + `accepts_paradigm`；`PlanMetric`/`PlanChart` 加新字段
- `_common.yaml` timeseries_plot 加 accepts_paradigm=true
- `resolve.py:PlanChart.output` 改为 `/mnt/user-data/outputs/`；按 accepts_paradigm 生成 --paradigm arg
- schema_version 升 "1.1"
- ethoinsight 测试：345/0/51 ✅

### PR-2 (#29): chart-maker workflow + skill 渐进披露

- chart_maker.py `skills=[..., "ethoinsight-charts"]` 挂孤儿
- prompt 模板改 `python -m <entry.script> <entry.args 拼接>`（files-are-facts，对齐 code_executor）
- workflow 加 Step 3（read ethoinsight-charts/SKILL.md 作为渐进披露入口）
- `ethoinsight-chart-maker/SKILL.md` 删除 "通用图必须追加 --paradigm" 错指示
- 两个 SKILL description 改写：charts = L2 know-what，chart-maker = L3 how-to-execute

### PR-3 (#30): lead 鲁棒性

- `task_tool.subagent_type` 改为运行时 Literal[...] 强类型化（enum = chart-maker/code-executor/data-analyst/report-writer/knowledge-assistant，不含 general-purpose）
- `_LEAD_EXCLUDED_TOOLS = frozenset({"bash", "write_file", "str_replace"})`
- 新建 `IntentPostStepAskGateProvider`：拦 task(chart-maker) 当 INTENT=E2E_FULL_ASKVIZ + data-analyst 完成 + gate3 未 ack；deny 消息含"请改调 ask_clarification(...) 因为 ..."
- 新建 `set_viz_choice(choice)` 工具：写 gate3_viz_acknowledged 到 experiment-context.json
- 删 `ethoinsight-planning` skill，quality-gates/failure-recovery references 搬到 lead-interaction
- IntentClassification 诊断打点 + 假阳 repro 单测（M2：diagnose only，修留 PR-4）
- Agent 测试：582/1，57 个新测试全绿

### 唯一个失败

`test_client_live.py::test_agent_uses_bash_tool` —— IntentClassification 拦截 lead 首次 bash（正确行为），测试 `tr_events[0]` 断言拿 deny ToolMessage 而非 `LIVE_TEST_OK`（预期内）。应更新测试跳过 deny 消息。

## 当前仓库现场

```bash
git log --oneline -5
# eb724699 Merge PR #30 (PR-3 lead robustness)
# a3e9e677 IntentClassification 诊断打点
# 4f187478 lead prompt 追加 set_viz_choice 指引
# 3c7b74bb IntentPostStepAskGateProvider
# 2f4e7ffe set_viz_choice 工具

git stash list
# stash@{0}: 本地有冲突的临时文件（merge 前 stashed）

git status --short
# 未提交的新文件（本会话产物）：
# - docs/handoffs/2026-05/*.md (3 个 handoff + 纠正版 + 前轮原版)
# - docs/specs/2026-05-22-pr{1,2,3}-*.md (3 个 spec)
# - docs/problems/2026-05-22-fst-e2e-issues-and-solutions.md
# - docs/refs/2026-05-22-mousegpt-paper-review.md
# - CLAUDE.md (modified)
```

## 关键决策记录

| 决策 | 时间 | 理由 |
|---|---|---|
| B4（plan 带 args，yaml 声明 accepts_paradigm）| Round Q3-b | B2 破坏 ethoinsight 库独立性；B4 保持库纯净 + single source of truth |
| 删 planning skill | Round Q5-a | 与 lead prompt INTENT 7 类重叠 + 无 guardrail 校验；SSOT 要求只有一份 |
| 不引入 chart purpose 枚举 | Round Q3-a | 用户明确拒绝（"不必"） |
| M2 方案（IntentClassification bug 诊断不修）| Round Q5-b | 不阻塞新 ask gate；574 假阳不阻塞执行 |
| lead 完全 disallow bash | Round Q5-b | 4/4 thread 主流水线没合法 bash 用例；遇合法需要再放开 |
| experiment-context.json gate_completed 范式 | Round Q5-b | deerflow 既有范式，跨 turn 持久、归档不丢、单调累加 |
| deny 消息必须含「请改用 X 因为 Y 然后做 Z」| Round Q5-b | 189e7840 实证：信息完备 deny 不足引导 LLM 自我纠正 |
| 单 PR 拆 3 PR（catalog / workflow / lead）| Grill 过程中 | PR-1 独立可测（纯库），PR-2 依赖 PR-1，PR-3 独立 |

## 未提交的产物文件

本会话产生的文档，尚未 commit，需要提交到 dev：

```bash
cd /home/wangqiuyang/noldus-insight

# 已提交的（来自 other agent 的 PR）：
# PR-1~3 的代码 + 测试

# 未提交的（本会话写的文档）：
git add docs/handoffs/2026-05/2026-05-22-tool-vs-script-grill-handoff.md \
        docs/handoffs/2026-05/2026-05-22-chart-maker-grill-corrected-handoff.md \
        docs/handoffs/2026-05/2026-05-22-intent-pipeline-consistency-grill-handoff.md \
        docs/handoffs/2026-05/2026-05-22-fst-infra-fixes-handoff.md \
        docs/specs/2026-05-22-pr1-catalog-schema-upgrade-final-spec.md \
        docs/specs/2026-05-22-pr2-chart-maker-workflow-final-spec.md \
        docs/specs/2026-05-22-pr3-lead-robustness-final-spec.md \
        docs/problems/2026-05-22-fst-e2e-issues-and-solutions.md \
        docs/refs/2026-05-22-mousegpt-paper-review.md
git add CLAUDE.md docs/handoffs/2026-05/2026-05-21-deerflow-sync-complete-and-next-handoff.md
git commit -m "docs: 5/22 3 PR spec + grill handoff 纠正 + 问题记录"

# stash 中有 merge 冲突导致的临时文件（扩展配置 + lead-interaction SKILL）
# PR 合入后 dev 已有更新版，locally stashed 不再需要：
#   git stash drop stash@{0}  # 如果确认不需要
```

## 下一步建议

### 优先级 1：dogfood 验证（立即可做）

运行服务后，用 thread 843fe2b8 相同的用户输入（"帮我分析一下大鼠强迫游泳的实验数据"）重跑，验证：

1. lead 声明 `[intent] E2E_FULL_ASKVIZ`
2. code-executor → data-analyst **完成**
3. **IntentPostStepAskGate 拦截第一次 chart-maker 派遣**（如果还未 ask）→ 强引导 deny 指示 lead 调 ask_clarification
4. lead 调 ask_clarification → 模拟用户回答 "yes" → lead 调 set_viz_choice
5. chart-maker 成功（args 数组来自 plan_charts.json，含 --paradigm fst 仅对 timeseries）
6. PlanChart.output 已是 outputs/，chart_files 全在 outputs/
7. lead 不再有 bash 工具可用（工具列表中无 bash）

### 优先级 2：修复 test_client_live.py 失败

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
.venv/bin/python -m pytest tests/test_client_live.py::TestLiveToolUse::test_agent_uses_bash_tool -v --tb=long
```

问题：`tr_events[0]` 现在是 IntentClassification deny ToolMessage，不是 `LIVE_TEST_OK`。修法：在 `tr_events` 中找第一个 content 含 `LIVE_TEST_OK` 的 ToolMessage。

### 优先级 3：PR-4（IntentClassification 假阳 bug 修复）

PR-3 已加诊断打点 + repro 单测（`tests/test_intent_classification_false_positive.py` 8 个 test）。修法待确定——真因可能在 ContextVar 跨 async/sync 边界或 ArchivingSummarization 把 [intent] 行压进 summary 后 IntentClassification 扫不到。

启动前先跑 repro 单测：如果 8 个已全绿，说明问题不在单测覆盖的路径中，需要线上生产复现。

## 所有 spec 速查

| 文件 | 内容 |
|---|---|
| [docs/specs/2026-05-22-pr1-catalog-schema-upgrade-final-spec.md](docs/specs/2026-05-22-pr1-catalog-schema-upgrade-final-spec.md) | PR-1 spec：schema v1.1 + display_name_zh + accepts_paradigm + 验证 |
| [docs/specs/2026-05-22-pr2-chart-maker-workflow-final-spec.md](docs/specs/2026-05-22-pr2-chart-maker-workflow-final-spec.md) | PR-2 spec：chart_manager.py prompt 模板 + skill 渐进披露 + 验证 |
| [docs/specs/2026-05-22-pr3-lead-robustness-final-spec.md](docs/specs/2026-05-22-pr3-lead-robustness-final-spec.md) | PR-3 spec：task_tool Literal + IntentPostStepAskGate + set_viz_choice + 验证 |
| [docs/handoffs/2026-05/2026-05-22-chart-maker-grill-corrected-handoff.md](docs/handoffs/2026-05/2026-05-22-chart-maker-grill-corrected-handoff.md) | PR 拆分总览 + 前轮误判记录（取代前轮版）|

## 所有 memory 条目（跨会话生效）

| 记忆 | 短描述 |
|---|---|
| `feedback_grill_handoff_must_be_verified.md` | grill handoff 必须先现场核实证据，不能基于前轮结论直接继续 |
| `feedback_deny_messages_must_direct.md` | deny 必须含"请改用 X 因为 Y 然后做 Z"，不能裸列 Available |
| `project_2026-05-22_chart_maker_grill_true_root_cause.md` | chart-maker 4 处契约错乱 + B4 + 删 planning + 3 PR |
| `project_2026-05-22_pr3_lead_robustness_grill_conclusions.md` | PR-3 grill 5 轮结论 + deerflow infra 调研 + 4 thread fact-check |
| MEMORY.md 4 条修正 | 已更新 index |
