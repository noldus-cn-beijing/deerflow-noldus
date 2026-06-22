# Spec：chart 全画 vs 代表性子集由 lead 反问，不再由 chart-maker 擅自限流

> **状态**：待实施
> **来源**：EPM dogfood thread `339512dd`（28 subject，用户事后质疑「上传这么多数据为什么只画了几张」）
> **优先级**：🟡 P1 产品行为（不是崩溃，但默认行为对认真分析的研究员不合理）
> **依赖**：与 `2026-06-22-chart-representative-subset-group-balanced`（#3，已实施）正交但相关——#3 修「限流时子集不均衡」，本 spec 修「该不该默认限流」。

## 〇、给实施 agent 的一句话

当用户出图意图「未明确指定全画还是子集」且 subject 数 > 阈值时，**lead 在派遣 chart-maker 前多问一档**「全部个体图 vs 代表性子集」，把答案作为 `chart_budget`（或省略=全画）传进派遣 prompt。**chart-maker 不再自行决定 `chart_budget`**——它只执行 lead 给定的预算。反问由 lead 发起（subagent 不能 HITL）。

## 一、根因（实证，dogfood thread `339512dd`）

1. **工具默认是全画**：`prep_chart_plan` 的 `chart_budget` 默认 `None` = 不限制（[prep_chart_plan_tool.py:64,88](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_chart_plan_tool.py)）。
2. **是 chart-maker（subagent LLM）擅自传了 `chart_budget=8`**：`ethoinsight-chart-maker/SKILL.md` 第 26 行 + `fallback-decision-tree.md` 指示它「用户没明确要『所有个体图』时用预算（如 6-8）只画代表性子集」。
3. 用户当时只说「A. 把结论画成图」→ 意图被判「未明确指定」→ chart-maker 走代表子集 → budget=8 → 28 subject 的 113 张图被砍到 8 张（每种个体图只剩 2 个 subject），105 张进 `charts_budget_remaining[]`。
4. **后果**：用户上传 28 个样本、跑完完整流水线，却只看到每类 2 张个体图，且（在 #3 修复前）全偏向 control 组。用户合理地认为「图不全」。

**职责错位**：subagent 替用户做了「画多少」这个本该用户定的决策。这违反项目 HITL 铁律（判断/决策归用户，工具不替用户选——见 memory `feedback_oft_single_zone_must_ask_not_guess`、lead prompt L480 范式必须反问）。

## 二、设计

### 2.1 反问归属：lead，不是 chart-maker

subagent 在本 harness 里**不能调 `ask_clarification`**（只有 lead 链挂了 `ClarificationMiddleware`）。所以反问必须在 lead 派遣 chart-maker **之前**完成。chart-maker 退化为纯执行者：lead 给 `chart_budget` 就按它筛，不给就全画。

### 2.2 何时反问（避免每次都问、避免无意义地问）

只在**同时满足**时反问：
- 用户出图意图未明确「全画 / 子集」（即没说「所有个体图」「每只都画」也没说「代表性几张就行」），且
- per_subject 图会很多（`total_subjects > N_ASK_THRESHOLD`，建议 `N_ASK_THRESHOLD = 6`——小样本全画无所谓，不值得多问一轮）。

否则：
- subject 数 ≤ 阈值 → 直接全画（不问，反正不多）。
- 用户已明确「全画」或「代表性子集就行」→ 按其意愿，不问。

> lead 在这一步已知 `total_subjects`（来自 `handoff_code_executor.json` / `plan_metrics.json`），可以问得具体：「这批 28 个个体，个体图（轨迹/热区等）要**全画**（每个个体各一张，约 N 张）还是**代表性子集**（每组前 K 个）？」

### 2.3 反问与现有 E2E_FULL_ASKVIZ 的关系

现有 `E2E_FULL_ASKVIZ` 已在 data-analyst 完成后问「要不要出图 A/B」。本 spec 的「全画/子集」问题应：
- **E2E_FULL_ASKVIZ 路径**：用户答「要图」后，若命中 2.2 条件，**在同一轮或紧接一轮**追问「全画/子集」（不要拆成两次独立中断，体验差）。可合并成一个多选：「要出图吗？① 全部个体图 ② 代表性子集 ③ 不用」。
- **E2E_FULL 路径**（用户明确说了「画」）：此时用户要图已确定，但「全画/子集」仍可能未定。若命中 2.2 条件仍需追问「全画/子集」一次（除非用户原话已含「所有/每只」「几张代表」等限定词）。

### 2.4 chart-maker 侧改动

- SKILL.md 第 26 行 `chart_budget` 说明改为：「**`chart_budget` 由 lead 在派遣 prompt 中给定**（lead 已就『全画/子集』反问过用户）。派遣 prompt 含明确预算数字 → 用它；派遣 prompt 说『全画/所有个体图』或未给预算 → 省略 `chart_budget`（全画）。**不要自行揣测一个预算数字。**」
- `fallback-decision-tree.md` 同步：去掉「用户没明确要所有个体图时自行用 6-8 预算」的自主限流指示。

### 2.5 lead 把答案传给 chart-maker

lead 派遣 chart-maker 的 task prompt 增加一行明确预算意图，例如：
- 用户选全画 → prompt 写「用户要全部个体图，prep_chart_plan 省略 chart_budget（全画）」。
- 用户选子集 → prompt 写「用户要代表性子集，prep_chart_plan 传 chart_budget=<N>」（N 由 lead 按组数 × 每组代表数定，或固定默认）。

## 三、改动清单

| 文件 | 改动 |
|---|---|
| `agents/lead_agent/prompt.py` | E2E_FULL_ASKVIZ / E2E_FULL 出图分支：命中 2.2 条件时加「全画/子集」反问档；把用户选择写进 chart-maker 派遣 prompt 的预算意图 |
| `skills/custom/ethoinsight-chart-maker/SKILL.md` | 第 26 行 `chart_budget` 说明改为「由 lead 给定，不自行揣测」 |
| `skills/custom/ethoinsight-chart-maker/references/fallback-decision-tree.md` | 去掉 chart-maker 自主限流指示 |

## 四、测试（TDD）

1. **lead prompt 契约测试**（`backend/tests/test_lead_agent_prompt.py` 扩展）：断言出图分支 prompt 含「全画/代表性子集」反问语义 + 「subject 数多时」触发条件文字。
2. **chart-maker skill 契约测试**（`backend/tests/test_chart_maker_skill.py` 扩展）：断言 SKILL.md 不再含「自行用 6-8 预算」类自主限流文字，且含「chart_budget 由 lead 给定」。
3. **行为回归（可选，偏集成）**：模拟「未明确意图 + total_subjects=28」→ lead 走反问；「total_subjects=3」→ 直接全画不问。

## 五、验收标准

- 用户上传多个 subject、出图意图未明确「全画/子集」时，**lead 会反问**，不再静默砍到 8 张。
- 用户选「全画」→ chart-maker 省略 chart_budget → 113 张全画。
- 用户选「子集」→ chart-maker 按 lead 给的预算筛（且按组均衡，#3 已保证）。
- 小样本（≤ 阈值）不触发额外反问。

## 六、风险与注意事项

1. **别把反问拆成多次中断**——和 E2E_FULL_ASKVIZ 的「要不要图」合并成一个多选，避免连续两次打断用户。
2. **deepseek 正面提示**：lead prompt 写「当 X 且 Y 时，反问用户选全画或子集」，chart-maker skill 写「按 lead 给定的预算执行」，不要写「不要自己设预算」。
3. **全画的代价要让用户知情**：反问选项里点明「全部 ≈ N 张，生成更久」，让用户带成本做选择。
4. 守 SSOT：`chart_budget` 的语义 SSOT 仍在 `select_charts_by_priority` + prep_chart_plan_tool，本 spec 只改「谁决定预算值」，不改预算筛选算法本身。

## milestone 建议

本 spec 属 chart-maker 交互完善，可与 #3（组均衡）、chart-maker 丢参修复（#1）合并到「EPM dogfood 图表流水线打磨」milestone 摘要里。
