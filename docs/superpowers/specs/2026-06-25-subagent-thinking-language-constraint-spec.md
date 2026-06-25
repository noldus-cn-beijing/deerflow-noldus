# Spec：thinking channel 语言约束 —— lead + 4 subagent 思考语言跟随用户语言

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-25
> 代码基线：dev HEAD `b00c39e4`
> 性质：🟡 中 · prompt 约束补强（v0.1 演示观感硬伤）。
> **方向（用户拍板 2026-06-25）**：EPM dogfood（thread `0e72d605`，用户全程中文）暴露——lead + 全部 4 个 subagent 的 **thinking（思考过程）全用英文**，只有最终给用户的输出是中文。用户规则明确：「用户说中文→think 中文，用户说英文→think 英文」。现有语言约束**只管「输出」不管「thinking」**，被模型理解成「思考可随意」。修法=语言约束显式点名 thinking channel，进各 agent 的 system_prompt。
> 受保护文件（sync surgical）：`subagents/builtins/{code_executor,data_analyst,chart_maker,report_writer}.py`、`agents/lead_agent/prompt.py`。

---

## ⚠️ 〇、根因（取证坐实）

> 本 spec 来自 EPM dogfood（thread `0e72d605`，2026-06-25）。已核 5 个 agent 的 system_prompt 语言段措辞坐实，不凭推断。

### 现象（dogfood trace）
用户**全程中文**（"分析数据"、"fewzones，xx是对照组…"、"A. 是，画图"）。但所有 agent 的 thinking 全英文：

| 角色 | thinking 实际语言（trace 逐字） |
|---|---|
| lead | **英文**："Let me check the group sizes... I'll dispatch code-executor." |
| code-executor | **英文**："The file is very large... Let me try run_metric_plan." |
| data-analyst | **英文**："Let me check the groups... compute the statistics." |
| chart-maker | **英文**："113 charts! Let me read the plan_charts.json." |
| report-writer | **英文**："Let me start by reading the required files." |

最终给用户的输出全是中文——**说明模型把语言约束理解成「管最终产出，thinking 随意」**。

### 真根因（逐 agent 核 system_prompt 措辞坐实）

| agent | 现有语言段 | 缺陷 |
|---|---|---|
| code_executor | [code_executor.py:19-21](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py#L19)：`<语言>中文优先，确保你输出的语言一致</语言>` | 只说「**输出**」，未点名 thinking；「中文优先」是模糊倾向非硬约束 |
| chart_maker | [chart_maker.py:15-17](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py#L15)：同上 | 同上 |
| data_analyst | [data_analyst.py:13-20](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py#L13)：「**所有输出**（最终消息、handoff 自由文本）都用同一种语言」 | **显式列举「输出」范围时把 thinking 排除在外**——模型据此认为 thinking 不在约束内 |
| report_writer | [report_writer.py:15-25](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py#L15)：同 data_analyst | 同上 |
| lead | [prompt.py:539-546](../../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L539) `<thinking_style>` 段**全英文写**，无「thinking 跟随用户语言」要求 | thinking_style 段本身英文，且不含语言约束 |

**核心根因**：5 个 agent 的语言约束**全部只覆盖「输出」channel，无一点名「thinking/reasoning」channel**。data_analyst/report_writer 更糟——显式列举输出范围（最终消息/handoff/write_file）时，**反向暗示 thinking 不受约束**。模型（GLM/deepseek 系）thinking 默认倾向英文（训练数据英文为主），无显式约束 → 全英文 thinking。

> 守 memory `feedback_subagent_system_prompt_higher_authority_than_skill`：subagent system_prompt 是最高权威。lead 派遣 prompt 里虽写了「思考过程（thinking）必须中文」（dogfood trace 可见派遣 prompt 含此句），但**被 subagent 自己 system_prompt 的「只管输出」措辞覆盖**——派遣 prompt 权威低于 system_prompt。所以**必须改 system_prompt 本身**，不能只靠派遣 prompt。

---

## 一、给实施 agent 的一句话

把 5 个 agent（lead + 4 subagent）的语言约束**显式扩展到 thinking channel**：用正面指令（守「deepseek 正面提示」原则）声明「**你的思考过程（thinking/reasoning）、最终输出、write_file 内容、handoff 文本，全部使用与用户相同的语言**」。subagent 改各自 `<语言>` 段；lead 改 `<thinking_style>` 段加一句语言约束。TDD 断言各 system_prompt 含 thinking 语言约束。

---

## 二、修法（M1：5 个 agent system_prompt）

### M1a：code_executor + chart_maker（措辞最弱，重写）

**改 [code_executor.py:19-21](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py#L19) + [chart_maker.py:15-17](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py#L15)**：

```
<语言>
使用与用户相同的语言，覆盖你的【全部】产出通道：
- 思考过程（thinking / reasoning）
- 最终消息
- write_file 内容、handoff_*.json 里的自由文本字段
lead 派发任务时会在 prompt 开头声明用户语言；未声明则从任务描述推断（中文任务→全程中文，英文任务→全程英文）。
</语言>
```

> 关键：**第一条就是「思考过程」**——把 thinking 提到约束范围最前，消除「只管输出」的误读。用正面「使用与用户相同的语言」不用「禁止英文 thinking」（守正面提示原则，「禁止 X」会反向激活）。

### M1b：data_analyst + report_writer（已较全，补 thinking 一行）

**改 [data_analyst.py:13-20](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py#L13) + [report_writer.py:15-25](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py#L15)**：在「所有输出…都用同一种语言」**之前**加 thinking 行：

```
<语言>
**思考过程（thinking / reasoning）和所有输出都必须与用户语言一致**：
- lead agent 派发任务时，会在 prompt 开头声明用户使用的语言
- 如果 lead 未明确声明，从任务描述中推断：中文任务用中文、英文任务用英文
- 【思考过程】+ 所有输出（最终消息、write_file 内容、handoff_*.json 里的自由文本字段）
  都用同一种语言
...（report_writer 的 sections_written 固定中文章节名规则保留不变）
```

> report_writer 的 `sections_written` 字段固定中文章节名（[report_writer.py:23-24](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py#L23)）**保留不变**——那是下游消费的固定 key，不跟随用户语言（守现有设计）。

### M1c：lead（thinking_style 段加语言约束）

**改 [prompt.py:539-546](../../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L539) `<thinking_style>` 段**，加一句（中文，因为 lead 主要服务中文用户但需跟随）：

```
<thinking_style>
- Think and reason in the SAME language as the user (用户说中文你就用中文思考，用户说英文就用英文思考). This applies to your thinking/reasoning process, not only the final response.
{subagent_thinking}- Never write down your full final answer or report in thinking process...
...
</thinking_style>
```

> lead 的 thinking_style 段是英文骨架（上游 deerflow 定制点），加一句双语语言约束，不重写整段（守 sync surgical，保留上游骨架）。

---

## 三、TDD（红→绿）

测试文件：扩 `test_chart_maker_system_prompt_budget.py` 思路，新 `test_subagent_thinking_language.py`。

### T1：4 subagent system_prompt 含 thinking 语言约束
```python
def test_t1_subagents_constrain_thinking_language():
    for agent in [code_executor, data_analyst, chart_maker, report_writer]:
        sp = get_system_prompt(agent)
        assert "思考" in sp or "thinking" in sp.lower()
        # 断言语言约束覆盖 thinking，不只是「输出」
        # 红：当前 code_executor/chart_maker 只有「输出」；绿：含「思考过程」
```

### T2：lead thinking_style 含语言约束
```python
def test_t2_lead_thinking_style_constrains_language():
    sp = lead_system_prompt()
    assert "same language" in sp.lower() or "用户说中文" in sp
```

### T3（回归）：report_writer sections_written 固定中文规则保留
```python
def test_t3_report_writer_sections_written_still_fixed_chinese():
    sp = get_system_prompt(report_writer)
    assert "sections_written" in sp and "中文章节名" in sp
```

### T4（导入环）：改 4 subagent + lead prompt 后裸导入两入口
```bash
PYTHONPATH=. python -c "import app.gateway"
PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
```

---

## 四、风险与注意事项

1. **prompt 约束不是 100% 保证**：thinking 语言是模型倾向，强约束能大幅改善但 GLM/deepseek 偶尔仍可能漏英文。**这是 prompt 能做到的上限**——thinking channel 无法像 seal 那样上确定性门（思考内容不是结构化产物，无法机械校验语言）。验收看 dogfood thinking 语言占比改善，不强求 100%。
2. **正面指令优先**：守 `feedback_skill_describing_tool_output_enables_hallucination`——用「使用与用户相同的语言」不用「禁止用英文」。「禁止 X」对 deepseek 系反向激活。
3. **不动 lead 中文播报规则**：[prompt.py:473](../../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L473)「每次 task 前先用简短中文播报」是面向中文用户的现有规则，与本 spec thinking 语言约束正交，不改。
4. **sync surgical**：5 个文件都是受保护文件（含 Noldus 定制 prompt）。改时只动语言段，保留其余定制（守「sync 受保护 prompt 退化成 paraphrase-merge」教训 `feedback_sync_protected_file_paraphrase_merge_weakens_constitution`）。
5. **与 spec C 的 code_executor.py 重叠**：spec C 的 M3 改 code_executor system_prompt 执行流程段，本 spec M1a 改其 `<语言>` 段——**不同段，但同文件**。建议 spec C 和 spec D 顺序合并（先合一个，另一个 rebase），避免 system_prompt 字符串冲突。

---

## 五、实施步骤

1. TDD 红：写 T1，确认 code_executor/chart_maker 当前红（只有「输出」无「思考」）。
2. M1a/M1b/M1c：改 5 个文件语言段，跑 T1/T2/T3 绿。
3. 全量 `make test`（重点 `test_*_system_prompt*` / `test_*_skill*`）+ 裸导入两入口（T4）。
4. dogfood 验收：重跑 EPM dogfood（中文），核 thinking 语言占比——应大幅转中文（不强求 100%，见风险 1）。

---

## 六、与其他 spec 的关系

- 与 spec C（read_file + code-executor）**同源**（同 dogfood thread `0e72d605`）+ **同文件部分重叠**（`code_executor.py`，但改不同段）→ 建议**同 PR 或顺序合并**。
- 与 spec A/B 正交。

---

## 七、milestone 建议

归入「输出宪法 / 交互语言一致性」track。checkpoint：「EPM dogfood（thread 0e72d605，用户全程中文）暴露 lead + 4 subagent 的 thinking 全英文（只有最终输出中文）。取证坐实根因=5 个 agent 语言约束只覆盖「输出」channel、无一点名「thinking」，data_analyst/report_writer 更显式列举输出范围反向暗示 thinking 不受约束。修法=语言约束显式扩展到 thinking channel（正面指令「使用与用户相同的语言」覆盖思考+输出+write_file+handoff），进 5 个 agent system_prompt。**thinking 语言无法上确定性门（非结构化产物），prompt 是上限**，验收看占比改善不强求 100%。」
