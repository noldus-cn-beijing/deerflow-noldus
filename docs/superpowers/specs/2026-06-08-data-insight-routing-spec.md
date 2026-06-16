# Spec A — 数据判读职责收窄：初次洞察归 data-analyst，knowledge-assistant 不从零判读

> 日期：2026-06-08 ｜ 目标分支：从 `dev` 新建 worktree（独立于 Spec B / Spec C）
> 来源：EPM n=1 dogfood 第三轮（thread `7d4d9b8e`）根因 A
> 性质：**职责收窄 + 路由澄清**，不是"加路由规则"。核心是删除 knowledge-assistant 的越界能力 + 让 lead 把初次数据判读导向 data-analyst。
> 这是给执行 agent 的施工单，不是给用户的总结。

---

## 0. 设计哲学（执行 agent 必读，决定改法方向）

本次连环 bug 的元教训：**不要用"加 prompt 规则 / 加 guardrail 特判"打补丁**——每个补丁制造下一个冲突点（Spec2 加 n=1 特判 → 本轮冒出 n=1 路由洞）。

本 spec 的方向是**减少职责重叠**，不是增加路由规则：
- knowledge-assistant 本就**不该**从零产出数据判读（那是 data-analyst 的活）。删除这个越界能力，比加一条"n=1 时不要派 knowledge-assistant"的规则更治本。
- data-analyst **已经**自带行为学知识 skill（已核实，见 §1），它做判读不丢知识、且受输出宪法约束。

---

## 1. 背景与证据（已现场核实）

### 根因 A：初次"数据洞察"被误路由给 knowledge-assistant

**现象**（thread 7d4d9b8e）：n=1 场景，data-analyst 被跳过（Spec2 正确行为）。用户首次发"帮我进行数据洞察"，lead 判为 QA_FACT → 派 **knowledge-assistant** 从零产出完整判读（`knowledge_response.md`，7548 字节）。该文件含 **14 处输出宪法禁止词**（正常范围×4/焦虑样行为×4/焦虑水平×4/高焦虑×1/参考范围×1），污染下游 report-writer（详见 Spec C 的连锁）。

### 已核实的关键事实

1. **data-analyst 自带行为学知识，不需要调别人**（`subagents/builtins/data_analyst.py`）：
   - line 386：`skills=["ethoinsight", "ethoinsight-metric-catalog", "ethovision-paradigm-knowledge"]`
   - line 146-148：prompt 教它 `read_file by-experiment/<paradigm>.md`（epm→epm.md）拿范式判读知识
   - line 116：它**已被要求读输出宪法**（`output-constitution.md`）
   - line 11：自我定位"具有深厚的 Noldus 领域知识"
   - **核实**：`skills/custom/ethovision-paradigm-knowledge/references/by-experiment/epm.md` 真有 EPM 判读知识（45 行，11 处判读关键词）。
   - **结论**：知识通过 **skill 渐进披露**（read_file）到达 data-analyst，不需要"subagent 调 subagent"。你担心的"data-analyst 拿不到 Noldus 知识"在现状已解决。

2. **data-analyst 本就支持 n=1**（`data_analyst.py`）：
   - line 80：`n<2 → Emit handoff status="partial". Report descriptive statistics only.`
   - line 289：`statistical_validity="skipped"`（单样本/n<2）→ 按"不做组间推断"路径解读
   - **结论**：n=1 时 data-analyst 完全能做描述性判读并 seal partial（Spec1 已让 partial 合法）。

3. **knowledge-assistant 的"场景 A"是越界点**（`subagents/builtins/knowledge_assistant.py`，prompt 约 line 33-40）：
   ```
   ### 场景 A：基于已有分析结果的追问
   用户之前已经完成了数据分析，现在对结果有疑问。
   - read_file ... handoff JSON ... 结合 handoff 中的具体数据 + 领域知识回答
   - 例如："这个 p 值为什么不显著"、"NND 偏高说明什么"
   ```
   - **关键区分**：场景 A 的**正当用途**是"对**已完成**判读的概念追问"（用户已有 data-analyst 结论，再问'为什么 p 不显著'）。
   - **被滥用的场景**：本次是用户**首次**要判读（还没有任何 data-analyst 结论），lead 却把它当 QA_FACT 派给 knowledge-assistant，让后者**从零产出完整判读**。这超出了"解释已有结果"的范围。
   - knowledge-assistant **不读输出宪法**（grep `knowledge_assistant.py` 对 `output-constitution`/`焦虑样`/`正常范围`/`绝对阈值` 零命中）→ 它从零判读时必然产违禁词。

---

## 2. 改动清单

### 改动 A1：收窄 knowledge-assistant 场景 A —— 只解释已有结论，不从零产判读

**文件**：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py`

把"场景 A"的描述从"基于已有分析结果的追问"**收窄并明确边界**。改后场景 A 必须明确：
- 它的输入**前提**是 workspace 已存在 `handoff_data_analyst.json`（或等价的已完成判读结论）。
- 它只**解释/补充**已有判读（回答"为什么"、"这个术语什么意思"、"这个数值在领域里的一般含义"），**不从零产出完整的多指标数据判读**。
- 若 lead 派来的任务是"对一批数据做完整判读/洞察"而 workspace **没有** data-analyst handoff → 这是**派错了**，knowledge-assistant 应在回复里**明确说明"完整数据判读应由 data-analyst 完成"**，只给有限的、不含绝对判读的概念性说明，不写一份完整判读报告。

> 实现提示：在场景 A 段落补充上述边界。用**正面措辞**（CLAUDE.md §6 deepseek 正面提示原则），例如：
> ```
> ### 场景 A：对【已完成】分析结果的概念追问
> 前提：workspace 已有 data-analyst 的判读结论（handoff_data_analyst.json）。用户对【已得出的结论】有疑问。
> - 你的职责是【解释】已有结论：回答"为什么这个 p 值不显著""这个术语什么意思""这个指标在领域里一般反映什么"。
> - read_file lead 授权的 handoff JSON，结合领域知识【解释】，而非【重新生成】一份完整判读。
> - 若 workspace 尚无 data-analyst 判读、而任务要求"对整批数据做完整洞察/判读"：完整的数据判读由 data-analyst 负责。此时你只提供概念性背景说明（不下绝对结论、不逐指标判读），并在回复开头说明"完整判读建议由 data-analyst 完成"。
> ```

> **注意**：不要删掉场景 A——"解释已有结论"是 knowledge-assistant 的正当价值。只收窄"从零产完整判读"这个越界用法。

### 改动 A2：lead 路由 —— "对本批数据的判读/洞察" → data-analyst（含 n=1）

**文件**：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

lead 当前把"帮我进行数据洞察"判成 QA_FACT → knowledge-assistant（line 1020 "追问/闲聊/概念问题"→knowledge-assistant；line 1029 `QA→knowledge-assistant`）。需要让 lead 区分：

- **初次数据判读/洞察**（用户要对**本批上传数据**做专业解读，且 workspace 尚无 `handoff_data_analyst.json`）→ **派 data-analyst**。
  - n=1 / n<2 时同样派 data-analyst（它走 partial 描述性路径，见 Spec1 已让 partial 合法）。这覆盖了 n=1 fast-path 此前"跳过 data-analyst"留下的洞察真空：**fast-path 跳过的是"自动流水线里的 data-analyst 步骤"，但用户【主动】要洞察时仍应派 data-analyst**。
- **对已有判读的概念追问**（workspace 已有 data-analyst 结论，用户问"为什么 p 不显著"）→ knowledge-assistant 场景 A。
- **纯通用知识**（"什么是 EPM"、"Noldus 有哪些产品"）→ knowledge-assistant 场景 B（QA_KNOWLEDGE）。

> 实现提示：找到 lead prompt 里 QA_FACT / QA_KNOWLEDGE 的判定段（line 247、1020、1029 附近）+ n=1 fast-path 段（line 1007-1016）。补充判定逻辑（正面措辞）：
> - 关键区分信号：**workspace 是否已有 `handoff_data_analyst.json`**。
>   - 无 + 用户要"判读/洞察/解读/分析这批数据" → data-analyst（这是初次判读，data-analyst 的活）。
>   - 有 + 用户问"为什么/这个术语/一般含义" → knowledge-assistant 场景 A（解释已有结论）。
> - 在 n=1 fast-path 段补一句：**fast-path 自动跳过 data-analyst 仅指"自动流水线"；若用户随后主动要数据洞察/判读，仍派 data-analyst（走 partial）**，不要派 knowledge-assistant 从零判读。
> - **不要**新增一个 intent 枚举（避免膨胀）。用 workspace 状态 + 用户语义在现有 QA_FACT 判定里区分即可。

> ⚠️ 措辞红线：deepseek 正面提示。不要写"不要派 knowledge-assistant"（反向激活），要写"数据判读派 data-analyst"。

### 改动 A3（可选，建议做）：knowledge-assistant 的 `when_to_use` 同步收窄

`knowledge_assistant.py` 的 `when_to_use`（约 line 76）当前写：
```
- 已有分析结果,用户追问'为什么 p 不显著' / 'NND 偏高说明什么'(QA_FACT)
```
建议补充"前提是已有 data-analyst 判读；完整初次判读不在此列，派 data-analyst"。让 lead 读 subagent 描述时就有正确预期。

---

## 3. 测试（TDD）

放 `packages/agent/backend/tests/`。这部分主要是 prompt 行为，难做纯单测，但有可断言的结构点：

### 测试 A（prompt 内容回归，防止改歪）
新建或追加 `test_routing_data_insight.py`：
```python
def test_knowledge_assistant_scene_a_scoped_to_explaining_existing():
    """knowledge-assistant 场景A 必须声明'前提是已有判读结论 + 只解释不重新生成'。"""
    from deerflow.subagents.builtins.knowledge_assistant import KNOWLEDGE_ASSISTANT_CONFIG  # 按实际导出名
    prompt = KNOWLEDGE_ASSISTANT_CONFIG.system_prompt
    # 收窄信号：场景A 提到"已完成/已有"判读 + "完整判读由 data-analyst"
    assert "data-analyst" in prompt
    assert ("已完成" in prompt or "已有" in prompt)

def test_lead_routes_initial_insight_to_data_analyst():
    """lead prompt 必须包含'初次数据判读→data-analyst（含n=1）'的指引。"""
    from deerflow.agents.lead_agent.prompt import <实际 prompt 常量或加载函数>
    text = <加载 lead system prompt>
    assert "data-analyst" in text
    # n=1 仍派 data-analyst 做洞察的指引存在
    # （执行 agent 按实际措辞写断言，关键是 workspace 无 handoff_data_analyst 时初次洞察导向 data-analyst）
```
> 执行 agent：先读 `knowledge_assistant.py` / `prompt.py` 确认 config 的真实导出名和 prompt 获取方式，按实际写断言。这些是"防回归锚点"，不求覆盖全部语义。

### 测试 B（行为验证，dogfood 复跑）
prompt 类改动最终靠 dogfood 验证（见 §5）。单测只锁结构，不锁语义。

---

## 4. 影响面与风险

- **不丢知识**：data-analyst 自带 paradigm-knowledge skill，收归判读后知识照样到达（read_file epm.md）。
- **不破坏场景 A 正当用途**：用户对已有结论追问"为什么"仍走 knowledge-assistant。
- **与 Spec B 的关系**：Spec B 松绑宪法定性术语后，data-analyst 产判读会更顺（不用在"焦虑样行为"上纠结）。两 spec 互补但独立：A 管"谁判读"，B 管"判读用什么词"。
- **与 Spec C 的关系**：A 修好后，违宪的 knowledge_response.md 不再进入 report-writer 输入 → C 的触发路径之一被堵。但 C 仍需独立兜底（seal 黑洞是底层机制问题）。
- **风险**：lead 路由是 prompt 行为，deepseek 可能仍偶尔误判。缓解：测试锁结构 + dogfood 验证 + 测辞用 workspace 状态（确定性信号）而非纯语义。

---

## 5. 验收标准

1. 结构测试绿（§3 测试 A）。
2. 全量回归（改 prompt 不影响逻辑，但跑一遍确认无 import/加载错误）：`cd packages/agent/backend && make test`（已知 4 污染 + config.yaml 依赖测试需 symlink，见下）。
   - ⚠️ worktree 需 `ln -sf /home/wangqiuyang/noldus-insight/packages/agent/config.yaml <worktree>/packages/agent/config.yaml`，否则 ~5 个 config 依赖测试假红（上轮实证）。
3. **dogfood 验证（关键）**：重跑 EPM n=1，用户发"帮我进行数据洞察"：
   - 期望：lead 派 **data-analyst**（不是 knowledge-assistant），data-analyst 产 `handoff_data_analyst.json`（status=partial，受宪法约束的描述性判读）。
   - 反例不该出现：knowledge-assistant 从零写 `knowledge_response.md` 完整判读。
4. **不新增 intent 枚举**、**不删场景 A**、**措辞正面**。

---

## 6. 提交

- worktree 名建议：`worktree-data-insight-routing-to-data-analyst`
- commit message（中文）：`fix(routing): 初次数据判读归 data-analyst，knowledge-assistant 场景A 收窄为只解释已有结论`
- 全量绿（除已知污染）后建 PR 合入 dev。

---

## 7. 关联

- dogfood findings：`docs/handoffs/2026-06/2026-06-08-epm-dogfood-findings.md`
- memory：`project_2026-06-08_epm_dogfood_routing_and_constitution_leak.md`（根因 A/B/C 全链）
- Spec B（宪法术语对齐）：`docs/superpowers/specs/2026-06-08-output-constitution-term-alignment-spec.md`
- Spec C（seal harness 硬保障）：`docs/superpowers/specs/2026-06-08-seal-harness-hard-guarantee-spec.md`
- data-analyst n=1 partial 已合法：Spec1（`2026-06-08-handoff-status-partial-and-file-perms-spec.md`）
- 知识到 subagent 走 skill 渐进披露：CLAUDE.md 第 12 条 "Skill 渐进披露"；`feedback_subagent_consumption_via_first_party_tool.md`（subagent 消费走 first-party 模式）
