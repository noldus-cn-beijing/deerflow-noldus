# Spec：prompt / 工具文案里的硬编码数字根治 —— 修 code-executor「2 个指标」嘴瓢 + 全量扫描修复

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-22
> 性质：prompt 纪律 + 文案 SSOT。两类硬编码病：① **工具文案里裸写会漂移的数字**（「v0.1 已实现 **5** 个范式」的「5」是字面量，权威源 `SUPPORTED_PARADIGMS_V01` 在旁却没引用）—— 范式清单变了数字不变 → 漂移；② **subagent prompt 没约束「摘要里的数字必须从结构化数据取」**，导致 code-executor 在 handoff 摘要里自由生成「2 个指标 × 28 × 2」嘴瓢（实际 5 个指标，5×28=140）。两者同构：**数字写死在文本里，不跟随权威来源**。
> 关联：
> - 调研来源：`docs/problems/2026-06-22-thread-3a41e483-four-issues-investigation.md` C 段（**文档原判「code-executor 硬编码了 2 个指标模板」有误**——grep 证实 code_executor.py 无此硬编码，是 LLM 嘴瓢；但扫描发现 identify_ev19 / prep_metric_plan 有真硬编码）
> - SSOT：memory `feedback_single_source_of_truth`、`feedback_skill_describing_tool_output_enables_hallucination`（prompt 用正面指令）

---

## 〇、给实施 agent 的一句话

两类问题：

**A 类（工具文案裸数字，真硬编码）**：`identify_ev19_template_tool.py:458/463` 的 `f"v0.1 已实现 5 个范式"` 和 `prep_metric_plan_tool.py:92` 的 `v0.1 仅支持以下 5 个` —— 「5」是字面量，权威源 `SUPPORTED_PARADIGMS_V01`（[ev19_facts.py](../../../packages/ethoinsight/ethoinsight/ev19_facts.py)）就在旁边却没引用。v0.1 扩到 6 个范式时这两处不会自动跟 → lead / 用户看到错的「5 个」。

**B 类（subagent 摘要嘴瓢，prompt 无约束）**：code-executor handoff 摘要写「总任务数: 140（**2 个指标** × 28 × 2）」，实际是 5 个指标。`code_executor.py` prompt **没有这个硬编码**（grep 证实），是 deepseek 生成摘要文本时自由估算嘴瓢，prompt 缺「数字必须从 plan 结构化字段取、禁止估算」的约束。lead 自己 grep plan 发现并纠正了（无下游影响），但这是纪律缺口。

治本：A 类把裸数字改成引用权威源（`len(SUPPORTED_PARADIGMS_V01)` / 动态生成清单）；B 类给 code-executor prompt 加「摘要数字纪律」铁律。另**全量扫描所有 subagent prompt + 工具文案**，把同类硬编码一次清干净。

---

## 一、根因（实证）

### 1.1 A 类：工具文案裸数字（3 处，grep 精确定位）

#### `identify_ev19_template_tool.py:455-465`（unsupported 范式提示）

```python
"supported_paradigms": sorted(SUPPORTED_PARADIGMS_V01),   # ← 权威源在这（L455）
...
"message": (
    f"当前版本暂不支持「{paradigm_label}」范式分析。"
    f"v0.1 已实现 5 个范式: 高架十字迷宫 (EPM)、旷场 (OFT)、明暗箱 (LDB)、"   # ← L458 「5」+ 清单都是字面量
    f"强迫游泳 (FST)、零迷宫 (Zero Maze)。"
),
"hint": (
    "请用 ask_clarification 告知用户当前不支持的范式名称, 并询问: "
    "(a) 数据是否实际属于已支持的 5 个范式之一(用户用错名称); "                # ← L463 裸「5」
    ...
),
```

讽刺的是：L455 已经把 `sorted(SUPPORTED_PARADIGMS_V01)` 放进了结构化返回字段，L458/463 的文案却手写「5」+ 手举清单。清单变了（v0.1 → v1.0 扩范围）→ 结构化字段对了、文案错了。

#### `prep_metric_plan_tool.py:88-92`（paradigm 参数 docstring）

```
paradigm: 范式 canonical key（学术名）, v0.1 仅支持以下 5 个:        # ← L92 裸「5」+ 手举清单
          'epm' / 'open_field' / 'forced_swim' / 'light_dark_box' / 'zero_maze'
          （filename-style 缩写如 'oft'/'fst'/'ldb' 也接受，向后兼容）
```

这是 **`@tool(parse_docstring=True)` 的 docstring**，会进 lead 可见的工具描述（memory `feedback_pydantic_field_description_not_in_prompt_vs_tool_docstring`：改 LLM 可见字段说明改 docstring）。lead 据此判断「支持哪些范式」。手举清单 + 裸「5」→ 加第 6 个范式（如 `tail_suspension` 进 v0.1）忘了改 → lead 以为不支持。

### 1.2 B 类：code-executor 摘要嘴瓢

前端 232 行（code-executor handoff 摘要）：`总任务数: 140（2 个指标 × 28 个受试者 × 2 次计算 + 统计检验）` —— 写「2 个指标」。

实际 plan 是 **5 个指标**（前端 399 行 data-analyst、1393 行 report-writer 都复述 5 个），140 = 28 × 5。

**grep `code_executor.py` 证实无「2 个指标」「总任务数」「× 28」任何硬编码**。这是 deepseek 生成 handoff 摘要 AIMessage 文本时，自由拼「指标数 × 受试者数 × 计算次数」估算嘴瓢。code-executor prompt（[`code_executor.py`](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py)）**没有约束「摘要里涉及指标/任务数量的数字必须从 plan 结构化字段取、禁止估算」**。

lead 在 [23]/前端 359 行自检发现不一致，读 handoff 纠正成「5/5」（[25]/前端 364）。**lead 行为正确，无下游影响**——但 prompt 纪律缺口仍在。

### 1.3 同构本质

两类都是「**数字写死在文本里，不跟随权威来源**」：
- A 类：权威源（`SUPPORTED_PARADIGMS_V01`）存在，文案没引用它
- B 类：权威源（`plan_metrics.json` 的 `metrics[]` 计数 / `run_metric_plan` 返回的 failures 统计）存在，摘要没引用它，靠 LLM 估算

---

## 二、设计

### 2.1 A 类：工具文案裸数字 → 引用权威源

#### `identify_ev19_template_tool.py:455-465`

`message` / `hint` 里的「5 个范式」+ 手举清单，改成从 `SUPPORTED_PARADIGMS_V01` 动态生成：

```python
supported = sorted(SUPPORTED_PARADIGMS_V01)
# 复用已有的 paradigm_cn_map（L441-449 那段）把 key 翻成中文名
supported_labels = "、".join(paradigm_cn_map_short.get(k, k) for k in supported)
...
"message": (
    f"当前版本暂不支持「{paradigm_label}」范式分析。"
    f"v0.1 已实现 {len(supported)} 个范式: {supported_labels}。"
),
"hint": (
    "请用 ask_clarification 告知用户当前不支持的范式名称, 并询问: "
    f"(a) 数据是否实际属于已支持的 {len(supported)} 个范式之一(用户用错名称); "
    ...
),
```

这样 v0.1 扩范围时，`SUPPORTED_PARADIGMS_V01` 一改，文案自动跟。

#### `prep_metric_plan_tool.py:88-92`

docstring 的「v0.1 仅支持以下 5 个」+ 手举清单。两种修法（选一）：

- **修法 1（推荐，静态 docstring 友好）**：去掉具体数字，措辞改为「v0.1 仅支持 `identify_ev19_template` 返回的 supported 范式集合（见该工具返回的 `supported_paradigms` 字段）」，不在 docstring 手举清单。清单权威源是 `identify_ev19_template` / `SUPPORTED_PARADIGMS_V01`，docstring 只指向它，不复制（守 SSOT：清单只一份）。
- **修法 2**：docstring 保留清单但去掉「5 个」数字，措辞「v0.1 仅支持以下范式（清单随版本扩展，以 identify_ev19_template 返回为准）」。清单仍是双存（docstring + ev19_facts），但数字不漂移。

**推荐修法 1**——彻底守 SSOT，清单不在 docstring 复制。若担心 lead 看不到清单，docstring 指向 `identify_ev19_template` 工具（lead 必然先调它）。

### 2.2 B 类：code-executor prompt 加「摘要数字纪律」

[`code_executor.py`](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py) system_prompt 增加一段（放 `<contract>` 或摘要相关段附近）：

```
<summary_number_discipline>
最终 AIMessage / handoff 摘要里涉及【指标数量】【受试者数量】【任务总数】【失败数】等数字时，
必须从结构化数据原样取，禁止估算或心算：

- 指标数量 = plan_metrics.json 的 metrics[] 按 id 去重后的计数（不是 ×subject 的展开数）
- 受试者数量 = plan 的 inputs.raw_files 计数（或 groups.json 的 subject 总数）
- 任务总数 = run_metric_plan 返回的 total / executed 计数
- 失败数 = run_metric_plan 返回的 failures 数组长度

若要写「N 个指标 × M 个受试者 = N×M」这类算式，N 和 M 都必须来自上述结构化字段，
不要凭印象拼「2 × 28 × 2」。不确定就不写算式，只写结构化字段原值。
</summary_number_discipline>
```

> 守 `feedback_skill_describing_tool_output_enables_hallucination`：用**正面指令**（「从结构化字段取」「必须来自」），不用「禁止估算」做主指令（那句是边界紧跟正面替代，可保留）。

### 2.3 全量扫描：把同类硬编码一次清干净

实施时 grep 全部 subagent prompt + 工具文案，找「**裸写会随版本/数据漂移的数字**」：

```bash
# 工具文案里裸数字（会进 LLM 可见描述 / handoff 文本）
grep -rn "[0-9]\+ 个范式\|[0-9]\+ 个指标\|v0.1 仅支持\|已实现 [0-9]" \
    packages/agent/backend/packages/harness/deerflow/{subagents,tools}/

# f-string 里有权威源却没用 len() 的
grep -rn 'f".*[0-9].*范式' packages/agent/backend/packages/harness/deerflow/tools/
```

**判断标准**（区分「无害描述性数字」vs「会漂移的硬编码」）：
- **无害**（不改）：prompt 里给人读的措辞性数字，如「拼 140 行 bash」「2-3 个 AI message」「每条 <80 字」「140 任务挤 4 worker ≈107s」——这些不进 handoff、不声称事实、随实现变化也无害。
- **要改**（会漂移）：声称事实且权威源存在的数字，如「v0.1 已实现 5 个范式」（权威源 `SUPPORTED_PARADIGMS_V01`）、「支持以下 N 个」（清单会扩）。

实施时把扫到的「要改」项全部按 §2.1 模式修。

---

## 三、改动清单

### 3.1 `identify_ev19_template_tool.py:455-465` —— f-string 裸数字引用 `SUPPORTED_PARADIGMS_V01`

§2.1：`message` / `hint` 的「5 个范式」+ 手举清单改成 `len(supported)` + 动态生成 labels。复用已有的 `paradigm_cn_map_short`（L441-449）。

### 3.2 `prep_metric_plan_tool.py:88-92` —— docstring 去清单去数字

§2.1 修法 1：docstring 指向 `identify_ev19_template` 返回的 supported 集合，不手举清单、不写「5」。

### 3.3 `code_executor.py` —— 加 `<summary_number_discipline>` 段

§2.2：摘要数字纪律铁律。

### 3.4 全量扫描修复

§2.3：grep 全 prompt + 工具文案，按判断标准修所有「会漂移的硬编码」。实施时在 spec PR 描述里列出扫到并修复的所有点（可审计）。

### 3.5 不改无害的描述性数字

`code_executor.py:4`「拼 140 行 bash」、`:67`「2-3 个 AI message」、`:84`「<80 字」、`run_metric_plan_tool.py:58`「140 任务挤 4 worker ≈107s」（代码注释）——不改（措辞性，不进 handoff）。

---

## 四、测试（红→绿坐实，TDD 强制）

新建 `packages/agent/backend/tests/test_prompt_no_hardcoded_counts.py`：

1. **`test_identify_ev19_unsupported_message_uses_len_not_literal`**：
   monkeypatch `SUPPORTED_PARADIGMS_V01` 为 6 个范式（含一个假的），触发 unsupported 路径，断言 `message` 含「6 个范式」且含第 6 个范式的 label（证明是动态生成，不是字面量「5」）。改动前红（硬编码「5」）。

2. **`test_identify_ev19_message_matches_supported_paradigms`**：
   断言 `message` 里的范式数量 == `len(SUPPORTED_PARADIGMS_V01)`（防回归：清单变了文案自动跟）。

3. **`test_prep_metric_plan_docstring_has_no_literal_count`**（静态契约）：
   断言 `prep_metric_plan` 的 paradigm docstring **不含**「5 个」字面量 + **不含**手举的范式清单串（指向 identify_ev19_template 即可）。

4. **`test_code_executor_prompt_has_number_discipline`**（静态契约）：
   断言 `code_executor.py` system_prompt 含 `<summary_number_discipline>` 措辞 + 含「按 id 去重」措辞（约束指标计数口径）。

5. **`test_no_hardcoded_drifting_counts_in_prompts`**（全量扫描契约，防回归）：
   扫描所有 subagent prompt + 工具文案，断言**不含**「已实现 N 个范式」「仅支持 N 个」这类裸数字模式（用 regex 匹配 `已实现\s*\d+\s*个` / `仅支持.{0,10}\d+\s*个`）。**白名单**：`code_executor.py` 的「140 行 bash」「2-3 个 AI message」等无害描述性数字（测试里显式排除）。

6. **裸导入两生产入口** + 全量回归（改 tools/builtins + subagents，铁律）。

---

## 五、验收标准

1. `identify_ev19_template` unsupported 路径的 `message` / `hint` 里范式数量 == `len(SUPPORTED_PARADIGMS_V01)`，monkeypatch 扩到 6 个时文案自动变「6 个」。
2. `prep_metric_plan` docstring 不含裸「5」+ 不手举范式清单。
3. code-executor prompt 含摘要数字纪律铁律。
4. 全量扫描契约绿：所有 prompt / 工具文案无「会漂移的硬编码数字」（无害描述性数字白名单除外）。
5. 全量回归绿（除已知污染）。

---

## 六、风险与注意事项

1. **区分无害 vs 会漂移**（§2.3 判断标准）：别把「140 行 bash」「2-3 个 AI message」这类措辞性数字也改了——它们不声称事实、不进 handoff、改了反而增加 diff 噪音。只改「声称事实 + 权威源存在 + 会随版本/数据漂移」的。
2. **守 SSOT**：范式清单权威源是 `SUPPORTED_PARADIGMS_V01`（ev19_facts），文案/docstring 引用它或指向它，**不复制清单**。修法 1（docstring 指向 identify_ev19_template）比修法 2（保留清单去数字）更彻底守 SSOT。
3. **措辞守正面指令**（`feedback_skill_describing_tool_output_enables_hallucination`）：code-executor 数字纪律用「从结构化字段取」「必须来自」为主指令，「禁止估算」仅作边界。
4. **B 类是 prompt 纪律非硬编码 bug**：别去找「code-executor 里硬编码的 2 个指标模板」——它不存在（调研文档 C.1 误判已在此修正）。治法是加约束让 LLM 不嘴瓢，不是删一个不存在的模板。
5. **静态契约测试防回归**：`test_no_hardcoded_drifting_counts_in_prompts` 是关键——以后新加 prompt 若再写裸数字，CI 抓住。

---

## milestone 建议

本 spec 是小修批次（P2），不构成 feature track checkpoint，但含**可复用纪律**。建议在 harness 鲁棒性 milestone 记一条：① 此问题 + 本 spec；② **可复用教训**：prompt / 工具文案里凡「声称事实的数字」（范式数 / 指标数 / 任务数）必须引用权威源（`SUPPORTED_PARADIGMS_V01` / plan 计数 / run_metric_plan 统计），不能写字面量——「权威源在旁却没引用」是典型硬编码漂移（identify_ev19 L455 已有 `sorted(SUPPORTED_PARADIGMS_V01)` 却在 L458 手写「5」）；③ subagent 摘要里的数字嘴瓢（code-executor「2 个指标」）治法是 prompt 加「数字从结构化字段取」纪律 + 静态契约测试防回归，不是删一个不存在的硬编码模板（调研文档 C.1 误判已修正）；④ 全量扫描契约 `test_no_hardcoded_drifting_counts_in_prompts` 守住「别再写裸数字」。
