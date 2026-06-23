# Spec：identify_ev19_template 别把检测好的列信号丢掉 —— 落盘 zone_info + lead 带依据反问

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-22
> 性质：正确性 + 体验。`identify_ev19_template` 花 `parse_header` 成本检测出的 `zone_info`（含 `suspect_columns=["open","closed"]`），**只在当次返回的内存 `evidence` 里出现，落盘 `template_candidates.json` 时被剥掉**（三条落盘路径 L609/619/643 都没存 zone_info）。后果：① lead 在 unknown 后为了带依据反问，得自己读 evidence 推断 open/closed → EPM，发生在 thinking 里 → 烧 turn（thread `3a41e483` 实证）；② 检测出的列信号是一次性的，不沉淀，guardrail / 将来下游要用拿不到。**修法：落盘时把 zone_info 一起存（别剥掉）+ lead prompt 加铁律「范式反问带列依据、不瞎猜」。不碰工具返回契约——仍 unknown，守 L480「不猜」。**
> 关联：
> - 调研来源：`docs/problems/2026-06-22-thread-3a41e483-four-issues-investigation.md` D 段
> - 列检测前序：commit `afe70e24`（`_SUSPECT_ZONE_COLUMN_PATTERNS` 已落地，检测 open/closed/center/head_dip）
> - HITL 铁律：lead prompt L480「范式推断失败必须 ask_clarification、不允许默认猜测」、L483「未明确选模板必须重问、不许默认推荐项」
> - SSOT：memory `feedback_single_source_of_truth`（zone_info 只检测一次，落盘沉淀）、`feedback_skill_describing_tool_output_enables_hallucination`（prompt 正面指令）
> - 撤回：本 spec 替代撤回的 `2026-06-22-column-signals-paradigm-candidates-spec.md`（旧版误判「工具该造 ambiguous 候选」——核实 L480 后确认范式识别本就必须反问用户、不许猜，工具造 ambiguous 是过度工程且违反 HITL 铁律）

---

## 〇、给实施 agent 的一句话

`identify_ev19_template` 的 Step 3 `_detect_zone_config` 已经检测出 `zone_info.suspect_columns=["open","closed"]`，但 Step 9 三条落盘路径（[L609 unknown](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py#L609) / [L619 ok](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py#L619) / [L643 ambiguous](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py#L643)）写 `template_candidates.json` 时**都没带 zone_info**。花成本检测出的列信号被剥掉、不沉淀。lead 在 unknown 后为了带依据反问，得自己读当次 ToolMessage 的内存 evidence 推断「open/closed → EPM」，这发生在 thinking 里 → 烧 turn；且一旦 evidence 被 summarize 截断，lead 要用列信号就得重读文件。

**修法两刀**：① 三条落盘路径的 `data` 都加 `"zone_info": zone_info`（检测了就别丢）；② lead prompt 加铁律——范式反问时若 zone_info 有列信号，必须把列信号支撑的所有范式都摆给用户（EPM + Zero Maze），不许只摆一个、不许替用户猜。**工具仍返回 unknown**（守 L480「不猜范式」），只是把检测出的列信号沉淀下来。

---

## 一、根因（逐字节实证，dogfood thread `3a41e483`）

### 1.1 现象

- [04] `identify_ev19_template` 返回 `status="unknown"`（evidence 里如实标了 `has_suspect_zone_columns: true, suspect_columns: ["open","closed"]`）。
- [05] lead **自己**从 evidence 里的 `open/closed` 推断「→ EPM」反问用户。这部分推理在 lead thinking 里 → 烧 turn。
- 用户确认 EPM → 继续。

### 1.2 真 bug：检测出的列信号被落盘剥掉

[`identify_ev19_template_tool.py`](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py) Step 9 三条落盘路径：

```python
# L607-614 unknown
_write_template_candidates(real_workspace, {"status": "unknown", "paradigm_key": paradigm_key})  # ← 没 zone_info
return {"status": "unknown", "evidence": evidence, ...}   # evidence 只在内存返回值里

# L616-639 ok
_write_template_candidates(real_workspace, {"status": "ok", "paradigm_key": ..., "ev19_template": ...})  # ← 没 zone_info

# L641-657 ambiguous
_write_template_candidates(real_workspace, {"status": "ambiguous", "paradigm_key": ..., "candidates": ..., ...})  # ← 没 zone_info
```

`zone_info`（含花 `parse_header` 成本检测出的 `suspect_columns`）**只在当次返回的内存 `evidence` 字段里**，落盘的 `template_candidates.json` 把它剥掉了。

### 1.3 后果1：lead thinking 烧 turn（thread `3a41e483` 实证）

lead 拿到 unknown 后，按 L480 必须反问、不许猜。但为了**带依据**反问（而非开放式「您做的是什么实验」），lead 得自己读当次 ToolMessage 的 `evidence.zone_info.suspect_columns`、自己推断「open/closed → 可能是 EPM」。这部分领域推理在 lead thinking 里展开 → 烧 turn。如果工具直接在落盘/返回里带好「列信号支撑的范式候选」，lead 只需转述，不用自己当侦探。

### 1.4 后果2：检测是一次性的，不沉淀

`template_candidates.json` 是给 `Ev19TemplateGuardrailProvider` 强制「用户必须确认范式」用的（[ev19_template_provider.py:86-89](../../../packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py#L86-L89) 只读 `status`/`candidates`）。落盘没存 zone_info 意味着：
- lead 想用列信号只能靠当次 ToolMessage 内存 evidence；一旦 evidence 被 summarize 截断出 context，要重读文件。
- guardrail / 将来任何下游想用「已检测的列信号」（如列对齐预填、防 lead 猜），都拿不到，得重检测。

### 1.5 为什么不修「工具返回 ambiguous」（撤回旧 D spec 的理由）

旧版 D spec 提议工具用列信号造 ambiguous 候选。核实 lead prompt 后撤回：
- L480：「范式推断失败 → ask_clarification：上传数据但无法推断 EV19 模板时，**必须反问让用户指定范式；不允许默认猜测**」
- L483：「用户未明确选模板 → 重问...**即使模板有"推荐"标记也不允许默认选推荐项**」
- L456：把「没指定范式」列为 `missing_info` 标准例，**必须 ask_clarification**

范式识别这一步**无论 unknown 还是 ambiguous，lead 都必须反问用户、都不许猜**。所以「unknown → ambiguous」**不减少交互轮次**（旧 spec 的核心价值主张站不住）。工具造 ambiguous 是在工具层重复 lead 该做的事，且可能诱导「工具给了候选就当确定了」违反 L480。**工具应诚实返回 unknown（不猜），但把检测成本沉淀（不丢）。**

### 1.6 与同事 PR `afe70e24` 的关系（避免「这不是做过了？」误判）

`afe70e24` 把列信号接入 `_filter_candidates_by_zone`（L217-219，`has_suspect` 时剔 NoZones），但**只做过滤，没做落盘**。本 spec 补的是「检测出的 zone_info 落盘沉淀」——同事 PR 用 zone_info 过滤候选（内存），本 spec 让 zone_info 沉淀进文件（持久）。两层正交，不重复。

---

## 二、设计

### 2.1 修法 A：三条落盘路径都带 zone_info（治本「检测了就丢」）

[`identify_ev19_template_tool.py`](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py) Step 9 三条 `_write_template_candidates` 调用，`data` 都加 `"zone_info": zone_info`：

```python
# L607 unknown
_write_template_candidates(real_workspace, {
    "status": "unknown",
    "paradigm_key": paradigm_key,
    "zone_info": zone_info,          # ← 新增：沉淀检测出的列信号（含 suspect_columns）
})

# L619 ok
_write_template_candidates(real_workspace, {
    "status": "ok",
    "paradigm_key": paradigm_key,
    "ev19_template": candidates[0]["template_id"],
    "zone_info": zone_info,          # ← 新增
})

# L643 ambiguous
_write_template_candidates(real_workspace, {
    "status": "ambiguous",
    "paradigm_key": paradigm_key,
    "candidates": candidates,
    "clarification_question": ...,
    "zone_info": zone_info,          # ← 新增
})
```

`zone_info` 已含 `{has_zone_columns, has_novobj_columns, has_suspect_zone_columns, zone_columns, novobj_columns, suspect_columns}`（[L180-187](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py#L180-L187)）。整个 dict 直接存，可 JSON 序列化。

> 工具返回值（`return {...}`）也已带 `evidence`（evidence 里含 zone_info），**内存返回 + 落盘双写一致**——lead 当次用内存 evidence，后续/截断后用落盘。

### 2.2 修法 B：lead prompt 加「带列依据反问」铁律

[`lead_agent/prompt.py`](../../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py)「EthoInsight 硬性反问场景」段（L479-484）补一条：

```
- **范式反问带列依据、不瞎猜**：identify_ev19_template 返回 unknown/ambiguous 时，
  若其 evidence（或落盘 template_candidates.json 的 zone_info）含 suspect_columns
  （如 open/closed/center/...），反问必须【把列信号支撑的所有范式都列为选项】，
  不许只列一个、不许替用户猜。
  例：suspect_columns 含 open/closed → 这对列同时支撑 EPM 和 Zero Maze（结构上无法区分），
  反问必须「可能是 EPM（高架十字）或 Zero Maze（零迷宫），请确认」，不许只问「是不是 EPM」。
  列信号→范式候选的依据来自 _facts.json 的 zone_template（同事维护 SSOT），
  你只转述工具/落盘给的列信号，不自己推断范式判定。
```

> 守 `feedback_skill_describing_tool_output_enables_hallucination`：用**正面指令**（「把所有范式都列为选项」「只转述工具给的列信号」），不用「禁止猜测」做主指令。

### 2.3 不改工具返回契约（守 L480）

工具**仍返回 unknown**（paradigm_key=None 时），不造 ambiguous 候选。改的只是「落盘多带 zone_info」+「lead 反问带依据」。守 HITL 铁律：范式由用户确认，工具/lead 都不猜。

### 2.4 不改 guardrail、不改 `_filter_candidates_by_zone`、不改 `_detect_zone_config`

- guardrail（ev19_template_provider.py）只读 `status`/`candidates`，多出的 `zone_info` 字段它不读不坏（向前兼容）。
- `_filter_candidates_by_zone`（过滤）、`_detect_zone_config`（检测）已正确，不动。

---

## 三、改动清单

### 3.1 `identify_ev19_template_tool.py` —— 三条落盘路径加 zone_info

§2.1：L609/619/643 三处 `_write_template_candidates` 的 `data` dict 各加 `"zone_info": zone_info`。`zone_info` 变量在函数体内已存在（L498 `_detect_zone_config` 返回值），直接引用。

### 3.2 `lead_agent/prompt.py` —— 加「带列依据反问」铁律

§2.2：L479-484「EthoInsight 硬性反问场景」段补一条「范式反问带列依据、不瞎猜」。

### 3.3 不改其他

不改工具返回契约、不改 guardrail、不改列检测/过滤逻辑。

---

## 四、测试（红→绿坐实，TDD 强制）

新建 `packages/agent/backend/tests/test_identify_ev19_zone_info_persisted.py`：

1. **`test_unknown_persists_zone_info`**（红线，复现 dogfood）：
   构造列含 `open`/`closed`（无范式关键词、文件名 `trial01.xlsx`）的 header，调 `identify_ev19_template`，读 `template_candidates.json`，断言含 `zone_info.suspect_columns == ["open","closed"]`。改动前红（现状 unknown 路径只存 status+paradigm_key）。

2. **`test_ok_and_ambiguous_also_persist_zone_info`**：
   构造能识别范式的数据（如文件名含 epm），断言 ok / ambiguous 路径落盘的 `template_candidates.json` 也含 `zone_info`。

3. **`test_zone_info_json_serializable`**：
   断言落盘的 `zone_info` 是合法 JSON（含全部 6 字段），可被后续读取方反序列化。

4. **`test_lead_prompt_has_column_evidence_rule`**（静态契约）：
   断言 `lead_agent/prompt.py` 含「范式反问带列依据」措辞 + 「open/closed 同时支撑 EPM 和 Zero Maze」措辞（钉死列信号→多范式的事实，防 lead 猜成单一范式）。

5. **`test_guardrail_ignores_extra_zone_info_field`**（向前兼容）：
   构造含 `zone_info` 字段的 `template_candidates.json`，断言 `Ev19TemplateGuardrailProvider` 正常工作（不读 zone_info 也不坏）。

6. **裸导入两生产入口** + 全量回归（改 tools/builtins + lead prompt，铁律）。

---

## 五、验收标准

1. `identify_ev19_template` 任何返回路径（unknown/ok/ambiguous）落盘的 `template_candidates.json` 都含 `zone_info.suspect_columns`。
2. dogfood EPM 复跑（thread `3a41e483` 同款）：lead 反问时直接读落盘的列信号、转述「可能是 EPM 或 Zero Maze」，**不在 thinking 里自己推断 open/closed → EPM**（省 turn）。
3. lead 反问把列信号支撑的所有范式都列为选项（open/closed → EPM + Zero Maze），不猜成单一范式。
4. 工具仍返回 unknown（paradigm_key=None 时），不造 ambiguous 候选（守 L480）。
5. guardrail 对多出的 zone_info 字段向前兼容（不坏）。
6. 全量回归绿（除已知污染）。

---

## 六、风险与注意事项

1. **守 L480 不猜范式**（§2.3）：工具仍 unknown，不在工具层造 ambiguous 候选。列信号只沉淀、只供 lead 带依据反问，不替用户定范式。
2. **守 SSOT**（§2.2）：列信号→范式的依据来自 `_facts.json` 的 zone_template（同事维护），lead 只转述工具/落盘给的列信号 + 已知的「open/closed 同见 EPM+ZeroMaze」事实，不自己拍范式判定。
3. **向前兼容**（§3.3）：guardrail 不读 zone_info 也不坏；新字段只是多出来供 lead/将来下游用。
4. **内存与落盘双写一致**：工具返回的 `evidence.zone_info` 与落盘的 `zone_info` 是同一份（变量复用），lead 当次用内存、截断后用落盘，不会漂移。
5. **别把「落盘 zone_info」当成「工具造候选」**（撤回旧 D spec 教训）：本 spec 只沉淀检测产物 + 治理 lead 反问行为，不让工具替用户猜范式。
6. **zone_info（本 spec 落盘）vs column_aliases（已落盘）是两阶段产物，别混淆**：
   - **zone_info**（identify 的 `_detect_zone_config` 产物，本 spec 要沉淀进 `template_candidates.json`）= **对齐前的原始检测**：纯模式匹配判定「这文件有 `open/closed` 等疑似归属列」，**不知含义**，可信度低。产生于 Gate 1 **之前**（范式还没确认时），作用是让 lead 在对齐前的反问带依据（「您数据里有 open/closed 列，可能是 EPM 或 ZeroMaze」）。
   - **column_aliases**（`set_experiment_paradigm(column_semantics=...)` 产物，已落盘进 `experiment-context.json`，[prep_metric_plan_tool.py:218](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py#L218) / [prep_chart_plan_tool.py:191](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_chart_plan_tool.py#L191) 自读）= **对齐后的确认结果**：用户确认「`zone x` = open_arm」后的映射表 `{"zone x": "open_arm"}`，可信度高（用户背书）。产生于 Gate 1 **之后**（用户确认范式+列语义后），作用是让 resolve 注入 `open_arm_zones=["zone x"]`（值是物理列名，源文件不改一个字），下游 compute 在 df 里找 `df["zone x"]` 命中。
   - **本 spec 不重复造 column_aliases**——那只落盘「对齐前检测」，对齐后确认早有 column_aliases 沉淀（SSOT 在 experiment-context.json，两处 prep 自读同一份）。本 spec 补的是更早一阶段（对齐前）的检测产物沉淀，时间上先于 column_aliases。两阶段、两文件、两种可信度，正交不重复。

---

## milestone 建议

本 spec 属于「EV19 模板识别 / 范式路由」track，小修。建议在该 milestone 记一条：① 此问题 + 本 spec；② **撤回教训**：初版 D spec 误判「工具该用列信号造 ambiguous 候选」，核实 lead prompt L480/L483（范式识别必须反问、不许猜）后撤回——范式识别这一步无论 unknown/ambiguous 都要问用户，工具造候选是过度工程且违反 HITL 铁律；③ **真根因**：`identify_ev19_template` 检测出的 `zone_info` 落盘时被剥掉（三条路径都没存），导致 lead 为带依据反问得在 thinking 里自己推断 → 烧 turn，且检测成本不沉淀；④ **可复用边界**：工具花成本检测的中间产物（zone_info/列信号）应落盘沉淀，别只放内存返回值——lead 后续、guardrail、下游都可能在 ToolMessage evidence 被截断后还要用它；⑤ open/closed 同见 EPM+ZeroMaze 是事实（`_facts.json` zone_template 实证），lead 反问必须把两个都列，不许猜成单一范式。
