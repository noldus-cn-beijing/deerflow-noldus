# Spec A — code-executor 封存边界对账兜底（扩 auto-seal 白名单 + 完整性判据）

> 日期：2026-06-10 ｜ 状态：待实施 ｜ 层级：harness ｜ 独立于行为学同事方法论阻塞
> 前置 review brief：`docs/handoffs/2026-06/2026-06-10-oft-dogfood-two-failures-seal-replan-review-brief.md`
> 评审：Fable 5（项目目录外，通用稿 `/tmp/multiagent-orchestration-two-failures-discussion.md`）三条判断已吸收
> 主仓库 `/home/wangqiuyang/noldus-insight`，dev HEAD `1d4a9d25`
> harness 根：`packages/agent/backend/packages/harness/deerflow/`（下文 file:line 均相对此根，除非另注）

---

## 0. 一句话目标

`code-executor` 跑完全部指标却漏调 `seal_code_executor_handoff` 时（"terminated without emitting handoff"），harness 在**封存边界**用**已落盘的命名规整结果文件 + plan 对账**确定性重建 handoff，不再判 FAILED 白等重派——但**只有 plan 期望集与磁盘实际产出对齐才标 `completed`，缺项标 `partial`**，并带来源标记 `sealed_by=framework_rebuild` + 触发率可观测。

**不做**：不改 data-analyst（认知产物，见 §6 边界判断）；不改退出协议（不加 tool_choice / structured_output / 硬循环到封存为止）；不改 `validate_catalog`（red herring）。

---

## 1. 根因（已逐字核实，2026-06-10）

合进同一条 SystemMessage（`subagents/executor.py:856-876`）的两份指令打架，制造"叙述黑洞"：

- **system_prompt（权威）**：`subagents/builtins/code_executor.py:89-92` step 6 明确"调 `seal_code_executor_handoff` tool……**严禁直接 write_file 写 handoff_code_executor.json，必须走本 tool**"。
- **过时 skill（矛盾）**：`skills/custom/ethoinsight-code/SKILL.md:30` 仍教 `write_file handoff`，全文 0 提 seal 工具；模板 `skills/custom/ethoinsight-code/templates/output-contract.md:22` 同。
- 模型把"写好叙述 + 输出 `[gate_signals]`"误当完成（同 `feedback_subagent_seal_deadlock_is_prompt_not_budget`，本轮触发源是过时 skill）。
- 加重项：write_file 旧路线被 `guardrails/script_invocation_only_provider.py:222-236`（`_HANDOFF_WRITE_FILE_DENY`，文案 :109-115）拒，退化成"完全没产物"。

**决定性证据（必须 harness 兜底）**：下游 data-analyst 的 prompt 已加固到顶（`subagents/builtins/data_analyst.py:153`「seal 是必达」、`:187`「发出 seal tool_call 才是落库」）**仍漏调** → 纯 prompt 不可靠，是系统性而非随机故障 → 重派对系统性故障无效（花两倍钱再现一次）。

**现有兜底显式排除 code-executor**：`subagents/executor.py:282-285` `_AUTO_SEALABLE = {report-writer, chart-maker}`；:270-278 注释"code-executor / data-analyst 的 handoff 是认知产物，不能从文件重建 → 永不 auto-seal"；:307 非白名单 early-return False。

**被推翻的假设（Fable 判据修正）**：注释的"认知产物不能重建"对"结果按命名约定落盘"的声明式管线是**错的**——值就躺在命名 JSON 里。正确判据不是"认知 vs 机械"，而是**每个必填字段是否存在"模型封存意图之外的忠实来源"**；code-executor 的 `metrics_summary` 来源是 `m_*.json`（文件），齐全可重建。

---

## 2. 对账可行性（已核实，2026-06-10）

机械重建的承重墙是"plan 期望集可枚举 + 与磁盘产出 1:1 对账"。已核实成立：

- **plan 可枚举**：`tools/builtins/prep_metric_plan_tool.py:290-297` 写 `plan_metrics.json`，顶层 `plan["metrics"]` 是数组；每项含 `id` / `script` / `output`（**plan 阶段就构造好的期望输出绝对路径**）/ `subject_index` / `display_name_zh`。
- **output 命名确定**：`packages/ethoinsight/ethoinsight/catalog/resolve.py:520-530`——单 subject `m_<id>.json`，多 subject `m_<id>_s<idx>.json`，**纯 index 派生、不随参数变化**。
- **结果文件原子写 1:1**：`packages/ethoinsight/ethoinsight/scripts/_cli.py:111-129` `save_output_json`（tmpfile + `os.replace` 原子）；schema = `{"metric": str, "value": float|int|null, "parameters_used": dict?}`（`_cli.py:25-45` `emit_result`）。
- **per_subject 可恢复**：`groups.json`（`prep_metric_plan_tool.py:225-262`，`{subject_file: group_name}`）+ plan 的 `inputs.raw_files[subject_index]` join → subject→group→value 全确定。

→ **对账算法确定可实现**（见 §3 A2），无缺口。

---

## 3. 实施方案

### A1（PROMPT，治本，必做）— 删 skill 旧流，SSOT 守护

**问题**：同一份"如何提交 handoff"的指令分散三处（skill / 模板 / system_prompt），只改一处留矛盾。

**改动**：
1. `skills/custom/ethoinsight-code/SKILL.md:30`：删 `write_file handoff` 旧流，改为 **defer 到封存工具**——"分析完成后由 system_prompt 指定的 `seal_*` 工具落库；本 skill 不描述 handoff 文件写法"。**不在 skill 重写 handoff 字段结构**（SSOT 红线，`feedback_single_source_of_truth`：权威只在 system_prompt + 工具 schema）。
2. `skills/custom/ethoinsight-code/templates/output-contract.md:22`：同步删 write_file handoff 段，"输出 `[gate_signals]`"措辞**降级为"中间产物，非完成信号"**，明确"只有 `seal_*` 工具返回 OK 才算完成"。
3. `subagents/builtins/code_executor.py` step 6：把封存重构为**唯一且最后**动作，正面措辞（CLAUDE.md §6，deepseek 不用"禁止 X"）——"完成全部 `m_*.json` 后，最后一步发出 `seal_code_executor_handoff` 的 tool_call，工具返回 OK 即任务完成"。

**红线**：三处合进同一 SystemMessage，**必须一致改**（红线清单 §5），只改一处留矛盾。

### A2（HARNESS，关键，必做）— 扩 auto-seal 到 code-executor + 完整性对账

**核心改动**：`subagents/executor.py`

**A2.1 扩白名单**（:282-285）：
```python
_AUTO_SEALABLE: dict[str, str] = {
    "report-writer": "handoff_report_writer.json",
    "chart-maker": "handoff_chart_maker.json",
    "code-executor": "handoff_code_executor.json",   # 新增
}
```
更新 :270-278 注释：从"code-executor / data-analyst 永不 auto-seal"改为"**code-executor 可重建（值在命名 JSON 里、plan 可对账）；data-analyst 仍永不 auto-seal（判读结论无文件来源，见 Spec 边界判断）**"。

**A2.2 在 `_attempt_auto_seal_from_artifacts`（:288-399）加 code-executor 分支**，逻辑：
1. 读 `plan_metrics.json` → 枚举 `expected = {m["output"] for m in plan["metrics"]}`（虚拟路径经 `resolve_sandbox_path` / `scripts/_cli.py` 按 `DEERFLOW_PATH_*` 解析成真实路径，复用 `validate_catalog` 已用的同一解析器，防 :30 那类 `result_file_unreadable` 误报）。
2. glob 实际 `m_*.json` → `actual`。
3. **完整性判据（Fable 判据，承重）**：
   - `missing = expected - actual`
   - `missing == ∅` → `status = "completed"`
   - `missing != ∅` → `status = "partial"`，把缺失的 `(metric_id, subject_index)` 列进 `errors` 字段（**绝不把半截尸体标 completed**）。
4. 读每个存在的 `m_*.json` 取 `{metric, value}`，join `groups.json` + plan 的 `subject_index→raw_files` → 重建 `metrics_summary` / `per_subject`。
5. `summary` 用确定性模板填（"机械重建：N/M 指标产出，status=X"），`paradigm` / `ev19_template` 从 `plan_metrics.json` 或 `experiment-context.json` 读。
6. **来源标记**：`sealed_by = "framework_rebuild"`（CodeExecutorHandoff 加该可选字段，见 A2.3）。
7. 走现成 `_seal_handoff_to_workspace`（`seal_handoff_tools.py:354-397`，已 Pydantic + tmp-rename 原子 + SHA256 manifest——**复用现成 transactional outbox，不新造**）。

**A2.3 schema**：`subagents/handoff_schemas.py:386` `CodeExecutorHandoff` 加可选 `sealed_by: Literal["model","framework_rebuild"] = "model"`（默认 model，重建路径显式置 framework_rebuild）。**只加字段不动必填**（`status`+`summary` 仍唯二必填）。同步检查 `errors` 字段已存在可承载 missing 列表。

**A2.4 字段名规范化（顺带消除已知故障类）**：重建直接产出**校验器消费的规范字段名** `metrics_summary`（`executor.py:119` `_CODE_EXECUTOR_METRICS_FIELDS` 三分裂之一）——绕开 `feedback_handoff_metrics_field_divergence_mislabels_failed` 的"字段三分裂、校验器只认一种"故障类。重建出的 handoff 反而比 worker 自封的更可靠。

**红线（实施铁律）**：
- ① 不加 turn 预算；② 不给 seal-resume 加 tool_choice（产空 args）/ 不用 structured_output（strip runtime 注入）——`executor.py:904` 已显式探明拒绝，照旧。
- ③ **动 `executor.py` 顶层 import 必须裸跑**（`feedback_conftest_mock_hides_circular_import`，前例 commit `3ffaf672`）：
  ```bash
  cd packages/agent/backend
  PYTHONPATH=. python -c "import app.gateway"
  PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
  ```
  新 helper 的 import 放函数体惰性，别放模块顶层。
- ④ A2 auto-seal 与 seal-resume 的执行顺序**不变**：`_validate_handoff_emitted`（事后校验，:174-266）→ `_attempt_seal_resume`（1 轮补救，`_SEAL_RESUME_MAX_ATTEMPTS=1` :171）→ `_attempt_auto_seal_from_artifacts`（确定性兜底）。auto-seal 只在 seal-resume 仍失败后跑一次，**不双封存、不多跑一轮**。

### A3（PROMPT，便宜，必做）— 补轮措辞去专有名词

`subagents/executor.py:915-919` 的 seal-resume 补轮 prompt 当前对所有 subagent 通用，含 `key_findings`（那是 DataAnalystHandoff 字段，给 code-executor 是错 schema）。
- 改为**通用措辞**："把你刚才得出的各结构化字段填入工具参数并落库"，去掉 `key_findings` 专有名词。
- **红线**：`:913` 有 in-code 守卫锁定该段——**只去专有名词、不重构**。

### A4（不做）— validate_catalog 是 red herring

`validate_catalog.py:346` 只遍历 `plan['metrics']`，结构上不可能对 plan 外项报错。**不改**。

---

## 4. 红 / 绿测试锚点（TDD 强制）

放 `packages/agent/backend/tests/`：

1. **`test_auto_seal_code_executor_completed`**：造 `plan_metrics.json`（3 指标 × 2 subject = 6 期望）+ 6 个 `m_*.json` 全在 + `groups.json` → auto-seal 产 `status=completed`、`sealed_by=framework_rebuild`、`metrics_summary` / `per_subject` 非空、与文件值一致。
2. **`test_auto_seal_code_executor_partial_on_missing`（完整性判据红锚点）**：同上但删 1 个 `m_*.json` → `status=partial`、`errors` 含缺失的 `(metric_id, subject_index)`、**绝不 completed**。
3. **`test_auto_seal_code_executor_field_name_canonical`**：重建产出落 `metrics_summary`（规范名），过 `_check_code_executor_content`（:122-135）非空校验、不触发字段三分裂误标。
4. **`test_auto_seal_excludes_data_analyst`（边界红锚点）**：data-analyst 漏封存 → auto-seal 返回 False（保持 FAILED 或走 §6 决定的最弱兜底），**绝不机械重建判读结论**。
5. **`test_seal_resume_prompt_no_key_findings_for_code_executor`**：补轮 prompt 不含 `key_findings` 专有名词。
6. **`test_gateway_import_no_cycle`** 已存在——确认改完仍过（subprocess 裸导入两入口）。

**验收项（写进 spec、非备注 —— `feedback_fallback_trigger_rate_must_be_observable_acceptance_criterion`）**：
- **V1 触发率可观测**：auto-seal(code-executor) 触发时 log 一条结构化记录（`subagent=code-executor sealed_by=framework_rebuild status=X`），可被 training-stats / gateway.log grep 统计。
- **V2 回归探针**：A1 skill 修复后，真实 dogfood 里 code-executor auto-seal 触发率应**趋近 0**（prompt 修对了就不该再漏封存）；触发率回升 = prompt/skill 回归信号。**这是验收标准，不是 nice-to-have**。

---

## 5. 实施顺序与验证

1. 起 worktree：`git worktree add <path> -b spec-a-code-executor-autoseal dev`（**显式基于 dev**，否则默认 origin/main 卷误删，`project_2026-06-08_three_specs_review_acdf`）。
2. A1（skill/模板/system_prompt 三处一致改）→ A2（executor + schema）→ A3（补轮措辞）。
3. 先写红测试（§4 #1-5）再实现。
4. 跑 `make test`（全量，改了共享 executor，不能只跑新测试，`feedback_pr_merge_must_run_full_suite_on_shared_logic`）。
5. **裸导入验证**（§3 A2 红线③）。
6. 已知 5 个基线红（deferred_tool_registry ×2 + inspect_gate/paradigm async ×2 + chart_maker_config ×1）非本改动引入，别归因自己。
7. dogfood OFT 真实数据复现原故障 → 确认 auto-seal 触发且 status 正确 / A1 修复后触发率趋零。
8. **§6.1 依据核实（写进实施记录）**：确认 data-analyst 加固实验时其 SystemMessage 确实加载了矛盾 skill 文本——成立则 §6 顺序坐实；不成立则按 §6.1 末条翻转、转录兜底同批提上。
9. **§6.3 触发判据存档**：A1 dogfood 通过后，若 data-analyst 在**指令一致条件下**仍漏调封存复发一次 → 机械启动转录兜底 follow-up spec（不开会、不设次数门槛）。

---

## 6. 边界判断：data-analyst 不接 auto-seal（已答，Fable 二轮确认）

**结论**：**不接，保持 prompt-only；转录兜底留作 follow-up，触发判据已定死（见 §6.3）。**

### 6.1 主依据（承重）：根因尚未隔离，决定性实验还没跑过

> ⚠️ 这是"暂不做"的**主依据**。"只复用不自造"是过程纪律、是**附带**收口——如果证据真已证明 prompt 不可救，纪律不该挡必要机制（大不了单开一批）。真正让"暂不做"成立的是下面这条根因隔离论证。

"data-analyst 加固到顶仍漏调封存"这条决定性证据，是**在矛盾指令仍在其 SystemMessage 里的条件下**采集的（过时 skill `SKILL.md:30` write_file 旧流 + system_prompt seal 工具，合进同一 SystemMessage 是**两个 worker 共享的污染源**）。该实验测的是"更多正确指令能否压过矛盾指令"——答案是不能（符合 LLM 指令遵循的一般认知：矛盾不会被平均掉，歧义被模型不可预测地消解）。**它没测"指令一致时模型是否会正常封存"——干净条件下的决定性实验还没跑过。** 同批上转录兜底 = 基于一份尚不存在的证据行动。

**显式依据栏（实施者必须核实，别让它隐含）**：
- ☑ 已核实：data-analyst 加固实验进行时，其 SystemMessage 确实加载了那条 / 同类"写好叙述=完成"的 skill/模板文本（`SKILL.md:30` + `output-contract.md:22`，与 code-executor 共享 ethoinsight-code skill）。
- **若此核实成立** → "prompt 不可能痊愈"降级为"prompt 叠加在矛盾之上不可能痊愈"，先后顺序是唯一正确顺序。
- **若发现它其实没加载那条 skill、加固是在干净指令下做的仍失败** → 结论翻转，转录兜底立刻提上来（同批）。

### 6.2 三条独立支撑顺序的论据
1. **本系统故障史在喊"先隔离"**：data-analyst seal 卡死已有≥2 个互不相同真根因——6-08 那次被报告误判为"忘记调工具/叙述黑洞"，**真相是 schema 缺 partial 三态、Pydantic 拒了一次诚实封存调用**（`feedback_dataanalyst_reportwriter_handoff_status_missing_partial`，gateway.log trace=acdfb7e5）。**seal 故障表象高度同质、根因高度异质**；根因未隔离时叠加新机制 = 给下次误判铺路。
2. **故障可见性不对称**：现状失败是**响的**（无 handoff / 框架报错 / FAILED）；仓促的转录兜底失败是**哑的**（幻觉 handoff 看起来像成功）。用一轮 dogfood（一次响亮 FAILED + 重派）换"不引入哑故障通道"，交易明显划算。
3. **顺序改善转录兜底的未来输入质量**：拔掉矛盾指令后若 worker 仍漏封存，它落在叙述里的内容会更贴近 seal schema 形状（上下文只剩描述正确 schema 的指令）→ 转录步骤输入更干净。**先修 skill 再上转录 = 依赖排序，不只是风险排序。**

### 6.3 follow-up 触发判据（现在定死，不留到 dogfood 后再议）
> **拔除矛盾指令（A1）并 dogfood 通过一次之后，任何一次"指令一致条件下的封存漏调"复发，即启动转录兜底 spec——不设次数门槛、不重新开会。** 理由：决定性实验是二值的，干净条件下复发一次就证明"一致指令仍不充分"，等第二次无信息增量。判据先定死，到时决策机械执行。

### 6.4 转录兜底设计约束 stub（写 stub ≠ 建机制，不破坏本 spec 收口）
若 §6.3 触发，转录兜底（起全新上下文、单一职责子代理，只把 worker 最终叙述转录成封存调用）的**首要可预测失败模式 = 转录幻觉**（schema 必填字段 + "产出合法输出"指令 = 编造压力）。三形态：①**压扁犹豫**（含糊叙述→确定结论字段）；②**把推演当结论**（探索性分支→挑一支封存）；③**把半截当完整**（worker 中途死掉，残缺叙述→形状完整 handoff；与 A2 完整性判据同问题但更难，无 `m_*.json` 清单可对账，顶多拿 `gate_signals` 期望集做部分对账）。

**核心设计约束 = 把"信任转录器"改造成"验证转录器"**：
- **C-α 引用锚定 + 确定性后验**：转录器为每个填入字段附一段叙述原文逐字引用；harness 做**子串校验**（引用非 transcript 子串即拒该字段，确定性、不信任任何 LLM）；引用不到锚点的必填字段一律置空。同时缓解全部三形态。
- **C-β status 由 harness 推导、不许转录器断言**：转录器只填内容字段；`completed/partial` 由框架按"字段完整度 vs 期望集"计算——**与 A2"plan 对账才标 completed"同一条设计语言，两个兜底共用**。
- **C-γ 默认偏置向缺省**：指令不对称写死"存疑则不填"，让 partial 成阻力最小路径、编造成阻力最大路径。

**`sealed_by=transcription_fallback` 承重三层（缺一不可）**：
- **路由层**：落进 orchestrator 实读的信号位（我们拓扑 = `gate_signals`）——不在实读位置的标记等于不存在（lead 不读 handoff 正文的实证教训换个出口）。
- **gate 层**：任何基于 handoff 完整性自动放行的 gate，对 transcription_fallback **一律不自动放行**——HITL 确认后报告才出，或报告带可见标注出。**落点天然在 Sprint S7 假设暴露面板**（转录来源本质是一条"该结论封存由框架代办"的假设，同板展示，不新造 UI）。
- **遥测层**：触发率可观测、写进验收项（同 `feedback_fallback_trigger_rate_must_be_observable_acceptance_criterion`）。

**一句话收口标准**：一份转录产出的 handoff，在它的内容影响任何决策（机器的 gate 路由、人的报告阅读）的**每一个点**上，都必须**无法被误认为 worker 自封的**。达到 = 诚实代笔；达不到 = 把响亮 FAILED 换成无声幻觉，**比不修更糟**。

---

## 7. 与 Spec B 的关系

A、B 是**同一元问题的两个实例**（正确持久化机制已存在、prompt 指向错误终结动作、harness 只事后检测），但**修复形状不同**（Fable 判断）：
- **共享 write 侧原语**：框架在**它能观测的边界**上对账规范持久化是否发生、没发生就确定性补上。A 的边界是 subagent 退出；B 的边界是"ask_clarification 后下一条用户消息到达"。
- **read 侧只 B 需要**：A 的持久化消费者是框架机器（校验器 / 下游派遣，不需回灌）；B 的消费者是下一轮 LLM（无内生持久读取，必须注入回灌）。
- **不造 `BoundaryReconciler` 基类**：n=2 抽象早熟，两个具体对账器共享一个概念即可（合"智能在节点内、约束在节点间"编排哲学）。

详见 `docs/superpowers/specs/2026-06-10-spec-b-resolved-facts-readback.md`。
