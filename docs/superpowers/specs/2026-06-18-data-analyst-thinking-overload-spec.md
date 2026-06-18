# Spec：data-analyst thinking 过载撞超时 —— 把「计算/搬运」逐出判读路径，让它只判读

## 〇、给实施 agent 的一句话

data-analyst 把本该「读结论」的活变成了「在 thinking 里重算+逐条搬运 63 条 outlier+手跑参数审计 a–f 决策树」，单个 model turn 撑爆 50K thinking、撞 900s wall-clock 超时、永远走不到 seal、lead 重派 → **原地打转**。本 spec 三刀：① **移除参数审计（step 2.8 整段）**——它要的「参数值 vs 数据分布」数值判据事实上造不出来（行为学同事拿 680 页书都没产出），留在判读路径上是纯负担；② **outlier 搬运下沉到 code-executor**——`_outliers.json` 直接写真实 `Trial N`（而非 `subject #i`），且这份成品就是最终 `outlier_findings`，data-analyst 只**引用**最该警惕的几条，不逐条搬；③ **data-analyst prompt 收窄成纯判读**——强化 read `by-experiment/<paradigm>.md`（判据已在 repo），删一切「遍历/映射/决策树」机械步骤。**不碰 reward YAML、不新建消费链**（守 SSOT + 不越界）。

> ⚠️ 关键边界（值得记进 memory）：本次失败是 **turn 内超时**，SealGateMiddleware（`after_model` 门）**结构上救不了**——它只在「model 完成一个 turn、要以纯文本收尾」时触发；turn 根本没结束，after_model 永不执行。这与之前 4 次 seal 漏调（收尾时漏 call）是**不同类**故障：那是「收不了尾时漏 call」，这是「**根本收不了尾**」。治本不在 seal 门，在「别让 thinking 背计算/搬运」。

---

## 一、根因（逐字节实证，dogfood thread `2e40514c-f276-4e13-a172-dd90bbbaeb4b`，HEAD `79ff9f2f`）

### 1.1 现象

```
code-executor → Task Succeeded (135/135 metrics)
data-analyst  → Task timed out. Error: Execution timed out after 900 seconds   ← 第一次
lead 重派     → data-analyst 再次进入同样穷举 reasoning（用户实时看到的"打转"）  ← 第二次
```

`handoff_data_analyst.json` 从未产出（workspace 里不存在）。lead 层 checkpoint 解码坐实上述序列。

### 1.2 排除：不是 Bug4（handoff 截断），handoff 完全健康

`handoff_code_executor.json` 实测 **18260 字节**（远低于 50K），Bug4 瘦身 spec 完全生效：旁路 `_outliers.json`(31K)/`_outputs.json`(8.3K) 已拆出；data-analyst fast-fail 必读的 `status:"completed"` / `statistics`（5 指标全有检验+效应量）/ `data_quality_warnings:[]` / `gate_signals.quality_warnings_critical_count:0` 全部在主文件、单次 read 即达。**数据 100% 可分析，链路无毛病。**

### 1.3 真根因：thinking 当产出，撞 wall-clock

用户实时贴出的 data-analyst 内部独白即铁证——它在**单个 model turn 的 thinking 里**完整且反复地：

- 把 63 条 outlier 逐条**重新映射** Trial 号（`subject #2` → `Trial 3`、`subject #4` → `Trial 12`…）
- 逐条**重算** counterfactual（"排除后 0.253→0.129"…），尽管 `_outliers.json` 里每条都有现成 `counterfactual` 预格式化串
- 把 `seal_data_analyst_handoff` 的 JSON 参数**用散文反复草拟两三遍**
- 末尾 `[Message truncated - exceeded 50,000 character limit]`——**爆 50K 的是它自己的 reasoning，不是 handoff**

`thinking_enabled=True` + `max_turns=12` + 实际 900s 硬墙（config `timeout_seconds=600`，外层放大到 900）。它在第一个 model turn 内就把 thinking token 写到天花板，生成耗时撞墙，**turn 永远没结束** → `after_model` 的 SealGate 永不触发 → seal 永远漏调。

### 1.4 prompt 自身就是诱因（自相矛盾）

`data_analyst.py` 的 system_prompt 处处写「只读不算」「统计层已算好」，**却同时命令它亲手跑完整搬运/审计流程**：

- **step 2.7b**：「把每条 outlier_diagnostics **映射进** outlier_findings 数组」——要求逐条搬运 63 条
- **step 2.8 a–f**：参数审计决策树——遍历每参数、判 Phase1/Phase2、算 p10/p90、套公式、5 选 1 `mismatch_kind`、按 subject 比例定 severity、外加一堆降级分支

对 thinking 模型，「逐条映射 63 条」「遍历参数走决策树」这些**机械动作本身**必然在 thinking 里全展开。**它不是不听话，是 prompt 的任务结构逼它进 thinking 黑洞。**

### 1.5 参数审计（step 2.8）的判据事实上不存在

参数审计要的是「参数值 vs 数据分布」尺子（如「velocity_threshold 对小鼠该 ≤X mm/s」）。核实行为学同事产出（`~/behavioral-book/reward-criteria/*.yaml` 6 范式 + 680 页书）：判据只有 `design_criteria`/`metric_completeness`/`interpretation_direction`/`confound_checks`/`forbidden` 五类，**无一条数值参数判据**。这不是偶然——「阈值该设多少」高度依赖装置/帧率/品系，是**现场标定的工程问题**，不是普适领域判据，最懂的人拿全书也产不出。prompt step 2.8 里满是 `判据待补，参见 issue #63` 即此现实的自证。

且本次数据 `parameters_used` 全是 `open_arm_zones:["open"]` 这种**离散列名选择**（HITL 已确认），无「配不配分布」可言——参数审计 100% 空转，正确产出仅一条占位 `info` finding，却为此烧掉大半个 turn。

消费端核实：`parameter_audit_findings` 唯一消费方是 `present_assumptions.py`（注释明写「Sprint 7 轻量不强制，lead 按需调用」的可折叠卡片），不进 gate、不阻断、不进正式 report。**核心燃料不存在 + 消费端可选不强制 + 占判读关键路径** = 纯负担。

### 1.6 判读判据已在 repo（data-analyst 不缺判据，缺「强制查表」）

`packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/epm.md`（6269B，commit `5b61fafd`/`2c79cc3d`/`96d6e2c4`，已合）已含 data-analyst 判读所需全部：运动抑制混杂排查、解读方向（开臂↓=焦虑↑）、组间比较口径、脱险点、风险评估指标取舍。**data-analyst 之所以在 thinking 里现编判读，部分是因为 prompt 没把它钉死到「先 read 这份判据、据此判读」。**

---

## 二、设计原则

### 2.1 判读归 data-analyst，计算/搬运归 code-executor（铁律落地）

「计算归计算、判读归判读」。counterfactual / LOO 均值 / subject→Trial 映射全是**确定性计算/数据变换**，归 code-executor 用代码做完、写成可直接消费的成品。data-analyst 只产出**意义**（哪条最该警惕、效应量解释、混杂判断），不产出**搬运**。

### 2.2 thinking 不背机械工作量

凡是「遍历 N 条逐条处理」「走多分支决策树」的步骤，对 thinking 模型都是膨胀源——要么下沉成代码（确定性），要么从 prompt 删除。data-analyst 的 thinking 只用来做**判断**（少量、高价值），不用来跑算法。

### 2.3 缺判据的 feature 不占判读关键路径

参数审计的领域判据事实上造不出来、消费端可选。判据未就绪前它不该在 data-analyst 路径上空转。`parameter_audit_findings` 保留 schema 字段（向前兼容 + 将来以确定性代码接入），但 data-analyst **不再产出**它（恒空数组）。

### 2.4 判据已在 repo，强化 read 而非新建消费链

判读判据已在 `by-experiment/<paradigm>.md`（叙述层，给 agent）。本 spec 只**强化** data-analyst 必 read 它并据此判读，**不碰** reward YAML（规则层，给 RL），**不新建**结构化判据消费链——守 single-source-of-truth + 双消费物理分离。

### 2.5 不破坏 outlier_findings 下游契约

`report-writer` 消费 `handoff_data_analyst.json.outlier_findings`。下沉后该字段**结构不变**（subject/metric/value/deviation/counterfactual），只是 `subject` 从 `subject #i` 变成 `Trial N`、由 code-executor 预填而非 data-analyst 手搬。

---

## 三、改动清单（按文件）

### 3.1 `run_metric_plan_tool.py`（及 outlier 旁路写出处）—— `subject #i` 解析为真实 `Trial N`

**问题**：`_outliers.json` 每条 `subject` 是组内序号 `subject #i`，逼 data-analyst 在 thinking 里用 `groups.json`/`per_subject` 翻译成文件名。

**改动**：写 `_outliers.json` 时，用本次已有的 `groups.json` + `per_subject` 的真实文件名（如 `Raw data-EPM-Xuhui-Trial 3`）把 `subject #i` 解析为人类可读的 subject 标识（取文件名或其末段，如 `Trial 3`）。`counterfactual` 串里的 `subject #i` 同步替换。**这是纯确定性映射**（组内序号→该组第 i 个文件名），在写旁路文件那一步用代码完成。

> 实现位置：outlier 旁路落盘的函数（spec 2026-06-18 handoff-slimming 引入的 `_slim_payload_into_sidecars` 或其上游产 outlier_diagnostics 的统计层）。映射所需的 group→files 关系在 `groups.json` 已有，per_subject 的 key 即文件名。

### 3.2 `run_metric_plan_tool.py` / 统计层 —— `_outliers.json` 成品对齐 `OutlierFinding` 形态

**目标**：让 data-analyst 无需「逐条搬运/重组字段」，可直接 `OutlierFinding(**entry)` 构造或最小投影。

**现状字段缺口（实证）**：`OutlierFinding`（`handoff_schemas.py:611`）字段为 `subject / metric / value / deviation / counterfactual`，其中 **`deviation` 是定性字符串**（如 `"2x group median"` / `"> 1.5 SD above mean"`）。但 `_outliers.json` 现有条目是 `group / subject / value / deviation_sd(数值) / deviation_median_ratio(数值) / group_mean / group_std / group_median / loo_mean / loo_std / counterfactual / metric`——**有两个数值 deviation 字段，没有现成的定性 `deviation` 串**。

**改动**：在写 `_outliers.json` 那一步（确定性代码），为每条**合成定性 `deviation` 串**（如由 `deviation_median_ratio=2.0` → `"2x group median"`、或由 `deviation_sd` → `"> 1.5 SD above mean"`），使每条含 `OutlierFinding` 所需的 5 个键且可直接构造。保留原数值字段（`extra="allow"`，不破坏已有读取方）。`subject` 用 §3.1 解析后的真实 Trial 号。**这样 data-analyst 引用时无需在 thinking 里把 `deviation_median_ratio=2.0` 翻译成 "2x group median"**——又一个本属机械变换、不该进 thinking 的点。

### 3.3 `data_analyst.py` SKILL —— 删 step 2.8 整段 + 收窄 step 2.7b + 强化 step 2.6

**删除**：

- **step 2.8 参数适配性审计整段**（Phase1/Phase2 分流、a–f 决策树、5 种 mismatch_kind、降级填法、捷径）——全删。
- `<gate_signals_contract>` 中 `parameter_audit_findings_count` / `parameter_audit_critical_count` 两行：保留输出但**恒为 0**（data-analyst 不再产参数审计）。
- `<handoff_field_format>` 里 `step 2.8 parameter_audit_findings 每条` 说明段——删。

**改写 step 2.7b（核心价值段）**为「**只引用不搬运**」：

```
b. 离群判读（不搬运、不重算、不重映射）：
   - 主 handoff 的 outlier_diagnostics_count == 0 → outlier_findings = []，跳过。
   - count > 0 → read_file <outlier_diagnostics_ref>。该文件每条已是【成品】：
     subject 已是真实标识（如 "Trial 3"）、counterfactual 已预格式化、LOO 数值已算好。
   - 你的职责是【判读】：从这些成品里挑出【最该警惕的 2-3 条】（如 total_entry_count=1
     的可疑个体、驱动组均值的极端值），解释它们对结论稳健性的影响，写进 key_findings /
     outlier_findings。outlier_findings 直接【引用】旁路文件对应条目，不要逐字段重组、
     不要把 63 条全部搬进数组、不要在 thinking 里重算 counterfactual 或重新映射 subject 号。
   - 兜底：旁路文件缺失/读取失败 → 只定性指出哪些 subject 看起来离群 + 方向，不给精确数字。
```

**强化 step 2.6**（判读判据 read）为**硬前置**：

```
2.6 【必读，判读的判据来源】read_file
    /mnt/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/<paradigm>.md
    （paradigm slug 取自 handoff_code_executor.json 的 paradigm 字段）
    该文档含本范式的【混杂排查清单】【解读方向】【组间比较口径】【脱险点】【指标取舍】——
    你的 key_findings / method_warnings / recommendations 必须据此判读，不要在 thinking 里
    现编范式判据。例如 EPM 必查"开臂↓是否伴随总入臂↓（运动抑制混杂）"——该判据来自此文档。
```

**改写 step 2.7 引导句**：删除「读 outlier_diagnostics 旁路文件获取离群受试者 + leave-one-out 反事实 + 选定判据 + 比对参数」中的「**比对参数**」（参数审计已删）；明确「thinking 只做判断，不做搬运/重算/映射」。

### 3.4 `handoff_schemas.py` —— 保留字段、更新注释（向前兼容，不删 schema）

`DataAnalystHandoff.parameter_audit_findings` 字段**保留**（默认空数组），`parameter_audit_findings_count`/`parameter_audit_critical_count` gate_signals 字段**保留**（默认 0）。更新 docstring：标注「2026-06-18 起 data-analyst 不再产出参数审计；判据未就绪，将来以确定性代码（code-executor/独立 validator）接入；present_assumptions 消费方对空数组已 graceful」。`ParameterAuditFinding` 类**保留**（present_assumptions 仍 import；将来复用）。

### 3.5 `present_assumptions.py` —— 确认对空 `parameter_audit_findings` graceful（很可能无需改）

核实：`parameter_audit_findings` 为空数组时不渲染「参数审计」段（应已是 graceful，因其「简单分析→不出面板」契约）。若已 graceful 仅加测试；若否，补空数组短路。

---

## 四、测试（红→绿坐实，TDD 强制）

### 4.1 红线：`_outliers.json` subject 已解析为真实 Trial 号

用本次真实 payload（27 subject / `groups.json` / per_subject 文件名），断言 `_outliers.json` 每条 `subject` **不含** `subject #`、**含**真实标识（如 `Trial 3`），且 `counterfactual` 串内同步替换。改动前该测试红（现状是 `subject #i`）。

### 4.2 `_outliers.json` 字段与 OutlierFinding 对齐

断言每条含 outlier_findings 所需全部键，可被 `OutlierFinding(**entry)` 直接构造（或经最小投影）。

### 4.3 data-analyst SKILL 静态契约（防回归）

- 断言 `data_analyst.py` system_prompt **不含** `参数适配性审计` / `mismatch_kind` / `Phase 2 优先路径` / `a–f` 等 step 2.8 标志串。
- 断言 system_prompt **含** step 2.6 的 `by-experiment/<paradigm>.md` 硬前置措辞 + step 2.7b 的「不搬运/不重算/不重映射」措辞。

### 4.4 schema 向前兼容

断言 `DataAnalystHandoff(parameter_audit_findings=[])` 合法、`gate_signals` 两个 audit count 默认 0；`ParameterAuditFinding` 仍可 import。

### 4.5 present_assumptions 空数组 graceful

断言 `parameter_audit_findings=[]` 时不渲染参数审计段、不抛错。

### 4.6 裸导入两生产入口（改 tools/builtins + subagents 核心，铁律）

`PYTHONPATH=. python -c "import app.gateway"` + `python -c "from deerflow.agents import make_lead_agent"` 均 0 退出。

### 4.7 全量回归（铁律）

改 `run_metric_plan_tool.py`（共享）后跑 backend 全量 + ethoinsight 全量；已知 pre-existing 污染（`test_chart_maker_config_basic_fields` 等）排除，不归因本批。

---

## 五、验收标准

1. **端到端**：同一份 27-subject EPM 数据复跑，data-analyst **单次** model turn 内完成判读并调 `seal_data_analyst_handoff`，**不超时、不打转**，产出 `handoff_data_analyst.json`。
2. **thinking 瘦身可观测**：data-analyst 不再逐条搬运 63 条 / 不再跑参数审计决策树；`outlier_findings` 仅含**精选的 2-3 条**（最该警惕的），`subject` 为真实 Trial 号。
3. **判读质量**：key_findings 体现 epm.md 判据（如显式判定「总入臂无差异 → 排除运动抑制」），非泛泛复述统计数字。
4. **参数审计归零**：`parameter_audit_findings == []`、两个 audit count == 0；present_assumptions 不报错。
5. 全量回归绿（除已知污染）。

---

## 六、风险与注意事项

1. **SealGate 救不了 turn 内超时**（§〇 边界）：本 spec 是治本（让 turn 能正常结束），不要试图靠调 SealGate / 加兜底解决——根因未隔离前别叠加兜底。
2. **不调 thinking_enabled / timeout 旋钮**（用户已定「先定根治不止血」）：调旋钮会把响亮的超时故障变哑故障（思考变浅、结论变水却不报错）。本 spec 落地后若仍偶发超时，再单独评估旋钮，但不在本 spec 内。
3. **subject→Trial 映射的边界**：映射依赖 `groups.json` 组内顺序与 per_subject 文件名一致。若某范式/降级路径 groups 为空或文件名缺失 → 映射降级为保留 `subject #i`（不阻断），并在该条 `deviation` 注明。测试需覆盖此降级。
4. **不删 ParameterAuditFinding / present_assumptions**：将来判据若以确定性代码就绪，复用此 schema + 消费方。本 spec 只断开「data-analyst 产出它」这条路径，不拆基础设施。
5. **report-writer 契约**：`outlier_findings` 结构不变（仅 subject 标识更友好、条数更少）。核实 report-writer 不依赖「全部 outlier 都在数组里」的假设（它本就只引用 data-analyst 精选的）。

---

## milestone 建议

本会话让「harness 鲁棒性 / dogfood 根因治理」track 再进一步：第二轮 4 根因（#154-157）全合后，第三轮复跑暴露**第 9 个结构根因**——**data-analyst thinking 把计算/搬运当产出 → turn 内超时 → SealGate（after_model 门）结构性救不了**。建议在该 milestone 记录：① 此根因 + 本 spec；② **可复用边界教训**：`after_model` 类 seal 门只覆盖「收尾时漏 call」，不覆盖「turn 内超时根本收不了尾」——判 seal 漏调先分「收尾漏 call / turn 内超时」两类；③ **判据现实**：「参数值 vs 数据分布」数值判据行为学上造不出来（依赖装置/品系现场标定），参数审计 feature 应等确定性标定代码而非领域判据；④ data-analyst 的判读判据已在 `by-experiment/*.md`（repo 内），缺的是 prompt 强制查表而非缺判据。
