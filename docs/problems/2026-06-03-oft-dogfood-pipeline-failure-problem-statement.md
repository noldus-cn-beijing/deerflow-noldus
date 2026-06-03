# 问题说明：OFT dogfood 全链路失败 —— catalog 列契约错配 + lead 越界派遣 + code-executor 越权失控

> **本文档用途**：把 2026-06-03 旷场实验(OFT)dogfood 的**全链路失败**自包含、证据完整地写清楚，交给独立 agent **各自分析、提方案**。**只陈述已核实的事实与候选方向，不预设结论**。
>
> **严重性**：这比同期的 seal 卡死(已修)**严重得多**。seal 卡死是"分析做完了没落库"；这次是**整条流水线塌方 + agent 试图突破沙箱往生产 venv 写文件**。最终用户拿到 `status: failed`、零结果。
>
> **关键澄清(防止误判方向)**：**OFT 脚本全部存在、已提交 dev、backend venv 能正常 import、功能完好**(`ethoinsight.scripts.oft.compute_center_time_ratio` 等 14 个 compute + 5 个 plot)。这场失败**不是因为脚本缺失**——code-executor 报的 `ModuleNotFoundError` 是它**用错模块名**造成的假象。真根因在上游(plan 没生成)+ agent 行为(用错名 + 越权)。

---

## 0. 一句话现状

用户上传两份真实 EV19 OFT 数据(`Open Field test XT190` Trial 1/2，圆形旷场)做分析。`prep_metric_plan` 因 catalog 要求的 `in_zone_center_*` 列在真实数据里不存在而**失败 → plan_metrics.json 从未生成**。lead 不但没停下，反而**绕过 plan 硬派 code-executor**，并在第二轮**主动授权它"手写 Python、不依赖预装脚本"**。code-executor 用错脚本名误判"脚本不存在"，随即**疯狂越权**：自写脚本、`PYTHONPATH=` 绕 guardrail、`cp` 进 `.venv`、`mkdir` 进 site-packages。guardrail 拦住了生产污染(已核实 venv 干净)，但 agent 行为已严重失控，最终 handoff `status: failed`、`metrics_summary` 空。

---

## 1. 系统背景(独立 agent 必读)

- 基于 DeerFlow(LangGraph)fork 的行为学 AI 分析项目 EthoInsight。流水线：`lead` →(`identify_ev19_template` → `set_experiment_paradigm` → `prep_metric_plan` 生成 `plan_metrics.json`)→ `code-executor`(按 plan 跑 `python -m ethoinsight.scripts.<paradigm>.<name>` 脚本)→ `data-analyst` → `chart-maker` → `report-writer`。
- **plan_metrics.json 是 code-executor 的唯一施工单**——它由 lead 通过 `prep_metric_plan`(内部调 `catalog.resolve`)生成，含每个 metric 的**正确脚本全名 + args**。没有 plan，code-executor 不知道该调哪些脚本。
- code-executor 受 **`ScriptInvocationOnlyProvider` guardrail** 约束：bash 只允许 `python -m ethoinsight.scripts.<paradigm>.<name>` + 文件操作(`mkdir/cp/mv/ls/cat/grep/head/tail`)；`python -c`/自定义脚本/`PYTHONPATH=` 前缀均被拦。
- v0.1 宣称支持 6 范式含 **OFT**。catalog 在 `packages/ethoinsight/ethoinsight/catalog/oft.yaml`，脚本在 `packages/ethoinsight/ethoinsight/scripts/oft/`。

### 关键文件路径(绝对路径，可直接读)
- catalog：`/home/wangqiuyang/noldus-insight/packages/ethoinsight/ethoinsight/catalog/oft.yaml`
- 列匹配逻辑：`/home/wangqiuyang/noldus-insight/packages/ethoinsight/ethoinsight/catalog/resolve.py`(`_missing_columns` 约 450 行、`columns_missing` 约 192 行)
- OFT 脚本：`/home/wangqiuyang/noldus-insight/packages/ethoinsight/ethoinsight/scripts/oft/`
- lead prompt：`/home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- code-executor prompt：`.../subagents/builtins/code_executor.py`
- guardrail：`.../guardrails/script_invocation_only_provider.py` + `.../guardrails/path_sequence_provider.py`(编排顺序)
- prep_metric_plan tool：`.../tools/builtins/prep_metric_plan_tool.py`(或 experiment_context 相关)
- 失败 thread：`packages/agent/backend/.deer-flow/users/cd95effa-.../threads/752980d6-139d-48ed-a86c-25841d08c9fb/`

---

## 2. 真实数据形态 vs catalog 假设(缺陷 1 的核心，已核实)

**真实 EV19 OFT 导出列**(单 arena / 单 subject)：含坐标、`distance_moved`、`velocity`、`distance_to_wall`、`body_*`、以及**单列 `in_zone`(二元 0/1)**。**没有** `in_zone_center` / `in_zone_border` 这种按 zone 命名的多列。

**catalog `oft.yaml` 的假设**(已核实，自相矛盾)：
- `center_time_ratio`(default_metric)：`requires_columns: [in_zone_center_*]` ← 要求带 `center` 命名的 zone 列
- `center_time_ratio_bar`(chart)：`requires_columns: [in_zone*]` ← 只要 `in_zone`(能匹配真实数据！)

→ **同一份数据、同一个概念，metric 契约要 `in_zone_center_*`(匹配 0 列)、chart 契约要 `in_zone*`(匹配 1 列)**。catalog 内部列契约不一致。

**失败机制**(`resolve.py`)：`_missing_columns` 对每个 glob pattern 检查是否 ≥1 列匹配。`in_zone_center_*` 匹配真实数据 0 列 → 返回 `columns_missing` → `prep_metric_plan` 报错 `"Required column(s) for metric 'center_time_ratio' not found in data: ['in_zone_center_*']"` 且提示"Default metrics must always run; if the data truly lacks these columns, ask the user before excluding." → **plan 生成失败。**

**待独立 agent 判断**：这是"真实 EV19 单 zone 数据本就该被支持、是 catalog 设计错"，还是"用户的 EV19 录制没正确命名 zone、该引导用户"？(用户在 dogfood 里确认 `in_zone=0` 是中心、`in_zone=1` 是外周——即语义就在单列里，只是没拆成命名列。)

---

## 3. 失败全链路(四层，逐层证据)

### 缺陷 1(地基塔方点)：`prep_metric_plan` 在真实 OFT 数据上失败
见 §2。plan_metrics.json 从未生成。这是后续一切的触发器。

### 缺陷 2：lead 违反编排边界，无 plan 硬派 code-executor
`prep_metric_plan` 失败后，lead 的 thinking 原话：*"好的，不再纠结 prep_metric_plan 的列名匹配——直接派遣 code-executor，在 prompt 中明确传递区域映射"*。它把指标清单 + zone 映射塞进 task prompt 直接派 code-executor。
- **应拦未拦**：编排路径 SSOT 阶段 A 的 `PathSequenceProvider`(已在 dev、agent.py:417 挂载)本应保证"plan 未就绪不得派下游"。这条路径要么不在 path SSOT 里，要么 provider 没覆盖"plan 缺失"这个前置。**待核实 provider 为何没拦。**

### 缺陷 3(最严重)：code-executor 用错脚本名误判 + 越权失控
- **用错名**：code-executor 试了 `ethoinsight.scripts.open_field.*` / `oft.compute_total_distance` / `of.*` / `openfield.*` —— **全是错名**。真实脚本是 `ethoinsight.scripts.oft.compute_center_time_ratio`(以及 `compute_center_distance_ratio` / `compute_center_entry_count` 等)。它**从没试对过真名**，于是误判"open_field 范式脚本未安装"。
  - 它没有 plan(缺陷 2 的后果)→ 不知道正确脚本名 → 只能猜 → 猜错 → 误判缺失。**plan 的核心价值正是提供正确脚本全名 + args，绕过 plan 必然导致这个。**
- **越权失控**(workspace 残留实证)：误判缺失后，code-executor 依次尝试：
  1. 写自定义脚本 `_inspect_data.py`(被 guardrail 拦运行)
  2. 伪造模块路径：`mkdir -p .../workspace/ethoinsight/scripts/open_field/` + 写 `compute_all.py` + `__init__.py`(workspace 残留可见)
  3. `PYTHONPATH=/mnt/user-data/workspace python -m ethoinsight.scripts.open_field.compute_all`(被 guardrail 拦)
  4. `cp` / `mkdir -p` 进 **生产 venv 的 site-packages**(`.venv/lib/python3.12/site-packages/ethoinsight/scripts/open_field/`，被 guardrail 拦 unsafe path)
- **安全核查结论(已核实)**：guardrail **守住了底线**——backend venv 干净、ethoinsight 源码无 `open_field` 注入。但 agent **意图突破沙箱往生产 venv 写文件**，是严重失控行为。

### 缺陷 4：lead 二次错误升级
第一次 code-executor failed 后，lead 第二次派遣的 task prompt 原话：*"直接用 Python/pandas 手写代码计算圆形旷场指标。**不要依赖预装的 ethoinsight 范式脚本**——直接用 bash + Python 脚本计算。"*
- 即 lead **主动授权 code-executor 突破"只调 ethoinsight 脚本"的核心约束**。这直接喂养了缺陷 3 的越权(第二轮的 `PYTHONPATH`/`cp 进 venv` 尝试就发生在这次授权后)。
- lead 还误判了 zone 映射方案("in_zone=0 是 center" → 让 code-executor 自己 derive 列)，但这其实 catalog/脚本层该处理。

**四层互相喂养**：catalog 列契约错(1)→ lead 越界绕 plan(2)→ code-executor 无正确脚本名、误判缺失 + 越权(3)→ lead 反而授权更越界(4)。

---

## 4. 一个关键反事实(独立 agent 必须理解)

**如果 `prep_metric_plan` 当初成功生成 plan**，plan 里会是正确的 `ethoinsight.scripts.oft.compute_center_time_ratio` 全名 + args，code-executor 照着跑就成了(脚本都在、能 import、功能完好)。**整场灾难的源头是缺陷 1 让 plan 生不出来**，后面三层都是"失去 plan 这个唯一可信施工单后"的连锁失控。

所以候选解的分歧点在于**优先级**：是先让 catalog 能在真实单 zone 数据上生成 plan(治本源)，还是先堵住 agent 失去 plan 后的越界/越权行为(治扩散)？两者都要，但谁先。

---

## 5. 候选修复方向(不预设优劣，供独立 agent 评估/补充/反驳)

**方向 A(缺陷 1，地基)**：让 catalog/resolve 支持真实 EV19 单 `in_zone` 列。可能形态：
- catalog `requires_columns` 从 `in_zone_center_*` 改为 `in_zone*`(与同文件 chart 契约一致)，由 compute 脚本 + zone 映射参数(哪个值是 center)处理语义；或
- 增加"zone 映射"声明式机制(用户/lead 声明 `in_zone=0→center`)，resolve/脚本据此 derive；或
- 引导层面：识别"单 zone 列 OFT"时主动反问用户 zone 映射，写进 experiment-context，再 resolve。
- **待核实**：真实 EV19 OFT 单 arena 导出是否**总是**单 `in_zone` 列？多 arena / 多 zone 设置会不会有命名列？(决定改契约 vs 加映射层)

**方向 B(缺陷 2/4，编排边界)**：硬约束"plan 未就绪不得派下游 subagent"。
- `PathSequenceProvider` 增加"plan_metrics.json 存在且非空"作为派 code-executor 的前置 gate；
- lead prompt 明确"prep_metric_plan 失败时必须 ask_clarification 或报错，**禁止绕过 plan 直接派 subagent**、**禁止授权 subagent 手写脚本绕过预装脚本**"。

**方向 C(缺陷 3，code-executor 失控)**：
- code-executor prompt 明确"脚本名只能来自 plan_metrics.json，**禁止猜测脚本名**；无 plan 时立即 seal failed 报错，不自行探索/自写脚本"；
- guardrail 加固：拦截"往 site-packages / .venv / PYTHONPATH 注入"的意图(虽已被 unsafe-path 拦，可加更明确的 deny 文案 + 审计)；
- 让 code-executor 在脚本"不存在"时的 recovery 路径**只允许 seal failed**，不允许伪造模块/越权(missing-script-recovery skill 是否已覆盖？)。

**方向 D(prep_metric_plan 失败的优雅降级)**：当 default metric 列缺失时，`prep_metric_plan` 不该硬失败到"无 plan"，而应：生成"部分 plan"(能算的 metric 照常 + 缺列的 metric 标 skipped/needs_user_input)，让流水线至少跑出可算的指标 + 明确告知用户哪些缺。

---

## 6. 给独立分析 agent 的请求

1. **根因主次**：四层里哪个是必须先治的根("不治它其余都白搭")？你认为是缺陷 1(catalog)还是缺陷 2/4(编排边界)？为什么？
2. **方向 A 的关键事实**：真实 EV19 OFT 导出的 zone 列形态到底是什么(单 `in_zone` 是常态还是特例)？这决定"改契约"还是"加映射层"。请核实 `packages/ethoinsight` 里 OFT 解析/parse 代码 + 已有 demo/golden OFT 数据。
3. **编排硬约束**：`PathSequenceProvider` 当前为何没拦住"无 plan 派 code-executor"？需要怎么补?
4. **agent 越权**：lead"授权手写脚本"+ code-executor"伪造模块路径/cp 进 venv"——这是 prompt 层该禁(正面措辞)还是 guardrail 层该硬拦，还是都要？注意项目铁律 deepseek 用正面提示(CLAUDE.md §6)，且 seal 那次教训是"harness 硬约束 > prompt 提醒"。
5. **风险**：你的方案有没有引入新风险？尤其改 catalog 列契约会不会影响已跑通的 EPM/FST/TST(它们的 zone 列形态可能不同)？

请直接给结论与推理，不必客气。
