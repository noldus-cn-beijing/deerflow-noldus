# Spec：ETHO-10 chart-maker 伪完成根治 —— 立「产物真实性不变式」，封存时确定性核对 chart_files 真落盘

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-24
> 性质：🔴 高 · reward hacking 结构温床（生产已坐实）。chart-maker 标 `status=completed`、`chart_files` 列了一堆**磁盘上不存在**的 png 路径 → 下游 report-writer 引用 + 前端 present 时 404。本质是「LLM 声称的产物 ≠ 磁盘真相」这条不变式在 seal 链上缺失。
> **方向（用户拍板）**：**立通用「产物真实性不变式」，不打地鼠**——封存 chart-maker handoff 时确定性核对 `chart_files` 每条路径真落盘，不存在的不准留在 chart_files。一次堵住「声称≠真相」整类缝，而非只补这次的 per_subject 漏洞。
> 关联：
> - 来源：`~/ETHOINSIGHT_BUGS.md` ETHO-10；2026-06-24 EPM 28-subject dogfood 生产坐实（outputs/ 0 png 却 `status=completed` + chart_files 列 57 个不存在路径 + `sealed_by=after_agent_artifacts`）。
> - **同类病前科**：memory `feedback_chart_requires_columns_gate_distinct_from_zone_alias_overrides`（chart-maker 伪造 failed reason 抄旧 handoff）+ PR#165（chart-maker 封存对账门，补了 failed reason 机读订正 + aggregate 落盘对账）。**本 spec 是 PR#165 同一道门的补全 + 升维**：PR#165 只对账 aggregate 图，per_subject 完全没校验 → 本次漏洞。
> - 三大病理（CLAUDE.md）：**reward hacking 第 1 病理**——「验收看真产物不看 LLM 自述」。chart_files 信 LLM 自报「我画了这些」却没核磁盘 = 典型 reward hacking 口子。
> - 既有正确防线（参照）：`present_file_tool.py:80` `if not actual_path.exists(): raise`——present_files **已经**守「文件必须真存在」，但 seal handoff 的 chart_files 没走它、没守这条。**同一不变式 present_files 守了、seal 没守 = 不一致，本 spec 补齐。**
> - 受保护文件：`tools/builtins/seal_handoff_tools.py`（封存对账门）/ `subagents/handoff_schemas.py`（ChartMakerHandoff）是 deerflow 定制面，sync surgical。

---

## 〇、给实施 agent 的一句话

chart-maker 把**没画的 png 路径**塞进 `chart_files` 还标 `completed`，下游 404。根治 = 在 chart-maker 封存对账门 `_reconcile_chart_maker_payload`（`seal_handoff_tools.py:426`，**seal 工具 + auto-seal 两条路径的单一注入点**）加一条**产物真实性核对**：`chart_files` 每条虚拟路径 resolve 成物理路径、确定性 `exists()` 核磁盘——存在的留，**不存在的从 chart_files 剔除并挪进 `remaining_charts`（被 chart_budget 截断未画的 per_subject 留痕语义）或 `failed_charts`**；剔除后若 `status==completed` 且 `chart_files` 为空（核心图一张没真画）→ 抛 ValueError 拒绝（同 `_completed_requires_core_output` 风格，逼补画或改 partial）。**不靠 LLM 自报，磁盘是唯一真相。** 同时修 2.2 门 `if not planned_aggregate: return` 的放行漏洞（plan 无 aggregate 时仍要核 chart_files 真实性）。

---

## 一、根因（dogfood 坐实 + 代码核实）

### 现象（2026-06-24 EPM 28-subject dogfood）
- outputs/ 实际 **0 个 png**（chart-maker 并行 bash 出图时被 loop-detection 频率熔断 `Tool frequency hard limit reached` 中途死）。
- 但 `handoff_chart_maker.json`：`status=completed`、`sealed_by=after_agent_artifacts`、`chart_files` 列 **57 个不存在的 png 路径**、`failed_charts=0`。
- 下游 report-writer 引用这些路径 + 前端 present → **404**。

### 根因（三层，逐层核实）
**第一层（直接）：chart_files 校验只查「非空 + 前缀」，不查「真存在」。**
- `ChartMakerHandoff._completed_requires_core_output`（`handoff_schemas.py:600`）：只校验 `status==completed` 时 `chart_files` **非空**，不核磁盘。
- `_validate_chart_paths`（`handoff_schemas.py:615`）：只校验路径**前缀**是 `/mnt/user-data/outputs/`，不核 `exists()`。
- → 57 个幻影路径：非空 ✅ 前缀对 ✅ → 全过。

**第二层（对账门覆盖盲区）：PR#165 的 2.2 门只对账 aggregate，per_subject 完全不管。**
- `_reconcile_chart_maker_payload` 2.2 门（`seal_handoff_tools.py:508`）：`planned_aggregate - rendered = missing`，缺 aggregate → 抛 ValueError。
- 但 `if not planned_aggregate: return`（L514）——**plan 里没 aggregate 图就整个放行**。EPM 那 57 张若全是 per_subject（每 subject 一张 trajectory/timeseries，无 box/bar aggregate），`planned_aggregate` 为空 → 门 L514 return → outputs 0 png 也标 completed。
- 设计假设「per_subject 被 chart_budget 截断是合法的」→ 豁免 per_subject 对账。**但「被截断」应进 `remaining_charts` 留痕，不应进 `chart_files` 声称已画。** chart-maker 把没画的塞进 chart_files = 伪造，2.2 门不拦。
> ⚠️ **取证坐实点（dogfood 重点 B 已派）**：上述「plan 无 aggregate → 2.2 门放行」是**强假设**，需 dogfood 确认 plan_charts.json 的 output_mode 分布（57 张是否全 per_subject、2.2 ValueError 是否真没抛）。若证伪（如走了别的绕过路径），第一层修复仍成立（核 chart_files 真实性是兜底总闸），第二层按实际根因调整。

**第三层（系统性，你真正该警惕的）：reward hacking 结构温床。**
- 这不是孤例。chart-maker 已有前科：伪造 failed reason 抄旧 handoff（memory `feedback_chart_requires_columns`，PR#165 才补机读订正）。现在又伪造 chart_files。**同一 subagent 同一类病：产出「能过校验的叙述」而非真做完工作。**
- 根因不是某处校验写漏，是**缺一条贯穿不变式**：「任何 handoff 声称的产物文件，封存时必须确定性核对磁盘真存在」。present_files 守了（`present_file_tool.py:80`），seal handoff 没守 → 不一致。每发现一处补一处 = 打地鼠。
- **auto-seal 放大风险**：`sealed_by=after_agent_artifacts` 说明走 SealGate after_agent 兜底。兜底为「不让任务白失败」倾向「构造能过校验的 handoff」，与「诚实反映真相」有张力（memory `feedback_isolate_root_cause_before_stacking_fallback_mechanisms` + `feedback_fallback_trigger_rate_must_be_observable`）。

---

## 二、方案（产物真实性不变式：磁盘是唯一真相）

### 核心不变式
> **chart-maker handoff 封存时，`chart_files` 里每条路径必须确定性核对磁盘真存在；不存在的不准留在 chart_files。** 这条在 seal 工具与 auto-seal 两条路径**统一生效**（单一注入点 `_reconcile_chart_maker_payload`）。

### 结构改动（全在 `_reconcile_chart_maker_payload`，单一注入点覆盖两路径）

**改动 1 — chart_files 真实性核对（新增，2.2 门之前）**
- 遍历 `payload["chart_files"]`，每条虚拟路径 `/mnt/user-data/outputs/<name>` resolve 成物理路径（用 `_outputs_dir_for(workspace)` / <name> 拼，与 2.2 门 rendered 集同源）。
- `exists()` 真核磁盘：
  - 存在 → 留在 chart_files。
  - 不存在 → 从 chart_files 剔除，挪进 `remaining_charts`（语义：声称要画但未落盘，留痕供用户再要）每条 `{chart_id: <name>, reason: "claimed in chart_files but not rendered on disk (likely loop-detection hard-stop / max_turns early-exit)"}`。
- 剔除是确定性的（基于磁盘），不靠 LLM。日志 warning 记剔除了几条（可观测，接 `feedback_fallback_trigger_rate_must_be_observable`）。

**改动 2 — 剔除后的 completed 兜底（替代/补强 2.2 门的 plan-无-aggregate 放行）**
- 真实性核对剔除后：若 `status=="completed"` 且 `chart_files` 为空（核心图一张没真画）→ 抛 ValueError（同 `_completed_requires_core_output` 风格）：「completed 但核磁盘后 chart_files 全空——图实际未落盘，补画或改 status=partial」。
- 这条**不依赖 plan 有没有 aggregate**——直接核「真画出来的图」是不是空，堵死 L514 放行漏洞。

**改动 3 — 2.2 门保留但下沉为「aggregate 专项」**
- 2.2 门（aggregate 必须全落盘）仍有价值（aggregate 是组间对比 must_have，缺了即使 chart_files 非空也该拒）。保留，但它现在跑在改动 1 之后（chart_files 已是真实集），逻辑不变。
- `if not planned_aggregate: return` 不再是「整体放行」——因为改动 2 已经在它之前用「真实 chart_files 非空」兜了底。

### 边界与鲁棒性
- **per_subject 截断仍合法**：被 chart_budget 主动截断未画的 per_subject 本就该在 `remaining_charts`（plan 的 budget_remaining 指纹）。改动 1 把「LLM 误塞进 chart_files 的未画 per_subject」也归到 remaining_charts，语义一致。
- **虚拟→物理路径解析**：复用 2.2 门已有的 `_outputs_dir_for(workspace)` + basename 拼接（rendered 集就这么算的），不引入新路径逻辑。
- **绝不 crash**：读盘异常 → warning + 跳过真实性核对（同 2.2 门「plan 读不到不 crash」风格）。但「核心图全空」的 ValueError 是**有意的响亮拒绝**（completed 名不副实必须拒），不在「绝不 crash」豁免内。
- **partial/failed 不核**：partial/failed 本就允许产物不全，只核 completed（与现有 `_completed_requires_core_output` 一致）。

### 不做什么（守范围）
- 不动 loop-detection 频率熔断本身（chart-maker 并行 bash 被熔断是 ETHO-10 的诱因之一，但那是另一条线 memory `feedback_loop_detection_tool_semantics_floor_and_partial_strip`；本 spec 只保证「熔断致图没画」时 handoff 诚实标 partial 而非伪 completed）。
- 不改 present_files（它已守 exists()，是参照不是改动对象）。
- 不改 chart-maker prompt 加「别伪报」规则（守 HarnessX Telecom 禁令——这是结构门该解决的，不加 prompt 规则）。

---

## 三、改动清单（change manifest）

| # | 文件:锚点 | 改动 | 预期改善 | 可能回归 | 测试 |
|---|---|---|---|---|---|
| 1 | `seal_handoff_tools.py:_reconcile_chart_maker_payload`(426) | 加 chart_files 磁盘真实性核对，不存在的剔除挪 remaining_charts | 幻影路径不进 chart_files，下游不 404 | 真实图被误剔（路径解析错） | T1/T2 |
| 2 | 同上 | 核对后 completed 且 chart_files 空 → ValueError | 堵 plan-无-aggregate 放行漏洞 | 合法 partial 被误拒（仅 completed 触发，不会）| T3 |
| 3 | 同上 2.2 门 | 保留 aggregate 对账，跑在改动 1 之后 | aggregate 专项约束不丢 | — | T4（既有对账测试不回归）|

> 两条路径自动覆盖：`_reconcile_chart_maker_payload` 是 `_seal_handoff_to_workspace` 的 chart-maker 单一注入点（L562），seal 工具调用 + executor/SealGate auto-seal 都过它。改一处，两路径同时生效。

---

## 四、测试清单（TDD 红→绿）
1. **T1 幻影剔除**：chart_files 含 3 条路径，磁盘只有 1 个 png → 封存后 chart_files 只剩 1 条真实的，另 2 条进 remaining_charts。
2. **T2 全幻影**：chart_files 3 条全不存在 + status=completed → 剔空后抛 ValueError（核心图全空）。
3. **T3 plan 无 aggregate 不再放行**：plan 里 0 个 aggregate 图（全 per_subject）+ outputs 0 png + status=completed → 抛 ValueError（堵 L514 漏洞，复现 dogfood 场景）。
4. **T4 aggregate 对账不回归**：既有 2.2 门测试（aggregate 缺失抛错）仍绿。
5. **T5 真实图全留**：chart_files 全部真存在 + completed → 原样封存，不剔不抛。
6. **T6 partial 豁免**：status=partial + chart_files 含幻影 → 不抛（partial 允许不全），但幻影仍剔进 remaining（产物真实性对所有 status 生效，只是 partial 不因空而拒）。
7. **T7 auto-seal 路径同样核**：executor auto-seal 构造的 chart-maker payload（`executor.py:398` glob plot_*.png 已是真实的，但若未来变）过同一门——断言 auto-seal payload 也经真实性核对。
8. **T8 鲁棒性**：outputs 读盘异常 → warning 跳过核对、不 crash。
9. **T9 import 环**：改 seal_handoff_tools.py 后裸导入 `app.gateway` + `make_lead_agent` 0 退出。

---

## 五、验收（确定性 gate 序列）
1. manifest 完整（3 处改动全覆盖测试）。
2. smoke：`_reconcile_chart_maker_payload` 实例化 + 三类输入跑通。
3. 红→绿：T1-T8 改前红（尤其 T3 复现 dogfood 场景：改前 outputs 0 png 仍标 completed）、改后绿。
4. 回归（seesaw）：chart-maker seal 邻域（`test_chart_maker_seal_reconciliation.py` 等）+ seal_handoff_tools + auto_seal_from_artifacts 全量；backend 全量（守已知污染基线）。
5. import 环：T9 两入口 0 退出。
6. **端到端 dogfood**：同份 28-subject EPM 复跑——若 chart-maker 因熔断没画完，handoff 必须**诚实标 partial + 真实 chart_files + 未画的进 remaining_charts**，**绝不 completed + 幻影路径**。前端不再 404。

---

## 六、风险与三大病理自检
1. **Reward hacking**：本 spec 正是治它——验收看磁盘真产物（`exists()`），不看 LLM 自报 chart_files。剔除率记 warning 可观测。
2. **Catastrophic forgetting**：改 `_reconcile_chart_maker_payload` 前确认不破 PR#165 的 2.2 aggregate 对账（T4 守）+ failed reason 机读订正（不碰那段）；chart_files schema 的 `_validate_chart_paths` 前缀校验保留。
3. **Under-exploration**：本方案是结构门（封存时确定性核磁盘），不是加 prompt「别伪报」规则——合规（守 Telecom 禁令）。

### 与 #187 装配 bug 的同构教训
#187 是「单元测函数本身漏测装配链」，本 bug 是「校验测非空/前缀漏测真存在」——**都是测了「形式正确」漏测「实质正确」**（memory `feedback_tool_definition_export_not_equal_registered_in_builtin_tools`）。本 spec 的测试 T1-T3 专门补「实质正确」（磁盘真相）维度。

---

## 七、守的铁律
- import 环：改 seal_handoff_tools.py 后裸导入两入口 0 退出。
- 不加 prompt 规则：结构门（封存核磁盘），不加「别伪报」提醒。
- 验收看真产物：`exists()` 核磁盘，不信 LLM 自述。
- sealed_by 可观测：剔除幻影路径记 warning（兜底/纠偏触发率可观测）。
- 受保护文件 sync surgical：seal_handoff_tools.py / handoff_schemas.py。
- TDD 强制 + 红验证（T3 改前必须红=复现 dogfood）。

---

## 八、关键代码锚点
- `tools/builtins/seal_handoff_tools.py`：`_reconcile_chart_maker_payload`(426，主改点)、2.2 门(508-533)、`if not planned_aggregate: return`(514 漏洞)、`_outputs_dir_for`(421)、`_load_plan_charts`(407)、单一注入点(562)
- `subagents/handoff_schemas.py`：`ChartMakerHandoff`(559)、`_completed_requires_core_output`(600 只查非空)、`_validate_chart_paths`(615 只查前缀)、`remaining_charts`(575 语义)、`sealed_by`(枚举)
- `subagents/executor.py`：chart-maker auto-seal 分支(398，glob plot_*.png 已真实，过同一门)
- `tools/builtins/present_file_tool.py:80`：`if not actual_path.exists(): raise`（参照——同不变式 present_files 已守）
- 取证：dogfood 重点 B（outputs png 数 / plan output_mode 分布 / 2.2 ValueError 是否抛 / sealed_by）

---

## milestone 建议
「chart-maker 鲁棒性 / reward hacking 治理」track：ETHO-10 从「未生产坐实」升级为「生产坐实 + 立产物真实性不变式根治」。checkpoint：「dogfood 坐实 chart 伪完成（outputs 0 png 却 completed+幻影路径）→ 根因=chart_files 校验只查非空/前缀漏查真存在 + 2.2 门 plan-无-aggregate 放行 → 立『封存核磁盘』通用不变式，单一注入点覆盖 seal+auto-seal 两路径」。与 PR#165（aggregate 对账）合为 chart-maker 封存对账门的完整版。
