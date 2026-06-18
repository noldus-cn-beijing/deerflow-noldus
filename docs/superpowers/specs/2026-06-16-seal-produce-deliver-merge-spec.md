# Spec: 消除认知型 subagent 的"产出/交付分离"——seal 漏调的结构性根治

> ⚠️ **已被取代**：本纯 prompt 版只做了辅助层（L2），缺结构性主防线。完整两层工程方案见 [2026-06-16-seal-gate-middleware-engineering-spec.md](2026-06-16-seal-gate-middleware-engineering-spec.md)（SealGateMiddleware after_model 闸门 + prompt 合一）。本文件保留作 L2 prompt 改法的细节参考。


> 日期：2026-06-16
> 类型：prompt 结构重构（根治反复复发的 seal 漏调）
> 范围：data-analyst（主）+ chart-maker + report-writer。**code-executor 不改**（已合一，作正面范例）。
> 约束：**纯 prompt 重构，零兜底/防护/harness 代码**。不加 auto-seal、不改 executor、不动 schema。
> 状态：待 review → 批准后实施。

---

## 0. 为什么这次不同：停止打地鼠

seal 漏调（subagent「terminated without emitting handoff」）在本仓库已被"修复"≥4 次，每次判一个新"真根因"、改 prompt/skill/schema，然后换个诱因复发：

| 次 | 当时判的根因 | 改法 | 为何复发 |
|---|---|---|---|
| 1 | prompt 矛盾（空参数降级） | 改 prompt | 只修空参数那条路 |
| 2 | 非空参数走 a–f 长审计 | 又改 prompt | 只修那条触发路 |
| 3 | skill 教 write_file 与 seal 矛盾 | 改 skill | 只修那个 skill 措辞 |
| 4 | schema 拒 list 型 zone 参数 | 改 schema | 只修那个数据形状 |
| 今天 | 空 statistics → 手算耗尽预算漏 seal | —— | —— |

**模式**：每次都在修"这一次让模型在思考里多消耗的那个具体诱因"。但诱因无穷无尽（空参数/旧 skill/schema/空 statistics/下一个未知）。**真根因不是任何一个诱因，是 seal 这个动作在结构上可被跳过。** 用户原话一针见血："真问题不是因为并没有 seal 吗？应该解决这个问题，而不是想办法兜底写屎山代码。"

本 spec 不修任何诱因、不加任何兜底，只改一件事：**让 seal 在结构上不可跳过——产出分析的唯一动作就是调 seal 工具。**

---

## 1. 根因：产出与交付被拆成两步，而 seal 参数 = 产出内容

### data-analyst（重度，本次失败者，已逐字核实 `data_analyst.py`）

workflow 把分析拆成两个先后阶段：
- **step 2.7（L165）**"一次性完成核心分析推理" + **step 2.8（L183-269）参数审计 → 模型在「思考」里写出全部 key_findings / outlier_findings / method_warnings / recommendations / parameter_audit。
- **step 3（L270-278）**"封存 handoff" → 把上面思考里**已得出的结论**作为 seal 工具参数**重新填一遍**。

`seal_data_analyst_handoff` 的参数恰好是 `key_findings / outlier_findings / method_warnings / recommendations / ...`（实测）——**与 step 2 思考产出的内容完全相同**。

→ **对模型，分析在 step 2 已完成；step 3 是把同一批内容誊抄进工具参数的冗余动作。** 模型写完 step 2 的思考就认为任务完成而停止（"现在让我来完成分析和封存"→ 停，无 tool_call）。

**症状指纹（结构病的铁证）**：prompt 在 **7+ 处**反复打补丁求模型别忘 seal——L88-90 / L143 / L165 / L192 / L199 / L270-274。L272 自己写道"把你**已经得出的结论**作为该工具的参数填入"——**承认了结论先于 seal 得出**。一个 prompt 需要 7 处哀求"请真的调那个工具"，正是产出/交付分离的指纹。

**本次 dogfood 精确踩中**（thread `51ff5eb8`）：data-analyst 在 thinking 里把 key_findings / outlier / counterfactual / Cohen's d 全算完写完，正文写到 "Key Findings…" 停止，未发 seal tool_call → `terminated without emitting handoff_data_analyst.json`。

### chart-maker / report-writer（轻度，已核实）

- chart-maker：step 10 跑绘图脚本（png **落盘** outputs/）→ step 11 调 seal 登记 → step 12 present_files → step 13 输出。
- report-writer：step 3 `write_file` 写 report.md（**落盘**）→ step 4 调 seal 登记 → step 5 输出。

分离较轻：**产出物有磁盘载体**（png / md 已真实落盘），seal 只是"登记已存在的文件路径"，且 seal 紧随产出。所以它们漏 seal 时 harness auto-seal 还能从文件重建（有安全网）。但"产出后还要记得再调 seal"的窗口仍在——同模式隐患，只是暂时被兜着没爆。

### code-executor（已合一，不改，作正面范例）

step 2 调 `run_metric_plan` 一个工具**既执行计算又落盘 handoff**（`sealed_by="run_plan"`）→ 正常路径**根本不需要调 seal**（L49："run_metric_plan 已落盘 handoff，无需再调 seal_code_executor_handoff"）。**产出与交付由同一个工具调用完成 → 结构上无法漏。** 这是本 spec 的设计依据：当产出动作本身就落盘交付时，漏 seal 不可能发生。

---

## 2. 统一原则

**调 seal 工具 = 产出分析的唯一动作，不是产出之后的额外登记。**

- 认知产物无磁盘载体（data-analyst）→ **思考只用于推导，结论直接作为 seal 参数产出**，取消"先在思考里组织完整结论文本"这一步。
- 有磁盘载体（chart-maker / report-writer）→ **产出文件后的下一个动作必须是 seal**，消除"产出→（其他步骤）→才 seal"的中间窗口。

---

## 3. 修法（逐 subagent，纯 prompt）

### 3.1 data-analyst（主体重构）

**核心改动：把 step 2.7/2.8（思考产出）与 step 3（seal 誊抄）从"两步"合成"一步"。**

1. **改 workflow 框架措辞**：
   - 删除 / 改写 step 2.7 标题"一次性完成核心分析推理"——改成 **"用思考推导分析素材（识别离群 + 算 leave-one-out + 选判据 + 比对参数）。不要在思考里撰写最终的 key_findings / 各字段文本。"**
   - step 3 从"封存 handoff（把已得结论填进参数）"改成 **"产出分析 = 发出 seal_data_analyst_handoff tool_call。你对本次分析的全部结论，第一次成文就直接写在该工具的参数里（key_findings / outlier_findings / ...）。没有'先在别处写一遍再填进来'——调用这个工具就是你产出分析的方式，也是唯一方式。"**
   - 顺序明确：思考结束 → 直接发 seal tool_call（中间不存在"撰写结论正文"的步骤）。

2. **删除 7+ 处"seal 必达 / 别忘 / 写文本≠落库"补丁**（L88-90 后半 / L143 / L165 括号 / L192 / L199 / L270-274 的哀求性表述）。结构合一后，这些补丁是病的症状，留着反而强化"seal 是个容易忘的额外动作"的心智模型。**保留**纯事实约束："严禁 write_file 写 handoff，必须走 seal 工具"（这是工具选择约束，非分离症状）。

3. **fast-fail 路径同样合一**（L72-122, L135-145, L1a）：触发硬失败 → **直接发 seal tool_call 输出 partial/failed 结论**（status + 简短 key_findings/errors 作为参数），不存在"先写描述性分析、再 seal"。把 1a 的"给出描述性判读后用描述结果填 seal 参数"改成"描述性结论直接作为 seal 参数发出"。删掉"封存是终止动作，分析写完必须调 seal 工具""手算是禁区"这类哀求——改为正向单一出口表述。

4. **gate_signals / 最终 AIMessage（step 4, L280）保持**：seal 之后输出 2-3 段摘要 + [gate_signals]。这部分不属于"产出分析"，是给 lead 的播报，顺序在 seal 之后，不冲突。

**不改**：schema、fast-fail 的判定逻辑、参数审计的判据内容（a–f 的领域判据照旧）、范式文档 read。只改"分析在哪里成文"——从思考改到 seal 参数。

### 3.2 chart-maker（收紧窗口）

- step 10（绘图）与 step 11（seal）之间不留遐想空间：把 step 11 措辞改成 **"绘图脚本一旦执行完（无论成功/失败条目），下一个动作就是 seal_chart_maker_handoff——chart_files 直接列 step 10 写到 outputs/ 的 png 路径，failed_charts 列失败项。这是产出图表工作的交付动作。"**
- present_files（step 12）/ 输出（step 13）顺序不变，但明确它们在 seal 之后。
- 强调：**seal 紧贴绘图，不要先 present_files 或先写摘要再回头 seal。**

### 3.3 report-writer（收紧窗口）

- step 3（write_file report.md）与 step 4（seal）合紧：把 step 4 改成 **"report.md 一旦 write_file 完成，下一个动作就是 seal_report_writer_handoff——report_path 填刚写的 md 路径，sections_written 列已写章节。写完报告文件的交付动作就是这次 seal 调用。"**
- 强调：**write_file 报告后的下一步必须是 seal，不要先写最终 AIMessage 摘要再回头 seal。**

---

## 4. 验收（行为验证为主，纯 prompt 改无单元测试可断言）

prompt 重构无法用 pytest 断言"模型会调 seal"。验收靠 **dogfood 行为复现 + 结构审查**：

1. **结构审查（可立即做）**：改后 grep 三个 prompt，确认：
   - data-analyst 无"先在思考/正文写出完整 key_findings 再填 seal"的表述；step 3 是"产出=调 seal"单一出口；"seal 必达/别忘"哀求补丁已删（保留 write_file 禁令）。
   - chart-maker / report-writer 的 seal 步骤紧贴产出动作，措辞为"产出后下一个动作就是 seal"。
2. **dogfood 复现（治本验证）**：用本次失败的同一数据（`/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`，control=7/treatment=21，列名 open/closed）重跑：
   - 注意：本数据的空 statistics 由 **statistics 列对齐缺口**（另一 spec，用户在修）导致。验证 data-analyst seal 时，**无论 statistics 空与否**，data-analyst 都应一次发出 seal tool_call（走 partial 或 completed），不再 `terminated without emitting`。
   - 即两条 spec 正交：列对齐 spec 让 statistics 不空（数据自洽）；本 spec 让 data-analyst 即便遇到空也结构性地必调 seal。
3. **多场景 dogfood**：跑正常（statistics 非空）+ 空 statistics + n<3 三种，确认 data-analyst 三种都一次 seal 成功（partial/completed/failed 均可，关键是**发出 tool_call**）。chart-maker / report-writer 各跑一次确认 seal 紧随产出。
4. **裸导入回归**：prompt 是模块级字符串，改后 `python -c "from deerflow.agents import make_lead_agent"` + `import app.gateway` exit 0（防手滑破坏模块）。

---

## 5. 不做 / 范围外（守纪律）

- **不改 code-executor**（已合一、今天多次 dogfood 无问题，作正面范例）。
- **不加任何 harness 兜底**：不把 data-analyst 加进 `_AUTO_SEALABLE`、不改 seal-resume、不改 max_turns 计数、不加 tool_choice 强制。**用户明确否决兜底方向**——兜底是在脆弱结构上叠代码，本 spec 修结构本身。
- **不改 schema / 不改 fast-fail 判定逻辑 / 不改参数审计判据**：只改"分析在哪成文"，不改"分析判什么"。
- **不碰 statistics 列对齐**（另一 spec，用户在修，正交）。
- **不引入 thinking 开关改动**（memory 实证 thinking ON/OFF 非根因）。

---

## 6. 改动文件清单

```
subagents/builtins/data_analyst.py    （workflow step 2.7/2.8/3 + fast_fail 合一；删 7+ 处 seal 哀求补丁）
subagents/builtins/chart_maker.py     （step 11 seal 紧贴 step 10 绘图）
subagents/builtins/report_writer.py   （step 4 seal 紧贴 step 3 write_file）
```

无测试文件改动（纯 prompt；验收靠 dogfood 行为 + 结构 grep + 裸导入）。

---

## 7. 一句话总结

seal 反复漏调的真根因不是任何一个诱因，是 prompt 把"产出分析"（思考成文）和"交付分析"（调 seal 誊抄）拆成两步、而 seal 参数恰是产出内容 → 模型产出完就认为完成、跳过冗余的交付步。治本是让"调 seal = 产出分析的唯一动作"（code-executor 的 run_metric_plan 已证明此模式不漏），对 data-analyst 合并思考与 seal、对 chart/report 让 seal 紧贴文件产出。纯 prompt 重构，零兜底代码。
