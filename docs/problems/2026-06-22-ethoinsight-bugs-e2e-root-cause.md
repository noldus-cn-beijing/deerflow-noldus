# ETHOINSIGHT E2E Bug 清单 — 根因调查与修复对照

> **来源**：用户在 `dev.ethoinsight.com` 跑 2 轮 EPM E2E（28 个 xlsx，数据飞轮模式，FewZones）发现的 10 个 bug，原始清单 `~/ETHOINSIGHT_BUGS.md`（2026-06-22）。
> **本文档**：对每条做**代码级根因取证**，对照当前 dev 代码（`45824364`，含 PR#168~#172）。
> **判断标准**：「这个现象在当前代码里还会不会发生」，**不是**「有没有相关 PR」。
> **重要免责**：以下「已消除/仍存在」均为**代码取证**结论，非站点实证。最终确认需 `make deploy-tar` 部署 #168~#172 后重跑那 2 轮 E2E。站点测试时跑的镜像早于这批 PR 部署。

---

## 分档总览

| 档 | 编号 | 一句话 |
|---|---|---|
| ✅ 已消除 | ETHO-6 | chart 参数裁剪（#168 确定性裁剪） |
| ✅ 主流程消除（残留缺口） | ETHO-3 | report-writer 侧消除；code-executor 分诊场景仍可能截断 |
| ◐ 设计如此非 bug | ETHO-4 | identify 故意不用 experiment 字段，守 HITL 反问铁律 |
| 🔴 高·仍存在 | ETHO-1, ETHO-2, ETHO-7 | seal 漏调降级未根治 / 派非法 bash / 决策点非确定性（病根） |
| 🟡 中·仍存在 | ETHO-5, ETHO-8 | 分组单文件外推 / A/B 顺序不稳定 |
| 🟢 低·仍存在 | ETHO-9, ETHO-10 | 反问缺 options / 并行 bash 被 loop-detection 中断 |

**需解决 = 7 条**（ETHO-1/2/5/7/8/9/10）+ ETHO-3 残留缺口。**不需解决 = 3 条**（ETHO-6、ETHO-3 主流程、ETHO-4）。

---

## ✅ 已消除

### ETHO-6（中）chart 脚本参数不兼容 → 出图失败 — **已坐实消除**
- **现象**：`open_arm_time_ratio_bar` 的 `parameters-json` 含 compute 函数不接受的字段 → 图失败，移除多余参数后才成功。
- **根因**：`resolve_charts` 原本把全范式 zone 参数**并集**无条件注入每张图的 `--parameters-json`；EPM 的 `compute_open_arm_time_ratio` 只接受 `open_arm_zones`，多余的 `closed_arm_zones` 在 `compute_fn(df, **params)` splat 时触发 **TypeError**。
- **为何消除**：#168 在 `resolve.py:465/497` 新增 `_filter_zone_overrides_for_chart`，按 `chart.requires_columns` 声明的 zone 概念裁剪（narrow pattern `in_zone_open_arm*` 精确匹配后只留 `open_arm_zones`；宽通配如 OFT `in_zone*` 不裁剪避免回归）。确定性裁剪，不靠 LLM。
- **证据**：`packages/ethoinsight/ethoinsight/catalog/resolve.py:465,497,747-798`、`packages/ethoinsight/ethoinsight/scripts/epm/plot_open_arm_time_ratio_bar.py:34-37`；测试 `test_chart_zone_overrides_filtered.py`（189 行）。
- **判定**：✅ 当前代码不会再发生。

---

## ✅ 主流程消除（有残留缺口）

### ETHO-3（高）读 plan_metrics.json（~133K）反复截断 — **report-writer 侧消除，code-executor 分诊场景仍可能复现**
- **现象**：code-executor / report-writer 读 `plan_metrics.json` 反复被截断、来回重读、陷入「读不到结尾」循环。
- **根因**：`plan_metrics.json` 按 subject 重复（28×5=140 条 ≈133K），触发 read_file 的 **50K 字符截断**（`sandbox/tools.py:1738`），啃不到尾部统计段。
- **report-writer 侧已消除**：#169 生成 `_metric_metadata.json`（去重 ~5 条），report-writer prompt 明确指向它并禁读 plan_metrics。
  - 证据：`subagents/builtins/report_writer.py:178-208`（L190-191「不读 plan_metrics.json，它是施工文件」）、生成处 `tools/builtins/prep_metric_plan_tool.py:328-341`。
- **⚠️ 残留缺口（code-executor 侧）**：happy path 不截断（走 `run_metric_plan` 工具内部读、不透传 LLM）；但**分诊失败场景** `code_executor.py:59`「用 read_file / ls 查证失败细节」若读 133K plan_metrics，**仍触发截断**。metadata sidecar 没覆盖这一侧。
- **判定**：◐ 正常流程已消除；分诊失败场景仍可能复现（需补：分诊也走 metadata，或给 read_file 加 offset/grep 在分诊侧使用）。

---

## ◐ 设计如此，不是 bug

### ETHO-4（中）identify_ev19_template 无法自动识别 EPM — **设计裁决：不是 bug**
- **现象**：即便 EV19 头 `experiment="EPM-Xuhui"` 且含 open/closed 列，仍返回 `status="unknown"`，被迫人工确认。
- **根因坐实**：`_extract_paradigm_from_hints`（`identify_ev19_template_tool.py:214-230`）**只扫用户消息 + 文件名**，完全不读 EV19 头的 `experiment` 字段。`paradigm_key=None` → 候选空 → `status="unknown"`（`:567-583`）。
- **为何是设计而非 bug**：
  1. `experiment` 字段自由文本无标准化（"EPM-Xuhui"/"强迫游泳"/"Elevated Plus Maze" 等变体），可靠性弱；
  2. 命中 HITL 铁律 `prompt.py:476`「范式推断失败必须 ask_clarification、不许默认猜测」；
  3. #170 spec 明写「工具仍返回 unknown，守 L480」（上一轮已撤回「工具该造 ambiguous 候选」的旧版，确认范式识别本就必须反问）。
- **用户正确用法**：上传时消息说「这是 EPM」或文件名含 epm/高架/plus-maze → 命中 hint → `status=ok`。
- **判定**：❌ 不当 bug 修（改成自动判定会违反反问铁律）。**需回给用户对齐这是预期行为。**

---

## 🔴 高优先级·仍存在

### ETHO-1（高）report-writer 偶发漏调 seal — **降级未根治（曾误判为已根治）**
- **现象**：report-writer 完成推理后未调 `seal_report_writer_handoff`，报 `terminated without emitting ... forgot to call the seal tool`，Lead 自动重试一整轮才成功（间歇性）。
- **根因坐实**：ReAct loop 的**终止时机由 LLM 决定，框架无法用 prompt 100% 约束**。三道防线只降低触发率：
  1. SealGateMiddleware（`seal_gate_middleware.py`）after_model 提醒，但提醒上限 `_MAX_REMINDERS=2` 后放行模型退出；
  2. seal-resume（`executor.py:1407`）补一轮，LLM 仍可能只输出文本不调 seal；
  3. auto-seal（`executor.py:1426` → `_attempt_auto_seal_from_artifacts:301`）**前提是 report.md 已落盘**。
- **为何仍复现**：`_validate_handoff_emitted`（`executor.py:1400`）**只查 workspace 的 handoff JSON、不查 outputs/report.md**。auto-seal 兜得住「有 report.md 没 handoff」，但兜不住「连 report.md 都没写盘」——那条 fall through 到 FAILED，靠 lead 重试（`prompt.py:258/272` 匹配 "terminated without emitting"）。
- **真实状态**：从「致命无限重派」降级成「偶发重试一轮（~2-4min）」。清单「结果可复现 + lead 重试有效兜底」印证。
- **判定**：🔴 根因（LLM 决定何时停）架构上无法 prompt 消除，只能继续降低触发率（如 auto-seal 也认 outputs/report.md 缺失时从 AIMessage 内容重建？需评估）。**别再报"已根治"。**

### ETHO-2（高）Lead 派遣非法子代理 `bash` — **完全没修，改动小**
- **现象**：Lead 尝试 `subagent_type='bash'` 批量扫描分组 → 被拒（`Input should be 'chart-maker', ...`）→ 退化成 28 次逐个 inspect。
- **根因坐实**：两个缺口叠加——
  1. lead prompt（`prompt.py:217`）「bash 仅在以上 subagent 都不适合时用」是**软提示非禁令**，没有「分析流程只许派这 5 个、批量扫描用 inspect 工具」的硬约束；
  2. **系统没有批量扫描多文件的工具**，只有逐文件 `inspect_uploaded_file_tool`。lead 想批量扫只能自创 `bash` → 被 `_SubagentLiteral` 枚举校验拒（`task_tool.py:334-350,403`）。
- **判定**：🔴 仍存在。修法：① prompt 加明确禁令；②（可选）加批量扫描工具替代 bash 路径。**性价比高。**

### ETHO-7（高）决策点数量/形态运行间不一致 — **病根·无硬约束**
- **现象**：同输入，Run1 出 4 个决策点，Run2 只 2 个且跳过「是否出图」。
- **根因坐实**：决策点数量/合并 **100% 由 lead LLM 自由裁量，零确定性约束**。
  - `prompt.py:483`「反问合并规则」是**建议不是强制**（无 middleware/guardrail 执行）；
  - `prompt.py:307` intent 分类（E2E_FULL vs E2E_FULL_ASKVIZ）凭 LLM 对「模糊程度」的理解，每轮不同 → 合并/不合并、跳不跳「是否出图」随机。
  - 现有 `IntentPostStepAskGateProvider`（`:114/145`）仅拦截 task() 派遣，**不约束反问本身的决策点和顺序**。
- **判定**：🔴 仍存在。**是 ETHO-5/8/9 的共同病根。**

---

## 🟡 中优先级·仍存在

### ETHO-5（中）分组检测单文件外推 — **prompt 没强制全扫**
- **现象**：Run2 lead 只 inspect 单文件就断言「Group 均为 XX」，后才逐个扫出 4 组。
- **根因坐实**：`identify_ev19_template` **本就一次性扫全部文件**返回 `per_file_grouping`（`identify_ev19_template_tool.py:554-566`），但 lead prompt（`prompt.py:485`）只说「**优先用**」不是「**必须用**」；单文件 `inspect_uploaded_file_tool` 给了偷懒入口（看一个文件就外推）。对比范式有「禁止猜」硬约束（`prompt.py:476`），分组**没有「禁止单文件外推」对应约束**。
- **判定**：🟡 仍存在。修法：prompt.py:485 段改「必须用 per_file_grouping + 禁止单文件外推全局结论」。改动小。

### ETHO-8（中）多选 A/B 顺序不稳定 — **候选列表无排序**
- **现象**：Run1 A=AllZones/B=FewZones，Run2 对调。
- **根因坐实**：`identify_ev19_template_tool.py:600` 处 `target_ids = filtered_ids if filtered_ids else candidate_ids` 后**直接 `for tid in target_ids` 遍历构造候选，无 `sort()`**。顺序来自 `EV19_TEMPLATE_PARADIGM_MAP`（`ethoinsight.ev19_facts`）的 dict 值列表迭代序；A/B 标签按候选索引分配 → 候选序变则 A/B 对调。
- **判定**：🟡 仍存在。修法：候选列表加确定性排序（按 template_id 字典序，或推荐项恒为 A）。**改动极小，纯收益。**

---

## 🟢 低优先级·仍存在

### ETHO-9（低）部分决策点缺快捷按钮 — **options 参数 LLM 自由传**
- **现象**：Run2「是否出报告」只给文本框、无 A/B 按钮（Run1 有）。
- **根因坐实**：`ask_clarification` 的 `options` 是**可选参数**（`clarification_tool.py:17`），传不传由 lead LLM 决定。`prompt.py:333` 只在「是否出图」样板硬编码了 options，其他反问点（尤其合并反问）无强制。
- **判定**：🟢 仍存在（ETHO-7 病根的症状）。

### ETHO-10（低）chart-maker 并行 bash 被中断未返回 — **loop-detection 误伤**
- **现象**：Run1 chart-maker 并行出图时「bash 调用被中断，工具调用没返回结果」，需事后遍历 outputs 补救。
- **根因方向**（低优先级未穷尽）：chart-maker 并行出图后逐个检查/重试失败 png 时，bash 高频调用触发 `loop_detection_middleware.py` 频率熔断（`_DEFAULT_TOOL_FREQ_WARN=3`）→ partial strip 掉 bash 的 tool_call（`:628-631`）→ ToolMessage 永不产生 → 「被中断未返回」。
- **同类病**：memory `feedback_loop_detection_tool_semantics_floor_and_partial_strip`（loop-detection 误伤高频记账工具）。
- **判定**：🟢 仍可能（取决于是否真进高频重试循环）。修法：chart-maker 检查也压进一个 bash call，或给 chart-maker 调高频率阈值 override。

---

## ETHO-7/8/9 共同病根

子代理一致确认：**交互流程决策完全由 lead prompt 的自由形 LLM 推理驱动，无确定性约束机制。**

| 环节 | 现状 | 缺陷 |
|---|---|---|
| 反问合并（ETHO-7） | prompt 建议 `prompt.py:483` | 非强制，无 middleware guard |
| 候选排序（ETHO-8） | 无 sort `identify_ev19_template_tool.py:600` | dict 迭代序不确定 |
| options（ETHO-9） | 可选参数 `clarification_tool.py:17` | LLM 自由传 |

**统一修复方向**：用已有的 `path_registry.PATHS` + guardrail provider 模式，把「反问顺序/合并/选项格式」从 prompt 自由裁量升级成结构化可执行约束（类比 intent 驱动的路径编排）。

---

## 修复优先级建议（按性价比）

| 序 | 编号 | 改动量 | 类型 |
|---|---|---|---|
| 1 | ETHO-8 | 极小（候选加 sort） | 确定性，纯收益 |
| 2 | ETHO-2 | 小（prompt 禁令 + 可选批量工具） | 🔴 高 |
| 3 | ETHO-5 | 小（prompt 全扫约束） | 中 |
| 4 | ETHO-7/9 | 大（guardrail 设计） | 🔴 病根 |
| 5 | ETHO-1 | 中（评估 auto-seal 兜底扩展） | 🔴 只能降级 |
| 6 | ETHO-3 残留 | 小（分诊侧走 metadata） | 缺口收尾 |
| 7 | ETHO-10 | 小（chart-maker 检查合并/阈值） | 低 |

**需对齐用户**：ETHO-4 是预期行为（范式识别故意反问）；ETHO-1 是降级非根治（解释清楚）。

---

*调查方法：4 个并行 Explore 子代理按代码区域分组取证，主代理抽查关键 file:line 防幻觉。教训见 memory `feedback_code_has_fix_not_equal_bug_eliminated_seal_react_floor`。*
