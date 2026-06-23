# 调研任务：EPM dogfood thread `3a41e483` 三问题根因深挖

> **交接对象**：接手调研的 AI Agent。
> **来源**：用户在生产 ECS 跑的 EPM dogfood（thread `3a41e483-aea1-4699-987f-0be2b5fb487f`，display_name「Experimental paradigm identification from trial data」，28 只 × 4 组 × 7）。
> **数据来源**：① 从 ECS dump 的完整对话 JSON（`/home/wangqiuyang/3a41e483-aea1-4699-987f-0be2b5fb487f.json`，59 条消息）；② 用户保存的前端输出全文（`docs/e2e/Experimental paradigm(1).txt`，1792 行，含完整 thinking + 命令）；③ 本地仓库代码。
> **前置 SOP**：从 ECS 拉数据走 [docs/sop/prod-thread-triage-sop.md](../sop/prod-thread-triage-sop.md)。
> **本会话已确认的事**：EV19 skill（`ethovision-paradigm-knowledge`）**已实施**——CLAUDE.md 第 10 条「设计已完成、待实施」的措辞过时。代码层验证完毕（见下 D）。

---

## 调研目标

本 thread 端到端跑通了（图出、报告出），但暴露 **4 个独立问题**。前 3 个（A/C/D）是用户点的调查焦点；第 4 个（B）是顺带发现但最值得修的真 bug。每个都要**确认根因 → 判断该不该立 spec → 给出修复方向**。

---

## A. report-writer「漏调 seal」— 疑似 memory 第 4 类根因复发（thinking 过载撞 turn 超时）

### 现象
- [49] / 前端 1736 行：`Error: Subagent 'report-writer' terminated without emitting 'handoff_report_writer.json'`，lead 自动重试一次才成功（[51]）。

### 本会话已查清的根因（从前端 1359-1735 行 report-writer 完整 thinking）
**不是「忘了调 seal」，是 thinking 过载撑爆 turn、turn 没正常结束 → SealGate 永不触发**。报告写作 subagent 的 thinking 轨迹（前端 1359-1735 行）显示它在**反复读 133K 字符的 `plan_metrics.json` 找 3 个指标的 `display_name_zh`**：

```
前端 1378: 读到 open_arm_time_ratio: 开放臂时间比例 ✅
前端 1380: 读到 open_arm_time: 开放臂时间 ✅
前端 1382-1383: open_arm_entry_count / open_arm_entry_ratio → "I need to find the display name"
前端 1500-1730: 反复估算"这 metric 在文件第几行"（"each entry is probably ~950 chars"、"140 entries"、"let me read line 900"、"133K chars too large to read at once"）...
前端 1736: 报错 terminated without emitting handoff（=turn 超时/撑爆）
```

**关键矛盾**：5 个指标的 display_name **第一次跑就已经在 `metrics_summary` 里全拿到了**（前端 1562-1566 行它自己列出来了），它完全不需要再去啃 133K 的原始 plan。这是 prompt/教学缺陷——没告诉它「display_name 看 metrics_summary 就够，别读原始 plan_metrics.json」。

### 对应 memory
- `feedback_seal_fourth_root_cause_thinking_overload_turn_timeout`：第 4 类 seal 根因 = thinking 当产出撞 turn 内超时。**那条 spec 只治理了 data-analyst，report-writer 没被覆盖。这是同根因复发，对象换成 report-writer。**

### 待调研（agent 要确认）
1. 读 report-writer 的 prompt（`packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py` 或对应 skill），**确认 prompt 里是否有诱导读 `plan_metrics.json` 拿 display_name 的指令**。
2. 确认 report-writer 的 turn 上限 / 思考预算配置，与 data-analyst 对比。
3. 确认 SealGateMiddleware 对 report-writer 是否生效（memory 说 SealGate 是结构性的，但 turn 内超时它拦不住——核对是否真拦不住还是没配）。
4. **修复方向**：把 `2026-06-18-data-analyst-thinking-overload-spec` 的治理模式（prompt 铁律「不含遍历 N 条 / 不读大文件」+ 明确「display_name 在 metrics_summary 取」）扩展到 report-writer。

---

## B.（顺带，最该修）chart 层 `--parameters-json` 注入过宽致 `plot_open_arm_time_ratio_bar` 全失败

### 现象（前端 1035-1098 行，dump JSON 看不到，因为 tool_result 被截断）
chart-maker 第一次并行跑 10 张图：
- `plot_box_open_arm`（aggregate）✅ 成功
- `plot_trajectory_s0-s3`（per_subject）✅ 4 张成功
- `plot_open_arm_time_ratio_bar_s0-s4`（per_subject）❌ **5 张全失败**：`TypeError: compute_open_arm_time_ratio() got an unexpected keyword argument 'closed_arm_zones'`

chart-maker 自己诊断出 `--parameters-json` 里塞了 `{"closed_arm_zones": ["closed"], "open_arm_zones": ["open"]}`，但底层 `compute_open_arm_time_ratio()` **只收 `open_arm_zones`**。它手动改成只带 `open_arm_zones` 重跑 5 次才成功（前端 1098-1178）。

### 根因（已定位）
`resolve_charts` → `_build_zone_aliases_overrides`（[resolve.py:644-739](packages/agent/backend/packages/ethoinsight/ethoinsight/catalog/resolve.py#L644-L739)）把**所有** zone param 一次性注入每个 chart 的 `--parameters-json`，不区分该 chart 实际用到的 metric 需要哪些 zone 参数。`box_open_arm` 需要 open+closed 两个，但 `open_arm_time_ratio_bar` 只需要 open——后者被强塞了 `closed_arm_zones`，透传到底层 compute 函数就 TypeError。

### 待调研
1. plot 脚本层是否**应该容忍多余 kwargs**（`compute_open_arm_time_ratio(**kwargs)` 加 `**_` 吞掉）——这是最便宜的兜底，但治标。
2. **治本**：`_chart_to_plan` 注入 `--parameters-json` 时，按该 chart 关联的 metric 的 `parameters_used` 裁剪，只注入该 metric 真正用到的 zone 参数。需要看 catalog chart entry 与 metric 的关联关系。
3. 确认这是否是 **chart 层第二条独立根因**（与上次 `d58adb40` thread 的 chart-maker turn 耗尽不同根因——见 `docs/handoffs/2026-06/2026-06-22-ecs-dogfood-chart-maker-turn-exhaustion-per-subject-path-handoff.md`）。两者都该进同一个 chart spec。
4. **守工程纪律**：动 `_build_zone_aliases_overrides` / `_chart_to_plan` 前先 grep 同事的 review-packages（memory `feedback_ssot_lives_in_review_packages`），范式/图表 SSOT 在那边。

---

## C. 「5 个指标 vs 2 个指标」叙述不一致 — code-executor handoff 摘要文本 bug

### 现象
- 前端 232 行（code-executor handoff 摘要）：`总任务数: 140（2 个指标 × 28 个受试者 × 2 次计算 + 统计检验）` —— **写「2 个指标」**
- 实际 plan 是 **5 个指标**：`open_arm_time_ratio / open_arm_time / open_arm_entry_count / open_arm_entry_ratio / total_entry_count`（前端 399 行 data-analyst、前端 1393 行 report-writer 都复述 5 个）
- 140 = 28 × 5，对得上「5」。code-executor 的「2 × 28 × 2」叙述**算错嘴瓢**，但**实际算的是 5 个**（per_subject 里 28×5=140 个值齐全）。

### 根因
- code-executor 生成摘要时**硬编码了「2 个指标」表述**（可能来自模板），没从 plan 实际 `metric_count` 取。
- lead 在 [23] / 前端 359 行**自检发现不一致**（「plan 有 5 个但 code-executor 汇报提到 2 个」），读 handoff 确认后纠正成「5/5」（[25] / 前端 364）。**lead 行为正确，无下游影响**。

### 待调研
1. 找 code-executor 生成 handoff 摘要的代码（`subagents/builtins/code_executor.py` 或 seal 工具），定位「2 个指标」这种硬编码模板从哪来。
2. 改为从 `plan_metrics.json` 的实际 metric_count 动态生成。
3. **小修，不单独立 spec**，并入下次 dogfood 回归修复批次即可。

---

## D. 范式识别绕圈 — `identify_ev19_template` 列结构信号未回流到范式推断

### 现象
- [04] / 前端 109-134：`identify_ev19_template` 返回 `status="unknown"`（evidence 里如实标了 `has_suspect_zone_columns: true, suspect_columns: ["open","closed"]`）。
- lead 自己从 evidence 里的 `open/closed` 推断「→ EPM」反问用户（[05]），用户确认。
- [08] 用户确认 EPM 后再跑一次 `identify_ev19_template`，这次返回 `ambiguous`（候选 PlusMaze-AllZones / PlusMaze-FewZones），用户选 B。

### 本会话已查清的根因（纠正前两轮错判）
**EV19 skill 已实施**（`ethovision-paradigm-knowledge` skill 在、`identify_ev19_template_tool.py` 在、`ev19_template_provider.py` guardrail 在）。`_filter_candidates_by_zone`（[identify_ev19_template_tool.py:215-218](packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py#L215-L218)）**已正确处理 `has_suspect`**（有疑似归属列时保留划区变体、剔除 NoZones）。

**真根因 = `paradigm_key` 提取纯靠字符串关键词，列结构完全没参与范式推断**：

1. Step 2（[identify_ev19_template_tool.py:147-160](packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py#L147-L160) `_extract_paradigm_from_hints`）只查 `_PARADIGM_CN_HINTS`（中文词）+ `_FILENAME_PARADIGM_PATTERNS`（文件名 regex 如 `epm|plus.?maze`）。
2. 本 thread：文件名 `trial01.xlsx` 无范式词、用户消息「帮我分析这些数据」无范式词 → `paradigm_key=None` → `candidate_ids=[]` → 直接走 Step 9 的 `status="unknown"`（[identify_ev19_template_tool.py:585-591](packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py#L585-L591)）。
3. 数据里的 `open`/`closed` 列是 **EPM 的强信号**，但**只能进 `evidence.zone_info.suspect_columns` 做「展示」，不能驱动范式识别**（`_detect_zone_config` 的结果没回流到 `_extract_paradigm_from_hints`）。

### 待调研
1. 设计一个「列结构 → 范式候选」的反向映射：哪些列组合是某范式的强信号（如 `open`+`closed` → epm；`center`+`periphery` → open_field；`light`+`dark` → light_dark_box）。**SSOT 纪律**：先 grep review-packages / 问行为学同事，别自己拍列→范式映射（memory `feedback_no_cross_paradigm_reuse_accept_duplication` + `feedback_ssot_lives_in_review_packages`）。
2. 在 `_extract_paradigm_from_hints` 返回 None 时，加一个基于 `zone_info.suspect_columns` 的兜底范式候选生成（返回 `ambiguous` 列出候选，而不是 `unknown`）。
3. **重要判别**：`open`+`closed` 同时出现在 EPM 和 Zero Maze——不能只凭这两列就唯一判 EPM，必须返回 `ambiguous` 让用户选范式。这正是当前工具「宁 unknown 不猜」的谨慎所在，谨慎是对的，但**应该从 unknown 升级到 ambiguous（带列依据）**，减少一轮交互。
4. 确认 `_PARADIGM_CN_HINTS` 是否该补「plus maze」「十字迷宫」（当前只有「高架十字」「十字高架」）。

---

## 优先级建议

| 问题 | 优先级 | 理由 |
|---|---|---|
| **B: parameters-json 注入过宽** | 🔴 P0 | 真 bug，每次都会触发，chart-maker 靠手改参数兜底；治本在 resolve_charts 按 metric 裁剪参数 |
| **A: report-writer thinking 过载** | 🔴 P0 | memory 第 4 类根因复发，结构性，立 spec 把 data-analyst 的治理套到 report-writer |
| **D: 范式识别列信号未回流** | 🟡 P1 | 体验问题（多一轮交互），非阻塞；修前必须先确认列→范式映射的 SSOT |
| **C: handoff 摘要 5 vs 2** | 🟢 P2 | 小修，无下游影响，并入回归批次 |

---

## 给接手 agent 的起步指令

1. 读本文件 + `docs/sop/prod-thread-triage-sop.md`（拉数据链路）。
2. 本地就能查 A/B/C/D 的代码侧（不需要 ECS）：
   - A：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py` + report-writer skill
   - B：`packages/agent/backend/packages/ethoinsight/ethoinsight/catalog/resolve.py` 的 `_build_zone_aliases_overrides` + `_chart_to_plan`
   - C：code-executor seal 工具 / handoff 摘要生成
   - D：`packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py` 的 `_extract_paradigm_from_hints`
3. 需要核对前端完整轨迹时读 `docs/e2e/Experimental paradigm(1).txt`（比 dump JSON 全，含 thinking + 完整命令）。
4. **守纪律**：动 catalog / SSOT 前先 grep review-packages；改 harness 核心后裸导入验证（memory 有大量教训）。

## 相关文件索引
- dump 对话：`/home/wangqiuyang/3a41e483-aea1-4699-987f-0be2b5fb487f.json`
- 前端全文：[docs/e2e/Experimental paradigm(1).txt](e2e/Experimental%20paradigm(1).txt)
- 取证 SOP：[docs/sop/prod-thread-triage-sop.md](../sop/prod-thread-triage-sop.md)
- 上次 chart dogfood handoff（B 的姊妹问题）：[docs/handoffs/2026-06/2026-06-22-ecs-dogfood-chart-maker-turn-exhaustion-per-subject-path-handoff.md](../handoffs/2026-06/2026-06-22-ecs-dogfood-chart-maker-turn-exhaustion-per-subject-path-handoff.md)
- memory 第 4 类 seal 根因（A 的依据）：`~/.claude/projects/-home-wangqiuyang/memory/feedback_seal_fourth_root_cause_thinking_overload_turn_timeout.md`
