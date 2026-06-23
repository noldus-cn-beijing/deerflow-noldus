# 交接：data-analyst seal handoff 太大撞 max_tokens —— 真根因已挖到底，待选 provider-agnostic 修法

> **会话日期**：2026-06-23（接 forensics 取证之后的深挖会话）
> **一句话**：data-analyst 的 seal handoff **结构化输出太冗杂/太大**，和模型 reasoning 共享单次响应的 `max_tokens=4096` 输出预算 → seal args 被挤穿、腰斩成残缺 JSON → invalid_tool_calls → handoff 永不落盘 → FAILED → lead 降级跳过专业判读（判读层功能性失效）。
> **要找的修法（用户定调）**：**既保持 handoff 结构化输出，又不让单次输出预算爆**。且必须 **provider-agnostic**（不绑 dashscope——将来可能自部署 LLM / 换别的 provider）。
> **为什么交接**：上下文快满。根因已挖到底、方向已收敛到 3 个候选，待实施。

---

## 〇、给接手 agent 的一句话

**别再质疑根因、别再走弯路**——本会话已把根因从「收尾漏 call」→「args 大」→**「reasoning_tokens 占满单次输出预算挤穿 args」**逐层挖到底，有 forensics dump 的 `response_metadata` 铁证。直接进**修法选型 + 实施**。

**两条已定的事实 + 一条用户指令：**
1. **真根因（铁证）**：seal 的 tool_call arguments 和 reasoning 共享单次响应 `max_tokens`。data-analyst 的 flash **默认就产 reasoning**（3000-4096 token/turn），把 4096 吃到只剩 ~770 给 args → args 写到 1179/2162 char 撞穿总闸腰斩。**handoff 体积 = 单次输出体积，被 seal 的 tool-args 设计绑死**——这是 provider-agnostic 的结构问题。
2. **用户裁决（明确指令）**：**max_tokens 可以增大，向下兼容，大了无所谓、小了出问题**。→ 提 max_tokens 作为第一道防线**直接做，无副作用**（4096 → 至少 12288，留足 reasoning 峰值 4096 + args + content）。
3. **但提 max_tokens 不是结构保证**（reasoning 是 LLM 自由裁量，#15 reasoning 单独吃满 4096；更大数据集/更复杂判读仍可能撞更高墙）→ **必须配一个结构修法**让 handoff 不依赖「单次响应装得下」。

---

## 一、完整根因追查链（本会话挖的，逐层纠正）

| 阶段 | 当时的判断 | 被什么推翻 |
|---|---|---|
| 最初（我接 handoff） | ETHO-1 prod trace = data-analyst「收尾漏 call」，SealGate 能拦 | forensics agent 独立复现 data-analyst 内部 turn：是 **args 截断成 invalid_tool_calls**，不是没调 seal。我只 dump 了 lead thread 看不到 subagent 内部。 |
| forensics spec | seal args **本身太大**（3000+ token）撞 4096 | 量 dump 实测：args 只有 **1179/2162 char（~400-750 token）**，根本不大 |
| 我（误判1） | flash 无 thinking 通道，判读写进 content 占预算 | config 有 `PatchedReasoningChatOpenAI`，我推断「该开 enable_thinking 根治」 |
| 我（误判2，被用户当场推翻） | 开 enable_thinking 让 reasoning 走独立通道根治 | `data_analyst.py:312-323` 注释：关 thinking 是 **06-18 故意的修复**（v4-pro thinking 撞预算），重开 = 重蹈覆辙 |
| **最终（铁证）** | **reasoning_tokens 计入 completion_tokens，与 args 共享 max_tokens** | forensics dump 的 `response_metadata`：`completion=4097, reasoning=3325`（#9）/ `reasoning=4096`（#15）。reasoning_content 字段非空（9K-14K 字符）。**flash 默认就 reasoning，`supports_thinking:false` 管不到它**（那开关只控制是否主动发 enable_thinking 参数） |

**铁证数据**（`/tmp/pw-driver/repro/full/data-analyst-repro000-full.json` 的 response_metadata）：
```
#9:  completion=4097, reasoning=3325, reasoning_content=12089字符  INVALID_SEAL(args@1179char断)
#13: completion=4097, reasoning=2727, reasoning_content=9474字符   INVALID_SEAL(args@2162char断)
#15: completion=4097, reasoning=4096                               ← reasoning几乎吃光全部4096
```

**06-18「换 flash 消除 thinking 过载」是假修复**：注释以为「换 flash 后 4096 全留给 args」，实际 flash 照样 reasoning、照样抢预算，根因从 v4-pro 延续到 flash，当时没看 reasoning_tokens 字段。

---

## 二、为什么是 provider-agnostic 问题（用户的关键提醒）

用户明确：**「不要在意 dashscope，这只是现在用的一个 LLM API，将来可能自部署 / 换别的」**。这淘汰了一整类 provider-specific 方案：

- ❌ 关 dashscope 的 reasoning（`enable_thinking:false`）—— 换 vLLM/Qwen 是 `chat_template_kwargs`，换 OpenAI o 系列根本关不掉，provider-specific。
- ❌ 给 reasoning 单独预算 —— 多数 provider 不暴露这旋钮。
- ❌ 只靠提 max_tokens 到某个数 —— 不同模型 reasoning 体量天差地别（o1 几万、flash 几千），一个数字适配不了所有。

**经得起「换任何 LLM」的只有一类**：不管模型怎么生成、reasoning 多长、args 在哪截断，**harness 都能确定性拿到完整 handoff**。根子在于——我们要求模型在**一次响应**里把完整结构化 handoff 作为 tool_call arguments 吐出，**handoff 体积 = 单次输出体积**，绑死了。任何 LLM 只要 reasoning+args 超过单次输出上限就腰斩。

---

## 三、seal 工具现状（问题的结构根源，已核实）

`tools/builtins/seal_handoff_tools.py:663` `seal_data_analyst_handoff`：
```python
def seal_data_analyst_handoff(
    status, key_findings, outlier_findings, excluded_metrics,
    method_warnings, recommendations, errors, gate_signals,
    quality_warnings, parameter_audit_findings, ...):
    # 整个 handoff 所有字段 = 模型必须在单次响应里吐的 tool_call arguments
    payload = {...}  # 组装后 _seal_handoff → 写 handoff_data_analyst.json
```
docstring 含禁令：**「严禁直接用 write_file 写 handoff_data_analyst.json，必须走本 tool」**（memory `feedback_code_executor_skill_writefile_contradicts_seal_tool` 踩过的坑——所以方案 C「让它 write_file」要解决和这条禁令的矛盾）。

**架构不一致**（关键洞察）：code-executor 的 140 个指标产物是写进 `handoff_code_executor.json` **文件**的，不是当 tool args 传；subagent 间本就走 workspace 文件传 hard fact（memory `feedback_subagent_consumption_via_first_party_tool`）。**唯独 data-analyst 的 seal 把整个判读塞进 args，是异类。**

---

## 四、三个候选修法（都 provider-agnostic，待选）

| 方案 | 做法 | 改动 | 评价 |
|---|---|---|---|
| **A. 确定性抢救** | executor 检测截断的 seal JSON（invalid_tool_calls + Unterminated）→ 补全未闭合括号 `]}` → 容错 parse 落盘 | 改 executor 核心，中 | 兜底层。救「已经截断的」。reasoning/数据规模无关。但救回的可能丢截断处半条 finding |
| **B. 分步/增量 seal** | handoff 拆多次小 tool_call（先 seal 骨架 status+key_findings，再 append outlier/warnings/recommendations） | 改 seal schema+工具+下游消费方，大 | 真结构根治、无预算假设。但改契约面大，跨 report-writer 等下游 |
| **C. handoff 走文件（推荐方向）** | data-analyst 把判读 `write_file` 到 workspace，seal 工具只接收**文件路径**（或读它刚写的文件）→ args 永远几十 token | 改 seal 工具 + data-analyst prompt + **解除 write_file 禁令矛盾**，中 | **直接砍断「体积=单次输出」绑定**，args 永远小，reasoning 多长/数据多大/换什么 LLM 都不腰斩。**且让 data-analyst 回归既有 workspace 文件传递模式，消除异类** |

**我（上一 agent）的判断**：**C 是最根本的 provider-agnostic 解**（砍断 handoff 体积 = 单次输出体积的绑定 + 架构归一），A 是更轻的兜底（可与 C 叠加），B 改契约面太大。**但用户尚未在 A/B/C 间最终拍板**——接手 agent 应把这三个连同「提 max_tokens 第一道防线」一起呈现给用户定夺，或按用户已倾向的方向实施。

**C 的实施要点（若选 C）**：
- data-analyst prompt 改为「把判读写进 workspace 文件（如 `da_findings.json`），再调 seal 传该路径」。
- seal 工具加一个「从文件读 payload」的路径，**保留**「禁止直接 write_file 写最终 `handoff_data_analyst.json`」禁令——区别在于：中间产物文件（da_findings.json）允许 write_file，最终 handoff 仍只由 seal 工具落盘（seal 读中间文件 → 校验 → 写 handoff_data_analyst.json）。这样既解耦输出体积，又不破坏「handoff 唯一由 seal 落盘」的契约。
- 守 memory `feedback_subagent_system_prompt_higher_authority_than_skill`：改 data-analyst 行为要改 `data_analyst.py` 的 system_prompt（最高权威），不只 SKILL.md。

---

## 五、立即可做（用户已授权，无副作用）

**提 data-analyst max_tokens**（用户明确「大了无所谓、向下兼容」）：
- `config.yaml` 两处 `max_tokens: 4096`（L14 是 deepseek-v4-pro，L29 是 deepseek-v4-pro-summary = data-analyst 用的）。**改 L29（data-analyst 那个）到 ≥12288**（reasoning 峰值实测 4096 + args ~2000 + content + 余量）。L14 是否改看是否同源问题，按需。
- **dev/prod 对齐**：config.yaml 改动须同步 docker-compose.yaml 的 env/CLI（memory `feedback_dev_prod_behavior_alignment`）。
- 这是**第一道防线降触发率**，不是结构保证——必须配 A/B/C 之一才算 generic 根治。
- 验证：用 `/tmp/pw-driver/repro-data-analyst.py`（forensics agent 的复现脚本，建议入库 `scripts/forensics/`）改 max_tokens 重跑，断言 seal 不再 invalid、handoff 落盘、判读质量（混杂排查/效应量/离群三项）不退化。

---

## 六、本会话新增/更新的 memory（接手先读）

- `feedback_data_analyst_seal_truncation_real_cause_reasoning_tokens_share_max_tokens` — **真根因铁证**（reasoning 占 completion 预算挤穿 args）+ provider-agnostic 三层方案 + factory.py 为何没关 reasoning。**最重要,先读这条。**
- `feedback_data_analyst_reasoning_channel_off_is_real_root_cause_of_seal_truncation` — **已撤回的错误推断**（开 thinking 根治），保留作「凭代码片段推断没读全上下文」教训。
- `feedback_etho1_real_root_cause_is_seal_args_truncated_by_max_tokens_4096` — args 截断根因（对了一半，没找到 reasoning 是元凶）。
- `feedback_etho1_prod_trace_seal_miss_both_dataanalyst_and_reportwriter` — prod 4thread 实证（其中 data-analyst 部分已被深挖纠正）。

---

## 七、关键文件锚点（已核实）

- `config.yaml:21-32` — data-analyst 用的 `deepseek-v4-pro-summary`（`use: PatchedReasoningChatOpenAI`、`model: deepseek-v4-flash`、`max_tokens:4096`、`supports_thinking:false`）。L29 是要改的 max_tokens。
- `data_analyst.py:312-323` — model= 注释（06-18 换 flash 的"假修复"现场，记着别重开 thinking）。
- `tools/builtins/seal_handoff_tools.py:663` `seal_data_analyst_handoff` — 全字段当 args 的异类设计；`_seal_handoff_to_workspace`(L535)。
- `models/factory.py:135-150` — `if not thinking_enabled:` 关 reasoning 的分支，**前提 `has_thinking_settings`（config 有 when_thinking_enabled/thinking）**；data-analyst 两者皆无 → 从没发过"关 reasoning"信号 → flash 默认 reasoning 一直跑。
- `models/patched_reasoning.py` — 接 dashscope `reasoning_content` 字段进 `additional_kwargs`。
- forensics 实证产物：`/tmp/pw-driver/repro/`（ai_messages.json + full/data-analyst-repro000-full.json，含 response_metadata 铁证）+ `repro-data-analyst.py` 复现脚本。

---

## 八、本会话的全局上下文（这条 forensics 线之外的待办，别丢）

本会话还接了 ETHOINSIGHT_BUGS 10 条 E2E bug 的 spec 工作（与本 forensics 线**正交但同源于 seal/harness**）：

- **已 merge dev**：ETHO-1（SealGate after_agent，仅 report-writer/chart-maker 可重建产物兜底）、ETHO-3、ETHO-8。
- **7 份 spec 已落盘 dev**（commit `639821d0`）：ETHO-1/2/3/5/7/8/9。
- **剩余未实施 spec**：ETHO-2（prompt 批量扫描）、ETHO-5（prompt 分组全量）、ETHO-7（intent 确定化，分 A/B 两子改动）、ETHO-9（path_registry Step 加 requires_options）。
- **并行计划**（两个冲突点）：① `prompt.py` 被 ETHO-2/5/7-A 三抢（建议合一个 prompt PR）；② `IntentPostStepAskGateProvider` 被 ETHO-9/7-B 两抢（ETHO-9 先合、7-B 基于其上）。详见 `docs/handoffs/2026-06/2026-06-23-ethoinsight-bugs-7specs-prod-trace-harnessx-handoff.md`。
- **ETHO-1 的 data-analyst 部分** = 本 forensics 线（SealGate 对 data-analyst 无效，走本 handoff 的 A/B/C）。

**本 forensics 线与 ETHO-1 的关系**：ETHO-1 SealGate after_agent 只兜 report-writer/chart-maker（可重建产物）；**data-analyst 的 seal 腰斩是独立问题，走本 handoff**。两者治本点正交。

---

## milestone 建议
- 「subagent 派遣/生命周期 infra 加固」track：data-analyst seal handoff 解耦（A/B/C）+ 提 max_tokens，与 ETHO-1（SealGate）同属 seal 鲁棒性系列。
- **方法论 checkpoint**：这条根因追查（收尾漏call→args大→reasoning占预算→handoff体积绑死单次输出）是「看 subagent 内部必须独立复现 + 量 response_metadata 的 reasoning_tokens 字段」的范例；也是「provider-agnostic 设计」原则的落地案例（不绑特定 LLM 的 reasoning 行为）。
