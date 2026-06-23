# data-analyst seal handoff 腰斩：讨论过程 + 当前方案 + 待定问题

> **用途**：供别的 agent / 人 review 的讨论存档。记录我们如何从一份 grill 文档逐步推到这里，当前方案 B 的完整设计，以及仍未拍板的问题。
> **日期**：2026-06-23
> **仓库**：`/home/wangqiuyang/noldus-insight`（DeerFlow fork，行为学数据分析 agent）
> **代码基线**：dev HEAD `1651c859`
> **相关文档**：
> - grill 文档：`~/.claude/plans/swirling-tumbling-parasol.md`
> - forensics handoff：`docs/handoffs/2026-06/2026-06-23-data-analyst-seal-args-truncation-forensics-handoff.md`
> - 取证产物：`/tmp/pw-driver/repro-data-analyst.py` + `repro/full/data-analyst-repro000-full.json`

---

## 〇、一句话现状

data-analyst 的 `seal_data_analyst_handoff` tool_call arguments 被腰斩成未终止 JSON → handoff 永不落盘 → FAILED → lead 降级跳过专业判读。根因已坐实（**reasoning_tokens 与 args 共享单次 max_tokens=4096**，reasoning 吃掉大头，留给 args 的不够 → 腰斩）。修复方向收敛为：**提 max_tokens（放宽狭颈）+ 方案 B「模板填充式分步 seal」（结构根治，砍断"args 体积=单次输出"的绑定）**。方案 B 的设计已基本成形，有几个待定问题。

---

## 一、根因（已坐实，有铁证）

### 现象
- data-analyst 派遣后连续 2 次 FAILED。
- `seal_data_analyst_handoff` 的 tool_call arguments 超 `max_tokens=4096` 单 turn 输出预算 → 被腰斩成未终止 JSON（char 1178/2142 处 `Unterminated string`）→ LangChain 判 `invalid_tool_calls` → 不执行 → 无 ToolMessage → SealGate 催回 → 再试再腰斩 → handoff 永不落盘 → FAILED。
- **seal 内容本身正确**（status=completed + 5 条含 Mann-Whitney U/p/Cohen's d 的 key_findings + outliers + method_warnings），纯粹长度撞墙。

### 根因机制（机制甲，已坐实）
**reasoning_tokens 与 args 共享单次 `max_tokens=4096`。**

证据链（双重坐实）：
1. **config.yaml**：data-analyst 用 `deepseek-v4-pro-summary`（底层 `deepseek-v4-flash`），`max_tokens: 4096`，`supports_thinking: false`，class `PatchedReasoningChatOpenAI`。
2. **`PatchedReasoningChatOpenAI`**（`models/patched_reasoning.py`）：只保留 `reasoning_content` 字段到 `additional_kwargs` 给前端显示——**证明 reasoning 确实在跑**（flash 即使 `supports_thinking:false` 也默认产 reasoning）。
3. **`response_metadata` 铁证**：
   ```
   #9:  completion_tokens=4097, reasoning_tokens=3325  → args 腰斩
   #13: completion_tokens=4097, reasoning_tokens=2727  → args 腰斩
   ```
   `completion_tokens=4097 > max_tokens=4096`，且 `reasoning_tokens(3325) < 4097` → **completion_tokens 同时包含 reasoning 和 args，reasoning 计入 max_tokens 预算**。reasoning 吃掉大头(3325)，留给 args+content 只剩 ~770 → args 必然腰斩。

两个独立 agent + 两个独立复现路径（playwright dogfood + repro-data-analyst.py）都指向同一根因。

### 关键澄清：狭颈只在「内容是 LLM 生成的自然语言且体积大」时存在
4 个 subagent 的 handoff 内容来源不同：
| subagent | handoff 内容来源 | 过 max_tokens 狭颈？ |
|---|---|---|
| **code-executor** | bash 脚本算的数值 | 不过（脚本产出，seal 只搬运，从没腰斩）|
| **data-analyst** | LLM 当场生成的自然语言判读 | **过**（命中）|
| **chart-maker** | 脚本画图 + seal 装文件清单 | 不过 |
| **report-writer** | LLM 写报告，但报告主体走 report.md 文件，seal 只装元数据（小）| 当前不过（无腰斩证据）|

**只有 data-analyst 命中狭颈。**

---

## 二、讨论过程（如何一步步推到这里）

### 阶段 1：grill swirl 文档（~/.claude/plans/swirling-tumbling-parasol.md）
swirl 文档提出 **C+A 方案**：
- **C**：data-analyst 把判读 `write_file` 到中间文件 `da_findings.json`，seal 只收文件路径。
- **A**：C 落地后，executor 把 data-analyst 加进 `_AUTO_SEALABLE`，seal 失败时读 da_findings.json 确定性兜底。

grill 发现 swirl 文档的问题：
1. swirl 文档把 `completion=4097, reasoning=3325` **过度诠释**成"reasoning 共享 max_tokens 是铁证"——但 forensics handoff 本身用中性措辞"args 超 4096 单 turn 预算"，没下这个因果断言。**（后来用户确认这个因果是有 config+代码依据的，grill 这点撤回——根因机制甲确实成立。）**
2. swirl 文档的 A 方案与 **5 处明确共识**冲突（forensics handoff §22-23 / etho1 §2.2 / PR#178 / `seal_gate_middleware.py:85` `_RECONSTRUCTABLE` / `executor.py:308` 注释），这些共识都明确"data-analyst 是认知产物，不 auto-seal/兜底"。

### 阶段 2：用户的关键一问——「难道现在 seal 的 handoff 不是 json 文件？」
这一问推翻了 C 方案的前提。查代码确认：
- `seal_data_analyst_handoff` 内部调 `_seal_handoff(DataAnalystHandoff, "handoff_data_analyst.json", payload, runtime)`（`seal_handoff_tools.py:708`）——**最终产物 `handoff_data_analyst.json` 本来就是文件**。
- 腰斩发生在"模型把判读吐进 tool args"这一步，**不在"落盘成文件"**（落盘是 seal 工具的纯函数行为，不经过 LLM max_tokens）。
- **致命发现**：`write_file` 工具的 `content` 字段就是 tool_call arguments 的一部分（`sandbox/tools.py:1800`，`args_schema=_WriteFileArgs` 的一个 Field）——**和 seal args 走同一个 max_tokens 狭颈**。代码注释甚至自己记录了"write_file content 太大模型吐不全"的已知失败模式。

**结论：C 方案无效。** 把 seal args 换成 write_file 的 content，只是把腰斩从 seal 这步挪到 write_file 这步，没绕过狭颈。"让 handoff 走文件"的前提不存在——handoff 早就是文件了。任何"用工具把 LLM 生成的自然语言吐出来"的路径都过狭颈。

### 阶段 3：方案 B「分步 seal」的提出
真正能绕过狭颈的，只有放宽（提 max_tokens）或拆小（把一次大输出拆成多次小输出）。提 max_tokens 是放宽瓶颈不是结构保证（reasoning 自由裁量，更大问题可能复现）。方案 B「分步 seal」是结构根治。

用户对方案 B 的三个关键决策输入：
1. **batch 大小怎么控**：用户指出"核心还是需要一个 tool use 工程化方法"，并让参考 HarnessX 文章（arXiv 2606.14249）+ claude code/hermes 的做法。
2. **中间态怎么落盘**：用户提出"subagent 一开始就给他准备好一个 seal handoff.json，他 parse 看哪空，不断补空，roadmap 就是这份模板"。
3. **SealGate 误判**：用户明确"只有彻底写完才叫 seal"。

### 阶段 4：HarnessX 文章 + 业界调研给出设计原则
- HarnessX line 320：**"Language-model subagents explore, hypothesize, and propose; typed structure and deterministic gates determine what ships."**（LLM 提议，确定性结构定生死）
- HarnessX line 610（Telecom 案例）：连加 5 条 reminder prompt → 第 6 条崩 → **别用 prompt 规则改结构**。
- 业界（Anthropic 等）：structured output 用 **tool-use-based 机制**（一个 schema 工具，模型分次调），不是让模型一次吐全 JSON。

这些和用户三个输入完全对齐，凝练成方案 B 的核心：**用工具签名的粒度（每个字段是天然最小 batch）+ 确定性 gate（finalize 判核心字段填足）控制行为，零 prompt 规则。**

### 阶段 5：用户的两个进一步约束
1. **"两套 handoff 并存会不会迷惑 agent / 负担未来开发？"** —— 拆解后：迷惑 data-analyst 自己（不会，工具集排他）、迷惑 lead/下游（不会，对外文件契约不变）、负担未来开发者（有一点真实成本，用"机制统一、只 data-analyst 启用"化解）。
2. **"in_progress 不是最终交付物，就不应该被读"** —— 固化为 spec 硬约束。

---

## 三、当前方案（方案 B 完整设计）

### 一句话
**harness 预置合法空模板 → data-analyst 用 `fill_handoff_field`（set 为主）逐字段补全 → `finalize` 确定性 gate 判核心字段填足才改 status=completed → SealGate 只认终态/finalize。全程用工具签名粒度 + 确定性 gate 控制行为，零 prompt 规则。**

### 四个确定性结构

**结构 1 — harness 预置空模板（executor 派遣时确定性生成，不经 LLM）**
- executor 在派遣 data-analyst 时（subagent 启动前），用纯 Python 生成 `handoff_data_analyst.json`：`status="in_progress"` + 所有字段空默认值 + 运行时填 `analysis_config_id`。
- Pydantic 先校验（确保模板本身合法，in_progress 态合法）→ 原子写（tmp+rename）。
- **不放静态文件**：`analysis_config_id` 是运行时字段（每个 thread 不同）、路径是 per-thread 的，静态文件行不通。executor 生成是纯代码、无 max_tokens、无腰斩、确定性 100% 成功。
- executor 生成后把字段清单写进 data-analyst 初始上下文，省 agent 第一轮探查。

**结构 2 — `fill_handoff_field` 工具（set 为主，append 兜底）**
```python
@tool("fill_handoff_field")
def fill_handoff_field(
    field: Literal["key_findings","outlier_findings","method_warnings",
                   "recommendations","excluded_metrics","errors",
                   "gate_signals","quality_warnings"],
    mode: Literal["set","append"],   # set=整体覆盖；append=追加一条
    value: ...,                        # set=该字段完整值；append=单条
    runtime: Runtime = None,
) -> str:
    """读模板 → 填/追加该字段 → Pydantic 重新校验 → 原子重写。
    返回当前各字段填充状态，供模型 parse 看哪还空。"""
```
- 每次只填一个字段 → args 天然小，reasoning 再长也吃得过。**用工具签名粒度强制 batch 小，不靠模型自律。**
- set 模式一次填一个字段完整内容（提 max_tokens=8192 后，5 条 finding 的 key_findings 完全装得下），省 turn。

**结构 3 — `finalize` 工具（唯一能把 status 改成终态的入口）**
```python
@tool("finalize_data_analyst_seal")  # 命名待定，见待定问题 2
def finalize_data_analyst_seal(
    final_status: Literal["completed","partial","failed"],
    runtime: Runtime = None,
) -> str:
    """确定性 gate: completed→必须 key_findings 非空
    (DataAnalystHandoff._completed_requires_core_output 既有 validator)。
    改 status → 写 manifest(sealed_by=finalize) → 返回 OK。"""
```

**结构 4 — SealGate 判据改"看 finalize/终态"**
- "已 seal"判定 = `finalize_*` ToolMessage 存在 **或** workspace 文件 `status ∈ {completed,partial,failed}`。
- `in_progress` → 未 seal，SealGate 继续催回。

### 配套改动
- `config.yaml` max_tokens 4096 → **8192**（方案 A，与 B 一起上，一次根治）。
- `DataAnalystHandoff.status` 加 `"in_progress"` 态。
- `data_analyst.py` `max_turns` 12 → **16-18**（fill/finalize 多步，已确认只要不打转提 max_turns 不是问题）。
- `data_analyst.py` system_prompt line 204-211 "产出与交付合一" **反转**为"填模板流程"（read 模板 → fill 逐字段 → finalize）。
- **loop detection lenient 名单**加 `fill_handoff_field`/`finalize_*`（否则多次 fill 被 per-tool frequency 误杀——有前科，memory `feedback_loop_detection_tool_semantics_floor_and_partial_strip`）。

### 为什么只上 data-analyst（"两套并存"的处理）
- 只 data-analyst 命中狭颈；code-executor/chart-maker 产出不过狭颈，给它们上 fill+finalize 是过度设计 + 引入回归。
- **机制层通用**（fill/finalize 底层复用 `_seal_handoff_to_workspace` 同一个纯函数），**实例只启用 data-analyst**。未来开发者看到的是"一套机制 + 一个启用实例"，启用名单解释为什么只 data-analyst（狭颈判据）。report-writer 将来若被证实腰斩，复用同代码开关即可。
- **不选"4 个全改"**：给没病的 subagent 堆机制，引入回归 + reward hacking（漏 finalize 新增失败模式）。

### in_progress 硬约束
> `status="in_progress"` 的 handoff 不是交付物。任何下游读到 in_progress 必须当"未交付"，不得消费字段内容。
- **流程层（主保证）**：report-writer 等 data-analyst finalize 后才被 lead 派遣 → 下游正常路径读不到 in_progress。
- **防御层（异常保证）**：grep 所有读 `handoff_data_analyst.json` status 的地方，加 `if status == "in_progress": treat as not-delivered` 守卫。spec 列为实施第一步核实项。

### 守的铁律
- import 环：工具加在 seal_handoff_tools.py，helper 惰性 import；改完裸导入 `app.gateway` + `make_lead_agent` 0 退出。
- HarnessX 不加 prompt 规则：全程结构（工具签名+gate+status 态）。
- 认知产物不兜底：in_progress 崩溃 = FAILED/降级，**不 auto-seal**（A 方案砍掉）。
- sealed_by 可观测：finalize 写 sealed_by；fill 不写。
- reward hacking：验收看真产物（字段内容）不看 LLM 自述。
- 三指令源：system_prompt（最高权威）+ SKILL.md 同改。
- prompt 契约测试用 importlib 读被测源。

### 测试清单（9 项）
1. 模板预置：executor 派遣后 workspace 有合法 in_progress 模板。
2. fill set：填 key_findings 后字段非空、Pydantic 校验过、原子更新。
3. fill append：追加不覆盖、超长字段不腰斩（args 小）。
4. finalize gate：key_findings 空 + completed → reject。
5. SealGate：in_progress → 催回；finalize 后 → 不催。
6. 崩溃：填一半 + in_progress → executor 判 FAILED/降级（不伪装 completed）。
7. loop detection：多次 fill 不被误杀。
8. import 环：裸导入两入口 0 退出。
9. repro 脚本入库 scripts/forensics/ + 断言：fill/finalize 路径 args 不腰斩。

---

## 四、待定问题（请 review 的 agent 重点看这些）

### 待定 1：旧 `seal_data_analyst_handoff` 工具怎么处理？
data-analyst 改造后，旧的一次性 seal 工具：
- 选项 A：**从 data-analyst 工具集移除**（data-analyst 拿不到旧工具=不可能选错，彻底消除"两套迷惑 agent"），但旧工具代码保留（通用机制一部分，别的 subagent 还用同名 `seal_*_handoff` 模式）。
- 选项 B：旧 `seal_data_analyst_handoff` 代码也删掉，完全由 fill+finalize 取代。但 code-executor/chart-maker/report-writer 的 `seal_*_handoff` 保留（它们不换）。

倾向 A（移除出工具集即可，代码保留）。需 review 确认。

### 待定 2：finalize 工具命名
- 选项 A：`finalize_data_analyst_seal`（per-subagent 名，明确，将来 report-writer 上时另起 `finalize_report_writer_seal`）。
- 选项 B：通用 `finalize_seal`（按当前 subagent 上下文推断或传 subagent_name），机制统一感强，但模型可能传错 subagent_name。

倾向 A（明确）。需 review 确认。

### 待定 3：「机制统一、只 data-analyst 启用」这个分层对不对？
即：不把 4 个 subagent 全改成 fill+finalize，而是做一套通用 fill+finalize 机制代码、只对 data-analyst 开启用开关。理由是只 data-analyst 命中狭颈，全改给没病的 subagent 堆机制引入回归 + reward hacking。
- 这是对"两套并存迷惑"问题的回答。
- 需 review 判断：分层（只 data-analyst）对，还是彻底统一（4 个全改）对？

### 待定 4：fill 的 value 类型怎么表达 list 字段？
`fill_handoff_field(field="key_findings", mode="set", value=???)` —— value 是 `list[str]`（key_findings）还是 `list[dict]`（outlier_findings）还是 `dict`（gate_signals）？一个通用 value 参数怎么容纳异构类型 + Pydantic 校验？
- 选项：value 用 `Any`，fill 工具内部按 field 名查 schema 做类型校验。
- 选项：per-field 拆成多个工具（fill_key_findings / fill_outlier_findings / ...），每个 value 类型明确，但工具数多。
- 需 review 给建议。

### 待定 5：fill 返回的"字段进度"格式
工具返回里带"哪空哪满"供模型 parse 决定下一个 fill。格式是 JSON 进度（`{"key_findings":"filled(5)","outlier_findings":"empty",...}`）还是人类可读文本？
- 影响：JSON 进度更结构化但占 token；人类可读更省但模型可能解析不稳。
- 需 review 给建议。

### 待定 6：finalize 的 partial 触发条件
data-analyst 什么情况下该 finalize 成 partial 而非 completed？现状（`data_analyst.py` system_prompt）已有 fast-fail 规则（statistics 空 / n<3 / 全指标 invalid → partial/failed）。这些规则在新流程下是否照旧适用？fill+finalize 不应改变这些判读语义，只是改变交付机制。需 review 确认语义不漂移。

### 待定 7：max_turns 提到多少有依据？
现状 `max_turns=12`。fill+finalize 多步（read 模板 1 + fill 各字段 ~5-6 + finalize 1 ≈ 8-12 turn，纯 append 会超）。用户同意"不打转就提"，但具体提多少（16? 18? 20?）需要估算 + 实测。需 review 给经验值。

### 待定 8：与 ETHO-1 PR#178 / SealGate `_RECONSTRUCTABLE` 的关系
PR#178 刚把 data-analyst 明确排除在 `_RECONSTRUCTABLE` 外（认知产物不兜底）。方案 B 不做 auto-seal（A 砍了），但 SealGate 判据要改（结构 4）。需确认：方案 B 的 SealGate 改动**只动"已 seal 判定"**（加 finalize/终态识别），**不动 `_RECONSTRUCTABLE`**（data-analyst 仍不在可重建集合，after_agent 仍不兜底）。两者正交，不冲突。需 review 确认这个正交判断对。

---

## 五、关键代码锚点（review 可直接核）
- `config.yaml:21-32`（L29 max_tokens；data-analyst 用 deepseek-v4-pro-summary / deepseek-v4-flash）
- `tools/builtins/seal_handoff_tools.py`：`seal_data_analyst_handoff`(L663)、`_seal_handoff_to_workspace`(L535 纯函数)、`_seal_handoff`(L590)、`_resolve_workspace`(L126)
- `sandbox/tools.py:1800`（write_file 的 content 是 args_schema 的 Field → 过同一 max_tokens 狭颈，证明 C 方案无效）
- `subagents/builtins/data_analyst.py`：system_prompt(L11起)、产出交付合一段(L204-211)、fast_fail(L80-92/141/145)、max_turns=12(L324)、关thinking注释(L312-323)
- `subagents/handoff_schemas.py`：`DataAnalystHandoff`(L658)、`_completed_requires_core_output`(L717-728，完成判据=status==completed and key_findings 非空)
- `subagents/executor.py`：派遣钩子（预置模板要插这里）、`_AUTO_SEALABLE`(L295)、`_attempt_auto_seal_from_artifacts`(L301)、`calculate_subagent_recursion_limit`(L53)、max_turns 终止(L1322)
- `agents/middlewares/seal_gate_middleware.py`：`_REQUIRES_SEAL`(L68，含 data-analyst)、`_RECONSTRUCTABLE`(L85，不含 data-analyst)、after_agent(L170)、"已 seal"判定(seal ToolMessage 出现，L27)
- `agents/middlewares/loop_detection_middleware.py`：per-tool frequency 阈值(L200，write_todos:(15,30))→ fill/finalize 要加 lenient 名单
- `models/patched_reasoning.py`：保留 reasoning_content（证明 reasoning 在跑）
- repro：`/tmp/pw-driver/repro-data-analyst.py` + `repro/full/data-analyst-repro000-full.json`

---

## 六、参考来源
- HarnessX 报告（arXiv 2606.14249）：`/home/wangqiuyang/resources/parsed/2606.14249v1/vlm/2606.14249v1.md`
  - line 320：「LLM 提议，确定性 gate 定生死」
  - line 610：Telecom 连加 reminder → 崩（别用 prompt 规则改结构）
  - line 568：Digester 把 10M trace 压成 10K structured summary（结构化中间态再消费）
- 业界 structured output：Anthropic 等用 tool-use-based 机制（schema 工具分次调），非一次吐全 JSON
