# Spec：data-analyst seal handoff 腰斩根治 —— 分步填模板（产物 + 封口），DeerFlow 原生

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-23
> 性质：🔴 高 · data-analyst 判读层功能性失效（连续 FAILED、lead 被迫降级跳过专业判读）的结构根治
> 代码基线：dev HEAD `5a622865`（含 ETHO-1 PR#178 / ETHO-2 #180 / ETHO-9 #179 已合）
> 来源 / 讨论存档：
> - 取证：根因取证（response_metadata 铁证、playwright dogfood + `repro-data-analyst.py` 双复现路径）见 `docs/problems/2026-06-23-data-analyst-seal-args-truncation-discussion.md` §一 + 取证产物 `/tmp/pw-driver/`（可入库 `scripts/forensics/`）
> - 讨论：`docs/problems/2026-06-23-data-analyst-seal-args-truncation-discussion.md`（grill 推翻 C、收敛到方案 B 的全过程）
> - grill：`~/.claude/plans/swirling-tumbling-parasol.md`
> - 取证产物：`/tmp/pw-driver/repro-data-analyst.py` + `repro/full/data-analyst-repro000-full.json`

---

## 〇、给实施 agent 的一句话

data-analyst 的判读是**唯一**"LLM 当场生成、直接当 `seal_data_analyst_handoff` 的 tool_call arguments 一次性吐出"的内容；它和模型的 reasoning_tokens 共享单次响应 `max_tokens=4096`，flash 默认产 3300-4096 reasoning → args 被挤穿腰斩成 `Unterminated string` → `invalid_tool_calls` → handoff 永不落盘 → FAILED → lead 降级。**根治 = 让 data-analyst 像其他三个 subagent 一样「先把工作变成落盘产物、再封口」**：harness 预置一个合法的 `in_progress` 结构化 handoff 模板 → data-analyst 用 `fill_*` 工具**逐字段**填（每次 args 天然小，绕过狭颈）→ `finalize` 确定性 gate 判核心字段填足才改 status=completed。配套提 `max_tokens` 4096→8192 作第一道防线。**全程用工具签名粒度 + 确定性 gate 控制行为，零新 prompt 规则**（守 HarnessX Telecom 禁令）。**只改 data-analyst**（三个代码铁证：另三个 subagent 不撞狭颈，全替换会让 code-executor 退化）。

---

## 一、根因（已坐实，有铁证）

### 现象
data-analyst 派遣后连续 2 次确定性 FAILED。`seal_data_analyst_handoff` 的 tool_call arguments 在 char 1178/2142 处被腰斩成未终止 JSON → LangChain 判 `invalid_tool_calls` → 不执行 → 无 ToolMessage → SealGate 催回 → 再试再腰斩 → handoff 永不落盘 → FAILED → lead 放弃 data-analyst、自己 read code-executor 结果给统计摘要，**跳过专业判读**（混杂排查/效应量/离群警示全丢）。seal 内容本身正确（status=completed + 5 条含 Mann-Whitney U/p/Cohen's d 的 key_findings），纯粹长度撞墙。

### 根因机制（双重坐实）
**reasoning_tokens 与 args 共享单次 `max_tokens=4096`。**
1. `config.yaml:29` data-analyst 用 `deepseek-v4-pro-summary`（底层 `deepseek-v4-flash`），`max_tokens:4096`，`supports_thinking:false`，class `PatchedReasoningChatOpenAI`。
2. `models/patched_reasoning.py` 保留 `reasoning_content` 到 `additional_kwargs` → 证明 reasoning 在跑（flash 即使 `supports_thinking:false` 也默认产 reasoning；`supports_thinking` 只控制是否主动发 `enable_thinking` 参数）。
3. `response_metadata` 铁证：
   ```
   #9:  completion_tokens=4097, reasoning_tokens=3325  → args@1178char 腰斩
   #13: completion_tokens=4097, reasoning_tokens=2727  → args@2142char 腰斩
   #15: completion_tokens=4097, reasoning_tokens=4096  → reasoning 几乎吃光全部预算
   ```
   `completion_tokens(4097) > max_tokens(4096)`，且 `reasoning(3325) < completion` → completion 同时含 reasoning + args，reasoning 计入预算，吃掉大头 → 留给 args 只剩 ~770 → 必然腰斩。

两个独立 agent + 两条独立复现路径（playwright dogfood + `repro-data-analyst.py`）指向同一根因。

### 为什么只有 data-analyst 撞狭颈（核心，决定本 spec 只改它）
四个 subagent 的 handoff 内容来源不同——**只有 data-analyst 把大段 LLM 当场生成的自然语言直接当 tool args 吐**：

| subagent | handoff 内容来源 | 现在怎么落盘（已核实代码） | 撞狭颈？ |
|---|---|---|---|
| **code-executor** | bash 脚本算的数值 | `run_metric_plan` 工具进程池跑完 compute+stats+聚合，**纯 Python 确定性落盘**（`sealed_by="run_plan"`，`code_executor.py:40/49`），LLM 正常路径**无需调 seal** | **否**（LLM 0 次调用）|
| **chart-maker** | 脚本画图 | 图（png）脚本直接写 `/mnt/user-data/outputs/*.png`（`chart_maker.py:22`），seal 只装**文件路径清单 + 一句话 summary** | **否**（seal 装小元数据）|
| **report-writer** | LLM 写报告 | 报告主体 write_file 到 `report.md`，**超 8000 字符按 chunking 分段写**（`report_writer.py:261/288`）；seal 只装 `report_path`+6 个章节名 | **否**（大产物走文件+分段，seal 小）|
| **data-analyst** | LLM 当场生成的自然语言判读 | LLM 一次性把全部判读塞进 seal args | **是**（唯一命中）|

**统一心智模型**：其他三个都"**先把工作变成落盘产物、seal 只封口记元数据**"。data-analyst 缺"先变产物"这一步——它的判读唯一成文就在 seal args 里。**本 spec 给它补上这一步**。

### 为什么"全替换成 fill/finalize"是错的（坐实，非倾向）
- 给 code-executor 上 fill/finalize = 把"LLM 0 次调用的确定性落盘"**退化**成"LLM 多步 fill"，凭空加回 LLM 失败面，违反 CLAUDE.md「LLM 提议，确定性门定生死」（确定性能做的不该退回让 LLM 做）。
- 给 chart/report 的小 seal 上 fill/finalize = 无收益纯加复杂度 + 引入"漏 finalize"新失败模式。
- **现状不是冗余的"两套"，是三种内容形态的三种最优解**（确定性数值→代码落盘；大文件产物→走文件+分段；LLM 判读无文件产物→分步填结构化模板）。本 spec 把这套分工写成显式知识（§三.6），不强行物理统一。

### 为什么不是"判读 write_file 成自由文本产物"（用户思路的修正）
"把判读 write_file 成一个文件"行不通两点：① write_file 的 `content` 也是 tool_call args（`sandbox/tools.py:1800`，`_WriteFileArgs.content` Field），走**同一个 max_tokens 狭颈**，大判读照样腰斩（除非分段=回到分步）；② 下游消费方（report-writer `report_writer.py:32/114/119/134` 逐字段读 `key_findings`/`outlier_findings`/`recommendations` 写不同章节；lead `prompt.py:348` 搬运 key_findings；`quality_warning_broadcast_middleware` 读 quality_warnings）**全依赖结构化字段**，读不了自由文本。所以 data-analyst 的产物**必须是结构化的**——即现有 `DataAnalystHandoff` 形状，只是改成**分步填**。

---

## 二、方案（分步填模板：产物 + 封口）

### 一句话
**harness 预置合法空模板（`status="in_progress"`）→ data-analyst 用 `fill_data_analyst_field`（按类型分 3 工具，set 为主）逐字段补全 → `finalize_data_analyst_handoff` 确定性 gate 判 key_findings 非空才改 status=completed → SealGate 对 data-analyst 认 finalize/终态而非旧 seal。配套提 max_tokens 8192。**

### 四个确定性结构

**结构 1 — harness 预置 in_progress 空模板（executor 派遣时纯 Python 生成，不经 LLM）**
- executor 在 data-analyst 启动前（`_build_initial_state` 之后、agent 跑之前的派遣路径），用纯 Python 生成 `handoff_data_analyst.json`：`status="in_progress"` + 所有字段空默认 + 运行时填 `analysis_config_id`（从 `experiment-context.json` 读，复用 `_read_analysis_config_id`）。
- Pydantic 先校验模板合法（需 `in_progress` 入 status Literal，见结构改动）→ 原子写（tmp+rename，复用 `_seal_handoff_to_workspace` 的写盘路径或其纯函数变体）。
- **不放静态文件**：`analysis_config_id` + 路径都是 per-thread 运行时值。executor 生成是纯代码、无 max_tokens、确定性 100% 成功。
- 把已预置事实写进 data-analyst 初始上下文（"模板已就绪在 workspace，请逐字段 fill"），省 agent 第一轮探查。
- **预置失败 = 派遣失败**（grill 问题 1）：模板写失败（磁盘满/权限/workspace_path 缺）→ **响亮报错、终止 data-analyst 派遣**（让 lead 重试/降级），**不放行让 subagent 自己生成**——绝不进入"subagent 起了但没模板"的半残态（subagent 后续 fill 会找不到模板）。
- **幂等覆盖**（grill 问题 1）：每次派遣 data-analyst 都是独立分析。派遣时**无条件覆盖**预置（无论 workspace 已有上次的 `in_progress`（FAILED 残留，覆盖防污染）还是终态模板（新分析，覆盖重置）），原子写保证不留半新半旧。

**结构 2 — `fill_data_analyst_field` 工具（按值类型分 3 个，避免 `value: Any` 丢类型引导）**

> ⚠️ **与 discussion §四待定 4 分歧的决议**：discussion 倾向"一个 value: Any 工具内部按 field 查 schema"。本 spec 决议**按值类型拆 3 个工具**——理由：`value: Any` 在 args_schema 里让 LLM 失去类型引导（不知该吐 list 还是 dict），更易吐错结构；且 memory `feedback_langchain_tool_args_schema_strips_injected` 踩过 args_schema 坑。DataAnalystHandoff 字段实测 4 类型，归并为 3 工具：

```python
# 工具 A：5 个 list[str] 字段
@tool("fill_data_analyst_text_list")
def fill_data_analyst_text_list(
    field: Literal["key_findings","excluded_metrics","method_warnings",
                   "recommendations","errors"],
    mode: Literal["set","append"],
    value: list[str],
    runtime: Runtime = None,
) -> str: ...

# 工具 B：list[dict] 字段（outlier_findings: list[OutlierFinding] / quality_warnings: list[DataQualityWarning]）
@tool("fill_data_analyst_record_list")
def fill_data_analyst_record_list(
    field: Literal["outlier_findings","quality_warnings"],
    mode: Literal["set","append"],
    value: list[dict],            # 工具内按 field 选 OutlierFinding/DataQualityWarning 校验
    runtime: Runtime = None,
) -> str: ...

# 工具 C：gate_signals（GateSignals|None，单 dict）
@tool("fill_data_analyst_gate_signals")
def fill_data_analyst_gate_signals(value: dict, runtime: Runtime = None) -> str: ...
```
- **gate_signals 一次填全**（grill 问题 3）：无 mode（单 dict 无 append 语义）。prompt 指引「gate_signals 一次整体填」——字段不多、一次装得下，不给分次补子键留口（否则每次重传整 dict、args 可能变大）。
- 公共行为：读模板 → 填/追加该字段 → 用对应 Pydantic 子模型校验该字段 → 整体 DataAnalystHandoff 重校验（保持 in_progress 合法）→ 原子重写 → **返回极简 JSON 进度**（见结构改动「进度格式」）。
- `set` 一次填一个字段完整内容（提 max_tokens=8192 后，5 条 key_findings 完全装得下）；`append` 兜底超长字段。每次只填一个字段 → args 天然小，reasoning 再长也吃得过。**用工具签名粒度强制 batch 小，不靠模型自律。**
- `parameter_audit_findings` 恒空（2026-06-18 起不产出），**不给 fill 入口**（模板预置为 `[]`，finalize 不要求）。

**结构 3 — `finalize_data_analyst_handoff` 工具（唯一能把 status 改成终态的入口）**
```python
@tool("finalize_data_analyst_handoff")
def finalize_data_analyst_handoff(
    final_status: Literal["completed","partial","failed"],
    runtime: Runtime = None,
) -> str:
    """确定性 gate：final_status=completed → 必须 key_findings 非空
    （复用 DataAnalystHandoff._completed_requires_core_output 既有 validator）。
    读模板 → 改 status=final_status → 重校验 → 写 manifest(sealed_by="finalize")
    → 返回 OK。gate 拒绝时返回 ValueError（LangChain 转 error ToolMessage 引导补 fill）。"""
```
- partial/failed 不要求 key_findings 非空（fast-fail 路径合法）。
- 命名用 per-subagent 名（`finalize_data_analyst_handoff`）：工具集排他，模型无法传错 subagent（决议 discussion §四待定 2 选 A）。
- **gate 走整体重校验、不自己写判空**（grill 问题 2，守 SSOT）：`_completed_requires_core_output` 是 `@model_validator(mode="after")`（`handoff_schemas.py:716`，非可单独调的函数）。finalize 实现 = `DataAnalystHandoff(**payload)` **整体重校验让 validator 自然触发**，**绝不自己写 `if final_status=="completed" and not key_findings`**——否则两处 key_findings 判空逻辑漂移（memory `feedback_single_source_of_truth`）。validator raise 的 ValueError 由 LangChain 转 error ToolMessage 引导补 fill。

**结构 4 — SealGate 对 data-analyst 认 finalize/终态**
- `seal_gate_middleware.py` 的 data-analyst 实例：`_seal_tool` 语义从 `seal_data_analyst_handoff` 换成 `finalize_data_analyst_handoff`；`_check`（after_model）的"已 seal 判定"对 data-analyst = `finalize_*` ToolMessage 存在 **或** workspace 文件 `status ∈ {completed,partial,failed}`。
- `status=="in_progress"` → 未 seal，继续催回（催"调 finalize"，不催旧 seal）。
- **不动 `_RECONSTRUCTABLE`**（仍 `{report-writer, chart-maker}`，data-analyst 不在）——本方案不做 auto-seal，after_agent 对 data-analyst 仍 pass-through（与 PR#178 正交，§六.Q9）。

### 配套改动
- `config.yaml:29` `deepseek-v4-pro-summary` `max_tokens: 4096` → **8192**（第一道防线降触发率；docker-compose 无 max_tokens 覆盖，config.yaml 直接 ship，无 dev/prod 对齐风险）。
- `DataAnalystHandoff.status` Literal 加 `"in_progress"`（`handoff_schemas.py:669`）。**下游守卫见 §三.5**。
- `subagents/builtins/data_analyst.py`：
  - `system_prompt`（最高权威源）：L204-211「产出与交付合一」段**反转**为「填模板流程」（模板已预置 → 逐字段 fill → finalize）；fast-fail 路径 L88/L145「立即调 seal_data_analyst_handoff」改「fill 完调 finalize(partial/failed)」。守 §6 deepseek 正面提示（用"填模板再 finalize"，不用"不要直接 seal"）。
  - SKILL.md（`skills/custom/ethoinsight/`）同改（三指令源对齐）。
  - `max_turns` 12 → **待 repro 实测定**（read 模板 1 + fill ~5-6 + finalize 1 ≈ 8-12，纯 append 会超；用 repro 实测真实 turn 数 ×1.5 余量后填，不在 spec 拍死数字）。**依赖顺序（grill 问题 4，防搞反）**：max_turns 必须在 **T-repro 改造成走新 fill/finalize 路径之后**实测（repro 现在跑旧流程，turn 数不可用）；实测定值前用 **18 作临时上限**。
  - 从 data-analyst 工具集**移除** `seal_data_analyst_handoff`、**加入** 3 个 fill + 1 个 finalize（旧 seal 工具代码保留——别的 subagent 不用它但纯函数共用，决议 discussion §四待定 1 选 A）。
- `loop_detection_middleware.py:_TOOL_FREQ_SEMANTIC_OVERRIDES`（L196）加 `fill_data_analyst_text_list`/`fill_data_analyst_record_list`/`fill_data_analyst_gate_signals` lenient（多次 fill 不被 per-tool frequency 误杀；前科 memory `feedback_loop_detection_tool_semantics_floor_and_partial_strip`）。
- `_attempt_seal_resume`（`executor.py:1076`）：data-analyst 的补轮 `seal_tool` 名换成 `finalize_data_analyst_handoff`（**隐藏耦合点**，§六.Q-resume）。

---

## 三、改动清单（change manifest：每处「预期改善 + 可能回归 + 测试」）

| # | 文件:锚点 | 改动 | 预期改善 | 可能回归 | 测试 |
|---|---|---|---|---|---|
| 1 | `config.yaml:29` | max_tokens 4096→8192 | 降腰斩触发率 | 成本/延迟↑（可接受，低频高价值） | 手动核 + repro |
| 2 | `handoff_schemas.py:669` | status Literal 加 in_progress | 模板合法 | 下游误消费 in_progress | T1 + §3.5 守卫 |
| 3 | `seal_handoff_tools.py` | 加 3 fill + 1 finalize 工具（复用 `_seal_handoff_to_workspace`）；helper 惰性 import | args 天然小绕狭颈 | import 环 | T2/T3/T4 + import guard |
| 4 | `executor.py:_build_initial_state` 后派遣路径 | 预置 in_progress 模板（纯 Python） | 模板确定性就绪 | 模板写失败致 subagent 无 handoff | T1 |
| 5 | `seal_gate_middleware.py` data-analyst 实例 | _seal_tool→finalize；已 seal 判定认 finalize/终态 | in_progress 正确催回 | 误判已 seal/漏催 | T5 |
| 6 | `data_analyst.py` system_prompt + SKILL.md | 反转为填模板流程；fast-fail 改 finalize | 行为对齐新机制 | prompt 削别处约束 | T-contract（importlib）|
| 7 | `data_analyst.py` 工具集 | 移除 seal、加 fill/finalize；max_turns 实测 | 排他无误选 | max_turns 设低卡死 | T7 + repro 实测 |
| 8 | `loop_detection_middleware.py:196` | fill/finalize 加 lenient | 多次 fill 不误杀 | — | T7 |
| 9 | `executor.py:_attempt_seal_resume:1076` | data-analyst 补轮催 finalize | 补轮催对工具 | 催错→补轮白做 | T-resume |

---

### 3.5 in_progress 硬约束（discussion §五固化）
> `status="in_progress"` 的 handoff **不是交付物**。任何下游读到 in_progress 必须当"未交付"，不得消费字段内容。
- **流程层（主保证）**：report-writer 等在 data-analyst finalize 后才被 lead 派遣 → 正常路径读不到 in_progress。
- **防御层（异常保证）**：实施**第一步**先 grep 所有读 `handoff_data_analyst.json` 的 status 判定点（已知：`executor.py:141` _check_data_analyst_content、`quality_warning_broadcast_middleware.py:42`、`report_writer.py:226`、lead 路由 `prompt.py:1119/1128`），逐个加 `if status=="in_progress": treat as not-delivered`。

### 3.6 三套机制形态映射（写进 spec/注释，化解"维护负担"担忧）
| 内容形态 | 最优落盘 | 谁用 |
|---|---|---|
| 确定性数值 | 代码直接落（run_metric_plan） | code-executor |
| 大文件产物（图/报告） | 产物落文件 + seal 装小元数据（+分段） | chart-maker / report-writer |
| LLM 当场判读（无文件产物） | **分步填结构化模板 + finalize 封口**（本 spec 新增） | **data-analyst** |
- 启用名单做成显式常量 + 注释写清判据（"只 data-analyst 命中 max_tokens 狭颈"），未来开发者一眼懂为何只它用 fill/finalize；report-writer 将来若被证实腰斩，复用同代码开关即可。

---

## 四、测试清单（TDD 红→绿 + smoke）
1. **T1 模板预置**：executor 派遣 data-analyst 后，workspace 有合法 `in_progress` 模板（Pydantic 校验过）。
2. **T2 fill set**：填 key_findings 后字段非空、整体校验过、原子更新、返回进度。
3. **T3 fill append**：追加不覆盖；超长 list 字段分多次 append 不腰斩（args 小）。
4. **T4 finalize gate**：key_findings 空 + completed → reject（ValueError）；partial 不要求。
5. **T5 SealGate**：in_progress → 催回（催 finalize）；finalize 后 → 不催；旧 seal ToolMessage 不再被 data-analyst 当终结。
6. **T6 崩溃诚实**：填一半 + in_progress + turn 用完 → executor 判 FAILED/降级（**不伪装 completed、不 auto-seal**）。
7. **T7 loop detection**：多次 fill 不被 per-tool frequency 误杀。
8. **T8 import 环**：裸导入 `import app.gateway` + `from deerflow.agents import make_lead_agent` 0 退出。
9. **T-contract**：prompt 契约（importlib 读被测 data_analyst.py 源）断言 system_prompt 含填模板流程、不含旧"产出交付合一/立即 seal"措辞。
10. **T-resume**：data-analyst 补轮 seal_tool 名 = finalize_data_analyst_handoff。
11. **T-repro（回归基线）**：`repro-data-analyst.py` 入库 `scripts/forensics/`，改造为走 fill/finalize 路径，断言 **args 不再 invalid_tool_calls、handoff finalize 落盘、key_findings ≥3 含三项 judgment（混杂排查/效应量/离群）**。

---

## 五、验收（确定性 gate 序列，第一个失败即停）
1. manifest 完整（§三 9 处全覆盖测试）。
2. smoke：3 个 fill + finalize 工具实例化跑通（纯函数路径）。
3. 红→绿：T1-T11 改前红、改后绿。
4. 回归（seesaw）：seal 邻域 + executor + data_analyst + loop_detection 全量；backend 全量（守 memory：native 主仓全量 3 failed 为已知污染基线，比文件集不比数）。
5. import 环：T8 两入口 0 退出。
6. **端到端 dogfood**（关键）：同份 28-subject EPM 复跑，data-analyst 一次派遣内 finalize 成功、handoff 落盘、lead 搬运判读（非降级）。
7. **净增失败面实测**（§六.R1 生死点）：repro 改造成新路径后跑 smoke（≥5 次快速验"args 不再腰斩、handoff finalize 落盘"）；**漏调率的真实判据靠上线后 `sealed_by`/finalize 触发率长期观测**（grill 问题 5：5 次统计意义太弱，漏调率若 ~10% 则 5 次里 0 命中也正常；长期观测才是正解，接 memory `feedback_fallback_trigger_rate_must_be_observable_acceptance_criterion`）。若 smoke 即暴露多步漏调高发 → 回退「只提 max_tokens + invalid-args 抢救 parse」轻方案。

---

## 六、风险与三大病理自检

### 三大病理自检（CLAUDE.md 固定段）
1. **Reward hacking**：finalize gate 看**真产物**（key_findings 字段内容非空）不看 LLM 自述；崩溃 = FAILED/降级**不 auto-seal**（不给"学会依赖兜底"的口子）；`sealed_by="finalize"` 可观测。
2. **Catastrophic forgetting**：改 system_prompt 前 grep 所有 seal 指引点（L88/L145/L204-211 + SKILL.md）一并对齐；不削弱 fast-fail 判读语义（n<3/statistics 空/全 critical 三态照旧，只改交付动作从 seal→finalize）。
3. **Under-exploration**：本方案是**结构改动**（工具签名+gate+status 态），不是加提醒 prompt——合规。守 HarnessX Telecom 禁令（discussion §四引 line 610）。

### 新发现的 3 个真风险（discussion 未单列，本 spec 列为生死点）
- **R1（净增失败面，最关键）**：多步（read 模板→fill ×5-6→finalize）比一步漏调点更多。方案赌"小 args 永不腰斩 > 多步漏调风险"——**必须 §五.7 repro 实测坐实**，不能只靠推理。若实测净增，回退到"只提 max_tokens + invalid-args 抢救 parse"的更轻方案。
- **R2（seal-resume 隐藏耦合，§三 #9）**：`_attempt_seal_resume:1076` 硬编码催 `seal_{name}_handoff`；data-analyst 改 finalize 后这条必须同步改成催 finalize，否则补轮催一个 data-analyst 已无的工具 → 永远失败。
- **R3（短 handoff 过度复杂化）**：fast-fail 路径（n<3 只描述统计）判读本就短、不撞狭颈；新机制别把"本来一步能成"的简单场景强制拖成多步。**缓解**：fast-fail 路径允许一次 `fill(key_findings, set)` + `finalize(partial)` 两步即可（不强制填满所有字段）。

### Q9（与 ETHO-1 PR#178 / `_RECONSTRUCTABLE` 关系，正交确认）
PR#178 把 data-analyst 明确排除在 `_RECONSTRUCTABLE` 外（认知产物不 auto-seal）。本方案**不做 auto-seal**（不碰 `_RECONSTRUCTABLE`），只改 SealGate 的"已 seal 判定"（结构 4，after_model 路径，认 finalize）。两者**正交不冲突**：after_agent 对 data-analyst 仍 pass-through（`_after_agent_check:204` `if name not in _RECONSTRUCTABLE: return None`），本方案不改这条。

---

## 七、守的铁律
- import 环：fill/finalize 加在 `seal_handoff_tools.py`，helper 惰性 import；改完裸导入两入口 0 退出。
- HarnessX 不加 prompt 规则：全程结构（工具签名+gate+status 态）。
- 认知产物不兜底：in_progress 崩溃 = FAILED/降级，**不 auto-seal**。
- sealed_by 可观测：finalize 写 `sealed_by="finalize"`；fill 不写。
- 三指令源：system_prompt（最高权威）+ SKILL.md 同改；改前 grep `builtins/data_analyst.py`。
- 受保护文件 sync surgical：seal_handoff_tools.py / executor.py / data_analyst.py / config.yaml / seal_gate_middleware.py / loop_detection_middleware.py / handoff_schemas.py 全是 deerflow 定制面。
- prompt 契约测试用 importlib 读被测源。

---

## 八、关键代码锚点（实施可直接核）
- `config.yaml:21-32`（L29 max_tokens）
- `tools/builtins/seal_handoff_tools.py`：`seal_data_analyst_handoff`(L663)、`_seal_handoff_to_workspace`(L535 纯函数)、`_seal_handoff`(L595)、`_resolve_workspace`(L126)、`_read_analysis_config_id`(L298)、`_update_manifest`(L315)
- `subagents/builtins/data_analyst.py`：system_prompt(L11起)、产出交付合一段(L204-211)、fast_fail 立即seal(L88/L145)、model+max_turns(L323-324)、关thinking注释(L312-323)、工具集(L339/L350)
- `subagents/executor.py`：`_build_initial_state`(L1020 预置模板插入点)、`_validate_handoff_emitted`(L176)、`_check_data_analyst_content`(L140 查key_findings)、`_attempt_seal_resume`(L1076 R2耦合点)、`_AUTO_SEALABLE`(L295 不动)、FAILED链(L1414-1465)
- `subagents/handoff_schemas.py`：`DataAnalystHandoff`(L658)、status Literal(L669)、`_completed_requires_core_output`(finalize gate 复用)、`OutlierFinding`(L630)、`GateSignals`(L341)、`DataQualityWarning`(L131)
- `agents/middlewares/seal_gate_middleware.py`：`_REQUIRES_SEAL`(L68)、`_RECONSTRUCTABLE`(L85 不动)、`_seal_in_history`(L103)、`_check`(L243)、after_agent(L170/204)、`__init__`(L121)、`_seal_tool`(L124 由 `_seal_tool_name` 算)、**per-instance**（`executor.py:977` 每 subagent 一实例 → 改 data-analyst 实例的 `_seal_tool` 即可，无需单实例分支，grill 问题 6 已核实）
- `agents/middlewares/loop_detection_middleware.py`：`_TOOL_FREQ_SEMANTIC_OVERRIDES`(L196)
- `sandbox/tools.py:1800`（write_file content 过同一狭颈，证 C 无效）
- repro：`/tmp/pw-driver/repro-data-analyst.py` + `repro/full/data-analyst-repro000-full.json`

---

## milestone 建议
「seal 漏调 / subagent handoff 鲁棒性」track 第 5 类根因（flash seal args 撞输出预算腰斩）的**结构根治**：
- 第 1-3 类：收尾漏 call → SealGate 结构门（已修）。
- 第 4 类：thinking 撞 turn 超时（v4-pro）→ 06-18 换 flash（PR#161，部分有效，留 args-truncation 镜像）。
- **第 5 类（本 spec）：分步填模板（产物+封口）**，把 data-analyst 从"一次性大 seal 撞狭颈"对齐到其他 subagent 的"先产物后封口"心智模型。checkpoint：「根因坐实 reasoning 共享 max_tokens → C(走文件)被证无效(write_file 同狭颈) → 方案 B(分步填结构化模板) → 待实施 PR」。
