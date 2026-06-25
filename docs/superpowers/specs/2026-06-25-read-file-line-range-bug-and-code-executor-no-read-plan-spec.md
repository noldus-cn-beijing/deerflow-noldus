# Spec：read_file 越界静默 bug 工程修 + code-executor「不 read 大 plan」prompt 双管

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-25
> 代码基线：dev HEAD `b00c39e4`
> 性质：🔴 高 · 工具行为 bug（影响所有 subagent 读大文件）+ code-executor 策略 prompt。
> **方向（用户拍板 2026-06-25）**：EPM dogfood（thread `0e72d605`）暴露 code-executor 反复 read_file 找 plan 的 statistics 段、陷入 20+ 轮死循环（最终自己醒悟改调 run_metric_plan 才跑通）。用户追问「纯 prompt 能彻底改善吗」→ 结论：**不能，必须 prompt + 工程化双管**。read_file 有两个真 bug（工程层，prompt 治不了）；code-executor 选错策略（prompt 层）。两层各治一层，缺一不彻底。
> 受保护文件（sync surgical）：`sandbox/tools.py`、`subagents/builtins/code_executor.py`。

---

## ⚠️ 〇、根因（取证坐实，守 ETHO-2 教训）

> 本 spec 来自 EPM dogfood（thread `0e72d605`，2026-06-25，端到端成功但 code-executor 阶段死循环 20+ 轮）。**不凭 LLM trace 自述，已核 read_file 工具代码坐实。**

### 现象（dogfood trace 逐字）
code-executor 收到「按 plan_metrics.json 算指标」任务后，**没有直接调 run_metric_plan**，而是反复 read_file 想确认 plan 末尾有 `statistics` 段：
> *"The file is 140360 chars... truncated at 49892 chars each time... I keep reading the same data because the file is truncated - the read_file is showing the beginning of the file again (it's wrapping around)."*
> *"each read is starting at line 1 when the range goes beyond the file length."*

循环 20+ 轮思考 + read_file 后才醒悟：
> *"Let me just run run_metric_plan. The tool reads the plan file and handles all the details internally including finding the statistics section."*

一调 run_metric_plan → **140 任务一次成功**（statistics 段从头到尾都在，`has statistics: True` 已验）。

### 真根因（三层，分别坐实）

| 层 | 根因 | 代码铁证 | prompt 能治吗 |
|---|---|---|---|
| **R1（工程）** | read_file **只传 start_line 不传 end_line** 时，切片逻辑**整个被跳过**（`if start_line is not None and end_line is not None`），返回**全文件**经头部截断 → LLM 永远看到**开头** | [tools.py:1726-1727](../../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py#L1726)：`if start_line is not None and end_line is not None: content = "\n".join(content.splitlines()[start_line-1:end_line])`。只传 start_line → 不进 if → content=全文件 → `_truncate_read_file_output` 头部截断保留**开头**49892 字节 | ❌ 工具行为缺陷 |
| **R2（工程）** | read_file **start_line 越界**（超过文件行数）时，切片返回**空列表** → content="" → 静默返回空，**无「已超过文件末尾（共 N 行）」提示** | [tools.py:1727](../../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py#L1727)：`splitlines()[2894:end]` = `[]` → `""`。LLM 拿到空内容**分不清「超末尾」还是「这段真空」**，换 start_line 再试 → 死循环 | ❌ 工具行为缺陷 |
| **R3（策略）** | code-executor **选错策略**：去 read_file「肉眼验货」plan 有没有 statistics，而 plan 是给 run_metric_plan 吃的、不是给人看的 | system_prompt 无「不要 read 大 plan」指引 | ✅ prompt 可治 |

**关键洞察（回应用户「纯 prompt 够不够」）**：
- 即使 R3 的 prompt 100% 让 code-executor 不 read plan，**R1/R2 仍存在**——任何 subagent（report-writer 读 handoff、data-analyst 读 stats.json 44K、lead 读任何大文件）只传 start_line 或越界都会踩。**prompt 只让 code-executor 这一个场景绕开，没消除 bug。**
- 因此**彻底改善 = R1+R2 工程修（治本，全 subagent 受益）+ R3 prompt（治标，减少 code-executor 触发）**。守 CLAUDE.md「LLM 提议，确定性门定生死」：R1/R2 是结构缺陷上工程修，R3 是策略指引改 prompt。

### 旁支（本 spec 不修，记录待查）
dogfood trace 显示 run_metric_plan **第一次被中断**（"Tool call was interrupted and did not return a result"），code-executor 查无 handoff 后重跑才成功。中断根因未知（超时？SSE 断？）。**本 spec 范围只治 read_file + 策略，中断单列待查**（见 §七）。

---

## 一、给实施 agent 的一句话

修 `read_file_tool`（`sandbox/tools.py`）两个越界静默行为：M1（只传 start_line 时按「从 start_line 到末尾」切，不再整段忽略）+ M2（start_line 越界时返回明确错误 `「start_line=N 超过文件末尾（共 M 行）」`，不静默返回空）。同步 M3 改 code-executor system_prompt：明示「plan_metrics.json 是 run_metric_plan 的输入、不是给你 read 的，**直接调 run_metric_plan，不要 read_file 读 plan 内容**」。TDD 覆盖三个修法。

---

## 二、修法（M1/M2/M3）

### M1：只传 start_line 时按「start_line 到末尾」切（治 R1）

**改 [tools.py:1726-1727](../../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py#L1726)**：

```python
# 当前（bug）：必须两个都传才切片，只传 start_line 静默返回全文件头部
if start_line is not None and end_line is not None:
    content = "\n".join(content.splitlines()[start_line - 1 : end_line])

# 改后：支持单独传 start_line（从 start_line 到末尾）/ 单独传 end_line（从头到 end_line）
if start_line is not None or end_line is not None:
    lines = content.splitlines()
    total_lines = len(lines)
    s = (start_line - 1) if start_line is not None else 0
    e = end_line if end_line is not None else total_lines
    # M2 的越界检查在这里（见下）
    content = "\n".join(lines[s:e])
```

### M2：start_line 越界返回明确错误（治 R2）

**在切片前加越界检查**（接 M1）：

```python
if start_line is not None or end_line is not None:
    lines = content.splitlines()
    total_lines = len(lines)
    # M2：start_line 越界 fail-loud，不静默返回空
    if start_line is not None and start_line > total_lines:
        return (
            f"Error: start_line={start_line} 超过文件末尾（文件共 {total_lines} 行）。"
            f"该文件较小，无需分段读——直接不传 start_line/end_line 读全文，"
            f"或传 start_line ≤ {total_lines}。"
        )
    s = (start_line - 1) if start_line is not None else 0
    e = end_line if end_line is not None else total_lines
    content = "\n".join(lines[s:e])
```

> **为什么 fail-loud 而非静默空**：LLM 拿到空内容会「换个 start_line 再试」陷入死循环（dogfood 实证）。明确的「超过末尾共 N 行」让 LLM 立刻知道文件只有 N 行、停止猜测。守 memory `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`（响亮故障 > 哑故障）。

### M3：code-executor 不 read 大 plan（治 R3，prompt 减少触发）

**改 `code_executor.py` system_prompt**，在执行流程段明示（用正面指令，守「deepseek 正面提示」原则）：

```
- plan_metrics.json 是 run_metric_plan 工具的【输入施工单】，由它内部 json.load 完整读取并执行——
  你【直接调 run_metric_plan】即可，它会自己读 plan、跑全部 compute 脚本、算统计、落盘 handoff。
- plan 文件可能很大（百 KB 量级），read_file 会头部截断（只显示开头）——
  【不要 read_file 去读 plan 内容确认 statistics/metrics】，run_metric_plan 内部会处理。
- 你唯一需要确认的是 plan 文件【存在】（ls 或直接调 run_metric_plan，它缺 plan 会 fail-loud 报错）。
```

> M3 是人因面（减少 code-executor 触发 read 大 plan）；M1/M2 是结构面（read_file 本身不再骗任何 subagent）。即使 M3 没完全压住、code-executor 仍 read plan，M1/M2 让它至少不陷死循环（越界立刻收到「文件共 N 行」）。

---

## 三、TDD（红→绿，强制）

测试文件：扩 `backend/tests/test_sandbox_tools.py`（或新 `test_read_file_line_range.py`）。

### T1（R1 红→绿）：只传 start_line 返回「从 start_line 到末尾」
```python
def test_t1_start_line_only_reads_to_end():
    # 文件 100 行，read_file(start_line=90) → 返回第 90-100 行
    # 红：当前只传 start_line 整段忽略，返回全文件头部
    # 绿：M1 后返回第 90-100 行
```

### T2（R1）：只传 end_line 返回「从头到 end_line」
```python
def test_t2_end_line_only_reads_from_start():
    # read_file(end_line=10) → 返回第 1-10 行
```

### T3（R2 红→绿）：start_line 越界返回明确错误
```python
def test_t3_start_line_beyond_eof_errors():
    # 文件 100 行，read_file(start_line=2895) → "Error: start_line=2895 超过文件末尾（文件共 100 行）"
    # 红：当前静默返回空串（LLM 死循环根源）
    # 绿：M2 后 fail-loud
```

### T4（回归）：两个都传仍正常
```python
def test_t4_both_lines_still_work():
    # read_file(start_line=10, end_line=20) → 第 10-20 行（行为不变）
```

### T5（R3 prompt）：code-executor system_prompt 含「不 read 大 plan」指引
```python
def test_t5_code_executor_prompt_warns_against_reading_plan():
    # assert "不要 read_file" / "直接调 run_metric_plan" 在 code_executor system_prompt
```

### T6（导入环）：改 sandbox/tools.py + code_executor.py 后裸导入两入口
```bash
PYTHONPATH=. python -c "import app.gateway"
PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
```

---

## 四、风险与注意事项

1. **read_file 是全 subagent + lead 共用工具**：M1/M2 改 read_file 切片逻辑，影响面广。grep 所有传 start_line/end_line 的调用方（应该都是 LLM 动态传，无硬编码调用），跑全量 `make test` 确认无回归。
2. **`(empty)` 短路位置**：[L1724](../../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py#L1724) `if not content: return "(empty)"` 在切片**前**——空文件仍返回 `(empty)`。M1/M2 的切片在其后，逻辑不冲突。但注意：切片后内容为空（如合法 range 但该段全空行）不该报 M2 错误，只有 **start_line 越界**才报。
3. **行号 1-indexed 边界**：`start_line=total_lines`（最后一行）合法，`start_line=total_lines+1` 越界。`>` 不是 `>=`。T3 用 `start_line > total_lines`。
4. **M3 是 prompt 不是门**：code-executor 仍可能违反（LLM 自由）。但 M1/M2 兜底——违反也不死循环。**不要为 R3 上结构门**（read plan 不是非法操作，只是低效），守「结构已对、只缺指引→改 prompt」（CLAUDE.md 三大病理自检反向）。
5. **async 路径同步**：`_read_file_tool_async`（[L1754](../../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py#L1754)）转发 `read_file_tool.func`，M1/M2 改 func 体即两路径生效，无需单独改 async。

---

## 五、实施步骤

1. TDD 红：先写 T1+T3，确认当前红（T1 只传 start_line 返回头部；T3 越界返回空）。
2. M1+M2：改 read_file 切片逻辑，跑 T1-T4 绿。
3. M3：改 code_executor system_prompt，跑 T5 绿 + `test_code_executor_*` 不破。
4. 全量 `make test` + 裸导入两入口（T6）+ grep read_file 调用方。
5. dogfood 验收：重跑 EPM dogfood，核 code-executor **不再 read 大 plan 死循环**（trace 应直接 ls→run_metric_plan）。

---

## 六、与其他 spec 的关系

- 与 spec A（seal-once）/ spec B（render decouple）正交——本 spec 改 read_file + code-executor，不碰 seal/run_chart_plan。
- 与 spec D（think 语言）正交但同源（同一 dogfood thread `0e72d605` 暴露）——D 改 subagent system_prompt 语言段，本 spec 改 code-executor 执行流程段，文件 `code_executor.py` 有重叠但改不同段，**建议同 PR 或顺序合并避免 system_prompt 冲突**。

---

## 七、待查（不在本 spec 范围）

- **run_metric_plan 第一次中断根因**：dogfood trace「Tool call was interrupted and did not return a result」。可能：subagent max_turns/超时、SSE 断、工具内异常被吞。需独立复现 subagent（守 `feedback_diagnose_subagent_behavior_replay_subagent_not_dump_lead`）查 SubagentExecutor 日志。**单列 problem doc，不混进本 spec。**

---

## 八、milestone 建议

归入「harness 工具鲁棒性」track。checkpoint：「EPM dogfood（thread 0e72d605）端到端成功但 code-executor 阶段 read_file 找 plan statistics 死循环 20+ 轮。取证坐实 read_file 两个越界静默 bug（只传 start_line 被忽略返回头部 / start_line 越界静默返回空）+ code-executor 策略错（read 给工具吃的大 plan）。修法=M1/M2 工程修 read_file 切片（fail-loud 越界，全 subagent 受益）+ M3 code-executor prompt（不 read 大 plan）。**回应用户「纯 prompt 够不够」：不够，必须 prompt+工程双管**——prompt 只让 code-executor 绕开，工程修才消除 bug 本身。」
