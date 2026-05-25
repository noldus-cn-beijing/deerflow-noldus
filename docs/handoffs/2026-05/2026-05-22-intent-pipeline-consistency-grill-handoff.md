# 2026-05-22 IntentPipelineConsistencyProvider 设计 grill 起点交接

> **状态**：未开始 grill。本文是 Q5-b 待 grill 子树的起点文档，下一会话进入实际 grill。
> **前置依赖**：先读 [2026-05-22-chart-maker-grill-corrected-handoff.md](2026-05-22-chart-maker-grill-corrected-handoff.md) 理解 PR-3 整体目标。

## 起点

用户观察到的端到端不稳定现象：

> "现在看端到端也能实现。但是可能流程不是很固定鲁棒，比如有时候停在了画图，有的时候没画图直接出分析结果，有的时候画图了"

前轮 chart-maker grill 现场核实结论：

| 假设 | 仓库证据 |
|---|---|
| 缺类似 claude-code `/plan` 的统一规划框架 | ❌ `ethoinsight-planning` skill 已有 6 步规划，但**全仓库零引用**；lead prompt 第 254-258 + 818 行已有完整 INTENT 7 类状态机 + 派遣链 |
| 真因 | ✅ **lead 不遵守自己写下的 INTENT 状态机**——78ccb52b 实证：标 `[intent] E2E_MIN` 却派 chart-maker，违反第 259 行 `E2E_MIN → code-executor → ask(four-choice)` |

→ 不能靠加更多文档/skill 治；必须 **harness 级 guardrail 拦** —— 5/18 [[project_2026-05-18_lead_not_reading_handoff]] 已吃过亏，prompt-only fix 不够，需要 `HandoffEnforcementGuardrailProvider` 同模式的硬约束。

## 已有 guardrail 拓扑

| Provider / Middleware | 职责 | 挂在哪 |
|---|---|---|
| `IntentClassificationGuardrailProvider` | lead 第一个非 read_file tool 前必须输出 `[intent] <NAME>`，校验该行格式 + 必须是 7 类之一 | lead 中间件链 |
| `HandoffEnforcementGuardrailProvider` (5/18 项) | 强制 subagent 派遣前 lead 已 read 上一棒 handoff | lead 中间件链 |
| `ScriptInvocationOnlyProvider` | code-executor 的 bash 只能跑 `python -m ethoinsight.scripts.*` + ls/cp 等 | **只挂 code-executor**（不约束 lead；78ccb52b 故障的关键空隙） |
| `LoopDetectionMiddleware` | 防重复 tool call 死循环 | 默认全启用 |
| `GuardrailMiddleware` | pre-tool-call 授权决策（统一拦截器，dispatch 到各 provider） | 全局 |

**待新增**：`IntentPipelineConsistencyProvider` —— **校验 lead 派遣的 subagent 与当前 INTENT + 已派遣历史一致**，挂在 lead 中间件链。

## INTENT 状态机规约（来自 lead_prompt.py:254-280 + lead-interaction:14-23）

```
E2E_FULL_ASKVIZ → code-executor → data-analyst → ask(viz?)
                  → [yes] chart-maker → ask(report?) → [yes] report-writer / [no] 终止
                  → [no] ask(report?) → [yes] report-writer / [no] 终止

E2E_FULL → code-executor → data-analyst → chart-maker → ask(report?)
           → [yes] report-writer / [no] 终止

E2E_MIN → code-executor → ask(four-choice)
          → [choose A] data-analyst → ... （回到 ASKVIZ 流分支）
          → [choose B] chart-maker → ...
          → [choose C] report-writer
          → [choose D] 终止

CHART → chart-maker（单派，已有 handoff）
REPORT → report-writer（单派，已有 handoff）
QA_FACT → knowledge-assistant（授权 handoff 占位符）
QA_KNOWLEDGE → knowledge-assistant（不授权 handoff）
CLARIFY → ask_clarification（不派 subagent）
```

## 必须 grill 的设计问题

### 1. 状态如何累积

guardrail 是 stateless provider（pre_tool_call hook），但 INTENT 状态机需要"已派遣序列"。两种方案：

- **(A)** 从 messages history 反向扫——按 ToolMessage 顺序找出过往 `task(subagent_type=X)` 的成功调用，组装当前序列
- **(B)** lead 状态字段——`thread_state` 加 `intent_pipeline_history: list[str]`，每次 task() 成功后 middleware 追加；guardrail 读该字段
- **(C)** 引入 INTENT 状态机的"当前 step"概念——`thread_state` 加 `intent_current_step: Literal["code-executor", "data-analyst", "ask-viz", ...]`

**关键 grill 点**：(A) 实现最简但每次 turn 都要重扫；(B) 写入侧增加耦合；(C) 状态机最直接但需要明确"ask_clarification 中断后 step 如何 resume"。

### 2. ask_clarification 中断的处理

INTENT 状态机里有 `ask(viz?)` / `ask(report?)` / `ask(four-choice)` 这些**用户决策点**。用户答完后 lead 进入新 turn，下一个 `task()` 派遣时：

- guardrail 怎么知道是从「ask(viz?) yes 分支」继续，而不是「ask(viz?) no 分支」？
- 是看用户答复字面（"是"/"不"/"画"/"不画"），还是看 lead 自己有没有再次输出 `[intent]` 行？
- 中间用户给追加要求（"再画一张趋势图"），是新 INTENT（CHART）还是继续原 INTENT？

**关键 grill 点**：状态机的 transition 需要明确"由谁触发 + 触发条件"。

### 3. 拒绝行为：硬拒 vs 软提示

guardrail 拦到不合法派遣时：

- **硬拒**：抛 `GuardrailError`，lead 收到拒绝消息后必须重新决策
- **软提示**：transform 派遣（如把 `task(chart-maker)` 在 E2E_MIN 下转成 `ask_clarification`）
- **混合**：第一次软提示+警告，第二次硬拒

5/18 `HandoffEnforcementGuardrailProvider` 是哪种模式？应该参考其设计。

### 4. 状态机字典的 single source of truth

状态机当前在 3 处定义：
- `lead_prompt.py:254-280`（自然语言）
- `lead_prompt.py:818`（一行压缩版）
- `ethoinsight-lead-interaction/SKILL.md:14-23`（7 分类表）

`IntentPipelineConsistencyProvider` 要校验状态机就**需要一份机器可读**的事实来源。两种方案：
- **(A)** 把状态机定义为 Python 字典（如 `INTENT_PIPELINE_GRAPH: dict[INTENT, list[Step]]`）放在 provider 文件里
- **(B)** 把状态机定义为 YAML 放在 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/intent_pipeline.yaml`，prompt 渲染 + guardrail 都读这份 yaml
- **(C)** 保持 prompt 是 source，guardrail 解析 prompt 字符串（**不推荐**，违反 single source of truth）

**(B) 与 catalog yaml 模式一致**（PR-1 的 B4 决策同模式）。

### 5. 与 IntentClassificationGuardrailProvider 的协同

- IntentClassification 校验 `[intent] X` 行格式 + 是 7 类之一 → 给 X
- IntentPipelineConsistency 校验"既定 X 下能派 Y" → 用 X 检查 Y
- 两者**串联还是合并**？合并是一个 provider 两个 hook（pre_intent_decl + pre_task_dispatch）？

### 6. 失败降级路径

`ethoinsight-planning/references/failure-recovery.md` 列了多种降级（PR-3 要搬到 lead-interaction）：
- code-executor 失败 → 按失败类型分支
- data-analyst 超时/空返回 → lead 自己读 handoff_code_executor.json 汇总
- report-writer 超时/空返回 → 用 data-analyst 摘要作输出
- chart-maker 失败 → 跳过图表，继续 report

这些**是 INTENT 状态机的"异常 transition"还是"状态机外的应急逻辑"**？guardrail 怎么区分"派遣不合法"和"前一棒失败后跳过派下一棒"？

## 必须现场核实的代码点

| 验证 | 命令 |
|---|---|
| IntentClassificationGuardrailProvider 实现 | `grep -rn "class IntentClassification" packages/agent/backend/packages/harness/deerflow/guardrails/` |
| HandoffEnforcementGuardrailProvider 实现 + 拒绝模式 | `grep -rn "class HandoffEnforcement" packages/agent/backend/packages/harness/deerflow/guardrails/` |
| GuardrailProvider 协议 | `grep -rn "class GuardrailProvider" packages/agent/backend/packages/harness/deerflow/guardrails/` |
| lead 中间件链顺序 | `grep -n "middleware" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` |
| thread_state 字段（看能不能加 intent_pipeline_history） | `grep -n "class ThreadDataState\|@dataclass" packages/agent/backend/packages/harness/deerflow/runtime/thread_state.py` |
| 实证 thread 78ccb52b 解码后的"违规派遣"链 | 见 [2026-05-22-chart-maker-grill-corrected-handoff.md](2026-05-22-chart-maker-grill-corrected-handoff.md) 末段，已有 ormsgpack 解码方法 |

## 与 PR-1/PR-2 的依赖关系

**PR-3 这部分独立**。但建议：
- PR-1（catalog schema）和 PR-2（chart-maker workflow）先合，验证 chart-maker 一支没问题后
- PR-3 实施时把 78ccb52b 那种故障重跑 dogfood，看 IntentPipelineConsistencyProvider 是否能正确拦截

## Q5-b grill 入口问题

> **Q-G5b-1（最关键）**：状态如何累积？
>
> - **(A)** stateless guardrail，每次扫 messages history 找过往 `task()` ToolMessage
> - **(B)** thread_state 加 `intent_pipeline_history: list[str]`，task() 后 middleware 追加
> - **(C)** thread_state 加 `intent_current_step: str`，状态机驱动 transition
>
> 你怎么选？(A) 实现最快但每 turn 重扫；(B) 写入侧耦合；(C) 状态机干净但需要明确 transition 触发条件。
>
> **Q-G5b-2**：拒绝行为是硬拒还是软提示？参考 5/18 `HandoffEnforcementGuardrailProvider` 的模式。
>
> **Q-G5b-3**：状态机字典放在哪？(A) provider Python 字典 (B) intent_pipeline.yaml + prompt 与 guardrail 共读 (C) 解析 prompt 字符串。

---

## 教训沉淀

- 任何"流程不固定"先看 prompt + skill 是否已有规约，**不要假设需要新框架文档**
- 文档（skill / prompt 段）只能让 LLM 知道"应该做什么"，**不能强制做**——只有 harness 级 guardrail 能强制
- single source of truth：状态机定义只能有一份机器可读源；prompt 渲染 / guardrail 校验 / 文档说明都从这一份生成
- 类比 5/18 HandoffEnforcement、5/15 lead-bash-removal、本次 ScriptInvocationOnly——**所有"让 LLM 不能跨界"的约束都必须 harness 实现**
