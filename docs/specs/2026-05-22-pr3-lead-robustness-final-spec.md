# PR-3 最终 spec —— INTENT 鲁棒性 + lead 边界硬约束 + planning skill 处置

> **生成于**：2026-05-22
> **依赖**：PR-1（catalog schema 升级）、PR-2（chart-maker workflow）合入后实施 dogfood 验证
> **范围**：lead 行为硬约束 + planning skill 删除 + 修正版 handoff 验证 thread
> **来源**：[2026-05-22-chart-maker-grill-corrected-handoff.md](../handoffs/2026-05/2026-05-22-chart-maker-grill-corrected-handoff.md) + [2026-05-22-intent-pipeline-consistency-grill-handoff.md](../handoffs/2026-05/2026-05-22-intent-pipeline-consistency-grill-handoff.md) Q5-b grill 5 轮的全部决策

## 0. 现实校准（grill fact-check 沉淀）

Q5-b grill 期间对 4 个真实 thread（843fe2b8 / 189e7840 / f3fbce44 / 7456611e）做了 ormsgpack 解码 + bytes 扫描，结论：

| 假设 | 证据 | 结论 |
|---|---|---|
| 「lead 不遵守自己的 INTENT 状态机」 | 4/4 thread 全标 `[intent] E2E_FULL_ASKVIZ`，用户原话全是「帮我分析一下大鼠强迫游泳的实验数据」（含 `分析` vague trigger，无 viz trigger），分类正确 | ❌ **没有派错人的证据**，前轮 grill 假设的故障类型在新样本中零复现 |
| 「LLM 跑过头跳过 ask(viz?)」 | 4/4 thread 的 INTENT=E2E_FULL_ASKVIZ 都跳过 ask(viz?) 直接派 chart-maker；`ask_clarification` tool_call 真实调用 0 次 | ✅ **真问题** —— 这是 LLM 系统性指令遵从度缺陷 |
| 「lead 越界 bash + 派幽灵 subagent」 | 189e7840 反复试 `task(bash)` 135 次 + `task(general-purpose)` 113 次；f3fbce44 试 general-purpose 81 次；deny 消息明确列出 Available 列表，**LLM thinking 内明确"看懂"了 ("there's no general-purpose subagent listed")，但仍反复重试** | ✅ **真问题** —— deny 信息完备但 LLM 知错不改；该用 schema-level 约束 |
| 「summarization 把 ASKVIZ 规则压掉了」 | 843fe2b8 token 未到 60k 阈值，整 thread 没归档；ASKVIZ 反问模板每次推理新鲜注入 system prompt | ❌ **不是归档问题** |
| 「IntentClassificationGuardrailProvider 全程工作」 | 843fe2b8: 231 次假阳 reject + 流水线照常跑完；f3fbce44: 343 次假阳 reject + 仍跑完 | ⚠️ **有 bug 但不阻塞**（fail-soft 模式让派遣继续） |

## 1. 核心原则（PR-3 全程遵循）

| 原则 | 来源 |
|---|---|
| **Schema 硬约束优于 runtime deny** | Q5-b Round 5 验证：信息完备的 deny 不足以让 LLM 自我纠正（189e7840 反复试 general-purpose） |
| **deny 消息必须含「请改用 X 因为 Y」** | Q5-b Round 5 对比：`task_tool` 裸 Available 列表无效 vs `IntentClassification` "请输出 [intent] X 行" 一般有效 |
| **门控状态走 experiment-context.json gate_completed**（deerflow 既有范式） | GateEnforcementMiddleware + set_experiment_paradigm_tool 已建立的模式，跨 turn 持久、归档不丢、单调累加 |
| **affordance 移除优于 deny 提醒** | lead 完全 disallow bash 比"加日志警告"更鲁棒 |
| **single source of truth** | INTENT 状态机定义不允许同时存在于 prompt 与文档双份 |

## 2. 改动清单

### 2.1 task_tool subagent_type schema 强类型化

**目标**：把"派幽灵 subagent"在 LLM 调用 schema 层禁掉，不依赖 runtime deny 拦截。

**文件**：`packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py`

**改动**：
- `task_tool` 函数签名的 `subagent_type` 参数从 `str` 改为运行时构造的 `Literal[...]`
- 通过 LangChain `@tool` 的 args schema 注入 `BUILTIN_SUBAGENTS.keys()` 枚举值
- 保留现有 `if config is None` 的 deny 分支作为安全网（防止 schema 绕过）

**关键约束**：
- 不能硬编码 5 个 builtin 名字——`get_available_subagent_names()` 是动态的（含 config 启用/禁用）
- LangChain Tool 的 args_schema 用 Pydantic Field，可用 `enum=` 或 `Literal` 注解；需要确认 args_schema 支持运行时动态枚举

**实施方法**（推荐）：
```python
# 在 task_tool 模块加载时动态生成 Literal type
from typing import Literal
def _make_subagent_literal():
    names = tuple(get_available_subagent_names())
    return Literal[names]  # type: ignore

# tool 注册时把 args_schema 中 subagent_type 改为该 Literal
```

替代方法：用 LangChain v1 的 `@tool` + Pydantic Field(default=..., examples=[...], json_schema_extra={"enum": [...]}) 透传 JSON Schema enum 字段给 LLM。

**验证**：dogfood 复跑 189e7840 / f3fbce44 用例，确认 LLM 调 task 时 schema 强制选 5 个之一。

### 2.2 lead disallowed_tools=["bash"]

**目标**：lead 完全不能调 bash 工具（5/15 lead-bash-removal 真正硬约束）。

**文件**：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py`（lead config 构建处）

**改动**：lead 的 `agent_config` 或 `tools` 构建处把 `bash` 工具排除掉。

**注意**：lead **仍然可以**通过 `task(subagent_type="code-executor", prompt=...)` 间接调 bash —— 这是设计意图（code-executor 有 ScriptInvocationOnlyProvider 限制）。

**边界**：用户明确确认 **完全 disallow，遇到合法需要再放开**。4/4 thread 主流水线没用过 lead bash，目前没有合法用例。

### 2.3 新 guardrail：IntentPostStepAskGateProvider

**目标**：当 INTENT=E2E_FULL_ASKVIZ 且 data-analyst 已完成但 ask(viz?) 未完成时，拦 `task(chart-maker)` + 注入强引导 deny。

**新建文件**：`packages/agent/backend/packages/harness/deerflow/guardrails/intent_post_step_ask_gate_provider.py`

**实施细节**：

```python
class IntentPostStepAskGateProvider:
    name = "intent_post_step_ask_gate"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # 只拦 task(chart-maker)
        if request.tool_name != "task":
            return GuardrailDecision(allow=True)
        if request.tool_input.get("subagent_type") != "chart-maker":
            return GuardrailDecision(allow=True)

        # 从 messages history 提取 INTENT（复用 IntentClassification 的 ContextVar 范式）
        messages = _lead_messages.get()
        intent = _extract_latest_intent(messages)
        if intent != "E2E_FULL_ASKVIZ":
            return GuardrailDecision(allow=True)

        # 读 experiment-context.json 看 gate3 是否已 ack
        workspace = _resolve_workspace_from_request(request)
        if workspace is None:
            return GuardrailDecision(allow=True)  # fail-open: 无 workspace 不拦
        ctx = read_context(workspace)
        if ctx is None:
            return GuardrailDecision(allow=True)
        gate_completed = ctx.get("gate_completed", [])
        if "gate3_viz_acknowledged" in gate_completed:
            return GuardrailDecision(allow=True)

        # 仅当 data-analyst 已完成时才拦（不然 lead 还在前流程，没必要拦）
        handoff_data_analyst = Path(workspace) / "handoff_data_analyst.json"
        if not handoff_data_analyst.exists():
            return GuardrailDecision(allow=True)

        return GuardrailDecision(
            allow=False,
            reasons=[GuardrailReason(
                code="ethoinsight.viz_choice_not_acknowledged",
                message=(
                    "INTENT=E2E_FULL_ASKVIZ 要求 data-analyst 完成后先反问用户是否需要图表。"
                    "请改调 `ask_clarification(question='📊 指标和解读已完成。需要我把结果可视化成图吗?', "
                    "options=['A. 是,把刚才的结论画成图(默认推荐,箱线图/轨迹图/时序图)', "
                    "'B. 不用,直接给我报告'])`；"
                    "用户回答后再调 `set_viz_choice(choice='yes'|'no')` 落盘 gate3，"
                    "之后才能派 chart-maker（或跳过直接派 report-writer）。"
                ),
            )],
            policy_id="intent_post_step_ask_gate",
        )
```

**关键设计点**：
- **deny 消息含完整后续指令**（请改调 ask_clarification + set_viz_choice + 后续派遣），LLM 看到就能照做
- **fail-open**（无 workspace / 无 context 时放行）——避免影响测试 / bootstrap
- **复用 IntentBridgeMiddleware ContextVar 范式**（注：该 middleware 有假阳 bug，PR-4 修；本 PR 接受同样的 bug 模式，因为本 provider 拦的是更窄场景，假阳影响小）

**挂载**：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` 中间件链——在 IntentBridge 之后、IntentClassification 之后追加：

```python
middlewares.append(GuardrailMiddleware(
    provider=IntentPostStepAskGateProvider(),
    fail_closed=guardrails_cfg.fail_closed,
))
```

### 2.4 新工具：set_viz_choice

**目标**：lead 在 ask(viz?) 用户答复后调用，把 viz 选择落盘到 experiment-context.json。

**新建工具**（在 `experiment_context.py` 中或新文件，与 `set_experiment_paradigm_tool` 同模式）：

```python
@tool("set_viz_choice", parse_docstring=True)
def set_viz_choice_tool(
    choice: Literal["yes", "no"],
    workspace_dir: str = "/mnt/user-data/workspace/",
    runtime: ToolRuntime[ContextT, ThreadState] = None,
) -> str:
    """Record the user's answer to the 'do you want a chart?' clarification.

    Use this AFTER ask_clarification has presented the viz question to the user
    and the user has replied. Writes gate3_viz_acknowledged + viz_choice to
    experiment-context.json.

    Args:
        choice: "yes" if user wants charts; "no" otherwise.
        workspace_dir: Workspace directory. Default: "/mnt/user-data/workspace/".
    """
    actual_workspace = _resolve_workspace(runtime, workspace_dir)
    existing = read_context(actual_workspace)
    if existing is None:
        return json.dumps({"status": "error", "message": "experiment-context.json missing; call set_experiment_paradigm first."}, ensure_ascii=False)
    gate_completed = existing.get("gate_completed", [])
    if "gate3_viz_acknowledged" not in gate_completed:
        gate_completed.append("gate3_viz_acknowledged")
    data = {**existing, "gate_completed": gate_completed, "viz_choice": choice, "viz_acknowledged_at": datetime.now(UTC).isoformat()}
    path = Path(actual_workspace) / "experiment-context.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return json.dumps({"status": "ok", "viz_choice": choice, "gate_completed": gate_completed}, ensure_ascii=False)
```

**注册**：在 `tools/builtins/__init__.py` 把 `set_viz_choice_tool` 加进 BUILTIN_TOOLS。

**lead prompt 改动**（最小化）：

在 lead prompt §E2E_FULL_ASKVIZ 反问模板段的末尾追加一句：

```
用户回答后,**必须立即**调 `set_viz_choice(choice='yes' | 'no')` 落盘 gate3,然后再决定派 chart-maker 或跳到 ask(report?)。否则后续 task(chart-maker) 会被 IntentPostStepAskGateProvider 拦截。
```

### 2.5 IntentPostStepAskGateProvider 假阳 bug —— 仅诊断不修

**目标**：用户确认 M2 方案——本 PR 加诊断打点 + 单测 reproduce，**修留给 PR-4**。

**改动**：
- `intent_classification_provider.py` IntentBridgeMiddleware：在 `_extract_and_set_messages` 调用前后加 `logger.debug` 打点（messages 数量、最后一条 AIMessage 是否含 `[intent]`）
- 新建单测 `tests/test_intent_classification_false_positive.py`：用 mock messages 列表（含 `[intent] E2E_FULL_ASKVIZ` 的 AIMessage 在 messages[-2]）reproduce 假阳 reject。如果单测 reproduce 不出来，则真因可能在 ContextVar 跨 async/sync 边界或 middleware 顺序——单测在该情况下应至少**捕获 messages 的实际内容**，留给 PR-4 调研

**不在本 PR 修复**理由：bug 不阻塞流水线（fail-soft 模式让派遣继续）；574 次假阳是日志噪音 + 性能影响（每次重 evaluate），优先级低于新 ask gate 落地。

### 2.6 删除 ethoinsight-planning skill + 内容并入 lead-interaction

**目标**：single source of truth；INTENT 状态机只能有 lead prompt 这一份事实来源。

**改动**：

| 文件 | 操作 |
|---|---|
| `packages/agent/skills/custom/ethoinsight-planning/` | **整目录删除** |
| **新建** `packages/agent/skills/custom/ethoinsight-lead-interaction/references/quality-gates.md` | 内容**搬自** `ethoinsight-planning/references/quality-gates.md` |
| **新建** `packages/agent/skills/custom/ethoinsight-lead-interaction/references/failure-recovery.md` | 内容**搬自** `ethoinsight-planning/references/failure-recovery.md` |
| `packages/agent/extensions_config.json` | 删 `"ethoinsight-planning": { "enabled": true }` 一行 |
| `packages/agent/skills/custom/ethoinsight-lead-interaction/SKILL.md` | 在末尾 references 列表加新两项指引；如有"Lead 不读本 skill"等遗留段落删除 |

**注意**：planning skill 的 6 步 workflow / 单行用户契约 / 派遣链模板等内容**不搬**——因为已被 lead prompt §INTENT 状态机覆盖且更完整。只搬两个真正独有的 references。

## 3. 改动文件总数

| # | 类型 | 文件 |
|---|---|---|
| 1 | 改 | `tools/builtins/task_tool.py`（subagent_type Literal）|
| 2 | 改 | `agents/lead_agent/agent.py`（lead disallowed_tools 加 bash + 挂 IntentPostStepAskGateProvider 中间件）|
| 3 | 新 | `guardrails/intent_post_step_ask_gate_provider.py` |
| 4 | 改 | `agents/middlewares/experiment_context.py`（加 set_viz_choice_tool）|
| 5 | 改 | `tools/builtins/__init__.py`（注册新 tool）|
| 6 | 改 | `agents/lead_agent/prompt.py`（追加 set_viz_choice 调用指引）|
| 7 | 改 | `guardrails/intent_classification_provider.py`（加诊断打点，**不修 bug**）|
| 8 | 新 | `tests/test_intent_classification_false_positive.py`（reproduce 单测）|
| 9 | 新 | `tests/test_intent_post_step_ask_gate.py`（新 provider 单测）|
| 10 | 删 | `packages/agent/skills/custom/ethoinsight-planning/`（整目录） |
| 11 | 新 | `skills/custom/ethoinsight-lead-interaction/references/quality-gates.md` |
| 12 | 新 | `skills/custom/ethoinsight-lead-interaction/references/failure-recovery.md` |
| 13 | 改 | `packages/agent/extensions_config.json` |
| 14 | 改 | `skills/custom/ethoinsight-lead-interaction/SKILL.md` |
| 15 | 新 | `tests/test_set_viz_choice.py`（新 tool 单测）|
| 16 | 新 | `tests/test_lead_disallowed_bash.py`（lead 不能调 bash 单测）|

## 4. 实施顺序（同 PR 内推荐顺序）

1. **删 ethoinsight-planning**（独立、零依赖）—— 文件搬迁 + extensions_config 清理
2. **task_tool Literal**（schema 强约束基础）—— 验证 dogfood 不能派 general-purpose
3. **lead disallowed_tools 加 bash**（affordance 移除）—— 验证 lead 无 bash 工具
4. **set_viz_choice 工具**（先建工具）—— 单测落盘 gate3
5. **IntentPostStepAskGateProvider** + 挂中间件 —— 完整 dogfood 验证
6. **lead prompt 追加 set_viz_choice 调用指引**
7. **IntentClassification 诊断打点 + 单测**（最后做）

## 5. dogfood 验证用例

| Thread | 期望复跑后表现 |
|---|---|
| **843fe2b8**（FST 跑通）| 期望：`E2E_FULL_ASKVIZ` 在 data-analyst 后**真的 ask** + 用户答 yes → set_viz_choice → chart-maker → ask(report?) → report-writer。**与原 thread 区别**：原 thread 跳过 ask，新流程必须 ask |
| **189e7840**（lead 越界 bash + general-purpose）| 期望：(a) lead 无 bash 工具，不能调；(b) task subagent_type Literal 强约束，不能写 general-purpose；(c) chart-maker 完成后 lead 不需要善后（PR-1 PlanChart.output 已是 outputs/） |
| **f3fbce44**（同 189e7840 模式）| 同上 |
| **7456611e**（流水线干净但跳过 ask）| 期望：在 data-analyst 后被 IntentPostStepAskGateProvider 拦截 → 强引导 deny 指示调 ask_clarification + set_viz_choice |

## 6. 不在本 PR 范围

| 项 | 留到 |
|---|---|
| 修 IntentClassificationGuardrailProvider 假阳 bug | PR-4（diag-only 本 PR 做）|
| claude-code 风格 `present_plan` 工具（派遣前预览）| v0.2+ |
| `ethoinsight-charts` / `ethoinsight-chart-maker` skill 渐进披露重构 | PR-2 |
| ChartEntry yaml accepts_paradigm 等字段 | PR-1 |

## 7. 风险与回滚

| 风险 | 缓解 |
|---|---|
| task_tool Literal schema 实现需要 LangChain @tool 支持运行时 enum | 实施前 spike 验证；若不支持，回退到 args_schema Pydantic + Field json_schema_extra |
| lead 完全 disallow bash 可能影响未发现的合法用例 | 用户已确认"遇到合法需要再放开"；保留 prompt 指引说明用 task(code-executor) 间接 |
| IntentPostStepAskGateProvider 假阳（intent 不在 messages 中但确实是 E2E_FULL_ASKVIZ）| 复用 IntentBridgeMiddleware 已知 bug，但本 provider 只拦 chart-maker 一个 subagent + 仅 INTENT=ASKVIZ + data-analyst 已完成的窄场景，假阳影响小 |
| 删除 ethoinsight-planning skill 可能误删未发现的内容引用 | 删除前 `grep -rn ethoinsight-planning` 全量扫描，确认无未知引用 |
| 单 PR 改 16 个文件 review 量大 | 实施时建议按 §4 顺序分 commit，每个 commit 单测保护 |

## 8. 教训沉淀

| 教训 | 来源 |
|---|---|
| **信息完备的 deny 不等于有效 deny** | Q5-b Round 5：189e7840 lead thinking 明确写"there's no general-purpose subagent listed"但仍反复试 |
| **schema 硬约束优于 runtime deny + LLM 自纠** | 同上 |
| **deny 消息必须含「请改用 X 因为 Y」+ 后续指令** | 对比 IntentClassification（有效）vs task_tool（无效）|
| **门控状态用 workspace 文件 gate_completed 持久化最稳** | deerflow 现成 experiment-context.json 范式 |
| **grill 必须现场扫真实 thread 而非仅基于前轮 handoff** | Q5-b 4 轮 grill 推翻多次基于假设的设计 |
| **affordance 移除优于"加告警的 deny"** | lead disallowed_tools 比"prompt 写不要用 bash"硬约束多了 |

---

## 附录 A：scheme 强类型化 spike 笔记

LangChain `@tool` 装饰器对运行时动态 Literal 的支持需要确认。三种实施备选：

1. **运行时构造 Literal**（推荐）：
```python
from typing import Literal
_AVAILABLE_SUBAGENTS_LITERAL = Literal[tuple(get_available_subagent_names())]
@tool("task", parse_docstring=True)
def task_tool(subagent_type: _AVAILABLE_SUBAGENTS_LITERAL, ...) -> str: ...
```

风险：模块加载时机和 builtin 注册时机的耦合；测试 mock 时可能需要重置。

2. **args_schema 显式 Pydantic 模型**：
```python
class TaskArgs(BaseModel):
    subagent_type: str = Field(..., description="...", json_schema_extra={"enum": [...]})
@tool(args_schema=TaskArgs) def task_tool(...): ...
```

兼容性高，但 LLM 可能不严格遵守 JSON Schema enum（取决于模型 + binding）。

3. **维持现状 + 强化 deny 消息**：
```
"Unknown subagent type 'general-purpose'. Valid options are: chart-maker (图表生成), code-executor (脚本执行) ... 请改调 task(subagent_type='chart-maker', ...)."
```

降级方案：如 1/2 实施成本过高或 LangChain 版本不支持，至少把 deny 消息从裸列表升级为"请改调 X"模式。

实施前 spike `grep -rn args_schema packages/agent/backend/packages/harness/deerflow/tools/builtins/` 看其他 builtin tool 怎么用 args_schema。
