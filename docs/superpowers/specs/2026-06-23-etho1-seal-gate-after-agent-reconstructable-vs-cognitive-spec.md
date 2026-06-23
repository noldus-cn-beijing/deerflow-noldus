# Spec：ETHO-1 report-writer / data-analyst 偶发漏调 seal —— 升级 SealGate 分两类堵漏，生产级工程化

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-23
> 性质：🔴 高 · harness 控制安全（D7）加固。subagent 完成推理后未调 `seal_<name>_handoff`，报 `terminated without emitting ... forgot to call the seal tool`，Lead 自动重试一整轮才成功（间歇性，~2-4min 代价 + 用户可见 `Task failed` 中间态）。
> **方向（用户拍板）**：**升级现有 `SealGateMiddleware`，不新建 middleware**。根因「LLM 决定何时停」prompt 约束不了，但 **infra 结构能大幅堵死**——分两类处理（产物可重建 vs 认知产物）。
> 关联：
> - 来源：`~/ETHOINSIGHT_BUGS.md` ETHO-1；根因对照 `docs/problems/2026-06-22-ethoinsight-bugs-e2e-root-cause.md`
> - **生产实证**（2026-06-23 dump 4 prod thread）：memory `feedback_etho1_prod_trace_seal_miss_both_dataanalyst_and_reportwriter`——`terminated without emitting` 中 **3/4 thread**（高频）；漏调跨 **data-analyst（e6ea7946 #38）+ report-writer（3a41e483 #49）两类**；重试 7 步即成功（**纯收尾漏 call，非 thinking 超时**）；runs 全 `success`（lead 重试有效）。
> - **物理边界实证**：memory `feedback_etho1_after_agent_cannot_jump_seal_gate_upgrade_boundary`——`after_agent` hook 不能 jump（5 个用它的 middleware 无一标 `can_jump_to`）；唯一能逼补调的窗口是 `after_model`。
> - **业界最佳实践**：memory `reference_harnessx_report_and_etho_spec_application`（HarnessX arXiv 2606.14249）——§6.6 Telecom 案例（累加 reminder 致崩）= 禁止再加提醒 prompt；「LLM 提议，确定性门定生死」= 用结构门不用 prompt；trace richness = 留 `sealed_by` 标记。
> - 受保护文件：`seal_gate_middleware.py` / `executor.py` 是 deerflow 定制中间件，sync 时 surgical 守护。
> - **import 环铁律**（CLAUDE.md）：改 executor/middleware 核心，验收必含裸导入 `app.gateway` + `make_lead_agent`。

---

## ‼️ 2026-06-23 重大修正：data-analyst 漏 seal 有第二个更严重的变体 = seal args 撞 max_tokens=4096 被腰斩

> 另一 agent 用 playwright 端到端 + `repro-data-analyst.py` **独立复现 data-analyst subagent**（看到 lead thread 看不到的内部 turn），逐字节锁定一个**比本 spec 原判断更严重的变体**。**data-analyst 治本的权威 spec = `docs/superpowers/specs/2026-06-23-data-analyst-seal-stepwise-fill-template-spec.md`（分步填模板）；完整取证见 `docs/problems/2026-06-23-data-analyst-seal-args-truncation-discussion.md`**（本节只摘要 + 划清边界）。memory `feedback_etho1_real_root_cause_is_seal_args_truncated_by_max_tokens_4096`。
>
> **两个结局是同一故障在不同数据规模下的表现（不矛盾，治本点正交）**：
> - **本 spec 原判断（prod e6ea7946，重试 7 步成功）对 prod 那次成立**：seal args **较短没撞墙**，是真·偶发收尾漏 call，补一轮即好——SealGate/after_agent 对这类有效。
> - **本地 28-subject EPM（连续 2 次 FAILED）是更严重变体**：5 条详尽 key_findings 把 seal args 撑到 char 1178/2142 **撞 `max_tokens=4096` 墙** → 腰斩成未终止 JSON → LangChain 判 `invalid_tool_calls` → 不执行 → SealGate 催回 → **再产再腰斩**（count=1→2 铁证）→ FAILED → lead 降级跳过判读 = **判读层功能性失效**。
>
> **对本 spec 的边界修正**：
> 1. **本 spec（SealGate after_agent 升级）只对 report-writer/chart-maker 负责**（产物可重建，§2.1 成立）。
> 2. **data-analyst 的治本不在本 spec** —— SealGate L1 催回 / after_agent 对 args 腰斩**全无效**（催回后再腰斩）。data-analyst 走 **forensics spec §四：首选提 `max_tokens` 4096→8192**（args 截断是确定性预算问题，可根治）。本 spec 下文 §2.0 / §2.2 关于 data-analyst 的内容**以 forensics spec 为准**，此处保留仅为说明边界。
> 3. 这是 **06-18 PR#161「换 flash」的镜像残留**：换 flash 没消根因（单 turn 4096 < 判读+seal 体积），只把载体从 thinking-truncation 换成 args-truncation。

---

## 〇、给实施 agent 的一句话

现状有 **4 层防线**（[`seal_gate_middleware.py`](../../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/seal_gate_middleware.py) L1 + [`executor.py:1400-1444`](../../../packages/agent/backend/packages/harness/deerflow/subagents/executor.py#L1400) L2/L3/L4）：

```
L1 SealGate.after_model: 检测「该 seal 但纯文本想退出」→ 提醒+jump_to=model 逼补调。
                          漏网口：_MAX_REMINDERS=2（规则5），提醒 2 次还不调 → return None 放行。
L2 executor._attempt_seal_resume: 补一轮 HumanMessage「现在请 seal」。
L3 _attempt_auto_seal_from_artifacts: 从 outputs/ 机械重建 handoff。仅 {report-writer, chart-maker, code-executor}；data-analyst 跳过（认知产物不可重建）。
L4 FAILED → lead 重试整个 subagent（生产实证：重试成功）。
```

**问题**：L1 的规则 5 放行口 + L3 触发太晚（埋在 L2 失败后）→ 用户看到 `Task failed` 中间态 + 多烧一轮。

**升级（两类分治）**：
1. **可重建产物**（report-writer / chart-maker / code-executor）：给 SealGate 加 `after_agent` hook，在 agent 真终止前调 `_attempt_auto_seal_from_artifacts` 兜底 → **结构性堵死，消除 `Task failed` 中间态**。
2. **认知产物**（data-analyst）：`after_agent` 既不能 jump 又不能伪造判读结论 → **不在 after_agent 造假**；只靠 L1 逼补调（已有）+ L2 补轮（已有）+ L4 lead 重试。**诚实承认：降触发率不归零。**
3. **L1 cap 不动**（保持 `_MAX_REMINDERS=2`）：提高 cap 的前提是 thinking 超时由另一条线解决（用户已让别的 agent 处理），否则 thinking 超时类会把更多 jump 烧在 turn 内超时上。本 spec **不碰 cap**，只补 after_agent 终止点兜底。

**禁止**：再加任何 seal 提醒 prompt 规则（HarnessX §6.6 Telecom：累加 reminder 会亚阈值耦合致崩；我们已改 4 次 prompt 打地鼠）。

---

## 〇·五、框架契合声明（DeerFlow-first，HarnessX 只借思想不借机制）

> ⚠️ 本仓库 infra 是 **DeerFlow（上层 LangGraph）**。本 spec 引用 HarnessX（arXiv 2606.14249）**仅借其工程化方法论思想**（change manifest / 确定性 gate 序列 / 三大病理自检 / trace richness / Telecom 累加 reminder 禁令），**不引入 HarnessX 的任何运行时机制**（processor 协议 / AEGIS 引擎 / substitution algebra / 9 维 taxonomy 对象）。见 memory `feedback_harnessx_ideas_on_deerflow_not_harnessx_mechanisms`。

| HarnessX 思想 | 本 spec 的 DeerFlow/LangGraph 落地 |
|---|---|
| D7 控制安全维度（防漏 seal） | 复用现有 `SealGateMiddleware`（LangChain `AgentMiddleware`），扩 `after_agent` hook |
| 「LLM 提议，确定性门定生死」 | after_model `jump_to='model'`（LangGraph 原生）逼补调 + 确定性 `_attempt_auto_seal_from_artifacts`（executor 现有纯函数） |
| Digester 结构化摘要 | 不涉及（ETHO-3 才用 sidecar） |
| trace richness | `sealed_by` 元字段写进**现有 handoff JSON**，无新机制 |
| change manifest / gate 序列 / 病理自检 | **仅用于组织本 spec 的文档结构与验收**，不落地成代码 |

**一句话**：所有运行时改动都是「在现有 DeerFlow middleware/executor 上加 hook、复用现有纯函数、写现有 JSON 字段」；HarnessX 只在「这份 spec 怎么写、怎么验」的元层面起作用。

---

## 一、根因（逐字节 + 生产实证）

### 1.1 现象与生产实证

`terminated without emitting 'handoff_<name>.json' ... forgot to call the seal tool`。Lead prompt（`prompt.py:258/272` 匹配 "terminated without emitting"）自动重试整个 subagent。

生产 4 thread（2026-06-23 dump）：
- **3a41e483 #48-51**：report-writer 漏 seal → #50 lead「按规则自动重试一次」→ #51 成功。
- **e6ea7946 #37-41**：data-analyst 漏 seal → #40 lead 重派 → #41 成功（7 步：read×6 → seal → 输出）。
- 4 thread runs 全 `success`；重试均 7 步内成功 = **纯收尾漏 call，非 thinking 超时**（超时会 turn 内炸、重试也救不了）。

### 1.2 根因：ReAct 终止时机由 LLM 决定（结构问题，非 prompt）

ReAct agent 在模型发出**无 tool_calls 的 AIMessage** 那一刻终止。「写一句纯文本 `分析完成`」自然结束循环——在 seal 工具被调之前。prompt 改「第 3 步必须 seal」无法强制 ReAct 本不具备的顺序（`seal_gate_middleware.py:7-15` docstring 已点明）。

### 1.3 现有 4 层防线的两个缺口

**缺口 A —— L1 规则 5 放行口**（[`seal_gate_middleware.py:159-161`](../../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/seal_gate_middleware.py#L159)）：
```python
# 5. Reminder cap reached → allow; existing seal-resume + 5.7 FAILED catch it
if self._get_reminder_count(runtime) >= _MAX_REMINDERS:   # _MAX_REMINDERS=2
    return None
```
提醒 2 次还不调 → 放行 → ReAct 纯文本退出 → 漏调发生。docstring 自称「L1 = structural 100%」与此放行口**矛盾**（memory `feedback_seal_missing_root_cause...` 已记此落差）。

**缺口 B —— L3 auto-seal 触发太晚、且只在 executor 层**（[`executor.py:1426`](../../../packages/agent/backend/packages/harness/deerflow/subagents/executor.py#L1426)）：auto-seal 埋在 L2 seal-resume 失败之后才跑。在它兜住之前，lead 已经收到 `Task failed`（L4 路径的报错先冒出来给用户看到）。`_attempt_auto_seal_from_artifacts`（[L301](../../../packages/agent/backend/packages/harness/deerflow/subagents/executor.py#L301)）docstring 明确：report-writer/chart-maker/code-executor 可机械重建，**data-analyst 判读结论无文件来源、永不 auto-seal**。

### 1.4 物理边界：after_agent 不能 jump

`after_agent` hook 在 agent「已决定终止」后跑——5 个用它的 middleware（token_budget/loop_detection/memory/todo/training_data）**无一标 `can_jump_to`**。∴ after_agent **不能 jump 回 model 逼补调**，只能做副作用（如调 auto-seal 重建文件）。这决定：
- 可重建产物：after_agent 可调 auto-seal（纯副作用，不需 jump）→ 有效。
- 认知产物：after_agent 既不能 jump 补调、又不能伪造 → **无法在此堵死**。

---

## 二、设计

### 2.1 修法 A：SealGate 加 after_agent 兜底（仅可重建产物）

给 `SealGateMiddleware` 加 `after_agent` / `aafter_agent` hook（**不标 `can_jump_to`**，纯副作用）。逻辑：

```python
def after_agent(self, state, runtime):
    # 仅 seal-requiring subagent；且仅「产物可机械重建」的类型。
    if self._subagent_name not in _RECONSTRUCTABLE:   # {report-writer, chart-maker}
        return None                                    # 认知产物/code-executor 不在此兜底
    messages = state.get("messages", []) if hasattr(state, "get") else []
    if _seal_in_history(messages, self._seal_tool):
        return None                                    # 已 seal，无需兜底
    # 漏调 + 产物可重建 → 在终止点调 auto-seal（复用 executor 已有纯函数）
    workspace = self._workspace_from_state(state)
    sealed = _attempt_auto_seal_from_artifacts(self._subagent_name, workspace)
    if sealed:
        logger.warning("SealGate.after_agent auto-sealed %s from artifacts", self._subagent_name)
    # 不能 jump；sealed 与否都返回 None（executor L3/L4 仍是最终裁决）
    return None
```

**关键设计决策**：
- `_RECONSTRUCTABLE = frozenset({"report-writer", "chart-maker"})` —— 与 `_attempt_auto_seal_from_artifacts` 实际能重建的集合对齐。**code-executor 虽也可重建但本就被 `_REQUIRES_SEAL` 排除**（run_metric_plan produce-and-deliver 合一，结构上不漏 seal），不纳入。
- **不重复造重建逻辑**：直接调 executor 的 `_attempt_auto_seal_from_artifacts`（复用优先，CLAUDE.md §12）。需把它的 import 放 **after_agent 函数体内惰性 import**（守 import 环铁律——seal_gate_middleware 不在顶层 import executor）。
- **after_agent 是兜底不是主路**：主堵漏仍是 L1（after_model 逼补调）。after_agent 只在 L1 cap 放行后、agent 真终止前，对可重建产物补一次 auto-seal，**把 executor L3 提前到终止点 + 消除 `Task failed` 中间态**。

### 2.2 修法 B：data-analyst 诚实不兜底（只降触发率）

data-analyst 漏调：
- L1 after_model 逼补调（已有，生产实证有效——重试就成功说明补调能成）。
- L2 executor seal-resume 补轮（已有）。
- after_agent **不做任何事**（不能 jump、不能伪造判读）。
- 仍漏 → L4 FAILED → lead 重试（生产实证有效）。

**不为 data-analyst 造 after_agent auto-seal**——判读结论是认知产物，伪造比漏调更糟（HarnessX reward hacking：别让系统学会「反正有兜底」）。**spec 明确写：data-analyst 的 seal 漏调是降级非根治，符合 memory `feedback_code_has_fix_not_equal_bug_eliminated`。**

### 2.3 可观测性：sealed_by 四分类 + 触发率（HarnessX trace richness）

每次 seal 完成时记录 `sealed_by` ∈ {`model_direct`（LLM 自己调）, `gate_reminder`（L1 jump 后补调）, `seal_resume`（L2 补轮）, `after_agent_artifacts`（修法 A）, `executor_artifacts`（L3）}。落进 handoff JSON 的元字段 + 日志。

**为什么必需**（HarnessX §7.2 + memory `feedback_fallback_trigger_rate_must_be_observable`）：兜底静默工作太好，会把上游 seal 漏调回归藏在全绿下。`after_agent_artifacts` 触发率上升 = 上游 L1 在退化的信号，必须可观测。验收把它做成可查询指标。

### 2.4 不改的东西

- **不改** `_MAX_REMINDERS=2`（§一末：提高 cap 依赖 thinking 超时线先解决，本 spec 不碰）。
- **不加** 任何 seal 提醒 prompt 规则（Telecom 禁令）。
- **不改** `_attempt_auto_seal_from_artifacts` 的重建逻辑（复用，只新增调用点）。
- **不改** L2 seal-resume / L4 FAILED→lead 重试（data-analyst 仍走这条）。
- **不为 data-analyst/code-executor 造 after_agent auto-seal**。

---

## 三、改动清单（change manifest 格式 — HarnessX Evolver）

### 3.1 `seal_gate_middleware.py` —— 加 after_agent 兜底
- **编辑**：加 `_RECONSTRUCTABLE` frozenset；加 `after_agent` + `aafter_agent`（不标 can_jump_to）；惰性 import `_attempt_auto_seal_from_artifacts`；加 `_workspace_from_state` helper。
- **预期改善**：report-writer/chart-maker 漏调时，终止点自动 auto-seal → 消除 `Task failed` 中间态（3a41e483 类场景）。
- **可能回归**：after_agent 调 auto-seal 若 workspace 解析错 → 静默不兜底（fail-open，不抛）。需 smoke test 覆盖 workspace=None。
- **病理风险**：reward hacking（LLM 学会依赖兜底）→ 由 §2.3 sealed_by 触发率可观测兜住。

### 3.2 seal 完成路径 —— 记录 sealed_by
- **编辑**：seal 工具 / executor 终止路径写入 `sealed_by` 元字段。
- **预期改善**：兜底触发率可观测，上游退化不被藏。
- **可能回归**：handoff schema 加字段 → 确认下游消费方容忍未知/可选字段（已有 partial 三态经验，schema 加可选字段安全）。

### 3.3 `_MAX_REMINDERS` —— 不改（显式声明）
- 文档注明：cap 提高依赖 thinking-overload 线（另一 agent 处理）。

---

## 四、测试（TDD 红→绿 + smoke test — HarnessX 强制）

> ⚠️ 改 middleware/executor 核心：除单测外**必须裸导入两生产入口**（`PYTHONPATH=. python -c "import app.gateway"` + `make_lead_agent`），守 import 环铁律。conftest mock executor 会藏环，裸导入才抓得到。

测试文件：`tests/test_seal_gate_after_agent.py`（新增）。

1. **test_after_agent_auto_seals_reconstructable_on_miss**（核心，红→绿）
   - 构造 report-writer：outputs/report.md 存在、history 无 seal ToolMessage。
   - 调 SealGate.after_agent → 断言 `_attempt_auto_seal_from_artifacts` 被调、handoff 文件被建。改前红（无 after_agent）、改后绿。

2. **test_after_agent_skips_data_analyst**（守边界）
   - data-analyst 漏调 → after_agent 断言**不调** auto-seal、返回 None（不伪造认知产物）。

3. **test_after_agent_noop_when_already_sealed**
   - history 已有 seal ToolMessage → after_agent 返回 None，不重复 seal。

4. **test_after_agent_fail_open_on_bad_workspace**（smoke / 鲁棒）
   - workspace=None / 解析异常 → after_agent 不抛、返回 None（fail-open）。

5. **smoke test_seal_gate_instantiates_and_runs**（HarnessX Evolver 要求）
   - SealGate(subagent_name='report-writer') 实例化 + after_model/after_agent 喂合成 state 跑通不抛。

6. **test_sealed_by_recorded**（可观测性）
   - 各路径（model_direct / gate_reminder / after_agent_artifacts）seal 后断言 handoff 元字段含正确 `sealed_by`。

7. **test_max_reminders_unchanged**（防 scope 漂移）
   - 断言 `_MAX_REMINDERS == 2`（防实施 agent 顺手改 cap，违反 §2.4）。

---

## 五、验收标准（确定性 gate 序列 — HarnessX Critic+Gate，第一个失败即停）

1. **manifest 完整性**：§三每处改动有预期改善 + 可能回归 + 对应测试。
2. **smoke**：测试 5（SealGate 实例化跑通）绿。
3. **红→绿**：测试 1 改前红、改后绿；测试 2/3/4/6/7 绿。
4. **回归（seesaw）**：现有 seal 相关测试全绿（`test_seal_gate*` / executor seal-resume / auto-seal 邻域）；backend 全量按 memory `feedback_pr_merge_must_run_full_suite_on_shared_logic` 跑 + grep `_attempt_auto_seal_from_artifacts` 调用方。
5. **import 环**：裸导入 `app.gateway` / `make_lead_agent` 0 退出。
6. **可观测性**：`sealed_by` 五分类可查询；`after_agent_artifacts` 触发率有日志/指标。
7. **scope**：`_MAX_REMINDERS` 未改；无新增 seal 提醒 prompt 规则。

---

## 六、风险与注意事项（三大病理自检 — HarnessX operational mirror）

1. **reward hacking**：after_agent 兜底会不会让 LLM 学会「反正有兜底就不好好 seal」？→ §2.3 sealed_by 触发率可观测兜住；且 after_agent 只对**可重建产物**（report.md/png 是真产出，不是伪造），不构成「糊弄验收」。data-analyst 明确不兜底，杜绝伪造判读。
2. **catastrophic forgetting**：加 after_agent 会不会回归现有 after_model L1 行为？→ 两 hook 正交（after_model 逼补调、after_agent 终止点兜底），测试 4/7 守住 L1 不变；现有 seal 测试全绿守 seesaw。
3. **under-exploration**：本 spec **敢动结构**（加 after_agent hook、改终止点兜底），不是又加一条 prompt 规则——正面避开 HarnessX §6.6 Telecom 的累加 reminder 陷阱。
4. **import 环**：`_attempt_auto_seal_from_artifacts` 的 import 必须在 after_agent 函数体内惰性 import（seal_gate_middleware 顶层不 import executor，否则闭环 → Gateway 起不来）。
5. **受保护文件**：seal_gate_middleware / executor 是 deerflow 定制中间件，sync 时 surgical 守护这段 after_agent。
6. **thinking 超时线协调**：本 spec 只管「收尾漏 call」类（生产实证的类型）；thinking 超时类（turn 内炸、after hook 不触发）由另一条线（spec 2026-06-18-data-analyst-thinking-overload，另一 agent）处理。两者**不能在本 spec 提高 cap** —— 提高 cap 会让 thinking 超时类烧更多 jump。

---

## milestone 建议
- 「subagent 派遣/生命周期 infra 加固」track：ETHO-1（SealGate after_agent 兜底，可重建产物结构堵死 + 认知产物诚实降级）与 ETHO-2（registry/prompt）同属 subagent 生命周期 infra 系列。checkpoint = 「seal 漏调从『偶发 Task failed 中间态 + 多烧一轮』降到『可重建产物零中间态、认知产物可观测降级』」。
