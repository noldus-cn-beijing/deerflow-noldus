# 编排路径 SSOT — 阶段 A 实施 spec（A1 prompt 渲染 + A2 派遣/ask 校验）

**类型**：代码核验版实施 spec（行号/锚点已对 dev HEAD `e7724e5b` 核验）
**对应**：[编排路径 SSOT 诊断](2026-05-29-orchestration-path-ssot-diagnosis-design.md) §3 阶段 A + [roadmap v2 优先级重排](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) P0 队头
**估期**：A1 ~1 天 + A2 ~2 天 = ~3 天（含 dogfood）
**前置**：无（地基已就位，见 §0）
**执行者**：独立 agent，TDD

---

## 0. 核心决策（已锁定，实施 agent 不得偏离）

| 决策 | 锁定值 | 理由 |
|---|---|---|
| **SSOT 形态** | **声明式 Python 数据**（dataclass + dict），**不是命令式代码**（无 `await`/`while`/`if` 控制流） | path SSOT 要被**三个消费者读取**（prompt 渲染 / 顺序 provider / ask provider）。命令式代码只能被"执行"，不能被多视角"读取分析"。这是 catalog YAML（声明式 SSOT）模式的同构物，**不是** Claude Code Workflow（命令式 JS）模式。详见诊断决策记录 |
| **SSOT 位置** | harness 内 `subagents/path_registry.py`，与 `SubagentConfig`（`subagents/config.py`）同处 | 依赖（`required_upstream_handoffs`）已在 `SubagentConfig`；路径（path spec）放它旁边，依赖与路径 single source。编排是 harness 职责，不塞进 ethoinsight 数据包 |
| **范围** | A1（prompt 从 SSOT 渲染，纯重构零行为变化）+ A2（派遣顺序 provider + ask 点 provider 从 SSOT 生成，有行为变化） | 全做。A1 先合（低风险纯重构），A2 后合（新拦截，需 dogfood） |
| **不做** | 不碰 `interrupt()`（阶段 B）；不动 catalog；不删现有 5.7/5.8 兜底 | 阶段 A 只消除"路径双存"；唤醒点仍走"结束+重入"，但重入读结构化路径进度而非重新解释 prompt |

**地基已核验（dev HEAD `e7724e5b`）**：
- `gate_completed` 数组已是运行中的"路径进度指针"（`experiment_context.py`：gate1_paradigm / gate2_quality_acknowledged / gate3_viz_acknowledged）——A2 的 provider 读它判断"走到哪步"
- `SubagentConfig` 在 `subagents/config.py:11`，含 `required_upstream_handoffs`
- prompt 意图状态机在 `lead_agent/prompt.py:283-294`（8 条路径箭头图，当前**手写**）
- prompt 已有声明式渲染范例：`prompt.py:200-203` 用 `noldus_order` 列表渲染 capability 块（A1 照此手法）
- provider 挂载范式：`agent.py:351-376`（`XxxBridge() + GuardrailMiddleware(provider=Xxx(), fail_closed=...)`）

---

## 1. SSOT 数据结构（`subagents/path_registry.py`，新建）

```python
"""编排路径 SSOT — 8 条 INTENT 路径的声明式定义。

唯一真相源：prompt 箭头图、派遣顺序 provider、ask 点 provider 三个消费者
都从本模块读取，不得各自硬编码路径逻辑。改路径 = 改本文件，三处自动同步，
CI 哨兵（tests/test_path_registry_ssot.py）验证一致。

声明式数据，非命令式代码：这里没有 await/while/if 控制流；路径的"执行"由
deerflow 现有 middleware/provider 读本数据驱动，不在此运行。
"""
from dataclasses import dataclass
from typing import Literal

StepKind = Literal["dispatch", "ask"]


@dataclass(frozen=True)
class Step:
    kind: StepKind
    # dispatch: subagent 名（用 prompt 侧的连字符命名 "code-executor"，
    #           对 required_upstream_handoffs 的下划线名做映射，见 §1.1）
    # ask:      ask key（"viz" / "report" / "four_choice"），对应 gate_completed 里的
    #           "gate3_viz_acknowledged" 等（映射见 §1.2）
    target: str
    # 仅 dispatch 用：该步是否条件触发（如 chart-maker 仅在 viz=yes）。None=无条件
    condition: str | None = None


# 8 个 INTENT → 有序 step 序列。从 prompt.py:286-294 箭头图逐条转写而来。
PATHS: dict[str, list[Step]] = {
    "E2E_FULL_ASKVIZ": [
        Step("dispatch", "code-executor"),
        Step("dispatch", "data-analyst"),
        Step("ask", "viz"),
        Step("dispatch", "chart-maker", condition="viz==yes"),
        Step("ask", "report"),
    ],
    "E2E_FULL": [
        Step("dispatch", "code-executor"),
        Step("dispatch", "data-analyst"),
        Step("dispatch", "chart-maker"),
        Step("ask", "report"),
    ],
    "E2E_MIN": [
        Step("dispatch", "code-executor"),
        Step("ask", "four_choice"),
    ],
    "CHART":        [Step("dispatch", "chart-maker")],
    "REPORT":       [Step("dispatch", "report-writer")],
    "QA_FACT":      [Step("dispatch", "knowledge-assistant")],
    "QA_KNOWLEDGE": [Step("dispatch", "knowledge-assistant")],
    "CLARIFY":      [Step("ask", "clarify")],
}

VALID_INTENTS = frozenset(PATHS.keys())  # ← intent_classification_provider 改 import 此处
```

### 1.1 命名映射（消除诊断「洞 3」）
- prompt 侧用连字符（`code-executor`），`required_upstream_handoffs` 用下划线（`code_executor`）
- path_registry 的 `Step.target` 用**连字符**（与 prompt 一致，因为主要消费者是 prompt 渲染）
- 提供一个 helper `to_handoff_name(target: str) -> str`（`"code-executor" → "code_executor"`），A2 的顺序 provider 用它对接 `required_upstream_handoffs`
- **import-time 校验**：每个 `Step(kind="dispatch")` 的 target 必须是 `BUILTIN_SUBAGENTS` 的合法 key（fail-fast，仿 `subagents/builtins/__init__.py:31` 现有的 fail-fast 校验）

### 1.2 ask key → gate_completed 映射
| ask key | gate_completed 标记 | 现状 |
|---|---|---|
| `viz` | `gate3_viz_acknowledged` | ✅ 已用（`intent_post_step_ask_gate_provider`） |
| `report` | `gate4_report_acknowledged` | 🆕 A2 新增（现在无 provider 保护——诊断「洞 2」） |
| `four_choice` | `gate_min_choice_acknowledged` | 🆕 A2 新增 |
| `clarify` | （不走 gate，直接 ask_clarification 中断） | 不变 |

---

## 2. A1：prompt 从 SSOT 渲染（纯重构，零行为变化，先合）

**目标**：删掉 `prompt.py:286-294` 手写箭头图，改成从 `PATHS` 渲染。渲染出的文本**必须与现有箭头图语义等价**（lead 读到的内容不变 → agent 行为不变）。

### 2.1 新增渲染函数（`prompt.py`，仿 `noldus_order` 渲染手法 line 200-203）
```python
def _render_intent_state_machine() -> str:
    """从 path_registry.PATHS 渲染意图状态机箭头图段。
    替代原 prompt.py:286-294 的手写 markdown。"""
    from deerflow.subagents.path_registry import PATHS
    lines = []
    for intent, steps in PATHS.items():
        chain = " → ".join(_render_step(s) for s in steps)
        lines.append(f"{intent} → {chain}")
    return "\n".join(lines)
```
- `_render_step`：dispatch → `target`；ask → `ask(<key>?)`；condition → `[<condition>]target`
- 渲染结果嵌回 `### 意图状态机` 段的代码块（line 285-294 区域）
- **保留** §296-304 的"复合语义判定 / Fast-path"自然语言（那是**意图分类规则**，不是路径拓扑，不在 SSOT 范围——SSOT 只管"INTENT→路径"，不管"用户语言→INTENT"）

### 2.2 `intent_classification_provider` 的 `_VALID_INTENTS` 改 import
- 现状 `intent_classification_provider.py:31` 硬编码 `_VALID_INTENTS = frozenset({...})`
- 改为 `from deerflow.subagents.path_registry import VALID_INTENTS as _VALID_INTENTS`
- 消除「8 个 INTENT 定义双存」（现在 prompt + provider 各一份）

### 2.3 A1 验收
- [ ] `_render_intent_state_machine()` 输出与原 prompt.py:286-294 语义等价（CI 哨兵 §4，逐条比对）
- [ ] `_VALID_INTENTS` 来自 path_registry（grep 确认 provider 不再硬编码）
- [ ] 全量测试零退化（A1 是纯重构，**不应有任何行为测试变化**）
- [ ] dogfood：跑一次 E2E_FULL（FST），确认 lead 读到的意图状态机段内容与改前一致、派遣行为不变

---

## 3. A2：派遣顺序 + ask 点 provider 从 SSOT 生成（新拦截，需 dogfood，后合）

### 3.1 新建 `PathSequenceProvider`（堵诊断「洞 1」：跳过 data-analyst 无人拦）
**位置**：`guardrails/path_sequence_provider.py`（新建，仿 `intent_post_step_ask_gate_provider.py` 结构）

**逻辑**：
1. 只拦 `task` 工具；从 messages 提取 latest `[intent]`（复用 `_extract_latest_intent`，可从 ask_gate provider 抽公共 helper）
2. 查 `PATHS[intent]`，找出本次 `task(subagent_type=X)` 中 X 在路径里的位置
3. 校验 **X 之前的所有 `dispatch` step 对应的 handoff 都已落盘**（读 workspace 的 `handoff_<prev>.json` 是否存在）
4. 若有前序 dispatch step 的 handoff 缺失 → **deny**，含明确指令（[[feedback_deny_messages_must_direct]]）：
   > `"按 <intent> 路径，<X> 之前必须先完成 <prev>。请先 task(<prev>)。"`
5. Fail-open：workspace/context 不可用时 allow（仿 ask_gate provider 的 fail-open）

**与现有 `task_handoff_authorization_provider` 的区别（必须说清，避免重复）**：
- 后者校验"prompt 里有没有写 `{{handoff://X}}` 占位符"（依赖**声明**）
- `PathSequenceProvider` 校验"路径中 X 的前序 dispatch 是否**真完成**了"（顺序**事实**）
- 两者正交：一个管"你声明了依赖吗"，一个管"你按路径顺序走了吗"

### 3.2 ask 点 provider 从 SSOT 生成（堵诊断「洞 2」：8 个 ask 点只保护了 1 个）
**现状**：`intent_post_step_ask_gate_provider` 只硬保护 `E2E_FULL_ASKVIZ` 的 `ask(viz?)`。
**改造**：泛化为读 `PATHS` —— 对任意 `intent`，若路径中某 `ask` step 之前的 dispatch 都已完成、但该 ask 对应的 `gate_completed` 标记未落盘，且 lead 试图跳过它派下一个 dispatch → deny。
- 保留现有 viz 拦截行为（它是迁移样板，回归测试必须证明 viz 行为不变）
- 新覆盖 `report`（gate4）、`four_choice`（E2E_MIN）
- **实现选择**：扩展现有 `IntentPostStepAskGateProvider` 使其数据驱动（读 PATHS），而非新建第 N 个 provider（否则又是打补丁——这正是本 spec 要消除的模式）

### 3.3 挂载（`agent.py`，仿 351-376 范式）
- `PathSequenceProvider` 在 W18（task_handoff_authorization）**之前**挂（顺序校验先于占位符校验，给更早更具体的反馈）
- 若 `PathSequenceProvider` 需要 ContextVar bridge（读 lead messages），仿 `IntentBridgeMiddleware`
- 全部走 `workflow_mode` 一致的挂载条件（确认与现有 intent provider 同档启用）

### 3.4 A2 验收
- [ ] E2E_FULL 下 lead 跳过 data-analyst 直接 task(chart-maker) → `PathSequenceProvider` deny，含"先 task(data-analyst)"指令
- [ ] viz 拦截行为与改前**完全一致**（回归，迁移样板不能破）
- [ ] E2E_FULL 下漏 `ask(report?)` 直接结束 → 被 ask provider 拦（新覆盖，洞 2 堵上）
- [ ] 正常顺序流（code→analyst→chart→ask report）全程不被误拦（fail-open + 正确路径放行）
- [ ] 与 `task_handoff_authorization` 不重复 deny（两 provider 正交，不双重拦同一问题）

---

## 4. CI 哨兵（关键，catalog 同款，A1 就要建）

新建 `tests/test_path_registry_ssot.py`：

| 测试 | 验证 |
|---|---|
| `test_render_matches_legacy_arrow_diagram` | `_render_intent_state_machine()` 输出语义等价于原 286-294（防 A1 渲染漂移；可对一份 frozen 期望字符串） |
| `test_valid_intents_single_source` | `intent_classification_provider._VALID_INTENTS is path_registry.VALID_INTENTS`（防 INTENT 定义双存复活） |
| `test_every_dispatch_target_is_registered_subagent` | 每个 `Step("dispatch", X)` 的 X 在 `BUILTIN_SUBAGENTS`（import-time fail-fast 的测试镜像） |
| `test_every_ask_key_has_gate_mapping` | 每个 `Step("ask", k)` 的 k 在 §1.2 映射表 |
| `test_path_sequence_provider_reads_paths` | PathSequenceProvider 的顺序判断确实来自 PATHS（改 PATHS 一条，provider 行为随之变）——证明 SSOT 真的驱动 provider，不是各写各的 |

**哨兵的意义**：改 `path_registry.PATHS` → prompt 渲染 + 两个 provider 自动同步；任何一处脱离 SSOT 手改 → CI 红。这是 catalog `test_tuning_section_lists_catalog_tunable_params` 哨兵的同款思路。

---

## 5. 实施顺序（TDD task 拆分）

| Task | 内容 | 阶段 | 估时 |
|---|---|---|---|
| T1 | 建 `path_registry.py`（Step/PATHS/VALID_INTENTS/helper + import-time 校验）+ 4 个 schema 测试 | A1 | 0.5 天 |
| T2 | `prompt.py` 加 `_render_intent_state_machine()`，替换手写箭头图；`intent_classification_provider` 改 import VALID_INTENTS | A1 | 0.25 天 |
| T3 | CI 哨兵 `test_render_matches_legacy_arrow_diagram` + 全量回归（A1 零行为变化）+ FST dogfood | A1 | 0.25 天 |
| **— A1 合 dev —** | | | |
| T4 | 新建 `PathSequenceProvider` + bridge + 测试（堵洞 1） | A2 | 0.75 天 |
| T5 | 泛化 `IntentPostStepAskGateProvider` 数据驱动读 PATHS（堵洞 2，保 viz 回归） | A2 | 0.75 天 |
| T6 | `agent.py` 挂载 + 挂载顺序 + 与 W18 正交性测试 | A2 | 0.25 天 |
| T7 | dogfood：复现"跳过 data-analyst""漏 ask(report)"被拦 + 正常流不误拦 + 全量回归 | A2 | 0.5 天 |
| **合计** | | | **~3 天** |

---

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| A1 渲染文本与原箭头图不等价 → lead 行为偷偷变了 | T3 哨兵逐条比对 frozen 期望；A1 定义为"纯重构"，任何行为测试变化都是 bug |
| `PathSequenceProvider` 与 `task_handoff_authorization` 重复 deny | §3.1 明确两者正交边界；T6 专门测"不双重拦" |
| ASKVIZ viz 拦截在 A2 泛化时被破坏 | viz 是迁移样板，T5 回归测试必须证明 viz 行为逐字不变 |
| 条件 step（chart-maker if viz==yes）的 condition 语义在 provider 里难判定 | A2 先只支持"无条件前序 dispatch 必须完成"的校验；condition 语义判定（viz==yes 才要求 chart-maker）留作 provider 的保守放行（condition 步不强制），不阻塞 |
| 改 prompt.py 触发 [[feedback_sync_protected_files_registry_loss]]（受保护文件） | prompt.py 本就是 Noldus 受保护文件，本 spec 是 Noldus 定制，正常改；注意未来 sync 时 surgical merge |
| 与并发的 Sprint 3/4 撞 prompt.py | Sprint 3 改 prompt 的"播报段"（§2.6），本 spec 改"意图状态机段"（283-294），不同片区；约定谁先合谁后 rebase，且 Sprint 3 在其播报段留 `# TODO(orchestration-ssot)` |

---

## 7. 不在阶段 A 范围
- ❌ `interrupt()` 改造唤醒点（阶段 B / P2）
- ❌ 把"用户语言→INTENT"的意图分类规则（prompt §296-304 Fast-path）也 SSOT 化（那是分类逻辑，非路径拓扑；本 spec 只管 INTENT→路径）
- ❌ 删除任何现有 provider 或 5.7/5.8 seal 兜底
- ❌ 动 catalog / ethoinsight 数据包（编排是 harness 职责）
- ❌ DataQualityGuardrail（Sprint 5；但 Sprint 5 实施时应复用本 spec 建立的"provider 从 SSOT 生成"模式，见 Sprint 5 迁移注记）

---

## 8. 给实施 agent 的提示
1. **SSOT 是声明式数据不是命令式代码**——若你在 path_registry.py 里写 `await`/`while`/`if` 控制流，方向就错了（那是 Claude Code Workflow 的 JS 模式，不是我们的）
2. **A1 是纯重构**：渲染出的 prompt 必须语义等价，任何 agent 行为变化都是 bug，不是 feature
3. **A2 的两个 provider 必须数据驱动**：行为来自读 PATHS，不是硬编码 if intent==...；T 测试要能证明"改 PATHS → provider 行为随之变"
4. **viz 拦截是迁移样板**：A2 泛化 ask provider 时，viz 行为必须逐字不变（回归红线）
5. **行号会漂**：本 spec 行号对 dev HEAD `e7724e5b` 核验；实施前 git pull 后重新 grep 锚点（prompt.py 意图状态机段、agent.py 351-376 挂载段）
6. 全量基线：ethoinsight 504 / backend 当前基线（实施前 `make test` 取真值，相对增量准、绝对值不跨环境比）
