# Subagent 角色拆分 + Capability-Exposure 实施 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Spec source:** [docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md](../specs/2026-05-18-subagent-role-split-capability-exposure-spec.md)
>
> **Worktree:** `.claude/worktrees/subagent-role-split-impl`(branch `worktree-subagent-role-split-impl`), rebased onto `dev` at HEAD `31275138`.

**Goal:** 用 capability-exposure 重构 lead → 5 subagent 调度,把 5-14/5-18 同类故障(lead 不读 handoff / catalog 无 fallback / code-executor 角色过载)用 harness 级 guardrail 硬约束消除。

**Architecture:** SubagentConfig 加 4 个 capability metadata 字段;按用户意图(E2E_FULL/E2E_MIN/CHART/REPORT/QA_FACT/QA_KNOWLEDGE/CLARIFY)切 5 个 subagent(chart-maker 新建);Plan dataclass 拆 PlanMetrics+PlanCharts,catalog CLI 加 `--mode` 与 `_common.yaml` fallback;2 个新 GuardrailProvider(IntentClassification + TaskHandoffAuthorization)在 lead 层强约束;lead prompt 从 1243 行瘦身到 ~200 行,detail 移到 ethoinsight-lead-interaction skill。

**Tech Stack:** Python 3.12 (deerflow harness) / Python 3.10 (ethoinsight) / LangGraph 0.x / dataclass / pytest / YAML catalog / Markdown skills

---

## Operating Constraints (实施前必读)

| # | Constraint | 强制度 |
|---|---|---|
| **C1** | Single source of truth:指标/范式/展示元数据**只存一份**,skill 不内嵌结构化知识(只链接到 catalog YAML / Python 常量) | 硬性 |
| **C2** | 不动 `HandoffIsolationProvider` / `Ev19TemplateGuardrailProvider`(现有 guardrail,职责正交) | 硬性 |
| **C3** | 不动 data-analyst / report-writer / knowledge-assistant 内部 system_prompt 主体逻辑,只补 capability metadata + 改 workflow 里 read 哪份 handoff | 硬性 |
| **C4** | 不碰 worktree `worktree-spec-phase-1-handoff`(并行任务) | 硬性 |
| **C5** | 不重新走 grill-me — Q1-Q12 决策已锁 | 硬性 |
| **C6** | dataclass 命名:catalog 端 `@dataclass(frozen=True)`、plan 端 `@dataclass`;不加 pydantic 依赖 | 硬性 |
| **C7** | 加新 chart/metric:YAML 注册与 `ethoinsight/scripts/...` 脚本**同一 commit** 落地(避免 "YAML 说有但脚本不存在") | 硬性 |
| **C8** | TDD 强制:每个 WI 红→绿→commit,test 先写、先看失败、再实现 | 硬性 |
| **C9** | harness 层(`packages/harness/deerflow/...`)**不允许** `from app.*`;`tests/test_harness_boundary.py` CI 拦截 | 硬性 |
| **C10** | 边界含糊时用 AskUserQuestion 反问,不要默认猜(spec §13.5) | 硬性 |

---

## File Structure (改动文件全景)

### 新建

```
packages/ethoinsight/ethoinsight/catalog/_common.yaml                              # W3
packages/ethoinsight/ethoinsight/scripts/_common/plot_timeseries.py                # W6
packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py # W13
packages/agent/backend/packages/harness/deerflow/guardrails/intent_classification_provider.py  # W17
packages/agent/backend/packages/harness/deerflow/guardrails/task_handoff_authorization_provider.py  # W18
packages/agent/skills/custom/ethoinsight/references/execution-conventions.md       # W7
packages/agent/skills/custom/ethoinsight-lead-interaction/SKILL.md                 # W8
packages/agent/skills/custom/ethoinsight-lead-interaction/references/*.md          # W8
packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md                      # W21
packages/agent/skills/custom/ethoinsight-chart-maker/references/*.md               # W21
```

### 改动

```
packages/agent/backend/packages/harness/deerflow/subagents/config.py               # W1
packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py     # W11
packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py      # W12
packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py     # W14
packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py  # W15
packages/agent/backend/packages/harness/deerflow/subagents/builtins/__init__.py    # W13(注册 chart_maker)
packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py       # W16(瘦身)
packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py        # W17/W18(挂载新 guardrail)
packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py       # W18 注册 chart_maker / W19 自动注入
packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py # W20(改输出 plan_metrics.json)
packages/ethoinsight/ethoinsight/catalog/schema.py                                 # W2(PlanMetrics + PlanCharts)
packages/ethoinsight/ethoinsight/catalog/resolve.py                                # W4/W5(resolve_charts + total_subjects)
packages/ethoinsight/ethoinsight/catalog/cli.py                                    # W4(--mode 参数)
packages/ethoinsight/ethoinsight/catalog/loader.py                                 # W3(load_common_catalog)
packages/agent/skills/custom/ethoinsight/SKILL.md                                  # W7(链接到 execution-conventions)
packages/agent/skills/custom/ethoinsight-code/SKILL.md                             # W11(删 chart 段)
packages/agent/skills/custom/ethoinsight-charts/SKILL.md                           # W9(重定位至 chart-maker)
packages/agent/skills/custom/ethovision-paradigm-knowledge/                        # W10(删 default-fallback)
```

### 测试新建

```
packages/agent/backend/tests/test_subagent_config_capability.py                    # W1
packages/agent/backend/tests/test_chart_maker_config.py                            # W13
packages/agent/backend/tests/test_code_executor_config.py                          # W11
packages/agent/backend/tests/test_data_analyst_config.py                           # W12
packages/agent/backend/tests/test_report_writer_config.py                          # W14
packages/agent/backend/tests/test_knowledge_assistant_config.py                    # W15
packages/agent/backend/tests/test_lead_prompt_capability_render.py                 # W16
packages/agent/backend/tests/test_intent_classification_provider.py                # W17
packages/agent/backend/tests/test_task_handoff_authorization.py                    # W18
packages/agent/backend/tests/test_task_tool_auto_inject.py                         # W19
packages/ethoinsight/tests/test_catalog_schema_plan_split.py                       # W2
packages/ethoinsight/tests/test_common_catalog.py                                  # W3
packages/ethoinsight/tests/test_resolve_charts.py                                  # W4
packages/ethoinsight/tests/test_evaluate_when.py                                   # W5
packages/ethoinsight/tests/test_plot_timeseries_cli.py                             # W6
```

测试改动:`packages/agent/backend/tests/test_prep_metric_plan.py` (W20 扩展)

---

## 依赖与执行顺序

```
Layer 0 (无依赖,可并行):     W1   W2   W3   W7
Layer 1 (依赖 L0):           W5(W2)  W6  W8(W7)  W9(W7)  W10(W7)  W21(W7,W9)
                              W4(W2,W3,W5)
Layer 2 (依赖 L0/L1):         W11(W1,W2,W8,W7)  W12(W1)  W14(W1)  W15(W1)
                              W13(W1,W2,W3,W4,W21,W7)
                              W20(W2)
Layer 3 (依赖 L2):            W16(W1,W11-W15,W8)  W17(W1)  W18(W1,W11-W15)
Layer 4 (依赖 L3):            W19(W1,W11-W15,W18)
Layer 5 (依赖全部):           W22 (dogfood)
```

**推荐线性执行顺序**(subagent-driven-development 一次只 dispatch 一个 task):

`W1 → W7 → W2 → W3 → W5 → W6 → W4 → W20 → W8 → W9 → W10 → W21 → W11 → W12 → W14 → W15 → W13 → W16 → W17 → W18 → W19 → W22`

**关键顺序约束**:
- **W20(prep_metric_plan 改输出 plan_metrics.json)必须在 W11(code-executor 读 plan_metrics.json)之前**。否则 W11 commit 到 W20 commit 之间 code-executor 期望 `plan_metrics.json` 但 tool 仍写 `metric_plan.json` → 中间状态 broken。本顺序已把 W20 提前到 W4 之后、W11 之前。
- **W18 + W19 应同 PR 合 dev**,中间不要单独发 W18(否则 W18 fail-closed 期间 lead 全部 task() 被 deny)。Plan 把 W18 排在 W19 紧前 1 个 commit,subagent-driven-development 可以连续跑两个 commit 然后一起 push。
- W16(lead prompt 瘦身)依赖 W11-W15 capability 都到位(prompt 渲染要靠 SubagentConfig.capability 字段),不要提前到 W11 之前。

---

## Per-Task Commit Convention

每个 task 完成且测试通过后 commit。Message 格式:

```
<type>(<scope>): <subject> (WI<N>)

<details — 改了什么、为什么>

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §<section>
```

`<type>` 取 `feat` / `refactor` / `test` / `docs`;`<scope>` 取 `subagent` / `catalog` / `guardrail` / `skill` / `lead` 等。

---

## Task 1 (W1): SubagentConfig — capability metadata 4 字段扩展

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/config.py:11-35`
- Test: `packages/agent/backend/tests/test_subagent_config_capability.py` (new)

**依赖:** 无 (Layer 0)

**Spec refs:** §3.1 / §3.2 / §3.3

- [ ] **Step 1: 红 — 写失败测试**

Create `packages/agent/backend/tests/test_subagent_config_capability.py`:

```python
"""W1: SubagentConfig 4 capability metadata 字段验收。

Spec §3.3:
- 4 新字段都可选,不破坏现有 builtin
- format_subagent_capability 在缺字段时输出 "(未声明)" 而非崩溃
- required_upstream_handoffs 中每个 key 必须在 HANDOFF_FILE_REGISTRY (fail-fast)
"""
from __future__ import annotations

import pytest

from deerflow.subagents.config import SubagentConfig, format_subagent_capability
from deerflow.subagents.builtins import BUILTIN_SUBAGENTS
from deerflow.tools.builtins.task_tool import HANDOFF_FILE_REGISTRY


def test_subagent_config_accepts_capability_fields():
    cfg = SubagentConfig(
        name="probe",
        description="d",
        when_to_use="when X",
        input_contract="prompt template",
        output_contract="returns Y",
        required_upstream_handoffs=["code_executor"],
    )
    assert cfg.when_to_use == "when X"
    assert cfg.input_contract == "prompt template"
    assert cfg.output_contract == "returns Y"
    assert cfg.required_upstream_handoffs == ["code_executor"]


def test_capability_fields_default_to_none_or_empty():
    cfg = SubagentConfig(name="bare", description="d")
    assert cfg.when_to_use is None
    assert cfg.input_contract is None
    assert cfg.output_contract is None
    assert cfg.required_upstream_handoffs == []


def test_existing_builtins_still_instantiate():
    # general-purpose / code-executor / data-analyst 等 unmodified config 仍能 import 而不报错
    for name, cfg in BUILTIN_SUBAGENTS.items():
        assert isinstance(cfg, SubagentConfig)
        assert isinstance(cfg.required_upstream_handoffs, list)


def test_format_subagent_capability_renders_known_fields():
    cfg = SubagentConfig(
        name="x",
        description="desc",
        when_to_use="适合 A",
        input_contract="传 B",
        output_contract="返回 C",
    )
    rendered = format_subagent_capability(cfg)
    assert "x" in rendered
    assert "desc" in rendered
    assert "适合 A" in rendered
    assert "传 B" in rendered
    assert "返回 C" in rendered


def test_format_subagent_capability_renders_placeholder_for_missing():
    cfg = SubagentConfig(name="x", description="desc")
    rendered = format_subagent_capability(cfg)
    assert "(未声明)" in rendered


def test_required_upstream_handoffs_must_reference_known_keys():
    # W1 实施加一个 module-level helper validate_subagent_handoff_refs(BUILTIN_SUBAGENTS, HANDOFF_FILE_REGISTRY)
    # 在 import builtins 时调用,如果 key 不在 registry 应 raise ValueError
    from deerflow.subagents.config import validate_subagent_handoff_refs

    bad = {
        "test-bad": SubagentConfig(
            name="test-bad",
            description="x",
            required_upstream_handoffs=["nonexistent_agent"],
        )
    }
    with pytest.raises(ValueError, match="nonexistent_agent"):
        validate_subagent_handoff_refs(bad, HANDOFF_FILE_REGISTRY)


def test_required_upstream_handoffs_validator_accepts_known():
    from deerflow.subagents.config import validate_subagent_handoff_refs

    good = {
        "test-good": SubagentConfig(
            name="test-good",
            description="x",
            required_upstream_handoffs=["code_executor"],
        )
    }
    validate_subagent_handoff_refs(good, HANDOFF_FILE_REGISTRY)  # should not raise
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_subagent_config_capability.py -v
```

Expected: 全部 FAIL,理由 "SubagentConfig.__init__() got an unexpected keyword argument 'when_to_use'" 等。

- [ ] **Step 3: 绿 — 实现**

Edit `packages/agent/backend/packages/harness/deerflow/subagents/config.py` — 在 `SubagentConfig` dataclass 字段后追加 4 个新字段(可选默认值),并补 helper:

```python
@dataclass
class SubagentConfig:
    # ---- 现有字段 (line 27-35,保持不变) ----
    name: str
    description: str
    system_prompt: str | None = None
    tools: list[str] | None = None
    disallowed_tools: list[str] | None = field(default_factory=lambda: ["task"])
    skills: list[str] | None = None
    model: str = "inherit"
    max_turns: int = 50
    timeout_seconds: int = 900
    # ---- W1 capability metadata 新增 ----
    when_to_use: str | None = None
    input_contract: str | None = None
    output_contract: str | None = None
    required_upstream_handoffs: list[str] = field(default_factory=list)


def format_subagent_capability(config: SubagentConfig) -> str:
    """Render a SubagentConfig as a Markdown block for lead prompt injection.

    格式:
        ### <name>
        - description: <description>
        - when_to_use: <when_to_use or '(未声明)'>
        - input_contract: <input_contract or '(未声明)'>
        - output_contract: <output_contract or '(未声明)'>

    required_upstream_handoffs 和 system_prompt 不暴露给 lead(harness 内部)。
    """
    def _or_placeholder(v: str | None) -> str:
        return v.strip() if v else "(未声明)"

    return (
        f"### {config.name}\n"
        f"- description: {_or_placeholder(config.description)}\n"
        f"- when_to_use: {_or_placeholder(config.when_to_use)}\n"
        f"- input_contract: {_or_placeholder(config.input_contract)}\n"
        f"- output_contract: {_or_placeholder(config.output_contract)}\n"
    )


def validate_subagent_handoff_refs(
    configs: dict[str, "SubagentConfig"],
    registry: dict[str, str],
) -> None:
    """Fail-fast: every `required_upstream_handoffs` entry must be a key in
    HANDOFF_FILE_REGISTRY. Called at module import time of builtins/__init__.py
    so wrong keys break test runs early instead of silently mis-routing.
    """
    for sub_name, cfg in configs.items():
        for upstream in cfg.required_upstream_handoffs:
            if upstream not in registry:
                raise ValueError(
                    f"Subagent '{sub_name}' references unknown upstream handoff '{upstream}'. "
                    f"Known: {sorted(registry)}"
                )
```

Then add a `validate_subagent_handoff_refs(BUILTIN_SUBAGENTS, HANDOFF_FILE_REGISTRY)` call at the bottom of `packages/agent/backend/packages/harness/deerflow/subagents/builtins/__init__.py`(import-time fail-fast).

- [ ] **Step 4: 跑测试确认通过**

```bash
PYTHONPATH=. uv run pytest tests/test_subagent_config_capability.py -v
```

Expected: 7 passed.

- [ ] **Step 5: 跑全套相关回归确认无破坏**

```bash
PYTHONPATH=. uv run pytest tests/test_task_tool_handoff_placeholders.py tests/test_subagent_executor.py -v 2>&1 | tail -30
```

Expected: 既有测试全绿。

- [ ] **Step 6: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/config.py \
        packages/agent/backend/packages/harness/deerflow/subagents/builtins/__init__.py \
        packages/agent/backend/tests/test_subagent_config_capability.py
git commit -m "$(cat <<'EOF'
feat(subagent): SubagentConfig 加 4 capability metadata 字段 (WI1)

- when_to_use / input_contract / output_contract / required_upstream_handoffs
  四字段全部可选,不破坏现有 builtin
- format_subagent_capability() helper 为 lead prompt 渲染 markdown
- validate_subagent_handoff_refs() 在 BUILTIN_SUBAGENTS import 时
  fail-fast 校验每个 required_upstream_handoffs entry 都在
  HANDOFF_FILE_REGISTRY 里

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §3
EOF
)"
```

---

## Task 2 (W7): ethoinsight 根 skill — execution-conventions.md 新建

**Files:**
- Create: `packages/agent/skills/custom/ethoinsight/references/execution-conventions.md`
- Modify: `packages/agent/skills/custom/ethoinsight/SKILL.md` (在底部追加引用)

**依赖:** 无 (Layer 0)

**Spec refs:** §8.1

无 Python 代码,无单元测试。Skill 文档质量靠 W11 / W13 实施时 read 该文件能 actionable 来验证。

- [ ] **Step 1: 创建 execution-conventions.md**

`packages/agent/skills/custom/ethoinsight/references/execution-conventions.md`:

```markdown
# EthoInsight 执行约束 (执行类 subagent 必读)

服务对象:code-executor / chart-maker 等"执行类"subagent。

## bash 调用形式

只允许两种形式:

1. **脚本调用**:`python -m ethoinsight.scripts.<paradigm | _common>.<name> --input ... --output ...`
2. **文件操作**:`mkdir / cp / mv / ls / cat / grep / head / tail` (常规 POSIX,不含 `python -c` / `pip install` / `bash -c '...'`)

CLI 例外(可调,但只限本进程):
- `python -m ethoinsight.catalog.resolve --mode metrics ...` — 由 `prep_metric_plan` 工具内部调,subagent 不直接用此 mode
- `python -m ethoinsight.catalog.resolve --mode charts ...` — chart-maker 自跑

其他形式会被 `ScriptInvocationOnlyProvider` 拦截。

## handoff JSON 写入规则

- 文件名严格:`handoff_<subagent_name>.json`(下划线,与 SubagentConfig.name 中的连字符替换后一致)
- 路径:`/mnt/user-data/workspace/handoff_<name>.json`
- 编码:UTF-8,`ensure_ascii=False`,2-space indent
- schema:见各 subagent 自己 skill 的 `templates/output-contract.md` (本文件不重复 schema)

## error recovery

- 脚本 stderr 非空 → 读 traceback → 决定是否重试
- 同一脚本(同一 metric_id / chart_id)最多重试 **2 次**;再失败则把 error 写入 handoff.errors[] 并继续后续步骤
- 不要"探索式地" `ls` skill 目录或 `--help`;该跑哪些脚本由 plan_metrics.json / plan_charts.json 决定

## gate_signals 块通用格式

执行类 subagent 完工后必须在最终 AIMessage 末尾输出:

```
[gate_signals]
constitution_acknowledged: true
<其他字段...>
errors_count: <int>
```

具体字段由各 subagent 自己 contract 决定。lead 用块的存在性判定走 gate_signals 路径。
```

- [ ] **Step 2: 修改根 SKILL.md 末尾追加引用**

In `packages/agent/skills/custom/ethoinsight/SKILL.md` 最后追加:

```markdown

## 操作约束

执行类 subagent (code-executor / chart-maker) **必读** `references/execution-conventions.md`(bash 形式、handoff 写入、error recovery、gate_signals 格式)。
```

- [ ] **Step 3: 手动校验 markdown 渲染**

```bash
ls packages/agent/skills/custom/ethoinsight/references/execution-conventions.md
wc -l packages/agent/skills/custom/ethoinsight/references/execution-conventions.md   # 期待 30-50 行
grep -n "execution-conventions" packages/agent/skills/custom/ethoinsight/SKILL.md    # SKILL.md 应有 1 处引用
```

- [ ] **Step 4: Commit**

```bash
git add packages/agent/skills/custom/ethoinsight/references/execution-conventions.md \
        packages/agent/skills/custom/ethoinsight/SKILL.md
git commit -m "$(cat <<'EOF'
docs(skill): 根 skill 加 execution-conventions.md (WI7)

执行类 subagent (code-executor / chart-maker) 必读的通用约束:
- bash 调用形式白名单
- handoff JSON 写入规则
- error recovery 上限 2 次
- gate_signals 块通用格式

C1 single source of truth:具体 handoff schema 仍归各 subagent
自己的 templates/output-contract.md,本文件只放跨 subagent 通用规则。

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §8.1
EOF
)"
```

---

## Task 3 (W2): Plan dataclass 拆 PlanMetrics + PlanCharts

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/catalog/schema.py` (add PlanMetrics, PlanCharts, keep old Plan as alias temporarily)
- Test: `packages/ethoinsight/tests/test_catalog_schema_plan_split.py` (new)

**依赖:** 无 (Layer 0)

**Spec refs:** §7.1

**注意:** spec §12 说"Plan dataclass 旧字段的 backward-compat 兼容层不保留 — 直接 W11+W13 同步改"。所以 W2 同时干两件事:
(a) 新增 PlanMetrics + PlanCharts dataclass
(b) 删除老 `Plan` dataclass

但 W2 单独 commit 时仍要让 `resolve()` / `plan_to_dict()` / `prep_metric_plan_tool` / 现有 tests 不破坏。策略:**W2 阶段保留 Plan 作为 PlanMetrics 的别名**(等价 dataclass),W4 落地后再删。这样不需要在 W2 一并改 resolve.py / cli.py / prep_metric_plan_tool.py。

- [ ] **Step 1: 红 — 写失败测试**

`packages/ethoinsight/tests/test_catalog_schema_plan_split.py`:

```python
"""W2: Plan dataclass 拆为 PlanMetrics + PlanCharts。"""
from __future__ import annotations

import pytest

from ethoinsight.catalog.schema import (
    PlanChart,
    PlanCharts,
    PlanInputs,
    PlanMetric,
    PlanMetrics,
    PlanSkipped,
    PlanStatistics,
)


def _sample_inputs() -> PlanInputs:
    return PlanInputs(raw_files=["/tmp/raw.txt"], groups_file=None, columns_file=None)


def test_plan_metrics_dataclass_has_required_fields():
    pm = PlanMetrics(
        paradigm="epm",
        ev19_template=None,
        generated_at="2026-05-18T00:00:00Z",
        inputs=_sample_inputs(),
        metrics=[],
        statistics=None,
        skipped=[],
        notes=[],
    )
    assert pm.paradigm == "epm"
    assert pm.schema_version == "1.0"   # default
    assert pm.metrics == []
    assert pm.statistics is None


def test_plan_charts_dataclass_has_required_fields():
    pc = PlanCharts(
        paradigm="epm",
        ev19_template=None,
        generated_at="2026-05-18T00:00:00Z",
        inputs=_sample_inputs(),
        charts=[],
        charts_fallback_available=[],
        skipped=[],
        user_intent=None,
        notes=[],
    )
    assert pc.paradigm == "epm"
    assert pc.schema_version == "1.0"
    assert pc.charts_fallback_available == []
    assert pc.user_intent is None


def test_plan_metrics_can_hold_real_entries():
    metric = PlanMetric(id="open_arm_time", script="ethoinsight.scripts.epm.compute_open_arm_time",
                       input="/tmp/raw.txt", output="/tmp/m.json", required=True, reason="paradigm.default")
    pm = PlanMetrics(
        paradigm="epm", ev19_template="EPM_v1", generated_at="2026-05-18T00:00:00Z",
        inputs=_sample_inputs(), metrics=[metric], statistics=None, skipped=[], notes=["n=3"],
    )
    assert pm.metrics[0].id == "open_arm_time"


def test_plan_charts_can_hold_charts_and_fallback():
    main = PlanChart(id="box_open_arm", script="ethoinsight.scripts.epm.plot_box", input="/tmp/raw.txt", output="/tmp/p.png")
    fallback = PlanChart(id="trajectory_plot", script="ethoinsight.scripts._common.plot_trajectory", input="/tmp/raw.txt", output="/tmp/p.png")
    pc = PlanCharts(
        paradigm="epm", ev19_template=None, generated_at="2026-05-18T00:00:00Z",
        inputs=_sample_inputs(), charts=[main], charts_fallback_available=[fallback],
        skipped=[], user_intent="再画几个图", notes=[],
    )
    assert pc.charts[0].id == "box_open_arm"
    assert pc.charts_fallback_available[0].id == "trajectory_plot"
    assert pc.user_intent == "再画几个图"


def test_plan_alias_backward_compat():
    """W2 过渡期:`Plan` 仍可 import 作为 PlanMetrics 别名,直到 W4/W11/W13 完成后再彻底删。"""
    from ethoinsight.catalog.schema import Plan
    # Plan 应等价于 PlanMetrics(或仍包含 charts 字段以兼容现有 resolve)
    # 选 PlanMetrics 别名:让现有 resolve 通过 plan.charts = [] 兼容,W4 时 resolve.py 拆函数
    assert Plan is PlanMetrics or hasattr(Plan, "metrics")
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd packages/ethoinsight
PYTHONPATH=. uv run pytest tests/test_catalog_schema_plan_split.py -v
```

Expected: ImportError "PlanMetrics" / "PlanCharts" 找不到。

- [ ] **Step 3: 绿 — 实现**

Edit `packages/ethoinsight/ethoinsight/catalog/schema.py` — 在文件末尾追加(保留旧 `Plan` dataclass 作为 alias 兼容期):

```python
# ============================================================================
# Plan split (W2): 拆 Plan → PlanMetrics + PlanCharts
# 老 Plan dataclass(line 128-139)保留作为 PlanMetrics 的过渡期别名,
# 等 W4/W11/W13 完成后 W22 dogfood 前彻底删。
# ============================================================================


@dataclass
class PlanMetrics:
    paradigm: str
    ev19_template: str | None
    generated_at: str
    inputs: PlanInputs
    metrics: list[PlanMetric]
    statistics: PlanStatistics | None
    skipped: list[PlanSkipped]
    notes: list[str]
    schema_version: str = "1.0"


@dataclass
class PlanCharts:
    paradigm: str
    ev19_template: str | None
    generated_at: str
    inputs: PlanInputs
    charts: list[PlanChart]
    charts_fallback_available: list[PlanChart]
    skipped: list[PlanSkipped]
    user_intent: str | None
    notes: list[str]
    schema_version: str = "1.0"
```

旧 `Plan` 留作 alias —— 在文件最末加:

```python
# Backward-compat alias during W2-W4 transition. Will be deleted after W11/W13
# stop importing Plan directly (use PlanMetrics).
# NOTE: do NOT alias `Plan = PlanMetrics` directly — old Plan has `charts: list[PlanChart]`
# which PlanMetrics does NOT have. Keep the original Plan dataclass intact for now.
# (i.e. **no code change** to existing Plan dataclass, just add the two new ones above.)
```

Actually re-read spec §7.1: "Plan(旧)保留为 backward-compatible 别名或在过渡期临时存活直到 W11/W13 完成 — 由 W2 实施 agent 决定(推荐:直接删旧 Plan dataclass,让 W11+W13 同步改)。"

我们选 **保留旧 Plan dataclass 不动**,只新增 PlanMetrics + PlanCharts。W22 之前再决定是否删。理由:W2 单独 commit 要保证 resolve.py / cli.py / prep_metric_plan_tool.py / 现有 ethoinsight 全套测试不破坏。

- [ ] **Step 4: 调整最后一条测试用例**

把 `test_plan_alias_backward_compat` 改为:

```python
def test_plan_metrics_and_plan_charts_coexist_with_legacy_plan():
    """W2 过渡期:旧 Plan dataclass 保留,新 PlanMetrics + PlanCharts 并列。"""
    from ethoinsight.catalog.schema import Plan
    # 老 Plan 仍含 metrics + charts 字段(给 resolve 用,直到 W4 拆 resolve_metrics/resolve_charts)
    assert hasattr(Plan, "__dataclass_fields__")
    assert "metrics" in Plan.__dataclass_fields__
    assert "charts" in Plan.__dataclass_fields__
    # 新 PlanMetrics 不再含 charts
    assert "charts" not in PlanMetrics.__dataclass_fields__
    # 新 PlanCharts 不含 metrics
    assert "metrics" not in PlanCharts.__dataclass_fields__
```

- [ ] **Step 5: 跑测试确认通过**

```bash
PYTHONPATH=. uv run pytest tests/test_catalog_schema_plan_split.py -v
PYTHONPATH=. uv run pytest tests/ -k "catalog" -v
```

Expected: 新测试全绿,既有 catalog 测试全绿。

- [ ] **Step 6: Commit**

```bash
git add packages/ethoinsight/ethoinsight/catalog/schema.py \
        packages/ethoinsight/tests/test_catalog_schema_plan_split.py
git commit -m "$(cat <<'EOF'
feat(catalog): 加 PlanMetrics + PlanCharts dataclass (WI2)

新 dataclass 取代将来 W4 拆出来的 resolve_metrics / resolve_charts
返回值。老 Plan dataclass 保留作过渡期别名 — W11/W13 完成,
W22 dogfood 前彻底删除。

PlanCharts 额外字段:
- charts_fallback_available: 单被试 / 组间不可对比场景的 fallback 候选
- user_intent: 来自 catalog CLI --user-intent

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §7.1
EOF
)"
```

---

## Task 4 (W3): _common.yaml + loader 加载

**Files:**
- Create: `packages/ethoinsight/ethoinsight/catalog/_common.yaml`
- Modify: `packages/ethoinsight/ethoinsight/catalog/loader.py` (add `load_common_catalog`)
- Test: `packages/ethoinsight/tests/test_common_catalog.py` (new)

**依赖:** 无 (Layer 0)

**Spec refs:** §7.2 / §7.3

**C7 提示**:本 task 同时落地 trajectory_plot + timeseries_plot 两条 YAML entry。trajectory_plot 的脚本已存在(`scripts/_common/plot_trajectory.py`),timeseries_plot 的脚本由 W6 同 commit/同分支落地。本 task 完成时 timeseries 脚本可能还不存在 — W3 只校验 YAML 加载,不调脚本;W6 才实际跑脚本。所以 W3+W6 可以分两个 commit,但**两个都进同一 PR 前不能合 dev**(避免 dev 上出现"YAML 说有 timeseries 但脚本不存在"的孤儿状态)。

- [ ] **Step 1: 红 — 写失败测试**

`packages/ethoinsight/tests/test_common_catalog.py`:

```python
"""W3: _common.yaml 加载 + 校验。"""
from __future__ import annotations

import pytest

from ethoinsight.catalog.loader import CatalogError, load_common_catalog


def test_load_common_catalog_returns_two_charts():
    cc = load_common_catalog()
    chart_ids = [c.id for c in cc.common_charts]
    assert "trajectory_plot" in chart_ids
    assert "timeseries_plot" in chart_ids


def test_common_charts_have_when_field():
    cc = load_common_catalog()
    for c in cc.common_charts:
        assert c.when, f"common chart '{c.id}' missing 'when' field"
        # 单被试 fallback 至少要 total_subjects >= 1
        # 该判断在 W5 _evaluate_when 扩展 total_subjects 后才能跑得通,
        # 这里只检查 YAML 字面量
        assert "total_subjects" in c.when or c.when == "always", (
            f"common chart '{c.id}' should use total_subjects-based when "
            f"(got '{c.when}')"
        )


def test_common_chart_script_path_format():
    cc = load_common_catalog()
    for c in cc.common_charts:
        # _common 通用脚本必须放在 ethoinsight.scripts._common.* 下
        assert c.script.startswith("ethoinsight.scripts._common."), (
            f"chart '{c.id}' script '{c.script}' must be under ethoinsight.scripts._common"
        )


def test_load_common_catalog_handles_missing_file(tmp_path):
    """Custom catalog_dir 指向无 _common.yaml 的目录 → CatalogError。"""
    with pytest.raises(CatalogError, match="_common.yaml"):
        load_common_catalog(catalog_dir=tmp_path)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd packages/ethoinsight
PYTHONPATH=. uv run pytest tests/test_common_catalog.py -v
```

Expected: ImportError `load_common_catalog` 不存在。

- [ ] **Step 3: 绿 — 实现 _common.yaml**

Create `packages/ethoinsight/ethoinsight/catalog/_common.yaml`:

```yaml
# 通用 charts — 范式无关,作为单被试 / 组间不可对比场景的 fallback 候选。
# C7 要求:加新 chart 时同步在 ethoinsight/scripts/_common/ 下添加 CLI 脚本,
# YAML 注册与脚本必须在同一 PR 落地。

common_charts:
  - id: trajectory_plot
    script: ethoinsight.scripts._common.plot_trajectory
    when: total_subjects >= 1
    rationale: 单被试或组间数据不全时的轨迹可视化

  - id: timeseries_plot
    script: ethoinsight.scripts._common.plot_timeseries
    when: total_subjects >= 1
    rationale: 单被试或组间数据不全时的时间动态
```

- [ ] **Step 4: 绿 — 实现 load_common_catalog**

Edit `packages/ethoinsight/ethoinsight/catalog/loader.py` — 末尾追加:

```python
# ============================================================================
# Common catalog — paradigm-agnostic fallback charts (W3)
# ============================================================================


@dataclass(frozen=True)
class CommonCatalog:
    """Paradigm-agnostic fallback resources."""
    common_charts: list["ChartEntry"]


# Use existing dataclass import at top of file; if missing, add:
# from dataclasses import dataclass


def load_common_catalog(catalog_dir: str | Path | None = None) -> CommonCatalog:
    """Load _common.yaml from catalog directory.

    Returns:
        CommonCatalog with common_charts list.

    Raises:
        CatalogError: file missing or malformed.
    """
    catalog_dir = Path(catalog_dir) if catalog_dir else _DEFAULT_CATALOG_DIR
    yaml_path = catalog_dir / "_common.yaml"
    if not yaml_path.is_file():
        raise CatalogError(
            f"_common.yaml not found in catalog directory: {yaml_path}"
        )
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise CatalogError(f"YAML parse failed for {yaml_path}: {e}") from e

    if not isinstance(raw, dict):
        raise CatalogError(
            f"{yaml_path}: top-level must be a mapping, got {type(raw).__name__}"
        )

    common_charts = _parse_chart_list_under_key(raw, "common_charts", yaml_path)
    return CommonCatalog(common_charts=common_charts)


def _parse_chart_list_under_key(raw: dict, key: str, source: Path) -> list[ChartEntry]:
    """Variant of _parse_chart_list that uses configurable key (charts vs common_charts)."""
    items = raw.get(key, []) or []
    if not isinstance(items, list):
        raise CatalogError(f"{source}: '{key}' must be a list")
    out: list[ChartEntry] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise CatalogError(f"{source}: {key}[{i}] must be a mapping")
        for f in ("id", "script", "when"):
            if f not in it:
                raise CatalogError(f"{source}: {key}[{i}] missing '{f}'")
        out.append(ChartEntry(id=it["id"], script=it["script"], when=it["when"]))
    return out
```

注意 import:文件顶部 already has `from dataclasses import dataclass`? 看现状(line 1-15)再决定是否加 import。

- [ ] **Step 5: 跑测试确认通过**

```bash
PYTHONPATH=. uv run pytest tests/test_common_catalog.py tests/test_catalog_schema_plan_split.py -v
PYTHONPATH=. uv run pytest tests/ -k "catalog" -v
```

Expected: 全绿。

- [ ] **Step 6: Commit**

```bash
git add packages/ethoinsight/ethoinsight/catalog/_common.yaml \
        packages/ethoinsight/ethoinsight/catalog/loader.py \
        packages/ethoinsight/tests/test_common_catalog.py
git commit -m "$(cat <<'EOF'
feat(catalog): _common.yaml + load_common_catalog (WI3)

注册两个 paradigm-agnostic fallback chart:
- trajectory_plot (脚本已存在)
- timeseries_plot (脚本由 W6 同 PR 落地)

C7 single source of truth:本 YAML 是 "fallback chart 注册表" 唯一来源,
chart-maker 的 skill 不再重复列举可用 fallback。

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §7.2/§7.3
EOF
)"
```

---

## Task 5 (W5): `_evaluate_when` 扩展 `total_subjects` 变量

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/catalog/resolve.py:295-326` (`_evaluate_when` + `_evaluate_atomic_when`)
- Test: `packages/ethoinsight/tests/test_evaluate_when.py` (new)

**依赖:** W2 (本 task 不直接 import PlanMetrics/PlanCharts,但语义上属于 plan-split 配套)

**Spec refs:** §7.4

- [ ] **Step 1: 红 — 写失败测试**

`packages/ethoinsight/tests/test_evaluate_when.py`:

```python
"""W5: _evaluate_when 扩展 total_subjects 变量。"""
from __future__ import annotations

from ethoinsight.catalog.resolve import _evaluate_when


def test_total_subjects_passes_when_threshold_met():
    assert _evaluate_when("total_subjects >= 1", n_per_group=None, n_groups=None, total_subjects=1) is True
    assert _evaluate_when("total_subjects >= 1", n_per_group=None, n_groups=None, total_subjects=10) is True


def test_total_subjects_fails_when_threshold_not_met():
    assert _evaluate_when("total_subjects >= 3", n_per_group=None, n_groups=None, total_subjects=1) is False


def test_total_subjects_none_evaluates_false():
    assert _evaluate_when("total_subjects >= 1", n_per_group=None, n_groups=None, total_subjects=None) is False


def test_compound_with_total_subjects():
    assert _evaluate_when("n_per_group >= 1 and total_subjects >= 1",
                          n_per_group=1, n_groups=None, total_subjects=1) is True
    assert _evaluate_when("n_per_group >= 1 and total_subjects >= 3",
                          n_per_group=1, n_groups=None, total_subjects=1) is False


def test_backward_compat_n_per_group_still_works():
    """W5 不允许破坏现有 n_per_group / n_groups 语义。"""
    assert _evaluate_when("n_per_group >= 3", n_per_group=3, n_groups=None, total_subjects=None) is True
    assert _evaluate_when("n_per_group >= 3", n_per_group=2, n_groups=None, total_subjects=None) is False
    assert _evaluate_when("n_groups >= 2", n_per_group=None, n_groups=2, total_subjects=None) is True
    assert _evaluate_when("always", n_per_group=None, n_groups=None, total_subjects=None) is True


def test_n_per_group_call_without_total_subjects_kwarg_still_works():
    """既有 resolve() 调用未传 total_subjects 时不应崩。
    实现策略:total_subjects=None 是默认值。
    """
    assert _evaluate_when("n_per_group >= 1", n_per_group=1, n_groups=None) is True
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd packages/ethoinsight
PYTHONPATH=. uv run pytest tests/test_evaluate_when.py -v
```

Expected: TypeError "unexpected keyword 'total_subjects'" 或 last test 中 var=='total_subjects' 永远返 False。

- [ ] **Step 3: 绿 — 实现**

Edit `packages/ethoinsight/ethoinsight/catalog/resolve.py:295-326`:

```python
def _evaluate_when(
    condition: str, *, n_per_group: int | None, n_groups: int | None,
    total_subjects: int | None = None,
) -> bool:
    cond = condition.strip()
    if cond == "always":
        return True

    parts = [p.strip() for p in cond.split(" and ")]
    for part in parts:
        if not _evaluate_atomic_when(
            part, n_per_group=n_per_group, n_groups=n_groups, total_subjects=total_subjects,
        ):
            return False
    return True


def _evaluate_atomic_when(
    part: str, *, n_per_group: int | None, n_groups: int | None,
    total_subjects: int | None = None,
) -> bool:
    tokens = part.split()
    if len(tokens) != 3:
        return False
    var, op, val_str = tokens
    if op != ">=":
        return False
    try:
        val = int(val_str)
    except ValueError:
        return False
    if var == "n_per_group":
        return n_per_group is not None and n_per_group >= val
    if var == "n_groups":
        return n_groups is not None and n_groups >= val
    if var == "total_subjects":
        return total_subjects is not None and total_subjects >= val
    return False
```

注意:`resolve()` 函数现状(line 54-229)调用 `_evaluate_when(ch.when, n_per_group=..., n_groups=...)`(line 184, 190)未传 total_subjects 是 OK 的(默认 None,只 affect total_subjects-based expressions)。**W5 不改 resolve() 主体** — 让 W4 改。

- [ ] **Step 4: 跑测试确认通过**

```bash
PYTHONPATH=. uv run pytest tests/test_evaluate_when.py -v
PYTHONPATH=. uv run pytest tests/ -k "catalog or resolve" -v
```

Expected: 全绿,既有 resolve 测试不破坏(`_evaluate_when` 默认 `total_subjects=None`)。

- [ ] **Step 5: Commit**

```bash
git add packages/ethoinsight/ethoinsight/catalog/resolve.py \
        packages/ethoinsight/tests/test_evaluate_when.py
git commit -m "$(cat <<'EOF'
feat(catalog): _evaluate_when 扩展 total_subjects 变量 (WI5)

新表达式 'total_subjects >= K' 支持单被试 / 跨组合并场景的 chart fallback
when 条件判断。

向后兼容:total_subjects 是默认 None 关键字参数,既有 resolve()
调用不传也能跑。

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §7.4
EOF
)"
```

---

## Task 6 (W6): plot_timeseries.py CLI 包装

**Files:**
- Create: `packages/ethoinsight/ethoinsight/scripts/_common/plot_timeseries.py`
- Test: `packages/ethoinsight/tests/test_plot_timeseries_cli.py` (new)

**依赖:** 无,但与 W3 配对(C7:YAML 注册与脚本同 PR);本 task 也要参考 W3 落地的 `_common.yaml` entry 命名

**Spec refs:** §7.8

**预读**:`packages/ethoinsight/ethoinsight/scripts/_common/plot_trajectory.py` 是参考样例。`packages/ethoinsight/ethoinsight/charts.py` 包含 `timeseries_plot` 库函数(待 W6 实施 agent 确认存在或决定新建)。

- [ ] **Step 0: 探查 ethoinsight.charts 是否有 timeseries_plot 函数**

```bash
cd packages/ethoinsight
grep -n "def timeseries_plot\|def plot_timeseries" ethoinsight/charts.py 2>&1 | head -5
```

**两种情况**:
- (A) `timeseries_plot` 已存在 → CLI 仅 wrap
- (B) 不存在 → W6 同时落地 charts.py 的 `timeseries_plot` 库函数(单 subject 时间序列图,默认 y_col 按 paradigm 查 `_DEFAULT_Y_COL_BY_PARADIGM`)

**W6 实施 agent 决策点**:如果 (B),按 spec §7.8 加 `_DEFAULT_Y_COL_BY_PARADIGM` 映射(EPM 默认 `open_arm_time_ratio`,OFT 默认 `center_time_ratio`,其他默认 `distance_moved`)。

- [ ] **Step 1: 红 — 写失败测试**

`packages/ethoinsight/tests/test_plot_timeseries_cli.py`:

```python
"""W6: plot_timeseries CLI 包装。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def sample_trajectory_file(tmp_path):
    """Minimal EthoVision-like CSV that parse_trajectory can consume."""
    f = tmp_path / "subject1.txt"
    # 用 demo-data 中的真实示例 fixture(由实施 agent 复制)
    # 或者最小 inline CSV:Trial time + X center + Y center
    f.write_text(
        "Trial time,X center,Y center\n"
        "0.0,1.0,1.0\n"
        "0.1,1.1,1.0\n"
        "0.2,1.2,1.1\n",
        encoding="utf-8",
    )
    return f


def test_plot_timeseries_cli_runs_with_single_input(sample_trajectory_file, tmp_path):
    output = tmp_path / "out.png"
    result = subprocess.run(
        [
            sys.executable, "-m", "ethoinsight.scripts._common.plot_timeseries",
            "--input", str(sample_trajectory_file),
            "--output", str(output),
        ],
        capture_output=True, text=True,
    )
    # 输出 png + emit [result] line
    if result.returncode != 0:
        pytest.fail(f"CLI failed: stderr={result.stderr}")
    assert output.exists(), f"png not generated: stderr={result.stderr}"
    assert "[result]" in result.stdout


def test_plot_timeseries_cli_accepts_y_col_override(sample_trajectory_file, tmp_path):
    """--y-col 参数显式指定 y 轴列名时不报错。"""
    output = tmp_path / "out.png"
    result = subprocess.run(
        [
            sys.executable, "-m", "ethoinsight.scripts._common.plot_timeseries",
            "--input", str(sample_trajectory_file),
            "--output", str(output),
            "--y-col", "X center",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI with --y-col failed: stderr={result.stderr}")
    assert output.exists()
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONPATH=. uv run pytest tests/test_plot_timeseries_cli.py -v
```

Expected: `No module named 'ethoinsight.scripts._common.plot_timeseries'`。

- [ ] **Step 3: 绿 — 实现 plot_timeseries.py**

Create `packages/ethoinsight/ethoinsight/scripts/_common/plot_timeseries.py`:

```python
"""通用:时间序列图(单 subject 或多 subject 叠加)。

CLI:
  单文件: python -m ethoinsight.scripts._common.plot_timeseries \
            --input <轨迹文件> --output <png> [--y-col <列名>]
  多文件: python -m ethoinsight.scripts._common.plot_timeseries \
            --inputs <inputs.json> --output <png> [--y-col <列名>]

--y-col 未传时,按 paradigm 默认 y_col 映射(--paradigm 也未传时回退 distance_moved)。

输出:PNG 图像文件 + stdout `[result] {...}` 行。
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ethoinsight.charts import timeseries_plot
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, read_inputs_json


# Spec §7.8 决策 6:范式 → 默认 y_col 映射。
# chart-maker skill 决策时可以通过 --y-col 显式覆盖。
_DEFAULT_Y_COL_BY_PARADIGM: dict[str, str] = {
    "epm": "open_arm_time_ratio",
    "oft": "center_time_ratio",
    "zero_maze": "open_arm_time_ratio",
    # 其他 paradigm 没明确"主指标",fallback 到通用 distance_moved
}
_GLOBAL_DEFAULT_Y_COL = "distance_moved"


def _resolve_y_col(args: argparse.Namespace) -> str:
    if args.y_col:
        return args.y_col
    if args.paradigm and args.paradigm in _DEFAULT_Y_COL_BY_PARADIGM:
        return _DEFAULT_Y_COL_BY_PARADIGM[args.paradigm]
    return _GLOBAL_DEFAULT_Y_COL


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ethoinsight.scripts._common.plot_timeseries",
                                 description="时间序列图 CLI 包装")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="单文件 trajectory")
    src.add_argument("--inputs", help="多文件 JSON 数组路径")
    ap.add_argument("--output", required=True, help="输出 PNG 路径")
    ap.add_argument("--y-col", default=None, help="y 轴列名(可选,默认按 paradigm 选)")
    ap.add_argument("--paradigm", default=None, help="paradigm 名(用于决定默认 y_col)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    y_col = _resolve_y_col(args)

    if args.input:
        df = parse_trajectory(args.input)
    else:
        paths = read_inputs_json(args.inputs)
        dfs = []
        for p in paths:
            sub_df = parse_trajectory(p)
            subject_attr = sub_df.attrs.get("subject", p)
            sub_df = sub_df.assign(subject=subject_attr)
            dfs.append(sub_df)
        df = pd.concat(dfs, ignore_index=True)

    output_path = timeseries_plot(df, y_col=y_col, output_path=args.output)
    emit_result({"plot": "timeseries", "path": output_path, "y_col": y_col})
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

如果 Step 0 发现 `timeseries_plot` 不存在,W6 实施 agent 必须**同 commit** 在 `ethoinsight/charts.py` 加 `timeseries_plot(df, *, y_col, output_path)` 函数(matplotlib 画时间序列,subject 多 → 不同颜色),否则 import 会失败。

- [ ] **Step 4: 跑测试确认通过**

```bash
PYTHONPATH=. uv run pytest tests/test_plot_timeseries_cli.py -v
```

Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add packages/ethoinsight/ethoinsight/scripts/_common/plot_timeseries.py \
        packages/ethoinsight/tests/test_plot_timeseries_cli.py
# 若同时改了 charts.py 也要 add
git add -u packages/ethoinsight/ethoinsight/charts.py 2>/dev/null || true
git commit -m "$(cat <<'EOF'
feat(scripts): plot_timeseries CLI 包装 + 默认 y_col 按 paradigm 映射 (WI6)

通用时间序列图脚本,服务 chart-maker 单被试 fallback 场景。

--y-col 未显式传时,按 _DEFAULT_Y_COL_BY_PARADIGM 映射选默认列:
- EPM/zero_maze → open_arm_time_ratio
- OFT → center_time_ratio
- 其他 → distance_moved

C7 与 _common.yaml(WI3)同 PR 落地。

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §7.8
EOF
)"
```

---

## Task 7 (W4): catalog CLI `--mode` + resolve_charts 函数 + 拆 resolve_metrics

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/catalog/cli.py` (add `--mode`, `--user-intent`, `--total-subjects`)
- Modify: `packages/ethoinsight/ethoinsight/catalog/resolve.py` (拆 `resolve` → `resolve_metrics` + new `resolve_charts`)
- Test: `packages/ethoinsight/tests/test_resolve_charts.py` (new)
- Test: `packages/ethoinsight/tests/test_catalog_cli_modes.py` (new)

**依赖:** W2 (PlanMetrics, PlanCharts), W3 (load_common_catalog), W5 (_evaluate_when total_subjects)

**Spec refs:** §7.5 / §7.6

**Design 决策**:
- `resolve_metrics(...)` 等价于现有 `resolve(...)` 行为(返回 PlanMetrics,无 charts)。为了不破坏 prep_metric_plan_tool / 现有调用者,**保留旧 `resolve()` 作为 thin wrapper**:它内部调 `resolve_metrics(...)` 再补一个空 `plan.charts=[]` 字段(W11/W20 完成后再彻底删 `resolve`)。
- `resolve_charts(...)` 是新函数,返回 PlanCharts,负责 fallback 决策。

### Fallback 触发条件(spec §7.6 注释复审决定)

- 主路径:`for ch in cat.charts: if _evaluate_when(ch.when, ..., n_per_group, n_groups, total_subjects)` → charts.append
- Fallback 触发:`if len(charts) == 0`(不要求 cat.charts 原本非空 — 单被试场景 cat.charts 常为空也应 fallback)
- Fallback 路径:`load_common_catalog() → for ch in common_charts: if _evaluate_when(ch.when, ..., total_subjects=...)` → charts_fallback_available.append

- [ ] **Step 1: 红 — 写 resolve_charts 失败测试**

`packages/ethoinsight/tests/test_resolve_charts.py`:

```python
"""W4: resolve_charts 函数验收。"""
from __future__ import annotations

from pathlib import Path

import pytest

from ethoinsight.catalog.resolve import resolve_charts, ResolveError


# 假设 epm catalog YAML 至少含 1 个 chart with `when: n_per_group >= 3`,
# 让单被试场景能触发 fallback。
EPM_COLUMNS_SAMPLE = [
    "Trial time", "X center", "Y center",
    "In zone(Open arms / center-point)", "In zone(Closed arms / center-point)",
]
RAW_FILES_SAMPLE = ["/tmp/raw1.txt"]


def test_single_subject_triggers_fallback(tmp_path):
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path),
        user_intent="再画几个图",
        total_subjects=1, n_per_group=1, n_groups=1,
    )
    # 单被试,EPM 的 catalog charts (when: n_per_group >= 3) 都不满足 → charts 为空
    # → fallback 路径触发 → charts_fallback_available 含 trajectory + timeseries
    assert pc.charts == [], f"expected no main charts for single subject, got {pc.charts}"
    fallback_ids = {c.id for c in pc.charts_fallback_available}
    assert "trajectory_plot" in fallback_ids
    assert "timeseries_plot" in fallback_ids
    assert pc.user_intent == "再画几个图"
    assert pc.paradigm == "epm"


def test_group_data_uses_catalog_charts_not_fallback(tmp_path):
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path),
        user_intent=None,
        total_subjects=6, n_per_group=3, n_groups=2,
    )
    # 6 个 subject,3v3 满足 catalog charts 的 when → charts 非空 → fallback 不触发
    assert len(pc.charts) > 0, "expected catalog charts for 3v3 EPM"
    assert pc.charts_fallback_available == [], (
        "fallback should NOT trigger when main charts present"
    )


def test_unknown_paradigm_raises_resolve_error(tmp_path):
    with pytest.raises(ResolveError) as exc:
        resolve_charts(
            paradigm="nonexistent_paradigm", columns=[], raw_files=[],
            workspace_dir=str(tmp_path), total_subjects=1,
        )
    assert exc.value.code == "unknown_paradigm"


def test_plan_charts_schema_version(tmp_path):
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path), total_subjects=1,
    )
    assert pc.schema_version == "1.0"
```

- [ ] **Step 2: 红 — 写 CLI mode 失败测试**

`packages/ethoinsight/tests/test_catalog_cli_modes.py`:

```python
"""W4: catalog CLI --mode metrics / charts。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def workspace(tmp_path):
    cols_file = tmp_path / "columns.json"
    cols_file.write_text(json.dumps({"columns": [
        "Trial time", "X center", "Y center",
        "In zone(Open arms / center-point)",
        "In zone(Closed arms / center-point)",
    ]}), encoding="utf-8")
    raws_file = tmp_path / "raws.json"
    raws_file.write_text(json.dumps([str(tmp_path / "raw1.txt")]), encoding="utf-8")
    return tmp_path, cols_file, raws_file


def test_mode_metrics_writes_plan_metrics_json(workspace):
    ws, cols, raws = workspace
    out = ws / "plan_metrics.json"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.catalog.resolve",
         "--mode", "metrics",
         "--paradigm", "epm",
         "--columns-file", str(cols),
         "--raw-files-json", str(raws),
         "--workspace-dir", str(ws),
         "--output", str(out),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "metrics" in payload
    assert "charts" not in payload   # plan_metrics 不再含 charts


def test_mode_charts_writes_plan_charts_json(workspace):
    ws, cols, raws = workspace
    out = ws / "plan_charts.json"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.catalog.resolve",
         "--mode", "charts",
         "--paradigm", "epm",
         "--user-intent", "再画几个图",
         "--total-subjects", "1", "--n-per-group", "1", "--n-groups", "1",
         "--columns-file", str(cols),
         "--raw-files-json", str(raws),
         "--workspace-dir", str(ws),
         "--output", str(out),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "charts" in payload
    assert "charts_fallback_available" in payload
    assert payload["user_intent"] == "再画几个图"
    assert "metrics" not in payload


def test_default_mode_is_metrics_for_backward_compat(workspace):
    """未传 --mode 时,默认 metrics 模式,既有调用者(如 prep_metric_plan)继续工作。"""
    ws, cols, raws = workspace
    out = ws / "plan_legacy.json"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.catalog.resolve",
         "--paradigm", "epm",
         "--columns-file", str(cols),
         "--raw-files-json", str(raws),
         "--workspace-dir", str(ws),
         "--output", str(out),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "metrics" in payload
```

- [ ] **Step 3: 跑测试确认失败**

```bash
cd packages/ethoinsight
PYTHONPATH=. uv run pytest tests/test_resolve_charts.py tests/test_catalog_cli_modes.py -v
```

Expected: ImportError `resolve_charts`、`unrecognized arguments: --mode --user-intent --total-subjects`。

- [ ] **Step 4: 绿 — 实现 resolve_charts**

Edit `packages/ethoinsight/ethoinsight/catalog/resolve.py`:

(1) 新增 import 顶部:

```python
from ethoinsight.catalog.loader import (
    CatalogError, CommonCatalog, load_catalog, load_common_catalog,
)
from ethoinsight.catalog.schema import (
    ChartEntry, MetricEntry, Plan, PlanChart, PlanCharts, PlanInputs,
    PlanMetric, PlanMetrics, PlanSkipped, PlanStatistics, StatisticsEntry,
)
```

(2) 在文件末尾追加新函数:

```python
def resolve_charts(
    paradigm: str,
    columns: list[str],
    raw_files: list[str],
    workspace_dir: str,
    *,
    user_intent: str | None = None,
    total_subjects: int | None = None,
    n_per_group: int | None = None,
    n_groups: int | None = None,
    groups_file: str | None = None,
    columns_file: str | None = None,
    ev19_template: str | None = None,
    virtual_workspace_dir: str | None = None,
) -> PlanCharts:
    """生成 PlanCharts。

    Step:
      1. load_catalog(paradigm) → cat.charts
      2. for ch in cat.charts: if _evaluate_when → charts.append
      3. if not charts (fallback 触发):
           load_common_catalog() → for ch in common_charts:
             if _evaluate_when(ch.when, total_subjects=...) → fallback.append
      4. 组装 PlanCharts dataclass
    """
    try:
        cat = load_catalog(paradigm)
    except CatalogError as e:
        if (
            "file not found" in str(e).lower()
            or "not found for paradigm" in str(e).lower()
        ):
            raise ResolveError(
                code="unknown_paradigm",
                message=f"Unknown paradigm '{paradigm}'.",
                details={"requested": paradigm},
            ) from e
        raise ResolveError(
            code="schema_violation",
            message=f"Catalog YAML for '{paradigm}' is malformed: {e}",
            details={"paradigm": paradigm},
        ) from e

    effective_workspace = virtual_workspace_dir or workspace_dir
    charts: list[PlanChart] = []
    for ch in cat.charts:
        if _evaluate_when(
            ch.when, n_per_group=n_per_group, n_groups=n_groups, total_subjects=total_subjects,
        ):
            charts.append(_chart_to_plan(ch, raw_files, workspace_dir,
                                          virtual_workspace_dir=virtual_workspace_dir))

    fallback: list[PlanChart] = []
    if not charts:
        # Fallback 触发
        try:
            common = load_common_catalog()
        except CatalogError:
            # _common.yaml 缺失或损坏不致命 — 返回空 fallback,让 chart-maker 反问
            common = CommonCatalog(common_charts=[])
        for ch in common.common_charts:
            if _evaluate_when(
                ch.when, n_per_group=n_per_group, n_groups=n_groups, total_subjects=total_subjects,
            ):
                fallback.append(_chart_to_plan(ch, raw_files, workspace_dir,
                                                virtual_workspace_dir=virtual_workspace_dir))

    notes: list[str] = []
    if charts:
        notes.append(f"Generated {len(charts)} catalog charts")
    elif fallback:
        notes.append(f"Fallback path: {len(fallback)} common charts available")
    else:
        notes.append("No charts matched; chart-maker should ask user")

    return PlanCharts(
        paradigm=cat.paradigm,
        ev19_template=ev19_template,
        generated_at=_utcnow_iso(),
        inputs=PlanInputs(
            raw_files=list(raw_files),
            groups_file=groups_file,
            columns_file=columns_file,
        ),
        charts=charts,
        charts_fallback_available=fallback,
        skipped=[],   # charts 层面 skip(when 不满足)目前不单独记录;
                     # 主路径只放命中的 chart,fallback 放未命中但可用的 common chart。
                     # W22 dogfood 后视需要再加详细 skip 记录。
        user_intent=user_intent,
        notes=notes,
    )


def plan_charts_to_dict(plan: PlanCharts) -> dict:
    return {
        "schema_version": plan.schema_version,
        "paradigm": plan.paradigm,
        "ev19_template": plan.ev19_template,
        "generated_at": plan.generated_at,
        "inputs": {
            "raw_files": plan.inputs.raw_files,
            "groups_file": plan.inputs.groups_file,
            "columns_file": plan.inputs.columns_file,
        },
        "charts": [
            {"id": c.id, "script": c.script, "input": c.input, "output": c.output}
            for c in plan.charts
        ],
        "charts_fallback_available": [
            {"id": c.id, "script": c.script, "input": c.input, "output": c.output}
            for c in plan.charts_fallback_available
        ],
        "skipped": [
            {"id": s.id, "reason": s.reason, "detail": s.detail} for s in plan.skipped
        ],
        "user_intent": plan.user_intent,
        "notes": plan.notes,
    }
```

注意:`_chart_to_plan` 已存在(line 268),复用即可。

- [ ] **Step 5: 绿 — 改 CLI 加 --mode**

Edit `packages/ethoinsight/ethoinsight/catalog/cli.py:76-90`:

```python
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ethoinsight.catalog.resolve")
    p.add_argument("--mode", choices=["metrics", "charts"], default="metrics",
                   help="metrics(默认,兼容)| charts(W4 新增)")
    p.add_argument("--paradigm", required=True)
    p.add_argument("--columns-file", required=True)
    p.add_argument("--raw-files-json", required=True)
    p.add_argument("--workspace-dir", required=True)
    p.add_argument("--virtual-workspace-dir", default=None)
    p.add_argument("--groups-file", default=None)
    p.add_argument("--include", action="append", default=[])
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--n-per-group", type=int, default=None)
    p.add_argument("--n-groups", type=int, default=None)
    p.add_argument("--total-subjects", type=int, default=None,
                   help="单被试场景 fallback 触发依据(W4 新增)")
    p.add_argument("--user-intent", default=None,
                   help="用户语义原话(chart-maker 用,W4 新增)")
    p.add_argument("--ev19-template", default=None)
    p.add_argument("--output", required=True)
    return p
```

(2) 在 main() 内根据 args.mode 分支:

```python
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Read columns.json + raw-files-json (现有逻辑保留)
    # ... (line 102-135 不变)

    virtual_workspace_dir = _resolve_virtual_workspace_dir(
        args.virtual_workspace_dir, args.workspace_dir
    )

    try:
        if args.mode == "metrics":
            plan = resolve_metrics(  # 改名:旧 resolve → resolve_metrics
                paradigm=args.paradigm,
                columns=columns,
                raw_files=raw_files,
                workspace_dir=args.workspace_dir,
                include=args.include,
                exclude=args.exclude,
                n_per_group=args.n_per_group,
                n_groups=args.n_groups,
                groups_file=args.groups_file,
                columns_file=args.columns_file,
                ev19_template=args.ev19_template,
                virtual_workspace_dir=virtual_workspace_dir,
            )
            plan_dict = plan_metrics_to_dict(plan)   # 改名:plan_to_dict → plan_metrics_to_dict
        else:  # mode == "charts"
            plan = resolve_charts(
                paradigm=args.paradigm,
                columns=columns,
                raw_files=raw_files,
                workspace_dir=args.workspace_dir,
                user_intent=args.user_intent,
                total_subjects=args.total_subjects,
                n_per_group=args.n_per_group,
                n_groups=args.n_groups,
                groups_file=args.groups_file,
                columns_file=args.columns_file,
                ev19_template=args.ev19_template,
                virtual_workspace_dir=virtual_workspace_dir,
            )
            plan_dict = plan_charts_to_dict(plan)
    except ResolveError as e:
        return _emit_error(e.code, str(e), e.details)
    except Exception as e:
        return _emit_error("schema_violation", f"Unexpected resolver failure: {e}", {})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # mode-specific summary
    if args.mode == "metrics":
        summary = (
            f"Plan written to {args.output}: paradigm={plan.paradigm}, "
            f"metrics={len(plan.metrics)}, skipped={len(plan.skipped)}, statistics="
            f"{'skip' if (plan.statistics and plan.statistics.skip_reason) else 'run' if plan.statistics else 'none'}"
        )
    else:
        summary = (
            f"Plan written to {args.output}: paradigm={plan.paradigm}, "
            f"charts={len(plan.charts)}, fallback_available={len(plan.charts_fallback_available)}"
        )
    print(summary)
    return 0
```

- [ ] **Step 6: 绿 — 拆 resolve → resolve_metrics**

Edit `packages/ethoinsight/ethoinsight/catalog/resolve.py:54`:

```python
def resolve_metrics(
    paradigm: str,
    columns: list[str],
    raw_files: list[str],
    workspace_dir: str,
    *,
    include: list[str] | tuple[str, ...] = (),
    exclude: list[str] | tuple[str, ...] = (),
    n_per_group: int | None = None,
    n_groups: int | None = None,
    groups_file: str | None = None,
    columns_file: str | None = None,
    ev19_template: str | None = None,
    virtual_workspace_dir: str | None = None,
) -> PlanMetrics:
    """生成 PlanMetrics dataclass(原 resolve 拆出,去掉 charts 段)。

    workflow:
      1. load_catalog
      2. process default_metrics
      3. process user-include
      4. resolve statistics
      5. notes
    (NO chart step — chart-maker 自己跑 resolve_charts)
    """
    # 把现有 resolve() (line 54-229) 的实现复制过来,
    # **删掉 line 181-185 的 "Step 4: charts" 段**,return 改为 PlanMetrics(无 charts 字段)。
    # ... (实施时按现有 resolve 全文复制 + 删 charts 段)
```

旧 `resolve(...)` 保留作 wrapper(向后兼容现有调用者:`prep_metric_plan_tool.py` 仍调 `resolve`,W20 才彻底切走):

```python
def resolve(*args, **kwargs) -> Plan:
    """Backward-compat wrapper. Calls resolve_metrics + returns legacy Plan
    (with empty charts list). DEPRECATED — use resolve_metrics/resolve_charts.
    Will be removed in W20+W22.
    """
    pm = resolve_metrics(*args, **kwargs)
    return Plan(
        schema_version=pm.schema_version,
        paradigm=pm.paradigm,
        ev19_template=pm.ev19_template,
        generated_at=pm.generated_at,
        inputs=pm.inputs,
        metrics=pm.metrics,
        statistics=pm.statistics,
        charts=[],   # 老 callers 假定有 charts 字段
        skipped=pm.skipped,
        notes=pm.notes,
    )


def plan_metrics_to_dict(plan: PlanMetrics) -> dict:
    """Serialize PlanMetrics → JSON dict (NO charts field)."""
    return {
        "schema_version": plan.schema_version,
        "paradigm": plan.paradigm,
        "ev19_template": plan.ev19_template,
        "generated_at": plan.generated_at,
        "inputs": {
            "raw_files": plan.inputs.raw_files,
            "groups_file": plan.inputs.groups_file,
            "columns_file": plan.inputs.columns_file,
        },
        "metrics": [
            {
                "id": m.id, "script": m.script, "input": m.input,
                "output": m.output, "required": m.required, "reason": m.reason,
            } for m in plan.metrics
        ],
        "statistics": (
            None if plan.statistics is None else {
                "id": plan.statistics.id, "script": plan.statistics.script,
                "input": plan.statistics.input, "output": plan.statistics.output,
                "skip_reason": plan.statistics.skip_reason,
            }
        ),
        "skipped": [{"id": s.id, "reason": s.reason, "detail": s.detail} for s in plan.skipped],
        "notes": plan.notes,
    }


# 老 plan_to_dict 仍保留作 wrapper:
def plan_to_dict(plan: Plan) -> dict:
    """DEPRECATED: 旧 Plan dataclass serializer。W20 切到 plan_metrics_to_dict 后删除。"""
    d = plan_metrics_to_dict_compat(plan)  # 见下
    d["charts"] = [
        {"id": c.id, "script": c.script, "input": c.input, "output": c.output}
        for c in plan.charts
    ]
    return d


def plan_metrics_to_dict_compat(plan) -> dict:
    """Variant that handles legacy Plan(with charts field) by ignoring charts.
    Used by plan_to_dict wrapper only.
    """
    return {
        "schema_version": plan.schema_version,
        "paradigm": plan.paradigm,
        "ev19_template": plan.ev19_template,
        "generated_at": plan.generated_at,
        "inputs": {
            "raw_files": plan.inputs.raw_files,
            "groups_file": plan.inputs.groups_file,
            "columns_file": plan.inputs.columns_file,
        },
        "metrics": [
            {"id": m.id, "script": m.script, "input": m.input,
             "output": m.output, "required": m.required, "reason": m.reason}
            for m in plan.metrics
        ],
        "statistics": (
            None if plan.statistics is None else {
                "id": plan.statistics.id, "script": plan.statistics.script,
                "input": plan.statistics.input, "output": plan.statistics.output,
                "skip_reason": plan.statistics.skip_reason,
            }
        ),
        "skipped": [{"id": s.id, "reason": s.reason, "detail": s.detail} for s in plan.skipped],
        "notes": plan.notes,
    }
```

- [ ] **Step 7: 跑测试确认通过**

```bash
PYTHONPATH=. uv run pytest tests/test_resolve_charts.py tests/test_catalog_cli_modes.py tests/test_evaluate_when.py tests/test_common_catalog.py tests/test_catalog_schema_plan_split.py -v
PYTHONPATH=. uv run pytest tests/ -v 2>&1 | tail -20
```

Expected: 新测试全绿,既有 catalog/CLI 测试全绿(因为 `resolve` wrapper 兼容)。

- [ ] **Step 8: Commit**

```bash
git add packages/ethoinsight/ethoinsight/catalog/cli.py \
        packages/ethoinsight/ethoinsight/catalog/resolve.py \
        packages/ethoinsight/tests/test_resolve_charts.py \
        packages/ethoinsight/tests/test_catalog_cli_modes.py
git commit -m "$(cat <<'EOF'
feat(catalog): --mode {metrics,charts} CLI + resolve_charts (WI4)

CLI:
- 新增 --mode (默认 metrics 兼容旧调用方)
- 新增 --user-intent / --total-subjects(charts 模式专用)
- mode=charts 写 plan_charts.json,含 charts + charts_fallback_available + user_intent
- mode=metrics 写 plan_metrics.json,无 charts 字段

API:
- 拆 resolve() → resolve_metrics() + resolve_charts()
- 旧 resolve() 保留作 wrapper(W20 prep_metric_plan_tool 切走后删)
- resolve_charts fallback 触发:len(charts)==0 → 加载 _common.yaml 候选

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §7.5/§7.6
EOF
)"
```

---

## Task 8 (W8): ethoinsight-lead-interaction skill 新建

**Files:**
- Create: `packages/agent/skills/custom/ethoinsight-lead-interaction/SKILL.md`
- Create: `packages/agent/skills/custom/ethoinsight-lead-interaction/references/intent-decision-tree.md`
- Create: `packages/agent/skills/custom/ethoinsight-lead-interaction/references/paradigm-identification.md`
- Create: `packages/agent/skills/custom/ethoinsight-lead-interaction/references/clarification-templates.md`

**依赖:** W7 (引用 execution-conventions.md 的格式约定)

**Spec refs:** §8.5

无单元测试(纯 markdown);质量靠 W16 实施时 read 该文件能 actionable 来验证。

- [ ] **Step 1: 创建 SKILL.md**

`packages/agent/skills/custom/ethoinsight-lead-interaction/SKILL.md`:

```markdown
---
name: ethoinsight-lead-interaction
description: >
  EthoInsight lead agent 交互手册:意图分类、范式识别、反问模板、调度规则。
  Lead 持有"如何与用户交互"的 know-how,subagent 持有"如何执行"的 know-how。
  本 skill 是 lead 的副本知识库,补完 lead system prompt 瘦身后(~200 行)留下的细节。
version: 1.0.0
author: noldus-insight
---

# Lead Agent 交互手册

## 用户意图 7 分类(决策树见 references/intent-decision-tree.md)

| 意图 | 触发条件 | 派遣链 |
|---|---|---|
| `E2E_FULL` | 上传数据 + 用户语义为"分析并画图/出报告/全套" | code-executor → data-analyst → chart-maker → ask_clarification(report?) |
| `E2E_MIN` | 上传数据 + 用户语义为"分析一下" | code-executor → data-analyst → ask_clarification(4-choice) |
| `CHART` | 已有 handoff + 用户要图 | chart-maker(单派) |
| `REPORT` | 已有 handoff + 用户要报告 | report-writer(单派) |
| `QA_FACT` | 已有 handoff + 追问具体数据 | knowledge-assistant(授权 handoff 占位符) |
| `QA_KNOWLEDGE` | 领域知识 / 概念问题 | knowledge-assistant(不授权 handoff) |
| `CLARIFY` | 意图模糊 / 缺范式 / 缺数据列 | `ask_clarification`(不派 subagent) |

## 意图分类硬规则

- **第一个非 read_file tool call 之前**,lead 必须输出 `[intent] <INTENT_NAME>` 行,被 `IntentClassificationGuardrailProvider` 拦截校验
- 不能默认猜测意图,模糊时 → `CLARIFY`

## 范式识别(详见 references/paradigm-identification.md)

1. read `ethovision-paradigm-knowledge` skill 的 SKILL.md 决策树
2. 用 skill 知识 + 用户消息 + 文件名推断 EV19 模板变体(**不 read raw txt**)
3. 唯一高置信 → `set_experiment_paradigm` 落盘
4. 多候选 / 无法分辨 → `ask_clarification` 带证据反问
5. 用户确认 → 落盘 → 派 subagent

## 反问模板(详见 references/clarification-templates.md)

- 4-choice ask_clarification(E2E_MIN 后):看结果就够了 / 加图 / 加报告 / 都要
- 范式候选反问:"我从数据中看到 X / Y / Z 列,可能是 EPM 或 OFT,请确认实验类型"
- 数据列缺失反问:"数据缺关键列 X(用于计算 metric Y),请确认导出设置是否漏了 zone configuration"

## Capability-exposure 调度规则

**Lead 不知道**:
- 各 subagent 内部脚本路径 / handoff JSON schema / 图种选择逻辑 / 指标计算细节

**Lead 知道**(通过 SubagentConfig 注入到 system prompt 的 capability 表):
- 每个 subagent 的 description / when_to_use / input_contract / output_contract

**Lead 派遣时**:
- 派 task() 不写 `{{handoff://X}}` 占位符 — harness task_tool 按 SubagentConfig.required_upstream_handoffs 自动注入
- task prompt 用用户语言原话 + 简短引导(见 SubagentConfig.input_contract 示例)
- task 调用前必须输出 `[intent] <INTENT>` 行

## 意图转移规则

哪些用户消息会让 lead 重新分类意图:

- 上一意图 == CHART 完成 → 用户说"再给个报告" → REPORT
- 上一意图 == E2E_MIN 完成,ask_clarification 已问 → 用户选"都要" → 派 chart-maker 后再派 report-writer
- 任何意图中 → 用户切换数据(上传新文件)→ 重新走 E2E_FULL/E2E_MIN 分类

## 不要做的事

- ❌ 不要自己 `write_file` 写 Python 脚本(`bash` / `write_file` / `str_replace` 已从 lead tool 移除,但即便加回也不要)
- ❌ 不要 `read_file` raw EthoVision txt 文件(交给 ethoinsight 库解析)
- ❌ 不要默认猜测范式(范式不明 → ask_clarification)
- ❌ 不要替 subagent 决定"该跑哪些图/指标"(那是 subagent + catalog 的职责)
```

- [ ] **Step 2: 创建 references/intent-decision-tree.md**

```markdown
# 意图分类决策树

```
[user message arrives]
   │
   ├── 有 <uploaded_files> 的 "uploaded in this message"?
   │     │
   │     ├── Yes + 用户语义包含"画图"+"报告"+"全套" → E2E_FULL
   │     ├── Yes + 用户语义为"分析一下" / "看看" → E2E_MIN
   │     ├── Yes + 用户只问候(无分析意图)→ QA_KNOWLEDGE(派 knowledge-assistant 让它询问意图)
   │     └── (转下一层)
   │
   ├── workspace 有 handoff_code_executor.json?
   │     │
   │     ├── Yes + 用户要图("画" / "trajectory" / "可视化")→ CHART
   │     ├── Yes + 用户要报告("出报告" / "Discussion" / "markdown")→ REPORT
   │     ├── Yes + 用户问数据(具体 p 值 / NND 偏高什么意思)→ QA_FACT
   │     └── (转下一层)
   │
   ├── 用户问领域知识(EPM 是什么 / 焦虑模型有哪些)→ QA_KNOWLEDGE
   │
   └── 信息不够(范式模糊 / 数据列缺 / 用户语义不清)→ CLARIFY
```

## 边界 case

- 上传数据 + "我先了解一下 EPM" → QA_KNOWLEDGE(用户在问知识,不是分析数据)
- 没上传 + "继续上次的分析" → 读 workspace 看是否有历史 handoff → 走 CHART/REPORT/QA_FACT 之一
- 上传新数据 + 已有旧 handoff → 重新走 E2E_FULL/E2E_MIN(新数据覆盖旧 handoff 是用户预期)
```

- [ ] **Step 3: 创建 references/paradigm-identification.md**

```markdown
# 范式识别流程

## 主决策

1. **read** `/mnt/skills/ethovision-paradigm-knowledge/SKILL.md` 拿 EV19 模板决策树
2. 综合证据(不 read raw txt):
   - 用户文字提示("EPM" / "open arms" / "fear conditioning"...)
   - 文件名("EPM_data.txt" / "shoaling_results.csv"...)
   - 已知 EV19 模板列表(skill 提供)
3. 决策:
   - 唯一高置信(单一 paradigm 评分 ≥ 阈值)→ `set_experiment_paradigm(paradigm, paradigm_cn, category, subject, ev19_template)`
   - 多候选(2-3 个范式都可能)→ `ask_clarification` 带证据反问
   - 完全不知道 → `ask_clarification` 让用户从范式列表中选

## 反问最多 1 次

如用户答"不知道":告诉用户"我需要范式信息才能选指标,请联系实验设计者确认",不要默认猜测(spec §13.5)。

## 已知 paradigm key 列表(来自 catalog)

`epm` / `oft` / `fst` / `tst` / `ldb` / `zero_maze` / `shoaling`

新增 paradigm 必须先在 `packages/ethoinsight/ethoinsight/catalog/<name>.yaml` 注册。
```

- [ ] **Step 4: 创建 references/clarification-templates.md**

```markdown
# ask_clarification 反问模板

## E2E_MIN 完成后 4-choice

```
我已完成 <N> 个被试的 <paradigm> 数据分析。指标摘要见上一条 message。
接下来你希望:

A. 看结果就够了(本次会话到此结束)
B. 加图表(我将派 chart-maker 生成可视化)
C. 加 markdown 报告(我将派 report-writer 写 6 段研究报告)
D. 都要(B + C 都做)
```

字段对应:`question_id="post_e2e_min"`, `options=["A","B","C","D"]`.

## 范式多候选反问

```
我从数据中看到 <列名 1>, <列名 2>, <列名 3> 列,文件名是 <文件名>。
这个数据可能是:

A. <候选范式 1>(<候选 1 的中文名>)
B. <候选范式 2>(<候选 2 的中文名>)
C. 其他 — 我会让你输入

请确认实验类型,我才能选对指标计算脚本。
```

## 数据列缺失反问

```
分析 <paradigm> 需要的关键列 <列名> 在你的数据中找不到。可能原因:

A. EthoVision 导出时漏选了对应 zone 的统计
B. 区域命名跟 <paradigm> 标准模板不一致
C. 这份数据不是 <paradigm> 范式

请检查 EthoVision 导出设置或重新导出后上传。
```

## 范式不明 + 用户也不知道(spec §13.5 Gate before guess)

```
对不起,我无法从文件名 / 列名推断实验范式,也没有得到你的明确确认。
为避免选错指标(导致分析结果误导你的实验解读),我必须暂停在这里。

请联系实验设计者确认范式名(EPM / OFT / FST / TST / LDB / zero_maze / shoaling),
或粘贴 EthoVision 导出时的 paradigm template 名称给我。
```

**不要**默认选最像的范式继续跑 — 错误的范式选错指标比报错暂停更糟。
```

- [ ] **Step 5: 验证文件存在 + lint**

```bash
ls packages/agent/skills/custom/ethoinsight-lead-interaction/
test -f packages/agent/skills/custom/ethoinsight-lead-interaction/SKILL.md
test -f packages/agent/skills/custom/ethoinsight-lead-interaction/references/intent-decision-tree.md
test -f packages/agent/skills/custom/ethoinsight-lead-interaction/references/paradigm-identification.md
test -f packages/agent/skills/custom/ethoinsight-lead-interaction/references/clarification-templates.md
```

- [ ] **Step 6: Commit**

```bash
git add packages/agent/skills/custom/ethoinsight-lead-interaction/
git commit -m "$(cat <<'EOF'
docs(skill): 新建 ethoinsight-lead-interaction skill (WI8)

Lead agent 的交互手册,补完 lead system prompt 瘦身(W16 1243→200 行)
后留下的细节。包含:
- 意图 7 分类决策树 (references/intent-decision-tree.md)
- 范式识别流程 (references/paradigm-identification.md)
- ask_clarification 反问模板 (references/clarification-templates.md)
- 意图状态转移规则
- Capability-exposure 调度规则
- Gate before guess 哲学 (不默认猜范式)

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §8.5
EOF
)"
```

---

## Task 9 (W9): ethoinsight-charts skill 重定位至 chart-maker 专有

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight-charts/SKILL.md`

**依赖:** W7

**Spec refs:** §8.3

只改 SKILL.md 顶部,声明服务对象。内容主体保留作为 chart-maker 的查询源(图种 → 适用场景对照表)。

- [ ] **Step 1: read 现有内容**

```bash
head -30 packages/agent/skills/custom/ethoinsight-charts/SKILL.md
```

- [ ] **Step 2: 修改 frontmatter / 顶部**

把 SKILL.md 前 30 行替换为:

```markdown
---
name: ethoinsight-charts
description: >
  服务对象:**chart-maker subagent**。图种 → 适用场景对照表,chart-maker
  决策选哪些图时的查询源。Lead 不读本 skill — capability-exposure 后
  "用户语义 → 图种" 归 chart-maker,lead 不持图选择 know-how。
version: 2.0.0
author: noldus-insight
---

# EthoInsight 图表指南(chart-maker 用)

**变更说明 (2026-05-18 W9)**:
- 服务对象从"lead agent"变为"chart-maker subagent"
- lead 不再读本 skill 决策图种
- "用户语义 → 图种" 决策树移到 `ethoinsight-chart-maker` skill (W21)
- 本 skill 保留作为图种适用性查询源 (chart-maker 在 W21 skill 决策时 reference 这里)

## 图种 → 适用场景对照表

(保留现有内容)

...
```

剩余内容(图种描述、适用场景、references/*.md 等)**保持不变**。

- [ ] **Step 3: 验证**

```bash
grep -n "服务对象" packages/agent/skills/custom/ethoinsight-charts/SKILL.md   # 期 1 处
grep -n "chart-maker" packages/agent/skills/custom/ethoinsight-charts/SKILL.md  # 至少 2 处
```

- [ ] **Step 4: Commit**

```bash
git add packages/agent/skills/custom/ethoinsight-charts/SKILL.md
git commit -m "$(cat <<'EOF'
docs(skill): ethoinsight-charts 重定位为 chart-maker 专有 (WI9)

- frontmatter 改 description / version → 2.0.0
- 顶部声明 "服务对象:chart-maker subagent,lead 不读"
- 内容主体(图种适用场景对照表)保留作为 chart-maker 查询源

C1 single source of truth:图种知识仍只存这一份,
chart-maker skill(W21)只放"用户语义 → 图种"决策树,
不复制本 skill 的图种内容。

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §8.3
EOF
)"
```

---

## Task 10 (W10): ethovision-paradigm-knowledge 删 default-fallback.md + 加强反问

**Files:**
- Delete: `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/default-fallback.md`(if exists; spec §13.5)
- Modify: `packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md`(删 fallback 引用,加反问模板提示)

**依赖:** W7

**Spec refs:** §8.6

- [ ] **Step 1: 探查 default-fallback.md 是否存在**

```bash
ls packages/agent/skills/custom/ethovision-paradigm-knowledge/references/ 2>&1
test -f packages/agent/skills/custom/ethovision-paradigm-knowledge/references/default-fallback.md && echo "EXISTS" || echo "NOT_FOUND"
test -f packages/agent/skills/custom/ethovision-paradigm-knowledge/references/default-template-fallback.md && echo "alt EXISTS"
```

**如果文件名是 `default-template-fallback.md`**(注意 spec §8.6 写的是 `default-fallback.md`,但 lead prompt 当前(line 274)引用的是 `default-template-fallback.md` — W10 实施 agent 必须以**实际存在的文件名为准**)。

- [ ] **Step 2: 删除 fallback 文档**

```bash
# 用实际文件名替换 <FILENAME>
git rm packages/agent/skills/custom/ethovision-paradigm-knowledge/references/<FILENAME>.md
```

- [ ] **Step 3: 修改 SKILL.md**

(a) 删除 SKILL.md 中所有 `default-fallback.md` / `default-template-fallback.md` 的 reference(grep 找一下)
(b) 加强反问模板段。在 SKILL.md 适当位置(可作为新 section "反问 vs 默认猜测")追加:

```markdown
## 反问哲学:Gate before guess (spec §13.5)

**不允许**:在范式不明时默认选最像的范式继续。

**必须**:范式无法唯一确定时,`ask_clarification` 带证据反问。

反问模板见 `ethoinsight-lead-interaction` skill 的
`references/clarification-templates.md` 的 "范式多候选反问" /
"范式不明 + 用户也不知道" 两段。
```

- [ ] **Step 4: 验证 lead prompt 不再引用 default-fallback**

```bash
grep -rn "default-fallback\|default-template-fallback" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/ packages/agent/skills/custom/ethovision-paradigm-knowledge/
```

Expected: 输出为空(或只剩 W10 实施 agent 自己加的"删除说明"注释)。lead prompt 中 line 274 的 `references/default-template-fallback.md` 引用需要在 W16(瘦身)时一并处理 — 但**为了避免在 W10 完成后 W16 完成前 lead 引用 dead link**,W10 此处建议 lead prompt 同步删 line 273-274 那两行(W16 还会再大改一次,但先保证 dev 上无 dead link)。

```bash
# W10 实施 agent 操作:
grep -n "default-template-fallback\|default-fallback" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
# 找到对应行后,在 prompt.py 中删掉那两行(单独 diff,trivial change)
```

- [ ] **Step 5: 跑 lead prompt 测试确保未破坏**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ -k "lead_prompt or apply_prompt" -v 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add -u packages/agent/skills/custom/ethovision-paradigm-knowledge/ \
        packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
git commit -m "$(cat <<'EOF'
docs(skill): ethovision-paradigm-knowledge 删 fallback + 加 Gate before guess (WI10)

- 删除 references/default-template-fallback.md (spec §13.5 不允许默认猜测)
- SKILL.md 加 "Gate before guess" section,引用 lead-interaction skill 的
  反问模板
- lead prompt.py 顺手删 line 273-274 dead link(W16 还会大改,先去重)

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §8.6
EOF
)"
```

---

## Task 11 (W21): ethoinsight-chart-maker skill 新建

**Files:**
- Create: `packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md`
- Create: `packages/agent/skills/custom/ethoinsight-chart-maker/references/user-intent-parsing.md`
- Create: `packages/agent/skills/custom/ethoinsight-chart-maker/references/fallback-decision-tree.md`

**依赖:** W7, W9 (引用 ethoinsight-charts 作为图种查询源)

**Spec refs:** §8.4

- [ ] **Step 1: 创建 SKILL.md**

`packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md`:

```markdown
---
name: ethoinsight-chart-maker
description: >
  chart-maker subagent 的可视化决策手册。基于 catalog plan_charts.json
  和用户语义,决策跑哪些图脚本。
  Use when chart-maker receives a visualization task with handoff_code_executor.json
  available.
version: 1.0.0
author: noldus-insight
---

# Chart-Maker 可视化决策手册

## 服务对象

`chart-maker` subagent。**lead 不读本 skill**。

## 前置阅读

执行前必读:`/mnt/skills/ethoinsight/references/execution-conventions.md`(bash 调用约束 + handoff 写入规则 + error recovery)。

## 工作流(由 SubagentConfig.system_prompt 重述,本 skill 是"如何决策"细节)

1. read `handoff_code_executor.json` → 拿 paradigm / 数据列名 / n_per_group / n_groups / total_subjects
2. bash `python -m ethoinsight.catalog.resolve --mode charts --paradigm <p> --user-intent "<原话>" ... --output plan_charts.json`
3. read `plan_charts.json` → `charts[]` + `charts_fallback_available[]`
4. **决策**(本 skill 重点):见 references/fallback-decision-tree.md
5. for each selected chart: bash 跑脚本
6. write `handoff_chart_maker.json`(schema 详见 references/output-contract.md,继承根 skill 的 gate_signals 格式)
7. `present_files(<生成的 png 列表>)` — 复制到 outputs/
8. 输出 `OK: <N> charts generated\n[gate_signals]\n...`

## 用户语义解析(细节见 references/user-intent-parsing.md)

| 用户原话特征 | 解析 |
|---|---|
| "再画几个图" / "再来几张" | 复数 + 模糊 → 选 fallback 全部 |
| "画 trajectory" / "轨迹图" | 单选明确 → 只跑 trajectory |
| "箱线图比较" / "compare" | 组间比较 + 要 box_* (catalog charts) |
| 无明确语义(E2E 流程默认) | 跑 charts[] 全部;无 charts 则跑 fallback 全部 |

## Fallback 决策树(细节见 references/fallback-decision-tree.md)

```
plan_charts.charts != []  → 跑 charts[](catalog 命中的图)
plan_charts.charts == [] AND fallback != []:
  ├── user_intent 明确(指定图种)→ 在 fallback 中选匹配的子集
  ├── user_intent 模糊("几个图")→ 跑 fallback 全部
  └── 都没匹配的 → 写 handoff status=failed,在最终 message 让 lead 反问用户
plan_charts.charts == [] AND fallback == []:
  → 写 handoff status=failed,在最终 message 让 lead 反问用户范式 / 数据
```

## 不允许的操作

- ❌ 不要 read 别的 subagent 的 handoff(`handoff_data_analyst.json` 等)— HandoffIsolationProvider 拦截
- ❌ 不要尝试 catalog 之外的图(写新 chart 必须通过 catalog YAML + 同 PR 加脚本,见 C7)
- ❌ 不要把图保存到 `/mnt/user-data/workspace/` 之外的路径;`present_files` 负责复制到 `outputs/`
```

- [ ] **Step 2: 创建 references/user-intent-parsing.md**

```markdown
# 用户语义解析规则

## 复数 vs 单数

- "画一个图" / "画 trajectory" → 单选
- "画几个图" / "画一些图" → 复数 → 选多张
- "画图" / "可视化" → 模糊 → 默认全选 charts[] 或 fallback 全部

## 图种关键词

| 关键词 | 匹配图 id |
|---|---|
| "trajectory" / "轨迹" / "运动路径" | `trajectory_plot` |
| "timeseries" / "时间序列" / "动态" | `timeseries_plot` |
| "box" / "箱线图" / "组间比较" | `box_*`(catalog 中的箱线图,通常 when: n_per_group >= 3) |
| "violin" / "小提琴图" | `violin_*` |
| "heatmap" / "热图" | `heatmap_*` |

## 反向匹配(fallback 中只有 trajectory + timeseries 时)

| 用户原话 | 选 |
|---|---|
| 含 "trajectory" / "轨迹" | trajectory_plot |
| 含 "timeseries" / "时间" / "动态" | timeseries_plot |
| 含 "几个" / "都" / "全" | 两个都跑 |
| 其他模糊 | 两个都跑 |
```

- [ ] **Step 3: 创建 references/fallback-decision-tree.md**

```markdown
# Fallback 决策树(完整)

## 输入

- `plan_charts.charts: list[PlanChart]` — catalog 命中
- `plan_charts.charts_fallback_available: list[PlanChart]` — fallback 候选(_common.yaml)
- `plan_charts.user_intent: str | None` — 用户原话

## 决策

### Case A:catalog 命中非空

```
if plan_charts.charts:
    selected = plan_charts.charts  # 全选 catalog 命中的
    fallback_used = False
```

跳过 fallback 路径。理由:catalog charts 是范式专家版本,优先于通用 fallback。

### Case B:catalog 空 + fallback 非空 + 用户语义明确

```
if not plan_charts.charts and plan_charts.charts_fallback_available:
    if user_intent_specific(plan_charts.user_intent):
        selected = filter_by_intent(plan_charts.charts_fallback_available, user_intent)
        fallback_used = True
```

参考 `references/user-intent-parsing.md` 反向匹配表。

### Case C:catalog 空 + fallback 非空 + 用户语义模糊

```
if not plan_charts.charts and plan_charts.charts_fallback_available:
    if user_intent_vague(plan_charts.user_intent):
        selected = plan_charts.charts_fallback_available  # 全选
        fallback_used = True
```

### Case D:catalog 空 + fallback 空

```
if not plan_charts.charts and not plan_charts.charts_fallback_available:
    write handoff_chart_maker.json status="failed"
    final_message:
      "无法生成图表 — catalog 无命中且 _common.yaml fallback 都不满足
       (total_subjects=<N>)。请检查数据是否符合范式预期,或在 plan_charts.notes
       看具体原因。"
```

让 lead 看到 status=failed → ask_clarification 用户。

## 错误处理

- 跑脚本失败(stderr) → 重试 ≤ 2 次 → 仍失败 → 进入 handoff.errors
- 超过 6 次 bash 调用未完工 → ScriptInvocationOnlyProvider / LoopDetection 中断
- chart-maker 的总 bash budget(spec §10.1):≤ 6 次(resolve + 最多 4 个 plot + 文件操作)
```

- [ ] **Step 4: 验证 + Commit**

```bash
ls -R packages/agent/skills/custom/ethoinsight-chart-maker/
git add packages/agent/skills/custom/ethoinsight-chart-maker/
git commit -m "$(cat <<'EOF'
docs(skill): 新建 ethoinsight-chart-maker skill (WI21)

chart-maker subagent 的可视化决策手册:
- 用户语义解析(复数 / 单数 / 关键词 → 图 id)
- Fallback 决策树(catalog 命中 / fallback / 空)
- bash budget ≤ 6 次

C1 single source of truth:图种适用性细节仍归 ethoinsight-charts skill
(W9 重定位后供 chart-maker 查),本 skill 只放"用户语义 → 决策"逻辑。

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §8.4
EOF
)"
```

---

## Task 12 (W11): code-executor 加 capability + 删 chart 段 + 改 plan 文件名

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`
- Modify: `packages/agent/skills/custom/ethoinsight-code/SKILL.md`
- Test: `packages/agent/backend/tests/test_code_executor_config.py` (new)

**依赖:** W1 (capability 字段), W2 (PlanMetrics 概念), W7 (execution-conventions skill), W8 (lead-interaction skill)

**Spec refs:** §5.1

**改动总览**:
- 加 4 capability metadata 字段
- `skills` 删 `ethoinsight-charts`
- system_prompt 中
  - read `metric_plan.json` → 改为 read `plan_metrics.json` (W20 配套)
  - workflow line 32-34 删 charts 循环段
  - <critical_rules> 删 plan.charts 相关警告

- [ ] **Step 1: 红 — 写失败测试**

`packages/agent/backend/tests/test_code_executor_config.py`:

```python
"""W11: code-executor SubagentConfig 验收。"""
from __future__ import annotations

from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG


def test_capability_metadata_set():
    cfg = CODE_EXECUTOR_CONFIG
    assert cfg.when_to_use is not None and "分析" in cfg.when_to_use
    assert cfg.input_contract is not None and "plan_metrics" in cfg.input_contract
    assert cfg.output_contract is not None and "handoff_code_executor.json" in cfg.output_contract
    assert "gate_signals" in cfg.output_contract
    assert cfg.required_upstream_handoffs == []  # 第一棒,无上游


def test_skills_no_longer_include_ethoinsight_charts():
    cfg = CODE_EXECUTOR_CONFIG
    assert cfg.skills is not None
    assert "ethoinsight-charts" not in cfg.skills
    assert "ethoinsight-code" in cfg.skills


def test_system_prompt_reads_plan_metrics_not_metric_plan():
    cfg = CODE_EXECUTOR_CONFIG
    assert "plan_metrics.json" in cfg.system_prompt
    assert "metric_plan.json" not in cfg.system_prompt   # 旧文件名彻底剥离


def test_system_prompt_no_longer_runs_charts():
    """W11 删 workflow step 4 'for chart in plan.charts:' 段。"""
    cfg = CODE_EXECUTOR_CONFIG
    # 不能含运行 chart 脚本的指令
    assert "for chart in plan.charts" not in cfg.system_prompt
    # 也不要保留 "plan.charts=[]" 类的警告 — 字段已不存在
    assert "plan.charts" not in cfg.system_prompt


def test_system_prompt_still_runs_metrics_and_stats():
    cfg = CODE_EXECUTOR_CONFIG
    # 主流程的 metrics + stats 不能删
    assert "plan.metrics" in cfg.system_prompt
    assert "plan.statistics" in cfg.system_prompt
    assert "handoff_code_executor.json" in cfg.system_prompt


def test_disallowed_tools_unchanged():
    cfg = CODE_EXECUTOR_CONFIG
    assert "task" in (cfg.disallowed_tools or [])
    assert "ask_clarification" in (cfg.disallowed_tools or [])
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_code_executor_config.py -v
```

Expected: AttributeError(`when_to_use` None)、`ethoinsight-charts` still in skills、`metric_plan.json` 字符串仍在 prompt。

- [ ] **Step 3: 绿 — 改 code_executor.py**

(a) `skills=["ethoinsight-code"]`(删 ethoinsight-charts)

(b) 增 capability metadata(放在 `timeout_seconds=900,` 后):

```python
    when_to_use=(
        "适合:\n"
        "- 用户上传 EthoVision 数据并要求'分析' / '算指标' / '做统计'\n"
        "- 已经派过本 subagent 后又要'重算某个指标' / '改 include/exclude 重跑'\n"
        "不适合:\n"
        "- 画图(派 chart-maker)\n"
        "- 解读统计结果(派 data-analyst)\n"
        "- 写报告(派 report-writer)"
    ),
    input_contract=(
        "派遣 prompt 模板:\n"
        '  "请按 plan_metrics.json 算指标和统计。范式: <paradigm>"\n'
        "配套:必须在 prompt 前先调 set_experiment_paradigm + prep_metric_plan tool"
    ),
    output_contract=(
        "- 写 /mnt/user-data/workspace/handoff_code_executor.json\n"
        "  (schema 详见 ethoinsight-code skill templates/output-contract.md)\n"
        "- 最终 AIMessage 形如 `OK: handoff written\\n[gate_signals]\\n...`\n"
        "- [gate_signals] 字段:constitution_acknowledged / data_quality{critical_count, "
        "warning_count, critical_items[]} / statistical_validity / errors_count"
    ),
    required_upstream_handoffs=[],
```

(c) system_prompt 改动(diff):

- 现状 line 24-25(workflow step 2):
  ```
  2. read `${workspace_path}/metric_plan.json` — 这是 lead 已经生成好的施工单,含 paradigm、metrics[]、statistics、charts[]、skipped[]
  ```
  改为:
  ```
  2. read `${workspace_path}/plan_metrics.json` — 这是 lead 通过 prep_metric_plan 工具生成的施工单,含 paradigm、metrics[]、statistics、skipped[](无 charts)
  ```

- 现状 line 32-34(workflow step 4):
  ```
  4. for chart in plan.charts:
       bash `python -m <chart.script> --input ... --output ...`
     注意:如果 plan.charts 是空数组,**直接跳过这一步**,不要去探索"还有什么 chart 可以跑"。
  ```
  **删除**(整段)。后续 step 编号顺移:原 step 5→4,原 step 6→5,原 step 7→6。

- <critical_rules>(line 41-46)删与 charts 相关的两条:
  - "**不要因为 plan 不完美就追加工作**。plan.charts=[]..."(line 43)整条删除。
  - 其他 critical_rules 保留。

(d) workflow 中文末加一句强调:"chart-maker 已接手图表执行,本 subagent **只跑 metrics + stats**。"

- [ ] **Step 4: 绿 — 改 ethoinsight-code skill SKILL.md**

`packages/agent/skills/custom/ethoinsight-code/SKILL.md`:

- 找到提到 `metric_plan.json` 的所有位置 → 改为 `plan_metrics.json`
- 找到 step 4 "for each chart in plan.charts" 段 → 整段删除
- 在 workflow 顶部加一句:"本 skill 服务于 code-executor,**只跑 metrics + stats**。图表归 chart-maker(见 ethoinsight-chart-maker skill,W21)。"
- 在 "## 通用资源" 段加一句:"执行约束(bash 形式 / handoff 写入 / error recovery / gate_signals 格式)见根 skill `references/execution-conventions.md`(W7),不在本 skill 重复(C1)。"

- [ ] **Step 5: 跑测试确认通过**

```bash
PYTHONPATH=. uv run pytest tests/test_code_executor_config.py -v
PYTHONPATH=. uv run pytest tests/ -k "code_executor or subagent" -v 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py \
        packages/agent/skills/custom/ethoinsight-code/SKILL.md \
        packages/agent/backend/tests/test_code_executor_config.py
git commit -m "$(cat <<'EOF'
refactor(subagent): code-executor 加 capability + 删 charts 段 (WI11)

- 加 4 capability metadata 字段
- skills 删 'ethoinsight-charts' (chart-maker 接手图表)
- system_prompt:
  * read metric_plan.json → read plan_metrics.json (W20 配套改名)
  * 删 workflow step 4 'for chart in plan.charts' 整段
  * 删 critical_rules 中 'plan.charts=[]' 相关警告
- ethoinsight-code SKILL.md 删 chart 段 + 引用根 skill execution-conventions

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §5.1
EOF
)"
```

---

## Task 13 (W12): data-analyst 加 capability

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`
- Test: `packages/agent/backend/tests/test_data_analyst_config.py` (new)

**依赖:** W1

**Spec refs:** §5.2

C3 约束:不动 system_prompt 主体。

- [ ] **Step 1: 红 — 写失败测试**

`packages/agent/backend/tests/test_data_analyst_config.py`:

```python
"""W12: data-analyst SubagentConfig 验收。"""
from __future__ import annotations

from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG


def test_capability_metadata_set():
    cfg = DATA_ANALYST_CONFIG
    assert cfg.when_to_use and "code-executor 刚完成" in cfg.when_to_use
    assert cfg.input_contract and "用户语言" in cfg.input_contract
    assert cfg.output_contract and "handoff_data_analyst.json" in cfg.output_contract
    assert "gate_signals" in cfg.output_contract


def test_required_upstream_handoffs_is_code_executor():
    cfg = DATA_ANALYST_CONFIG
    assert cfg.required_upstream_handoffs == ["code_executor"]


def test_system_prompt_unchanged_in_substance():
    """C3:不动 prompt 主体。这里检查关键骨架字段还在。"""
    cfg = DATA_ANALYST_CONFIG
    # 至少包含 contract / workflow / principles 等核心 section
    # (实际 anchor 字符串以 data_analyst.py 现状为准)
    assert cfg.system_prompt and len(cfg.system_prompt) > 200
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONPATH=. uv run pytest tests/test_data_analyst_config.py -v
```

- [ ] **Step 3: 绿 — 改 data_analyst.py**

加 capability metadata:

```python
    when_to_use=(
        "适合:\n"
        "- code-executor 刚完成、有 handoff_code_executor.json,要对统计结果做专业解读 / 方法学把关 / 离群诊断\n"
        "不适合:\n"
        "- 用户问纯领域知识(派 knowledge-assistant)\n"
        "- 画图(派 chart-maker)"
    ),
    input_contract=(
        "派遣 prompt 模板(用户语言原话 + 简短引导):\n"
        '  "请基于 code-executor 的结果做专业解读,关注效应量和混杂因素。"'
    ),
    output_contract=(
        "- 写 /mnt/user-data/workspace/handoff_data_analyst.json\n"
        "  (schema 详见 data_analyst system_prompt)\n"
        "- 最终 AIMessage:2-3 段自然语言摘要 + [gate_signals] 块\n"
        "- [gate_signals] 字段:constitution_acknowledged / method_warnings_count / "
        "outlier_count / excluded_metrics_count / statistical_validity / errors_count"
    ),
    required_upstream_handoffs=["code_executor"],
```

- [ ] **Step 4: 跑测试 + Commit**

```bash
PYTHONPATH=. uv run pytest tests/test_data_analyst_config.py -v
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py \
        packages/agent/backend/tests/test_data_analyst_config.py
git commit -m "feat(subagent): data-analyst 加 capability metadata (WI12)

required_upstream_handoffs=['code_executor'] — 必读 code-executor handoff。
system_prompt 主体未动(C3)。

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §5.2"
```

---

## Task 14 (W14): report-writer 加 capability + workflow 加可选 chart_maker

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`
- Test: `packages/agent/backend/tests/test_report_writer_config.py` (new)

**依赖:** W1

**Spec refs:** §5.4

**注意 R5**:`required_upstream_handoffs` 暂不支持 optional。chart_maker handoff 是"可选"依赖 — W14 不放进 required_upstream_handoffs,而是 lead 派遣时手动加 `{{handoff://chart_maker}}` 占位符(W19 自动注入完成后此 case 仍要手动,因为 chart_maker 不在 required list)。短期 workaround,长期 future spec 再补 `optional_upstream_handoffs`。

- [ ] **Step 1: 红 — 写失败测试**

```python
"""W14: report-writer SubagentConfig 验收。"""
from __future__ import annotations

from deerflow.subagents.builtins.report_writer import REPORT_WRITER_CONFIG


def test_capability_metadata_set():
    cfg = REPORT_WRITER_CONFIG
    assert cfg.when_to_use and "报告" in cfg.when_to_use
    assert cfg.input_contract and "code-executor" in cfg.input_contract
    assert cfg.output_contract and "report.md" in cfg.output_contract


def test_required_upstream_handoffs_is_code_and_data():
    cfg = REPORT_WRITER_CONFIG
    assert sorted(cfg.required_upstream_handoffs) == ["code_executor", "data_analyst"]


def test_system_prompt_mentions_chart_maker_handoff_optional():
    """workflow 应提到 chart_maker handoff 是可选的。"""
    cfg = REPORT_WRITER_CONFIG
    # 至少有提示:可选 read_file handoff_chart_maker.json 取 chart_paths
    assert "handoff_chart_maker" in cfg.system_prompt
    assert "可选" in cfg.system_prompt or "optional" in cfg.system_prompt.lower()
```

- [ ] **Step 2: 跑测试确认失败 → 绿 — 改 report_writer.py**

(a) 加 capability metadata:

```python
    when_to_use=(
        "适合:\n"
        "- 已有 code-executor + data-analyst handoff,用户要'出报告' / '写 Discussion' / "
        "'我要 markdown 报告给导师看'\n"
        "不适合:\n"
        "- 没有 data-analyst 解读(先派 data-analyst)\n"
        "- 只要图(派 chart-maker)"
    ),
    input_contract=(
        "派遣 prompt 模板:\n"
        '  "请基于 code-executor 数据 + data-analyst 解读撰写 6 段骨架报告。"'
    ),
    output_contract=(
        "- 写 /mnt/user-data/outputs/report.md(6 段骨架,见 system_prompt <structure>)\n"
        "- 写 /mnt/user-data/workspace/handoff_report_writer.json\n"
        "- 最终 AIMessage:报告路径 + 章节摘要 + [gate_signals]\n"
        "- [gate_signals] 字段:constitution_acknowledged / sections_written_count / "
        "sections_missing[] / statistical_validity / errors_count"
    ),
    required_upstream_handoffs=["code_executor", "data_analyst"],
```

(b) 在 system_prompt workflow 段适当位置加(同 commit):

```
<optional_chart_handoff>
如果 lead 在 task prompt 中包含 `/mnt/user-data/workspace/handoff_chart_maker.json` 路径
(注意这是可选输入,只有走过 chart-maker 的会话才有),你可以 read_file 拿 chart_paths
然后在 report.md "Figures" section 用 `![](path)` 引用。
若该文件不存在,Figures section 写"(无可视化输出)"即可,不报错。
</optional_chart_handoff>
```

- [ ] **Step 3: 跑测试 + Commit**

```bash
PYTHONPATH=. uv run pytest tests/test_report_writer_config.py -v
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py \
        packages/agent/backend/tests/test_report_writer_config.py
git commit -m "feat(subagent): report-writer 加 capability + 可选 chart handoff (WI14)

required_upstream_handoffs=['code_executor', 'data_analyst']
chart_maker handoff 是 optional — lead 派遣时手动加 {{handoff://chart_maker}}
占位符(短期 workaround,R5)。

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §5.4"
```

---

## Task 15 (W15): knowledge-assistant 加 capability

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py`
- Test: `packages/agent/backend/tests/test_knowledge_assistant_config.py` (new)

**依赖:** W1

**Spec refs:** §5.5

- [ ] **Step 1: 红 — 写失败测试**

```python
"""W15: knowledge-assistant SubagentConfig 验收。"""
from deerflow.subagents.builtins.knowledge_assistant import KNOWLEDGE_ASSISTANT_CONFIG


def test_capability_metadata_set():
    cfg = KNOWLEDGE_ASSISTANT_CONFIG
    assert cfg.when_to_use and "QA_KNOWLEDGE" in cfg.when_to_use
    assert cfg.input_contract
    assert cfg.output_contract


def test_required_upstream_handoffs_empty():
    """QA_KNOWLEDGE 不需要 handoff;QA_FACT 时 lead 手动加占位符。"""
    cfg = KNOWLEDGE_ASSISTANT_CONFIG
    assert cfg.required_upstream_handoffs == []
```

- [ ] **Step 2: 跑测试确认失败 → 绿**

加 capability metadata:

```python
    when_to_use=(
        "适合:\n"
        "- 用户问范式 / 术语 / 方法论概念问题(QA_KNOWLEDGE)\n"
        "- 已有分析结果,用户追问'为什么 p 不显著' / 'NND 偏高说明什么'(QA_FACT)\n"
        "不适合:\n"
        "- 用户要重新算指标 / 出新报告(派对应 subagent)"
    ),
    input_contract=(
        "派遣 prompt 模板:\n"
        "  QA_KNOWLEDGE: '用户问题: <原话>'\n"
        "  QA_FACT: '用户问题: <原话>。相关数据见 upstream handoff 文件。'"
    ),
    output_contract=(
        "- 简单问题:直接在最终 AIMessage 回答\n"
        "- 深度问题:write_file /mnt/user-data/workspace/knowledge_response.md + 摘要\n"
        "- 不强制 [gate_signals] 块(QA 不进入 gate 决策路径)"
    ),
    required_upstream_handoffs=[],
```

- [ ] **Step 3: 跑测试 + Commit**

```bash
PYTHONPATH=. uv run pytest tests/test_knowledge_assistant_config.py -v
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py \
        packages/agent/backend/tests/test_knowledge_assistant_config.py
git commit -m "feat(subagent): knowledge-assistant 加 capability metadata (WI15)

required_upstream_handoffs=[] — QA_KNOWLEDGE 不依赖 handoff,
QA_FACT 时 lead 派遣手动加 {{handoff://X}} 占位符(区分两种 QA 子意图)。

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §5.5"
```

---

## Task 16 (W13): chart-maker builtin 新建 + 注册

**Files:**
- Create: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/__init__.py` (import + 注册)
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py` (HANDOFF_FILE_REGISTRY 加 chart_maker)
- Test: `packages/agent/backend/tests/test_chart_maker_config.py` (new)

**依赖:** W1, W2, W3, W4, W7, W21

**Spec refs:** §5.3

- [ ] **Step 1: 红 — 写失败测试**

`packages/agent/backend/tests/test_chart_maker_config.py`:

```python
"""W13: chart-maker SubagentConfig + 注册到 BUILTIN_SUBAGENTS。"""
from __future__ import annotations

from deerflow.subagents.builtins.chart_maker import CHART_MAKER_CONFIG
from deerflow.subagents.builtins import BUILTIN_SUBAGENTS
from deerflow.tools.builtins.task_tool import HANDOFF_FILE_REGISTRY


def test_chart_maker_config_basic_fields():
    cfg = CHART_MAKER_CONFIG
    assert cfg.name == "chart-maker"
    assert "可视化" in cfg.description or "chart" in cfg.description.lower()
    assert cfg.model == "inherit"


def test_chart_maker_capability_metadata():
    cfg = CHART_MAKER_CONFIG
    assert cfg.when_to_use and "画图" in cfg.when_to_use
    assert cfg.input_contract and "chart" in cfg.input_contract.lower()
    assert cfg.output_contract and "handoff_chart_maker.json" in cfg.output_contract
    assert cfg.required_upstream_handoffs == ["code_executor"]


def test_chart_maker_tools():
    cfg = CHART_MAKER_CONFIG
    assert "bash" in cfg.tools
    assert "read_file" in cfg.tools
    assert "write_file" in cfg.tools
    assert "present_files" in cfg.tools
    assert "task" in (cfg.disallowed_tools or [])
    assert "ask_clarification" in (cfg.disallowed_tools or [])


def test_chart_maker_skills():
    cfg = CHART_MAKER_CONFIG
    assert "ethoinsight" in cfg.skills
    assert "ethoinsight-chart-maker" in cfg.skills


def test_chart_maker_registered_in_builtins():
    assert "chart-maker" in BUILTIN_SUBAGENTS
    assert BUILTIN_SUBAGENTS["chart-maker"] is CHART_MAKER_CONFIG


def test_chart_maker_handoff_registered():
    assert "chart_maker" in HANDOFF_FILE_REGISTRY
    assert HANDOFF_FILE_REGISTRY["chart_maker"] == "handoff_chart_maker.json"


def test_chart_maker_system_prompt_workflow():
    cfg = CHART_MAKER_CONFIG
    # workflow 必读 execution-conventions
    assert "execution-conventions" in cfg.system_prompt
    # 必读 chart-maker skill
    assert "ethoinsight-chart-maker" in cfg.system_prompt
    # 自跑 catalog.resolve --mode charts
    assert "catalog.resolve" in cfg.system_prompt
    assert "--mode charts" in cfg.system_prompt
    # 写 handoff
    assert "handoff_chart_maker.json" in cfg.system_prompt
    # present_files
    assert "present_files" in cfg.system_prompt
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_chart_maker_config.py -v
```

Expected: ImportError chart_maker / "chart-maker" 不在 BUILTIN_SUBAGENTS / "chart_maker" 不在 HANDOFF_FILE_REGISTRY。

- [ ] **Step 3: 绿 — 创建 chart_maker.py**

`packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py`:

```python
"""Chart-maker subagent — 行为数据可视化执行专家。"""

from deerflow.subagents.config import SubagentConfig


CHART_MAKER_CONFIG = SubagentConfig(
    name="chart-maker",
    description=(
        "行为数据可视化执行专家。"
        "基于 code-executor handoff 自跑 catalog charts + skill 决策图种,bash 编排 plot 脚本。"
    ),
    system_prompt="""你是行为数据可视化执行专家。

<语言>
中文优先,确保你输出的语言一致。
</语言>

<environment>
ethoinsight 是 pre-installed Python 库(无需 pip install)。
工作目录由 lead 提供(workspace_path)。
不用 venv;每个 plot 脚本通过 `python -m ethoinsight.scripts.<paradigm | _common>.<name> --input ... --output ...` 独立调用。
</environment>

<workflow>
1. **开工前必读执行约束**: read_file `/mnt/skills/ethoinsight/references/execution-conventions.md`
2. **read 决策手册**: read_file `/mnt/skills/ethoinsight-chart-maker/SKILL.md`
   (含用户语义解析 + fallback 决策树;细节在 references/*.md)
3. read `/mnt/user-data/workspace/handoff_code_executor.json`
   → 拿 paradigm / columns / n_per_group / n_groups / total_subjects / raw_files
   (该 handoff 由 lead 通过 {{handoff://code_executor}} 占位符授权访问 — W19 自动注入)
4. bash 自跑 catalog charts:
   ```
   python -m ethoinsight.catalog.resolve \\
     --mode charts \\
     --paradigm <paradigm> \\
     --user-intent "<用户原话>" \\
     --total-subjects <N> --n-per-group <N> --n-groups <N> \\
     --columns-file <path> --raw-files-json <path> \\
     --workspace-dir /mnt/user-data/workspace \\
     --output /mnt/user-data/workspace/plan_charts.json
   ```
5. read `/mnt/user-data/workspace/plan_charts.json` → 拿 charts[] + charts_fallback_available[] + user_intent
6. 按 ethoinsight-chart-maker skill 决策树选图:
   - a) charts != [] → 跑 charts[](catalog 命中,通常组间场景)
   - b) charts == [] 且 fallback != [] 且 user_intent 明确 → 在 fallback 中选匹配子集
   - c) charts == [] 且 fallback != [] 且 user_intent 模糊 → 跑 fallback 全部
   - d) charts == [] 且 fallback == [] → 写 status=failed,让 lead 反问
7. for each selected chart:
   bash `python -m <chart.script> --input <chart.input> --output <chart.output>`
8. write `/mnt/user-data/workspace/handoff_chart_maker.json` (schema 见下)
9. `present_files(<生成的 png 列表>)` — 把图复制到 /mnt/user-data/outputs/
10. 输出最终 AIMessage:`OK: <N> charts generated\\n[gate_signals]\\n...`
</workflow>

<handoff_schema>
{
  "status": "completed" | "failed",
  "charts_generated": [
    {"id": "...", "script": "...", "output": "...", "reason": "user-requested" | "fallback"}
  ],
  "charts_skipped": [
    {"id": "...", "reason": "when_not_satisfied: ..."}
  ],
  "fallback_used": true | false,
  "user_intent_parsed": "<本 subagent 对用户语义的解析摘要>",
  "errors": []
}
</handoff_schema>

<bash_constraints>
你的 bash 命令必须是以下三种之一:
- 脚本调用: `python -m ethoinsight.scripts.<paradigm | _common>.<name> --input ... --output ...`
- catalog CLI(本 subagent 例外): `python -m ethoinsight.catalog.resolve --mode charts ...`
- 文件操作: mkdir / cp / mv / ls / cat / grep / head / tail

其他形式被 ScriptInvocationOnlyProvider 拦截。
</bash_constraints>

<output>
完工后最终 AIMessage 含:
1. 一行 `OK: <N> charts generated`(N 是 charts_generated 数)或 `FAILED: <reason>`
2. `[gate_signals]` 块,字段:
   ```
   [gate_signals]
   constitution_acknowledged: true
   charts_generated_count: <int>
   charts_skipped_count: <int>
   fallback_used: <bool>
   errors_count: <int>
   ```
</output>

<failure>
- catalog.resolve 失败 → 读 stderr → 把 error 写进 handoff.errors → status=failed,让 lead 反问
- 单个 plot 脚本失败 → 重试 ≤ 2 次 → 仍失败则 skip,记入 errors → 继续后续 chart
- 全部脚本失败 → status=failed + errors 全记 + 让 lead 反问
- 超过 6 次 bash 调用未完工 → 写 handoff status=partial + 输出已生成内容
</failure>""",
    tools=[
        "bash",
        "read_file",
        "write_file",
        "ls",
        "str_replace",
        "present_files",
    ],
    disallowed_tools=[
        "task",
        "ask_clarification",
        "web_search",
        "web_fetch",
        "image_search",
    ],
    model="inherit",
    max_turns=15,
    timeout_seconds=600,
    skills=["ethoinsight", "ethoinsight-chart-maker"],
    # ---- capability ----
    when_to_use=(
        "适合:\n"
        "- 用户要求'画图' / '可视化' / 'trajectory' / '再补几个图'\n"
        "- E2E 流程中 data-analyst 完成后的图表生成阶段\n"
        "不适合:\n"
        "- 第一次分析(先派 code-executor,本 subagent 依赖 handoff_code_executor.json)\n"
        "- 解读图意义(派 data-analyst 或 knowledge-assistant)"
    ),
    input_contract=(
        "派遣 prompt 模板:\n"
        '  "请基于 code-executor 的结果生成可视化图表。用户原话: <用户语义>"\n'
        "不需要 lead 指定图种 — 本 subagent 自己解析用户语义 + 跑 catalog 决定。"
    ),
    output_contract=(
        "- 写 /mnt/user-data/workspace/handoff_chart_maker.json (schema 见 system_prompt)\n"
        "- 写 /mnt/user-data/workspace/plot_*.png(图文件)\n"
        "- 用 present_files 把图复制到 /mnt/user-data/outputs/ 让用户可见\n"
        "- 最终 AIMessage: `OK: <N> charts generated` + [gate_signals]\n"
        "- [gate_signals] 字段:constitution_acknowledged / charts_generated_count / "
        "charts_skipped_count / fallback_used / errors_count"
    ),
    required_upstream_handoffs=["code_executor"],
)
```

- [ ] **Step 4: 绿 — 注册到 BUILTIN_SUBAGENTS + HANDOFF_FILE_REGISTRY**

Edit `packages/agent/backend/packages/harness/deerflow/subagents/builtins/__init__.py`:

```python
"""Built-in subagent configurations."""

from .bash_agent import BASH_AGENT_CONFIG
from .chart_maker import CHART_MAKER_CONFIG          # W13 新增
from .code_executor import CODE_EXECUTOR_CONFIG
from .data_analyst import DATA_ANALYST_CONFIG
from .general_purpose import GENERAL_PURPOSE_CONFIG
from .knowledge_assistant import KNOWLEDGE_ASSISTANT_CONFIG
from .report_writer import REPORT_WRITER_CONFIG

__all__ = [
    "GENERAL_PURPOSE_CONFIG",
    "BASH_AGENT_CONFIG",
    "CODE_EXECUTOR_CONFIG",
    "DATA_ANALYST_CONFIG",
    "CHART_MAKER_CONFIG",          # W13 新增
    "KNOWLEDGE_ASSISTANT_CONFIG",
    "REPORT_WRITER_CONFIG",
]

BUILTIN_SUBAGENTS = {
    "code-executor": CODE_EXECUTOR_CONFIG,
    "data-analyst": DATA_ANALYST_CONFIG,
    "chart-maker": CHART_MAKER_CONFIG,    # W13 新增
    "report-writer": REPORT_WRITER_CONFIG,
    "knowledge-assistant": KNOWLEDGE_ASSISTANT_CONFIG,
}

# W1 fail-fast 校验:每个 required_upstream_handoffs entry 必须在 HANDOFF_FILE_REGISTRY
from deerflow.subagents.config import validate_subagent_handoff_refs
from deerflow.tools.builtins.task_tool import HANDOFF_FILE_REGISTRY
validate_subagent_handoff_refs(BUILTIN_SUBAGENTS, HANDOFF_FILE_REGISTRY)
```

Edit `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py:54-59`:

```python
HANDOFF_FILE_REGISTRY: dict[str, str] = {
    "code_executor": "handoff_code_executor.json",
    "data_analyst": "handoff_data_analyst.json",
    "chart_maker": "handoff_chart_maker.json",   # W13 新增
    "report_writer": "handoff_report_writer.json",
    "planning": "handoff_planning.json",
}
```

- [ ] **Step 5: 跑全套测试**

```bash
PYTHONPATH=. uv run pytest tests/test_chart_maker_config.py tests/test_subagent_config_capability.py tests/test_task_tool_handoff_placeholders.py -v
PYTHONPATH=. uv run pytest tests/ -k "subagent or chart" -v 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py \
        packages/agent/backend/packages/harness/deerflow/subagents/builtins/__init__.py \
        packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py \
        packages/agent/backend/tests/test_chart_maker_config.py
git commit -m "$(cat <<'EOF'
feat(subagent): chart-maker builtin 新建 + 注册 (WI13)

- chart_maker.py: 行为数据可视化执行专家
  * 自跑 catalog.resolve --mode charts
  * 按 ethoinsight-chart-maker skill 决策树选图(catalog / fallback / failed)
  * bash budget ≤ 6 次(resolve + 最多 4 plot + 文件操作)
  * 写 handoff_chart_maker.json + present_files(*.png)
  * 输出 OK + [gate_signals]
- BUILTIN_SUBAGENTS 注册 "chart-maker"
- HANDOFF_FILE_REGISTRY 注册 "chart_maker" → "handoff_chart_maker.json"
- validate_subagent_handoff_refs 在 import 时 fail-fast 校验

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §5.3
EOF
)"
```

---

## Task 17 (W20): prep_metric_plan tool 改输出 plan_metrics.json + 调 resolve_metrics

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py`
- Modify: `packages/agent/backend/tests/test_prep_metric_plan.py` (扩展)

**依赖:** W2 (PlanMetrics), W4 (resolve_metrics, plan_metrics_to_dict)

**Spec refs:** §7.7

- [ ] **Step 1: 红 — 扩展现有测试**

`packages/agent/backend/tests/test_prep_metric_plan.py`(扩展,read 一下现状):

```python
# 加新测试 case:
def test_prep_metric_plan_writes_plan_metrics_json(tmp_path, monkeypatch):
    """W20: 输出文件名从 metric_plan.json 改为 plan_metrics.json。"""
    # ... 设置 thread_data 指向 tmp_path、上传 fixture 文件 ...
    result = prep_metric_plan_tool.invoke({"uploaded_file": "/mnt/user-data/uploads/sample.txt", "paradigm": "epm"})
    assert result["status"] == "ok"
    assert result["plan_path"] == "/mnt/user-data/workspace/plan_metrics.json"
    # 物理文件按新名
    assert (tmp_path / "plan_metrics.json").exists()
    # 老文件名不再生成
    assert not (tmp_path / "metric_plan.json").exists()


def test_plan_metrics_json_has_no_charts_field(tmp_path):
    # 跑 prep_metric_plan,read 输出 JSON,检查 'charts' 不在顶层
    ...
    payload = json.loads((tmp_path / "plan_metrics.json").read_text())
    assert "metrics" in payload
    assert "statistics" in payload
    assert "charts" not in payload   # plan_metrics 不再有 charts
```

- [ ] **Step 2: 跑确认失败 → 绿**

Edit `packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py`:

(a) Line 20 import 改:
```python
from ethoinsight.catalog.resolve import ResolveError, plan_metrics_to_dict, resolve_metrics
```
(b) Line 133 调用改:
```python
        plan = resolve_metrics(
            paradigm=paradigm,
            ...
        )
```
(c) Line 154-156 序列化改:
```python
    plan_dict = plan_metrics_to_dict(plan)
    plan_path = Path(real_workspace_path) / "plan_metrics.json"
```
(d) Line 76 docstring + Line 176 return dict 中 plan_path 字符串:
```python
        "plan_path": "/mnt/user-data/workspace/plan_metrics.json",
```

- [ ] **Step 3: 跑测试 + Commit**

```bash
PYTHONPATH=. uv run pytest tests/test_prep_metric_plan.py -v
git add packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py \
        packages/agent/backend/tests/test_prep_metric_plan.py
git commit -m "feat(tool): prep_metric_plan 输出 plan_metrics.json (WI20)

- import resolve_metrics + plan_metrics_to_dict (W4 拆出来的新函数)
- 输出文件名 metric_plan.json → plan_metrics.json
- return dict 中 plan_path 字符串同步更新
- 输出 JSON 不再含 charts 字段(chart-maker 自跑 plan_charts.json)

兼容性:不保留 metric_plan.json 别名(C7 single source of truth),
前端 / agent 跟着改 — code-executor (W11) / lead prompt (W16) 同步切。

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §7.7"
```

---

## Task 18 (W16): Lead prompt 大瘦身 (1243 → ~200 行) + capability 注入

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- Test: `packages/agent/backend/tests/test_lead_prompt_capability_render.py` (new)

**依赖:** W1 (format_subagent_capability), W11-W15 (5 subagent capability), W8 (lead-interaction skill 已落地)

**Spec refs:** §8.5(skill 接管细节), §3.2(capability 注入位置)

**改动总览**:
- `_build_subagent_section()` (line 180-) 整段重构 → 调 `format_subagent_capability()` 渲染每个 subagent
- 删除 `noldus_rules` 整大段(line 219-562 左右,~340 行)
- 保留 intent 分类 hard rule 段(写"必须输出 [intent] X 行")
- 保留 EV19 模板识别开关(set_experiment_paradigm 触发条件)
- 删除"如何反问 / 4-choice / 范式默认 fallback"细节 — 移到 ethoinsight-lead-interaction skill(W8 已建)
- 删除 chart 段相关引导(chart-maker 自决策)
- 删除 default-template-fallback / default-fallback 引用(W10 已部分删)

目标行数:从 1243 行降到 ~200 行(含 skill 部分 + capability 注入 + intent 状态机骨架 + 3 guardrail 错误码 hint)。

**注意**:lead_agent prompt 不 read 任何 EthoVision skill 之外的具体执行细节 — 那些移到 `ethoinsight-lead-interaction` skill;lead system prompt 只放"必须做 / 不能做"的硬约束。

- [ ] **Step 1: 红 — 写失败测试**

`packages/agent/backend/tests/test_lead_prompt_capability_render.py`:

```python
"""W16: Lead prompt 瘦身后的 capability 注入验收。"""
from __future__ import annotations

import pytest

from deerflow.agents.lead_agent.prompt import apply_prompt_template, _build_subagent_section


def test_prompt_renders_capability_for_each_subagent():
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    # 五个 subagent name 都应出现在 prompt 中
    for name in ["code-executor", "data-analyst", "chart-maker", "report-writer", "knowledge-assistant"]:
        assert name in prompt, f"subagent '{name}' missing from lead prompt"


def test_prompt_renders_when_to_use_input_output_contract():
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    # capability 字段都应渲染
    assert "when_to_use" in prompt or "适合" in prompt
    assert "input_contract" in prompt or "派遣 prompt 模板" in prompt
    assert "output_contract" in prompt or "返回" in prompt or "handoff" in prompt


def test_prompt_does_NOT_contain_handoff_placeholder_instructions():
    """W19 auto-inject 之后,lead 不需要知道 {{handoff://X}} 语法。
    Prompt 不应教用户怎么写占位符。
    """
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    assert "{{handoff://" not in prompt
    # 但允许 prompt 提"upstream handoff 自动注入"作为说明
    # (具体检查的措辞由 W16 实施 agent 定)


def test_prompt_contains_intent_classification_hard_rule():
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    assert "[intent]" in prompt
    # 7 类意图至少出现 4 类 keyword(防 typo)
    intent_keywords = ["E2E_FULL", "E2E_MIN", "CHART", "REPORT", "QA_FACT", "QA_KNOWLEDGE", "CLARIFY"]
    found = [kw for kw in intent_keywords if kw in prompt]
    assert len(found) >= 4, f"intent keywords missing: only found {found}"


def test_prompt_references_lead_interaction_skill():
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    assert "ethoinsight-lead-interaction" in prompt


def test_prompt_no_chart_selection_logic():
    """图选择逻辑已移到 chart-maker;lead prompt 不应含 'plot_trajectory' 等图脚本名。"""
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    for chart_keyword in ["plot_trajectory", "plot_timeseries", "plot_box", "trajectory_plot"]:
        assert chart_keyword not in prompt, f"chart-specific keyword '{chart_keyword}' leaked into lead prompt"


def test_prompt_line_count_drastically_reduced():
    """W16 目标:1243 → ~200 行。给个宽松上限 400。"""
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    line_count = prompt.count("\n")
    assert line_count < 400, f"lead prompt too long after W16 (got {line_count} lines, expect <400)"


def test_prompt_no_default_fallback_reference():
    """W10 已删 default-template-fallback.md;prompt 不应再引用。"""
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    assert "default-template-fallback" not in prompt
    assert "default-fallback" not in prompt


def test_prompt_no_metric_plan_only_plan_metrics():
    """W11+W20 改文件名:metric_plan.json → plan_metrics.json。lead prompt 应跟。"""
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    assert "metric_plan.json" not in prompt
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_lead_prompt_capability_render.py -v
```

Expected: 多条 FAIL(行数太多 / capability 没注入 / chart 关键词泄漏 / 等)。

- [ ] **Step 3: 绿 — 大改 prompt.py**

(由于 W16 是 plan 中最大单一改动,这一步分子步进行)

(3a) **重构 `_build_subagent_section`**(line 180-562 整段重写):

```python
def _build_subagent_section(max_concurrent: int) -> str:
    """Build the subagent system prompt section.

    W16(2026-05-18):瘦身后只渲染 SubagentConfig.capability 字段,
    不再硬编码"何时派 X / 怎么反问 Y"等细节(移到 ethoinsight-lead-interaction skill)。
    """
    from deerflow.subagents.config import format_subagent_capability
    from deerflow.subagents.builtins import BUILTIN_SUBAGENTS
    from deerflow.subagents import get_available_subagent_names

    available_names = set(get_available_subagent_names())

    # 按固定顺序渲染 5 个 Noldus subagent capability
    noldus_order = ["code-executor", "data-analyst", "chart-maker", "report-writer", "knowledge-assistant"]
    capability_blocks: list[str] = []
    for name in noldus_order:
        if name not in available_names:
            continue
        cfg = BUILTIN_SUBAGENTS.get(name)
        if cfg is None:
            continue
        capability_blocks.append(format_subagent_capability(cfg))

    # general-purpose / bash 作为 fallback subagent,简短列出
    other_lines: list[str] = []
    if "general-purpose" in available_names:
        other_lines.append("- **general-purpose**: For ANY non-trivial task outside the EthoInsight pipeline.")
    if "bash" in available_names:
        other_lines.append("- **bash**: For command execution (limited cases — prefer Noldus subagents).")

    section = f"""
## Subagent 调度 (最多 {max_concurrent} 个并发)

你是 EthoInsight 调度员。**不直接执行**,通过 `task(subagent_type, prompt, description)` 派遣。

### Capability-Exposure: 5 个 EthoInsight subagent

{chr(10).join(capability_blocks)}

### 其他 subagent

{chr(10).join(other_lines) if other_lines else "(none)"}

### 派遣规则 (硬约束)

- **第一个非 read_file tool call 之前,必须输出 `[intent] <INTENT>` 行**
  (INTENT ∈ E2E_FULL / E2E_MIN / CHART / REPORT / QA_FACT / QA_KNOWLEDGE / CLARIFY)
  违反会被 `IntentClassificationGuardrailProvider` 拦截(错误码 `ethoinsight.intent_not_declared`)
- 派遣 task() 时**不写 {{handoff://X}} 占位符** —— harness 按 SubagentConfig.required_upstream_handoffs 自动注入授权
  违反会被 `TaskHandoffAuthorizationProvider` 拦截(错误码 `ethoinsight.required_handoff_missing`)
- 范式不明确时 **必须 ask_clarification**,**不允许默认猜测**(Gate before guess)
- 已有 set_experiment_paradigm 之前 **不可** task(code-executor)
  违反会被 `Ev19TemplateGuardrailProvider` 拦截

### 意图状态机

```
[ANY] → 上传数据 + 复合语义 → E2E_FULL → ... → ask(report?)
[ANY] → 上传数据 + 单语义     → E2E_MIN  → ... → ask(four-choice)
[ANY+handoff] → 要图          → CHART (派 chart-maker)
[ANY+handoff] → 要报告        → REPORT (派 report-writer)
[ANY+handoff] → 追问数据      → QA_FACT (派 knowledge-assistant + 占位符)
[ANY]         → 问知识         → QA_KNOWLEDGE (派 knowledge-assistant)
[ANY]         → 信息缺失       → CLARIFY (ask_clarification)
```

### 详细交互手册

意图分类决策树 / 范式识别流程 / 反问模板 / 意图转移规则 见
`/mnt/skills/ethoinsight-lead-interaction/SKILL.md` 及其 references/ 目录。

### 禁止行为

- ❌ 不要 read_file raw EthoVision txt 文件(交给 ethoinsight 库解析)
- ❌ 不要默认猜测范式
- ❌ 不要替 subagent 决定具体跑哪些图 / 指标(那是 subagent + catalog 的职责)
"""
    return section
```

(3b) **删除大段 noldus_rules**:把现状 line 219 起一直到 noldus_rules 段结束的所有"路由判断 / 反问决策 / 失败处理 / EV19 模板识别说明 / 端到端范式默认 fallback"等内容**全部删除** — 这些细节已经在 `ethoinsight-lead-interaction` skill 里。

(3c) **保留**:
- 顶部 imports
- skill loading / memory injection 等 helper(不动)
- `apply_prompt_template` 函数本身的 wiring(只是它调的 `_build_subagent_section` 已重写)
- 任何与 `ev19_template_provider` / `GuardrailMiddleware` / `Ev19TemplateGuardrailProvider` 错误码相关的提示(已在 (3a) 派遣规则中)

(3d) **删除其他 dead reference**:
- `default-template-fallback.md` / `default-fallback.md` 引用(W10 已部分删,本 task 兜底)
- `metric_plan.json` 字符串改为 `plan_metrics.json`(W11+W20 配套)
- `plan.charts` / chart 相关说明全删
- "harness 自动注入 {{handoff://X}}" 等技术细节 — 保留 1-2 句说明即可,不展开

- [ ] **Step 4: 跑测试**

```bash
PYTHONPATH=. uv run pytest tests/test_lead_prompt_capability_render.py -v
```

如果行数仍超 400,检查是否有未删的 noldus_rules 残留或 skill / memory 段超大(skill 段是动态注入,不在 W16 控制范围 — 可在测试中只对 `_build_subagent_section` 段计算行数)。

如果 capability 内容渲染不出,检查 `BUILTIN_SUBAGENTS` import 路径和 `get_available_subagent_names` 是否返回所有 5 个 name。

- [ ] **Step 5: 全套回归**

```bash
PYTHONPATH=. uv run pytest tests/ -k "lead or prompt" -v 2>&1 | tail -30
```

特别注意:`test_apply_prompt_template_*` 等既有测试若 break,需要 W16 实施 agent 判断是测试断言过时还是 prompt 真的丢了关键内容。**两次反问 user**:如果断言看似过时(测试期望"端到端 default 派 code-executor → data-analyst" 的特定字符串),改测试为新的 capability 注入断言。

- [ ] **Step 6: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py \
        packages/agent/backend/tests/test_lead_prompt_capability_render.py
git commit -m "$(cat <<'EOF'
refactor(lead): prompt 大瘦身 (1243 → ~200 行) + capability 注入 (WI16)

- _build_subagent_section 整段重写,调 format_subagent_capability
  渲染 5 个 SubagentConfig (W11-W15 已落地 metadata)
- 删除 noldus_rules / 4-choice 反问模板 / 范式默认 fallback / chart 选择
  逻辑等 ~340 行细节 — 移到 ethoinsight-lead-interaction skill (W8 已建)
- 派遣规则压成 5 条硬约束:
  * [intent] 行(W17 IntentClassificationGuardrailProvider 拦截)
  * 不写 {{handoff://X}}(W19 自动注入)
  * Gate before guess (不默认猜)
  * ev19 template 前置 (Ev19TemplateGuardrailProvider 不动)
  * task() 调用前必须分类意图
- 删除 default-template-fallback.md / plan.charts / metric_plan.json 等
  dead reference
- 意图状态机骨架内联 (7 类意图 → 派遣链)

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §8.5
EOF
)"
```

---

## Task 19 (W17): IntentClassificationGuardrailProvider + Bridge middleware

**Files:**
- Create: `packages/agent/backend/packages/harness/deerflow/guardrails/intent_classification_provider.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` (挂载)
- Test: `packages/agent/backend/tests/test_intent_classification_provider.py` (new)

**依赖:** W1

**Spec refs:** §6.1

### 设计要点

- Provider:扫描 thread messages 历史(通过 ContextVar),匹配最近一条 lead AIMessage 是否含 `[intent] <NAME>` 行
- BridgeMiddleware:类比 `Ev19WorkspaceBridgeMiddleware`,把 messages 放到 ContextVar
- read_file 永远 allow(lead 在分类前需要 read skill 文档)
- Thread 刚开始无 messages 时 allow(避免死锁)
- INTENT 枚举:`E2E_FULL` / `E2E_MIN` / `CHART` / `REPORT` / `QA_FACT` / `QA_KNOWLEDGE` / `CLARIFY`

### Lead messages 的获取方式

`request.state` 是 LangChain `AgentState` 包(由 `Ev19WorkspaceBridgeMiddleware` 验证过可用 `request.state.get("thread_data")`)。AgentState 还含 `messages` 字段 — 即 thread 的完整消息历史。BridgeMiddleware 把它放到 ContextVar:

```python
from contextvars import ContextVar
_lead_messages: ContextVar[list | None] = ContextVar("_lead_messages", default=None)
```

- [ ] **Step 1: 红 — 写失败测试**

`packages/agent/backend/tests/test_intent_classification_provider.py`:

```python
"""W17: IntentClassificationGuardrailProvider 验收。"""
from __future__ import annotations

import pytest
from contextvars import copy_context
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.guardrails.intent_classification_provider import (
    IntentClassificationGuardrailProvider,
    _lead_messages,
)
from deerflow.guardrails.provider import GuardrailRequest


def _make_request(tool_name: str, args: dict | None = None) -> GuardrailRequest:
    return GuardrailRequest(tool_name=tool_name, tool_input=args or {})


@pytest.fixture
def provider():
    return IntentClassificationGuardrailProvider()


def test_allow_read_file_always(provider):
    _lead_messages.set([HumanMessage(content="hi")])
    # 无任何 lead AIMessage [intent] X,但 read_file 永远放行
    decision = provider.evaluate(_make_request("read_file", {"file_path": "/mnt/skills/x/SKILL.md"}))
    assert decision.allow


def test_allow_when_intent_declared(provider):
    _lead_messages.set([
        HumanMessage(content="分析这个数据"),
        AIMessage(content="[intent] E2E_MIN\n我开始分析..."),
    ])
    decision = provider.evaluate(_make_request("task", {"subagent_type": "code-executor"}))
    assert decision.allow


def test_deny_when_intent_missing_and_non_read_tool(provider):
    _lead_messages.set([
        HumanMessage(content="分析数据"),
        AIMessage(content="我马上派 subagent。"),   # 没 [intent] 行
    ])
    decision = provider.evaluate(_make_request("task", {"subagent_type": "code-executor"}))
    assert not decision.allow
    assert decision.reasons[0].code == "ethoinsight.intent_not_declared"


def test_deny_for_prep_metric_plan_when_intent_missing(provider):
    """非 read_file 工具都要校验:prep_metric_plan / set_experiment_paradigm / task / ask_clarification 等。"""
    _lead_messages.set([HumanMessage(content="分析")])  # 只有 user message,无 lead AIMessage
    decision = provider.evaluate(_make_request("prep_metric_plan",
                                                {"uploaded_file": "/tmp/x.txt", "paradigm": "epm"}))
    assert not decision.allow


def test_allow_when_messages_empty(provider):
    """Thread 刚开始,_lead_messages 是 None 或空 list → fail-open。"""
    _lead_messages.set(None)
    decision = provider.evaluate(_make_request("task", {"subagent_type": "code-executor"}))
    assert decision.allow   # fail-open,不锁死


def test_intent_pattern_recognizes_all_seven(provider):
    intents = ["E2E_FULL", "E2E_MIN", "CHART", "REPORT", "QA_FACT", "QA_KNOWLEDGE", "CLARIFY"]
    for intent in intents:
        _lead_messages.set([
            HumanMessage(content="x"),
            AIMessage(content=f"[intent] {intent}\nrouting..."),
        ])
        decision = provider.evaluate(_make_request("task", {"subagent_type": "code-executor"}))
        assert decision.allow, f"failed to recognize intent '{intent}'"


def test_intent_recognized_even_if_not_last_message(provider):
    """Intent 可以在历史 messages 中任何 lead AIMessage,
    不必是最近一条 — 同一 turn 内 lead 可能先输出文本再派 tool。"""
    _lead_messages.set([
        HumanMessage(content="x"),
        AIMessage(content="[intent] E2E_MIN\nstart"),
        AIMessage(content="now dispatching..."),   # 后续无 [intent] 也 OK
    ])
    decision = provider.evaluate(_make_request("task", {"subagent_type": "code-executor"}))
    assert decision.allow


def test_intent_invalid_name_does_not_count(provider):
    """[intent] FOO 不是合法枚举 → 视为未分类 → deny。"""
    _lead_messages.set([
        HumanMessage(content="x"),
        AIMessage(content="[intent] DOSOMETHING\nrouting"),
    ])
    decision = provider.evaluate(_make_request("task", {"subagent_type": "code-executor"}))
    assert not decision.allow
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONPATH=. uv run pytest tests/test_intent_classification_provider.py -v
```

Expected: ImportError `intent_classification_provider` 模块不存在。

- [ ] **Step 3: 绿 — 实现 provider + bridge**

`packages/agent/backend/packages/harness/deerflow/guardrails/intent_classification_provider.py`:

```python
"""IntentClassificationGuardrailProvider — 强制 lead 在派遣前声明意图。

Spec §6.1:lead 在第一个非 read_file tool call 之前必须输出
`[intent] <INTENT_NAME>` 行,否则拦截并注入 reminder。

Mechanism:类比 Ev19WorkspaceBridgeMiddleware,Bridge middleware 把
thread messages 写入 ContextVar,Provider 读 ContextVar 扫描历史。
"""
from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.guardrails.provider import (
    GuardrailDecision,
    GuardrailReason,
    GuardrailRequest,
)


# ContextVar:per-thread lead messages 历史,由 IntentBridgeMiddleware 写入。
_lead_messages: ContextVar[list | None] = ContextVar("_lead_messages", default=None)


# 7 个合法意图
_VALID_INTENTS = frozenset({
    "E2E_FULL", "E2E_MIN", "CHART", "REPORT",
    "QA_FACT", "QA_KNOWLEDGE", "CLARIFY",
})

# 匹配 lead AIMessage 中 `[intent] <NAME>` 行
_INTENT_LINE_RE = re.compile(
    r"\[intent\]\s+([A-Z_]+)", re.MULTILINE
)


def _extract_declared_intents(messages: list | None) -> set[str]:
    """从 lead AIMessage 中提取所有 [intent] X 行的合法枚举值。"""
    if not messages:
        return set()
    declared: set[str] = set()
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if not isinstance(content, str):
            # content 可能是 list[block] — flatten 文本块
            if isinstance(content, list):
                content = "\n".join(
                    str(b.get("text", "")) if isinstance(b, dict) else str(b)
                    for b in content
                )
            else:
                content = str(content)
        for match in _INTENT_LINE_RE.finditer(content):
            name = match.group(1)
            if name in _VALID_INTENTS:
                declared.add(name)
    return declared


class IntentClassificationGuardrailProvider:
    """Block lead's non-read_file tool calls until [intent] X declared.

    Allows: read_file (lead needs to read skill / handoff before classifying);
    empty messages (fail-open during thread bootstrap).
    """

    name = "intent_classification"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # read_file 永远放行 — lead 在分类前需读 skill / handoff
        if request.tool_name == "read_file":
            return GuardrailDecision(allow=True)

        messages = _lead_messages.get()
        # Thread 刚开始无 messages — fail-open(避免死锁)
        if not messages:
            return GuardrailDecision(allow=True)

        declared = _extract_declared_intents(messages)
        if declared:
            return GuardrailDecision(allow=True)

        return GuardrailDecision(
            allow=False,
            reasons=[
                GuardrailReason(
                    code="ethoinsight.intent_not_declared",
                    message=(
                        "在派遣 subagent / 调 prep_metric_plan / "
                        "set_experiment_paradigm 等工具前,请先在 message "
                        "中输出 `[intent] <INTENT>` 行(INTENT ∈ {"
                        + ", ".join(sorted(_VALID_INTENTS)) + "})。"
                    ),
                )
            ],
            policy_id="intent_classification",
        )

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)


class IntentBridgeMiddleware(AgentMiddleware[AgentState]):
    """Extract lead messages history into the _lead_messages ContextVar before
    GuardrailMiddleware runs. Must sit BEFORE the IntentClassification GuardrailMiddleware
    in the lead's middleware chain.
    """

    def __init__(self):
        super().__init__()

    def _extract_and_set_messages(self, request: ToolCallRequest) -> None:
        state = request.state
        if state is not None and isinstance(state, dict):
            msgs = state.get("messages")
            if isinstance(msgs, list):
                _lead_messages.set(msgs)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        self._extract_and_set_messages(request)
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        self._extract_and_set_messages(request)
        return await handler(request)
```

- [ ] **Step 4: 绿 — 挂载到 lead_agent middleware 链**

Edit `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py:317-326`(`if guardrails_cfg.enabled:` 块内,Ev19 加载之后):

```python
    if guardrails_cfg.enabled:
        from deerflow.guardrails.ev19_template_provider import (
            Ev19TemplateGuardrailProvider,
            Ev19WorkspaceBridgeMiddleware,
        )
        from deerflow.guardrails.intent_classification_provider import (
            IntentBridgeMiddleware,
            IntentClassificationGuardrailProvider,
        )
        from deerflow.guardrails.middleware import GuardrailMiddleware

        # Ev19 (现有,不动)
        provider = Ev19TemplateGuardrailProvider()
        middlewares.append(Ev19WorkspaceBridgeMiddleware())
        middlewares.append(GuardrailMiddleware(provider=provider, fail_closed=guardrails_cfg.fail_closed))

        # W17: Intent classification(新)
        middlewares.append(IntentBridgeMiddleware())
        middlewares.append(GuardrailMiddleware(
            provider=IntentClassificationGuardrailProvider(),
            fail_closed=guardrails_cfg.fail_closed,
        ))
```

- [ ] **Step 5: 跑测试**

```bash
PYTHONPATH=. uv run pytest tests/test_intent_classification_provider.py tests/test_harness_boundary.py -v
PYTHONPATH=. uv run pytest tests/ -k "lead_agent or guardrail" -v 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/guardrails/intent_classification_provider.py \
        packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py \
        packages/agent/backend/tests/test_intent_classification_provider.py
git commit -m "$(cat <<'EOF'
feat(guardrail): IntentClassificationGuardrailProvider (WI17)

Lead 在第一个非 read_file tool call 之前必须输出 `[intent] <NAME>` 行,
否则被拦截(错误码 ethoinsight.intent_not_declared)。

Mechanism:
- IntentBridgeMiddleware 把 thread messages 写到 ContextVar
  (类比 Ev19WorkspaceBridgeMiddleware,sit before GuardrailMiddleware)
- Provider 扫历史 lead AIMessage 中所有 [intent] X 行
- 7 类合法意图:E2E_FULL/E2E_MIN/CHART/REPORT/QA_FACT/QA_KNOWLEDGE/CLARIFY
- read_file 永远放行(lead 在分类前需 read skill)
- 空 messages fail-open(避免 thread bootstrap 死锁)

挂载在 Ev19TemplateGuardrailProvider 之后(both are lead-only,
中间件链顺序:Ev19Bridge → GuardrailMid(Ev19) → IntentBridge → GuardrailMid(Intent))。

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §6.1
EOF
)"
```

---

## Task 20 (W18): TaskHandoffAuthorizationProvider

**Files:**
- Create: `packages/agent/backend/packages/harness/deerflow/guardrails/task_handoff_authorization_provider.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` (挂载)
- Test: `packages/agent/backend/tests/test_task_handoff_authorization.py` (new)

**依赖:** W1, W11-W15(BUILTIN_SUBAGENTS 各 subagent 的 required_upstream_handoffs)

**Spec refs:** §6.2

### 设计要点

- 拦 lead 的 `task(subagent_type=X, prompt=Y)` 调用
- 校验 `prompt` 是否含 `BUILTIN_SUBAGENTS[X].required_upstream_handoffs` 中每个 name 对应的 `{{handoff://X}}` 占位符
- 与 W19(自动注入)的关系:**W18 是安全网**。W19 完成后,正常情况下 lead 不写占位符也能跑(harness 自动加);本 provider 抓 W19 bug 或 lead 手写绕过情况
- W19 完成前,**W18 是主要约束**(lead 必须手写占位符 — 但 W16 已删 "教 lead 写占位符" 段,所以 W18 单独存在时 lead 会被这个 provider 不断 deny → fail-closed)。**为避免 W18 落地到 W19 落地之间这个窗口期 lead 完全瘫痪**,W19 在 W18 之后立刻落地(spec §9.2 层次 W19 在 W18 之后)

- [ ] **Step 1: 红 — 写失败测试**

`packages/agent/backend/tests/test_task_handoff_authorization.py`:

```python
"""W18: TaskHandoffAuthorizationProvider 验收。"""
from __future__ import annotations

from deerflow.guardrails.task_handoff_authorization_provider import (
    TaskHandoffAuthorizationProvider,
)
from deerflow.guardrails.provider import GuardrailRequest


def _make_task_request(subagent_type: str, prompt: str) -> GuardrailRequest:
    return GuardrailRequest(
        tool_name="task",
        tool_input={"subagent_type": subagent_type, "prompt": prompt, "description": "x"},
    )


def test_allow_when_no_required_handoffs():
    """code-executor 的 required_upstream_handoffs = []; 无占位符也放行。"""
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(_make_task_request("code-executor", "请分析数据"))
    assert decision.allow


def test_allow_when_required_handoff_present():
    """data-analyst 需要 code_executor;prompt 含占位符。"""
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(
        _make_task_request("data-analyst", "请解读 {{handoff://code_executor}}")
    )
    assert decision.allow


def test_deny_when_required_handoff_missing():
    """data-analyst 需要 code_executor;prompt 缺占位符。"""
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(_make_task_request("data-analyst", "请解读结果"))
    assert not decision.allow
    assert decision.reasons[0].code == "ethoinsight.required_handoff_missing"
    assert "code_executor" in decision.reasons[0].message


def test_deny_when_one_of_multiple_required_missing():
    """report-writer 需要 code_executor + data_analyst;只有其一时 deny。"""
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(
        _make_task_request("report-writer", "写报告 {{handoff://code_executor}}")
    )
    assert not decision.allow
    assert "data_analyst" in decision.reasons[0].message


def test_allow_when_all_required_present():
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(_make_task_request(
        "report-writer",
        "写报告 {{handoff://code_executor}} {{handoff://data_analyst}}",
    ))
    assert decision.allow


def test_unknown_subagent_type_passes_through():
    """task subagent_type 未知时,本 provider 不拦(其他逻辑会拦)。"""
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(_make_task_request("nonexistent", "x"))
    assert decision.allow


def test_non_task_tools_pass_through():
    """本 provider 只针对 task tool。"""
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(GuardrailRequest(tool_name="read_file", tool_input={}))
    assert decision.allow
```

- [ ] **Step 2: 跑测试确认失败 → 绿**

`packages/agent/backend/packages/harness/deerflow/guardrails/task_handoff_authorization_provider.py`:

```python
"""TaskHandoffAuthorizationProvider — 校验 lead task() 是否含必需占位符。

Spec §6.2:lead 派遣 task(subagent_type=X) 时,prompt 必须含
BUILTIN_SUBAGENTS[X].required_upstream_handoffs 中每个 name 的
{{handoff://X}} 占位符。

与 W19 的关系:W19 完成后,task_tool 在 prompt 进入 SubagentExecutor 之前
自动注入缺失的占位符 → 本 provider 实际不再 deny;但本 provider 作为
**安全网**保留,抓 W19 bug 或 lead 手写绕过情况。

W19 必须紧跟 W18 落地,避免窗口期 lead 全部 task() 都被 deny。
"""
from __future__ import annotations

from deerflow.guardrails.provider import (
    GuardrailDecision,
    GuardrailReason,
    GuardrailRequest,
)


class TaskHandoffAuthorizationProvider:
    name = "task_handoff_authorization"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if request.tool_name != "task":
            return GuardrailDecision(allow=True)

        subagent_type = request.tool_input.get("subagent_type", "")
        prompt = request.tool_input.get("prompt", "") or ""

        # Lazy import 避开 cyclic dependency
        from deerflow.subagents.builtins import BUILTIN_SUBAGENTS

        config = BUILTIN_SUBAGENTS.get(subagent_type)
        if config is None or not config.required_upstream_handoffs:
            return GuardrailDecision(allow=True)

        missing = [
            name for name in config.required_upstream_handoffs
            if f"{{{{handoff://{name}}}}}" not in prompt
        ]
        if missing:
            return GuardrailDecision(
                allow=False,
                reasons=[GuardrailReason(
                    code="ethoinsight.required_handoff_missing",
                    message=(
                        f"subagent '{subagent_type}' 需要 upstream handoff "
                        f"{missing}。在 prompt 中加 {{{{handoff://<name>}}}} 占位符,"
                        f"或检查 task_tool 自动注入(W19)是否启用。"
                    ),
                )],
                policy_id="task_handoff_authorization",
            )
        return GuardrailDecision(allow=True)

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)
```

- [ ] **Step 3: 挂载到 middleware 链**

Edit `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py:317-326` 块内,接 W17 之后:

```python
        # W18: Task handoff authorization(新)
        from deerflow.guardrails.task_handoff_authorization_provider import (
            TaskHandoffAuthorizationProvider,
        )
        middlewares.append(GuardrailMiddleware(
            provider=TaskHandoffAuthorizationProvider(),
            fail_closed=guardrails_cfg.fail_closed,
        ))
```

- [ ] **Step 4: 跑测试 + Commit**

```bash
PYTHONPATH=. uv run pytest tests/test_task_handoff_authorization.py tests/test_harness_boundary.py -v
git add packages/agent/backend/packages/harness/deerflow/guardrails/task_handoff_authorization_provider.py \
        packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py \
        packages/agent/backend/tests/test_task_handoff_authorization.py
git commit -m "$(cat <<'EOF'
feat(guardrail): TaskHandoffAuthorizationProvider (WI18)

校验 lead task() 调用是否含 SubagentConfig.required_upstream_handoffs
对应的 {{handoff://X}} 占位符。

错误码 ethoinsight.required_handoff_missing。

与 W19 关系:W19 task_tool auto-inject 完成后本 provider 实际不会 deny,
但保留作安全网。W19 在 W18 之后紧跟落地,避免窗口期 task() 全 deny。

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §6.2
EOF
)"
```

---

## Task 21 (W19): task_tool 自动注入 required_upstream_handoffs 占位符

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py` (line 357-363 之前注入)
- Test: `packages/agent/backend/tests/test_task_tool_auto_inject.py` (new)

**依赖:** W1, W11-W15, W18(让 fail-closed 测试 baseline 是 deny)

**Spec refs:** §2.2 / R3

### 设计要点

- 注入位置:`task_tool` 函数内,line 358 `_resolve_placeholders` 之前(也可以在 `_resolve_handoff_placeholders` 之前)
- **不双注入**:lead 已手写 `{{handoff://X}}` 占位符的不重复加 — grep `prompt` 找出 `_HANDOFF_PLACEHOLDER_RE.findall(prompt)` 已有的 name,跳过
- 算法:
  ```
  config = BUILTIN_SUBAGENTS[subagent_type]
  existing = set(_HANDOFF_PLACEHOLDER_RE.findall(prompt))  # 已存在的 name
  for name in config.required_upstream_handoffs:
    if name not in existing:
      prompt += f"\n\n[Upstream handoff (auto-injected)]: {{handoff://{name}}}"
  ```

- [ ] **Step 1: 红 — 写失败测试**

`packages/agent/backend/tests/test_task_tool_auto_inject.py`:

```python
"""W19: task_tool 自动注入 required_upstream_handoffs 占位符。"""
from __future__ import annotations

from deerflow.tools.builtins.task_tool import (
    _auto_inject_handoff_placeholders,
    _HANDOFF_PLACEHOLDER_RE,
)


def test_auto_inject_for_data_analyst_when_missing():
    """data-analyst 需要 code_executor;lead 未写占位符 → harness 加。"""
    new_prompt = _auto_inject_handoff_placeholders("请解读结果", "data-analyst")
    found = _HANDOFF_PLACEHOLDER_RE.findall(new_prompt)
    assert "code_executor" in found


def test_no_inject_when_no_required_handoffs():
    """code-executor 不需要 upstream handoff。"""
    new_prompt = _auto_inject_handoff_placeholders("请分析", "code-executor")
    assert _HANDOFF_PLACEHOLDER_RE.findall(new_prompt) == []
    assert new_prompt == "请分析" or new_prompt.strip() == "请分析"


def test_no_double_inject():
    """lead 已手写 {{handoff://code_executor}} 时不再追加。"""
    original = "请解读 {{handoff://code_executor}} 的结果"
    new_prompt = _auto_inject_handoff_placeholders(original, "data-analyst")
    # 仍然只有 1 处 code_executor 占位符
    matches = _HANDOFF_PLACEHOLDER_RE.findall(new_prompt)
    assert matches.count("code_executor") == 1


def test_multi_required_all_injected():
    """report-writer 需要 code_executor + data_analyst;两个都注入。"""
    new_prompt = _auto_inject_handoff_placeholders("写报告", "report-writer")
    found = _HANDOFF_PLACEHOLDER_RE.findall(new_prompt)
    assert "code_executor" in found
    assert "data_analyst" in found


def test_partial_handwritten_injects_only_missing():
    """report-writer 需要 code+data;lead 手写了 code → harness 加 data。"""
    original = "写报告 {{handoff://code_executor}}"
    new_prompt = _auto_inject_handoff_placeholders(original, "report-writer")
    found = _HANDOFF_PLACEHOLDER_RE.findall(new_prompt)
    assert found.count("code_executor") == 1  # 没双注入
    assert "data_analyst" in found


def test_unknown_subagent_type_passthrough():
    """task subagent_type 未注册 → 不注入,prompt 原样返回。"""
    result = _auto_inject_handoff_placeholders("x", "nonexistent")
    assert result == "x"


def test_chart_maker_gets_code_executor():
    new_prompt = _auto_inject_handoff_placeholders("画图", "chart-maker")
    found = _HANDOFF_PLACEHOLDER_RE.findall(new_prompt)
    assert "code_executor" in found
```

并加 end-to-end 测试(可选,本 task 实施 agent 决定是否值得 fixturize):

```python
def test_task_tool_integration_auto_inject_before_placeholders_resolve():
    """task_tool 调用完成后,执行链:
       1. _auto_inject_handoff_placeholders(prompt, subagent_type)
       2. _resolve_handoff_placeholders(...)  →  authorized_handoff_paths
    最终 authorized_handoff_paths 含 required_upstream_handoffs 对应路径。
    """
    # 该 test 可能需要 mock SubagentExecutor,本 task 实施 agent 决定是否值得
    pass
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PYTHONPATH=. uv run pytest tests/test_task_tool_auto_inject.py -v
```

Expected: ImportError `_auto_inject_handoff_placeholders` 不存在。

- [ ] **Step 3: 绿 — 实现 helper**

Edit `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py`:

(a) 在 line 60(`_resolve_handoff_placeholders` 函数之后)加新 helper:

```python
def _auto_inject_handoff_placeholders(prompt: str, subagent_type: str) -> str:
    """W19: 按 SubagentConfig.required_upstream_handoffs 自动注入 {{handoff://X}}
    占位符,跳过已存在的(避免双注入)。

    Spec §2.2 + R3:
    - lead 派遣 task() 时不写占位符,harness 自动加
    - 已手写的占位符不重复添加(grep prompt 检测)
    - 未注册的 subagent_type 原样返回

    返回新 prompt(字符串)。后续仍走 _resolve_handoff_placeholders 替换为路径 +
    生成 authorized_handoff_paths。
    """
    # Lazy import 避免 cyclic dependency
    from deerflow.subagents.builtins import BUILTIN_SUBAGENTS

    config = BUILTIN_SUBAGENTS.get(subagent_type)
    if config is None or not config.required_upstream_handoffs:
        return prompt

    existing = set(_HANDOFF_PLACEHOLDER_RE.findall(prompt))
    additions: list[str] = []
    for name in config.required_upstream_handoffs:
        if name not in existing:
            additions.append(f"{{{{handoff://{name}}}}}")

    if not additions:
        return prompt

    # 追加注入声明 + 占位符到 prompt 末尾
    return (
        f"{prompt}\n\n"
        f"[Upstream handoff (auto-injected by harness)]\n"
        + "\n".join(f"- {p}" for p in additions)
    )
```

(b) 在 `task_tool` 函数内,line 358 之前加调用:

```python
    # Noldus: resolve {{shared://...}} placeholders to /mnt/shared/... paths.
    prompt = _resolve_placeholders(prompt)

    # W19: 自动注入 required_upstream_handoffs 对应的 {{handoff://X}} 占位符,
    # 避免 lead 派遣时遗漏(由 SubagentConfig 声明 → harness 加 → 后续
    # _resolve_handoff_placeholders 转路径 + 授权)。
    prompt = _auto_inject_handoff_placeholders(prompt, subagent_type)

    # Noldus: resolve {{handoff://...}} placeholders and capture authorized paths.
    # The authorized set is consumed by HandoffIsolationProvider so the subagent
    # can read_file only the handoff JSONs the lead explicitly granted.
    prompt, authorized_handoff_paths = _resolve_handoff_placeholders(prompt)
```

- [ ] **Step 4: 跑测试 + 全套回归**

```bash
PYTHONPATH=. uv run pytest tests/test_task_tool_auto_inject.py tests/test_task_tool_handoff_placeholders.py tests/test_task_handoff_authorization.py -v
PYTHONPATH=. uv run pytest tests/test_harness_boundary.py -v
PYTHONPATH=. uv run pytest tests/ -k "task_tool or subagent" -v 2>&1 | tail -30
```

特别注意:`test_harness_boundary.py` 要求 `packages/harness/deerflow/` 不 import `app.*` — W19 添加的 import 都是 `deerflow.*`,安全。

- [ ] **Step 5: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py \
        packages/agent/backend/tests/test_task_tool_auto_inject.py
git commit -m "$(cat <<'EOF'
feat(task_tool): 自动注入 required_upstream_handoffs 占位符 (WI19)

Lead 派遣 task() 时不写 {{handoff://X}} 占位符 — harness 按
SubagentConfig.required_upstream_handoffs 自动追加,然后
_resolve_handoff_placeholders 转物理路径 + 加入 authorized_handoff_paths。

防双注入:已存在占位符的 name 跳过(R3)。
未注册的 subagent_type pass-through(不抛异常)。

与 W18 TaskHandoffAuthorizationProvider 互补:
- W19 主动注入 → 正常 case 不被 W18 拦
- W18 作为安全网 → 抓 W19 bug 或 lead 手写绕过

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §2.2/R3
EOF
)"
```

---

## Task 22 (W22): Dogfood 三场景 E2E 验证

**Files:** 无代码改动,本 task 是手动 + 半自动的 E2E 验收 + bug 修复回环

**依赖:** 全部前置 task(W1-W21)

**Spec refs:** §10

### 前置准备

- [ ] **Step 0a: 跑全套 unit test 确保 baseline 绿**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run make test 2>&1 | tail -30
cd ../../ethoinsight
PYTHONPATH=. uv run pytest tests/ 2>&1 | tail -20
```

如有 fail 必须先修;不允许带病 dogfood。

- [ ] **Step 0b: 启动 dev 环境**

```bash
cd packages/agent
make dev
```

确认 frontend `localhost:2026` 可访问,后端 LangGraph + Gateway 都 up。

### S1: 5-18 复现(单被试 + "再画几个图")

**前提**:从 demo-data 单被试 EPM 数据,先跑一次 E2E_MIN 让 workspace 有 `handoff_code_executor.json` 和 `plan_metrics.json`。

- [ ] **Step S1.1: 上传单被试 EPM 数据 + "帮我分析一下"**

期望路径:
1. Lead 输出 `[intent] E2E_MIN`(被 W17 通过)
2. `set_experiment_paradigm(paradigm="epm", ...)`
3. `prep_metric_plan(uploaded_file=..., paradigm="epm")` → `plan_metrics.json`
4. `task("code-executor", ...)` — W19 注入空 list(code-executor 无上游),W18 通过
5. code-executor 跑 metrics → `handoff_code_executor.json` + `[gate_signals]`
6. `task("data-analyst", ...)` — W19 注入 `{{handoff://code_executor}}`
7. data-analyst 解读 → `handoff_data_analyst.json` + `[gate_signals]`
8. `ask_clarification` 4-choice

**S1.1 验收**:
- ✅ 没有 LOOP DETECTED 中断
- ✅ ev19_template 已设
- ✅ workspace 含 `plan_metrics.json` (无 `metric_plan.json`)
- ✅ `handoff_code_executor.json` 存在
- ✅ `handoff_data_analyst.json` 存在
- ✅ `ask_clarification` 出现 4 选项

- [ ] **Step S1.2: 用户回复"再画几个图"**

期望路径:
1. Lead 输出 `[intent] CHART`
2. `task("chart-maker", prompt="...再画几个图...")` — W19 注入 `{{handoff://code_executor}}`,W18 通过
3. chart-maker 读 `/mnt/skills/ethoinsight/references/execution-conventions.md` + `/mnt/skills/ethoinsight-chart-maker/SKILL.md`
4. read `handoff_code_executor.json`
5. bash `python -m ethoinsight.catalog.resolve --mode charts --paradigm epm --user-intent "再画几个图" --total-subjects 1 --n-per-group 1 --n-groups 1 ... --output plan_charts.json`
6. read `plan_charts.json` → `charts=[]`, `charts_fallback_available=[trajectory_plot, timeseries_plot]`
7. skill 决策:"几个图" 复数 + 模糊 + 单被试 → 选 fallback 全部
8. bash 跑 plot_trajectory + plot_timeseries
9. write `handoff_chart_maker.json` (fallback_used=true, charts_generated_count=2)
10. `present_files([trajectory.png, timeseries.png])`
11. 输出 `OK: 2 charts generated\n[gate_signals]\n...`

**S1.2 验收(spec §10.1)**:
- ✅ **0 次 LOOP DETECTED**
- ✅ chart-maker 总 bash 调用 **≤ 6 次**(catalog resolve + 2 plot + 最多 2-3 ls/cat)
- ✅ lead **没有** `write_file`(lead 工具已剥除 bash/write_file,Skill 加固后也确认)
- ✅ 用户在前端 outputs/ 看到 trajectory.png + timeseries.png
- ✅ `handoff_chart_maker.json` `fallback_used: true`, `charts_generated_count: 2`

### S2: 正常端到端(EPM 3v3 + "帮我分析一下")

- [ ] **Step S2.1: 上传 EPM 3v3 数据 + "帮我分析一下"**

期望路径同 S1.1,但 `n_per_group=3, n_groups=2` → code-executor 不再走 fallback,catalog 中 EPM charts (`when: n_per_group >= 3`) 满足。

**S2.1 验收(spec §10.2)**:
- ✅ 6 个被试 EPM 默认指标 + 组间 t-test 跑完无错
- ✅ 4-choice ask_clarification 在 turn 4 左右出现
- ✅ statistical_validity = "ok"

- [ ] **Step S2.2: 用户回复"都要"**

期望:lead 先派 chart-maker 后派 report-writer。

- chart-maker 拿到 `plan_charts.charts != []`(catalog 命中,不是 fallback)→ 跑 catalog charts(box_open_arm 等)
- report-writer 拿到 `{{handoff://code_executor}} {{handoff://data_analyst}}` 自动注入 + lead 手动加 `{{handoff://chart_maker}}`(R5 短期 workaround)

**S2.2 验收**:
- ✅ chart-maker `fallback_used: false`(用 catalog 命中)
- ✅ report.md 6 段全写
- ✅ `present_files(report.md)`

### S8: 范式模糊(只说"帮我分析一下" + 文件名不明确)

- [ ] **Step S8.1: 上传 `mouse_data_2026.txt`(无范式线索)+ "帮我分析一下"**

期望路径:
1. Lead 输出 `[intent] CLARIFY`(或先 E2E_MIN 然后转 CLARIFY)
2. read `ethovision-paradigm-knowledge` skill
3. 文件名 + 列名无法唯一确定范式
4. `ask_clarification("我从数据中看到 X / Y / Z 列,可能是 EPM 或 OFT...")`

**S8.1 验收(spec §10.3)**:
- ✅ Lead **不**调 `set_experiment_paradigm(paradigm="epm")`(不默认猜)
- ✅ ask_clarification 文案**带证据**(列名 / 文件名,而非空泛"请问范式是什么")
- ✅ `Ev19TemplateGuardrailProvider` 未触发(因为 lead 在分类后没派 code-executor)
- ✅ W10 删除的 default-template-fallback 没被引用

### Bug 回环:发现问题 → 单独 commit 修

如果 S1/S2/S8 任一失败,**不要在 dogfood task 内修复**,而是:

1. 写新 task `W22.1-fix-<issue>`(在 plan 末尾加 Step 描述)
2. 修复完跑相应 unit test + 重新 dogfood 失败场景
3. 全绿后 commit `fix(...): ... (W22 dogfood)` message

### S0/S?: 额外回归 case(可选)

如果时间允许,补:
- 用户上传后只说"你好" → QA_KNOWLEDGE 派 knowledge-assistant 询问意图
- 已分析后用户问"NND 是什么" → QA_KNOWLEDGE
- 已分析后用户问"开放臂时间为什么 p 不显著" → QA_FACT (knowledge-assistant + lead 手动加 `{{handoff://code_executor}}` 占位符)

### 完工 Commit + 收尾

- [ ] **Step S?.1: 所有 dogfood 场景全绿后**

写一条总结性 commit:

```bash
git commit --allow-empty -m "$(cat <<'EOF'
chore(dogfood): W22 三场景全绿 (S1 5-18 复现 + S2 EPM 3v3 + S8 范式模糊)

S1 (单被试 + 再画几个图):
- 0 次 LOOP DETECTED
- chart-maker bash ≤ 6 次
- lead 未自己 write_file Python
- fallback_used=true, 2 charts (trajectory + timeseries)

S2 (EPM 3v3 + 分析一下):
- 6 被试 metrics + t-test 跑完
- 4-choice ask_clarification at turn ~4
- chart-maker fallback_used=false (用 catalog 命中)
- report.md 6 段齐全

S8 (范式模糊):
- lead 不默认猜
- ask_clarification 带证据(列名 / 文件名)
- Ev19TemplateGuardrailProvider 未触发

dogfood 验收完毕,可以 PR 合 dev。

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md §10
EOF
)"
```

- [ ] **Step S?.2: 删 worktree 上的过渡 alias**

W2 时保留的 `Plan` 老 dataclass 现在可以删了(`resolve()` wrapper / `plan_to_dict()` wrapper 也跟着删,因为已经没人调它们 — 前置 task 全部切到 `resolve_metrics` / `resolve_charts` / `plan_metrics_to_dict` / `plan_charts_to_dict`)。

```bash
# 检查是否还有 caller
grep -rn "from ethoinsight.catalog.resolve import.*resolve\b\|plan_to_dict\b" packages/ 2>/dev/null | grep -v __pycache__ | head -20
grep -rn "from ethoinsight.catalog.schema import Plan\b" packages/ 2>/dev/null | head -20
```

如果有 leftover caller,先迁移再删。删后跑全套测试:

```bash
cd packages/ethoinsight && PYTHONPATH=. uv run pytest tests/ 2>&1 | tail -10
cd ../agent/backend && PYTHONPATH=. uv run make test 2>&1 | tail -10
```

Commit:`refactor(catalog): 删除过渡期 Plan / resolve / plan_to_dict (W22 收尾)`。

- [ ] **Step S?.3: PR 合 dev**

```bash
# 推送 branch
git push -u origin worktree-subagent-role-split-impl

# 创建 PR
gh pr create --title "feat: Subagent 角色拆分 + Capability-Exposure 重构 (W1-W22)" --body "$(cat <<'EOF'
## Summary

- 用 capability-exposure 重构 lead → 5 subagent 调度
- SubagentConfig 加 4 capability metadata 字段
- 新建 chart-maker subagent(承担"画图"职责,从 code-executor 剥离)
- 新建 2 个 GuardrailProvider:IntentClassification + TaskHandoffAuthorization
- task_tool 自动注入 required_upstream_handoffs 占位符
- Plan dataclass 拆 PlanMetrics + PlanCharts;catalog CLI 加 --mode
- _common.yaml 加 trajectory + timeseries 作为 fallback chart
- Lead prompt 从 1243 行瘦身到 ~200 行,detail 移到 ethoinsight-lead-interaction skill
- 删除 default-template-fallback(Gate before guess)

## Test plan

- [x] 全套 unit test 绿(W1-W21 各 task 自带)
- [x] S1: 单被试 + 再画几个图 → 0 LOOP DETECTED,chart-maker bash ≤ 6 次
- [x] S2: EPM 3v3 + 分析一下 → 4-choice ask_clarification,catalog charts 命中
- [x] S8: 范式模糊 → ask_clarification 带证据,不默认猜

Refs: docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## 备忘:风险 + 应急

(spec §11 风险登记的实施期具体应对)

| Risk ID | 应急 |
|---|---|
| R1 (W16 瘦身丢约束) | S1/S2/S8 任一 fail → 找出丢失约束,加回 prompt(以最小增量) |
| R2 (W17 误拦 read_file) | 测试已覆盖 `test_allow_read_file_always`,实施时再加 fuzz fixture |
| R3 (W19 双注入) | 测试已覆盖 `test_no_double_inject` |
| R4 (chart-maker sandbox 路径) | W22 S1 在 sandbox 中跑 catalog.resolve --mode charts 验 |
| R5 (chart_maker handoff optional) | 短期 lead 手动加 `{{handoff://chart_maker}}`;长期 future spec 加 `optional_upstream_handoffs` 字段 |
| R6 (lead read raw txt 复发) | ethoinsight-lead-interaction skill 已写"禁止 read_file raw 数据";S8 验收时检查 |
| R7 (merge conflict) | 每个 WI 单独 commit + 每个 PR 合 dev 前 rebase |
| R8 (默认 y_col 选错) | W6 `_DEFAULT_Y_COL_BY_PARADIGM` 保守选择;chart-maker --y-col 可覆盖 |
| R9 (第 3 次复现) | S1 必须 0 LOOP DETECTED;有 W17/W18 兜底拦截 |
