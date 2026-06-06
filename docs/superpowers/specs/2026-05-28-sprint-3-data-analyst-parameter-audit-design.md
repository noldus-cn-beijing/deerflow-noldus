# Sprint 3 实施 spec — data-analyst 参数审计

**关联**：[2026-05-28 SOTA agent 路线图 v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) Sprint 3
**估期**：1 周
**前置**：**Sprint 2b 必须先合 dev**（Sprint 2b 填实 `MetricStat.parameters_used`；Sprint 3 消费这字段做参数适配性比对）
**执行者**：交给独立 agent 执行

---

## 1. 背景与目标

### 现状（Sprint 2b 之后）

- 每条 `MetricStat.parameters_used` 含本次执行的实际参数值（`{"velocity_threshold": 30.0, "velocity_min_duration": 25}`）
- `analysis_config_id` 已与 handoff 关联（Sprint 4.5 之前可能仍是 PENDING 占位，不影响 Sprint 3）
- code-executor 的 handoff 已含 `per_subject` 数据分布、`metrics_summary` 统计摘要
- 用户可调参数已在 catalog 标 `tunable_by_user: true`（Sprint 2a 加的 `ParamSpec.tunable_by_user`）

### Sprint 3 要解决的问题

参数虽可调，但 agent 不会判断"当前参数适不适合这批数据"。例子：

| 场景 | 当前 | Sprint 3 后 |
|---|---|---|
| FST 用 `velocity_threshold=30 mm/s`，但本批数据 velocity 中位数 5 mm/s（鱼游速本就慢）| 不警告，继续算 → immobility 时间会被严重低估 | data-analyst 发警告 "参数与数据分布不匹配：阈值 30 但中位数仅 5，建议调到 ≤10" |
| EPM 默认 `total_entry_threshold=8`（运动抑制警戒），但数据 entries 普遍 3-5 | 多数 subject 被打 LOW_ENTRIES 警告 | data-analyst 警告"全样本 entries 偏低，可能 paradigm 实验设计或 protocol 问题，并非个体异常" |

### 关键原则

1. **只警告，不调参** — Sprint 3 不允许 data-analyst 自动改 overrides。调参权交给用户（Sprint 4 提供调参指南，Sprint 4.5 提供 overrides 写入路径）
2. **诚实标注边界** — 参数审计只能识别"分布层面不匹配"，识别不出深层 protocol 问题。审计结论必须含"建议研究者确认 X、Y、Z 后再调"
3. **不替代 data quality warnings** — 已有 `DataQualityWarning`（dispatcher 端 SAMPLE/MOTOR/SIGNAL/METHOD）是"数据本身的"问题，Sprint 3 的 `ParameterAuditFinding` 是"参数与数据匹配度"问题，两者正交

---

## 2. 文件改动清单

### 2.1 新增 schema `ParameterAuditFinding`

**位置**：`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`

在 `DataQualityWarning`（约 line 80）之后插入：

```python
class ParameterAuditFinding(BaseModel):
    """Single parameter-vs-data-distribution mismatch finding from data-analyst."""

    parameter: str = Field(
        description=(
            "Parameter name as it appears in MetricStat.parameters_used, "
            "e.g. 'velocity_threshold', 'total_entry_threshold'."
        ),
    )
    metric: str = Field(
        description="Affected metric slug, e.g. 'immobility_time', 'total_entry_count'."
    )
    severity: Literal["critical", "warning", "info"]
    used_value: float | int | str = Field(
        description="Parameter value actually used in the run (from MetricStat.parameters_used)."
    )
    observed_distribution: dict[str, float | int] = Field(
        description=(
            "Snapshot of the data distribution that triggered the finding, e.g. "
            "{'median': 5.0, 'p90': 12.0, 'max': 25.0, 'n_subjects': 12}. "
            "Used by the report writer and the hypothesis panel (Sprint 7)."
        ),
    )
    mismatch_kind: Literal[
        "threshold_too_high",  # 阈值远高于数据上限/中位数
        "threshold_too_low",   # 阈值远低于数据下限/中位数
        "window_too_wide",     # 窗口超出 trial 时长
        "window_too_narrow",   # 窗口过窄无法捕捉事件
        "category_mismatch",   # 离散参数取值与 paradigm 不符
    ]
    suggestion: str = Field(
        description=(
            "Plain-Chinese guidance for the researcher. e.g. "
            "'当前阈值 30 mm/s 高于本批中位数 5 mm/s 的 6 倍，建议改至 ≤10 mm/s 后重跑'. "
            "MUST NOT include exact override values — that's Sprint 4 paradigm md's job."
        ),
    )
    blocks_downstream: bool = Field(
        default=False,
        description=(
            "When True, chart-maker / report-writer should annotate the affected "
            "metric as 'parameter-suspect'. Sprint 5 GuardrailProvider may also "
            "block downstream subagent dispatch in manual mode."
        ),
    )

    @field_validator("parameter")
    @classmethod
    def _validate_parameter(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("parameter must be a non-empty identifier")
        return v
```

**注意**：`mismatch_kind` 是 5 元枚举，确保 LLM 不能自由发明（首发只覆盖 5 类，未来需要扩枚举值时显式扩 Literal）。

### 2.2 扩展 `DataAnalystHandoff`

**位置**：`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py:323`

加 `parameter_audit_findings` 字段：

```python
class DataAnalystHandoff(BaseModel):
    # ... existing fields ...
    parameter_audit_findings: list[ParameterAuditFinding] = Field(
        default_factory=list,
        description=(
            "Sprint 3 新增。data-analyst 比对 MetricStat.parameters_used 与 "
            "handoff_code_executor 中的 per_subject 数据分布后产出的不匹配清单。"
            "下游 report-writer 会读此字段写入'数据质量与局限'段；前端 "
            "QualityWarningBanner 不读这个字段（它只显示 quality_warnings）。"
        ),
    )
```

### 2.3 扩展 `GateSignals`

**位置**：`handoff_schemas.py:129`

加 `parameter_audit_findings_count`：

```python
class GateSignals(BaseModel):
    # ... existing fields ...
    parameter_audit_findings_count: int = Field(
        default=0,
        description=(
            "Sprint 3 新增。data-analyst 看到的 parameter_audit_findings 总数 "
            "(critical+warning+info 合计)。lead 据此决定是否在播报模板中提及。"
        ),
    )
    parameter_audit_critical_count: int = Field(
        default=0,
        description=(
            "Sprint 3 新增。parameter_audit_findings 中 severity=='critical' 且 "
            "blocks_downstream=True 的条目数。Sprint 5 manual 模式下 guardrail "
            "可据此拦截下游 subagent。"
        ),
    )
```

### 2.4 扩展 `seal_data_analyst_handoff` tool

**位置**：`packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py`

```python
@tool("seal_data_analyst_handoff", parse_docstring=True)
def seal_data_analyst_handoff(
    status: str,
    key_findings: list[str] | None = None,
    outlier_findings: list[dict[str, Any]] | None = None,
    excluded_metrics: list[str] | None = None,
    method_warnings: list[str] | None = None,
    recommendations: list[str] | None = None,
    errors: list[str] | None = None,
    gate_signals: dict[str, Any] | None = None,
    quality_warnings: list[dict[str, Any]] | None = None,
    # === Sprint 3 新增 ===
    parameter_audit_findings: list[dict[str, Any]] | None = None,
    runtime: ToolRuntime[ContextT, ThreadState] = None,
) -> str:
    """..."""
    payload = {
        # ... existing fields ...
        "parameter_audit_findings": parameter_audit_findings or [],
    }
    return _seal_handoff(DataAnalystHandoff, "handoff_data_analyst.json", payload, runtime)
```

### 2.5 改动 `data_analyst.py` workflow

**位置**：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`

在步骤 2.5（quality warnings）和 2.6（按范式 read 判读文档）之间插入步骤 2.7：

```
2.7 **参数适配性审计**（Sprint 3 新增）：
    从 handoff_code_executor.json 取 metric_stats，逐条比对
    parameters_used 与 per_subject 数据分布：

    a. 对每个有 parameters_used 的 metric：
       - velocity_threshold：取 per_subject[*].velocity 的中位数 / p90 / max
       - total_entry_threshold：取 per_subject[*].total_entry_count 分布
       - window_seconds：取 trial 时长（per_subject[*].duration_seconds）
       - (其他参数按需扩展)

    b. 判定 mismatch_kind：
       - threshold_too_high：used_value > p90 数据值 × 3
       - threshold_too_low：used_value < p10 数据值 ÷ 3
       - window_too_wide：window > trial_duration × 0.9
       - window_too_narrow：window < trial_duration × 0.05
       - category_mismatch：枚举参数取值与本范式标准不符

    c. severity 判定（保守）：
       - critical（blocks_downstream=true）：所有 subject 的某指标都受影响
       - warning：≥50% subject 受影响
       - info：单纯阈值落在边界值附近

    d. suggestion 字段：
       - 描述偏差量（如 "阈值 30 高于中位数 5 的 6 倍"）
       - 提示用户参考 paradigm md 的"参数调整指南"段（Sprint 4 产出）
       - **严禁**自己给出具体调到多少的数字 — 那是 paradigm md 的职责

    e. 写到 ParameterAuditFinding list；
       gate_signals.parameter_audit_findings_count = len(list)；
       gate_signals.parameter_audit_critical_count = sum(critical+blocks=true)；
       透传到 seal_data_analyst_handoff 的 parameter_audit_findings 参数。
```

在 system_prompt 的 `[gate_signals]` 模板段（约 line 130）加：

```
parameter_audit_findings_count: <int>      # Sprint 3 新增
parameter_audit_critical_count: <int>      # Sprint 3 新增
```

### 2.6 改动 lead prompt 播报模板

**位置**：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

在 Sprint 1 加的"data-analyst 阻断级质量警告播报"段之后追加：

```
- **data-analyst 参数审计播报**（Sprint 3）：收到 data-analyst handoff 后，
  如果 gate_signals.parameter_audit_findings_count > 0，向用户播报：
  "已收到 data-analyst 结果，发现 <N> 条参数适配性问题:
  - [<severity>] <parameter> 与数据分布不匹配: <suggestion>
  ..."
  用 suggestion 字段呈现（**不念 observed_distribution dict** — 那是给后续
  hypothesis panel 用的）。
  如果 parameter_audit_findings_count = 0，不额外提及参数审计。

  注意：parameter_audit_findings 与 quality_warnings 是**正交**两类信号：
  - quality_warnings: 数据本身的问题（n<3, motor low, signal poor）
  - parameter_audit_findings: 参数与数据分布不匹配
  播报时各自一段，不混在一起。
```

### 2.7 frontend 渲染（占位，非阻塞）

**位置**：`packages/agent/frontend/src/components/workspace/messages/`

Sprint 3 **不强制**前端 banner 集成（避免 banner 越来越花）。但为 Sprint 7 假设面板做准备：

- `utils.ts` 加 `extractParameterAuditFindings(message)` helper（读 `additional_kwargs.parameter_audit_findings`，但 Sprint 3 阶段后端**不注入** — `QualityWarningBroadcastMiddleware` 不扩展到 parameter_audit）
- 前端短期内只能通过 report.md 看 parameter_audit 内容

**为何不复用 QualityWarningBroadcastMiddleware**：banner 是 critical+blocks 的"中断信号"，parameter_audit 多数是 info/warning，前端如果两种都用 banner 会噪声化。Sprint 7 假设面板（独立 UI 控件）是更合适的容器。

### 2.8 单元测试

新增 `packages/agent/backend/tests/test_parameter_audit_schema.py`：

| 测试 | 验证 |
|---|---|
| `test_valid_finding_construction` | 5 个字段都能正常构造 |
| `test_mismatch_kind_must_be_enum` | mismatch_kind 必须是 5 元枚举之一 |
| `test_parameter_must_be_nonempty` | parameter 字段非空 |
| `test_observed_distribution_accepts_numeric_only` | observed_distribution 值必须是 float/int |
| `test_blocks_downstream_default_false` | blocks_downstream 默认 False |
| `test_severity_critical_does_not_auto_block` | severity=critical 不自动让 blocks_downstream=True（必须显式设） |
| `test_data_analyst_handoff_carries_findings` | DataAnalystHandoff 能携带 list[ParameterAuditFinding] |
| `test_gate_signals_carries_counts` | parameter_audit_findings_count + parameter_audit_critical_count 默认 0 |

新增 `packages/agent/backend/tests/test_seal_data_analyst_parameter_audit.py`：

| 测试 | 验证 |
|---|---|
| `test_seal_writes_findings_to_handoff` | seal_data_analyst_handoff 透传 findings 到 JSON 文件 |
| `test_seal_validates_finding_schema` | 非法 mismatch_kind → Pydantic 拒绝 |
| `test_seal_handles_empty_findings` | 不传 findings → 默认 [] |

### 2.9 集成测试

新增 `packages/agent/backend/tests/test_lead_parameter_audit_broadcast.py`：

| 测试 | 验证 |
|---|---|
| `test_lead_prompt_template_mentions_audit_when_count_positive` | 模拟 gate_signals.parameter_audit_findings_count=3 → lead prompt 模板含 "参数审计" 播报段 |
| `test_lead_prompt_skips_audit_when_count_zero` | count=0 → 不出现"参数审计"段 |

---

## 3. 实施顺序（task 拆分）

| Task | 内容 | 估时 |
|---|---|---|
| T1 | handoff_schemas.py 加 `ParameterAuditFinding` schema | 0.5 天 |
| T2 | 扩展 DataAnalystHandoff.parameter_audit_findings + GateSignals 加 2 个计数字段 | 0.25 天 |
| T3 | seal_data_analyst_handoff 加 parameter_audit_findings 参数 | 0.25 天 |
| T4 | tests/test_parameter_audit_schema.py 编写 8 个 schema 测试 | 0.5 天 |
| T5 | data_analyst.py prompt 加步骤 2.7（参数审计 workflow）+ gate_signals 模板更新 | 1 天 |
| T6 | lead prompt 加参数审计播报段 | 0.25 天 |
| T7 | frontend/utils.ts 加 extractParameterAuditFindings helper（占位） | 0.25 天 |
| T8 | 集成测试（lead 模板触发 + dispatcher 不退化） | 0.5 天 |
| T9 | dogfood：FST n=5（velocity 中位数低）触发 threshold_too_high 警告 | 0.5 天 |
| T10 | 全量回归 + 修退化 | 0.5 天 |
| **合计** | | **4.5 天 ≈ 1 周** |

---

## 4. 风险与缓解

| 风险 | 缓解 |
|---|---|
| data-analyst LLM 计算分布数字错误（如把 p90 算成 mean）| T5 prompt 明确给出"取 numpy.percentile(data, 90)"等指令；T9 dogfood 时验证 observed_distribution 数字与原始数据一致 |
| data-analyst 自作主张给出具体 override 数字（违反"只警告不调参"原则）| T5 prompt 明确"严禁数字 suggestion"；T9 dogfood 时扫 suggestion 字段不含 `=` 后跟数字的形式 |
| ParameterAuditFinding 与 quality_warnings 在 UX 上混淆 | Sprint 3 不集成 banner，T6 lead prompt 明确"两类信号各自一段不混合"；Sprint 7 假设面板提供独立容器 |
| Sprint 2b 完工后 MetricStat.parameters_used 仍有部分 metric 为 `{}`（不可调参数的 metric）| T5 prompt 跳过空 parameters_used 的 metric，不产生 audit 结论 |
| LLM 自由发明 mismatch_kind 值（如 "data_outlier"）| T1 用 Literal[5 元枚举]，Pydantic 直接拒绝 |
| 与 Sprint 4 paradigm md "参数调整指南"段时序耦合 | Sprint 3 的 suggestion 只指向"请参考 paradigm md 的参数调整指南段"——即便 Sprint 4 还没产，suggestion 文字仍然合理（只是用户看到指引时找不到内容，Sprint 4 完工后自动补齐） |

---

## 5. 验收 checklist

实施完成时，确认全部通过：

- [ ] `ParameterAuditFinding` schema 已加，6 字段 + 5 元枚举 mismatch_kind + Pydantic 校验
- [ ] DataAnalystHandoff 增 `parameter_audit_findings` 字段
- [ ] GateSignals 增 `parameter_audit_findings_count` + `parameter_audit_critical_count`
- [ ] seal_data_analyst_handoff 支持 parameter_audit_findings 参数 + atomic write 落 handoff JSON
- [ ] data_analyst subagent prompt 含步骤 2.7（5 类 mismatch 判定 + severity 三档 + 严禁数字 suggestion）
- [ ] lead prompt 含 parameter_audit_findings_count>0 时播报模板
- [ ] frontend utils.ts 含 extractParameterAuditFindings helper（即使后端不注入，前端 type 就位）
- [ ] 单元测试 8 个 schema case + 3 个 seal case 全绿
- [ ] 集成测试 2 个 prompt 模板 case 全绿
- [ ] dogfood：FST（mock 数据 velocity 中位数 5、threshold 30）→ data-analyst handoff 出现 `threshold_too_high` finding + lead 播报包含参数审计段
- [ ] dogfood：EPM（数据 velocity 与 catalog default 匹配）→ data-analyst handoff parameter_audit_findings 为 [] + lead 不播报参数审计段
- [ ] 全量测试通过（ethoinsight ≥477，agent backend ≥3111 + Sprint 3 新增 ~13）
- [ ] `git grep "parameter_audit"` 在 handoff_schemas.py / seal_handoff_tools.py / data_analyst.py / lead_agent/prompt.py 均能找到，证明四处契约就位

---

## 6. 不在 Sprint 3 范围

明确**不做**的事：

- ❌ 给 ParameterAuditFinding 加 banner 渲染（Sprint 7 假设面板做，避免 banner 噪声化）
- ❌ data-analyst 自动改 overrides（违反"只警告不调参"原则）
- ❌ paradigm md "参数调整指南"段的具体内容（Sprint 4 做）
- ❌ Guardrail 拦截 `task` 派遣（Sprint 5 做，可读 parameter_audit_critical_count）
- ❌ analysis_config_id 计算（Sprint 4.5 做）
- ❌ Lineage 封印（Sprint 5.5 做）
- ❌ 扩展 mismatch_kind 枚举（首发只 5 类；如果 dogfood 暴露新类型再加，本 sprint 不预扩）

---

## 7. 参考

- [SOTA agent 路线图 v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) — Sprint 3 章节
- [Sprint 0 spec](2026-05-28-sprint-0-handoff-schema-foundation-design.md) — `MetricStat.parameters_used` 字段定义
- [Sprint 1 spec](2026-05-28-sprint-1-data-quality-structured-design.md) — DataQualityWarning（与 ParameterAuditFinding 正交参考）
- [Sprint 2b spec](2026-05-28-sprint-2b-parameter-pipeline-design.md) — parameters_used 实际填值路径
- handoff_schemas.py：`MetricStat`（line 47）、`DataQualityWarning`（line 80）、`GateSignals`（line 129）、`DataAnalystHandoff`（line 323）
- data-analyst subagent：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`
- lead prompt：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
