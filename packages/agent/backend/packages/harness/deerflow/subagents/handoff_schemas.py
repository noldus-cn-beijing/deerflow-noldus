"""Pydantic schemas for subagent handoff JSON contracts.

These schemas define the contract between subagents (code-executor, data-analyst,
report-writer) and the lead agent. Lead validates handoff files against these
schemas to detect contract failures clearly, rather than absorbing malformed
output silently.

See docs/plans/2026-04-20-ethoinsight-pipeline-redesign.md section 5 (L1 —
Handoff contract enforcement).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Virtual path contract: all subagent handoff JSON files must reference user-data
# files via virtual paths (e.g. /mnt/user-data/uploads/x.txt), never host absolute
# paths (e.g. /home/.../user-data/uploads/x.txt). Downstream consumers like
# chart-maker pass these paths to sandbox-enforced CLIs, which reject host paths.
#
# This invariant is enforced at the schema level so that a subagent leaking a host
# path via Path.resolve() fails fast at handoff parse time rather than silently
# propagating through the pipeline (see project_2026-05-26 path-pollution-defense).
_VIRTUAL_USER_DATA_PREFIX = "/mnt/user-data/"

# Allowed DOT-prefixes for DataQualityWarning.code.
# Imported by downstream normalization logic — single source of truth.
WARNING_CODE_PREFIXES = frozenset({"SAMPLE", "MOTOR", "SIGNAL", "METHOD"})


def _validate_virtual_user_data_paths(paths: list[str]) -> list[str]:
    """Validate that every path starts with /mnt/user-data/ (the sandbox virtual prefix).

    Empty list is accepted (handoff may legitimately reference no raw files).
    Raises ValueError listing every offender so the subagent can fix them all at once.
    """
    if not paths:
        return paths
    offenders = [p for p in paths if not (isinstance(p, str) and p.startswith(_VIRTUAL_USER_DATA_PREFIX))]
    if offenders:
        raise ValueError(
            f"raw_files must use virtual paths under {_VIRTUAL_USER_DATA_PREFIX!r}, "
            f"got host-side or malformed entries: {offenders}. "
            "Copy paths verbatim from plan_metrics.json; do not run Path.resolve() / realpath."
        )
    return paths


class TaskContext(BaseModel):
    """任务状态包——只装「产出方独有、消费方无法自行推导」的执行事实。

    由 seal 工具在封存时确定性组装（不由 LLM 填）。
    刻意不含 objective/next_steps/constraints：前两者的真相源是 lead 的意图判断
    （知识双存禁忌），constraints 无干净确定性源。详见 spec 设计原则。

    ⚠️ 价值现状（2026-06-04 三轮真实数据核实）：当前 EthoInsight 拓扑是
    「subagent → lead 中转 → 下一个」，非「subagent 直接接力」，且 partial 多为
    「统计因样本量被跳过」而非「未完成」。经核实，本结构当前字段对下游 subagent
    冗余（下游消费原始字段），对 lead 多被 gate_signals 覆盖。故：
    - 下游 subagent prompt **不消费** task_context（保持现状，勿加"教消费"指引）。
    - 真实用户暂定为 audit/lineage（handoff 自描述）。
    - pending_items 当前恒空（无可靠未完成明细源，见 seal_handoff_tools._build_task_context 注释）。
    TODO: task_context 的整体价值待 v1.0「subagent 直接接力」拓扑或真实
    「脚本失败型 partial」场景出现后重评估。届时参考本类的字段设计。
    """

    model_config = ConfigDict(extra="allow")

    file_changes: list[str] = Field(
        default_factory=list,
        description="本 subagent 创建/修改的产物文件虚拟路径（seal 从 output_files 自动提取）。",
    )
    verify_commands: list[str] = Field(
        default_factory=list,
        description="下游验证本 handoff 完整性的命令（seal 按模板自动生成）。",
    )
    failed_paths: list[str] = Field(
        default_factory=list,
        description="已尝试且失败、下游不应重试的方法（seal 从 errors 自动派生）。",
    )
    pending_items: list[str] = Field(
        default_factory=list,
        description="未完成项。当前恒空（真实 partial 为统计跳过非未完成，无可靠数据源）。详见类 docstring。",
    )


class MetricStat(BaseModel):
    """Summary statistics for a single metric within a single group.

    Allow arbitrary extra fields from ethoinsight so new metric-level statistics
    (e.g. median, IQR) do not break older handoff files.
    """

    model_config = ConfigDict(extra="allow")

    mean: float | None = None
    std: float | None = None
    n: int | None = None
    applicable: bool = Field(
        default=True,
        description="False when the metric is not applicable (e.g. group metric on single-subject file).",
    )
    reason: str | None = Field(
        default=None,
        description="Present when applicable=False; explains why.",
    )
    parameters_used: dict[str, float | int | str | None] = Field(
        default_factory=dict,
        description=(
            "Actual parameters used to compute this metric, e.g. "
            "{'velocity_threshold': 30.0, 'velocity_min_duration': 25}. "
            "Populated by Sprint 2b execution pipeline. "
            "Sprint 0 only defines the field; defaults to empty dict. "
            "None is allowed for individual values when the param is not "
            "applicable to this metric (e.g. pendulum params on EPM)."
        ),
    )


class DataQualityWarning(BaseModel):
    """Single quality warning attached to a handoff.

    Normalisation (underscores→DOT, metric=None→"all", evidence=str→wrapped)
    is applied transparently by the model_validator(mode="before") above,
    before the field_validator("code") runs strict namespace checks.
    """

    model_config = ConfigDict(extra="allow")

    severity: Literal["critical", "warning", "info"]
    metric: str = Field(
        description="Metric name, or 'all' / 'pipeline' when warning applies broadly.",
    )
    message: str
    code: str = Field(
        description=(
            "Warning code in dotted form, e.g. 'SAMPLE.TOO_SMALL', 'MOTOR.LOW_VELOCITY'. "
            "First segment must be one of: SAMPLE / MOTOR / SIGNAL / METHOD. "
            "See Sprint 1 spec for full code taxonomy."
        ),
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured numeric evidence, e.g. "
            "{'velocity_median_mm_s': 5.2, 'threshold_mm_s': 30.0}. "
            "Frontend / data-analyst should not parse `message` for numbers."
        ),
    )
    blocks_downstream: bool = Field(
        default=False,
        description=(
            "True = downstream subagents (data-analyst, chart-maker, report-writer) "
            "should not be dispatched without user acknowledgement. "
            "Used by Sprint 5 DataQualityGuardrailProvider (manual mode) "
            "and Sprint 1 frontend (red vs orange rendering)."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_llm_typeros(cls, values: dict[str, Any] | Any) -> Any:
        """Conservatively normalise common LLM typos before strict validation.

        Only touches fields that are unambiguous — never guesses semantics.
        Triggered by key presence, so missing keys are left untouched.
        Runs before field_validator("code") so underscore→DOT happens first.
        """
        if not isinstance(values, dict):
            return values

        # 1. code: underscore → DOT when first segment is a known prefix.
        #    e.g. "SAMPLE_TOO_SMALL" → "SAMPLE.TOO_SMALL"
        #    Pure semantic errors like "insufficient_sample" are left as-is
        #    (no known prefix match → no transformation).
        code_val = values.get("code")
        if isinstance(code_val, str) and "_" in code_val:
            first_segment = code_val.split("_")[0].upper()
            if first_segment in WARNING_CODE_PREFIXES:
                values["code"] = code_val.replace("_", ".", 1)

        # 2. metric: None / missing → "all"
        metric_val = values.get("metric")
        if metric_val is None:
            values["metric"] = "all"

        # 3. evidence: str → {"note": <original>}; non-dict non-str → {}
        evidence_val = values.get("evidence")
        if isinstance(evidence_val, str):
            values["evidence"] = {"note": evidence_val}
        elif evidence_val is not None and not isinstance(evidence_val, dict):
            values["evidence"] = {}

        return values

    @field_validator("code")
    @classmethod
    def _validate_code_namespace(cls, v: str) -> str:
        allowed = {"SAMPLE", "MOTOR", "SIGNAL", "METHOD"}
        head = v.split(".", 1)[0] if "." in v else ""
        if head not in allowed:
            raise ValueError(
                f"warning code must use literal DOT '.' as separator, e.g. "
                f"'SAMPLE.TOO_SMALL', 'MOTOR.LOW_VELOCITY', 'SIGNAL.TRACKING_LOST', "
                f"'METHOD.SHAPIRO_INAPPLICABLE'. NOT underscore-separated ('SAMPLE_X' is INVALID). "
                f"First segment must be one of {sorted(allowed)} followed by '.'. "
                f"Got {v!r}."
            )
        return v


class ParameterAuditFinding(BaseModel):
    """Single parameter-vs-data-distribution mismatch finding from data-analyst.

    Sprint 3: data-analyst compares MetricStat.parameters_used against
    per_subject data distribution to detect mismatches (e.g. velocity_threshold=30
    but data median=5). This is distinct from DataQualityWarning (which flags
    data-level problems); ParameterAuditFinding flags parameter-vs-data mismatches.

    Phase 1.5: model_validator(mode="before") normalises common LLM mistakes
    (used_value=None → "", non-numeric observed_distribution values stripped)
    before strict field validation runs — mirrors DataQualityWarning._normalize_llm_typeros.
    """

    model_config = ConfigDict(extra="allow")

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
        "threshold_too_low",  # 阈值远低于数据下限/中位数
        "window_too_wide",  # 窗口超出 trial 时长
        "window_too_narrow",  # 窗口过窄无法捕捉事件
        "category_mismatch",  # 离散参数取值与 paradigm 不符
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

    @model_validator(mode="before")
    @classmethod
    def _normalize_audit_finding(cls, values: dict[str, Any] | Any) -> Any:
        """Conservatively normalise common LLM mistakes before strict validation.

        Only touches fields that are unambiguous — never guesses semantics.
        Mirrors DataQualityWarning._normalize_llm_typeros pattern.

        Handles degenerate findings from step 2.8 "skip/info" exits where
        deepseek may fill used_value=None or put explanatory text in
        observed_distribution (e.g. {"note": "文字"} instead of numeric dict).
        """
        if not isinstance(values, dict):
            return values

        # 1. used_value=None → "" (schema requires float|int|str; degenerate
        #    scene prompt should fill real param value, this catches edge leaks)
        if values.get("used_value") is None:
            values["used_value"] = ""

        # 2. observed_distribution: strip non-numeric values (schema requires
        #    dict[str, float|int]). {"note": "文字"} / any str values → removed;
        #    if all keys stripped → {}. Explicit None / non-dict (when the key is
        #    present) → {} (degenerate skip exit: deepseek may send null/text).
        #    A *missing* key is left untouched so genuine omissions still fail loud.
        if "observed_distribution" in values:
            od = values["observed_distribution"]
            if isinstance(od, dict):
                values["observed_distribution"] = {
                    k: v for k, v in od.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
            else:
                # None, str, list, etc. → empty numeric dict
                values["observed_distribution"] = {}

        return values

    @field_validator("parameter")
    @classmethod
    def _validate_parameter(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("parameter must be a non-empty identifier")
        return v


class GateSignals(BaseModel):
    """Structured decision signals from subagent to lead.

    Lead reads these from subagent's final AIMessage (not from the handoff
    file itself) to make Step 1.5 quality-gate decisions without inflating
    context with the full handoff JSON. Also persisted in handoff JSON for
    audit/replay (optional field).
    """

    model_config = ConfigDict(extra="allow")

    data_quality: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Summary of data_quality_warnings: "
            "{'critical_count': int, 'warning_count': int, "
            "'critical_items': [str, ...]  # 关键 critical 条目摘要，每条 <80 字}"
        ),
    )
    statistical_validity: Literal["ok", "warning", "failed", "skipped"] = "ok"
    errors_count: int = 0
    quality_warnings_critical_count: int = Field(
        default=0,
        description=(
            "data-analyst 看到的 critical + blocks_downstream=true 警告数, "
            "lead 据此判断是否需要 ask_clarification (Sprint 5 in manual mode)。"
        ),
    )
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


class CodeExecutorInputs(BaseModel):
    """Inputs block recorded by code-executor (raw files + grouping + columns).

    Subagent must copy raw_files verbatim from plan_metrics.json.inputs.raw_files;
    field_validator enforces that paths stay virtual.
    """

    model_config = ConfigDict(extra="allow")

    raw_files: list[str] = Field(
        default_factory=list,
        description="Virtual paths (/mnt/user-data/uploads/...) of input trajectory files.",
    )
    groups: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional group assignment dict, e.g. {'Arena 1': 'Treatment'}.",
    )

    @field_validator("raw_files")
    @classmethod
    def _enforce_virtual_paths(cls, v: list[str]) -> list[str]:
        return _validate_virtual_user_data_paths(v)


class CodeExecutorHandoff(BaseModel):
    """Handoff JSON produced by the code-executor subagent (SOTA glue-script architecture).

    Written by the glue script to handoff_code_executor.json in the workspace.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["completed", "partial", "failed"]
    summary: str
    inputs: CodeExecutorInputs | None = Field(
        default=None,
        description=(
            "Inputs block (raw_files, groups, ...). Optional for backwards compatibility "
            "with older handoff files; new code-executor invocations should populate it."
        ),
    )
    output_files: dict[str, Any] = Field(
        default_factory=dict,
        description="Map of category (metrics/statistics/charts) to path(s).",
    )
    metrics_summary: dict[str, dict[str, MetricStat]] = Field(
        default_factory=dict,
        description="Nested map: group -> metric -> stats.",
    )
    per_subject: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Raw per-subject metric values: {subject_name: {metric: value, ...}}. "
            "Downstream data-analyst uses this to identify outlier subjects by "
            "name and compute leave-one-out counterfactual group statistics. "
            "Phase 2: subject dict may also contain '_signal_distributions' key "
            "(namespace prefix '_') mapping metric → {p10, p90, median, max, n_frames, signal_key}, "
            "providing per-subject frame-level signal distribution for parameter audit. "
            "Old code that iterates metric scalar values should skip '_'-prefixed keys."
        ),
    )
    statistics: dict[str, Any] = Field(default_factory=dict)
    assessment: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    data_quality_warnings: list[DataQualityWarning] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional overall confidence score in [0,1].",
    )
    gate_signals: GateSignals | None = Field(
        default=None,
        description=(
            "Structured signals for lead's decision-making. "
            "Optional in JSON file (lead reads from final AIMessage instead), "
            "but recommended to include for audit/replay."
        ),
    )
    paradigm: str = Field(
        description=(
            "Experiment paradigm (e.g. 'fst', 'epm'). Redundant with "
            "experiment-context.json so handoff is self-contained for replay."
        ),
    )
    ev19_template: str | None = Field(
        default=None,
        description="EV19 template ID, or None for paradigms not mapped to EV19.",
    )
    analysis_config_id: str = Field(
        description=(
            "16-char hex hash of (catalog_default + parameter_overrides). "
            "Populated by code-executor from experiment-context.json."
        ),
    )
    task_context: TaskContext | None = Field(
        default=None,
        description="任务状态包（seal 工具确定性组装，向后兼容：旧 handoff 为 None）。",
    )
    sealed_by: Literal["model", "framework_rebuild"] = Field(
        default="model",
        description=(
            "Handoff 来源标记。model = subagent 自行调 seal 工具封存（正常路径）；"
            "framework_rebuild = harness 在 auto-seal 兜底中从 m_*.json 机械重建。"
            "用于触发率可观测 + 回归探针（Spec A V1/V2 验收项）。"
        ),
    )


class FailedChart(BaseModel):
    """One failed chart entry."""

    chart_id: str = Field(description="Chart ID from catalog, e.g. 'trajectory_heatmap'.")
    reason: str = Field(description="Free-text failure reason.")


class ChartMakerHandoff(BaseModel):
    """Handoff JSON produced by chart-maker subagent.

    Fields align with the schema currently declared in chart-maker subagent prompt
    (subagents/builtins/chart_maker.py <handoff_schema> section).
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["completed", "partial", "failed"] = "completed"
    paradigm: str = Field(description="Experiment paradigm.")
    chart_files: list[str] = Field(
        default_factory=list,
        description="Virtual paths under /mnt/user-data/outputs/.",
    )
    failed_charts: list[FailedChart] = Field(default_factory=list)
    summary: str = Field(description="One-liner describing generated charts.")
    gate_signals: GateSignals | None = None
    analysis_config_id: str = Field(
        default="PENDING",
        description="Inherited from CodeExecutorHandoff via seal tool.",
    )
    task_context: TaskContext | None = Field(
        default=None,
        description="任务状态包（seal 工具确定性组装，向后兼容：旧 handoff 为 None）。",
    )

    @field_validator("chart_files")
    @classmethod
    def _validate_chart_paths(cls, v: list[str]) -> list[str]:
        if not v:
            return v
        prefix = "/mnt/user-data/outputs/"
        offenders = [p for p in v if not p.startswith(prefix)]
        if offenders:
            raise ValueError(
                f"chart_files must be under {prefix!r}, "
                f"got: {offenders}. Outputs must be in outputs/, not workspace/."
            )
        return v


class OutlierFinding(BaseModel):
    """One flagged outlier subject with counterfactual support.

    Used by data-analyst to surface per-subject diagnostics: which subject
    deviates on which metric, by how much, and what the group statistics
    look like with that subject excluded.
    """

    model_config = ConfigDict(extra="allow")

    subject: str = Field(description="Subject identifier, e.g. 'Subject 3'.")
    metric: str = Field(description="Metric on which this subject is an outlier.")
    value: float = Field(description="Raw per-subject value on that metric.")
    deviation: str = Field(
        description=(
            "Qualitative description of deviation, e.g. '2x group median', "
            "'CV=35%', '> 1.5 SD above mean'."
        ),
    )
    counterfactual: str | None = Field(
        default=None,
        description=(
            "Leave-one-out group stats if this subject is excluded, e.g. "
            "'treatment mean_nnd drops 48.2 → 37.2 mm if Subject 3 excluded'."
        ),
    )


class DataAnalystHandoff(BaseModel):
    """Handoff JSON produced by the data-analyst subagent.

    Structured return type so downstream consumers (report-writer, lead agent
    rendering) can act on the analyst's findings without re-parsing natural
    language. This is the single source of truth for data-analyst output —
    the subagent writes nothing else to disk.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["completed", "partial", "failed"]
    key_findings: list[str] = Field(
        default_factory=list,
        description="1-5 bullet findings surfaced to the user.",
    )
    outlier_findings: list[OutlierFinding] = Field(
        default_factory=list,
        description=(
            "Per-subject outlier diagnostics with leave-one-out counterfactual "
            "context. Empty list when no outlier is flagged."
        ),
    )
    excluded_metrics: list[str] = Field(
        default_factory=list,
        description="Metrics skipped due to applicability or data-quality concerns.",
    )
    method_warnings: list[str] = Field(
        default_factory=list,
        description="Statistical-method concerns raised during quality review.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Suggested next steps or follow-up analyses.",
    )
    errors: list[str] = Field(default_factory=list)
    gate_signals: GateSignals | None = Field(default=None)
    analysis_config_id: str = Field(
        default="PENDING",
        description="Inherited from CodeExecutorHandoff via seal tool.",
    )
    task_context: TaskContext | None = Field(
        default=None,
        description="任务状态包（seal 工具确定性组装，向后兼容：旧 handoff 为 None）。",
    )
    quality_warnings: list[DataQualityWarning] = Field(
        default_factory=list,
        description=(
            "从 handoff_code_executor.json 透传的 data_quality_warnings, "
            "保留完整结构供下游(report-writer / lead UI / 假设面板)按 code 分组渲染。"
        ),
    )
    parameter_audit_findings: list[ParameterAuditFinding] = Field(
        default_factory=list,
        description=(
            "Sprint 3 新增。data-analyst 比对 MetricStat.parameters_used 与 "
            "handoff_code_executor 中的 per_subject 数据分布后产出的不匹配清单。"
            "下游 report-writer 会读此字段写入'数据质量与局限'段；前端 "
            "QualityWarningBanner 不读这个字段（它只显示 quality_warnings）。"
        ),
    )


class ReportWriterHandoff(BaseModel):
    """Handoff JSON produced by the report-writer subagent."""

    model_config = ConfigDict(extra="allow")

    status: Literal["completed", "partial", "failed"]
    report_path: str
    sections_written: list[str] = Field(
        default_factory=list,
        description="E.g. ['Results', 'Discussion'].",
    )
    errors: list[str] = Field(default_factory=list)
    gate_signals: GateSignals | None = Field(default=None)
    analysis_config_id: str = Field(
        default="PENDING",
        description="Inherited from CodeExecutorHandoff via seal tool.",
    )
    task_context: TaskContext | None = Field(
        default=None,
        description="任务状态包（seal 工具确定性组装，向后兼容：旧 handoff 为 None）。",
    )


__all__ = [
    "ChartMakerHandoff",
    "CodeExecutorHandoff",
    "CodeExecutorInputs",
    "DataAnalystHandoff",
    "DataQualityWarning",
    "FailedChart",
    "GateSignals",
    "MetricStat",
    "OutlierFinding",
    "ParameterAuditFinding",
    "ReportWriterHandoff",
    "TaskContext",
    "WARNING_CODE_PREFIXES",
]
