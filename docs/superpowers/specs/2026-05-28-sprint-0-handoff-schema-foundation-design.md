# Sprint 0 实施 spec — handoff 全面 schema 化 + seal_*_handoff first-party tool 集

**关联**：[2026-05-28 SOTA agent 路线图 v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) Sprint 0
**估期**：2 周（含 grill 锁定的 atomic write + 4 个 seal tool + 三档 strict mode）
**前置**：无；本 sprint 是后续 9 个 sprint 的下游承接器
**执行者**：交给独立 agent 执行（请独立阅读本 spec + roadmap + 关联代码）

---

## 1. 背景与目标

Sprint 0 是 v2 整条路线的下游承接器。**不做对**这一步，后续 sprint 全部受波及：
- 参数 lineage（Sprint 2b）没地方挂 `parameters_used`
- analysis_config_id（Sprint 4.5）没地方写
- data_quality_warnings 升级（Sprint 1）没地方放新字段
- lineage 封印（Sprint 5.5）没东西可 hash
- 跨会话 memory（Sprint 6）没确定性 fact 注入点

### 三个互锁的核心改动

1. **Pydantic schema 全套**：4 个 handoff 都有 Pydantic 类，含 grill 锁定的所有新字段
2. **4 个 seal_*_handoff first-party tool**：subagent 改用 tool 调用（结构化参数）seal handoff，不再用 write_file 写 JSON 字符串 — LLM 漂移面归零
3. **三档 strict mode**：OFF/WARN/FAIL_CLOSED，默认 WARN（1 周观察期），灰度切 FAIL_CLOSED

### 核心架构决策（grill 锁定）

| 决定 | 出处 |
|---|---|
| ChartMakerHandoff 字段以 chart-maker prompt 现有契约为准（paradigm/chart_files/failed_charts/summary）| Q1 = A |
| atomic write（tmp + os.rename）+ 4 个 seal tool 全走（LLM 不碰 JSON）| Q2 = A+E |
| 不在 Sprint 0 加 ExperimentSummary 类（Sprint 6 走 deerflow facts 通道）| Q9 = B |
| analysis_config_id 字段在 handoff 顶层（Sprint 4.5 自动填）| Q6 = B |
| DataQualityWarning 加 code/evidence/blocks_downstream 三字段（Sprint 1 用，Sprint 0 加结构）| Q3 |

---

## 2. 文件改动清单

### 2.0 字段来源责任表（review 反馈后新增）

`analysis_config_id` 等"非 subagent 输入"字段需要明确**谁负责填**，避免 Pydantic 必填字段在 LLM 漏传时炸。

| 字段 | 出现位置 | 来源 | LLM/subagent 是否传 | Pydantic 必填 |
|---|---|---|---|---|
| `analysis_config_id` | 所有 4 个 Handoff 类顶层 | `_seal_handoff` helper 内 `payload.setdefault(...)` 自动从 experiment-context.json 读，没值则用 `"PENDING_SPRINT_4.5"` 占位 | **不传**（helper 注入） | 必填（helper 保证有值，Pydantic constructor 不炸） |
| `paradigm` | CodeExecutorHandoff / ChartMakerHandoff | subagent 通过 seal tool 参数传入 | **必传**（subagent prompt 明示） | 必填（漏传 → Pydantic 报错 → LLM 看到 ValueError 重试） |
| `ev19_template` | CodeExecutorHandoff | subagent 传入；某些范式可为 None | 传或不传都可 | `Optional[str] = None` |
| 其他业务字段 | 各 Handoff 类 | subagent 业务输出 | 必传 | 各自定义 |

**helper 注入顺序（关键）**：

```python
def _seal_handoff(model_cls, filename, payload, runtime):
    workspace = _resolve_workspace(runtime)

    # 1. 先注入 helper 负责的字段（在 Pydantic validate 之前!）
    payload.setdefault("analysis_config_id", _read_analysis_config_id(workspace))

    # 2. 再走 Pydantic constructor，此时 payload 已经有 analysis_config_id
    handoff = model_cls(**payload)  # ← 必填字段已被 helper 填入,不会炸
    ...
```

**Sprint 4.5 落地后**：helper 内 `_read_analysis_config_id()` 自动返回实际 hash，不再需要 `"PENDING_SPRINT_4.5"` 占位，向前兼容（旧 handoff 文件含占位 id 仍能读）。

### 2.1 改动 `subagents/handoff_schemas.py`

**位置**：`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`

#### a) DataQualityWarning 加 3 字段

```python
class DataQualityWarning(BaseModel):
    severity: Literal["critical", "warning", "info"]
    metric: str
    message: str

    # === Sprint 0 新增 ===
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

    @field_validator("code")
    @classmethod
    def _validate_code_namespace(cls, v: str) -> str:
        # Sprint 0/1 过渡期白名单含 LEGACY；Sprint 1 完工时从白名单删除 LEGACY
        # （seal_code_executor_handoff 的过渡兜底同步删，见 §2.3）
        allowed = {"SAMPLE", "MOTOR", "SIGNAL", "METHOD", "LEGACY"}
        head = v.split(".", 1)[0] if "." in v else ""
        if head not in allowed:
            raise ValueError(
                f"warning code must start with one of {allowed}.*, got {v!r}. "
                "See Sprint 1 for code taxonomy."
            )
        return v
```

#### b) MetricStat 加 `parameters_used` 字段

```python
class MetricStat(BaseModel):
    model_config = ConfigDict(extra="allow")
    mean: float | None = None
    std: float | None = None
    n: int | None = None
    applicable: bool = Field(default=True, ...)
    reason: str | None = Field(default=None, ...)

    # === Sprint 0 新增 ===
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
```

#### c) CodeExecutorHandoff 加 paradigm/ev19_template/analysis_config_id

```python
class CodeExecutorHandoff(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: Literal["completed", "partial", "failed"]
    summary: str
    inputs: CodeExecutorInputs | None = None
    output_files: dict[str, Any] = Field(default_factory=dict)
    metrics_summary: dict[str, dict[str, MetricStat]] = Field(default_factory=dict)
    per_subject: dict[str, dict[str, Any]] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)
    assessment: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    data_quality_warnings: list[DataQualityWarning] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    gate_signals: GateSignals | None = None

    # === Sprint 0 新增 ===
    paradigm: str = Field(
        description=(
            "Experiment paradigm (e.g. 'fst', 'epm'). Redundant with "
            "experiment-context.json so handoff is self-contained for replay. "
            "**Subagent MUST pass this via seal tool args** — LLM 知道本次范式。"
        ),
    )
    ev19_template: str | None = Field(
        default=None,
        description="EV19 template ID, or None for paradigms not mapped to EV19.",
    )
    analysis_config_id: str = Field(
        description=(
            "16-char hex hash of (catalog_default + parameter_overrides). "
            "**Subagent does NOT pass this** — _seal_handoff helper auto-injects "
            "from experiment-context.json. Sprint 0 阶段占位 'PENDING_SPRINT_4.5'，"
            "Sprint 4.5 实施后自动正常填入。"
        ),
    )
```

#### d) 新增 ChartMakerHandoff 类（grill Q1 锁定字段）

```python
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

    # === Sprint 0 新增 ===
    analysis_config_id: str = Field(
        description=(
            "Inherited from CodeExecutorHandoff. "
            "**Subagent does NOT pass** — helper auto-injects from "
            "experiment-context.json."
        ),
    )

    @field_validator("chart_files")
    @classmethod
    def _validate_chart_paths(cls, v: list[str]) -> list[str]:
        # chart_files must be under /mnt/user-data/outputs/ (per chart-maker prompt L43)
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
```

#### e) DataAnalystHandoff / ReportWriterHandoff 加 analysis_config_id

**注意**：这两个类**现有字段已就位**（见 handoff_schemas.py 第 208-244 / 247-259 行）。Sprint 0 只新增 `analysis_config_id` 一个字段，不动其他字段：

```python
class DataAnalystHandoff(BaseModel):
    # ... existing fields (status / key_findings / outlier_findings /
    # excluded_metrics / method_warnings / recommendations / errors /
    # gate_signals) 保持不变 ...

    analysis_config_id: str = Field(  # Sprint 0 新增
        description=(
            "Inherited from CodeExecutorHandoff. "
            "**Subagent does NOT pass** — helper auto-injects."
        ),
    )


class ReportWriterHandoff(BaseModel):
    # ... existing fields (status / report_path / sections_written /
    # errors / gate_signals) 保持不变 ...

    analysis_config_id: str = Field(  # Sprint 0 新增
        description=(
            "Inherited from CodeExecutorHandoff. "
            "**Subagent does NOT pass** — helper auto-injects."
        ),
    )
```

**向前兼容性**：两个类都已经有 `model_config = ConfigDict(extra="allow")`（见 handoff_schemas.py:217, 250），所以**多字段不会炸**——但 `extra="allow"` **不解决缺字段问题**。旧的 handoff_data_analyst.json / handoff_report_writer.json 没有 `analysis_config_id` 字段 → Pydantic constructor 会触发 schema violation → 三档 strict mode 在 WARN 期间只 log + 继续运行（不影响旧 thread），FAIL_CLOSED 期间确实会 raise（但 review 已确认旧会话不兼容是 v0.1 频繁变更期可接受的代价）。Sprint 0 启动后所有新 handoff 都带这字段。

**实际作用范围**：DataAnalystHandoff / ReportWriterHandoff 的 schema 违规**不会**在 read_handoff helper 里触发（该函数只读 code_executor handoff，见 §2.6）；只在 seal tool 写入时的 Pydantic constructor 校验。所以这两个类加 `analysis_config_id` 必填字段的运行时影响范围是「seal 时」，不是「read 时」。

#### f) `__all__` 更新

```python
__all__ = [
    "ChartMakerHandoff",           # 新增
    "CodeExecutorHandoff",
    "CodeExecutorInputs",
    "DataAnalystHandoff",
    "DataQualityWarning",
    "FailedChart",                 # 新增
    "GateSignals",
    "MetricStat",
    "OutlierFinding",
    "ReportWriterHandoff",
]
```

### 2.2 改动 `subagents/handoff_registry.py`

文件已有 `"chart_maker": "handoff_chart_maker.json"` 注册，无需改动。**确认即可**。

### 2.3 新建 `tools/builtins/seal_handoff_tools.py`

**位置**：`packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py`

```python
"""4 个 first-party tool — subagent 调用本 tool 结构化 seal handoff 到 workspace。

设计原则（grill 锁定 Sprint 0）：
1. LLM 只填 tool 参数（LangChain tool_call schema 自动校验类型/必填）
2. tool 内部 Pydantic 校验 + atomic write + .lineage/manifest.json 记录
3. 4 个 tool 共享 _seal_handoff helper,避免重复
4. 调用方:
    - code-executor → seal_code_executor_handoff
    - data-analyst → seal_data_analyst_handoff
    - chart-maker → seal_chart_maker_handoff
    - report-writer → seal_report_writer_handoff
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT
from pydantic import ValidationError

from deerflow.agents.thread_state import ThreadState
from deerflow.subagents.handoff_schemas import (
    ChartMakerHandoff,
    CodeExecutorHandoff,
    DataAnalystHandoff,
    ReportWriterHandoff,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 内部 helper
# ============================================================================

def _resolve_workspace(runtime: ToolRuntime[ContextT, ThreadState]) -> Path:
    """从 runtime state 取 host-side workspace 路径。"""
    state = runtime.state
    if not isinstance(state, dict):
        raise RuntimeError("seal_*_handoff: runtime.state is not a dict")
    thread_data = state.get("thread_data")
    if not isinstance(thread_data, dict):
        raise RuntimeError("seal_*_handoff: thread_data missing from state")
    workspace_path = thread_data.get("workspace_path")
    if not workspace_path:
        raise RuntimeError("seal_*_handoff: workspace_path missing")
    return Path(workspace_path)


def _read_analysis_config_id(workspace: Path) -> str:
    """从 experiment-context.json 读 analysis_config_id (Sprint 4.5 填)。

    Sprint 0 阶段:experiment-context.json 可能还没有此字段,返回 "PENDING_SPRINT_4.5"
    占位,Sprint 4.5 实施后会自动正常填入。
    """
    ctx_path = workspace / "experiment-context.json"
    if not ctx_path.exists():
        return "PENDING_SPRINT_4.5"
    try:
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
        return ctx.get("analysis_config_id", "PENDING_SPRINT_4.5")
    except Exception as e:
        logger.warning("read experiment-context.json failed: %s", e)
        return "PENDING_SPRINT_4.5"


def _update_manifest(workspace: Path, handoff_filename: str, sha256: str, analysis_config_id: str) -> None:
    """写 .lineage/manifest.json。

    Sprint 5.5 会进一步用本 manifest 做下游 hash 校验;Sprint 0 只负责写。
    """
    lineage_dir = workspace / ".lineage"
    lineage_dir.mkdir(exist_ok=True)
    manifest_path = lineage_dir / "manifest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                manifest = {}
        except Exception:
            manifest = {}
    else:
        manifest = {}

    manifest[handoff_filename] = {
        "sha256": sha256,
        "analysis_config_id": analysis_config_id,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # atomic write for manifest itself
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(manifest_path)


def _seal_handoff(
    model_cls: type,
    filename: str,
    payload: dict[str, Any],
    runtime: ToolRuntime[ContextT, ThreadState],
) -> str:
    """共享 helper:Pydantic 校验 → atomic write → 写 manifest → 返回 OK。

    所有失败路径都返回 ValueError(LangChain 会自动转 error ToolMessage 给 LLM)。
    """
    workspace = _resolve_workspace(runtime)

    # 1. 注入 analysis_config_id (subagent 不用手动传)
    payload.setdefault("analysis_config_id", _read_analysis_config_id(workspace))

    # 2. Pydantic 校验
    try:
        handoff = model_cls(**payload)
    except ValidationError as e:
        raise ValueError(
            f"seal_{filename}: schema validation failed: {e}. "
            f"Check field names/types against {model_cls.__name__} schema."
        ) from e

    # 3. Atomic write (tmp + rename)
    final_path = workspace / filename
    tmp_path = workspace / f"{filename}.tmp"
    json_bytes = handoff.model_dump_json(indent=2, exclude_none=False).encode("utf-8")
    tmp_path.write_bytes(json_bytes)
    os.rename(tmp_path, final_path)  # POSIX atomic

    # 4. 写 manifest
    sha256 = hashlib.sha256(json_bytes).hexdigest()
    _update_manifest(workspace, filename, sha256, payload["analysis_config_id"])

    return f"OK: sealed {filename} (sha256={sha256[:12]}...)"


# ============================================================================
# 4 个 first-party tool
# ============================================================================

@tool("seal_code_executor_handoff", parse_docstring=True)
def seal_code_executor_handoff(
    status: str,
    summary: str,
    paradigm: str,
    metrics_summary: dict[str, dict[str, dict[str, Any]]] | None = None,
    per_subject: dict[str, dict[str, Any]] | None = None,
    statistics: dict[str, Any] | None = None,
    output_files: dict[str, Any] | None = None,
    data_quality_warnings: list[dict[str, Any]] | None = None,
    errors: list[str] | None = None,
    confidence: float | None = None,
    ev19_template: str | None = None,
    inputs: dict[str, Any] | None = None,
    gate_signals: dict[str, Any] | None = None,
    runtime: ToolRuntime[ContextT, ThreadState] = None,
) -> str:
    """Code-executor 完成指标计算后,封存 handoff_code_executor.json。

    严禁直接用 write_file 写 handoff_code_executor.json,必须走本 tool。

    Args:
        status: 执行状态: "completed" / "partial" / "failed"
        summary: 一句话总结
        paradigm: 范式名,如 "fst" / "epm"
        metrics_summary: 嵌套 dict: group -> metric -> {mean, std, n, parameters_used, ...}
        per_subject: 每个 subject 的原始数据: {subject_name: {metric: value}}
        statistics: 组间统计检验结果
        output_files: 产物文件路径表
        data_quality_warnings: 警告列表,每条含 severity/code/metric/message/evidence/blocks_downstream
        errors: 错误信息列表
        confidence: 整体置信度 [0,1]
        ev19_template: EV19 模板 ID,如 'fst-modified'
        inputs: 输入信息: {raw_files: [...], groups: {...}}
        gate_signals: 决策信号
    """
    # === Sprint 0/1 过渡兜底（Sprint 1 完工时务必删除整块）===
    # 当前 dispatcher.py 输出的 warning 只有 {severity, metric, message},
    # 缺 Sprint 0 新增的 code/evidence/blocks_downstream 三字段。Sprint 1 在
    # dispatcher 端正式填这三个字段;在此之前 seal tool 自动补占位,避免
    # Pydantic constructor 因必填字段缺失而炸,导致 code-executor 死循环。
    # Sprint 1 完工时:
    #   1. 删除本 block（grep "LEGACY.UNCATEGORIZED" 一键定位）
    #   2. 从 DataQualityWarning._validate_code_namespace 白名单删除 "LEGACY"
    if data_quality_warnings:
        for w in data_quality_warnings:
            w.setdefault("code", "LEGACY.UNCATEGORIZED")
            w.setdefault("evidence", {})
            # 老语义:severity=critical 时阻断下游(对齐 GateEnforcementMiddleware
            # 现有行为);其他默认 False
            w.setdefault("blocks_downstream", w.get("severity") == "critical")
    # === 过渡兜底结束 ===

    payload = {
        "status": status,
        "summary": summary,
        "paradigm": paradigm,
        "metrics_summary": metrics_summary or {},
        "per_subject": per_subject or {},
        "statistics": statistics or {},
        "output_files": output_files or {},
        "data_quality_warnings": data_quality_warnings or [],
        "errors": errors or [],
        "confidence": confidence,
        "ev19_template": ev19_template,
        "inputs": inputs,
        "gate_signals": gate_signals,
    }
    return _seal_handoff(CodeExecutorHandoff, "handoff_code_executor.json", payload, runtime)


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
    runtime: ToolRuntime[ContextT, ThreadState] = None,
) -> str:
    """Data-analyst 完成分析后,封存 handoff_data_analyst.json。

    严禁直接用 write_file 写 handoff_data_analyst.json,必须走本 tool。

    Args:
        status: "completed" / "failed"
        key_findings: 1-5 条核心发现
        outlier_findings: 异常 subject 列表,每条含 subject/metric/value/deviation/counterfactual
        excluded_metrics: 因质量问题被排除的指标
        method_warnings: 统计方法警告
        recommendations: 建议后续操作
        errors: 错误信息
        gate_signals: 决策信号
    """
    payload = {
        "status": status,
        "key_findings": key_findings or [],
        "outlier_findings": outlier_findings or [],
        "excluded_metrics": excluded_metrics or [],
        "method_warnings": method_warnings or [],
        "recommendations": recommendations or [],
        "errors": errors or [],
        "gate_signals": gate_signals,
    }
    return _seal_handoff(DataAnalystHandoff, "handoff_data_analyst.json", payload, runtime)


@tool("seal_chart_maker_handoff", parse_docstring=True)
def seal_chart_maker_handoff(
    paradigm: str,
    summary: str,
    chart_files: list[str] | None = None,
    failed_charts: list[dict[str, str]] | None = None,
    status: str = "completed",
    gate_signals: dict[str, Any] | None = None,
    runtime: ToolRuntime[ContextT, ThreadState] = None,
) -> str:
    """Chart-maker 完成绘图后,封存 handoff_chart_maker.json。

    严禁直接用 write_file 写 handoff_chart_maker.json,必须走本 tool。

    Args:
        paradigm: 范式名
        summary: 一句话描述生成的图表
        chart_files: 成功的图表 png 路径(必须在 /mnt/user-data/outputs/ 下)
        failed_charts: 失败列表,每条 {chart_id, reason}
        status: "completed" / "partial" / "failed"(全部失败时为 failed)
        gate_signals: 决策信号
    """
    payload = {
        "status": status,
        "paradigm": paradigm,
        "summary": summary,
        "chart_files": chart_files or [],
        "failed_charts": failed_charts or [],
        "gate_signals": gate_signals,
    }
    return _seal_handoff(ChartMakerHandoff, "handoff_chart_maker.json", payload, runtime)


@tool("seal_report_writer_handoff", parse_docstring=True)
def seal_report_writer_handoff(
    status: str,
    report_path: str,
    sections_written: list[str] | None = None,
    errors: list[str] | None = None,
    gate_signals: dict[str, Any] | None = None,
    runtime: ToolRuntime[ContextT, ThreadState] = None,
) -> str:
    """Report-writer 完成写报告后,封存 handoff_report_writer.json。

    严禁直接用 write_file 写 handoff_report_writer.json,必须走本 tool。

    Args:
        status: "completed" / "failed"
        report_path: 报告 md 文件路径
        sections_written: 已写的段落,如 ["Results", "Discussion"]
        errors: 错误信息
        gate_signals: 决策信号
    """
    payload = {
        "status": status,
        "report_path": report_path,
        "sections_written": sections_written or [],
        "errors": errors or [],
        "gate_signals": gate_signals,
    }
    return _seal_handoff(ReportWriterHandoff, "handoff_report_writer.json", payload, runtime)
```

### 2.4 注册 4 个 seal tool

**位置**：`packages/agent/backend/packages/harness/deerflow/tools/builtins/__init__.py`

把 4 个 tool 导出并加入 `BUILTIN_TOOLS`（注意：Noldus 受保护文件，不能整文件覆盖上游；做 surgical merge）：

```python
from .seal_handoff_tools import (
    seal_chart_maker_handoff,
    seal_code_executor_handoff,
    seal_data_analyst_handoff,
    seal_report_writer_handoff,
)

# 在 BUILTIN_TOOLS 字典里加 4 项
BUILTIN_TOOLS = {
    # ... existing ...
    "seal_code_executor_handoff": seal_code_executor_handoff,
    "seal_data_analyst_handoff": seal_data_analyst_handoff,
    "seal_chart_maker_handoff": seal_chart_maker_handoff,
    "seal_report_writer_handoff": seal_report_writer_handoff,
}

# 在 __all__ 加 4 项
```

### 2.5 改 4 个 subagent 的 SubagentConfig

#### a) code_executor.py

**位置**：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`

改动 3 处：

1. `tools=[...]` 加入 `"seal_code_executor_handoff"`
2. system_prompt 中所有"写 handoff_code_executor.json"步骤改为"调 seal_code_executor_handoff tool"
3. `<handoff_schema>` 段改为指向 `CodeExecutorHandoff` Pydantic 模型字段说明（含 paradigm/analysis_config_id 自动注入说明）

例如（具体段落需要 read_file 当前 code_executor.py 看上下文再改）：

```
12. **封存 handoff**: 调 seal_code_executor_handoff tool,传入 status/summary/paradigm/
    metrics_summary/per_subject/statistics/data_quality_warnings/output_files/...,
    工具会自动写入 /mnt/user-data/workspace/handoff_code_executor.json 并落 manifest hash。
    严禁直接 write_file 写 handoff_code_executor.json。
```

#### b) data_analyst.py / chart_maker.py / report_writer.py

同 a，分别加入对应 seal tool，prompt 工作流末尾改为"调 seal_X_handoff"，禁止 write_file 直接写 handoff。

### 2.6 改 `experiment_context.py` — 三档 strict mode

**位置**：`packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py`

**作用范围说明**（review 反馈后澄清）：

当前 `read_handoff()` 是**专门读 `handoff_code_executor.json`** 的（文件名写死在 line 74），被 `get_critical_warnings()` → `GateEnforcementMiddleware` 调用以做 Gate 2 决策。**它不是通用 handoff reader**。

所以 Sprint 0 的 strict mode 作用范围是：**lead 通过 GateEnforcementMiddleware 主动读 handoff_code_executor.json 时校验 CodeExecutorHandoff**。

**为什么其他 3 个 handoff 不在 Sprint 0 校验**：

- data-analyst / chart-maker / report-writer 的 handoff 文件**没有任何 host-side reader 主动 load**（lead 通过 LLM read_file 工具读，那是普通 sandbox tool，不经过 read_handoff helper）
- 下游 subagent 读上游 handoff 也是用 sandbox read_file（绕过本 helper）
- 这 3 个 handoff 的 schema 校验由 **Sprint 0 的 seal tool 在写入时**保证（Pydantic constructor 校验）—— **写时正确则永远正确**（atomic write + 后续 manifest hash）
- Sprint 5.5 引入 `LineageIntegrityGuardrailProvider` 在派遣 subagent 前验 hash，那是另一条独立路径，不用 read_handoff helper

**结论**：Sprint 0 阶段不需要把 read_handoff 改成通用 reader；strict mode 只覆盖 code_executor handoff 即可。如果未来需要在 host-side 主动校验其他 handoff，再通用化（YAGNI）。

```python
import enum
from pathlib import Path


class HandoffStrictMode(str, enum.Enum):
    OFF = "off"
    WARN = "warn"
    FAIL_CLOSED = "fail_closed"


_EMERGENCY_DOWNGRADE_FILE = "/tmp/disable_strict_handoff"


def _get_strict_mode() -> HandoffStrictMode:
    """读 config + 紧急降级文件。"""
    # 紧急降级:文件存在时强制 WARN
    if Path(_EMERGENCY_DOWNGRADE_FILE).exists():
        logger.warning("emergency downgrade file present, forcing WARN mode")
        return HandoffStrictMode.WARN
    # 从 config 读 (默认 WARN)
    from deerflow.config import get_app_config
    cfg = get_app_config()
    mode_str = getattr(cfg, "handoff_strict_mode", "warn")
    try:
        return HandoffStrictMode(mode_str)
    except ValueError:
        logger.warning("invalid handoff_strict_mode %r, falling back to WARN", mode_str)
        return HandoffStrictMode.WARN


class HandoffSchemaError(Exception):
    """Raised in FAIL_CLOSED mode when handoff schema validation fails."""


# 改 read_handoff 内部 validation 段(仅对 code_executor handoff 生效):
def read_handoff(workspace_dir, thread_data=None):
    # ... existing read + reverse-mask logic ...

    if not isinstance(data, dict):
        return None

    mode = _get_strict_mode()
    if mode == HandoffStrictMode.OFF:
        # 旧行为:不做 validation
        return data

    # Validate against CodeExecutorHandoff (本函数专读 code_executor handoff)
    try:
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff
        CodeExecutorHandoff.model_validate(data)
    except Exception as e:
        if mode == HandoffStrictMode.FAIL_CLOSED:
            raise HandoffSchemaError(
                f"handoff_code_executor.json schema violation (FAIL_CLOSED): {e}"
            ) from e
        # WARN 模式
        logger.warning("handoff schema violation (WARN mode): %s", e)
        # TODO: counter (Prometheus 或 简单 file counter,Sprint 0 阶段先 log)
        violations = data.setdefault("_schema_violations", [])
        if isinstance(violations, list):
            violations.append(str(e))

    return data
```

### 2.7 加 config schema

**位置**：`packages/agent/backend/packages/harness/deerflow/config/app_config.py`（或对应文件 — 需要 read 实际位置）

在 AppConfig 加：

```python
handoff_strict_mode: Literal["off", "warn", "fail_closed"] = Field(
    default="warn",
    description=(
        "Handoff schema validation strictness. "
        "'warn' (default) logs violations but returns dict; "
        "'fail_closed' raises HandoffSchemaError; "
        "'off' restores legacy behavior. "
        "Override at runtime by creating /tmp/disable_strict_handoff (forces 'warn')."
    ),
)
```

`config.example.yaml` 也加：

```yaml
# Sprint 0 新增:handoff schema 校验级别
handoff_strict_mode: warn  # off | warn | fail_closed
```

### 2.8 改 `script_invocation_only_provider.py` — 拦截 write_file 写 handoff

**位置**：`packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py`

read 当前实现，在 4 个 subagent 调 write_file 时检查 path 是否命中 `handoff_*.json`，若命中 → deny + 含明确指令：

```python
# 在 evaluate() 加分支
if tool_name == "write_file":
    path = tool_input.get("path", "")
    if "handoff_" in path and path.endswith(".json"):
        return GuardrailDecision(
            allow=False,
            reasons=[GuardrailReason(
                code="handoff.write_file_forbidden",
                message=(
                    f"严禁用 write_file 写 {path}。"
                    f"请改用对应的 first-party tool: "
                    f"seal_code_executor_handoff / seal_data_analyst_handoff / "
                    f"seal_chart_maker_handoff / seal_report_writer_handoff,"
                    f"按结构化参数调用。"
                ),
            )],
        )
```

### 2.9 ~~前端 bash 命令展示美化~~ ← 挪出 Sprint 0

**审查决定**（review 反馈）：这是纯前端 cosmetic 改动，与 handoff schema 核心架构无关，**不放在 Sprint 0**。

挪到独立小 spec 处理，建议路径：

- 选项 A：合并进 Sprint 1（与 data_quality_warnings 红字标注一起做，都是前端展示层）
- 选项 B：作为独立的 0.5 天 quick win，任意空档实施

实施时的位置参考（仅供后续 spec 使用）：

> 前端 SSE stream 渲染层：`packages/agent/frontend/`（具体路径需要 read frontend/）
>
> 渲染 tool_call 详情时，对 bash 命令字符串做 regex 替换：
> ```typescript
> // e.g. apps/agent/frontend/src/lib/format-bash.ts
> export function formatBashCommand(cmd: string): string {
>     return cmd.replace(
>         /python\s+-m\s+ethoinsight\.scripts\.(\w+)\.(\w+)([^\n]*)/g,
>         "ethoinsight: $1.$2$3"
>     );
> }
> ```
>
> 后端不动，纯前端展示层 cosmetic。

---

## 3. 测试要求

### 3.1 单元测试

新建 `packages/agent/backend/tests/test_seal_handoff_tools.py`：

| 测试 | 期望 |
|---|---|
| `test_seal_code_executor_happy_path` | 调 tool → handoff_code_executor.json 落盘 + manifest 含 sha256 |
| `test_seal_chart_maker_path_validation` | chart_files 包含 `/mnt/user-data/workspace/x.png` → ValueError |
| `test_seal_data_analyst_minimum_fields` | 只传 `status="completed"` → 通过（其他默认 empty list）|
| `test_seal_atomic_write_no_partial` | mock os.rename 抛错 → 落盘不出现 `.json`，只剩 `.json.tmp` |
| `test_manifest_includes_sha256_and_config_id` | manifest 文件含正确 sha256 和 analysis_config_id |
| `test_manifest_atomic_when_concurrent` | 两次 seal 调用并发 → manifest 不损坏 |

新建 `packages/agent/backend/tests/test_handoff_strict_mode.py`：

| 测试 | 期望 |
|---|---|
| `test_strict_mode_off_returns_dict_on_violation` | 写错 schema → 返回原 dict + _schema_violations |
| `test_strict_mode_warn_logs_violation` | WARN + caplog 含 WARNING |
| `test_strict_mode_fail_closed_raises` | FAIL_CLOSED + 写错 schema → raise HandoffSchemaError |
| `test_emergency_downgrade_file_forces_warn` | 创建 `/tmp/disable_strict_handoff` → 即使 config=FAIL_CLOSED 也走 WARN |

新建 `packages/agent/backend/tests/test_data_quality_warning_schema.py`：

| 测试 | 期望 |
|---|---|
| `test_warning_code_taxonomy_enforced` | code="FOO.BAR" → ValueError（首段必须 SAMPLE/MOTOR/SIGNAL/METHOD/LEGACY）|
| `test_warning_code_legacy_accepted_during_transition` | code="LEGACY.UNCATEGORIZED" → 通过（Sprint 0/1 过渡期白名单）|
| `test_warning_evidence_dict` | evidence={"velocity": 5.2} → 通过；evidence=5.2 → ValueError |
| `test_warning_blocks_downstream_default_false` | 不传 → False |

新增 `packages/agent/backend/tests/test_seal_handoff_legacy_fallback.py`（Sprint 0/1 过渡兜底专测）：

| 测试 | 期望 |
|---|---|
| `test_seal_auto_injects_legacy_code_for_old_dispatcher_warnings` | 传 `{severity: "critical", metric: "all", message: "..."}` → seal 不抛错；落盘 handoff 内 warning 含 `code="LEGACY.UNCATEGORIZED"` + `evidence={}` + `blocks_downstream=true` |
| `test_seal_auto_blocks_downstream_for_critical_only_in_legacy` | 传 severity=warning 旧 dict → blocks_downstream 兜底为 False |
| `test_seal_preserves_explicit_code_when_present` | 传含 `code="SAMPLE.TOO_SMALL"` 的 dict → 不被 "LEGACY.UNCATEGORIZED" 覆盖（验 setdefault 语义）|

**注**：第二个测试文件在 Sprint 1 完工删 LEGACY 兜底时**整文件删除**（避免误以为兜底还在）。

新建 `packages/agent/backend/tests/test_chart_maker_handoff_schema.py`：

| 测试 | 期望 |
|---|---|
| `test_chart_maker_minimum_fields` | 只传 paradigm/summary → 通过 |
| `test_chart_maker_path_must_be_outputs` | chart_files=["/mnt/user-data/workspace/x.png"] → ValueError |
| `test_chart_maker_failed_chart_structure` | failed_charts=[{"chart_id": "x", "reason": "y"}] → 通过 |

### 3.2 集成测试

| 测试 | 期望 |
|---|---|
| `test_dogfood_fst_end_to_end_with_seal_tools` | 跑完整 FST 流程，4 个 handoff 全部走 seal tool，落盘 OK，manifest 4 条 |
| `test_write_file_handoff_path_denied` | code-executor 试图 write_file `handoff_code_executor.json` → guardrail deny |
| `test_emergency_downgrade_takes_effect_without_restart` | FAIL_CLOSED 模式 + 故障 handoff → 先报错，touch 降级文件 → 不再报错 |

### 3.3 测试基线

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight && .venv/bin/python -m pytest tests/ -q
cd /home/wangqiuyang/noldus-insight/packages/agent/backend && source .venv/bin/activate && make test
```

实施前 + 实施后必须全绿。当前基线：ethoinsight 439 passed / agent backend 3043 passed。

---

## 4. 实施顺序（建议给 agent 的 task 拆分）

| Task | 内容 | 估时 |
|---|---|---|
| T1 | handoff_schemas.py 加 3 字段（DataQualityWarning code/evidence/blocks_downstream）+ field_validator | 0.5 天 |
| T2 | handoff_schemas.py 加 MetricStat.parameters_used | 0.25 天 |
| T3 | handoff_schemas.py 加 ChartMakerHandoff + FailedChart 类 + path validator | 0.5 天 |
| T4 | handoff_schemas.py 给 4 个 handoff 加 analysis_config_id 字段 + CodeExecutorHandoff 加 paradigm/ev19_template | 0.5 天 |
| T5 | T1-T4 的单元测试（4 个测试文件）| 1 天 |
| T6 | 新建 seal_handoff_tools.py（4 个 tool + _seal_handoff helper）| 1.5 天 |
| T7 | seal_handoff_tools 单元测试（含 atomic write + manifest 测试）| 1 天 |
| T8 | 注册 4 个 seal tool 到 tools/builtins/\_\_init\_\_.py（surgical merge）| 0.25 天 |
| T9 | 改 4 个 subagent SubagentConfig（tools / prompt 中"写 handoff"步骤）| 1 天 |
| T10 | experiment_context.py 加三档 strict mode + HandoffSchemaError | 0.5 天 |
| T11 | strict_mode 单元测试 | 0.5 天 |
| T12 | AppConfig 加 handoff_strict_mode 字段 + config.example.yaml 加示例 | 0.25 天 |
| T13 | script_invocation_only_provider.py 拦截 write_file handoff 路径 + 单元测试 | 0.5 天 |
| T14 | 集成测试：dogfood FST 走通新 seal 通道 | 1 天 |
| T15 | 全量测试 + 修复退化（缓冲）| 1 天 |
| **合计** | | **9.75 天 ≈ 2 周（前端 bash 美化已挪出，独立做）** |

**注**：原 T14 前端 bash 命令美化已挪出 Sprint 0（见 §2.9 spec 末尾，作为独立小改动，0.5 天）。Sprint 0 净工期 9.75 天。

---

## 5. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 改 subagent prompt 后 dogfood 退化 | T9 后立刻跑 dogfood FST + EPM，发现退化立刻修；保留 git revert 路径 |
| LLM 调 seal tool 时漏字段 | LangChain tool_call schema 自动校验（参数缺失立即报错给 LLM），LLM 一次重试基本能修；LoopDetectionMiddleware 兜底 |
| analysis_config_id 字段 Sprint 0 阶段没值 | 占位 `"PENDING_SPRINT_4.5"`，Sprint 4.5 落地后自动正常 |
| Pydantic `extra="allow"` 配 ChartMakerHandoff 会接收新字段 | 故意保留 — chart-maker 上游迭代可能加字段；只严格校验已声明的关键字段 |
| WARN 模式下没有 counter 难统计 violation | Sprint 0 先 log；Sprint 1 实施时把 counter 加到 Prometheus（如果生产已部署）|
| 紧急降级文件 `/tmp/disable_strict_handoff` 在 docker 容器内不持久 | feature 而非 bug — 降级仅当次生效，重启容器自动恢复 strict |
| Noldus 受保护文件（tools/builtins/__init__.py 等）surgical merge 错误 | 改前 grep 确认 Noldus 定制段；改后跑全量测试 |

---

## 6. 验收 checklist

实施完成时，确认下列全部通过：

- [ ] 4 个 handoff schema 在 handoff_schemas.py 中（含 ChartMakerHandoff + FailedChart）
- [ ] DataQualityWarning 有 code/evidence/blocks_downstream 三新字段，code 走 4 一级分类校验
- [ ] MetricStat 有 parameters_used 字段（默认 empty dict）
- [ ] CodeExecutorHandoff 有 paradigm/ev19_template/analysis_config_id
- [ ] 其他 3 个 handoff 有 analysis_config_id
- [ ] seal_handoff_tools.py 含 4 个 tool + _seal_handoff helper
- [ ] 4 个 seal tool 注册进 BUILTIN_TOOLS（surgical merge __init__.py 保留 Noldus 定制）
- [ ] 4 个 subagent SubagentConfig 含对应 seal tool，prompt 步骤改为"调 seal_X_handoff"
- [ ] experiment_context.py 三档 strict mode + HandoffSchemaError 实现
- [ ] AppConfig + config.example.yaml 加 handoff_strict_mode
- [ ] script_invocation_only_provider.py 拦截 write_file 写 handoff_*.json
- [ ] 所有新增单元测试通过（约 20+ test cases）
- [ ] **Sprint 0/1 过渡兜底就位**：seal_code_executor_handoff 含 LEGACY 自动兜底 block + DataQualityWarning 白名单含 "LEGACY"；过渡测试 `test_seal_handoff_legacy_fallback.py` 存在并通过
- [ ] **Sprint 1 完工提醒已写入** §7.1，清理步骤清晰可执行
- [ ] 全量测试通过（ethoinsight ≥439 + agent backend ≥3043 + 新增 ~20）
- [ ] dogfood FST + EPM 跑通，4 个 handoff 都走 seal 通道，manifest 落盘
- [ ] FAIL_CLOSED 模式下故意写错 handoff 字段名 → 流程被拦截
- [ ] touch /tmp/disable_strict_handoff → FAIL_CLOSED 即时降级 WARN

---

## 7. 不在 Sprint 0 范围

明确**不做**的事，避免 scope creep：

- ❌ data_quality_warnings 的 code/evidence 实际填写（Sprint 1 做）
- ❌ MetricStat.parameters_used 实际写入（Sprint 2b 做；Sprint 0 只加字段）
- ❌ analysis_config_id 的实际计算（Sprint 4.5 做；Sprint 0 占位 "PENDING_SPRINT_4.5"）
- ❌ DataQualityGuardrailProvider（Sprint 5 做）
- ❌ LineageIntegrityGuardrailProvider（Sprint 5.5 做；Sprint 0 只写 manifest，不验 hash）
- ❌ ExperimentSummary 新顶层结构（grill Q9 砍掉；Sprint 6 走 facts 通道）
- ❌ AssumptionPanelGateProvider（grill Q10 砍掉）
- ❌ Manifest 的下游 hash 验证（Sprint 5.5 做）

## 7.1 Sprint 1 完工时务必清理（过渡兜底）

**Sprint 0/1 过渡期为了避免上线即挂，spec 内有两处兜底机制。Sprint 1 dispatcher.py 正式填充 code/evidence/blocks_downstream 三字段后，必须把这两处兜底删干净，否则永久兜底掩盖 dispatcher 端 bug**：

1. **删 seal_handoff_tools.py 内的 LEGACY 兜底 block**
   - 文件：`packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py`
   - 函数：`seal_code_executor_handoff`
   - 定位：grep `LEGACY.UNCATEGORIZED` 一键找到
   - 删除注释块 `# === Sprint 0/1 过渡兜底（Sprint 1 完工时务必删除整块）===` 到 `# === 过渡兜底结束 ===` 之间所有代码

2. **删 DataQualityWarning 白名单的 LEGACY**
   - 文件：`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`
   - 函数：`DataQualityWarning._validate_code_namespace`
   - 改 `allowed = {"SAMPLE", "MOTOR", "SIGNAL", "METHOD", "LEGACY"}` → `allowed = {"SAMPLE", "MOTOR", "SIGNAL", "METHOD"}`
   - 删 `LEGACY` 相关注释

3. **删过渡测试文件**
   - 删 `packages/agent/backend/tests/test_seal_handoff_legacy_fallback.py` 整文件
   - 删 `test_data_quality_warning_schema.py` 的 `test_warning_code_legacy_accepted_during_transition` 测试函数

4. **加单元测试验证 dispatcher 真填了 code**
   - Sprint 1 的 dispatcher 单元测试覆盖：跑各范式 critical case → handoff 内 warning 含正确 code（不是 LEGACY.UNCATEGORIZED）

**Sprint 1 的 spec 务必在验收 checklist 加：**

> - [ ] `git grep LEGACY.UNCATEGORIZED packages/agent/backend/` 返回空（兜底已清理）

---

## 8. 参考

- [SOTA agent 路线图 v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) — Sprint 0 章节
- 现有 handoff_schemas.py（`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`）
- 现有 4 个 subagent：`subagents/builtins/{code_executor,data_analyst,chart_maker,report_writer}.py`
- Ev19TemplateGuardrailProvider（参考 GuardrailProvider 模板）：`guardrails/ev19_template_provider.py`
- script_invocation_only_provider.py：`guardrails/script_invocation_only_provider.py`
- experiment_context.py：`packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py`
- LangChain `@tool` + ToolRuntime API：见 `view_image_tool.py` / `present_file_tool.py` 等现有 first-party tool 范例
