"""Catalog resolver: catalog + columns + 用户偏差 → Plan.

输入：
  paradigm  : 范式 key（来自 lead 的 Gate 1 识别）
  columns   : raw 数据真实列名（来自 dump_headers）
  raw_files : raw 文件绝对路径列表
  include   : 用户额外要的 metric ids
  exclude   : 用户排除的 metric ids
  n_per_group / n_groups : 用于 charts/statistics 的 when 条件评估

输出：Plan dataclass（含 metrics / statistics / charts / skipped / notes）
       由 plan_to_dict() 序列化为 metric_plan.json schema_version="1.0"
"""

from __future__ import annotations

import datetime as dt
import fnmatch
from pathlib import Path

from ethoinsight.catalog.loader import CatalogError, load_catalog
from ethoinsight.catalog.schema import (
    ChartEntry,
    MetricEntry,
    Plan,
    PlanChart,
    PlanInputs,
    PlanMetric,
    PlanSkipped,
    PlanStatistics,
    StatisticsEntry,
)

SCHEMA_VERSION = "1.0"


class ResolveError(Exception):
    """Resolver 失败，含结构化 code 供 lead 反问。

    code 枚举:
      unknown_paradigm  - paradigm 不在 catalog 目录
      unknown_metric    - include 里的 id 不在 catalog
      columns_missing   - default 或 user-include 指标缺必需列
      empty_plan        - 所有 default + include 都被剪光
      schema_violation  - catalog YAML 损坏（fallback 自 CatalogError）
    """

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        self.code = code
        self.details = details or {}
        super().__init__(message)


def resolve(
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
) -> Plan:
    """生成 Plan dataclass。失败抛 ResolveError。"""
    try:
        cat = load_catalog(paradigm)
    except CatalogError as e:
        if "file not found" in str(e).lower() or "not found for paradigm" in str(e).lower():
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

    include_set = set(include or ())
    exclude_set = set(exclude or ())

    all_known_ids = {m.id for m in cat.default_metrics} | {m.id for m in cat.optional_metrics}
    unknown = include_set - all_known_ids
    if unknown:
        raise ResolveError(
            code="unknown_metric",
            message=f"Metric(s) not found in {paradigm} catalog: {sorted(unknown)}",
            details={"requested": sorted(unknown), "available": sorted(all_known_ids)},
        )

    plan_metrics: list[PlanMetric] = []
    skipped: list[PlanSkipped] = []

    # Step 1: process default_metrics
    for m in cat.default_metrics:
        if m.id in exclude_set:
            skipped.append(PlanSkipped(
                id=m.id, reason="user.exclude",
                detail=f"User explicitly excluded {m.id}.",
            ))
            continue
        missing = _missing_columns(m.requires_columns, columns)
        if missing:
            raise ResolveError(
                code="columns_missing",
                message=(
                    f"Required column(s) for metric '{m.id}' not found in data: "
                    f"{missing}. Default metrics must always run; if the data "
                    f"truly lacks these columns, ask the user before excluding."
                ),
                details={
                    "metric": m.id, "missing_patterns": missing,
                    "available_columns": columns,
                },
            )
        plan_metrics.append(_metric_to_plan(m, raw_files, workspace_dir, required=True, reason="paradigm.default"))

    # Step 2: process user-include
    for inc_id in include_set:
        if inc_id in exclude_set:
            skipped.append(PlanSkipped(
                id=inc_id, reason="user.exclude",
                detail=f"User specified both include and exclude for {inc_id}; honoring exclude.",
            ))
            continue
        if inc_id in {m.id for m in cat.default_metrics}:
            continue
        m = next(m for m in cat.optional_metrics if m.id == inc_id)
        missing = _missing_columns(m.requires_columns, columns)
        if missing:
            skipped.append(PlanSkipped(
                id=m.id, reason="columns.missing",
                detail=f"User-included metric {m.id} skipped: missing columns {missing}.",
            ))
            continue
        plan_metrics.append(_metric_to_plan(m, raw_files, workspace_dir, required=False, reason="user.include"))

    if not plan_metrics:
        raise ResolveError(
            code="empty_plan",
            message=f"All metrics for paradigm '{paradigm}' were excluded or unavailable.",
            details={"paradigm": paradigm},
        )

    # Step 4: charts
    plan_charts: list[PlanChart] = []
    for ch in cat.charts:
        if _evaluate_when(ch.when, n_per_group=n_per_group, n_groups=n_groups):
            plan_charts.append(_chart_to_plan(ch, raw_files, workspace_dir))

    # Step 5: statistics
    plan_stats: PlanStatistics | None = None
    if cat.statistics_default is not None:
        if _evaluate_when(cat.statistics_default.when, n_per_group=n_per_group, n_groups=n_groups):
            plan_stats = _stats_to_plan(cat.statistics_default, workspace_dir, skip_reason=None)
        else:
            plan_stats = _stats_to_plan(
                cat.statistics_default, workspace_dir,
                skip_reason=f"condition '{cat.statistics_default.when}' not met (n_per_group={n_per_group}, n_groups={n_groups})",
            )

    notes: list[str] = []
    if include_set:
        notes.append(f"User include: {sorted(include_set)}")
    if exclude_set:
        notes.append(f"User exclude: {sorted(exclude_set)}")
    notes.append(f"Data columns: {len(columns)}; metrics planned: {len(plan_metrics)}; skipped: {len(skipped)}")

    return Plan(
        schema_version=SCHEMA_VERSION,
        paradigm=cat.paradigm,
        ev19_template=ev19_template,
        generated_at=_utcnow_iso(),
        inputs=PlanInputs(raw_files=list(raw_files), groups_file=groups_file, columns_file=columns_file),
        metrics=plan_metrics,
        statistics=plan_stats,
        charts=plan_charts,
        skipped=skipped,
        notes=notes,
    )


# ============================================================================
# Helpers
# ============================================================================


def _missing_columns(patterns: list[str], available: list[str]) -> list[str]:
    """对 requires_columns 中的每个 glob pattern，检查是否 ≥1 列匹配；返回未匹配的 pattern 集。"""
    missing: list[str] = []
    for pat in patterns:
        if not any(fnmatch.fnmatchcase(col, pat) for col in available):
            missing.append(pat)
    return missing


def _metric_to_plan(m: MetricEntry, raw_files: list[str], workspace_dir: str, *, required: bool, reason: str) -> PlanMetric:
    input_path = raw_files[0]
    output_path = str(Path(workspace_dir) / f"m_{m.id}.json")
    return PlanMetric(
        id=m.id, script=m.script,
        input=input_path, output=output_path,
        required=required, reason=reason,
    )


def _chart_to_plan(ch: ChartEntry, raw_files: list[str], workspace_dir: str) -> PlanChart:
    return PlanChart(
        id=ch.id, script=ch.script,
        input=raw_files[0],
        output=str(Path(workspace_dir) / f"plot_{ch.id}.png"),
    )


def _stats_to_plan(st: StatisticsEntry, workspace_dir: str, skip_reason: str | None) -> PlanStatistics:
    return PlanStatistics(
        id=st.id, script=st.script,
        input=str(Path(workspace_dir) / "handoff_code_executor.json"),
        output=str(Path(workspace_dir) / "stats.json"),
        skip_reason=skip_reason,
    )


def _evaluate_when(condition: str, *, n_per_group: int | None, n_groups: int | None) -> bool:
    cond = condition.strip()
    if cond == "always":
        return True

    parts = [p.strip() for p in cond.split(" and ")]
    for part in parts:
        if not _evaluate_atomic_when(part, n_per_group=n_per_group, n_groups=n_groups):
            return False
    return True


def _evaluate_atomic_when(part: str, *, n_per_group: int | None, n_groups: int | None) -> bool:
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
    return False


def _utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================================
# Serialization
# ============================================================================


def plan_to_dict(plan: Plan) -> dict:
    """Plan dataclass → JSON-serializable dict."""
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
                "id": m.id, "script": m.script,
                "input": m.input, "output": m.output,
                "required": m.required, "reason": m.reason,
            } for m in plan.metrics
        ],
        "statistics": (
            None if plan.statistics is None else {
                "id": plan.statistics.id, "script": plan.statistics.script,
                "input": plan.statistics.input, "output": plan.statistics.output,
                "skip_reason": plan.statistics.skip_reason,
            }
        ),
        "charts": [
            {"id": c.id, "script": c.script, "input": c.input, "output": c.output}
            for c in plan.charts
        ],
        "skipped": [
            {"id": s.id, "reason": s.reason, "detail": s.detail}
            for s in plan.skipped
        ],
        "notes": plan.notes,
    }


if __name__ == "__main__":
    import sys
    from ethoinsight.catalog.cli import main
    sys.exit(main())
