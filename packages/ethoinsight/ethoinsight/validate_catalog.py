"""Catalog-driven metric validation — L-B 语义范围校验。

两层分工：
  L-A（validate.py）    — 进程内安全网，只查 NaN/Inf
  L-B（本模块）          — 按 output_unit 查范围 + 复合_stats + 孤儿/未知单位

两个入口：
  1. validate_metrics_against_catalog(results, paradigm)
     —— 直接调函数：load_catalog 查 output_unit，能检出 catalog 之外的孤儿指标
        （catalog_unknown）。供单测 / 程序内调用。
  2. validate_plan_results(plan)  ←  CLI 用
     —— 吃 plan_metrics.json：用每条 metric 自带的 output_unit（resolve.py 从
        catalog 透传），不再 load_catalog（P1-1）。按 subject 逐条验证，同 metric_id
        多 subject 不互相覆盖（P0-2）。注意：plan 由 resolve 生成，只含 catalog 内
        指标，故 CLI 路径不会遇到孤儿（孤儿检测仅在入口 1 生效）。

CLI： python -m ethoinsight.validate_catalog --plan <plan_metrics.json>

TODO: radians 上限待确认（关联指标定义）。当前只验 ≥0。
TODO: plausible_max 机制预留，等 catalog MetricEntry 增加可选字段后启用。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from ethoinsight.catalog.loader import load_catalog
from ethoinsight.scripts._cli import resolve_sandbox_path


# ============================================================================
# B.3: output_unit → 规则表（6 种已知单位，枚举强制）
# ============================================================================

_RANGE_RULES: dict[str, dict[str, Any]] = {
    "ratio": {
        "lower": 0.0,
        "upper": 1.0,
        "integer_only": False,
        "description": "ratio (0–1)",
    },
    "seconds": {
        "lower": 0.0,
        "upper": None,  # plausible_max: 预留，当前等价于无上限
        "integer_only": False,
        "description": "seconds (≥0)",
    },
    "count": {
        "lower": 0,
        "upper": None,
        "integer_only": True,
        "description": "count (≥0, integer)",
    },
    "radians": {
        "lower": 0.0,
        "upper": None,  # TODO: 上限待确认（关联指标定义），先只验 ≥0
        "integer_only": False,
        "description": "radians (≥0, upper bound TBD)",
    },
    "cm": {
        "lower": 0.0,
        "upper": None,  # plausible_max: 预留
        "integer_only": False,
        "description": "cm (≥0)",
    },
    "mm_s2": {
        "lower": 0.0,
        "upper": None,  # plausible_max: 预留
        "integer_only": False,
        "description": "mm/s² (≥0)",
    },
}


# ============================================================================
# B.2: 复合 _stats 字段白名单
# ============================================================================

# 字段名组件匹配：含这些 token → 套 output_unit 范围（上下限）
_RANGE_FIELD_TOKENS = frozenset({"mean", "median", "min", "max", "p25", "p75"})

# 字段名组件匹配：含这些 token → 只验 ≥0 + 非 NaN（不套上限）
# n / count 额外验整数
_NON_NEGATIVE_ONLY_TOKENS = frozenset({"std", "stdev", "sem", "var"})

# 精确匹配的计数类字段：≥0 + 整数
_COUNT_LIKE_FIELDS = frozenset({"n", "count"})


def _is_std_like_field(field_name: str) -> bool:
    """Check if field name contains a std/stdev/sem/var token."""
    parts = field_name.lower().split("_")
    return bool(_NON_NEGATIVE_ONLY_TOKENS & set(parts))


def _is_count_like_field(field_name: str) -> bool:
    """Check if field name is a count-like field (n, count)."""
    return field_name.lower() in _COUNT_LIKE_FIELDS


def _check_numeric_strictness(
    metric_id: str, value: float, output_unit: str
) -> list[dict[str, str]]:
    """Check integer-only constraint for count-type units."""
    violations: list[dict[str, str]] = []
    rule = _RANGE_RULES.get(output_unit, {})
    if rule.get("integer_only") and isinstance(value, float):
        if not value.is_integer():
            violations.append({
                "metric": metric_id,
                "issue": "value_not_integer",
                "value": str(value),
            })
    return violations


def _apply_range_rule(
    metric_id: str, value: float, output_unit: str
) -> list[dict[str, str]]:
    """Apply output_unit range rule to a single scalar value.

    Returns list of violations (empty = ok).
    """
    violations: list[dict[str, str]] = []

    # NaN/Inf first (same as L-A but we also catch these in composite fields)
    if math.isnan(value):
        violations.append({"metric": metric_id, "issue": "NaN", "value": "NaN"})
        return violations
    if math.isinf(value):
        violations.append({"metric": metric_id, "issue": "Inf", "value": str(value)})
        return violations

    rule = _RANGE_RULES.get(output_unit)
    if rule is None:
        return violations  # unknown unit handled at higher level

    lower = rule["lower"]
    upper = rule["upper"]

    if value < lower:
        violations.append({
            "metric": metric_id,
            "issue": "value_below_lower_bound",
            "value": str(value),
        })

    if upper is not None and value > upper:
        violations.append({
            "metric": metric_id,
            "issue": "value_above_upper_bound",
            "value": str(value),
        })

    # Integer-only check (count output_unit)
    violations.extend(_check_numeric_strictness(metric_id, value, output_unit))

    return violations


def _validate_composite_stats(
    metric_id: str, value: dict[str, Any], output_unit: str
) -> list[dict[str, str]]:
    """Validate fields inside a composite _stats dict value.

    Range-applied fields (mean, median, min, max, p25, p75): full range check.
    Std-like fields (std, stdev, sem, var): only ≥0 + non-NaN.
    Count-like fields (n, count): ≥0 + integer + non-NaN.
    Unknown fields: range-applied (conservative).
    """
    violations: list[dict[str, str]] = []
    rule = _RANGE_RULES.get(output_unit)

    for field_name, field_value in value.items():
        if not isinstance(field_value, (int, float)):
            continue

        sub_metric_id = f"{metric_id}.{field_name}"

        # NaN/Inf check for every field
        if math.isnan(field_value):
            violations.append({"metric": sub_metric_id, "issue": "NaN", "value": "NaN"})
            continue
        if math.isinf(field_value):
            violations.append({"metric": sub_metric_id, "issue": "Inf", "value": str(field_value)})
            continue

        if _is_count_like_field(field_name):
            # Count-like: ≥0 + integer
            if field_value < 0:
                violations.append({
                    "metric": sub_metric_id,
                    "issue": "value_below_lower_bound",
                    "value": str(field_value),
                })
            if isinstance(field_value, float) and not field_value.is_integer():
                violations.append({
                    "metric": sub_metric_id,
                    "issue": "value_not_integer",
                    "value": str(field_value),
                })
        elif _is_std_like_field(field_name):
            # Std-like: only ≥0 (no upper bound)
            if field_value < 0:
                violations.append({
                    "metric": sub_metric_id,
                    "issue": "value_below_lower_bound",
                    "value": str(field_value),
                })
        else:
            # Range-applied: full output_unit range
            if rule is not None:
                lower = rule["lower"]
                upper = rule["upper"]
                if field_value < lower:
                    violations.append({
                        "metric": sub_metric_id,
                        "issue": "value_below_lower_bound",
                        "value": str(field_value),
                    })
                if upper is not None and field_value > upper:
                    violations.append({
                        "metric": sub_metric_id,
                        "issue": "value_above_upper_bound",
                        "value": str(field_value),
                    })

    return violations


# ============================================================================
# B.1: 主验证函数
# ============================================================================


def validate_metrics_against_catalog(
    results: dict[str, Any],
    paradigm: str,
    catalog_dir: str | Path | None = None,
) -> list[dict[str, str]]:
    """按 catalog output_unit 校验指标范围。

    Args:
        results: {metric_id: value} dict. value 可以是标量或 dict（复合 _stats）。
        paradigm: canonical paradigm key（如 "epm", "open_field", "forced_swim"）。
        catalog_dir: catalog YAML 目录，默认用内置 catalog。

    Returns:
        违规列表，每条 {"metric", "issue", "value"}（与 L-A 同 schema）。
    """
    violations: list[dict[str, str]] = []

    # 1. Load catalog
    cat = load_catalog(paradigm, catalog_dir)

    # 2. Build metric_id → output_unit map
    unit_map: dict[str, str] = {}
    for m in cat.default_metrics + cat.optional_metrics:
        unit_map[m.id] = m.output_unit

    # 3. Validate each result
    for metric_id, value in results.items():
        # --- orphan check ---
        if metric_id not in unit_map:
            violations.append({
                "metric": metric_id,
                "issue": "catalog_unknown",
                "value": str(value),
            })
            continue

        output_unit = unit_map[metric_id]

        # --- unknown output_unit check ---
        if output_unit not in _RANGE_RULES:
            violations.append({
                "metric": metric_id,
                "issue": "unknown_output_unit",
                "value": f"output_unit={output_unit}",
            })
            continue

        # --- None / non-numeric → skip ---
        if value is None or not isinstance(value, (int, float, dict)):
            continue

        # --- composite _stats ---
        if isinstance(value, dict):
            violations.extend(_validate_composite_stats(metric_id, value, output_unit))
            continue

        # --- scalar ---
        violations.extend(_apply_range_rule(metric_id, value, output_unit))

    return violations


# ============================================================================
# B.6: plan-driven 验证（CLI 用，直接吃 plan 条目的 output_unit）
# ============================================================================


def _validate_one_value(
    label: str, value: Any, output_unit: str
) -> list[dict[str, str]]:
    """Validate a single value (scalar or composite dict) against an output_unit.

    ``label`` is the violation's metric label — may carry a subject suffix
    (e.g. "center_time_ratio#0") so per-subject violations are distinguishable.
    """
    if output_unit not in _RANGE_RULES:
        return [{
            "metric": label,
            "issue": "unknown_output_unit",
            "value": f"output_unit={output_unit}",
        }]
    if value is None or not isinstance(value, (int, float, dict)):
        return []
    if isinstance(value, dict):
        return _validate_composite_stats(label, value, output_unit)
    return _apply_range_rule(label, value, output_unit)


def validate_plan_results(plan: dict[str, Any]) -> list[dict[str, str]]:
    """Validate every metric output declared in a plan_metrics.json dict.

    Uses each metric entry's own ``output_unit`` field (resolve.py emits it
    verbatim from the catalog), so this does NOT re-load the catalog — the
    plan IS the resolved-catalog projection (P1-1).

    plan_metrics.json expands one PlanMetric PER SUBJECT (resolve.py: each
    subject gets its own output path + subject_index). The same metric_id
    therefore appears multiple times; each is validated independently and
    labelled with a ``#<subject_index>`` suffix so per-subject violations
    are not collapsed (P0-2).

    Returns: violation list, each {"metric", "issue", "value"}.
    """
    violations: list[dict[str, str]] = []

    for entry in plan.get("metrics", []):
        metric_id = entry.get("id", "")
        output_path = entry.get("output", "")
        output_unit = entry.get("output_unit", "")
        subject_index = entry.get("subject_index")
        if not metric_id or not output_path:
            continue

        # Plan entry is missing output_unit → cannot range-check; surface it.
        if not output_unit:
            violations.append({
                "metric": metric_id,
                "issue": "plan_missing_output_unit",
                "value": f"subject_index={subject_index}",
            })
            continue

        try:
            data = json.loads(resolve_sandbox_path(output_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Missing/unreadable result file — completeness gap, surface it.
            violations.append({
                "metric": metric_id,
                "issue": "result_file_unreadable",
                "value": str(output_path),
            })
            continue

        # Extract value from [result]-style output: {"metric": ..., "value": ...}
        value = data.get("value")
        if value is None and metric_id in data:
            value = data[metric_id]

        # Per-subject label so multiple subjects of the same metric don't collide.
        label = (
            f"{metric_id}#{subject_index}" if subject_index is not None else metric_id
        )
        violations.extend(_validate_one_value(label, value, output_unit))

    return violations


# ============================================================================
# CLI 入口
# ============================================================================


def main(argv: list[str] | None = None) -> None:
    """CLI: python -m ethoinsight.validate_catalog --plan <plan.json>

    Reads plan_metrics.json (paradigm + metrics[] with per-entry output_unit
    + output path), validates each metric output PER SUBJECT, and prints
    VALIDATION_ERROR lines to stdout.

    Exit code is always 0 (informational; downstream decides how to handle).
    """
    ap = argparse.ArgumentParser(
        description="Catalog-driven metric range validation (L-B)"
    )
    ap.add_argument(
        "--plan", required=True,
        help="Path to plan_metrics.json (paradigm + metrics[] with output_unit + output)",
    )
    args = ap.parse_args(argv)

    plan_path = Path(args.plan)
    if not plan_path.is_file():
        print(f"VALIDATION_ERROR: plan file not found: {plan_path}", file=sys.stderr)
        sys.exit(0)

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"VALIDATION_ERROR: plan file read failed: {e}", file=sys.stderr)
        sys.exit(0)

    try:
        violations = validate_plan_results(plan)
    except Exception as e:
        print(f"VALIDATION_ERROR: plan validation failed: {e}", file=sys.stderr)
        sys.exit(0)

    for v in violations:
        print(
            f"VALIDATION_ERROR: {v['metric']}: {v['issue']} (value={v['value']})"
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
