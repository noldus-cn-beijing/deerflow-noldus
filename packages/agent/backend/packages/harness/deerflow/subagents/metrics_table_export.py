"""Deterministic export of the metrics results table (CSV + JSON, same source).

Spec 2026-06-30 C1 (module 1): after ``run_metric_plan`` aggregates metrics, this
module exports a **clean user-facing** results table — the computed metric values
per subject + per-group summary — so the artifact gallery can show *results*
(not process files / raw uploads / internal handoff viscera).

Two artifacts, same in-memory source (SSOT — values computed once, serialized
twice; CSV and JSON numbers can never drift):

- ``metrics_table.csv`` — one row per subject; columns
  ``subject, group, <metric1>, <metric2>, ...``. Full table for researchers
  to download into SPSS/Prism.
- ``metrics_table.json`` — clean structured data for the frontend table:
  ``{paradigm, metric_names, groups[], per_subject[]}`` with IQR outlier flags.

**Strip viscera by construction**: this module only ever sees
``metrics_summary`` / ``per_subject`` / ``subject_groups`` / ``paradigm`` — it
physically cannot emit ``gate_signals`` / ``statistics`` / ``assessment`` /
``confidence`` / ``inputs`` / ``sealed_by`` / ``handoff``. Those live in the
internal handoff; the export reads from the pre-viscera aggregation dict.

Outlier detection (spec §三 module 1): Tukey IQR per (group, metric) —
``value < Q1 - 1.5*IQR`` or ``value > Q3 + 1.5*IQR``. Q1/Q3 via median of the
lower/upper halves (exclusive median, the boxplot convention researchers expect).
``None`` values (metric not applicable for that subject) are excluded from the
IQR computation AND never flagged. n<2 → no outliers detectable (all False).

This module is **stdlib-only** (csv / json / os / statistics / pathlib): no
deerflow imports, so it is picklable (ProcessPoolExecutor parity) and cannot
close any import cycle. Callers lazy-import it inside the tool function body.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any

# ============================================================================
# IQR outlier flags (Tukey, per group × metric)
# ============================================================================


def _median_of_half(sorted_vals: list[float]) -> float:
    """Median of a sorted list (exclusive-median style halves fed in already sorted)."""
    return statistics.median(sorted_vals)


def _iqr_outlier_flags(
    per_subject: dict[str, dict[str, float | None]],
    subject_groups: dict[str, str],
) -> dict[str, dict[str, bool]]:
    """Per (subject, metric) outlier flag via IQR over the subject's GROUP values.

    For each (group, metric): collect non-None values of subjects in that group.
    n<2 → all flags False for that (group, metric) (no outliers detectable).
    Else Q1 = median of lower half, Q3 = median of upper half (exclusive median
    split, Tukey), IQR = Q3-Q1, lo = Q1-1.5*IQR, hi = Q3+1.5*IQR. A subject's
    value is an outlier iff value is not None and (value < lo or value > hi).

    Returns: ``{subject_name: {metric_name: bool}}``.
    """
    # Bucket non-None values by (group, metric) for the IQR reference set.
    bucket: dict[tuple[str, str], list[float]] = {}
    for subject, metrics in per_subject.items():
        group = subject_groups.get(subject, "unknown")
        for metric, value in metrics.items():
            if value is None:
                continue
            bucket.setdefault((group, metric), []).append(float(value))

    fences: dict[tuple[str, str], tuple[float, float]] = {}
    for (group, metric), vals in bucket.items():
        if len(vals) < 2:
            # n<2 → no outliers detectable; leave absent (defaults to False).
            continue
        ordered = sorted(vals)
        mid = len(ordered) // 2
        lower = ordered[:mid]
        upper = ordered[mid + (1 if len(ordered) % 2 == 1 else 0):]
        if not lower or not upper:
            continue
        q1 = _median_of_half(lower)
        q3 = _median_of_half(upper)
        iqr = q3 - q1
        fences[(group, metric)] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    flags: dict[str, dict[str, bool]] = {}
    for subject, metrics in per_subject.items():
        group = subject_groups.get(subject, "unknown")
        per_metric: dict[str, bool] = {}
        for metric, value in metrics.items():
            is_outlier = False
            if value is not None:
                fence = fences.get((group, metric))
                if fence is not None:
                    lo, hi = fence
                    v = float(value)
                    is_outlier = v < lo or v > hi
            per_metric[metric] = is_outlier
        flags[subject] = per_metric
    return flags


# ============================================================================
# Shared in-memory table (SSOT for both CSV + JSON serialization)
# ============================================================================


def _sorted_metric_names(
    metrics_summary: dict[str, dict[str, dict]],
    per_subject: dict[str, dict[str, float | None]],
) -> list[str]:
    """Sorted union of all metric names across groups + subjects (CSV column order)."""
    names: set[str] = set()
    for group_metrics in metrics_summary.values():
        names.update(group_metrics.keys())
    for subject_metrics in per_subject.values():
        names.update(subject_metrics.keys())
    return sorted(names)


def _write_json(
    outputs_dir: Path,
    metric_names: list[str],
    metrics_summary: dict[str, dict[str, dict]],
    per_subject: dict[str, dict[str, float | None]],
    subject_groups: dict[str, str],
    outlier_flags: dict[str, dict[str, bool]],
    paradigm: str,
) -> Path:
    """Serialize the clean JSON (groups summary + per-subject rows + outlier flags).

    mean/std/n read straight from ``metrics_summary`` (the aggregator's
    ``_compute_stat`` — sample stdev n-1, ignoring None); they are NOT recomputed
    here. Only outlier_flags are newly derived (IQR).
    """
    groups_payload: list[dict[str, Any]] = []
    for group in sorted(metrics_summary.keys()):
        gm = metrics_summary[group]
        metrics_payload: dict[str, dict[str, Any]] = {}
        for metric in sorted(gm.keys()):
            stat = gm[metric]
            # stat shape: {mean, std, n, parameters_used} (mean/std may be None).
            metrics_payload[metric] = {
                "mean": stat.get("mean"),
                "std": stat.get("std"),
                "n": stat.get("n", 0),
            }
        # n at group level = the max n across its metrics (groups carry per-metric n;
        # use the metric with the most applicable subjects as the group's headline n).
        group_n = max((m.get("n", 0) for m in metrics_payload.values()), default=0)
        groups_payload.append({"group": group, "n": group_n, "metrics": metrics_payload})

    per_subject_payload: list[dict[str, Any]] = []
    for subject in sorted(per_subject.keys()):
        per_subject_payload.append(
            {
                "subject": subject,
                "group": subject_groups.get(subject, "unknown"),
                "values": dict(per_subject[subject]),
                "outlier_flags": dict(outlier_flags.get(subject, {})),
            }
        )

    data = {
        "paradigm": paradigm,
        "metric_names": metric_names,
        "groups": groups_payload,
        "per_subject": per_subject_payload,
    }
    return _atomic_write_json(outputs_dir / "metrics_table.json", data)


def _write_csv(
    outputs_dir: Path,
    metric_names: list[str],
    per_subject: dict[str, dict[str, float | None]],
    subject_groups: dict[str, str],
) -> Path:
    """Serialize the CSV: header ``subject,group,<metric>,...``, one row per subject."""
    final = outputs_dir / "metrics_table.csv"
    tmp = outputs_dir / "metrics_table.csv.tmp"
    header = ["subject", "group", *metric_names]
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for subject in sorted(per_subject.keys()):
            row_metrics = per_subject[subject]
            cells: list[str] = [subject, subject_groups.get(subject, "unknown")]
            for metric in metric_names:
                value = row_metrics.get(metric)
                cells.append("" if value is None else _format_number(value))
            writer.writerow(cells)
    os.replace(tmp, final)
    os.chmod(final, 0o644)
    return final


def _format_number(value: float) -> str:
    """Render a number for CSV (use repr-ish precision; floats as-is, ints as ints).

    ``str()`` on a float keeps full precision (no silent rounding) and renders
    integer-valued floats with a trailing ``.0``; researchers expect raw values.
    """
    return str(value)


def _atomic_write_json(path: Path, data: Any) -> Path:
    """Atomic JSON write (tmp + os.replace + chmod 0o644). Deterministic key order."""
    tmp = path.parent / f"{path.name}.tmp"
    tmp.write_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
    os.replace(tmp, path)
    os.chmod(path, 0o644)
    return path


# ============================================================================
# Public entry point
# ============================================================================


def export_metrics_table(
    *,
    metrics_summary: dict[str, dict[str, dict]],
    per_subject: dict[str, dict[str, float | None]],
    subject_groups: dict[str, str],
    outputs_dir: Path,
    paradigm: str = "",
) -> tuple[Path, Path]:
    """Deterministically export the clean metrics results table (CSV + JSON).

    Writes ``outputs_dir/metrics_table.csv`` and ``outputs_dir/metrics_table.json``
    atomically. Both files derive from the SAME in-memory build (SSOT: values
    computed once, serialized twice — CSV/JSON numbers can never drift).

    Args:
      metrics_summary: ``{group: {metric: {mean, std, n, parameters_used}}}``
        (from the aggregator; mean/std may be None, n is applicable-subject count).
      per_subject: ``{subject_name: {metric: value | None}}`` (None = not applicable).
      subject_groups: ``{subject_name: group}`` (from the aggregator — the
        additive ``subject_groups`` return key; SSOT for the subject→group mapping).
      outputs_dir: host Path to ``outputs/`` (created if missing).
      paradigm: optional paradigm tag (JSON header only; not in CSV).

    Returns:
      ``(csv_path, json_path)`` — the two written absolute Paths.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)

    metric_names = _sorted_metric_names(metrics_summary, per_subject)
    outlier_flags = _iqr_outlier_flags(per_subject, subject_groups)

    csv_path = _write_csv(outputs_dir, metric_names, per_subject, subject_groups)
    json_path = _write_json(
        outputs_dir,
        metric_names,
        metrics_summary,
        per_subject,
        subject_groups,
        outlier_flags,
        paradigm,
    )
    return csv_path, json_path
