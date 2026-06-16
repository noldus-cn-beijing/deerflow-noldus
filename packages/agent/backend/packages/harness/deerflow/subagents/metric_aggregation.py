"""Pure metric-aggregation function — single source of truth for handoff reconstruction.

Spec S4 (2026-06-12): the code-executor aggregation logic that reads m_*.json +
plan_metrics.json + groups.json and builds the CodeExecutorHandoff payload is shared
by exactly two callers:

1. ``run_metric_plan_tool`` — the deterministic first-party tool that executes the
   plan via ProcessPoolExecutor then aggregates the on-disk artifacts.
2. ``subagents/executor.py::_attempt_auto_seal_from_artifacts`` — the harness
   auto-seal fallback when a subagent finishes but forgets to call the seal tool.

Both MUST produce byte-identical aggregation from the same m_*.json (Spec S4 §1.4 /
test #6: "抽出复用无行为变化"). Extracting this into one pure function is how that
invariant is enforced — there is no second copy to drift.

Function is pure (filesystem read only, no LLM, no I/O side effects beyond reads).
It deliberately does NOT call validate here for the auto-seal path's historical
contract; callers that want validation pass ``run_validation=True``. The auto-seal
caller passes False to preserve byte-identical output with pre-S4 behaviour (test #6
locks the existing payload; adding validation warnings to auto-seal would change it).

Run-metric-plan caller passes True (Spec S4 §1.4 VALIDATION_ERROR) because the tool's
contract is "run compute + statistics + aggregate + validate, all deterministic".
"""

from __future__ import annotations

import logging
import statistics as _stats
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Validation helpers (Spec S4 §1.4): in-process, no stdout scraping.
# Scripts already save_output_json(payload) the same payload they emit_result print,
# so reading m_*.json gives us value + parameters_used. validate here is on the
# read values, not on captured stdout.
# ============================================================================


def _collect_validation_warnings(
    plan: dict[str, Any], workspace: Path
) -> list[dict[str, Any]]:
    """Run L-A (NaN/Inf) + L-B (catalog range) validation on disk artifacts.

    Mirrors the prompt's historical VALIDATION_ERROR aggregation: each metric gets
    at most one warning listing all its issues; merges are per-metric not per-line.

    Returns a list of DataQualityWarning-shaped dicts (code=METRIC_VALIDATION.*).
    Empty list when everything is clean or when ethoinsight is unavailable.
    """
    try:
        from ethoinsight.validate_catalog import validate_plan_results
    except Exception:
        # ethoinsight not installed (harness-only env) → validation is best-effort;
        # the prompt-level L-B step would also be a no-op. Never block aggregation.
        return []

    warnings: list[dict[str, Any]] = []
    try:
        # validate_plan_results reads each entry.output via resolve_sandbox_path
        # (DEERFLOW_PATH_* env). In run_metric_plan the workspace env is inherited
        # by the process pool; in tests the paths are real host paths (resolve is
        # a passthrough for non-/mnt paths). Either way it returns violations.
        violations = validate_plan_results(plan)
    except Exception:
        logger.warning("[aggregate] validate_plan_results failed; skipping L-B", exc_info=True)
        return warnings

    # Group violations by metric base id (strip the per-subject ``#idx`` suffix L-B
    # adds so multiple subjects of one metric don't collide).
    by_metric: dict[str, list[str]] = {}
    for v in violations:
        metric = str(v.get("metric", "")).split("#", 1)[0] or "unknown"
        issue = v.get("issue", "unknown")
        value = v.get("value", "")
        by_metric.setdefault(metric, []).append(f"{issue} (value={value})")

    for metric, issues in by_metric.items():
        warnings.append({
            "severity": "critical",
            "code": "METHOD.METRIC_VALIDATION",
            "metric": metric,
            "message": f"{metric}: {'; '.join(issues)}",
            "evidence": {"source": "validate_catalog"},
            "blocks_downstream": True,
        })
    return warnings


# ============================================================================
# Core pure aggregation
# ============================================================================


def _compute_stat(values: list[Any]) -> dict[str, Any]:
    """Compute mean/std/n over a list of per-subject values for one metric.

    Spec 2026-06-16 缺陷 2：旧的聚合循环只 n+=1，mean 恒等于首个 subject 的值、
    std 恒为 null，致 metrics_summary 与 per_subject 矛盾（data-analyst 手算螺旋）。

    语义锁定（spec §2，避免下游 LLM 重算）：
      - ``None``（compute 脚本报"不适用"）忽略——不计入 mean/std/n，与下游
        ``MetricStat.applicable`` 语义对齐。
      - **mean** = 算术平均（``statistics.mean``），忽略 None。
      - **std** = **样本标准差**（``statistics.stdev``，n−1 分母），n<2 时 None。
        与 scipy/pandas ddof=1 默认一致，行为学统计惯例。
      - **n** = 非 None 值个数（applicable subject 数）。

    抽成纯函数便于直接单测（spec §2 末允许），避免构造整份 plan/workspace。
    """
    vals_clean = [v for v in values if v is not None]
    n = len(vals_clean)
    return {
        "mean": _stats.mean(vals_clean) if vals_clean else None,
        "std": _stats.stdev(vals_clean) if n >= 2 else None,
        "n": n,
    }


def aggregate_metrics_to_handoff(
    plan: dict[str, Any],
    workspace: Path,
    *,
    run_validation: bool = False,
) -> dict[str, Any]:
    """Aggregate on-disk m_*.json artifacts into a CodeExecutorHandoff payload.

    Single source of truth (Spec S4 §1.4). Reads plan_metrics.json (passed in as
    ``plan``) + groups.json + every m_*.json present in ``workspace``; reconstructs
    metrics_summary / per_subject / output_files / data_quality_warnings / errors
    deterministically. Does NOT touch stdout (scripts already save_output_json).

    Completeness (Spec S4 §1.5, pure function — not LLM judgement):
      - expected output set == actual m_*.json set       → status "completed"
      - some expected outputs missing                     → status "partial"
      - (the "failed" status is set by callers when the
         failure ratio exceeds their threshold, not here)

    Args:
        plan: parsed plan_metrics.json dict.
        workspace: host-side workspace dir (where m_*.json / groups.json live).
        run_validation: when True, also run L-A + L-B validation and merge any
            VALIDATION_ERROR findings into ``data_quality_warnings``. The auto-seal
            caller passes False to stay byte-identical with pre-S4 behaviour; the
            run_metric_plan caller passes True (its contract includes validation).

    Returns:
        A dict with keys: status, metrics_summary, per_subject, output_files,
        data_quality_warnings, errors, paradigm, ev19_template, missing_expected,
        n_total, n_present. The caller adds ``summary``/``sealed_by`` and any
        extra fields (statistics, gate_signals, confidence, inputs) before sealing.
    """
    import json

    plan_metrics = plan.get("metrics", [])
    if not plan_metrics:
        return {
            "status": "failed",
            "metrics_summary": {},
            "per_subject": {},
            "output_files": {"metrics": []},
            "data_quality_warnings": [],
            "errors": ["aggregate: plan has no metrics[] to reconcile"],
            "paradigm": plan.get("paradigm", ""),
            "ev19_template": plan.get("ev19_template"),
            "missing_expected": [],
            "n_total": 0,
            "n_present": 0,
        }

    # Enumerate plan's expected output filenames + metric_id lookup.
    expected: set[str] = set()
    metric_id_to_entry: dict[str, dict] = {}
    for m in plan_metrics:
        output = m.get("output", "")
        if output:
            fname = Path(output).name
            expected.add(fname)
            metric_id_to_entry[fname] = m

    if not expected:
        return {
            "status": "failed",
            "metrics_summary": {},
            "per_subject": {},
            "output_files": {"metrics": []},
            "data_quality_warnings": [],
            "errors": ["aggregate: plan metrics have no output paths to reconcile"],
            "paradigm": plan.get("paradigm", ""),
            "ev19_template": plan.get("ev19_template"),
            "missing_expected": [],
            "n_total": 0,
            "n_present": 0,
        }

    # glob actual m_*.json present on disk.
    actual: set[str] = set()
    for f in workspace.glob("m_*.json"):
        actual.add(f.name)

    missing = expected - actual
    status = "completed" if not missing else "partial"

    # Read groups.json (subject_file -> group_name).
    groups: dict[str, str] = {}
    groups_path = workspace / "groups.json"
    if groups_path.exists():
        try:
            groups = json.loads(groups_path.read_text(encoding="utf-8"))
        except Exception:
            groups = {}

    # subject_index -> raw file path from plan inputs (for per_subject naming).
    raw_files: list[str] = []
    plan_inputs = plan.get("inputs", {})
    if isinstance(plan_inputs, dict):
        raw_files = plan_inputs.get("raw_files", []) or []

    metrics_summary: dict[str, dict[str, dict]] = {}
    per_subject: dict[str, dict[str, Any]] = {}
    output_files_metrics: list[str] = []
    errors: list[str] = []

    for fname in sorted(actual):
        fpath = workspace / fname
        try:
            mdata = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue

        metric_name = mdata.get("metric", "")
        value = mdata.get("value")
        params = mdata.get("parameters_used", {}) or {}

        entry = metric_id_to_entry.get(fname, {})
        subject_index = entry.get("subject_index", 0)
        subject_file = (
            raw_files[subject_index]
            if subject_index < len(raw_files)
            else f"subject_{subject_index}"
        )
        subject_name = Path(subject_file).stem
        group_name = groups.get(
            subject_file, groups.get(Path(subject_file).name, "unknown")
        )

        # per_subject
        if subject_name not in per_subject:
            per_subject[subject_name] = {}
        per_subject[subject_name][metric_name] = value

        # metrics_summary: group -> metric -> MetricStat-shaped dict
        # 累加器：循环内只收集 values 到临时 _values 列表，循环结束后统一算
        # mean/std/n（spec 2026-06-16 缺陷 2：旧实现只 n+=1，mean 恒=首个 subject、
        # std 恒=null，致 data-analyst 读到与 per_subject 矛盾的 mean 手算螺旋）。
        if group_name not in metrics_summary:
            metrics_summary[group_name] = {}
        if metric_name not in metrics_summary[group_name]:
            metrics_summary[group_name][metric_name] = {
                "_values": [value],          # 临时累加，末尾 _finalize_stats 清除
                "mean": None,
                "std": None,
                "n": 1,
                "parameters_used": params,
            }
        else:
            existing = metrics_summary[group_name][metric_name]
            existing["_values"].append(value)

        output_files_metrics.append(f"/mnt/user-data/workspace/{fname}")

    data_quality_warnings: list[dict] = []

    # Completeness errors (auto-seal parity: missing items → errors + AUTO_SEAL_INCOMPLETE
    # is added by the auto-seal caller's own naming; here we keep the generic missing list).
    if missing:
        for mf in sorted(missing):
            entry = metric_id_to_entry.get(mf, {})
            mid = entry.get("id", mf)
            si = entry.get("subject_index", "?")
            errors.append(
                f"expected output {mf} (metric_id={mid}, subject_index={si}) "
                f"not found in workspace"
            )

    # Finalize statistics: 循环内只收 _values，这里统一算 mean/std/n 并删临时字段。
    # _compute_stat 忽略 None（不适用 subject 不计入 mean/std/n）。
    for group_metrics in metrics_summary.values():
        for stat in group_metrics.values():
            vals = stat.pop("_values", None)
            if vals:
                computed = _compute_stat(vals)
                stat["mean"] = computed["mean"]
                stat["std"] = computed["std"]
                stat["n"] = computed["n"]

    # Optional validation (run_metric_plan path only).
    if run_validation:
        data_quality_warnings.extend(_collect_validation_warnings(plan, workspace))

    return {
        "status": status,
        "metrics_summary": metrics_summary,
        "per_subject": per_subject,
        "output_files": {"metrics": output_files_metrics},
        "data_quality_warnings": data_quality_warnings,
        "errors": errors,
        "paradigm": plan.get("paradigm", ""),
        "ev19_template": plan.get("ev19_template"),
        "missing_expected": sorted(missing),
        "n_total": len(expected),
        "n_present": len(actual & expected),
    }
