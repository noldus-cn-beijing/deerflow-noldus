"""ethoinsight.assess — result assessment based on between-group statistical comparison.

判读统计结果时仅基于组间显著差异 + 效应量方向，**不参考绝对阈值 / 常模 /
文献基线**（2026-05-13 同事反馈硬要求 + CLAUDE.md §9）。

保留 _infer_phenotype 用于把"显著差异 + 效应方向"映射到表型标签（如
"Anxiety-like phenotype (EPM open arm avoidance)"），但映射本身依赖
显著性，不依赖绝对值大小。
"""

from __future__ import annotations


_SEVERITY_LABELS = {
    1.0: "mild",
    1.5: "moderate",
    2.0: "severe",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_results(
    statistics_results: dict,
    paradigm: str,
    metrics_result: dict | None = None,
) -> dict:
    """Assess statistical results based on between-group comparisons.

    Args:
        statistics_results: Output of ``statistics.compare_groups()``.
        paradigm: Paradigm name (e.g. "epm", "open_field", "forced_swim").
        metrics_result: Optional output of ``metrics.compute_paradigm_metrics()``
            (reserved for future use, e.g. subject-level effect direction).

    Returns:
        {
            "paradigm": str,
            "overall_assessment": str,
            "findings": [{"metric", "finding", "severity", "evidence"}],
            "phenotype_indicators": [str],
            "recommendations": [str],
            "confidence": float,
        }
    """
    findings: list[dict] = []
    phenotype_indicators: list[str] = []
    recommendations: list[str] = []

    # 1. Assess statistical comparisons
    comparisons = statistics_results.get("comparisons", {})
    for metric_name, comps in comparisons.items():
        for comp in comps:
            if comp.get("significant"):
                effect = comp.get("effect_size", {})
                effect_d = effect.get("d")
                effect_mag = effect.get("magnitude", "unknown")

                finding = {
                    "metric": metric_name,
                    "finding": f"Significant difference between "
                    f"{comp.get('group1', '?')} and {comp.get('group2', '?')}",
                    "test": comp.get("test_used", "unknown"),
                    "p_value": comp.get("p_value"),
                    "effect_size_d": effect_d,
                    "effect_magnitude": effect_mag,
                    "severity": "notable"
                    if effect_mag in ("medium", "large")
                    else "minor",
                    "evidence": f"p = {comp.get('p_value', '?'):.4f}, "
                    f"d = {effect_d:.2f}"
                    if effect_d
                    else f"p = {comp.get('p_value', '?'):.4f}",
                }
                findings.append(finding)

                # Phenotype indicators based on metric + paradigm
                indicator = _infer_phenotype(metric_name, paradigm, comp)
                if indicator:
                    phenotype_indicators.append(indicator)

    # 2. Generate overall assessment
    sig_count = sum(
        1 for f in findings if f.get("severity") in ("notable", "moderate", "severe")
    )
    total_metrics = len(comparisons)

    if sig_count == 0:
        overall = "No significant differences detected between groups."
        confidence = 0.3
    elif sig_count <= total_metrics * 0.3:
        overall = f"Mild effects detected: {sig_count} of {total_metrics} metrics show significant group differences."
        confidence = 0.5
    else:
        overall = f"Strong effects detected: {sig_count} of {total_metrics} metrics show significant group differences."
        confidence = 0.8

    # 3. Recommendations
    if not findings:
        recommendations.append(
            "Consider increasing sample size for more statistical power."
        )
    if any(f.get("effect_magnitude") == "large" for f in findings):
        recommendations.append(
            "Large effect sizes suggest robust biological effects. "
            "Consider replication with independent cohorts."
        )

    return {
        "paradigm": paradigm,
        "overall_assessment": overall,
        "findings": findings,
        "phenotype_indicators": phenotype_indicators,
        "recommendations": recommendations,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_phenotype(metric_name: str, paradigm: str, comp: dict) -> str | None:
    """Infer phenotype indicator from a significant comparison."""
    effect_mag = comp.get("effect_size", {}).get("magnitude", "")
    if effect_mag not in ("medium", "large"):
        return None

    indicators = {
        (
            "epm",
            "open_arm_time_ratio",
        ): "Anxiety-like phenotype (EPM open arm avoidance)",
        (
            "open_field",
            "center_time_ratio",
        ): "Anxiety-like phenotype (open field center avoidance)",
        (
            "open_field",
            "thigmotaxis_index",
        ): "Anxiety-like phenotype (increased thigmotaxis)",
        (
            "o_maze",
            "open_arm_time_ratio",
        ): "Anxiety-like phenotype (O-maze open area avoidance)",
    }
    return indicators.get((paradigm, metric_name))
