"""ethoinsight.assess — result assessment using domain knowledge.

Interprets statistical results in context of behavioral neuroscience
domain knowledge: normal ranges, anxiety/depression thresholds, and
insight frameworks from EthoInsight technical documents.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Reference thresholds (fallback when knowledge YAML not available)
# Source: docs/EthoInsight-技术文档/小鼠焦虑样行为范式-20260112.md
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS = {
    "epm": {
        "open_arm_time_ratio": {
            "normal_range": (0.15, 0.25),
            "high_anxiety": {"below": 0.10},
            "low_anxiety": {"above": 0.30},
            "unit": "ratio",
        },
    },
    "open_field": {
        "center_time_ratio": {
            "normal_range": (0.20, 0.30),
            "high_anxiety": {"below": 0.15},
            "unit": "ratio",
        },
        "distance_moved": {
            "normal_range": (2000, 3500),
            "hypoactivity": {"below": 1500},
            "hyperactivity": {"above": 4000},
            "unit": "cm (5-min test)",
        },
    },
    "o_maze": {
        "open_arm_time_ratio": {
            "normal_range": (0.15, 0.25),
            "high_anxiety": {"below": 0.10},
            "unit": "ratio",
        },
    },
    "shoaling": {
        # Shoaling doesn't have fixed anxiety thresholds;
        # assessment is relative (group comparison based)
    },
}

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
    knowledge_dir: str = "knowledge/",
    metrics_result: dict | None = None,
) -> dict:
    """Assess statistical results using domain knowledge.

    Args:
        statistics_results: Output of ``statistics.compare_groups()``.
        paradigm: Paradigm name (e.g. "epm", "shoaling").
        knowledge_dir: Path to YAML knowledge base directory.
        metrics_result: Optional output of ``metrics.compute_paradigm_metrics()``
            for threshold-based assessment.

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
    # Try to load paradigm-specific knowledge
    thresholds = _load_thresholds(paradigm, knowledge_dir)

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
                    "severity": "notable" if effect_mag in ("medium", "large") else "minor",
                    "evidence": f"p = {comp.get('p_value', '?'):.4f}, "
                                f"d = {effect_d:.2f}" if effect_d else
                                f"p = {comp.get('p_value', '?'):.4f}",
                }
                findings.append(finding)

                # Phenotype indicators based on metric + paradigm
                indicator = _infer_phenotype(metric_name, paradigm, comp)
                if indicator:
                    phenotype_indicators.append(indicator)

    # 2. Threshold-based assessment (if metrics_result provided)
    if metrics_result and thresholds:
        threshold_findings = _assess_thresholds(
            metrics_result, paradigm, thresholds,
        )
        findings.extend(threshold_findings)

    # 3. Generate overall assessment
    sig_count = sum(1 for f in findings if f.get("severity") in ("notable", "moderate", "severe"))
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

    # 4. Recommendations
    if not findings:
        recommendations.append("Consider increasing sample size for more statistical power.")
    if any(f.get("effect_magnitude") == "large" for f in findings):
        recommendations.append("Large effect sizes suggest robust biological effects. "
                               "Consider replication with independent cohorts.")
    if paradigm in ("epm", "open_field", "o_maze"):
        recommendations.append("For anxiety phenotyping, cross-validate with at least "
                               "one additional anxiety paradigm for directional consistency.")

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


def _load_thresholds(paradigm: str, knowledge_dir: str) -> dict:
    """Load paradigm thresholds from YAML knowledge base, falling back to defaults."""
    yaml_path = os.path.join(knowledge_dir, f"{paradigm}.yaml")
    if os.path.isfile(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data.get("thresholds", {})
        except Exception:
            pass
    return _DEFAULT_THRESHOLDS.get(paradigm, {})


def _infer_phenotype(metric_name: str, paradigm: str, comp: dict) -> str | None:
    """Infer phenotype indicator from a significant comparison."""
    effect_mag = comp.get("effect_size", {}).get("magnitude", "")
    if effect_mag not in ("medium", "large"):
        return None

    indicators = {
        ("epm", "open_arm_time_ratio"): "Anxiety-like phenotype (EPM open arm avoidance)",
        ("open_field", "center_time_ratio"): "Anxiety-like phenotype (open field center avoidance)",
        ("open_field", "thigmotaxis_index"): "Anxiety-like phenotype (increased thigmotaxis)",
        ("o_maze", "open_arm_time_ratio"): "Anxiety-like phenotype (O-maze open area avoidance)",
        ("shoaling", "mean_iid"): "Altered social cohesion (shoaling group spacing)",
        ("shoaling", "mean_nnd"): "Altered social proximity (nearest-neighbor distance)",
        ("shoaling", "mean_polarity"): "Altered group coordination (movement alignment)",
    }
    return indicators.get((paradigm, metric_name))


def _assess_thresholds(
    metrics_result: dict,
    paradigm: str,
    thresholds: dict,
) -> list[dict]:
    """Assess per-subject metrics against reference thresholds."""
    findings = []
    per_subject = metrics_result.get("per_subject", {})

    for subject, subject_metrics in per_subject.items():
        for metric_name, value in subject_metrics.items():
            if not isinstance(value, (int, float)) or value is None:
                continue
            if metric_name not in thresholds:
                continue

            ref = thresholds[metric_name]
            normal_range = ref.get("normal_range")
            if normal_range and (value < normal_range[0] or value > normal_range[1]):
                # Check severity thresholds
                for key in ("high_anxiety", "hypoactivity"):
                    threshold = ref.get(key, {})
                    if "below" in threshold and value < threshold["below"]:
                        findings.append({
                            "metric": metric_name,
                            "finding": f"{subject}: {metric_name} = {value:.3f} "
                                       f"below threshold ({threshold['below']})",
                            "severity": "notable",
                            "evidence": f"Reference range: {normal_range}",
                        })
                for key in ("low_anxiety", "hyperactivity"):
                    threshold = ref.get(key, {})
                    if "above" in threshold and value > threshold["above"]:
                        findings.append({
                            "metric": metric_name,
                            "finding": f"{subject}: {metric_name} = {value:.3f} "
                                       f"above threshold ({threshold['above']})",
                            "severity": "notable",
                            "evidence": f"Reference range: {normal_range}",
                        })

    return findings
