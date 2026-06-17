"""ethoinsight.statistics — statistical testing for behavioral metrics.

Provides normality tests, group comparisons, effect sizes, and a
high-level dispatcher that works with the output of ``metrics.compute_paradigm_metrics``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


# ============================================================================
# Normality
# ============================================================================


def test_normality(values: list[float] | np.ndarray, alpha: float = 0.05) -> dict:
    """Shapiro-Wilk normality test.

    Returns:
        {"statistic", "p_value", "is_normal", "test": "shapiro-wilk", "n"}
    """
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 3:
        return {
            "statistic": None,
            "p_value": None,
            "is_normal": None,
            "test": "shapiro-wilk",
            "n": n,
            "note": "n < 3, cannot test",
        }
    stat, p = sp_stats.shapiro(arr)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "is_normal": bool(p > alpha),
        "test": "shapiro-wilk",
        "n": n,
    }


def test_homogeneity(
    *groups: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Levene's test for homogeneity of variances.

    Returns:
        {"statistic", "p_value", "is_homogeneous", "test": "levene", "n_groups"}
    """
    arrays = [np.array(g, dtype=float) for g in groups]
    arrays = [a[~np.isnan(a)] for a in arrays]
    arrays = [a for a in arrays if len(a) >= 2]
    if len(arrays) < 2:
        return {
            "statistic": None,
            "p_value": None,
            "is_homogeneous": None,
            "test": "levene",
            "n_groups": len(arrays),
            "note": "need at least 2 groups with n >= 2",
        }
    stat, p = sp_stats.levene(*arrays)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "is_homogeneous": bool(p > alpha),
        "test": "levene",
        "n_groups": len(arrays),
    }


# ============================================================================
# Two-group comparison
# ============================================================================


def compare_two_groups(
    g1: list[float] | np.ndarray,
    g2: list[float] | np.ndarray,
    alpha: float = 0.05,
    paired: bool = False,
) -> dict:
    """Compare two groups, auto-selecting parametric or non-parametric test.

    Uses Shapiro-Wilk to check normality of both groups.
    If both normal and unpaired: Levene's test decides equal-var t vs Welch t.
    If both normal and paired: paired t-test.
    If either non-normal: Mann-Whitney U (or Wilcoxon signed-rank if paired).

    Returns:
        {"test_used", "statistic", "p_value", "significant", "alpha",
         "normality_g1", "normality_g2", "effect_size", "effect_size_hedges_g",
         "variance_homogeneity" (when applicable)}
    """
    a1 = np.array(g1, dtype=float)
    a2 = np.array(g2, dtype=float)
    a1 = a1[~np.isnan(a1)]
    a2 = a2[~np.isnan(a2)]

    norm1 = test_normality(a1, alpha)
    norm2 = test_normality(a2, alpha)
    both_normal = norm1.get("is_normal", False) and norm2.get("is_normal", False)

    variance_test = None

    if both_normal:
        if paired:
            stat, p = sp_stats.ttest_rel(a1, a2)
            test_name = "paired-t-test"
        else:
            variance_test = test_homogeneity(a1, a2, alpha=alpha)
            equal_var = variance_test.get("is_homogeneous", False)
            if equal_var:
                stat, p = sp_stats.ttest_ind(a1, a2, equal_var=True)
                test_name = "independent-t-test"
            else:
                stat, p = sp_stats.ttest_ind(a1, a2, equal_var=False)
                test_name = "welch-t-test"
    else:
        if paired:
            stat, p = sp_stats.wilcoxon(a1, a2)
            test_name = "wilcoxon-signed-rank"
        else:
            stat, p = sp_stats.mannwhitneyu(a1, a2, alternative="two-sided")
            test_name = "mann-whitney-u"

    effect = compute_cohens_d(a1, a2)
    hedges = compute_hedges_g(a1, a2)

    result = {
        "test_used": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < alpha),
        "alpha": alpha,
        "normality_g1": norm1,
        "normality_g2": norm2,
        "effect_size": effect,
        "effect_size_hedges_g": hedges,
    }
    if variance_test is not None:
        result["variance_homogeneity"] = variance_test
    return result


# ============================================================================
# Multi-group comparison
# ============================================================================


def compare_multiple_groups(
    groups: list[list[float] | np.ndarray],
    alpha: float = 0.05,
) -> dict:
    """Compare 3+ groups, auto-selecting ANOVA or Kruskal-Wallis.

    Returns:
        {"test_used", "statistic", "p_value", "significant", "alpha",
         "all_normal", "variance_homogeneity", "effect_size",
         "effect_size_omega_squared"}
    """
    arrays = [np.array(g, dtype=float) for g in groups]
    arrays = [a[~np.isnan(a)] for a in arrays]

    normality = [test_normality(a, alpha) for a in arrays]
    all_normal = all(n.get("is_normal", False) for n in normality)
    variance_test = test_homogeneity(*arrays, alpha=alpha)

    if all_normal:
        stat, p = sp_stats.f_oneway(*arrays)
        test_name = "one-way-anova"
    else:
        stat, p = sp_stats.kruskal(*arrays)
        test_name = "kruskal-wallis"

    effect = compute_eta_squared(arrays)
    omega = compute_omega_squared(arrays)

    return {
        "test_used": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < alpha),
        "alpha": alpha,
        "all_normal": bool(all_normal),
        "variance_homogeneity": variance_test,
        "effect_size": effect,
        "effect_size_omega_squared": omega,
    }


# ============================================================================
# Effect sizes
# ============================================================================


def compute_cohens_d(
    g1: list[float] | np.ndarray,
    g2: list[float] | np.ndarray,
) -> dict:
    """Cohen's d effect size for two groups.

    Uses pooled standard deviation.

    Returns:
        {"d", "magnitude"} where magnitude is "negligible"/"small"/"medium"/"large".
    """
    a1 = np.array(g1, dtype=float)
    a2 = np.array(g2, dtype=float)
    a1 = a1[~np.isnan(a1)]
    a2 = a2[~np.isnan(a2)]
    n1, n2 = len(a1), len(a2)
    if n1 < 2 or n2 < 2:
        return {"d": None, "magnitude": None}

    pooled_std = np.sqrt(
        ((n1 - 1) * a1.std(ddof=1) ** 2 + (n2 - 1) * a2.std(ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return {"d": 0.0, "magnitude": "negligible"}

    d = float(abs(a1.mean() - a2.mean()) / pooled_std)
    if d < 0.2:
        mag = "negligible"
    elif d < 0.5:
        mag = "small"
    elif d < 0.8:
        mag = "medium"
    else:
        mag = "large"
    return {"d": d, "magnitude": mag}


def compute_hedges_g(
    g1: list[float] | np.ndarray,
    g2: list[float] | np.ndarray,
) -> dict:
    """Hedges' g: bias-corrected Cohen's d for small samples.

    Uses correction factor J = 1 - 3 / (4*(n1+n2) - 9).

    Returns:
        {"g", "magnitude"} where magnitude is "negligible"/"small"/"medium"/"large".
    """
    d_result = compute_cohens_d(g1, g2)
    if d_result["d"] is None:
        return {"g": None, "magnitude": None}
    a1 = np.array(g1, dtype=float)
    a2 = np.array(g2, dtype=float)
    a1 = a1[~np.isnan(a1)]
    a2 = a2[~np.isnan(a2)]
    n1, n2 = len(a1), len(a2)
    df = n1 + n2 - 2
    if df < 1:
        return {"g": None, "magnitude": None}
    j = 1 - 3 / (4 * df - 1)
    g = d_result["d"] * j
    if g < 0.2:
        mag = "negligible"
    elif g < 0.5:
        mag = "small"
    elif g < 0.8:
        mag = "medium"
    else:
        mag = "large"
    return {"g": float(g), "magnitude": mag}


def compute_eta_squared(
    groups: list[list[float] | np.ndarray],
) -> dict:
    """Eta-squared effect size for multiple groups.

    Returns:
        {"eta_squared", "magnitude"} where magnitude is "small"/"medium"/"large".
    """
    arrays = [np.array(g, dtype=float) for g in groups]
    arrays = [a[~np.isnan(a)] for a in arrays]
    all_vals = np.concatenate(arrays)
    if len(all_vals) < 2:
        return {"eta_squared": None, "magnitude": None}

    grand_mean = all_vals.mean()
    ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
    ss_total = np.sum((all_vals - grand_mean) ** 2)

    if ss_total == 0:
        return {"eta_squared": 0.0, "magnitude": "negligible"}

    eta_sq = float(ss_between / ss_total)
    if eta_sq < 0.01:
        mag = "negligible"
    elif eta_sq < 0.06:
        mag = "small"
    elif eta_sq < 0.14:
        mag = "medium"
    else:
        mag = "large"
    return {"eta_squared": eta_sq, "magnitude": mag}


def compute_omega_squared(
    groups: list[list[float] | np.ndarray],
) -> dict:
    """Omega-squared: less biased alternative to eta-squared for ANOVA.

    ω² = (SS_between - df_between * MS_within) / (SS_total + MS_within)

    Returns:
        {"omega_squared", "magnitude"}
    """
    arrays = [np.array(g, dtype=float) for g in groups]
    arrays = [a[~np.isnan(a)] for a in arrays]
    all_vals = np.concatenate(arrays)
    n_total = len(all_vals)
    k = len(arrays)
    if n_total < 2 or k < 2:
        return {"omega_squared": None, "magnitude": None}

    grand_mean = all_vals.mean()
    ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
    ss_within = sum(np.sum((a - a.mean()) ** 2) for a in arrays)
    ss_total = np.sum((all_vals - grand_mean) ** 2)

    df_between = k - 1
    df_within = n_total - k
    ms_within = ss_within / df_within if df_within > 0 else 0

    denom = ss_total + ms_within
    if denom == 0:
        return {"omega_squared": 0.0, "magnitude": "negligible"}

    omega_sq = float((ss_between - df_between * ms_within) / denom)
    omega_sq = max(0.0, omega_sq)  # floor at 0

    if omega_sq < 0.01:
        mag = "negligible"
    elif omega_sq < 0.06:
        mag = "small"
    elif omega_sq < 0.14:
        mag = "medium"
    else:
        mag = "large"
    return {"omega_squared": omega_sq, "magnitude": mag}


# ============================================================================
# Outlier + leave-one-out diagnostics（确定性，下沉自 data-analyst prompt）
# ============================================================================
#
# spec 2026-06-17-data-analyst-loo-counterfactual-pushdown：识别离群 subject +
# 算 leave-one-out 反事实是纯确定性数值计算，曾在 data-analyst 自然语言推理里手算
# （反复验算耗尽预算）。这里把它做成纯函数，由 compare_groups 产出、落盘进
# statistics.json，data-analyst 只读不算。判据与 prompt step 2.7 b 字面一致：
# 偏离组均值 ≥ 1.5 SD，或偏离组中位数 ≥ 2×（取并集）。

_OUTLIER_SD_THRESHOLD = 1.5
_OUTLIER_MEDIAN_RATIO_THRESHOLD = 2.0


def _nonnan_indices(values: list[float]) -> list[int]:
    """返回非 NaN 值在原列表里的下标，保持与去 NaN 后数组的顺序对齐。"""
    return [i for i, v in enumerate(values) if not (v != v)]


def compute_outlier_diagnostics(
    group_values: dict[str, list[float]],
    subject_names: dict[str, list[str]] | None = None,
    sd_threshold: float = _OUTLIER_SD_THRESHOLD,
    median_ratio_threshold: float = _OUTLIER_MEDIAN_RATIO_THRESHOLD,
) -> list[dict]:
    """对每组识别离群 subject 并预算 leave-one-out 反事实。纯确定性，无 LLM。

    判据（与 data-analyst prompt step 2.7 b 完全一致，取并集）：
      - ``|value - group_mean| / group_std >= sd_threshold``（默认 1.5），或
      - ``max(value/median, median/value) >= median_ratio_threshold``（默认 2×），
        方向独立判，极小值与极大值相对中位数的偏离都覆盖。

    Args:
        group_values: ``{group_name: [float, ...]}``，每组某 metric 的 per-subject
            标量值（与 ``group_summary[grp][metric]["values"]`` 同结构）。
        subject_names: 可选 ``{group_name: [subject_id, ...]}``，顺序须与
            ``group_values`` 里该组的值顺序一致。调用方有真名时透传，data-analyst
            引用更可读；缺失则用组内 index 兜底（``subject #i``）。subject 真名映射
            属 Issue #98 列对齐家族另一轴，本函数不负责重建。
        sd_threshold / median_ratio_threshold: 离群判据阈值，默认与 prompt 字面一致。

    Returns:
        每个离群 subject 一条 dict：
        ``{group, subject, value, deviation_sd, deviation_median_ratio,
           group_mean, group_std, group_median, loo_mean, loo_std,
           counterfactual}``。
        ``counterfactual`` 是预格式化串，如
        ``"control mean 0.2530 → 0.1285 (std 0.3356 → 0.0701) if subject #2 excluded"``，
        data-analyst 原样引用不重算。

    Notes:
        - 组内 <2 值时 SD/LOO 无意义，跳过该组（与 compare_groups 的 ``len>=2`` 门一致）。
        - ``std`` 用 ddof=1（与 dispatcher.compute_paradigm_metrics 一致）。
    """
    diagnostics: list[dict] = []
    names = subject_names or {}
    for grp_name, vals in group_values.items():
        arr = np.array(vals, dtype=float)
        nonnan_idx = _nonnan_indices(vals)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2:
            continue
        grp_mean = float(arr.mean())
        grp_std = float(arr.std(ddof=1))
        grp_median = float(np.median(arr))
        grp_subjects = names.get(grp_name) or [f"subject #{i}" for i in range(len(vals))]
        for i, (raw_idx, value) in enumerate(zip(nonnan_idx, arr.tolist())):
            deviation_sd = abs(value - grp_mean) / grp_std if grp_std > 0 else 0.0
            if grp_median != 0:
                ratio = max(value / grp_median, grp_median / value)
            elif value != 0:
                ratio = float("inf")
            else:
                ratio = 1.0
            is_outlier = deviation_sd >= sd_threshold or ratio >= median_ratio_threshold
            if not is_outlier:
                continue
            # leave-one-out：排除该 subject 后重算组 mean/std
            loo_arr = np.delete(arr, i)
            loo_mean = float(loo_arr.mean()) if len(loo_arr) > 0 else grp_mean
            loo_std = float(loo_arr.std(ddof=1)) if len(loo_arr) > 1 else 0.0
            subject = (
                grp_subjects[raw_idx]
                if raw_idx < len(grp_subjects)
                else f"subject #{raw_idx}"
            )
            counterfactual = (
                f"{grp_name} mean {grp_mean:.4f} → {loo_mean:.4f} "
                f"(std {grp_std:.4f} → {loo_std:.4f}) if {subject} excluded"
            )
            diagnostics.append(
                {
                    "group": grp_name,
                    "subject": subject,
                    "value": value,
                    "deviation_sd": float(deviation_sd),
                    "deviation_median_ratio": float(ratio),
                    "group_mean": grp_mean,
                    "group_std": grp_std,
                    "group_median": grp_median,
                    "loo_mean": loo_mean,
                    "loo_std": loo_std,
                    "counterfactual": counterfactual,
                }
            )
    return diagnostics


# ============================================================================
# High-level dispatcher
# ============================================================================


def compare_groups(
    metrics_result: dict,
    groups: list[str] | None = None,
    metrics_to_test: list[str] | None = None,
    alpha: float = 0.05,
    correction: str = "bonferroni",
) -> dict:
    """Run statistical comparisons across groups for all requested metrics.

    Args:
        metrics_result: Output of ``metrics.compute_paradigm_metrics()``.
        groups: Group names to compare. If None, uses all groups.
        metrics_to_test: Metric names to test. If None, tests all.
        alpha: Significance level.
        correction: Multiple comparison correction ("bonferroni" or "none").

    Returns:
        {
            "comparisons": {metric_name: [{"group1", "group2", "p_value", ...}]},
            "summary": str,
            "alpha": float,
            "correction": str,
        }

    This return format is compatible with ``charts.py``'s ``significance``
    parameter.
    """
    group_summary = metrics_result.get("group_summary", {})
    if groups is None:
        groups = list(group_summary.keys())

    if len(groups) < 2:
        return {
            "comparisons": {},
            "summary": "Need at least 2 groups for comparison.",
            "alpha": alpha,
            "correction": correction,
            "outlier_diagnostics": [],
        }

    # Collect all testable metric names
    all_metrics = set()
    for grp_name in groups:
        if grp_name in group_summary:
            all_metrics.update(group_summary[grp_name].keys())
    if metrics_to_test:
        all_metrics = all_metrics.intersection(metrics_to_test)

    # Count total tests for Bonferroni correction
    n_tests = len(all_metrics)
    adjusted_alpha = (
        alpha / n_tests if correction == "bonferroni" and n_tests > 0 else alpha
    )

    comparisons: dict[str, list[dict]] = {}
    sig_count = 0
    total_count = 0
    # 每个离群 subject 一条（spec 2026-06-17 LOO 下沉）：data-analyst 只读不算。
    # subject 真名 compare_groups 拿不到（dispatcher 落盘时丢了 group→subject 映射），
    # 纯函数内用组内 index 兜底；真名映射属 Issue #98 另一轴。
    outlier_diagnostics: list[dict] = []

    for metric_name in sorted(all_metrics):
        # Gather values per group
        group_values = {}
        for grp_name in groups:
            grp = group_summary.get(grp_name, {})
            if metric_name in grp:
                group_values[grp_name] = grp[metric_name].get("values", [])

        valid_groups = {k: v for k, v in group_values.items() if len(v) >= 2}
        if len(valid_groups) < 2:
            continue

        # 离群 + LOO 诊断：用全 group_values（纯函数内部对 <2 值组自行跳过），
        # 与 comparisons 的 valid_groups 门独立，但只在 ≥2 组参与比较时才有意义。
        metric_outliers = compute_outlier_diagnostics(group_values)
        for diag in metric_outliers:
            diag["metric"] = metric_name
            outlier_diagnostics.append(diag)

        grp_names = list(valid_groups.keys())
        grp_arrays = list(valid_groups.values())

        metric_comparisons = []

        if len(grp_names) == 2:
            result = compare_two_groups(
                grp_arrays[0], grp_arrays[1], alpha=adjusted_alpha
            )
            result["group1"] = grp_names[0]
            result["group2"] = grp_names[1]
            metric_comparisons.append(result)
            total_count += 1
            if result["significant"]:
                sig_count += 1
        else:
            # Omnibus test
            omnibus = compare_multiple_groups(grp_arrays, alpha=adjusted_alpha)
            omnibus["groups"] = grp_names
            metric_comparisons.append(omnibus)
            total_count += 1
            if omnibus["significant"]:
                sig_count += 1

            # Pairwise post-hoc if omnibus significant
            if omnibus["significant"]:
                from itertools import combinations

                is_parametric = omnibus["test_used"] == "one-way-anova"
                is_homogeneous = omnibus.get("variance_homogeneity", {}).get(
                    "is_homogeneous", False
                )

                if is_parametric and is_homogeneous:
                    # Try Tukey HSD (scipy >= 1.8)
                    try:
                        tukey = sp_stats.tukey_hsd(*grp_arrays)
                        for i, j in combinations(range(len(grp_names)), 2):
                            pw = {
                                "test_used": "tukey-hsd",
                                "statistic": float(tukey.statistic[i][j])
                                if hasattr(tukey, "statistic")
                                else None,
                                "p_value": float(tukey.pvalue[i][j]),
                                "significant": bool(
                                    tukey.pvalue[i][j] < adjusted_alpha
                                ),
                                "group1": grp_names[i],
                                "group2": grp_names[j],
                                "post_hoc": True,
                                "effect_size": compute_cohens_d(
                                    grp_arrays[i], grp_arrays[j]
                                ),
                                "effect_size_hedges_g": compute_hedges_g(
                                    grp_arrays[i], grp_arrays[j]
                                ),
                            }
                            metric_comparisons.append(pw)
                    except Exception:
                        # Fallback to pairwise comparisons
                        for i, j in combinations(range(len(grp_names)), 2):
                            pw = compare_two_groups(
                                grp_arrays[i],
                                grp_arrays[j],
                                alpha=adjusted_alpha,
                            )
                            pw["group1"] = grp_names[i]
                            pw["group2"] = grp_names[j]
                            pw["post_hoc"] = True
                            metric_comparisons.append(pw)
                else:
                    # Non-parametric or unequal variance: pairwise comparisons
                    for i, j in combinations(range(len(grp_names)), 2):
                        pw = compare_two_groups(
                            grp_arrays[i],
                            grp_arrays[j],
                            alpha=adjusted_alpha,
                        )
                        pw["group1"] = grp_names[i]
                        pw["group2"] = grp_names[j]
                        pw["post_hoc"] = True
                        metric_comparisons.append(pw)

        comparisons[metric_name] = metric_comparisons

    summary_parts = [
        f"Compared {len(groups)} groups across {len(comparisons)} metrics.",
        f"Correction: {correction} (adjusted α = {adjusted_alpha:.4f}).",
        f"Significant results: {sig_count}/{total_count}.",
    ]

    return {
        "comparisons": comparisons,
        "summary": " ".join(summary_parts),
        "alpha": alpha,
        "correction": correction,
        "outlier_diagnostics": outlier_diagnostics,
    }
