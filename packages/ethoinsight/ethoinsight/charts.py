"""ethoinsight.charts — publication-quality chart generation.

All functions save figures as PNG and return the file path.
Uses matplotlib Agg backend for headless environments.
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Colorblind-friendly palette (Okabe-Ito)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]

_DEFAULT_OUTPUT_DIR = "/tmp/ethoinsight"


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _resolve_output_path(output_path: str | None, prefix: str) -> str:
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        return output_path
    os.makedirs(_DEFAULT_OUTPUT_DIR, exist_ok=True)
    ts = int(time.time() * 1000)
    return os.path.join(_DEFAULT_OUTPUT_DIR, f"{prefix}_{ts}.png")


def _extract_group_data(
    metrics: dict,
    metric_name: str,
) -> tuple[list[str], list[list[float]]]:
    """Extract per-group values for a metric from metrics result dict."""
    groups = []
    values = []
    for grp_name, grp_metrics in metrics.get("group_summary", {}).items():
        if metric_name in grp_metrics:
            groups.append(grp_name)
            values.append(grp_metrics[metric_name].get("values", []))
    return groups, values


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------


def box_plot(
    metrics: dict,
    metrics_to_plot: list[str],
    significance: dict | None = None,
    output_path: str | None = None,
) -> str:
    """Box plot comparing groups for each metric.

    Args:
        metrics: Output of ``compute_paradigm_metrics()``.
        metrics_to_plot: List of metric names.
        significance: Optional statistics result with comparison p-values.
        output_path: Save path. Auto-generated if None.

    Returns:
        Path to saved PNG.
    """
    _setup_style()
    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(4 * max(n, 1), 5), squeeze=False)
    axes = axes.flatten()

    for idx, mname in enumerate(metrics_to_plot):
        ax = axes[idx]
        groups, values = _extract_group_data(metrics, mname)
        if not groups:
            ax.set_title(mname)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        bp = ax.boxplot(values, tick_labels=groups, patch_artist=True, widths=0.6)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PALETTE[i % len(PALETTE)])
            patch.set_alpha(0.7)
        ax.set_title(mname.replace("_", " ").title())
        ax.set_ylabel(mname)

        if significance:
            _add_significance_from_stats(ax, groups, mname, significance)

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    path = _resolve_output_path(output_path, "box_plot")
    fig.savefig(path)
    plt.close(fig)
    return path


def bar_chart(
    metrics: dict,
    metrics_to_plot: list[str],
    error_type: str = "sem",
    significance: dict | None = None,
    output_path: str | None = None,
) -> str:
    """Bar chart with error bars comparing groups.

    Args:
        error_type: "sem" (standard error) or "std" (standard deviation).
    """
    _setup_style()
    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(4 * max(n, 1), 5), squeeze=False)
    axes = axes.flatten()

    for idx, mname in enumerate(metrics_to_plot):
        ax = axes[idx]
        group_summary = metrics.get("group_summary", {})
        groups = []
        means = []
        errors = []
        for grp_name, grp_metrics in group_summary.items():
            if mname in grp_metrics:
                groups.append(grp_name)
                info = grp_metrics[mname]
                means.append(info["mean"])
                std = info.get("std", 0)
                nn = info.get("n", 1)
                if error_type == "sem" and nn > 0:
                    errors.append(std / np.sqrt(nn))
                else:
                    errors.append(std)

        if not groups:
            ax.set_title(mname)
            continue

        x = np.arange(len(groups))
        bars = ax.bar(x, means, yerr=errors, capsize=4, width=0.6,
                       color=[PALETTE[i % len(PALETTE)] for i in range(len(groups))],
                       alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_title(mname.replace("_", " ").title())
        ax.set_ylabel(mname)

        if significance:
            _add_significance_from_stats(ax, groups, mname, significance)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    path = _resolve_output_path(output_path, "bar_chart")
    fig.savefig(path)
    plt.close(fig)
    return path


def violin_plot(
    metrics: dict,
    metrics_to_plot: list[str],
    significance: dict | None = None,
    output_path: str | None = None,
) -> str:
    """Violin plot comparing groups for each metric."""
    _setup_style()
    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(4 * max(n, 1), 5), squeeze=False)
    axes = axes.flatten()

    for idx, mname in enumerate(metrics_to_plot):
        ax = axes[idx]
        groups, values = _extract_group_data(metrics, mname)
        if not groups or all(len(v) == 0 for v in values):
            ax.set_title(mname)
            continue
        # Filter out empty groups
        valid = [(g, v) for g, v in zip(groups, values) if len(v) > 0]
        if not valid:
            continue
        groups_v, values_v = zip(*valid)
        parts = ax.violinplot(values_v, showmeans=True, showmedians=True)
        for i, body in enumerate(parts.get("bodies", [])):
            body.set_facecolor(PALETTE[i % len(PALETTE)])
            body.set_alpha(0.7)
        ax.set_xticks(range(1, len(groups_v) + 1))
        ax.set_xticklabels(groups_v)
        ax.set_title(mname.replace("_", " ").title())
        ax.set_ylabel(mname)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    path = _resolve_output_path(output_path, "violin_plot")
    fig.savefig(path)
    plt.close(fig)
    return path


def trajectory_plot(
    df: pd.DataFrame,
    color_by: str = "subject",
    output_path: str | None = None,
) -> str:
    """Plot X/Y trajectories from parsed data.

    Args:
        df: DataFrame with x_center, y_center columns. May have a
            ``subject`` column for coloring.
        color_by: Column name to color trajectories by.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    if "x_center" not in df.columns or "y_center" not in df.columns:
        ax.text(0.5, 0.5, "No position data", ha="center", va="center",
                transform=ax.transAxes)
        path = _resolve_output_path(output_path, "trajectory")
        fig.savefig(path)
        plt.close(fig)
        return path

    if color_by in df.columns:
        groups = df[color_by].unique()
        for i, grp in enumerate(groups):
            sub = df[df[color_by] == grp]
            ax.plot(sub["x_center"], sub["y_center"],
                    alpha=0.5, linewidth=0.5,
                    color=PALETTE[i % len(PALETTE)], label=str(grp))
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.plot(df["x_center"], df["y_center"], alpha=0.5, linewidth=0.5,
                color=PALETTE[0])

    ax.set_xlabel("X (position)")
    ax.set_ylabel("Y (position)")
    ax.set_title("Trajectory Plot")
    ax.set_aspect("equal")
    fig.tight_layout()

    path = _resolve_output_path(output_path, "trajectory")
    fig.savefig(path)
    plt.close(fig)
    return path


def timeseries_plot(
    timeseries_df: pd.DataFrame,
    y_col: str,
    x_col: str = "trial_time",
    output_path: str | None = None,
) -> str:
    """Line plot of a timeseries metric (e.g. IID, polarity over time)."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    if x_col not in timeseries_df.columns or y_col not in timeseries_df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        path = _resolve_output_path(output_path, "timeseries")
        fig.savefig(path)
        plt.close(fig)
        return path

    ax.plot(timeseries_df[x_col], timeseries_df[y_col],
            linewidth=0.8, color=PALETTE[0], alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(y_col.replace("_", " ").title())
    fig.tight_layout()

    path = _resolve_output_path(output_path, "timeseries")
    fig.savefig(path)
    plt.close(fig)
    return path


def add_significance_markers(ax: plt.Axes, comparisons: list[dict]) -> None:
    """Add significance bracket markers to an axes.

    Each comparison dict should have:
        group1: int (x position), group2: int, p_value: float, text: str
    """
    y_max = ax.get_ylim()[1]
    step = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08

    for i, comp in enumerate(comparisons):
        x1 = comp.get("group1", 0)
        x2 = comp.get("group2", 1)
        p = comp.get("p_value", 1.0)
        text = comp.get("text", _p_to_stars(p))

        y = y_max + step * (i + 1)
        ax.plot([x1, x1, x2, x2], [y - step * 0.2, y, y, y - step * 0.2],
                lw=0.8, c="black")
        ax.text((x1 + x2) / 2, y, text, ha="center", va="bottom", fontsize=9)

    # Extend y-axis to fit markers
    if comparisons:
        ax.set_ylim(top=y_max + step * (len(comparisons) + 1.5))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _add_significance_from_stats(
    ax: plt.Axes,
    groups: list[str],
    metric_name: str,
    significance: dict,
) -> None:
    """Extract comparisons from a statistics result dict and add markers."""
    comparisons_data = significance.get("comparisons", {})
    if metric_name not in comparisons_data:
        return
    comps = comparisons_data[metric_name]
    markers = []
    for comp in comps:
        g1 = comp.get("group1", "")
        g2 = comp.get("group2", "")
        if g1 in groups and g2 in groups:
            markers.append({
                "group1": groups.index(g1),
                "group2": groups.index(g2),
                "p_value": comp.get("p_value", 1.0),
            })
    if markers:
        add_significance_markers(ax, markers)
