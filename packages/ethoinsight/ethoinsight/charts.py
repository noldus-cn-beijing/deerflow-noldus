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
import seaborn as sns  # noqa: E402

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Colorblind-friendly palette (Okabe-Ito)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]

_DEFAULT_OUTPUT_DIR = "/tmp/ethoinsight"


def _setup_style() -> None:
    plt.rcParams.update(
        {
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
        }
    )


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
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
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
        bars = ax.bar(
            x,
            means,
            yerr=errors,
            capsize=4,
            width=0.6,
            color=[PALETTE[i % len(PALETTE)] for i in range(len(groups))],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
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
        ax.text(
            0.5,
            0.5,
            "No position data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        path = _resolve_output_path(output_path, "trajectory")
        fig.savefig(path)
        plt.close(fig)
        return path

    if color_by in df.columns:
        groups = df[color_by].unique()
        for i, grp in enumerate(groups):
            sub = df[df[color_by] == grp]
            ax.plot(
                sub["x_center"],
                sub["y_center"],
                alpha=0.5,
                linewidth=0.5,
                color=PALETTE[i % len(PALETTE)],
                label=str(grp),
            )
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.plot(
            df["x_center"], df["y_center"], alpha=0.5, linewidth=0.5, color=PALETTE[0]
        )

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
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        path = _resolve_output_path(output_path, "timeseries")
        fig.savefig(path)
        plt.close(fig)
        return path

    ax.plot(
        timeseries_df[x_col],
        timeseries_df[y_col],
        linewidth=0.8,
        color=PALETTE[0],
        alpha=0.8,
    )
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
        ax.plot(
            [x1, x1, x2, x2], [y - step * 0.2, y, y, y - step * 0.2], lw=0.8, c="black"
        )
        ax.text((x1 + x2) / 2, y, text, ha="center", va="bottom", fontsize=9)

    # Extend y-axis to fit markers
    if comparisons:
        ax.set_ylim(top=y_max + step * (len(comparisons) + 1.5))


def raincloud_plot(
    metrics: dict,
    metrics_to_plot: list[str],
    significance: dict | None = None,
    output_path: str | None = None,
) -> str:
    """Raincloud plot — half-violin + box + jittered scatter.

    A publication-quality composite plot that shows distribution shape,
    summary statistics, and individual data points simultaneously.
    """
    _setup_style()
    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(4.5 * max(n, 1), 5), squeeze=False)
    axes = axes.flatten()

    for idx, mname in enumerate(metrics_to_plot):
        ax = axes[idx]
        groups, values = _extract_group_data(metrics, mname)
        if not groups or all(len(v) == 0 for v in values):
            ax.set_title(mname)
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        positions = np.arange(len(groups))
        for i, (grp, vals) in enumerate(zip(groups, values)):
            if not vals:
                continue
            vals_arr = np.array(vals, dtype=float)
            color = PALETTE[i % len(PALETTE)]

            # Half-violin (right side only)
            parts = ax.violinplot(
                [vals_arr],
                positions=[positions[i]],
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=0.6,
            )
            for body in parts.get("bodies", []):
                m_path = body.get_paths()[0]
                m_path.vertices[:, 0] = np.clip(
                    m_path.vertices[:, 0], positions[i], positions[i] + 0.4
                )
                body.set_facecolor(color)
                body.set_alpha(0.4)

            # Box plot (narrow, center)
            bp = ax.boxplot(
                [vals_arr],
                positions=[positions[i]],
                widths=0.12,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=color, alpha=0.8),
                medianprops=dict(color="white", linewidth=1.5),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
            )

            # Jittered scatter (left side)
            jitter = np.random.default_rng(42).uniform(-0.15, -0.02, size=len(vals_arr))
            ax.scatter(
                positions[i] + jitter,
                vals_arr,
                s=15,
                color=color,
                alpha=0.7,
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(groups)
        ax.set_title(mname.replace("_", " ").title())
        ax.set_ylabel(mname)

        if significance:
            _add_significance_from_stats(ax, groups, mname, significance)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    path = _resolve_output_path(output_path, "raincloud_plot")
    fig.savefig(path)
    plt.close(fig)
    return path


def beeswarm_plot(
    metrics: dict,
    metrics_to_plot: list[str],
    significance: dict | None = None,
    output_path: str | None = None,
) -> str:
    """Beeswarm plot — individual data points with mean ± SEM overlay.

    Each data point is shown as a non-overlapping dot, ideal for small
    sample sizes typical in animal behavior experiments (n=5-15).
    """
    _setup_style()
    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(4 * max(n, 1), 5), squeeze=False)
    axes = axes.flatten()

    for idx, mname in enumerate(metrics_to_plot):
        ax = axes[idx]
        groups, values = _extract_group_data(metrics, mname)
        if not groups or all(len(v) == 0 for v in values):
            ax.set_title(mname)
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        # Build DataFrame for seaborn
        rows = []
        for grp, vals in zip(groups, values):
            for v in vals:
                rows.append({"group": grp, "value": v})
        df = pd.DataFrame(rows)

        # Swarm plot
        sns.swarmplot(
            data=df,
            x="group",
            y="value",
            ax=ax,
            size=6,
            palette=PALETTE[: len(groups)],
            alpha=0.8,
            zorder=3,
        )

        # Mean ± SEM overlay
        for i, (grp, vals) in enumerate(zip(groups, values)):
            if not vals:
                continue
            vals_arr = np.array(vals, dtype=float)
            mean = np.mean(vals_arr)
            sem = (
                np.std(vals_arr, ddof=1) / np.sqrt(len(vals_arr))
                if len(vals_arr) > 1
                else 0
            )
            ax.hlines(mean, i - 0.25, i + 0.25, color="black", linewidth=1.5, zorder=4)
            ax.vlines(i, mean - sem, mean + sem, color="black", linewidth=1.2, zorder=4)

        ax.set_xlabel("")
        ax.set_title(mname.replace("_", " ").title())
        ax.set_ylabel(mname)

        if significance:
            _add_significance_from_stats(ax, groups, mname, significance)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    path = _resolve_output_path(output_path, "beeswarm_plot")
    fig.savefig(path)
    plt.close(fig)
    return path


def correlogram(
    metrics: dict,
    metrics_to_plot: list[str] | None = None,
    output_path: str | None = None,
) -> str:
    """Correlation matrix heatmap across behavioral metrics.

    Computes Pearson correlations between metrics using per-subject values
    across all groups. Upper triangle is masked to avoid redundancy.
    """
    _setup_style()

    # Build subject-level DataFrame from all groups
    group_summary = metrics.get("group_summary", {})
    all_metrics: dict[str, list[float]] = {}
    for grp_metrics in group_summary.values():
        for mname, minfo in grp_metrics.items():
            vals = minfo.get("values", [])
            if mname not in all_metrics:
                all_metrics[mname] = []
            all_metrics[mname].extend(vals)

    if metrics_to_plot:
        all_metrics = {k: v for k, v in all_metrics.items() if k in metrics_to_plot}

    if len(all_metrics) < 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(
            0.5,
            0.5,
            "Need ≥2 metrics for correlogram",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        path = _resolve_output_path(output_path, "correlogram")
        fig.savefig(path)
        plt.close(fig)
        return path

    # Align lengths (use min length across all metrics)
    min_len = min(len(v) for v in all_metrics.values())
    df = pd.DataFrame({k: v[:min_len] for k, v in all_metrics.items()})
    corr = df.corr()

    # Upper triangle mask
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.8), max(5, len(corr) * 0.7)))
    cmap = plt.cm.RdBu_r

    im = ax.imshow(
        corr.where(~mask, np.nan).values, cmap=cmap, vmin=-1, vmax=1, aspect="auto"
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    # Labels
    labels = [m.replace("_", " ") for m in corr.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells with values
    for i in range(len(corr)):
        for j in range(len(corr)):
            if mask[i, j]:
                continue
            val = corr.iloc[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color
            )

    ax.set_title("Metric Correlation Matrix")
    fig.tight_layout()
    path = _resolve_output_path(output_path, "correlogram")
    fig.savefig(path)
    plt.close(fig)
    return path


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
            markers.append(
                {
                    "group1": groups.index(g1),
                    "group2": groups.index(g2),
                    "p_value": comp.get("p_value", 1.0),
                }
            )
    if markers:
        add_significance_markers(ax, markers)
