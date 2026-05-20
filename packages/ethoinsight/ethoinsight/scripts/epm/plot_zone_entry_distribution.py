"""EPM: 单样本各区域进入次数分布柱状图。

CLI: python -m ethoinsight.scripts.epm.plot_zone_entry_distribution \
       --input <轨迹文件> --output <png>

单样本场景:画 open / closed 进入次数对比 + 合计。
反映动物对各区域的探索分布。

输出: PNG 图像。
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from ethoinsight.metrics.epm import (
    compute_open_arm_entry_count,
    compute_total_entry_count,
)
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_plot_parser


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=False).parse_args(argv)
    if not args.input:
        print("error: plot_zone_entry_distribution requires --input (single subject)", file=sys.stderr)
        return 2

    df = parse_trajectory(args.input)
    open_entries = compute_open_arm_entry_count(df) or 0
    total_entries = compute_total_entry_count(df) or 0
    closed_entries = max(total_entries - open_entries, 0)

    labels = ["Open arms", "Closed arms"]
    values = [open_entries, closed_entries]
    colors = ["#4C9F70", "#D97757"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5, width=0.55)
    ax.set_ylabel("Entry count")
    ax.set_title("Zone entry distribution")
    ymax = max(values) if max(values) > 0 else 1
    ax.set_ylim(0, ymax * 1.2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    emit_result(
        {
            "plot": "zone_entry_distribution",
            "path": args.output,
            "open_entries": open_entries,
            "closed_entries": closed_entries,
            "total_entries": total_entries,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
