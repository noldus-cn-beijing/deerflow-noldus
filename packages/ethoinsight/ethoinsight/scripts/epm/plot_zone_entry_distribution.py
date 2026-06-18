"""EPM: 单样本各区域进入次数分布柱状图。

CLI:
  单文件:  python -m ethoinsight.scripts.epm.plot_zone_entry_distribution \
             --input <轨迹文件> --output <png>
  多文件:  python -m ethoinsight.scripts.epm.plot_zone_entry_distribution \
             --inputs <inputs.json> --output <png>

单样本场景:画 open / closed 进入次数对比 + 合计。多文件 inputs.json 时读 paths[0]。
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
from ethoinsight.scripts._cli import emit_result, make_plot_parser, parse_parameters, resolve_per_subject_input, select_zone_kwargs


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=False).parse_args(argv)
    try:
        path = resolve_per_subject_input(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    parameters = parse_parameters(args)  # 与 compute 脚本同源（spec 2026-06-18）
    df = parse_trajectory(path)
    # 两函数签名不同：compute_open_arm_entry_count 只接 open_arm_zones，
    # compute_total_entry_count 接 open_arm_zones + closed_arm_zones。按函数筛 key。
    open_entries = compute_open_arm_entry_count(df, **select_zone_kwargs(parameters, ["open_arm_zones"])) or 0
    total_entries = compute_total_entry_count(df, **select_zone_kwargs(parameters, ["open_arm_zones", "closed_arm_zones"])) or 0
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
