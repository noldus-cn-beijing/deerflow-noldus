"""LDB: 亮室 vs 暗室进入次数分布柱状图（单样本 per-subject）。

CLI:
  单文件:  python -m ethoinsight.scripts.ldb.plot_zone_entry_distribution \\
             --input <轨迹文件> --output <png>
  多文件:  python -m ethoinsight.scripts.ldb.plot_zone_entry_distribution \\
             --inputs <inputs.json> --output <png>

单样本场景: 画 light / dark 进入次数对比 + 合计。多文件 inputs.json 时读 paths[0]。
反映动物对明暗两室的探索分布。

输出: PNG 图像。
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from ethoinsight.metrics.ldb import compute_light_entry_count, compute_transition_count
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
    # 两函数签名不同：compute_light_entry_count 只接 light_zone，
    # compute_transition_count 接 light_zone + dark_zone。按函数筛 key。
    light_entries = compute_light_entry_count(df, **select_zone_kwargs(parameters, ["light_zone"])) or 0
    total_transitions = compute_transition_count(df, **select_zone_kwargs(parameters, ["light_zone", "dark_zone"])) or 0
    dark_entries = max(total_transitions - light_entries, 0)

    labels = ["Light zone", "Dark zone"]
    values = [light_entries, dark_entries]
    colors = ["#F5C542", "#4A4A4A"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5, width=0.55)
    ax.set_ylabel("Entry count")
    ax.set_title("Zone entry distribution (Light-Dark Box)")
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
            "light_entries": light_entries,
            "dark_entries": dark_entries,
            "total_transitions": total_transitions,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
