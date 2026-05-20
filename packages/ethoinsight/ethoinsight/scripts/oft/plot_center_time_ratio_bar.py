"""OFT: 单样本中心区时间占比柱状图。

CLI: python -m ethoinsight.scripts.oft.plot_center_time_ratio_bar \
       --input <轨迹文件> --output <png>

输出: PNG 图像。
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from ethoinsight.metrics.oft import compute_center_time_ratio
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_plot_parser


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=False).parse_args(argv)
    if not args.input:
        print("error: plot_center_time_ratio_bar requires --input (single subject)", file=sys.stderr)
        return 2

    df = parse_trajectory(args.input)
    subject = df.attrs.get("subject", "Subject")
    value = compute_center_time_ratio(df)
    if value is None:
        print("error: could not compute center_time_ratio — no center zone columns", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([subject], [value], color="#4C9F70", edgecolor="black", linewidth=0.5, width=0.5)
    ax.set_ylabel("Center time ratio")
    ax.set_ylim(0, max(1.0, value * 1.15))
    ax.set_title("Center time ratio")
    ax.text(0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    emit_result({"plot": "center_time_ratio_bar", "path": args.output, "value": value})
    return 0


if __name__ == "__main__":
    sys.exit(main())
