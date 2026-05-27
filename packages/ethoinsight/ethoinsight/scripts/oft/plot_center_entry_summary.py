"""OFT: 单样本中心区进入次数与累计时间摘要图。

CLI:
  单文件:  python -m ethoinsight.scripts.oft.plot_center_entry_summary \
             --input <轨迹文件> --output <png>
  多文件:  python -m ethoinsight.scripts.oft.plot_center_entry_summary \
             --inputs <inputs.json> --output <png>

per-subject plot: reads ``paths[0]`` when given an inputs.json.
并排两个柱:进入次数 + 累计停留时间(秒),用双纵轴显示。

输出: PNG 图像。
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from ethoinsight.metrics.oft import compute_center_entry_count, compute_center_time
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_plot_parser, resolve_per_subject_input


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=False).parse_args(argv)
    try:
        path = resolve_per_subject_input(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    df = parse_trajectory(path)
    entries = compute_center_entry_count(df) or 0
    seconds = compute_center_time(df) or 0.0

    fig, ax_left = plt.subplots(figsize=(5, 4))
    ax_left.bar(["Entry count"], [entries], color="#4C9F70", edgecolor="black", linewidth=0.5, width=0.5)
    ax_left.set_ylabel("Entry count")
    ax_left.text(0, entries, str(entries), ha="center", va="bottom", fontsize=10)
    ax_left.set_ylim(0, max(entries * 1.2, 1))

    ax_right = ax_left.twinx()
    ax_right.bar(["Time (s)"], [seconds], color="#D97757", edgecolor="black", linewidth=0.5, width=0.5)
    ax_right.set_ylabel("Center time (s)")
    ax_right.text(1, seconds, f"{seconds:.2f}", ha="center", va="bottom", fontsize=10)
    ax_right.set_ylim(0, max(seconds * 1.2, 1.0))

    ax_left.set_title("Center entry summary")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    emit_result(
        {
            "plot": "center_entry_summary",
            "path": args.output,
            "entry_count": entries,
            "center_time_seconds": seconds,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
