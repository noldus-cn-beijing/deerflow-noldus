"""EPM: 单样本开臂时间占比柱状图。

CLI:
  单文件:  python -m ethoinsight.scripts.epm.plot_open_arm_time_ratio_bar \
             --input <轨迹文件> --output <png>
  多文件:  python -m ethoinsight.scripts.epm.plot_open_arm_time_ratio_bar \
             --inputs <inputs.json> --output <png>

单样本场景:画当前 subject 的 open_arm_time_ratio 柱状条。多文件 inputs.json 时
读 paths[0]。组间场景请改用 plot_box_open_arm。

输出: PNG 图像。
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from ethoinsight.metrics.epm import compute_open_arm_time_ratio
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
    subject = df.attrs.get("subject", "Subject")
    value = compute_open_arm_time_ratio(df)
    if value is None:
        print("error: could not compute open_arm_time_ratio — no open-arm zone columns", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([subject], [value], color="#4C9F70", edgecolor="black", linewidth=0.5, width=0.5)
    ax.set_ylabel("Open arm time ratio")
    ax.set_ylim(0, max(1.0, value * 1.15))
    ax.set_title("Open arm time ratio")
    ax.text(0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    emit_result({"plot": "open_arm_time_ratio_bar", "path": args.output, "value": value})
    return 0


if __name__ == "__main__":
    sys.exit(main())
