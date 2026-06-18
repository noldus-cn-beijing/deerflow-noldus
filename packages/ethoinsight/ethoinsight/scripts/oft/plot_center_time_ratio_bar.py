"""OFT: 单样本中心区时间占比柱状图。

CLI:
  单文件:  python -m ethoinsight.scripts.oft.plot_center_time_ratio_bar \
             --input <轨迹文件> --output <png>
  多文件:  python -m ethoinsight.scripts.oft.plot_center_time_ratio_bar \
             --inputs <inputs.json> --output <png>

per-subject plot: reads ``paths[0]`` when given an inputs.json.

输出: PNG 图像。
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from ethoinsight.metrics.oft import compute_center_time_ratio
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_plot_parser, parse_parameters, resolve_per_subject_input


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=False).parse_args(argv)
    try:
        path = resolve_per_subject_input(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    parameters = parse_parameters(args)  # 与 compute 脚本同源（spec 2026-06-18）
    df = parse_trajectory(path)
    subject = df.attrs.get("subject", "Subject")
    value = compute_center_time_ratio(df, **parameters)
    if value is None:
        hint = (
            "（已传 zone 参数仍无值，疑似该 subject 数据无中心区列）"
            if parameters
            else "（未收到 zone 对齐参数，疑似 prep_chart_plan 未注入 --parameters-json）"
        )
        print(f"error: could not compute center_time_ratio — no center zone columns {hint}", file=sys.stderr)
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
