"""Zero Maze: 开放区指标均值±SEM 柱状图（n<3 自动隐藏误差线）。

CLI: python -m ethoinsight.scripts.zero_maze.plot_bar_open_zone \
       --inputs <inputs.json> --groups <groups.json> --output <png>
"""

from __future__ import annotations

import sys

from ethoinsight.charts import bar_chart
from ethoinsight.metrics.dispatcher import compute_paradigm_metrics
from ethoinsight.parse import parse_batch
from ethoinsight.scripts._cli import emit_result, make_plot_parser, read_groups_json, read_inputs_json

METRICS_TO_PLOT = ["open_zone_time_ratio", "open_zone_distance", "hesitation_count"]


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=True).parse_args(argv)
    if not args.inputs:
        print("error: plot_bar_open_zone requires --inputs (multi-file)", file=sys.stderr)
        return 2
    paths = read_inputs_json(args.inputs)
    groups = read_groups_json(args.groups) if args.groups else None
    parsed = parse_batch(paths)
    metrics = compute_paradigm_metrics(parsed, paradigm="zero_maze", groups=groups)
    output_path = bar_chart(metrics, metrics_to_plot=METRICS_TO_PLOT, output_path=args.output)
    emit_result({"plot": "bar_open_zone", "path": output_path, "metrics": METRICS_TO_PLOT})
    return 0


if __name__ == "__main__":
    sys.exit(main())
