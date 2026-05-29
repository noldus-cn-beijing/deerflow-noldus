"""EPM: 头部朝向统计 (head direction circular stats)。

CLI: python -m ethoinsight.scripts.epm.compute_head_direction_stats \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "head_direction_stats", "value": {mean_rad/circular_stdev_rad/resultant_length/n} or null}
"""

from __future__ import annotations

import sys

from ethoinsight.metrics._common import compute_head_direction_stats
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_compute_parser, parse_parameters, save_output_json

METRIC_NAME = "head_direction_stats"


def main(argv: list[str] | None = None) -> int:
    parser = make_compute_parser(description=__doc__)
    args = parser.parse_args(argv)
    df = parse_trajectory(args.input)
    parameters = parse_parameters(args)
    value = compute_head_direction_stats(df, **parameters)
    payload = {"metric": METRIC_NAME, "value": value, "parameters_used": parameters}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
