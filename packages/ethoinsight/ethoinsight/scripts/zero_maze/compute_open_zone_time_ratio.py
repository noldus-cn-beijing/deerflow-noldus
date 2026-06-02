"""Zero Maze: 开放区时间占比 (open zone time ratio).

CLI: python -m ethoinsight.scripts.zero_maze.compute_open_zone_time_ratio \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "open_zone_time_ratio", "value": <float or null>}

stdout 末尾打印 [result] {json} 行供 subagent 抓取。
"""

from __future__ import annotations

import sys

from ethoinsight.metrics.zero_maze import compute_open_zone_time_ratio
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import (
    emit_result,
    make_compute_parser,
    parse_parameters,
    save_output_json,
)


METRIC_NAME = "open_zone_time_ratio"


def main(argv: list[str] | None = None) -> int:
    parser = make_compute_parser(description=__doc__)
    args = parser.parse_args(argv)

    df = parse_trajectory(args.input)
    parameters = parse_parameters(args)
    value = compute_open_zone_time_ratio(df, **parameters)

    payload = {"metric": METRIC_NAME, "value": value, "parameters_used": parameters}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
