"""EPM: 转角统计 (turn angle stats, unsigned)。

CLI: python -m ethoinsight.scripts.epm.compute_turn_angle_stats \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "turn_angle_stats", "value": {mean_abs_rad/mean_abs_deg/std_abs_rad/total_abs_rad/n} or null}
"""

from __future__ import annotations

import sys

from ethoinsight.metrics._common import compute_turn_angle_stats
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_compute_parser, save_output_json

METRIC_NAME = "turn_angle_stats"


def main(argv: list[str] | None = None) -> int:
    parser = make_compute_parser(description=__doc__)
    args = parser.parse_args(argv)
    df = parse_trajectory(args.input)
    value = compute_turn_angle_stats(df)
    payload = {"metric": METRIC_NAME, "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
