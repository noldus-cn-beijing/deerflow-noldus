"""通用：总移动距离 (distance moved, sum of per-frame distance)。

CLI: python -m ethoinsight.scripts._common.compute_distance_moved \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "distance_moved", "value": <float or null>}
"""

from __future__ import annotations

import sys

from ethoinsight.metrics._common import compute_distance_moved
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_compute_parser, save_output_json


METRIC_NAME = "distance_moved"


def main(argv: list[str] | None = None) -> int:
    args = make_compute_parser(description=__doc__).parse_args(argv)
    df = parse_trajectory(args.input)
    value = compute_distance_moved(df)
    payload = {"metric": METRIC_NAME, "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
