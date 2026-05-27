"""EPM: 身体拉伸度统计 (body elongation stats)。

CLI: python -m ethoinsight.scripts.epm.compute_body_elongation_stats \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "body_elongation_stats", "value": {mean/std/max/min/median} or null}
"""

from __future__ import annotations

import sys

from ethoinsight.metrics._common import compute_body_elongation_stats
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_compute_parser, save_output_json

METRIC_NAME = "body_elongation_stats"


def main(argv: list[str] | None = None) -> int:
    parser = make_compute_parser(description=__doc__)
    args = parser.parse_args(argv)
    df = parse_trajectory(args.input)
    value = compute_body_elongation_stats(df)
    payload = {"metric": METRIC_NAME, "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
