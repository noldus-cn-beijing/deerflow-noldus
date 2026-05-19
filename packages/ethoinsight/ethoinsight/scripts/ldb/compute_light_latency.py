"""LDB: 明箱潜伏期 (light latency)。

CLI: python -m ethoinsight.scripts.ldb.compute_light_latency \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "light_latency", "value": <float or null>}

stdout 末尾打印 [result] {json} 行供 subagent 抓取。
"""

from __future__ import annotations

import sys

from ethoinsight.metrics.ldb import compute_light_latency
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import (
    emit_result,
    make_compute_parser,
    save_output_json,
)


METRIC_NAME = "light_latency"


def main(argv: list[str] | None = None) -> int:
    parser = make_compute_parser(description=__doc__)
    args = parser.parse_args(argv)

    df = parse_trajectory(args.input)
    value = compute_light_latency(df)

    payload = {"metric": METRIC_NAME, "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
