"""Shoaling: 鱼群个体间距离 (inter-individual distance)。

CLI: python -m ethoinsight.scripts.shoaling.compute_inter_individual_distance \
       --inputs <inputs.json> --output <metric.json>

inputs.json 包含多个 trajectory 文件路径的 JSON 数组。

输出 JSON:
  {"metric": "inter_individual_distance", "value": {"mean_iid_mean": float | null, ...}}

stdout 末尾打印 [result] {json} 行供 subagent 抓取。
"""

from __future__ import annotations
import sys
from ethoinsight.metrics.shoaling import compute_inter_individual_distance
from ethoinsight.parse import parse_batch
from ethoinsight.scripts._cli import (
    emit_result,
    read_inputs_json,
    save_output_json,
)
import argparse

METRIC_NAME = "inter_individual_distance"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inputs", required=True, help="Path to JSON array of input file paths"
    )
    ap.add_argument("--output", required=True, help="Path to write metric JSON")
    args = ap.parse_args(argv)

    paths = read_inputs_json(args.inputs)
    parsed = parse_batch(paths)
    result = compute_inter_individual_distance(parsed["subjects"])

    if result is not None:
        summary = {
            "mean_iid_mean": float(result["mean_iid"].mean()),
            "mean_iid_std": float(result["mean_iid"].std()),
        }
    else:
        summary = None
    payload = {"metric": METRIC_NAME, "value": summary}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
