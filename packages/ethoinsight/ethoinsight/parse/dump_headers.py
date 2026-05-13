"""ethoinsight.parse.dump_headers — CLI: 提取 raw 数据列名清单到 JSON.

CLI: python -m ethoinsight.parse.dump_headers \\
       --input <raw_file.txt> --output <columns.json>

成功: exit 0, 写 JSON {file, columns, n_subjects?, duration_s?}
失败: exit 1, stderr 最后一行写 {"code": "...", "message": "...", "details": {...}}
  code 枚举: file.not_found / header.parse_failed / format.unrecognized
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ethoinsight.parse._core import detect_ethovision, parse_header, parse_trajectory


def _emit_error(code: str, message: str, details: dict | None = None) -> int:
    payload = {"code": code, "message": message, "details": details or {}}
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    return 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ethoinsight.parse.dump_headers")
    p.add_argument("--input", required=True, help="raw EthoVision trajectory file")
    p.add_argument("--output", required=True, help="path to write columns.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    raw_path = Path(args.input)

    if not raw_path.is_file():
        return _emit_error(
            "file.not_found", f"Input not found: {raw_path}", {"path": str(raw_path)}
        )

    if not detect_ethovision(str(raw_path)):
        return _emit_error(
            "format.unrecognized",
            f"File does not look like an EthoVision export: {raw_path}",
            {"path": str(raw_path)},
        )

    try:
        header = parse_header(str(raw_path))
    except Exception as e:
        return _emit_error(
            "header.parse_failed",
            f"parse_header failed: {e}",
            {"path": str(raw_path)},
        )

    try:
        df = parse_trajectory(str(raw_path))
        columns = list(df.columns)
    except Exception as e:
        return _emit_error(
            "header.parse_failed",
            f"parse_trajectory failed: {e}",
            {"path": str(raw_path)},
        )

    payload = {
        "file": str(raw_path),
        "columns": columns,
        "n_subjects": header.get("number_of_subjects"),
        "duration_s": header.get("trial_duration"),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Wrote {len(columns)} column names to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
