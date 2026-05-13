"""ethoinsight.parse — EthoVision XT data file parser.

Backward compat shim: re-exports public API from parse._core so that
`from ethoinsight.parse import parse_header, ...` continues to work.

新增子模块:
  - dump_headers : CLI 工具，输出 raw 文件列名清单到 JSON
"""

from __future__ import annotations

from ethoinsight.parse._core import (  # noqa: F401
    detect_ethovision,
    detect_paradigm,
    get_summary,
    normalize_columns,
    parse_batch,
    parse_header,
    parse_trajectory,
)
