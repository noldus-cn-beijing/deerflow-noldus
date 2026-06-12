"""ethoinsight.parse — EthoVision XT data file parser.

Backward compat shim: re-exports public API from parse._core so that
`from ethoinsight.parse import parse_header, ...` continues to work.

新增子模块:
  - dump_headers : CLI 工具，输出 raw 文件列名清单到 JSON
"""

from __future__ import annotations

# Excel 解析依赖 python-calamine（PR #125 起，硬依赖）。在 parse 模块 import 期
# 响亮断言其可用——缺依赖应在部署/启动当场失败，而非运行到第一次读 xlsx 才炸，
# 更不该被 _detect_ethovision_xlsx 的 try/except 误判成"非 EthoVision 文件"（哑故障）。
try:
    import python_calamine  # noqa: F401  # ⚠️ import 名是 python_calamine 不是 calamine
except ImportError as e:
    raise ImportError(
        "python-calamine is required for ethoinsight.parse (Excel XLSX/XLS reading via "
        "pandas engine='calamine', PR #125). Install: uv add 'python-calamine>=0.6,<0.7' "
        "or pip install 'python-calamine>=0.6,<0.7'."
    ) from e

from ethoinsight.parse._core import (  # noqa: F401
    detect_ethovision,
    detect_paradigm,
    get_summary,
    normalize_columns,
    parse_batch,
    parse_header,
    parse_trajectory,
)
