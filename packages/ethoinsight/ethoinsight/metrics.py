"""向后兼容 shim：保旧导入路径 `from ethoinsight.metrics import compute_*` 工作。

实际实现在 `ethoinsight/metrics/<范式>.py`，通过 `metrics/__init__.py` 重导出。

新代码推荐: from ethoinsight.metrics.epm import compute_open_arm_time
"""

# Python 解析时优先 metrics/ 子包 over metrics.py（同名子包优先）。
# 这个 shim 其实不会被加载，但保留作为文档锚点。
# 真正的转发由 metrics/__init__.py 完成。
