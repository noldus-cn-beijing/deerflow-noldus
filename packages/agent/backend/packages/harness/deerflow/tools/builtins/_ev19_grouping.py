"""EV19 分组字段提取 —— identify 和 inspect 共享的纯函数 (SSOT, 无双存)."""

from __future__ import annotations

# EV19 metadata 中表示"分组"的关键字段 (中英文都覆盖)
# 按出现频率排序: Treatment 是 Noldus 默认分组字段
_GROUPING_METADATA_KEYS = (
    "Treatment", "treatment",
    "Group", "group", "组别",
    "Drug", "drug",
    "Condition", "condition",
    "Dose", "dose", "剂量",
    "Compound", "compound",
    "Animal ID", "Animal", "动物 ID", "动物编号",
)


def extract_grouping_fields(raw_metadata: dict[str, str] | None) -> dict[str, str]:
    """从 EV19 raw_metadata 中提取分组相关字段。

    Args:
        raw_metadata: parse_header 返回的 raw_metadata dict, 可能为 None.

    Returns:
        匹配到的分组字段 dict (如 {"Treatment": "Drug", "Group": "XX"}), 无匹配时为空 dict.
    """
    if not raw_metadata:
        return {}
    result: dict[str, str] = {}
    for key in _GROUPING_METADATA_KEYS:
        if key in raw_metadata and raw_metadata[key]:
            result[key] = raw_metadata[key]
    return result
