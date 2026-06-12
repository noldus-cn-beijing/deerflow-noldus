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
)

# Animal ID 不是分组字段而是 subject identifier; 单独处理, 只取首个命中的变体
# (同一 header 一般只会有一个 Animal ID 变体, 取首个避免别名重复污染分组 dict)。
_ANIMAL_ID_KEYS = ("Animal ID", "Animal", "动物 ID", "动物编号")


def extract_grouping_fields(raw_metadata: dict[str, str] | None) -> dict[str, str]:
    """从 EV19 raw_metadata 中提取分组相关字段。

    Args:
        raw_metadata: parse_header 返回的 raw_metadata dict, 可能为 None.

    Returns:
        匹配到的分组字段 dict (如 {"Treatment": "Drug", "Group": "XX"}), 无匹配时为空 dict.
        额外附带首个命中的 Animal ID 变体 (subject identifier, 方便 lead 理解),
        若已在分组字段中命中则不重复。
    """
    if not raw_metadata:
        return {}
    result: dict[str, str] = {}
    for key in _GROUPING_METADATA_KEYS:
        if key in raw_metadata and raw_metadata[key]:
            result[key] = raw_metadata[key]
    # 额外: Animal ID 不一定是分组字段但是 subject identifier, 一起返回方便 lead 理解。
    # 只取首个命中的变体 (break), 避免多个别名同时出现时污染分组 dict。
    for k in _ANIMAL_ID_KEYS:
        if k in raw_metadata and raw_metadata[k] and k not in result:
            result[k] = raw_metadata[k]
            break
    return result
