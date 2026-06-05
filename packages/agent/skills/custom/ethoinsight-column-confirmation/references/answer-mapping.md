# 用户自然语言 → column_semantics 参数映射

## 格式

用户回答的自然语言 → 对应的 `column_semantics.columns` 条目。

`set_experiment_paradigm(column_semantics={...})` 的 schema：

```json
{
  "columns": {
    "<用户列名>": {
      "raw_name": "<原始列名>",
      "normalized": "<normalize 后>",
      "resolves_to": "<catalog 概念名 or null if ignore>",
      "meaning_zh": "<中文叙述语义>",
      "confirmed": true,
      "ignore": false  // true only when user says "忽略"
    }
  }
}
```

## 映射规则

### 用户说 "对" / "是的" / "确认" / "没问题"

→ 所有预填列 `confirmed: true`，值即预填的 `resolves_to`。

### 用户说 "中心区是 center，边缘区是 border，边缘区到中心区 忽略"

→ 逐列映射：
```json
{
  "columns": {
    "中心区": {"raw_name": "中心区", "normalized": "center", "resolves_to": "in_zone_center", "meaning_zh": "中心分析区", "confirmed": true},
    "边缘区": {"raw_name": "边缘区", "normalized": "边缘区", "resolves_to": "in_zone_border", "meaning_zh": "边缘分析区", "confirmed": true},
    "边缘区到中心区": {"raw_name": "边缘区到中心区", "normalized": "边缘区到center", "resolves_to": null, "ignore": true, "confirmed": true}
  }
}
```

### 用户说 "第一列中心 第二列边缘 第三列忽略/无关/不用"

→ 同上，按列顺序映射。

### 用户说 "X 列不对，应该是 Y"

→ 保持其他列不变，修正 X 列的 `resolves_to`，`confirmed` 变 true。

### 用户说 "忽略 X 列" / "X 列不用管"

→ X 列：`resolves_to: null, ignore: true, confirmed: true`。

### 用户说 "X 列不是分析区，是距离列"

→ X 列：`resolves_to: null, ignore: true, confirmed: true`。

## paradigm → catalog 概念名映射（速查）

| paradigm     | catalog 概念名       | meaning_zh   |
|-------------|---------------------|--------------|
| open_field  | in_zone_center_*     | 中心分析区    |
| open_field  | in_zone_border_*     | 边缘分析区    |
| open_field  | in_zone_corner_*     | 角落分析区    |
| epm         | in_zone_open_arms_*  | 开放臂       |
| epm         | in_zone_closed_arms_*| 封闭臂       |
| epm         | in_zone_center_*     | 中心区       |
| light_dark_box | in_zone_light_*   | 明区         |
| light_dark_box | in_zone_dark_*    | 暗区         |
| zero_maze   | in_zone_open_*       | 开放臂       |
| zero_maze   | in_zone_closed_*     | 封闭臂       |
| forced_swim | (无自定义区列)        | —            |

> 注意：`in_zone_*` 是 glob 模式，实际 catalog requires_columns 是 `in_zone_center_*`（可匹配 `in_zone_center_point`）等形式。
