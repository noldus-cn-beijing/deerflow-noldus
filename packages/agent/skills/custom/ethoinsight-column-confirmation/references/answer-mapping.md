# 用户回答 → 列对齐决议（语义层）

这份文档只讲**语义**：用户的自然语言 → 你理解成哪个**分析区概念**。
不讲 JSON 字段细节、不讲列名后缀——那些由 `set_experiment_paradigm` 工具和 catalog 代码负责。

## 你要做的

把每个待确认的用户列，对齐到该范式的一个**分析区概念关键词**（或判为忽略）。
概念关键词就是范式分析的区类型，**不带任何机器前缀/后缀**：

| paradigm | 分析区概念关键词 |
|----------|----------------|
| open_field | `center` / `border` / `corner` |
| epm | `open_arms` / `closed_arms` / `center` |
| light_dark_box | `light` / `dark` |
| zero_maze | `open` / `closed` |
| forced_swim | （无自定义分析区） |

> 概念关键词的权威来源是 catalog YAML 的 `requires_columns`；上表是速查。
> 你只需填概念关键词（如 `center`），系统自动把它翻译成数据能匹配的列名——
> 你**不需要**、也**不应该**填 `in_zone_center_point` 这种机器列名。

## 映射规则（语义）

| 用户这样说 | 你这样理解 |
|-----------|----------|
| "对" / "是的" / "确认" / "没问题" | 采纳你预填的全部对齐 |
| "中心区是 center，边缘区是 border" | 中心区→`center`，边缘区→`border` |
| "第一列中心 第二列边缘 第三列不用" | 按列序：→`center`、→`border`、→忽略 |
| "X 列不对，应该是 Y" | 只改 X 列的概念，其余不变 |
| "忽略 X 列" / "X 列不用管" / "X 是距离列不是分析区" | X 列判为**忽略** |

## 落盘

用 `set_experiment_paradigm(column_semantics={...})` 写决议。每列给：
- `raw_name`：与数据列头**一字不差**（如 `中心区`，不要写成 `center`）
- `resolves_to`：**概念关键词**（如 `center`），忽略的列填 `null` + `ignore: true`
- `meaning_zh`：中文叙述语义（如 "中心分析区"），供报告引用
- `confirmed: true`：含被忽略的列也要确认

具体字段格式由工具 docstring 说明，本文档不复述（避免知识双存）。
