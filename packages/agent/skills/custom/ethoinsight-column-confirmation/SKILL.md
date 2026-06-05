---
name: ethoinsight-column-confirmation
description: EV19 自定义分析区列的 HITL 列语义对齐 — 预填反诘话术、答案映射、落盘指引
allowed-tools: read_file, ask_clarification, set_experiment_paradigm, inspect_uploaded_file
---

# EthoInsight 列语义确认 Skill

## 触发条件

`inspect_uploaded_file` 返回 `column_assessment.open_questions` 非空时触发。

## 跳过条件

`open_questions` 为空 → 标准数据，**不触发列对齐反问**，流程同今天（零额外开销）。

## 方法论

### 1. 理解 catalog 合法概念菜单

先读 `identify_ev19_template` 结果中的 `paradigm_key`，按范式查 catalog：

- EPM: `open_arms` / `closed_arms` / `center`
- OFT: `center` / `border` / `corner`
- LDB: `light` / `dark`
- Zero Maze: `open_arms` / `closed_arms`
- FST: 无自定义分析区（mobility state 列 `mobility_*` 是系统标准列，不会被标 unrecognized）

> 具体的合法概念以 catalog YAML 的 `requires_columns` glob 为权威，上面只是速查。

### 2. 预填 + 反诘（D15）

对每列：
1. 从 `evidence` 获取取值分布（0/1 占比、连续值范围）
2. 从 catalog 合法概念菜单匹配最可能的概念
3. **预填你的最佳理解，明确说"请确认"**（不是直接采纳）
4. 给否决口子："以上理解对吗？如有错误请告诉我正确对应。"

**绝不从列名字面猜测**。证据 + catalog 菜单 → 预填，不来自 "中心区就有 center 字样所以是 center"。

### 3. 多列合并反问（D16）

所有未对齐列在一个表格中展示，一次问清。如果模板只有一个候选，同时宣告模板已识别免重复问。

### 4. 答案映射

用户自然语言回答 → `column_semantics` 参数映射，见 `references/answer-mapping.md`。

### 5. 落盘

调用 `set_experiment_paradigm(column_semantics={...})` 将决议写入 `experiment-context.json`。

- `resolves_to` 填 catalog 概念名（如 `in_zone_center`）
- 用户确认忽略的列：`resolves_to: null, ignore: true`
- `meaning_zh` 填中文叙述语义（如 "中心分析区"），喂给 report-writer
- 所有列（含忽略）都须 `confirmed: true`

### 6. 二次确认（用户否决时）

如果用户说 "不对，X 列不是 Y"，重新反问：
- 引用该列证据（取值占比）
- 指出用户标为 X
- 给出领域知识提示（"中心区通常是动物较少停留的区"）
- 问 "这是本意吗？回复'是'或更正。"

## 话术范例

见 `references/presentation-template.md`。
