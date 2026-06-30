---
name: ethoinsight-column-confirmation
description: EV19 自定义分析区列的 HITL 列语义对齐 — 预填反诘话术、答案映射、落盘指引
---

# EthoInsight 列语义确认 Skill

## 触发条件

inspect_uploaded_file 报告有**未被系统识别的自定义分析区列**时触发（典型：用户把分析区命名为「中心区」「Center」「zone_A」等，系统无法判断它对应哪个分析区）。

## 跳过条件

所有列都被识别（标准命名数据）→ **不触发列对齐反问**，流程同今天（零额外开销）。

## 方法论

**处理用户自定义列前，先 `read_file references/column-processing-methodology.md`** 了解三层列处理框架（固定列 / 自定义区 1:1 映射 / 子区聚合）与工具分工（identify → inspect → dump_headers）。

### 重要：列名非标准 ≠ 换模板

当 inspect 报告未识别的自定义分析区列（如 open/closed/中心区/zone_A）时，这是**列对齐**任务，不是重新判断模板的信号。模板变体（AllZones / FewZones / NoZones）由"录制时是否划分析区"决定：数据里有区归属列就属于划了区的变体（Few/AllZones）；NoZones 指数据完全没有任何区归属列（只有 x/y 轨迹）。**保持已选模板不变**，把已识别的归属列对齐到 catalog 概念即可（参考 `references/zone-concepts.generated.md` 菜单）。列名非标准是命名/对齐问题，换模板会丢掉全部区指标。

### 1. 理解 catalog 合法概念菜单

先读 `identify_ev19_template` 结果中的 `paradigm_key`，按范式查 catalog 的合法分析区概念：

各范式合法分析区概念见 `references/zone-concepts.generated.md`——该菜单由 catalog 自动生成、与 SSOT 同源，按此对齐。

> 概念关键词的权威来源是 catalog YAML 的 `requires_columns`，`zone-concepts.generated.md` 是其机械投影。

### 2. 预填 + 反诘（D15）

对每个待确认的列：
1. 参考该列的客观取值分布（占比 / 数值范围，由 inspect 提供）
2. 从 catalog 合法概念菜单匹配最可能的概念
3. **预填你的最佳理解，明确说"请确认"**（不是直接采纳）
4. 给否决口子："以上理解对吗？如有错误请告诉我正确对应。"

**绝不从列名字面猜测**。证据 + catalog 菜单 → 预填，不来自 "中心区就有 center 字样所以是 center"。

### 3. 多列合并反问（D16）

所有未对齐列在一个表格中展示，一次问清。如果模板只有一个候选，同时宣告模板已识别免重复问。

### 4. 答案映射

用户自然语言回答 → `column_semantics` 参数映射，见 `references/answer-mapping.md`。

### 5. 落盘

调用 `set_experiment_paradigm(column_semantics={...}, column_semantics_source=...)` 将决议写入 `experiment-context.json`。

- `resolves_to` 填**概念关键词**（如 `center` / `open_arms`），**不是**机器列名（系统自动翻译成数据能匹配的列）
- 用户确认忽略的列：`resolves_to: null, ignore: true`
- `meaning_zh` 填中文叙述语义（如 "中心分析区"），喂给 report-writer
- 用户本轮确认的列：`confirmed: true`
- **`column_semantics_source` 必须如实声明这些值的来源**（守 HITL 铁律：不能替用户确认用户本轮没回答的项）：
  - `"user_current_turn"`：用户本轮明确回答了这些列 → `confirmed: true` 生效，分析放行。
  - `"prefilled_from_memory"`：这些值来自 memory 历史偏好（用户本轮没回答列语义，例如只回了模板/分组）→ 系统确定性把每列降级为 `confirmed: false`（待确认预填），code-executor 派遣会被门拦下并触发 ask_clarification 重问；memory 历史偏好的正确用法是**预填进反问选项让用户一键确认**，不是替用户跳过确认。

### 6. 二次确认（用户否决时）

如果用户说 "不对，X 列不是 Y"，重新反问：
- 引用该列证据（取值占比）
- 指出用户标为 X
- 给出领域知识提示（"中心区通常是动物较少停留的区"）
- 问 "这是本意吗？回复'是'或更正。"

## 话术范例

见 `references/presentation-template.md`。
