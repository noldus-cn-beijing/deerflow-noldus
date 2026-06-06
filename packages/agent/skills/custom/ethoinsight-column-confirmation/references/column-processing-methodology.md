# EV19 列处理三层框架与工具分工

本文档描述 EthoInsight 处理 EthoVision XT 导出列的系统方法论。agent 处理用户自定义分析区列前，必须先读本文档了解框架与工具分工。

---

## 三层框架

### Layer 1 — 固定列（白名单匹配，零 HITL）

EthoVision XT 每个模板有一组**固定列**（如 `x_center`、`y_center`、`velocity`、`Mobility state`、`Elongation` 等），名称不受用户语言环境影响。这些列通过白名单匹配自动识别，无需用户确认。

**实现**：`ethoinsight.catalog.resolve` 的 `requires_columns` glob 匹配。

### Layer 2 — 自定义分析区 1:1 映射（Sprint 1 已实现）

用户可以为每个分析区自定义列名（如"中心区""边缘区""Center Zone"等）。Sprint 1 已实现 **HITL 列语义对齐**：

1. `inspect_uploaded_file` 报告未识别的自定义分析区列
2. agent 按 catalog 合法概念菜单 + 客观证据（取值分布）预填推测
3. 用户确认或纠正映射关系
4. 决议通过 `set_experiment_paradigm(column_semantics={...})` 写入 `experiment-context.json`

**每个自定义列对应一个 catalog 概念**（如"中心区"→`center`），1:1 映射，不涉及多列聚合。

**注意**：自定义分析区列的**合法概念菜单以 catalog YAML 的 `requires_columns` glob 为权威**，不在本文档硬编码。

### Layer 3 — 子区聚合（聚合语义待 Issue #98 专家确认）

某些 EV19 模板在一个分析区内有多个子区（如 EPM 的 4 条臂、Zero Maze 的 4 个开放区段）。这些子区的指标如何聚合为单一分析区值（OR 还是 sum），取决于行为学专家对"EV19 zone 是否空间重叠"的回答。

**当前状态**：聚合语义留给 Issue #98，本 skill 不含聚合规则。

**agent 职责（仅骨架）**：
1. **检测**：识别出需要聚合的场景（多列指向同一分析区概念）
2. **HITL 复述**：向用户说明"你的数据有 N 个 X 区子列，需要聚合。当前系统暂不支持自动聚合，请告知期望的聚合方式"
3. **留白**：不猜测聚合方式，等 Issue #98 结论后再实施

---

## 工具分工

处理 EV19 列名有三个工具，**职责不同，不能互相替代**：

| 工具 | 用途 | 获取什么 | 局限 |
|------|------|---------|------|
| `identify_ev19_template` | 识别 EV19 模板 | 模板 ID、paradigm key、前 20 列名（截断） | 列名被截断，**不可用于列对齐决策** |
| `inspect_uploaded_file` | 完整列名 + 几何证据 | 完整列名（不截断）、每个分析区的坐标范围、样本数据 | 需带 `paradigm` 参数才给出分析区几何验证 |
| `python -m ethoinsight.parse.dump_headers` | 原始列名导出 | 完整原始列名清单（JSON） | 无几何证据、无分析区识别 |

### 明确禁止

- **禁止 `bash head -1`**：UTF-16 编码 / 分号分隔符 / 多行标题行，三处全错。列名走 parser，不走 shell 文本处理。

---

## 列名来源链

完整列名**只有一个来源**：

1. `inspect_uploaded_file` 输出（含完整列名 + 几何证据，推荐）
2. `python -m ethoinsight.parse.dump_headers` 输出（纯列名 JSON，无几何证据）

**不从以下来源获取列名**：
- ❌ `head -1` / `cat` / shell 文本处理
- ❌ `identify_ev19_template` 的前 20 列截断
- ❌ 用户口头描述（必须从数据文件获取）

---

## 补充说明

- **不内嵌结构化知识**：固定列清单、逻辑区清单、合法分析区概念均在 catalog YAML 中（single source of truth），本文档不重复。
- **Issue #98**：Layer 3 聚合语义（OR vs sum）取决于行为学专家对 zone 空间重叠性的回答。在此之前不实施任何聚合逻辑。
