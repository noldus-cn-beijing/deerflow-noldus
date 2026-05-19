# Excel 文件解析支持 — 技术规格

- **创建日期**：2026-05-11
- **状态**：待实施
- **关联 plan**：`/home/wangqiuyang/.claude/plans/radiant-honking-snowflake.md`

## 背景

当前 `ethoinsight/parse.py` 仅处理 EthoVision XT 的 UTF-16 LE 文本导出格式（`.txt` / `.csv`）。用户需要 agent 也能分析 Excel 格式（`.xlsx` / `.xls`）的轨迹数据。

基础设施已 90% 就绪：前端、网关、sandbox 均接受 `.xlsx` 文件，`pandas` + `openpyxl` 已在依赖中。缺口仅在解析层。

## 目标

`parse_batch()` 和 `parse_trajectory()` 在不改变输出协议的前提下，透明支持 `.xlsx/.xls` 文件，下游 metrics/statistics/charts/templates 零改动。

## 设计决策

### 检测策略：扩展名 + 内容验证

- `.xlsx/.xls/.xlsm` 扩展名 → 读前 20 行扫描已知轨迹列名标记（如 "试用时间"、"X 中心"、"trial_time"、"x_center"）
- >= 2 个标记命中 → 识别为 EthoVision 轨迹 Excel
- `.txt/.csv` → 保持现有 UTF-16 BOM + "标题行数" 检测

### 表头行定位：自动检测

不硬编码行号。扫描前 30 行，对每行按已知列名标记匹配数评分，取最高分行。>= 2 分才认，否则回退到第 0 行。

### 元数据提取：尽力而为

表头行上方恰好 2 个非空单元格的行视为 key-value 对，匹配已知键名（"实验"/"Experiment"、"对象名称"/"Subject" 等）。匹配不到则回退为空字符串/文件名推断。

### 多 Sheet：仅 sheet 0

EthoVision 轨迹数据在第一个 sheet。多 sheet 场景后续按需扩展。

## 实施范围

| 文件 | 改动 |
|------|------|
| `packages/ethoinsight/pyproject.toml` | 加 `openpyxl>=3.0` |
| `packages/ethoinsight/ethoinsight/parse.py` | 新增 4 个函数，修改 2 个现有函数 |
| `packages/ethoinsight/tests/test_parse.py` | 新增 ~20 个 Excel 测试 |

### 新增函数

| 函数 | 职责 |
|------|------|
| `_detect_ethovision_excel(path)` | Excel 文件检测：读前 20 行，扫描轨迹列名标记 |
| `_find_header_row(df_raw)` | 自动定位表头行：按列名标记匹配数评分 |
| `_extract_excel_metadata(df_raw, header_row)` | 从表头上方提取 key-value 元数据 |
| `_parse_trajectory_excel(file_path)` | 核心解析：读 Excel → DataFrame，签名对齐 `parse_trajectory()` |

### 修改函数

| 函数 | 改动 |
|------|------|
| `detect_ethovision()` | 在 BOM 检查前插入 Excel 扩展名分支 |
| `parse_batch()` | 文件过滤和解析循环中加 Excel 后缀路由 |

## 不改变的部分

- `utils.py` — `normalize_columns()`、`detect_paradigm()`、`COLUMN_MAP` 完全复用
- `metrics/` — 消费 DataFrame，格式无关
- `statistics.py` / `charts.py` — 消费 metrics 输出
- `templates/` — 编排层，调用 `parse_batch()` 自动受益
- 前端 / 网关 / sandbox — 已接受 Excel

## 输出协议约束

`parse_trajectory()` 和 `parse_batch()` 的输出结构必须完全一致：

```
DataFrame.attrs = {experiment, trial_name, subject, start_time,
                   duration, arena, paradigm, columns, units}

parse_batch() → {metadata, subjects, all_data, summary}
```

Excel 解析器每个字段都必须填充，即使值为空字符串或空列表。

## Excel vs TXT 精度分析

| 维度 | TXT | Excel |
|------|-----|-------|
| 编码风险 | UTF-16 LE BOM 依赖，编码错误则解析失败 | 二进制格式，无编码歧义 |
| 类型推断 | 字符串 → `pd.to_numeric` 猜测 | 单元格自带类型，无需猜测 |
| 分隔符歧义 | 分号 + 引号 escaping | 单元格天然隔离 |
| 列名归一化 | `normalize_columns()` | 同左，完全一致 |
| 元数据提取 | 分号分隔 key-value 混注释行 | A 列 key + B 列 value，更结构化 |

**结论**：Excel 解析精度不低于 TXT，在类型推断和分隔符处理上更可靠。

## 风险

| 风险 | 缓解 |
|------|------|
| EthoVision 版本间 Excel 布局差异 | 自动检测表头行 + 尽力元数据提取 |
| 大文件内存 | 轨迹文件通常 < 5 万行，`read_excel` 足够；后续可按需加 `usecols` |
| 密码保护 Excel | `pd.read_excel()` 抛异常，捕获后返回清晰错误 |
| `.xls` 旧格式 | `xlrd` 需单独安装；初期仅支持 `.xlsx/.xlsm` |

## 测试策略

使用 `pd.DataFrame.to_excel(tmp_path)` 动态生成合成 Excel 文件，覆盖：

- **检测**：合法轨迹 / 无轨迹列 / 不存在 / 损坏文件
- **解析**：基本解析 / 无元数据 / 表头不同位置 / 中文列名 / 混合列名 / 空数据 / 缺失值
- **元数据提取**：标准两列布局 / 混合行跳过
- **混合格式**：同一批次 `.txt` + `.xlsx`

## 关联文档

- [实施 Plan](/home/wangqiuyang/.claude/plans/radiant-honking-snowflake.md)
- [CLAUDE.md](../../CLAUDE.md) — 仓库架构总览
