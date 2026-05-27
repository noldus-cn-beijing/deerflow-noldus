# Excel 文件解析支持 — 技术规格

- **创建日期**：2026-05-11
- **状态**：已完成（PR #58，2026-05-27 合入 dev）
- **关联 PR**：https://github.com/noldus-cn-beijing/noldus-insight/pull/58

## 背景

当前 `ethoinsight/parse.py` 仅处理 EthoVision XT 的 UTF-16 LE 文本导出格式（`.txt` / `.csv`）。用户需要 agent 也能分析 Excel 格式（`.xlsx` / `.xls`）的轨迹数据。

基础设施已 90% 就绪：前端、网关、sandbox 均接受 `.xlsx` 文件，`pandas` + `openpyxl` 已在依赖中。缺口仅在解析层。

## 目标

`parse_batch()` 和 `parse_trajectory()` 在不改变输出协议的前提下，透明支持 `.xlsx/.xls` 文件，下游 metrics/statistics/charts/templates 零改动。

## 设计决策

### 实际实现与原始规格的差异

| 原规格 | 实际实现 | 原因 |
|--------|----------|------|
| 列名标记扫描检测（读 20 行匹配 "试用时间" 等） | 首单元格 "标题行数" 检测 | 所有真实 Excel 导出的 R0C0 均是"标题行数："，更简洁可靠 |
| 自动检测表头行（前 30 行评分） | 固定布局：header_lines 来自 R0C1 | 真实数据布局固定，无需自适应 |
| 仅 sheet 0 | 完整多 sheet 支持（`::sheet` 后缀约定） | FST 一个 xlsx 含 2 个 sheet = 2 subject |
| `parse.py` | `parse/_core.py` | 代码已按模块重构 |

### 检测策略：扩展名 + 首单元格验证

- `.xlsx/.xls` 扩展名 → `pd.read_excel(nrows=1)` 读 R0C0 → 检查是否包含"标题行数"
- `.txt/.csv` → 保持现有 UTF-16 BOM + "标题行数" 检测（不变）

### 表头布局：固定偏移

基于真实数据（5 范式 14 文件）验证的固定布局：
- R0C0 = "标题行数：", R0C1 = N
- R1..R(N-3): key-value 元数据（col 0 = key, col 1 = value）
- R(N-2): 列名
- R(N-1): 单位
- RN+: 数据

### 元数据提取

表头区域逐行读 col 0 + col 1 作为 key-value，跳过空行。与 TXT 共用 `normalize_columns()` + `detect_paradigm()`。

### 多 Sheet：`::sheet_name` 后缀约定

`prep_metric_plan_tool` 的 Step 1.5 自动检测多 sheet XLSX 并展开：
- 单 sheet：行为不变
- 多 sheet（如 FST）：上传列表自动展开为 `file.xlsx::Sheet1`, `file.xlsx::Sheet2` ……
- `_parse_path_and_sheet()` 在所有入口函数中统一拆解 `::` 分隔符
- 下游 metric 脚本调用 `parse_trajectory(args.input)` 无需改动

## 实施范围（已完成）

| 文件 | 改动 |
|------|------|
| `packages/ethoinsight/pyproject.toml` | 加 `openpyxl>=3.1` |
| `packages/ethoinsight/ethoinsight/parse/_core.py` | 新增 `_parse_path_and_sheet`、`_detect_ethovision_xlsx`、`_parse_header_xlsx`、`_parse_trajectory_xlsx` 四个函数，修改 `detect_ethovision`/`parse_header`/`parse_trajectory`/`parse_batch` 加格式分派 |
| `packages/ethoinsight/tests/test_parse.py` | 新增 5 测试类 10 用例（35 passed 0 failed） |
| `packages/agent/backend/.../prep_metric_plan_tool.py` | Step 1.5 多 sheet XLSX 自动展开 + `::sheet` 路径处理 |
| `packages/agent/backend/.../uploads_middleware.py` | 按文件扩展名区分数据文件 vs 文档文件，数据文件不再被 `read_file`/`grep` 误导 |

总计 ~270 行，6 文件。

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
