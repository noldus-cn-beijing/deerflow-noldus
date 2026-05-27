# 2026-05-27 Excel 解析支持 handoff

## 完成内容

PR #58 已合入 dev（HEAD 0fd3e62），实现 EthoVision XT Excel 导出文件的解析支持。

### 改动文件（6 文件，~350 行）

| 文件 | 改动 |
|------|------|
| `packages/ethoinsight/ethoinsight/parse/_core.py` | 新增 `_parse_path_and_sheet`、`_detect_ethovision_xlsx`、`_parse_header_xlsx`、`_parse_trajectory_xlsx` 四个函数；修改 `detect_ethovision`/`parse_header`/`parse_trajectory`/`parse_batch` 加格式分发 + `::sheet` 后缀约定 |
| `packages/ethoinsight/pyproject.toml` | 加 `openpyxl>=3.1` 依赖 |
| `packages/agent/backend/.../prep_metric_plan_tool.py` | Step 1.5 多 sheet XLSX 自动展开为 N 个条目 + `::sheet` 路径处理 |
| `packages/agent/backend/.../uploads_middleware.py` | 按文件扩展名区分数据文件 vs 文档文件，数据文件不再被 `read_file`/`grep` 误导 |
| `packages/ethoinsight/tests/test_parse.py` | 新增 5 测试类 10 用例（35 passed 0 failed） |
| `packages/agent/backend/tests/test_uploads_middleware_core_logic.py` | 更新 grep hint 断言适配新行为（37 passed 0 failed） |

### 设计约定：`::sheet_name` 后缀

多 sheet XLSX（如 FST 的 Mobility_1 + Mobility_10 并存于一个文件）通过路径后缀指定：

```
/mnt/user-data/uploads/data.xlsx::轨迹-Arena 1-Subject 1
```

- `_parse_path_and_sheet()` 在所有入口函数中统一拆解 `::` 分隔符
- `prep_metric_plan_tool` 的 Step 1.5 自动检测多 sheet 并展开为 N 个条目
- 下游 metric 脚本无需改动（`parse_trajectory(args.input)` 透明处理）

### 测试情况

- **ethoinsight parse tests**: 35 passed, 16 skipped, 0 failed
- **prep_metric_plan tests**: 17 passed, 0 failed
- **uploads_middleware tests**: 37 passed, 0 failed
- **E2E**: 用户用 FST 多 sheet XLSX 实测通过（lead → prep_metric_plan → code-executor → data-analyst）
- **5 范式 14 XLSX 文件**: 全部 detect/parse_header/parse_trajectory 通过

### E2E 验证细节

FST Mobility_10 双 sheet xlsx → prep_metric_plan 自动展开为 2 条目 → code-executor 6 个 compute 脚本成功 → data-analyst 完成解读。

唯一小问题：lead agent 在 prep_metric_plan 前尝试 `read_file` 读 xlsx 失败后自己恢复，未卡死。已通过 UploadsMiddleware 修复。

## 下一步（无紧急事项）

1. **可选项**：`UploadsMiddleware._DATA_FILE_EXTENSIONS` 目前是 hard-coded `{".xlsx", ".xls", ".csv", ".tsv", ".txt"}`，如果后续有更多数据文件类型（如 `.json`、`.parquet`），需要扩展这个集合。更好的方案是从文件格式检测（magic bytes）自动判断，但当前最简 hard-code 足够。
2. **前端 UX**：Issue #25 还提到"上传前没有任何指示应该导入何种文件类型"。可以在前端上传区域加 accept 属性或 supported formats 提示文案。

## 相关 Issue

- Issue #25：[用户反馈] 无法识别 excel 文件 → PR #58 修复

## 关键文件路径

- 解析器核心：`packages/ethoinsight/ethoinsight/parse/_core.py`
- 工具入口：`packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py`
- 上传中间件：`packages/agent/backend/packages/harness/deerflow/agents/middlewares/uploads_middleware.py`
- Demo XLSX 数据：`/home/wangqiuyang/DemoData/newdemodata/`
- PR：https://github.com/noldus-cn-beijing/noldus-insight/pull/58
