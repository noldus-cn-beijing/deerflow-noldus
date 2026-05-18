---
name: ethoinsight-metric-catalog
description: >
  EthoInsight 范式指标 catalog 读取手册。lead 用 resolve CLI 生成 plan；
  data-analyst 取 direction_for_anxiety / statistical_default 做判读；
  report-writer 取 display_name_zh / unit_zh / one_liner 翻译展示。
  catalog 是 single source of truth：范式→指标清单 + 展示元数据 + 判读
  方向性集中在 packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml。
type: knowledge
version: 0.1.0
author: noldus-insight
---

# EthoInsight Metric Catalog 入口

## catalog 物理位置

```bash
python -c "import ethoinsight.catalog as c; print(c.__file__)"
# 输出: .../ethoinsight/catalog/__init__.py
# YAML 在同目录: epm.yaml / oft.yaml / fst.yaml / tst.yaml / ldb.yaml / zero_maze.yaml / shoaling.yaml
```

## 你是哪种 role？

### lead

在派遣 code-executor **之前**，调 `prep_metric_plan` 工具一步完成列名解析 + catalog resolve。

```
prep_metric_plan(uploaded_file="/mnt/user-data/uploads/<raw_file>.txt", paradigm="<epm|oft|fst|...>")
```

**参数**：
- `uploaded_file`：用户上传的数据文件虚拟路径（如 `/mnt/user-data/uploads/轨迹.txt`）
- `paradigm`：范式 ID（`epm` / `oft` / `fst` / `ldb` / `tst` / `zero_maze` / `shoaling`）

**成功返回** (`status="ok"`)：
```json
{
  "status": "ok",
  "plan_path": "/mnt/user-data/workspace/metric_plan.json",
  "plan_summary": {
    "paradigm": "epm",
    "metric_count": 5,
    "metric_ids": ["open_arm_time_ratio", "open_arm_entries_ratio", ...]
  }
}
```

**失败返回** (`status="error"`)：

| error_code | 含义 | 怎么反问（hint） |
|------------|------|------------------|
| `file_not_found` | 数据文件不存在，可能用户上传失败 | ask_clarification 让用户重新上传 |
| `format_unrecognized` | 文件不是 EthoVision XT 导出格式 | ask_clarification 让用户确认导出方式 |
| `parse_failed` | 数据文件损坏 | ask_clarification 让用户重新导出 |
| `unknown_paradigm` | 范式不在 catalog 内 | ask_clarification 让用户确认范式或检查 set_experiment_paradigm 调用 |
| `columns_missing` | 数据缺关键列（可能录制设置漏了 Open/Closed arms 进入次数） | ask_clarification 让用户确认实验录制设置 |
| `schema_violation` | catalog YAML 损坏——项目内部 bug | present_files 把错误信息呈现给用户，让他报 bug |
| `empty_plan` | 按当前参数一项指标都跑不了 | ask_clarification 确认用户需求 |
| `unknown_metric` | 用户要求的指标不在 catalog 中 | ask_clarification 让用户从可用指标中选择 |

工具返回的 `hint` 字段已包含上述反问话术，可直接用于 ask_clarification。

**派遣 code-executor**

派遣 prompt 中只需要：

```
范式：{paradigm}
plan 路径：/mnt/user-data/workspace/metric_plan.json
分组：/mnt/user-data/workspace/groups.json

请按 plan.metrics[]、plan.statistics、plan.charts[] 逐条执行
```

不要把指标清单展开在 prompt 里。

### data-analyst

判读某个指标时，按 metric id 查 catalog：

**Step 1: 定位 catalog YAML 路径**
（由 lead 在派遣时提供，或从本 skill 顶部读取）

**Step 2: read_file 对应的 YAML 文件**

关注字段：
- `direction_for_anxiety`: lower_is_anxious / higher_is_anxious / null
  - lower_is_anxious: 该指标越低 → 焦虑样行为越明显
  - higher_is_anxious: 该指标越高 → 焦虑样行为越明显
  - null: 该指标不属于焦虑判读维度（如运动量 control）
- `statistical_default`: groupwise_compare / paired_compare（验证 code-executor 用了正确的统计入口）

### report-writer

写 Results / Discussion 段时，按 metric id read catalog YAML 取展示字段：

- `display_name_zh`: 中文展示名
- `unit_zh`: 单位
- `one_liner`: 一句话解释（仅首次提及该指标时引用）

详细字段映射见 [`references/field-guide.md`](references/field-guide.md)。
