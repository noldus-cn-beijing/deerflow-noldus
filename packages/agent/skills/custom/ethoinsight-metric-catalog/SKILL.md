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

在派遣 code-executor **之前** 走以下两步 bash：

**Step 1: 列名预检**

```bash
python -m ethoinsight.parse.dump_headers \
    --input /mnt/user-data/uploads/<raw_file>.txt \
    --output /mnt/user-data/workspace/columns.json
```

失败时 stderr 最后一行是 JSON，含 `code` 字段：

| code | 含义 | 怎么反问 |
|------|------|----------|
| `file.not_found` | 文件路径错 | 让用户重新提供路径或确认上传成功 |
| `format.unrecognized` | 文件不是 EthoVision 导出 | "这个文件看起来不像 EthoVision XT 导出，能确认下吗？" |
| `header.parse_failed` | header 损坏 | 让用户检查文件完整性 |

**Step 2: 生成执行计划**

```bash
python -m ethoinsight.catalog.resolve \
    --paradigm <epm|oft|fst|...> \
    --columns-file /mnt/user-data/workspace/columns.json \
    --raw-files-json /mnt/user-data/workspace/raw_files.json \
    --workspace-dir /mnt/user-data/workspace \
    --groups-file /mnt/user-data/workspace/groups.json \
    --output /mnt/user-data/workspace/metric_plan.json \
    [--include METRIC_ID]* \
    [--exclude METRIC_ID]* \
    [--n-per-group N] \
    [--n-groups N] \
    [--ev19-template TEMPLATE_ID]
```

> 注：早期版本要求显式传 `--virtual-workspace-dir /mnt/user-data/workspace`。
> 现已改用 sandbox 注入的 env var（`DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE`）作为兜底，
> lead 无需再传该参数——sandbox 会自动确保 plan.json output 字段是虚拟路径。
> 该参数仍保留以兼容直接命令行调试（无 sandbox env var 的场景）。

完整参数文档见 [`references/resolve-cli.md`](references/resolve-cli.md)。

失败时按 stderr JSON 的 `code` 字段反问用户：

| code | 反问话术（中文） |
|------|------------------|
| `unknown_paradigm` | （内部错误，不应发生 —— Gate 1 已识别） |
| `unknown_metric` | "您要的指标 `<id>` 我们目前没有预制脚本，要不咱们换 `<details.available>` 中的一个？" |
| `columns_missing` | "您的数据里缺 `<details.missing_patterns>` 这些列，没法跑 `<details.metric>` 指标。能确认下数据是否完整、或者咱们跳过这个指标？" |
| `empty_plan` | "按这个组合一项指标都跑不了，咱们是不是哪里搞错了？" |
| `schema_violation` | （内部错误，向用户致歉、把 details 报给开发者） |

**Step 3: 派遣 code-executor**

派遣 prompt 中只需要：

```
范式：{paradigm}
plan 路径：/mnt/user-data/workspace/metric_plan.json
分组：/mnt/user-data/workspace/groups.json

请按 plan.metrics[]、plan.statistics、plan.charts[] 逐条 bash 执行
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
