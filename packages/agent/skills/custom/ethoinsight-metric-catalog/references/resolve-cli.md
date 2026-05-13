# catalog.resolve CLI 参数完整文档

## 必填参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--paradigm` | str | 范式 key: epm / oft / fst / tst / ldb / zero_maze / shoaling |
| `--columns-file` | path | dump_headers 产物路径 `columns.json` |
| `--raw-files-json` | path | JSON 数组文件 含 raw 文件绝对路径 |
| `--workspace-dir` | path | 工作区根目录 |
| `--output` | path | 输出 plan.json 路径 |

## 可选参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--groups-file` | path | null | 分组 JSON 路径 |
| `--include` | str | (可重复) | 用户额外要的 metric id |
| `--exclude` | str | (可重复) | 用户排除的 metric id |
| `--n-per-group` | int | null | 每组样本量（用于 when 条件） |
| `--n-groups` | int | null | 组数（用于 when 条件） |
| `--ev19-template` | str | null | EV19 模板 ID 透传 |

## 错误码

| code | 含义 |
|------|------|
| `unknown_paradigm` | paradigm 无对应 catalog YAML |
| `unknown_metric` | include 指定的 id 不在 catalog |
| `columns_missing` | 数据缺少 default metric 需要的列 |
| `empty_plan` | 所有指标都被排除或不可用 |
| `schema_violation` | catalog YAML 损坏 / columns.json 格式错 |
