# catalog.resolve CLI 参数完整文档

## 必填参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--paradigm` | str | 范式 key: epm / oft / fst / tst / ldb / zero_maze / shoaling |
| `--columns-file` | path | dump_headers 产物路径 `columns.json` |
| `--raw-files-json` | path | JSON 数组文件 含 raw 文件绝对路径 |
| `--workspace-dir` | path | 工作区根目录（物理路径，用于脚本执行） |
| `--virtual-workspace-dir` | path | plan.json output 字段使用的虚拟路径。**通常无需在 sandbox 中显式传入**——CLI 优先读 env var `DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE` 自动确定。显式传入只用于直接命令行调试（无 sandbox 环境）。fallback 顺序：(1) 显式 `--virtual-workspace-dir` 是虚拟路径 → 用它 (2) env var 存在 → 用硬编码 `/mnt/user-data/workspace` (3) 兜底物理 `--workspace-dir` |
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
