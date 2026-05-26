---
name: ethoinsight-metric-catalog
description: >
  EthoInsight 范式指标 catalog 消费手册。lead 通过 prep_metric_plan 工具
  在派遣 subagent 前生成 plan_metrics.json;data-analyst 和 report-writer
  从 plan_metrics.json 直接读判读 / 展示字段,不再 read catalog YAML 文件
  (它在 Python 包内,sandbox 不暴露)。
type: knowledge
version: 0.2.0
author: noldus-insight
---

# EthoInsight Metric Catalog 入口

## 设计契约(2026-05-27 起)

catalog YAML 仍是 single source of truth,但**只有 lead 通过 deerflow first-party
工具 `prep_metric_plan` 间接消费**(工具在 sandbox 外的 deerflow 进程内 import
ethoinsight.catalog,把解析结果写到 `/mnt/user-data/workspace/plan_metrics.json`)。

**subagent 不直接读 catalog YAML** — sandbox 的 read_file 白名单只放行
`/mnt/user-data/*` / `/mnt/skills/*` / `/mnt/acp-workspace/*` / `/mnt/shared/*`
和配置的 custom mounts。Python 包内路径会被 PermissionError 拒。

详见设计 spec:`docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md`

## 你是哪种 role?

### lead

在派遣 code-executor **之前**,调 `prep_metric_plan` 工具一步完成列名解析 + catalog resolve。

```
prep_metric_plan(uploaded_files=["/mnt/user-data/uploads/<raw_file>.txt", ...], paradigm="<epm|oft|fst|...>")
```

**参数**:
- `uploaded_files`: 用户上传的数据文件虚拟路径数组(把当前 `<uploaded_files>` 中所有相关文件都传进来)
- `paradigm`: 范式 ID(`epm` / `oft` / `fst` / `ldb` / `zero_maze`)

**成功返回** (`status="ok"`):
```json
{
  "status": "ok",
  "plan_path": "/mnt/user-data/workspace/plan_metrics.json",
  "plan_summary": {
    "paradigm": "epm",
    "metric_count": 5,
    "subject_count": 2,
    "metric_ids": ["open_arm_time_ratio", ...]
  }
}
```

**失败返回** (`status="error"`):

| error_code | 含义 | 怎么反问(hint) |
|------------|------|------------------|
| `file_not_found` | 数据文件不存在,可能用户上传失败 | ask_clarification 让用户重新上传 |
| `format_unrecognized` | 文件不是 EthoVision XT 导出格式 | ask_clarification 让用户确认导出方式 |
| `parse_failed` | 数据文件损坏 | ask_clarification 让用户重新导出 |
| `unknown_paradigm` | 范式不在 catalog 内 | ask_clarification 让用户确认范式或检查 set_experiment_paradigm 调用 |
| `columns_missing` | 数据缺关键列 | ask_clarification 让用户确认实验录制设置 |
| `schema_violation` | catalog YAML 损坏——项目内部 bug | present_files 让用户报 bug |
| `empty_plan` | 按当前参数一项指标都跑不了 | ask_clarification 确认用户需求 |
| `unknown_metric` | 用户要求的指标不在 catalog 中 | ask_clarification 让用户选择 |

工具返回的 `hint` 字段已含上述反问话术,可直接用于 ask_clarification。

**派遣 code-executor**

派遣 prompt 中只需要:

```
范式:{paradigm}
plan 路径:/mnt/user-data/workspace/plan_metrics.json
分组:/mnt/user-data/workspace/groups.json

请按 plan.metrics[]、plan.statistics、plan.charts[] 逐条执行
```

不要把指标清单展开在 prompt 里。

### data-analyst

判读某个指标时,read plan_metrics.json,按 metric id 在 `metrics[]` 数组中匹配:

```
read_file:
    /mnt/user-data/workspace/plan_metrics.json
```

关注字段(由 lead 派遣前 resolve 透传自 catalog YAML):

- `direction_for_anxiety`: `"lower_is_anxious"` / `"higher_is_anxious"` / `null`
  - lower_is_anxious: 该指标越低 → 焦虑样行为越明显
  - higher_is_anxious: 该指标越高 → 焦虑样行为越明显
  - null: 该指标不属于焦虑判读维度(如运动量 control)
- `statistical_default`: `"groupwise_compare"` / `"paired_compare"`(验证 code-executor 用了正确的统计入口)

多 subject 场景下同一 metric id 会出现多次(`subject_index` 区分),判读字段在所有同 id 行上一致,取首个即可。

**不要尝试 read catalog YAML 文件** — 它在 Python 包内,sandbox 不暴露。

### report-writer

写 Results / Discussion 段时,read plan_metrics.json,按 metric id 匹配:

```
read_file:
    /mnt/user-data/workspace/plan_metrics.json
```

展示字段(由 lead 派遣前 resolve 透传自 catalog YAML):

- `display_name_zh`: 中文展示名
- `unit_zh`: 中文单位
- `one_liner`: 一句话解释(仅首次提及该指标时引用)
- `output_unit`: 物理单位标识(seconds / count,辅助 LLM 理解 `unit_zh` 含义)

多 subject 场景下同一 metric id 会出现多次,展示字段在所有同 id 行上一致,取首个即可。

**不要尝试 read catalog YAML 文件** — 它在 Python 包内,sandbox 不暴露。

## metric 字段字典 reference(完整版)

每个 `metrics[]` 元素的全部字段:

| 字段 | 类型 | 用途 | 谁消费 |
|---|---|---|---|
| id | str | metric 唯一标识 | 所有 |
| script | str | Python 调用路径 | code-executor |
| input | str | 输入文件虚拟路径 | code-executor |
| output | str | 输出 JSON 虚拟路径 | code-executor |
| required | bool | 是否 default metric | code-executor |
| reason | str | 入选理由(paradigm.default / user.include / ...) | (诊断用) |
| subject_index | int | 0-based subject 序号 | code-executor / report-writer 区分 multi-subject |
| display_name_zh | str | 中文展示名 | report-writer / data-analyst |
| unit_zh | str | 中文单位 | report-writer |
| one_liner | str | 一句话解释 | report-writer(首次提及) |
| output_unit | str | 物理单位标识 | report-writer |
| direction_for_anxiety | "lower_is_anxious"/"higher_is_anxious"/null | 焦虑判读方向 | data-analyst |
| statistical_default | "groupwise_compare"/"paired_compare" | 默认统计入口 | data-analyst 校验 |
