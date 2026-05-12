# TST (Tail Suspension Test) 范式

> 学术范式 key: `tail_suspension`
> EV19 模板映射: `TailSuspension` 大类下所有变体
> 行为学同事维护的领域知识: `docs/review-packages/2026-04-29-ev19-templates/by-experiment/tail_suspension.md`

## 可用脚本清单

所有脚本以 `python -m <module_path> --input ... --output ...` 调用。

### 核心指标脚本（compute_*.py）

| 脚本 module | --input | --output | 输出 JSON | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts.tst.compute_immobility_time` | 单轨迹文件 | metric JSON | `{"metric": "immobility_time", "value": float \| null}` | 总不动时间（秒） |
| `ethoinsight.scripts.tst.compute_immobility_latency` | 单轨迹文件 | metric JSON | `{"metric": "immobility_latency", "value": float \| null}` | 首次不动潜伏期（秒） |
| `ethoinsight.scripts.tst.compute_immobility_bout_count` | 单轨迹文件 | metric JSON | `{"metric": "immobility_bout_count", "value": int \| null}` | 不动发作次数 |

### 通用指标脚本（任范式可用）

| 脚本 module | --input | --output | 输出 JSON | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts._common.compute_distance_moved` | 单轨迹文件 | metric JSON | `{"metric": "distance_moved", "value": float \| null}` | 总移动距离 |
| `ethoinsight.scripts._common.compute_velocity_stats` | 单轨迹文件 | metric JSON | `{"metric": "velocity_stats", "value": {mean, std, max, min, median} \| null}` | 速度描述统计 |

### 可视化脚本（plot_*.py）

| 脚本 module | --input / --inputs | --groups | --output | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts._common.plot_trajectory` | `--input <单文件>` 或 `--inputs <inputs.json>` | — | PNG | 轨迹图 |
| `ethoinsight.scripts.tst.plot_box_immobility` | `--inputs <inputs.json>` | `--groups <groups.json>` | PNG | 不动时间组间对比箱线图 |

### 统计脚本（run_*_stats.py）

| 脚本 module | --inputs | --groups | --output | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts.tst.run_groupwise_stats` | `<inputs.json>` | `<groups.json>` | stats JSON | 3 指标全 Shapiro-Wilk 决策树检验 |

## 输入文件格式约定

### `--input`（单文件）
直接传 EthoVision 导出 `.txt` 文件的路径。

### `--inputs`（多文件聚合）
传一个 JSON 文件路径，内容是文件路径数组。subagent 先用 `write_file` 生成此 JSON：

```json
["/mnt/user-data/uploads/subject1.txt", "/mnt/user-data/uploads/subject2.txt"]
```

### `--groups`
传一个 JSON 文件路径，内容是分组映射：

```json
{
  "control": ["Subject 1", "Subject 2", "Subject 3"],
  "treatment": ["Subject 4", "Subject 5", "Subject 6"]
}
```

subject 名称由 EthoVision 文件 header 的 `对象名称` 字段决定。

## 实验设计决策树

根据 lead 提供的 `实验设计` 字段裁剪脚本列表：

### n=1（单样本描述性分析）
- ✅ 跑全部 `compute_*.py`（3 个 TST 核心 + 2 个通用）
- ✅ 跑 `_common.plot_trajectory` 单文件版
- ❌ 跳过 `plot_box_immobility`（无组间对比意义）
- ❌ 跳过 `run_groupwise_stats`（无统计意义）

### n_per_group ∈ [3, 4]（小样本）
- ✅ 跑全部 `compute_*.py` for each subject
- ✅ 跑 `plot_box_immobility`（描述性，注意小样本）
- ⚠️ 跑 `run_groupwise_stats`，但在 handoff 标注 `data_quality_warnings: 小样本统计功效不足`

### n_per_group ≥ 5（标准）
- ✅ 跑全部脚本

### 用户特殊需求
- "跳过统计" → 跳过 `run_groupwise_stats`

## handoff JSON 必须字段

`${workspace_path}/handoff_code_executor.json` schema:

```json
{
  "paradigm": "tail_suspension",
  "per_subject": {
    "Subject 1": {"immobility_time": 100.2, "immobility_latency": 30.5, "immobility_bout_count": 6},
    ...
  },
  "charts": ["/mnt/user-data/workspace/outputs/tst_box.png", ...],
  "statistics": { /* 直接复制 run_groupwise_stats 的输出 JSON，可选 */ },
  "data_quality_warnings": [ /* 见下方 */ ],
  "errors": [ /* 脚本执行报错记录 */ ]
}
```

### data_quality_warnings 触发条件
- subject 数 < 5/组 → `{"severity": "warning", "message": "小样本统计功效不足"}`
- 某 subject 的 `immobility_latency == None`（从未进入不动状态）→ `{"severity": "warning", "message": "<subject> 从未进入不动状态，疑为运动亢进"}`
- 某指标在所有 subject 都返回 None → `{"severity": "critical", "message": "<metric> 列名识别失败，可能不是 TST 数据"}`

## 错误处理

脚本返回 non-zero 退出码或 stderr 非空时：

| 错误模式 | 处理 |
|---|---|
| `ValueError: must be a JSON array` | inputs/groups JSON 格式错 → 检查 write_file 生成的 JSON 内容 |
| `FileNotFoundError: <path>` | 路径不存在 → 用 `ls` 核对 |
| `activity_state` 列缺失 | 列名识别失败 → 写入 `data_quality_warnings`，标 critical，向 lead 反问 |
| `UnicodeDecodeError` | 文件编码不是 UTF-16-LE → 文件可能不是 EthoVision 导出 |

## 编排流程（subagent 工作流）

1. **read** 本文件，看清单和决策树
2. **裁剪**：根据 lead 给的实验设计 + 用户需求，决定要跑哪些脚本
3. **准备输入文件**：用 `write_file` 生成 `inputs.json` 和 `groups.json`（如需要）
4. **bash 循环调脚本**：每个脚本一个 bash 调用，**不要把多个脚本拼在一行**
5. **收集**：每个脚本调用后，stdout 含 `[result] {...}` 行；用 `read_file` 读各 metric JSON
6. **聚合**：构造 handoff JSON（schema 见上）
7. **写 handoff**：`write_file` 到 `${workspace_path}/handoff_code_executor.json`
8. **输出 [gate_signals]** 块（详见 code-executor system_prompt 的 `<output>` 段）

## 范式简介（领域知识）

- 抑郁样行为范式：不动时间 ↑ ＝ 行为绝望 ↑ ＝ 抑郁样表型
- 典型 EthoVision 列：`activity_state`（0=immobile, 1=mobile，注意 normalize_columns 小写）
- 标准实验：6 分钟、单次曝光（通常只分析后 4 分钟）
- TST vs FST：TST 仅用于小鼠（大鼠太重），无低体温混淆
- 关键混杂因素：尾部攀爬行为（可能被误判为 mobility）
- 详细领域知识：见 `docs/review-packages/2026-04-29-ev19-templates/by-experiment/tail_suspension.md`
