# OFT (Open Field Test) 范式

> 学术范式 key: `open_field`（注意：dispatcher 用 snake_case，不是 "oft"）
> EV19 模板映射: `OpenFieldRectangle` / `OpenFieldCircle` 大类下所有变体
> 行为学同事维护的领域知识: `docs/review-packages/2026-04-29-ev19-templates/by-experiment/open_field.md`

## 可用脚本清单

所有脚本以 `python -m <module_path> --input ... --output ...` 调用。

### 核心指标脚本（compute_*.py）

| 脚本 module | --input | --output | 输出 JSON | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts.oft.compute_center_time_ratio` | 单轨迹文件 | metric JSON | `{"metric": "center_time_ratio", "value": float \| null}` | 中心区域时间占比 |
| `ethoinsight.scripts.oft.compute_center_entry_count` | 单轨迹文件 | metric JSON | `{"metric": "center_entry_count", "value": int \| null}` | 进入中心区域次数 |
| `ethoinsight.scripts.oft.compute_center_distance_ratio` | 单轨迹文件 | metric JSON | `{"metric": "center_distance_ratio", "value": float \| null}` | 中心区域移动距离占比 |
| `ethoinsight.scripts.oft.compute_thigmotaxis_index` | 单轨迹文件 | metric JSON | `{"metric": "thigmotaxis_index", "value": float \| null}` | 趋触性指数（周边区滞留偏好） |

### 通用指标脚本（任范式可用）

| 脚本 module | --input | --output | 输出 JSON | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts._common.compute_distance_moved` | 单轨迹文件 | metric JSON | `{"metric": "distance_moved", "value": float \| null}` | 总移动距离 |
| `ethoinsight.scripts._common.compute_velocity_stats` | 单轨迹文件 | metric JSON | `{"metric": "velocity_stats", "value": {mean, std, max, min, median} \| null}` | 速度描述统计 |

### 可视化脚本（plot_*.py）

| 脚本 module | --input / --inputs | --groups | --output | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts._common.plot_trajectory` | `--input <单文件>` 或 `--inputs <inputs.json>` | — | PNG | 轨迹图（**用户提到"轨迹"必跑**） |
| `ethoinsight.scripts.oft.plot_box_center` | `--inputs <inputs.json>` | `--groups <groups.json>` | PNG | 中心区指标组间对比箱线图 |

### 统计脚本（run_*_stats.py）

| 脚本 module | --inputs | --groups | --output | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts.oft.run_groupwise_stats` | `<inputs.json>` | `<groups.json>` | stats JSON | 4 指标全 Shapiro-Wilk 决策树检验 |

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
- ✅ 跑全部 `compute_*.py`（4 个 OFT 核心 + 2 个通用）
- ✅ 跑 `_common.plot_trajectory` 单文件版
- ❌ 跳过 `plot_box_center`（无组间对比意义）
- ❌ 跳过 `run_groupwise_stats`（无统计意义）

### n_per_group ∈ [3, 4]（小样本）
- ✅ 跑全部 `compute_*.py` for each subject
- ✅ 跑 `plot_box_center`（描述性，注意小样本）
- ✅ 跑 `_common.plot_trajectory --inputs` 多文件版
- ⚠️ 跑 `run_groupwise_stats`，但在 handoff 标注 `data_quality_warnings: 小样本统计功效不足`

### n_per_group ≥ 5（标准）
- ✅ 跑全部脚本

### 用户特殊需求
- "只要轨迹图" → 仅跑 `_common.plot_trajectory`
- "跳过统计" → 跳过 `run_groupwise_stats`
- "我想看 X 指标" → 仅跑对应 `compute_X.py`

## handoff JSON 必须字段

`${workspace_path}/handoff_code_executor.json` schema:

```json
{
  "paradigm": "open_field",
  "per_subject": {
    "Subject 1": {"center_time_ratio": 0.25, "center_entry_count": 10, ...},
    ...
  },
  "charts": ["/mnt/user-data/workspace/outputs/oft_box.png", ...],
  "statistics": { /* 直接复制 run_groupwise_stats 的输出 JSON，可选 */ },
  "data_quality_warnings": [ /* 见下方 */ ],
  "errors": [ /* 脚本执行报错记录 */ ]
}
```

### data_quality_warnings 触发条件
- subject 数 < 5/组 → `{"severity": "warning", "message": "小样本统计功效不足"}`
- `center_entry_count < 5`（某 subject）→ `{"severity": "warning", "message": "<subject> 中心进入次数过低，疑为运动抑制"}`
- `distance_moved < 1000`（某 subject）→ `{"severity": "warning", "message": "<subject> 总移动距离异常低"}`
- 某指标在所有 subject 都返回 None → `{"severity": "critical", "message": "<metric> 列名识别失败，可能不是 OFT 数据"}`

## 错误处理

脚本返回 non-zero 退出码或 stderr 非空时：

| 错误模式 | 处理 |
|---|---|
| `ValueError: must be a JSON array` | inputs/groups JSON 格式错 → 检查 write_file 生成的 JSON 内容 |
| `FileNotFoundError: <path>` | 路径不存在 → 用 `ls` 核对 |
| `KeyError: 'in_zone_center'` 等 | 列名识别失败 → 写入 `data_quality_warnings`，标 critical，向 lead 反问 |
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

- 焦虑研究常用范式：中心区域探索 ↓ ＝ 焦虑 ↑
- 趋触性（thigmotaxis）是核心指标：正常啮齿类倾向于贴壁（周边），高焦虑时贴壁更显著
- 标准实验：5-10 分钟、单次曝光
- 关键混杂因素：运动抑制（看 `distance_moved` 和 `center_entry_count`）
- 详细领域知识：见 `docs/review-packages/2026-04-29-ev19-templates/by-experiment/open_field.md`
