# Shoaling（斑马鱼鱼群行为）范式

> 学术范式 key: `shoaling`
> EV19 模板映射: `Shoaling` 大类下所有变体
> 行为学同事维护的领域知识: `docs/review-packages/2026-04-29-ev19-templates/by-experiment/shoaling.md`

## 可用脚本清单

所有脚本以 `python -m <module_path> --input ... --output ...` 调用。

**注意：Shoaling 是多 subject 范式（5 鱼同时追踪）。compute 脚本使用 `--inputs`（JSON 数组）而非 `--input`（单文件）。**

### 核心指标脚本（compute_*.py）

| 脚本 module | --inputs | --output | 输出 JSON | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts.shoaling.compute_inter_individual_distance` | inputs JSON 数组 | metric JSON | `{"metric": "inter_individual_distance", "value": {"mean_iid_mean": float, "mean_iid_std": float} \| null}` | 鱼群个体间距离（均值、标准差） |
| `ethoinsight.scripts.shoaling.compute_nearest_neighbor_distance` | inputs JSON 数组 | metric JSON | `{"metric": "nearest_neighbor_distance", "value": {"mean_nnd": float, "std_nnd": float} \| null}` | 最近邻距离 |
| `ethoinsight.scripts.shoaling.compute_group_polarity` | inputs JSON 数组 | metric JSON | `{"metric": "group_polarity", "value": {"mean_polarity": float, "std_polarity": float} \| null}` | 群体极性（游动方向一致性，0–1） |

### 通用指标脚本（任范式可用）

| 脚本 module | --input | --output | 输出 JSON | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts._common.compute_distance_moved` | 单轨迹文件 | metric JSON | `{"metric": "distance_moved", "value": float \| null}` | 总移动距离（per subject） |
| `ethoinsight.scripts._common.compute_velocity_stats` | 单轨迹文件 | metric JSON | `{"metric": "velocity_stats", "value": {mean, std, max, min, median} \| null}` | 速度描述统计（per subject） |

### 可视化脚本（plot_*.py）

| 脚本 module | --input / --inputs | --groups | --output | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts._common.plot_trajectory` | `--input <单文件>` 或 `--inputs <inputs.json>` | — | PNG | 轨迹图 |
| `ethoinsight.scripts.shoaling.plot_box_iid` | `--inputs <inputs.json>` | `--groups <groups.json>` | PNG | 个体间距离组间对比箱线图 |

### 统计脚本（run_*_stats.py）

| 脚本 module | --inputs | --groups | --output | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts.shoaling.run_groupwise_stats` | `<inputs.json>` | `<groups.json>` | stats JSON | 3 指标全 Shapiro-Wilk 决策树检验 |

## 输入文件格式约定

### `--input`（单文件）
直接传 EthoVision 导出 `.txt` 文件的路径。（仅通用脚本使用）

### `--inputs`（多文件聚合）
传一个 JSON 文件路径，内容是文件路径数组。Shoaling 的 compute 脚本都需要多文件：

```json
["/mnt/user-data/uploads/fish1.txt", "/mnt/user-data/uploads/fish2.txt", "/mnt/user-data/uploads/fish3.txt", "/mnt/user-data/uploads/fish4.txt", "/mnt/user-data/uploads/fish5.txt"]
```

### `--groups`
传一个 JSON 文件路径，内容是分组映射：

```json
{
  "control": ["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5"],
  "treatment": ["Subject 6", "Subject 7", "Subject 8", "Subject 9", "Subject 10"]
}
```

subject 名称由 EthoVision 文件 header 的 `对象名称` 字段决定。

## 实验设计决策树

根据 lead 提供的 `实验设计` 字段裁剪脚本列表：

### n=1 组（单组描述性）
- ✅ 跑全部 `compute_*.py`（3 个 shoaling 核心）
- ✅ 跑 `_common.plot_trajectory --inputs` 多文件版
- ✅ 跑 `_common.compute_distance_moved` + `_common.compute_velocity_stats` for each subject
- ✅ 跑 `plot_box_iid`（无 groups 参数也可，看个体间分布）
- ❌ 跳过 `run_groupwise_stats`（无组间对比）

### n_per_group ≥ 1（有分组）
- ✅ 跑全部脚本
- ⚠️ n_per_group < 3 → 统计在 handoff 标注 `data_quality_warnings: 样本量不足，统计仅供参考`

### 用户特殊需求
- "只要轨迹图" → 仅跑 `_common.plot_trajectory --inputs`
- "跳过统计" → 跳过 `run_groupwise_stats`

## handoff JSON 必须字段

`${workspace_path}/handoff_code_executor.json` schema:

```json
{
  "paradigm": "shoaling",
  "per_subject": {
    "Subject 1": {"distance_moved": 450.2, ...},
    ...
  },
  "group_metrics": {
    "inter_individual_distance": {"mean_iid_mean": 15.3, "mean_iid_std": 5.1},
    "nearest_neighbor_distance": {"mean_nnd": 8.2, "std_nnd": 3.1},
    "group_polarity": {"mean_polarity": 0.72, "std_polarity": 0.15}
  },
  "charts": ["/mnt/user-data/workspace/outputs/shoaling_box.png", ...],
  "statistics": { /* 直接复制 run_groupwise_stats 的输出 JSON，可选 */ },
  "data_quality_warnings": [ /* 见下方 */ ],
  "errors": [ /* 脚本执行报错记录 */ ]
}
```

### data_quality_warnings 触发条件
- subject 数 < 5（shoaling 标准 5 鱼）→ `{"severity": "warning", "message": "鱼群 subject 数 < 5，IID 计算可能不准确"}`
- IID 标准差过大（某组）→ `{"severity": "warning", "message": "鱼群松散，可能受应激影响"}`
- 某指标返回 None → `{"severity": "critical", "message": "<metric> 计算失败，检查数据完整性"}`

## 错误处理

脚本返回 non-zero 退出码或 stderr 非空时：

| 错误模式 | 处理 |
|---|---|
| `ValueError: must be a JSON array` | inputs JSON 格式错 → 检查 write_file 生成的 JSON 内容 |
| `FileNotFoundError: <path>` | 路径不存在 → 用 `ls` 核对 |
| subject 数 < 2 | `compute_inter_individual_distance` 无法计算配对距离 → 返回 None |
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

- 斑马鱼天然群游行为：鱼群紧凑度（IID ↓）＝ 焦虑 ↓
- 极性（polarity）衡量游动方向一致性：0 = 随机，1 = 完全一致
- 标准实验：5 鱼同缸、5-10 分钟、自由游动
- 关键混杂因素：缸大小、水温、光照强度
- 与其他范式的关系：shoaling 是焦虑/社会行为的综合指标，常与 NTT (Novel Tank Test) 配合使用
- 详细领域知识：见 `docs/review-packages/2026-04-29-ev19-templates/by-experiment/shoaling.md`
