# 2026-05-27 Noldus JS 示例库深挖 + P3 实施 handoff

## 背景

Noldus 官方 GitHub 仓库 `https://github.com/noldus/EthoVision-JavaScriptCustomAnalysis` 包含 45 个 EthoVision XT 19 即用型 JS 自定义变量脚本。每个是 `.txt` 文件（内容为 JS 代码，复制粘贴到 EV19 Analysis Profile 中使用），分为 3 个目录。

这 45 个脚本是 **Noldus 官方编写的行为学分析算法**，代表了行为学领域的标准计算方法。

## 仓库结构

```
EthoVision-JavaScriptCustomAnalysis/
├── Activity (pixel change)/     # 3 个脚本：全 arena 像素变化分析
│   ├── Activity (Pixel change) - Continuous.txt
│   ├── Activity (Pixel change) - Smoothed - Continuous.txt
│   └── Number of pixels changed - Continuous.txt
├── Single-subject analysis/     # 32 个脚本：单 subject 行为分析
│   ├── Acceleration - Smoothed - Continuous.txt
│   ├── Angle formed by center point and nose point - Continuous.txt
│   ├── Body Area - Continuous.txt
│   ├── Body Area - Running Average - Continuous.txt
│   ├── Body Length - Direct - Continuous.txt
│   ├── Body Length - Sum of segments - Continuous.txt
│   ├── Body orientation in side view - Fish - Continuous.txt
│   ├── Body orientation in side view - Fish head down - State.txt
│   ├── Body orientation in side view - Fish head up - State.txt
│   ├── Center-Nose length - Continuous.txt
│   ├── Distance moved - Cumulative in last k samples - Continuous.txt
│   ├── Find when a proportion of zones was visited - Event.txt
│   ├── Head direction - Absolute - Continuous.txt
│   ├── Heading - Smoothed - Continuous.txt
│   ├── Heading relative to vector from point1 to point2 - Continuous.txt
│   ├── Heading relative to vector from point1 to point2 - binned - State.txt
│   ├── Mobility in interval defined by Devices - Continuous.txt
│   ├── Non-movement bouts with a minimum duration - State.txt ★
│   ├── Percentage of correct choices (zone entries) - Continuous.txt
│   ├── Percentage of correct choices reached (zone entries) - Event.txt
│   ├── Split the track in bins based on velocity - State.txt ★
│   ├── Time to reach percentage of correct choices (zone entries) - State.txt
│   ├── Turn Angle with Distance moved filter - Continuous.txt ★
│   ├── Turn angle - Based on center point - Running Total - Continuous.txt
│   ├── Turn angle - Based on head direction - Continuous.txt
│   ├── Turn angle - Based on head direction - Running Total - Continuous.txt
│   ├── X and Y coordinates - Continuous.txt
│   ├── X and Y coordinates of a zone center - Continuous.txt
│   ├── X and Y coordinates rescaled - Continuous.txt
│   ├── Zone transitions - Sum up transitions of different type - Event.txt
│   ├── Zone visits when at least two body points in zone - State.txt
│   └── Zone visits with a minimum duration - State.txt ★
└── Multi-subject analysis/      # 13 个脚本：多 subject 交互分析
    ├── Aggregation behavior - 5 Subjects - State.txt
    ├── Approaching - State.txt
    ├── Coordinates of Largest subject in a group.txt
    ├── Following - State.txt
    ├── Inter Individual Distance IID - Continuous.txt
    ├── Leaving - State.txt
    ├── More than k subjects in a zone - State.txt
    ├── Nearest Neighbour Distance NND - Continuous.txt
    ├── Number of Subjects - In a zone - Continuous.txt
    ├── Orientation of focal subject relative to another subject - Continuous.txt
    ├── Social Contact - Any body point - State.txt
    ├── Social Contact - Each body point - State.txt
    └── Velocity higher than threshold for two subjects - State.txt
```

## 需要移植到 ethoinsight 的脚本（按优先级）

### Priority 1 — TST/FST 第三种 immobility 检测路径

#### 1a. Non-movement bouts with a minimum duration

**文件**: `Single-subject analysis/Non-movement bouts with a minimum duration - State.txt`
**算法**: velocity ≤ 30mm/s 持续 ≥ 25 帧 → immobility
**输入**: center-point (x,y) + GetSampleTime()
**与现有路径的关系**:

| 路径 | 输入 | 方法 | 优先级 |
|------|------|------|--------|
| 主路径 | mobility_state 列 | EV19 预分析（用户配阈值） | 1st |
| pendulum fallback | activity 列 | 自相关周期性检测 | 2nd |
| **velocity fallback（新增）** | **x_center + y_center** | **velocity 阈值 + min duration** | **3rd** |

**改动**:
- `metrics/_common.py`: 新增 `_resolve_immobile_from_velocity()` 函数
- `_resolve_immobile_series` 的 fallback 链: mobility_state → pendulum(activity) → velocity(x_center, y_center)
- 移植算法: distance = √(dx²+dy²), velocity = distance/dt, counter ≥ min_duration
- 可配置参数: VELOCITY_THRESHOLD=30 (mm/s), MIN_DURATION=25 (samples), 帧率自适应
- `ev19-dependent-variables.md` §14 补充 velocity-based immobility 说明

#### 1b. Smoothed Activity

**文件**: `Activity (pixel change)/Activity (Pixel change) - Smoothed - Continuous.txt`
**算法**: GetPixelChange() 的 SMA 滑动平均，默认窗口=10
**用途**: activity_intensity 图可选平滑，减少单帧噪点
**改动**: `charts.py` activity_intensity_plot 增加 `smooth_window` 参数（None=不平滑）

### Priority 2 — OFT/EPM 运动模式分析

#### 2a. Turn angle with distance filter ★

**文件**: `Single-subject analysis/Turn Angle with Distance moved filter - Continuous.txt`
**算法**: TurnAngle(p0,p1,p2)，但 point 需要移动 ≥ 1mm 才参与计算（排除静止噪声）
**输出**: 转角序列（度），可用于统计 mean/std/median turn angle
**改动**:
- `metrics/_common.py`: 新增 `compute_turn_angle_stats(df)` — 使用已有 heading 列做 diff 得转角
- `catalog/*.yaml`: EPM/OFT 新增 optional_metric `turn_angle_stats`
- `ev19-dependent-variables.md`: 补充 TurnAngle 公式和 distance filter 说明

#### 2b. Head direction (已有 heading 列)

**文件**: `Single-subject analysis/Head direction - Absolute - Continuous.txt`
**当前状态**: parser 已有 `"方向": "heading"` mapping，heading 列在 EV19 导出中以度为单位
**改动**:
- `metrics/_common.py`: 新增 `compute_heading_stats(df)` — circular mean/std/median
- 注意: 需要用 circular statistics（scipy.stats.circmean），因为 359° 和 1° 的均值是 0° 不是 180°

#### 2c. Velocity bins

**文件**: `Single-subject analysis/Split the track in bins based on velocity - State.txt`
**用途**: 按速度分箱（0-100, 100-200... mm/s），统计各箱时间占比。用于 locomotion 分析
**改动**: `metrics/_common.py` 新增 `compute_velocity_bins(df, bin_edges=None)`

### Priority 3 — Body measurement

#### 3a. Body length (sum of segments)

**文件**: `Single-subject analysis/Body Length - Sum of segments - Continuous.txt`
**算法**: Nose→Center 距离 + Center→Tail 距离 = 身体长度（mm）
**用途**: 当 elongation 列缺失时的替代方案；需要 nose/tail 检测
**改动**: `metrics/_common.py` 新增 `compute_body_length(df)`

#### 3b. Body area stats

**文件**: `Single-subject analysis/Body Area - Continuous.txt`
**当前状态**: parser 已有 `"区域": "body_area"` mapping
**改动**: `metrics/_common.py` 新增 `compute_body_area_stats(df)` — mean/std/median

### Priority 4 — 社交/鱼类范式（未来）

#### 4a. Inter-Individual Distance (IID)

**文件**: `Multi-subject analysis/Inter Individual Distance IID - Continuous.txt`
**算法**: 焦点个体到所有其他个体 center-point 的平均欧氏距离
**用途**: Shoaling（鱼群）、社交范式

#### 4b. Nearest Neighbour Distance (NND)

**文件**: `Multi-subject analysis/Nearest Neighbour Distance NND - Continuous.txt`
**算法**: 焦点个体到最近其他个体的距离

#### 4c. Social Contact

**文件**: `Multi-subject analysis/Social Contact - Any body point - State.txt`
**文件**: `Multi-subject analysis/Social Contact - Each body point - State.txt`
**算法**: 两个 body point 距离 < 阈值 → social contact

#### 4d. Body orientation in side view (Fish)

**文件**: `Single-subject analysis/Body orientation in side view - Fish - Continuous.txt`
**算法**: 检测鱼类的身体朝向（侧视）

### Priority 5 — 学习记忆范式（未来）

- Percentage of correct choices（zone entries）
- Time to reach percentage of correct choices
- Zone transitions（sum up transitions of different type）

### Priority 6 — Zone 分析增强

#### 6a. Zone visits with a minimum duration

**文件**: `Single-subject analysis/Zone visits with a minimum duration - State.txt`
**算法**: 进入 zone 持续 ≥ N 帧才算 visit（排除短暂穿越）
**用途**: 减少 EPM/OFT 的 zone entry 噪声
**改动**: `metrics/_common.py` 的 `_count_zone_entries` 增加 `min_duration_frames` 参数

## 当前代码状态

### 已有实现（不需要改）

| 功能 | 覆盖 | 文件 |
|------|------|------|
| Body area | parser mapping `"区域": "body_area"` | `utils.py` |
| Heading | parser mapping `"方向": "heading"` | `utils.py` |
| Distance moved | EV19 内置 `distance_moved` 列 | 直接使用 |
| Elongation | parser mapping `"伸长": "elongation"` | `utils.py` |
| Immobility (state) | mobility_state 列 + pendulum fallback | `_common.py` |
| Activity | `_find_activity_column` + pendulum detect | `_common.py` |

### 需要新增

| 功能 | 新文件/函数 | 优先级 |
|------|------------|--------|
| Immobility (velocity) | `_common.py::_resolve_immobile_from_velocity` | P1 |
| Smoothed activity | `charts.py::activity_intensity_plot` 加 smooth_window | P1 |
| Turn angle stats | `_common.py::compute_turn_angle_stats` | P2 |
| Heading stats (circular) | `_common.py::compute_heading_stats` | P2 |
| Velocity bins | `_common.py::compute_velocity_bins` | P2 |
| Body length (segments) | `_common.py::compute_body_length` | P3 |
| Body area stats | `_common.py::compute_body_area_stats` | P3 |
| Zone min duration | `_common.py::_count_zone_entries` 加 min_duration | P6 |

## 实施建议

### 一个 PR：P1 + P2（核心新增，对现有 6 范式有直接价值）

1. **immobility_velocity**: `_common.py` 新增函数 + 接入 fallback 链
2. **smoothed activity**: `charts.py` 加 smooth_window 参数
3. **turn_angle_stats**: `_common.py` 新增 + catalog EPM/OFT 注册
4. **heading_stats**: `_common.py` 新增（需要 scipy circular stats）+ catalog 注册
5. **velocity_bins**: `_common.py` 新增 + catalog OFT 注册

### 第二个 PR：P3 + P6

6. **body_length**: `_common.py` 新增
7. **body_area_stats**: `_common.py` 新增
8. **zone_min_duration**: `_common.py` 现有函数增强

### 未来 PR：P4 + P5（社交/学习记忆范式）

## 关键参考

- 仓库地址: `https://github.com/noldus/EthoVision-JavaScriptCustomAnalysis`
- 所有文件通过 `https://raw.githubusercontent.com/noldus/EthoVision-JavaScriptCustomAnalysis/main/<目录>/<文件名>` 直接读取
- EV19 JS API 参考: `docs/review-packages/2026-0521-feedbacks/tstYoyo/manual/Commands and functions for JavaScript variables.html`
- 公式参考: `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/ev19-dependent-variables.md`
- 已有指标: `packages/ethoinsight/ethoinsight/metrics/_common.py`
- 图表: `packages/ethoinsight/ethoinsight/charts.py`
- Catalog: `packages/ethoinsight/ethoinsight/catalog/*.yaml`

## 注意事项

1. **JS → Python 移植要点**: EV19 JS 的 null 检查 → Python 的 None/pd.NA 检查；EV19 帧率 25fps 假定 → 用 `_estimate_dt()` 自适应
2. **Circular statistics**: heading 的 mean/std 必须用 `scipy.stats.circmean/circstd`，普通 mean 在 359°↔0° 边界会算错
3. **Catalog 更新**: 新增指标必须在 catalog YAML 中注册 `optional_metrics`，charts 同理
4. **ev19-dependent-variables.md 同步更新**: 新增指标的计算公式同步写入 §15+
5. **测试**: 每个新函数至少 2 个测试（happy path + edge case）
6. **git 操作**: 必须用 `git -C /home/wangqiuyang/noldus-insight`（CWD 在符号链接内）
7. **immobility fallback 链顺序**: mobility_state → pendulum(activity) → velocity。不能改变优先级，否则可能覆盖用户配置的阈值。
