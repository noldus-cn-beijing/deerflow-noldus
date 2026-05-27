# 2026-05-27 Noldus JS 示例库深挖 + P3 补全 handoff（v3）

## 当前状态

```
dev HEAD: 751428e4

已完成: P0 / P1 / P2 全部
P3 基础: body_elongation / head_direction / turn_angle / rose_plot 函数+测试已完成
P3 遗留: catalog YAML 注册 + CLI 脚本 wrapper 未做

ethoinsight: 436 passed  /  agent backend: 3043 passed
```

---

## Phase A: P3 基础补全（注册壳）

已有 4 个函数在 `metrics/_common.py` + `charts.py`，只需补 catalog 注册和 CLI 入口：

| 函数 | catalog 加到 | CLI 脚本 |
|------|-------------|---------|
| `compute_body_elongation_stats` | `epm.yaml` + `oft.yaml` optional_metrics | `scripts/epm/` + `scripts/oft/` |
| `compute_head_direction_stats` | `epm.yaml` + `zero_maze.yaml` optional_metrics | `scripts/epm/` + `scripts/zero_maze/` |
| `compute_turn_angle_stats` | `oft.yaml` + `epm.yaml` optional_metrics | `scripts/oft/` + `scripts/epm/` |
| `rose_plot` | `epm.yaml` + `zero_maze.yaml` charts (confidence: optional) | `scripts/epm/` + `scripts/zero_maze/` |

每组 CLI 脚本 2 行核心代码：
```python
from ethoinsight.metrics._common import compute_xxx
value = compute_xxx(df)
```

---

## Phase B: Noldus JS 示例库算法移植

Noldus 官方仓库 `https://github.com/noldus/EthoVision-JavaScriptCustomAnalysis`

### B1 — immobility_velocity ★ 最高优先

来源文件: `Single-subject analysis/Non-movement bouts with a minimum duration - State.txt`

**完整 JS 代码**:
```javascript
const g_MinimumDuration = 25;
const g_VelocityThreshold = 30;
var g_SleepDuration = 0;
var g_Sleep, g_ptPrev, g_tPrev;

function Distance(pt1, pt2) {
    var dx = pt1.x - pt2.x;
    var dy = pt1.y - pt2.y;
    return Math.sqrt(dx * dx + dy * dy);
}

function Process() {
   var pt = GetCenter();
   var t  = GetSampleTime();
   if (pt) {
       if (g_ptPrev) {
          var distance = Distance(pt, g_ptPrev);
          var velocity = distance / (t - g_tPrev);
          if (velocity <= g_VelocityThreshold) {
              g_SleepDuration += 1;
          } else {
              g_SleepDuration = 0;
          }
       }
       g_Sleep = (g_SleepDuration >= g_MinimumDuration);
       g_ptPrev = pt;
       g_tPrev  = t;
   }
   SetOutput(g_Sleep);
}
```

**算法**: 逐帧计算 center-point 位移速度，velocity ≤ 30mm/s → 计数器+1，超过阈值 → 计数器归零。计数器 ≥ 25 帧 → 输出 immobile=true。**关键**: 短暂停顿（<25帧）被过滤，不会算作 immobility。

**immobility fallback 链当前**: `mobility_state → pendulum(activity) → None`
**新增后**: `mobility_state → pendulum(activity) → velocity(x_center, y_center)`

**移植要点**:
- `metrics/_common.py`: 新增 `_resolve_immobile_from_velocity(df)`，在 `_resolve_immobile_series` 中接入作为最后 fallback
- VELOCITY_THRESHOLD = 30（mm/s），MIN_DURATION = 25（samples），帧率自适应（乘 `_estimate_dt()/0.04`）
- `ev19-dependent-variables.md` §14 补充
- `tests/test_pendulum.py` 新增 2 个测试

---

### B2 — smoothed_activity

来源文件: `Activity (pixel change)/Activity (Pixel change) - Smoothed - Continuous.txt`

**完整 JS 代码**:
```javascript
const Smoothing = 10;

function RunningAverageQueueMissing(nSamplesMax) {
    this.m_nMax     = nSamplesMax;
    this.m_aValues  = new Array;
    this.Add = function(value) {
        if (this.m_aValues.length >= this.m_nMax)
            this.m_aValues.shift();
        this.m_aValues.push(value);
    }
    this.GetAvg = function() {
        var sum = 0; var nTotal = 0;
        for (var i = 0; i < this.m_aValues.length; i++) {
            if (this.m_aValues[i] != null) {
                sum += this.m_aValues[i];
                nTotal++;
            }
        }
        if (nTotal == 0) return null;
        return sum / nTotal;
    }
}

var g_queue = new RunningAverageQueueMissing(Smoothing);

function Process() {
   var Activity = GetPixelChange();
   g_queue.Add(Activity);
   var avg = g_queue.GetAvg();
   if (avg != null) { SetOutput(avg); }
   else { SetOutputMissing(); }
}
```

**算法**: 定长队列（默认 10）存最近 N 个像素变化值。取平均时跳过 null，全 null → 输出 missing。Simple Moving Average，所有样本等权重。

**移植要点**:
- `metrics/_common.py`: 新增 `compute_smoothed_activity(df, window=10)` — 用 `pd.Series.rolling(window, min_periods=1).mean()`
- `charts.py`: `activity_intensity_plot` 增加 `smooth_window` 参数，None=不光滑（向后兼容），>0 时先平滑再画
- 不需要改 catalog

---

### B3 — turn_angle_filtered

来源文件: `Single-subject analysis/Turn Angle with Distance moved filter - Continuous.txt`

**完整 JS 代码**:
```javascript
const g_Distance = 1;  // mm — 小于此位移不参与转角计算
var g_aPoints = new Array;

function toDegrees(angle) { return angle * (180 / Math.PI); }
function Dm(pt1, pt2) {
    var dx = pt1.x - pt2.x;
    var dy = pt1.y - pt2.y;
    return Math.sqrt(dx * dx + dy * dy);
}

function Process() {
   var pt = GetCenter();
   if (pt) {
       var bAdd = true;
       if (g_aPoints.length > 0) {
           var ptPrev = g_aPoints[g_aPoints.length - 1];
           bAdd = Dm(pt, ptPrev) > g_Distance;
       }
       if (bAdd) {
           g_aPoints.push(pt);
           if (g_aPoints.length > 3)
               g_aPoints.shift();
       }
   }
   if (g_aPoints.length == 3) {
       var ta = TurnAngle(g_aPoints[0], g_aPoints[1], g_aPoints[2]);
       if (ta != null) { SetOutput(toDegrees(ta)); }
       else { SetOutputMissing(); }
   }
}
```

**算法**: 维护最近 3 个 point（滑动窗口）。新 point 只有位移 > 1mm 才入队（排除静止抖动）。窗口满 3 点后用 `TurnAngle(p0,p1,p2)` 计算转角（弧度→度）。

**与已有 `compute_turn_angle_stats` 的区别**: 已有函数用 heading 列做差分，可能包含静止时的微小波动。这个版本用 **位移过滤** 排除噪声。

**移植要点**:
- `metrics/_common.py`: 新增 `compute_turn_angle_filtered(df, min_displacement_mm=1.0)` — 直接用 x_center/y_center 计算
- `catalog/oft.yaml`: optional_metrics 新增
- `scripts/oft/compute_turn_angle_filtered.py`

---

### B4 — velocity_bins

来源文件: `Single-subject analysis/Split the track in bins based on velocity - State.txt`

**算法**: 每帧计算 velocity = distance/dt，按预设边界分箱（如 0-100, 100-200... mm/s）。输出每个 bin 的帧数/时间占比。

**移植要点**:
- `metrics/_common.py`: 新增 `compute_velocity_bins(df, bin_edges=None)` — bin_edges 默认 `[0, 50, 100, 200, 400]` mm/s
- 返回值: `dict[bin_label, ratio]`
- `catalog/oft.yaml`: optional_metrics 新增
- `scripts/oft/compute_velocity_bins.py`

---

### B5 — body_length_segments

来源文件: `Single-subject analysis/Body Length - Sum of segments - Continuous.txt`

**完整 JS 代码**:
```javascript
function Distance(pt1, pt2) {
    var dx = pt1.x - pt2.x;
    var dy = pt1.y - pt2.y;
    return Math.sqrt(dx * dx + dy * dy);
}

function Process() {
   var length;
   var ptNose = GetNose();
   var ptCog  = GetCenter();
   var ptTail = GetTail();
   if (ptNose != null && ptCog != null && ptTail != null) {
      length  = Distance(ptNose, ptCog);
      length += Distance(ptCog, ptTail);
   }
   if (length != null) { SetOutput(length); }
   else { SetOutputMissing(); }
}
```

**算法**: Nose→Center 欧氏距离 + Center→Tail 欧氏距离 = 身体总长度（mm）。需要 Nose、Center、Tail 三个 body point 全部检测到。

**移植要点**:
- `metrics/_common.py`: 新增 `compute_body_length(df)` — 使用 `x_nose/y_nose + x_center/y_center + x_tail/y_tail` 列
- parser 已有这些 mapping: `"X 鼻点": "x_nose"`, `"Y 鼻点": "y_nose"`, `"X 尾": "x_tail"`, `"Y 尾": "y_tail"`
- 注意: 不是所有 EV19 项目都启用多 body point 检测。函数需检查所有 6 列都存在
- `catalog/*.yaml`: 只在有 nose/tail 检测的范式注册（EPM? OFT?）
- `scripts/_common/compute_body_length.py`

---

### B6 — zone_min_duration

来源文件: `Single-subject analysis/Zone visits with a minimum duration - State.txt`

**算法**: 进入 zone 后必须持续 ≥ N 帧才算一次 visit。排除短暂穿越（如动物跑过中心区但没停留）。

**移植要点**:
- `metrics/_common.py`: 修改 `_count_zone_entries` 增加 `min_duration_frames` 参数，默认 0（不启用，向后兼容）
- 当 min_duration_frames > 0 时，bout 长度 < N → 不计入 entry
- `metrics/epm.py` + `metrics/oft.py`: 透传新参数

---

### B7 — cumulative_distance_k

来源文件: `Single-subject analysis/Distance moved - Cumulative in last k samples - Continuous.txt`

**算法**: 滑动窗口内累计移动距离（最近 K 帧的 distance_moved 之和）。反映"动物最近一段时间的活动量"，不是全场累计。

**对 OFT 的价值**: 可以看动物在实验不同阶段的实时活动水平变化（习惯化评估）。

**移植要点**:
- `metrics/_common.py`: 新增 `compute_cumulative_distance(df, window_samples=25)` — `distance_moved.rolling(window).sum()`
- `scripts/oft/compute_cumulative_distance.py`

---

### B8 — acceleration_stats

来源文件: `Single-subject analysis/Acceleration - Smoothed - Continuous.txt`

**算法**: velocity 的差分 → 加速度。先平滑 velocity（SMA），再求差分，减少帧间噪声。

**对 OFT/EPM 的价值**: 加速度反映运动模式变化——高加速度 = 突然启动/转向，低加速度 = 匀速。thigmotaxis 老鼠沿墙走时加速度稳定，进入中心区探索时加速度变化大。

**移植要点**:
- `metrics/_common.py`: 新增 `compute_acceleration_stats(df, smooth_window=5)` — diff of smoothed velocity
- `scripts/oft/compute_acceleration_stats.py`

---

### B9 — zone_visits_body_points

来源文件: `Single-subject analysis/Zone visits when at least two body points in zone - State.txt`

**算法**: 需要 ≥ 2 个 body point（如 center + nose）同时在 zone 内才算 visit。比单 point 判定更严格，排除"只有鼻子探进去"的假进入。

**对 EPM/OFT 的价值**: 论文中 open arm entry 的标准定义是"四爪都进入开放臂"。双 body point 是 EV19 能做到的最接近的近似。

**移植要点**:
- `metrics/epm.py`: 新增 `compute_open_arm_entry_two_points(df)` — 要求 `in_zone_open_arms_center AND in_zone_open_arms_nose` 同时为 1
- `catalog/epm.yaml`: optional_metrics 新增

---

### B10 — heading_smoothed

来源文件: `Single-subject analysis/Heading - Smoothed - Continuous.txt`

**算法**: heading 值的 SMA 平滑。heading 原始值在 ±180° 边界有 wrap-around，平滑前需要做 circular unwrap。

**移植要点**:
- `metrics/_common.py`: 新增 `compute_heading_smoothed(df, window=5)` — circular unwrap → rolling mean → re-wrap

---

### B11 — center_nose_length

来源文件: `Single-subject analysis/Center-Nose length - Continuous.txt`

**算法**: Center-point 到 Nose-point 的欧氏距离（mm）。用于衡量动物"探头"行为的幅度。

**对 EPM 的价值**: EPM 中老鼠在封闭臂内向开放臂探头时，center-nose 距离增大。可作为 risk-assessment 的补充指标。

**移植要点**:
- `metrics/_common.py`: 新增 `compute_center_nose_length(df)` — √((x_center-x_nose)² + (y_center-y_nose)²) 的 mean/std
- 需要 x_nose/y_nose 列（parser 已有 mapping）

---

### B12 — angle_center_nose

来源文件: `Single-subject analysis/Angle formed by center point and nose point - Continuous.txt`

**算法**: Center→Nose 向量与水平轴的夹角。反映身体局部朝向（不同于整体运动方向 heading）。

**移植要点**:
- `metrics/_common.py`: 新增 `compute_center_nose_angle(df)` — atan2(y_nose-y_center, x_nose-x_center) 转度
- 需要 nose 检测

---

## 完整优先级矩阵

### 对 v0.1 6 范式直接有用（Phase B 实施）

| # | 算法 | 适用范式 | 价值 |
|---|------|----------|------|
| B1 | immobility_velocity | TST/FST | 第三种 immobility 检测路径 |
| B2 | smoothed_activity | TST/FST | 活动强度图减噪 |
| B3 | turn_angle_filtered | OFT/EPM | 位移过滤排除静止噪声 |
| B4 | velocity_bins | OFT | locomotion profiling |
| B5 | body_length_segments | EPM/OFT | 无 elongation 列时的替代 |
| B6 | zone_min_duration | EPM/OFT/LDB | 排除短暂穿越的噪声 |
| B7 | cumulative_distance_k | OFT | 习惯化评估 |
| B8 | acceleration_stats | OFT/EPM | 运动模式变化 |
| B9 | zone_visits_body_points | EPM | 双 body point 严格 zone 进入 |
| B10 | heading_smoothed | EPM/Zero Maze | circular unwrap + 平滑 |
| B11 | center_nose_length | EPM | 探头行为幅度 |
| B12 | angle_center_nose | EPM | 身体局部朝向 |

### 社交/鱼类范式（未来，Phase C）

| 算法 | 用途 |
|------|------|
| Inter Individual Distance IID | 鱼群 shoaling |
| Nearest Neighbour Distance NND | 最近邻距离 |
| Social Contact (Any/Each body point) | 社交接触检测 |
| Following / Leaving / Approaching | 社交行为分类 |
| Aggregation behavior | 聚集行为 |
| Body orientation in side view (Fish) | 鱼类侧视朝向 |

### 学习记忆范式（未来，Phase C）

| 算法 | 用途 |
|------|------|
| Percentage of correct choices | MWM/Barnes/Y-maze 正确率 |
| Time to reach % correct choices | 学习曲线 |
| Zone transitions | 区域切换汇总 |
| Find when proportion of zones visited | 探索完成度 |

## 实施顺序（更新）

```
Phase A (1 PR): P3 基础 catalog + CLI 补全
Phase B PR1: B1 + B2（immobility_velocity + smoothed_activity）
Phase B PR2: B3 + B4 + B7 + B8（转角/速度/距离/加速度，运动模式 4 件套）
Phase B PR3: B5 + B11 + B12（身体测量 3 件套：body_length + center_nose + angle）
Phase B PR4: B6 + B9 + B10（zone/heading 增强）
Phase C (未来): 社交 + 学习记忆范式

## 关键文件

| 类别 | 路径 |
|------|------|
| P3 已有函数 | `metrics/_common.py`（grep `elongation\|head_direction\|turn_angle`） |
| P3 已有测试 | `tests/test_metrics_body_elongation.py` 等 4 个 |
| Rose plot | `charts.py::rose_plot` |
| 已有 immobility | `metrics/_common.py::_resolve_immobile_series` + `_resolve_immobile_from_activity` |
| 已有 activity_intensity | `charts.py::activity_intensity_plot` |
| 已有 entry 计数 | `metrics/epm.py::_count_zone_entries` |
| EV19 JS API 参考 | `docs/review-packages/2026-0521-feedbacks/tstYoyo/manual/Commands and functions for JavaScript variables.html` |
| EV19 公式参考 | `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/ev19-dependent-variables.md` |
| Catalog | `catalog/*.yaml`（epm/oft/zero_maze/ldb/tst/fst） |
| CLI 模板参考 | `scripts/tst/compute_immobility_time.py` |
| Parser 列映射 | `utils.py` COLUMN_MAP |

## 注意事项

1. **CLI 脚本模板**: 参考 `scripts/tst/compute_immobility_time.py`，2 行核心代码搞定
2. **Catalog requires_columns 是 AND**: 所有 pattern 必须匹配。B5 body_length 需要 `x_nose + x_tail`，不是所有范式都有
3. **immobility fallback 链顺序不能改**: `mobility_state → pendulum(activity) → velocity` 优先级固定
4. **JS null → Python None/pd.NA**: JS 版 `if (x != null)` → Python 版 dropna()
5. **帧率自适应**: JS 版假定 25fps → Python 版用 `_estimate_dt()` 自适应
6. **B3 位移过滤**: JS 版 `> g_Distance` (strict) → Python 版 `> min_displacement_mm` (same)
7. **B2 光滑窗口**: JS 版用定长队列手动实现 → Python 版直接用 `pd.Series.rolling()`
8. **git**: 必须用 `git -C /home/wangqiuyang/noldus-insight`
9. **测试**: `cd /home/wangqiuyang/noldus-insight/packages/ethoinsight && .venv/bin/python -m pytest tests/ -q`
10. **backend 测试**: `cd /home/wangqiuyang/noldus-insight/packages/agent/backend && source .venv/bin/activate && make test`
