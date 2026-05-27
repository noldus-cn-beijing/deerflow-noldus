# 2026-05-27 Noldus JS 算法增强 handoff（v4 final）

## 当前状态

```
dev HEAD: ceec0581
已完成: P0 / P1 / P2 / P3（含 catalog + CLI） / B1（velocity fallback）
ethoinsight: 439 passed  /  agent backend: 3043 passed
```

## 背景

Noldus 官方仓库 `https://github.com/noldus/EthoVision-JavaScriptCustomAnalysis` 包含 48 个 EV19 即用型 JS 分析脚本。这些是 Noldus 官方的行为学分析方法。

我们已逐项对照完成并修正了最紧急的问题（P0 bug、P1 immobility 三条路径、P2 图表增强、P3 基础指标、chart 置信度分级）。

接下来做 B2-B12：对照 Noldus 官方算法，修正偏差或补全缺失。

**原则**：
- 不建新 SSOT，不建新 tool，不改架构
- 每个算法：对照我们现有代码 → 如果缺失或不一致 → 修 Python 代码
- 每个 port 的算法：函数 + 测试 + catalog（如适用）+ CLI（如适用）同 PR 落地

---

## 待做算法（B2-B10）

### B2 — smoothed_activity

**Noldus JS**（`Activity (pixel change) - Smoothed - Continuous.txt`）：
```javascript
const Smoothing = 10;
function RunningAverageQueueMissing(nSamplesMax) { /* SMA queue */ }
function Process() {
   var Activity = GetPixelChange();
   g_queue.Add(Activity);
   var avg = g_queue.GetAvg();
   if (avg != null) SetOutput(avg); else SetOutputMissing();
}
```
**算法**: 定长队列（默认 10）做 SMA，跳过 null。

**我们的现状**: `activity_intensity_plot` 直接画原始 activity 值，没有平滑。
**改动**:
- `charts.py`: `activity_intensity_plot` 增加 `smooth_window=None` 参数（None=不平滑，>0 则先 `rolling(window).mean()`）
- 向后兼容，不改变 catalog

---

### B3 — turn_angle_filtered（带距离过滤的转角）

**Noldus JS**（`Turn Angle with Distance moved filter - Continuous.txt`）：
```javascript
const g_Distance = 1;  // mm — 位移小于此值不参与转角
function Process() {
   var pt = GetCenter();
   if (pt) {
       if (g_aPoints.length > 0) {
           bAdd = Dm(pt, ptPrev) > g_Distance;
       }
       if (bAdd) { g_aPoints.push(pt); if (g_aPoints.length > 3) g_aPoints.shift(); }
   }
   if (g_aPoints.length == 3) {
       var ta = TurnAngle(g_aPoints[0], g_aPoints[1], g_aPoints[2]);
       if (ta != null) SetOutput(toDegrees(ta)); else SetOutputMissing();
   }
}
```
**算法**: 滑动窗口 3 点计算 TurnAngle，新点位移 > 1mm 才入队（排除静止抖动）。

**我们的现状**: `compute_turn_angle_stats` 用 TurnAngle 列做统计，但没有距离过滤——静止时的微小波动被算进去了。
**改动**:
- `metrics/_common.py`: 新增 `compute_turn_angle_filtered(df, min_displacement_mm=1.0)` — 从 x_center/y_center 计算，位移 < 1mm 的点丢弃
- `catalog/oft.yaml`: optional_metrics 新增
- `scripts/oft/compute_turn_angle_filtered.py`
- 测试

---

### B4 — velocity_bins

**Noldus JS**（`Split the track in bins based on velocity - State.txt`）：
**算法**: 每帧计算 velocity，按预设边界分箱，输出各 bin 时间占比。

**我们的现状**: `compute_velocity_stats` 只有 mean/std/max/min/median，没有分布。
**改动**:
- `metrics/_common.py`: 新增 `compute_velocity_bins(df, bin_edges=None)` — 默认边界 `[0, 50, 100, 200, 400]` mm/s，返回 `{bin_label: ratio}`
- `catalog/oft.yaml`: optional_metrics 新增
- `scripts/oft/compute_velocity_bins.py`
- 测试

---

### B5 — body_length_segments

**Noldus JS**（`Body Length - Sum of segments - Continuous.txt`）：
```javascript
function Process() {
   var ptNose = GetNose(), ptCog = GetCenter(), ptTail = GetTail();
   if (ptNose && ptCog && ptTail) {
      length  = Distance(ptNose, ptCog);
      length += Distance(ptCog, ptTail);
   }
   SetOutput(length);
}
```
**算法**: Nose→Center + Center→Tail 的欧氏距离求和 = 身体长度（mm）。

**我们的现状**: 有 elongation（0-1 比例）但没有绝对身体长度。
**改动**:
- `metrics/_common.py`: 新增 `compute_body_length(df)` — 需要 x_nose/y_nose + x_center/y_center + x_tail/y_tail（parser 已有 mapping）
- 注意：不是所有范式都有多 body point 检测，函数需检查所有 6 列存在
- `catalog/*.yaml`: EPM/OFT 的 optional_metrics 新增
- `scripts/_common/compute_body_length.py`
- 测试

---

### B6 — zone_min_duration

**Noldus JS**（`Zone visits with a minimum duration - State.txt`）：
**算法**: 进入 zone 后持续 ≥ N 帧才算一次 visit，排除短暂穿越。

**我们的现状**: `_count_zone_entries` 数 0→1 跳变，1 帧掠过也算 entry（噪声）。
**改动**:
- `metrics/_common.py`: 修改 `_count_zone_entries` 增加 `min_duration_frames=0` 参数（0=不启用，向后兼容）
- 当 > 0 时，bout 长度 < N 不计入
- `metrics/epm.py` + `metrics/oft.py`: entry 函数透传参数

---

### B7 — cumulative_distance_k

**Noldus JS**（`Distance moved - Cumulative in last k samples - Continuous.txt`）：
**算法**: 滑动窗口内累计移动距离。`distance_moved.rolling(window).sum()`。
**用途**: OFT 习惯化——看动物在不同阶段的实时活动水平。
**改动**:
- `metrics/_common.py`: 新增 `compute_cumulative_distance(df, window_samples=25)`
- `scripts/oft/compute_cumulative_distance.py`

---

### B8 — acceleration_stats

**Noldus JS**（`Acceleration - Smoothed - Continuous.txt`）：
**算法**: 先平滑 velocity（SMA），再差分得加速度。高加速度 = 突然启动/转向。
**改动**:
- `metrics/_common.py`: 新增 `compute_acceleration_stats(df, smooth_window=5)`
- `scripts/oft/compute_acceleration_stats.py`

---

### B10 — heading_smoothed

**Noldus JS**（`Heading - Smoothed - Continuous.txt`）：
**算法**: heading 的 SMA，含 circular unwrap（±180° 边界处理）。
**改动**:
- `metrics/_common.py`: 新增 `compute_heading_smoothed(df, window=5)`

---

## 实施顺序

```
PR 1: B2 + B3（smoothed_activity + turn_angle_filtered）       — 运动平滑 2 件
PR 2: B4 + B7 + B8（velocity_bins + cumulative + acceleration） — 运动分析 3 件
PR 3: B5（body_length_segments）                                — 身体测量
PR 4: B6 + B10（zone_min_duration + heading_smoothed）           — zone/heading 增强
```

每 PR 独立，做完一个可以马上合。

B9（zone_visits_two_body_points）、B11（center_nose_length）、B12（angle_center_nose）优先级低，如果时间紧可以跳过。

## 注意

1. **帧率自适应**：Noldus JS 假定 25fps，Python 版用 `_estimate_dt()` 把所有"帧数"参数转为"时间秒"再换算回当前数据的帧数
2. **B5 需要 nose/tail 列**：不是所有数据都有，函数必须检查 6 列都存在
3. **B2 smooth_window=None 向后兼容**：默认不改变现有行为
4. **B6 zone_min_duration 默认 0**：不改变现有 entry 计数行为
5. **git**: 必须用 `git -C /home/wangqiuyang/noldus-insight`
6. **测试**: `cd /home/wangqiuyang/noldus-insight/packages/ethoinsight && .venv/bin/python -m pytest tests/ -q`
7. **backend 测试**: `cd /home/wangqiuyang/noldus-insight/packages/agent/backend && source .venv/bin/activate && make test`
