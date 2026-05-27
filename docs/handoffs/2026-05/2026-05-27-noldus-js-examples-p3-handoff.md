# 2026-05-27 Noldus JS 示例库深挖 + P3 补全 handoff（v2）

## 当前状态

P0/P1/P2 已全部完成。P3 基础版（body_elongation、head_direction、turn_angle、rose_plot）的函数和测试已写，但**缺少 catalog YAML 注册 + CLI 脚本 wrapper**。

```
当前 dev HEAD: 1e529799

已完成:
  P0: 3 bug 修复
  P1: pendulum detect + activity_intensity 修复 + struggle 双色 + _find_mobility_column 收紧
  P2: 4:3 宽高比 + EPM center point + LDB 区域进入分布图 + hesitation 扩展 + OFT group-aggregate
  P3 基础: body_elongation / head_direction / turn_angle / rose_plot 函数+测试（缺注册）

待做:
  Phase A: P3 基础补全（catalog + CLI）
  Phase B: Noldus JS 示例库新算法移植
```

## Phase A: P3 基础补全（catalog YAML + CLI wrapper）

P3 已有的 4 个函数和测试文件，需要补注册：

### A1. body_elongation_stats

**已有**: `metrics/_common.py::compute_body_elongation_stats` + `tests/test_metrics_body_elongation.py`
**需要**:
- `catalog/epm.yaml`: `optional_metrics` 新增 entry（SAP 检测是 EPM 经典补充指标）
- `catalog/oft.yaml`: 同上
- `scripts/epm/compute_body_elongation_stats.py` (CLI)
- `scripts/oft/compute_body_elongation_stats.py` (CLI)

### A2. head_direction_stats

**已有**: `metrics/_common.py::compute_head_direction_stats` + `tests/test_metrics_head_direction.py`
**需要**:
- `catalog/epm.yaml`: optional_metrics 新增（朝向开放臂的 risk-assessment）
- `catalog/zero_maze.yaml`: 同上
- `scripts/epm/compute_head_direction_stats.py` (CLI)
- `scripts/zero_maze/compute_head_direction_stats.py` (CLI)

### A3. turn_angle_stats

**已有**: `metrics/_common.py::compute_turn_angle_stats` + `tests/test_metrics_turn_angle.py`
**需要**:
- `catalog/oft.yaml`: optional_metrics 新增（thigmotaxis 旋转模式）
- `catalog/epm.yaml`: 同上
- `scripts/oft/compute_turn_angle_stats.py` (CLI)
- `scripts/epm/compute_turn_angle_stats.py` (CLI)

### A4. rose_plot

**已有**: `charts.py::rose_plot` + `tests/test_rose_plot.py`
**需要**:
- `catalog/epm.yaml`: charts 新增 `confidence: optional`
- `catalog/zero_maze.yaml`: charts 新增 `confidence: optional`
- `scripts/epm/plot_rose.py` (CLI)
- `scripts/zero_maze/plot_rose.py` (CLI)

---

## Phase B: Noldus JS 示例库新算法移植

Noldus 官方仓库 `https://github.com/noldus/EthoVision-JavaScriptCustomAnalysis` 含 45 个脚本。

所有文件通过 `https://raw.githubusercontent.com/noldus/EthoVision-JavaScriptCustomAnalysis/main/<目录>/<文件名>` 直接获取。

### B1 — immobility_velocity（TST/FST 第三种检测路径）★ 最高优先

**来源**: `Single-subject analysis/Non-movement bouts with a minimum duration - State.txt`
**算法**: velocity ≤ 30mm/s 持续 ≥ 25 帧 → immobility。从 center-point 计算 velocity = distance/dt
**用途**: 当 mobility_state 列缺失且 activity 列也缺失时的最后 fallback
**当前 immobility fallback 链**: `mobility_state → pendulum(activity) → ❌ None`
**新增后**: `mobility_state → pendulum(activity) → velocity(x_center, y_center)`

**改动**:
- `metrics/_common.py`: 新增 `_resolve_immobile_from_velocity(df)` + 接入 `_resolve_immobile_series` fallback 链
- 参数: VELOCITY_THRESHOLD=30 (mm/s, JS 版默认), MIN_DURATION=25 (samples), 帧率自适应
- `ev19-dependent-variables.md` §14 补充 velocity-based immobility 说明
- `tests/test_pendulum.py` 新增 2 个 velocity fallback 测试

### B2 — smoothed_activity（活动强度图平滑）

**来源**: `Activity (pixel change)/Activity (Pixel change) - Smoothed - Continuous.txt`
**算法**: GetPixelChange() 的 SMA 滑动平均，默认窗口=10
**用途**: activity_intensity 图可选平滑，减少单帧噪点

**改动**:
- `metrics/_common.py`: 新增 `compute_smoothed_activity(df, window=10)` — 返回平滑后的 activity series
- `charts.py`: `activity_intensity_plot` 增加 `smooth_window` 参数（None=不平滑，向后兼容）
- `catalog/tst.yaml` + `catalog/fst.yaml`: activity_intensity chart 不改变 requires_columns

### B3 — turn_angle_filtered（带 distance filter 的转角）

**来源**: `Single-subject analysis/Turn Angle with Distance moved filter - Continuous.txt`
**算法**: TurnAngle(p0,p1,p2)，但 point 需移动 ≥ 1mm 才参与计算（排除静止噪声）
**与已有 turn_angle_stats 的关系**: 已有函数计算的是 heading 列差分。这个新版本用 **distance filter** 排除微动噪声，更适合 OFT wall-following 分析。

**改动**:
- `metrics/_common.py`: 新增 `compute_turn_angle_filtered(df, min_displacement_mm=1.0)` — 从 x_center/y_center 计算
- `catalog/oft.yaml`: optional_metrics 新增
- `scripts/oft/compute_turn_angle_filtered.py`

### B4 — velocity_bins（速度分箱）

**来源**: `Single-subject analysis/Split the track in bins based on velocity - State.txt`
**算法**: 每帧按 velocity 分类到预设 bins（0-100, 100-200... mm/s），统计各 bin 时间占比
**用途**: OFT locomotion profiling

**改动**:
- `metrics/_common.py`: 新增 `compute_velocity_bins(df, bin_edges=None)`
- `catalog/oft.yaml`: optional_metrics 新增
- `scripts/oft/compute_velocity_bins.py`

### B5 — body_length_segments（身体长度，分段求和）

**来源**: `Single-subject analysis/Body Length - Sum of segments - Continuous.txt`
**算法**: Nose→Center 距离 + Center→Tail 距离 = 身体长度（mm）
**用途**: 当 elongation 列缺失时的替代。需要 nose/tail body point 检测。

**改动**:
- `metrics/_common.py`: 新增 `compute_body_length(df)` — 使用 x_nose/y_nose + x_center/y_center + x_tail/y_tail
- `catalog/*.yaml`: optional_metrics 新增（有 nose/tail 检测的范式）
- `scripts/_common/compute_body_length.py`

### B6 — zone_min_duration（Zone 进入最短持续）

**来源**: `Single-subject analysis/Zone visits with a minimum duration - State.txt`
**算法**: 进入 zone 持续 ≥ N 帧才算 visit（排除短暂穿越）
**用途**: 减少 EPM/OFT zone entry 噪声

**改动**:
- `metrics/_common.py`: `_count_zone_entries` 增加 `min_duration_frames` 参数（默认 0 = 不启用，向后兼容）
- `metrics/epm.py` + `metrics/oft.py`: entry 计算函数透传 min_duration

---

## 实施顺序

```
Phase A (1 PR): P3 基础 catalog + CLI 补全（8 个 CLI + catalog 注册）
    ↓
Phase B PR1: B1 + B2（immobility_velocity + smoothed_activity）
    ↓
Phase B PR2: B3 + B4（turn_angle_filtered + velocity_bins）
    ↓
Phase B PR3: B5 + B6（body_length + zone_min_duration）
```

## 参考文件

| 类别 | 路径 |
|------|------|
| P3 已有函数 | `packages/ethoinsight/ethoinsight/metrics/_common.py`（搜索 `elongation\|head_direction\|turn_angle`） |
| P3 已有测试 | `packages/ethoinsight/tests/test_metrics_body_elongation.py` 等 4 个 |
| Rose plot | `packages/ethoinsight/ethoinsight/charts.py::rose_plot` |
| JS 示例获取 | `https://raw.githubusercontent.com/noldus/EthoVision-JavaScriptCustomAnalysis/main/<目录>/<文件名>` |
| EV19 JS API | `docs/review-packages/2026-0521-feedbacks/tstYoyo/manual/Commands and functions for JavaScript variables.html` |
| EV19 公式参考 | `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/ev19-dependent-variables.md` |
| Catalog | `packages/ethoinsight/ethoinsight/catalog/*.yaml` |
| CLI 模板 | `packages/ethoinsight/ethoinsight/scripts/tst/compute_immobility_time.py`（参考 CLI 结构） |
| 现有 immobility | `packages/ethoinsight/ethoinsight/metrics/_common.py::_resolve_immobile_series` |

## 注意事项

1. **CLI 脚本模板**: 每个新指标需要一个 `scripts/<paradigm>/compute_<name>.py`，格式参考现有脚本
2. **Catalog 注册**: `optional_metrics` 新增，注意 `requires_columns` 标注正确的列依赖
3. **Catalog 的 `requires_columns` 是 AND 逻辑**: 所有 pattern 都必须匹配。不需要的列不要写进去
4. **immobility fallback 链顺序不能改**: `mobility_state → pendulum(activity) → velocity`。velocity 是最后一道防线
5. **B5 body_length 需要 nose/tail 检测**: parser 已有 `x_nose/y_nose/x_tail/y_tail` mapping，但不是所有范式都有这些 body point。注册时需标注 `requires_columns: [x_nose, x_tail]`
6. **JS → Python**: null check → None/pd.NA；25fps 假定 → `_estimate_dt()` 自适应
7. **Circular statistics**: heading 的 mean/std 必须用 `scipy.stats.circmean/circstd`
8. **git 操作**: 必须用 `git -C /home/wangqiuyang/noldus-insight`
9. **测试**: ethoinsight tests: `cd /home/wangqiuyang/noldus-insight/packages/ethoinsight && .venv/bin/python -m pytest tests/ -q`
10. **agent backend 测试**: `cd /home/wangqiuyang/noldus-insight/packages/agent/backend && source .venv/bin/activate && make test`
