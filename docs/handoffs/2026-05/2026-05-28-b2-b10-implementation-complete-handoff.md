# 2026-05-28 Noldus JS B2-B10 实施完成 handoff

## 当前状态

```
dev HEAD: 2d1ef1ef
ethoinsight: 463 passed / 64 skipped
post-review fixes: 5 项已修，全部合入
```

## 任务目标

对照 Noldus 官方 JS 算法仓库 (48 个 EV19 脚本)，将 B2-B10 算法 port 到 Python (`packages/ethoinsight/`)。

## 已完成 (✅)

### PR 1: B2 + B3 — 运动平滑
- **B2** `activity_intensity_plot(smooth_window=...)` — FST/TST 脚本传入 Noldus 默认值 10
- **B3** `compute_turn_angle_filtered(df, min_displacement_mm=1.0)` — 3 点滑动窗口 + 位移门控，过滤静止抖动

### PR 2: B4 + B7 + B8 — 运动分析
- **B4** `compute_velocity_bins(df, bin_edges=...)` — 速度分箱，含 `<{first}` 前导 bin
- **B7** `compute_cumulative_distance(df, window_samples=25)` — 滑动窗口累计距离，帧率自适应
- **B8** `compute_acceleration_stats(df, smooth_window=5)` — SMA 平滑后差分加速度

### PR 3: B5 — 身体测量
- **B5** `compute_body_length(df)` — Nose→Center + Center→Tail 欧式距离，需 6 列 body point

### PR 4: B6 + B10 — Zone/Heading 增强
- **B6** `_count_zone_entries` 从 epm.py 迁移到 _common.py，新增 `min_duration_frames=0`；EPM/OFT 入口函数透传
- **B10** `compute_heading_smoothed(df, window=5)` — ffill NaN + unwrap + SMA + re-wrap

### Post-review 修复 (HEAD 2d1ef1ef)
- `velocity_bins`: `<{first}` 前导 bin 防静默丢弃
- `acceleration_stats`: docstring 修正单元声明
- `body_length`: `std=0.0` (单帧时) 替代 NaN
- `heading_smoothed`: ffill NaN 间隙后再 unwrap
- `heading_smoothed`: 新增 5 个测试（含 ±π 边界 + NaN 间隙）

## 关键上下文

- **仓库**: `/home/wangqiuyang/noldus-insight`，分支 `dev`，remote `origin`
- **ethoinsight venv**: `.venv/bin/python`，pytest 在 `packages/ethoinsight/tests/`
- **Git**: 所有命令加 `git -C /home/wangqiuyang/noldus-insight`
- **实施计划**: `docs/plans/2026-05-27-b2-b10-implementation-plan.md` (v2, post Opus review)
- **Handoff**: `docs/handoffs/2026-05/2026-05-27-noldus-js-examples-p3-handoff.md`
- **CLAUDE.md**: 项目根 + `packages/agent/backend/CLAUDE.md`

## 新增文件清单

### scripts/ (6 个)
- `scripts/oft/compute_turn_angle_filtered.py`
- `scripts/oft/compute_velocity_bins.py`
- `scripts/oft/compute_cumulative_distance.py`
- `scripts/oft/compute_acceleration_stats.py`
- `scripts/oft/compute_body_length.py`
- `scripts/epm/compute_body_length.py`

### catalog entries (6 个)
- `catalog/oft.yaml`: turn_angle_filtered, velocity_bins, cumulative_distance, acceleration_stats, body_length
- `catalog/epm.yaml`: body_length

### tests (新增 ~20 个测试函数)
- `tests/test_metrics_turn_angle.py` — B3 4 tests
- `tests/test_metrics.py` — B4 5 tests + B7 2 tests + B8 3 tests
- `tests/test_metrics_body_elongation.py` — B5 3 tests
- `tests/test_metrics_oft.py` — B6 2 tests
- `tests/test_metrics_head_direction.py` — B10 5 tests

## 未完成

### 高优先级
- [ ] **backend 测试**: `cd packages/agent/backend && source .venv/bin/activate && make test`（handoff 要求但尚未执行）
- [ ] **B9, B11, B12**: handoff 标注"优先级低，时间紧可跳过"（zone_visits_two_body_points, center_nose_length, angle_center_nose）

### 低优先级（Opus review nit）
- [ ] `velocity_bins` / `acceleration_stats` catalog `requires_columns: [velocity]` 阻止仅有 `distance_moved` 的数据集通过 catalog 触发这些指标。函数内部有 fallback 但 catalog 的硬门控会先拦截。可选方案：移除函数 fallback（最简），或放宽 catalog。
- [ ] `cumulative_distance.min` 因 `min_periods=1` 预热导致误导（建议 discard 前 window-1 行）
- [ ] 魔数 `0.04`（Noldus 25fps dt）在 `compute_cumulative_distance` 和 `compute_acceleration_stats` 及现有的 `_resolve_immobile_from_velocity` 中重复，可提取为模块级常量
- [ ] `compute_turn_angle_filtered` 返回的是 unsigned angle（arccos），与 Noldus JS 的 signed TurnAngle 不同。docstring 已标注 `abs`，但用户可能期望方向性转角

## 验证命令

```bash
# ethoinsight 全量测试
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/ -q

# backend 测试（尚未执行）
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate && make test
```

## 下一位 Agent 第一步建议

1. 先跑 backend 测试: `cd /home/wangqiuyang/noldus-insight/packages/agent/backend && source .venv/bin/activate && make test`
2. 如果全绿，B2-B10 任务闭环，根据 handoff 判断是否继续 B9/B11/B12
3. 如果 backend 测试有失败，检查是否 catalog 新增条目影响了 resolve 行为
