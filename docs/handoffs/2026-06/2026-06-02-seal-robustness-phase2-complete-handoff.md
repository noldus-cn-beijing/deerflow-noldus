# Handoff: seal-robustness Phase 2 完成 — signal_distribution 数据通路

**日期**: 2026-06-02
**Worktree**: `/home/wangqiuyang/noldus-insight/.claude/worktrees/seal-robustness-phase2/`
**分支**: `worktree-seal-robustness-phase2`
**提交**: `ae3af113`
**规范文档**: `docs/superpowers/specs/2026-06-01-all-subagent-seal-robustness-design.md` §2.3 Phase 2

---

## 完成内容

### B-1: code-executor 产出逐帧信号分布统计量

| 文件 | 改动 |
|------|------|
| `metrics/_pendulum.py` | 新增 `pendulum_periodicity_series()` — 纯函数，返回逐帧 periodicity float 数组 |
| `metrics/_common.py` | 新增 `_compute_distribution_stats(vals, signal_key)` — 返回 {p10,p90,median,max,n_frames,signal_key} |
| `metrics/_common.py` | `_resolve_immobile_from_velocity` 加 `return_signal=True` 可选参数 — 返回 velocity 数组，默认 False 零波及 |
| `scripts/_signal_distribution.py` | **新增** — 共享提取逻辑（pendulum 优先 → velocity fallback） |
| `scripts/fst/compute_*.py` × 3 | payload 额外输出 `signal_distribution` 字段 |
| `scripts/tst/compute_*.py` × 3 | 同上，TST 独立脚本 |

### B-1e: schema SSOT 更新

| 文件 | 改动 |
|------|------|
| `handoff_schemas.py` | `per_subject` Field 描述文档化 `_signal_distributions` 命名空间约定 |
| `code_executor.py` prompt | 聚合指令教 code-executor 把 `signal_distribution` 收集到 `per_subject[subject]["_signal_distributions"][metric]` |

### B-2: data-analyst step 2.8 用真分布审计

| 文件 | 改动 |
|------|------|
| `data_analyst.py` prompt | step 2.8a 新增 **Phase 2 优先路径** — 从 `_signal_distributions` 取 p10/p90/median 做参数比对；`_signal_distributions` 不存在时走 Phase 1 降级路径（与阶段 1.5 行为一致） |

---

## 关键设计决策

### 1. 勘探结论：分布在 compute_*.py 脚本里算（非 dispatcher.py）

- **生产路径**: code-executor 对每个 metric 跑一次 `python -m ethoinsight.scripts.<fst|tst>.compute_*`
- **dispatcher.py**: 不在生产路径上，仅内部 Python API 使用
- **FST/TST 共享底层**: `fst.py`/`tst.py` → `_common.py` → `_resolve_immobile_series` → pendulum/velocity

### 2. 信号丢失点（已修复）

- `pendulum_immobility_series` 吞掉 periodicity → 新增 `pendulum_periodicity_series` 提取
- `_resolve_immobile_from_velocity` 吞掉逐帧 velocity → `return_signal=True` 提取

### 3. `_signal_distributions` 存储格式

```json
{
  "per_subject": {
    "Subject 1": {
      "immobility_time": 45.2,
      "immobility_latency": 12.5,
      "_signal_distributions": {
        "immobility_time": {"p10": 0.1, "p90": 0.7, "median": 0.35, "max": 0.95, "n_frames": 1250, "signal_key": "periodicity"},
        "immobility_latency": {"p10": 0.1, "p90": 0.7, "median": 0.35, "max": 0.95, "n_frames": 1250, "signal_key": "periodicity"}
      }
    }
  }
}
```

- `_` 前缀命名空间键 — 遍历标量 metric 只需 `if not k.startswith("_")` 一条规则
- per_subject 类型已容得下（`dict[str, dict[str, Any]]`）

### 4. `return_signal=True` 可选参数降低波及面

`_resolve_immobile_from_velocity` 默认 `return_signal=False` 保持原有二元组返回：
- 2 个 fallback 调用方（`_resolve_immobile_series` L219/L225）不受影响
- 1 个单测（`test_metrics_parameter_passing.py`）不受影响

---

## 测试结果

### ethoinsight
```
540 passed, 64 skipped, 0 failed
```

新增 `test_signal_distribution.py` 覆盖 3 个新函数（18 个测试）：
- `TestPendulumPeriodicitySeries` × 5
- `TestComputeDistributionStats` × 6
- `TestResolveImmobileFromVelocityWithSignal` × 7

### backend
```
3553 passed, 3 pre-existing failures, 0 regression
```
3 failures 均为 `config.yaml not found`（worktree 环境问题，与本任务无关）。

---

## 待做

1. **合并 PR**: 从 `worktree-seal-robustness-phase2` → `dev`，需要手动建 PR
2. **dogfood 验收（spec 验收标准）**:
   - FST 数据（n≥2）跑 E2E → code-executor handoff 的 per_subject 含 `_signal_distributions`
   - data-analyst step 2.8 用真分布产出有数据支撑的 `ParameterAuditFinding`（不再全跳过）
   - 全链路不卡死
3. **清理 worktree**: dogfood 通过后 `git worktree remove`
