# Handoff: seal-robustness Phase 2 — signal_distribution 数据通路

**日期**: 2026-06-02  
**Worktree**: `/home/wangqiuyang/noldus-insight/.claude/worktrees/seal-robustness-phase1/`  
**分支**: `worktree-seal-robustness-phase1`  
**规范文档**: `docs/superpowers/specs/2026-06-01-all-subagent-seal-robustness-design.md`

---

## 当前任务目标

实施规范 §2.3 的 **阶段 2（路径 B）**：

> 让 code-executor 在计算 pendulum/velocity 类 metric 时，把逐帧中间量的分布统计量（p10/p90/median/max/n_frames）写进 handoff，data-analyst step 2.8 就有真数据可比，参数审计**真正能做**，不再是无米之炊、不再卡死。

---

## 当前进展

### ✅ 已完成（阶段 1）
提交 `2b0f4e9d`（当前 HEAD）包含：
1. 4 个 subagent prompt 加 `<handoff_field_format>` 格式速查段
2. `DataQualityWarning` 加 `model_validator(mode="before")` 归一化（code 下划线→DOT / metric=None→"all" / evidence=str→wrap）
3. `WARNING_CODE_PREFIXES` 常量从 schema import（SSOT）
4. data-analyst step 2.8 加 "signal_distribution 缺失→info 跳过" 前置条件
5. 21 个新测试 `test_handoff_normalization.py`，全量 3307 passed

### ❌ 未完成（阶段 2，本 handoff 目标）
- **B-1**: code-executor 产出 `signal_distribution`
- **B-2**: data-analyst step 2.8 用真分布做参数审计
- `handoff_schemas.py` 加 `signal_distribution` 字段（schema SSOT）

---

## 关键架构发现

### 1. 数据流（pendulum 路径）

```
DataFrame (Activity 列)
  ↓
_common.py::_resolve_immobile_from_activity(df, **pendulum_kwargs)
  ↓
_pendulum.py::pendulum_immobility_series(activity, dt, **pendulum_kwargs)
  ↓  [内部]
_pendulum.py::detect_pendulum(activity, dt, ...) → list[dict]
  每帧: {"state": int, "periodicity": float, "is_pendulum": bool}
  ↓  [只用 state]
返回 immobility 二值序列 (np.ndarray)
  ↓
compute_immobility_time / latency / bout_count → 标量 float
```

**关键问题**：`detect_pendulum` 的 `periodicity` 逐帧值在 `pendulum_immobility_series` 中被丢弃，只用了 `state`。

### 2. 数据流（velocity 路径）

```
DataFrame (X center / Y center 列 或 velocity 列)
  ↓
_common.py::_resolve_immobile_from_velocity(df, velocity_threshold, ...)
  → 内部计算逐帧 velocity = sqrt(dx² + dy²) / dt
  → 但不返回 velocity 序列，只返回 immobility 二值序列
```

### 3. MetricStat schema（已有 parameters_used）

位置：`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py:52`

```python
class MetricStat(BaseModel):
    model_config = ConfigDict(extra="allow")   # ← 可加新字段不破坏旧文件
    mean: float | None = None
    std: float | None = None
    n: int | None = None
    applicable: bool = True
    parameters_used: dict[str, float | int | str | None] = {}
```

### 4. per_subject 当前结构

```json
{
  "per_subject": {
    "Subject 1": {"immobility_time": 45.2, "immobility_latency": 12.5},
    "Subject 2": {"immobility_time": 52.1, "immobility_latency": 8.3}
  }
}
```

data-analyst 当前用这些**标量**计算组内 p10/p90/median —— 这是问题所在：标量聚合后精度不足以审计 pendulum 参数（velocity_threshold 等需要逐帧分布）。

### 5. seal_code_executor_handoff tool 签名

位置：`seal_handoff_tools.py:231`

```python
def seal_code_executor_handoff(
    status, summary, paradigm,
    metrics_summary: dict[str, dict[str, dict[str, Any]]] | None = None,
    per_subject: dict[str, dict[str, Any]] | None = None,
    ...
) -> str
```

`metrics_summary` 是 `group → metric → MetricStat`。

---

## 设计决策（接手前需确认一次）

### B-1 实施位置：compute 脚本

**不修改 `detect_pendulum` / `pendulum_immobility_series` 的返回值**（保持 API 稳定）。

在每个 FST/TST compute 脚本中，**额外**调用 `detect_pendulum` 获取逐帧数据并计算分布统计：

```python
# compute_immobility_time.py 新增
from ethoinsight.metrics._pendulum import detect_pendulum
from ethoinsight.metrics._common import _find_activity_column, _estimate_dt

activity_col = _find_activity_column(df)
if activity_col is not None:
    activity = df[activity_col].to_numpy(dtype=float)
    dt = _estimate_dt(df)
    frames = detect_pendulum(activity, dt, **{k: v for k, v in parameters.items() if k.startswith("pendulum_")})
    periodicity_vals = [f["periodicity"] for f in frames]
    signal_distribution = _compute_distribution_stats(periodicity_vals, "periodicity")
    payload["signal_distribution"] = signal_distribution
```

### B-1 存储位置：per_subject[subject][metric]["signal_distribution"]

把 `per_subject` 从纯标量升级为混合结构：

```json
{
  "per_subject": {
    "Subject 1": {
      "immobility_time": 45.2,
      "immobility_time__signal_distribution": {
        "p10": 0.1, "p90": 0.7, "median": 0.35, "max": 0.95, "n_frames": 1250,
        "signal_key": "periodicity"
      }
    }
  }
}
```

**为什么用 `{metric}__signal_distribution` key 而非嵌套**：`per_subject[subject][metric]` 当前是标量，改成 dict 会破坏 data-analyst 的 scalar 读取逻辑。用 `{metric}__signal_distribution` 作为旁路 key，data-analyst 按 key 存在性决定是否走新路径，向后兼容。

**alternative**：加到 `metrics_summary[group][metric].signal_distribution`（MetricStat extra 字段）。data-analyst 从 metrics_summary 取，更符合"分组统计"语义。规范说"per_subject[subject][metric] 下加一个 signal_distribution 字段"。**接手前先确认用户意见**，或直接按规范实施。

### B-2 data-analyst step 2.8 修改

步骤 2.8a 中，检查 `per_subject` 是否有 `{metric}__signal_distribution` key：
- 有 → 用它的 `p10/p90/median` 做参数比对
- 无 → 走旧路（从 per_subject 标量计算组内 p10/p90）

---

## 需要修改的文件

| 文件 | 改动 |
|------|------|
| `packages/ethoinsight/ethoinsight/metrics/_pendulum.py` | 新增 `extract_periodicity_distribution(activity, dt, **kwargs)` 辅助函数 |
| `packages/ethoinsight/ethoinsight/metrics/_common.py` | 新增 `extract_velocity_distribution(df, **kwargs)` + `_compute_distribution_stats(vals)` |
| `packages/ethoinsight/ethoinsight/scripts/fst/compute_immobility_time.py` | 额外计算 periodicity 分布，写入 payload["signal_distribution"] |
| `packages/ethoinsight/ethoinsight/scripts/fst/compute_immobility_latency.py` | 同上 |
| `packages/ethoinsight/ethoinsight/scripts/fst/compute_immobility_bout_count.py` | 同上 |
| `packages/ethoinsight/ethoinsight/scripts/tst/compute_*.py` | 同上（如存在） |
| `packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py` | `MetricStat` 加 `signal_distribution` 字段 |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` | step 2.8 补"有 signal_distribution 时用真分布"路径 |
| `packages/ethoinsight/tests/` | 新增分布函数单测 |
| `packages/agent/backend/tests/` | 新增 handoff schema 字段测试 |

---

## 接手的第一步

1. **确认 Phase 1 状态**：
   ```bash
   cd /home/wangqiuyang/noldus-insight/.claude/worktrees/seal-robustness-phase1/packages/agent/backend
   PYTHONPATH=. uv run pytest tests/test_handoff_normalization.py tests/test_data_quality_warning_schema.py -q
   ```

2. **阅读规范**：
   `docs/superpowers/specs/2026-06-01-all-subagent-seal-robustness-design.md` §2.3

3. **核查关键文件**：
   - `packages/ethoinsight/ethoinsight/metrics/_pendulum.py`（detect_pendulum 函数）
   - `packages/ethoinsight/ethoinsight/scripts/fst/compute_immobility_time.py`（compute 脚本模式）
   - `packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`（MetricStat）

4. **检查 TST scripts 是否存在**：
   ```bash
   find packages/ethoinsight/ethoinsight/scripts/tst/ -name "compute_*.py"
   ```

5. **开始实施**：先写 `_compute_distribution_stats` 辅助函数和单测，再改 compute 脚本，最后改 schema 和 data-analyst。

---

## 风险与注意事项

1. **不要修改 `detect_pendulum` 的返回类型**：其他代码（`pendulum_immobility_series`）依赖 `state` 字段，不能破坏。
2. **per_subject 标量兼容性**：data-analyst 当前代码 `from per_subject 识别偏离组均值` 假设值是 `float`。新的 signal_distribution 必须是**旁路 key**（`{metric}__signal_distribution`），不能把 `per_subject[subject][metric]` 改成 dict。
3. **FST 和 TST 共用钟摆算法**：两者的 compute 脚本都需要同样的改动。
4. **velocity-based 指标**：`_resolve_immobile_from_velocity` 内部计算了逐帧 velocity，但不暴露。需要一个单独的辅助函数计算 velocity 分布。但要注意：该函数逻辑依赖 df 有 X center/Y center 列（非所有文件都有）。
5. **MetricStat extra="allow"**：`signal_distribution` 可以作为 extra 字段通过，但规范要求"schema 是 SSOT"，所以要显式声明。
6. **全量测试命令**：
   ```bash
   PYTHONPATH=/home/wangqiuyang/noldus-insight/.claude/worktrees/seal-robustness-phase1/packages/agent/backend \
   uv run pytest tests/ -q --tb=short
   # 预期：3307+ passed，3 pre-existing failures（config.yaml 环境问题，与本任务无关）
   ```
7. **ethoinsight 测试**：
   ```bash
   cd /home/wangqiuyang/noldus-insight/.claude/worktrees/seal-robustness-phase1/packages/ethoinsight
   pytest tests/ -q
   ```

---

## 规范摘要（Phase 2 核心）

> **B-1**: `_pendulum.py` 已逐帧产 `periodicity`/`activity`（`:140-144`）；compute 脚本聚合 metric 时，**额外计算这些逐帧量的 {p10,p90,median,max,n_frames} 并随 metric 输出**。写入位置：`per_subject[subject][metric]` 下加一个 `signal_distribution` 字段（dict），或 `metrics_summary` 的 `MetricStat` 加字段。
>
> **B-2**: data-analyst step 2.8 改为用真分布比对。若 `signal_distribution` 缺失（老 handoff / 该 metric 无此分布）→ 记 info 跳过，不卡死。
