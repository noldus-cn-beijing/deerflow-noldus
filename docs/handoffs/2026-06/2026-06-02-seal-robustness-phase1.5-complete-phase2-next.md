# 2026-06-02 阶段 1.5 完成交接 — 阶段 2 待启动

**状态**：阶段 1.5 已完成并 push（commit `834a0e3a`，分支 `worktree-seal-robustness-phase1.5`，待建 PR 合 dev）。阶段 2 未开始。

## 本次完成内容（阶段 1.5）

修复 data-analyst seal handoff ValidationError（thread `81051535` dogfood 实证）：

1. **`handoff_schemas.py` ParameterAuditFinding** — 加 `model_validator(mode="before")`
   - `used_value=None → ""`
   - `observed_distribution` 剔除非数字值（`{"note": "文字"} → {}`）
   - 加 `ConfigDict(extra="allow")`（与 DataQualityWarning 一致）

2. **`data_analyst.py` step 2.8 prompt** — 删"前置条件"独立出口，统一为"每参数一条"粒度，明确教退化 finding 合法字段值

3. **`test_parameter_audit_schema.py`** — 新增 7 条归一化测试（30/30 全通过）

**测试**：3522 passed / 33 pre-existing failed（与本改动无关）

## 阶段 2 任务（路径 B — 补逐帧分布数据通路）

### 目标

让 code-executor handoff 的 `per_subject` 包含逐帧分布统计量（`_signal_distributions`），使 data-analyst step 2.8 参数审计**真正有数据可比**，不再是"无米之炊"。

### Spec 位置

`docs/superpowers/specs/2026-06-01-all-subagent-seal-robustness-design.md` §2.3 / §7.3 / §7.4

### 关键设计决策（已锁定）

- 存储位置：`per_subject[subject]["_signal_distributions"][metric] = {p10, p90, median, max, n_frames, signal_key}`
  - 用 `_` 前缀命名空间键，老代码遍历 metric 标量只需跳过 `_` 前缀键
  - per_subject 类型已是 `dict[str, dict[str, Any]]`，`Any` 容得下，schema 只需更新 Field 描述
- 不放宽 schema 契约
- 参数审计只警告不调参（Sprint 3 铁律）
- 工程不编领域阈值数值（issue #63 归同事）

### 🔴 实施前必做的第一步（勘探）

Spec §7.4 明确标注：**真实流水线是 catalog 路径**，code-executor 逐个 bash 跑 `scripts/fst/compute_*.py`，`dispatcher.py` 不一定在该路径上。

**阶段 2 启动第一步必须钉死**：分布统计量是在 `dispatcher.py`（per_subject 确定性组装点）算，还是在 `compute_immobility_time_fst` 等脚本（`metrics/fst.py`）算并随 payload 输出？

勘探方法：
1. grep `compute_immobility_time` / `compute_immobility` 的调用链
2. 看 code-executor glue script（bash 调用 `python -m ethoinsight.scripts.fst.*`）的输出如何进入 per_subject
3. 确认 dispatcher.py 是否被 compute 脚本调用，还是 dispatcher 是另一条路径（plot/run_groupwise_stats）

### 具体改动项（spec §7.3 + §7.4）

**B-1a. periodicity 逐帧序列**（`metrics/_pendulum.py`，零 break）：
- 新增纯函数 `pendulum_periodicity_series(activity, dt=0.04) -> np.ndarray`
- `detect_pendulum` 返回的 dict 已含 `periodicity`，只需逐帧收集
- 0 个现有调用方受影响

**B-1b. velocity 逐帧序列**（`metrics/_common.py:226 _resolve_immobile_from_velocity`，改返回签名）：
- 当前 `:272 return series, 1`，velocity 在循环内 `:264` 算完即弃
- 改为循环内收集 `velocity_arr`，返回 `(series, 1, np.array(velocity_arr))`
- ⚠️ **波及 2 个 fallback 调用方**（`_common.py:182/188`）+ 1 单测
- 可考虑用可选返回（`return_signal=True` 参数）降低波及面

**B-1c. 分布组装点**（待勘探确认位置）：
- 在算 immobility 类 metric 时（FST/TST），额外调序列函数 → 算 `{p10, p90, median, max, n_frames}` → 写入 `per_subject[subject]["_signal_distributions"][metric]`

**B-2. data-analyst step 2.8 改为用真分布比对**（`data_analyst.py`）：
- 从 `per_subject[subject]["_signal_distributions"][metric]` 取真分布
- 对 pendulum/velocity 参数做 p90×3 / p10÷3 比对
- 产出有数据支撑的 ParameterAuditFinding（n≥2 时）
- `_signal_distributions` 缺失仍走阶段 1.5 的降级出口

**handoff_schemas.py**：
- `per_subject` Field 描述加 `_signal_distributions` 约定说明（类型已是 `dict[str, Any]`，容得下）

### 验收标准（spec §4）

- [ ] code-executor handoff per_subject 含 `_signal_distributions`（periodicity/activity/velocity 的 p10/p90/median）
- [ ] handoff_schemas 加该字段描述（schema SSOT）
- [ ] data-analyst step 2.8 用真分布产出有数据支撑的 ParameterAuditFinding（n≥2 时）
- [ ] 端到端：FST 数据参数审计**真正产出 finding**（不再全跳过），且不卡死
- [ ] 全量 make test + ethoinsight pytest 不退化
- [ ] `_resolve_immobile_from_velocity` 改签名后，grep 全调用方确认无遗漏

### 文件位置速查

所有路径相对于 `packages/agent/backend/`：
- Schema SSOT: `packages/harness/deerflow/subagents/handoff_schemas.py`
- Data-analyst prompt: `packages/harness/deerflow/subagents/builtins/data_analyst.py`
- Pendulum metrics: `packages/ethoinsight/ethoinsight/metrics/_pendulum.py`
- Common metrics (velocity): `packages/ethoinsight/ethoinsight/metrics/_common.py`
- FST compute scripts: `packages/ethoinsight/ethoinsight/scripts/fst/` 或 `packages/ethoinsight/ethoinsight/metrics/fst.py`
- Dispatcher: `packages/ethoinsight/ethoinsight/metrics/dispatcher.py`
- Seal tools: `packages/harness/deerflow/tools/builtins/seal_handoff_tools.py`
