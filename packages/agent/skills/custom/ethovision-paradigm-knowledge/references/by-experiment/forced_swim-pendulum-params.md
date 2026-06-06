# FST 钟摆运动检测算法说明 / FST Pendulum Motion Detection Algorithm

## 1. 问题背景 / Problem

强迫游泳实验 (Forced Swim Test, FST) 中，动物停止挣扎后身体会像钟摆一样规律摆动。这种摆动会引起画面像素变化，导致 Activity 指标出现波动，被误判为游泳/挣扎行为。

传统方法仅依赖 Activity 阈值来区分"不动"与"游泳/挣扎"，无法处理钟摆摆动的情况。

**核心发现**：停止挣扎后的钟摆运动具有显著的**周期性**特征——摆动频率稳定、幅度逐渐衰减。真实挣扎则表现为不规则的 Activity 波动。利用这一差异，可以用自相关分析来区分两者。

> **注意**：FST 与 TST 共用同一套钟摆检测算法（自相关周期性分析），但物理环境不同（水中 vs 空中悬吊）。水中阻力更大，钟摆频率和衰减特性可能与 TST 不同。以下参数默认值沿用 TST 的经验值，**FST 专属精确判据需同事确认后更新**。

---

## 2. 算法概要 / Algorithm Summary

算法与 TST 版本相同（6 阶段流水线：预处理平滑 → 滑动窗口 → 自相关检测 → 宽容期更新 → 状态判定 → 持续时间过滤）。

核心判定逻辑：

| 条件 | 判定 | 原因 |
|------|------|------|
| Activity 很低 (< MIN_STILL_ACTIVITY) | 不动（真不动，Activity 极低） | 动物完全不动 |
| 周期性强 (> PERIODICITY_THRESHOLD) | 不动（钟摆，周期性强） | 检测到钟摆振荡 |
| Activity 高 (> ACTIVITY_STRUGGLE_THRESHOLD) | 游泳/挣扎 | 不可能由钟摆产生 |
| Activity 中等 (> MODERATE_ACTIVITY_THRESHOLD)，无周期性 | 游泳/挣扎 | 不规则运动 |
| Activity 低，近期有钟摆（宽容期 > 0） | 不动 | 钟摆振荡衰减中的过渡 |
| Activity 低，无近期钟摆 | 游泳/挣扎 | 挣扎中的短暂停顿 |

---

## 3. 参数一览 / Parameters

> ⚠️ **以下参数值为从 TST 移植的默认值**。FST 在水中进行，水的阻力可能影响钟摆频率和衰减。**同事需确认这些参数是否需要针对 FST 调整**。确认后在本文档直接更新即可。

| 参数 | 默认值 | 说明 | FST 备注 |
|------|--------|------|----------|
| `SMOOTH_WINDOW` | 1 | 预处理平滑窗口（帧） | 建议不超过 3 |
| `ANALYSIS_WINDOW` | 25 | 自相关分析窗口（帧） | 应覆盖 3~5 个钟摆周期 |
| `PERIOD_MIN` | 4 | 最短搜索周期（帧） | 待确认：水中钟摆周期范围 |
| `PERIOD_MAX` | 12 | 最长搜索周期（帧） | 待确认：水中钟摆周期范围 |
| `PERIODICITY_THRESHOLD` | 0.55 | 周期性判定阈值 | 钟摆 > 0.5，挣扎 < 0.3 |
| `ACTIVITY_STRUGGLE_THRESHOLD` | 2.0 | 高 Activity 挣扎阈值 | 待确认 |
| `MIN_STILL_ACTIVITY` | 0.3 | 极低 Activity 不动阈值 | 待确认 |
| `MODERATE_ACTIVITY_THRESHOLD` | 1.0 | 中等 Activity 挣扎阈值 | 待确认 |
| `MIN_STATE_DURATION` | 25 | 状态最短持续帧数 | 基于 25fps |
| `PENDULUM_GRACE_PERIOD` | 20 | 钟摆宽容期（帧） | 0 = 禁用宽容期 |

> **帧率自适应**：`MIN_STATE_DURATION` 和 `PENDULUM_GRACE_PERIOD` 基于 25fps 填写。算法自动检测实际采样率并按比例缩放。

---

## 4. 参数审计判据 / Parameter Audit Criteria

data-analyst 在 step 2.8 参数审计时，对 FST pendulum 参数使用以下判据：

| 参数 | 期望特征 | 判据来源 |
|------|----------|----------|
| `PERIODICITY_THRESHOLD` | 钟摆段应 >0.5，挣扎段应 <0.3 | 算法设计（§3 自相关检测） |
| `ANALYSIS_WINDOW` | 应覆盖 3~5 个钟摆周期 | 算法设计（§2 滑动窗口） |
| `PERIOD_MIN` / `PERIOD_MAX` | 受品系/体重/水中阻力影响 | 待同事确认精确范围 |
| `MIN_STATE_DURATION` | 应 ≥ 1 个钟摆周期 | 算法设计（§6 持续时间过滤） |

**精确判据待补**：FST 水中环境的精确 pendulum 参数范围需同事根据实验数据确认后更新（关联 issue #63）。

---

## 5. 与 TST pendulum 的关系

- **共享**：算法原理（自相关周期性检测）和代码实现
- **独立**：参数默认值可能因物理环境（水 vs 空气）而不同
- **文档独立**：本文档（FST）与 `tail_suspension-pendulum-params.md`（TST）各自维护，互不引用
