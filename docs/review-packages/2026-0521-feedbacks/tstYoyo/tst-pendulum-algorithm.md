# TST 钟摆运动检测算法说明 / TST Pendulum Motion Detection Algorithm

## 1. 问题背景 / Problem

悬尾实验 (Tail Suspension Test, TST) 中，动物停止挣扎后身体会像钟摆一样规律摆动。这种摆动同样会引起画面像素变化，导致 Activity 指标出现波动，被误判为挣扎行为。

传统方法仅依赖 Activity 阈值来区分"静止"与"挣扎"，无法处理钟摆摆动的情况。

**核心发现**：停止挣扎后的钟摆运动具有显著的**周期性**特征——摆动频率稳定、幅度逐渐衰减。真实挣扎则表现为不规则的 Activity 波动。利用这一差异，可以用自相关分析来区分两者。

In Tail Suspension Test, after the animal stops struggling, its body oscillates like a pendulum. This oscillation causes pixel changes that inflate Activity readings, making the animal appear to still be struggling. The key insight is that pendulum oscillation is **periodic** (stable frequency, decaying amplitude), while real struggling is irregular. Autocorrelation analysis can distinguish the two.

---

## 2. 输入输出 / Input & Output

### 输入
- **Activity 时间序列**：逐帧的 Activity 百分比值（0~100），即相邻两帧之间的像素变化比例 × 100
- **采样间隔 (dt)**：帧之间的时间间隔（秒）

### 输出
- **逐帧状态标签**：每帧判定为 静止 (Still) 或 挣扎 (Struggling)

| 输出值 | 含义 |
|--------|------|
| 0 | 挣扎 (Struggling) |
| 1 | 静止 (Still，包含钟摆摆动) |

---

## 3. 算法流程 / Algorithm Flow

```
原始 Activity 序列
        │
        ▼
┌──────────────────────┐
│ Phase 1: 预处理平滑   │  移动平均，去除帧间噪声
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Phase 2: 滑动窗口     │  环形缓冲区，维护最近 N 帧数据
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Phase 3: 自相关检测   │  计算周期性强度 (0~1)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Phase 4: 宽容期更新   │  维护"近期是否检测到钟摆"的上下文
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Phase 5: 状态判定     │  综合周期性 + Activity 水平判定
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Phase 6: 持续时间过滤 │  防止状态频繁跳变
└──────────┬───────────┘
           ▼
       输出状态
```

### Phase 1: 预处理平滑 (Pre-smoothing)

对原始 Activity 做移动平均，减少单帧噪点干扰。窗口大小由 `SMOOTH_WINDOW` 控制，通常设为 1（不平滑）。

### Phase 2: 滑动窗口 (Sliding Window)

使用环形缓冲区（ring buffer）维护最近 `ANALYSIS_WINDOW` 帧的数据。每次新帧到来时写入缓冲区，覆盖最旧的数据。窗口未满时默认输出"挣扎"（实验初期动物不太可能放弃挣扎）。

计算窗口内的 **Activity 均值 (meanAct)**，用于后续判定。

### Phase 3: 自相关周期性检测 (Autocorrelation Periodicity Detection)

这是算法的核心。在滑动窗口内检测 Activity 曲线是否存在周期性振荡。

**步骤**：
1. 将窗口内数据减去均值（去均值化）
2. 计算信号能量 E = Σ(xᵢ)²
3. 对每个候选滞后值 lag（从 `PERIOD_MIN` 到 `PERIOD_MAX`），计算自相关值：
   - AC(lag) = Σ(xᵢ × xᵢ₊ₗₐg) / E
4. 取所有 lag 中 AC 的最大值作为 **周期性强度 (periodicity)**，范围 0~1

**直觉理解**：
- 如果 Activity 曲线呈周期性振荡，那么将它平移一个周期后，与原始曲线高度重合，自相关值接近 1
- 如果 Activity 是随机波动（挣扎），平移后不重合，自相关值接近 0
- 挣扎段落的 periodicity 通常 < 0.3，钟摆段落通常 > 0.5

### Phase 4: 宽容期更新 (Grace Period Update)

维护一个倒计时计数器 `g_graceCounter`：
- 每当检测到周期性 > 阈值时，重置为 `PENDULUM_GRACE_PERIOD`
- 否则每帧递减 1

**作用**：钟摆运动中周期性可能在阈值附近波动。宽容期让算法在周期性短暂低于阈值时，仍保持"近期检测到钟摆"的上下文，避免误判为挣扎。

### Phase 5: 状态判定 (State Decision)

综合 meanAct、periodicity 和宽容期，按以下优先级判定：

```
if meanAct < MIN_STILL_ACTIVITY:        → 静止（真静止，Activity 极低）
elif periodicity > PERIODICITY_THRESHOLD: → 静止（钟摆，周期性强）
elif meanAct > ACTIVITY_STRUGGLE_THRESHOLD: → 挣扎（Activity 高）
elif meanAct > MODERATE_ACTIVITY_THRESHOLD: → 挣扎（Activity 中等）
elif 宽容期 > 0:                         → 静止（钟摆过渡区）
else:                                     → 挣扎（挣扎暂停）
```

**判定逻辑说明**：

| 条件 | 判定 | 原因 |
|------|------|------|
| Activity 很低 (< 0.3) | 静止 | 动物完全不动 |
| 周期性强 (> 0.55) | 静止 | 检测到钟摆振荡 |
| Activity 高 (> 2.0) | 挣扎 | 不可能由钟摆产生 |
| Activity 中等 (> 1.0)，无周期性 | 挣扎 | 不规则运动 |
| Activity 低 (0.3~1.0)，近期有钟摆 | 静止 | 钟摆振荡衰减中的过渡 |
| Activity 低 (0.3~1.0)，无近期钟摆 | 挣扎 | 挣扎中的短暂停顿 |

### Phase 6: 持续时间过滤 (Duration Filter)

防止状态在"挣扎"和"静止"之间频繁跳变。新状态必须连续出现 `MIN_STATE_DURATION` 帧才会被确认输出。在确认之前，保持之前的状态。

---

## 4. 参数一览 / Parameters

| 参数 | 默认值 | 说明 | 调参建议 |
|------|--------|------|----------|
| `SMOOTH_WINDOW` | 1 | 预处理平滑窗口（帧） | 建议不超过 3 |
| `ANALYSIS_WINDOW` | 25 | 自相关分析窗口（帧） | 覆盖 3~5 个钟摆周期 |
| `PERIOD_MIN` | 4 | 最短搜索周期（帧） | 对应约 0.16s |
| `PERIOD_MAX` | 12 | 最长搜索周期（帧） | 对应约 0.48s |
| `PERIODICITY_THRESHOLD` | 0.55 | 周期性判定阈值 | 钟摆 > 0.5，挣扎 < 0.3 |
| `ACTIVITY_STRUGGLE_THRESHOLD` | 2.0 | 高 Activity 挣扎阈值 | 观察数据中挣扎的 Activity 范围 |
| `MIN_STILL_ACTIVITY` | 0.3 | 极低 Activity 静止阈值 | 真正不动时的 Activity 水平 |
| `MODERATE_ACTIVITY_THRESHOLD` | 1.0 | 中等 Activity 挣扎阈值 | 防止中等强度挣扎被判为静止 |
| `MIN_STATE_DURATION` | 25 | 状态最短持续帧数 | 越小越灵敏但越不稳定 |
| `PENDULUM_GRACE_PERIOD` | 20 | 钟摆宽容期（帧） | 0 = 禁用宽容期 |

> **帧率自适应**：`MIN_STATE_DURATION` 和 `PENDULUM_GRACE_PERIOD` 基于 25fps 填写。算法通过 `GetSampleTime()` 自动检测实际采样率并按比例缩放，确保不同帧率下等效时长一致。

---

## 5. 数据格式要求 / Data Format

### 输入数据

Activity 序列需满足：
- 每个值为像素变化比例 × 100（即 Activity %）
- 缺失值用 `NaN` 或 `null` 表示
- 帧间等间隔采样

### EthoVision XT 数据格式

EthoVision XT 导出文件为 UTF-16 LE 编码、分号分隔的文本文件。前 37 行为元数据头，第 36 行为列名，第 37 行为单位，第 38 行起为逐帧数据。关键列：

| 列名 | 含义 |
|------|------|
| `activity_sampleRate_1` | Activity %（原始采样率） |
| `X 坐标` / `Y 坐标` | 中心点像素坐标 |
| `面积` | 身体面积 |

读取时需指定 `encoding='utf-16'`、`sep=';'`，跳过前 35 行元数据。

---

## 6. 实现版本 / Implementations

本项目提供两个实现版本：

### 6.1 JavaScript 版本（EthoVision XT 实时检测）

文件：`ethovision/TST_PendulumDetect.js`

在 EthoVision XT 的 Analysis Profile 中添加此脚本，变量类型选择 "State"，输出数量设为 1。

**API 说明**：
- `GetPixelChange()` — 获取当前帧与上一帧的像素变化比例（0~1），× 100 即为 Activity %
- `GetSampleTime()` — 获取当前采样时间（秒），用于自动检测采样率
- `SetOutput(value)` — 设置输出值

### 6.2 Python 版本（离线批量分析）

文件：`analysis/pendulum_detect.py`

离线版本包含额外的后处理步骤（JS 实时版本无法实现）：
- **钟摆起始回溯** (`extend_pendulum_onset`)：向前延伸钟摆段开头，补偿滑动窗口延迟
- **最短静止时长过滤** (`apply_min_still_duration`)：过滤挣扎中的短暂静止误判

---

## 7. 固有限制 / Limitations

1. **检测延迟**：滑动窗口引入约 `ANALYSIS_WINDOW / 2 + MIN_STATE_DURATION` 帧的延迟（约 1~2 秒）。这是滑动窗口方法的固有特性，无法完全消除。

2. **挣扎暂停**：挣扎中的短暂停顿（Activity 接近零）可能被误判为静止。宽容期机制和中等 Activity 阈值可以缓解但不能完全消除。

3. **参数依赖性**：不同品系、体重的动物钟摆频率不同。`PERIOD_MIN`/`PERIOD_MAX` 需要覆盖实际钟摆周期范围。

4. **单变量局限**：算法仅使用 Activity 一个指标。如果同时使用身体角度、摆动幅度等信息，可以进一步提升准确性。
