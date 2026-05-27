# EV19 因变量公式参考

**来源**: EthoVision XT 19 官方帮助文档（`Dependent Variables in Detail` 章节）
**用途**: knowledge-assistant / data-analyst 回答"EV19 如何计算 X"时引用

---

## 1. Activity（活动量）

**定义**: 整个 arena 内当前帧与前一帧之间变化像素的百分比。

**公式**:

```
Activity_k = (CP_k / P_k) × 100
```

- `CP_k` = 第 k 帧的变化像素数
- `P_k` = arena 总像素数
- 输出单位: %（0–100）

**计算流程**（Activity state 三步）:

1. **Step 1 — 像素变化比例**: 确定 arena 内所有像素坐标（不仅是检测到的 subject），比较当前帧与前一帧的灰度值，计算变化像素数。
2. **Step 2 — 滑动平均**: 对 Activity 值应用 Averaging interval（默认=1，即不平滑）。公式: `running_avg = ΣActivity_i / interval`。
3. **Step 3 — 阈值分箱**: 将滑动平均后的 Activity 与用户定义的阈值比较，分为 2–4 个状态（Inactive / Moderately active / Active / Highly active）。

**关键规则**:
- Subject missing > 3 帧 → 当前 Activity state 终止，剩余 missing 帧忽略
- Averaging interval 缺失样本处理: 缺失帧的值在 avg() 内被跳过；两个相邻缺失 → 输出 [no value]

**比较陷阱**（来自 EV19 FAQ）:
- **不应跨不同视频分辨率比较 Activity**：分辨率越高，像素变化总数越大，Activity % 的绝对值不可跨分辨率对比
- **采样率影响 Activity**：采样率越高，相邻帧之间的像素变化越小；Activity 值与 sampling rate 负相关
- **Acquisition 中的 Activity 阈值不会被 Analysis 复用**：在 Analysis profile 中需重新输入阈值，两处阈值独立

### Activity vs Mobility 区别

| | Activity | Mobility |
|---|---|---|
| 计算范围 | 整个 arena 所有像素 | 仅检测到的 subject 身体像素 |
| 是否依赖 subject 检测 | 否 | 是 |
| 适用场景 | subject 检测困难、动物相对 arena 很大 | 默认首选 |
| TST/FST 推荐 | ✅（动物固定，像素变化反映挣扎） | — |

---

## 2. Velocity（速度）

**定义**: 基于 center-point 位置变化计算的速度。EV19 内置因变量，以 cm/s 为单位输出。

**注意**: Velocity ≠ Activity。TST/FST 中动物被固定，velocity ≈ 0，应用 Activity（像素变化%）衡量活动强度。

---

## 3. Distance（距离）

**公式**:

```
Distance(pt1, pt2) = √((pt1.x - pt2.x)² + (pt1.y - pt2.y)²)
```

**等价调用**: `pt1.Distance(pt2)`（Point.Distance 方法）

**单位**: mm（内部单位，独立于 Experiment Settings 中选择的单位）

**注意**: 如果任一点为 null，不计算。

---

## 4. Turn Angle（转向角）

**定义**: 三个点形成的角度（弧度）。

```
TurnAngle(Point1, Point2, Point3)
```

- Point1: 起始点（t-2 时刻）
- Point2: 中间点（t-1 时刻）
- Point3: 终点（当前 t 时刻）
- 返回值: 弧度

**转度数**:

```js
function toDegrees(angle) { return angle * (180 / Math.PI); }
```

---

## 5. Heading（朝向）

**定义**: 从 Point1 到 Point2 的方向角（弧度）。

```
Heading(Point1, Point2)
```

- Point1: 起始点（t-1 时刻）
- Point2: 终点（当前 t 时刻）
- 返回值: 弧度

---

## 6. Body Elongation（身体拉伸度）

**定义**: 身体拉伸程度，0–1 之间的值。

- `GetElongation()` 返回 0–1
- **EV19 因变量 Body elongation = GetElongation() × 100**（范围 0–100）
- 用途: 检测 stretched-attend posture (SAP)、grooming 等行为

---

## 7. Body Area（身体面积）

**定义**: subject 的检测面积（跟踪时的黄色 blob）。

- `GetArea()` 返回当前面积
- `GetChangedArea()` 返回相对上一帧的变化面积
- 单位: mm²（内部单位）

**注意**: Deep learning with two subjects per arena 模式下不可用（不记录身体轮廓）。

---

## 8. Head Direction / View Direction（头部朝向）

**定义**: subject 的头部朝向。

- `GetViewDirection()` 返回弧度
- EV19 因变量 **Head direction** 以**度**为单位
- 转换: `degrees = radians × 180 / π`

**注意**: Deep learning with two subjects per arena 模式下不可用。

---

## 9. Distance to Zone / POI（到区域/兴趣点的距离）

**定义**: 从指定 body point 到 zone 边界或 POI 的距离。

```
GetDistanceToZone(Zone, Point)
GetDistanceToPoi(PointOfInterest, Point)
```

- 如果 body point 在 zone 内，距离 = 0
- 单位: mm（内部单位）

---

## 10. Zone In/Out 判定

**单 subject**:

```
IsPointInZone(ZoneName, Point)
```

**多 subject**:

```
IsSubjectCenterInZone(SubjectName, ZoneName)
IsSubjectNoseInZone(SubjectName, ZoneName)
IsSubjectTailInZone(SubjectName, ZoneName)
```

**重要**: 论文惯例（Pellow et al. 1985 等）中 EPM 进臂判定的金标准是 **center point in zone**，不是 nose 或 tail。此约定非 EV19 文档定义，而是行为学领域的通用实践。如果 EthoVision 导出中包含多个 body point 的 in_zone 列，应优先使用 center point。

---

## 11. Pixel Change（像素变化，JS API）

```
GetPixelChange()
```

- 返回当前帧与前一帧的像素变化比例（0–1）
- `Activity % = GetPixelChange() × 100`
- 与导出文件中含 `activity` / `sample_rate` 前缀的列对应（具体列名以 EV19 实际导出为准，不同 Analysis 配置下列名可能不同）

---

## 12. 数据采样

### GetSampleTime

- 返回当前样本的时间（秒）
- **重要**: 时间从 **track 开始**（第一个样本）计算，不是从 trial 开始
- 用于帧率自适应: `dt = sampleTime_k - sampleTime_{k-1}`

### Averaging Interval（Outlier filter）

- 对因变量 V_k 做滑动平均: 用最近 N 个样本的平均值替换当前值
- 缺失样本: 只对有效值取平均；如果所有值都缺失 → 输出 [no value]
- **Outlier filter 在 Track Smoothing 之后应用**（Track Smoothing 先平滑原始 x,y 坐标）

---

## 13. JavaScript 变量类型

| 类型 | SetOutput 值 | 含义 |
|------|-------------|------|
| JavaScript continuous | 有理数 | 每个样本一个数值，聚合为 mean/std 等统计量 |
| JavaScript state | true/false (0/1) | on/off 状态，持续时间 = 1 帧的总数 × dt |
| JavaScript event | 0/1 | 点事件，用于计次 |

### SetOutput（多输出）

```js
SetOutput(0, bOutput1);  // 第一个输出
SetOutput(1, bOutput2);  // 第二个输出
SetOutput(2, bOutput3);  // 第三个输出
SetOutput(3, bOutput4);  // 第四个输出
```

需在 JS 窗口底部设置 **Number of outputs**。

---

## 14. Mobility state 列的编码（重要）

EV19 导出的 `Mobility state` 列是**多状态分类值**，不是 0/1 二值:

| 编码 | 2-state | 3-state | 4-state |
|------|---------|---------|---------|
| 0 | Inactive | Inactive | Inactive |
| 1 | Highly active | Moderately active | Moderately active |
| 2 | — | Highly active | Active |
| 3 | — | — | Highly active |

> **注意**: EV19 HTML 帮助文档给出了状态语义和名称（Activity state.html），但**未明确整数编码顺序**。上表的 0/1/2/3 编码是按阈值递增推断（Inactive → Moderately active → Active → Highly active），以 EV19 实际导出列名为权威（如 `mobility_state_immobile` 等 one-hot 列名后缀）。

**对代码的影响**: `immobile_value = 0` 只在 2-state 配置下正确；3/4-state 下会把 Moderately active / Active 也归为 immobile。应检查列名是否含 `immobile` 关键字来判定编码方式。
