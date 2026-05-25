# Chart catalog 出图缺口 — 实施清单 + 2 个参数确认

> **SSOT**：[2026-0521-feedbacks/tstYoyo/范式-图表对应关系.md](2026-0521-feedbacks/tstYoyo/范式-图表对应关系.md)
> **范围**：仅 chart-maker 出图层。指标层已 100% 对齐 SSOT，data-analyst 解读 skill 已挂载同事写的 6 份判读文档。
> **触发场景**：2026-05-25 E2E thread `ba798d22` FST n=1/1，catalog `charts=0`，只能跑 4 张 fallback。

## 一、缺口清单（按 SSOT 严格对照）

### 高架十字迷宫 (`epm`)
- ❌ 轨迹图
- ❌ 热区图
- ✅ 箱线图 (`box_open_arm`)
- ✅ 柱状图 (`open_arm_time_ratio_bar`)
- ✅ 区域进入分布图 (`zone_entry_distribution`)

### 旷场 (`oft`)
- ❌ 轨迹图
- ❌ 热区图
- ✅ 箱线图 (`box_center`)
- ✅ 柱状图 (`center_time_ratio_bar`)
- ✅ 中心区进入汇总图 (`center_entry_summary`)
- ❌ 时间进程图（3-5 分钟 time bin 折线图）

### O迷宫 (`zero_maze`)
- ❌ 轨迹图
- ❌ 热区图
- ❌ 箱线图（脚本已写 `plot_box_open_zone.py`，未注册到 catalog）
- ❌ 柱状图

### 明暗箱 (`ldb`)
- ❌ 轨迹图
- ❌ 热区图
- ❌ 箱线图（脚本已写 `plot_box_light.py`，未注册到 catalog）
- ❌ 柱状图

### 悬尾 (`tst`)
- ✅ 箱线图 (`box_immobility`)
- ❌ 柱状图（均值 ± SEM）
- ❌ 活动强度图（activity 像素变化波形图）
- ❌ 放弃挣扎分布图（挣扎 vs 放弃挣扎时间分布，类似 `plt.eventplot()`）

### 强迫游泳 (`fst`)
- ✅ 箱线图 (`box_immobility`)
- ❌ 柱状图（均值 ± SEM）
- ❌ 活动强度图
- ❌ 放弃挣扎分布图

**合计缺**：18 张图脚本待补 + 已写未注册的 2 个箱线脚本待挂 catalog。

## 二、实施步骤（每张图同一套流程）

按 SSOT 已经明确每张图要画什么，不需要再问同事。每张图实施分 3 步：

1. **写脚本** `packages/ethoinsight/ethoinsight/scripts/<paradigm>/plot_<chart_id>.py`
   - 参考已有的 `plot_box_*.py`、`plot_*_bar.py`、`plot_*_distribution.py` 同结构
   - 通用图（轨迹/热区/时间进程）可下沉到 `scripts/_common/`，按 SSOT 中"宽高比固定 4:3"等约束实现
2. **catalog 注册** 在对应 `packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml` 的 `charts:` 块加条目
3. **加单测** `packages/ethoinsight/tests/test_<paradigm>_<chart_id>.py`（覆盖 n=1/n=3 两条分支）

按优先级排序：

| 优先级 | 范围 | 图数量 |
|---|---|---|
| **P0** | fst + tst 的「柱状图、活动强度图、放弃挣扎分布图」 | 6 张 |
| **P1** | zero_maze + ldb 「轨迹、热区、箱线、柱状」 | 8 张 + 2 个已写脚本挂 catalog |
| **P2** | epm + oft 的「轨迹、热区」 | 4 张 |
| **P3** | oft 「时间进程图」 | 1 张 |

## 三、需要同事拍板的 2 个参数

SSOT 文档没明说、但实施时必须填具体值，请同事确认。其他全部"画什么、什么数据、什么含义"SSOT 已写清楚，照做即可。

### 问题 1：柱状图（均值 ± SEM）在 n<3 时怎么处理

SSOT 写"柱状图（均值 ± SEM）"——但 n=1 没有 SD/SEM、n=2 SEM 几乎无意义。两种处理：

- **A**：n<3 时**不出柱状图**（按 SSOT 字面意思，SEM 必需 n≥2，n≥3 才有意义）→ 严格保守
- **B**：n<3 时**不画误差线，柱顶直接标数值** → 让小样本探索也能看到方向性

E2E 实测里用户就是 n=1/1 的情况，A 还是 B 直接决定该场景能否拿到柱状图。**我倾向 B**（保留可用性，仅用注释/警告说明误差线不可计算），请同事确认。

### 问题 2：OFT 时间进程图的"实验较长"具体阈值 + time bin 长度

SSOT 原文：「若实验较长，以 3-5 分钟为 time bin，画折线图，看运动距离和中央区滞留之间的变化（评估习惯化）」

代码 catalog 的 `when:` 字段需要一个明确阈值，time bin 也需要一个固定数。请同事确认：

- 触发条件：实验总时长 ≥ ? 秒（**默认 600 秒/10 分钟**，请同事拍）
- time bin：3 / 4 / 5 分钟（**默认 5 分钟**，请同事拍）

## 四、命名沿用现有规范（不需同事再拍）

同事文档已给出所有图的中文名（轨迹图 / 热区图 / 箱线图 / 柱状图 / 区域进入分布图 / 中心区进入汇总图 / 时间进程图 / 活动强度图 / 分布图）。

- `display_name_zh`：直接用同事文档里的中文名
- `chart id`：照现有 catalog 风格延续（`<指标>_<图种>` 或 `<图种>_<范畴>`，如 `box_open_arm` / `open_arm_time_ratio_bar` / `zone_entry_distribution`）

## 五、不在本次范围

- 不动指标定义（已 100% 对齐 SSOT）
- 不改已有 `box_*` 图的 `n_per_group >= 3` 阈值
- 不加 SSOT 未提到的图种（如散点图、相关性热图、小提琴图）

---

**同事回复以上 2 个问题后**，我按 SSOT + 同事偏好分批补脚本 + catalog 注册 + 单测，提 PR 到 dev。
