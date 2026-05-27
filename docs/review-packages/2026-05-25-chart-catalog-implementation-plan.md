# Chart catalog 出图缺口 — 实施清单（同事已拍板，可开始实施）

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

## 三、同事已拍板的 2 个参数（2026-05-25 11:14）

### Q1：柱状图（均值 ± SEM）在 n<3 时怎么处理 → **B：不画误差线，柱图照出**

> 同事原话：「第一个样本数量过少的时候，不画那个方差的那个误差的误差线」

**实施约定**：
- 所有「柱状图（均值 ± SEM）」类 chart 脚本统一行为：
  - n ≥ 3：画柱体 + SEM 误差线
  - n < 3 (n=1 或 n=2)：**柱体照出**，**不画误差线**，柱顶标数值
- catalog `when` 阈值用 `total_subjects >= 1`（柱图本身能出，只是误差线由脚本内部按 n 自动决定）

### Q2：OFT 时间进程图触发条件 + time bin → **时长 > 5 分钟才画；按 5 分钟一个 bin 等分**

> 同事原话：「时长大于 5 的时候按照 3 等分和 5 等分来做。常见的就是 5 分钟往上的话，就是 10 分钟、15 分钟、20 分钟这样的话，比如说 15 分钟三等分就是五，二十分钟 4 等分的话就是四五」

**实施约定**：
- catalog `when`：`total_duration_seconds > 300`（实验总时长 > 5 分钟）
- bin 策略：**bin 长度固定 5 分钟**，bin 数 = `floor(总时长_秒 / 300)`
  - 10 min → 2 bins (5+5)
  - 15 min → 3 bins (5+5+5)
  - 20 min → 4 bins (5+5+5+5)
  - 25 min → 5 bins (5+5+5+5+5)
  - 末尾不足 5 分钟的尾段并入最后一个 bin
- 每个 bin 内画 2 条折线：**运动距离** + **中心区滞留**（同事原话「看运动距离和中央区滞留之间的变化」）

## 四、命名沿用现有规范（不需同事再拍）

同事文档已给出所有图的中文名（轨迹图 / 热区图 / 箱线图 / 柱状图 / 区域进入分布图 / 中心区进入汇总图 / 时间进程图 / 活动强度图 / 分布图）。

- `display_name_zh`：直接用同事文档里的中文名
- `chart id`：照现有 catalog 风格延续（`<指标>_<图种>` 或 `<图种>_<范畴>`，如 `box_open_arm` / `open_arm_time_ratio_bar` / `zone_entry_distribution`）

## 五、不在本次范围

- 不动指标定义（已 100% 对齐 SSOT）
- 不改已有 `box_*` 图的 `n_per_group >= 3` 阈值
- 不加 SSOT 未提到的图种（如散点图、相关性热图、小提琴图）

---

**同事 2 个参数已定，开始实施**。按 P0→P3 顺序补脚本 + catalog 注册 + 单测，提 PR 到 dev。
