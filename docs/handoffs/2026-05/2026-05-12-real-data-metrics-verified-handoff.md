# 2026-05-12 真数据指标验证 + 列名调校 完成交接

> **2026-05-13 校正补遗：** 同事已于 5-13 回 [`docs/review-packages/2026-05-12-feedback.md`](../../review-packages/2026-05-12-feedback.md)，本文档以下内容被推翻：
>
> - **第 50 行**（"EPM 8% 偏低但在生理范围"）— 同事："单只动物无对照组不做结论"。"偏低/合理"这类绝对判读作废。
> - **第 54 行**（"FST immobility 比文献典型 100s+ 小 10-100 倍 → Q4"）— 同事："还是陷入了和常模对比的幻觉。**不要参考常模或基线**。所有的数据指标要和对照组比。" 整条"和文献对比"的判读哲学作废。
> - **第 60、66 行**（Q4 决定新算法路径、Q5 决定 Mobility 含义）— Q4 整题作废、不再追求 `mobility_continuous` 新算法分支；Q5 真实含义是**采样率**（每帧 vs 每 10 帧平均），不是阈值。
> - **未完成事项 §🟡-2**（"从 user-defined variable 列自动建 groups"）— 概念错误：raw 最后一列其实是「数据选择配置中对结果块的命名」，需要按这个语义重写。
>
> 当前主线行动追踪：[`2026-05-13 同事反馈校正交接`](2026-05-13-feedback-corrections-handoff.md)（待写）。下文原文保留作历史。

---

## 背景

前置交接：
- [2026-05-12-plan-b-c-completed-handoff.md](2026-05-12-plan-b-c-completed-handoff.md) — Plan B 完成 6 范式 scripts，遗留"真数据列名 regex 调校 + e2e 验证"
- [../2026-05-11-sota-migration-completed-real-data-pending-handoff.md](../2026-05-11-sota-migration-completed-real-data-pending-handoff.md) — Step 1-6 的具体调校流程

行为学同事于 2026-05-12 提供真数据：`/home/wangqiuyang/DemoData/newdemodata/`，含 EPM / OFT / FST 三个范式真 EthoVision XT 19 导出。

## 已完成

### 列名诊断（Step 2）

对 3 个范式 7 个 .txt 文件做了 `parse_header` + raw 列对照：

| 范式 | 真列名（normalize 前 → 后） | 命中状态 |
|------|------------------------------|----------|
| EPM  | `分析区中(Open arms/中心点)` → `in_zone_open_arms_center`<br>`分析区中(Closed arms/中心点)` → `in_zone_closed_arms_center`<br>`Entries to Open Arms` → `entries_open_arms` | ✅ 现 regex 直接命中 |
| OFT  | `分析区中` → `in_zone`（**单列、无后缀**）<br>`到墙壁距离` → 残留中文未归一化<br>`Nose over Wall zone` → `nose_over_wall_zone` | ❌ 中文残留 + center 列识别失败 |
| FST  | `Mobility state(狂躁/活跃/静止)` → `mobility_state_狂躁/活跃/静止`（**3 个 one-hot 列、中文残留**）<br>**无 distance_moved 列、无 velocity 列** | ❌ 中文残留 + 不识别 one-hot |

### 列名 regex 调校（Step 3）— 3 个 fix commit

| Commit | 内容 |
|--------|------|
| `db630ec7` | `utils.py` `_slugify.subs` 补 4 个中文翻译：狂躁/活跃/静止 → highly_mobile/mobile/immobile；到墙壁距离 → distance_to_wall |
| `96b15e49` | `metrics/_common.py` 新增 `_resolve_immobile_series` 收口两种 mobility schema（合一列 immobile=0 / one-hot 列 immobile=1）；fst.py / tst.py 把 `DEFAULT_MOBILITY_COL` 改 None 委托自动发现 |
| `d8311bd3` | `metrics/oft.py` 新增 `_find_center_zone_column`（裸 `in_zone` fallback 为 center）+ `_find_periphery_zone_column`；thigmotaxis_index 三层 fallback（显式 periphery 列 → 1−center → x/y 几何） |

### 测试 + e2e 验证（Step 4）

- `ethoinsight pytest tests/ -k "not test_parse"`: **231 passed / 33 skipped / 0 failed**
- 真数据 e2e: 三个范式所有指标完整出来（详见 [review-packages/2026-05-12-real-data-metrics/review.html](../../review-packages/2026-05-12-real-data-metrics/review.html)）
- Scripts CLI e2e: 9 个 `compute_*` 脚本在真数据上跑通

### 真数据关键指标摘录

| 范式 | Subject | 指标 |
|------|---------|------|
| EPM 小鼠 | Subject 1（Drug） | open_arm_time_ratio=**8.0%**, open_arm_time=**23.6s**, total_entries=**21**, distance=1460cm |
| OFT 小鼠 T1 | Dark mouse | center_time=**10.07%**, thigmotaxis=**89.93%**, center_entries=**55**, distance=6775cm |
| OFT 小鼠 T2 | White mouse | center_time=**10.92%**, thigmotaxis=**89.08%**, center_entries=**65**, distance=6725cm |
| FST 大鼠 M1-A1 | Drug | immobility=**5.52s**, latency=**23.92s**, bouts=**52** |
| FST 大鼠 M1-A2 | Saline | immobility=**14.56s**, latency=**32.88s**, bouts=**126** |
| FST 大鼠 M10-A1 | Drug | immobility=**0.56s**, latency=**293.92s**, bouts=**1** |
| FST 大鼠 M10-A2 | Saline | immobility=**1.92s**, latency=**96.16s**, bouts=**7** |

方向性验证：
- ✅ EPM 8% 偏低但在生理范围（单只无对照组只能看合理性）
- ✅ OFT 两 trial 高度一致（thigmotaxis 都 ~90%）
- ✅ FST Saline > Drug 两组阈值都成立（药物有抗抑郁倾向）
- ✅ FST 阈值越长 immobility 越少（RLE 语义正确）
- ⚠️ FST immobility 比文献典型（100s+）小 10-100 倍 → 已写问题 Q4 让同事确认 EthoVision Mobility threshold 设定

### 给行为学同事的 review 包

- HTML: [docs/review-packages/2026-05-12-real-data-metrics/review.html](../../review-packages/2026-05-12-real-data-metrics/review.html)
- 入口: [docs/review-packages/2026-05-12-real-data-metrics/README.md](../../review-packages/2026-05-12-real-data-metrics/README.md)
- 6 个待确认问题（Q1-Q6），核心在 Q4（FST immobility 数值偏小根因）和 Q5（Mobility_1/10 目录命名真实含义）

## 未完成事项

### 🔴 阻塞 — 等同事 review 回流

- Q4 答复决定是否需要新算法路径（基于 mobility_continuous 连续值的 immobility 判断）
- Q5 答复决定 Mobility_1/Mobility_10 目录的实际含义（影响 by-paradigm/fst.md 文档）
- Q6 答复决定要不要补 swimming_time / climbing_time / 平均 bout 长度等 FST 衍生指标

### 🟡 中优先级

1. **`by-paradigm` 文档同步**：oft.md / fst.md / tst.md 加"真数据列名结构 + 单列 vs one-hot 兼容"小节
2. **`parse.py` 自动 group 推断**：从 user-defined variable 列（Drug/Saline/Treatment）自动建 groups dict，目前 dispatcher 收到 groups=None 退化为 "all"
3. **dispatcher 范式无关指标"适用性"标识**：FST 真数据没有 distance_moved / velocity 列，dispatcher 无脑调用返 None，污染 data_quality_warnings；建议加 metric 适用性 flag
4. **`utils.detect_paradigm` bug**：zero_maze 返回 `"o_maze"` 但 dispatcher / ev19_facts 用 `"zero_maze"`

### 🟢 低优先级

5. files-are-facts e2e dogfooding（Plan-B-C handoff §🟡 三个用例 A/B/C 待 `make dev` 实测）
6. golden-cases 落盘：本次 3 个范式的真数据指标可固化成 golden-case 候选（等同事 review 数字合理性后再做）

## 关键文件速查

| 用途 | 路径 |
|------|------|
| 真数据 | `/home/wangqiuyang/DemoData/newdemodata/{高架十字迷宫_小鼠_三点,旷场_小鼠_三点,强迫游泳_大鼠}/` |
| 跑指标的临时脚本 | `/tmp/run_metrics_on_real_data.py` |
| Review HTML | `docs/review-packages/2026-05-12-real-data-metrics/review.html` |
| 修改的代码 | `packages/ethoinsight/ethoinsight/{utils.py, metrics/_common.py, metrics/oft.py, metrics/fst.py, metrics/tst.py}` |
| 测试覆盖 | `packages/ethoinsight/tests/test_metrics_{oft,fst,tst}.py`（37 + 9 个，全绿） |
| 上一份交接 | [2026-05-12-plan-b-c-completed-handoff.md](2026-05-12-plan-b-c-completed-handoff.md) |

## Commit 历史

```
d8311bd3 fix(metrics/oft): 支持单列 in_zone + thigmotaxis 用 1-center fallback
96b15e49 fix(metrics): FST/TST 支持 one-hot mobility_state_immobile 列
db630ec7 fix(parse): _slugify 补 mobility state + distance to wall 中文翻译
```

3 个 fix commit + 1 个 review 包 + 1 个交接文档（本文件）= 5 个 commit 总计。

## 下一位 Agent 的第一步建议

1. **等同事 review 回流**（最重要是 Q4/Q5/Q6）
2. 同事 review 完后按 Q6 勾选结果补全指标 + 按 Q4/Q5 决定是否需新算法路径
3. 把 3 个范式真数据 + 验证后的指标固化到 `golden-cases/`，启动 SFT 训练数据飞轮
4. 完成 `by-paradigm/{oft,fst,tst}.md` 文档的"真数据列名"小节
