# 2026-05-27 P0 指标 Bug 修复 + EV19 文档审计 handoff

## 当前任务目标

1. 评估同事在 `docs/review-packages/2026-0521-feedbacks/tstYoyo/` 下的 EV19 JS 导出代码对项目指标计算/图表/Skill 的价值
2. 修复 Opus 审计发现的 P0 指标计算 bug

## 当前进展

✅ P0 三处 bug 已修复并推送到 `origin/dev`（commit `b8ff57d4`）
✅ 全量测试通过（401 passed, 64 skipped）
✅ EV19 manual HTML 文件完成全面审计（Opus 逐 JS 文件 md5 校验 + 内容审查）
✅ 完成了指标计算代码、图表代码 vs EV19 官方文档 + 同事规格的全面对比审计

## 关键发现

### EV19 manual/ HTML 文件

`docs/review-packages/2026-0521-feedbacks/tstYoyo/manual/` 下的 HTML 是 EthoVision XT 19 安装目录的离线帮助文档（浏览器"另存为完整网页"导出），包含：

- `Commands and functions for JavaScript variables.html` (83KB) — EV19 JS API 完整参考
- `Activity.html` / `Activity state.html` — Activity 因变量定义 + 公式 `Activity = CP_k / P_k × 100`
- `Averaging interval.html` — 滑动平均语义 + 缺失样本处理表
- `JavaScript custom variables.html` / `JavaScript state.html` — JS 变量概念模型

**所有 `_files/` 文件夹中的 JS 文件 100% 是 Adobe RoboHelp 2015 帮助系统框架代码**（经 Opus 逐文件 md5 校验，6 个文件夹是同一套 18 个 JS 文件的完整副本）。唯一的指标相关内容在 HTML 正文和一张公式图片 `Activity state_files/inset_4800920.jpg` 中。

### 同事的 TST pendulum detect

`tst_pendulum_example.py` 是完整可用的 Python 实现（6 阶段自相关周期性检测），优于当前单纯依赖 EV19 预分析 `mobility_state` 列的方式。JS 版本 `TST_PendulumDetect.js` 是 EV19 内实时检测版本。

### P0 Bug 审计发现的完整清单

#### 已修复（本次 commit `b8ff57d4`）

| Bug | 文件 | 影响 |
|-----|------|------|
| `compute_open_zone_distance` 返回 ratio 而非 cm | `metrics/zero_maze.py:130` | Zero Maze 报告中的开放区距离缩小两个数量级 |
| `compute_center_time` 列名 `"time"` → `"trial_time"` | `metrics/oft.py:190-192` | `center_time` 指标从实现以来从未生效（永远 return None） |
| `compute_center_distance_ratio` 曼哈顿距离 | `metrics/oft.py:142-154` | 中心区距离占比系统性偏高 |

#### 未修复（Opus 审计 P1/P2/P3，按优先级）

**P1 — 重要 Feature 缺失/语义错误：**
- P1-1: TST pendulum detect 未集成到 ethoinsight（`tst_pendulum_example.py` 已有完整实现，迁移 ~200 行）
- P1-2: `activity_intensity_plot` 用 velocity 代替 Activity 像素变化%（TST/FST 中 velocity ≈ 0，图无信息量）
- P1-3: parser 需识别 EV19 Activity 列名（`utils.py` COLUMN_MAP 增加 `"Activity": "activity"` 等）
- P1-4: `_find_mobility_column` fallback 太宽（可能误匹配 `mobility_continuous` 连续值列）
- P1-5: `struggle_distribution_plot` 名称/语义反（只画 immobility 段，同事要求"挣扎 vs 放弃挣扎对比"）

**P2 — 算法改进：**
- P2-1: 轨迹图/热区图未强制 4:3 宽高比（同事规格要求）
- P2-2: EPM entry 判定应优先用 center body point 列（论文金标准）
- P2-3: LDB 缺"区域进入分布图"
- P2-4: Zero Maze `compute_hesitation_count` 定义可扩展
- P2-5: OFT 时间进程图缺 group-aggregate 模式

**P3 — EV19 因变量补全（可选）：**
- body_elongation / heading / turn_angle / distance_to_zone 等 EV19 因变量未接入

### EV19 文档的最终用途

- **当前**：HTML 正文作为 EV19 公式权威参考（Activity = CP_k/P_k × 100、Averaging interval 语义等）
- **未来**：在实施 `ethovision-paradigm-knowledge` skill 时（CLAUDE.md 第 10 条，已设计未实施），将 HTML 搬到 `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/ev19-help/` 下

## 建议接手路径

1. **立即做 P1-1**：移植 `tst_pendulum_example.py` 到 `ethoinsight/metrics/_pendulum.py`，在 TST/FST 中增加基于 activity 列的 immobility fallback 路径
2. **然后 P1-2 + P1-3**：修复 `activity_intensity_plot` 数据源 + parser 列名识别，一个 PR 解决
3. **P1-4 + P1-5** 可拼一个"数据质量"PR

## 关键文件

- `packages/ethoinsight/ethoinsight/metrics/_common.py` — 共享指标（immobility、velocity、distance）
- `packages/ethoinsight/ethoinsight/metrics/oft.py` — OFT 指标（已修 2 处）
- `packages/ethoinsight/ethoinsight/metrics/zero_maze.py` — Zero Maze 指标（已修 1 处）
- `packages/ethoinsight/ethoinsight/charts.py` — 所有图表函数
- `packages/ethoinsight/ethoinsight/catalog/*.yaml` — 指标/图表注册
- `docs/review-packages/2026-0521-feedbacks/tstYoyo/tst_pendulum_example.py` — 待移植的钟摆检测 Python 实现
- `docs/review-packages/2026-0521-feedbacks/tstYoyo/tst-pendulum-algorithm.md` — 算法文档
- `docs/review-packages/2026-0521-feedbacks/tstYoyo/范式-图表对应关系.md` — 同事的 6 范式图表规格
- `docs/review-packages/2026-0521-feedbacks/tstYoyo/manual/` — EV19 官方帮助文档

## 当前分支状态

- Branch: `dev`
- Last commit: `b8ff57d4` (pushed to origin)
- 工作区干净（仅 3 个 untracked `.github/workflows/` yml，与本次无关）
