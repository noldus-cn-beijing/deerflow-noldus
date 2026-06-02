# 2026-05-27 P2+P3+EV19 Skill 收尾 — 实施完成

## TL;DR

在 2026-05-27 P0+P1 基础上完成全部后续任务：

- **P2**: 5 项算法/图表改进（1 个 commit each）
- **P3**: 4 项 EV19 因变量补全（1 个 commit）
- **EV19 收尾**: default-template-fallback 指针文档 + 软门 _gate.py + skill 启用（1 个 commit）

## 改动清单

| Commit | 内容 |
|---|---|
| `85d87322` | fix: 轨迹图和热区图强制 4:3 宽高比 — 加 set_box_aspect(3/4) |
| `e2b5497e` | fix: EPM/Zero Maze zone 列查找优先 center point — 论文金标准对齐 |
| `1a80512a` | feat: LDB 新增区域进入分布图（亮室 vs 暗室进入次数）+ compute_light_entry_count |
| `4dd32d32` | feat: Zero Maze hesitation_count 扩展 proximity_threshold_cm 近边界探测 + 算法假设文档化 |
| `c11625ec` | feat: OFT 时间进程图增加 group-aggregate 模式（均值±SEM）+ time_progress_plot 支持 group_label |
| `5e8e256b` | feat: P3 EV19 因变量补全 — body elongation / head direction / turn angle / rose plot |
| `0649846f` | feat: EV19 收尾 — default-template-fallback 指针文档 + 软门 _gate.py + skill 启用 |

## 测试结果

- ethoinsight: **436 passed, 64 skipped** (+24 from baseline 412)
- agent backend: **3043 passed, 19 skipped** (不变)

## 新增/修改文件

### 新增
- `packages/ethoinsight/ethoinsight/scripts/ldb/plot_zone_entry_distribution.py`
- `packages/ethoinsight/ethoinsight/templates/_gate.py`
- `packages/ethoinsight/tests/test_metrics_body_elongation.py`
- `packages/ethoinsight/tests/test_metrics_head_direction.py`
- `packages/ethoinsight/tests/test_metrics_turn_angle.py`
- `packages/ethoinsight/tests/test_rose_plot.py`
- `packages/ethoinsight/tests/test_template_soft_gate.py`
- `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/default-template-fallback.md`

### 修改
- `packages/ethoinsight/ethoinsight/charts.py` — 4:3 宽高比 + time_progress group_label + rose_plot
- `packages/ethoinsight/ethoinsight/metrics/epm.py` — center point 优先
- `packages/ethoinsight/ethoinsight/metrics/zero_maze.py` — center point 优先 + hesitation 扩展
- `packages/ethoinsight/ethoinsight/metrics/ldb.py` — compute_light_entry_count
- `packages/ethoinsight/ethoinsight/metrics/_common.py` — 3 个 EV19 因变量函数
- `packages/ethoinsight/ethoinsight/catalog/ldb.yaml` — zone_entry_distribution chart
- `packages/ethoinsight/ethoinsight/scripts/oft/plot_time_progress.py` — group-aggregate
- `packages/agent/extensions_config.json` — 启用 ethovision-paradigm-knowledge

## 已知遗留

1. **P2-4 proximity_threshold_cm** — 参数已实现并激活（默认 2.0 cm），但有 `distance_moved` 列时才生效。heuristic 在无距离数据时自动退回到 zone-column-only 检测。
2. **P3 指标未在 catalog YAML 中注册** — body elongation/head direction/turn angle 的函数和测试已完成，但 catalog 注册 + CLI 脚本 wrapper 留给后续 task。当前函数只能通过 Python API 直接调用。
3. **rose_plot 配套 P3-2 head direction** — 图表函数已就绪，但 catalog 注册 + CLI 脚本 wrapper 未做。
4. **EV19 软门 _gate.py** — 已就绪，但 6 范式 templates/*.py 尚未创建（templates/ 目录下只有 __init__.py 和 _gate.py）。等后续 E2 task（6 范式分析模板补全）时集成。
5. **quality-gates.md** — 已存在于 `ethoinsight-lead-interaction/references/quality-gates.md` 且含 ev19_template 引用，本次未修改（Opus 审查确认内容已充分）。
6. **LDB 默认变体** — `OpenFieldRectangle-Subdivided2x2` 是临时兜底，等行为学同事 PR 确认。

## 后续工作

- **6 范式分析模板补全**（templates/epm.py 等）— 依赖行为学同事 review PR
- **P3 指标 catalog 注册 + CLI 脚本** — 需为 body_elongation/head_direction/turn_angle/rose_plot 创建 scripts/*/compute_*.py 和 catalog YAML 条目
- **shoaling golden-case 校验**
- **EV19 E2E 手工验证** — 启动服务确认 skill 在 system prompt 中可见 + agent 能走通识别→分析流程

## Opus 审查关键决策

- **Task C4 (quality-gates) 已跳过**：内容已存在于 `ethoinsight-lead-interaction/references/quality-gates.md`
- **Task C1 (default-template-fallback) 改为指针文档**：不重复 `EV19_TEMPLATE_PARADIGM_MAP` 的数据，以 Python 代码为 SSOT
- **Task A4 (proximity_threshold_cm) 已实现**：不是 dead parameter，heuristic 在有 distance_moved 数据时激活
- **Task A5 (bin alignment) 已修复**：用众数 bin count 对齐（`max(set(bin_counts), key=bin_counts.count)`）
