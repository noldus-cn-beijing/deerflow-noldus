# 2026-05-27 P0+P1 完成 + 后续任务 handoff

## 本次会话做了什么

基于同事 `docs/review-packages/2026-0521-feedbacks/tstYoyo/` 下的 EV19 官方帮助文档和 TST 钟摆检测算法，完成了：

1. **EV19 manual 文件全面审计**：6 个 HTML 是 EV19 帮助文档正文，`_files/` 中的 JS 100% 是 Adobe RoboHelp 2015 框架代码（经 Opus md5 校验），不含计算逻辑
2. **P0 bug 修复**（3 处）：zero_maze 距离、oft center_time 列名、oft 欧氏距离
3. **EV19 因变量公式参考文件**：从 HTML 提取 14 章权威公式，写入 `ethovision-paradigm-knowledge/references/ev19-dependent-variables.md`
4. **公式引用接入**：3 个 subagent + 2 个 skill + lead_agent prompt + metrics docstring
5. **P1 全部 5 项**：钟摆检测移植、activity_intensity 修复、_find_mobility_column 收紧、struggle 图表双色
6. **两次 Opus 审查**：公式准确性 + 路径修正 + 算法忠实度验证

## dev 分支最终状态

```
7fcb1cbc fix: Opus 审查 — activity 全 NaN 边界修复
a1883304 fix: P1-2+P1-5 activity_intensity 改用 Activity 列 + struggle 图表双色
5f9fdae8 feat: P1-1 移植 TST 钟摆检测算法到 ethoinsight
b5e10138 fix: Opus 审查修复 — 路径补 custom/ + 去重 + 公式准确性改进
39cb80c2 feat: EV19 因变量公式引用接入 3 subagent + 2 skill + metrics docstring
61682d18 feat: 新增 EV19 因变量公式参考到 ethovision-paradigm-knowledge skill
b8ff57d4 fix: P0 指标计算三处 bug 修复
```

- ethoinsight tests: **412 passed, 64 skipped**
- agent backend tests: **3043 passed, 19 skipped**
- 全部已推送 origin/dev

---

## 后续待办任务

### P2 — 算法/图表改进（量小，建议一个 PR 收尾）

#### P2-1: 轨迹图/热区图强制 4:3 宽高比

**依据**：同事 `范式-图表对应关系.md` L26/L48 写"宽高比固定 4:3"

**当前问题**：`charts.py` 的 `trajectory_plot` (L254-310) 和 `heatmap_plot` (L670-692) 用 `figsize=(8,6)` + `set_aspect("equal")` 间接实现，数据 X/Y 范围非 4:3 时会留白或挤压。

**改动**：在 `ax.set_aspect("equal")` 之后加 `ax.set_box_aspect(3/4)` 强制 axes box 4:3。

**涉及文件**：
- `packages/ethoinsight/ethoinsight/charts.py` — `trajectory_plot` 和 `heatmap_plot` 两个函数

#### P2-2: EPM entry 判定优先 center point 列

**依据**：论文惯例（Pellow et al. 1985 等）EPM 进臂判定金标准是 center point in zone。EV19 导出可能有 `in_zone_open_arms_center`、`_nose`、`_tail`、`_all` 多个变体。

**当前问题**：`metrics/epm.py` 的 `_get_open_zone_cols` 用 `re.search(r"in_zone.*open", c)` 匹配所有 body point 变体，然后 `df[cols].max(axis=1)` OR 全部。这会导致 nose 在开放臂、body 还在封闭臂时也计为"在开放臂"——高估开放臂时间。

**改动**：
1. 在 `_get_open_zone_cols` / `_get_closed_zone_cols` 中加 `prefer_body_point` 参数
2. 默认优先匹配 `center` 后缀的列，找不到才 fallback 到 nose/tail/all
3. docstring 说明此决策依据（论文惯例 + ev19-dependent-variables.md §10）

**涉及文件**：
- `packages/ethoinsight/ethoinsight/metrics/epm.py` — `_get_open_zone_cols`、`_get_closed_zone_cols`
- `packages/ethoinsight/ethoinsight/metrics/zero_maze.py` — 同模式，也需检查是否需要

#### P2-3: LDB 缺"区域进入分布图"

**依据**：同事 `范式-图表对应关系.md` L147 总结表，LDB 行写了"其他结果图：区域进入分布图"

**当前状态**：`catalog/ldb.yaml` 的 charts 列表有 `box_light`、`bar_light`、`trajectory`、`heatmap`，没有 entry distribution chart。

**改动**：
1. 新增 `scripts/ldb/plot_zone_entry_distribution.py`（可参考 EPM 同名脚本）
2. 在 `catalog/ldb.yaml` charts 列表注册新 chart
3. 用 `extract_immobility_bouts` 类似的逻辑提取 light↔dark 穿梭事件的时间分布

**涉及文件**：
- `packages/ethoinsight/ethoinsight/scripts/ldb/` — 新增 `plot_zone_entry_distribution.py`
- `packages/ethoinsight/ethoinsight/catalog/ldb.yaml` — 注册新 chart

#### P2-4: Zero Maze hesitation_count 定义扩展（可选）

**依据**：同事规格只写了"犹豫次数"未给精确定义。当前实现用 `min_gap_frames=5` 判定（<5 帧的开放区 visit 算犹豫）。

**改动**：扩展为包含"开放臂边界附近 X cm 内的停留+迅速退回"，docstring 说明算法假设。

#### P2-5: OFT 时间进程图缺 group-aggregate 模式

**依据**：同事规格写"以 3-5 分钟为 time bin，画折线图，看运动距离和中央区滞留之间的变化（评估习惯化）"。评估习惯化需要看 control vs treatment 的 group 趋势对比。

**当前状态**：`scripts/oft/plot_time_progress.py` 是 per-subject 单图。

**改动**：增加 group-aggregate 模式：多 subject 时按 group mean ± SEM 画线。

**涉及文件**：
- `packages/ethoinsight/ethoinsight/charts.py` — `time_progress_plot` 函数
- `packages/ethoinsight/ethoinsight/scripts/oft/plot_time_progress.py`

---

### P3 — EV19 因变量补全（可选，v0.1 后）

| 指标 | EV19 API | 适用范式 | 改动 |
|------|----------|----------|------|
| Body elongation | `GetElongation` (0-1, ×100) | EPM/OFT（SAP 检测） | `metrics/_common.py` 新增 `compute_body_elongation_stats` |
| Head direction | `GetViewDirection` (弧度) | EPM/Zero Maze（朝向分析） | `metrics/_common.py` 新增 `compute_head_direction_stats` |
| Turn angle | `TurnAngle` (弧度) | OFT/EPM（旋转模式） | `metrics/_common.py` 新增 `compute_turn_angle_stats` |
| Rose plot | 极坐标方向分布图 | 与 heading 配套 | `charts.py` 新增 `rose_plot` |

---

### EV19 模板识别 Skill 实施（下一个里程碑，工作量大）

**背景**：CLAUDE.md 第 10 条标记 `ethovision-paradigm-knowledge` skill 的 EV19 模板识别功能"地基设计已完成、实施计划已就绪"。

**已有材料**：
- 设计文档：`docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md`
- 实施计划：`docs/superpowers/plans/2026-05-08-ev19-template-skill-foundation-plan.md`（14 task / 90 step）
- 当前 skill 已有：`SKILL.md` + `references/_facts.md`（62 变体）+ `references/identification-decision-tree.md` + `references/ev19-dependent-variables.md`（本次新增）+ `references/by-experiment/` + `references/by-template/`

**建议**：先阅读 spec + plan 文档了解全貌，再按 plan 的 14 个 task 顺序执行。

---

## 关键文件索引

| 类别 | 路径 |
|------|------|
| 指标计算 | `packages/ethoinsight/ethoinsight/metrics/_common.py`（共享）、`epm.py`、`oft.py`、`zero_maze.py`、`ldb.py`、`tst.py`、`fst.py`、`_pendulum.py`（新增） |
| 图表生成 | `packages/ethoinsight/ethoinsight/charts.py` |
| Catalog | `packages/ethoinsight/ethoinsight/catalog/*.yaml` |
| CLI 脚本 | `packages/ethoinsight/ethoinsight/scripts/{epm,oft,fst,tst,ldb,zero_maze,_common}/` |
| 测试 | `packages/ethoinsight/tests/`（重点关注 `test_metrics_oft.py`、`test_metrics_zero_maze.py`、`test_pendulum.py`） |
| EV19 公式参考 | `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/ev19-dependent-variables.md` |
| Subagent prompt | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/{data_analyst,knowledge_assistant,report_writer}.py` |
| Lead prompt | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` |
| Skill 入口 | `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md` |
| 同事规格 | `docs/review-packages/2026-0521-feedbacks/tstYoyo/范式-图表对应关系.md` |
| EV19 官方文档 | `docs/review-packages/2026-0521-feedbacks/tstYoyo/manual/`（6 个 HTML） |

## 测试命令

```bash
# ethoinsight 库测试
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/ -q

# agent backend 测试
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate && make test
```

## 注意事项

1. **`_files/` 中的 JS 文件不是计算代码**——是 Adobe RoboHelp 框架。只有 HTML 正文和 `Activity state_files/inset_4800920.jpg`（公式图）有知识价值。
2. **EV19 公式参考已接入**——knowledge-assistant、data-analyst、report-writer 三个 subagent 都可能 read `ev19-dependent-variables.md`，新增公式相关内容时同步更新该文件。
3. **SSOT 原则**（用户 MEMORY）：指标定义/公式只存一处。`ev19-dependent-variables.md` 是 EV19 公式的权威源，不要在 subagent prompt 和 SKILL.md 中重复写公式内容——只写"read_file 路径"引用。
4. **catalog `requires_columns` 是 AND 逻辑**——所有 pattern 都必须匹配。不能表达 OR，所以图表函数内部做 fallback（如 activity_intensity 的 Activity→velocity）。
5. **git 操作必须用 `git -C /home/wangqiuyang/noldus-insight`**——当前 shell 的 CWD 在 `packages/ethoinsight/` 下（符号链接），直接 `git add` 会路径不匹配。
