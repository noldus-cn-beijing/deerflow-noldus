# 2026-05-26 chart 列依赖过滤 + 子任务态机识别 gate_enforcement (PR 待建)

## 当前任务目标

修复 2026-05-26 FST 端到端 dogfood 揪出的两个独立真 bug:

| # | 症状 | 根因 |
|---|---|---|
| A | activity_intensity 出"velocity column missing"占位空图 | catalog 未声明 chart 列依赖 + resolve.py charts 路径未接 `_missing_columns` 过滤 |
| B | 子任务卡片永久停在"正在请专家解读"不结束 | 前端 message-list.tsx 态机只识 Task* 前缀, GateEnforcement 短路返回的特殊 ToolMessage 落入 in_progress 兜底 |

**状态**: HEAD `1ea5a82a` 在 worktree `.claude/worktrees/chart-catalog-cols-and-subtask-state` 分支 `fix/chart-catalog-cols-and-subtask-state`. 测试全绿待 push.

## Phase 1 调查与决定性证据

### A 根因证据

- 用户上传 FST 数据 `columns.json` 实证 (newdemodata/强迫游泳_大鼠 dump_headers):
  - 列: `trial_time / recording_time / x_center / y_center / body_area / area_change / elongation / mobility_continuous / mobility_state_highly_mobile / mobility_state_mobile / mobility_state_immobile / drug`
  - **无 velocity / 无 distance_moved / 无 zone**
- catalog/fst.yaml:57 activity_intensity 仅 `when: total_subjects >= 1`, 无 requires_columns
- resolve.py:386 `_missing_columns` 函数已存在但**只在 metrics 路径** L177/L213 调用; charts 路径完全跳过
- charts.py:706-715 脚本自身有 fallback 画"velocity column missing"占位图

### B 根因证据 (checkpoints msgpack 解码验证)

```
total msgs: 43, AI task tool_calls: 5, ToolMessages for tasks: 5
--- dispatch trace ---
[ 15] code-executor       ← Task Succeeded
[ 17] data-analyst        ← OTHER  preview='数据质量检查发现以下 critical 问题, 必须先获得用户确认才能继续...请调用 ask...'
[ 24] data-analyst        ← Task Succeeded
[ 31] chart-maker         ← Task Succeeded
[ 38] report-writer       ← Task Succeeded
```

- [17] data-analyst 第一次的 ToolMessage 是 GateEnforcementMiddleware 短路返回 (name=`gate_enforcement`), 不带 Task* 前缀
- message-list.tsx:207-211 兜底 `status: "in_progress"` -> 卡片永远卡在 in_progress

## Phase 2-4 修复

### A 修复 (catalog + resolve)

1. **schema.py**: ChartEntry 加 `requires_columns: list[str] = field(default_factory=list)`
2. **loader.py**: 两个 chart parser 解析 requires_columns (默认 [], 向后兼容)
3. **resolve.py**: charts 主路径 + fallback 路径都加 `_missing_columns` 过滤 + 写入 `skipped` 列表, notes 透明记录被跳过的 chart 与原因
4. **6 yaml audit** (基于 SSOT `docs/review-packages/2026-05-12-feedback.md` + newdemodata 真实列名):
   - fst/tst: `mobility_state*` (box/bar/struggle), `velocity` (activity_intensity)
   - epm: `in_zone_open_arms_*` (box/bar/distribution), `x_center y_center` (trajectory/heatmap)
   - oft: `in_zone*` (匹配真实 bare in_zone), `x_center y_center trial_time` (time_progress)
   - ldb: `in_zone*`, `x_center y_center`
   - zero_maze: `in_zone*`, `x_center y_center`
   - _common: `x_center y_center` (trajectory_plot), `trial_time` (timeseries_plot)

### B 修复 (前端态机)

message-list.tsx:169-213 改为:
- `Task Succeeded` → completed (既有)
- `Task failed.` → failed (既有)
- `Task timed out` → failed (既有)
- `Task cancelled` → failed (**新增**, 对齐 backend task_cancelled SSE)
- **其他**: completed + 原始 ToolMessage 内容作为 result (让被 gate 拦截的信息渲染出来)

## 关键决策

### SSOT: catalog requires_columns 严格反映真实 raw data 列存在与否

用户 5-26 反馈:"计算指标的代码和生图的代码, 不应该凭空想象, 应该根据我们现实存在的 raw data 而建立"。本 PR 严格按 newdemodata 真实 columns 对齐:

- **不**凭空写理想列名 (如 `in_zone_center_*`), 用真实数据匹配的通配 (`in_zone*`)
- requires_columns 解决的是"chart 需要哪些列存在", 不解决"列存在但语义歧义"
- OFT/LDB/zero_maze metric 函数 zone 识别歧义 (见 `2026-05-12-feedback.md` Q2: "不要猜要问") 是**独立 bug**, 不在本 PR

### 不跳过既有非本 PR 范围的问题

- ruff F401 (pathlib.Path 未用 import) 是 pre-existing, 留给后续清理
- hooks.ts 既有 lint 警告同上
- OFT/LDB/zero_maze metric 函数 bare in_zone fallback 移除 → 反问机制接入是另一个独立 PR

### worktree 测试 pitfall (复用 5-25 教训)

按 [[feedback_worktree_uv_editable_install_pitfall]] worktree 内**没有 .venv**, 正确用法:

```bash
cd packages/agent/backend
PYTHONPATH=$PWD/packages/harness:$PWD \
DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml \
/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv/bin/python -m pytest tests/ --tb=line -q
```

frontend pnpm 跑前需 `ln -s /home/wangqiuyang/noldus-insight/packages/agent/frontend/node_modules .`

## 测试结果

| 套件 | 结果 |
|---|---|
| ethoinsight pytest | **384 passed / 51 skipped** (新增 5 个回归测试) |
| backend pytest | **3016 passed / 19 skipped** |
| frontend pnpm typecheck | 无错 |
| ruff check (本 PR 改的文件) | clean (剩 1 个 F401 是 pre-existing) |

## 提交分支

- 分支: `fix/chart-catalog-cols-and-subtask-state` (基于 dev `9ca91e98`)
- HEAD: `1ea5a82a`
- 文件: 12 changed, +227 / −2 lines
- worktree: `.claude/worktrees/chart-catalog-cols-and-subtask-state`
- gh CLI 不可用; push 后请用户去 https://github.com/noldus-cn-beijing/noldus-insight/pull/new/fix/chart-catalog-cols-and-subtask-state 手工建 PR

## 未完成事项 (用户已确认范围, 不进本 PR)

### 🟡 中优先级

1. **shoaling 范式从仓库整体删除** (用户 5-26 指令)
   - 我们支持的 5 范式 = O迷宫 / 大鼠强迫游泳 / 旷场 / 明暗箱 / 高架十字迷宫 (即 zero_maze / fst / oft / ldb / epm)
   - shoaling 出现在 ~40 文件: catalog/shoaling.yaml, metrics/shoaling.py, scripts/shoaling/*, templates/, dispatcher.py, ev19_facts.py, assess.py, utils.py, loader.py, golden-cases/case-001-shoaling-baseline, lead_agent/prompt.py, identify_ev19_template_tool.py, prep_metric_plan_tool.py, 6 个 tests, scripts/build_ev19_experiment_drafts.py, scripts/validate_golden_case.py, memory/prompt.py, report_writer.py 等
   - 建议下一会话专门做; 单独 PR

2. **OFT/LDB/zero_maze metric 函数 zone 歧义反问机制接入**
   - 同事 5-13 feedback Q2 方向: 不要猜要问
   - 当前 metric 函数 (oft.py `_find_center_zone_column`) 拒绝裸 `in_zone` 静默返 None
   - 应改为抛 AmbiguousZoneError, 上层 agent 触发 ask_clarification
   - 单独 PR

### 🟢 低优先级

3. ruff F401 pre-existing 清理 (`pathlib.Path` 未用 import in test_resolve_charts.py)
4. SubtaskCard 当 status=completed 但被 gate 拦截时, label 文案"指标已完成, 正在请专家解读"与 ✓ icon 冲突, 后续 UX 改进
5. 写 milestone 更新 (本 PR 是 catalog v1.1 单源真理强化的 checkpoint)

## 风险与注意事项

### ⚠️ 不要做的事

1. 不要 force-push 到 main/master
2. 不要 amend 已合入 PR 的 commit
3. 不要在 worktree 内跑 backend pytest 不设 PYTHONPATH (`uv editable install` 副作用)
4. 新增 chart entry 必须声明 requires_columns (默认 [] 是向后兼容, 但 audit 完的 catalog 全部显式声明)

### ⚠️ 易混淆

- requires_columns 是 chart "至少需要哪些列存在"的门槛, 不解决"列存在但语义歧义"问题
- fnmatch.fnmatchcase: `in_zone_*` 不匹配单一 `in_zone` (要 1+ 字符 suffix); `in_zone*` 匹配
- ChartEntry.requires_columns 默认 `[]` 表示不依赖任何列 (向后兼容旧 yaml), 但所有 audit 过的 catalog 都应显式声明

## 关键文件清单

### Plan / Handoff

- 本文件 — 2026-05-26 chart 列过滤 + 子任务态机修复
- [/tmp/PR-chart-catalog-cols-description.md] — PR 描述 (创建后写到 /tmp)

### 改动核心

- [packages/ethoinsight/ethoinsight/catalog/schema.py](../../packages/ethoinsight/ethoinsight/catalog/schema.py) — `ChartEntry.requires_columns` 字段
- [packages/ethoinsight/ethoinsight/catalog/loader.py](../../packages/ethoinsight/ethoinsight/catalog/loader.py) — 两个 chart parser 接纳 requires_columns
- [packages/ethoinsight/ethoinsight/catalog/resolve.py](../../packages/ethoinsight/ethoinsight/catalog/resolve.py) — charts 主路径 + fallback 加 `_missing_columns` 过滤
- 6 yaml (epm/oft/ldb/fst/tst/zero_maze/_common) — 所有 chart entry 加 requires_columns
- [packages/ethoinsight/tests/test_resolve_charts.py](../../packages/ethoinsight/tests/test_resolve_charts.py) — 5 个新回归测试
- [packages/agent/frontend/src/components/workspace/messages/message-list.tsx](../../packages/agent/frontend/src/components/workspace/messages/message-list.tsx) — 态机识别 gate_enforcement 等非 Task* 前缀

### 相关 memory

- `feedback_single_source_of_truth.md` — SSOT 原则: catalog requires_columns 以真实 raw data 列名为准
- `feedback_ssot_lives_in_review_packages.md` — 5-25 SSOT 在 review-packages 下
- `feedback_worktree_uv_editable_install_pitfall.md` — worktree pytest 必须设 PYTHONPATH
- `project_2026-05-26_chart_maker_fst_5_root_causes.md` — 前一个 chart-maker FST 修复
