# 2026-05-25 FST E2E 5 处修复 + 范式 skill 挂载 + chart catalog gap 整理 handoff

> **状态**：所有代码修复已合入 dev (HEAD `3834a022`)，chart 出图 review 包 (`docs/review-packages/2026-05-25-chart-catalog-implementation-plan.md`) 已交同事，**同事已回复 2 个参数（2026-05-25 11:14）**，可立即开始按 P0→P3 实施补 chart。

## 当前任务目标

接 2026-05-25 早上的会话（task_tool args_schema bug 已修，commit 4d66f3a6 + 809350b7）。
用户重启服务跑端到端 FST n=1/1 测试，整体跑通了 6 段流水线（识别 → 指标 → 解读 → ask viz → chart → 报告），但暴露 5 个独立小问题。
本会话的目标：诊断 5 个症状 → 修代码 → 同事可拍板的产品 gap 单独抽出。

## 当前进展

### ✅ 5 个症状全部诊断 + 4 处代码 fix 已合入 dev (HEAD `3834a022`)

| # | 症状 | 根因 | fix |
|---|---|---|---|
| 1 | 前端"子任务运行中"卡死不消失 | commit `39bda9e6` (5/22) 改了 ToolMessage 模板从 `"Task Succeeded. Result: <...>"` 改成 `"Task Succeeded.\n\n## 最终结果\n<...>"`，前端 `message-list.tsx:173` 的 `startsWith("Task Succeeded. Result:")` 不再匹配 → 永远 `status: "in_progress"` | `message-list.tsx` parser 兼容新旧两种格式 |
| 2 | Gate 2 `set_experiment_paradigm(acknowledge_quality=True)` 被 guardrail deny | `ev19_template_provider.py:65-83` 只检查"ev19_template 已设置 + 没传 confirm_template_change → deny"，没区分 acknowledge_quality 模式 | guardrail 早返回放过 `acknowledge_quality=True` 且没传 paradigm/ev19_template 的调用 + 2 个 regression test |
| 3 | chart-maker 跑脚本时 plan_charts.json 里 input 是宿主路径，被 sandbox guardrail 拦 | **code-executor 自己写 handoff_code_executor.json 时**把 `inputs.raw_files` 写成了宿主路径（resolve.py 不做 realpath；chart-maker 是从上游 handoff 抄路径）。ethoinsight-code SKILL **没有像 chart-maker 那样**有"必须用虚拟路径"的警告 | `ethoinsight-code/SKILL.md` + `templates/output-contract.md` 加强制条款 |
| 4 | FST chart catalog 命中 0 个 | `fst.yaml charts:` 只有 `box_immobility` (`when: n_per_group >= 3`)，n=1 不命中 → 落 fallback。**产品 gap，不是代码 bug** | 整理为 review 包给同事（见下文） |
| 5 | report-writer 找不到 output-constitution.md | `data_analyst.py:73` 和 `report_writer.py:191` 写的是 `/mnt/skills/ethoinsight/...`，但实际路径是 `/mnt/skills/custom/ethoinsight/...`（code_executor + chart_maker 是对的） | 2 处加 `custom/` 前缀 |

### ✅ 范式解读 skill 已正确挂载到 data-analyst / report-writer

发现同事在 `docs/review-packages/2026-04-29-ev19-templates/by-experiment/` 写的 6 份范式判读文档（epm/open_field/zero_maze/light_dark_box/forced_swim/tail_suspension）**100% 已搬到 `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/`，内容字字一致**。但 data-analyst 和 report-writer 的 `skills=[...]` 列表里没列这个 skill，导致 subagent 看不到这些判读文档。

**修复**：
- `data_analyst.py:175` skills 加 `"ethovision-paradigm-knowledge"`
- `report_writer.py:259` 同上
- 两个 subagent 的 workflow 加了 **step 2.5**：从 handoff 的 paradigm 字段拿 slug，主动 read `/mnt/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/<paradigm>.md`，照"标准报告语言"段做术语来源

### ✅ FST chart catalog gap → 整理交同事 (#4)

文档：`docs/review-packages/2026-05-25-chart-catalog-implementation-plan.md`

内容：严格按同事 SSOT `docs/review-packages/2026-0521-feedbacks/tstYoyo/范式-图表对应关系.md` 对照当前 6 个 catalog yaml，列出 18 张图缺口 + 2 个已写未注册的脚本，按 P0~P3 排好优先级。

**同事 2 个参数已拍板（2026-05-25 11:14，会话中口头给出）**：
- **Q1**：柱状图（均值±SEM）n<3 时 → **不画误差线，柱体照出 + 柱顶标数值**
- **Q2**：OFT 时间进程图 → **总时长 > 5 分钟才画**；bin 长度**固定 5 分钟**，bin 数 = `floor(总时长_秒 / 300)`（10/15/20/25 min → 2/3/4/5 bins）；每 bin 画两条折线：运动距离 + 中心区滞留

详细实施约定已写入 `docs/review-packages/2026-05-25-chart-catalog-implementation-plan.md` 第三段。

### ✅ 测试通过

- backend pytest: **3008 passed, 18 skipped, 0 failed** (109s)（注：本会话 FST 修复时是 2749 passed，后 DeerFlow sync 5 PR 合入后涨到 3008）
- frontend `pnpm typecheck`: 干净通过

## 关键上下文

### 项目结构（与本任务相关）

```
noldus-insight/
├── packages/
│   ├── agent/
│   │   ├── backend/packages/harness/deerflow/
│   │   │   ├── subagents/builtins/
│   │   │   │   ├── data_analyst.py    ← 改了 skills + workflow step 2.5
│   │   │   │   ├── report_writer.py   ← 改了 skills + workflow step 2.5
│   │   │   │   └── (code_executor.py / chart_maker.py 路径本来就对)
│   │   │   ├── guardrails/ev19_template_provider.py  ← 加 ack 早返回
│   │   │   └── tools/builtins/task_tool.py  ← 上一会话已修 args_schema
│   │   ├── frontend/src/components/workspace/messages/
│   │   │   └── message-list.tsx  ← parser 兼容新旧格式
│   │   └── skills/custom/
│   │       ├── ethoinsight-code/
│   │       │   ├── SKILL.md             ← 加 raw_files 虚拟路径强制条款
│   │       │   └── templates/output-contract.md ← 加同样条款
│   │       └── ethovision-paradigm-knowledge/
│   │           └── references/by-experiment/
│   │               ├── epm.md / open_field.md / zero_maze.md
│   │               ├── light_dark_box.md / forced_swim.md
│   │               └── tail_suspension.md   ← 同事原稿已完整搬到这里
│   └── ethoinsight/ethoinsight/
│       ├── catalog/    ← 6 个 paradigm yaml，charts 块缺图（见 review 包）
│       └── scripts/<paradigm>/  ← 部分孤儿脚本（plot_box_open_zone / plot_box_light）
└── docs/
    ├── handoffs/2026-05/                        ← 本文件位置
    ├── review-packages/
    │   ├── 2026-05-25-chart-catalog-implementation-plan.md  ← 本会话新增 (#4)
    │   ├── 2026-04-29-ev19-templates/by-experiment/         ← 同事范式原稿（已搬到 skill）
    │   └── 2026-0521-feedbacks/tstYoyo/
    │       └── 范式-图表对应关系.md             ← chart 层 SSOT
```

### 三层对齐情况（严格按 SSOT）

| 层 | 状态 |
|---|---|
| **code-executor 指标计算** | ✅ 6 个范式全部对齐（catalog 5/5、4/4、3/3 全过） |
| **data-analyst 解读 skill** | ✅ skill 内容齐 + 本会话已挂载到 subagent + workflow 加引导 |
| **chart-maker 出图** | ❌ 6 个范式都不对齐，每个缺 2-4 张图（见 review 包） |

### 已解决但要注意的几个点

- **task_tool args_schema bug** 在前一会话 commit `4d66f3a6` 已修。如果再看到 `TypeError: task_tool() missing 2 required positional arguments`，说明退化了。
- **5/22 task_tool 模板改动 `39bda9e6`**：把 `"Task Succeeded. Result: ..."` 改成 `"Task Succeeded.\n\n## 最终结果\n..."` 已加进度时间线。前端 parser 本会话已兼容两种，**不要把后端模板改回**。
- **ethoinsight-code SKILL 新增 raw_files 虚拟路径强制条款**：跟 ethoinsight-chart-maker 是对称的。任何 subagent 写出 `inputs.raw_files` 都必须用虚拟路径。

## 关键发现

1. **同事 SSOT 文档已经把图的中文名给齐了**：轨迹图 / 热区图 / 箱线图 / 柱状图 / 区域进入分布图 / 中心区进入汇总图 / 时间进程图 / 活动强度图 / 分布图。补 chart 时 `display_name_zh` 直接用，英文 id 照现有 catalog 风格延续即可（`<指标>_<图种>`），**不需要重新发明命名规范**。

2. **SSOT 是 chart catalog 唯一真源**。**禁止**自己加 SSOT 没提的图（散点图、相关性热图、小提琴图等）。同事 review 时也是按这份 SSOT 验收。

3. **范式名映射**（中文 ↔ paradigm key）：
   - 高架十字迷宫 ↔ `epm`
   - 旷场 ↔ `oft`
   - O迷宫 ↔ `zero_maze`
   - 明暗箱 ↔ `ldb`
   - 悬尾 ↔ `tst`
   - 强迫游泳 ↔ `fst`

4. **arena → group 对应是用户在反问中给的**（不在文件名里）。本次实测 thread `ba798d22` 的 Arena 1 = treatment、Arena 2 = control，是用户回的"1是实验组,2是对照组"。data-analyst 的 group 信息从 lead 派遣 prompt 拿，不是从文件名解析。

## 未完成事项

### 高优先级（同事已拍板，可立即开始）

- [ ] **按 review 包 P0→P3 顺序补 18 张 chart 脚本 + 2 个孤儿脚本挂 catalog**
  - P0：fst + tst 各补「柱状图、活动强度图、放弃挣扎分布图」共 6 张
  - P1：zero_maze + ldb 各补 4 类 + 2 个箱线脚本挂 catalog 共 8 张 + 2 注册
  - P2：epm + oft 各补「轨迹图、热区图」共 4 张（通用脚本下沉到 `_common/`）
  - P3：oft「时间进程图」1 张

### 中优先级

- [ ] **跑端到端 dogfood 验证本会话 5 处修复**：用同样的 FST n=1/1 数据重跑 thread，验证：
  - lead Gate 2 `acknowledge_quality=True` 一次过（不再被 guardrail deny 后用 confirm_template_change 兜底）
  - 前端"子任务运行中"在 subagent 完成后消失（前端 parser fix 生效）
  - chart-maker 不再触发 host_path_blocked（code-executor 写 handoff 时走虚拟路径）
  - report-writer 能成功 read `/mnt/skills/custom/ethoinsight/references/output-constitution.md`（不再 file not found）
  - data-analyst 报告中术语用了同事文档"标准报告语言"段（如"开臂滞留时间百分比"而非简称"开臂时间"）

### 低优先级

- [ ] **`data_analyst.py:73` 那里 workflow 编号有重复**（两个 `2.` 步骤）：本会话只插了 step 2.5，原有的双 `2.` 编号问题没动。下次顺手修。
- [ ] 整理本会话教训到 memory：(a) 同事 SSOT 在 review-packages 下，要先 grep review-packages 再做范式相关判断；(b) skill 文件存在 ≠ subagent 能用，要确认 subagent `skills=[...]` 是否列入。

## 建议接手路径

### 第一步：dogfood 验证本会话 5 处修复确实生效

代码改完没做端到端验证，先确认 5 处修复都跑通，再开始补 chart 脚本。

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop && make dev
# 浏览器开 localhost:2026，上传同一批 FST 文件 (Arena 1 + Arena 2)
# 重复"帮我分析一下大鼠强迫游泳的实验数据"
# 重点验证：
#   - lead 调 set_experiment_paradigm(acknowledge_quality=True) 一次过吗？（不再被 guardrail deny 后用 confirm_template_change 兜底）
#   - chart-maker 跑脚本前 plan_charts.json 是不是虚拟路径？（不再 host_path_blocked）
#   - subagent 卡片完成后是否切换到 "已完成" 状态？（前端 parser 兼容新格式）
#   - report-writer 能 read /mnt/skills/custom/ethoinsight/references/output-constitution.md 吗？
#   - report 中术语是不是用同事文档的"标准报告语言"（如"开臂滞留时间百分比"而非简称）？
```

### 第二步：按 review 包 P0 起按范式实施补 chart

同事 2 个参数已定（见上文"当前进展"段），直接照做。

P0：fst + tst 各补「柱状图、活动强度图、放弃挣扎分布图」共 6 张
P1：zero_maze + ldb 各补 4 类 + 2 个孤儿脚本挂 catalog 共 8 张 + 2 注册
P2：epm + oft 各补「轨迹图、热区图」共 4 张（通用脚本下沉到 `_common/`）
P3：oft「时间进程图」1 张（5 分钟 bin 等分）

每张图 3 步：写脚本 → catalog 注册 → 单测。**参考 `plot_box_open_arm.py` / `plot_open_arm_time_ratio_bar.py` / `plot_zone_entry_distribution.py` 已有的脚本结构**。**禁止**自己设计图样，必须严格按 SSOT 文档相应章节实现。

**柱状图 n<3 行为统一约定**（同事 Q1 答复）：脚本内部按 n 自动决定是否画误差线 —— n≥3 画 SEM 误差线、n<3 仅柱体 + 柱顶标数值。catalog `when:` 不需特殊处理，用 `total_subjects >= 1`。

**OFT 时间进程图 bin 策略**（同事 Q2 答复）：catalog `when: total_duration_seconds > 300`；bin 长度固定 300 秒，bin 数 = `floor(总时长/300)`；末尾不足 5 min 并入最后一个 bin；每 bin 画运动距离 + 中心区滞留 2 条折线。

## 风险与注意事项

### ❌ 不要做的事

- **不要按"我觉得需要"补 chart**——同事 SSOT 没列的图，一律不加。本会话第一版 review 包里我自己加了"组间柱状图/点图"被用户当场叫停。
- **不要重新设计 chart id 命名规范**——同事文档没要求，照现有风格延续即可。
- **不要为了"修复"产品 gap 改 catalog `when:` 阈值**——`box_*` 的 `n_per_group >= 3` 是数学正确的（boxplot 至少要 3 点画 IQR），不要为了让 n=1 也命中就改阈值。
- **不要把后端 task_tool 模板改回 `"Task Succeeded. Result: ..."`** ——5/22 加进度时间线是正向改进，前端 parser 已兼容两种，往回改会丢失进度时间线。

### ⚠️ 容易混淆

- 6 份范式判读文档在两个地方：`docs/review-packages/2026-04-29-ev19-templates/by-experiment/` (review 原稿) 和 `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/` (运行时 skill)。**两者内容 100% 一致**，但 subagent 运行时只 read 后者（skill 路径）。同事更新前者后要同步到后者。
- "code-executor 写 handoff 时用虚拟路径"和"chart-maker 的 raw-files-json 必须虚拟路径"是同一类问题，但分布在两个 skill。两边都要警告，不能只警告一边。

## 下一位 Agent 的第一步建议

1. **Read 本文件**（你正在做）
2. **Read review 包**：`cat docs/review-packages/2026-05-25-chart-catalog-implementation-plan.md` 看 chart 缺口具体清单 + **同事已拍板的 2 个参数实施约定**（第三段）
3. **Read SSOT**：`cat docs/review-packages/2026-0521-feedbacks/tstYoyo/范式-图表对应关系.md` 看同事对每个范式图表的原始要求
4. **先 dogfood 验证 5 处修复**（避免开始补 chart 后才发现修复不完整）：启动服务 → 重跑 FST n=1/1 端到端 → 对照"建议接手路径"段的 5 个验证点逐一确认
5. **Verify 测试不退化**：`cd packages/agent/backend && .venv/bin/python -m pytest tests/ -q`（应 3008 passed）；`cd packages/agent/frontend && pnpm typecheck`（应 0 error）
6. **dogfood 全过后开始 P0**：补 fst + tst 各 3 张 chart（共 6 张），参考已有 `plot_box_*.py` 结构

## 当前 git 状态

```
HEAD: 37bcbba4 "Merge pull request #38 from noldus-cn-beijing/feat/langfuse-full-integration"
本会话 FST 5 处修复 commit: 3834a022 "重新按照范式文档进行了代码整理"
working tree: 仅本 handoff 自身 + chart review plan 同事答复更新未提交
本地 dev 与 origin/dev 同步（5/25 后续 DeerFlow sync 5 PR 已全合入）：
  37bcbba4 ← PR #38 langfuse-full-integration（最后合入）
  ...省略中间 DeerFlow sync 多个 commit
  3834a022 ← 本会话 FST 5 处修复（已 push）
  3e5a9800 ← merge origin/dev (PR #24 同事 TST feedback)
  809350b7 ← 上次会话补 5-22 docs
  4d66f3a6 ← 上次会话修 task_tool args_schema bug
```

**未 commit 的待提交文件**：
- `docs/review-packages/2026-05-25-chart-catalog-implementation-plan.md`（同事答复落实更新）
- `docs/handoffs/2026-05/2026-05-25-fst-e2e-fixes-and-chart-catalog-handoff.md`（本文件）
- `docs/handoffs/2026-05/2026-05-25-deerflow-sync-all-prs-merged-handoff.md`
- `docs/handoffs/2026-05/2026-05-25-deerflow-sync-pra-merged-prb-pending-prc-todo-handoff.md`

## 关键文件速查

| 任务 | 文件 |
|---|---|
| chart 实施清单 + 2 问题 | `docs/review-packages/2026-05-25-chart-catalog-implementation-plan.md` |
| chart SSOT | `docs/review-packages/2026-0521-feedbacks/tstYoyo/范式-图表对应关系.md` |
| 范式判读 skill（运行时） | `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/*.md` |
| 6 个范式 catalog | `packages/ethoinsight/ethoinsight/catalog/{epm,oft,zero_maze,ldb,fst,tst}.yaml` |
| data-analyst subagent | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` |
| report-writer subagent | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py` |
| ev19 guardrail | `packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py` |
| 前端 subtask parser | `packages/agent/frontend/src/components/workspace/messages/message-list.tsx` |
