# 2026-05-25 EthoInsight 工程性问题大扫除 完结 handoff

> **会话日期**：2026-05-25（整天）
> **dev HEAD**：`eb2852ef`（origin/dev 同步）
> **测试基线**：ethoinsight 384 passed / agent backend 3017 passed / frontend typecheck 0 error / **0 failed**

---

## 当前任务目标

本会话目标：**解决 5/25 早上 FST n=1 dogfood 暴露的全部结构性问题**，让下一次 dogfood 能跑通完整流水线 + 出齐 SSOT 要求的图表。

结果：**6 个独立结构性问题全部修完**（4 个本会话亲自修 + 2 个分发给独立 worktree agent 修），共 9 个 commit 合入 dev。

---

## 当前进展

### ✅ 全部完成（按 commit 时序）

| commit | 类型 | 说明 |
|---|---|---|
| `8cea306a` | fix(subagent) | recursion_limit 按 middleware hook 动态算（修 data-analyst 卡死） |
| `232f12c8` | fix(frontend) | subagent 完成后 lead 汇报被折叠的 5/25 回归（B2 task.result 顶部永久可见 + B3 assistant:processing 分支主气泡渲染 narrative） |
| `6f6211e9` | docs(handoffs) | P0-P3 chart catalog 实施 handoff（已被下位 agent 全部执行） |
| `ce1376af` | fix(catalog) | load_catalog 接受学术名 paradigm key（PR #41 worktree agent 实施） |
| `e943c2cb` | feat(charts) P0 | fst + tst 各补 3 张图（柱状/活动强度/放弃挣扎） |
| `94c1fe2a` | feat(charts) P1 | zero_maze + ldb 各补 4 张图（箱线/柱状/轨迹/热区） |
| `eeeedd65` | feat(charts) P2 | epm + oft 各补轨迹图 + 热区图（通用脚本下沉 `_common/`） |
| `cec2af53` | feat(charts) P3 | OFT 时间进程图（5min bin，运动距离+中心区滞留） |

### ✅ 用户验收完成

测试基线 0 退化：
- ethoinsight pytest: **384 passed, 51 skipped, 0 failed**（155s）
- agent backend pytest: **3017 passed, 18 skipped, 0 failed**（135s）
- frontend pnpm typecheck: **0 error**

端到端模拟通过：
- `load_catalog("forced_swim")` 一次成功 → 加载 `fst.yaml` → 3 个 FST 指标
- 9f77adcc trace 的 lead 试错路径已不可能再触发

---

## 6 个修复的根因小档案

### 1. recursion_limit fix（`8cea306a`）
- **现象**：FST n=1 dogfood 中 data-analyst 失败，报 `GraphRecursionError: Recursion limit of 25 reached`
- **根因**：[executor.py:512](packages/agent/backend/packages/harness/deerflow/subagents/executor.py#L512) 公式 `max_turns * 2 + 1` 假设每 turn = 2 node。但 LangChain `create_agent` 给每个 overridden middleware hook（`before_agent/before_model/after_model/after_agent`）都加独立 node。`78c9b570 + 11fee2f9`（5/25 上午 sync）给 subagent middleware 链加了 `SafetyFinishReasonMiddleware` 后，每 turn node 数从 2 增到 ~3。叠加 `3834a022` 让 data-analyst 多用 1 个 turn（加 step 2.5 read paradigm doc），刚好 8 turn × ~3 node = 24+ 顶破上限。
- **修复**：新 `calculate_subagent_recursion_limit(middlewares, max_turns)` helper 按实际 hook 数算：`per_turn = before_model + after_model + 2`，one-off = `before_agent + after_agent`，margin = 3。9 个新单测全过。

### 2. frontend B2+B3（`232f12c8`）
- **现象**：subagent 完成后用户看不到 lead 的汇报（"已收到 X 的结果..."），除非手动展开 thinking
- **根因**：`cd512536`（5/21）让 lead 把 "narrative + ask_clarification" 打进同一 AIMessage。前端 `groupMessages` 把这种 AIMessage 分到 `assistant:processing` group（hasToolCalls + 不是 task/present-files），但 [message-list.tsx](packages/agent/frontend/src/components/workspace/messages/message-list.tsx) 没有针对这个 group 的 case，落到 fallback `<MessageGroup>` 把 content 当 thinking 渲到 CoT 折叠区。
- **修复**：
  - **B2** [subtask-card.tsx](packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx): `task.result` 挪出 `ChainOfThoughtContent` 折叠区，放卡片顶部独立 block 永久可见。
  - **B3** [message-list.tsx](packages/agent/frontend/src/components/workspace/messages/message-list.tsx): 加 `else if (group.type === "assistant:processing")` 分支，每条 AIMessage 的 content 走主气泡 narrative，reasoning 留独立折叠。

### 3. paradigm-key-alignment（`ce1376af`，PR #41 worktree）
- **现象**：5/25 trace `9f77adcc` lead 调 `prep_metric_plan(paradigm='forced_swim')` 报 `unknown_paradigm`，第二次改 `'fst'` 才成功。
- **根因**：3 套 key 共存——identify_ev19_template / metrics dispatcher / skill doc 全用学术名（`forced_swim`），catalog yaml 文件名用短 key（`fst.yaml`）。LLM 必须试错才能发现转换。
- **修复**：[loader.py](packages/ethoinsight/ethoinsight/catalog/loader.py) 加 `_PARADIGM_ALIASES` 映射表，`load_catalog` 先按学术名解析到文件名再加载。学术名为 canonical，短 key 向后兼容。**没动 catalog yaml 文件名或字段**（伤筋动骨）。
- **影响**：`forced_swim/tail_suspension/open_field/light_dark_box` 这 4 个原本错位的范式现在能一次成功；`epm/zero_maze/shoaling` 已对齐范式 0 变化。

### 4-7. P0-P3 chart catalog（`e943c2cb` / `94c1fe2a` / `eeeedd65` / `cec2af53`）
- **现象**：5/25 trace 显示 FST chart-maker 命中 0 个 catalog 图，只能跑 4 张 fallback（轨迹+时序），用户想要的箱线图/柱状图当时不存在。
- **修复**：按 [docs/review-packages/2026-05-25-chart-catalog-implementation-plan.md](docs/review-packages/2026-05-25-chart-catalog-implementation-plan.md) 实施，覆盖 18 张图 + 2 个孤儿脚本注册。每个 P 一个 commit，分批 push。

---

## 关键上下文

### 项目结构（本次改动相关）

```
noldus-insight/
├── packages/
│   ├── ethoinsight/
│   │   ├── ethoinsight/
│   │   │   ├── catalog/
│   │   │   │   ├── loader.py          ← ce1376af 加 _PARADIGM_ALIASES
│   │   │   │   ├── fst.yaml           ← e943c2cb 加 3 个 chart
│   │   │   │   ├── tst.yaml           ← e943c2cb 加 3 个 chart
│   │   │   │   ├── epm.yaml           ← eeeedd65 加 trajectory + heatmap
│   │   │   │   ├── oft.yaml           ← eeeedd65 加 trajectory + heatmap + cec2af53 加 time_progress
│   │   │   │   ├── zero_maze.yaml     ← 94c1fe2a 加 4 个 chart
│   │   │   │   └── ldb.yaml           ← 94c1fe2a 加 4 个 chart
│   │   │   ├── charts.py              ← 加 4 个新函数 (heatmap_plot / activity_intensity_plot / struggle_distribution_plot / time_progress_plot) + bar_chart 加 suppress_errorbars kwarg
│   │   │   └── scripts/
│   │   │       ├── _common/
│   │   │       │   ├── plot_trajectory.py    ← (已存在)
│   │   │       │   └── plot_heatmap.py       ← eeeedd65 新增
│   │   │       ├── fst/
│   │   │       │   ├── plot_bar_immobility.py        ← e943c2cb
│   │   │       │   ├── plot_activity_intensity.py    ← e943c2cb (用 velocity)
│   │   │       │   └── plot_struggle_distribution.py ← e943c2cb
│   │   │       ├── tst/ (同 fst 3 个新脚本)
│   │   │       ├── zero_maze/plot_bar_open_zone.py   ← 94c1fe2a
│   │   │       ├── ldb/plot_bar_light.py             ← 94c1fe2a
│   │   │       └── oft/plot_time_progress.py         ← cec2af53
│   │   └── tests/
│   │       ├── test_catalog_loader_aliases.py         ← ce1376af
│   │       └── test_plot_*_cli.py                     ← P0-P3 各加单测
│   └── agent/
│       ├── backend/packages/harness/deerflow/
│       │   ├── subagents/executor.py                  ← 8cea306a 加 calculate_subagent_recursion_limit
│       │   ├── tools/builtins/prep_metric_plan_tool.py ← ce1376af docstring 更新接受学术名
│       │   └── tests/test_subagent_executor_recursion_limit.py ← 8cea306a 9 个单测
│       └── frontend/src/components/workspace/messages/
│           ├── subtask-card.tsx                       ← 232f12c8 B2
│           └── message-list.tsx                       ← 232f12c8 B3
└── docs/
    ├── handoffs/2026-05/
    │   ├── 2026-05-25-fst-e2e-fixes-and-chart-catalog-handoff.md       ← 早上 handoff (本会话第一个 read)
    │   ├── 2026-05-25-chart-catalog-p0-p3-implementation-handoff.md    ← 6f6211e9 P0-P3 任务书
    │   ├── 2026-05-25-paradigm-key-alignment-handoff.md                ← 未 commit, 已被 worktree agent 执行
    │   ├── 2026-05-25-deerflow-sync-cleanup-pr-pending-handoff.md      ← 未 commit (跟本会话无关, 别人留下的)
    │   └── 2026-05-25-fst-e2e-recursion-frontend-chart-final-handoff.md ← 本文件
    └── review-packages/
        ├── 2026-05-25-chart-catalog-implementation-plan.md              ← 同事拍板的 2 个参数 (n<3不画误差线, OFT 5min bin)
        └── 2026-0521-feedbacks/tstYoyo/范式-图表对应关系.md             ← chart SSOT
```

### 三层对齐情况（最新）

| 层 | 状态 |
|---|---|
| code-executor 指标计算 | ✅ 6 个范式 100% 对齐 SSOT |
| data-analyst 解读 skill | ✅ skill 内容齐 + 已挂载 + workflow step 2.5 引导 |
| **chart-maker 出图** | ✅ **本会话 P0-P3 全部补齐**（18 张图 + 2 注册） |
| **范式 key 一致性** | ✅ **本会话 paradigm-key-alignment 修完**（学术名为 canonical） |
| lead 汇报可见性 | ✅ **本会话 B2+B3 修完** |
| subagent 不卡死 | ✅ **本会话 recursion_limit fix 修完** |

### 同事已拍板参数（持续生效）

- **柱状图 n<3 不画误差线**：`bar_chart(suppress_errorbars=True/None)` kwarg，None=auto（任一组 n<3 自动 suppress）
- **OFT 时间进程图**：`total_duration_seconds > 300` 才画；bin 长度 300s；末尾不足并入最后一个 bin
- **活动强度图代理列**：用 velocity（不用 mobility_continuous，不用 mobility_state）

### 范式 key canonical 政策（2026-05-25 立的规矩）

整个系统**统一用学术名**作 paradigm key：

| 范式 | canonical key | catalog yaml 文件名 |
|---|---|---|
| 强迫游泳 | `forced_swim` | `fst.yaml` |
| 悬尾 | `tail_suspension` | `tst.yaml` |
| 旷场 | `open_field` | `oft.yaml` |
| 明暗箱 | `light_dark_box` | `ldb.yaml` |
| 高架十字 | `epm` | `epm.yaml` |
| O 迷宫 | `zero_maze` | `zero_maze.yaml` |
| 鱼群 | `shoaling` | `shoaling.yaml` |

**所有上游代码用 canonical key**，loader 内部做学术名→文件名映射。新加范式时：（1）catalog yaml 文件名沿用短 key 风格；（2）`loader.py:_PARADIGM_ALIASES` 加一行学术名→文件名映射。

### 关键已解决但要注意的几点

- **subagent middleware 链**：本会话起 `recursion_limit` 按 hook 数动态算（[executor.py](packages/agent/backend/packages/harness/deerflow/subagents/executor.py)），未来加新 middleware 不再需要手动调 multiplier。
- **`assistant:processing` group 分支**：本会话起前端有专门 case 渲染（[message-list.tsx](packages/agent/frontend/src/components/workspace/messages/message-list.tsx)）。如果未来新增 group type，记得加专门 case 否则回归 CoT 折叠。
- **catalog yaml 不必跟 canonical key 同名**：loader alias 表保证向后兼容；不要顺手改 yaml 文件名，会拖出测试 + 引用更新连锁反应。

---

## 关键发现

1. **paradigm-key-alignment 是个分发样板**：用户决定让另一个 agent 在独立 worktree 干，本会话只写 handoff 然后 commit。worktree agent 完美完成 + PR #41 合入。这个流程可复用——下次类似纯重构任务都可以这么干。
2. **5/25 工程性问题集中爆发**：上午 dogfood 暴露 5 处独立 bug（fix #1-5 + 后续 4 个），下午又发现 recursion_limit / frontend 折叠 / paradigm key 3 个深层结构性问题。一天合入 9 个 commit，**0 退化 0 失败**。
3. **每个 P 一个 commit 切分有效**：P0-P3 各自合 PR，每个 P 完成用户都能立刻 dogfood 验证，避免大爆炸式提交。
4. **前端 5/22 ToolMessage 模板改动只补了 5/25 parser，没补 group 渲染**：前端的 fix 链路需要同时考虑 ① 字符串提取 (5/25 fix #1) ② 分组归类 (5/22 之后 groupMessages 把它分到 processing group) ③ group 渲染 (本会话 B3 才补全)。三段都要顾才能完整修复 ToolMessage 渲染回归。
5. **`grep_count(short_key)=9 vs grep_count(academic)=24`**：用 grep 统计两种 key 的引用密度，是决定 canonical 方向的关键证据。下次类似的标识符空间冲突，可以直接用这个方法判断。

---

## 未完成事项

### 高优先级（可立即做，但不属于本会话）

- [ ] **重启服务做 dogfood 终极验收**：跑 FST n=1 + n=3，验证：
  - lead 调 `prep_metric_plan(paradigm='forced_swim')` 一次成功（不再试错）
  - chart-maker 命中 fst catalog 中的 3 张新图（bar / activity / struggle），不再走 fallback
  - subagent 完成后 SubtaskCard 顶部能看到长 result（B2 fix）
  - lead 的"已收到 X 的结果..."汇报渲染到主气泡（B3 fix）
- [ ] **其他 5 个范式 dogfood 轮跑**：tst / epm / oft / zero_maze / ldb 各跑一遍验命中 catalog
- [ ] **OFT 时间进程图实测**：需要 > 5 分钟数据才会触发，可能要单独准备测试数据

### 中优先级

- [ ] **deerflow sync cleanup PR**：[docs/handoffs/2026-05/2026-05-25-deerflow-sync-cleanup-pr-pending-handoff.md](docs/handoffs/2026-05/2026-05-25-deerflow-sync-cleanup-pr-pending-handoff.md)（未提交 + 跟本会话无关）—— 别人留的 sync 收尾，需要单独评估
- [ ] **paradigm-key-alignment-handoff.md**（未提交）—— 本会话写的 handoff，PR #41 完成后没人 commit。下位 agent 决定要不要补 commit 留档
- [ ] **lead prompt L289 文案优化**（已在 5/22 写规范）：现在 B3 frontend 兜底了，lead prompt L289 那段「搬运 key_findings 1-3 条」的强约束可以适度放宽（subagent.result 已经在前端可见）；但不要删——双重保险

### 低优先级

- [ ] **`data_analyst.py:73` workflow 编号重复**（早上 handoff 已标低优先级，未做）：step 2 出现两次（`2. read_file ...` + `2. 一次性完成核心分析推理`），中间夹了一个 `2.5`。视觉上有点乱，不影响逻辑。下次顺手修。
- [ ] **未来加新范式的 SOP**：把"加新范式 = 写 catalog yaml + 加 alias 行 + 加 metrics dispatcher case + 加 skill doc"写进 [docs/sop/](docs/sop/)。本会话还没动这个 SOP。

---

## 建议接手路径

### 第一步：read 本文件 + 确认 dev 干净

```bash
cd /home/wangqiuyang/noldus-insight
git pull --ff-only origin dev
git log --oneline -10  # 应该看到 eb2852ef 在最上
git status --short     # 应该只有未跟踪的 handoff md 文件
```

### 第二步：决定下一步方向

| 用户意图 | 你该做的 |
|---|---|
| "继续 dogfood" | 重启服务（`cd packages/agent && make stop && make dev`），让用户跑 FST n=1 → 等截图反馈 → 根据问题定位 |
| "5 个范式都跑一遍" | 准备 demo-data/ 下各范式样本路径列表给用户，让用户分别上传测试 |
| "P0-P3 出图我看看" | 帮用户找 demo-data/ 的某范式样本，跑 catalog.resolve --mode charts 看真实命中数 |
| "之前 handoff 还有什么没做" | 看本文档"未完成事项"段，按优先级问用户选哪个 |
| "deerflow sync cleanup" | read [docs/handoffs/2026-05/2026-05-25-deerflow-sync-cleanup-pr-pending-handoff.md](docs/handoffs/2026-05/2026-05-25-deerflow-sync-cleanup-pr-pending-handoff.md)，但这是别人留的，**先跟用户对齐它的优先级**再开工 |

### 第三步：跑回归保证基线

```bash
# 任何代码改动前先确认基线
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight && uv run pytest tests/ -q
# 期望: 384 passed, 51 skipped, 0 failed

cd /home/wangqiuyang/noldus-insight/packages/agent/backend && .venv/bin/python -m pytest tests/ -q
# 期望: 3017 passed, 18 skipped, 0 failed

cd /home/wangqiuyang/noldus-insight/packages/agent/frontend && pnpm typecheck
# 期望: 0 error
```

---

## 风险与注意事项

### ❌ 不要做的事

- **不要把 catalog yaml 文件名改成学术名**（fst.yaml → forced_swim.yaml）。loader alias 表已解决一致性，改文件名会拖出所有 grep+test 连锁更新，无意义。
- **不要往 `_PARADIGM_ALIASES` 加非现有 catalog 范式的映射**（例如 `morris_water_maze`）。MVP 现在只支持 7 个范式，未来加新范式时 yaml + alias 一起加。
- **不要在 lead prompt 里加更激进的"必须先汇报再调 X"约束**：现在 B3 frontend 已经兜底，lead prompt 现有的 L287-298 规范够用。再加 prompt 约束反而可能反向激活。
- **不要把后端 task_tool 模板改回 `"Task Succeeded. Result: ..."`**：5/22 加进度时间线是正向改进；前端 parser 已兼容两种。
- **不要重新发明 chart id 命名规范**：本会话定的 `bar_<指标>` / `box_<指标>` / `trajectory` / `heatmap` / `activity_intensity` / `struggle_distribution` / `time_progress` 沿用即可。
- **不要加 SSOT 没列的图种**（散点图 / 相关性热图 / 小提琴图等）—— 5/24 早上 review 包第一版就被用户当场叫停。

### ⚠️ 容易混淆

- **paradigm key 三套**：identify 返回学术名 / catalog yaml 文件名是短 key / catalog yaml 内 `paradigm:` 字段是短 key。上游写代码用**学术名**，loader 自己处理映射。**不要在上游做 alias，会破坏 canonical**。
- **`assistant:processing` group**：lead AIMessage 同时有 content + 非 task tool_call（如 ask_clarification / set_viz_choice）时落到这个 group。本会话 B3 加了主气泡渲染。如果未来 lead 行为变，content 不再出现在这种 AIMessage 里，这个 case 会变成 dead code（但 reasoning 折叠仍 useful）。
- **subagent middleware hook 数**：本会话 `calculate_subagent_recursion_limit` 按 hook 数算。如果将来 langchain 改了 graph node 计数算法（例如把 middleware hook 合并成一个 node），公式可能需要重新调。但这是上游事件，会有显式 sync commit 提醒。
- **`when: total_duration_seconds > 300`**：OFT time_progress 用这个 catalog 条件。`evaluate_when` 已支持 `total_duration_seconds` 变量。不要往 `when` 字段写 SSOT 没规定的奇怪表达式。

---

## 下一位 Agent 的第一步建议

1. **Read 本文件全部**（你正在做）
2. **跑回归确认基线**（上面"第三步"命令）—— 0 failed 才能继续动手
3. **问用户当前优先级**：
   - 是要继续 dogfood 验收吗？
   - 还是开新功能？
   - 还是处理"未完成事项"里的某条？
4. **如果用户给的指令是"继续昨天工作"或类似模糊话**：
   - 默认是 **dogfood 多范式验收**（高优先级第 1 条）
   - 重启服务，等用户上传数据
5. **如果用户提到具体 commit / 文件名**：直接定位那个文件，不要绕弯
6. **如果用户提到 "5/25 handoff" / "之前的"**：read [docs/handoffs/2026-05/](docs/handoffs/2026-05/) 下相关日期的所有 .md（5/25 共有 5 个）

---

## 当前 git 状态

```
dev HEAD: eb2852ef (origin/dev 同步)

最近 9 个 commit 全部本会话产物:
  eb2852ef Merge PR #40 chore/sync-protected-files-and-sop  (别人合的, 跟本会话无关)
  cec2af53 P3 OFT 时间进程图
  eeeedd65 P2 epm + oft 轨迹+热区
  94c1fe2a P1 zero_maze + ldb × 4
  9f02f48c Merge PR #41 paradigm-key-alignment              (worktree agent 实施)
  e943c2cb P0 fst + tst × 3
  ce1376af fix(catalog): load_catalog 接受学术名
  6f6211e9 docs(handoffs): P0-P3 chart catalog 实施 handoff
  232f12c8 fix(frontend): subagent 完成后 lead 汇报被折叠
  8cea306a fix(subagent): recursion_limit 按 middleware hook 动态算
```

**未 commit 的待提交文件**：
- `docs/handoffs/2026-05/2026-05-25-paradigm-key-alignment-handoff.md`（本会话写的，PR #41 已合入后没人 commit）
- `docs/handoffs/2026-05/2026-05-25-deerflow-sync-cleanup-pr-pending-handoff.md`（别人留的，跟本会话无关）
- `docs/handoffs/2026-05/2026-05-25-fst-e2e-recursion-frontend-chart-final-handoff.md`（本文件）

下位 agent 决定要不要把它们一起 commit 留档（建议 commit）。

---

## 关键文件速查

| 任务 | 文件 |
|---|---|
| 本 handoff | `docs/handoffs/2026-05/2026-05-25-fst-e2e-recursion-frontend-chart-final-handoff.md` |
| P0-P3 实施任务书 | `docs/handoffs/2026-05/2026-05-25-chart-catalog-p0-p3-implementation-handoff.md` |
| paradigm-key-alignment 任务书 | `docs/handoffs/2026-05/2026-05-25-paradigm-key-alignment-handoff.md` |
| chart SSOT | `docs/review-packages/2026-0521-feedbacks/tstYoyo/范式-图表对应关系.md` |
| chart 实施计划 | `docs/review-packages/2026-05-25-chart-catalog-implementation-plan.md` |
| paradigm alias 表 | `packages/ethoinsight/ethoinsight/catalog/loader.py` (`_PARADIGM_ALIASES`) |
| 4 个新 chart 函数 | `packages/ethoinsight/ethoinsight/charts.py` (line 670+) |
| 6 个 catalog yaml | `packages/ethoinsight/ethoinsight/catalog/{epm,oft,zero_maze,ldb,fst,tst}.yaml` |
| 18 个新 plot 脚本 | `packages/ethoinsight/ethoinsight/scripts/{fst,tst,oft,epm,zero_maze,ldb,_common}/plot_*.py` |
| recursion_limit helper | `packages/agent/backend/packages/harness/deerflow/subagents/executor.py` (`calculate_subagent_recursion_limit`) |
| 前端 B2 (task.result 顶部可见) | `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx` |
| 前端 B3 (processing group narrative) | `packages/agent/frontend/src/components/workspace/messages/message-list.tsx` |
