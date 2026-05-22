# 2026-05-21 FST E2E dogfood 7 处修复 + ASKVIZ 意图引入 + 前端 thinking 闪烁修

## 当前任务目标

**主线任务**: 用户对 5 个真实范式数据(`/home/wangqiuyang/DemoData/newdemodata/`)做端到端 dogfood,本次会话陪跑 FST 一个完整循环并修复发现的所有问题,然后用户继续跑其余 4 个范式(EPM / OFT / LDB / Zero Maze)。

**5 个真实范式 → 仓库映射(已确认全部对齐)**:

| 真实数据目录 | paradigm key | catalog yaml | scripts/ | review-pkg 状态 |
|---|---|---|---|---|
| `强迫游泳_大鼠/` | `forced_swim` (fst) | ✅ fst.yaml | ✅ fst/ | ✅ 同事已填 |
| `高架十字迷宫_小鼠_三点/` | `epm` | ✅ epm.yaml | ✅ epm/ | ✅ 同事已填 |
| `旷场_小鼠_三点/` | `open_field` (oft) | ✅ oft.yaml | ✅ oft/ | ✅ 同事已填 |
| `明暗箱/` | `light_dark_box` (ldb) | ✅ ldb.yaml | ✅ ldb/ | ✅ 同事已填(LDB ev19 facts 映射用 OFRect 兜底) |
| `O迷宫/` | `zero_maze` | ✅ zero_maze.yaml | ✅ zero_maze/ | ✅ 同事已填 |

**ethoinsight 全套测试**: 334 passed / 51 skipped / 0 failed
**agent backend 全套测试**: 2680 passed / 17 skipped / 1 failed(pre-existing memory router,与 baseline 一致)

## 当前进展(本次会话 6 个 commit 全 push 到 origin/dev)

```
HEAD = cd512536 (origin/dev)
├── cd512536 fix(intent): E2E_FULL_ASKVIZ 反问前 lead 必须先汇报 data-analyst 发现
├── 4d6fb37e fix(frontend): 流式 reasoning+content 时保持 group 连续性,消除 thinking 刷新闪烁
├── 58ae57ba fix(frontend): 统一双 thinking layout + 加 Lead Agent 徽章
├── bf6fd22d feat(intent): 新增 E2E_FULL_ASKVIZ 意图,模糊语义跑完解读再反问要不要出图
├── 7e144acb docs(catalog): raw-files-json 明示必须用虚拟路径,不可用宿主机绝对路径
├── 6bc554a4 fix(timeseries): FST/LDB/Shoaling 默认 y_col 映射 + 列不存在时防御性 fallback
└── 06c480c9 fix(sync-regression): 恢复 BUILTIN_TOOLS 的 3 个 EthoInsight 工具注册
```

### ✅ 已完成

1. **PR #23 sync regression 修复** (`06c480c9`) — 恢复被 sync 误删的 `BUILTIN_TOOLS` 3 个工具注册(set_experiment_paradigm / identify_ev19_template / prep_metric_plan)+ `tools/builtins/__init__.py` 的 2 个导出。Lead tools 从 5 个恢复到 8 个。

2. **timeseries 空图修复** (`6bc554a4`) — `_DEFAULT_Y_COL_BY_PARADIGM` 加 FST(mobility_continuous) / TST(mobility_continuous) / LDB(in_zone_light) / Shoaling(nearest_neighbor_distance) 默认映射 + 防御性 `_pick_fallback_y_col`(列不存在时自动选第一个数值列,避免未来新范式撞同样空图)。同时改 chart-maker SKILL 工作流明示 `--paradigm <p>` 必传。

3. **plan_charts.json 宿主机路径修复** (`7e144acb`) — 改 `resolve-cli.md` 明示 `--raw-files-json` 必须用虚拟路径(`/mnt/user-data/uploads/...`),不可填宿主机绝对路径。chart-maker SKILL 同步加约束。

4. **E2E_FULL_ASKVIZ 新意图** (`bf6fd22d`) — 引入第 3 个 E2E 意图,根据"出图意愿置信度"路由:
   - 模糊总称(分析/看看/研究下)→ `E2E_FULL_ASKVIZ` → code → data → ask(viz?) → ...
   - 明确出图(画/图/箱线/轨迹/表)→ `E2E_FULL` → code → data → chart → ask(report?)
   - 单 CALC → `E2E_MIN`
   配套改 `IntentClassificationGuardrailProvider._VALID_INTENTS` 加入新值。

5. **前端双 thinking layout 统一** (`58ae57ba`) — `MessageListItem` 的 `ReasoningPanel` 从 ai-elements/Reasoning(Shimmer 闪烁版)改成 ChainOfThought(与 MessageGroup 一致),两个位置都加 "Lead Agent" Badge。subagent 不动(已有 ClipboardListIcon + 子代理名)。

6. **前端 thinking 流式闪烁修复** (`4d6fb37e`) — `groupMessages` 加 `isStreaming` 参数。流式中最后一条 message 即使满足"reasoning+content+!tools"也保持 `assistant:processing`,避免组件 unmount/remount。`message-list.tsx` 调用时传 `{ isStreaming: thread.isLoading }`。

7. **lead ASKVIZ 反问前必须先汇报** (`cd512536`) — 修第 6 commit 引入的新缺陷:用户反馈"data-analyst 出结果后 lead 没在 think 外做总结"。改 prompt.py + intent-decision-tree.md,明示反问模板**两步流程**:先发 AIMessage 搬运 key_findings,再 ask_clarification。配套加防回归测试。

### ⏸ 暂不动(已分析,等用户拍板)

- **lead 调 task(general-purpose) 的现象** — thread f3fbce44 截图显示 lead 调过 general-purpose,根因是 `task_tool.py` 的 docstring 写死 "Built-in subagent types: general-purpose / bash"(LangChain `@tool(parse_docstring=True)` 把 docstring 灌给 LLM 作工具描述)。Noldus 在 prompt.py 已把 general-purpose 从 `available_names` 排除,但 docstring 仍诱导 LLM 调用。`BUILTIN_SUBAGENTS` dict 不含 general-purpose → task_tool 会返回 "Unknown subagent type"错误,但前端 stage broadcast fallback 显示成 "🛠 正在派遣 general-purpose..." 让人误以为成功了。

  **现状**: thread 7456611e(本次会话最新)未再复现,可能是 prompt 改动副作用让 lead 不再尝试。继续观察其余 4 范式是否复现,如复现再修(改 task_tool docstring 是受保护文件,需 surgical)。

- **n=1 时 lead 派 chart-maker / lead 派 subagent 改 experiment-context.json** — thread f3fbce44 的 2 个老问题。但 7456611e 的 ASKVIZ 流程下没复现(因为 data-analyst 已包含 method_warnings,lead 不再先派 subagent 改 gate)。继续观察。

## 关键上下文

### 仓库现状

- **主分支** dev = `cd512536`(本地 = 远程,完全同步)
- **没有 active worktree**,主仓干净
- HEAD 比 sync 完成时(`2b5100e8`)往前推了 6 个 dogfood 修复 commit

### 关键文件路径

| 用途 | 路径 |
|---|---|
| 项目根 CLAUDE.md | `/home/wangqiuyang/noldus-insight/CLAUDE.md` |
| Backend CLAUDE.md | `/home/wangqiuyang/noldus-insight/packages/agent/backend/CLAUDE.md` |
| Frontend CLAUDE.md | `/home/wangqiuyang/noldus-insight/packages/agent/frontend/CLAUDE.md` |
| Lead prompt | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` |
| Intent guardrail | `packages/agent/backend/packages/harness/deerflow/guardrails/intent_classification_provider.py` |
| Lead-interaction SKILL | `packages/agent/skills/custom/ethoinsight-lead-interaction/` |
| Chart-maker SKILL | `packages/agent/skills/custom/ethoinsight-chart-maker/` |
| Chart-maker 默认 y_col 映射 | `packages/ethoinsight/ethoinsight/scripts/_common/plot_timeseries.py` |
| BUILTIN_TOOLS 注册 | `packages/agent/backend/packages/harness/deerflow/tools/tools.py` |
| 前端 groupMessages | `packages/agent/frontend/src/core/messages/utils.ts` |
| 前端 message 渲染 | `packages/agent/frontend/src/components/workspace/messages/` |
| 真实数据目录 | `/home/wangqiuyang/DemoData/newdemodata/` |
| review-package | `/home/wangqiuyang/noldus-insight/docs/review-packages/2026-04-29-ev19-templates/` |
| 当前 langgraph.log | `/home/wangqiuyang/noldus-insight/packages/agent/logs/langgraph.log` |

### 关键测试基线

| 套件 | 命令 | 期望 |
|---|---|---|
| ethoinsight | `cd packages/ethoinsight && uv run pytest tests/ --tb=no -q` | 334 passed / 51 skipped / 0 failed |
| agent backend | `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/ --ignore=tests/test_client_live.py --tb=no -q` | 2680 passed / 17 skipped / 1 failed |
| frontend vitest | `cd packages/agent/frontend && pnpm test` | 15 passed |
| frontend typecheck | `cd packages/agent/frontend && pnpm typecheck` | 干净 |

**唯一已知 fail**: `test_memory_router::test_update_memory_fact_route_preserves_omitted_fields` — **pre-existing,与 baseline 一致**。要修的话改 `tests/test_memory_router.py:261` 的 `assert_called_once_with(...)` 加 `user_id='test-user-autouse'`。

### 测过 FST 的 thread

| thread_id | 何时 | 状态 |
|---|---|---|
| `21d5527c-e20f-4405-ab55-a5e5a80c56b2` | sync regression 触发前(2026-05-21 ~14:00) | ❌ Lead tools 5→撞 LoopDetection 硬限 |
| `f3fbce44-2b61-48a9-ae3d-442d9164dbd0` | sync regression 修复后第一次跑 | ✅ 完整跑通但有 5 个 UX/架构 issue |
| `7456611e-0d43-4deb-a858-231c1c7dcb53` | 本次会话 6 commit 修完后最后一次跑 | ✅ 跑通,thinking 闪烁 + 没汇报 2 个新问题 → 现已修(`4d6fb37e` + `cd512536`) |

## 关键发现

### 1. PR #23 sync 误删受保护注册项的教训

`tools/tools.py` 和 `tools/builtins/__init__.py` 在 sync `598667e6` 中被当成"安全文件"整体接受上游,洗掉了 Noldus 3 个工具注册。这条教训已写到 memory `feedback_sync_protected_files_registry_loss.md`。**含 `BUILTIN_*` / `__all__` / 聚合 import 块的文件都要 surgical merge**。

### 2. E2E_FULL_ASKVIZ 设计哲学

用户原话:「我觉得复合语义中,我们可以在 data-analyst 出结果后,反问 ask clarification 用户要不要出图。而不是直接做。除非用户明确说了出图,表格相关的话」。

实现:**置信度路由**,不是流程变更。最终都会问报告,只是"出图反问"的位置从"无脑画完再问"变成"低置信度时先问再画"。

**触发词清单**(高置信 = 跳过反问,直接 chart-maker):
- 直接"图"概念:画/图/可视化/画出来/出图/图表
- 表格类(视为可视化):表/表格/列出来/一览表
- 具体图种名:箱线/箱型/轨迹/趋势/热图/散点/柱状/时序
- 明确展示意图:用图说/展示/呈现

低置信(走 ASKVIZ):分析一下/帮我看看/研究下/整一下。

### 3. 前端 LLM 流式输出最佳实践

主流 web app(ChatGPT / Claude.ai / Cursor)的做法:
- **完成态切换一次**: 流式过程保持单一容器,完成后做最终态视觉切换
- **组件 unmount 是闪烁元凶**: React key 变化或父子组件类型切换都会 unmount,流式过程必须保证 message → group 映射稳定

我的修法保证 last message 流式中始终在 MessageGroup(processing) 渲染,流结束后才切到 MessageListItem,符合"完成态切换"模式。

### 4. task_tool docstring 是 LLM 看到的工具描述

LangChain `@tool(parse_docstring=True)` 把整段 docstring 灌给 LLM。task_tool 的 docstring 写死了"Built-in subagent types: general-purpose / bash",**即使 prompt.py 在 system prompt 里只列 5 个 EthoInsight subagent,LLM 仍会被 docstring 诱导**。这是上游 deerflow 的设计,改 docstring 是受保护文件 surgical merge。先观察是否复现。

### 5. dev server / frontend 修改后必须重启

- 改 prompt.py 必须 `make stop && make dev`(LangGraph server 重启加载新 prompt)
- 改前端代码必须 frontend dev server 重新 build
- 用户已经 push 到 origin/dev,但需要本地重启 dev server 才生效

## 未完成事项

### 🔴 高优先级: 继续跑 4 个范式 dogfood

用户原话:"a,我正在手动跑给你看,等我消息"。规则:每跑完一个,用户给 thread_id + 现象,看 langgraph.log 排坑。

| 顺序 | 范式 | 真实数据 | 注意点 |
|---|---|---|---|
| 2 | EPM | `/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点/` | catalog 已完整 |
| 3 | OFT | `/home/wangqiuyang/DemoData/newdemodata/旷场_小鼠_三点/` | catalog 已完整 |
| 4 | LDB | `/home/wangqiuyang/DemoData/newdemodata/明暗箱/` | ⚠️ ev19_facts 映射兜底 = OpenFieldRectangle-Subdivided2x2,行为学同事 review 后可能要修映射 |
| 5 | Zero Maze | `/home/wangqiuyang/DemoData/newdemodata/O迷宫/` | catalog 已完整 |

**前置**: 用户**重启 dev server + frontend dev server** 加载本次 6 commit 改动。

### 🟡 中优先级: 观察是否复现的 2 个老问题

1. **lead 调 task(general-purpose) 的现象** — 如果 EPM/OFT/LDB/Zero Maze 任一复现,需要改 task_tool docstring(受保护文件,surgical merge)
2. **n=1 数据时 lead 提前派 subagent 改 experiment-context.json** — ASKVIZ 流程下大概率不再出现,但若复现需要给 lead 加 `acknowledge_quality_gate` 专用工具

### 🟢 低优先级: 完成所有范式后

- 写一个 backend 测试套用 `DemoData/newdemodata/` 真实数据 + mock lead 决策,验证 5 范式 Gate 1 → set_experiment_paradigm → prep_metric_plan → code-executor 派遣链路。这能拦截未来的 sync regression / catalog schema 漂移。
- LDB ev19_facts 映射修正(待行为学同事 PR 后)

## 建议接手路径

### 第 1 步: 读 handoff + CLAUDE.md 摸清现状

```bash
cat /home/wangqiuyang/noldus-insight/docs/handoffs/2026-05/2026-05-21-fst-e2e-7-fixes-and-askviz-intent-handoff.md  # 本文档
cat /home/wangqiuyang/noldus-insight/CLAUDE.md
cat /home/wangqiuyang/noldus-insight/packages/agent/backend/CLAUDE.md
```

### 第 2 步: 确认基线

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline -3    # HEAD 应该是 cd512536
git status              # 应该干净

# ethoinsight
cd packages/ethoinsight && uv run pytest tests/ --tb=no -q | tail -3
# 期望: 334 passed, 51 skipped

# agent backend
cd ../agent/backend && PYTHONPATH=. uv run pytest tests/ --ignore=tests/test_client_live.py --tb=no -q | tail -3
# 期望: 2680 passed, 17 skipped, 1 failed (pre-existing)
```

### 第 3 步: 根据用户当前的关注点选择路径

**如果用户继续跑范式 dogfood (默认)**:
1. 用户跑完一个范式给 thread_id + 现象,你
2. 看 langgraph.log: `grep "<thread_id>" /home/wangqiuyang/noldus-insight/packages/agent/logs/langgraph.log | grep -oE "(Background run|frequency|Guardrail|identify_ev19|prep_metric_plan|set_experiment|task_)[^[]*"`
3. 排根因 + 提修复方案,**等用户授权再动代码**
4. 修完 commit(不 push,等用户 push)

**如果用户报新问题不在本文档范围**:
- 优先 grep 代码,定位代码路径再分析根因
- 不要假设,直接读 log 找事实
- 出具诊断报告 + 提 2-3 个可选方案让用户挑

## 风险与注意事项

### ⚠️ 易混淆点

1. **dev server 必须重启才能加载 prompt/code 改动** — 用户每次问"为什么没效果"先确认是否重启了
2. **frontend dev server 也要重启** — 前端代码改动需要重新 build
3. **测试 fail ≠ 你引入的** — 1 个 pre-existing memory router fail 是 baseline,与本次会话所有改动无关
4. **5 个真实数据的 paradigm key 映射** — 见上方第一张表,不要按目录中文名猜
5. **langgraph.log 是单个文件累积** — 多个 thread 的 log 混在一起,用 grep `<thread_id>` 过滤

### ⚠️ 不建议的方向

- ❌ 改 `task_tool.py` docstring(受保护文件,需 surgical merge,先观察是否真复现 general-purpose 问题)
- ❌ 直接覆盖 `prompt.py` / `agent.py` / `executor.py` 整文件
- ❌ 不重启 dev server 就让用户重测
- ❌ 改 `aupdate_memory` 回 async(已选 sync 委托,见 [[2026-05-21 DeerFlow upstream sync 完成]] memory)
- ❌ 在用户没明确说"出图"/"画图"/"表格"时跳过 ASKVIZ 反问直接派 chart-maker

## 下一位 Agent 的第一步建议

**如果用户继续此线程(发 thread_id 报范式 dogfood 结果)**:

1. 读 langgraph.log 那个 thread 段落:
```bash
grep "<thread_id>" /home/wangqiuyang/noldus-insight/packages/agent/logs/langgraph.log | grep -oE "(Background run|frequency|Guardrail|identify_ev19|prep_metric_plan|set_experiment|task_|error)[^[]*"
```

2. 看用户截图 / 描述的现象,对照 prompt.py + skill 文档查根因

3. 提出修复方案让用户授权

**如果是全新会话**:

```bash
# 1. 读本 handoff
cat /home/wangqiuyang/noldus-insight/docs/handoffs/2026-05/2026-05-21-fst-e2e-7-fixes-and-askviz-intent-handoff.md

# 2. 读两个 CLAUDE.md
cat /home/wangqiuyang/noldus-insight/CLAUDE.md
cat /home/wangqiuyang/noldus-insight/packages/agent/backend/CLAUDE.md

# 3. 确认基线
cd /home/wangqiuyang/noldus-insight && git log --oneline -1  # cd512536
git status                                                    # 干净

# 4. 问用户: EPM / OFT / LDB / Zero Maze 哪个先跑?
```

**用户当前的 mental model**: FST E2E 已基本跑通,剩 4 范式排队。前端 thinking 闪烁 + lead 没汇报 这 2 个新引入的问题已在 commit `4d6fb37e` + `cd512536` 修了,但**用户需要重启 dev server + frontend 才能验证修复**。

---

**本次会话产物**:
- 7 个 commit(`06c480c9` ... `cd512536`)已合 origin/dev
- ethoinsight 334 passed(+3 新 timeseries 测试)
- agent backend 2680 passed(+1 新 ASKVIZ 汇报测试)
- frontend vitest 15 passed(+3 新 streaming 连续性测试)
- 2 条新 memory: `feedback_sync_protected_files_registry_loss.md` + `project_2026-05-21_fst_tool_registration_regression_fix.md`
- 本 handoff 文档
