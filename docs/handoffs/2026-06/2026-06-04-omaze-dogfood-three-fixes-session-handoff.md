# 2026-06-04 O迷宫 dogfood 三修复会话 交接文档

> **面向下一个 AI Agent**。本会话以 O迷宫全链路 dogfood 为轴心，修复并验证了三个独立 harness 问题，全部推送到 `origin/dev`，O迷宫 dogfood 已跑通。仓库 HEAD = `48ac5003`（origin/dev）。

---

## 1. 当前任务目标

本会话已完成：O迷宫 dogfood 暴露的三个 harness 问题全部修复、测试绿、dogfood 真实验证、push。

无残留的「必须立即接手」事项——下一个 Agent 可从新的 dogfood 任务或其他优先级事项开始。

---

## 2. 当前进展

### ✅ 本会话三个修复（全部已 commit + push 到 origin/dev + dogfood 验证）

| commit | 问题 | 验证 |
|--------|------|------|
| `e5a0d53b` | data-analyst seal-deadlock（step 2.8 非空参数路：`open_zones` 触发 a–f 长分支卡死） | ✅ dogfood 一次 seal |
| `12cf2c79` | chart-maker 出图全失败（`catalog.resolve` + `parse.dump_headers` 未在 guardrail 白名单） | ✅ dogfood 出图跑通 |
| `d3026f31` | lead 反复重问"要不要画图"（`TodoMiddleware.after_model` 误判等待为提前退出 → 强制重唤起） | ✅ dogfood 反问只问一次 |

### ✅ 文档产出

- `docs/handoffs/2026-06/2026-06-04-upload-scale-and-zone-override-unification-handoff.md` — 读懂本会话起点（上一会话产出，文件上传扩容 + zone override 三范式统一）
- `docs/handoffs/2026-06/2026-06-04-clarification-reask-loop-investigation-brief.md` — 反问循环根因 brief（`TodoMiddleware`，含 checkpoint 解码证据）
- `docs/handoffs/2026-06/2026-06-04-chart-maker-catalog-resolve-guardrail-block-brief.md` — chart-maker guardrail 根因 brief（含 git 史证、精准修法）
- `docs/sop/2026-06-04-playwright-multi-paradigm-dogfood-test-guide.md` — 6 范式 Playwright dogfood 端到端测试指南

### ✅ Memory 落档

3 条 memory 新建/更新（均经真实 dogfood 验证）：
- `feedback_subagent_seal_deadlock_is_prompt_not_budget` — 加第 6 点：非空参数路是独立复发点
- `feedback_chart_maker_bash_guardrail_must_allow_resolve_dumpheaders`（新建）
- `feedback_todo_middleware_must_not_force_reengage_while_awaiting_clarification`（新建）

---

## 3. 关键上下文

### 仓库 HEAD 与分支

- 分支：`dev`，与 `origin/dev` 同步（HEAD `48ac5003`，并行工作有 2 个额外 commit 也已 push）
- **工作区有 2 个不属于本会话的 modified 文件**：`CLAUDE.md` + `docs/milestone/README.md`。这是并行工作的脏改动，**不要提交或回退**。

### O迷宫 demo data 路径

```
/home/wangqiuyang/DemoData/newdemodata/O迷宫/
  原始数据-oMaze-试验 1.xlsx   (Antagonist)
  原始数据-oMaze-试验 2.xlsx   (Seratonin4-saline)
  原始数据-oMaze-试验 3.xlsx   (Seratonin4-agonist)
```

### 成功 dogfood thread

- thread `ca86744f-39e6-4acb-95fa-bda8ec8593ed`（user `cd95effa-...`）
- 时间：2026-06-04 11:21（重启后第一次跑通的那次）
- workspace：`.deer-flow/users/cd95effa-.../threads/ca86744f-.../user-data/workspace/`
  - `handoff_code_executor.json` ✅
  - `handoff_data_analyst.json` ✅（status=completed，parameter_audit_findings=1 info，category_mismatch open_zones）
  - **无** `handoff_chart_maker.json`（chart-maker 在那次 thread 失败过，guardrail 修复在此 commit 之后）

### 三个修复技术速查

**① data-analyst step 2.8 非空参数路（e5a0d53b）**
- 文件：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`
- 在 step 2.8 非空分支入口（`若至少有一个 metric 的 parameters_used 非空` 之后、a–f 之前）插了：
  - "本段至多 2-3 轮思考，到点立即发 seal tool_call"前置
  - 离散/类别参数（`open_zones` 等）+ 范式无判据 → 直接记一条 info finding → 立即 seal 的捷径
- 测试：`tests/test_data_analyst_step28_contract.py`（10 条，含 2 个新红锚点）

**② chart-maker guardrail 白名单（12cf2c79）**
- 文件：`packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py:40`
- 旧正则：`ethoinsight\.scripts\.\w+\.\w+`
- 新正则：`ethoinsight\.(scripts\.\w+\.\w+|catalog\.resolve|parse\.dump_headers)`
- SKILL.md 同步补了 dump_headers 前置步骤：`packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md`
- 测试：`tests/test_script_invocation_only_provider.py`（25 条，含 5 个新红锚点含 lookalike 仍 deny）

**③ TodoMiddleware 等待澄清短路（d3026f31）**
- 文件：`packages/agent/backend/packages/harness/deerflow/agents/middlewares/todo_middleware.py`
- 加了 `_is_awaiting_clarification(messages)` helper：从消息尾向前扫，先遇非控制 Human=已答 False，先遇 `ToolMessage(name="ask_clarification")`=仍等待 True；自注入的控制 HumanMessage 跳过
- `after_model` step 2.5 处短路：等待态 `return None`（不催不 jump）
- prompt 兜底：`prompt.py:464` 附近加"已发 ask_clarification、用户未答时保持静默等待"
- 测试：`tests/test_todo_middleware.py`（58 条，+8）+ `tests/test_lead_prompt_interactive_pipeline.py`（17 条，+2）

### 全量回归基线

- **3693 passed**，4 failed 为已知顺序 flake（`test_deferred_tool_registry_promotion` ×2 + `test_async_delegates_to_sync` ×2），clean HEAD 单跑 4/4 passed，与这批改动无关。

---

## 4. 关键发现（已验证，可直接采信）

1. **zone override PR #89 完全生效**：`parameter_overrides: {"anonymous_zone_is":"in_zone"}` → `open_zones:["in_zone"]`（list），4 个指标出真值（6.4%/10.7%/14.2%），无 `~in_zone` 脑补。这是上一会话修复，本会话 dogfood 验证。

2. **data-analyst 的 open_zones 走离散参数捷径**：dogfood 里 data-analyst 产出 `parameter_audit_findings=[{parameter:"open_zones", mismatch_kind:"category_mismatch", severity:"info"}]`——这是正确行为，不是 bug。

3. **前端 console error `Unexpected tool message outside a processing group` 大概率是次生现象**：由之前的 chart-maker 失败 + 反问循环共同打乱前端 message 分组逻辑。后端两个根因修完后需复测确认是否消失；若仍在查 `frontend/src/core/messages/utils.ts:83`（`getMessageGroups`）对孤立 ToolMessage 的容错。

4. **并行工作产出**：HEAD `48ac5003` 有两个不属于本会话的额外 commit（`722b63bd` code-executor handoff 校验器加固 + `48ac5003` task_context.pending_items bug 修），已进 origin/dev，本会话修复在其之前。

5. **catalog.resolve `--columns-file` 是必填参数**（查 `--help` 实测）—— chart-maker 的 `dump_headers` 步骤是真实必需，不是 subagent 脑补。`parse.dump_headers` 模块真实存在可运行。

---

## 5. 未完成事项（按优先级）

### 🟡 中优先级

- **chart-maker 全链路 dogfood 复测**：guardrail 修了但本会话未验证 chart-maker 出图（那次成功 thread 是在修复前失败的）。需重跑 O迷宫（或其他范式）验 chart-maker → dump_headers → catalog.resolve → plot 脚本 → 出 png 全链路。参考 dogfood 指南：`docs/sop/2026-06-04-playwright-multi-paradigm-dogfood-test-guide.md`。

- **其他范式 dogfood**：本会话只验了 O迷宫。Playwright 测试文档已写好（6 范式，优先 FST + EPM + LDB）。

- **前端 console error**：`Unexpected tool message outside a processing group`。先跑一次完整 dogfood 看是否自动消失；若仍在查 `getMessageGroups (src/core/messages/utils.ts:83)`。

### 🟢 低优先级 / 可选

- **文件上传 prod 生效**：`~/ethoinsight-prod/config.yaml` 已加 uploads 段（上一会话），还需 `cd packages/agent && make deploy-tar` 才线上生效（未执行）。

- **提交 untracked spec**：`docs/superpowers/specs/2026-06-04-zone-override-unification-three-paradigm-design.md`（上一会话产出，untracked）。

- **OFT 范式 dogfood**：验 center_zone str 路径 + 不误报 zone_unnamed（OFT 有命名列，不该触发 zone 反问）。

---

## 6. 建议接手路径

### 如果接手 chart-maker 复测

```bash
# 1. 重启 dev（guardrail 修复需要重启生效）
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop && make dev

# 2. 确认服务起来
ss -ltn | grep -E ':(2024|2026|3000|8001)'

# 3. 上传 O迷宫数据，发"分析数据"，走完到"画图"反问
# 数据：/home/wangqiuyang/DemoData/newdemodata/O迷宫/ 3 个 xlsx
# 反问时答"是，画图"

# 4. 验 chart-maker 出图：
TID=<新thread_id>
WS=$(find .deer-flow/users -type d -path "*threads/$TID/user-data/workspace")
ls -la "$WS"
# 期望：handoff_chart_maker.json 存在 + .png 文件存在
```

### 如果接手其他任务

直接读 `docs/milestone/README.md` 获取项目全局状态，然后按需读具体范式的 handoff 或 spec。

---

## 7. 风险与注意事项

- **dogfood 前必须 `make stop && make dev` 重启**：harness / prompt 改动在 agent 创建时载入内存，不重启跑的是旧代码。
- **工作区 2 个 modified 文件**（`CLAUDE.md` / `docs/milestone/README.md`）不属于本会话，**不要提交、不要回退**。
- **非确定性**：deepseek 是概率性的。chart-maker 建议多跑 1-2 次确认 dump_headers → catalog.resolve → plot 流程稳定通过。
- **O迷宫 in_zone 几何结论**：`in_zone=1` = 开放臂（动物回避，占时 6–14%）。反问时必须答"开放臂"。永不走取反思路。
- **那 4 个全量顺序 flake** 不是回退，clean HEAD 单跑通过，跑全量时正常出现不用追。

---

## 8. 关键路径速查

```
# 三个修复文件
packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py  # ① step 2.8 捷径
packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py  # ② guardrail 白名单
packages/agent/backend/packages/harness/deerflow/agents/middlewares/todo_middleware.py  # ③ 等待澄清短路
packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py  # ③ prompt 兜底

# 三个测试文件（改动同步）
packages/agent/backend/tests/test_data_analyst_step28_contract.py
packages/agent/backend/tests/test_script_invocation_only_provider.py
packages/agent/backend/tests/test_todo_middleware.py
packages/agent/backend/tests/test_lead_prompt_interactive_pipeline.py

# Demo data
/home/wangqiuyang/DemoData/newdemodata/{O迷宫,高架十字迷宫_小鼠_三点,旷场_小鼠_三点,明暗箱,强迫游泳_大鼠,悬尾}

# Dogfood 测试指南
docs/sop/2026-06-04-playwright-multi-paradigm-dogfood-test-guide.md

# 全量回归命令
cd packages/agent/backend && PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
```
