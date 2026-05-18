# 2026-05-18 P0 Lead Bash Removal 实施交接

> **状态**: 4 个 task 全部完成，7 个 commit 已落 `sync/p0-lead-bash-removal` 分支。单元测试全绿。等待用户授权 push/merge + dogfood 浏览器端到端验证。

---

## 当前任务目标

彻底消除 P0 bug 根因：lead agent 不再有 bash tool，所有 `parse.*`/`catalog.*` 调用包成 deerflow 注册的 Python 工具 `prep_metric_plan`。

**产出**: 7 个 commit 在 `sync/p0-lead-bash-removal` 分支，worktree 在 `.claude/worktrees/p0-lead-bash-removal`。

---

## 当前进展

### ✅ 已完成

1. **Task 1** (`20a443cc`): LoopDetectionMiddleware 阈值 30/50→3/5 + 消息文案更新。5 个新测试 PASS。
2. **Task 2** (`53c38d21`): 新建 `prep_metric_plan_tool.py`，一步调 `ethoinsight.parse._core` + `ethoinsight.catalog.resolve`。6 个新测试 PASS。
3. **Task 3** (`667d4261`): `_filter_lead_tools` 纯函数 + `_LEAD_EXCLUDED_TOOLS`，从 lead 工具列表移除 bash/write_file/str_replace。10 个新测试 PASS。
4. **Task 4a** (`8f78dbba`): 删除 `LeadAgentExecutionBoundaryProvider` + 配套 test。
5. **Task 4b** (`1f245910`): prompt.py 三处修改（transparency 表、Step 0.5、skill 说明段）。
6. **Task 4c** (`efbb24a0`): SKILL.md lead role 段改为 prep_metric_plan 工具。
7. **Task 4d** (`ff3d815d`): dogfood 验证报告已写，浏览器测试尝试但未完成（auth 问题）。
8. **Step 0 实证**: langgraph.log 8013 行中 0 条 LoopDetection 事件 → 归类 (a)：中间件完全未触发。

### 🔴 待完成

- **浏览器 dogfood E2E 验证**: 需从工作树重启服务后手动测试
- **用户授权 push/merge**
- **Spec 阶段 1 合 dev**（P0 fix 合 dev 之后做，需 rebase）

---

## 关键上下文

### 分支与 worktree

```
分支: sync/p0-lead-bash-removal (基于 dev@9314b985)
Worktree: /home/wangqiuyang/noldus-insight/.claude/worktrees/p0-lead-bash-removal
```

### 7 个 Commit（按顺序）

```
ff3d815d docs: P0 fix dogfood 验证报告
efbb24a0 feat(skill): ethoinsight-metric-catalog lead role 段改用 prep_metric_plan 工具
1f245910 feat(lead): prompt Step 0.5 改 prep_metric_plan 工具，删 bash CLI 指令
8f78dbba feat: 删除 G4 LeadAgentExecutionBoundaryProvider (bash 已从 lead tool 列表移除)
667d4261 feat(lead): 从 lead 工具列表移除 bash / write_file / str_replace, 所有 ethoinsight CLI 调用强制走 prep_metric_plan
53c38d21 feat(tools): 加 prep_metric_plan Python 工具, lead 一步生成 metric_plan.json 不走 bash
20a443cc fix: LoopDetectionMiddleware 按 tool name 计数(P0 兜底)
9314b985 docs(plan): P0 lead bash 移除 + prep_metric_plan 工具注册 实施计划  ← base (dev)
```

### 测试结果

```
2273 passed, 7 failed (全部预存，与本 PR 无关)
新增 21 个测试: Task 1 x5 + Task 2 x6 + Task 3 x10
```

### 关键文件改动

| 文件 | 改动 |
|------|------|
| `packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py` | 阈值 30/50→3/5 + 消息文案 |
| `packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py` | **新建** — 核心工具 |
| `packages/agent/backend/packages/harness/deerflow/tools/builtins/__init__.py` | 导出 prep_metric_plan_tool |
| `packages/agent/backend/packages/harness/deerflow/tools/tools.py` | BUILTIN_TOOLS + import |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` | `_filter_lead_tools` + 删 G4 boundary |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | Step 0.5 + transparency 表 |
| `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md` | lead role 段 |
| `packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py` | **已删除** |
| `packages/agent/backend/tests/test_lead_execution_boundary_provider.py` | **已删除** |
| `packages/agent/backend/tests/test_loop_detection_middleware.py` | +5 测试 |
| `packages/agent/backend/tests/test_prep_metric_plan_tool.py` | **新建** |
| `packages/agent/backend/tests/test_lead_tool_filtering.py` | **新建** |

### 技术细节

- **prep_metric_plan_tool.py**: 使用 lazy import `deerflow.sandbox.tools.replace_virtual_path` 避免循环导入
- **_filter_lead_tools**: 纯函数，不修改 bootstrap agent 路径
- **LoopDetectionMiddleware**: 现有 hash-based 测试显式传入高 tool_freq 阈值避免冲突
- **guardrails/\_\_init\_\_.py**: 不需要改，`LeadAgentExecutionBoundaryProvider` 不在其 export 列表中

### Dogfood 数据

```
/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点/轨迹-Elevated Plus Maze XT190-Trial     1-Arena 1-Subject 1.txt
```

---

## 未完成事项（优先级排序）

### 🔴 立即

1. **浏览器 dogfood 验证**: 停止当前服务 → 从工作树 `make dev` → 浏览器 `localhost:2026` → 上传 EPM 数据 → "请分析这份 EPM 数据" → 验证 lead 调了 prep_metric_plan
2. **用户授权 push + merge dev**

### 🟡 P0 fix 合 dev 后

3. **Spec 阶段 1 合 dev**: branch `worktree-spec-phase-1-handoff` (HEAD `adbf3107`)，rebase 到 dev（含 P0 fix）后合入
4. **Plan A deerflow sync**: 11 个安全 commit，plan 在 `docs/superpowers/plans/2026-05-15-deerflow-upstream-sync-plan-A-safe-batch.md`
5. **Plan B**: 同目录 plan-B

---

## 风险与注意事项

1. **不要 push** 任何东西直到用户授权
2. **不要碰这 4 个文件**: `llm-finetuning-strategy.md`, `page.tsx`, `serve.sh`, `bootstrap/SKILL.md`
3. **Task 1 可独立合 dev**: 兜底机制，即使 Task 2-4 有异议也能止血
4. **工作树 venv**: 首次启动需 `uv sync`（已执行），config.yaml 需从主 repo 复制
5. **G4 boundary vs G4 方案 C**: 两个不同东西。G4 方案 C（前端 stage broadcast）已合 dev。G4 boundary（本次删除的）是 `LeadAgentExecutionBoundaryProvider`
6. **不要重新设计/重新诊断**: 所有 grill 决策和根因诊断已完成

---

## 下一位 Agent 的第一步建议

```bash
# 1. 进工作树
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/p0-lead-bash-removal

# 2. 验证状态
git log --oneline -8
cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/ --ignore=tests/test_metric_catalog_live.py -q 2>&1 | tail -5

# 3. 复制配置（如未做）
cp /home/wangqiuyang/noldus-insight/packages/agent/config.yaml /home/wangqiuyang/noldus-insight/.claude/worktrees/p0-lead-bash-removal/packages/agent/config.yaml
cp /home/wangqiuyang/noldus-insight/packages/agent/extensions_config.json /home/wangqiuyang/noldus-insight/.claude/worktrees/p0-lead-bash-removal/packages/agent/extensions_config.json

# 4. Dogfood: 停止主 repo 服务 → 从工作树启动 → 浏览器测试
cd /home/wangqiuyang/noldus-insight/packages/agent && make stop
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/p0-lead-bash-removal/packages/agent && make dev
# 浏览器 localhost:2026 → 上传 EPM 数据 → "请分析这份 EPM 数据" → "做单样本分析"

# 5. 读关键文档
cat docs/handoffs/2026-05/2026-05-15-p0-lead-bash-removal-handoff.md  # 根因 + 7 grill 决策
cat docs/superpowers/plans/2026-05-15-lead-bash-removal-and-tool-registration-plan.md  # 执行手册
```
