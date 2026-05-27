# 2026-05-26 会话交接文档 — chart 列过滤 + 子任务态机 + shoaling 下线

## 当前任务目标

修复 2026-05-26 FST 端到端 dogfood 揪出的两个 bug（PR #46），并将 shoaling 范式从"假支持"下线为"明示暂不支持"（PR #48）。

**三个 PR 全部合入 dev (HEAD `77875eee`)**，无未完成的正在进行工作。

---

## 当前进展

### ✅ PR #46: chart 列依赖过滤 + 子任务态机识别 gate_enforcement

**已合入 dev `2acae6da`**

- `ChartEntry` 加 `requires_columns: list[str]` 字段 ([schema.py:45](packages/ethoinsight/ethoinsight/catalog/schema.py#L45))
- `resolve.py` charts 路径接通 `_missing_columns` 过滤 ([resolve.py:334-340](packages/ethoinsight/ethoinsight/catalog/resolve.py#L334))
- 6 个 catalog yaml（epm/oft/ldb/fst/tst/zero_maze/_common）基于 newdemodata 真实列名审计补 `requires_columns`
- `message-list.tsx:207-211` 从 `in_progress` 兜底改为"任何 tool_call_id 匹配的 ToolMessage 抵达即视为终态"（识别 gate_enforcement 短路的非 Task* 前缀）
- **测试**: ethoinsight 384 passed / 51 skipped; backend 3016 passed

### ✅ PR #48: shoaling 范式下线 + v0.1 范围公告

**已合入 dev `77875eee`**

- 物理删除：catalog/shoaling.yaml + metrics/shoaling.py + scripts/shoaling/ + golden-cases/case-001-shoaling-baseline/
- 引用清理：metrics/__init__/dispatcher/_common + catalog/loader + templates/__init__ + ev19_facts/utils/assess
- `ev19_facts.py` 新增 `SUPPORTED_PARADIGMS_V01 = frozenset({'epm','open_field','zero_maze','light_dark_box','forced_swim'})`
- `identify_ev19_template_tool.py` Step 2.5：paradigm ∉ SUPPORTED_PARADIGMS_V01 → 直接 `status="unsupported"` 不走流水线
- `lead_agent/prompt.py` 加「当前支持的范式范围 (v0.1)」段，反问选项改成 5 个范式
- 测试：ethoinsight 374 passed / 64 skipped（14 个新 shoaling skip）; backend 3016 passed

---

## 关键上下文

### 仓库状态

- **主开发分支**: `dev` (HEAD `77875eee`)
- **主生产分支**: `main`（从 dev PR 到 main 触发 ACR build→ ECS 部署）
- **仓库根**: `/home/wangqiuyang/noldus-insight/`

### 当前支持的 5 个范式

| 范式 | Paradigm Key | Catalog |
|---|---|---|
| 高架十字迷宫 | epm | epm.yaml |
| 旷场 | open_field | oft.yaml |
| 明暗箱 | light_dark_box | ldb.yaml |
| 强迫游泳（大鼠） | forced_swim | fst.yaml |
| 零迷宫/O迷宫 | zero_maze | zero_maze.yaml |

所有其他范式（鱼类/TST/MWM/Barnes/Y-T maze/novel_object/sociability等）识别后 `status="unsupported"`。

### 真实数据位置

- **newdemodata（5 范式 demo 数据）**: `/home/wangqiuyang/DemoData/newdemodata/`
  - 高架十字迷宫_小鼠_三点/ → EPM
  - 旷场_小鼠_三点/ → OFT
  - 明暗箱/ → LDB
  - 强迫游泳_大鼠/ (Mobility_1 + Mobility_10 两个采样率) → FST
  - O迷宫/ → Zero Maze

### stash 中的 WIP（待下次处理）

- `stash@{0}: WIP: ev19_facts path fallback` — 让 ev19_facts.py 在 Docker 容器内也能解析路径；对应改动：`ev19_facts.py` + `_facts.json`。**小修，独立 PR**。

### 已知独立待解决问题

- **OFT/LDB/zero_maze zone 识别歧义**: 真实 OFT 数据 `in_zone`（bare 单列），metric 函数 `_find_center_zone_column` 找不到含 "center" 的列名 → 静默返回 None。同事 5-13 feedback Q2 方向：反问用户而非猜测，需要 `AmbiguousZoneError` 机制。

---

## 关键发现（本次会话学到）

### SSOT: catalog requires_columns 以真实 raw data 列为准

用户明确要求：「计算指标的代码和生图的代码，不应该凭空想象，应该根据我们现实存在的 raw data 而建立。」

- FST 真实列：`mobility_state_highly_mobile/mobile/immobile`，**无 velocity**
- OFT 真实列：`in_zone`（bare 单列，非 `in_zone_center_*`）
- EPM 真实列：`in_zone_open_arms_center`、`in_zone_closed_arms_center`
- LDB/Zero Maze 真实列：`in_zone`（bare 单列）
- EPM/OFT/Zero Maze/LDB 都有 `x_center`、`y_center`；FST/LDB **无** velocity、distance_moved

### Gate Enforcement 短路的 ToolMessage 让前端卡死

当 `GateEnforcementMiddleware` 检测到 critical warning 时短路返回 ToolMessage（content 前缀 `"数据质量检查发现以下 critical 问题..."`，name=`gate_enforcement`），frontend message-list.tsx 旧逻辑将其 fallback 到 `in_progress`，导致子任务卡片永远不结束。

### ev19_facts.SUPPORTED_PARADIGMS_V01 是唯一真源

`identify_ev19_template_tool.py` Step 2.5 判断 `paradigm_key ∉ SUPPORTED_PARADIGMS_V01` → `status="unsupported"`，不走 zone 解析或候选搜索。此设计清楚分离了"能识别"与"能执行"。

---

## 未完成事项（按优先级）

### 🔴 无阻塞性问题，但关键待办

1. **ev19_facts Docker 路径修复** (stash@{0})
   - 改动：`ev19_facts.py` 加双路径 fallback（包内 `_facts.json` 优先，repo 根次之）
   - `_facts.json` 在 `packages/ethoinsight/ethoinsight/` 下已生成
   - **操作**：`git stash pop stash@{0}` → 测试 → 单独 PR

### 🟡 中优先级

2. **OFT/LDB/zero_maze zone 歧义反问机制**
   - 文件：`packages/ethoinsight/ethoinsight/metrics/oft.py::_find_center_zone_column`
   - 方向：bare `in_zone` 应触发 `AmbiguousZoneError` → agent 层接 ask_clarification
   - 独立 PR

3. **E2E 验证全 5 范式**
   - 用 `/home/wangqiuyang/DemoData/newdemodata/` 每个子目录数据跑一遍端到端
   - 确认 activity_intensity 不再误出图（FST 无 velocity 被正确 skip）
   - 确认 shoaling 关键词触发 agent 立即反馈「暂不支持」

4. **TST (tail_suspension) 下线还是保留？**
   - 用户确认 5 个支持范式不含 TST，但代码层（catalog/tst.yaml, metrics/tst.py, scripts/tst/）仍存在
   - 待用户确认是否按 shoaling 相同方式下线

### 🟢 低优先级

5. identify_ev19_template_tool 3 个 pre-existing ruff 错误（F401/E501/F841）清理
6. worktree 清理：`.claude/worktrees/retire-shoaling/` 可 `git worktree remove`
7. SubtaskCard 当 status=completed 但被 gate 拦截时，label 文案与 ✓ icon 冲突的 UX 改进

---

## 建议接手路径

### 接手步骤

```bash
cd /home/wangqiuyang/noldus-insight
git fetch origin && git checkout dev && git pull --ff-only
# 确认 HEAD = 77875eee
git log --oneline -3

# 恢复 WIP（可选，独立小 PR）
git stash pop stash@{0}
# 查看改动
git diff packages/ethoinsight/ethoinsight/ev19_facts.py
```

### 主要工作入口

- **catalog 文档**: `packages/ethoinsight/ethoinsight/catalog/` — 每个范式 YAML
- **前端子任务态机**: `packages/agent/frontend/src/components/workspace/messages/message-list.tsx`
- **范式识别工具**: `packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py`
- **支持范式白名单**: `packages/ethoinsight/ethoinsight/ev19_facts.py::SUPPORTED_PARADIGMS_V01`

### 测试命令

```bash
# ethoinsight
cd packages/ethoinsight
/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv/bin/python -m pytest tests/ -q

# backend
cd packages/agent/backend
PYTHONPATH=$PWD/packages/harness:$PWD \
DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml \
/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv/bin/python -m pytest tests/ --tb=line -q
```

---

## 风险与注意事项

### ⚠️ worktree pytest 盲区

worktree 内**没有 .venv**，始终用主仓库 venv + PYTHONPATH，见上方命令。

### ⚠️ OFT/LDB zone 识别：catalog 写 `in_zone*`，metric 可能返回 None

- catalog 的 `requires_columns: ["in_zone*"]` 让 chart 进 plan（真实数据存在 `in_zone` 列）
- 但 `oft.py::_find_center_zone_column` 拒绝 bare `in_zone`（没有 "center" 字样），返回 None
- 表现：code-executor 跑完但 center_time_ratio 等指标为 None，数据分析误导性
- **根治方向**：抛 AmbiguousZoneError，让 agent 反问用户「您的数据中 in_zone 列代表哪个区域？」

### ⚠️ TST 代码仍存在

- TST (tail_suspension) catalog/metrics/scripts 均未删
- `SUPPORTED_PARADIGMS_V01` 不含 TST，identify tool 会返回 unsupported
- 但代码死代码残留（同 shoaling 之前的状态）
- 待用户确认是否下线

### ⚠️ shoaling identify tool 仍可"识别" shoaling

- `_PARADIGM_CN_HINTS` / `_FILENAME_PARADIGM_PATTERNS` 保留了鱼类关键词，识别后 Step 2.5 即 unsupported
- 这是预期行为，鱼类数据在识别阶段就被拦截

---

## 关键文件清单

| 文件 | 说明 |
|---|---|
| [packages/ethoinsight/ethoinsight/ev19_facts.py](packages/ethoinsight/ethoinsight/ev19_facts.py) | `SUPPORTED_PARADIGMS_V01` 白名单 |
| [packages/ethoinsight/ethoinsight/catalog/](packages/ethoinsight/ethoinsight/catalog/) | 5 个范式 YAML + schema + loader + resolve |
| [packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py](packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py) | Step 2.5 unsupported 分支 |
| [packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) | v0.1 范围公告段 (L252-268) |
| [packages/agent/frontend/src/components/workspace/messages/message-list.tsx](packages/agent/frontend/src/components/workspace/messages/message-list.tsx) | 子任务态机（gate_enforcement 识别） |
| [docs/handoffs/2026-05/2026-05-26-chart-cols-filter-and-subtask-state-handoff.md](docs/handoffs/2026-05/2026-05-26-chart-cols-filter-and-subtask-state-handoff.md) | PR #46 详细交接 |
| [docs/handoffs/2026-05/2026-05-26-shoaling-retirement-handoff.md](docs/handoffs/2026-05/2026-05-26-shoaling-retirement-handoff.md) | PR #48 详细交接 |
