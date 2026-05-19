# Handoff: Gate Enforcement 生产级改造 — Section 4 实施完成

**日期**: 2026-04-28
**状态**: Steps 0-6 全部完成，测试全绿，等待手动 E2E 验证
**基于设计**: [docs/superpowers/specs/2026-04-28-gate-enforcement-design.md](../specs/2026-04-28-gate-enforcement-design.md)

---

## 1. 本会话产出

1. ✅ Step 0: Schema rename `["gate1"]` → `["gate1_paradigm"]`
2. ✅ Step 1: 扩展 `experiment_context.py` 新增 3 个函数（`read_handoff`, `get_critical_warnings`, `is_quality_acknowledged`）
3. ✅ Step 2: 增强 `GateEnforcementMiddleware` — Gate 2 检查、结构化日志、subagent_type 分支
4. ✅ Step 3: 简化 `prompt.py` Gate 1 段 + Step 1.5 段
5. ✅ Step 4: 更新 `quality-gates.md` Gate 0 + Gate 2 段
6. ✅ Step 5: 14 个测试全部通过
7. ✅ Step 6: 全量测试 1725 passed, 0 regression

---

## 2. 关键改动

### 2.1 `experiment_context.py` — 新增 3 个数据质量函数

**文件**: [experiment_context.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py)

- `read_handoff(workspace_dir)` → dict | None — 读 `handoff_code_executor.json`
- `get_critical_warnings(workspace_dir)` → list[dict] — 提取 severity="critical" 条目，防御式访问（.get()），格式不匹配返回空列表（fail open）
- `is_quality_acknowledged(workspace_dir)` → bool — 检查 experiment-context.json 的 `gate_completed` 是否包含 `"gate2_quality_acknowledged"`
- 同时：`set_experiment_paradigm_tool` 中 `["gate1"]` → `["gate1_paradigm"]`

### 2.2 `gate_enforcement_middleware.py` — 双层 Gate 拦截

**文件**: [gate_enforcement_middleware.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/gate_enforcement_middleware.py)

核心变化：
- `_check_gate1(state)` — 原 `_should_block`，检查 experiment-context.json 是否存在
- `_check_gate2(state)` — 新增，检查 handoff 中是否有 critical warnings 且未 acknowledge
- `_build_quality_block_message(warnings, request)` — 新增，data quality 专用 error message，列出所有 critical 条目并提供 (a)(b)(c) 选项模板
- `wrap_tool_call()` / `awrap_tool_call()` — 按 `subagent_type` 分支：
  - `task(data-analyst)` → Gate 2 检查
  - 其余 `task(X)` / 无 subagent_type → Gate 1 检查
  - 非 task 工具 → 直接放行
- 所有 gate check 输出结构化日志：`gate_check | gate=... | thread=... | result=... | detail=...`

### 2.3 `prompt.py` — 退化为执行指导

**文件**: [prompt.py](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py)

- Gate 1 段 (L279-311): 措辞从"你必须先执行这两步检查"改为"你必须先确认实验范式（系统会在 task 调度时强制执行此检查）"，新增条件分支提示段落
- Step 1.5 段 (L1125-1136): data quality 检查改为引用中间件拦截，write_file 共享摘要保留完整指令

### 2.4 `quality-gates.md` — 对齐文档

**文件**: [quality-gates.md](packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md)

- Gate 0: 表格从两行扩展为三行（明确大类+细分 / 只明确大类 / 什么都没提供），新增强制执行说明
- Gate 2: 新增"由 GateEnforcementMiddleware 强制执行"说明

### 2.5 测试

**文件**: [test_gate_enforcement_middleware.py](packages/agent/backend/tests/test_gate_enforcement_middleware.py)

14 个测试，覆盖：
- TestGate1: blocks / allows / missing workspace_path
- TestGate2: blocks (critical+unacknowledged) / allows (no critical) / allows (acknowledged) / missing workspace_path
- TestNonTaskPassthrough: 5 种非 task 工具透传
- TestDisabledMiddleware: enabled=False 全部放行（含 data-analyst）
- TestGateCheckLogging: 4 个结构化日志验证

---

## 3. 架构要点（下一位 Agent 需理解）

1. **双层 enforcement**: middleware 是第一层（早拦截），prompt 退化为指导。未做的 defense-in-depth 第二层（task_tool 函数体内二次校验）在 spec 的 follow-up 里，当前不需要
2. **拦截链**: `GateEnforcementMiddleware(ToolMessage error)` → agent 读 error → 调 `ask_clarification` → `ClarificationMiddleware` 拦截 → `Command(goto=END)`，这是有意的设计（ToolMessage 而非直接 Command，让 agent 有机会解释）
3. **fail open**: 所有检查在 `resolve_workspace_from_state` 返回 None、thread_data 缺失、handoff 格式不匹配时都放行，不阻断老 thread
4. **workflow_mode**: `manual` 激活 middleware，`auto` 跳过。wiring 在 [agent.py:287-291](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py#L287-L291)

---

## 4. 测试结果

```
基线: 2 failed, 1716 passed, 14 skipped
变更后: 2 failed, 1725 passed, 14 skipped
```

2 个 failure 是 pre-existing，与本次改动无关：
- `test_ethoinsight_planning_skill.py::test_planning_skill_is_enabled_in_config`
- `test_lead_prompt_interactive_pipeline.py::TestDefaultPipelineIsTwoStep::test_usage_example_shows_ask_clarification_between_analyst_and_writer`

---

## 5. 下一步

### 高优先级：手动 E2E 验证

```bash
cd packages/agent && make dev
```

测试场景：
1. 创建 thread 时传 `configurable.workflow_mode: "manual"`
2. 上传 shoaling 数据，说"帮我分析" → 应被 Gate 1 拦截
3. 说"分析斑马鱼鱼群行为" → 应引导直接调 set_experiment_paradigm
4. code-executor 完成后如果 handoff 有 critical warning → 调 task(data-analyst) 时应被 Gate 2 拦截
5. 搜后端日志 `gate_check` 验证结构化日志

### 低优先级（spec follow-up，不阻塞当前）

- schema 版本号: experiment-context.json 加 `schema_version` 字段
- defense in depth 第二层: task_tool.py 函数体内补二次校验

---

## 6. 风险

- 老 thread 缺少 experiment-context.json → 已通过 fail-open 缓解
- handoff_code_executor.json 格式变化 → get_critical_warnings() 用 .get() 防御式访问
- 中间件复杂度: 当前约 130 行，spec 门槛 200 行，目前安全
