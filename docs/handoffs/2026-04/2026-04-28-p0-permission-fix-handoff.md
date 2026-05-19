# Handoff: P0 permission 修复 + Gate 1 / data quality 提示词改造分析

**日期**: 2026-04-28
**状态**: P0 已修 + 测试通过 + 提示词改造方案已讨论，待实施
**上一会话**: [2026-04-28-event-loop-fix-handoff.md](2026-04-28-event-loop-fix-handoff.md)

---

## 1. 本会话产出

1. ✅ 修复 `set_experiment_paradigm` tool permission error
2. ✅ 修复 `/mnt/shared/` 写入失败
3. ✅ 新增 8 个测试（路径验证 + 安全边界）
4. ✅ 全面梳理所有 prompt 和 skill（5 层提示词体系）
5. ✅ 分析 Gate 1 两级范式跳过 + data quality warning 未走的根因
6. ✅ 与用户对齐改造方案（条件分支 Gate 1 + 硬中断 data quality gate）

---

## 2. 关键改动（已 commit: `54453634`）

### 2.1 `set_experiment_paradigm` tool — 通过 ToolRuntime 解析 host 路径

**文件**: [experiment_context.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py)

**根因**: 工具是 langchain `@tool`，在 lead agent host 进程中运行。但 `workspace_dir` 默认值 `/mnt/user-data/workspace/` 是 sandbox 虚拟路径，`pathlib.Path` 直接写会落到 host 上不存在的字面路径。

**修复**:
- 新增 `ToolRuntime[ContextT, ThreadState]` 参数（langgraph 自动注入，不影响 tool schema）
- 从 `runtime.state["thread_data"]["workspace_path"]` 获取 host 实际 workspace 路径
- fallback：`runtime` 为 None 时仍用 `workspace_dir` 默认值（兼容测试/非 langgraph 环境）

### 2.2 `/mnt/shared/` 写入 — 两层安全校验补允

**文件**: [sandbox/tools.py](packages/agent/backend/packages/harness/deerflow/sandbox/tools.py)

**根因**:
- `validate_local_tool_path`（安全门）：只允许 `/mnt/user-data/`、`/mnt/skills/`、`/mnt/acp-workspace/`、自定义挂载路径，**没有 `/mnt/shared/`**
- `_validate_resolved_user_data_path`（路径范围校验）：allowed roots 只有 workspace/uploads/outputs，**缺少 `shared_path`**

**修复**（两处各加 ~3 行）:
1. `validate_local_tool_path` 加 `/mnt/shared/` 前缀检查（读写均允许）
2. `_validate_resolved_user_data_path` 的 allowed roots 加 `thread_data.get("shared_path")`

### 2.3 测试

**文件**: [test_sandbox_tools_security.py](packages/agent/backend/tests/test_sandbox_tools_security.py)（末尾新增 8 个测试）

覆盖 `/mnt/shared/` 的四层校验：
- `validate_local_tool_path`：允许写、允许读、允许 bare prefix、阻止路径穿越
- `replace_virtual_path`：映射 `/mnt/shared/file` 和 bare `/mnt/shared`
- `_resolve_and_validate_user_data_path`：正确解析、阻止穿越

**测试基线**: 1716 PASS, 2 FAIL (pre-existing), 14 skipped

---

## 3. Prompt 体系梳理（5 层）

| 层级 | 文件 | 注入方式 | 行数 |
|------|------|----------|------|
| Lead Agent 主 prompt | [prompt.py](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) | `SYSTEM_PROMPT_TEMPLATE` 直接注入 | 1215 |
| Lead Agent 调度指南 | 同上，`orchestration_guide` 动态拼接 | 直接注入（~150行） | — |
| ethoinsight-planning skill | [quality-gates.md](packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md) | 按需 `read_file`（披露式） | 90 |
| code-executor prompt | [code_executor.py](packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py) | subagent system prompt | 93 |
| data-analyst / report-writer / knowledge-assistant | 各自 `*_CONFIG.system_prompt` | subagent system prompt | 230 / 223 / 72 |

Skill 加载方式：
- **披露式**（lead agent 按需 `read_file`）：`ethoinsight-planning`、`ethoinsight`、`ethoinsight-charts`、`compaction-recovery`
- **注入式**：`ethoinsight-analysis` → code-executor 直接带（合理，code-executor 永远需要）

---

## 4. Gate 1 + Data Quality Warning — 已达成共识的改造方案

### Gate 1：从无条件两级改为条件分支

**当前问题**: prompt 写的是无条件"先问大类 → 再问细分"，但 E2E 中 agent 看到"鱼群行为"直接推断 shoaling 就跳过了。

**用户确认的方案**: 按三级分支：

| 用户已提供的信息 | 行为 |
|-----------------|------|
| 大类 + 细分范式都明确（如"斑马鱼鱼群行为"） | 直接 `set_experiment_paradigm`，不反问 |
| 只有一级明确（如"焦虑相关实验"） | 只问缺少的那一级 |
| 两级都不明确（如"帮我分析"） | 完整两级 UI 流程 |

当前 prompt 已经有完整的 7 大类→细分映射表（line 261-271），agent 有信息做匹配。需要修改 Gate 1 段的描述逻辑。

### Data Quality Warning：从"建议"改为"硬中断"

**当前问题**: prompt 说"如果有 warnings → ask_clarification"，但 E2E 中 agent 看到 critical（n=2<3）仍选择不阻塞。在 agent 看来这是建议，不是硬规则。

**用户确认的方案**:
- 强化 language：severity="critical" 的条目**必须先获得用户确认**才能继续
- 把 data quality gate 从 lead agent prompt 的 Step 1.5 中抽出，只留在 quality-gates.md 作为权威来源（减少重复、增强"来自外部知识"的感知）
- Gate 1 的范式选择逻辑保留在 lead agent prompt 内联（它是路由判断的一部分）

---

## 5. 未完成事项（按优先级）

### P0 — 实施已讨论的 prompt 改造

1. **Gate 1 加条件分支逻辑**
   - 文件: [lead_agent/prompt.py:279-311](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L279-L311)（Gate 1 节） + [quality-gates.md:5-16](packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md#L5-L16)（Gate 0 节）
   - 正面指令："用户在消息中**同时**提到了大类名和具体范式名（如'斑马鱼鱼群行为'），则直接调用 `set_experiment_paradigm`，不反问"
   - 正面指令："用户只提到大类但未指定细分范式（如只说'焦虑迷宫'没说具体是 EPM 还是零迷宫），则只问细分那一级"
   - 考虑：提前在 thinking_style 中加一句规则，让 agent 在判断前先"检查用户是否已提供完整信息"

2. **Data quality gate 强化为硬中断**
   - 文件: [lead_agent/prompt.py:1125-1136](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L1125-L1136)（Step 1.5） + [quality-gates.md:44-58](packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md#L44-L58)（Gate 2）
   - 改为："`data_quality_warnings` 中有 severity='critical' 的条目时，**必须先 ask_clarification 获得用户确认才能继续**。这一步不可跳过。（GLM-5.1 注意：这不是建议，是硬规则。）"

3. **减少重复**（可选优化）
   - quality-gates.md 的 Gate 2 和 prompt.py Step 1.5 描述重复
   - 建议：Step 1.5 只写 "检查 data_quality_warnings，具体规则见 ethoinsight-planning skill 的 quality-gates.md 中的 Gate 2"，不重复列规则

### P1 — 提交上游 issue（用户自己提）

4. **deerflow upstream issue 草稿就绪**
   - `docs/handoffs/2026-04-28-deerflow-upstream-issue-draft.md`
   - 目标: https://github.com/bytedance/deer-flow/issues/new/choose

---

## 6. 上游 issue 反馈

event loop issue 下有人回复了三个建议方案（contextvars、weakref、per-call creation），确认我们的根因分析正确。

**决策**: 不改。当前 `_evict_provider_async_client_caches()` 方案本质上等价于 per-call creation（option 3），且更优——lead agent run 内的多次 LLM 调用仍共享同一 client。等上游在 langchain_anthropic 层面修了，删掉 workaround 即可。

---

## 7. 文件位置速查

| 内容 | 路径 |
|------|------|
| P0-1 修复 | [experiment_context.py:57-105](packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py#L57-L105) |
| P0-2 修复 (validate) | [sandbox/tools.py:607-613](packages/agent/backend/packages/harness/deerflow/sandbox/tools.py#L607-L613) |
| P0-2 修复 (allowed roots) | [sandbox/tools.py:626-635](packages/agent/backend/packages/harness/deerflow/sandbox/tools.py#L626-L635) |
| 新增测试 | [test_sandbox_tools_security.py](packages/agent/backend/tests/test_sandbox_tools_security.py)（末尾 8 个函数） |
| Gate 1 prompt | [lead_agent/prompt.py:259-311](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L259-L311) |
| Data quality gate (prompt) | [lead_agent/prompt.py:1125-1136](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L1125-L1136) |
| Data quality gate (skill) | [quality-gates.md:44-58](packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md#L44-L58) |
| 上游 issue 草稿 | [2026-04-28-deerflow-upstream-issue-draft.md](docs/handoffs/2026-04-28-deerflow-upstream-issue-draft.md) |

---

## 8. 建议接手路径

```
第一步：确认当前状态
  cd /home/wangqiuyang/noldus-insight
  git log --oneline -3
  # 最新 commit: 54453634 "修复了permission denied"
  # 确认之前 event-loop 修复的 commit 也在

第二步：跑测试确认基线
  cd packages/agent/backend && source .venv/bin/activate
  PYTHONPATH=. python -m pytest tests/ --tb=short 2>&1 | tail -5
  # 预期: 1716 PASS, 2 FAIL (pre-existing), 14 skipped

第三步：实施 Gate 1 条件分支
  # 核心改动点：
  vim packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
  # 找到 line 279 "Gate 1 — 两级实验类型确认"
  # 在描述"分两步确认"之前插入条件判断逻辑：
  #   "用户在消息中同时提供了大类名和具体范式名 → 直接 set_experiment_paradigm"
  #   "用户只提供了大类但未指定细分 → 只 ask_clarification 细分选择"
  #   "都没提供 → 完整两级流程"
  # 同时更新 quality-gates.md Gate 0 节对齐

第四步：强化 data quality 硬中断
  # prompt.py line 1125-1136 (Step 1.5)
  # quality-gates.md line 44-58 (Gate 2)
  # 改 severity="critical" → 必须 ask_clarification，不可跳过
  # 考虑：Step 1.5 只写引用 quality-gates.md，减少重复

第五步：跑 E2E 验证（可选，改动较大时建议）
  # 手动测试一个包含 critical warning 的场景
  # 验证 agent 是否真的停下来问用户
```

---

## 9. 决策记录

- ✅ **不改 upstream issue 反馈的三个方案**: 当前方案等价于 option 3（per-call creation），且更优
- ✅ **Gate 1 从无条件两级改为条件分支**: 用户明确要求，两级都明确时直接跳过 UI
- ✅ **Data quality gate 强化**: 从建议改为硬中断，减少重复描述
- ✅ **Skill 加载方式维持现状**: 除 `ethoinsight-analysis`（注入 code-executor，合理）外均为披露式

---

## 10. 风险与注意事项

| 风险 | 应对 |
|------|------|
| **GLM-5.1 不严格遵循 prompt** | 改 prompt 时必须用正面指令（"必须 X"），不用否定句（"禁止不 X"）。考虑在 thinking_style 段加预检查指令 |
| **Gate 1 条件分支可能过于复杂** | 分支逻辑要简洁——3 个分支，每分支 1 句，不要写 if-else 伪代码 |
| **Data quality hard stop 可能影响体验** | 只对 critical 级别硬中断，warning/info 级别可以继续 |
| **quality-gates.md 和 prompt.py 重复** | 后续修改 gate 逻辑时记住只改 quality-gates.md，prompt.py 写引用 |
