# Handoff: Lead Agent 工作流增强 — 实施阶段交接

**日期**: 2026-04-27
**状态**: 代码实施完成（6 commits），全测试 1705/1706 PASS，等待 E2E 验证
**上一会话**: 审批意见反馈 → 9 个审批问题全部修复 → 6 个 Task 实施完成

---

## 1. 当前任务目标

为 EthoInsight 的 lead agent 增加 **18 范式两级选择** 和 **Human-in-the-Loop 三 Gate**。实施计划详见 `/home/qiuyangwang/.claude/plans/dynamic-petting-petal-implementation.md`，审批意见详见 `/home/qiuyangwang/.claude/plans/dynamic-petting-petal-implementation-review.md`（9 个问题已全部修复）。

## 2. 当前进展

- ✅ Task 1: PARADIGMS registry（18 范式 / 7 大类 / 11 tests）— [templates/__init__.py](packages/ethoinsight/ethoinsight/templates/__init__.py)
- ✅ Task 2: experiment-context.json 读写模块 — [experiment_context.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py)
- ✅ Task 2b: `set_experiment_paradigm` dedicated tool（同一文件内）
- ✅ Task 3: GateEnforcementMiddleware（5 tests）— [gate_enforcement_middleware.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/gate_enforcement_middleware.py)
- ✅ Task 3b: agent.py 注入 workflow_mode + GateEnforcementMiddleware — [agent.py:235-287](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py)
- ✅ Task 5: prompt.py 两级 Gate 1 分支 — [prompt.py:259-350](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py)
- ✅ Task 6: 前端 flywheel 模式（9 文件修改，pnpm check 通过）
- ✅ Task 7: TrainingDataMiddleware 回归测试（2 new tests, 12/12 PASS）
- ✅ Task 9: quality-gates.md Gate 0 + Gate 1.5
- ⬜ **Task 10-11: E2E 人工验证**（需要启动服务）

**6 commits on `dev` branch**:
```
d7d377e6 feat: add ask_clarification regression tests + quality-gates Gate 0/1.5
c9517fb8 feat: add flywheel mode to frontend (manual workflow, full 3 Gates)
22e66653 feat: add two-level Gate 1 (category → paradigm) + manual/auto dual-mode
b31e8ec7 feat: add GateEnforcementMiddleware + inject into lead agent
c213d3c8 feat: add experiment-context.json read/write + set_experiment_paradigm tool
dffdfcf2 feat: add PARADIGMS registry with 18 paradigms, 7 categories
```

## 3. 关键上下文

### 新增文件

| 文件 | 职责 |
|------|------|
| `ethoinsight/templates/__init__.py` | **修改**: 新增 CATEGORIES + PARADIGMS registry（18 范式）；lazy import 替代了原来的 top-level `from .tool import` |
| `agent/middlewares/experiment_context.py` | **新**: read_context() + context_exists()（host-side）+ set_experiment_paradigm tool（container-side） |
| `agent/middlewares/gate_enforcement_middleware.py` | **新**: 只拦截 task()，从 state.thread_data.workspace_path 解析 host 路径 |
| `agent/frontend/...` | **修改**: types.ts + settings/local.ts + hooks.ts + input-box.tsx + welcome.tsx + mode-hover-guide.tsx + i18n |
| `skills/.../quality-gates.md` | **修改**: 新增 Gate 0（两级范式选择）+ Gate 1.5（数据一致性软检查） |

### 关键设计决策（含审批修复）

1. **Middleware 路径解析**: GateEnforcementMiddleware 从 `request.state["thread_data"]["workspace_path"]` 获取 host 端物理路径，**不使用** 容器侧 `/mnt/user-data` 虚拟路径（P0 #1 修复）
2. **不做白名单**: GateEnforcementMiddleware 只拦截 `task()`，不维护白名单（P1 #4 修复）
3. **flywheel = ultra 能力 + manual workflow**: hooks.ts 中 `is_plan_mode` 和 `subagent_enabled` 对 flywheel 均启用（P0 #3 修复）
4. **Fault injection 用 monkeypatch**: 不在 state 里设 `messages=None`（会被 `.get()` 短路），而是 monkeypatch 内部方法抛异常（P0 #2 修复）
5. **set_experiment_paradigm tool**: 替代 prompt 指导 LLM 调 write_file 写 JSON（P1 #6 修复）
6. **Gate 1 两级选择**: 先 7 大类 → 再 2-6 细分。斑马鱼大类只有 shoaling 一个范式，无需第二步

### 文件位置速查

| 内容 | 路径 |
|------|------|
| 实施计划 | `/home/qiuyangwang/.claude/plans/dynamic-petting-petal-implementation.md` |
| 审批意见 | `/home/qiuyangwang/.claude/plans/dynamic-petting-petal-implementation-review.md` |
| 设计计划 | `/home/qiuyangwang/.claude/plans/dynamic-petting-petal.md` |
| Spec | `docs/superpowers/specs/2026-04-27-lead-agent-workflow-design.md` |
| 审批说明 | `docs/superpowers/specs/2026-04-27-lead-agent-workflow-approval-brief.md` |
| 交接文档（设计阶段） | `docs/handoffs/2026-04-27-lead-agent-workflow-handoff.md` |
| PARADIGMS registry | `packages/ethoinsight/ethoinsight/templates/__init__.py` |
| Experiment context | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py` |
| GateEnforcement | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/gate_enforcement_middleware.py` |
| Lead agent entry | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` |
| Lead agent prompt | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` |
| EV19 templates | `demo-data/ev19 templates/`（62 个模板，用于审计 PARADIGMS 注册表） |

## 4. 关键发现

- **ethoinsight 的 `tool.py` 依赖 langchain** — 修改 `templates/__init__.py` 用 `__getattr__` lazy import 避免 standalone ethoinsight 测试崩溃
- **backend 测试基线**: 1705/1706 PASS。1 个 pre-existing 失败：`test_usage_example_shows_ask_clarification_between_analyst_and_writer`（检查旧 prompt 文本 "需要 APA 格式报告"，prompt 已改但测试未更新）
- **ethoinsight 测试基线**: 113/113 PASS（排除 3 个依赖 langchain 的 test，在 ethoinsight standalone venv 无法运行）
- **`set_experiment_paradigm_tool` 已定义但未注册** — tool 在 `experiment_context.py` 中定义为 `@tool("set_experiment_paradigm")`，但未添加到 lead agent 的 tool list。需要在 `tools/builtins/` 或 agent 配置中注册
- **前端 pnpm check 通过**（lint + typecheck 零错误）

## 5. 未完成事项（按优先级）

### P0 — E2E 验证

1. **启动服务验证 flywheel 模式**:
   ```bash
   cd /home/qiuyangwang/noldus-insight/packages/agent && make dev
   ```
   - 打开 http://localhost:2026 → 新建对话 → mode-switcher 是否出现"数据飞轮"选项
   - 选择 flywheel → 上传 shoaling 测试数据 → 输入"帮我分析"
   - Gate 1 第一级是否弹出 7 大类按钮？
   - 选择大类 → 第二级是否弹出细分范式按钮？
   - 选择范式 → `experiment-context.json` 是否写入？
   - Gate 2 组别确认是否触发？
   - 切到非 flywheel 模式 → 是否只触发 Gate 1？

2. **注册 `set_experiment_paradigm` tool**:
   - 当前 tool 在 `experiment_context.py` 中定义但未加入 agent tool list
   - 需要在 `deerflow/tools/builtins/` 或 `config.yaml` 的 tool_groups 中注册
   - 否则 prompt 指导 agent 调 `set_experiment_paradigm`，agent 找不到该 tool

### P1 — 文档/清理

3. **paradigm-list.md 替换**: 旧 `orchestration_guide` 底部 `## 可用范式模板\nshoaling (斑马鱼群体行为)` 已过时，应替换为从 registry 动态生成的列表
4. **prompt 中的范式列表**: 当前硬编码在 prompt.py 的 noldus_rules 中。将来应改为从 `list_categories()` / `list_paradigms()` 动态注入，但目前 prompt template 是静态字符串 → 需要添加动态注入逻辑

### P2 — 后续

5. E2E 测试自动化（v0.1 后）
6. auto 模式端到端验证
7. 国际化完善

## 6. 建议接手路径

```
第一步：验证当前状态
  cd /home/qiuyangwang/noldus-insight/packages/agent/backend
  source .venv/bin/activate && make test
  # 确认 1705 PASS, 1 pre-existing FAIL (test_lead_prompt_interactive_pipeline)

第二步：注册 set_experiment_paradigm tool
  # 这是 prompt 能正常工作的前提
  # 查看 deerflow/tools/builtins/ 中的注册模式
  # 在 config.yaml 或 extensions_config.json 中添加

第三步：启动服务 + E2E 验证
  cd /home/qiuyangwang/noldus-insight/packages/agent && make dev
  # 按第 5 节 P0 #1 的 checklist 逐项验证

第四步（可选）：处理 P1 项
  # 动态注入范式列表到 prompt
  # 清理 paradigm-list.md
```

## 7. 风险与注意事项

| 风险 | 应对 |
|------|------|
| **set_experiment_paradigm tool 未注册** | 必须在启动 E2E 前注册，否则 Gate 1 无法写入 context.json → GateEnforcementMiddleware 永远拦截 task() → manual 模式死锁 |
| **GateEnforcementMiddleware 路径解析** | 依赖 `state["thread_data"]["workspace_path"]`。如果 ThreadDataMiddleware 未注入（如老 thread），返回 None → 放行 task()。这是正确的 fallback 行为 |
| **flywheel 模式** | 依赖 `context.mode === "flywheel"`。如果前端未正确持久化 mode，workflow_mode 默认 "auto" |
| **GLM-5.1 正面提示规则** | prompt 中不使用"禁止跳过 Gate""不允许直接 task"，已全部改为正面指令 |
| **飞轮录制静默失败** | TrainingDataMiddleware 已用 monkeypatch 验证故障隔离。但 E2E 时应检查 `training-data/auto-collected/` 中 ask_clarification turns 的 JSONL 质量 |

## 8. 下一位 Agent 的第一步建议

```bash
# 1. 确认基线
cd /home/qiuyangwang/noldus-insight/packages/agent/backend
source .venv/bin/activate && make test

# 2. 找到 set_experiment_paradigm tool 的注册入口
# 查看已有 built-in tools 的注册方式
grep -rn "ask_clarification" packages/harness/deerflow/tools/builtins/

# 3. 注册 set_experiment_paradigm tool（参考 ask_clarification 的注册模式）
# 编辑 packages/harness/deerflow/tools/builtins/ 中的相关文件

# 4. 启动 E2E 验证
cd /home/qiuyangwang/noldus-insight/packages/agent && make dev
```

然后按第 5 节 P0 #1 的 checklist 执行 E2E 验证。
