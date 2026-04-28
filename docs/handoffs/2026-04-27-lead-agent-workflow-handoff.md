# Handoff: Lead Agent 工作流增强 — 实验识别 + Human-in-the-Loop

**日期**: 2026-04-27
**状态**: 设计方案已定稿，已通过 Claude Opus 4.7 审查，等待实施
**上一会话**: brainstorming → 方案设计 → 审查修订

---

## 1. 当前任务目标

为 EthoInsight 的 lead agent 增加两个能力：

1. **实验类型识别**：用户上传 EV19 数据后，lead agent 通过 `ask_clarification` 让用户从 16 种范式模板中选择，而非靠数据推断（raw data 列结构都一样，推断不可靠）
2. **Human-in-the-Loop 机制**：在关键决策点插入确认 gate，飞轮期收集高质量反馈数据。通过 `workflow_mode` 开关支持 manual（飞轮期）和 auto（post-training 后）两种模式

预期产出：~250 行代码 + 11 单测，13 文件修改，零新前端组件/API 路由。

## 2. 当前进展

- ✅ 项目全面探索（CLAUDE.md、DeerFlow 框架、ethoinsight 库、DemoData 18 种范式）
- ✅ 方案选型：方案 C（手动/自动双模式），用户确认
- ✅ 设计定稿：3 个 Gate 全部用 `ask_clarification` + `ClarificationMiddleware` 实现
- ✅ Claude Opus 4.7 审查通过，7 个缺口已纳入修订
- ✅ 修订后计划写入 `/home/qiuyangwang/.claude/plans/dynamic-petting-petal.md`
- ⬜ 写正式 spec 到 `docs/superpowers/specs/`
- ⬜ `writing-plans` skill → 实施

## 3. 关键上下文

### 项目文件位置

| 内容 | 路径 |
|------|------|
| 项目 CLAUDE.md | `/home/qiuyangwang/noldus-insight/CLAUDE.md` |
| DeerFlow 后端 CLAUDE.md | `/home/qiuyangwang/noldus-insight/packages/agent/backend/CLAUDE.md` |
| 修订后计划 | `/home/qiuyangwang/.claude/plans/dynamic-petting-petal.md` |
| Opus 审查意见 | `/home/qiuyangwang/.claude/plans/dynamic-petting-petal-review.md` |
| Lead agent 入口 | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` |
| Lead agent prompt | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` |
| ClarificationMiddleware | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/clarification_middleware.py` |
| ask_clarification tool | `packages/agent/backend/packages/harness/deerflow/tools/builtins/clarification_tool.py` |
| ethoinsight-planning skill | `packages/agent/skills/custom/ethoinsight-planning/SKILL.md` |
| quality-gates.md | `packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md` |
| ethoinsight/templates/ | `packages/ethoinsight/ethoinsight/templates/`（目前只有 shoaling.py） |
| DemoData | `/home/qiuyangwang/ethoinsight-demo/DemoData/`（18 个范式目录） |
| frontend clarification UI | `packages/agent/frontend/src/components/workspace/messages/clarification-options.tsx` |
| frontend mode-switcher | `packages/agent/frontend/src/components/workspace/` 附近 |
| frontend hooks | `packages/agent/frontend/src/core/threads/hooks.ts` (line 546-562) |
| frontend types | `packages/agent/frontend/src/core/threads/types.ts` |
| frontend settings | `packages/agent/frontend/src/core/settings/local.ts` |
| 现有 handoff 文档 | `docs/handoffs/` |

### 核心设计决策

1. **所有 Gate 都用 `ask_clarification`**（不另造 UI）— DeerFlow 已有 tool → 中断 → 用户回复 → 继续执行 闭环
2. **`workflow_mode: "manual" | "auto"`** 照抄 `is_plan_mode` 模式，在 config.configurable 注入
3. **Gate 间状态存在 `experiment-context.json`**（`/mnt/user-data/workspace/`），写文件传状态
4. **GateEnforcementMiddleware** 强制 manual 模式下 Gate 1 完成后才能 task()，防止模型跳过
5. **`ethoinsight/templates/__init__.py` registry** 作单一事实源，替代 paradigm-list.md

### 已解决的架构问题

- **为什么不用自定义前端流程**？ask_clarification 已有完整的 tool call → 中断 → 按钮渲染 → 用户回复 → 继续执行闭环，零新 UI 代码，TrainingDataMiddleware 自动录制
- **为什么需要 enforcement middleware**？prompt 指令有概率漏调，一旦跳过 Gate，飞轮 SFT 标注数据丢失
- **为什么需要 experiment-context.json**？summarization 会丢掉对话历史中的 Gate 选择，且 TrainingDataMiddleware 需要结构化 label

## 4. 关键发现

- DeerFlow 的 `ClarificationMiddleware` 已经是完美的 human-in-the-loop 机制（`Command(goto=END)` 中断）
- 前端 `ClarificationOptions` 组件已支持 options 数组 → 按钮渲染，无需改 UI
- `is_plan_mode` 模式是实现 workflow_mode 的完美模板，直接照抄
- `ethoinsight-planning` skill 已有意图分类（Step 1）和完整性检查（Step 2），直接增强即可
- ethoinsight/templates/ 目前只实现了 shoaling，其他 15 种 status="planned"
- 飞轮录制（TrainingDataMiddleware）是 after_agent observer，失败吞异常 — 需要回归测试确认 ask_clarification turn 的录制正确性

## 5. 未完成事项（按优先级）

### 立即执行（Layer 1-7 顺序）

1. **Layer 1: 单一事实源** — 新建 `ethoinsight/templates/__init__.py` registry（PARADIGMS + list_paradigms() + verify_paradigm_columns()）+ 4 单测
2. **Layer 2: 状态存储** — experiment-context.json 读写逻辑 + 老 thread 兼容（文件不存在 → free-form fallback）
3. **Layer 3: Enforcement** — 新建 `GateEnforcementMiddleware`（拦截未写 context.json 的 task()）+ 4 单测
4. **Layer 4: Prompt** — `prompt.py` 加 manual/auto 分支（manual 三 Gate，auto 只 Gate 1）
5. **Layer 5: 前端** — mode-switcher 加 `flywheel` 选项 + hooks.ts 注入 workflow_mode + types.ts + settings
6. **Layer 6: 飞轮回归** — 3 个 TrainingDataMiddleware 回归测试（可与 Layer 3 并行）
7. **Layer 7: E2E 验证** — DemoData/shoaling 走完整 manual 流

### 后续（Not in Scope）

- 16 种 paradigm 实际模板实现（epm.py, oft.py 等）
- 范式知识（zone 阈值、判读逻辑）— 走 golden-cases
- auto 模式端到端测试 — v0.1 后
- 国际化 — 当前中文为主

## 6. 建议接手路径

```
第一步：读计划
  Read /home/qiuyangwang/.claude/plans/dynamic-petting-petal.md
  Read /home/qiuyangwang/.claude/plans/dynamic-petting-petal-review.md

第二步：读项目文档
  Read /home/qiuyangwang/noldus-insight/CLAUDE.md
  Read /home/qiuyangwang/noldus-insight/packages/agent/backend/CLAUDE.md

第三步：探索关键文件
  Read packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py
  Read packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py (关注 Noldus dispatch rules 部分)
  Read packages/agent/backend/packages/harness/deerflow/agents/middlewares/clarification_middleware.py
  Read packages/ethoinsight/ethoinsight/templates/shoaling.py (理解模板结构)

第四步：写 spec → writing-plans → 实施
  Write spec to docs/superpowers/specs/2026-04-27-lead-agent-workflow-design.md
  Invoke writing-plans skill to create implementation plan
  按 Layer 1-7 顺序实施
```

## 7. 风险与注意事项

| 风险 | 应对 |
|------|------|
| **飞轮录制静默失败** | Layer 6 必须执行 3 个回归测试，确认 ask_clarification turn 录制正确、options 结构化保留、middleware 出错吞异常 |
| **老 thread 崩溃** | experiment-context.json 不存在时必须返回 None，走 free-form fallback，不可抛异常 |
| **paradigm registry 与 templates/ 不同步** | registry 作唯一事实源，新加模板必须同时更新 registry status |
| **Gate 被模型跳过** | GateEnforcementMiddleware 在 tool call 层强制拦截，不依赖 prompt 遵守 |
| **summarization 吃掉 Gate 选择** | 写入 experiment-context.json 持久化，不依赖对话历史 |
| **GLM-5.1 正面提示规则** | prompt 中不用"禁止跳过 Gate""不允许直接 task"，用"必须先 ask_clarification 确认实验类型，再 task" |

## 8. 下一位 Agent 的第一步建议

```bash
# 1. 确认项目能跑
cd /home/qiuyangwang/noldus-insight/packages/agent
make check

# 2. 运行现有测试确保基线绿色
cd backend && source .venv/bin/activate && make test

# 3. 读三个核心文件建立心智模型
# - 计划文档
# - lead agent prompt（dispatch rules）
# - ClarificationMiddleware（理解 ask_clarification 闭环）
```

然后按第 6 节路径推进：写 spec → writing-plans → Layer 1 开始实施。
