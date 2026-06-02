# Sprint 5 设计骨架 — DataQualityGuardrail 数据质量门

**类型**：设计骨架版
**对应**：[roadmap v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) Sprint 5（重估降 2→1 周）+ 2026-05-29 grill 复审 🟡
**估期**：roadmap 标 1 周，**grill 复审：可能 >1 周（依赖未核验）**
**前置**：Sprint 1（data_quality_warnings 结构化 + blocks_downstream）

---

## 0. 目标

code-executor 生成 warning 后 data-analyst 照样跑。有 `severity=critical AND blocks_downstream=true` 时，manual 模式必须先让用户确认再继续。用 DeerFlow 已有 GuardrailMiddleware 拦截下游 subagent 派遣。

> ### 🟡 迁移注记（2026-05-29 编排路径 SSOT 诊断引入，不阻塞本 sprint）
> 本 sprint 会**再手写一个孤立 provider**（DataQualityGuardrailProvider）。这正是 [编排路径 SSOT 诊断](2026-05-29-orchestration-path-ssot-diagnosis-design.md) §2.2 批评的"打补丁式 provider"模式——目前 8 条路径的拦截规则散落在多个各管一段的 provider 里，无 SSOT 桥接。
> **本 sprint 仍按原计划先实现**（quality gate 是真实需求，不阻塞）。但实现时请：
> 1. 把"在什么 INTENT/什么 step 之后该拦什么"写成**数据驱动的判定**（读 gate_completed / handoff 状态），而非把路径逻辑硬编码进 provider 类
> 2. 在 spec/代码注释里标注：**未来编排路径 SSOT 落地后，本 provider 的拦截规则应迁移为从 path SSOT 生成**，而不是长期作为独立手写补丁
> 这样将来 SSOT 化时，本 provider 是"可迁移的样板"而非"又一块要拆的债"。

---

## 1. grill 复审发现（实施前必读）

### 🟡 估期乐观（5.8 同款风险）
roadmap 说"`Ev19TemplateGuardrailProvider` 是完美模板，克隆改改，1 周富余"。**5.8 也以为克隆现成的很简单，核验后冒出 3 个踩空点。** 别假设 1 周够。

### 依赖未核验（实施前必须确认就位）
Sprint 5 依赖一串**尚未确认实施**的东西：
- `blocks_downstream: bool` 字段 — Sprint 1 加的。**Sprint 1 已实施？**（roadmap 标已实施，但本 sprint 实施前要确认该字段真在 DataQualityWarning schema 里）
- `workflow_mode=auto/manual` — roadmap 说"deerflow 现成配置直接复用"。**这是真的吗？需 grep 确认 deerflow 有没有 workflow_mode 这个 config，还是要新建**（这是最大的未验假设）
- `gate2_quality_acknowledged` — 要新加到 experiment-context.json
- `set_experiment_paradigm(acknowledge_quality=True)` — 复用现有 gate 机制，确认该参数路径在

### 避免双重提示（Sprint 1 dogfood 教训，roadmap 已记）
manual 模式下 GateEnforcementMiddleware 已用粗粒度 ToolMessage 触发反问，Sprint 1 又给 banner 二次展示同一 warning。Sprint 5 替换 gate2 拦截路径时，**拒绝消息直接含结构化 warning payload，让前端用同一 banner 渲染**，guardrail 只贡献"中断状态"，不产生第二条文字。

---

## 2. 设计（骨架）

**Auto / Manual 双轨（grill 锁定）**：
| 模式 | 行为 | 实现 |
|---|---|---|
| `workflow_mode=auto`（默认日常用户）| 不阻断，UI 红字（Sprint 1 已做） | **不挂** DataQualityGuardrailProvider |
| `workflow_mode=manual`（飞轮专家）| guardrail 拦下游，必须 ask_clarification + acknowledge 才放行 | **挂** DataQualityGuardrailProvider（仿 GateEnforcementMiddleware 仅 manual 启用） |

**改动（骨架）**：
- `guardrails/data_quality_provider.py` 新建 — 仿 `Ev19TemplateGuardrailProvider`：pre-tool-call 拦 `task(subagent_type ∈ {data-analyst, chart-maker, report-writer})`，读 handoff 中 `critical AND blocks_downstream` warning，检查 `gate2_quality_acknowledged` → 未确认则 deny + 含明确指令（[[feedback_deny_messages_must_direct]]：deny 必须告知 lead 下一步做什么 + 含结构化 payload 给 banner）
- `guardrails/__init__.py` — 注册
- `lead_agent/agent.py` — **仅 manual 模式**挂（仿 GateEnforcementMiddleware 的条件挂载）
- `lead_agent/prompt.py` — manual 模式调度边界加 gate 规则

**拦截目标**：task(data-analyst/chart-maker/report-writer)；knowledge-assistant 免拦。

---

## 3. 实施前核验清单（升级为核验版必做）
1. **`workflow_mode` 是否是 deerflow 现成 config**（grep config schema）——决定是"复用"还是"新建"，影响估期最大
2. `blocks_downstream` 字段是否真在 DataQualityWarning（Sprint 1 产出）
3. `Ev19TemplateGuardrailProvider` 的真实结构（克隆模板，看 ContextVar bridge / evaluate / deny 怎么写）——5.8 教训：别假设克隆简单
4. GateEnforcementMiddleware 的 manual 条件挂载怎么写的（仿它）
5. `gate2_quality_acknowledged` / `set_experiment_paradigm(acknowledge_quality)` 路径

---

## 4. 验收（骨架）
1. manual 模式 + critical/blocks_downstream warning → 派 data-analyst 被 deny，含明确指令 + 结构化 payload
2. ask_clarification → set_experiment_paradigm(acknowledge_quality=True) → 放行
3. auto 模式不挂 guardrail（不阻断，仅红字）
4. 不产生双重提示（guardrail 只中断，banner 是唯一信息源）

## 5. 不在范围
- ❌ auto 模式挂 guardrail（auto 靠 UI 红字）
- ❌ knowledge-assistant 拦截
