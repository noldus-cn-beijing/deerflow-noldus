# Spec: Lead Agent 工作流增强 — 实验识别 + Human-in-the-Loop

**日期**: 2026-04-27
**状态**: 设计定稿，已通过 plan-eng-review，等待实施
**范围**: lead agent 工作流重构，不涉及 subagent 内部逻辑变更

---

## 1. 问题陈述

当前 lead agent 存在两个能力缺口：

1. **实验类型识别不可靠**：用户上传 EV19 数据后，raw data / processed data 的列结构完全一致，纯数据驱动推断无法区分范式。EV19 62 个模板覆盖 18 种行为学范式，需要两级选择（先大类后细分）引导用户确认。
2. **缺少 Human-in-the-Loop**：只有分析完成后的 report 选择是交互式的。关键决策点（实验类型确认、组别设计、结论方向）没有人类确认 gate，飞轮期无法收集高质量反馈数据。

## 2. 设计约束

- 最大程度复用 DeerFlow 已有机制，不另起炉灶
- 支持手动模式（飞轮期，收集反馈）和自动模式（post-training 后，端到端）
- 不改变 subagent 内部逻辑
- 不引入新的前端 UI 组件（复用已有 clarification 按钮渲染 + mode-switcher）
- 零新 API 路由

## 3. 设计决策

### 3.1 所有 Gate 用 ask_clarification 实现

DeerFlow 已有完整闭环：`ask_clarification` tool call → `ClarificationMiddleware` 中断（`Command(goto=END)`）→ 前端按钮渲染 → 用户点击回复 → agent 继续执行。零新 UI 代码，TrainingDataMiddleware 自动录制。

### 3.2 workflow_mode 照抄 is_plan_mode 模式

`is_plan_mode` 是已验证的运行时模式切换：`config.configurable` 注入 → prompt 分支 → middleware 条件启用。`workflow_mode: "manual" | "auto"` 照搬此模式。

### 3.3 GateEnforcementMiddleware 强制 Gate 顺序

prompt 指令有概率漏调，一旦 Gate 被跳过，飞轮 SFT 标注数据丢失。middleware 在 tool call 层拦截：未写 `experiment-context.json` 前不允许 `task()` 调用。

### 3.4 experiment-context.json 持久化 Gate 状态

Gate 之间的选择值不依赖对话历史（会被 summarization 丢弃），写入 `/mnt/user-data/workspace/experiment-context.json`。文件不存在时返回 None，走 free-form fallback 兼容老 thread。

### 3.5 templates/__init__.py registry 作单一事实源

独立维护 paradigm-list.md 和 templates/ 是双事实源，必然不同步。registry 同时服务选项列表和列名校验。

## 4. 架构

### 4.1 三 Gate 流程（Gate 1 为两级选择）

```
用户上传数据 → 选择"端到端数据分析"
    │
    ▼
Gate 1 — 两级实验类型选择
    ├─ 第一级: 7 大类（旷场及物体识别 / 焦虑迷宫 / 空间学习记忆迷宫 / 社会交互与偏好 / 抑郁绝望 / 恐惧条件化 / 斑马鱼行为）
    ├─ 第二级: 该大类下的 1-6 个细分范式
    ├─ 选项来源: ethoinsight.templates.list_categories() + list_paradigms(category=...)
    ├─ ask_clarification(type="approach_choice")
    ├─ 用户选择 → 写入 experiment-context.json（含 category + subject）
    └─ → 数据一致性 safety check
    │
    ▼
Gate 2 — 组别设计确认（仅 manual）
    ├─ ask_clarification(type="missing_info")
    ├─ 选项: 两组对照 / 多剂量组 / 重复测量 / 不确定
    ├─ 用户选择 → 更新 experiment-context.json
    └─ → dispatch code-executor
    │
    ▼
[code-executor → data-analyst]  # 不变
    │
    ▼
Gate 3 — 初步结论审查（仅 manual）
    ├─ ask_clarification(type="approach_choice")
    ├─ 选项: 生成 APA 报告 / 深入分析某指标 / 提出新问题 / 结束
    └─ 用户选择 → 按方向继续或结束
```

### 4.2 数据流

```
workflow_mode ──┬── config.configurable ──▶ prompt 分支 (manual/auto)
                │
                ├── config.configurable ──▶ GateEnforcementMiddleware 条件启用
                │
                └── hooks.ts ──▶ 前端 mode-switcher (flywheel = manual, 其他 = auto)

experiment-context.json ──▶ code-executor 读 (分析开始)
                          ──▶ data-analyst 读 (组别上下文)
                          ──▶ GateEnforcementMiddleware 检查 (拦截/放行)
```

### 4.3 experiment-context.json Schema

```json
{
  "paradigm": "shoaling",
  "paradigm_cn": "斑马鱼鱼群行为",
  "group_design": {
    "type": "two_group",
    "groups": ["对照组", "实验组"],
    "n_per_group": 5
  },
  "paradigm_confirmed_at": "2026-04-27T10:30:00Z",
  "workflow_mode": "manual",
  "gate_completed": ["gate1", "gate2"]
}
```

### 4.4 PARADIGMS Registry Schema（18 范式 / 7 大类）

```python
CATEGORIES = [
    {"name": "open_field", "cn": "旷场及物体识别", "en": "Open Field & Object Recognition"},
    {"name": "anxiety", "cn": "焦虑迷宫", "en": "Anxiety Mazes"},
    {"name": "spatial_memory", "cn": "空间学习记忆迷宫", "en": "Spatial Memory Mazes"},
    {"name": "social", "cn": "社会交互与偏好", "en": "Social Interaction & Preference"},
    {"name": "depression", "cn": "抑郁/绝望", "en": "Depression & Despair"},
    {"name": "fear", "cn": "恐惧条件化", "en": "Fear Conditioning"},
    {"name": "zebrafish", "cn": "斑马鱼行为", "en": "Zebrafish Behavior"},
]

PARADIGMS = {
    "shoaling": {
        "cn": "斑马鱼鱼群行为",
        "en": "Shoaling",
        "category": "zebrafish",
        "subject": "fish",       # rodent | fish | insect | other
        "zones": ["center", "periphery"],
        "expected_columns": ["X_center", "Y_center", "velocity", "IID", "NND"],
        "ev19_arena_templates": ["DanioVision DVOC 004x, 96 round wells"],
        "status": "ready",
    },
    # ... 共 18 种
}
```

新增字段：
- `category`：对应 CATEGORIES 中的 name，用于两级选择过滤
- `subject`：受试动物类型，影响 agent 分析语言（"anxiety-like" vs "shoaling cohesion"）
- `ev19_arena_templates`：对应的 EV19 模板 arena 名（`templateMetaData.xml` 的 `m_strArenaTemplate`），用于文档参考

## 5. 需要修改的文件

| 文件 | 改动量 | 类型 | 内容 |
|------|--------|------|------|
| `ethoinsight/templates/__init__.py` | ~60 行 | **新文件** | PARADIGMS registry + list_paradigms() + verify_paradigm_columns() |
| `agents/lead_agent/agent.py` | ~10 行 | 修改 | workflow_mode 运行时参数注入 + GateEnforcementMiddleware 条件启用 |
| `agents/lead_agent/prompt.py` | ~40 行 | 修改 | prompt 分支：manual/auto 模式下的 Gate 策略 + 模板选择引导 |
| `agents/middlewares/gate_enforcement_middleware.py` | ~80 行 | **新文件** | 强制 Gate 1 完成后才能 task() |
| `skills/custom/ethoinsight-planning/references/quality-gates.md` | ~30 行 | 修改 | 新增 Gate 0 (实验确认) + Gate 1.5 (数据一致性) |
| `frontend/core/threads/types.ts` | ~1 行 | 修改 | AgentThreadContext 新增 workflow_mode |
| `frontend/core/settings/local.ts` | ~1 行 | 修改 | 持久化 flywheel 模式 |
| `frontend/core/threads/hooks.ts` | ~3 行 | 修改 | workflow_mode 注入 configurable |
| `frontend mode-switcher 组件` | ~5 行 | 修改 | 新增 flywheel 选项 |
| `tests/test_templates.py` | 4 测试 | **新文件** | registry shape / verify match / mismatch / 文件不存在 |
| `tests/test_gate_enforcement_middleware.py` | 4 测试 | **新文件** | 拦截 / 放行 / auto 不启用 / ask_clarification 放行 |
| `tests/test_training_data_middleware.py` | 3 测试 | **新文件** | ask_clarification 录入 jsonl / options 保留 / 失败吞异常 |

**总计: ~250 行代码 + 11 单测。13 文件（含 4 新文件）。零新前端组件、零新 API 路由。**

## 6. 不变的部分

- Subagent 内部逻辑（code-executor, data-analyst, report-writer）完全不变
- ethoinsight 统计决策树、指标计算、图表生成完全不变
- MCP、Sandbox、Memory、训练数据录制框架完全不变
- 前端 UI 组件完全不变
- Gateway API 路由完全不变

## 7. Not in Scope

1. 16 种 paradigm 的实际模板实现（templates/epm.py, oft.py 等）— 范围内只有 shoaling
2. 范式知识（zone 阈值、判读逻辑）— 走 golden-cases 流程
3. auto 模式端到端无人值守测试 — v0.1 后
4. ask_clarification 国际化 — 当前中文为主

## 8. 实施顺序

1. **Layer 1 — 单一事实源**: templates/__init__.py registry → verify_paradigm_columns → 4 单测
2. **Layer 2 — 状态存储**: experiment-context.json 读写 + 老 thread 兼容
3. **Layer 3 — Enforcement**: GateEnforcementMiddleware + 4 单测
4. **Layer 4 — Prompt**: prompt.py 分支（manual/auto）
5. **Layer 5 — 前端**: mode-switcher + flywheel + hooks.ts 注入
6. **Layer 6 — 飞轮回归**: TrainingDataMiddleware 3 回归测试
7. **Layer 7 — E2E 验证**: DemoData/shoaling 走完整 manual 流

每层独立可测、可回滚。

## 9. 测试策略

### 单元测试（11 个）

| 测试文件 | 测试内容 |
|----------|----------|
| `test_templates.py` | registry shape / verify match / mismatch / 文件不存在 |
| `test_gate_enforcement_middleware.py` | 拦截未写 context 的 task() / 放行已写 / auto 不启用 / ask_clarification 放行 |
| `test_training_data_middleware.py` | ask_clarification 录入 jsonl / options 结构化保留 / 失败吞异常 |

### 集成测试

- DemoData/shoaling 走完整 manual 流程，确认 3 Gate 触发 + experiment-context.json 写入

### E2E 验证

- manual (flywheel) 模式：每个 Gate 弹出选项按钮，点击后 agent 正确继续
- auto 模式：只 Gate 1，其余端到端不中断
- 老 thread 兼容：无 experiment-context.json 不报错
- 飞轮录制：training-data/ 中 ask_clarification 的 tool call 和用户选择结构化保留

## 10. 失败模式与防御

| 失败场景 | 防御机制 | 用户可见？ |
|----------|----------|-----------|
| LLM 跳过 Gate 直接 task() | GateEnforcementMiddleware 拦截 | 可见：提示"请先确认实验类型" |
| experiment-context.json 不存在（老 thread） | 返回 None → free-form fallback | 不可见 |
| TrainingDataMiddleware 处理 ask_clarification 出错 | 吞异常，不拖累 agent turn | 不可见 |
| 用户选了未实现 paradigm | PARADIGMS status="planned" → 友好降级 | 可见：提示"该范式尚未就绪" |
| 数据列与 paradigm 不匹配 | verify_paradigm_columns → risk_confirmation | 可见：warning 但不阻止 |
| Summarization 丢弃 Gate 选择 | experiment-context.json 持久化 | 不可见 |

## 11. 参考文档

- 设计计划: `/home/qiuyangwang/.claude/plans/dynamic-petting-petal.md`
- 审查意见: `/home/qiuyangwang/.claude/plans/dynamic-petting-petal-review.md`
- 交接文档: `/home/qiuyangwang/noldus-insight/docs/handoffs/2026-04-27-lead-agent-workflow-handoff.md`
- 项目 CLAUDE.md: `/home/qiuyangwang/noldus-insight/CLAUDE.md`
- DeerFlow 后端 CLAUDE.md: `/home/qiuyangwang/noldus-insight/packages/agent/backend/CLAUDE.md`
