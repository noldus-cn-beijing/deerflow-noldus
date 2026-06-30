# 设计 spec：A2 前端分轨渲染（消费 A1 事件，阶段叙事内联对话流）（2026-06-30）

> 生成式 UX 路线图第二步。**依赖 A1**（`2026-06-30-generative-ux-roadmap-and-a1-event-track-foundation-design.md` 的后端事件分轨地基）。本 spec 只设计 A2，交别的 agent 实施。**实施需等 A1 上线**（A2 消费 A1 发出的 custom stage 事件）。

---

## 一、目标与核心决定

A1 在后端把信息分了轨（messages 正文 / reasoning 思考 / custom 阶段叙事 / values 产物）。**A2 让前端按轨道来源决定渲染形态**，并消除前端的翻译漂移源。

三个已拍板的核心决定：

1. **甲方案：阶段叙事内联进对话流**，不做独立横向进度轨。
   - 承接 `#231`（`2026-06-29-remove-progress-rails.md` / commit `bf9b48ed`）——用户已删除 `#214` 的 7 阶段常驻横轨，理由是「状态卡死 + 与对话流信息重复」。A2 不重蹈覆辙：不再造横轨，阶段叙事作为对话流的有机部分。
2. **A 方案：后端翻译取代前端查表**。删除 `stage-broadcast.ts`（前端把 tool_call 查表翻译成中文文案的机制），改为直接渲染 A1 事件携带的后端 `narration`。消除漂移源（工具改了前端翻译没跟上），守 single-source-of-truth。
3. **阶段状态只认 A1 后端事件**，前端不推导。`#214` 卡死的真根因正是「前端从 trace 推导阶段、推不出 run 结束信号」；A2 的阶段 active/completed **只由 A1 的 `stage_update` 驱动**，从根上避免卡死。

---

## 二、前端现状（已勘察坐实）

- `#231` 已删除：`src/components/workspace/analysis-rail/`、`src/core/workflow/`、`src/core/trace/`、`src/components/workspace/trace/`（四目录均已不存在）。所以 A2 不与旧进度轨冲突。
- custom 轨已被消费：`hooks.ts:539 onCustomEvent` 已处理 `task_running`/`llm_retry` 等 custom 事件——**A1 的 stage 事件在此新增分支即可**，分发点现成。
- 内联状态渲染位现成：`subtask-card.tsx:106-120` 用 `ChainOfThoughtStep` + `getStageBroadcastForSubagent(task.subagent_type, t)` 显示阶段文案（completed/failed/active 三态 + Shimmer 流式效果）——**这就是「甲方案」要接管的渲染位**。
- `stage-broadcast.ts` 消费点仅 2 处：`subtask-card.tsx`（subagent 翻译）+ `message-group.tsx:428`（bash 翻译）+ i18n `stageBroadcast.*` 段（types/zh-CN/en-US 三文件）。改动面小、可控。

---

## 三、模块设计

### 模块 1：custom 轨 stage 事件接入（`hooks.ts:onCustomEvent`）

在已有 `onCustomEvent` 分发点新增分支，处理 A1 的两类事件：
- `stage_update{stage, status:"active"|"completed", narration}` → 更新对应阶段的内联状态。
- `stage_plan{stages, skipped}` → 记录本次流水线的阶段全貌（供内联渲染知道顺序/总数）；非流水线意图 A1 不发 → 前端无阶段渲染（纯对话）。

状态存放：复用现有 subtask store（`core/tasks/context.tsx`）或并行一个轻量 stage store——实施时择一，但**stage 状态来源唯一 = A1 事件**，不掺前端推导。

### 模块 2：渲染改造——后端 narration 取代前端查表（A 方案）

- `subtask-card.tsx`：`getStageBroadcastForSubagent(task.subagent_type, t)` → 改为渲染 A1 `stage_update.narration`。保留现有三态视觉（completed/failed/active + Shimmer 心跳）。
- `message-group.tsx:428`：`getStageBroadcastForBash(...)` → 同改为消费后端 narration（或对纯 bash 步骤不再单独翻译，由 A1 的阶段叙事覆盖）。
- **删除** `src/core/tools/stage-broadcast.ts` + i18n 三文件的 `stageBroadcast.*` 段（消除漂移源）。删后 grep 确认无残留引用，`pnpm check` 0。

### 模块 3：三轨渲染分工（A2 全景，明确什么动什么不动）

| 轨 | 来源 | A2 处理 |
|---|---|---|
| `messages`（正文） | LLM token | **不动**——对话气泡逐字流式，保「还在跑」心跳 |
| `reasoning_content`（思考） | ThinkTagMiddleware | **不动/微调**——已有折叠渲染（`message-list-item.reasoning-collapse`） |
| `custom` stage 事件 | **A1** | **本 spec 核心**——内联对话流的阶段状态行（narration + active/completed） |
| `values`（产物） | state | **不动**——已有 thread-assets-panel |

---

## 四、边界与不做什么

- ❌ 不做独立横向进度轨（承接 #231 的删除决定）。
- ❌ 不动 messages 正文流式 / reasoning 折叠 / 产物面板（守流式心跳 + 不扩范围）。
- ❌ 不让前端推导阶段状态（#214 卡死根因）——只认 A1 事件。
- ❌ 不引入 ag-ui / CopilotKit 组件。
- ✅ 硬依赖 A1 已上线（A2 消费其 custom stage 事件）。

---

## 五、验收

1. **A1 已上线**为前置（A2 无法独立于 A1 验收——可用 A1 的事件 fixture 做单测）。
2. **TDD**（CLAUDE.md 强制）：
   - 喂 A1 的 `stage_plan` + 序列 `stage_update` fixture → 断言对话流内联渲染对应阶段叙事，active→completed 正确翻转。
   - 断言阶段结束后状态变 completed **不卡死**（喂「最后一个 stage_update completed、其后无更多事件」→ 断言不再 active）——直接覆盖 #214/report-writer 卡死那类回归。
   - **防 vacuous**：断言渲染的文案来自 A1 事件的 `narration`（后端来源），**不是** `stage-broadcast` 查表；断言工具名（`identify_ev19_template`）/gate 关键字不出现在对话流。
   - 断言非流水线意图（A1 不发 stage_plan）→ 前端无阶段渲染、纯对话。
3. `stage-broadcast.ts` + i18n `stageBroadcast.*` 已删，无残留引用。
4. `pnpm check` 0 + `npx vitest run` 绿（含新断言 + 不回归现有 subtask-card/message-group 测试）。

---

## 六、关联

- 上游依赖：A1（`2026-06-30-generative-ux-roadmap-and-a1-event-track-foundation-design.md` 第二部分）。
- 承接决定：`#231`（`2026-06-29-remove-progress-rails.md`，删 7 阶段横轨 + RunTrace 徽章）——A2 不复活横轨，只内联。
- 同病根：#214 进度轨卡死、report-writer 卡运行态（`2026-06-30-subtask-card-stuck-in-progress-after-run-success-fix-spec.md`，#248 已合）、dots 卡死（#247 已合）——共同根因「前端推导无 run 结束信号」，A2 用「只认 A1 后端事件」从根上规避。
- 后续：C1（画廊纳 code-executor 产物）、C2（画廊 UI/UX）、C3（产物框选追问）。
