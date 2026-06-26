# Spec：运行轨迹面板 · 档 A Live Trace（Phase 0 · 第 2 项）

> 类型：**一次性实施 spec**（前端，零后端依赖）
> 日期：2026-06-24
> 母方案：[docs/plans/2026-06-24-frontend-generative-ux-upgrade.md](../../plans/2026-06-24-frontend-generative-ux-upgrade.md)（§5.2 档 A）
> 依赖：[2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md](2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md)（用其 `--ease-*` / `--dur-*` / `--color-status-*`）
> 适用层：`packages/agent/frontend/src/components/workspace/` + `core/tasks` + `core/threads/hooks.ts`（只读消费,不改流式核心）
> 设计准则来源：`ui-ux-pro-max`（domain ux：AI-Interaction Streaming、progressive-disclosure；Quick Reference §7/§9）
> 一句话：新增一个**可切出的"运行轨迹"侧抽屉**,把本次正在跑的分析里 agent 的全部行为（派遣子代理 / 工具调用 / 质检 gate / HITL 决策）**实时**按时序铺成一条可展开的时间线。**全部数据来自已有实时事件,零后端改动**——这是用户"运行轨迹现在就做"的当期可交付件。

---

## 〇、为什么是档 A（live）而不是直接做 replay

母方案 §5.2 经后端核查（2026-06-24）确认：
- **lead 层 + subagent 内部步骤的实时事件,前端全收得到**（`task_started/running/completed/failed`、tool_calls、reasoning、gate_signals 实时 AIMessage、clarification）。→ **live trace 全保真,零后端依赖,今天能做**。
- 但 subagent 内部 `task_running` 是**临时事件,跑完即丢未持久化**（task_tool 只 SSE 不写 journal）。→ **历史回放（档 B）的 subagent 内部步骤需 ~30 行后端补丁**,故拆出。

所以档 A 先做（当期满足用户）、档 B（replay）放 Phase 1。**本 spec 只做档 A**。

> 一个关键洞察：现在前端**已经在消费这些事件**了——`SubtaskCard` 已渲染 subagent 时间线,`onCustomEvent` 已接 `task_running`（`hooks.ts:477-490`）。档 A 不是"接新数据",而是**把已在手的数据换一个聚合视角呈现**：从"散在消息流里的若干卡片"→"一条统一的、可纵览的运行时间线"。

---

## 一、现状（带证据）

### 1.1 已有的事件 / 数据（全部可复用，零后端改动）

后端实时 custom event（母方案后端调研 §2 + 本轮核查坐实，`task_tool.py:544-631`）：

| 事件 | 载荷 | 前端现状 |
|---|---|---|
| `task_started` | `{task_id, description}` | 部分消费 |
| `task_running` | `{task_id, message(AIMessage), message_index, total_messages}` | `hooks.ts:477-490` → `updateSubtask` |
| `task_completed` | `{task_id, result, usage}` | message-list 推导 |
| `task_failed` / `task_cancelled` / `task_timed_out` | `{task_id, error, usage?}` | message-list 推导 |

加上 thread message 流里已有的：lead 的 `tool_calls`、reasoning、`gate_signals`（实时 AIMessage 里）、`ask_clarification`、artifacts 变化。

### 1.2 已有的前端基建（复用）

| 资产 | 位置 | 复用方式 |
|---|---|---|
| `useSubtask` / `useUpdateSubtask` / Subtask 类型 | `core/tasks/context.tsx`、`core/tasks/types.ts` | 轨迹面板的 subagent 数据源 |
| `stage-broadcast.ts`（工具→中文语义文案） | `core/tools/stage-broadcast.ts` | 轨迹里每步的人类可读标题 |
| `ChainOfThought` 时间线视觉 | `ai-elements/chain-of-thought` | 轨迹时间线的视觉骨架 |
| `QualityWarningBanner` / `QualityWarning` 类型 | `messages/quality-warning-banner.tsx` | gate 卡明细 |
| `extractQualityWarnings` / `findToolCallResult` 等 | `core/messages/utils.ts` | 从消息抽 gate/工具结果 |
| `ResizablePanel`（artifacts 侧栏已用） | `chats/chat-box.tsx` | 抽屉布局参照（但见 §3.3 布局决策） |

### 1.3 缺口

- **没有"纵览"视角**：用户只能在消息流里顺着读,无法"一眼看完 agent 这次干了哪几步、卡在哪、每步多久"。
- subagent 完成后 `SubtaskCard` 折叠进消息流历史,**时序关系**（谁先谁后、间隔多久）丢失。
- gate 判定散在各 subagent 的 handoff 汇报里,无**集中的质检关卡视图**。

---

## 二、目标与非目标

### 目标
1. 新增**运行轨迹侧抽屉**（可开关,默认关,不占空间）,呈现**当前 thread 当前 run** 的实时 agent 行为时间线。
2. 时间线节点类型：`范式锁定` / `子代理派遣（含内部步骤）` / `工具调用` / `质检 gate（绿/黄/红）` / `HITL 决策` / `产物生成`。
3. 每节点：人类可读标题（走 `stage-broadcast`）+ 状态图标 + 相对时序 + 可展开看 input/output。
4. 实时流式更新（事件到达即追加节点,带 §spec1 的入场动效）。
5. 与母方案 §3 进度轨、§5.1 决策卡**数据同源**（同一套 subtask/事件,不同视角）。

### 非目标
- ❌ **不做历史 replay**（档 B → Phase 1）。本面板只反映"当前正在跑/刚跑完的本 run"。
- ❌ 不做"带真实时序的回放播放器"（那是档 B）。live 是"实时追加",不是"重播"。
- ❌ 不改 `useStream` / `mergeMessages` / `groupMessages` / 任何流式合并逻辑（只**额外消费**事件,不改既有渲染）。
- ❌ 不加后端事件 / 不加后端端点。
- ❌ 不动 `SubtaskCard` 在消息流里的现有渲染（轨迹面板是**并行的第二视角**,不替换消息流）。

---

## 三、设计

### 3.1 数据层：一个聚合 hook `useRunTrace`

新增 `core/tasks/use-run-trace.ts`（或 `core/trace/`）——**纯聚合,不发请求,不改流式核心**：

```
输入（全部已有）:
  - thread.messages（lead 的 tool_calls / reasoning / gate / clarification / present_file）
  - subtasks（来自 useSubtask context，含每个 subagent 的 messages[] + status）
  - onToolEnd / onCustomEvent 已捕获的事件（hooks.ts 现有）

输出: TraceEvent[]（按时序）
  TraceEvent = {
    id, kind: 'paradigm'|'dispatch'|'tool'|'gate'|'clarification'|'artifact',
    title: string,            // 走 stage-broadcast 得人类可读文案
    status: 'running'|'ok'|'warning'|'failed'|'waiting',
    detail?: {...},           // 展开内容：args / result / gate warnings / 子步骤
    subEvents?: TraceEvent[], // dispatch 节点下挂 subagent 内部 tool/reasoning 步骤
    order: number,            // 逻辑时序（用 message 在流里的顺序 + subtask message_index）
  }
```

- **时序来源**：用 message 在 `thread.messages` 里的顺序 + subagent `message_index`（`task_running` 载荷自带）。**live 不需要 created_at 墙钟**（那是档 B replay 才需要的"真实间隔"）——live 就是"到了就追加"。
- **gate 节点**：从 AIMessage 的 `extractQualityWarnings` + handoff 汇报里的 `gate_signals`（实时在消息里）抽。绿/黄/红映射 `statistical_validity` / `critical_count` / `blocks_downstream`。
- **subagent 内部步骤**：直接复用 subtask 的 `messages[]`（`convertToSteps` 已有逻辑,`message-group.tsx:579`）。

> ⚠️ 工程纪律：`useRunTrace` 是**只读派生**——它 `useMemo` 从已有 state 算 TraceEvent[],**不写任何 state、不碰 `thread.submit`、不改 dedupe/merge**。这保证不回归流式核心（CLAUDE.md 红线）。

### 3.2 视图层：时间线（日式克制）

- **垂直时间线**：一条**极细竖线**（`border-l` 1px,不是粗轴）+ 小圆点节点（母方案 §4.3 视觉降噪：不堆方框）。
- **节点状态色**：用 spec1 的 `--color-status-*`（running=brand 脉动 / ok=success / warning=warning / failed=danger / waiting=warning 脉动）。**色 + 图标 + 文字三件套**（`color-not-only`）。
- **dispatch 节点可展开**：默认折叠成一行"🔬 数据解读中…"（`stage-broadcast`）,点开露出该 subagent 的内部 tool/reasoning 子步骤（缩进二级时间线）。`progressive-disclosure`。
- **gate 节点**：绿✓/黄⚠/红✗ 一行,点开是 `QualityWarningBanner` 同款明细（severity/code/message/blocks_downstream）。
- **实时追加动效**：新节点用 spec1 的 `--animate-fade-in-up`（`ease-brand-out`,从下方进 = `hierarchy-motion`）+ stagger（连续多个节点逐个 30-50ms 入场,不齐刷）。
- **流式态**：当前进行中的节点用 `animate-pulse-soft`（brand 呼吸,已有）。对齐 `ui-ux-pro-max` AI-Interaction `Streaming`：**有实时进展可见,绝不是 10s+ 空转 spinner**。

### 3.3 布局：抽屉如何与现有 artifacts 侧栏共存（关键决策）

现有 `chat-box.tsx` 已用 `ResizablePanelGroup` 做 chat(60) / artifacts(40) 二分栏。运行轨迹是**第三个面**,不能和 artifacts 抢同一块。方案：

- **运行轨迹做成 overlay 抽屉**（从右侧滑入,盖在 chat 之上,`fixed` + scrim）,**不进 ResizablePanelGroup**——避免三栏 resize 复杂度 + 避免和 artifacts 互斥。
- 触发：workspace header 或输入框旁一个"运行轨迹"图标按钮（`ListTreeIcon` 类）。徽章提示"进行中 N 步"。
- 抽屉宽度 `clamp(360px, 33vw, 480px)`,从触发源方向滑入（spec1 `modal-motion`：从触发处展开,`ease-brand-out` + `duration-slow`）。
- scrim 仅在窄屏（<1024px）出现（覆盖式）;宽屏可作非模态浮层（不锁背景,`ui-ux-pro-max` `adaptive-navigation`）。
- ESC / 点 scrim / 再点按钮关闭（`modal-escape` / `escape-routes`）。

> 为什么不复用 artifacts 的 ResizablePanel：artifacts 是"产物消费"（看图看报告,需要大空间、常驻）;运行轨迹是"过程监督"（瞥一眼进度,瞬时）。两者心智/时机不同,叠进同一 resize 组会互相挤。overlay 抽屉更轻、更符合"按需查看"。

### 3.4 入口与默认态

- 默认**关闭**（不占空间,不打扰非技术研究员）。
- 运行中,入口按钮显**进度徽章**（"3 步进行中"/ 完成后"7 步"）——给想看的人一个钩子,不强推。
- 出错时（任一 gate=红 / subagent failed）,入口按钮**变 danger 色 + 轻脉动**,引导用户点开看"卡在哪"（`error-recovery`：清晰恢复路径）。

---

## 四、实施步骤

### Step 1：`useRunTrace` 聚合 hook（`core/`）
纯派生 hook,`useMemo` 从 `thread.messages` + subtasks 算 `TraceEvent[]`。复用 `convertToSteps` / `extractQualityWarnings` / `stage-broadcast` / `findToolCallResult`。单测：喂一组 mock messages（含 dispatch + gate + clarification）→ 断言 TraceEvent[] 时序/状态正确。

### Step 2：`RunTracePanel` 组件 + `TraceEventItem`（`components/workspace/trace/`）
时间线视觉（§3.2）。`TraceEventItem` 按 kind 分支渲染（dispatch 可展开挂子步骤 / gate 挂 QualityWarningBanner / tool 走 stage-broadcast）。

### Step 3：overlay 抽屉容器 + 入口按钮（§3.3 / §3.4）
- 抽屉用 Radix Dialog（非模态变体）或自实现 fixed + transform 滑入（复用 spec1 曲线）。
- 入口按钮放 workspace header 右侧或输入框工具区,带徽章 + danger 态。

### Step 4：接线
- 入口按钮开关抽屉;抽屉内挂 `RunTracePanel`,数据来自 `useRunTrace`。
- **不改 `hooks.ts` 的事件处理**——`useRunTrace` 直接读 `thread.messages` + `useSubtask` context（都是现成的)。若需 `onToolEnd` 的额外信号,从现有 context 取,不加新回调。

### Step 5：i18n
轨迹节点标题、状态文案、入口 tooltip 进 `core/i18n/locales/{zh-CN,en-US,types}.ts`（对齐现有 `t.toolCalls.stageBroadcast.*` 模式,**不硬编码中文**——CLAUDE.md / memory 铁律）。

### Step 6：reduced-motion + a11y
- 节点入场 / 脉动在 reduced-motion 下降级（spec1 机制 + motion 库 `useReducedMotion()` 若用 spring）。
- 抽屉键盘可达：focus trap、ESC 关、入口按钮 aria-label、时间线节点可 tab + 展开（`focusable-elements` / `keyboard-nav`）。

---

## 五、验收标准

### 功能
- [ ] 运行轨迹抽屉可从入口按钮开关,默认关。
- [ ] 运行中,抽屉实时追加节点：范式锁定 / 每次派遣 / 每次工具 / 每个 gate / 每次反问 / 产物生成,时序正确。
- [ ] dispatch 节点可展开,露出该 subagent 内部 tool/reasoning 步骤（复用 convertToSteps）。
- [ ] gate 节点绿/黄/红正确,展开见 DataQualityWarning 明细。
- [ ] 入口按钮徽章反映进行中步数;出错时变 danger + 脉动。
- [ ] 与消息流里现有 SubtaskCard **并存不冲突**（轨迹是第二视角,消息流不变）。
- [ ] 与 artifacts 侧栏**不互斥**（overlay,不进同一 ResizablePanelGroup）。

### a11y / 性能
- [ ] 抽屉 focus trap + ESC + scrim 关闭（窄屏）;入口 aria-label。
- [ ] 节点入场动效用 spec1 曲线;reduced-motion 下降级。
- [ ] 实时进展可见,无 10s+ 空转 spinner（AI-Interaction Streaming）。
- [ ] gate/状态"色 + 图标 + 文字"三件套。
- [ ] 大量节点（一次复杂分析几十步）滚动流畅;若单 run 节点 >50,时间线列表虚拟化（`virtualize-lists`）——**否则至少懒渲染折叠的子步骤**。

### 工程纪律
- [ ] `useRunTrace` 纯派生只读,**不写 state、不碰 submit/merge/dedupe**。
- [ ] **不改** `useStream` / `groupMessages` / `SubtaskCard` 现有渲染。
- [ ] **零后端改动**（不加事件、不加端点）。
- [ ] `pnpm check` 通过 + `useRunTrace` 有单测。
- [ ] i18n 不硬编码中文。

---

## 六、风险与回退

| 风险 | 缓解 |
|---|---|
| 轨迹与消息流"信息重复"惹烦 | 轨迹默认关 + 是"纵览/监督"视角（消息流是"叙事/阅读"）,两者时机不同;默认关不打扰非技术用户 |
| 聚合逻辑与消息流渲染漂移（同事件两处算法不一致） | `useRunTrace` 复用消息流同款 util（convertToSteps/extractQualityWarnings）,不另写解析;单测锁定 |
| 三面板（chat/artifacts/trace）布局打架 | trace 走 overlay 不进 ResizablePanelGroup（§3.3 决策） |
| 节点过多卡顿 | 虚拟化 + 折叠子步骤懒渲染 |

**回退**：纯新增组件 + 一个只读 hook + 一个入口按钮;移除入口按钮即"隐藏",`git revert` 即可。不动任何既有路径,回退零风险。

---

## 七、给实施 agent 的交接

- 新增：`core/.../use-run-trace.ts`（+ 单测）、`components/workspace/trace/{run-trace-panel,trace-event-item}.tsx`、入口按钮、i18n 条目。
- **只读消费**：`thread.messages`、`useSubtask` context、现有 util。**不改** `core/threads/hooks.ts` 的事件处理、不改 `message-list.tsx` / `subtask-card.tsx` 渲染。
- 复用 spec1 的 `--ease-*` / `--dur-*` / `--color-status-*`（**不重定义**）。
- 与 Phase 0 spec #4（进度轨）**数据同源**：进度轨是"7 阶段宏观",轨迹是"每步微观",都从 `useRunTrace`/subtask 派生——实施 #4 时复用本 spec 的聚合 hook,别另起一套。
- 档 B（replay）是 Phase 1 单独 spec：届时 `useRunTrace` 增加"从 history 端点喂历史 run"的输入源 + 后端两个小补丁。**本 spec 把 `useRunTrace` 的输入设计成可扩展**（live 源 now / history 源 later），为档 B 留口。

---

*依据：母方案 §5.2 档 A + 后端事件契约核查（task_tool.py / 实时 custom event 全可消费）+ `ui-ux-pro-max`（AI-Interaction Streaming、progressive-disclosure、modal/navigation）。未写代码。*
