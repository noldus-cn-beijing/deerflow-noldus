# 前端打磨 + 子任务状态 bug 修复交接文档

**日期**: 2026-05-06
**交接人**: Claude (本会话)
**接手对象**: 下一位 AI Agent / 开发者
**任务状态**: ✅ 全部已实施并 commit (`1837b0e2`),待人工验证视觉 + 子任务流程

---

## 1. 当前任务目标

继 [2026-04-30-frontend-aesthetic-upgrade-handoff.md](2026-04-30-frontend-aesthetic-upgrade-handoff.md) 之后,对工作区做细节打磨:

1. 输入框从「贴底卡条」改成「悬浮玻璃卡」
2. 收紧空态输入框的视觉比例(尺寸 + 占位文字位置)
3. 删除冗余 UI(欢迎描述段、SuggestionList 行)
4. 移除 SubtaskCard 上的浮夸彩虹特效,替换成符合「日式极简 × 冷绿米」设计语言的状态指示
5. 修复一个发现的真 bug:子任务卡片完成后不会从「运行中」翻到「已完成」状态

---

## 2. 当前进展

### ✅ 已完成 — 单个 commit (`1837b0e2`)

提交信息:**修复了前端的一些bug,修复了subagent的一些事件被静默丢弃的问题**

#### 2.1 输入框悬浮化

| 文件 | 改动 |
|---|---|
| [src/styles/globals.css](../../packages/agent/frontend/src/styles/globals.css) | 新增 `--shadow-float` token(三层柔和阴影,Forest `rgba(26,72,64,...)`)+ `.shadow-float` 工具类 |
| [src/components/workspace/input-box.tsx](../../packages/agent/frontend/src/components/workspace/input-box.tsx) | `bg-background/85` → `bg-card/90` + `shadow-float`;**删除底部那条 `bg-background` 遮罩条**(line 611 旧代码) |
| [src/app/workspace/chats/[thread_id]/page.tsx](../../packages/agent/frontend/src/app/workspace/chats/[thread_id]/page.tsx) | 容器加 `pb-4`、移除 `-translate-y-4`;占位骨架同步换成悬浮风格 |

#### 2.2 输入框尺寸优化

[input-box.tsx](../../packages/agent/frontend/src/components/workspace/input-box.tsx) 的 `PromptInputTextarea`:

```diff
- className={cn("size-full")}
+ className={cn("size-full min-h-14 py-4 text-[15px] leading-6")}
```

**几何**:`56px - 32px(padding) = 24px` 内部高度 = `leading-6`(24px)→ 占位「今天我能为你做些什么?」**垂直居中**,不再贴顶。

#### 2.3 删除冗余 UI

| 文件 | 删除内容 |
|---|---|
| [src/components/workspace/welcome.tsx](../../packages/agent/frontend/src/components/workspace/welcome.tsx) | 删除非 skill 模式下的 `t.welcome.description`(「欢迎使用 🦌 DeerFlow,一个完全开源的超级智能体...」段) |
| [src/components/workspace/input-box.tsx](../../packages/agent/frontend/src/components/workspace/input-box.tsx) | 移除 `<SuggestionList />` 渲染 + 删除整个 `SuggestionList` 函数定义 + 清理无用 import (`useSearchParams`、`PlusIcon`、`ConfettiButton`、`DropdownMenu*` 等) |

#### 2.4 子任务运行态:去除浮夸特效

| 文件 | 改动 |
|---|---|
| [src/components/workspace/messages/subtask-card.tsx](../../packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx) | **删除** `<div className="ambilight z-[-1]">` (45° 紫粉绿黄红渐变 + 60px blur,40s 循环);**删除** `<ShineBorder shineColor={["#A07CFE","#FE8FB5","#FFBE7B"]}>` (旋转虹色描边);**新增** `border-brand/50` + `animate-pulse-soft` 柔和 Lime 脉动 |
| [src/styles/globals.css](../../packages/agent/frontend/src/styles/globals.css) | **删除** `.ambilight` 的所有定义和 `@keyframes ambilight`;**新增** `.animate-pulse-soft` 工具类 + `@keyframes pulse-soft` (2.4s ease-in-out,Lime 阴影从 0.18 → 0.32 透明度);新增 `prefers-reduced-motion` fallback 静态描边 |
| import 清理 | 移除 `import { ShineBorder } from "@/components/ui/shine-border"` |

#### 2.5 子任务状态 bug 修复 ⚠️ **重要**

**症状**:多个子任务串行执行时,先启动的子任务后端已完成,UI 卡片仍卡在「子任务运行中」状态,不翻到「已完成」。

**根因**(基于 langgraph.log 和代码追踪):

后端 [task_tool.py:191](../../packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py) 在子任务完成时发送三种 SSE 事件:

```python
writer({"type": "task_completed", "task_id": ..., "result": ...})
writer({"type": "task_failed", "task_id": ..., "error": ...})
writer({"type": "task_timed_out", "task_id": ..., ...})
```

但前端 [hooks.ts:252-280](../../packages/agent/frontend/src/core/threads/hooks.ts) 之前**只处理 `task_running`**,其他三个事件被**静默丢弃**。

冗余路径(message-list.tsx 在每次重渲染时遍历 `thread.messages`,看到 `Task Succeeded. Result:...` 工具消息后调用 `updateSubtask({status: "completed"})`)在 [context.tsx:78-87](../../packages/agent/frontend/src/core/tasks/context.tsx) 走的是**渲染期路径**——in-place mutation 不调用 `setTasks`。原始注释假定「下一个 SSE 事件会传播状态」——但 `task_completed` 没被处理,所以根本不会有那个事件。

**修复**:

| 文件 | 改动 |
|---|---|
| [src/core/threads/hooks.ts](../../packages/agent/frontend/src/core/threads/hooks.ts) (`onCustomEvent`) | 新增 `task_completed` / `task_failed` / `task_timed_out` 三个 handler,分别调用 `updateSubtask({status: "completed"\|"failed"})` |
| [src/core/tasks/context.tsx](../../packages/agent/frontend/src/core/tasks/context.tsx) (`useUpdateSubtask`) | 新增「终态 SSE 路径」:`status` 变成 completed/failed 且**确实变化**时调用 `setTasks` 触发重渲染。状态未变化时(渲染期被反复调用)走原有 in-place mutation 路径,避免渲染循环 |

---

## 3. 关键上下文

### 设计原则(继承自 04-30 交接)

- **底色**: `oklch(0.972 0.012 145)` 浅绿米
- **品牌色**: Lime `#10DD8B` 仅 hero CTA + input-box Submit + **(本轮新增)** 子任务运行态描边脉动
- **阴影**: 之前只允许 modal (`--shadow-modal`),本轮新增 `--shadow-float` (输入框) + `pulse-soft` 动画(运行态),都用 Forest 透明色
- **不引入新色相**(明确删除了紫粉橙的 `ShineBorder`)

### 字体 / 容器宽度等保持不变

- OPPO Sans 4.0 (latin 40KB + zh 3.6MB)
- 新对话 max-width: `--container-width-sm = 576px`
- 对话中 max-width: `--container-width-md = 816px`

### 关键约束

- 包管理器 `pnpm`,Node 22+
- **不要用** `make dev` (端口 2026,全栈),**用** `pnpm dev` (端口 3000,Turbopack)
- 工作目录: `/home/wangqiuyang/noldus-insight/packages/agent/frontend/`
- typecheck + lint 已全部通过

---

## 4. 关键发现

### 4.1 子任务事件流不对称

后端发 4 种 SSE custom event(`task_started` / `task_running` / `task_completed` / `task_failed` / `task_timed_out`),前端之前**只处理 1 种** (`task_running`)。这是个长期存在的潜在 bug,只是单子任务场景下被「重渲染时遍历工具消息」这条冗余路径覆盖,多子任务串行时暴露。

### 4.2 `useUpdateSubtask` 双路径设计的脆弱性

[context.tsx](../../packages/agent/frontend/src/core/tasks/context.tsx) 的 `updateSubtask` 是「渲染期 in-place mutation」+「SSE 期 setTasks」的混合设计。注释明确说「渲染期路径依赖下一个 SSE 事件传播」。这个假设很脆弱——任何场景下 SSE 事件链断了,UI 状态就僵在中间态。

**未来改进方向**(P2):重构成全 setTasks,用 `useEffect` 或 `useMemo` 把 `thread.messages → tasks` 做成纯派生(derived state),消除 in-place mutation 的歧义。

### 4.3 多子任务排查信号

诊断时关键的判断方法:

```bash
# 1. 后端 task 数量(去重)
grep -E "task_id=call_00_\w+" packages/agent/logs/langgraph.log | grep -oE "task_id=call_00_\w+" | sort -u

# 2. 训练数据(已写入完整 input/output)
jq -s '.[] | {role, message_id, subagent_type}' \
  packages/agent/backend/.deer-flow/training-data/auto-collected/<thread_id>.jsonl
```

如果 backend 任务数量与 UI 卡片数量对不上 → frontend rendering bug。
如果 backend 任务全部 `status: completed` 而 UI 仍卡运行中 → SSE event handling bug(就是本次的 bug)。

### 4.4 关于「悬浮输入框」的视觉决策

明确选定**完全悬浮方案**(三选一):
- ❌ 贴底悬浮(底部圆角 + 仅向上阴影)
- ❌ 浮岛卡片(脱离底边 24-32px,阴影最强)
- ✅ **完全悬浮**(脱离底 16px,柔和向下阴影,消息可半透出现在卡片下方)

---

## 5. 未完成事项

### P0 — 阻塞

无。

### P1 — 人工验证

- [ ] **多子任务串行场景**回归测试:跑一次完整的 shoaling 分析(spawn code-executor → data-analyst → 可能还有 report-writer),确认每个卡片都能从「运行中」自动翻到「已完成」
- [ ] **子任务失败场景**:故意让某个子任务报错,确认卡片正确翻到红色「已失败」+ 错误信息
- [ ] **子任务超时场景**:罕见但需测试,确认 `task_timed_out` 事件被处理(走 `failed` 分支)
- [ ] **悬浮输入框视觉**:截图对比 04-30 vs 05-06,确认输入框真的「不突兀」了
- [ ] **占位文字垂直居中**:多分辨率(375 / 768 / 1280)下确认占位「今天我能为你做些什么?」始终垂直居中

### P2 — 可选优化

- [ ] **重构 `updateSubtask` 双路径**:把渲染期 in-place mutation 重构成纯派生(`useMemo` 从 `thread.messages` 计算 `tasks`),消除「依赖下一个 SSE 事件」的脆弱假设。预估改动:context.tsx 全重写 + message-list.tsx 不再调用 `updateSubtask`。
- [ ] **「执行 N 个子任务」React key 重复**:[message-list.tsx:216](../../packages/agent/frontend/src/components/workspace/messages/message-list.tsx) 用了 `key="subtask-count"` 常量字符串,如果一个 `assistant:subagent` group 有多个 AI 消息会触发 React 警告。修复:`key={"subtask-count-" + message.id}`。
- [ ] **`shine-border` 组件源码留存**:`src/components/ui/shine-border.tsx` 现在不再被任何文件 import,可以一并删除。需先 grep 确认。

### P3 — 架构

- [ ] dark mode token 升级(本轮明确不动,继承 04-30 的标记)

---

## 6. 建议接手路径

### 6.1 验证基础可用

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
pnpm dev
# 浏览器 http://localhost:3000
```

### 6.2 触发多子任务场景

最快的回归测试:

1. 用浏览器到 `localhost:3000`,新建对话
2. 上传 `demo-data/` 下的斑马鱼轨迹 txt 文件
3. 输入「分析斑马鱼群体行为」(或类似 prompt 触发 shoaling 范式)
4. 观察 lead-agent → code-executor → data-analyst 的串行执行
5. 验证:每个子任务卡片在 backend 完成后立即翻到「子任务已完成」(< 1s 延迟)

### 6.3 如果还是卡住

```bash
# 在浏览器 DevTools 中查看 SSE 事件流
# Network → 选中 langgraph stream 请求 → 观察 event:custom 数据中是否有 task_completed
# 如果有但前端没响应 → 说明 hooks.ts 改动没生效,检查 onCustomEvent 分支

# 或在 console 加日志
# 临时编辑 hooks.ts,在每个 task_* 分支加 console.log(e)
```

### 6.4 如果视觉不对

- 阴影太重 → 改 [globals.css](../../packages/agent/frontend/src/styles/globals.css) `--shadow-float` 透明度(当前 0.04 / 0.10 / 0.08)
- 脉动太亮 → 改 `pulse-soft` keyframes 里的 Lime 透明度(当前 0.18 → 0.32)
- 输入框还是「太大」 → 把 `min-h-14` 改成 `min-h-12`,但要同步把 `py-4` 改成 `py-3.5`(否则文字会被裁切)

---

## 7. 风险与注意事项

### ⚠️ 不要轻易动 `useUpdateSubtask` 的渲染期路径

[context.tsx:97-105](../../packages/agent/frontend/src/core/tasks/context.tsx) 的 in-place mutation **看起来**像 bug,但移除会导致**渲染循环 + React warning**(MessageList 重渲染时调用 `updateSubtask({status: "completed"})`,setTasks 触发 Provider 重渲染,Provider 重渲染让 MessageList 再渲染一次,无限循环)。本轮通过加「status 变化才 setTasks」的 guard 才安全引入了 setTasks。

### ⚠️ 后端 SSE 事件 schema 已假定

[hooks.ts](../../packages/agent/frontend/src/core/threads/hooks.ts) 的新 handler 假定:

```ts
{type: "task_completed", task_id: string, result?: string}
{type: "task_failed", task_id: string, error?: string}
{type: "task_timed_out", task_id: string, error?: string}
```

如果后端改了字段名(比如 `result` → `output`),前端会收事件但 `result` 是 undefined。**修改后端事件 payload 时记得同步前端**。

### ⚠️ `task_started` 事件目前仍未处理

CLAUDE.md 列了 `task_started` 是事件类型之一,但前端没处理。当前不阻塞(因为 `task_running` 隐含已开始),但未来如果后端依赖前端响应 `task_started` 做某些事会出问题。

### ✅ 视觉变更已自洽

`shadow-float` / `pulse-soft` / `border-brand/50` 都用了现有 token (`--brand`, Forest `rgba`),不引入新色相,不破坏「品牌色仅 2 处显式」的设计原则的精神(严格说现在 Lime 出现 3 处:hero CTA + input submit + 运行态描边,但状态指示器是必要例外)。

---

## 8. 下一位 Agent 的第一步建议

1. **读这份 handoff** ✅(你正在做)
2. **读 04-30 handoff** 了解视觉系统设计基线: [docs/handoffs/2026-04-30-frontend-aesthetic-upgrade-handoff.md](2026-04-30-frontend-aesthetic-upgrade-handoff.md)
3. **跑一次端到端测试**(见 §6.2),拍三张截图存档:
   - 新对话空态(看输入框尺寸 + 占位文字位置)
   - 子任务运行中(看 Lime 脉动是否柔和)
   - 子任务完成后(看是否正确翻到「子任务已完成」)
4. **如果一切正常**:把 P1 的待验证条目逐一勾选,开 PR/写测试报告
5. **如果发现新 bug**:优先看 P2 的「双路径架构脆弱性」是否是元凶,再决定是否触发 P2 重构

---

## 9. 附录:关键资源链接

| 资源 | 路径 |
|---|---|
| 上一轮交接(视觉基线) | [docs/handoffs/2026-04-30-frontend-aesthetic-upgrade-handoff.md](2026-04-30-frontend-aesthetic-upgrade-handoff.md) |
| 本轮 commit | `1837b0e2` (`git show 1837b0e2`) |
| 前端工作目录 | `/home/wangqiuyang/noldus-insight/packages/agent/frontend/` |
| 后端 task_tool 定义 | [packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py](../../packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py) |
| Subtask 类型 | [packages/agent/frontend/src/core/tasks/types.ts](../../packages/agent/frontend/src/core/tasks/types.ts) |
| 训练数据(诊断用) | `packages/agent/backend/.deer-flow/training-data/auto-collected/<thread_id>.jsonl` |
| 后端日志(诊断用) | `packages/agent/logs/langgraph.log` |
