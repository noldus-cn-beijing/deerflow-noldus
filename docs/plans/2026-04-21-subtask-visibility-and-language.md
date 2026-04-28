# Subtask 过程可见性 + 语言一致性 + 洞察深度 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 解决 fix4 E2E 暴露的三类问题：(1) 用户看不到 subagent 内部"专家工作过程"；(2) lead 仍会在正文写 `## Extracted Context` 英文 dump；(3) data-analyst 丢失了"按个体点名 + 反事实排除"的洞察深度。

**Architecture:**
- **前端**（C1-C3）：`Subtask.messages` 从单条 `latestMessage` 扩展为累积数组；`SubtaskCard` 展开态复用 `convertToSteps` 渲染统一 CoT 时间线；`HIDDEN_TOOL_CALL_NAMES` 拆成 lead 严格白名单 vs subtask 宽松白名单
- **Prompt 层**（C4-C5）：Lead prompt 加"用户语言锁定"+"不写结构化 dump"正面引导；三个 subagent prompt 统一加"匹配用户首条消息语言"规则
- **洞察深度**（C6）：`CodeExecutorHandoff` 明确保留 per_subject 原始指标；`data-analyst` prompt 增"离群个体逐一点名 + 反事实对比"要求
- **不动 DeerFlow 基建**：所有改动限定在 fork 受保护文件 + EthoInsight 包；不改 middleware 框架、消息路由、StreamBridge

**Tech Stack:** TypeScript / React 19（前端），Python 3.12 + Pydantic（后端 schema），LangGraph subagent prompts（纯字符串）

**分支 & 工作目录**:
- 分支 `dev`，从 `ec62f918` 继续
- `docs/e2e_tests/` 和这份 plan 文件本身不要动

**测试基线**:
- backend: `make test` → `1651 passed, 14 skipped`
- frontend: `pnpm check` + `pnpm build` 全绿

---

## Task 1: 扩展 `Subtask` 数据模型累积消息

**Files:**
- Modify: `packages/agent/frontend/src/core/tasks/types.ts`
- Modify: `packages/agent/frontend/src/core/tasks/context.tsx`
- Modify: `packages/agent/frontend/src/core/threads/hooks.ts` (around L263)

**Step 1: 读当前实现**

读 `core/tasks/types.ts`（12 行）、`core/tasks/context.tsx`（53 行）、`core/threads/hooks.ts` L256-280 了解 `task_running` 事件处理流。

**Step 2: 修改 `Subtask` 类型**

`packages/agent/frontend/src/core/tasks/types.ts` 完整重写为：

```typescript
import type { AIMessage } from "@langchain/langgraph-sdk";

export interface Subtask {
  id: string;
  status: "in_progress" | "completed" | "failed";
  subagent_type: string;
  description: string;
  prompt: string;
  result?: string;
  error?: string;

  /**
   * All AI messages emitted by the subagent, accumulated from each
   * `task_running` SSE event in the order they arrived. Replay-safe: entries
   * are deduplicated on `message.id`.
   */
  messages: AIMessage[];

  /**
   * Convenience pointer to messages[messages.length - 1]. Retained so the
   * collapsed card can keep rendering `explainLastToolCall(latestMessage)`
   * without iterating the array.
   */
  latestMessage?: AIMessage;
}
```

**Step 3: 修改 `useUpdateSubtask` 累积逻辑**

`packages/agent/frontend/src/core/tasks/context.tsx` 把 `updateSubtask` 重写为：

```typescript
export function useUpdateSubtask() {
  const { tasks, setTasks } = useSubtaskContext();
  const updateSubtask = useCallback(
    (update: Partial<Subtask> & { id: string }) => {
      const existing = tasks[update.id];
      const incoming = update.latestMessage;

      // If this is a task_running event with a new message, append it to
      // `messages` (dedup on message.id so React strict-mode or SSE retry
      // replays do not double-count).
      if (incoming) {
        const prevMessages = existing?.messages ?? [];
        const alreadyHas =
          incoming.id != null &&
          prevMessages.some((m) => m.id === incoming.id);
        const nextMessages = alreadyHas
          ? prevMessages
          : [...prevMessages, incoming];
        tasks[update.id] = {
          ...(existing ?? ({} as Subtask)),
          ...update,
          messages: nextMessages,
          latestMessage: nextMessages[nextMessages.length - 1],
        } as Subtask;
        setTasks({ ...tasks });
        return;
      }

      // Non-message update (status change, prompt init, result, error).
      tasks[update.id] = {
        ...(existing ?? ({ messages: [] } as unknown as Subtask)),
        ...update,
        messages: existing?.messages ?? [],
      } as Subtask;
      setTasks({ ...tasks });
    },
    [tasks, setTasks],
  );
  return updateSubtask;
}
```

（注意：原代码只在 `task.latestMessage` 存在时才 `setTasks`，这是 bug——status-only 更新不会刷新 UI。顺手修掉。）

**Step 4: Thread 切换时清理**

确认 `core/threads/hooks.ts` 里 thread 切换时 subtask state 是否已经重置。查找：

```bash
grep -n "setTasks" packages/agent/frontend/src/core/threads/hooks.ts
```

如果没有 reset 调用，在 `cachedThreadIdRef.current !== threadId` 分支里加 `setTasks({})`。如果已经有，跳过这步。

**Step 5: 前端类型检查**

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent/frontend
pnpm check
```

Expected: 类型错误只在 Task 2 要改的 `subtask-card.tsx` 和 Task 3 可能涉及的 `message-group.tsx`（因为这些文件可能依赖 `latestMessage` 非空）。其他文件应全绿。

**Step 6: Commit**

```bash
cd /home/qiuyangwang/noldus-insight
git add packages/agent/frontend/src/core/tasks/types.ts \
        packages/agent/frontend/src/core/tasks/context.tsx \
        packages/agent/frontend/src/core/threads/hooks.ts
git commit -m "$(cat <<'EOF'
C1: Subtask 数据模型累积保留全部内部消息

问题：SubtaskCard 展开后看不到 subagent 的完整工作过程。根因是
Subtask 只保留 latestMessage（单条滚动覆盖），后端 task_running 每
推送一条就丢掉前一条。日志显示一次 shoaling 分析后端发了 16 条消息，
前端只留了第 16 条。

改动：
- types.ts: 新增 messages: AIMessage[]，保留 latestMessage 作为指针
- context.tsx: updateSubtask 按 message.id 去重追加到 messages；
  同时修复原代码 status-only 更新不触发 setTasks 的 bug

数据流不变（仍是 task_running 自定义 event，非 thread messages
state），所以：
- 不改 DeerFlow 基建
- 不膨胀 lead 上下文
- 刷新后 subtask 过程消失（可接受——原本 latestMessage 也丢）

下一个 commit（C2）用这份 messages 渲染完整 CoT。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `SubtaskCard` 展开态渲染统一 CoT 时间线

**Files:**
- Modify: `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx`
- Modify: `packages/agent/frontend/src/components/workspace/messages/message-group.tsx`（导出 `convertToSteps` + `CoTStep` 类型，或抽到独立模块）

**Step 1: 确认 `convertToSteps` 可复用**

读 `packages/agent/frontend/src/components/workspace/messages/message-group.tsx` L471-518。
当前是模块内 `function convertToSteps(messages)` + 模块内 `CoTStep` 类型，没有 export。需要把它们导出给 subtask-card.tsx 用。

**Step 2: 决定过滤白名单分层（和 Task 3 协同）**

Task 3 会把 `HIDDEN_TOOL_CALL_NAMES` 拆成两个集合。本 Task 只需要让 `convertToSteps` **接受一个过滤集合参数**，默认用 lead 的集合（向后兼容）。签名改为：

```typescript
export function convertToSteps(
  messages: Message[],
  hiddenToolNames: Set<string> = LEAD_HIDDEN_TOOL_CALL_NAMES,
): CoTStep[] { ... }
```

（`LEAD_HIDDEN_TOOL_CALL_NAMES` 是 Task 3 重命名后的 `HIDDEN_TOOL_CALL_NAMES`；本 Task 中先保持旧名，Task 3 再重命名。）

**Step 3: 修改 `convertToSteps` 签名 + 导出**

在 `message-group.tsx` 把：

```typescript
const HIDDEN_TOOL_CALL_NAMES = new Set<string>([...]);

function convertToSteps(messages: Message[]): CoTStep[] {
  ...
  if (HIDDEN_TOOL_CALL_NAMES.has(tool_call.name)) {
    continue;
  }
  ...
}
```

改为：

```typescript
export const HIDDEN_TOOL_CALL_NAMES = new Set<string>([...]);  // 暂不重命名

export type CoTStep = CoTReasoningStep | CoTToolCallStep;
export type CoTReasoningStep = { ... };  // 已有，补 export
export type CoTToolCallStep = { ... };   // 已有，补 export

export function convertToSteps(
  messages: Message[],
  hiddenToolNames: Set<string> = HIDDEN_TOOL_CALL_NAMES,
): CoTStep[] {
  ...
  if (hiddenToolNames.has(tool_call.name)) {
    continue;
  }
  ...
}
```

**Step 4: 重写 `SubtaskCard` 展开态**

完整替换 `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx` L125-173 的 `<ChainOfThoughtContent>` 块。新结构：

```tsx
<ChainOfThoughtContent className="px-4 pb-4">
  {/* 任务描述：折叠 */}
  {task.prompt && (
    <ChainOfThoughtStep
      label={<span className="text-muted-foreground">{t.subtasks.taskDescription ?? "任务描述"}</span>}
    >
      <div className="pt-1">
        <Streamdown
          {...streamdownPluginsWithWordAnimation}
          components={{ a: CitationLink }}
        >
          {task.prompt}
        </Streamdown>
      </div>
    </ChainOfThoughtStep>
  )}

  {/* 专家工作过程：统一 CoT 时间线（复用 lead 的 convertToSteps） */}
  {task.messages.length > 0 && (
    <SubtaskCoTTimeline
      messages={task.messages}
      isLoading={task.status === "in_progress"}
    />
  )}

  {/* 任务结果：折叠；仅完成时显示 */}
  {task.status === "completed" && task.result && (
    <ChainOfThoughtStep
      label={<span className="text-muted-foreground">{t.subtasks.taskResult ?? "任务结果"}</span>}
      icon={<CheckCircleIcon className="size-3" />}
    >
      <div className="pt-1">
        <MarkdownContent
          content={task.result}
          isLoading={false}
          rehypePlugins={rehypePlugins}
        />
      </div>
    </ChainOfThoughtStep>
  )}

  {/* 失败态 */}
  {task.status === "failed" && (
    <ChainOfThoughtStep
      label={<div className="text-red-500">{task.error}</div>}
      icon={<XCircleIcon className="size-4 text-red-500" />}
    />
  )}
</ChainOfThoughtContent>
```

**Step 5: 新建 `SubtaskCoTTimeline` 子组件（同文件内）**

在 `subtask-card.tsx` 文件底部加：

```tsx
import {
  convertToSteps,
  HIDDEN_TOOL_CALL_NAMES,  // Task 3 会改名；本 Task 先引用旧名
} from "./message-group";

// Subtask 内部过滤集合：比 lead 主线更宽松（展示 ethoinsight 专家工具）
// 详细名单见 Task 3。此处临时用 lead 的集合作为占位，Task 3 替换。
const SUBTASK_HIDDEN_TOOL_CALL_NAMES = HIDDEN_TOOL_CALL_NAMES;  // placeholder

function SubtaskCoTTimeline({
  messages,
  isLoading,
}: {
  messages: AIMessage[];
  isLoading: boolean;
}) {
  const steps = useMemo(
    () => convertToSteps(messages, SUBTASK_HIDDEN_TOOL_CALL_NAMES),
    [messages],
  );
  // Minimal renderer: iterate steps, render reasoning as Reasoning block,
  // tool calls as ChainOfThoughtStep. Full styling reuses lead patterns;
  // copy the minimum subset of message-group.tsx render logic.
  // (Concrete rendering code omitted for brevity—follow the patterns in
  // message-group.tsx L80+ for each step.type branch.)
  return (
    <div className="flex flex-col gap-2">
      {steps.map((step) => (
        <CoTStepRenderer key={step.id} step={step} isLoading={isLoading} />
      ))}
    </div>
  );
}
```

**实施者注意：`CoTStepRenderer` 需要从 `message-group.tsx` 中抽取或复制，要求渲染三种形态：**

- **reasoning 步骤**：用 `@/components/ai-elements/reasoning` 的 `Reasoning` + `ReasoningTrigger` + `ReasoningContent`（参考 `message-group.tsx` 内的用法）
- **tool call 步骤**：`ChainOfThoughtStep` + icon（根据 `step.name` 映射），`result` 可用 `<CodeBlock>` 折叠展示
- **AI 消息正文**：当前 `convertToSteps` 不返回"正文文本"步骤。**这里需要扩展 `convertToSteps`**：新增 `CoTTextStep`（`type: "text", content: string`）捕获 `message.content` 非空部分

扩展 `convertToSteps` 让它同时捕获 AIMessage 正文为 `CoTTextStep`：

```typescript
// 在 message-group.tsx 的 convertToSteps 里，在 reasoning 之后加：
if (typeof message.content === "string" && message.content.trim()) {
  steps.push({
    id: `${message.id}-text`,
    messageId: message.id,
    type: "text",
    content: message.content,
  });
}
```

并扩展 `CoTStep` 类型：

```typescript
export type CoTTextStep = {
  id: string;
  messageId: string | undefined;
  type: "text";
  content: string;
};
export type CoTStep = CoTReasoningStep | CoTToolCallStep | CoTTextStep;
```

Lead 主线是否展示 `CoTTextStep` 取决于它自己的渲染器；如果 lead 本来就把 AIMessage 正文渲染成最终答案气泡，就在 `message-group.tsx` 的渲染处 `filter(s => s.type !== "text")` 跳过；或者更清晰：`convertToSteps` 增加第三参数 `includeText: boolean = false`。

**Step 6: 前端 check + build**

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent/frontend
pnpm check
SKIP_ENV_VALIDATION=1 pnpm build
```

Expected: 全绿。所有类型错误都在本 Task 改动的文件里解决。

**Step 7: 起服务手动验证**

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent
make dev
```

然后在浏览器访问 `localhost:2026`，跑一次 shoaling 5 文件的分析。展开 "执行 shoaling 数据分析" 卡片，确认：
- 任务描述可折叠
- 展开出现完整 CoT 时间线，至少 5-10 个步骤（16 条消息不全是 tool call，有的只是 reasoning）
- `parse_trajectories` 之类工具**暂时被隐藏**（Task 3 会放开）

**Step 8: Commit**

```bash
git add packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx \
        packages/agent/frontend/src/components/workspace/messages/message-group.tsx
git commit -m "$(cat <<'EOF'
C2: SubtaskCard 展开态渲染统一 CoT 时间线

展开后用户能看到 subagent 完整工作过程而不只是"读取 skill 文件"这一行。
复用 lead 主线的 convertToSteps + CoT 原语，结构为：
- 任务描述（折叠）
- 专家工作过程（CoT 时间线：reasoning / tool calls / AI 正文）
- 任务结果（完成后折叠）
- 失败态（红色错误）

扩展 convertToSteps 签名：
- 新增第二参数 hiddenToolNames（默认 lead 白名单）
- 新增 CoTTextStep 捕获 AIMessage content（lead 主线不渲染 text 步骤）

过滤集合暂时复用 lead 的 HIDDEN_TOOL_CALL_NAMES（placeholder）；
C3 会拆成 lead 严格 vs subtask 宽松两套。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 过滤白名单作用域分层（lead 严格 / subtask 宽松）

**Files:**
- Modify: `packages/agent/frontend/src/components/workspace/messages/message-group.tsx` L452-469
- Modify: `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx`（替换 placeholder）

**Step 1: 重命名 lead 集合 + 新建 subtask 集合**

`message-group.tsx` 把：

```typescript
export const HIDDEN_TOOL_CALL_NAMES = new Set<string>([
  "read_file", "write_file", "str_replace", "bash", "ls", "glob", "grep",
  "get_analysis_template", "parse_trajectories", "compute_metrics",
  "run_statistics", "generate_charts", "assess_and_handoff",
]);
```

拆成：

```typescript
/**
 * Lead-agent timeline: hide low-level I/O plumbing AND ethoinsight
 * fine-grained tools. Lead shouldn't be calling the latter directly;
 * if it does it's internal shuffling of handoff files.
 */
export const LEAD_HIDDEN_TOOL_CALL_NAMES = new Set<string>([
  "read_file", "write_file", "str_replace", "bash", "ls", "glob", "grep",
  "get_analysis_template", "parse_trajectories", "compute_metrics",
  "run_statistics", "generate_charts", "assess_and_handoff",
]);

/**
 * Subtask timeline (SubtaskCard expanded state): only hide pure
 * filesystem noise. KEEP ethoinsight domain tools visible — those ARE
 * the "expert at work" steps users want to see.
 */
export const SUBTASK_HIDDEN_TOOL_CALL_NAMES = new Set<string>([
  "read_file", "write_file", "str_replace", "ls", "glob", "grep",
  // bash stays visible in subtask: often diagnostic commands worth showing
  // ethoinsight tools (parse_trajectories etc.) NOT in this set
]);

// Backward-compat alias for C2 placeholder (remove in same commit)
/** @deprecated use LEAD_HIDDEN_TOOL_CALL_NAMES */
export const HIDDEN_TOOL_CALL_NAMES = LEAD_HIDDEN_TOOL_CALL_NAMES;
```

实际提交时删除 `@deprecated` 别名，只保留两个新名字——`subtask-card.tsx` 会同时更新引用。

**Step 2: 更新 `convertToSteps` 默认参数**

```typescript
export function convertToSteps(
  messages: Message[],
  hiddenToolNames: Set<string> = LEAD_HIDDEN_TOOL_CALL_NAMES,  // 改名
  includeText: boolean = false,
): CoTStep[] { ... }
```

**Step 3: 更新 `subtask-card.tsx` 的 import**

```typescript
import {
  convertToSteps,
  SUBTASK_HIDDEN_TOOL_CALL_NAMES,  // 替换占位
} from "./message-group";

// 删掉原 placeholder：
// const SUBTASK_HIDDEN_TOOL_CALL_NAMES = HIDDEN_TOOL_CALL_NAMES;
```

并在 `SubtaskCoTTimeline` 里：

```typescript
const steps = useMemo(
  () => convertToSteps(messages, SUBTASK_HIDDEN_TOOL_CALL_NAMES, true),  // includeText=true
  [messages],
);
```

**Step 4: 前端 check + build**

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent/frontend
pnpm check
SKIP_ENV_VALIDATION=1 pnpm build
```

Expected: 全绿。

**Step 5: 手动验证**

重启服务跑同一个 shoaling，展开 subtask 卡片确认：
- 能看到 `parse_trajectories` / `compute_metrics` / `run_statistics` / `generate_charts` / `assess_and_handoff` 的调用序列
- 看不到 `read_file`/`write_file`/`ls`（纯文件噪声）
- Lead 主时间线仍然隐藏以上全部

**Step 6: Commit**

```bash
git add packages/agent/frontend/src/components/workspace/messages/message-group.tsx \
        packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx
git commit -m "$(cat <<'EOF'
C3: 拆分 lead vs subtask 的工具过滤白名单

原单一 HIDDEN_TOOL_CALL_NAMES 对 subtask 过度隐藏，导致展开态看不到
ethoinsight 专家工具（parse_trajectories / compute_metrics 等）的工作
过程，却保留了琐碎的"读取 skill 文件"。

改为两组：
- LEAD_HIDDEN_TOOL_CALL_NAMES: 严格集合，继承原语义（用于 lead 主线）
- SUBTASK_HIDDEN_TOOL_CALL_NAMES: 只藏纯文件 I/O（read_file/write_file
  /str_replace/ls/glob/grep），保留 ethoinsight 域工具和 bash

subtask-card.tsx SubtaskCoTTimeline 改用 subtask 集合 + includeText=true，
让 AI 正文片段（"发现 Subject 3 为离群个体..."之类）也进入时间线。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Lead prompt — 用户语言锁定 + 不写结构化 dump

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- Add test: `packages/agent/backend/tests/test_lead_prompt_language_and_style.py`

**Step 1: 读现状**

```bash
grep -n "语言\|language\|dump\|## Extracted\|## 提取" \
  packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
```

确认当前 prompt 里没有这两类约束。

**Step 2: 写失败测试**

`packages/agent/backend/tests/test_lead_prompt_language_and_style.py`：

```python
"""Lead prompt contract: language lock + no structured dump output.

These are prompt-level contract tests; they assert that key instruction
phrases are present in the generated system prompt, not that the model
behaves a certain way (that requires E2E).
"""
from __future__ import annotations

from deerflow.agents.lead_agent.prompt import (
    SYSTEM_PROMPT_TEMPLATE,
    _build_subagent_section,
)


def test_language_lock_rule_present_in_system_prompt():
    assert "用户语言" in SYSTEM_PROMPT_TEMPLATE or "user language" in SYSTEM_PROMPT_TEMPLATE
    # Positive phrasing only (GLM-5.1 rule): no "禁止/不要 X" directives
    assert "用和用户相同的语言回答" in SYSTEM_PROMPT_TEMPLATE


def test_no_dump_style_rule_present():
    # Positive phrasing: tell lead HOW to reply, not what to avoid.
    # E.g. "用自然段落和项目符号回答" instead of "不要写 ## Extracted Context 格式"
    assert "自然段落" in SYSTEM_PROMPT_TEMPLATE
    assert "项目符号" in SYSTEM_PROMPT_TEMPLATE or "bullet" in SYSTEM_PROMPT_TEMPLATE


def test_subagent_section_still_intact():
    # Guard: language rule changes should not break existing subagent block.
    section = _build_subagent_section(max_concurrent=3)
    assert "data-analyst" in section
    assert "code-executor" in section
    assert "report-writer" in section
```

**Step 3: 运行测试确认失败**

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent/backend
source .venv/bin/activate
PYTHONPATH=. pytest tests/test_lead_prompt_language_and_style.py -v
```

Expected: 前两个 FAIL（phrase 不在 prompt 里），第三个 PASS。

**Step 4: 在 prompt.py 加"过程透明 / 回答风格"小节**

在 `SYSTEM_PROMPT_TEMPLATE` 里找到已有的 `<过程透明原则>` / `<分析结果呈现模板>` 小节附近（commit 5 加的），插入新小节：

```python
# 在 SYSTEM_PROMPT_TEMPLATE 里，紧接 <过程透明原则> 之后加：
<用户语言锁定>
检测用户首条消息的主要语言（中文 / 英文 / 其他），之后整个会话**用和
用户相同的语言回答**，包括：
- 你自己的 AIMessage 正文
- 调用 ask_clarification 时的 question 和 options
- 派 subagent 时 prompt 里给它的指示

派 subagent 时在 prompt 开头明确声明用户语言，例如：
"用户使用中文交流。你的回答、write_file 内容、handoff 摘要都必须使用中文。"

这让下游 subagent 与用户保持一致，避免中英文交错。
</用户语言锁定>

<回答风格>
对用户的每一条回答**用自然段落和项目符号**组织，不要用以下结构化形式：
- `## Extracted Context` / `## 提取的关键上下文` / `### Task` 这类"给自己看的状态 dump"
- 大段的键值对列表（`**Task**:` / `**Status**:` / `**Next**:`）

如果你需要内部整理思路，放到 `<think>` 标签里——ThinkTagMiddleware 会
自动搬到 reasoning，用户默认不可见。最终对用户说的话一律用自然语言。
</回答风格>
```

**Step 5: 运行测试确认通过**

```bash
PYTHONPATH=. pytest tests/test_lead_prompt_language_and_style.py -v
```

Expected: 3 PASS.

**Step 6: 全量回归**

```bash
make test
```

Expected: `1654 passed, 14 skipped`（原 1651 + 本次新增 3）。如数字不对，检查是否误伤 prompt 相关的其他测试。

**Step 7: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py \
        packages/agent/backend/tests/test_lead_prompt_language_and_style.py
git commit -m "$(cat <<'EOF'
C4: Lead prompt 加用户语言锁定 + 回答风格正面引导

Phase B 废弃了 InternalNotesMiddleware 的 heading 白名单（治标不治本），
残留风险是 lead 仍可能在 AIMessage 正文主动写 ## Extracted Context
风格的 dump。本 commit 用正面 prompt 引导封住这条路。

新增两小节：
1. <用户语言锁定>：检测用户首条消息语言，整轮用同语言回答；派 subagent
   时在其 prompt 开头声明用户语言。解决 fix4 里中英文混用问题
   （##Extracted Context: Zebrafish Shoaling Analysis 这类）
2. <回答风格>：用自然段落 + 项目符号回答；需要内部整理用 <think> 标签
   由 ThinkTagMiddleware 自动搬到 reasoning 字段，前端折叠不打扰用户

遵守用户已确认的 GLM-5.1 原则：只用正面指令，不用"禁止 X"。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Subagent prompts — 匹配用户语言

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`
- Add test: `packages/agent/backend/tests/test_subagent_language_rule.py`

**Step 1: 写失败测试**

`packages/agent/backend/tests/test_subagent_language_rule.py`：

```python
"""Each builtin subagent prompt must contain a user-language-matching rule
so handoff text, write_file content, and final messages do not leak
English into a Chinese conversation (or vice versa).
"""
from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG
from deerflow.subagents.builtins.report_writer import REPORT_WRITER_CONFIG


def _assert_language_rule(prompt: str, subagent_name: str):
    # Positive phrasing: match user language
    assert "用户语言" in prompt, (
        f"{subagent_name} prompt missing user-language rule"
    )
    assert "与用户语言一致" in prompt or "匹配用户语言" in prompt, (
        f"{subagent_name} prompt missing explicit language-matching instruction"
    )


def test_code_executor_has_language_rule():
    _assert_language_rule(CODE_EXECUTOR_CONFIG.system_prompt, "code-executor")


def test_data_analyst_has_language_rule():
    _assert_language_rule(DATA_ANALYST_CONFIG.system_prompt, "data-analyst")


def test_report_writer_has_language_rule():
    _assert_language_rule(REPORT_WRITER_CONFIG.system_prompt, "report-writer")
```

**Step 2: 运行测试确认失败**

```bash
PYTHONPATH=. pytest tests/test_subagent_language_rule.py -v
```

Expected: 3 FAIL.

**Step 3: 改三个 subagent prompt**

在每个 subagent 的 `system_prompt` 靠顶部（`<contract>` 块之前）加一致的小节：

```python
# data_analyst.py 示例，其他两个同样处理
system_prompt="""你是行为数据分析与洞察专家...

<语言>
**输出语言必须与用户语言一致**：
- lead agent 派发任务时，会在 prompt 开头声明用户使用的语言
- 如果 lead 未明确声明，从任务描述中推断：中文任务用中文、英文任务用英文
- 所有输出（最终消息、write_file 内容、handoff_*.json 里的自由文本字段）
  都用同一种语言
- 统计术语、变量名、文件路径可以保留英文（它们是专有名词）
</语言>

<contract>
...
""",
```

三个文件都加同一段（完全相同的文案，未来维护好统一）。

**Step 4: 测试通过**

```bash
PYTHONPATH=. pytest tests/test_subagent_language_rule.py -v
```

Expected: 3 PASS.

**Step 5: 全量回归 + subagent 契约测试**

```bash
make test
```

Expected: `1657 passed, 14 skipped`（原 1654 + 本次 3）。顺手确认 `test_subagent_contracts.py` 14 个契约测试全绿（我们没改 handoff schema，不应该动）。

**Step 6: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py \
        packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py \
        packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py \
        packages/agent/backend/tests/test_subagent_language_rule.py
git commit -m "$(cat <<'EOF'
C5: Subagent prompt 加用户语言匹配规则

配合 C4 lead 侧的"派任务时声明用户语言"，三个 EthoInsight subagent
（code-executor / data-analyst / report-writer）在 prompt 里加
<语言> 小节，约束其：
- 从 lead 传入的任务描述推断用户语言
- 最终消息 / write_file 内容 / handoff JSON 自由文本字段都用同一语言
- 统计术语 / 变量名 / 路径保持英文（专有名词例外）

解决 fix4 里 data-analyst handoff 英文 Extracted Context 继续污染上下文
的问题。和 C4 一起构成"全链路语言一致"。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: data-analyst 洞察深度 — per-subject 数据 + 反事实分析

**Files:**
- Read: `packages/ethoinsight/ethoinsight/templates/tool.py`（确认 `assess_and_handoff_tool` 写进 `code_summary.json` 时是否保留 per_subject）
- Read: 最新 E2E 产物 `/mnt/user-data/workspace/handoff_code_executor.json`（可 `find . -name "code_summary.json" -o -name "handoff_code_executor.json"` 定位本地样本）
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`（若现 schema 不含 per_subject，补字段）
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`（prompt 增加"按个体点名 + 反事实"要求）
- Modify（可能）: `packages/ethoinsight/ethoinsight/templates/tool.py`（若产物缺 per_subject，补写）
- Add test: `packages/agent/backend/tests/test_data_analyst_insight_contract.py`
- Add test: `packages/ethoinsight/tests/test_code_summary_per_subject.py`（如果要改 ethoinsight 产物）

**Step 1: 诊断 — `code_summary.json` 是否有 per_subject？**

读一份最近的 E2E 产物确认数据是否齐全：

```bash
# 本地 thread dir
ls packages/agent/backend/.deer-flow/threads/*/user-data/workspace/code_summary.json 2>/dev/null | tail -1

# 取最新一份查看结构
find packages/agent/backend/.deer-flow/threads -name "code_summary.json" -printf '%T@ %p\n' 2>/dev/null \
  | sort -n | tail -1 | cut -d' ' -f2 | xargs -I {} python -c "
import json; d = json.load(open('{}'))
print('top-level keys:', list(d.keys()))
print('metrics_summary groups:', list(d.get('metrics_summary', {}).keys()))
print('has per_subject:', 'per_subject' in d)
if 'per_subject' in d:
    print('per_subject sample:', next(iter(d['per_subject'].items()))[1])
"
```

**决策树：**

- **若 `per_subject` 存在且数据完整** → data-analyst 拿得到，问题在 prompt 引导 → 跳到 Step 3
- **若 `per_subject` 不存在或数据残缺** → 需要先补 ethoinsight 产物 → 做 Step 2

**Step 2（条件执行）: 补 ethoinsight 产物的 per_subject**

如果 Step 1 显示缺字段，定位写出逻辑：

```bash
grep -rn "code_summary" packages/ethoinsight/ethoinsight/
grep -n "per_subject" packages/ethoinsight/ethoinsight/
```

找到 `assess_and_handoff_tool` 产出 `code_summary.json` 的函数。按 TDD 加测试：

`packages/ethoinsight/tests/test_code_summary_per_subject.py`：

```python
def test_code_summary_includes_per_subject_metrics():
    """code_summary.json must expose per-subject raw metric values so
    data-analyst can identify outlier subjects by name and do
    leave-one-out counterfactual analysis (e.g., 'excluding Subject 3,
    NND drops from 48.2 → 37.2 mm')."""
    from ethoinsight.templates.tool import assess_and_handoff_tool
    # ... build minimal shoaling input with 5 subjects
    # Assert the returned JSON has per_subject[subject_name][metric] = value
    ...
```

然后改 `assess_and_handoff_tool` 让它把 per_subject 数据从 metrics.csv 读出来嵌入 JSON。**如果 Step 1 已经发现 per_subject 存在，本 Step 不做。**

**Step 3: 扩展 `DataAnalystHandoff` schema（必做）**

`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py` 给 `DataAnalystHandoff` 加字段：

```python
class OutlierFinding(BaseModel):
    """One flagged outlier subject with counterfactual support."""

    model_config = ConfigDict(extra="allow")

    subject: str = Field(description="Subject identifier, e.g. 'Subject 3'")
    metric: str = Field(description="Metric on which subject is an outlier")
    value: float = Field(description="Raw value on that metric")
    deviation: str = Field(
        description="Qualitative description, e.g. '2x group median', 'CV=35%'",
    )
    counterfactual: str | None = Field(
        default=None,
        description="Group stats if this subject is excluded, e.g. 'NND drops 48.2 → 37.2 mm'",
    )


class DataAnalystHandoff(BaseModel):
    ...
    outlier_findings: list[OutlierFinding] = Field(
        default_factory=list,
        description="Per-subject outlier diagnostics with leave-one-out context.",
    )
```

**Step 4: 写契约测试**

`packages/agent/backend/tests/test_data_analyst_insight_contract.py`：

```python
from deerflow.subagents.handoff_schemas import DataAnalystHandoff, OutlierFinding


def test_outlier_finding_schema():
    f = OutlierFinding(
        subject="Subject 3",
        metric="mean_nnd",
        value=70.02,
        deviation="~2x group median",
        counterfactual="NND drops 48.2 → 37.2 mm if excluded",
    )
    assert f.subject == "Subject 3"
    assert f.counterfactual is not None


def test_data_analyst_handoff_accepts_outlier_findings():
    h = DataAnalystHandoff(
        status="completed",
        analysis_summary_path="/mnt/x.md",
        outlier_findings=[
            OutlierFinding(
                subject="Subject 3", metric="mean_nnd", value=70.02,
                deviation="2x median",
            ),
        ],
    )
    assert len(h.outlier_findings) == 1


def test_data_analyst_prompt_requires_per_subject_review():
    from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG
    p = DATA_ANALYST_CONFIG.system_prompt
    # Prompt must explicitly direct per-subject outlier inspection
    assert "逐一" in p or "按受试者" in p or "per_subject" in p or "per-subject" in p
    # And counterfactual / leave-one-out analysis
    assert "排除" in p or "leave-one-out" in p or "反事实" in p
```

**Step 5: 运行测试确认失败**

```bash
PYTHONPATH=. pytest tests/test_data_analyst_insight_contract.py -v
```

Expected: 前两个 PASS（schema 已加），第三个 FAIL（prompt 还没加这段）。

**Step 6: 改 data-analyst prompt**

在 `data_analyst.py` 的 `<workflow>` 里新增第 6.5 步（"数据洞察"之前）+ 扩展 `<principles>`：

```python
# system_prompt 修改示例 —— 在 <workflow> 6. 之后、7. 之前加：

6. **按受试者逐一检查**（关键！）：
   - 从 code_summary.json 的 `per_subject` 字段拿各受试者原始指标值
   - 对每个指标，识别哪条受试者偏离组均值 ≥ 1.5 SD 或偏离组中位数 ≥ 2 倍
   - 对发现的离群个体，计算**反事实统计**：如果排除这条受试者，该组 mean/std
     变成多少？组间差异是否还存在？
   - 在 handoff JSON 的 outlier_findings 字段记录：subject / metric / value /
     deviation / counterfactual

   示例：
     Subject 3 的 mean_nnd = 70.02（组均值 48.16，组内 std=18.95）
     排除 Subject 3 后，treatment 组 mean_nnd 降至 37.23 mm，
     与 control（37.97）差异不足 2%，提示组间差异几乎完全由该个体驱动。
```

并在 `<principles>` 加：

```
- **具名诊断**：发现问题时必须点名具体 subject（"Subject 3"），不要只说
  "存在至少一个异常个体"
- **反事实支撑**：对每个指出的离群个体，给出"排除后组间差异变化"的量化
  支撑，便于研究员判断该发现是否稳健
```

**Step 7: 测试全绿**

```bash
PYTHONPATH=. pytest tests/test_data_analyst_insight_contract.py -v
```

Expected: 3 PASS.

```bash
make test
```

Expected: `1660 passed, 14 skipped`（原 1657 + 本次 3）。

**Step 8: E2E 回归验证**

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent
make dev
```

重跑一次 shoaling 5 文件，确认 data-analyst 输出里：
- 点名了 `Subject 3`（而不是"某个异常个体"）
- 给出了 `70.02 vs 48.16` 具体数值
- 做了"排除后降至 37.23 mm"类反事实
- handoff_data_analyst.json 里 `outlier_findings` 数组非空

**Step 9: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py \
        packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py \
        packages/agent/backend/tests/test_data_analyst_insight_contract.py
# 如果 Step 2 条件执行过，也要加：
# packages/ethoinsight/ethoinsight/templates/tool.py \
# packages/ethoinsight/tests/test_code_summary_per_subject.py
git commit -m "$(cat <<'EOF'
C6: data-analyst 洞察深度 — 按受试者点名 + 反事实排除

fix4 vs fix3 对照发现洞察深度退化：
- fix3 明确点出 Subject 3 的 NND=70.02，给出排除后组均值降至 37.23 mm
- fix4 只笼统说"存在至少一个运动量异常的离群个体"

改动：
- handoff_schemas.py: DataAnalystHandoff 新增 outlier_findings: list[OutlierFinding]
  字段，结构化承载（subject, metric, value, deviation, counterfactual）
- data_analyst.py prompt:
  - <workflow> 加 Step 6 "按受试者逐一检查 + 反事实计算"
  - <principles> 加 "具名诊断" + "反事实支撑"

[如果触发 Step 2：] 同时补齐 ethoinsight assess_and_handoff_tool 输出的
per_subject 字段，确保 data-analyst 拿得到原始单个体数据。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: 综合 E2E 验证 + 文档归档

**Step 1: 跑完整 make test 回归**

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent/backend
source .venv/bin/activate
make test
make lint
```

Expected:
- `1660 passed, 14 skipped`（C4+3 / C5+3 / C6+3 累计 +9）
- lint 只报 prompt.py / data_analyst.py 的 pre-existing 问题（F841 / E501），不引入新 violation

**Step 2: 跑 ethoinsight 回归**

```bash
cd /home/qiuyangwang/noldus-insight/packages/ethoinsight
python -m pytest tests/ -q
```

Expected: `130 passed, 3 skipped`（如果未触发 Task 6 Step 2，数字不变；触发了则按新增测试数增加）。

**Step 3: 前端回归**

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent/frontend
pnpm check
SKIP_ENV_VALIDATION=1 pnpm build
```

Expected: 全绿。

**Step 4: 端到端跑一遍 + 写验收清单**

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent
make dev
```

跑 shoaling 5 文件场景，对照清单：

| # | 验收点 | 证据位置 |
|---|---|---|
| 1 | Subtask 卡片展开后显示完整 CoT 时间线（reasoning + tool + text） | 前端界面 |
| 2 | `parse_trajectories` / `compute_metrics` 等可见；`read_file` 不可见 | 前端界面 |
| 3 | Lead 主时间线仍然简洁，没有 ethoinsight 细粒度工具 | 前端界面 |
| 4 | 整轮对话语言统一（中文输入 → 中文输出，无英文 `## Extracted Context` 等 dump） | 前端导出 |
| 5 | data-analyst 点名 Subject 3 + 给出反事实数据 | 前端界面 + handoff JSON |
| 6 | 无后端 error / traceback / 额外 429 | `logs/langgraph.log` |

**Step 5: 导出 E2E → 归档**

把新的 E2E 对话导出到 `docs/e2e_tests/斑马鱼鱼群行为轨迹数据分析-fix5.md`（用户手动做，不在 commit 范围）。

**Step 6: 写交接文档**

`docs/handoffs/2026-04-21-subtask-visibility-handoff.md`：简要 6 个 commit 清单 + E2E 验收结果 + 风险与注意事项。参考上一版 `2026-04-20-phase-ab-handoff.md` 风格。

**Step 7: Commit 交接文档**

```bash
git add docs/handoffs/2026-04-21-subtask-visibility-handoff.md
git commit -m "docs: 4/21 subtask 可见性 + 语言一致性 + 洞察深度 交接"
```

---

## Task 8: 最后一步 — 决定是否 push

**不要自己 push。** 问用户：
- 本轮 7 commit + Phase A/B 15 commit + pipeline 1-6a 的 7 commit，总共 29 个 commit 等待 push
- 是否 `git push origin dev`

---

## 风险与注意事项

### 容易跑偏

- **不要动 DeerFlow 基建**：不改 middleware 框架、StreamBridge、消息路由、agent graph 结构
- **不要回退到白名单**：C4 用正面 prompt 引导，不要重新引入 heading 白名单 middleware
- **不要只让 data-analyst 改 prompt 而不改 schema**：C6 的 `outlier_findings` 结构化字段是让下游 report-writer/前端可机器读的关键
- **前端 `convertToSteps` 签名变更有连锁影响**：改它之前 `grep -rn "convertToSteps" packages/agent/frontend/src` 确认所有调用点都更新

### 容易误判

- `docs/e2e_tests/` 和本文件 `docs/plans/2026-04-21-...md` **都是 untracked / 不要 git add**
- `extensions_config.json` 是 gitignored
- Task 6 Step 1 诊断输出如果显示 per_subject 已存在，**不要**改 ethoinsight——省掉一个 commit 的工作量
- Sonnet 不用 `<think>` 标签，C4 的 `<think>` 引导对当前模型**影响有限**，但对未来模型或 vLLM/Qwen 有效

### Prompt 工程效果需要 E2E 验证

C4 / C5 / C6 都是 prompt 改动，单测只验证 phrase 在 prompt 里，不能保证模型真的听。**必须** E2E 跑一次确认行为。如果某条规则失效：
- 先确认 prompt phrase 确实渲染到了系统消息（LangGraph debug 输出 / lang​graph.log）
- 调整用词强度（中文 → 更 imperative；英文 → MUST vs should）
- 仍不生效时考虑把约束拆到 skill 示例里（示例胜过规则）

### 测试基线数字

| 阶段 | backend 预期 |
|---|---|
| 起点 | 1651 passed |
| C4 后 | 1654 passed |
| C5 后 | 1657 passed |
| C6 后 | 1660 passed |

如果 backend 数字不对，先看是否误伤已有测试（prompt 里某些 phrase 被其他测试 assert）。

### 执行顺序

C1 → C2 → C3 严格按序（C2 placeholder 依赖 C1，C3 替换 C2 placeholder）；
C4 / C5 互相独立；
C6 与 C1-C5 独立；

并行空间有限，按单线推进更安全。
