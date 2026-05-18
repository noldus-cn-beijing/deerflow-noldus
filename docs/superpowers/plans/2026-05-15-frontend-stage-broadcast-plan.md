# G4 方案 C: 前端 tool_call 事件自动播报 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把"每个工具调用前向用户播报当前在做什么"从 lead prompt 自觉约束转移到**前端 UI 自动渲染**。dogfood 测试证明 prompt 层的 FIRST-TOKEN emoji 规则在 deepseek 多步推理中不稳定（0→1 emoji，未达 ≥4），且会让 lead 输出生硬。本 plan 用前端 UI 在收到 tool_call event 时自动展示业务语义的状态条（"🧮 正在计算 N 个高架十字迷宫指标..."），从根本上把"是否播报"从 LLM 行为问题转化为 UI 渲染问题。

**Architecture:** 前端已有 `SubtaskCard`（处理 `task(subagent_type=...)` 的 subagent 派遣）和 `ToolCall`（处理 bash / read_file / write_file 等通用工具调用）。本 plan 在这两个组件上加 **业务语义播报层**：
- 给 `SubtaskCard` 卡片标题加 subagent_type → 中文播报模板的 mapping（覆盖 code-executor / data-analyst / report-writer / knowledge-assistant）
- 给 `ToolCall` 的 bash 分支加 EthoInsight CLI command pattern → 中文播报的识别（覆盖 dump_headers / catalog.resolve）
- 给 `ask_clarification` 拦截展示加状态条
- 全部新文案进 i18n（`zh-CN.ts` + `en-US.ts` + `types.ts`），不硬编码

不改后端任何代码，纯前端改动。

**Tech Stack:** Next.js 16 / React 19 / TypeScript 5.8 / Tailwind 4 / pnpm 10.26.2，无新增依赖。

**Context links（实现时必读）：**
- frontend 架构总览：`packages/agent/frontend/CLAUDE.md`（强调："`ui/` 和 `ai-elements/` 是从 registries 生成的——不要手工编辑这些"——本 plan 完全不动这两个目录）
- 现有 SubtaskCard：`packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx`（256 行，本 plan 改 ~15 行）
- 现有 ToolCall：`packages/agent/frontend/src/components/workspace/messages/message-group.tsx:182-...`（本 plan 改 bash 分支约 ~30 行）
- 现有 i18n 类型：`packages/agent/frontend/src/core/i18n/locales/types.ts:189-208`（既有 toolCalls 节点）
- 现有 i18n zh-CN：`packages/agent/frontend/src/core/i18n/locales/zh-CN.ts:246-265`
- 现有 i18n en-US：`packages/agent/frontend/src/core/i18n/locales/en-US.ts:256-265`
- Subtask 类型：`packages/agent/frontend/src/core/tasks/types.ts`（含 `subagent_type` 字段）
- explainLastToolCall：`packages/agent/frontend/src/core/tools/utils.ts`（29 行，本 plan 改 ~10 行加 ethoinsight CLI 识别）
- 项目约束：`CLAUDE.md`（中文 commit）+ frontend CLAUDE.md（pnpm check 必跑、ui/ai-elements 不动）

**Scope（明确不做）：**
- **不改后端** `lead_agent/prompt.py`——FIRST-TOKEN 回退由独立 plan 处理（`docs/superpowers/plans/2026-05-15-revert-first-token-rule-plan.md`）
- **不动 `ui/` 和 `ai-elements/`**（CLAUDE.md 明令禁改）
- **不写单元测试**（frontend "No test framework is configured"），改用 manual QA 步骤
- **不改 backend i18n / prompt 中的播报模板表**（lead prompt 第 442 行的"过程透明原则"段表格保留作为推荐参考，不强制）
- **不改 SSE 事件流 / LangGraph SDK 用法**——只在现有渲染层加业务语义识别
- **不实现"播报历史记录可回放"功能**——状态条是 ephemeral 的，刷新页面不要求恢复

**前置假设（执行前用 git log -1 验证）：**
- 当前在 `dev` 分支
- FIRST-TOKEN 回退 plan 已经执行完毕（依赖 prompt 那边已经回退；不强依赖，但顺序建议先回退再做本 plan）
- backend 服务能起来供 manual QA 用
- `pnpm install` 已跑过、`pnpm check` 在基线全绿

---

## File Structure

**修改 6 个文件**：

| 文件 | 改动 |
|---|---|
| `packages/agent/frontend/src/core/i18n/locales/types.ts` | 给 `toolCalls` 类型加 `stageBroadcast: { dispatchSubagent / parseHeaders / resolveCatalog / askClarification }` 子节点 |
| `packages/agent/frontend/src/core/i18n/locales/zh-CN.ts` | 加上述节点的中文翻译 |
| `packages/agent/frontend/src/core/i18n/locales/en-US.ts` | 加上述节点的英文翻译 |
| `packages/agent/frontend/src/core/tools/utils.ts` | `explainToolCall` 增加 `task` (subagent_type 映射) + `bash` (识别 EthoInsight CLI) 分支 |
| `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx` | 卡片标题改用 `getStageBroadcastForSubagent(subagent_type, t)` 函数替代纯 `task.description` |
| `packages/agent/frontend/src/components/workspace/messages/message-group.tsx` | ToolCall bash 分支加 EthoInsight CLI command 识别 + 用 `t.toolCalls.stageBroadcast.*` 替代当前简单 `description` |

**新建 1 个文件**：
- `packages/agent/frontend/src/core/tools/stage-broadcast.ts`（~60 行）— 新 module，导出：
  - `getStageBroadcastForSubagent(subagentType: string, t: Translations): string` — subagent_type → 状态文案
  - `detectEthoinsightCli(command: string): "parse" | "catalog" | null` — bash command pattern 识别
  - `getStageBroadcastForBash(command: string, t: Translations): string | null` — 若识别为 EthoInsight CLI 返回状态文案，否则 null

**0 个删除文件**

---

### Task 1: 加 i18n 类型定义和翻译

**Files:**
- Modify: `packages/agent/frontend/src/core/i18n/locales/types.ts:189-208` (toolCalls 节点)
- Modify: `packages/agent/frontend/src/core/i18n/locales/zh-CN.ts:246-265`
- Modify: `packages/agent/frontend/src/core/i18n/locales/en-US.ts:256-265`

**为什么先做 i18n**：所有后续步骤都依赖新增的字符串模板。先把类型和翻译落地，TypeScript 类型系统会引导后续 task 不写错 key。

- [ ] **Step 1: 改 `types.ts` 给 `toolCalls` 加 `stageBroadcast` 子节点**

用 Edit 工具修改 `packages/agent/frontend/src/core/i18n/locales/types.ts`。

old_string（line 189-208 之间的精确锚点，包含 `useTool` 结尾以确保唯一匹配）：

```typescript
  // Tool calls
  toolCalls: {
    moreSteps: (count: number) => string;
    lessSteps: string;
    executeCommand: string;
    presentFiles: string;
    needYourHelp: string;
    useTool: (toolName: string) => string;
    searchForRelatedInfo: string;
    searchForRelatedImages: string;
    searchFor: (query: string) => string;
    searchForRelatedImagesFor: (query: string) => string;
    searchOnWebFor: (query: string) => string;
    viewWebPage: string;
    listFolder: string;
    readFile: string;
    writeFile: string;
```

new_string：

```typescript
  // Tool calls
  toolCalls: {
    moreSteps: (count: number) => string;
    lessSteps: string;
    executeCommand: string;
    presentFiles: string;
    needYourHelp: string;
    useTool: (toolName: string) => string;
    searchForRelatedInfo: string;
    searchForRelatedImages: string;
    searchFor: (query: string) => string;
    searchForRelatedImagesFor: (query: string) => string;
    searchOnWebFor: (query: string) => string;
    viewWebPage: string;
    listFolder: string;
    readFile: string;
    writeFile: string;
    /**
     * Business-semantic stage broadcasts (G4 方案 C):
     * UI renders these automatically when EthoInsight subagent/CLI tool calls fire,
     * so users see "正在请专家解读..." instead of raw tool names. Maps to lead prompt
     * §"过程透明原则" 表格，但实施层从 prompt 自觉转移到 UI 自动渲染。
     */
    stageBroadcast: {
      dispatchSubagent: (subagentType: string) => string;
      parseHeaders: string;
      resolveCatalog: string;
      askClarification: string;
      runScript: (scriptName: string) => string;
      genericBash: string;
    };
```

- [ ] **Step 2: 改 `zh-CN.ts` 加中文翻译**

用 Edit 工具修改 `packages/agent/frontend/src/core/i18n/locales/zh-CN.ts`。

old_string（line 246-265 之间）：

```typescript
  // Tool calls
  toolCalls: {
    moreSteps: (count: number) => `查看其他 ${count} 个步骤`,
    lessSteps: "隐藏步骤",
    executeCommand: "执行命令",
    presentFiles: "展示文件",
    needYourHelp: "需要你的协助",
    useTool: (toolName: string) => `使用 “${toolName}” 工具`,
    searchFor: (query: string) => `搜索 “${query}”`,
    searchForRelatedInfo: "搜索相关信息",
    searchForRelatedImages: "搜索相关图片",
    searchForRelatedImagesFor: (query: string) => `搜索相关图片 “${query}”`,
    searchOnWebFor: (query: string) => `在网络上搜索 “${query}”`,
    viewWebPage: "查看网页",
    listFolder: "列出文件夹",
    readFile: "读取文件",
    writeFile: "写入文件",
```

new_string：

```typescript
  // Tool calls
  toolCalls: {
    moreSteps: (count: number) => `查看其他 ${count} 个步骤`,
    lessSteps: "隐藏步骤",
    executeCommand: "执行命令",
    presentFiles: "展示文件",
    needYourHelp: "需要你的协助",
    useTool: (toolName: string) => `使用 “${toolName}” 工具`,
    searchFor: (query: string) => `搜索 “${query}”`,
    searchForRelatedInfo: "搜索相关信息",
    searchForRelatedImages: "搜索相关图片",
    searchForRelatedImagesFor: (query: string) => `搜索相关图片 “${query}”`,
    searchOnWebFor: (query: string) => `在网络上搜索 “${query}”`,
    viewWebPage: "查看网页",
    listFolder: "列出文件夹",
    readFile: "读取文件",
    writeFile: "写入文件",
    stageBroadcast: {
      dispatchSubagent: (subagentType: string) => {
        const labels: Record<string, string> = {
          "code-executor": "🧮 正在计算指标，预计 30-60 秒…",
          "data-analyst": "🔬 指标已完成，正在请专家解读，预计 1-2 分钟…",
          "report-writer": "📝 解读已完成，正在生成中文研究报告…",
          "knowledge-assistant": "📚 正在查阅领域知识…",
        };
        return labels[subagentType] ?? `🛠 正在派遣 ${subagentType}…`;
      },
      parseHeaders: "📂 正在解析 EthoVision 文件结构…",
      resolveCatalog: "📋 正在生成指标计划…",
      askClarification: "⚠️ 我需要先确认一件事…",
      runScript: (scriptName: string) => `⚙️ 正在运行 ${scriptName}…`,
      genericBash: "💻 正在执行命令…",
    },
```

- [ ] **Step 3: 改 `en-US.ts` 加英文翻译**

用 Edit 工具修改 `packages/agent/frontend/src/core/i18n/locales/en-US.ts`。

old_string（line 256-265 之间，按 useTool 结尾锚定，与 zh-CN 同款 14 项）：

```typescript
  // Tool calls
  toolCalls: {
    moreSteps: (count: number) => `View other ${count} steps`,
    lessSteps: "Hide steps",
    executeCommand: "Execute command",
    presentFiles: "Present files",
    needYourHelp: "Need your help",
    useTool: (toolName: string) => `Use "${toolName}" tool`,
```

new_string（仅在节点末尾追加，不改原 14 项；执行 agent 注意 — 用 Edit 之前先在 en-US.ts 里 grep 到 `writeFile:` 这一行，看清楚最后一项是哪一行后再操作）：

实际操作：先 `grep -n "writeFile\|toolCalls:" packages/agent/frontend/src/core/i18n/locales/en-US.ts` 定位 toolCalls 节点末尾，然后在最后一项（`writeFile: "Write file"` 或类似）后面追加：

```typescript
    stageBroadcast: {
      dispatchSubagent: (subagentType: string) => {
        const labels: Record<string, string> = {
          "code-executor": "🧮 Computing metrics, ~30-60 seconds…",
          "data-analyst": "🔬 Metrics ready, consulting domain expert, ~1-2 minutes…",
          "report-writer": "📝 Insights ready, drafting research report…",
          "knowledge-assistant": "📚 Looking up domain knowledge…",
        };
        return labels[subagentType] ?? `🛠 Dispatching ${subagentType}…`;
      },
      parseHeaders: "📂 Parsing EthoVision file structure…",
      resolveCatalog: "📋 Generating metric plan…",
      askClarification: "⚠️ Need to confirm one thing first…",
      runScript: (scriptName: string) => `⚙️ Running ${scriptName}…`,
      genericBash: "💻 Executing command…",
    },
```

确保该追加块插入在原 toolCalls 节点的 `}` 之前（与 zh-CN.ts 的结构对齐）。

- [ ] **Step 4: TypeScript 类型检查**

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
pnpm typecheck
```

Expected: 无错误。types.ts 和两个 locale 必须 schema 一致，否则会报缺字段。

- [ ] **Step 5: Commit i18n 改动**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/frontend/src/core/i18n/locales/types.ts \
        packages/agent/frontend/src/core/i18n/locales/zh-CN.ts \
        packages/agent/frontend/src/core/i18n/locales/en-US.ts
git commit -m "$(cat <<'EOF'
feat(frontend): G4 方案 C — i18n 加 stageBroadcast 节点

为前端自动业务语义播报准备 i18n 字符串。覆盖 4 类场景：
- dispatchSubagent: subagent_type → 中文状态（code-executor / data-analyst /
  report-writer / knowledge-assistant），未匹配走通用 fallback
- parseHeaders / resolveCatalog: EthoInsight 两个 CLI 的专属文案
- askClarification: 反问场景
- runScript / genericBash: 兜底 bash 文案

仅加类型 + zh-CN + en-US 翻译，渲染逻辑下一 commit 接入。
EOF
)"
```

---

### Task 2: 新建 `stage-broadcast.ts` module（业务逻辑）

**Files:**
- Create: `packages/agent/frontend/src/core/tools/stage-broadcast.ts`

**为什么独立成 module**：把"识别 subagent_type / bash command pattern 并返回播报文案"的业务逻辑集中在一处，避免散落在 SubtaskCard / ToolCall 两个组件里。未来加新 subagent 或新 CLI 只改这一份文件。

- [ ] **Step 1: 写完整 module**

完整内容：

```typescript
/**
 * G4 方案 C: 业务语义播报识别
 *
 * 把 tool_call event 的"被动渲染"升级为"语义化状态条"。
 * UI 调本 module 函数把 tool_call 翻译成用户友好的中文文案：
 *
 *   tool_call: task(subagent_type="data-analyst", ...)
 *   → "🔬 指标已完成，正在请专家解读…"
 *
 *   tool_call: bash("python -m ethoinsight.catalog.resolve ...")
 *   → "📋 正在生成指标计划…"
 *
 * 设计原则：
 *   1. 单一识别入口——不在 SubtaskCard / ToolCall 散落识别逻辑
 *   2. fallback 友好——未匹配的 subagent_type / bash 命令返回通用文案，不抛错
 *   3. 文案走 i18n——本 module 不硬编码中文，所有文案从 t.toolCalls.stageBroadcast.* 取
 */

import type { Translations } from "@/core/i18n";

/**
 * Subagent type → 状态播报文案。
 * 未知 subagent_type 走通用 "正在派遣 <type>…" fallback。
 */
export function getStageBroadcastForSubagent(
  subagentType: string,
  t: Translations,
): string {
  return t.toolCalls.stageBroadcast.dispatchSubagent(subagentType);
}

/**
 * 识别 EthoInsight CLI command pattern。
 * 仅对带 `python -m ethoinsight.<module>.<script>` 模式的命令返回非 null。
 */
export function detectEthoinsightCli(
  command: string,
): "parse" | "catalog" | "scripts" | null {
  // 容忍前导空白、python/python3
  const match = command
    .trim()
    .match(/^python3?\s+-m\s+ethoinsight\.(parse|catalog|scripts)\.(\w+)/);
  if (!match) return null;
  return match[1] as "parse" | "catalog" | "scripts";
}

/**
 * bash command → 状态播报文案（仅识别 EthoInsight CLI，其他返回 null 表示"用通用文案"）。
 */
export function getStageBroadcastForBash(
  command: string,
  t: Translations,
): string | null {
  const cliKind = detectEthoinsightCli(command);
  if (cliKind === "parse") {
    return t.toolCalls.stageBroadcast.parseHeaders;
  }
  if (cliKind === "catalog") {
    return t.toolCalls.stageBroadcast.resolveCatalog;
  }
  if (cliKind === "scripts") {
    // 提取脚本名 (ethoinsight.scripts.epm.compute_open_arm_time_ratio → compute_open_arm_time_ratio)
    const scriptMatch = command.match(/ethoinsight\.scripts\.\w+\.(\w+)/);
    const scriptName = scriptMatch?.[1] ?? "script";
    return t.toolCalls.stageBroadcast.runScript(scriptName);
  }
  return null;
}
```

- [ ] **Step 2: lint + typecheck**

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
pnpm check
```

Expected: 无错误。

- [ ] **Step 3: Commit module**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/frontend/src/core/tools/stage-broadcast.ts
git commit -m "$(cat <<'EOF'
feat(frontend): G4 方案 C — 新建 stage-broadcast module

集中识别 subagent_type 和 EthoInsight CLI bash 命令 pattern，
返回用户友好的中文播报文案。三个导出函数：
- getStageBroadcastForSubagent: 处理 task() 派遣
- detectEthoinsightCli: 识别 python -m ethoinsight.* 命令
- getStageBroadcastForBash: 翻译 bash 命令为状态文案

文案全部从 i18n 取，新增 subagent / CLI 只改此 module 不改 UI 组件。
EOF
)"
```

---

### Task 3: 接入 SubtaskCard——subagent 派遣场景

**Files:**
- Modify: `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx:80-92`（折叠态卡片标题段）

**改动思路**：`SubtaskCard` 当前展示 `task.description`（lead 写在 `task()` 调用里的中文描述）作为卡片标题。改成**优先**展示 `getStageBroadcastForSubagent(task.subagent_type, t)` 的业务播报；如果 lead 给的 description 与播报不同则显示在播报下方作为副标题。

- [ ] **Step 1: 加 import 到 subtask-card.tsx**

用 Edit 工具，在 `subtask-card.tsx` 的 import 段（line 21-26 附近，找 `import { useI18n }`）后追加：

old_string：
```typescript
import { useI18n } from "@/core/i18n/hooks";
import { hasToolCalls } from "@/core/messages/utils";
```

new_string：
```typescript
import { useI18n } from "@/core/i18n/hooks";
import { hasToolCalls } from "@/core/messages/utils";
import { getStageBroadcastForSubagent } from "@/core/tools/stage-broadcast";
```

- [ ] **Step 2: 改卡片标题渲染**

找 `ChainOfThoughtStep` 的 `label` prop（约 line 80-92），原本是直接显示 `task.description`。

old_string：
```typescript
              <ChainOfThoughtStep
                className="font-normal"
                label={
                  task.status === "in_progress" ? (
                    <Shimmer duration={3} spread={3}>
                      {task.description}
                    </Shimmer>
                  ) : (
                    task.description
                  )
                }
                icon={<ClipboardListIcon />}
              ></ChainOfThoughtStep>
```

new_string：
```typescript
              <ChainOfThoughtStep
                className="font-normal"
                label={
                  task.status === "in_progress" ? (
                    <Shimmer duration={3} spread={3}>
                      {getStageBroadcastForSubagent(task.subagent_type, t)}
                    </Shimmer>
                  ) : (
                    getStageBroadcastForSubagent(task.subagent_type, t)
                  )
                }
                icon={<ClipboardListIcon />}
              ></ChainOfThoughtStep>
```

**关键点**：把 `task.description` 替换为 `getStageBroadcastForSubagent(task.subagent_type, t)`。这一改之后，无论 lead 是否给出 description，前端都展示业务语义文案。原 `task.description` 仍然在折叠态打开后的 prompt 区域可见（line 125-141），不会丢失。

- [ ] **Step 3: lint + typecheck**

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
pnpm check
```

Expected: 无错误。

- [ ] **Step 4: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): G4 方案 C — SubtaskCard 展示业务语义播报

折叠态卡片标题从纯 task.description 改为 getStageBroadcastForSubagent(),
按 subagent_type 显示"🔬 正在请专家解读…"等中文播报。lead 写的
description 仍在展开态的 prompt 区可见。

这是从 prompt 自觉播报转为 UI 自动播报的核心入口。
EOF
)"
```

---

### Task 4: 接入 ToolCall bash 分支——EthoInsight CLI 场景

**Files:**
- Modify: `packages/agent/frontend/src/components/workspace/messages/message-group.tsx`（bash 工具渲染分支）

**前置步骤**：先 grep 找到 bash 分支的精确位置。

- [ ] **Step 1: 精确定位 bash 分支**

Run:
```bash
grep -n '"bash"\|name === "bash"' /home/wangqiuyang/noldus-insight/packages/agent/frontend/src/components/workspace/messages/message-group.tsx
```

记下 bash 分支的起始行号（应该在 line 400-470 之间）。

- [ ] **Step 2: 查看 bash 分支的当前实现**

Run:
```bash
sed -n '<bash_start_line>,$p' packages/agent/frontend/src/components/workspace/messages/message-group.tsx | head -40
```

确认 bash 分支当前显示什么（很可能是 `executeCommand` 或 `args.description`）。

- [ ] **Step 3: 加 import 到 message-group.tsx**

用 Edit 工具，在 message-group.tsx 的 import 段加：

```typescript
import { getStageBroadcastForBash } from "@/core/tools/stage-broadcast";
```

（按 imports 排序规则插入：external imports 后、internal `@/` imports 中、按字母序）

- [ ] **Step 4: 改 bash 分支，加 EthoInsight CLI 识别**

找到 bash 分支（基于 Step 1 的行号），将分支顶部的 label 计算逻辑改为：

**找到当前 bash 分支类似的代码段**（具体形式视当前实现）：
```typescript
  } else if (name === "bash") {
    // 当前实现：直接用 args.description 或 t.toolCalls.executeCommand
    const description = (args as { description?: string })?.description
      ?? t.toolCalls.executeCommand;
    return (
      <ChainOfThoughtStep key={id} label={description} ... >
        ...
      </ChainOfThoughtStep>
    );
```

**改为**：

```typescript
  } else if (name === "bash") {
    const command = (args as { command?: string })?.command ?? "";
    const description = (args as { description?: string })?.description;
    // G4 方案 C: 优先识别 EthoInsight CLI 业务语义播报
    const stageBroadcast = getStageBroadcastForBash(command, t);
    const label = stageBroadcast ?? description ?? t.toolCalls.executeCommand;
    return (
      <ChainOfThoughtStep key={id} label={label} ... >
        ...
      </ChainOfThoughtStep>
    );
```

**关键点**：
- `stageBroadcast` 不为 null（识别到 EthoInsight CLI）→ 用业务文案
- 否则回到 lead 写的 description
- 都没有时回到通用 executeCommand 文案
- **如果 bash 分支已经显示了 `description`，仅在 label 前面**加上业务识别——保持向后兼容

如果 bash 分支当前的渲染结构与上述假设不同（例如分多个 case），按"优先用 stageBroadcast，否则保留现有逻辑"的原则调整。**核心是 `getStageBroadcastForBash(command, t)` 应该被调用，且其非 null 返回值优先于现有 description**。

- [ ] **Step 5: lint + typecheck**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
pnpm check
```

Expected: 无错误。

- [ ] **Step 6: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/frontend/src/components/workspace/messages/message-group.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): G4 方案 C — ToolCall bash 分支识别 EthoInsight CLI

bash 工具渲染时优先调 getStageBroadcastForBash() 识别命令模式。
匹配的 ethoinsight.parse.* / ethoinsight.catalog.* / ethoinsight.scripts.*
显示业务语义文案（"📂 正在解析 EthoVision 文件结构…"等），
非 EthoInsight 命令回退到 description 或通用 executeCommand 文案。
EOF
)"
```

---

### Task 5: 接入 ask_clarification 场景（如果还没有）

**Files:**
- Possibly modify: `packages/agent/frontend/src/components/workspace/messages/message-group.tsx`（ask_clarification 渲染分支）

**前置探查**：先看看 ask_clarification 现在怎么渲染的——它可能已经有专门 UI（interrupt 流），那本 plan 不必动；也可能走通用 ToolCall fallback，那要补一下。

- [ ] **Step 1: 查现状**

Run:
```bash
grep -n '"ask_clarification"\|ask_clarification' /home/wangqiuyang/noldus-insight/packages/agent/frontend/src/components/workspace/messages/message-group.tsx /home/wangqiuyang/noldus-insight/packages/agent/frontend/src/components/workspace/messages/message-list.tsx 2>/dev/null
```

- [ ] **Step 2: 判断**

- 如果**已有专门渲染分支**（例如 interrupt 处理面板）：**跳过 Task 5**——ask_clarification 已经有显著 UI，不需再加播报
- 如果**走通用 ToolCall fallback**：用 Edit 给 message-group.tsx 加 `} else if (name === "ask_clarification") {` 分支，label 用 `t.toolCalls.stageBroadcast.askClarification`

- [ ] **Step 3（仅当 Step 2 判断需要加时）: 实施 ask_clarification 分支**

找到 ToolCall 的 else-if 链末尾，在通用 fallback 之前加：

```typescript
  } else if (name === "ask_clarification") {
    return (
      <ChainOfThoughtStep
        key={id}
        label={t.toolCalls.stageBroadcast.askClarification}
        icon={ClipboardListIcon}  // 或 AlertTriangleIcon, 选已有 import 的
      />
    );
```

- [ ] **Step 4: lint + typecheck**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
pnpm check
```

- [ ] **Step 5: Commit（仅当 Step 3 实施）**

```bash
git add packages/agent/frontend/src/components/workspace/messages/message-group.tsx
git commit -m "feat(frontend): G4 方案 C — ask_clarification 显示业务语义播报"
```

---

### Task 6: Manual QA 端到端验证

**Files:**
- 不修改任何文件（纯测试）

**目的**：用 thread b0d3a611 / 8ff3be6d 同款入口跑一次完整 dogfood，肉眼观察 UI 上是否在每个 tool_call 触发时显示业务语义播报。

- [ ] **Step 1: 起服务**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop
make dev  # 后台跑
until curl -sf --max-time 2 http://localhost:2026/ -o /dev/null 2>/dev/null; do sleep 2; done && echo "ready"
```

如 gateway 30s 起不来按 `docs/handoffs/2026-05/2026-05-14-e2e-test-checklist.md` §1.4 备选。

- [ ] **Step 2: 浏览器跑完整 dogfood**

用 Playwright MCP 或自己开浏览器 `http://localhost:2026`：
1. 新建 thread
2. 上传 `/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点/轨迹-Elevated Plus Maze XT190-Trial     1-Arena 1-Subject 1.txt`
3. 发：`请分析这个 EPM 单只数据`
4. 跟着流程走，**每一步观察 UI 上是否出现业务播报**：
   - lead 跑 `python -m ethoinsight.parse.dump_headers` → UI 应显示 `📂 正在解析 EthoVision 文件结构…`
   - lead 跑 `python -m ethoinsight.catalog.resolve` → UI 应显示 `📋 正在生成指标计划…`
   - lead 派 code-executor → 卡片应显示 `🧮 正在计算指标，预计 30-60 秒…`
   - lead 派 data-analyst → 卡片应显示 `🔬 指标已完成，正在请专家解读…`
   - lead 派 report-writer → 卡片应显示 `📝 解读已完成，正在生成中文研究报告…`
   - lead 调 ask_clarification → 应显示 `⚠️ 我需要先确认一件事…`（如 Task 5 实施了）

- [ ] **Step 3: 截图保留**

完整对话截图（Playwright `browser_take_screenshot fullPage=true`）保存到 `docs/handoffs/2026-05/screenshots/2026-05-15-g4-method-c-verified.png`（若 screenshots 目录不存在，先 mkdir）。

- [ ] **Step 4: 停服务 + 记录验证**

```bash
make stop
```

在 `docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md` 末尾追加：

```markdown
## G4 方案 C 修复复测（YYYY-MM-DD）

修复 commits: <列 Task 1-5 的 commit hash>

dogfood thread: <UUID>

观察结果：
- code-executor 派遣 → UI 显示 "🧮 正在计算指标…" ✅ / ❌
- data-analyst 派遣 → UI 显示 "🔬 正在请专家解读…" ✅ / ❌
- report-writer 派遣 → UI 显示 "📝 正在生成研究报告…" ✅ / ❌
- dump_headers → UI 显示 "📂 正在解析 EthoVision 文件结构…" ✅ / ❌
- catalog.resolve → UI 显示 "📋 正在生成指标计划…" ✅ / ❌

截图：`screenshots/2026-05-15-g4-method-c-verified.png`

判定：G4 修复 ✅ / 部分 ⚠️ / 失败 ❌

Batch A/B 表格 G4 行：从 ⚠️ partial 改为对应结果。
```

- [ ] **Step 5: Commit 验证记录**

```bash
cd /home/wangqiuyang/noldus-insight
git add docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md \
        docs/handoffs/2026-05/screenshots/  # 如有截图
git commit -m "$(cat <<'EOF'
docs(dogfood): G4 方案 C 端到端验证 — 前端 UI 自动播报生效

dogfood thread <UUID> 复测：所有 tool_call 触发时 UI 上正确显示业务语义播报，
不再依赖 lead prompt 自觉。Batch A/B 表格 G4 行从 ⚠️ partial 改为 ✅。
EOF
)"
```

---

### Task 7: 收尾 push

- [ ] **Step 1: 全量 frontend check**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
pnpm check
```

Expected: 全绿。

- [ ] **Step 2: backend 测试（防止 frontend 改动意外影响）**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make test
```

Expected: 全绿（除已知 6 个预存失败 auth/live/skill）。

- [ ] **Step 3: commit 自查**

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline origin/dev..dev
```

Expected 看到本 plan 的 commit（4-6 个，按 Task 1/2/3/4/5/6 拆分）：
1. `feat(frontend): G4 方案 C — i18n 加 stageBroadcast 节点`
2. `feat(frontend): G4 方案 C — 新建 stage-broadcast module`
3. `feat(frontend): G4 方案 C — SubtaskCard 展示业务语义播报`
4. `feat(frontend): G4 方案 C — ToolCall bash 分支识别 EthoInsight CLI`
5. `feat(frontend): G4 方案 C — ask_clarification 显示业务语义播报`（可选，看 Task 5 判断）
6. `docs(dogfood): G4 方案 C 端到端验证 — 前端 UI 自动播报生效`

- [ ] **Step 4: Push**

```bash
git push origin dev 2>&1 | tail -5
```

Expected: `<旧hash>..<新hash>  dev -> dev`，所有 commit 入 origin。

- [ ] **Step 5: 回报用户**

```
# G4 方案 C 落地完成

## Commits 列表
<贴 git log 输出>

## Dogfood 验证
- thread: <UUID>
- 5 个 tool_call 事件全部显示业务播报 ✅ / 部分 ⚠️
- 截图：screenshots/2026-05-15-g4-method-c-verified.png

## Batch A/B 状态
- G4 阶段播报: ⚠️ partial → ✅ (UI 机制层兜底，不再依赖 LLM 自觉)
- 整体 8/8 ✅

## 遗留
无。所有 Batch A/B 检查通过。下一步候选：spec 阶段 1 双层 handoff 协议落地。
```

---

## 不要做的事（防止越权）

- ❌ **不要修后端 `lead_agent/prompt.py`**——FIRST-TOKEN 回退在独立 plan
- ❌ **不要修 `ui/` 和 `ai-elements/` 目录**（frontend CLAUDE.md 明令禁改）
- ❌ **不要加单元测试**（frontend 无 test framework，CLAUDE.md 第 12 行说明）
- ❌ **不要改 SSE 事件流或 LangGraph SDK 用法**——本 plan 只在现有渲染层补业务识别
- ❌ **不要"顺便修"其他 UI 问题**——保持 plan scope 收紧
- ❌ **不要用 `--no-verify`**
- ❌ **不要 force push**
- ❌ **不要碰这 3 个无关文件**：
  - `docs/specs/llm-finetuning-strategy.md`
  - `docs/plans/2026-05-13-base-model-decision-memo.md`
  - `packages/agent/frontend/src/app/page.tsx`（即使在 frontend 目录里）

---

## 实施完成后的状态

- 新建 1 个文件：`stage-broadcast.ts`
- 改 6 个文件：3 个 i18n + 2 个组件 + 1 个 dogfood doc
- 4-6 个 commit 入 origin
- G4 修复**机制化**：前端 UI 自动在 tool_call 触发时显示业务语义播报
- 不再依赖 lead prompt 自觉
- backend 完全不动
- Batch A/B 8/8 ✅，dogfood 闭环完整收尾
