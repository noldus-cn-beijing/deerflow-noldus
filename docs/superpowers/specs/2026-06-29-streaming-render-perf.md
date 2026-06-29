# Spec: 流式 Markdown 渲染性能 —— 消除每 token O(n) 重扫 + 稳定 markdown 渲染

> 状态：待实施（根因已坐实、业界/DeerFlow 上游最佳实践已调研）
> 归属：前端性能 / dogfood 修复批（2026-06-29）
> 红线：**不动流式核心**（useStream / mergeMessages / dedupe），守 `feedback_perf_is_efficient_impl_not_visual_downgrade`——只省开销、不降视觉。

## Context（为什么做）

dogfood：agent 流式输出到一定内容后**浏览器单 tab 奇卡**（主线程被占满），尤其 subagent think（reasoning）流式时。#212（Phase0#7）已加 `useDeferredValue(messages)` + `useMemo(groupMessages)` + 热组件 `React.memo`，**仍卡** → 瓶颈在别处。

## 根因（已读码坐实）

整条流式渲染：`useStream`(hooks.ts) → `useDeferredValue` → `useMemo(groupMessages)` → `MessageList` → `markdown-content.tsx` → streamdown(Shiki)。

1. **头号：`splitInlineReasoning()`（`core/messages/utils.ts:294`）对整条消息内容跑正则，且在一趟里被调 4-6 次**（`hasContent`/`hasReasoning`/`extractContentFromMessage`/`extractReasoningContentFromMessage`，分别在 utils.ts:338/369/411/426），而这趟 **每个 deferred 批次都重跑**。消息累积越长，每批次重扫 O(n) → 整条流摊销 O(n²)。**无任何按 message 缓存**。
   - 调用点密集在 `message-list.tsx`（groupMessages mapper 内）+ `message-list-item.tsx:153/154`（render 内）。
   - 上游 DeerFlow 的 message-list **不在 render 热路径做 content 抽取**（把抽取留给 copy/export），所以上游不显此症。
2. **次因：markdown 渲染层 prop 不稳定**。`markdown-content.tsx:42` 的 `components` 对象 deps 含未 memo 的 `componentsFromProps`+`threadId`，每次重建 → `message.tsx:307` 的 `MessageResponse` memo 只比 `children` 字符串，`components` 新身份会击穿 memo → 整段 markdown 可能每 token 重解析（含 Shiki 语法高亮）。
3. **零散**：`subtask-card.tsx:179` inline `components={{a:CitationLink}}`、`message-list.tsx:495` inline style，每 render 新身份。

## 业界 / 上游最佳实践（调研结论）

- streamdown 2.5.0（我们用的）内置 `remend` 增量解析 + 块级 memo + Shiki，**「committed 前缀 + live 尾部只重解析尾块」这层库已做**——不必自造，但**前提是传给它的 props 身份稳定**（命中根因 2）。
- `useDeferredValue` 已是 RAF 批处理的现代等价（根因层面已对，不必再加节流）。
- 通用法则命中我们缺口：**避免每 token 全量扫整条消息**（根因 1）、**稳定渲染 props 让块级 memo 生效**（根因 2）、长列表虚拟化（已有 `VirtualizedGroups`，按阈值生效，无需动）。

## 方案（结构性、最小、同视觉）

按确认顺序——**先 CDP 实测坐实，再改**（handoff 铁律；守「代码有修复≠现象消除」）：

### Step 0（前置，不可跳）：CDP perf trace 坐实
在 **prod build**（`pnpm build && pnpm start` 或 prod compose :2026，dev build 的 perf 是噪声）上，用 `noldus-insight-e2e` 的 CDP perf 采集跑一轮长流式 run，确认主线程 self-time 大头落在 `splitInlineReasoning`/正则/markdown 解析。**坐实后再动手**，给出 before 数字作为验收基线。

### Step 1（头号修）：按 message 缓存 content/reasoning 抽取
- 新 `core/messages/extraction-cache.ts`：以 `message.id` 为键缓存 `extractContentFromMessage` / `extractReasoningContentFromMessage` 结果。
- **缓存失效规则（关键正确性）**：in-flight 流式消息内容在变 → 缓存 key 必须含「内容长度或 isStreaming」维度，**只对已终态（非流式）消息长缓存**；流式中那一条不缓存（它本来就在变，缓存无益且会过期）。即：缓存只挡「已完成的历史消息每批次被重扫」——而长 thread 里历史消息占绝大多数，收益就在这。
- 合并 `splitInlineReasoning` + `stripControlSignalLines` 为单趟扫描（少一次 O(n)）。
- 消费点（message-list.tsx 的 mapper、message-list-item.tsx）改调缓存版。

### Step 2（次因修）：稳定 markdown 渲染 props
- `markdown-content.tsx`：把 `components` 的 `useMemo` deps 收敛到真正变化的量；调用方传入的 `componentsFromProps` 要么 memo 要么不传。
- `MessageResponse`（message.tsx:307）memo 比较扩展到 `{children, components}` 身份，或把 `MarkdownContent` 自身 `React.memo`。
- 让 streamdown 的块级 memo 真正生效（已完成消息不再每 token 重解析）。

### Step 3（零散，低优先）
- `subtask-card.tsx:179` / `message-list.tsx:495` 的 inline object/style 提到 `useMemo`/模块常量。

## 改动文件
- 新增 `src/core/messages/extraction-cache.ts`
- 改 `src/core/messages/utils.ts`（单趟扫描；可选导出缓存友好的纯函数）
- 改 `src/components/workspace/messages/message-list.tsx`、`message-list-item.tsx`（用缓存）
- 改 `src/components/workspace/messages/markdown-content.tsx`、`src/components/ai-elements/message.tsx`（稳定 prop + memo）
- 零散：`subtask-card.tsx`

## 验收
- **CDP perf（prod build）**：Step 0 基线 vs 改后，长流式 run 主线程 `splitInlineReasoning`/抽取 self-time 降 ≥50%，longtask 数下降；交互延迟（INP）改善。**dev build 的 perf 不作数**。
- 功能不回归：`pnpm check` 0 error；`npx vitest run`（现有 + 新增缓存单测；2 个 isStreaming pre-existing 红 baseline 不变）。reasoning/think 流式显示、content strip（[intent]/[gate_signals]）行为字节级不变（TDD：缓存版与原函数对同一 message 输出相等）。
- 守红线：useStream/mergeMessages/dedupe 未碰（git diff 证）。

## 不做（non-goals）
- 不重写 streamdown（库已做块级增量）。
- 不动 useStream/SSE/merge 核心。
- 不调虚拟化阈值（已有）。
