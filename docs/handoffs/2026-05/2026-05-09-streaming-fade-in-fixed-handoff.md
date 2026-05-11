# 2026-05-09 Streaming 吞字 — 修复完成 + 上游 PR 交接

> **前置文档**：
> - [v1 假设链](./2026-05-09-streaming-fade-in-debug-handoff.md) — 初步分析
> - [v2 环境准备](./2026-05-09-streaming-fade-in-debug-handoff-v2.md) — chromium MCP 配置 + 直接执行清单

本文档收尾本次会话：完整定性 + noldus-insight 临时修复 + deerflow-noldus 永久 PR + 下次 sync 注意事项。

## TL;DR

- ✅ **noldus-insight `dev` 分支**：commit `d4171eed`，**临时止血**：删 `rehypeSplitWordsIntoSpans` 插件链，丢失 word fade-in 视觉，但吞字完全消失。**未 push**
- ✅ **deerflow-noldus `fix-streaming-word-animation-remount` 分支**：commit `9fd3ec7b`，**永久修复**：streamdown `1.4.0` → `^2.5.0` + 官方 `animated` prop，已 push 到 origin，**PR 用户已提**到 `bytedance/deer-flow:main`
- ⚠️ **下次 deerflow sync** 需特别处理：见本文末"sync 注意"

## 根因（再次确认）

deerflow 上游 `frontend/src/core/rehype/index.ts` 自家 `rehypeSplitWordsIntoSpans` 把 markdown text node 按 word 包成 `<span class="animate-fade-in">`，让 streaming 文字逐 word 淡入。但 react-markdown 没给这些 span 稳定 React key，streamdown@1.4.0 在每个 SSE chunk 重 diff block 时，**已经渲染过的内容词被 React 当新 mount**，重新触发 1.1s fade-in。用户感官 = 字"先出来 → 消失 → 又出来"。

### 实测证据（25 秒英文 streaming，MutationObserver 全程跟踪）

| 指标 | 值 |
|---|---|
| span 总挂载次数 | 2,121 |
| mount 时 `opacity` 全部 | `0`（再 fade 到 1） |
| `animationDuration` 全部 | `1.1s`（默认） |
| **同一内容词重复 mount** | `the` 41× · `distance` 7× · `velocity` 7× · `Eth` 8× · `time` 10× · `zone` 9× |

详见对应会话日志。

### 为什么上游 #1726 没修

`952059eb fix(ui): avoid over-segmenting cjk messages (#1726)` 加了 CJK regex skip，**只解决中文被过度切分**，没动重 mount 行为。英文 streaming 仍然吞字。

## noldus-insight：临时修复

**Commit**：`d4171eed fix(frontend): 删除 rehypeSplitWordsIntoSpans 临时止血 streaming 吞字`，dev 分支，**本地，未 push**。

**改动文件**（8 个）：

| 文件 | 改动 |
|---|---|
| `packages/agent/frontend/src/core/rehype/index.ts` | 整文件删 |
| `packages/agent/frontend/src/core/streamdown/plugins.ts` | 删 `rehypeSplitWordsIntoSpans` import + `streamdownPluginsWithWordAnimation` export |
| `packages/agent/frontend/src/components/workspace/messages/markdown-content.tsx` | `rehypePlugins` 改可选，缺省 `streamdownPlugins.rehypePlugins` |
| `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx` | 删 `useRehypeSplitWordsIntoSpans` 调用链 + `streamdownPluginsWithWordAnimation` 用 |
| `packages/agent/frontend/src/components/workspace/messages/message-list.tsx` | 删 `useRehypeSplitWordsIntoSpans` 调用 + `rehypePlugins=` 透传 |
| `packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx` | 删 `useRehypeSplitWordsIntoSpans` + 多余的 `[rehypeKatex, ...]` 拼接 |
| `packages/agent/frontend/src/components/workspace/messages/message-group.tsx` | 删 `useRehypeSplitWordsIntoSpans` |
| `packages/agent/frontend/src/styles/globals.css` | 删 `--animate-fade-in` keyframe（保留 `--animate-fade-in-up`） |

**质量门**：`pnpm typecheck` ✅、`pnpm lint` ✅。

**验证**：浏览器 25s 英文 streaming，MutationObserver 报 `0` 个 `.animate-fade-in` 挂载，bodyTextLen 5370 字符 + 6 个 H2/H3 markdown 标题正常渲染（即流式渲染没出问题）。

**代价**：失去 word-level fade-in 视觉。流式渲染回退到 streamdown 1.4 的 block-level 默认行为（即 markdown block 整块出现）。

## deerflow-noldus：永久修复 + 上游 PR

**仓库**：`/home/wangqiuyang/deerflow-noldus`
**分支**：`fix-streaming-word-animation-remount`
**Commit**：`9fd3ec7b fix(frontend): use streamdown's built-in animation to stop word re-mounts during streaming`（已 push 到 `noldus-cn-beijing/deerflow-noldus`）
**PR**：用户已提交到 `bytedance/deer-flow:main`（PR 编号待补）

**核心方案**：streamdown 1.4.0 → `^2.5.0` + 用官方 `animated` prop。
- streamdown 2.3.0 引入 `animated={{ animation: 'fadeIn' | 'blurIn' | 'slideUp', duration, easing, sep: 'word' | 'char' }}` + `isAnimating` prop
- 内部用 per-block `prevContentLength` 跟踪机制识别已渲染文本，自动给已渲染部分设 `--sd-duration:0ms` 跳过动画
- **保留 word fade-in 视觉**，**根除重 mount 现象**

**改动文件**（12 个，+153/-154）：
- `frontend/package.json` — streamdown bump
- `frontend/src/app/layout.tsx` — 加 `import "streamdown/styles.css"`
- `frontend/src/core/rehype/` — 整目录删
- `frontend/src/core/streamdown/plugins.ts` — 删 `streamdownPluginsWithWordAnimation`
- `frontend/src/styles/globals.css` — 删 `--animate-fade-in`
- 4 个消费者改 `<Streamdown animated={{...}} isAnimating={isLoading}>`
- 2 处 `components={{ a: ... }}` 加 `as never`（streamdown 2.x `Components` 类型 narrow 了）
- `<SubtaskCard>` 删不再被消费的 `isLoading` prop

**质量门**：`pnpm typecheck` ✅、`pnpm lint` ✅、`pnpm test` 40/40 ✅、`pnpm build` ✅。

**有意未做**（避免 PR 失焦）：
- 没切 streamdown 2.x 的 `plugins={code,math,mermaid,cjk}` 命名插件 API
- 没删 `rehype-katex` / `rehype-raw` / `remark-gfm` / `remark-math` / `unist-util-visit` 等 npm 直接依赖（虽然有些已变成间接依赖）
- 没动 `humanMessagePlugins` / `reasoningPlugins` 自定义（这些有 deerflow 项目特定 rationale，与本 bug 正交）

这些清理可作为后续 PR。

**PR 草稿**：见 `/home/wangqiuyang/deerflow-noldus/PR-DRAFT.md`（**git 未追踪**，仅本地参考）

## 下次 deerflow sync 注意事项

`scripts/sync-deerflow.sh` 拉上游时**会**碰到这些文件冲突。**必读** [docs/sop/deerflow-sync-sop.md](../sop/deerflow-sync-sop.md) 中关于 streaming word animation 的章节（已加）。

简版：

1. **如果上游已 merge 我们的 streamdown 升级 PR**（bytedance/deer-flow 主线已含 streamdown ^2.5.0 + animated prop）：
   - **接受上游版本**，覆盖 noldus-insight 的临时修复
   - 然后**回退** noldus-insight 的 `d4171eed` commit（业务恢复 word fade-in 视觉，根因已由上游修复）
2. **如果上游还没 merge**：
   - 上游同步**跳过**这 8 个文件（保留 noldus-insight 的临时修复）
   - 注意 `streamdown` 仍是 1.4.0，没有 word fade，但没有吞字

**判断方法**：sync 前先 `git log deerflow/main -- frontend/src/core/streamdown/plugins.ts | head` 看上游有没有"animated prop"或"streamdown 2.x"相关 commit。或直接打开 PR 链接看 merged 状态。

## 关键文件路径

| 用途 | noldus-insight 路径 | deerflow-noldus 路径（上游对应） |
|---|---|---|
| 自家 word-split 插件（已删） | `packages/agent/frontend/src/core/rehype/index.ts` | `frontend/src/core/rehype/index.ts` |
| streamdown plugin profile | `packages/agent/frontend/src/core/streamdown/plugins.ts` | `frontend/src/core/streamdown/plugins.ts` |
| markdown 渲染入口 | `packages/agent/frontend/src/components/workspace/messages/markdown-content.tsx` | 同 |
| Subagent CoT 渲染 | `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx` | 同 |
| 主消息列表 | `packages/agent/frontend/src/components/workspace/messages/message-list.tsx` | 同 |
| AI 消息 + KaTeX 拼接 | `packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx` | 同 |
| Reasoning step | `packages/agent/frontend/src/components/workspace/messages/message-group.tsx` | 同 |
| 全局动画变量 | `packages/agent/frontend/src/styles/globals.css` | `frontend/src/styles/globals.css` |
| Streamdown 包装 | `packages/agent/frontend/src/components/ai-elements/message.tsx` | 同 |

## 时间线（本次会话）

1. 接 v2 交接 → MCP 进程检查 → Playwright 进入 chrome devtools-mcp 替代品
2. 浏览器登录 → 多轮 streaming → MutationObserver 确认 41× `the` 重 mount
3. 提议永久修复"删 plugin"，验证有效 → 用户提议反向追溯：Noldus 改的 vs 上游的
4. git log 确认 plugin 来自 deerflow 上游（`9f2b94ed feat: implement basic web app`）
5. 用户提议 PR 走 deerflow-noldus → bytedance/deer-flow
6. 调研 streamdown 现状 → 发现 streamdown 2.3+ 已修同款 bug + 官方 `animated` prop
7. deerflow-noldus 切分支 + commit + push
8. noldus-insight 落临时修复 commit
9. 写本文档

## 后续 follow-up（可选）

- [ ] 监控上游 PR review。如有 reviewer 反馈，按计划修
- [ ] 上游 merge 后，回退 noldus-insight 的 `d4171eed`，让业务恢复 word fade-in
- [ ] 上游 merge + sync 后，可考虑发后续清理 PR：迁到 `plugins={code,math,mermaid,cjk}` 命名插件 API + 删多余 npm 依赖
