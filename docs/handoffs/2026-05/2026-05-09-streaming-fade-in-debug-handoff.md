# 2026-05-09 Streaming 吞字现象调查 — 进度交接

## 用户报告的两个问题

1. **Streaming 吞字** — AI 输出过程中，先出 1-2 行文字，然后"消失"，又继续出。视觉上像被吞。
2. **图中带图标的"框"** — 用户截图显示 "执行 1 个子任务 / 解读 Shoaling 分析结果" 灰底卡片。

## 已确认结论

### 问题 2：不是 bug，是设计

那个框是 [packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx](../../packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx) 渲染的 `SubtaskCard`，lead agent 调 `task` 工具派遣 subagent 时显示。可点击展开看 chain-of-thought。视觉是否要调整可另议。

### 问题 1：高度怀疑根因 = `rehypeSplitWordsIntoSpans` + `animate-fade-in: 1.1s`

**证据链**：
1. AI streaming 渲染走 [markdown-content.tsx](../../packages/agent/frontend/src/components/workspace/messages/markdown-content.tsx) → `MessageResponse` (Streamdown)
2. `isLoading=true` 时启用 [rehype/index.ts](../../packages/agent/frontend/src/core/rehype/index.ts) 的 `rehypeSplitWordsIntoSpans`，把每个非 CJK 文本节点切成 word + 包 `<span class="animate-fade-in">`
3. [globals.css:83](../../packages/agent/frontend/src/styles/globals.css#L83) `--animate-fade-in: fade-in 1.1s` (opacity 0 → 1, 1.1s)
4. 每来一个 streaming chunk → content 变 → streamdown 重切 markdown blocks → 每个 block 重过 react-markdown → rehype 重新生成全部 span → React diffing 对无 key 的 span 视为新挂载 → 重新触发 1.1s 渐显
5. 用户感知：刚渲染好的字 opacity 又变 0，1.1s 内重新出现 = "消失再出现"

注意 CJK 文本（中文整段）在 [rehype/index.ts:21-24](../../packages/agent/frontend/src/core/rehype/index.ts#L21-L24) 被跳过不切，所以**纯中文段落不受影响**；但中英混排、含数字/标点/Markdown 标记、英文术语都会触发。

## 已做的改动（待验证）

- [globals.css:83](../../packages/agent/frontend/src/styles/globals.css#L83) `1.1s` → `0.15s`
- 改动还未提交 git
- Dev server 仍在跑（`make dev`，nginx 入口 :2026），turbopack HMR 应该已应用

## 验证方案

下次会话用 Playwright MCP（已装好但当前会话未加载，重启后可用）：

1. `playwright_browser_navigate` 到 `http://localhost:2026/`
2. 创建/进入一个 thread，发起一次 streaming（比如让 lead agent 跑一个会出英文/Markdown 输出的任务，或者直接复现上次的 shoaling 分析）
3. 在流式过程中：
   - 录屏或定时 snapshot 看 token 渲染节奏
   - 用 `evaluate` 抓 `<span class="animate-fade-in">` 节点的 computed `opacity` 和 `animation` 状态，看是否持续重置
   - 网络面板看 SSE 是不是有真断流（应该没有）
4. 对比：如果吞字现象大幅减弱/消失 → 假设确认；如果仍存在 → 假设错误，回 Phase 1 重新查（可能是 streamdown block 切分导致整 block 卸载，或后端真断流）

## 永久修复方向（验证通过后再做）

按代价和效果排序：

1. **保持 0.15s**（最简，已做）— 时长缩短到接近 chunk 间隔，重渲染叠加问题被掩盖。代价：失去渐显视觉效果。
2. **rehype 给 span 加稳定 key**（中等）— `properties.key = ${index}-${word}` 或哈希，让 React diff 复用 DOM。可能不可靠因为 word 内容流式中会变化。
3. **彻底删 rehypeSplitWordsIntoSpans**（最干净）— streamdown 已自带增量 token 渲染，重复加渐显反而打架。建议这条。
4. **只对 reasoning 内容启用**（折中）— 把启用条件从 `isLoading` 收紧到只在 reasoning 模块用，普通正文不切词。

## 环境状态

- 前端 dev server 在跑（pid 详见 `ps aux | grep "next dev"`）
- LangGraph :2024 / Gateway :8001 / Nginx :2026 都健康
- Playwright MCP 装好（chromium runtime 在 `~/.cache/ms-playwright/chromium-1217/`），重启 Claude Code 会话后工具可用
- 之前 npx 缓存损坏 (`.playwright-core-XQxHSCe3`) 已清理

## 下一步（重启会话后）

用户会说："接着验证 streaming 吞字"。立即用 playwright 打开页面跑验证，按上面"验证方案"执行。
