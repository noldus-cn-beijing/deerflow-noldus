# 2026-05-09 Streaming 吞字验证 — 进度交接 v2

> **直接前置文档**：[2026-05-09-streaming-fade-in-debug-handoff.md](./2026-05-09-streaming-fade-in-debug-handoff.md)（v1，假设链 + 改动），本文档只补充本次会话做的「环境准备」+ 给下一会话的「直接执行清单」。

## 上下文回顾（30 秒）

- 用户报告：AI streaming 时字"先出来 → 消失 → 又出来"
- 假设：[rehype/index.ts](../../packages/agent/frontend/src/core/rehype/index.ts) 把每个非 CJK 词包 `<span class="animate-fade-in">`，每来一个 chunk 全 block 重 diff 时 React 把无 key 的 span 当新挂载，重新触发 `--animate-fade-in: 1.1s` 渐显
- 已做改动：[globals.css:83](../../packages/agent/frontend/src/styles/globals.css#L83) `1.1s` → `0.15s`，**未提交，未验证**

## 本次会话做了什么

**没做验证。** 因为环境里没有系统 Chrome（Playwright/chrome-devtools MCP 默认要找 `/opt/google/chrome/chrome`），花了一个 turn 在让 MCP 用上 `~/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome`（Playwright 已经下好的 chromium runtime）。

### ✅ 已完成的环境准备

> ⚠️ **2026-05-09 二次修正**：v2 文档原本改的两个 `.mcp.json` 路径**都不是 Claude Code 实际加载的入口** —— 重启再多次也不会生效。真正生效的位置是 `~/.claude/plugins/cache/<marketplace>/<plugin>/<version>/.mcp.json`（每个 plugin 装好后 Claude Code 把 marketplace 的 `.mcp.json` 拷到 cache 目录，启动时只读 cache 里的副本）。

1. **chrome-devtools MCP 配置（真入口）** — `/home/wangqiuyang/.claude/plugins/cache/claude-plugins-official/chrome-devtools-mcp/0.22.0/.mcp.json` 加了 `--executablePath /home/wangqiuyang/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome --isolated --headless`
2. **Playwright MCP 配置（真入口）** — `/home/wangqiuyang/.claude/plugins/cache/claude-plugins-official/playwright/unknown/.mcp.json` 加了 `--executable-path ... --isolated --headless`
3. **JSON 合法性已验证** (`jq -e .` 通过)
4. **chromium binary 烟测通过** — `chrome --headless --dump-dom http://localhost:2026/` 拿到完整 HTML（标题 `EthoInsight`，next.js 资源齐）

**校验方法**（下次会话重启后第一件事）：

```bash
ps -ef | grep -E "(chrome-devtools-mcp|playwright/mcp)" | grep -v grep
```

如果命令行里看到 `--executablePath` / `--executable-path` 参数，配置就吃上了。如果没有，说明 plugin 升级后 cache 路径变了（比如 chrome-devtools-mcp 从 0.22.0 升到 0.23.0），需要重新找新的 cache 路径再改。

### ⚠️ 需要重启 Claude Code 会话

MCP server 进程是 Claude Code 启动时拉起来的，改完 `.mcp.json` 后**当前会话仍用旧参数**。下次会话启动时 MCP 会用新参数，两个 MCP 工具就都能用了。

## 环境状态（仍健康）

```
✅ LangGraph :2024  (pid 3514932 起)
✅ Gateway   :8001  (pid 3514998 起)
✅ Frontend dev (next dev --turbo) (pid 3515140 起)
✅ Nginx     :2026  → curl 返 200
✅ globals.css 改动仍在工作目录（git status 显示 ` M`，未 commit）
```

服务都在 pts/9，**别 kill**。

## 下一会话第一步（不要再绕路）

**用户预期一句话**："接着验证 streaming 吞字"。**直接做以下步骤**，不要再讨论方案：

### Step 1：确认 MCP 用上 chromium

```
mcp__plugin_chrome-devtools-mcp_chrome-devtools__list_pages
```

或

```
mcp__plugin_playwright_playwright__browser_navigate(url="http://localhost:2026/")
```

任一不报"找不到 chrome"即说明配置生效。**如果还报错**：检查 `.mcp.json` 是否被覆盖（用户可能 git pull 过），按本文档第二节的 4 个改动重新加上。

### Step 2：打开 thread 触发 streaming

页面 `/` 是 landing。需要进入 chat。建议：

1. `browser_snapshot` 看主页结构，找到"开始对话"或类似按钮
2. 也可以直接 `browser_navigate` 到 `http://localhost:2026/chat` 或 `http://localhost:2026/threads/new`（看具体路由，前端代码在 `packages/agent/frontend/src/app/`）
3. 输入一个会产生**英文/Markdown 混排**输出的 prompt，比如「用英文写一段关于 zebrafish shoaling 的 200 字解读，用 markdown 列点」—— 因为 [rehype/index.ts:21-24](../../packages/agent/frontend/src/core/rehype/index.ts#L21-L24) 跳过纯 CJK，必须中英混排或纯英文才会触发 fade-in span 切词

### Step 3：在 streaming 期间抓 DOM 状态

每隔 200-300ms 跑一次：

```javascript
mcp__plugin_playwright_playwright__browser_evaluate(function: `() => {
  const spans = document.querySelectorAll('.animate-fade-in');
  const sample = Array.from(spans).slice(-20).map(el => ({
    text: el.textContent.slice(0, 30),
    opacity: window.getComputedStyle(el).opacity,
    animationName: window.getComputedStyle(el).animationName,
    animationDuration: window.getComputedStyle(el).animationDuration,
    animationPlayState: window.getComputedStyle(el).animationPlayState,
  }));
  return { count: spans.length, sample };
}`)
```

**判读规则**：
- ✅ **假设确认（修复有效）**：`animationDuration` = `0.15s`，吞字感官消失/明显减弱（用户观察）
- ❌ **假设错误**：`animationDuration` 仍是 `1.1s`（HMR 没生效，硬刷新 + Ctrl+F5 重试），或仍 0.15s 但吞字依旧（说明根因不是 fade-in 时长，回看 v1 文档"永久修复方向"第 2/3 条 — span key 不稳 / streamdown 整 block 重挂载 / SSE 真断流）
- ⚠️ **注意**：opacity 在 0.15s 内多次接近 0 = 仍在重复触发动画 → 即使时长变短也是 cosmetic fix，**根因仍是无 key span 重挂载**，建议走永久修复方向第 3 条「彻底删 rehypeSplitWordsIntoSpans」

### Step 4：用户层验证

抓一段 streaming 录屏（`browser_take_screenshot` 多帧），或问用户："现在感觉怎么样？还有吞字吗？"

### Step 5：根据结果决定

| 结果 | 下一步 |
|---|---|
| 吞字消失 | 问用户：保持 0.15s 临时修复 commit，还是直接做永久修复（删 rehypeSplitWordsIntoSpans）？ |
| 吞字减弱但仍有 | 永久修复方向第 3 条最干净 — 删 [rehype/index.ts](../../packages/agent/frontend/src/core/rehype/index.ts) 里 `rehypeSplitWordsIntoSpans` 的注册和实现，跑一遍 streaming 验证 |
| 吞字不变 | 假设错。**回滚** globals.css 改动（`git checkout packages/agent/frontend/src/styles/globals.css`），重查：(a) 看 SSE Network 面板是否真断；(b) 看 streamdown 切 block 时是否整 block 卸载（在 React DevTools 看 `MessageResponse` 的 children re-render；或用 `browser_evaluate` 监听 MutationObserver 抓 removeChild） |

## 关键文件路径速查

| 用途 | 路径 |
|---|---|
| fade-in span 切词逻辑 | `packages/agent/frontend/src/core/rehype/index.ts` |
| markdown 渲染入口 | `packages/agent/frontend/src/components/workspace/messages/markdown-content.tsx` |
| 动画时长定义 | `packages/agent/frontend/src/styles/globals.css:83` |
| chrome-devtools MCP 配置（真入口） | `~/.claude/plugins/cache/claude-plugins-official/chrome-devtools-mcp/0.22.0/.mcp.json` |
| Playwright MCP 配置（真入口） | `~/.claude/plugins/cache/claude-plugins-official/playwright/unknown/.mcp.json` |
| chromium binary | `~/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome` |

## 风险与坑

1. **不要 sudo 装 chrome** — 已经走偏过一次。chromium runtime 完全够用，配 `--executablePath` 就行。
2. **不要 kill dev server** — pts/9 上跑着，pid 见上，HMR 应该能直接接收 globals.css 改动。如果发现 `animationDuration` 还是 `1.1s`，先在浏览器硬刷新（headless 模式：`browser_navigate(url=...)` 重新拉一遍）。
3. **headless 看不到视觉吞字** — DOM 状态可以抓，但"用户感官上是否消失"最终还是要用户层确认。可以多帧截屏拼成对比。
4. **CJK 不触发 span 切词** — 测试 prompt 必须含英文/数字/Markdown 标记，纯中文段落看不出问题。
5. **MCP 改完要重启会话才生效** — 已经在本会话改了，但本会话拿不到新 MCP 进程。下次会话启动会自动用新配置。

## TL;DR

下一会话开场：

> 用户："接着验证 streaming 吞字"
>
> 你：直接 `mcp__plugin_chrome-devtools-mcp_chrome-devtools__list_pages` 或 Playwright `browser_navigate` 测 MCP 是否用上 chromium → 进 chat → 触发英文 streaming → `browser_evaluate` 抓 `.animate-fade-in` span 的 `animationDuration` + `opacity` → 报告结果。
