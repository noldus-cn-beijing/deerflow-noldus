# Spec: 修复 HTML 报告完全无样式（prose 死类 + typography 未装 + title 裸露）

> 状态：待实施（确定性 bug，活体坐实）
> 归属：#234 回归修复（2026-06-29）
> 用户决策：**前端 prose 方案**（装 typography 插件让 prose 生效；报告本身不带 `<style>`）。

## Context（活体证据）

dogfood thread `73b41dc3` 的 `report.html` 前端渲染后**完全没有格式**——纯文本堆叠，`<table>` 无边框、`<h2>` 无层级、列表/间距全无（用户截图证实）。但 HTML 结构本身**完全合规**（`<!DOCTYPE>`/`<head>`/`<body>`、`<h2>`/`<ul>`/`<strong>`/`<blockquote>`/`<table><thead><th>`）。问题在样式层。

## 根因（坐实）

1. **前端 `prose` 是死类**：`html-content.tsx` 与 `report-card.tsx:88/90` 都套了 `prose prose-sm max-w-none`，但 **`@tailwindcss/typography` 插件根本没装**——`package.json` 无、`node_modules/@tailwindcss/typography` 不存在、`src/styles/globals.css` 无 `@plugin`。→ `prose` 不生成任何 CSS → 语义标签只剩浏览器默认弱样式。同族坑见 memory `feedback_tailwind_v4_dur_namespace_not_real_duration_utility_dead_class`（看着像真类、实则未定义→渲染无效）。
   - **连带影响**：`MarkdownContent` 也套 `prose`（report-card.tsx:90 + 消息流），所以 **markdown 报告/消息的 prose 样式可能一直也没生效**——实施时一并核实（可能是更大范围的潜在缺陷）。
2. **后端报告无内联 `<style>`**：report.html 是纯语义标签，样式全指望前端——而前端那条路断了。（用户已选"报告不带 style、前端控样式"，故此条不改后端。）
3. **附带 bug：`<title>` 裸露**。LLM 写了 `<title>EPM...报告</title>`，但后端 sanitizer 的 `_STRIP_TAGS_KEEP_CONTENT`（seal_handoff_tools.py:311 含 `title`）**剥标签留内容** → 标题文字裸露在 `<head>` 里（前端渲染时若 head 内容被渲染会冒出一行裸文本）。

## 方案

### 1. 装 Tailwind Typography（v4 方式——关键，别用 v3 config）
本前端是 **Tailwind v4**（`globals.css` 用 `@import "tailwindcss"`，无 `tailwind.config.js`）。v4 接插件**在 CSS 里**，不是 config 文件：
```css
/* src/styles/globals.css */
@import "tailwindcss";
@plugin "@tailwindcss/typography";   /* ← 新增这行 */
```
+ `pnpm add -D @tailwindcss/typography`。**别**新建 `tailwind.config.js` 走 v3 `plugins:[]`（v4 不读、是常见误装）。

### 2. 确认 prose 真正命中 dangerouslySetInnerHTML 子树
`HTMLContent` 的 `prose` 在外层 div，内容经 `dangerouslySetInnerHTML` 注入——prose 的后代选择器（`.prose h2`、`.prose table` 等）对注入的子树**生效**（CSS 后代选择器不区分来源）。装好插件后即应生效。实施后**活体核实** report.html 渲染有层级/表格边框。
- 日式简洁观感（守 `feedback_frontend_design_japanese_minimal_motion_craft`）：用 `prose prose-sm`，必要时配项目色板微调（`prose-headings:`、`prose-table:` 等 modifier），不要花哨。

### 3. 修 title 裸露
后端 sanitizer：`title` 从 `_STRIP_TAGS_KEEP_CONTENT` 移到 **drop-with-content**（`<title>` 内容不该出现在 body 渲染区），或前端只渲染 `<body>` 内容（剥 head）。**推荐前端**：`HTMLContent` 渲染前取 `<body>` 内部（或 DOMPurify 配置 `WHOLE_DOCUMENT:false` 只留 body），避免 head 里的裸标题/meta 混入。实施时二选一并加测。

## 改动文件
- `packages/agent/frontend/package.json`（加 typography 依赖）
- `packages/agent/frontend/src/styles/globals.css`（`@plugin`）
- `src/components/workspace/artifacts/html-content.tsx`（确认 prose + 只渲染 body / DOMPurify `WHOLE_DOCUMENT:false`）
- （核实）`report-card.tsx` / `markdown-content.tsx` 的 prose 是否本就失效
- （可选）后端 `seal_handoff_tools.py` title 处理

## TDD / 验收
- `html-content.test.tsx`：渲染含 `<h2>/<table>/<ul>` 的 HTML，断言外层有 prose 类且 DOMPurify 未剥这些结构标签；title/head 内容不出现在输出。
- **活体（关键，守「代码有修复≠现象消除」）**：本地 localhost:2026 打开该 report.html → `<h2>` 有层级、`<table>` 有边框/表头样式、列表有缩进、整体日式简洁。截图确认。
- 构建坐实：`pnpm build` 后 typography 工具类真生成（grep 产物 CSS 含 `.prose` 规则）——避免又一个"看着装了实则没进 build"（守 token PR 探针教训）。
- `pnpm check` 0；vitest 绿。

## 不做
- 报告 HTML 不内联 `<style>`（用户选前端控样式）。
- 不引入重型 CSS 框架，复用 Tailwind typography + 项目色板。
