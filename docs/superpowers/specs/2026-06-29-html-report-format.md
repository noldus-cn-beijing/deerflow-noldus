# Spec: 分析报告从 Markdown 改为 HTML（含少量代表性图表内联）

> 状态：待实施（产物链已摸清；端点基础设施已半就位）
> 归属：后端 report-writer + 前端渲染（2026-06-29）
> 自检：改 report-writer prompt 前过 HarnessX 三病理（尤其 reward hacking：验收看真产物不看自述；catastrophic forgetting：保留判读语言宪法/术语约束全部原样）。

## Context（为什么做）

当前 report 是 Markdown（`report.md`），**下载 markdown 会丢图**（图是分离的 .png，md 只存路径引用），且 markdown 限制了报告的视觉表达与图表嵌入能力。用户要改 HTML：更 fancy、能内联图表、有视觉表达。

**用户关键约束**：HTML 报告里**只放少量"代表性"图**，不是把全部生成的图（113 张）都塞进去。→ 这绕开了"base64 全量几十 MB"的顾虑：**少量代表性图 base64 内联完全可行**，下载即自包含、离线可看，彻底治"下载丢图"。

## 现状产物链（已读码）

- **report-writer subagent**：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`（`REPORT_WRITER_CONFIG`）。当前输出 `/mnt/user-data/outputs/report.md`，6 段中文骨架，图用 `{{img:<filename>}}` 占位符；handoff `handoff_report_writer.json` 的 `report_path` 指向 .md。disallowed_tools 禁 bash/str_replace（防 LLM 瞎写路径）。
- **seal 解析占位符**：`tools/builtins/seal_handoff_tools.py`——regex 把 `{{img:basename}}` 按 `handoff_chart_maker.json` 的 chart 映射换成 canonical 路径 `/mnt/user-data/outputs/<file>`，并规整坏路径。
- **图与报告关联**：chart-maker 生成 .png 落 outputs/，元数据在 `handoff_chart_maker.json:chart_files`；report 通过占位符引用。
- **端点**：`app/gateway/routers/artifacts.py` `list_report_artifacts`（GET `/threads/{tid}/artifacts/reports`）**已 glob `.md/.htm/.html`**——HTML 列举免改。内容投递：`.html` 在 `ACTIVE_CONTENT_MIME_TYPES` 里**强制 attachment 下载**（防内联 XSS）。
- **前端渲染**：`src/components/workspace/artifacts/report-card.tsx` 用 `MarkdownContent` 渲染。`isReportArtifact` 已认 `.html`。**无 HTML 渲染器**。

## 方案

### 嵌图策略：base64 内联 + 仅代表性图（用户拍板）
- report-writer 在 HTML 里用 `{{img:<filename>}}` 占位**仅代表性图**（prompt 指导它从 chart_maker 产物里挑 1-N 张关键图，如每范式核心箱线/轨迹各一，不全量）。
- seal 时新增逻辑：把占位符对应的 .png 读出 → base64 → 内联成 `<img src="data:image/png;base64,...">`。少量图 → 文件可控。
- 自包含 HTML：下载即离线可看全部内联图，治"下载丢图"。

### 后端改动
1. `report_writer.py`：
   - 输出路径 .md → `.html`；handoff `report_path` 同步。
   - system prompt：6 段骨架改产合法 HTML5（`<!DOCTYPE html>`…`<h2>`/`<table>`/`<p>`/`<ul>`）。**判读语言宪法 / 禁止写法 / 绝对阈值禁令全部原样保留**（只换载体，不松约束——守 catastrophic forgetting）。
   - 图指导：明示"只内联少量代表性图，用 `{{img:<basename>}}` 占位，seal 会转 base64"。
2. `seal_handoff_tools.py`：`{{img:}}` 解析分支扩展——HTML 模式读 png→base64→`<img src="data:...">`；保留路径规整。**HTML 消毒**（去 `<script>`/on* 事件）在 seal 时做一层（防注入）。
3. `handoff_schemas.py`：`report_path` 容忍 .md/.html（或直接切 .html）。
4. 端点 `artifacts.py`：列举免改；**决策**——HTML 报告是「前端内联渲染」还是「强制下载」？建议**前端内联渲染**（见下），则需放开 `.html` 的 attachment 强制 or 走前端 fetch 文本再渲染（report-card 已是 fetch 文本→渲染模式，不触发浏览器 MIME 内联，安全）。

### 前端改动
- 新 `src/components/workspace/.../html-content.tsx`：渲染 HTML 报告，**必须 XSS 消毒**（引 `dompurify`/`sanitize-html`，或后端已消毒+前端二次 sanitize）。
- `report-card.tsx`：按扩展名 `.html` 走 `HTMLContent`，`.md` 仍走 `MarkdownContent`（互不影响、旧 md 报告不回归）。

### TDD
- 后端：`test_report_writer_config.py`（prompt 出 .html + 只代表性图指导）；新 `test_seal_html_report.py`（占位符→base64 内联、`{{img:}}` 全消、`<script>` 被消毒）；`test_artifacts_reports_endpoint.py` 加 .html case。
- 前端：`HTMLContent` 消毒单测（`<script>`/onerror 被剥）、扩展名路由单测。

## 改动文件
- 后端：`report_writer.py`、`seal_handoff_tools.py`、`handoff_schemas.py`、（可能）`artifacts.py` + 三测试。
- 前端：新 `html-content.tsx`、改 `report-card.tsx` + 测试，`package.json` 加 sanitizer 依赖。

## 风险
- **XSS**：HTML 来自 LLM 产出，前端 `dangerouslySetInnerHTML` 必须 sanitize（硬要求）。
- **HTML 合法性**：LLM 可能产残缺 HTML → seal 时 `html.parser` 校验/兜底包 `<pre>`。
- **catastrophic forgetting**：prompt 重写**只换 HTML 载体**，判读宪法/术语/阈值禁令逐条保留——改完 grep 三镜像文案在不在。
- **reward hacking**：验收看真 report.html 内联图能渲染，不看 handoff 自述。

## 验收
- dogfood 一轮：产出 `report.html`，前端 ReportCard 内联渲染、代表性图可见；下载该 html 离线打开图仍在（base64）。
- 后端 130+ artifact 测试 + 新测全绿；裸导入两入口 0 退出。
- 旧 `.md` 报告仍正常渲染（不回归）。

## 不做
- 不全量内联 113 图（用户明确只代表性）。
- 全量图仍走右侧 gallery 面板（见 spec 4）。
