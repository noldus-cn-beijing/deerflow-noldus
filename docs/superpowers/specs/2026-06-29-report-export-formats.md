# Spec (feature): 报告多格式导出 —— PDF / Word / LaTeX

> 状态：feature 立项（**实施前 spike 已完成，见下**；选型与端点已据 spike 细化，可进入实施）
> 归属：报告体验（2026-06-29）
> 定位澄清（用户）：**HTML 用于「展现」**（交互界面的可能性，保留）；**导出面向研究员实际使用的三种格式 PDF / Word / LaTeX**。当前完全没有导出能力（只能下原始 .html/.md）。

## ✅ 实施前 spike 已完成（2026-06-29）

结论全文：[spikes/2026-06-29-report-export-formats-spike.md](../spikes/2026-06-29-report-export-formats-spike.md)

**一句话**：PDF / docx 开箱保真（图、中文、表格全进），LaTeX 必须**预处理 data-uri**（否则 pandoc 静默丢图，已验证治法）；选型 **WeasyPrint（PDF，+12MB）+ pandoc（docx/tex，+155MB）**，端点设计已细化。

**前置依赖**：样式 #236 ✅ 已合；图内联 #234 逻辑已合但间歇性丢图 bug 由 **PR #237**（seal fail-loud）修——**#237 应先于导出实施落地**（导出依赖报告有图）。

**实施前仍需人工确认**：用 #237 修复后新生成的「图正确内联」真 report.html，各转一次肉眼比对视觉（本次 spike 受限于无该样本，用构造样本代）。

## Context（为什么做）

研究员最终交付/投稿用 PDF（阅读/归档）、Word（协作/期刊投稿模板）、LaTeX（学术排版）。当前报告产物只有 `report.html`（在线展现）和遗留 `.md`，**无任何转换导出**——研究员拿不到能直接用的格式。HTML 作为「单一真相源」很适合做转换源（语义结构 + 内联图）。

> 前置依赖：本 feature 的转换质量依赖报告 HTML **结构正确 + 图能内联**。所以实施顺序上，先落地这两个修复再做导出：
> - [[2026-06-29-fix-html-report-inline-img]]（图全坏——否则导出的 PDF/Word 也没图）
> - [[2026-06-29-fix-html-report-styling]]（无样式——PDF 视觉直接受影响）

## 方案概览：HTML 作为转换源（single source）

report.html（自包含、base64 内联图、结构化）作为唯一源，后端按需转 3 种格式。新增导出端点 + 前端导出菜单（替换/扩展现有"下载"按钮）。

### 转换链选型（实施前 spike 各跑一遍真报告验证质量）

| 目标 | 候选工具 | 备注 |
|---|---|---|
| **PDF** | ① Playwright/headless Chromium 打印（`page.pdf()`）② WeasyPrint（纯 Python，CSS 分页好） | Playwright 渲染保真最高（和前端展现一致，含 base64 图、prose 样式）；WeasyPrint 无浏览器依赖更轻。**倾向 Playwright**（保真=所见即所得，且部署已可能有 chromium for e2e） |
| **Word (.docx)** | Pandoc（`html → docx`） | 成熟、支持图（base64 需先落临时文件或 pandoc 处理 data-uri）；可挂 reference.docx 模板对齐期刊样式 |
| **LaTeX (.tex)** | Pandoc（`html → latex`） | 输出 .tex 源（研究员自行编译/改）；表格/公式映射需验证 |

- **Pandoc** 一个依赖覆盖 Word+LaTeX 两种，优先。PDF 单独用 Playwright/WeasyPrint。
- **base64 内联图的处理**：pandoc 对 `data:image` 的支持需 spike 验证（可能需预处理把 data-uri 还原成临时文件再转）。这是转换质量的关键风险点。

### 后端
- 新端点：`GET /threads/{tid}/artifacts/report/export?format=pdf|docx|tex`（或 POST），读 report.html → 转换 → 返回 attachment。
- 转换器封装在一处（SSOT），可缓存（同 report 同格式不重复转）。
- 依赖：pandoc（系统二进制，需进部署镜像）+ Playwright chromium 或 weasyprint。**部署影响**：镜像要带这些转换工具——评估镜像体积（守 `.dockerignore`/镜像瘦身那批刚做的纪律，别让导出工具把镜像又撑大；pandoc ~150MB、chromium 大）。可考虑导出走单独 service 或按需安装。

### 前端
- 报告卡/产物面板的"下载"改成**导出菜单**：HTML（原文）/ PDF / Word / LaTeX。
- 点击 → 调导出端点 → 浏览器下载。转换可能耗时 → loading 态 + 失败提示。

## 影响 / 风险
- **部署体积**：pandoc + chromium/weasyprint 进镜像，需权衡（见上）。这是最大工程成本。
- **转换保真**：base64 图、表格、prose 样式在三种格式里的还原度——**实施前必须 spike**（拿真 report.html 各转一遍人工看），别假设 pandoc/playwright 开箱即完美。
- **大报告/含图导出耗时**：异步 + 超时 + 缓存。
- **安全**：转换在服务端跑 LLM 产出的 HTML（已 sanitize），但 Playwright 渲染要确保沙箱/无外联。

## 实施前 spike（不可跳）
1. 拿 thread `73b41dc3` 修好图+样式后的 report.html，分别用 Playwright `page.pdf()`、`pandoc html docx`、`pandoc html latex` 各转一次，**人工评估**图/表/样式还原度。
2. 评估镜像体积增量，定"进主镜像 vs 单独 service vs 按需"。
3. 据 spike 结果再细化本 spec 的最终选型与端点设计。

## 验收
- 三种格式各能导出、能被对应软件正常打开（PDF 可读、docx 在 Word 打开有格式、tex 能编译）。
- 报告里的图在 PDF/Word 中可见（依赖图修复先落地）。
- 部署镜像体积增量在可接受范围（给出数字）。
- 端点有 TDD（mock 转换器 + 真 pandoc smoke）。

## 不做（本期）
- 不做在线富文本编辑后导出（只导出 agent 产出的报告）。
- 不做自定义期刊模板系统（先固定一套样式；reference.docx 模板可作后续增强）。
