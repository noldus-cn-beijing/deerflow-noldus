# Spike 报告：报告多格式导出（PDF / Word / LaTeX）实施前验证

> 对应 spec：[2026-06-29-report-export-formats.md](2026-06-29-report-export-formats.md)
> 日期：2026-06-29
> 方法：拿真 `report.html`（EPM thread 73b41dc3 正文 + #234 修复后形态的正确 base64 内联图 + #236 prose 样式）各转一遍，权威工具核保真，量工具体积。

## 1. 转换源

- 真正文取自 dogfood thread `73b41dc3` 的 `report.html`（EPM，4 组 × 7，含 6 段骨架 + 1 个 5×6 描述性统计表）。
- 该 thread 原报告的图是**坏的嵌套 `<img>`**（`<img src="<img src=data:image/png...">">` 经 sanitize 拆坏）——是 #234 修复**前**的旧产物。spike 手工构造了 #234 修复**后**的正确形态：`<figure><img src="data:image/png;base64,...">`（1 张真 `plot_box_open_arm.png`，19KB），以代表生产将产出的产物。
- 加了一段内联 CSS 模拟 #236 prose 样式（typography、表格边框、blockquote），让 PDF 保真评估有意义。
- 源文件：`/tmp/spike-export/report.html`（120KB）。

## 2. 转换结果（保真度）

| 目标 | 工具 | 产物 | 图保留 | 文本/表格 | 结论 |
|---|---|---|---|---|---|
| **PDF** | **Playwright `page.pdf()`** | 641KB | ✅ `pdfimages` 确认 1 张 2370×1466 @144ppi | ✅ `pdftotext` 中文/表格全在 | ✅ 保真最高（所见即所得） |
| **PDF** | **WeasyPrint** | 418KB | ✅ `pdfimages` 确认 1 张 2370×1466 @96ppi + alpha smask | ✅ 中文/表格全在 | ✅ 纯 Python、透明度处理更好 |
| **Word** | **pandoc `html→docx`** | 150KB | ✅ `word/media/rId14.png`（84KB，pandoc 原生处理 data-uri→嵌入 media） | ✅ 表格 6 行、M±SD 单元格正确 | ✅ 开箱即用 |
| **LaTeX** | **pandoc `html→latex`**（裸跑） | 6.8KB | ❌ **图静默丢弃**（产空 `\begin{figure}`，无 `\includegraphics`） | ✅ 表格→`tabular`、列表→`itemize` 正确 | ⚠️ 需预处理 |
| **LaTeX** | **pandoc + data-uri 预提取** | — | ✅ `\includegraphics{rep_chart.png}` 正确生成 | ✅ | ✅ 预处理后可用 |

### 关键风险点（spec 点名）的结论

- **base64 data-uri 在各格式的处理**（spec 最大风险）：
  - **docx**：pandoc 原生支持，data-uri 自动落成 `word/media/*.png`，无需预处理。✅
  - **PDF（Playwright/WeasyPrint）**：两者都渲染 `<img src="data:...">`，图正确进 PDF（`pdfimages` 实证）。✅
  - **LaTeX**：pandoc 的 latex writer **不支持 data-uri**（`\includegraphics` 要文件路径，且不会像 docx writer 那样把 data-uri 落盘）→ **图静默丢、产空 figure**。❌
- **LaTeX 图丢的治法（已验证）**：pandoc 前先把 base64 data-uri 还原成临时 `.png` 文件、`src` 改相对路径、pandoc 跑时传 `--resource-path` → 正确产 `\includegraphics{rep_chart.png}`。✅ 这条预提取逻辑是导出后端 LaTeX 路径的**必经步骤**。

## 3. 工具体积（部署镜像增量）

| 方案 | 增量（粗略，未扣镜像已有通用库） |
|---|---|
| **pandoc**（`pypandoc_binary` 自带二进制，docx+tex 共用） | **~155MB** |
| **WeasyPrint** + 纯 py 依赖（pydyf/tinycss2/tinyhtml5/cssselect2/pyphen/fonttools/zopfli） | **~12MB**（brotli 等通用库多半镜像已有，未计） |
| **Playwright** py 包 | **~137MB** |
| **chromium full** | **~379MB** |
| **chromium headless-shell**（PDF 只需这个） | **~64MB** |

### 选型增量对比

- **方案 A：WeasyPrint（PDF）+ pandoc（docx/tex）** ≈ **+167MB**（155 pandoc + 12 weasyprint）
- **方案 B：Playwright + headless-shell（PDF）+ pandoc（docx/tex）** ≈ **+356MB**（155 + 137 + 64）
- **方案 B'：复用镜像已有的 e2e chromium（PDF）+ pandoc** ≈ **+292MB**（155 + 137，chromium 增量 0）

> 本机 `~/.cache/ms-playwright/` 已有 chromium（e2e 测试用），但生产**部署镜像**是否已带 chromium 取决于 Dockerfile——需核 `packages/agent/` 的镜像构建链。若 e2e chromium 已在镜像，Playwright 路径的边际成本仅 +137MB（py 包）。

## 4. 选型建议（供 spec 终审）

### PDF：**WeasyPrint 优先，Playwright 备选**

- **WeasyPrint 胜出**：纯 Python、无浏览器进程（部署/运维简单、无沙箱安全面）、+12MB 几乎零成本、CSS 分页好、透明度（smask）处理优于 Playwright。
- **Playwright 备选**：保真理论上最高（所见即所得、和前端展现一致），但 +137MB~+516MB 成本高、需跑 chromium 进程（安全沙箱、内存）。
- **决策点**：若 WeasyPrint 对真报告的 prose 样式/分页还原足够好（本次 spike 的简单 CSS 通过），选 WeasyPrint；若复杂样式（多列、特殊字体）还原不达标，退 Playwright。**实施时需用一份图正确内联的真 report.html 再各转一次人工比对视觉**（本次 spike 受限于无「图正确」真样本，用构造样本代）。

### Word + LaTeX：**pandoc 一个依赖覆盖两种**

- **Word**：开箱即用，data-uri 原生处理。✅
- **LaTeX**：**必须加 data-uri 预提取步骤**（base64→临时 .png + `--resource-path`），否则图丢。这是后端转换器 LaTeX 分支的硬要求。

## 5. 端点设计（据 spike 细化 spec §后端）

```
GET /threads/{tid}/artifacts/report/export?format=pdf|docx|tex
```

- 读 thread 的 `outputs/report.html`（转换源 = HTML single source）。
- 转换器封装在一处（SSOT），三种格式分支：
  - `pdf` → WeasyPrint（`HTML(filename=...).write_pdf()`）
  - `docx` → pandoc（`pypandoc.convert_file(src, "docx")`）
  - `tex` → 预处理（data-uri→临时 .png）→ pandoc（`convert_file(src, "latex", extra_args=["--resource-path", tmp])`）
- 返回 `Content-Disposition: attachment`（与现有 artifacts 路由的下载语义一致）。
- 缓存：同 report 同格式不重复转（按 report.html 内容 hash + format 作 key）。
- 大报告/含图耗时：同步即可（本次 120KB 含图 <1s）；超时/异步留作后续增强（spec 验收未要求）。

## 6. 前置依赖状态（spec line 11-13）

- ✅ **样式**（`2026-06-29-fix-html-report-styling` / #236）：**已合 dev**——prose typography 已装。
- ⚠️ **图内联**（`2026-06-29-fix-html-report-inline-img` / #234）：**逻辑已合 dev**，但 dogfood thread `a2a14b8f` 暴露**间歇性丢图 bug**（lead 没把 chart handoff 路径塞给 report-writer → 报告零图，假成功藏全绿下）。**PR #237 加 seal 确定性 fail-loud 门**修复（让漏放可见）。导出功能依赖报告有图，故 #237 应先于导出实施落地。

## 7. 待人工确认（spike 局限）

1. **视觉保真**：本次无「图正确内联」的真 report.html 样本（磁盘上 a2a14b8f 零图、73b41dc3 图坏），用构造样本代。实施前需用 #237 修复后新生成的报告各转一次、人工肉眼比对 PDF/docx 视觉。
2. **复杂表格/多图**：本次只 1 表 1 图。多表、嵌套表、>3 张代表性图的还原度待真样本验证。
3. **生产镜像是否已带 chromium**：需核 Dockerfile / `make deploy-tar` 镜像清单，定 Playwright 路径真实增量。
