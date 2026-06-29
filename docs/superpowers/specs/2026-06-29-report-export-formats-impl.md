# Spec (实施): 报告多格式导出 —— PDF / Word / LaTeX

> 状态：**可实施**（实施前 spike 已完成，选型与端点已定，见 [spike](2026-06-29-report-export-formats-spike.md) + 上游 feature spec [2026-06-29-report-export-formats.md](2026-06-29-report-export-formats.md)）
> 归属：报告体验（2026-06-29）
> 前置依赖：✅ 样式 #236 已合 dev；✅ 图内联 fail-loud #237 已合 dev（导出依赖报告有图，#237 让间歇性丢图可见）

## 〇、给实施 agent 的一句话

在后端加一个 `GET /threads/{tid}/artifacts/report/export?format=pdf|docx|tex` 端点，用 WeasyPrint（PDF）+ pandoc（docx/tex，LaTeX 分支须预处理 base64 data-uri）把线程的 `outputs/report.html` 转成对应格式返回 attachment；前端把报告卡的单一「下载」按钮改成「导出菜单」（HTML / PDF / Word / LaTeX），含 loading 态与失败提示。

---

## 一、根因 / 为什么这么做

研究员最终交付/投稿用 PDF（阅读/归档）、Word（协作/期刊模板）、LaTeX（学术排版）。当前只能下原始 `.html`（spec 上游已论证）。spike 实证：

- **HTML 是单一转换源**（自包含、base64 内联图、结构化语义）——后端不重新生成报告，只转换现有 `report.html`。
- **PDF 选 WeasyPrint**（纯 Python、+12MB、无浏览器进程/沙箱、CSS 分页好、透明度 smask 优）；Playwright 保真更高但 +200MB+ 且需 chromium 进程，spike 判 WeasyPrint 对当前 prose 样式足够。
- **docx/tex 选 pandoc**（一个依赖覆盖两种）。
- **LaTeX 的坑**：pandoc 的 latex writer **不支持 data-uri**（`\includegraphics` 要文件路径，且不会像 docx writer 那样自动落盘）→ **裸跑静默丢图**。spike 验证的治法：pandoc 前把 base64 data-uri 解码成临时 `.png`、`src` 改相对路径、传 `--resource-path`。

## 二、设计要点

- **转换器封装在一处（SSOT）**：`ReportExporter` 纯函数模块，三种格式分支 + LaTeX 预处理，可单测、可缓存。
- **复用现有 artifacts 路由模式**：`@require_permission("threads","read",owner_check=True)` + `_build_attachment_headers`（RFC 5987 中文文件名）+ `Response(content=bytes)`。不新造权限/响应工具。
- **惰性 import 转换依赖**（守导入环铁律）：`weasyprint` / `pypandoc` 的 import 放函数体内，模块顶层不引——否则可能闭成 harness/app 导入环（CLAUDE.md §导入环风险）。
- **缓存**：同 report 同格式不重复转（report.html mtime + format 作 key，进程内 LRU）。首版可先不缓存（同步转换 <1s，spike 实测），留接口。
- **同步转换**：120KB 含图报告 <1s（spike 实测），首版同步即可；异步/超时留后续增强（不在本 spec 验收）。

---

## 三、改动清单

### 3.1 新增依赖（pyproject）

文件：`packages/agent/backend/packages/harness/pyproject.toml`（或 backend 根 `pyproject.toml`，以 weasyprint/pypandoc 实际该声明的位置为准——实施时 `grep` 现有依赖段确认）

- 新增 `weasyprint>=69`（PDF）
- 新增 `pypandoc_binary>=1.17`（docx/tex；`_binary` 变体自带 pandoc 二进制，免系统装 pandoc，部署镜像确定性更好）

> ⚠️ 部署镜像影响：pypandoc_binary ~155MB、weasyprint ~12MB。实施时核 `Dockerfile` / `make deploy-tar`，确认这两个 wheel 进了 backend 镜像（守镜像瘦身纪律，但这是导出能力的必要成本，spike 已量化）。

### 3.2 新增转换器模块（SSOT，纯函数 + 惰性 import）

**新文件**：`packages/agent/backend/packages/harness/deerflow/tools/builtins/report_exporter.py`

```python
"""报告多格式导出转换器（spec 2026-06-29-report-export-formats-impl）。

把 report.html 转成 PDF / docx / tex。纯函数 + 惰性 import（守导入环铁律：
weasyprint / pypandoc 的 import 全放函数体内，模块顶层不引）。

选型据 spike：PDF=WeasyPrint、docx/tex=pandoc；LaTeX 分支必须预处理 base64
data-uri（pandoc latex writer 不支持 data-uri，裸跑静默丢图）。
"""
from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path

# {{img:}} 占位符与 seal_handoff_tools 同源正则——但导出面对的是已 seal 的 report.html，
# 图已是 data:image/png;base64,... 形态（占位符早已被 seal 解析）。这里只处理 data-uri。
_DATA_URI_RE = re.compile(r"data:image/(png|jpeg|jpg);base64,([A-Za-z0-9+/=]+)")

# 三种导出格式
EXPORT_FORMATS = ("pdf", "docx", "tex")

# 格式 → (mime_type, extension, 下载文件名后缀)
_FORMAT_META = {
    "pdf": ("application/pdf", ".pdf"),
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
    "tex": ("application/x-tex", ".tex"),
}


class ExportError(Exception):
    """转换失败（依赖缺失 / pandoc 报错 / 源 HTML 不可读）。"""


def export_report(report_html_path: Path, fmt: str, *, base_filename: str = "report") -> tuple[bytes, str, str]:
    """转 report.html → 指定格式。

    Args:
        report_html_path: 已 seal 的 report.html 物理路径（含 base64 内联图）。
        fmt: "pdf" | "docx" | "tex"。
        base_filename: 下载文件名主干（不含扩展名），默认 "report"。

    Returns:
        (content_bytes, mime_type, filename) —— 供端点构造 Response + Content-Disposition。

    Raises:
        ExportError: 源文件不存在 / 格式非法 / 转换器报错。
    """
    if fmt not in EXPORT_FORMATS:
        raise ExportError(f"unsupported export format: {fmt!r} (allowed: {EXPORT_FORMATS})")
    if not report_html_path.is_file():
        raise ExportError(f"report.html not found: {report_html_path}")

    if fmt == "pdf":
        content = _to_pdf(report_html_path)
    elif fmt == "docx":
        content = _to_docx(report_html_path)
    else:  # tex
        content = _to_tex(report_html_path)

    mime, ext = _FORMAT_META[fmt]
    return content, mime, f"{base_filename}{ext}"


def _read_html(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        raise ExportError(f"report.html unreadable: {e}") from e


def _to_pdf(report_html_path: Path) -> bytes:
    """PDF via WeasyPrint（spike 选型）。惰性 import。"""
    try:
        from weasyprint import HTML  # 惰性：守导入环 + 缺依赖时清晰报错
    except ImportError as e:
        raise ExportError("weasyprint not installed (pip install weasyprint)") from e
    try:
        return HTML(filename=str(report_html_path)).write_pdf()
    except Exception as e:
        raise ExportError(f"weasyprint PDF conversion failed: {e}") from e


def _to_docx(report_html_path: Path) -> bytes:
    """docx via pandoc。data-uri pandoc 原生处理（落 word/media），无需预处理。惰性 import。"""
    try:
        import pypandoc
    except ImportError as e:
        raise ExportError("pypandoc not installed (pip install pypandoc_binary)") from e
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "report.docx"
        try:
            pypandoc.convert_file(str(report_html_path), "docx", outputfile=str(out))
        except Exception as e:
            raise ExportError(f"pandoc docx conversion failed: {e}") from e
        return out.read_bytes()


def _to_tex(report_html_path: Path) -> bytes:
    """tex via pandoc。**必须预处理 base64 data-uri**（spike：latex writer 不支持 data-uri，
    裸跑静默丢图）。把每个 data-uri 解码成临时 .png、src 改相对路径、传 --resource-path。"""
    try:
        import pypandoc
    except ImportError as e:
        raise ExportError("pypandoc not installed (pip install pypandoc_binary)") from e
    html = _read_html(report_html_path)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # 预处理：data-uri → 临时 png + 相对 src
        def _extract(m: re.Match[str]) -> str:
            ext, b64 = m.group(1), m.group(2)
            ext = "jpg" if ext == "jpeg" else ext
            name = f"img_{abs(hash(m.group(0))) % 10**6}.{ext}"
            (td_path / name).write_bytes(base64.b64decode(b64))
            return name

        preprocessed = _DATA_URI_RE.sub(_extract, html)
        src_in_td = td_path / "report.html"
        src_in_td.write_text(preprocessed, encoding="utf-8")
        out = td_path / "report.tex"
        try:
            pypandoc.convert_file(
                str(src_in_td), "latex", outputfile=str(out),
                extra_args=[f"--resource-path={td}"],
            )
        except Exception as e:
            raise ExportError(f"pandoc tex conversion failed: {e}") from e
        return out.read_bytes()
```

### 3.3 新增导出端点（artifacts router）

文件：`packages/agent/backend/app/gateway/routers/artifacts.py`

在 `get_report_artifacts`（line 371）之后、catch-all `get_artifact`（line 388）**之前**新增（顺序重要：catch-all 会吞掉后面的 `/artifacts/...` 路由）：

```python
# 顶部 import 区追加（惰性 import 在 report_exporter 函数体内，此处只 import 本模块自身）
from deerflow.tools.builtins.report_exporter import EXPORT_FORMATS, ExportError, export_report


@router.get(
    "/threads/{thread_id}/artifacts/report/export",
    summary="Export Report (PDF/Word/LaTeX)",
    description="Convert the thread's report.html to PDF (WeasyPrint) / docx / tex (pandoc) and return as attachment. LaTeX path pre-extracts base64 data-uris (pandoc latex writer drops them otherwise).",
)
@require_permission("threads", "read", owner_check=True)
async def export_report_artifact(
    thread_id: str, request: Request, format: str = "pdf"
) -> Response:
    """报告多格式导出（spec 2026-06-29-report-export-formats-impl）。

    读 outputs/report.html → 转 format → 返回 attachment。源文件缺失/格式非法/
    转换器报错 → 404/400/500。
    """
    fmt = format.lower()
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(status_code=400, detail=f"unsupported format: {format} (allowed: {', '.join(EXPORT_FORMATS)})")
    outputs_dir = resolve_thread_virtual_path(thread_id, _OUTPUTS_VIRTUAL_PREFIX)
    report_html = outputs_dir / "report.html"
    if not report_html.is_file():
        raise HTTPException(status_code=404, detail="report.html not found for this thread")
    try:
        content, mime, filename = export_report(report_html, fmt)
    except ExportError as e:
        raise HTTPException(status_code=500, detail=f"report export failed: {e}") from e
    return Response(
        content=content,
        media_type=mime,
        headers=_build_attachment_headers(filename),
    )
```

> 路由顺序自检：`export_report_artifact` 的路径 `/threads/{tid}/artifacts/report/export` 必须注册在 catch-all `/threads/{tid}/artifacts/{path:path}` **之前**（FastAPI 按注册顺序匹配，catch-all 会吞）。现有 `get_report_artifacts`（`/artifacts/reports`）已是这个模式，照搬即可。

### 3.4 前端：报告卡「下载」→「导出菜单」

文件：`packages/agent/frontend/src/components/workspace/artifacts/report-card.tsx`

把第 70-79 行的单一下载 `<Button>` 换成下拉菜单（HTML 原文 / PDF / Word / LaTeX）。用 shadcn `DropdownMenu`（`ui/dropdown-menu.tsx`，copy-in 可组合）。点 PDF/Word/LaTeX → `window.open(导出端点 URL)`；点 HTML → 现有 `urlOfArtifact({download:true})`。

```tsx
// 顶部 import 追加
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";

// 导出端点 URL 构造（与 urlOfArtifact 同基础，backend base 由现有 API 客户端定）
function exportUrl(threadId: string, format: "pdf" | "docx" | "tex"): string {
  // 复用现有 backend base 解析；若 urlOfArtifact 不支持自定义路径，按其模式构造：
  //   /api/threads/{tid}/artifacts/report/export?format=...
  // 实施时核 urlOfArtifact 是否可复用，否则照其实现抽一个 exportUrl helper。
  return `/api/threads/${threadId}/artifacts/report/export?format=${format}`;
}

// 替换原 <Button>...</Button> 下载块为：
<DropdownMenu>
  <DropdownMenuTrigger asChild>
    <Button type="button" variant="outline" size="sm" className="gap-1">
      <DownloadIcon className="size-4" />
      {t.gallery.reportExport}
    </Button>
  </DropdownMenuTrigger>
  <DropdownMenuContent align="end">
    <DropdownMenuItem asChild>
      <a href={urlOfArtifact({ filepath: meta.path, threadId, download: true })} target="_blank" rel="noopener noreferrer">
        {t.gallery.exportHtml}
      </a>
    </DropdownMenuItem>
    <DropdownMenuItem onClick={() => window.open(exportUrl(threadId, "pdf"))}>{t.gallery.exportPdf}</DropdownMenuItem>
    <DropdownMenuItem onClick={() => window.open(exportUrl(threadId, "docx"))}>{t.gallery.exportWord}</DropdownMenuItem>
    <DropdownMenuItem onClick={() => window.open(exportUrl(threadId, "tex"))}>{t.gallery.exportLatex}</DropdownMenuItem>
  </DropdownMenuContent>
</DropdownMenu>
```

> loading 态/失败提示：`window.open` 触发浏览器下载，PDF/Word 转换可能耗时 1-2s。首版用浏览器原生下载反馈即可（导出端点同步返回，浏览器转圈）。若需明确 loading 态，改成 fetch + blob 下载 + try/catch toast，但首版不做（守 simplicity，spike 实测 <1s）。

### 3.5 前端 i18n 文案（zh-CN + en-US 两齐）

文件：`packages/agent/frontend/src/core/i18n/locales/zh-CN.ts` + `en-US.ts`（`gallery` 段）

- `reportExport`: "导出" / "Export"
- `exportHtml`: "HTML（原文）" / "HTML (original)"
- `exportPdf`: "PDF" / "PDF"
- `exportWord`: "Word" / "Word"
- `exportLatex`: "LaTeX" / "LaTeX"

> 保留现有 `reportDownload`（不删，避免悬空引用——grep 确认无其他消费者再决定是否删；首版保留更安全）。

### 3.6 文档同步

- `packages/agent/backend/CLAUDE.md` 的 Artifacts router 表格：补 `/export?format=` 端点行。
- `docs/sop/` 若有报告相关 SOP，补导出能力说明（按需）。

---

## 四、测试（TDD，先红后绿）

### 4.1 后端转换器单测

**新文件**：`packages/agent/backend/tests/test_report_exporter.py`

用与 `test_seal_html_report.py` 同款 fixture（tmp_path workspace + outputs + 真 png + 构造 report.html 含 base64 图 + 表格）：

1. `test_export_pdf_returns_bytes_and_image_present` —— `export_report(report, "pdf")` 返回非空 bytes，`pdfimages` 不可用时用 `b"%PDF"` 头 + 字节大小断言（spike 实证 WeasyPrint 产合法 PDF）。
2. `test_export_docx_returns_valid_zip_with_media` —— docx 是 zip，`zipfile` 打开含 `word/media/` 且至少 1 个 png（spike 实证 data-uri 落 word/media）。
3. `test_export_tex_contains_includegraphics` —— tex 文本含 `\includegraphics`（spike 实证预处理后正确生成；这条是 LaTeX 坑的回归门——裸跑会产空 figure，测试会红）。
4. `test_export_tex_without_image_still_works` —— 无图 report 转 tex 不崩（边界）。
5. `test_unsupported_format_raises` —— `export_report(report, "xlsx")` 抛 `ExportError`。
6. `test_missing_report_raises` —— 源文件不存在抛 `ExportError`。

> **pandoc/weasyprint 是真依赖**：这些测试真跑 pandoc/weasyprint（非 mock），spike 已验证本机环境可跑。CI 需确保镜像装了这两个依赖（见 3.1）。转换器内部不 mock（守"验收看真产物"——spike 教训：pandoc data-uri 行为只能真跑才知）。

### 4.2 后端端点测试

文件：`packages/agent/backend/tests/test_artifacts_reports_endpoint.py`（已存在，追加）或新文件

7. `test_export_endpoint_pdf_returns_attachment` —— `GET /threads/{tid}/artifacts/report/export?format=pdf` 返回 200、`Content-Disposition: attachment; filename*=...report.pdf`、body 是 PDF。
8. `test_export_endpoint_bad_format_400` —— `?format=xlsx` → 400。
9. `test_export_endpoint_missing_report_404` —— 线程无 report.html → 404。
10. `test_export_endpoint_registered_before_catchall` —— 路由顺序门：确认 `/artifacts/report/export` 不被 catch-all `/artifacts/{path:path}` 吞（用 `app.routes` 检查注册顺序，或实测端点可达）。

> 用 `test_client.py` 的 TestClient 模式（现有 artifacts 端点测试同款）。owner_check 权限按现有测试的 mock 用户方式。

### 4.3 前端组件测试（vitest）

文件：`packages/agent/frontend/src/components/workspace/artifacts/report-card.test.tsx`（新建）

11. `test_renders_export_dropdown_with_four_options` —— 报告卡渲染「导出」下拉，展开含 HTML/PDF/Word/LaTeX 四项。
12. `test_pdf_item_opens_export_url` —— 点 PDF → `window.open` 被以 `?format=pdf` URL 调用（mock `window.open`）。

> 前端测试用 vitest（`npx vitest run`，CLAUDE.md "No test framework" 已过时，见 memory）。注意 shadcn DropdownMenu 的交互测试可能需 `@testing-library/user-event`，实施时核 `package.json` 是否已装（spike 未核，实施第一步确认）。

---

## 五、验收标准

1. ✅ `GET /threads/{tid}/artifacts/report/export?format=pdf|docx|tex` 三种格式各返回合法产物（PDF 可被 PDF 阅读器打开、docx 在 Word 打开有表格+图、tex 含 `\includegraphics` 可编译）——四、的测试 1-3 + 7 覆盖。
2. ✅ 报告里的图在 PDF/Word 中可见（spike 已验证 data-uri 路径；tex 经预处理保留）——测试 1-3。
3. ✅ 端点 TDD 全绿（mock 转换器边界 + 真 pandoc/weasyprint smoke）——四、全部。
4. ✅ 路由顺序正确（export 端点不被 catch-all 吞）——测试 10。
5. ✅ 前端报告卡导出菜单四选项可点、PDF 触发导出 URL——测试 11-12。
6. ✅ i18n zh-CN + en-US 两齐——四、3.5 手动 grep 核。
7. ✅ 裸导入两生产入口无导入环（`import app.gateway` + `make_lead_agent`，weasyprint/pypandoc 惰性 import）——改完必跑（CLAUDE.md 铁律）。
8. ✅ `make test`（后端）+ `pnpm check`（前端）全绿。
9. ⏳ **需人工 dogfood**：用一份 #237 修复后新生成的「图正确内联」真 report.html，三种格式各导出一次肉眼比对视觉（spike 受限于无该样本）。这条进 PR body 标 ⏳。

---

## 六、风险与注意事项（实施时守）

1. **惰性 import 铁律**：`weasyprint` / `pypandoc` 的 import **必须放 `report_exporter.py` 函数体内**，模块顶层不引。否则可能闭成 `app.gateway → report_exporter → weasyprint → ...` 的导入环，生产启动崩（conftest mock 假绿，CLAUDE.md §导入环）。改完裸导入两入口验证（验收 7）。

2. **路由顺序**：`export_report_artifact` 必须注册在 catch-all `get_artifact`（`/artifacts/{path:path}`）**之前**，否则被吞 404。现有 `get_report_artifacts` 已是这个模式，照搬（验收 4 + 测试 10）。

3. **pandoc/weasyprint 进部署镜像**：`pypandoc_binary`（~155MB）+ `weasyprint`（~12MB）必须进 backend 镜像。实施时核 `Dockerfile` / `make deploy-tar`，确认 wheel 在镜像里、运行时 `import` 不报缺。spike 已量化体积（spec 上游 line 33 验收"镜像体积增量在可接受范围"——给数字：~167MB）。

4. **LaTeX 预处理是硬要求**：`_to_tex` 必须先解码 data-uri。漏了 = 图静默丢（spike 实证）。测试 3 是这条的回归门。

5. **转换器看真产物不看 mock**：data-uri 在各格式的行为只能真跑 pandoc/weasyprint 才知（spike 教训）。转换器内部不 mock；端点测试可 mock 转换器边界（验路由/权限/响应头）。

6. **不自动注入代表性图 / 不改 report-writer**：导出只**转换**现有 report.html，不修补报告内容。报告有没有图是 #237 的职责（已落地）。导出端点读到的 report.html 是什么样就转什么样。

7. **同步转换超时**：首版同步，120KB 含图 <1s（spike 实测）。极大报告（几百张图全内联？report.html 几十 MB）可能慢——但 #234 的 report-writer prompt 限制只内联 1-3 张代表性图，正常 report.html 不会大。若实测有超时风险，加 FastAPI `BackgroundTasks` + 轮询，但**首版不做**（守 simplicity，验收未要求）。

8. **Catastrophic forgetting 自检**：改 artifacts router 是共享文件——改前 `grep` 现有端点消费者，确认新端点不破坏 catch-all / reports / charts 端点；改完跑全量 artifacts 相关测试（不只新测试）。

9. **deepseek 正面提示**：任何 prompt/文案改动用正面指令（本 spec 主要改代码，prompt 不动，风险低）。

---

## 不做（本期）

- 不做异步导出 / 进度条（同步 <1s 够用）。
- 不做转换结果缓存（首版每次实时转；留接口）。
- 不做自定义期刊模板（reference.docx 模板作后续增强，spec 上游 line 58 已划走）。
- 不做在线富文本编辑后导出（只导出 agent 产出的 report.html，spec 上游 line 57）。
- 不做 Playwright PDF 路径（选型已定 WeasyPrint；若真报告视觉不达标另起 spec）。
