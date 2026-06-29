"use client";

import { ChevronDownIcon, ChevronRightIcon, DownloadIcon, FileTextIcon } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { HTMLContent } from "@/components/workspace/artifacts/html-content";
import { MarkdownContent } from "@/components/workspace/messages/markdown-content";
import { loadArtifactContent } from "@/core/artifacts/loader";
import { type ArtifactMeta } from "@/core/artifacts/types";
import { reportExportURL, urlOfArtifact } from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";

/**
 * 报告产物判定：report.md 可能是 lead present 的裸 string（normalize 成 {path}，kind 缺失），
 * 也可能来自 /artifacts/reports 端点（kind="report"）。按 kind 或 .md/.html 扩展名判。
 */
export function isReportArtifact(meta: ArtifactMeta): boolean {
  if (meta.kind === "report") return true;
  const p = meta.path.toLowerCase();
  return p.endsWith(".md") || p.endsWith(".html") || p.endsWith(".htm");
}

/** spec 2026-06-29: HTML 报告（report.html）走 HTMLContent，旧 md 报告仍走 MarkdownContent。 */
function isHtmlReportPath(path: string): boolean {
  const p = path.toLowerCase();
  return p.endsWith(".html") || p.endsWith(".htm");
}

/**
 * 报告卡：点开懒拉报告文本用 MarkdownContent 渲染（含 citation 链接、图片路径规范化，
 * 与对话流渲染同构）。资产面板 + （历史）对话流摘要共用此组件，SSOT 一处。
 */
export function ReportCard({ meta, threadId }: { meta: ArtifactMeta; threadId: string }) {
  const { t } = useI18n();
  const [expanded, setExpanded] = useState(false);
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleToggle() {
    const next = !expanded;
    setExpanded(next);
    if (next && content === null) {
      setLoading(true);
      try {
        const { content: text } = await loadArtifactContent({ filepath: meta.path, threadId });
        setContent(text);
      } catch {
        setContent("");
      } finally {
        setLoading(false);
      }
    }
  }

  // 空内容守护：报告文本为空时不渲染 MarkdownContent 外壳（否则出现一个带下载/复制图标的
  // 空白 box——dogfood 实测的「报告末尾空白方块」即此）。改显式提示「报告为空」。
  const hasContent = (content ?? "").trim().length > 0;

  return (
    <div className="rounded-lg border bg-background p-3">
      <div className="flex items-center gap-2">
        <FileTextIcon className="size-4 shrink-0 text-primary" />
        <span className="text-sm font-medium">{meta.filename ?? t.gallery.reportTitle}</span>
        <div className="ml-auto flex items-center gap-2">
          <Button type="button" variant="ghost" size="sm" className="gap-1" onClick={handleToggle}>
            {expanded ? <ChevronDownIcon className="size-4" /> : <ChevronRightIcon className="size-4" />}
            {t.gallery.reportOpen}
          </Button>
          {/* spec 2026-06-29-report-export-formats-impl：单一下载 → 导出菜单（HTML / PDF / Word / LaTeX）。
              HTML 走现有 download 链接；PDF/Word/LaTeX 走后端导出端点（同步返回，浏览器原生下载反馈）。 */}
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
              <DropdownMenuItem onClick={() => window.open(reportExportURL(threadId, "pdf"))}>{t.gallery.exportPdf}</DropdownMenuItem>
              <DropdownMenuItem onClick={() => window.open(reportExportURL(threadId, "docx"))}>{t.gallery.exportWord}</DropdownMenuItem>
              <DropdownMenuItem onClick={() => window.open(reportExportURL(threadId, "tex"))}>{t.gallery.exportLatex}</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      {expanded && (
        <div className="mt-3 border-t pt-3">
          {loading ? (
            <p className="text-muted-foreground text-sm">{t.gallery.reportOpen}…</p>
          ) : hasContent ? (
            isHtmlReportPath(meta.path) ? (
              <HTMLContent content={content ?? ""} className="prose prose-sm max-w-none" />
            ) : (
              <MarkdownContent content={content ?? ""} isLoading={false} className="prose prose-sm max-w-none" threadId={threadId} />
            )
          ) : (
            <p className="text-muted-foreground text-sm">{t.gallery.noArtifacts}</p>
          )}
        </div>
      )}
    </div>
  );
}
