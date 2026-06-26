"use client";

import { AlertTriangleIcon, ChevronDownIcon, ChevronRightIcon, DownloadIcon, FileTextIcon, ImagesIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { MarkdownContent } from "@/components/workspace/messages/markdown-content";
import { useChartArtifacts } from "@/core/artifacts/hooks";
import { loadArtifactContent } from "@/core/artifacts/loader";
import {
  isImageArtifact,
  normalizeArtifacts,
  selectRepresentativeCharts,
  type ArtifactInput,
  type ArtifactMeta,
} from "@/core/artifacts/types";
import {
  archiveArtifactsURL,
  urlOfArtifact,
} from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

/**
 * 聊天流 inline 产物摘要（spec 2026-06-24-frontend-phase0-3-artifact-gallery §3.2）。
 *
 * 第 0 层（零渲染压力）：只 inline aggregate 代表图（通常 ≤6），其余全进画廊。
 * 两个并列主动作（§3.1.7 分层，不分主次）：
 *   - 「打开画廊」→ 第 2 层自由探索（独立路由，触发渲染）
 *   - 「下载全部 ZIP」→ 第 1 层带走全部（零渲染，浏览器直下）
 * 失败/截断显式呈现（守「无声截断」铁律）。
 *
 * 数据源（spec 2026-06-26-conversation-gallery-empty §一修复）：图元数据走 **磁盘端点**
 * `/artifacts/charts`（useChartArtifacts），与 /gallery 独立页同源——不再吃 thread.values.artifacts
 * （state 冒泡，chart-maker subagent 的 artifacts 不上行到 lead，恒空）。磁盘端点空/失败时回退
 * props.artifacts（lead present 的代表图仍在 state，至少 1 张）。报告（report.md）仍从 props
 * artifacts 里认（它是 lead present 的裸 string，不在磁盘 charts 端点里）。
 *
 * 触发时机：present-files 消息出现即挂载本组件→拉一次；run 完成（onFinish）后由 message-list
 * 传 refetchSignal 触发 refetch 补全量（图陆续落盘）。不每帧拉、不轮询（性能）。
 */
export function InlineArtifactSummary({
  threadId,
  artifacts,
  chartsStatus,
  className,
  refetchSignal,
}: {
  threadId: string;
  artifacts: ArtifactInput[] | null | undefined;
  chartsStatus?:
    | {
        n_rendered?: number;
        failed?: { chart_id: string; reason: string }[];
        remaining?: { chart_id: string; reason: string }[];
      }
    | null;
  className?: string;
  /**
   * run 完成（onFinish）后递增 → 触发 refetch 拉全量图（spec §一触发时机）。
   * message-list 把 run 完成事件折成一个递增序号传进来，避免每帧拉。
   */
  refetchSignal?: number;
}) {
  const { t } = useI18n();
  const router = useRouter();

  // 磁盘端点是图的唯一真相（spec §一）；state（props.artifacts）作回退 + 报告来源。
  const fallbackArtifacts = useMemo(() => normalizeArtifacts(artifacts), [artifacts]);
  const { artifacts: diskArtifacts, loaded, refetch } = useChartArtifacts(threadId, {
    fallbackArtifacts,
  });
  // run 完成（refetchSignal 变化）后补拉一次全量（图陆续落盘，首轮可能不全）。
  useEffect(() => {
    if (refetchSignal !== undefined && refetchSignal > 0) {
      void refetch();
    }
  }, [refetchSignal, refetch]);

  // 报告卡（report.md）走 state（lead present 的裸 string，不在磁盘 charts 端点）；图走磁盘。
  const report = useMemo(() => fallbackArtifacts.find(isReportArtifact) ?? null, [fallbackArtifacts]);
  // 加载中先用回退（state 代表图）撑住首屏，避免闪烁；加载完用磁盘全量。
  const metas = loaded ? diskArtifacts : fallbackArtifacts;
  const images = useMemo(() => metas.filter(isImageArtifact), [metas]);
  const representatives = useMemo(
    () => selectRepresentativeCharts(metas).slice(0, 6),
    [metas],
  );

  const nAggregate = images.filter((m) => m.output_mode === "aggregate").length;
  const nPerSubject = images.filter((m) => m.output_mode === "per_subject").length;
  const nTotal = images.length;

  const failedCount = (chartsStatus?.failed?.length ?? 0) + (chartsStatus?.remaining?.length ?? 0);

  if (nTotal === 0 && failedCount === 0 && !report) {
    return null;
  }

  return (
    <div
      className={cn(
        "flex w-full flex-col gap-3 rounded-xl border bg-muted/20 p-3",
        className,
      )}
    >
      {/* 报告卡（spec §2 方案 A：对话流内嵌报告，研究员最关心，主路径可见） */}
      {report && <ReportCard meta={report} threadId={threadId} />}

      {/* 摘要行（tabular-nums，对齐数字） */}
      <div className="text-muted-foreground text-sm tabular-nums">
        {nAggregate > 0 && <span className="mr-3">{t.gallery.summaryCharts(nAggregate)}</span>}
        {nPerSubject > 0 && <span className="mr-3">{t.gallery.summaryPerSubject(nPerSubject)}</span>}
        {nTotal > 0 && <span>{t.gallery.summaryAll(nTotal)}</span>}
      </div>

      {/* 代表图（aggregate 全部，≤6，eager；其余进画廊） */}
      {representatives.length > 0 && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {representatives.map((meta) => (
            <RepresentativeThumb key={meta.path} meta={meta} threadId={threadId} />
          ))}
        </div>
      )}

      {/* 两个并列主动作（§3.1.7：画廊第 2 层 / ZIP 第 1 层，不分主次） */}
      <div className="flex flex-wrap items-center gap-2 pt-1">
        {nTotal > 0 && (
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => router.push(`/workspace/chats/${threadId}/gallery`)}
            aria-label={t.gallery.openGallery(nTotal)}
          >
            <ImagesIcon className="size-4" />
            {t.gallery.openGallery(nTotal)}
          </Button>
        )}
        {nTotal > 0 && (
          <Button type="button" variant="outline" size="sm" asChild>
            <a
              href={archiveArtifactsURL(threadId)}
              aria-label={t.gallery.downloadAll(nTotal)}
              // 浏览器直下，零渲染（spec §3.1.7 第 1 层）。
            >
              <DownloadIcon className="size-4" />
              {t.gallery.downloadAll(nTotal)}
            </a>
          </Button>
        )}
      </div>

      {/* 失败/截断显式（不静默少图） */}
      {failedCount > 0 && (
        <div className="flex items-start gap-2 rounded-lg border border-[var(--color-status-warning)]/40 bg-[var(--color-status-warning)]/10 p-2 text-sm">
          <AlertTriangleIcon className="mt-0.5 size-4 shrink-0 text-[var(--color-status-warning)]" />
          <div className="min-w-0">
            <div className="font-medium">{t.gallery.failedGenerated(failedCount)}</div>
            <ul className="text-muted-foreground mt-1 space-y-0.5 text-xs">
              {(chartsStatus?.failed ?? []).slice(0, 5).map((f, i) => (
                <li key={`f-${i}`} className="truncate">
                  {f.chart_id}: {f.reason}
                </li>
              ))}
              {(chartsStatus?.remaining ?? []).slice(0, 3).map((r, i) => (
                <li key={`r-${i}`} className="truncate">
                  {r.chart_id}: {r.reason}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

/** 代表图缩略图：优先 thumb_path（治成本 ①），无则退化原图 + decoding=async。 */
function RepresentativeThumb({
  meta,
  threadId,
}: {
  meta: ArtifactMeta;
  threadId: string;
}) {
  const thumbUrl = meta.thumb_path
    ? urlOfArtifact({ filepath: meta.thumb_path, threadId })
    : null;
  const originalUrl = urlOfArtifact({ filepath: meta.path, threadId });
  // alt 有意义：chart_id/metric/subject 组合（非文件名）。
  const alt = [meta.chart_id, meta.metric, meta.subject].filter(Boolean).join(" · ") || meta.path;

  return (
    <a
      href={urlOfArtifact({ filepath: meta.path, threadId, download: true })}
      target="_blank"
      rel="noopener noreferrer"
      className="group relative block overflow-hidden rounded-lg border bg-background"
    >
      <img
        src={thumbUrl ?? originalUrl}
        alt={alt}
        loading="eager"
        decoding="async"
        className="aspect-square w-full object-contain transition-transform duration-200 ease-[var(--ease-brand-out)] group-hover:scale-[1.03]"
      />
    </a>
  );
}

/**
 * 报告产物判定（spec §2）：report.md 是 lead present 的裸 string，normalize 成 {path}，
 * kind 缺失。按 path 扩展名 .md 推 report（types.ts ArtifactKind 已含 "report"，但后端
 * present 报告时不带 kind，故前端按扩展名补判，不依赖后端先改 present_file）。
 */
export function isReportArtifact(meta: ArtifactMeta): boolean {
  if (meta.kind === "report") return true;
  return meta.path.toLowerCase().endsWith(".md");
}

/**
 * 报告卡（spec §2 方案 A）：对话流内嵌，点开懒拉 report.md 文本用 MarkdownContent 渲染。
 * 复用 messages/markdown-content（含 citation 链接、图片路径规范化），与对话流渲染同构。
 */
function ReportCard({ meta, threadId }: { meta: ArtifactMeta; threadId: string }) {
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

  return (
    <div className="rounded-lg border bg-background p-3">
      <div className="flex items-center gap-2">
        <FileTextIcon className="size-4 shrink-0 text-primary" />
        <span className="text-sm font-medium">{t.gallery.reportTitle}</span>
        <div className="ml-auto flex items-center gap-2">
          <Button type="button" variant="ghost" size="sm" className="gap-1" onClick={handleToggle}>
            {expanded ? <ChevronDownIcon className="size-4" /> : <ChevronRightIcon className="size-4" />}
            {t.gallery.reportOpen}
          </Button>
          <Button type="button" variant="outline" size="sm" asChild>
            <a
              href={urlOfArtifact({ filepath: meta.path, threadId, download: true })}
              target="_blank"
              rel="noopener noreferrer"
            >
              <DownloadIcon className="size-4" />
              {t.gallery.reportDownload}
            </a>
          </Button>
        </div>
      </div>
      {expanded && (
        <div className="mt-3 border-t pt-3">
          {loading ? (
            <p className="text-muted-foreground text-sm">{t.gallery.reportOpen}…</p>
          ) : (
            <MarkdownContent content={content ?? ""} isLoading={loading} className="prose prose-sm max-w-none" threadId={threadId} />
          )}
        </div>
      )}
    </div>
  );
}
