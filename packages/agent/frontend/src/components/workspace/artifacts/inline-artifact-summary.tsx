"use client";

import { AlertTriangleIcon, DownloadIcon, ImagesIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { useMemo } from "react";

import { Button } from "@/components/ui/button";
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
 * 元数据来自 thread.values.artifacts（后端 ArtifactMeta，plan_charts.json SSOT），
 * 不从消息裸路径猜分类。
 */
export function InlineArtifactSummary({
  threadId,
  artifacts,
  chartsStatus,
  className,
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
}) {
  const { t } = useI18n();
  const router = useRouter();

  const metas = useMemo(() => normalizeArtifacts(artifacts), [artifacts]);
  const images = useMemo(() => metas.filter(isImageArtifact), [metas]);
  const representatives = useMemo(
    () => selectRepresentativeCharts(metas).slice(0, 6),
    [metas],
  );

  const nAggregate = images.filter((m) => m.output_mode === "aggregate").length;
  const nPerSubject = images.filter((m) => m.output_mode === "per_subject").length;
  const nTotal = images.length;

  const failedCount = (chartsStatus?.failed?.length ?? 0) + (chartsStatus?.remaining?.length ?? 0);

  if (nTotal === 0 && failedCount === 0) {
    return null;
  }

  return (
    <div
      className={cn(
        "flex w-full flex-col gap-3 rounded-xl border bg-muted/20 p-3",
        className,
      )}
    >
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
