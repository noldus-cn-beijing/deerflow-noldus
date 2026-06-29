"use client";

import { useThreadAssets } from "@/core/artifacts/hooks";
import { useI18n } from "@/core/i18n/hooks";
import type { AgentThreadState } from "@/core/threads";
import { cn } from "@/lib/utils";

import { ArtifactGallery } from "./gallery/artifact-gallery";
import { ReportCard } from "./report-card";

/**
 * thread 级「产出物」资产面板（右侧 Artifacts 面板内容）。
 *
 * 数据**全部来自磁盘端点**（useThreadAssets：/artifacts/charts + /artifacts/reports），
 * 完全不读 streaming 消息流 / LangGraph state.artifacts。产物因此稳定——不随 present_files
 * 消息、流式 re-render、切 tab 而出现/消失/漂移（本会话 dogfood 反复暴露的不稳定家族根治）。
 *
 * 结构：顶部「报告」区（每个 report 一张可展开 ReportCard）+ 下面「图表」区（复用 ArtifactGallery：
 * 分面筛选 + 条件虚拟化网格 + lightbox + ZIP 下载）。两者皆空 → 克制空态。
 */
export function ThreadAssetsPanel({
  threadId,
  chartsStatus,
  refetchSignal,
  className,
}: {
  threadId: string;
  chartsStatus?: AgentThreadState["charts_status"];
  /** run 完成后递增 → 补拉磁盘端点确保全量（产物陆续落盘）。 */
  refetchSignal?: number;
  className?: string;
}) {
  const { t } = useI18n();
  const { charts, reports, loaded } = useThreadAssets(threadId, { refetchSignal });

  const isEmpty = loaded && charts.length === 0 && reports.length === 0;

  return (
    <div className={cn("flex size-full flex-col gap-4 overflow-y-auto p-4", className)}>
      <header className="shrink-0">
        <h2 className="text-lg font-medium">{t.gallery.assetsTitle}</h2>
      </header>

      {isEmpty && (
        <p className="text-muted-foreground text-sm">{t.gallery.assetsEmpty}</p>
      )}

      {reports.length > 0 && (
        <section className="flex flex-col gap-2">
          <h3 className="text-muted-foreground text-xs font-medium uppercase tracking-wide">
            {t.gallery.reportsSection}
          </h3>
          <div className="flex flex-col gap-2">
            {reports.map((meta) => (
              <ReportCard key={meta.path} meta={meta} threadId={threadId} />
            ))}
          </div>
        </section>
      )}

      {charts.length > 0 && (
        <section className="flex min-h-0 flex-1 flex-col gap-2">
          <h3 className="text-muted-foreground text-xs font-medium uppercase tracking-wide">
            {t.gallery.chartsSection}
          </h3>
          <ArtifactGallery artifacts={charts} threadId={threadId} chartsStatus={chartsStatus} />
        </section>
      )}
    </div>
  );
}
