"use client";

import type { ReactNode } from "react";

import { useThreadAssets } from "@/core/artifacts/hooks";
import { useI18n } from "@/core/i18n/hooks";
import type { AgentThreadState } from "@/core/threads";
import { cn } from "@/lib/utils";

import { EmptyState } from "../kit/empty-state";

import { ArtifactGallery } from "./gallery/artifact-gallery";
import { MetricsTableCard } from "./metrics-table-card";
import { ReportCard } from "./report-card";

/**
 * thread 级「产出物」资产面板（右侧 Artifacts 面板内容）。
 *
 * 数据**全部来自磁盘端点**（useThreadAssets：/artifacts/charts + /reports + /data），
 * 完全不读 streaming 消息流 / LangGraph state.artifacts。产物因此稳定——不随 present_files
 * 消息、流式 re-render、切 tab 而出现/消失/漂移（本会话 dogfood 反复暴露的不稳定家族根治）。
 *
 * **最小可扩展挂载（spec 2026-06-30 C1）**：原本是「报告 / 图表」两段硬编码上下叠，
 * 加第三类产物若继续硬编码 = 加一类改一次结构。这里改成按产物类型的 section 描述符数组
 * ——第 4 类只需加一个数组项，不是新 JSX 块（不造 plugin registry，避免过度工程）。
 * 完整画廊 layout 重设计（留白/层级/分类导航/日式高级感）是 C2 的核心任务，不在 C1 做。
 */

/** 单个产物区段描述符（最小可扩展挂载单元）。 */
interface AssetSection {
  key: string;
  title: string;
  items: unknown[];
  /** 整段渲染（charts 段用 ArtifactGallery 吃整个数组，故不走 per-item）。 */
  render: () => ReactNode;
}

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
  const { charts, reports, data, loaded } = useThreadAssets(threadId, { refetchSignal });

  const isEmpty = loaded && charts.length === 0 && reports.length === 0 && data.length === 0;

  // 顺序即视觉顺序：报告 → 数据表 → 图表（图表常占大空间，放最后）。
  const sections: AssetSection[] = [
    {
      key: "reports",
      title: t.gallery.reportsSection,
      items: reports,
      render: () => (
        <div className="flex flex-col gap-2">
          {reports.map((meta) => (
            <ReportCard key={meta.path} meta={meta} threadId={threadId} />
          ))}
        </div>
      ),
    },
    {
      key: "data",
      title: t.gallery.dataSection,
      items: data,
      render: () => (
        <div className="flex flex-col gap-2">
          {data.map((meta) => (
            <MetricsTableCard key={meta.path} meta={meta} threadId={threadId} />
          ))}
        </div>
      ),
    },
    {
      key: "charts",
      title: t.gallery.chartsSection,
      items: charts,
      render: () => (
        <ArtifactGallery artifacts={charts} threadId={threadId} chartsStatus={chartsStatus} />
      ),
    },
  ].filter((s) => s.items.length > 0);

  return (
    <div className={cn("flex size-full flex-col gap-4 overflow-y-auto p-4", className)}>
      <header className="shrink-0">
        <h2 className="text-lg font-medium">{t.gallery.assetsTitle}</h2>
      </header>

      {isEmpty && <EmptyState title={t.gallery.assetsEmpty} />}

      {sections.map((section) => (
        <section
          key={section.key}
          className={cn("flex flex-col gap-2", section.key === "charts" && "min-h-0 flex-1")}
          data-testid={`assets-section-${section.key}`}
        >
          <h3 className="text-muted-foreground text-xs font-medium uppercase tracking-wide">{section.title}</h3>
          {section.render()}
        </section>
      ))}
    </div>
  );
}
