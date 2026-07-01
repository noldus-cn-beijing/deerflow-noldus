"use client";

import { ChevronDownIcon, ChevronRightIcon, ColumnsIcon, XIcon } from "lucide-react";
import { useMemo, useState } from "react";


import { Button } from "@/components/ui/button";
import { isImageArtifact, type ArtifactMeta } from "@/core/artifacts/types";
import { urlOfArtifact } from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";
import type { AgentThreadState } from "@/core/threads";
import { cn } from "@/lib/utils";

import { STATUS_ICON } from "../../kit/status-badge";

import { GalleryFacetBar } from "./gallery-facet-bar";
import { GalleryGrid } from "./gallery-grid";
import { GalleryLightbox } from "./gallery-lightbox";
import { useGalleryFilters } from "./use-gallery-filters";

interface ArtifactGalleryProps {
  artifacts: ArtifactMeta[];
  threadId: string;
  chartsStatus?: AgentThreadState["charts_status"];
}

export function ArtifactGallery({ artifacts, threadId, chartsStatus }: ArtifactGalleryProps) {
  const { t } = useI18n();
  const g = t.gallery;

  const images = artifacts.filter(isImageArtifact);
  const { filters, setFilter, filteredMetas, facetOptions } = useGalleryFilters(images);

  const [perSubjectExpanded, setPerSubjectExpanded] = useState(false);
  // spec 2026-06-29 问题3：per_subject 按 chart_type 二次分组，每类型一个独立折叠子区。
  // 各子区折叠态独立（box 展开不影响 trajectory），虚拟化也按子区各自触发。
  const [expandedTypes, setExpandedTypes] = useState<Set<string>>(new Set());
  const [compareMode, setCompareMode] = useState(false);
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const [lightboxMeta, setLightboxMeta] = useState<ArtifactMeta | null>(null);

  const aggregateMetas = filteredMetas.filter((m) => m.output_mode === "aggregate" || !m.output_mode);
  const perSubjectMetas = filteredMetas.filter((m) => m.output_mode === "per_subject");

  // per_subject 按 chart_type 分组（spec 2026-06-29 问题3）。
  // 有 chart_type 的图进对应类型子区；无类型的图（chart_type 空）不建类型子区，
  // 单独留在 untyped 列表里平铺渲染（守「空类型不建子区」且不让它们消失）。
  const perSubjectByType = useMemo(() => {
    const groups = new Map<string, ArtifactMeta[]>();
    const untyped: ArtifactMeta[] = [];
    for (const m of perSubjectMetas) {
      const t = m.chart_type?.trim();
      if (!t) {
        untyped.push(m);
      } else {
        const arr = groups.get(t);
        if (arr) arr.push(m);
        else groups.set(t, [m]);
      }
    }
    // 稳定排序：按类型名字母序，便于眼睛定位。
    const sorted = [...groups.entries()].sort(([a], [b]) => a.localeCompare(b));
    return { typed: sorted, untyped };
  }, [perSubjectMetas]);

  function toggleType(type: string) {
    setExpandedTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  }

  const failedCount = (chartsStatus?.failed?.length ?? 0) + (chartsStatus?.remaining?.length ?? 0);

  function handleSelect(path: string) {
    setSelectedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }

  function handleOpen(meta: ArtifactMeta) {
    setLightboxMeta(meta);
  }

  const compareModeItems = filteredMetas.filter((m) => selectedPaths.has(m.path));

  return (
    <div className="flex flex-col gap-4">
      <GalleryFacetBar
        filters={filters}
        facetOptions={facetOptions}
        totalCount={filteredMetas.length}
        onFilter={setFilter}
        threadId={threadId}
      />

      {failedCount > 0 && (
        // 失败图表数提示：颜色/图标走 kit danger SSOT（status→token 单一来源）。
        // spec D2 Task6：只动视觉状态外壳，不碰 6 条 dogfood 不变式（chart_type 分组 /
        // per_subject 折叠 / aggregate 段 / lightbox / compare / ZIP）。
        // border-/bg- 带透明度用与 kit BAR_CLASS 同源的 D1 token（bg-status-danger），
        // Tailwind 静态抽取需字面量，故不动态拼。
        <div className="flex items-start gap-2 rounded-lg border border-status-danger/40 bg-status-danger/10 p-2 text-sm">
          {(() => {
            const FailedIcon = STATUS_ICON.danger;
            return <FailedIcon className="text-status-danger mt-0.5 size-4 shrink-0" />;
          })()}
          <div className="min-w-0">
            <div className="font-medium">{g.failedGenerated(failedCount)}</div>
            <ul className="text-muted-foreground mt-1 space-y-0.5 text-xs">
              {(chartsStatus?.failed ?? []).slice(0, 5).map((f, i) => (
                <li key={`f-${i}`} className="truncate">{f.chart_id}: {f.reason}</li>
              ))}
              {(chartsStatus?.remaining ?? []).slice(0, 3).map((r, i) => (
                <li key={`r-${i}`} className="truncate">{r.chart_id}: {r.reason}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {aggregateMetas.length > 0 && (
        <section>
          <div className="mb-2 flex items-center gap-2">
            <h2 className="text-sm font-medium">{g.aggregate} ({aggregateMetas.length})</h2>
            <div className="ml-auto flex gap-2">
              <Button
                size="sm"
                variant={compareMode ? "secondary" : "outline"}
                onClick={() => {
                  setCompareMode((v) => !v);
                  setSelectedPaths(new Set());
                }}
                className="h-7 gap-1 text-xs"
              >
                <ColumnsIcon className="size-3.5" />
                {compareMode ? g.exitCompare : g.compareMode}
              </Button>
            </div>
          </div>

          <GalleryGrid
            items={aggregateMetas}
            threadId={threadId}
            selectedPaths={selectedPaths}
            onSelect={compareMode ? handleSelect : undefined}
            onOpen={compareMode ? undefined : handleOpen}
            compareMode={compareMode}
          />
        </section>
      )}

      {perSubjectMetas.length > 0 && (
        <section>
          <button
            type="button"
            // 折叠入口可发现性（spec 2026-06-26）：保持默认折叠（防图墙），但把入口
            // 做成明显的可点 affordance——边框 + hover 抬升 + 圆角，文案带「展开 + 计数」
            // 动作提示。日式克制：是「明确可点」，不是「花哨大按钮」。
            className={cn(
              "mb-2 flex w-full items-center gap-2 rounded-lg border bg-card px-3 py-2 text-left text-sm font-medium shadow-rest",
              "duration-fast ease-[var(--ease-brand-out)]",
              "hover:shadow-raised hover:bg-accent/50",
              "focus-visible:ring-ring focus-visible:outline-none focus-visible:ring-2",
            )}
            onClick={() => setPerSubjectExpanded((v) => !v)}
            aria-expanded={perSubjectExpanded}
            title={perSubjectExpanded ? g.collapsePerSubject : g.expandPerSubject(perSubjectMetas.length)}
          >
            {perSubjectExpanded ? (
              <ChevronDownIcon className="size-4 shrink-0 text-muted-foreground" />
            ) : (
              <ChevronRightIcon className="size-4 shrink-0 text-muted-foreground" />
            )}
            <span className="truncate">
              {perSubjectExpanded
                ? `${g.perSubject} (${perSubjectMetas.length}) · ${g.collapsePerSubject}`
                : g.expandPerSubject(perSubjectMetas.length)}
            </span>
          </button>

          {perSubjectExpanded && (
            <div className="flex flex-col gap-3">
              {perSubjectByType.typed.map(([type, metas]) => {
                const isOpen = expandedTypes.has(type);
                return (
                  <div key={type} className="rounded-lg border bg-card/40 p-2">
                    <button
                      type="button"
                      // spec 2026-06-29 问题3：每类型一个独立折叠子区。data-chart-subsection
                      // 让测试/无障碍能定位子区；aria-expanded 体现各自独立折叠态。
                      data-chart-subsection={type}
                      aria-expanded={isOpen}
                      className={cn(
                        "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm font-medium",
                        "duration-fast ease-[var(--ease-brand-out)]",
                        "hover:bg-accent/50",
                        "focus-visible:ring-ring focus-visible:outline-none focus-visible:ring-2",
                      )}
                      onClick={() => toggleType(type)}
                    >
                      {isOpen ? (
                        <ChevronDownIcon className="size-4 shrink-0 text-muted-foreground" />
                      ) : (
                        <ChevronRightIcon className="size-4 shrink-0 text-muted-foreground" />
                      )}
                      <span className="truncate capitalize">{type} ({metas.length})</span>
                    </button>
                    {isOpen && (
                      <GalleryGrid
                        items={metas}
                        threadId={threadId}
                        selectedPaths={selectedPaths}
                        onSelect={compareMode ? handleSelect : undefined}
                        onOpen={compareMode ? undefined : handleOpen}
                        compareMode={compareMode}
                      />
                    )}
                  </div>
                );
              })}

              {perSubjectByType.untyped.length > 0 && (
                // 无 chart_type 的图不建类型子区，但仍可见（平铺在末尾，不丢图）。
                <GalleryGrid
                  items={perSubjectByType.untyped}
                  threadId={threadId}
                  selectedPaths={selectedPaths}
                  onSelect={compareMode ? handleSelect : undefined}
                  onOpen={compareMode ? undefined : handleOpen}
                  compareMode={compareMode}
                />
              )}
            </div>
          )}
        </section>
      )}

      {compareMode && selectedPaths.size > 0 && (
        <div className="bg-background/80 fixed bottom-4 left-1/2 z-40 flex -translate-x-1/2 items-center gap-3 rounded-xl border px-4 py-2 shadow-lg backdrop-blur">
          <span className="text-sm font-medium">{selectedPaths.size} selected</span>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => setSelectedPaths(new Set())}
            className="gap-1"
          >
            <XIcon className="size-3.5" />
            {g.clearFilters}
          </Button>
          {compareModeItems.length >= 2 && (
            <div className="grid grid-cols-2 gap-2 sm:flex sm:flex-wrap">
              {compareModeItems.slice(0, 4).map((item) => {
                const src = item.thumb_path
                  ? urlOfArtifact({ filepath: item.thumb_path, threadId })
                  : urlOfArtifact({ filepath: item.path, threadId });
                return (
                  <img
                    key={item.path}
                    src={src}
                    alt={item.chart_id ?? item.path}
                    className="h-16 w-16 rounded object-contain border"
                    decoding="async"
                  />
                );
              })}
            </div>
          )}
        </div>
      )}

      <GalleryLightbox
        open={lightboxMeta !== null}
        meta={lightboxMeta}
        allItems={filteredMetas}
        threadId={threadId}
        onClose={() => setLightboxMeta(null)}
        onNavigate={setLightboxMeta}
      />
    </div>
  );
}
