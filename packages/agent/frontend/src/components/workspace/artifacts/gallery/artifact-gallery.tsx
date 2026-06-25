"use client";

import { AlertTriangleIcon, ChevronDownIcon, ChevronRightIcon, ColumnsIcon, XIcon } from "lucide-react";
import { useState } from "react";


import { Button } from "@/components/ui/button";
import { isImageArtifact, type ArtifactMeta } from "@/core/artifacts/types";
import { urlOfArtifact } from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";
import type { AgentThreadState } from "@/core/threads";

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
  const [compareMode, setCompareMode] = useState(false);
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const [lightboxMeta, setLightboxMeta] = useState<ArtifactMeta | null>(null);

  const aggregateMetas = filteredMetas.filter((m) => m.output_mode === "aggregate" || !m.output_mode);
  const perSubjectMetas = filteredMetas.filter((m) => m.output_mode === "per_subject");

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
        <div className="flex items-start gap-2 rounded-lg border border-[var(--color-status-warning)]/40 bg-[var(--color-status-warning)]/10 p-2 text-sm">
          <AlertTriangleIcon className="mt-0.5 size-4 shrink-0 text-[var(--color-status-warning)]" />
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
            className="mb-2 flex items-center gap-1 text-sm font-medium"
            onClick={() => setPerSubjectExpanded((v) => !v)}
            aria-expanded={perSubjectExpanded}
          >
            {perSubjectExpanded ? (
              <ChevronDownIcon className="size-4" />
            ) : (
              <ChevronRightIcon className="size-4" />
            )}
            {g.perSubject} ({perSubjectMetas.length})
          </button>

          {perSubjectExpanded && (
            <GalleryGrid
              items={perSubjectMetas}
              threadId={threadId}
              selectedPaths={selectedPaths}
              onSelect={compareMode ? handleSelect : undefined}
              onOpen={compareMode ? undefined : handleOpen}
              compareMode={compareMode}
            />
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
