"use client";

import { ChevronLeftIcon, ChevronRightIcon, DownloadIcon, ExternalLinkIcon, XIcon } from "lucide-react";
import { useEffect } from "react";


import { Button } from "@/components/ui/button";
import { type ArtifactMeta } from "@/core/artifacts/types";
import { urlOfArtifact } from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";

interface GalleryLightboxProps {
  open: boolean;
  meta: ArtifactMeta | null;
  allItems: ArtifactMeta[];
  threadId: string;
  onClose: () => void;
  onNavigate: (meta: ArtifactMeta) => void;
}

export function GalleryLightbox({
  open,
  meta,
  allItems,
  threadId,
  onClose,
  onNavigate,
}: GalleryLightboxProps) {
  const { t } = useI18n();

  const currentIndex = meta ? allItems.findIndex((m) => m.path === meta.path) : -1;
  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex >= 0 && currentIndex < allItems.length - 1;

  useEffect(() => {
    if (!open) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
      } else if (e.key === "ArrowLeft" && hasPrev) {
        onNavigate(allItems[currentIndex - 1]!);
      } else if (e.key === "ArrowRight" && hasNext) {
        onNavigate(allItems[currentIndex + 1]!);
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, hasPrev, hasNext, currentIndex, allItems, onClose, onNavigate]);

  if (!open || !meta) return null;

  const originalUrl = urlOfArtifact({ filepath: meta.path, threadId });
  const downloadUrl = urlOfArtifact({ filepath: meta.path, threadId, download: true });
  const alt = [meta.chart_id, meta.metric, meta.subject].filter(Boolean).join(" · ") || meta.path;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
      onClick={onClose}
      aria-modal="true"
      role="dialog"
      aria-label={alt}
    >
      <div
        className="relative flex max-h-[90vh] max-w-[90vw] flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 按钮行永远可见：shrink-0 防它被图区挤掉（spec 2026-06-29 问题1）。 */}
        <div className="mb-2 flex shrink-0 items-center justify-end gap-2">
          <Button
            size="icon"
            variant="secondary"
            asChild
            className="size-8"
            aria-label={t.common.download}
          >
            <a href={downloadUrl} download>
              <DownloadIcon className="size-4" />
            </a>
          </Button>
          <Button
            size="icon"
            variant="secondary"
            asChild
            className="size-8"
            aria-label={t.common.openInNewWindow}
          >
            <a href={originalUrl} target="_blank" rel="noopener noreferrer">
              <ExternalLinkIcon className="size-4" />
            </a>
          </Button>
          <Button
            size="icon"
            variant="secondary"
            className="size-8"
            aria-label={t.common.close}
            onClick={onClose}
          >
            <XIcon className="size-4" />
          </Button>
        </div>

        {/* 图区占满剩余空间、按需滚动；min-h-0 让 flex 子项真能收缩，图随框自适应。 */}
        <div className="relative flex min-h-0 flex-1 items-center justify-center gap-2 overflow-auto">
          <Button
            size="icon"
            variant="secondary"
            className="size-8 shrink-0 disabled:opacity-30"
            aria-label="Previous"
            disabled={!hasPrev}
            onClick={() => hasPrev && onNavigate(allItems[currentIndex - 1]!)}
          >
            <ChevronLeftIcon className="size-5" />
          </Button>

          <img
            src={originalUrl}
            alt={alt}
            decoding="async"
            className="max-h-full max-w-full rounded-lg object-contain"
          />

          <Button
            size="icon"
            variant="secondary"
            className="size-8 shrink-0 disabled:opacity-30"
            aria-label="Next"
            disabled={!hasNext}
            onClick={() => hasNext && onNavigate(allItems[currentIndex + 1]!)}
          >
            <ChevronRightIcon className="size-5" />
          </Button>
        </div>

        {(meta.metric ?? meta.subject) && (
          <p className="text-muted-foreground mt-2 shrink-0 text-center text-sm">
            {[meta.chart_id, meta.metric, meta.subject].filter(Boolean).join(" · ")}
          </p>
        )}
      </div>
    </div>
  );
}
