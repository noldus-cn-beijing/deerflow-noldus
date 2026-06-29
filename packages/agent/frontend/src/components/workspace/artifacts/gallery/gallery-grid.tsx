"use client";

import { useVirtualizer } from "@tanstack/react-virtual";
import { useRef } from "react";


import { type ArtifactMeta } from "@/core/artifacts/types";
import { urlOfArtifact } from "@/core/artifacts/utils";
import { cn } from "@/lib/utils";

const COLS = 3;
const ROW_HEIGHT = 200;
// 小于此数量的网格直接平铺（自然高度，无内层滚动容器）。只有真正的大图集（per_subject
// 几十~上百张）才值得虚拟化。早先所有网格都套固定 `height: calc(100vh - 320px)` 的滚动容器，
// 导致只有 1 张图的「汇总图」网格也撑出近一屏高的空白——汇总图与单样本图之间巨大空隙的根因。
const VIRTUALIZE_THRESHOLD = 24;

interface GalleryGridProps {
  items: ArtifactMeta[];
  threadId: string;
  selectedPaths?: Set<string>;
  onSelect?: (path: string) => void;
  onOpen?: (meta: ArtifactMeta) => void;
  compareMode?: boolean;
}

interface ThumbCardProps {
  meta: ArtifactMeta;
  threadId: string;
  selected: boolean;
  onSelect?: (path: string) => void;
  onOpen?: (meta: ArtifactMeta) => void;
  compareMode: boolean;
}

function ThumbCard({ meta, threadId, selected, onSelect, onOpen, compareMode }: ThumbCardProps) {
  const thumbUrl = meta.thumb_path
    ? urlOfArtifact({ filepath: meta.thumb_path, threadId })
    : null;
  const originalUrl = urlOfArtifact({ filepath: meta.path, threadId });
  const alt = [meta.chart_id, meta.metric, meta.subject].filter(Boolean).join(" · ") || meta.path;

  function handleClick() {
    if (compareMode && onSelect) {
      onSelect(meta.path);
    } else if (onOpen) {
      onOpen(meta);
    }
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      className={cn(
        "group relative block w-full overflow-hidden rounded-lg border bg-background text-left",
        "focus-visible:ring-ring focus-visible:outline-none focus-visible:ring-2",
        selected && "ring-primary ring-2",
      )}
      aria-pressed={compareMode ? selected : undefined}
    >
      <img
        src={thumbUrl ?? originalUrl}
        alt={alt}
        decoding="async"
        className="aspect-square w-full object-contain motion-safe:transition-transform motion-safe:duration-200 motion-safe:ease-[var(--ease-brand-out)] motion-safe:group-hover:scale-[1.03]"
      />
      {selected && (
        <div className="bg-primary/80 absolute inset-0 flex items-center justify-center rounded-lg">
          <svg
            className="size-8 text-white"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2.5}
            aria-hidden
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
        </div>
      )}
      {meta.metric && (
        <div className="absolute right-0 bottom-0 left-0 truncate bg-black/40 px-2 py-1 text-xs text-white">
          {meta.metric}
        </div>
      )}
    </button>
  );
}

export function GalleryGrid({
  items,
  threadId,
  selectedPaths,
  onSelect,
  onOpen,
  compareMode = false,
}: GalleryGridProps) {
  const parentRef = useRef<HTMLDivElement>(null);
  const rowCount = Math.ceil(items.length / COLS);
  const shouldVirtualize = items.length > VIRTUALIZE_THRESHOLD;

  const rowVirtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 2,
    // 不虚拟化时禁用，避免它读取 parentRef（已无固定高度容器）产生无意义测量。
    enabled: shouldVirtualize,
  });

  if (items.length === 0) {
    return null;
  }

  // 小图集：自然高度平铺，无内层滚动容器（页面整体滚动）。彻底避免「1 张图撑出一屏空白」。
  if (!shouldVirtualize) {
    return (
      <div className="grid grid-cols-3 gap-2 p-1 sm:grid-cols-4">
        {items.map((item) => (
          <ThumbCard
            key={item.path}
            meta={item}
            threadId={threadId}
            selected={selectedPaths?.has(item.path) ?? false}
            onSelect={onSelect}
            onOpen={onOpen}
            compareMode={compareMode}
          />
        ))}
      </div>
    );
  }

  // 大图集（per_subject 几十~上百张）：虚拟化 + 有界滚动容器，DOM/显存恒定。
  return (
    <div
      ref={parentRef}
      style={{ height: "calc(100vh - 320px)", overflow: "auto" }}
    >
      <div
        style={{
          height: `${rowVirtualizer.getTotalSize()}px`,
          position: "relative",
        }}
      >
        {rowVirtualizer.getVirtualItems().map((virtualRow) => {
          const startIdx = virtualRow.index * COLS;
          const rowItems = items.slice(startIdx, startIdx + COLS);
          return (
            <div
              key={virtualRow.key}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              <div className="grid grid-cols-3 gap-2 p-1 sm:grid-cols-4">
                {rowItems.map((item) => (
                  <ThumbCard
                    key={item.path}
                    meta={item}
                    threadId={threadId}
                    selected={selectedPaths?.has(item.path) ?? false}
                    onSelect={onSelect}
                    onOpen={onOpen}
                    compareMode={compareMode}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
