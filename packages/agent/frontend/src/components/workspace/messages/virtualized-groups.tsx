"use client";

import { useVirtualizer } from "@tanstack/react-virtual";
import { useRef, type ReactNode } from "react";
import type { StickToBottomContext } from "use-stick-to-bottom";

import { cn } from "@/lib/utils";

/**
 * Phase0#7 Step 4 — message-list windowing (spec §3.1).
 *
 * The chat scroll container is owned by `StickToBottom` (from the ai-elements
 * `Conversation`), which exposes its scroll/content elements via a `contextRef`
 * (StickToBottomContext). `VirtualizedGroups` reads `scrollRef.current` off
 * that context to drive `@tanstack/react-virtual`'s `useVirtualizer`, so only
 * the visible message groups (+ overscan) are mounted — DOM node count stays
 * roughly constant regardless of how long the thread gets.
 *
 * Dynamic heights are handled with `measureElement` (reasoning panels expand/
 * collapse, images, tables) plus an `estimateSize` fallback. Stick-to-bottom
 * is left to the owning `Conversation` (it already tracks "are we at the
 * bottom" and auto-scrolls on growth); we only render the windowed slice here.
 *
 * Risk / degradation (spec §3.1 / Step 4): virtualizing a dynamic-height,
 * streaming-append list on top of a third-party scroll manager is the
 * highest-risk item in the perf spec. MessageList only opts into this
 * component when `groups.length >= VIRTUALIZATION_THRESHOLD`; below that the
 * plain map path is used (no measurement overhead). If the dynamic-measurement
 * integration proves unstable in dogfood, set the threshold to `Infinity` (or
 * drop this component from MessageList) — Step 1-3 (defer + memo) already
 * remove the dominant streaming jank, so virtualization is additive.
 */
export const VIRTUALIZATION_THRESHOLD = 30;

export function VirtualizedGroups({
  groups,
  scrollContext,
  paddingBottomPx,
  className,
  children,
}: {
  groups: ReactNode[];
  /**
   * The StickToBottom context captured via Conversation's `contextRef`. The
   * scroll element lives inside the registry Conversation component; we only
   * read its ref here (no edit to the registry component).
   */
  scrollContext: StickToBottomContext | null;
  paddingBottomPx: number;
  className?: string;
  /** Trailing nodes (streaming indicator + bottom padding) rendered after the windowed groups. */
  children?: ReactNode;
}) {
  // The scroll element isn't available on first render (StickToBottom populates
  // context.scrollRef after mount). useVirtualizer tolerates a null scroll
  // element initially and re-measures once it arrives (it polls getScrollElement).
  const getScrollElement = useRefCallback(() => {
    return scrollContext?.scrollRef?.current ?? null;
  });

  const virtualizer = useVirtualizer({
    count: groups.length,
    getScrollElement,
    estimateSize: () => 160,
    overscan: 6,
    measureElement:
      typeof window !== "undefined" &&
      navigator.userAgent.includes("Firefox")
        ? undefined
        : (el) => el?.getBoundingClientRect().height ?? 160,
    getItemKey: (index) => index,
  });

  const totalSize = virtualizer.getTotalSize();

  return (
    <div className={cn("relative w-full", className)}>
      <div
        style={{
          height: `${Math.max(0, totalSize + paddingBottomPx)}px`,
        }}
        className="relative"
      >
        {virtualizer.getVirtualItems().map((virtualItem) => {
          const node = groups[virtualItem.index];
          if (node == null) {
            return null;
          }
          return (
            <div
              key={virtualItem.key}
              data-index={virtualItem.index}
              ref={virtualizer.measureElement}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                transform: `translateY(${virtualItem.start}px)`,
              }}
            >
              {node}
            </div>
          );
        })}
      </div>
      {children}
    </div>
  );
}

/**
 * Stable identity getter so useVirtualizer's `getScrollElement` option doesn't
 * change every render (it would otherwise tear down/recreate the instance).
 */
function useRefCallback<T>(fn: () => T) {
  const ref = useRef(fn);
  ref.current = fn;
  return ref.current;
}
