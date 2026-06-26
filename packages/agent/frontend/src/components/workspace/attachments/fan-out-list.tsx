"use client";

import { useVirtualizer } from "@tanstack/react-virtual";
import { Fragment, useRef } from "react";

import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

import { AttachmentChip, type Attachment } from "./attachment-chip";

const ROW_HEIGHT = 40; // h-8 chip + gap; ≥44px touch target via row padding
const VIRTUALIZE_THRESHOLD = 50; // spec §3.1: virtualize the fan-out list beyond 50

export interface FanOutListProps {
  items: Attachment[];
  /**
   * Remove handler. Required in the editable input-box context (store-backed);
   * OPTIONAL in read-only contexts like the message-flow attachments, where a
   * sent message's files cannot be removed. When omitted, no ✕ control renders.
   */
  onRemove?: (id: string) => void;
}

/**
 * Scrollable list of the stacked (overflow) attachments shown when the stack
 * fans out. Each row is an `AttachmentChip` with a persistent ✕ (touch users
 * have no hover). Beyond `VIRTUALIZE_THRESHOLD` rows the list virtualizes with
 * `@tanstack/react-virtual` — the same engine as the artifact gallery (#3) and
 * the message stream (#7) — so hundreds of files stay scrollable.
 *
 * Rows stagger their entrance (spec §3.3 `stagger-sequence`) via a CSS var per
 * index; reduced-motion collapses this through the global transition override.
 */
export function FanOutList({ items, onRemove }: FanOutListProps) {
  const { t } = useI18n();
  const scrollRef = useRef<HTMLDivElement>(null);
  const shouldVirtualize = items.length > VIRTUALIZE_THRESHOLD;

  const virtualizer = useVirtualizer({
    count: items.length,
    enabled: shouldVirtualize,
    estimateSize: () => ROW_HEIGHT,
    getScrollElement: () => scrollRef.current,
    overscan: 4,
  });

  const rows = shouldVirtualize
    ? virtualizer.getVirtualItems().map((v) => ({
        key: v.key,
        index: v.index,
        transform: `translateY(${v.start}px)`,
      }))
    : items.map((_, index) => ({
        key: items[index]!.id,
        index,
        transform: undefined as string | undefined,
      }));

  return (
    <Fragment>
      <div
        // Title is a real heading for the fan-out (keyboard users landing here).
        className="text-muted-foreground mb-1 px-1 text-xs font-medium"
      >
        {t.inputBox.stackFanOutTitle.replace(
          "{count}",
          String(items.length),
        )}
      </div>
      <div
        ref={scrollRef}
        className={cn(
          "max-h-72 overflow-auto pr-1",
          // Only virtualized mode needs a positioned spacer; plain mode is a column.
          shouldVirtualize && "relative",
        )}
      >
        {shouldVirtualize ? (
          <div style={{ height: `${virtualizer.getTotalSize()}px` }}>
            {rows.map((row) => (
              <FanOutRow
                key={row.key}
                item={items[row.index]!}
                index={row.index}
                onRemove={onRemove}
                absolute={row.transform}
              />
            ))}
          </div>
        ) : (
          <div className="flex flex-col gap-1">
            {rows.map((row) => (
              <FanOutRow
                key={row.key}
                item={items[row.index]!}
                index={row.index}
                onRemove={onRemove}
              />
            ))}
          </div>
        )}
      </div>
    </Fragment>
  );
}

interface FanOutRowProps {
  item: Attachment;
  index: number;
  /** Optional: when omitted the row is read-only and renders no ✕ control. */
  onRemove?: (id: string) => void;
  absolute?: string;
}

function FanOutRow({ item, index, onRemove, absolute }: FanOutRowProps) {
  return (
    <div
      className={cn(
        "flex min-h-11 items-center",
        // Absolute positioning only in virtualized mode.
        absolute !== undefined &&
          "absolute left-0 top-0 w-full motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-bottom-1",
      )}
      style={{
        ...(absolute !== undefined
          ? {
              transform: absolute,
              // Stagger the entrance: each row delays ~30ms (cap to keep it snappy).
              animationDelay: `${Math.min(index, 12) * 30}ms`,
            }
          : {
              animationDelay: `${Math.min(index, 12) * 30}ms`,
            }),
      }}
    >
      <AttachmentChip
        // In the fan-out the remove control is always visible when removal is
        // possible (touch users have no hover). Read-only callers pass no
        // onRemove → AttachmentChip renders no ✕ at all, so alwaysShowRemove
        // is only meaningful when a handler exists.
        alwaysShowRemove={Boolean(onRemove)}
        className="motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-bottom-1 w-full"
        data={item}
        onRemove={onRemove}
      />
    </div>
  );
}
