"use client";

import { type ReactNode, useMemo } from "react";

import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

export interface AttachmentStackProps {
  /** Number hidden behind the stack (the "+N" badge value). */
  count: number;
  /** Aggregated upload progress, optional. When set, shown under the badge. */
  progressLabel?: string;
  /** Controlled open state — parent drives both hover (desktop) and tap (touch). */
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Fan-out panel content (virtualized list of remaining files). */
  children: ReactNode;
  className?: string;
}

/**
 * The collapsed stack visual: 2–3 slightly offset, rotated card "shadows" with
 * `--shadow-overlap`, a tabular-nums "+N" badge, and an optional aggregated
 * progress line. Triggers the fan-out via a Radix HoverCard whose `open` state
 * is controlled by the parent so BOTH desktop hover and touch tap open it
 * (spec §3.2 `Hover vs Tap` High red line — hover is never the only entry).
 *
 * Deliberately quiet (spec §3.3 一屏一主角): low-saturation, small, sits in one
 * fixed slot so the input area height stays constant as uploads grow.
 */
export function AttachmentStack({
  count,
  progressLabel,
  open,
  onOpenChange,
  children,
  className,
}: AttachmentStackProps) {
  const { t } = useI18n();
  // Up to 2 underlying layers behind the top card (Seedance-style stack).
  const layers = useMemo(() => (count > 1 ? [0, 1] : [0]), [count]);

  return (
    <HoverCard
      closeDelay={120}
      onOpenChange={onOpenChange}
      open={open}
      openDelay={90}
    >
      <HoverCardTrigger asChild>
        <button
          aria-expanded={open}
          aria-haspopup="dialog"
          aria-label={t.inputBox.stackLabel.replace("{count}", String(count))}
          className={cn(
            // Visual badge is compact (h-8) but the button pads out to a ≥44px
            // touch target (spec §3.2 touch-target-size) without growing layout.
            "focus-visible:ring-ring group relative inline-flex min-h-11 cursor-pointer items-center",
            "rounded-md px-1.5 py-1.5 outline-none focus-visible:ring-2",
            "motion-safe:transition-transform motion-safe:duration-fast motion-safe:ease-brand-out",
            "active:scale-[0.97]",
            className,
          )}
          // Touch tap toggles; desktop hover is handled by HoverCard delays.
          onClick={() => onOpenChange(!open)}
          type="button"
        >
          {/* Layered card shadows behind the top card. */}
          <span className="relative inline-flex">
            {layers.map((layer) => (
              <span
                aria-hidden
                key={layer}
                className={cn(
                  "border-border bg-card absolute h-8 w-7 rounded-md border",
                  "shadow-overlap",
                )}
                style={{
                  // Each underlying card offsets + rotates slightly; top card at 0.
                  transform: `translate(${layer * 3}px, ${layer * -2}px) rotate(${layer * -4}deg)`,
                  opacity: 1 - layer * 0.18,
                  zIndex: layers.length - layer,
                }}
              />
            ))}
            {/* Top card with the +N badge. */}
            <span
              className="border-border bg-card relative z-10 inline-flex h-8 w-7 items-center justify-center rounded-md border"
            >
              <span className="flex flex-col items-center leading-none">
                <span className="text-muted-foreground text-[11px] font-semibold tabular-nums">
                  +{count}
                </span>
                {progressLabel ? (
                  <span className="text-muted-foreground/70 mt-0.5 max-w-16 truncate text-[9px] tabular-nums">
                    {progressLabel}
                  </span>
                ) : null}
              </span>
            </span>
          </span>
        </button>
      </HoverCardTrigger>
      <HoverCardContent
        align="start"
        className="w-72 p-2"
        // ESC closes (modal-escape); Radix handles Escape natively.
        onEscapeKeyDown={() => onOpenChange(false)}
        onInteractOutside={() => onOpenChange(false)}
      >
        {children}
      </HoverCardContent>
    </HoverCard>
  );
}
