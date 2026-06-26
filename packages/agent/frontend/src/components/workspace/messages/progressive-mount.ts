"use client";

import { useEffect, useRef, useState } from "react";

/**
 * chat-render-jank-on-open fix (Fix 2, 2026-06-26) — defer the one-shot mount
 * of a heavy historical message list.
 *
 * Opening a thread commits every historical group in a single React commit
 * (markdown + charts + Collapsible subtrees). That one-shot mount is the bulk
 * of the open-jank CPU profile (jsxDEV + commit*Effects). For the
 * non-virtualized path (groups below `VIRTUALIZATION_THRESHOLD`), this hook
 * mounts only the last `initialVisible` groups synchronously — the user's
 * viewport lands on the most recent messages — and reveals the earlier groups
 * in idle-time batches wrapped in `startTransition`. The browser paints the
 * visible slice first, then low-priority backfills the rest.
 *
 * Gating (all must hold, else it's a transparent no-op that renders everything):
 *  - `enabled === false` → render all immediately (e.g. while streaming, or
 *    when the list is short enough that splitting isn't worth it).
 *  - Once `visibleCount` reaches `total` it never shrinks and the hook stops
 *    scheduling — steady-state cost is zero.
 *  - The hook only ever GROWS `visibleCount`. Appended groups during streaming
 *    are already covered because streaming runs with `enabled=false`.
 *
 * The slice is taken from the END (`groups.length - visibleCount`) so the
 * newest messages — what the user opened the thread to see — paint first.
 *
 * Scrolling to an earlier stage node (#214 analysis rail) finds its target as
 * soon as the next idle batch reveals it (a few frames); we don't try to be
 * smarter than that here, which keeps the integration with stick-to-bottom and
 * the virtualized path risk-free.
 */
export const PROGRESSIVE_MOUNT_INITIAL_VISIBLE = 8;
/** Only split mounts when the list is meaningfully heavier than the initial slice. */
export const PROGRESSIVE_MOUNT_MIN_TOTAL = 12;
/** Groups revealed per idle tick. Small enough to keep each frame under budget. */
export const PROGRESSIVE_MOUNT_BATCH_SIZE = 6;

export interface ProgressiveMountOptions {
  total: number;
  /**
   * Master switch. Callers pass `false` while the thread is actively streaming
   * (so appended messages never get hidden behind a progressive reveal) and for
   * lists below `PROGRESSIVE_MOUNT_MIN_TOTAL` (no point splitting a short list).
   */
  enabled: boolean;
  initialVisible?: number;
  batchSize?: number;
  /**
   * Injectable idle scheduler (defaults to `requestIdleCallback` /
   * `requestAnimationFrame` fallback) so the hook is deterministic in tests.
   * Receives a callback and returns a cancel handle.
   */
  scheduleIdle?: (cb: () => void) => () => void;
}

/**
 * Returns how many groups (counted from the END of the list) to render right
 * now. Starts at `initialVisible` (or `total` if disabled) and grows toward
 * `total` across idle ticks until the whole list is mounted.
 */
export function useProgressiveMount({
  total,
  enabled,
  initialVisible = PROGRESSIVE_MOUNT_INITIAL_VISIBLE,
  batchSize = PROGRESSIVE_MOUNT_BATCH_SIZE,
  scheduleIdle = defaultScheduleIdle,
}: ProgressiveMountOptions): number {
  const start = enabled ? Math.min(initialVisible, total) : total;
  const [visibleCount, setVisibleCount] = useState(start);
  // Track the latest total so the idle callback (captured once per reveal
  // cycle) reads a fresh value without being re-created every render.
  const totalRef = useRef(total);
  totalRef.current = total;

  useEffect(() => {
    if (!enabled) {
      // Disabled path: make sure everything is visible (no-op if already).
      setVisibleCount(totalRef.current);
      return;
    }
    if (visibleCount >= totalRef.current) {
      return;
    }
    let cancelled = false;
    const revealNext = () => {
      if (cancelled) return;
      const t = totalRef.current;
      setVisibleCount((prev) => {
        if (prev >= t) return prev;
        return Math.min(prev + batchSize, t);
      });
    };
    const cancel = scheduleIdle(revealNext);
    return () => {
      cancelled = true;
      cancel();
    };
  }, [enabled, visibleCount, batchSize, scheduleIdle]);

  return Math.min(visibleCount, total);
}

function defaultScheduleIdle(cb: () => void): () => void {
  if (typeof window !== "undefined" && typeof window.requestIdleCallback === "function") {
    const handle = window.requestIdleCallback(() => cb());
    return () => window.cancelIdleCallback(handle);
  }
  if (typeof window !== "undefined" && typeof window.requestAnimationFrame === "function") {
    const handle = window.requestAnimationFrame(() => cb());
    return () => window.cancelAnimationFrame(handle);
  }
  // SSR / no timers: fire next microtask. The reveal still completes, just not
  // broken across frames. `void` — the promise is intentionally fire-and-forget
  // (the cancel flag guards late invocation).
  let done = false;
  void Promise.resolve().then(() => {
    if (!done) cb();
  });
  return () => {
    done = true;
  };
}
