"use client";

import { useEffect, useState } from "react";

/**
 * Tracks whether the document (browser tab/window) is currently hidden.
 *
 * Used by the message render layer (spec 2026-06-29-fix-tab-switchback-jank) to
 * detect the "user switched away to another app" state. The render layer uses
 * this to stop accumulating deferred streaming updates while the tab is hidden,
 * so that on switchback React does not have to flush a large backlog of
 * deferred batches in one go (the confirmed cause of switchback jank).
 *
 * Render-layer only. The SSE/merge core (core/threads/hooks.ts) is never aware
 * of visibility — this hook is consumed solely by MessageList to choose its
 * render strategy.
 *
 * SSR-safe: returns `false` on the server and during the first client render
 * (before the effect runs), so the initial mount always uses the default
 * (visible) strategy and there is no hydration mismatch.
 */
export function useDocumentVisibility(): boolean {
  // `document` is undefined during SSR; default to visible (false = not hidden).
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    if (typeof document === "undefined") return;
    // Sync once on mount in case the tab was already hidden when the component
    // mounted (e.g. background-tab restore).
    setHidden(document.hidden);
    const onChange = () => setHidden(document.hidden);
    document.addEventListener("visibilitychange", onChange);
    return () => document.removeEventListener("visibilitychange", onChange);
  }, []);

  return hidden;
}
