// @vitest-environment jsdom
import { act } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  PROGRESSIVE_MOUNT_BATCH_SIZE,
  PROGRESSIVE_MOUNT_INITIAL_VISIBLE,
  useProgressiveMount,
} from "@/components/workspace/messages/progressive-mount";

/**
 * chat-render-jank-on-open Fix 2 (2026-06-26) — `useProgressiveMount`.
 *
 * We test the hook in isolation with an injected SYNCHRONOUS idle scheduler so
 * the reveal sequence is deterministic (no real requestIdleCallback timing).
 * The integration (last-N-first slicing, only-on-non-virtualized, disabled
 * while streaming) lives in message-list.tsx and is exercised by the dogfood
 * checklist; here we pin the contract:
 *  - disabled → render everything immediately,
 *  - enabled → starts at the initial slice and grows by `batchSize` per idle
 *    tick until it reaches `total`, then stops scheduling,
 *  - never exceeds `total`.
 */

// A scheduler that runs the callback synchronously the first time, then on each
// subsequent call. Each `tick()` drains one queued reveal.
function makeSyncScheduler() {
  const queue: Array<() => void> = [];
  const scheduleIdle = (cb: () => void) => {
    queue.push(cb);
    // automatic cancel handle (no-op)
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    return () => {};
  };
  const tick = () => {
    const cb = queue.shift();
    if (cb) cb();
  };
  const pending = () => queue.length;
  return { scheduleIdle, tick, pending };
}

describe("useProgressiveMount (Fix 2, 2026-06-26)", () => {
  it("renders everything immediately when disabled", async () => {
    const { renderHook } = await import("@testing-library/react");
    const { result } = renderHook(() =>
      useProgressiveMount({ total: 30, enabled: false }),
    );
    expect(result.current).toBe(30);
  });

  it("starts at the initial visible slice and reveals the rest in idle batches", async () => {
    const { renderHook } = await import("@testing-library/react");
    const sched = makeSyncScheduler();
    const { result } = renderHook(() =>
      useProgressiveMount({ total: 25, enabled: true, scheduleIdle: sched.scheduleIdle }),
    );

    // First paint: only the initial slice (the most recent groups).
    expect(result.current).toBe(PROGRESSIVE_MOUNT_INITIAL_VISIBLE);

    // Drain idle ticks one at a time; each grows by batchSize.
    act(() => sched.tick());
    expect(result.current).toBe(
      PROGRESSIVE_MOUNT_INITIAL_VISIBLE + PROGRESSIVE_MOUNT_BATCH_SIZE,
    );

    act(() => sched.tick());
    expect(result.current).toBe(
      PROGRESSIVE_MOUNT_INITIAL_VISIBLE + 2 * PROGRESSIVE_MOUNT_BATCH_SIZE,
    );

    // Tick until the full list is mounted.
    act(() => sched.tick());
    act(() => sched.tick());
    expect(result.current).toBe(25);
    // No more pending reveals once we've reached total.
    expect(sched.pending()).toBe(0);
  });

  it("never reports more than total", async () => {
    const { renderHook } = await import("@testing-library/react");
    const sched = makeSyncScheduler();
    const { result } = renderHook(() =>
      useProgressiveMount({ total: 10, enabled: true, scheduleIdle: sched.scheduleIdle }),
    );
    // initialVisible (8) < total (10) → first paint is 8, then clamps at total.
    expect(result.current).toBe(8);
    act(() => sched.tick());
    // 8 + batchSize 6 = 14 would overshoot → clamp at total (10).
    expect(result.current).toBe(10);
    act(() => sched.tick());
    expect(result.current).toBe(10);
  });

  it("stops scheduling once fully revealed (no busy-loop)", async () => {
    const { renderHook } = await import("@testing-library/react");
    const sched = makeSyncScheduler();
    renderHook(() =>
      useProgressiveMount({ total: 9, enabled: true, scheduleIdle: sched.scheduleIdle }),
    );
    // total (9) > MIN path handled by the hook internally only when enabled and
    // total > visibleCount; here initialVisible (8) < 9 so one reveal is queued.
    expect(sched.pending()).toBe(1);
    act(() => sched.tick());
    expect(sched.pending()).toBe(0);
  });
});
