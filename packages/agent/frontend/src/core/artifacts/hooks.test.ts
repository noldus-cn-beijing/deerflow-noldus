// @vitest-environment jsdom
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ArtifactMeta } from "@/core/artifacts/types";

import { useChartArtifacts } from "./hooks";

/**
 * spec 2026-06-26-conversation-gallery-empty §一：对话流内嵌图廊改走磁盘端点。
 *
 * 对话流的 InlineArtifactSummary 原吃 thread.values.artifacts（state 冒泡，恒空，
 * 因为 chart-maker subagent 的 artifacts 不上行到 lead）。现统一走 /artifacts/charts
 * 磁盘端点（同 /gallery 独立页），磁盘空/失败时回退 state 代表图。
 *
 * 这里只测 hook 的「磁盘主 + state 回退」契约（SSOT：两入口同源，不复制 fetch 逻辑）。
 */

function mockChart(count: number): ArtifactMeta[] {
  return Array.from({ length: count }, (_, i) => ({
    path: `/mnt/user-data/outputs/chart_${i}.png`,
    kind: "chart",
    output_mode: i % 2 === 0 ? "aggregate" : "per_subject",
    chart_id: `c_${i}`,
  }));
}

/** Coerce a fetch input (string | Request | URL) to its URL string. */
function urlOfInput(input: RequestInfo | URL): string {
  if (typeof input === "string") return input;
  // Request has .url; URL has .href — both string-typed, no default toString.
  if ("url" in input) return input.url;
  return input.href;
}

/** Intercept global fetch and return a JSON body for matching URLs. */
function mockFetch(urlPattern: RegExp, body: unknown, ok = true) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = urlOfInput(input);
    if (urlPattern.test(url)) {
      return new Response(ok ? JSON.stringify(body) : "boom", {
        status: ok ? 200 : 500,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response("not found", { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("useChartArtifacts (disk-primary + state fallback)", () => {
  beforeEach(() => {
    vi.stubGlobal("getBackendBaseURL", () => "http://test");
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("fetches the disk endpoint and normalizes 113 chart metas (the dogfood case)", async () => {
    const charts = mockChart(113);
    mockFetch(/\/artifacts\/charts$/, charts);

    const { result } = renderHook(() => useChartArtifacts("bd7ca7f7"));

    await waitFor(() => expect(result.current.loaded).toBe(true));
    expect(result.current.artifacts).toHaveLength(113);
    expect(result.current.artifacts[0]?.path).toBe(charts[0]!.path);
    expect(result.current.error).toBe(false);
  });

  it("falls back to the provided state artifacts when the disk endpoint returns empty", async () => {
    mockFetch(/\/artifacts\/charts$/, []);
    const stateFallback = mockChart(1);

    const { result } = renderHook(() =>
      useChartArtifacts("t1", { fallbackArtifacts: stateFallback }),
    );

    await waitFor(() => expect(result.current.loaded).toBe(true));
    // 磁盘空 → 退回 state 代表图（至少 1 张）。
    expect(result.current.artifacts).toHaveLength(1);
    expect(result.current.artifacts[0]?.path).toBe(stateFallback[0]!.path);
  });

  it("falls back to state artifacts when the disk endpoint errors (500 / network)", async () => {
    mockFetch(/\/artifacts\/charts$/, null, false);
    const stateFallback = mockChart(2);

    const { result } = renderHook(() =>
      useChartArtifacts("t1", { fallbackArtifacts: stateFallback }),
    );

    await waitFor(() => expect(result.current.loaded).toBe(true));
    expect(result.current.artifacts).toHaveLength(2);
    // 端点失败不静默崩——loaded 仍真，消费方拿到回退数据。
    expect(result.current.error).toBe(true);
  });

  it("exposes a refetch that re-hits the disk endpoint (run-finished → pull full set)", async () => {
    const charts = mockChart(50);
    const fetchMock = mockFetch(/\/artifacts\/charts$/, charts);

    const { result } = renderHook(() => useChartArtifacts("t1"));
    await waitFor(() => expect(result.current.loaded).toBe(true));
    const callsAfterFirst = fetchMock.mock.calls.length;

    await result.current.refetch();
    expect(fetchMock.mock.calls.length).toBeGreaterThan(callsAfterFirst);
    expect(result.current.artifacts).toHaveLength(50);
  });

  it("returns empty list (not null) when both disk and fallback are empty", async () => {
    mockFetch(/\/artifacts\/charts$/, []);
    const { result } = renderHook(() => useChartArtifacts("t1"));
    await waitFor(() => expect(result.current.loaded).toBe(true));
    expect(result.current.artifacts).toEqual([]);
  });

  it("ignores state fallback when the disk endpoint has charts (disk is the SSOT)", async () => {
    const diskCharts = mockChart(10);
    mockFetch(/\/artifacts\/charts$/, diskCharts);
    const stateFallback = mockChart(1);

    const { result } = renderHook(() =>
      useChartArtifacts("t1", { fallbackArtifacts: stateFallback }),
    );

    await waitFor(() => expect(result.current.loaded).toBe(true));
    // 磁盘有数据 → 不混入 state 代表图（磁盘是唯一真相）。
    expect(result.current.artifacts).toHaveLength(10);
    expect(result.current.artifacts[0]?.path).toBe(diskCharts[0]!.path);
  });
});
