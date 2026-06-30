import type { Run } from "@langchain/langgraph-sdk";
import { describe, expect, it } from "vitest";

import {
  clearStaleReconnectRunId,
  findRunStatus,
  getStoredReconnectRunId,
  isReconnectingToTerminalRun,
  isTerminalRunStatus,
} from "./reconnect";

/**
 * Minimal in-memory Storage stub — only the two methods the SDK +
 * reconnect helpers use (getItem / removeItem). Lets the pure helpers be tested
 * without a DOM.
 */
function makeStorage(initial: Record<string, string> = {}): Storage {
  const store = new Map<string, string>(Object.entries(initial));
  return {
    get length() {
      return store.size;
    },
    clear() {
      store.clear();
    },
    getItem(key: string) {
      return store.has(key) ? store.get(key)! : null;
    },
    key(i: number) {
      return Array.from(store.keys())[i] ?? null;
    },
    removeItem(key: string) {
      store.delete(key);
    },
    setItem(key: string, value: string) {
      store.set(key, value);
    },
  };
}

function makeRun(runId: string, status: Run["status"]): Run {
  return {
    run_id: runId,
    thread_id: "thread-1",
    status,
    assistant_id: "lead_agent",
    created_at: "",
    updated_at: "",
  } as Run;
}

describe("getStoredReconnectRunId", () => {
  // The SDK persists the in-flight run id under `lg:stream:<threadId>` in
  // sessionStorage (reconnectOnMount). After a browser crash this key survives
  // and points at the LAST run — which, per the 2026-06-30 logs, had already
  // reached `success`.

  it("reads the run id the SDK persisted for the thread", () => {
    const storage = makeStorage({ "lg:stream:thread-1": "run-success" });
    expect(getStoredReconnectRunId(storage, "thread-1")).toBe("run-success");
  });

  it("returns null when no run id is persisted", () => {
    const storage = makeStorage({});
    expect(getStoredReconnectRunId(storage, "thread-1")).toBeNull();
  });

  it("returns null for an empty-string value (treated as absent)", () => {
    const storage = makeStorage({ "lg:stream:thread-1": "" });
    expect(getStoredReconnectRunId(storage, "thread-1")).toBeNull();
  });

  it("is scoped per thread — does not return another thread's run id", () => {
    const storage = makeStorage({ "lg:stream:thread-other": "run-other" });
    expect(getStoredReconnectRunId(storage, "thread-1")).toBeNull();
  });
});

describe("isTerminalRunStatus", () => {
  it.each(["success", "error", "timeout", "cancelled"])(
    "treats %s as terminal (a cancel against it 409s)",
    (status) => {
      expect(isTerminalRunStatus(status)).toBe(true);
    },
  );

  it.each(["running", "pending"])(
    "treats %s as live (a real run we can cancel / rejoin)",
    (status) => {
      expect(isTerminalRunStatus(status)).toBe(false);
    },
  );

  it("treats 'interrupted' as NOT terminal — it is a paused HITL run the SDK SHOULD rejoin", () => {
    // A clarification/awaiting-user run is interrupted; reconnecting to it is
    // correct behaviour and must not be classified as a stale-terminal spin.
    expect(isTerminalRunStatus("interrupted")).toBe(false);
  });

  it("is case-insensitive", () => {
    expect(isTerminalRunStatus("SUCCESS")).toBe(true);
  });

  it("returns false for unknown / missing statuses", () => {
    expect(isTerminalRunStatus(undefined as unknown as string)).toBe(false);
    expect(isTerminalRunStatus("")).toBe(false);
    expect(isTerminalRunStatus("bogus")).toBe(false);
  });
});

describe("findRunStatus", () => {
  it("returns the status of the matching run", () => {
    const runs = [makeRun("run-a", "success"), makeRun("run-b", "running")];
    expect(findRunStatus(runs, "run-b")).toBe("running");
  });

  it("returns undefined when the run is not in the list", () => {
    const runs = [makeRun("run-a", "success")];
    expect(findRunStatus(runs, "run-missing")).toBeUndefined();
  });

  it("returns undefined for an empty run list", () => {
    expect(findRunStatus([], "run-a")).toBeUndefined();
  });
});

describe("isReconnectingToTerminalRun", () => {
  // 修法 A core: after a crash the SDK re-joins the persisted run id; if that
  // run is already terminal, the join spins (no events) and isLoading stays true.
  // This predicate lets the consumer layer override the UI back to "ready" and
  // allow a new interaction WITHOUT touching useStream internals.

  it("is true when the persisted run id is a terminal-status run", () => {
    const storage = makeStorage({ "lg:stream:thread-1": "run-success" });
    const runs = [makeRun("run-success", "success")];
    expect(
      isReconnectingToTerminalRun({ storage, threadId: "thread-1", runs }),
    ).toBe(true);
  });

  it("is false when the persisted run is still running (a legitimate rejoin)", () => {
    const storage = makeStorage({ "lg:stream:thread-1": "run-live" });
    const runs = [makeRun("run-live", "running")];
    expect(
      isReconnectingToTerminalRun({ storage, threadId: "thread-1", runs }),
    ).toBe(false);
  });

  it("is false when the persisted run is interrupted (HITL awaiting user)", () => {
    const storage = makeStorage({ "lg:stream:thread-1": "run-waiting" });
    const runs = [makeRun("run-waiting", "interrupted")];
    expect(
      isReconnectingToTerminalRun({ storage, threadId: "thread-1", runs }),
    ).toBe(false);
  });

  it("is false when nothing is persisted (fresh mount, no reconnect)", () => {
    const storage = makeStorage({});
    const runs = [makeRun("run-success", "success")];
    expect(
      isReconnectingToTerminalRun({ storage, threadId: "thread-1", runs }),
    ).toBe(false);
  });

  it("is false when the persisted run id is absent from the runs list (status unknown)", () => {
    // If we cannot confirm the run is terminal, do not claim a stale spin —
    // default to trusting the SDK's isLoading so we never hide a live stream.
    const storage = makeStorage({ "lg:stream:thread-1": "run-vanished" });
    const runs = [makeRun("run-other", "success")];
    expect(
      isReconnectingToTerminalRun({ storage, threadId: "thread-1", runs }),
    ).toBe(false);
  });
});

describe("clearStaleReconnectRunId", () => {
  // 修法 B: once we detect the stale-terminal spin we drop the persisted key so
  // (a) the SDK's `stop()` won't POST a cancel against the dead run → no 409,
  // and (b) a subsequent mount won't re-join the dead run.

  it("removes the persisted run id for the thread", () => {
    const storage = makeStorage({
      "lg:stream:thread-1": "run-success",
      "lg:stream:thread-other": "run-other",
    });
    clearStaleReconnectRunId(storage, "thread-1");
    expect(storage.getItem("lg:stream:thread-1")).toBeNull();
    // Other threads are untouched.
    expect(storage.getItem("lg:stream:thread-other")).toBe("run-other");
  });

  it("is a no-op when nothing is persisted", () => {
    const storage = makeStorage({});
    expect(() => clearStaleReconnectRunId(storage, "thread-1")).not.toThrow();
    expect(storage.getItem("lg:stream:thread-1")).toBeNull();
  });
});
