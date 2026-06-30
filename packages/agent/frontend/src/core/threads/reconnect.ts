/**
 * Crash-reconnect stale-run helpers (spec 2026-06-30).
 *
 * After a browser crash on a heavy page (113 charts + long thread), the
 * LangGraph SDK's `useStream({ reconnectOnMount: true })` re-joins the run id it
 * persisted in `sessionStorage["lg:stream:<threadId>"]`. That key survives the
 * crash and points at the LAST run — which, per the gateway logs, had already
 * reached `success`. Joining an already-terminal run's stream yields no events,
 * so `thread.isLoading` stays true and the UI spins ("暂停输出" is the only
 * escape). Worse, the SDK `stop()` reads the SAME key and POSTs a cancel against
 * the dead run → backend correctly 409s ("is not cancellable (status: success)").
 *
 * These helpers let the consumer layer (hooks.ts / chat page) detect and break
 * that spin WITHOUT touching the `useStream` / `mergeMessages` / dedupe core
 * (redline). They are pure functions over an injectable Storage so they can be
 * unit-tested with no DOM.
 */
import type { Run } from "@langchain/langgraph-sdk";

/** sessionStorage key the SDK uses to persist the in-flight run id per thread. */
export const RECONNECT_RUN_STORAGE_KEY = (threadId: string) =>
  `lg:stream:${threadId}`;

/**
 * Terminal-run statuses. A cancel POSTed against any of these 409s server-side.
 * Note: `interrupted` is intentionally excluded — it is a paused HITL run
 * (clarification awaiting user) that the SDK SHOULD rejoin; classifying it as
 * terminal would break legitimate pause/resume reconnects.
 * `pending` / `running` are live runs we can legitimately cancel.
 */
const TERMINAL_RUN_STATUSES = new Set([
  "success",
  "error",
  "timeout",
  "cancelled",
]);

/** Read the run id the SDK persisted for `threadId`, or null if absent/empty. */
export function getStoredReconnectRunId(
  storage: Storage,
  threadId: string,
): string | null {
  const runId = storage.getItem(RECONNECT_RUN_STORAGE_KEY(threadId));
  return runId && runId.length > 0 ? runId : null;
}

/** True iff `status` is a terminal run status (success/error/timeout/cancelled). */
export function isTerminalRunStatus(status: string | undefined): boolean {
  return typeof status === "string" && TERMINAL_RUN_STATUSES.has(status.toLowerCase());
}

/** Look up a run's status in the thread's run list, or undefined if not found. */
export function findRunStatus(
  runs: readonly Run[],
  runId: string,
): Run["status"] | undefined {
  return runs.find((run) => run.run_id === runId)?.status;
}

type ReconnectCheckArgs = {
  storage: Storage;
  threadId: string;
  runs: readonly Run[];
};

/**
 * 修法 A predicate: is the SDK currently re-joining an already-terminal run?
 *
 * True iff a run id is persisted for `threadId` AND that run's status (looked up
 * in `runs`) is terminal. When true, the consumer layer should treat the UI as
 * "ready" (not "streaming") so the user can immediately start a new interaction,
 * and treat a stop-press as a no-op (don't cancel the dead run).
 *
 * Returns false when the persisted run is `running`/`pending`/`interrupted`
 * (legitimate rejoin), when nothing is persisted (fresh mount), or when the run
 * id is absent from `runs` (status unknown — default to trusting the SDK so we
 * never hide a genuinely live stream).
 */
export function isReconnectingToTerminalRun({
  storage,
  threadId,
  runs,
}: ReconnectCheckArgs): boolean {
  const runId = getStoredReconnectRunId(storage, threadId);
  if (!runId) return false;
  const status = findRunStatus(runs, runId);
  if (status === undefined) return false;
  return isTerminalRunStatus(status);
}

/**
 * 修法 B: drop the persisted reconnect run id for `threadId`.
 *
 * Call this once a stale-terminal spin is detected (or once a new run is
 * submitted). Removing the key means:
 *   - the SDK `stop()` will not POST a cancel against the dead run → no 409;
 *   - a subsequent mount will not re-join the dead run.
 * Idempotent; safe to call when nothing is persisted.
 */
export function clearStaleReconnectRunId(storage: Storage, threadId: string): void {
  storage.removeItem(RECONNECT_RUN_STORAGE_KEY(threadId));
}
